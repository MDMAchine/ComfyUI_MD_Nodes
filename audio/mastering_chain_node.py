# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# ████ MD_Nodes/MasteringChainNode – Professional Mastering Suite ████▓▒░
# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
# ░▒▓ ORIGIN & DEV:
#    • Cast into the void by: MDMAchine (sonic architect)
#    • Enhanced by: Gemini (Adapter/Fixes), Claude (v1.3 Overhaul), MDMAchine (QA)
#    • License: Apache 2.0 — No cursed floppy disks allowed

# ░▒▓ DESCRIPTION:
#    Professional-grade mastering suite for ComfyUI. Contains the full
#    Mastering Chain (All-in-One) as well as individual modular nodes for
#    Gain, EQ, Compression, and Limiting.

# ░▒▓ FEATURES:
#    ✓ Modular architecture: Use the full chain or individual components
#    ✓ Global gain control
#    ✓ Butterworth low-pass and high-pass cutoff filters
#    ✓ High-shelf and low-shelf filters (RBJ coefficients)
#    ✓ Four parametric EQ bands
#    ✓ Single-band compressor (Pedalboard)
#    ✓ 3-band stereo-linked multiband compressor (Pedalboard)
#    ✓ Linkwitz-Riley crossovers with configurable order
#    ✓ Lookahead limiter (Pedalboard)
#    ✓ Before/after waveform visualizations with RMS/Peak analysis

# ░▒▓ CHANGELOG:
#    - v1.5.1 (Plotting & Stability Fixes):
#        • FIXED: Added smart downsampling to waveform plots for performance.
#        • ENHANCED: Added RMS and Peak markers to visualization.
#        • FIXED: Added explicit dtype to all tensor initializations.
#        • ENHANCED: Added comprehensive tooltips to modular nodes.
#    - v1.5.0 (Modularization):
#        • ADDED: Individual nodes: Gain, EQ, Compressor, Limiter.
#        • REFACTOR: Moved core logic to `MasteringBase` class.
#    - v1.4.3a (Guide Update):
#        • UPDATED: Converted to ComfyUI_MD_Nodes Guide v1.4.3a standards.

# ░▒▓ CONFIGURATION:
#    → Primary Use: Professional audio mastering for normalized material (-14 LUFS).
#    → Secondary Use: Creative audio shaping and frequency sculpting.

# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄


# =================================================================================
# == Standard Library Imports                                                    ==
# =================================================================================
import io
import sys
import traceback
import logging
from typing import Dict, Tuple, Any, Optional, List

# =================================================================================
# == Third-Party Imports                                                         ==
# =================================================================================
import torch
import numpy as np
from scipy import signal
import matplotlib
matplotlib.use('Agg') # Set backend before pyplot import
import matplotlib.pyplot as plt
from PIL import Image
import pedalboard

# =================================================================================
# == Constants                                                                   ==
# =================================================================================
BLANK_PLOT_SIZE = (1, 256, 512, 3) # [B, H, W, C]
MAX_PLOT_SAMPLES = 150000 # Max samples to plot before downsampling

# Setup logger (Defensive)
logger = logging.getLogger("ComfyUI_MD_Nodes.MasteringChain")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('[%(name)s] %(levelname)s: %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# =================================================================================
# == Base Processing Class (Shared Logic)                                        ==
# =================================================================================

class MasteringBase:
    """
    Shared signal processing logic for all Mastering nodes.
    """

    def _db_to_amplitude(self, db: float) -> float:
        """Converts dB to linear amplitude."""
        return 10**(db / 20.0)

    def _amplitude_to_db(self, amplitude: float) -> float:
        """Converts linear amplitude to dB."""
        return 20 * np.log10(np.maximum(1e-9, np.abs(amplitude)))

    def _apply_gain(self, audio_data: np.ndarray, gain_db: float) -> np.ndarray:
        """Applies gain in dB to audio data."""
        if gain_db == 0.0:
            return audio_data
        return audio_data * self._db_to_amplitude(gain_db)

    # --- Filter Design Helpers ---

    def _design_lowpass_filter(self, cutoff_freq: float, sample_rate: int, order: int = 4):
        nyquist = 0.5 * sample_rate
        normal_cutoff = np.clip(cutoff_freq / nyquist, 0.01, 0.99)
        b, a = signal.butter(order, normal_cutoff, btype='lowpass', analog=False)
        return signal.tf2sos(b, a)

    def _design_highpass_filter(self, cutoff_freq: float, sample_rate: int, order: int = 4):
        nyquist = 0.5 * sample_rate
        normal_cutoff = np.clip(cutoff_freq / nyquist, 0.01, 0.99)
        b, a = signal.butter(order, normal_cutoff, btype='highpass', analog=False)
        return signal.tf2sos(b, a)

    def _design_peaking_filter(self, gain_db: float, freq: float, Q: float, sample_rate: int):
        if abs(gain_db) < 0.001 or Q <= 0 or freq <= 0 or freq >= sample_rate / 2:
            return None
        
        A = 10**(gain_db / 40.0)
        omega = 2 * np.pi * freq / sample_rate
        sn = np.sin(omega)
        cs = np.cos(omega)
        alpha = sn / (2 * Q)
        a0 = 1 + alpha / A

        if abs(a0) < 1e-9: return None

        b0 = 1 + alpha * A
        b1 = -2 * cs
        b2 = 1 - alpha * A
        a1 = -2 * cs
        a2 = 1 - alpha / A

        b = np.array([b0 / a0, b1 / a0, b2 / a0], dtype=np.float64)
        a = np.array([1.0, a1 / a0, a2 / a0], dtype=np.float64)
        
        if not np.all(np.isfinite(b)) or not np.all(np.isfinite(a)): return None
        return signal.tf2sos(b, a)

    def _design_low_shelf_filter(self, gain_db: float, freq: float, sample_rate: int):
        if abs(gain_db) < 0.001 or freq <= 0 or freq >= sample_rate / 2: return None
        
        A = 10**(gain_db / 40.0)
        w0 = 2 * np.pi * freq / sample_rate
        cos_w0 = np.cos(w0)
        sin_w0 = np.sin(w0)
        alpha = sin_w0 / (2 * 0.707)
        a0 = (A+1) + (A-1)*cos_w0 + 2*np.sqrt(A)*alpha

        if abs(a0) < 1e-9: return None

        b0 = A * ((A+1) - (A-1)*cos_w0 + 2*np.sqrt(A)*alpha)
        b1 = 2 * A * ((A-1) - (A+1)*cos_w0)
        b2 = A * ((A+1) - (A-1)*cos_w0 - 2*np.sqrt(A)*alpha)
        a1 = -2 * ((A-1) + (A+1)*cos_w0)
        a2 = (A+1) + (A-1)*cos_w0 - 2*np.sqrt(A)*alpha

        b = np.array([b0 / a0, b1 / a0, b2 / a0], dtype=np.float64)
        a = np.array([1.0, a1 / a0, a2 / a0], dtype=np.float64)
        
        if not np.all(np.isfinite(b)) or not np.all(np.isfinite(a)): return None
        return signal.tf2sos(b, a)

    def _design_high_shelf_filter(self, gain_db: float, freq: float, sample_rate: int):
        if abs(gain_db) < 0.001 or freq <= 0 or freq >= sample_rate / 2: return None
        
        A = 10**(gain_db / 40.0)
        w0 = 2 * np.pi * freq / sample_rate
        cos_w0 = np.cos(w0)
        sin_w0 = np.sin(w0)
        alpha = sin_w0 / (2 * 0.707)
        a0 = (A+1) - (A-1)*cos_w0 + 2*np.sqrt(A)*alpha

        if abs(a0) < 1e-9: return None

        b0 = A * ((A+1) + (A-1)*cos_w0 + 2*np.sqrt(A)*alpha)
        b1 = -2 * A * ((A-1) + (A+1)*cos_w0)
        b2 = A * ((A+1) + (A-1)*cos_w0 - 2*np.sqrt(A)*alpha)
        a1 = 2 * ((A-1) - (A+1)*cos_w0)
        a2 = (A+1) - (A-1)*cos_w0 - 2*np.sqrt(A)*alpha

        b = np.array([b0 / a0, b1 / a0, b2 / a0], dtype=np.float64)
        a = np.array([1.0, a1 / a0, a2 / a0], dtype=np.float64)

        if not np.all(np.isfinite(b)) or not np.all(np.isfinite(a)): return None
        return signal.tf2sos(b, a)

    def _design_linkwitz_riley_crossover(self, cutoff_freq: float, sample_rate: int, order: int = 8):
        if order % 2 != 0: order = max(4, order + 1)
        nyquist = 0.5 * sample_rate
        normal_cutoff = np.clip(cutoff_freq / nyquist, 0.01, 0.99)
        try:
            b_lp, a_lp = signal.butter(order // 2, normal_cutoff, btype='lowpass')
            b_hp, a_hp = signal.butter(order // 2, normal_cutoff, btype='highpass')
            sos_lp = signal.tf2sos(b_lp, a_lp)
            sos_hp = signal.tf2sos(b_hp, a_hp)
            if np.any(np.abs(np.roots(a_lp)) >= 1.0) or np.any(np.abs(np.roots(a_hp)) >= 1.0):
                 raise ValueError("Unstable filter")
            return np.vstack([sos_lp, sos_lp]), np.vstack([sos_hp, sos_hp])
        except ValueError as e:
            logger.error(f"Crossover design failed: {e}")
            raise

    def _apply_filters_to_audio(self, audio_data: np.ndarray, sos_filters: List) -> np.ndarray:
        if not sos_filters or all(s is None for s in sos_filters): return audio_data
        filtered_audio = audio_data.copy()
        for sos in sos_filters:
             if sos is not None:
                try:
                    filtered_audio = signal.sosfiltfilt(sos, filtered_audio, axis=-1)
                except ValueError:
                     continue
        filtered_audio = np.nan_to_num(filtered_audio, nan=0.0, posinf=1.0, neginf=-1.0)
        return np.clip(filtered_audio, -1.0, 1.0)

    def _plot_waveform_to_tensor(self, audio_data: np.ndarray, sample_rate: int, title: str = "Waveform") -> torch.Tensor:
        """
        Plots waveform and returns as tensor using the reliable PIL buffer method.
        Includes downsampling and enhanced stats (Peak/RMS).
        """
        if audio_data is None or audio_data.size == 0:
            return torch.zeros(BLANK_PLOT_SIZE, dtype=torch.float32)
        
        fig = None
        try:
            plt.style.use('dark_background')
            fig, ax = plt.subplots(figsize=(10, 3))
            
            # Use first channel for plotting
            plot_data = audio_data[0, :] if audio_data.ndim == 2 else audio_data
            
            # [NODE_FIX] Smart Downsampling
            num_samples_original = len(plot_data)
            if num_samples_original > MAX_PLOT_SAMPLES:
                ds_factor = num_samples_original // MAX_PLOT_SAMPLES
                plot_data = plot_data[::ds_factor]
                time_axis = np.linspace(0, num_samples_original / sample_rate, len(plot_data))
                logger.debug(f"Plot downsampled {ds_factor}x for '{title}'")
            else:
                time_axis = np.linspace(0, num_samples_original / sample_rate, len(plot_data))
            
            ax.plot(time_axis, plot_data, color='#87CEEB', linewidth=0.5)
            
            # [NODE_FIX] Enhanced Stats (Peak/RMS)
            peak_val = np.max(np.abs(plot_data)) if plot_data.size > 0 else 0.0
            rms = np.sqrt(np.mean(plot_data**2)) if plot_data.size > 0 else 0.0
            
            # Add markers
            if peak_val > 0.8:
                ax.axhline(y=peak_val, color='orangered', ls='--', lw=0.7, alpha=0.6, label=f'Peak: {peak_val:.3f}')
                ax.axhline(y=-peak_val, color='orangered', ls='--', lw=0.7, alpha=0.6)
            
            ax.axhline(y=rms, color='mediumseagreen', ls=':', lw=0.7, alpha=0.6, label=f'RMS: {rms:.3f}')
            ax.axhline(y=-rms, color='mediumseagreen', ls=':', lw=0.7, alpha=0.6)

            ax.set_title(f"{title} | Peak: {peak_val:.3f} | RMS: {rms:.3f}", fontsize=10)
            ax.set_xlabel("Time (s)", fontsize=8)
            ax.set_ylabel("Amplitude", fontsize=8)
            ax.set_ylim(-1.05, 1.05)
            ax.set_yticks([-1, -0.5, 0, 0.5, 1])
            ax.grid(True, ls=':', lw=0.5, alpha=0.3)
            ax.legend(loc='upper right', fontsize=7, framealpha=0.5)
            
            # Styling
            ax.tick_params(axis='both', which='major', labelsize=7)
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            plt.tight_layout()
            
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=96, facecolor=fig.get_facecolor())
            buf.seek(0)
            img = Image.open(buf).convert("RGB")
            img_np = np.array(img).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_np).unsqueeze(0)
            buf.close()
            plt.close(fig)
            return img_tensor
        except Exception:
            if fig: plt.close(fig)
            return torch.zeros(BLANK_PLOT_SIZE, dtype=torch.float32)

    # --- Common Processing Blocks ---

    def process_eq(self, processed, sample_rate, **params):
        """Executes EQ logic."""
        sos_filters = []
        # Cutoffs
        if params.get('enable_lowpass'):
            sos_filters.append(self._design_lowpass_filter(params['lowpass_freq'], sample_rate, params['lowpass_order']))
        if params.get('enable_highpass'):
            sos_filters.append(self._design_highpass_filter(params['highpass_freq'], sample_rate, params['highpass_order']))
        # Shelves
        sos = self._design_high_shelf_filter(params['eq_high_shelf_gain_db'], params['eq_high_shelf_freq'], sample_rate)
        if sos is not None: sos_filters.append(sos)
        if params.get('enable_low_shelf_eq'):
            sos = self._design_low_shelf_filter(params['eq_low_shelf_gain_db'], params['eq_low_shelf_freq'], sample_rate)
            if sos is not None: sos_filters.append(sos)
        # Parametric
        for i in range(1, 5):
            if params.get(f'enable_param_eq{i}'):
                sos = self._design_peaking_filter(params[f'param_eq{i}_gain_db'], params[f'param_eq{i}_freq'], params[f'param_eq{i}_q'], sample_rate)
                if sos is not None: sos_filters.append(sos)
        
        return self._apply_filters_to_audio(processed, sos_filters)

    def process_compression(self, processed, sample_rate, **params):
        """Executes Compression logic."""
        if not params.get('enable_comp', False): return processed
        
        audio_float = processed.astype(np.float32)
        comp_type = params.get('comp_type', "Multiband")
        
        if comp_type == "Single-Band":
            compressor = pedalboard.Compressor(
                threshold_db=params['comp_threshold_db'], ratio=params['comp_ratio'],
                attack_ms=params['comp_attack_ms'], release_ms=params['comp_release_ms']
            )
            if audio_float.ndim == 2 and audio_float.shape[0] < audio_float.shape[1]: # Detect [channels, samples]
                 result = compressor(audio_float.T, sample_rate=sample_rate).T
            else: # Assume [samples] or [samples, channels]
                 result = compressor(audio_float, sample_rate=sample_rate)
            
            if params['comp_makeup_gain_db'] != 0:
                 gain = pedalboard.Gain(gain_db=params['comp_makeup_gain_db'])
                 if result.ndim == 2: result = gain(result.T, sample_rate=sample_rate).T
                 else: result = gain(result, sample_rate=sample_rate)
            return result
        else:
            # Multiband logic
            try:
                sos_lm = self._design_linkwitz_riley_crossover(params['mb_crossover_low_mid_hz'], sample_rate, params['mb_crossover_order'])
                sos_mh = self._design_linkwitz_riley_crossover(params['mb_crossover_mid_high_hz'], sample_rate, params['mb_crossover_order'])
            except ValueError: return processed

            bands = {'low': signal.sosfiltfilt(sos_lm[0], processed, axis=-1), 
                     'high': signal.sosfiltfilt(sos_mh[1], processed, axis=-1)}
            temp_mid = signal.sosfiltfilt(sos_lm[1], processed, axis=-1)
            bands['mid'] = signal.sosfiltfilt(sos_mh[0], temp_mid, axis=-1)
            
            accumulator = np.zeros_like(processed, dtype=np.float32)
            
            # Compress Low
            comp_l = pedalboard.Compressor(threshold_db=params['mb_low_threshold_db'], ratio=params['mb_low_ratio'], attack_ms=params['mb_low_attack_ms'], release_ms=params['mb_low_release_ms'])
            band_l = bands['low'].astype(np.float32)
            res_l = comp_l(band_l.T, sample_rate=sample_rate).T if band_l.ndim==2 else comp_l(band_l, sample_rate=sample_rate)
            if params['mb_low_makeup_gain_db'] != 0:
                 gain_l = pedalboard.Gain(gain_db=params['mb_low_makeup_gain_db'])
                 res_l = gain_l(res_l.T, sample_rate=sample_rate).T if res_l.ndim==2 else gain_l(res_l, sample_rate=sample_rate)
            accumulator += res_l

            # Compress Mid
            comp_m = pedalboard.Compressor(threshold_db=params['mb_mid_threshold_db'], ratio=params['mb_mid_ratio'], attack_ms=params['mb_mid_attack_ms'], release_ms=params['mb_mid_release_ms'])
            band_m = bands['mid'].astype(np.float32)
            res_m = comp_m(band_m.T, sample_rate=sample_rate).T if band_m.ndim==2 else comp_m(band_m, sample_rate=sample_rate)
            if params['mb_mid_makeup_gain_db'] != 0:
                 gain_m = pedalboard.Gain(gain_db=params['mb_mid_makeup_gain_db'])
                 res_m = gain_m(res_m.T, sample_rate=sample_rate).T if res_m.ndim==2 else gain_m(res_m, sample_rate=sample_rate)
            accumulator += res_m

            # Compress High
            comp_h = pedalboard.Compressor(threshold_db=params['mb_high_threshold_db'], ratio=params['mb_high_ratio'], attack_ms=params['mb_high_attack_ms'], release_ms=params['mb_high_release_ms'])
            band_h = bands['high'].astype(np.float32)
            res_h = comp_h(band_h.T, sample_rate=sample_rate).T if band_h.ndim==2 else comp_h(band_h, sample_rate=sample_rate)
            if params['mb_high_makeup_gain_db'] != 0:
                 gain_h = pedalboard.Gain(gain_db=params['mb_high_makeup_gain_db'])
                 res_h = gain_h(res_h.T, sample_rate=sample_rate).T if res_h.ndim==2 else gain_h(res_h, sample_rate=sample_rate)
            accumulator += res_h
            
            return np.clip(np.nan_to_num(accumulator), -1.0, 1.0)

    def process_limiting(self, processed, sample_rate, **params):
        """Executes Limiting logic."""
        if not params.get('enable_limiter', False): return processed
        audio_float = processed.astype(np.float32)
        try:
            limiter = pedalboard.Limiter(threshold_db=params['limiter_ceiling_db'], release_ms=params['limiter_release_ms'])
            if audio_float.ndim == 2: return np.clip(limiter(audio_float.T, sample_rate=sample_rate).T, -1.0, 1.0)
            else: return np.clip(limiter(audio_float, sample_rate=sample_rate), -1.0, 1.0)
        except: return np.clip(audio_float, -1.0, 1.0)


    def _unpack_audio(self, audio_dict):
        if not isinstance(audio_dict, dict) or 'waveform' not in audio_dict:
            raise ValueError("Invalid audio input")
        audio_np = audio_dict['waveform'].cpu().float().numpy()
        if audio_np.ndim == 3: audio_np = audio_np[0]
        if audio_np.ndim == 1: audio_np = audio_np[np.newaxis, :]
        return audio_np, audio_dict['waveform'].device


# =================================================================================
# == Modular Nodes                                                               ==
# =================================================================================

class MasteringGainNode(MasteringBase):
    """Standalone Gain Node"""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "Input audio."}),
                "gain_db": ("FLOAT", {"default": 0.0, "min": -60.0, "max": 20.0, "step": 0.1, "tooltip": "GAIN (dB): Amplitude adjustment."}),
            }
        }
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "apply_gain"
    CATEGORY = "MD_Nodes/Audio Processing"

    def apply_gain(self, audio, gain_db):
        try:
            audio_np, device = self._unpack_audio(audio)
            processed = self._apply_gain(audio_np, gain_db)
            out_tensor = torch.from_numpy(processed).to(device).float().unsqueeze(0)
            return ({"waveform": out_tensor, "sample_rate": audio.get("sample_rate", 44100)},)
        except Exception as e:
            logger.error(f"Gain failed: {e}")
            return (audio,)

class MasteringEQNode(MasteringBase):
    """Standalone EQ/Filter Node"""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "Input Audio."}),
                "sample_rate": ("INT", {"default": 44100, "tooltip": "Audio sample rate (Hz)."}),
                "enable_lowpass": ("BOOLEAN", {"default": False, "tooltip": "Enable low-pass filter."}),
                "lowpass_freq": ("FLOAT", {"default": 18000.0, "tooltip": "Low-pass cutoff frequency (Hz)."}),
                "lowpass_order": ("INT", {"default": 4, "tooltip": "Low-pass filter order."}),
                "enable_highpass": ("BOOLEAN", {"default": False, "tooltip": "Enable high-pass filter."}),
                "highpass_freq": ("FLOAT", {"default": 20.0, "tooltip": "High-pass cutoff frequency (Hz)."}),
                "highpass_order": ("INT", {"default": 4, "tooltip": "High-pass filter order."}),
                "eq_high_shelf_gain_db": ("FLOAT", {"default": 0.0, "tooltip": "High shelf gain (dB)."}),
                "eq_high_shelf_freq": ("FLOAT", {"default": 12000.0, "tooltip": "High shelf frequency (Hz)."}),
                "enable_low_shelf_eq": ("BOOLEAN", {"default": False, "tooltip": "Enable low shelf filter."}),
                "eq_low_shelf_gain_db": ("FLOAT", {"default": 0.0, "tooltip": "Low shelf gain (dB)."}),
                "eq_low_shelf_freq": ("FLOAT", {"default": 75.0, "tooltip": "Low shelf frequency (Hz)."}),
                "enable_param_eq1": ("BOOLEAN", {"default": False, "tooltip": "Enable EQ Band 1."}),
                "param_eq1_gain_db": ("FLOAT", {"default": 0.0, "tooltip": "EQ1 Gain (dB)."}),
                "param_eq1_freq": ("FLOAT", {"default": 55.0, "tooltip": "EQ1 Frequency (Hz)."}),
                "param_eq1_q": ("FLOAT", {"default": 2.0, "tooltip": "EQ1 Q-Factor."}),
                "enable_param_eq2": ("BOOLEAN", {"default": False, "tooltip": "Enable EQ Band 2."}),
                "param_eq2_gain_db": ("FLOAT", {"default": 0.0, "tooltip": "EQ2 Gain (dB)."}),
                "param_eq2_freq": ("FLOAT", {"default": 125.0, "tooltip": "EQ2 Frequency (Hz)."}),
                "param_eq2_q": ("FLOAT", {"default": 2.0, "tooltip": "EQ2 Q-Factor."}),
                "enable_param_eq3": ("BOOLEAN", {"default": False, "tooltip": "Enable EQ Band 3."}),
                "param_eq3_gain_db": ("FLOAT", {"default": 0.0, "tooltip": "EQ3 Gain (dB)."}),
                "param_eq3_freq": ("FLOAT", {"default": 1250.0, "tooltip": "EQ3 Frequency (Hz)."}),
                "param_eq3_q": ("FLOAT", {"default": 1.0, "tooltip": "EQ3 Q-Factor."}),
                "enable_param_eq4": ("BOOLEAN", {"default": False, "tooltip": "Enable EQ Band 4."}),
                "param_eq4_gain_db": ("FLOAT", {"default": 0.0, "tooltip": "EQ4 Gain (dB)."}),
                "param_eq4_freq": ("FLOAT", {"default": 5000.0, "tooltip": "EQ4 Frequency (Hz)."}),
                "param_eq4_q": ("FLOAT", {"default": 1.0, "tooltip": "EQ4 Q-Factor."}),
            }
        }
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "apply_eq"
    CATEGORY = "MD_Nodes/Audio Processing"

    def apply_eq(self, audio, sample_rate, **kwargs):
        try:
            audio_np, device = self._unpack_audio(audio)
            processed = self.process_eq(audio_np, sample_rate, **kwargs)
            out_tensor = torch.from_numpy(processed).to(device).float().unsqueeze(0)
            return ({"waveform": out_tensor, "sample_rate": sample_rate},)
        except Exception as e:
            logger.error(f"EQ failed: {e}")
            return (audio,)

class MasteringCompressorNode(MasteringBase):
    """Standalone Compressor Node"""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "Input Audio."}),
                "sample_rate": ("INT", {"default": 44100, "tooltip": "Sample rate (Hz)."}),
                "enable_comp": ("BOOLEAN", {"default": True, "tooltip": "Enable compressor."}),
                "comp_type": (["Single-Band", "Multiband"], {"default": "Multiband", "tooltip": "Compressor type."}),
                "comp_threshold_db": ("FLOAT", {"default": -8.0, "tooltip": "Single-band Threshold (dB)."}),
                "comp_ratio": ("FLOAT", {"default": 2.5, "tooltip": "Single-band Ratio."}),
                "comp_attack_ms": ("FLOAT", {"default": 20.0, "tooltip": "Single-band Attack (ms)."}),
                "comp_release_ms": ("FLOAT", {"default": 250.0, "tooltip": "Single-band Release (ms)."}),
                "comp_makeup_gain_db": ("FLOAT", {"default": 0.0, "tooltip": "Single-band Makeup Gain (dB)."}),
                "mb_crossover_low_mid_hz": ("FLOAT", {"default": 250.0, "tooltip": "Low/Mid Crossover (Hz)."}),
                "mb_crossover_mid_high_hz": ("FLOAT", {"default": 4000.0, "tooltip": "Mid/High Crossover (Hz)."}),
                "mb_crossover_order": ("INT", {"default": 8, "tooltip": "Crossover order."}),
                "mb_low_threshold_db": ("FLOAT", {"default": -10.0, "tooltip": "Low Band Threshold (dB)."}),
                "mb_low_ratio": ("FLOAT", {"default": 3.0, "tooltip": "Low Band Ratio."}),
                "mb_low_attack_ms": ("FLOAT", {"default": 30.0, "tooltip": "Low Band Attack (ms)."}),
                "mb_low_release_ms": ("FLOAT", {"default": 300.0, "tooltip": "Low Band Release (ms)."}),
                "mb_low_makeup_gain_db": ("FLOAT", {"default": 0.0, "tooltip": "Low Band Makeup (dB)."}),
                "mb_mid_threshold_db": ("FLOAT", {"default": -8.0, "tooltip": "Mid Band Threshold (dB)."}),
                "mb_mid_ratio": ("FLOAT", {"default": 2.5, "tooltip": "Mid Band Ratio."}),
                "mb_mid_attack_ms": ("FLOAT", {"default": 20.0, "tooltip": "Mid Band Attack (ms)."}),
                "mb_mid_release_ms": ("FLOAT", {"default": 180.0, "tooltip": "Mid Band Release (ms)."}),
                "mb_mid_makeup_gain_db": ("FLOAT", {"default": 0.0, "tooltip": "Mid Band Makeup (dB)."}),
                "mb_high_threshold_db": ("FLOAT", {"default": -6.0, "tooltip": "High Band Threshold (dB)."}),
                "mb_high_ratio": ("FLOAT", {"default": 2.0, "tooltip": "High Band Ratio."}),
                "mb_high_attack_ms": ("FLOAT", {"default": 10.0, "tooltip": "High Band Attack (ms)."}),
                "mb_high_release_ms": ("FLOAT", {"default": 120.0, "tooltip": "High Band Release (ms)."}),
                "mb_high_makeup_gain_db": ("FLOAT", {"default": 0.0, "tooltip": "High Band Makeup (dB)."}),
            }
        }
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "apply_comp"
    CATEGORY = "MD_Nodes/Audio Processing"

    def apply_comp(self, audio, sample_rate, **kwargs):
        try:
            audio_np, device = self._unpack_audio(audio)
            processed = self.process_compression(audio_np, sample_rate, **kwargs)
            out_tensor = torch.from_numpy(processed).to(device).float().unsqueeze(0)
            return ({"waveform": out_tensor, "sample_rate": sample_rate},)
        except Exception as e:
            logger.error(f"Compressor failed: {e}")
            return (audio,)

class MasteringLimiterNode(MasteringBase):
    """Standalone Limiter Node"""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "Input Audio."}),
                "sample_rate": ("INT", {"default": 44100, "tooltip": "Sample rate (Hz)."}),
                "enable_limiter": ("BOOLEAN", {"default": True, "tooltip": "Enable limiter."}),
                "limiter_ceiling_db": ("FLOAT", {"default": -0.1, "tooltip": "Ceiling (dB)."}),
                "limiter_release_ms": ("FLOAT", {"default": 50.0, "tooltip": "Release (ms)."}),
            }
        }
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "apply_limit"
    CATEGORY = "MD_Nodes/Audio Processing"

    def apply_limit(self, audio, sample_rate, **kwargs):
        try:
            audio_np, device = self._unpack_audio(audio)
            processed = self.process_limiting(audio_np, sample_rate, **kwargs)
            out_tensor = torch.from_numpy(processed).to(device).float().unsqueeze(0)
            return ({"waveform": out_tensor, "sample_rate": sample_rate},)
        except Exception as e:
            logger.error(f"Limiter failed: {e}")
            return (audio,)


# =================================================================================
# == Original All-in-One Chain Node (Preserved & Refactored)                     ==
# =================================================================================

class MasteringChainNode(MasteringBase):
    """
    Advanced ComfyUI audio mastering chain node (All-in-One).
    Uses the same logic as modular nodes via MasteringBase.
    """
    @classmethod
    def INPUT_TYPES(cls):
        # Retaining original INPUT_TYPES for backward compatibility
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "Input audio data."}),
                "sample_rate": ("INT", {"default": 44100, "min": 8000, "max": 192000}),
                "master_gain_db": ("FLOAT", {"default": 0.0, "min": -20.0, "max": 20.0}),
                "enable_eq": ("BOOLEAN", {"default": True}),
                "enable_lowpass": ("BOOLEAN", {"default": False}),
                "lowpass_freq": ("FLOAT", {"default": 18000.0}),
                "lowpass_order": ("INT", {"default": 4}),
                "enable_highpass": ("BOOLEAN", {"default": False}),
                "highpass_freq": ("FLOAT", {"default": 20.0}),
                "highpass_order": ("INT", {"default": 4}),
                "eq_high_shelf_gain_db": ("FLOAT", {"default": 0.0}),
                "eq_high_shelf_freq": ("FLOAT", {"default": 12000.0}),
                "enable_low_shelf_eq": ("BOOLEAN", {"default": False}),
                "eq_low_shelf_gain_db": ("FLOAT", {"default": 0.0}),
                "eq_low_shelf_freq": ("FLOAT", {"default": 75.0}),
                "enable_param_eq1": ("BOOLEAN", {"default": False}),
                "param_eq1_gain_db": ("FLOAT", {"default": 0.0}),
                "param_eq1_freq": ("FLOAT", {"default": 55.0}),
                "param_eq1_q": ("FLOAT", {"default": 2.0}),
                "enable_param_eq2": ("BOOLEAN", {"default": False}),
                "param_eq2_gain_db": ("FLOAT", {"default": 0.0}),
                "param_eq2_freq": ("FLOAT", {"default": 125.0}),
                "param_eq2_q": ("FLOAT", {"default": 2.0}),
                "enable_param_eq3": ("BOOLEAN", {"default": False}),
                "param_eq3_gain_db": ("FLOAT", {"default": 0.0}),
                "param_eq3_freq": ("FLOAT", {"default": 1250.0}),
                "param_eq3_q": ("FLOAT", {"default": 1.0}),
                "enable_param_eq4": ("BOOLEAN", {"default": False}),
                "param_eq4_gain_db": ("FLOAT", {"default": 0.0}),
                "param_eq4_freq": ("FLOAT", {"default": 5000.0}),
                "param_eq4_q": ("FLOAT", {"default": 1.0}),
                "enable_comp": ("BOOLEAN", {"default": False}),
                "comp_type": (["Single-Band", "Multiband"], {"default": "Multiband"}),
                "comp_threshold_db": ("FLOAT", {"default": -8.0}),
                "comp_ratio": ("FLOAT", {"default": 2.5}),
                "comp_attack_ms": ("FLOAT", {"default": 20.0}),
                "comp_release_ms": ("FLOAT", {"default": 250.0}),
                "comp_makeup_gain_db": ("FLOAT", {"default": 0.0}),
                "mb_crossover_low_mid_hz": ("FLOAT", {"default": 250.0}),
                "mb_crossover_mid_high_hz": ("FLOAT", {"default": 4000.0}),
                "mb_crossover_order": ("INT", {"default": 8}),
                "mb_low_threshold_db": ("FLOAT", {"default": -10.0}),
                "mb_low_ratio": ("FLOAT", {"default": 3.0}),
                "mb_low_attack_ms": ("FLOAT", {"default": 30.0}),
                "mb_low_release_ms": ("FLOAT", {"default": 300.0}),
                "mb_low_makeup_gain_db": ("FLOAT", {"default": 0.0}),
                "mb_mid_threshold_db": ("FLOAT", {"default": -8.0}),
                "mb_mid_ratio": ("FLOAT", {"default": 2.5}),
                "mb_mid_attack_ms": ("FLOAT", {"default": 20.0}),
                "mb_mid_release_ms": ("FLOAT", {"default": 180.0}),
                "mb_mid_makeup_gain_db": ("FLOAT", {"default": 0.0}),
                "mb_high_threshold_db": ("FLOAT", {"default": -6.0}),
                "mb_high_ratio": ("FLOAT", {"default": 2.0}),
                "mb_high_attack_ms": ("FLOAT", {"default": 10.0}),
                "mb_high_release_ms": ("FLOAT", {"default": 120.0}),
                "mb_high_makeup_gain_db": ("FLOAT", {"default": 0.0}),
                "enable_limiter": ("BOOLEAN", {"default": False}),
                "limiter_ceiling_db": ("FLOAT", {"default": -0.1}),
                "limiter_release_ms": ("FLOAT", {"default": 50.0}),
            }
        }

    RETURN_TYPES = ("AUDIO", "IMAGE", "IMAGE")
    RETURN_NAMES = ("audio", "waveform_before", "waveform_after")
    FUNCTION = "apply_mastering_chain"
    CATEGORY = "MD_Nodes/Audio Processing"

    def apply_mastering_chain(self, audio, sample_rate, master_gain_db, enable_eq, **kwargs):
        try:
            logger.info(f"MASTERING CHAIN START | Sample Rate: {sample_rate} Hz")
            
            audio_np, device = self._unpack_audio(audio)
            if audio_np.size == 0 or audio_np.shape[-1] == 0:
                return (audio, torch.zeros(BLANK_PLOT_SIZE, dtype=torch.float32), torch.zeros(BLANK_PLOT_SIZE, dtype=torch.float32))

            before_plot = self._plot_waveform_to_tensor(audio_np, sample_rate, "Original Waveform")

            # 1. Gain
            processed = self._apply_gain(audio_np, master_gain_db)
            processed = np.clip(processed, -1.0, 1.0)

            # 2. EQ
            if enable_eq:
                processed = self.process_eq(processed, sample_rate, **kwargs)

            # 3. Compression
            processed = self.process_compression(processed, sample_rate, **kwargs)

            # 4. Limiting
            processed = self.process_limiting(processed, sample_rate, **kwargs)

            # Finalize
            after_plot = self._plot_waveform_to_tensor(processed, sample_rate, "Processed Waveform")
            
            out_tensor = torch.from_numpy(processed).to(device).float().unsqueeze(0)
            return ({"waveform": out_tensor, "sample_rate": sample_rate}, before_plot, after_plot)

        except Exception as e:
            logger.error(f"MasteringChainNode failed: {e}", exc_info=True)
            return (audio, torch.zeros(BLANK_PLOT_SIZE, dtype=torch.float32), torch.zeros(BLANK_PLOT_SIZE, dtype=torch.float32))

# =================================================================================
# == Node Registration                                                           ==
# =================================================================================

NODE_CLASS_MAPPINGS = {
    "MasteringChainNode": MasteringChainNode,
    "MD_Mastering_Gain": MasteringGainNode,
    "MD_Mastering_EQ": MasteringEQNode,
    "MD_Mastering_Compressor": MasteringCompressorNode,
    "MD_Mastering_Limiter": MasteringLimiterNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MasteringChainNode": "MD: Mastering Chain (Full)",
    "MD_Mastering_Gain": "MD: Mastering Gain",
    "MD_Mastering_EQ": "MD: Mastering EQ & Filters",
    "MD_Mastering_Compressor": "MD: Mastering Compressor",
    "MD_Mastering_Limiter": "MD: Mastering Limiter"
}