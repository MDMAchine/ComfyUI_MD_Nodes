# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
# ████ MD_Nodes/AudioAutoMasterPro – Iterative Mastering Chain v6.9.2 ████▓▒░
# © 2025 MDMAchine
# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
# ░▒▓ ORIGIN & DEV:
#   • Cast into the void by: MDMAchine
#   • Enhanced by: Gemini, Claude
#   • License: Apache 2.0 — Sharing is caring

# ░▒▓ DESCRIPTION:
#   An all-in-one, iterative audio mastering node. It automatically analyzes
#   and processes audio using a chain of filtering, iterative spectral EQ,
#   de-essing, 3-band compression, stereo widening, and limiting.

# ░▒▓ FEATURES:
#   ✓ Hits a user-defined target_lufs (via Pyloudnorm).
#   ✓ Iterative spectral EQ to tame harsh highs and muddy lows.
#   ✓ 3-band multiband compressor with Linkwitz-Riley crossovers.
#   ✓ Built-in de-esser, stereo widener, and brickwall limiter.
#   ✓ 5+ mastering presets (Standard, Podcast, Aggressive, etc.).
#   ✓ Returns detailed text log and before/after waveform images.

# ░▒▓ CHANGELOG:
#   - v6.9.2 (Final Polish - Nov 2025):
#       • COMPLIANCE: Removed version number from runtime logs.
#       • UX: Added comprehensive tooltips to all optional parameters.
#       • REFACTOR: Extracted magic numbers to constants.
#   - v6.9.1 (Critical Fix):
#       • FIXED: Normalized SOS filter coefficients (a0=1.0).
#       • FIXED: Error handler returns blank images instead of None.

# ░▒▓ CONFIGURATION:
#   → Primary Use: One-click "Standard" or "Podcast" profile mastering to -14 LUFS.
#   → Edge Use: Disabling all modules to use as a "smart" normalizer.

# ░▒▓ WARNING:
#   This node may trigger:
#   ▓▒░ A sudden, god-like complex after turning a muddy .wav into a -14 LUFS banger.
#   ▓▒░ Compulsively checking the analysis log, whispering "just one more iteration."
#   Consult your nearest demoscene vet if hallucinations persist.
# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄


# =================================================================================
# == Standard Library Imports                                                     ==
# =================================================================================
import io
import logging
import traceback

# =================================================================================
# == Third-Party Imports                                                          ==
# =================================================================================
import torch
import numpy as np
import pyloudnorm as pln
import librosa
from scipy import signal
from scipy.signal import lfilter, sosfilt
from pedalboard import Pedalboard, Compressor, Limiter, Gain
import matplotlib as mpl
# CRITICAL: Set non-interactive backend BEFORE importing pyplot
mpl.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

# =================================================================================
# == Constants                                                                    ==
# =================================================================================

# Magic Numbers
MAX_SAMPLES_PLOT = 150000
LIMITER_RELEASE_MS = 5.0
CLIPPING_THRESHOLD = 0.99

MASTERING_PROFILES = {
    "Standard": {
        "hp": 30, "lp": 0, "eq": True, "bass": 1.5, "high": 0.4, "adapt": True,
        "deess": True, "deess_db": -10.0, "mbc": True,
        "x_low": 300, "x_high": 3000, "x_order": 8,
        "mbc_L": (-18.0, 2.0), "mbc_M": (-18.0, 2.0), "mbc_H": (-18.0, 2.0),
        "lim": True, "lim_db": -0.1, "width": 1.0
    },
    "Aggressive": {
        "hp": 40, "lp": 0, "eq": True, "bass": 1.2, "high": 0.3, "adapt": True,
        "deess": True, "deess_db": -12.0, "mbc": True,
        "x_low": 250, "x_high": 2500, "x_order": 10,
        "mbc_L": (-20.0, 3.0), "mbc_M": (-20.0, 3.0), "mbc_H": (-19.0, 2.5),
        "lim": True, "lim_db": -0.1, "width": 1.1
    },
    "Podcast (Clarity)": {
        "hp": 80, "lp": 16000, "eq": True, "bass": 2.0, "high": 0.5, "adapt": True,
        "deess": True, "deess_db": -15.0, "mbc": True,
        "x_low": 400, "x_high": 3500, "x_order": 6,
        "mbc_L": (-16.0, 2.5), "mbc_M": (-18.0, 3.0), "mbc_H": (-18.0, 2.0),
        "lim": True, "lim_db": -1.0, "width": 0.8
    },
    "Gentle (Tame)": {
        "hp": 20, "lp": 0, "eq": True, "bass": 2.0, "high": 0.6, "adapt": True,
        "deess": False, "deess_db": 0.0, "mbc": True,
        "x_low": 300, "x_high": 3000, "x_order": 8,
        "mbc_L": (-16.0, 1.8), "mbc_M": (-17.0, 1.8), "mbc_H": (-17.0, 1.5),
        "lim": True, "lim_db": -0.5, "width": 1.0
    },
    "Mastering (Transparent)": {
        "hp": 20, "lp": 0, "eq": True, "bass": 1.8, "high": 0.5, "adapt": True,
        "deess": True, "deess_db": -12.0, "mbc": True,
        "x_low": 200, "x_high": 4000, "x_order": 8,
        "mbc_L": (-15.0, 1.5), "mbc_M": (-16.0, 1.5), "mbc_H": (-16.0, 1.4),
        "lim": True, "lim_db": -0.3, "width": 1.0
    }
}

# =================================================================================
# == Core Node Class                                                              ==
# =================================================================================

class MD_AutoMasterNode:
    """
    Audio Auto Master Pro (MD_AutoMasterNode)
    An all-in-one, iterative audio mastering node.
    """
    
    def __init__(self):
        self.analysis_log = []

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "AUDIO INPUT\n- The audio data to process."}),
                "target_lufs": ("FLOAT", {
                    "default": -14.0, "min": -24.0, "max": -6.0, "step": 0.1,
                    "tooltip": "TARGET LOUDNESS (LUFS)\n- -14.0 LUFS is standard for streaming."
                }),
                "profile": (["Custom"] + list(MASTERING_PROFILES.keys()), {
                    "default": "Standard",
                    "tooltip": "MASTERING PROFILE\n- Select a preset to auto-configure all parameters."
                }),
            },
            "optional": {
                "input_gain_db": ("FLOAT", {
                    "default": 0.0, "min": -12.0, "max": 12.0, "step": 0.1, 
                    "tooltip": "INPUT GAIN (dB)\n- Pre-processing gain adjustment."
                }),
                "highpass_freq": ("FLOAT", {
                    "default": 0, "min": 0, "max": 500, "step": 1, 
                    "tooltip": "HIGH-PASS FILTER (Hz)\n- Cuts sub-bass rumble.\n- 30-40Hz is typical."
                }),
                "lowpass_freq": ("FLOAT", {
                    "default": 0, "min": 0, "max": 20000, "step": 100, 
                    "tooltip": "LOW-PASS FILTER (Hz)\n- Cuts ultra-high hiss.\n- 18kHz+ is typical."
                }),
                "do_eq": ("BOOLEAN", {"default": True, "label_on": "Run Iterative EQ", "label_off": "Skip EQ"}),
                "eq_bass_target": ("FLOAT", {
                    "default": 1.5, "min": 0.0, "max": 30.0, "step": 0.1, 
                    "tooltip": "EQ BASS TARGET\n- Lower = Tighter bass.\n- Higher = Fuller bass."
                }),
                "eq_high_target": ("FLOAT", {
                    "default": 0.4, "min": 0.0, "max": 30.0, "step": 0.01, 
                    "tooltip": "EQ HIGH TARGET\n- Lower = Darker.\n- Higher = Brighter/Airy."
                }),
                "eq_adaptive": ("BOOLEAN", {"default": True, "tooltip": "ADAPTIVE EQ\n- Dynamically scales EQ cuts based on analysis."}),
                "do_deess": ("BOOLEAN", {"default": True, "label_on": "Run De-Esser", "label_off": "Skip De-Esser"}),
                "deess_amount_db": ("FLOAT", {"default": -10.0, "min": -40.0, "max": 0.0, "step": 0.5}),
                "do_mbc": ("BOOLEAN", {"default": True, "label_on": "Run Single-Pass MBC", "label_off": "Skip MBC"}),
                
                # MBC Params - Added Tooltips
                "mbc_crossover_low": ("FLOAT", {
                    "default": 300, "min": 100, "max": 1000, "step": 10,
                    "tooltip": "MBC CROSSOVER LOW (Hz)\n- Split point between Low and Mid bands."
                }),
                "mbc_crossover_high": ("FLOAT", {
                    "default": 3000, "min": 1000, "max": 8000, "step": 100,
                    "tooltip": "MBC CROSSOVER HIGH (Hz)\n- Split point between Mid and High bands."
                }),
                "mbc_crossover_order": ("INT", {
                    "default": 8, "min": 4, "max": 12, "step": 2,
                    "tooltip": "MBC CROSSOVER ORDER\n- Filter steepness. Higher = sharper separation."
                }),
                "mbc_low_thresh_db": ("FLOAT", {"default": -18.0, "min": -60.0, "max": 0.0, "step": 0.5, "tooltip": "Low Band Threshold"}),
                "mbc_low_ratio": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 20.0, "step": 0.1, "tooltip": "Low Band Ratio"}),
                "mbc_mid_thresh_db": ("FLOAT", {"default": -18.0, "min": -60.0, "max": 0.0, "step": 0.5, "tooltip": "Mid Band Threshold"}),
                "mbc_mid_ratio": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 20.0, "step": 0.1, "tooltip": "Mid Band Ratio"}),
                "mbc_high_thresh_db": ("FLOAT", {"default": -18.0, "min": -60.0, "max": 0.0, "step": 0.5, "tooltip": "High Band Threshold"}),
                "mbc_high_ratio": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 20.0, "step": 0.1, "tooltip": "High Band Ratio"}),
                
                "do_limiter": ("BOOLEAN", {"default": True, "label_on": "Run Final Limiter", "label_off": "Skip Limiter"}),
                "limiter_threshold_db": ("FLOAT", {
                    "default": -0.1, "min": -3.0, "max": 0.0, "step": 0.1,
                    "tooltip": "LIMITER CEILING (dB)\n- Final brickwall limit."
                }),
                "stereo_width": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "STEREO WIDTH\n- 1.0 = Normal, <1.0 = Narrower, >1.0 = Wider."
                }),
                "max_iterations_eq": ("INT", {
                    "default": 5, "min": 1, "max": 20,
                    "tooltip": "EQ ITERATIONS\n- Max passes for the auto-EQ loop."
                }),
                "fast_mode": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "FAST MODE\n- Skips intermediate normalization steps."
                }),
                "mix": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "DRY/WET MIX\n- Blend original signal (0.0) with mastered signal (1.0)."
                }),
            }
        }

    RETURN_TYPES = ("AUDIO", "STRING", "IMAGE", "IMAGE")
    RETURN_NAMES = ("audio", "analysis_details", "waveform_before", "waveform_after")
    FUNCTION = "master_audio"
    CATEGORY = "MD_Nodes/Audio"

    # --- Helper Methods ---

    def _normalize(self, audio, sr, meter_obj, target):
        loudness = meter_obj.integrated_loudness(audio)
        if loudness == -float('inf'):
            self.analysis_log.append(f"⚠️ Skipped normalization (silence)")
            return audio, -float('inf')
        normalized_audio = pln.normalize.loudness(audio, loudness, target)
        self.analysis_log.append(f"🎚️ Normalized to {target} LUFS (was {loudness:.2f} LUFS)")
        return normalized_audio, loudness
    
    def _check_for_clipping(self, audio, stage_name):
        if isinstance(audio, torch.Tensor):
            audio_np = audio.cpu().numpy()
        else:
            audio_np = audio

        if isinstance(audio_np, np.ndarray) and audio_np.size > 0:
            clips = np.sum(np.abs(audio_np) > CLIPPING_THRESHOLD)
            peak = np.max(np.abs(audio_np))
            if clips > 0:
                self.analysis_log.append(f"⚠️ WARNING: {clips} samples clipped after {stage_name} (peak: {peak:.3f})")
        else:
            self.analysis_log.append(f"⚠️ Skipping clipping check after {stage_name}: Invalid audio data")


    def _analyze(self, audio, sr, meter_obj):
        analysis_channel = audio[:, 0] if audio.ndim > 1 else audio
        if analysis_channel.size == 0:
            return {"bass": np.nan, "high": np.nan}

        stft = np.abs(librosa.stft(analysis_channel))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=stft.shape[0]*2-2)
        
        bass_mask = freqs < 100
        high_mask = freqs > 10000
        
        bass_energy = np.mean(stft[bass_mask]) if np.any(bass_mask) else 0.0
        high_energy = np.mean(stft[high_mask]) if np.any(high_mask) else 0.0
        
        self.analysis_log.append(f"📊 Analysis: Bass={bass_energy:.2f}, Highs={high_energy:.2f}")
        return {"bass": bass_energy, "high": high_energy}


    def _apply_filters(self, audio, sr, highpass_freq, lowpass_freq):
        try:
            if highpass_freq > 0:
                if highpass_freq >= sr / 2:
                    self.analysis_log.append(f"⚠️ Highpass freq {highpass_freq}Hz >= Nyquist ({sr/2}Hz). Skipping.")
                else:
                    sos = signal.butter(4, highpass_freq, btype='highpass', fs=sr, output='sos')
                    audio = signal.sosfiltfilt(sos, audio, axis=0)
                    self.analysis_log.append(f"🔊 Applied highpass filter @ {highpass_freq} Hz")
            
            if lowpass_freq > 0 and lowpass_freq < sr / 2:
                sos = signal.butter(4, lowpass_freq, btype='lowpass', fs=sr, output='sos')
                audio = signal.sosfiltfilt(sos, audio, axis=0)
                self.analysis_log.append(f"🔉 Applied lowpass filter @ {lowpass_freq} Hz")
            elif lowpass_freq >= sr / 2:
                self.analysis_log.append(f"⚠️ Lowpass freq {lowpass_freq}Hz >= Nyquist ({sr/2}Hz). Skipping.")

        except ValueError as e:
            self.analysis_log.append(f"❌ Error applying filters: {e}")
            raise ValueError(f"Filter design failed: {e}")
        
        return audio

    def _design_shelf_sos(self, gain_db, freq, sample_rate, shelf_type='low'):
        """
        Designed as SOS (Second-Order Sections) with strict a0=1.0 normalization.
        """
        if abs(gain_db) < 0.001 or freq <= 0 or freq >= sample_rate / 2: return None
        try:
            A = 10**(gain_db / 40.0)
            w0 = 2 * np.pi * freq / sample_rate
            cos_w0 = np.cos(w0)
            sin_w0 = np.sin(w0)
            alpha = sin_w0 / 2.0 * np.sqrt(2)
            
            # RBJ Cookbook Formulas
            if shelf_type == 'low':
                b0 = A*((A+1)-(A-1)*cos_w0+2*np.sqrt(A)*alpha)
                b1 = 2*A*((A-1)-(A+1)*cos_w0)
                b2 = A*((A+1)-(A-1)*cos_w0-2*np.sqrt(A)*alpha)
                a0 = (A+1)+(A-1)*cos_w0+2*np.sqrt(A)*alpha
                a1 = -2*((A-1)+(A+1)*cos_w0)
                a2 = (A+1)+(A-1)*cos_w0-2*np.sqrt(A)*alpha
            elif shelf_type == 'high':
                b0 = A*((A+1)+(A-1)*cos_w0+2*np.sqrt(A)*alpha)
                b1 = -2*A*((A-1)+(A+1)*cos_w0)
                b2 = A*((A+1)+(A-1)*cos_w0-2*np.sqrt(A)*alpha)
                a0 = (A+1)-(A-1)*cos_w0+2*np.sqrt(A)*alpha
                a1 = 2*((A-1)-(A+1)*cos_w0)
                a2 = (A+1)-(A-1)*cos_w0-2*np.sqrt(A)*alpha
            else:
                return None

            # CRITICAL: Normalize coefficients so a0 = 1.0
            # Scipy requires the 4th element (index 3) to be exactly 1.
            norm_factor = a0
            b0 /= norm_factor
            b1 /= norm_factor
            b2 /= norm_factor
            a1 /= norm_factor
            a2 /= norm_factor
            # a0 becomes 1.0 (implicit in SOS division)

            # SOS format: [b0, b1, b2, 1.0, a1, a2]
            sos = np.array([[b0, b1, b2, 1.0, a1, a2]])
            return sos
        except Exception as e:
            self.analysis_log.append(f"⚠️ Shelf filter design failed: {e}")
            return None

    def _apply_eq(self, audio, sr, adjustments, adaptive=False):
        processed_audio = audio.copy()
        
        # Low Shelf
        if adjustments.get("bass_cut_db"):
            cut = adjustments["bass_cut_db"]
            if adaptive and "bass_scale" in adjustments: cut *= adjustments["bass_scale"]
            sos = self._design_shelf_sos(cut, 120, sr, 'low')
            if sos is not None:
                processed_audio = sosfilt(sos, processed_audio, axis=0)
                self.analysis_log.append(f"🎛️ Applied EQ: Low-shelf cut of {cut:.1f} dB @ 120 Hz")

        # High Shelf
        if adjustments.get("high_cut_db"):
            cut = adjustments["high_cut_db"]
            if adaptive and "high_scale" in adjustments: cut *= adjustments["high_scale"]
            sos = self._design_shelf_sos(cut, 8000, sr, 'high')
            if sos is not None:
                processed_audio = sosfilt(sos, processed_audio, axis=0)
                self.analysis_log.append(f"🎛️ Applied EQ: High-shelf cut of {cut:.1f} dB @ 8000 Hz")
            
        return processed_audio

    def _design_linkwitz_riley_crossover(self, cutoff_freq, sample_rate, order=8):
        if order % 2 != 0: order = order + 1
        nyquist = 0.5 * sample_rate
        normal_cutoff = np.clip(cutoff_freq / nyquist, 0.01, 0.99)
        try:
            sos_lp = signal.butter(order // 2, normal_cutoff, btype='lowpass', output='sos')
            sos_hp = signal.butter(order // 2, normal_cutoff, btype='highpass', output='sos')
            return [sos_lp, sos_lp], [sos_hp, sos_hp]
        except ValueError as e:
            raise

    def _apply_single_band_compression(self, audio_data, threshold_db, ratio, 
                                    attack_ms, release_ms, makeup_gain_db, sample_rate,
                                    log_details=False): 
        if log_details: self.analysis_log.append(f"     ↳ Band Comp: T={threshold_db:.1f}dB, R={ratio:.1f}:1")
        audio_float = audio_data.astype(np.float32)
        try:
            compressor = Compressor(threshold_db=threshold_db, ratio=ratio, attack_ms=attack_ms, release_ms=release_ms)
            if audio_float.ndim == 2: processed = compressor(audio_float.T, sample_rate=sample_rate).T
            else: processed = compressor(audio_float, sample_rate=sample_rate)
            if makeup_gain_db != 0.0: processed *= 10**(makeup_gain_db / 20.0)
            return processed
        except Exception as e:
            self.analysis_log.append(f"❌ Error in single-band compression: {e}")
            return audio_float


    def _apply_multiband_compression_manual(self, audio_data, sample_rate, 
                                            crossover_low_mid, crossover_mid_high,
                                            crossover_order, low_params, mid_params, high_params):
        self.analysis_log.append(f"🗜️ Applying Manual MBC (Single Pass)...") 
        is_stereo = audio_data.ndim > 1 and audio_data.shape[1] == 2
        if is_stereo: audio_ch = audio_data.T
        else: audio_ch = audio_data[np.newaxis, :]
        num_channels = audio_ch.shape[0]
        processed_audio_accumulator = np.zeros_like(audio_ch, dtype=np.float64) 
        
        try:
            sos_lm_lp_list, sos_lm_hp_list = self._design_linkwitz_riley_crossover(crossover_low_mid, sample_rate, crossover_order)
            sos_mh_lp_list, sos_mh_hp_list = self._design_linkwitz_riley_crossover(crossover_mid_high, sample_rate, crossover_order)
        except ValueError as e:
            self.analysis_log.append(f"❌ Critical Error: MBC Crossover design failed. {e}")
            return audio_data 
            
        for c in range(num_channels):
            ch_data = audio_ch[c, :].astype(np.float64) 
            try:
                # SOS Filtering (More stable than previous implementation)
                low_band = signal.sosfiltfilt(sos_lm_lp_list[0], ch_data); low_band = signal.sosfiltfilt(sos_lm_lp_list[1], low_band)
                high_band = signal.sosfiltfilt(sos_mh_hp_list[0], ch_data); high_band = signal.sosfiltfilt(sos_mh_hp_list[1], high_band)
                mid_band_tmp = signal.sosfiltfilt(sos_lm_hp_list[0], ch_data); mid_band_tmp = signal.sosfiltfilt(sos_lm_hp_list[1], mid_band_tmp)
                mid_band = signal.sosfiltfilt(sos_mh_lp_list[0], mid_band_tmp); mid_band = signal.sosfiltfilt(sos_mh_lp_list[1], mid_band)
                
                if not np.all(np.isfinite(low_band)) or not np.all(np.isfinite(mid_band)) or not np.all(np.isfinite(high_band)):
                    processed_audio_accumulator[c, :] = ch_data
                    continue

            except Exception:
                processed_audio_accumulator[c, :] = ch_data
                continue

            base_attack_release = [(10.0, 200.0), (10.0, 150.0), (5.0, 100.0)]
            
            comp_low = self._apply_single_band_compression(low_band, low_params['threshold_db'], low_params['ratio'], base_attack_release[0][0], base_attack_release[0][1], 0.0, sample_rate)
            comp_mid = self._apply_single_band_compression(mid_band, mid_params['threshold_db'], mid_params['ratio'], base_attack_release[1][0], base_attack_release[1][1], 0.0, sample_rate)
            comp_high = self._apply_single_band_compression(high_band, high_params['threshold_db'], high_params['ratio'], base_attack_release[2][0], base_attack_release[2][1], 0.0, sample_rate)
            
            summed_bands = comp_low + comp_mid + comp_high
            processed_audio_accumulator[c, :] = summed_bands

        if is_stereo: return processed_audio_accumulator.T.astype(np.float32)
        else: return processed_audio_accumulator[0, :].astype(np.float32)
            
            
    def _apply_deesser_manual(self, audio, sr, max_cut_db):
        self.analysis_log.append(f"✂️ Manual De-Esser: Applying cuts up to {max_cut_db:.1f} dB")
        processed_audio = audio.copy()
        deess_targets = [(5500, 2.5), (7500, 3.0), (9500, 3.5)]
        for freq, Q in deess_targets:
            gain_for_filter = max_cut_db 
            if freq >= sr / 2: continue
            try:
                # Peaking EQ cut for "De-essing" effect
                A = 10**(gain_for_filter / 40.0); w0 = 2 * np.pi * freq / sr; alpha = np.sin(w0) / (2 * Q)
                b0 = 1 - alpha * A; b1 = -2 * np.cos(w0); b2 = 1 + alpha * A
                a0 = 1 + alpha / A; a1 = -2 * np.cos(w0); a2 = 1 - alpha / A
                b = np.array([b0/a0, b1/a0, b2/a0]); a = np.array([1.0, a1/a0, a2/a0])
                
                processed_audio = lfilter(b, a, processed_audio, axis=0)
            except Exception:
                pass
        return processed_audio


    def _apply_stereo_width(self, audio, width):
        if audio.ndim < 2 or audio.shape[1] < 2: return audio
        if width == 1.0: return audio
        mid = (audio[:, 0] + audio[:, 1]) / 2.0; side = (audio[:, 0] - audio[:, 1]) / 2.0
        side *= width
        left = mid + side; right = mid - side
        processed = np.column_stack([left, right])
        if width > 1.0:
            max_peak = np.max(np.abs(processed))
            if max_peak > 1.0:
                processed = np.clip(processed, -1.0, 1.0)
        self.analysis_log.append(f"↔️ Applied stereo width: {width:.2f}")
        return processed


    def _plot_waveform_to_tensor(self, audio_data, sample_rate, title="Waveform", max_samples=MAX_SAMPLES_PLOT): 
        if audio_data is None or audio_data.size == 0:
            return torch.zeros((1, 64, 64, 3), dtype=torch.float32)

        fig = None
        try: 
            plt.style.use('dark_background'); fig, ax = plt.subplots(figsize=(10, 3)) 
            if audio_data.ndim == 2: plot_data = audio_data[:, 0]
            else: plot_data = audio_data
            
            if plot_data.size == 0:
                plt.close(fig); return torch.zeros((1, 64, 64, 3), dtype=torch.float32)

            num_samples_original = len(plot_data)
            if num_samples_original > max_samples:
                ds_factor = num_samples_original//max_samples; plot_data = plot_data[::ds_factor]
                time_axis = np.linspace(0, num_samples_original/sample_rate, len(plot_data))
            else: time_axis = np.linspace(0, num_samples_original/sample_rate, len(plot_data))
            
            if plot_data.size == 0:
                plt.close(fig); return torch.zeros((1, 64, 64, 3), dtype=torch.float32)

            ax.plot(time_axis, plot_data, color='#87CEEB', linewidth=0.5) 
            peak_val = np.max(np.abs(plot_data)); rms = np.sqrt(np.mean(plot_data**2))
            if peak_val > 0.8: 
                ax.axhline(y=peak_val, color='orangered', ls='--', lw=0.7, alpha=0.6, label=f'Peak: {peak_val:.3f}')
                ax.axhline(y=-peak_val, color='orangered', ls='--', lw=0.7, alpha=0.6)
            ax.axhline(y=rms, color='mediumseagreen', ls=':', lw=0.7, alpha=0.6, label=f'RMS: {rms:.3f}')
            ax.axhline(y=-rms, color='mediumseagreen', ls=':', lw=0.7, alpha=0.6)
            ax.set_title(f"{title} | Peak: {peak_val:.3f} | RMS: {rms:.3f}", fontsize=10)
            ax.set_xlabel("Time (s)", fontsize=8); ax.set_ylabel("Amplitude", fontsize=8)
            ax.set_xlim(0, num_samples_original/sample_rate); ax.set_ylim(-1.05, 1.05); ax.set_yticks([-1, -0.5, 0, 0.5, 1])
            ax.grid(True, ls=':', lw=0.5, alpha=0.3, color='gray'); ax.legend(loc='upper right', fontsize=7, framealpha=0.5)
            ax.tick_params(axis='both', which='major', labelsize=7); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            plt.tight_layout()
            
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=96, facecolor=fig.get_facecolor())
            buf.seek(0)
            img = Image.open(buf).convert("RGB")
            img_np = np.array(img).astype(np.float32) / 255.0
            
            buf.close()
            plt.close(fig)
            
            return torch.from_numpy(img_np).unsqueeze(0)
        
        except Exception as e:
            logging.error(f"[MD_AutoMasterNode] Plotting Error: {e}")
            if fig: plt.close(fig)
            return torch.zeros((1, 64, 64, 3), dtype=torch.float32)

    # --- Main Execution Function ---

    def master_audio(self, audio, target_lufs, profile, 
                      input_gain_db, highpass_freq, lowpass_freq,
                      do_eq, eq_bass_target, eq_high_target, eq_adaptive,
                      do_deess, deess_amount_db,
                      do_mbc, mbc_crossover_low, mbc_crossover_high, mbc_crossover_order, 
                      mbc_low_thresh_db, mbc_low_ratio, 
                      mbc_mid_thresh_db, mbc_mid_ratio,
                      mbc_high_thresh_db, mbc_high_ratio,
                      do_limiter, limiter_threshold_db,
                      stereo_width, max_iterations_eq, fast_mode, mix): 
        """
        Main execution function for iterative audio mastering.
        
        Returns:
            Tuple[dict, str, Tensor, Tensor]: Processed audio, log, before img, after img.
        """
        
        waveform_before = None
        waveform_after = None

        try:
            self.analysis_log = []; self.analysis_log.append(f"{'='*50}")
            self.analysis_log.append(f"🎧 MD AutoMaster - Processing Started")
            self.analysis_log.append(f"{'='*50}")
            
            waveform_tensor = audio['waveform']; sample_rate = audio['sample_rate']
            audio_data = waveform_tensor[0].T.cpu().numpy().astype(np.float32)
            
            if audio_data.shape[0] == 0: return (audio, "❌ Error: Empty audio input.", None, None)
            if sample_rate < 44100: return (audio, f"❌ Error: Sample rate {sample_rate}Hz < 44100 Hz.", None, None)

            meter = pln.Meter(sample_rate)
            channels = audio_data.shape[1] if audio_data.ndim > 1 else 1
            
            original_audio_data = audio_data.copy()
            waveform_before = self._plot_waveform_to_tensor(original_audio_data, sample_rate, "Original Waveform")

            # Apply Profile Settings (Refactored using Dictionary)
            if profile in MASTERING_PROFILES:
                self.analysis_log.append(f"🎛️ Loading Profile: '{profile}'")
                p = MASTERING_PROFILES[profile]
                highpass_freq, lowpass_freq = p["hp"], p["lp"]
                do_eq, eq_bass_target, eq_high_target, eq_adaptive = p["eq"], p["bass"], p["high"], p["adapt"]
                do_deess, deess_amount_db = p["deess"], p["deess_db"]
                do_mbc = p["mbc"]
                mbc_crossover_low, mbc_crossover_high, mbc_crossover_order = p["x_low"], p["x_high"], p["x_order"]
                mbc_low_thresh_db, mbc_low_ratio = p["mbc_L"]
                mbc_mid_thresh_db, mbc_mid_ratio = p["mbc_M"]
                mbc_high_thresh_db, mbc_high_ratio = p["mbc_H"]
                do_limiter, limiter_threshold_db = p["lim"], p["lim_db"]
                stereo_width = p["width"]
            
            # --- Processing Chain ---
            processed_audio = original_audio_data.copy()

            if input_gain_db != 0.0:
                self.analysis_log.append(f"📈 Input Gain: {input_gain_db:+.1f} dB")
                board = Pedalboard([Gain(gain_db=input_gain_db)])
                processed_audio = board(processed_audio, sample_rate=sample_rate)
                original_audio_data = processed_audio.copy() 

            if highpass_freq > 0 or lowpass_freq > 0:
                processed_audio = self._apply_filters(processed_audio, sample_rate, highpass_freq, lowpass_freq)

            processed_audio, _ = self._normalize(processed_audio, sample_rate, meter, target_lufs)
            
            if do_eq:
                current_analysis = self._analyze(processed_audio, sample_rate, meter)
                eq_iteration = 0; bass_ok = current_analysis["bass"] <= eq_bass_target; high_ok = current_analysis["high"] <= eq_high_target
                while (not bass_ok or not high_ok) and eq_iteration < max_iterations_eq:
                    eq_adjustments = {}
                    if not bass_ok:
                        if eq_adaptive: overage=np.clip((current_analysis["bass"]-eq_bass_target),0,None)/max(eq_bass_target,0.1); eq_adjustments["bass_cut_db"]=-2.0; eq_adjustments["bass_scale"]=min(overage,2.0)
                        else: eq_adjustments["bass_cut_db"] = -2.0
                    if not high_ok:
                        if eq_adaptive: overage=np.clip((current_analysis["high"]-eq_high_target),0,None)/max(eq_high_target,0.1); eq_adjustments["high_cut_db"]=-1.5; eq_adjustments["high_scale"]=min(overage,2.0)
                        else: eq_adjustments["high_cut_db"] = -1.5
                    
                    if not eq_adjustments: break 

                    processed_audio = self._apply_eq(processed_audio, sample_rate, eq_adjustments, eq_adaptive)
                    self._check_for_clipping(processed_audio, "EQ")
                    if not fast_mode: processed_audio, _ = self._normalize(processed_audio, sample_rate, meter, target_lufs)
                    current_analysis = self._analyze(processed_audio, sample_rate, meter)
                    bass_ok = current_analysis["bass"] <= eq_bass_target; high_ok = current_analysis["high"] <= eq_high_target
                    eq_iteration += 1
                self.analysis_log.append(f"✅ EQ completed after {eq_iteration} iteration(s)")
            else: self.analysis_log.append("⏭️ Skipped Iterative EQ")

            if do_deess: processed_audio = self._apply_deesser_manual(processed_audio, sample_rate, deess_amount_db) 
            else: self.analysis_log.append("⏭️ Skipped De-Esser")

            if stereo_width != 1.0: processed_audio = self._apply_stereo_width(processed_audio, stereo_width)

            if do_mbc:
                if not fast_mode: processed_audio, _ = self._normalize(processed_audio, sample_rate, meter, target_lufs)
                low_params = {'threshold_db': mbc_low_thresh_db, 'ratio': mbc_low_ratio}
                mid_params = {'threshold_db': mbc_mid_thresh_db, 'ratio': mbc_mid_ratio}
                high_params = {'threshold_db': mbc_high_thresh_db, 'ratio': mbc_high_ratio}
                processed_audio = self._apply_multiband_compression_manual(processed_audio, sample_rate, mbc_crossover_low, mbc_crossover_high, mbc_crossover_order, low_params, mid_params, high_params)
                self._check_for_clipping(processed_audio, "Manual MBC")
                self.analysis_log.append(f"✅ Manual MBC applied (Single Pass)") 
            else: self.analysis_log.append("⏭️ Skipped MBC")

            needs_final_processing = fast_mode or mix < 1.0
            if not fast_mode and mix == 1.0:
                if do_limiter:
                    limiter = Limiter(threshold_db=limiter_threshold_db, release_ms=LIMITER_RELEASE_MS)
                    board = Pedalboard([limiter])
                    processed_audio = board(processed_audio.astype(np.float32), sample_rate=sample_rate)
                else: self.analysis_log.append("⏭️ Skipped Final Limiter")

            if mix < 1.0:
                self.analysis_log.append(f"🧪 Applying {mix*100:.0f}% wet mix")
                L = min(processed_audio.shape[0], original_audio_data.shape[0])
                processed_audio = (processed_audio[:L] * mix) + (original_audio_data[:L] * (1.0 - mix))
            
            if needs_final_processing:
                log_prefix = "⚡ Fast Mode:" if fast_mode else "🧪 Post-mix:"
                self.analysis_log.append(f"{log_prefix} Applying final normalization & limiting")
                processed_audio, _ = self._normalize(processed_audio, sample_rate, meter, target_lufs)
                if do_limiter:
                    limiter = Limiter(threshold_db=limiter_threshold_db, release_ms=LIMITER_RELEASE_MS)
                    board = Pedalboard([limiter])
                    processed_audio = board(processed_audio.astype(np.float32), sample_rate=sample_rate)

            # --- Final Analysis ---
            final_analysis = self._analyze(processed_audio, sample_rate, meter) 
            final_lufs = meter.integrated_loudness(processed_audio)
            final_peak = np.max(np.abs(processed_audio)) if processed_audio.size > 0 else 0.0
            final_peak_db = -np.inf if final_peak <= 0 else 20*np.log10(final_peak)

            self.analysis_log.append(f"\n{'='*50}"); self.analysis_log.append(f"🏁 FINAL RESULTS"); self.analysis_log.append(f"{'='*50}")
            self.analysis_log.append(f"  LUFS: {final_lufs:.2f} (target: {target_lufs})")
            self.analysis_log.append(f"  Peak: {final_peak:.3f} ({final_peak_db:.1f} dBFS)")
            self.analysis_log.append(f"  Bass: {final_analysis.get('bass', np.nan):.2f} (target: ≤{eq_bass_target})")
            self.analysis_log.append(f"  Highs: {final_analysis.get('high', np.nan):.2f} (target: ≤{eq_high_target})")
            
            waveform_after = self._plot_waveform_to_tensor(processed_audio, sample_rate, "Processed Waveform")

            if channels == 1 and processed_audio.ndim == 2: processed_audio = processed_audio[:, 0]
            if processed_audio.ndim == 1: final_tensor = torch.from_numpy(processed_audio).unsqueeze(0)
            else: final_tensor = torch.from_numpy(processed_audio.T)
            final_tensor = final_tensor.unsqueeze(0).to(waveform_tensor.device)
            output_audio = {"waveform": final_tensor, "sample_rate": sample_rate}
            
            return (output_audio, "\n".join(self.analysis_log), waveform_before, waveform_after)

        except Exception as e:
            error_msg = f"❌ [MD_AutoMasterNode] Error: {e}"
            logging.error(error_msg)
            logging.debug(traceback.format_exc())
            print(f"[MD_AutoMasterNode] ⚠️ Error encountered, returning input unchanged")
            
            safe_waveform_before = waveform_before if waveform_before is not None else torch.zeros((1, 64, 64, 3))
            safe_waveform_after = torch.zeros((1, 64, 64, 3))
            
            return (audio, f"{error_msg}\n{traceback.format_exc()}", safe_waveform_before, safe_waveform_after)

# =================================================================================
# == Node Registration                                                            ==
# =================================================================================

NODE_CLASS_MAPPINGS = {
    "MD_AutoMasterNode": MD_AutoMasterNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MD_AutoMasterNode": "MD: Audio Auto Master Pro"
}