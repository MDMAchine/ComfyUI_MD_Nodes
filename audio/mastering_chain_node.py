# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# ████ MASTERING CHAIN NODE v1.3 – Optimized for Ace-Step Audio/Video ████▓▒░
# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀

# ░▒▓ ORIGIN & DEV:
#   • Cast into the void by: MDMAchine (sonic architect)
#   • Original mind behind Mastering Chain Node
#   • Initial ComfyUI adaptation by: Gemini
#   • Enhanced & refined by: MDMAchine & Gemini
#   • Critical optimizations & bugfixes: Gemini
#   • v1.3 overhaul: Claude (Anthropic AI Assistant)
#   • Final polish: MDMAchine QA/Sanity Hero
#   • License: Apache 2.0 — No cursed floppy disks allowed

# ░▒▓ DESCRIPTION:
#   Professional-grade mastering chain for ComfyUI with Ace-Step precision.
#   Engineered for advanced audio processing with true stereo-linked multiband
#   compression, surgical EQ control, and transparent limiting. Optimized for
#   post-LUFS normalized audio (-14 to -20 LUFS input). Built to channel the
#   chaos of demoscene nights into clean, powerful sound transformations.
#   May work with other models, but don't expect mercy or miracles.

# ░▒▓ FEATURES:
#   ✓ Global gain control for precise level adjustment
#   ✓ True low-pass and high-pass Butterworth cutoff filters (NEW in v1.3)
#   ✓ High-shelf and low-shelf filters with corrected RBJ coefficients
#   ✓ Four parametric EQ bands for surgical frequency sculpting
#   ✓ Single-band compressor with RMS detection via Pedalboard
#   ✓ 3-band multiband compressor with stereo-linked gain reduction (FIXED in v1.3)
#   ✓ Linkwitz-Riley crossovers with configurable order for phase coherence
#   ✓ Lookahead limiter to prevent clipping and overs
#   ✓ Mono and stereo support with automatic channel handling
#   ✓ Before/after waveform visualizations for instant feedback
#   ✓ Optimized defaults for -14 LUFS normalized input (NEW in v1.3)
#   ✓ Robust numerical stability and NaN/Inf protection

# ░▒▓ CHANGELOG:
#   - v1.3 (Current Release - Major Overhaul):
#       • Fixed stereo phasing with stereo-linked multiband compression
#       • Added true low-pass and high-pass cutoff filters (Butterworth)
#       • Corrected shelf filter coefficient normalization (RBJ cookbook)
#       • Fixed tensor boolean evaluation error in audio extraction
#       • Optimized compression defaults for -14 LUFS normalized input
#       • Added configurable crossover filter order (4-12th order)
#       • Improved numerical stability throughout signal chain
#       • Enhanced console output with detailed processing logs
#       • Better filter validation and bounds checking
#   - v1.2a:
#       • Re-implemented shelf filters with RBJ EQ Cookbook formulas
#       • Refactored code for clarity & maintainability
#   - v1.0-1.1:
#       • Switched compression to pedalboard library for speed
#       • Fixed pedalboard compressor parameters
#   - v0.1-0.9.1:
#       • Initial development and feature additions
#       • EQ bands, compression modes, waveform plotting
#       • See previous versions for detailed history

# ░▒▓ CONFIGURATION:
#   → Primary Use: Professional audio mastering for normalized material (-14 LUFS)
#   → Secondary Use: Creative audio shaping and frequency sculpting
#   → Advanced Use: Stereo-linked multiband dynamics for mix glue
#   → Edge Use: Live performance or experimental audio hacking

# ░▒▓ RECOMMENDED WORKFLOW:
#   1. LUFS Normalize your audio (-20 to -14 LUFS recommended)
#   2. Apply Mastering Chain with surgical EQ and gentle compression
#   3. Use limiter for final peak control (-0.3dB to -0.1dB ceiling)
#   4. Export at desired loudness target

# ░▒▓ WARNING:
#   This node may trigger:
#   ▓▒░ Temporal distortion and memories of ANSI art
#   ▓▒░ Unstoppable urges to tweak Q factors at 3 AM
#   ▓▒░ Spontaneous understanding of Linkwitz-Riley topology
#   ▓▒░ Compulsive A/B comparison syndrome
#   ▓▒░ Dangerous levels of sonic confidence

# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄

import torch
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import io
from PIL import Image
import pedalboard


class MasteringChainNode:
    """
    Advanced ComfyUI audio mastering chain with improved EQ and stereo-linked compression.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO", {"audio_sampling_rate_input": True, "tooltip": "Input audio waveform. Supports mono or stereo."}),
                "sample_rate": ("INT", {"default": 44100, "min": 8000, "max": 192000, "step": 1, "tooltip": "Sampling rate of the audio in Hz."}),

                # --- Global Gain ---
                "master_gain_db": ("FLOAT", {"default": 0.0, "min": -20.0, "max": 20.0, "step": 0.1, "tooltip": "Overall gain adjustment (dB)."}),

                # --- Equalization ---
                "enable_eq": ("BOOLEAN", {"default": True, "tooltip": "Enable/bypass Equalizer."}),

                # Low-Pass Filter (NEW)
                "enable_lowpass": ("BOOLEAN", {"default": False, "tooltip": "Enable low-pass cutoff filter."}),
                "lowpass_freq": ("FLOAT", {"default": 18000.0, "min": 1000.0, "max": 20000.0, "step": 100.0, "tooltip": "Low-pass cutoff frequency (Hz)."}),
                "lowpass_order": ("INT", {"default": 4, "min": 2, "max": 8, "step": 2, "tooltip": "Filter order (higher = steeper rolloff)."}),

                # High-Pass Filter (NEW)
                "enable_highpass": ("BOOLEAN", {"default": False, "tooltip": "Enable high-pass cutoff filter."}),
                "highpass_freq": ("FLOAT", {"default": 20.0, "min": 10.0, "max": 500.0, "step": 1.0, "tooltip": "High-pass cutoff frequency (Hz)."}),
                "highpass_order": ("INT", {"default": 4, "min": 2, "max": 8, "step": 2, "tooltip": "Filter order (higher = steeper rolloff)."}),

                # High-Shelf
                "eq_high_shelf_gain_db": ("FLOAT", {"default": 0.0, "min": -12.0, "max": 12.0, "step": 0.1, "tooltip": "High-shelf gain (dB)."}),
                "eq_high_shelf_freq": ("FLOAT", {"default": 12000.0, "min": 1000.0, "max": 20000.0, "step": 100.0, "tooltip": "High-shelf frequency (Hz)."}),

                # Low-Shelf
                "enable_low_shelf_eq": ("BOOLEAN", {"default": False, "tooltip": "Enable low-shelf EQ."}),
                "eq_low_shelf_gain_db": ("FLOAT", {"default": 0.0, "min": -12.0, "max": 12.0, "step": 0.1, "tooltip": "Low-shelf gain (dB)."}),
                "eq_low_shelf_freq": ("FLOAT", {"default": 75.0, "min": 20.0, "max": 500.0, "step": 1.0, "tooltip": "Low-shelf frequency (Hz)."}),

                # Parametric EQ Bands
                "enable_param_eq1": ("BOOLEAN", {"default": False, "tooltip": "Enable Parametric EQ 1."}),
                "param_eq1_gain_db": ("FLOAT", {"default": 0.0, "min": -20.0, "max": 20.0, "step": 0.1}),
                "param_eq1_freq": ("FLOAT", {"default": 55.0, "min": 20.0, "max": 20000.0, "step": 1.0}),
                "param_eq1_q": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 20.0, "step": 0.1}),

                "enable_param_eq2": ("BOOLEAN", {"default": False, "tooltip": "Enable Parametric EQ 2."}),
                "param_eq2_gain_db": ("FLOAT", {"default": 0.0, "min": -20.0, "max": 20.0, "step": 0.1}),
                "param_eq2_freq": ("FLOAT", {"default": 125.0, "min": 20.0, "max": 20000.0, "step": 1.0}),
                "param_eq2_q": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 20.0, "step": 0.1}),

                "enable_param_eq3": ("BOOLEAN", {"default": False, "tooltip": "Enable Parametric EQ 3."}),
                "param_eq3_gain_db": ("FLOAT", {"default": 0.0, "min": -20.0, "max": 20.0, "step": 0.1}),
                "param_eq3_freq": ("FLOAT", {"default": 1250.0, "min": 20.0, "max": 20000.0, "step": 1.0}),
                "param_eq3_q": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 20.0, "step": 0.1}),

                "enable_param_eq4": ("BOOLEAN", {"default": False, "tooltip": "Enable Parametric EQ 4."}),
                "param_eq4_gain_db": ("FLOAT", {"default": 0.0, "min": -20.0, "max": 20.0, "step": 0.1}),
                "param_eq4_freq": ("FLOAT", {"default": 5000.0, "min": 20.0, "max": 20000.0, "step": 1.0}),
                "param_eq4_q": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 20.0, "step": 0.1}),

                # --- Compression ---
                "enable_comp": ("BOOLEAN", {"default": False, "tooltip": "Enable Compressor."}),
                "comp_type": (["Single-Band", "Multiband"], {"default": "Multiband"}),

                # Single-Band Compressor (optimized for -14 LUFS input)
                "comp_threshold_db": ("FLOAT", {"default": -8.0, "min": -60.0, "max": 0.0, "step": 0.1, "tooltip": "Threshold optimized for normalized input (-14 LUFS)"}),
                "comp_ratio": ("FLOAT", {"default": 2.5, "min": 1.0, "max": 20.0, "step": 0.1}),
                "comp_attack_ms": ("FLOAT", {"default": 20.0, "min": 1.0, "max": 500.0, "step": 1.0}),
                "comp_release_ms": ("FLOAT", {"default": 250.0, "min": 10.0, "max": 2000.0, "step": 10.0}),
                "comp_makeup_gain_db": ("FLOAT", {"default": 0.0, "min": -12.0, "max": 12.0, "step": 0.1}),

                # Multiband Crossovers
                "mb_crossover_low_mid_hz": ("FLOAT", {"default": 250.0, "min": 20.0, "max": 1000.0, "step": 10.0}),
                "mb_crossover_mid_high_hz": ("FLOAT", {"default": 4000.0, "min": 1000.0, "max": 15000.0, "step": 100.0}),
                "mb_crossover_order": ("INT", {"default": 8, "min": 4, "max": 12, "step": 2, "tooltip": "Crossover filter order (higher = steeper)."}),

                # Low Band (tighter control on bass)
                "mb_low_threshold_db": ("FLOAT", {"default": -10.0, "min": -60.0, "max": 0.0, "step": 0.1, "tooltip": "Lower threshold for tighter bass control"}),
                "mb_low_ratio": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 20.0, "step": 0.1}),
                "mb_low_attack_ms": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 500.0, "step": 1.0}),
                "mb_low_release_ms": ("FLOAT", {"default": 300.0, "min": 10.0, "max": 2000.0, "step": 10.0}),
                "mb_low_makeup_gain_db": ("FLOAT", {"default": 0.0, "min": -12.0, "max": 12.0, "step": 0.1}),

                # Mid Band (transparent handling of vocals/instruments)
                "mb_mid_threshold_db": ("FLOAT", {"default": -8.0, "min": -60.0, "max": 0.0, "step": 0.1, "tooltip": "Balanced for vocal/instrument range"}),
                "mb_mid_ratio": ("FLOAT", {"default": 2.5, "min": 1.0, "max": 20.0, "step": 0.1}),
                "mb_mid_attack_ms": ("FLOAT", {"default": 20.0, "min": 1.0, "max": 500.0, "step": 1.0}),
                "mb_mid_release_ms": ("FLOAT", {"default": 180.0, "min": 10.0, "max": 2000.0, "step": 10.0}),
                "mb_mid_makeup_gain_db": ("FLOAT", {"default": 0.0, "min": -12.0, "max": 12.0, "step": 0.1}),

                # High Band (gentle on highs to preserve air)
                "mb_high_threshold_db": ("FLOAT", {"default": -6.0, "min": -60.0, "max": 0.0, "step": 0.1, "tooltip": "Higher threshold to preserve high-frequency detail"}),
                "mb_high_ratio": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 20.0, "step": 0.1}),
                "mb_high_attack_ms": ("FLOAT", {"default": 10.0, "min": 1.0, "max": 500.0, "step": 1.0}),
                "mb_high_release_ms": ("FLOAT", {"default": 120.0, "min": 10.0, "max": 2000.0, "step": 10.0}),
                "mb_high_makeup_gain_db": ("FLOAT", {"default": 0.0, "min": -12.0, "max": 12.0, "step": 0.1}),

                # --- Limiting ---
                "enable_limiter": ("BOOLEAN", {"default": False, "tooltip": "Enable Limiter."}),
                "limiter_ceiling_db": ("FLOAT", {"default": -0.1, "min": -6.0, "max": 0.0, "step": 0.01}),
                "limiter_release_ms": ("FLOAT", {"default": 50.0, "min": 1.0, "max": 500.0, "step": 1.0}),
            }
        }

    RETURN_TYPES = ("AUDIO", "IMAGE", "IMAGE")
    RETURN_NAMES = ("audio", "waveform_before", "waveform_after")
    FUNCTION = "apply_mastering_chain"
    CATEGORY = "MD_Nodes/Audio Processing"

    # ==================== UTILITY METHODS ====================

    def _db_to_amplitude(self, db):
        return 10**(db / 20.0)

    def _amplitude_to_db(self, amplitude):
        return 20 * np.log10(np.abs(amplitude) + 1e-9)

    def _apply_gain(self, audio_data, gain_db):
        if gain_db == 0.0:
            return audio_data
        return audio_data * self._db_to_amplitude(gain_db)

    # ==================== FILTER DESIGN ====================

    def _design_lowpass_filter(self, cutoff_freq, sample_rate, order=4):
        """Butterworth low-pass filter."""
        nyquist = 0.5 * sample_rate
        normal_cutoff = cutoff_freq / nyquist
        if normal_cutoff >= 1.0:
            normal_cutoff = 0.99
        b, a = signal.butter(order, normal_cutoff, btype='lowpass', analog=False)
        return signal.tf2sos(b, a)

    def _design_highpass_filter(self, cutoff_freq, sample_rate, order=4):
        """Butterworth high-pass filter."""
        nyquist = 0.5 * sample_rate
        normal_cutoff = cutoff_freq / nyquist
        if normal_cutoff <= 0.0:
            normal_cutoff = 0.01
        b, a = signal.butter(order, normal_cutoff, btype='highpass', analog=False)
        return signal.tf2sos(b, a)

    def _design_peaking_filter(self, gain_db, freq, Q, sample_rate):
        """Peaking (bell) EQ filter."""
        if abs(gain_db) < 0.001:
            return None
            
        A = 10**(gain_db / 40.0)
        omega = 2 * np.pi * freq / sample_rate
        sn = np.sin(omega)
        cs = np.cos(omega)
        alpha = sn / (2 * Q) if Q > 1e-6 else sn / (2 * 0.707)

        b0 = 1 + alpha * A
        b1 = -2 * cs
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * cs
        a2 = 1 - alpha / A

        b = np.array([b0 / a0, b1 / a0, b2 / a0], dtype=np.float64)
        a = np.array([1.0, a1 / a0, a2 / a0], dtype=np.float64)
        return signal.tf2sos(b, a)

    def _design_low_shelf_filter(self, gain_db, freq, sample_rate):
        """Low-shelf filter using RBJ cookbook (corrected)."""
        if abs(gain_db) < 0.001:
            return None
            
        A = 10**(gain_db / 40.0)
        w0 = 2 * np.pi * freq / sample_rate
        cos_w0 = np.cos(w0)
        sin_w0 = np.sin(w0)
        alpha = sin_w0 / (2 * 0.707)

        b0 = A * ((A+1) - (A-1)*cos_w0 + 2*np.sqrt(A)*alpha)
        b1 = 2 * A * ((A-1) - (A+1)*cos_w0)
        b2 = A * ((A+1) - (A-1)*cos_w0 - 2*np.sqrt(A)*alpha)
        a0 = (A+1) + (A-1)*cos_w0 + 2*np.sqrt(A)*alpha
        a1 = -2 * ((A-1) + (A+1)*cos_w0)
        a2 = (A+1) + (A-1)*cos_w0 - 2*np.sqrt(A)*alpha

        b = np.array([b0 / a0, b1 / a0, b2 / a0], dtype=np.float64)
        a = np.array([1.0, a1 / a0, a2 / a0], dtype=np.float64)
        return signal.tf2sos(b, a)

    def _design_high_shelf_filter(self, gain_db, freq, sample_rate):
        """High-shelf filter using RBJ cookbook (corrected)."""
        if abs(gain_db) < 0.001:
            return None
            
        A = 10**(gain_db / 40.0)
        w0 = 2 * np.pi * freq / sample_rate
        cos_w0 = np.cos(w0)
        sin_w0 = np.sin(w0)
        alpha = sin_w0 / (2 * 0.707)

        b0 = A * ((A+1) + (A-1)*cos_w0 + 2*np.sqrt(A)*alpha)
        b1 = -2 * A * ((A-1) + (A+1)*cos_w0)
        b2 = A * ((A+1) + (A-1)*cos_w0 - 2*np.sqrt(A)*alpha)
        a0 = (A+1) - (A-1)*cos_w0 + 2*np.sqrt(A)*alpha
        a1 = 2 * ((A-1) - (A+1)*cos_w0)
        a2 = (A+1) - (A-1)*cos_w0 - 2*np.sqrt(A)*alpha

        b = np.array([b0 / a0, b1 / a0, b2 / a0], dtype=np.float64)
        a = np.array([1.0, a1 / a0, a2 / a0], dtype=np.float64)
        return signal.tf2sos(b, a)

    def _design_linkwitz_riley_crossover(self, cutoff_freq, sample_rate, order=8):
        """Linkwitz-Riley crossover (must be even order)."""
        if order % 2 != 0:
            order = order + 1
            
        nyquist = 0.5 * sample_rate
        normal_cutoff = cutoff_freq / nyquist
        normal_cutoff = np.clip(normal_cutoff, 0.01, 0.99)

        b_lp, a_lp = signal.butter(order // 2, normal_cutoff, btype='lowpass')
        b_hp, a_hp = signal.butter(order // 2, normal_cutoff, btype='highpass')

        return signal.tf2sos(b_lp, a_lp), signal.tf2sos(b_hp, a_hp)

    def _apply_filters_to_audio(self, audio_data, sos_filters):
        """Apply chain of SOS filters."""
        if not sos_filters:
            return audio_data

        filtered_audio = audio_data.copy()
        
        if audio_data.ndim == 1:  # Mono
            for sos in sos_filters:
                if sos is not None:
                    filtered_audio = signal.sosfilt(sos, filtered_audio)
                    filtered_audio = np.nan_to_num(filtered_audio, nan=0.0, posinf=1.0, neginf=-1.0)
        else:  # Stereo
            for i in range(audio_data.shape[0]):
                channel_data = audio_data[i, :].copy()
                for sos in sos_filters:
                    if sos is not None:
                        channel_data = signal.sosfilt(sos, channel_data)
                        channel_data = np.nan_to_num(channel_data, nan=0.0, posinf=1.0, neginf=-1.0)
                filtered_audio[i, :] = channel_data

        return np.clip(filtered_audio, -1.0, 1.0)

    # ==================== PROCESSING STAGES ====================

    def _apply_eq(self, audio_data, sample_rate, **params):
        """Apply complete EQ chain."""
        print("Applying EQ Filters...")
        sos_filters = []

        # Low-pass cutoff
        if params.get('enable_lowpass'):
            print(f"  - Low-pass: {params['lowpass_freq']}Hz, Order={params['lowpass_order']}")
            sos_filters.append(self._design_lowpass_filter(
                params['lowpass_freq'], sample_rate, params['lowpass_order']
            ))

        # High-pass cutoff
        if params.get('enable_highpass'):
            print(f"  - High-pass: {params['highpass_freq']}Hz, Order={params['highpass_order']}")
            sos_filters.append(self._design_highpass_filter(
                params['highpass_freq'], sample_rate, params['highpass_order']
            ))

        # Shelving filters
        sos = self._design_high_shelf_filter(params['eq_high_shelf_gain_db'], 
                                             params['eq_high_shelf_freq'], sample_rate)
        if sos is not None:
            print(f"  - High-shelf: {params['eq_high_shelf_gain_db']}dB @ {params['eq_high_shelf_freq']}Hz")
            sos_filters.append(sos)

        if params.get('enable_low_shelf_eq'):
            sos = self._design_low_shelf_filter(params['eq_low_shelf_gain_db'],
                                               params['eq_low_shelf_freq'], sample_rate)
            if sos is not None:
                print(f"  - Low-shelf: {params['eq_low_shelf_gain_db']}dB @ {params['eq_low_shelf_freq']}Hz")
                sos_filters.append(sos)

        # Parametric bands
        for i in range(1, 5):
            if params.get(f'enable_param_eq{i}'):
                sos = self._design_peaking_filter(
                    params[f'param_eq{i}_gain_db'],
                    params[f'param_eq{i}_freq'],
                    params[f'param_eq{i}_q'],
                    sample_rate
                )
                if sos is not None:
                    print(f"  - Parametric {i}: {params[f'param_eq{i}_gain_db']}dB @ {params[f'param_eq{i}_freq']}Hz, Q={params[f'param_eq{i}_q']}")
                    sos_filters.append(sos)

        return self._apply_filters_to_audio(audio_data, sos_filters)

    def _apply_single_band_compression(self, audio_data, threshold_db, ratio, 
                                       attack_ms, release_ms, makeup_gain_db, sample_rate):
        """Single-band compression with Pedalboard."""
        print(f"  Single-band: Thresh={threshold_db}dB, Ratio={ratio}:1, Attack={attack_ms}ms, Release={release_ms}ms")
        
        audio_data = audio_data.astype(np.float32)
        
        compressor = pedalboard.Compressor(
            threshold_db=threshold_db,
            ratio=ratio,
            attack_ms=attack_ms,
            release_ms=release_ms
        )

        if audio_data.ndim == 2:
            processed = compressor(audio_data.T, sample_rate=sample_rate).T
        else:
            processed = compressor(audio_data, sample_rate=sample_rate)

        if makeup_gain_db != 0.0:
            processed *= self._db_to_amplitude(makeup_gain_db)

        return processed

    def _apply_multiband_compression_stereo_linked(self, audio_data, sample_rate, 
                                                   crossover_low_mid, crossover_mid_high,
                                                   crossover_order, low_params, mid_params, high_params):
        """
        Multiband compression with stereo-linked gain reduction to prevent phasing.
        """
        print(f"Applying Stereo-Linked Multiband Compression...")
        print(f"  Crossovers: {crossover_low_mid}Hz | {crossover_mid_high}Hz (Order={crossover_order})")

        if audio_data.ndim == 1:
            audio_data = audio_data[np.newaxis, :]

        num_channels = audio_data.shape[0]
        is_stereo = num_channels == 2
        processed_audio = np.zeros_like(audio_data, dtype=np.float32)

        # Design crossovers
        sos_lm_lp, sos_lm_hp = self._design_linkwitz_riley_crossover(crossover_low_mid, sample_rate, crossover_order)
        sos_mh_lp, sos_mh_hp = self._design_linkwitz_riley_crossover(crossover_mid_high, sample_rate, crossover_order)

        # Split all channels into bands
        bands = {'low': [], 'mid': [], 'high': []}
        
        for c in range(num_channels):
            ch = audio_data[c, :].astype(np.float32)
            
            # Low band
            low = signal.sosfiltfilt(sos_lm_lp, ch)
            
            # High band
            high = signal.sosfiltfilt(sos_mh_hp, ch)
            
            # Mid band
            mid_hp = signal.sosfiltfilt(sos_lm_hp, ch)
            mid = signal.sosfiltfilt(sos_mh_lp, mid_hp)
            
            bands['low'].append(low)
            bands['mid'].append(mid)
            bands['high'].append(high)

        # Compress each band
        for band_name, params in [('low', low_params), ('mid', mid_params), ('high', high_params)]:
            print(f"  {band_name.upper()} band: Thresh={params['threshold_db']}dB, Ratio={params['ratio']}:1")
            
            if is_stereo:
                # Stack stereo for linked processing
                stereo_band = np.stack([bands[band_name][0], bands[band_name][1]], axis=0)
                
                # Compress with stereo linking
                compressed = self._apply_single_band_compression(
                    stereo_band, params['threshold_db'], params['ratio'],
                    params['attack_ms'], params['release_ms'],
                    params['makeup_gain_db'], sample_rate
                )
                
                processed_audio[0, :] += compressed[0, :]
                processed_audio[1, :] += compressed[1, :]
            else:
                # Mono processing
                compressed = self._apply_single_band_compression(
                    bands[band_name][0], params['threshold_db'], params['ratio'],
                    params['attack_ms'], params['release_ms'],
                    params['makeup_gain_db'], sample_rate
                )
                processed_audio[0, :] += compressed

        processed_audio = np.clip(processed_audio, -1.0, 1.0)

        if num_channels == 1:
            return processed_audio.flatten()
        return processed_audio

    def _apply_limiter(self, audio_data, ceiling_db, release_ms, sample_rate):
        """Brickwall limiter with Pedalboard."""
        print(f"Applying Limiter: Ceiling={ceiling_db}dB, Release={release_ms}ms")
        
        audio_data = audio_data.astype(np.float32)
        
        limiter = pedalboard.Limiter(
            threshold_db=ceiling_db,
            release_ms=release_ms
        )

        if audio_data.ndim == 2:
            processed = limiter(audio_data.T, sample_rate=sample_rate).T
        else:
            processed = limiter(audio_data, sample_rate=sample_rate)

        return np.clip(processed, -1.0, 1.0)

    def _plot_waveform_to_tensor(self, audio_data, sample_rate, title="Waveform"):
        """Plot waveform and return as tensor."""
        plt.figure(figsize=(10, 4))

        if audio_data.ndim == 2:
            mono = np.mean(audio_data, axis=0)
        else:
            mono = audio_data

        time_axis = np.linspace(0, len(mono) / sample_rate, len(mono))
        plt.plot(time_axis, mono, color='blue', linewidth=0.5)
        
        plt.title(title)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.xlim(0, len(mono) / sample_rate)
        plt.ylim(-1.05, 1.05)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        plt.close()

        img = Image.open(buf).convert("RGB")
        img_np = np.array(img).astype(np.float32) / 255.0
        return torch.from_numpy(img_np).unsqueeze(0)

    # ==================== MAIN PROCESSING ====================

    def apply_mastering_chain(self, audio, sample_rate, master_gain_db,
                              enable_eq, enable_lowpass, lowpass_freq, lowpass_order,
                              enable_highpass, highpass_freq, highpass_order,
                              eq_high_shelf_gain_db, eq_high_shelf_freq,
                              enable_low_shelf_eq, eq_low_shelf_gain_db, eq_low_shelf_freq,
                              enable_param_eq1, param_eq1_gain_db, param_eq1_freq, param_eq1_q,
                              enable_param_eq2, param_eq2_gain_db, param_eq2_freq, param_eq2_q,
                              enable_param_eq3, param_eq3_gain_db, param_eq3_freq, param_eq3_q,
                              enable_param_eq4, param_eq4_gain_db, param_eq4_freq, param_eq4_q,
                              enable_comp, comp_type,
                              comp_threshold_db, comp_ratio, comp_attack_ms, comp_release_ms, comp_makeup_gain_db,
                              mb_crossover_low_mid_hz, mb_crossover_mid_high_hz, mb_crossover_order,
                              mb_low_threshold_db, mb_low_ratio, mb_low_attack_ms, mb_low_release_ms, mb_low_makeup_gain_db,
                              mb_mid_threshold_db, mb_mid_ratio, mb_mid_attack_ms, mb_mid_release_ms, mb_mid_makeup_gain_db,
                              mb_high_threshold_db, mb_high_ratio, mb_high_attack_ms, mb_high_release_ms, mb_high_makeup_gain_db,
                              enable_limiter, limiter_ceiling_db, limiter_release_ms):
        
        print(f"\n{'='*60}")
        print(f"MASTERING CHAIN v1.3 | Sample Rate: {sample_rate} Hz")
        print(f"{'='*60}\n")

        # Extract audio tensor (avoid 'or' with tensors - causes ambiguous boolean error)
        if isinstance(audio, dict):
            audio_tensor = audio.get('waveform')
            if audio_tensor is None:
                audio_tensor = audio.get('audio')
            if audio_tensor is None:
                audio_tensor = audio.get('samples')
            if audio_tensor is None:
                raise ValueError("Could not find audio data in dictionary. Expected keys: 'waveform', 'audio', or 'samples'")
        else:
            audio_tensor = audio

        audio_np = audio_tensor.cpu().numpy()

        # Normalize to 2D
        if audio_np.ndim == 3:
            audio_np = audio_np[0]
        if audio_np.ndim == 1:
            audio_np = audio_np[np.newaxis, :]

        # Generate before plot
        before_plot = self._plot_waveform_to_tensor(audio_np, sample_rate, "Original Waveform")

        # Process
        processed = audio_np.copy()

        # 1. Global Gain
        if master_gain_db != 0.0:
            print(f"Applying Global Gain: {master_gain_db:+.1f} dB")
            processed = self._apply_gain(processed, master_gain_db)
            processed = np.clip(processed, -1.0, 1.0)

        # 2. EQ
        if enable_eq:
            eq_params = {
                'enable_lowpass': enable_lowpass, 'lowpass_freq': lowpass_freq, 'lowpass_order': lowpass_order,
                'enable_highpass': enable_highpass, 'highpass_freq': highpass_freq, 'highpass_order': highpass_order,
                'eq_high_shelf_gain_db': eq_high_shelf_gain_db, 'eq_high_shelf_freq': eq_high_shelf_freq,
                'enable_low_shelf_eq': enable_low_shelf_eq, 'eq_low_shelf_gain_db': eq_low_shelf_gain_db, 
                'eq_low_shelf_freq': eq_low_shelf_freq,
                'enable_param_eq1': enable_param_eq1, 'param_eq1_gain_db': param_eq1_gain_db, 
                'param_eq1_freq': param_eq1_freq, 'param_eq1_q': param_eq1_q,
                'enable_param_eq2': enable_param_eq2, 'param_eq2_gain_db': param_eq2_gain_db,
                'param_eq2_freq': param_eq2_freq, 'param_eq2_q': param_eq2_q,
                'enable_param_eq3': enable_param_eq3, 'param_eq3_gain_db': param_eq3_gain_db,
                'param_eq3_freq': param_eq3_freq, 'param_eq3_q': param_eq3_q,
                'enable_param_eq4': enable_param_eq4, 'param_eq4_gain_db': param_eq4_gain_db,
                'param_eq4_freq': param_eq4_freq, 'param_eq4_q': param_eq4_q,
            }
            processed = self._apply_eq(processed, sample_rate, **eq_params)
            processed = np.clip(processed, -1.0, 1.0)

        # 3. Compression
        if enable_comp:
            if comp_type == "Single-Band":
                processed = self._apply_single_band_compression(
                    processed, comp_threshold_db, comp_ratio,
                    comp_attack_ms, comp_release_ms, comp_makeup_gain_db, sample_rate
                )
            else:  # Multiband
                low_params = {
                    'threshold_db': mb_low_threshold_db, 'ratio': mb_low_ratio,
                    'attack_ms': mb_low_attack_ms, 'release_ms': mb_low_release_ms,
                    'makeup_gain_db': mb_low_makeup_gain_db
                }
                mid_params = {
                    'threshold_db': mb_mid_threshold_db, 'ratio': mb_mid_ratio,
                    'attack_ms': mb_mid_attack_ms, 'release_ms': mb_mid_release_ms,
                    'makeup_gain_db': mb_mid_makeup_gain_db
                }
                high_params = {
                    'threshold_db': mb_high_threshold_db, 'ratio': mb_high_ratio,
                    'attack_ms': mb_high_attack_ms, 'release_ms': mb_high_release_ms,
                    'makeup_gain_db': mb_high_makeup_gain_db
                }
                processed = self._apply_multiband_compression_stereo_linked(
                    processed, sample_rate, mb_crossover_low_mid_hz, mb_crossover_mid_high_hz,
                    mb_crossover_order, low_params, mid_params, high_params
                )
            processed = np.clip(processed, -1.0, 1.0)

        # 4. Limiting
        if enable_limiter:
            processed = self._apply_limiter(processed, limiter_ceiling_db, limiter_release_ms, sample_rate)
            processed = np.clip(processed, -1.0, 1.0)

        # Generate after plot
        after_plot = self._plot_waveform_to_tensor(processed, sample_rate, "Processed Waveform")

        # Convert back to tensor
        if processed.ndim == 1:
            processed = processed[np.newaxis, :]

        output_tensor = torch.from_numpy(processed).to(audio_tensor.device).float().unsqueeze(0)
        
        output_audio = {
            "waveform": output_tensor,
            "sample_rate": sample_rate
        }

        print(f"\n{'='*60}")
        print("MASTERING COMPLETE")
        print(f"{'='*60}\n")

        return (output_audio, before_plot, after_plot)


NODE_CLASS_MAPPINGS = {
    "MasteringChainNode": MasteringChainNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MasteringChainNode": "Mastering Chain v1.3 (Enhanced)"
}