# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# ████ MASTERING CHAIN NODE v1.2a – Optimized for Ace-Step Audio/Video ████▓▒░
# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀

# ░▒▓ ORIGIN & DEV:
#   • Cast into the void by: MDMAchine
#   • Original mind behind Mastering Chain Node
#   • Initial ComfyUI adaptation by: Gemini
#   • Enhanced & refined by: MDMAchine & Gemini
#   • Critical optimizations & bugfixes: Gemini
#   • Final polish: MDMAchine QA/Sanity Hero
#   • License: Apache 2.0 — No cursed floppy disks allowed

# ░▒▓ DESCRIPTION:
#   A mastering chain node for ComfyUI, designed with Ace-Step precision.
#   Built for advanced audio processing, it channels the chaos of
#   demoscene nights into clean, powerful sound transformations.
#   May work with other models, but don't expect mercy or miracles.

# ░▒▓ FEATURES:
#   ✓ Global gain control to crank up or dial down output
#   ✓ Multi-band equalizer for precise frequency sculpting
#   ✓ Advanced compression with RMS detection & soft knee
#   ✓ Lookahead limiter to prevent clipping and overs
#   ✓ Mono and stereo support for flexible channel handling
#   ✓ Before/after waveform visualizations for instant feedback

# ░▒▓ CHANGELOG:
#   - v0.1 (Initial Release):
#       • Core gain implemented
#       • Basic stability ensured
#   - v0.2:
#       • Added simple EQ and limiter
#       • UI/UX improvements
#   - v0.3:
#       • Compressor upgraded with RMS detection & soft knee
#   - v0.4:
#       • Multi-band compression added
#   - v0.5:
#       • Linkwitz-Riley crossovers for phase coherence
#   - v0.6:
#       • Added Low-Shelf and Parametric EQ bands
#   - v0.7:
#       • Waveform plotting implemented
#   - v0.8:
#       • Numerical stability & tooltip improvements
#   - v0.9:
#       • Stability tweaks, code cleanup, default parameter adjustments
#   - v0.9.1:
#       • Fixed shelf filter boost/cut behavior
#       • Added extra parametric EQ bands
#       • Changed EQ inputs to text fields
#   - v1.0:
#       • Switched compression to pedalboard library for speed
#   - v1.1:
#       • Fixed pedalboard compressor parameters
#   - v1.2a:
#       • Re-implemented shelf filters with RBJ EQ Cookbook formulas
#       • Refactored code for clarity & maintainability

# ░▒▓ CONFIGURATION:
#   → Primary Use: Audio mastering and dynamic processing
#   → Secondary Use: Creative audio shaping and experimentation
#   → Edge Use: Live performance or experimental audio hacking

# ░▒▓ WARNING:
#   This node may trigger:
#   ▓▒░ Temporal distortion
#   ▓▒░ Memories of ANSI art & screaming modems
#   ▓▒░ A sense of unstoppable creative power

# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄



import torch
import numpy as np
from scipy import signal
import os
import matplotlib.pyplot as plt
import io
from PIL import Image # For converting image buffer to tensor
import pedalboard # New import for faster DSP


class MasteringChainNode:
    """
    A custom ComfyUI node for applying a mastering chain to audio.

    This node provides a structured framework for common mastering steps:
    Global Gain, Equalization, Compression (single-band or multiband),
    and Limiting.

    The DSP logic for EQ, Compression, and Limiting are improved,
    with an enhanced single-band compressor (RMS detection, soft knee),
    a 3-band multiband compressor using Linkwitz-Riley crossovers,
    and a more controlled limiter with a configurable release.
    It supports both mono and stereo audio processing.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        """
        Defines the input ports and their types that will appear in the ComfyUI UI.
        Includes tooltips for better user guidance.
        """
        return {
            "required": {
                # Input audio waveform (expected to be a PyTorch tensor, but handled if wrapped in dict)
                "audio": ("AUDIO", {"audio_sampling_rate_input": True, "tooltip": "Input audio waveform. Supports mono or stereo."}),
                # Sample rate is crucial for all DSP operations
                "sample_rate": ("INT", {"default": 44100, "min": 8000, "max": 192000, "step": 1, "tooltip": "Sampling rate of the audio in Hz. Must match the input audio."}),

                # --- Global Gain ---
                "master_gain_db": ("FLOAT", {"default": 0.0, "min": -20.0, "max": 20.0, "step": 0.1, "tooltip": "Overall gain adjustment for the entire mastering chain (in dB)."}),

                # --- Equalization (Expanded) ---
                "enable_eq": ("BOOLEAN", {"default": True, "tooltip": "Enable or bypass the entire Equalizer section."}),

                # High-Shelf (existing)
                "eq_high_shelf_gain_db": ("FLOAT", {"default": 0.0, "min": -12.0, "max": 12.0, "step": 0.1, "tooltip": "Gain for the high-shelf filter in dB. Affects frequencies above the shelf frequency."}),
                "eq_high_shelf_freq": ("FLOAT", {"default": 12000.0, "min": 1000.0, "max": 20000.0, "step": 100.0, "tooltip": "Corner frequency for the high-shelf filter in Hz. Frequencies above this are boosted/cut."}),
                "eq_high_shelf_q": ("FLOAT", {"default": 0.707, "min": 0.1, "max": 5.0, "step": 0.01, "tooltip": "Q factor for the high-shelf filter. (Fixed at 0.707 for standard behavior)."}),

                # Low-Shelf (New)
                "enable_low_shelf_eq": ("BOOLEAN", {"default": False, "tooltip": "Enable or bypass the Low-Shelf Equalizer."}),
                "eq_low_shelf_gain_db": ("FLOAT", {"default": -12.0, "min": -12.0, "max": 12.0, "step": 0.1, "tooltip": "Gain for the low-shelf filter in dB. Tone down specified hz."}),
                "eq_low_shelf_freq": ("FLOAT", {"default": 75.0, "min": 20.0, "max": 500.0, "step": 1.0, "tooltip": "Corner frequency for the low-shelf filter in Hz."}),
                "eq_low_shelf_q": ("FLOAT", {"default": 0.707, "min": 0.1, "max": 5.0, "step": 0.01, "tooltip": "Q factor for the low-shelf filter. (Fixed at 0.707 for standard behavior)."}),

                # Parametric EQ Band 1
                "enable_param_eq1": ("BOOLEAN", {"default": False, "tooltip": "Enable or bypass Parametric EQ Band 1."}),
                "param_eq1_gain_db": ("FLOAT", {"default": 0.0, "min": -20.0, "max": 20.0, "step": 0.1, "tooltip": "Gain for Parametric EQ Band 1 (e.g., -6dB for 77Hz cut)."}),
                "param_eq1_freq": ("FLOAT", {"default": 55.0, "min": 20.0, "max": 20000.0, "step": 1.0, "tooltip": "Center frequency for Parametric EQ Band 1."}),
                "param_eq1_q": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 20.0, "step": 0.1, "tooltip": "Q factor for Parametric EQ Band 1 (e.g., 9.0 for 77Hz)."}),

                # Parametric EQ Band 2
                "enable_param_eq2": ("BOOLEAN", {"default": False, "tooltip": "Enable or bypass Parametric EQ Band 2."}),
                "param_eq2_gain_db": ("FLOAT", {"default": 0.0, "min": -20.0, "max": 20.0, "step": 0.1, "tooltip": "Gain for Parametric EQ Band 2 (e.g., +3dB for 130Hz boost)."}),
                "param_eq2_freq": ("FLOAT", {"default": 125.0, "min": 20.0, "max": 20000.0, "step": 1.0, "tooltip": "Center frequency for Parametric EQ Band 2."}),
                "param_eq2_q": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 20.0, "step": 0.1, "tooltip": "Q factor for Parametric EQ Band 2."}),

                # Parametric EQ Band 3 (New)
                "enable_param_eq3": ("BOOLEAN", {"default": False, "tooltip": "Enable or bypass Parametric EQ Band 3."}),
                "param_eq3_gain_db": ("FLOAT", {"default": 0.0, "min": -20.0, "max": 20.0, "step": 0.1, "tooltip": "Gain for Parametric EQ Band 3."}),
                "param_eq3_freq": ("FLOAT", {"default": 1250.0, "min": 20.0, "max": 20000.0, "step": 1.0, "tooltip": "Center frequency for Parametric EQ Band 3."}),
                "param_eq3_q": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 20.0, "step": 0.1, "tooltip": "Q factor for Parametric EQ Band 3."}),

                # Parametric EQ Band 4 (New)
                "enable_param_eq4": ("BOOLEAN", {"default": False, "tooltip": "Enable or bypass Parametric EQ Band 4."}),
                "param_eq4_gain_db": ("FLOAT", {"default": 0.0, "min": -20.0, "max": 20.0, "step": 0.1, "tooltip": "Gain for Parametric EQ Band 4."}),
                "param_eq4_freq": ("FLOAT", {"default": 5000.0, "min": 20.0, "max": 20000.0, "step": 1.0, "tooltip": "Center frequency for Parametric EQ Band 4."}),
                "param_eq4_q": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 20.0, "step": 0.1, "tooltip": "Q factor for Parametric EQ Band 4."}),

                # --- Compression (Single-Band or Multiband) ---
                "enable_comp": ("BOOLEAN", {"default": False, "tooltip": "Enable or bypass the Compressor section."}),
                "comp_type": (["Single-Band", "Multiband"], {"default": "Multiband", "tooltip": "Select between a single-band or 3-band multiband compressor."}),

                # Single-Band Compressor Parameters (visible when comp_type is Single-Band)
                "comp_threshold_db": ("FLOAT", {"default": -14.0, "min": -60.0, "max": 0.0, "step": 0.1, "tooltip": "Single-band: Input level (in dB) above which compression begins."}),
                "comp_ratio": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 20.0, "step": 0.1, "tooltip": "Single-band: The ratio of input level change to output level change (e.g., 4:1 means 4dB in = 1dB out above threshold)."}),
                "comp_attack_ms": ("FLOAT", {"default": 25.0, "min": 1.0, "max": 500.0, "step": 1.0, "tooltip": "Single-band: Time (in ms) it takes for the compressor to reach full gain reduction after the signal crosses the threshold."}),
                "comp_release_ms": ("FLOAT", {"default": 300.0, "min": 10.0, "max": 2000.0, "step": 10.0, "tooltip": "Single-band: Time (in ms) it takes for the compressor to return to unity gain after the signal falls below the threshold."}),
                "comp_makeup_gain_db": ("FLOAT", {"default": 0.0, "min": -12.0, "max": 12.0, "step": 0.1, "tooltip": "Single-band: Adds gain after compression to compensate for reduction."}),
                "comp_soft_knee_db": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 12.0, "step": 0.1, "tooltip": "Single-band: The range (in dB) over which the compressor gradually transitions into full compression, rather than instantly (hard knee). A value of 0.0 means hard knee."}),

                # Multiband Compressor Crossover Frequencies (only active if comp_type is Multiband)
                "mb_crossover_low_mid_hz": ("FLOAT", {"default": 250.0, "min": 20.0, "max": 1000.0, "step": 10.0, "tooltip": "Multiband: Crossover frequency between the Low and Mid bands (in Hz)."}),
                "mb_crossover_mid_high_hz": ("FLOAT", {"default": 4000.0, "min": 1000.0, "max": 15000.0, "step": 100.0, "tooltip": "Multiband: Crossover frequency between the Mid and High bands (in Hz)."}),

                # Multiband Compressor Parameters (for each band)
                # Low Band
                "mb_low_threshold_db": ("FLOAT", {"default": -16.0, "min": -60.0, "max": 0.0, "step": 0.1, "tooltip": "Multiband Low: Threshold for the low-frequency band."}),
                "mb_low_ratio": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 20.0, "step": 0.1, "tooltip": "Multiband Low: Ratio for the low-frequency band."}),
                "mb_low_attack_ms": ("FLOAT", {"default": 40.0, "min": 1.0, "max": 500.0, "step": 1.0, "tooltip": "Multiband Low: Attack time for the low-frequency band."}),
                "mb_low_release_ms": ("FLOAT", {"default": 350.0, "min": 10.0, "max": 2000.0, "step": 10.0, "tooltip": "Multiband Low: Release time for the low-frequency band."}),
                "mb_low_makeup_gain_db": ("FLOAT", {"default": 0.0, "min": -12.0, "max": 12.0, "step": 0.1, "tooltip": "Multiband Low: Makeup gain for the low-frequency band."}),
                "mb_low_soft_knee_db": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 12.0, "step": 0.1, "tooltip": "Multiband Low: Soft knee range for the low-frequency band."}),

                # Mid Band
                "mb_mid_threshold_db": ("FLOAT", {"default": -14.0, "min": -60.0, "max": 0.0, "step": 0.1, "tooltip": "Multiband Mid: Threshold for the mid-frequency band."}),
                "mb_mid_ratio": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 20.0, "step": 0.1, "tooltip": "Multiband Mid: Ratio for the mid-frequency band."}),
                "mb_mid_attack_ms": ("FLOAT", {"default": 25.0, "min": 1.0, "max": 500.0, "step": 1.0, "tooltip": "Multiband Mid: Attack time for the mid-frequency band."}),
                "mb_mid_release_ms": ("FLOAT", {"default": 200.0, "min": 10.0, "max": 2000.0, "step": 10.0, "tooltip": "Multiband Mid: Release time for the mid-frequency band."}),
                "mb_mid_makeup_gain_db": ("FLOAT", {"default": 0.0, "min": -12.0, "max": 12.0, "step": 0.1, "tooltip": "Multiband Mid: Makeup gain for the mid-frequency band."}),
                "mb_mid_soft_knee_db": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 12.0, "step": 0.1, "tooltip": "Multiband Mid: Soft knee range for the mid-frequency band."}),

                # High Band
                "mb_high_threshold_db": ("FLOAT", {"default": -12.0, "min": -60.0, "max": 0.0, "step": 0.1, "tooltip": "Multiband High: Threshold for the high-frequency band."}),
                "mb_high_ratio": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 20.0, "step": 0.1, "tooltip": "Multiband High: Ratio for the high-frequency band."}),
                "mb_high_attack_ms": ("FLOAT", {"default": 10.0, "min": 1.0, "max": 500.0, "step": 1.0, "tooltip": "Multiband High: Attack time for the high-frequency band."}),
                "mb_high_release_ms": ("FLOAT", {"default": 120.0, "min": 10.0, "max": 2000.0, "step": 10.0, "tooltip": "Multiband High: Release time for the high-frequency band."}),
                "mb_high_makeup_gain_db": ("FLOAT", {"default": 0.0, "min": -12.0, "max": 12.0, "step": 0.1, "tooltip": "Multiband High: Makeup gain for the high-frequency band."}),
                "mb_high_soft_knee_db": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 12.0, "step": 0.1, "tooltip": "Multiband High: Soft knee range for the high-frequency band."}),

                # --- Limiting ---
                "enable_limiter": ("BOOLEAN", {"default": False, "tooltip": "Enable or bypass the Limiter."}),
                "limiter_ceiling_db": ("FLOAT", {"default": -0.1, "min": -6.0, "max": 0.0, "step": 0.01, "tooltip": "Maximum output peak level (in dBFS) the limiter will allow."}),
                "limiter_lookahead_ms": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 10.0, "step": 0.1, "tooltip": "Time (in ms) the limiter looks ahead to detect peaks and apply gain reduction preventively. Reduces clipping."}),
                "limiter_release_ms": ("FLOAT", {"default": 50.0, "min": 1.0, "max": 500.0, "step": 1.0, "tooltip": "Time (in ms) it takes for the limiter to smoothly return to unity gain after reducing a peak."}),
            }
        }

    RETURN_TYPES = ("AUDIO", "IMAGE", "IMAGE")
    FUNCTION = "apply_mastering_chain"
    CATEGORY = "MD_Nodes/Audio Processing"

    def _db_to_amplitude(self, db):
        """Converts decibels to an amplitude multiplier."""
        return 10**(db / 20.0)

    def _amplitude_to_db(self, amplitude):
        """Converts amplitude multiplier to decibels."""
        # Add a small epsilon to avoid log(0) for silent samples
        return 20 * np.log10(np.abs(amplitude) + 1e-9)

    def _apply_gain(self, audio_data, gain_db):
        """
        Applies a simple gain adjustment to the audio data.
        Supports both mono (1D) and stereo (2D: channels x samples) data.
        """
        if gain_db == 0.0:
            return audio_data
        gain_multiplier = self._db_to_amplitude(gain_db)
        return audio_data * gain_multiplier

    def _design_peaking_filter(self, gain_db, freq, Q, sample_rate):
        """
        Designs a peaking (bell) filter coefficients.
        Ensures float64 precision for coefficients.
        """
        A = 10**(gain_db / 40.0)
        omega = 2 * np.pi * freq / sample_rate
        sn = np.sin(omega)
        cs = np.cos(omega)
        alpha = sn / (2 * Q) if Q > 1e-6 else sn / (2 * 0.707)

        # Coefficients for peaking EQ filter (derived from Audio EQ Cookbook)
        b0 = 1 + alpha * A
        b1 = -2 * cs
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * cs
        a2 = 1 - alpha / A

        # Ensure coefficients are float64 for better numerical stability
        b = np.array([b0, b1, b2], dtype=np.float64) / a0
        a = np.array([1.0, a1 / a0, a2 / a0], dtype=np.float64)
        return signal.tf2sos(b, a) # Convert to SOS

    def _design_low_shelf_filter_rbj(self, gain_db, freq, sample_rate):
        """
        Designs a low-shelf filter coefficients based on RBJ Audio EQ Cookbook.
        Q is implicitly handled for standard shelf response (Q=0.707).
        """
        A = 10**(gain_db / 40.0) # Amplitude gain
        w0 = 2 * np.pi * freq / sample_rate
        cos_w0 = np.cos(w0)
        sin_w0 = np.sin(w0)
        
        # This alpha corresponds to Q = 1/sqrt(2) (0.707) for the shelf.
        # It's derived from sin(w0) / (2*Q) from the cookbook.
        alpha_val = sin_w0 / (2 * 0.707) 

        # Coefficients for Low-Shelf Filter
        b0_num = A * ( (A+1) - (A-1)*cos_w0 + 2*np.sqrt(A)*alpha_val )
        b1_num = 2 * A * ( (A-1) - (A+1)*cos_w0 )
        b2_num = A * ( (A+1) - (A-1)*cos_w0 - 2*np.sqrt(A)*alpha_val )
        
        a0_den = (A+1) + (A-1)*cos_w0 + 2*np.sqrt(A)*alpha_val
        a1_den = -2 * ( (A-1) + (A+1)*cos_w0 )
        a2_den = (A+1) + (A-1)*cos_w0 - 2*np.sqrt(A)*alpha_val

        # Normalize coefficients by a0_den
        b = np.array([b0_num, b1_num, b2_num], dtype=np.float64) / a0_den
        a = np.array([a0_den, a1_den, a2_den], dtype=np.float64) / a0_den # a0 should be 1.0
        
        return signal.tf2sos(b, a)


    def _design_high_shelf_filter_rbj(self, gain_db, freq, sample_rate):
        """
        Designs a high-shelf filter coefficients based on RBJ Audio EQ Cookbook.
        Q is implicitly handled for standard shelf response (Q=0.707).
        """
        A = 10**(gain_db / 40.0)
        w0 = 2 * np.pi * freq / sample_rate
        cos_w0 = np.cos(w0)
        sin_w0 = np.sin(w0)

        # This alpha corresponds to Q = 1/sqrt(2) (0.707) for the shelf.
        alpha_val = sin_w0 / (2 * 0.707) 

        # Coefficients for High-Shelf Filter
        b0_num = A * ( (A+1) + (A-1)*cos_w0 + 2*np.sqrt(A)*alpha_val )
        b1_num = -2 * A * ( (A-1) + (A+1)*cos_w0 )
        b2_num = A * ( (A+1) + (A-1)*cos_w0 - 2*np.sqrt(A)*alpha_val )

        a0_den = (A+1) - (A-1)*cos_w0 + 2*np.sqrt(A)*alpha_val
        a1_den = -2 * ( (A-1) + (A+1)*cos_w0 )
        a2_den = (A+1) - (A-1)*cos_w0 - 2*np.sqrt(A)*alpha_val
        
        # Normalize coefficients by a0_den
        b = np.array([b0_num, b1_num, b2_num], dtype=np.float64) / a0_den
        a = np.array([a0_den, a1_den, a2_den], dtype=np.float64) / a0_den # a0 should be 1.0
        
        return signal.tf2sos(b, a)


    def _apply_filters_to_audio(self, audio_data, sos_filters):
        """Applies a chain of SOS filters to audio data. sos_filters should be a list of SOS arrays."""
        if not sos_filters:
            return audio_data

        if audio_data.ndim == 1: # Mono
            filtered_audio = audio_data
            for sos in sos_filters:
                # Add a small epsilon before filtering if data is all zeros to prevent potential issues
                if np.all(filtered_audio == 0):
                    filtered_audio += 1e-12 # Add a tiny amount to avoid perfect zeros leading to NaNs later

                filtered_audio = signal.sosfilt(sos, filtered_audio)
                # Ensure no NaNs or Infs propagate immediately after each filter
                filtered_audio = np.nan_to_num(filtered_audio, nan=0.0, posinf=1.0, neginf=-1.0)
                filtered_audio = np.clip(filtered_audio, -1.0, 1.0)
        else: # Stereo (channels x samples)
            filtered_audio = np.zeros_like(audio_data)
            for i in range(audio_data.shape[0]):
                channel_data = audio_data[i, :]
                
                for sos in sos_filters:
                    # Add a small epsilon before filtering if data is all zeros
                    if np.all(channel_data == 0):
                        channel_data += 1e-12

                    channel_data = signal.sosfilt(sos, channel_data)
                    # Ensure no NaNs or Infs propagate immediately after each filter
                    channel_data = np.nan_to_num(channel_data, nan=0.0, posinf=1.0, neginf=-1.0)
                filtered_audio[i, :] = channel_data
            # Ensure the output stays within valid audio range [-1.0, 1.0]
            # This is a good practice after applying multiple filters
            filtered_audio = np.clip(filtered_audio, -1.0, 1.0)
        return filtered_audio

    def _apply_eq(self, audio_data, sample_rate,
                  eq_high_shelf_gain_db, eq_high_shelf_freq, eq_high_shelf_q, # eq_high_shelf_q is now nominal
                  enable_low_shelf_eq, eq_low_shelf_gain_db, eq_low_shelf_freq, eq_low_shelf_q, # eq_low_shelf_q is now nominal
                  enable_param_eq1, param_eq1_gain_db, param_eq1_freq, param_eq1_q,
                  enable_param_eq2, param_eq2_gain_db, param_eq2_freq, param_eq2_q,
                  enable_param_eq3, param_eq3_gain_db, param_eq3_freq, param_eq3_q,
                  enable_param_eq4, param_eq4_gain_db, param_eq4_freq, param_eq4_q):
        """
        Applies multiple EQ stages: High-shelf, Low-shelf, and four Parametric bands.
        Q for shelving filters is fixed at 0.707 for stability.
        """
        print("Applying EQ Filters...")
        
        all_sos_filters = []

        # High-Shelf Filter
        if eq_high_shelf_gain_db != 0.0:
            fixed_high_shelf_q = 0.707
            print(f"  - High-shelf: Gain={eq_high_shelf_gain_db}dB, Freq={eq_high_shelf_freq}Hz, Q={fixed_high_shelf_q} (fixed)")
            all_sos_filters.append(self._design_high_shelf_filter_rbj(eq_high_shelf_gain_db, eq_high_shelf_freq, sample_rate))

        # Low-Shelf Filter
        if enable_low_shelf_eq and eq_low_shelf_gain_db != 0.0:
            fixed_low_shelf_q = 0.707
            print(f"  - Low-shelf: Gain={eq_low_shelf_gain_db}dB, Freq={eq_low_shelf_freq}Hz, Q={fixed_low_shelf_q} (fixed)")
            all_sos_filters.append(self._design_low_shelf_filter_rbj(eq_low_shelf_gain_db, eq_low_shelf_freq, sample_rate))

        # Parametric EQ Band 1
        if enable_param_eq1 and param_eq1_gain_db != 0.0:
            print(f"  - Parametric EQ 1: Gain={param_eq1_gain_db}dB, Freq={param_eq1_freq}Hz, Q={param_eq1_q}")
            all_sos_filters.append(self._design_peaking_filter(param_eq1_gain_db, param_eq1_freq, param_eq1_q, sample_rate))

        # Parametric EQ Band 2
        if enable_param_eq2 and param_eq2_gain_db != 0.0:
            print(f"  - Parametric EQ 2: Gain={param_eq2_gain_db}dB, Freq={param_eq2_freq}Hz, Q={param_eq2_q}")
            all_sos_filters.append(self._design_peaking_filter(param_eq2_gain_db, param_eq2_freq, param_eq2_q, sample_rate))

        # Parametric EQ Band 3
        if enable_param_eq3 and param_eq3_gain_db != 0.0:
            print(f"  - Parametric EQ 3: Gain={param_eq3_gain_db}dB, Freq={param_eq3_freq}Hz, Q={param_eq3_q}")
            all_sos_filters.append(self._design_peaking_filter(param_eq3_gain_db, param_eq3_freq, param_eq3_q, sample_rate))

        # Parametric EQ Band 4
        if enable_param_eq4 and param_eq4_gain_db != 0.0:
            print(f"  - Parametric EQ 4: Gain={param_eq4_gain_db}dB, Freq={param_eq4_freq}Hz, Q={param_eq4_q}")
            all_sos_filters.append(self._design_peaking_filter(param_eq4_gain_db, param_eq4_freq, param_eq4_q, sample_rate))

        # Apply all collected filters sequentially
        return self._apply_filters_to_audio(audio_data, all_sos_filters)

    def _apply_single_band_compression(self, audio_data, threshold_db, ratio, attack_ms, release_ms, makeup_gain_db, soft_knee_db, sample_rate):
        """
        Applies single-band compression using Pedalboard's Compressor.
        Note: Pedalboard's Compressor does not directly expose a 'knee_db' parameter in its constructor.
        The soft_knee_db parameter from ComfyUI node is thus noted for user's information but not passed to Pedalboard.
        Makeup gain is applied separately after compression.
        """
        print(f"Applying Single-Band Compression with Pedalboard: Thresh={threshold_db}dB, Ratio={ratio}:1, Attack={attack_ms}ms, Release={release_ms}ms, Makeup={makeup_gain_db}dB, Soft Knee={soft_knee_db}dB (Note: Soft Knee not directly applied by Pedalboard Compressor constructor)")

        # Pedalboard expects samples as float32
        audio_data = audio_data.astype(np.float32)
        
        # Initialize Pedalboard Compressor with supported arguments.
        # The 'knee_db' parameter is NOT passed to the constructor as it's not supported.
        compressor = pedalboard.Compressor(
            threshold_db=threshold_db,
            ratio=ratio,
            attack_ms=attack_ms,
            release_ms=release_ms
        )

        # Pedalboard works with (num_samples, num_channels) if stereo, or (num_samples,) if mono
        # Our internal representation is (num_channels, num_samples), so transpose for stereo
        if audio_data.ndim == 2: # Stereo (channels, samples)
            processed_audio = compressor(audio_data.T, sample_rate=sample_rate).T
        else: # Mono (samples,)
            processed_audio = compressor(audio_data, sample_rate=sample_rate)
        
        # Apply makeup gain manually after compression
        if makeup_gain_db != 0.0:
            makeup_gain_linear = self._db_to_amplitude(makeup_gain_db)
            processed_audio *= makeup_gain_linear

        return processed_audio

    def _design_linkwitz_riley_crossover(self, cutoff_freq, sample_rate, order=4):
        """
        Designs a Linkwitz-Riley crossover filter (Butterworth squared).
        Returns low-pass and high-pass SOS filter coefficients.
        Order must be even. A 4th order LR filter uses two 2nd order Butterworth filters.
        Using `sosfiltfilt` will apply the filter twice (forward and backward)
        to ensure zero-phase response, which is ideal for crossovers.
        """
        if order % 2 != 0:
            raise ValueError("Linkwitz-Riley filter order must be even.")

        nyquist = 0.5 * sample_rate
        normal_cutoff = cutoff_freq / nyquist

        # Design Butterworth filter coefficients for half the desired LR order
        b_lp, a_lp = signal.butter(order // 2, normal_cutoff, btype='lowpass', analog=False)
        b_hp, a_hp = signal.butter(order // 2, normal_cutoff, btype='highpass', analog=False)

        # Convert to SOS (Second-Order Sections) for numerical stability
        sos_lp = signal.tf2sos(b_lp, a_lp)
        sos_hp = signal.tf2sos(b_hp, a_hp)

        return sos_lp, sos_hp

    def _apply_multiband_compression(self, audio_data, sample_rate,
                                      crossover_low_mid_hz, crossover_mid_high_hz,
                                      low_params, mid_params, high_params):
        """
        Implements a 3-band compressor using SciPy for band splitting and Pedalboard for per-band compression.
        """
        print(f"Applying Multiband Compression with Pedalboard: Low-Mid Crossover={crossover_low_mid_hz}Hz, Mid-High Crossover={crossover_mid_high_hz}Hz")

        # Ensure audio_data is at least 2D for consistent channel handling
        if audio_data.ndim == 1:
            audio_data = audio_data[np.newaxis, :] # Make it (1, num_samples)

        num_channels = audio_data.shape[0]
        num_samples = audio_data.shape[1]
        processed_audio = np.zeros_like(audio_data, dtype=np.float32) # Ensure float32 for Pedalboard

        # Design crossover filters (4th order Linkwitz-Riley by default, meaning 2nd order Butterworth applied twice)
        sos_low_mid_lp, sos_low_mid_hp = self._design_linkwitz_riley_crossover(crossover_low_mid_hz, sample_rate, order=4)
        sos_mid_high_lp, sos_mid_high_hp = self._design_linkwitz_riley_crossover(crossover_mid_high_hz, sample_rate, order=4)

        for c in range(num_channels):
            channel_audio = audio_data[c, :].astype(np.float32) # Ensure channel data is float32

            # --- Band Splitting using sosfiltfilt for zero-phase distortion ---
            # Low band: Low-pass at low-mid crossover
            low_band = signal.sosfiltfilt(sos_low_mid_lp, channel_audio)

            # High band: High-pass at mid-high crossover
            high_band = signal.sosfiltfilt(sos_mid_high_hp, channel_audio)

            # Mid band: High-pass at low-mid crossover AND Low-pass at mid-high crossover
            mid_band_hp = signal.sosfiltfilt(sos_low_mid_hp, channel_audio)
            mid_band = signal.sosfiltfilt(sos_mid_high_lp, mid_band_hp)

            # --- Apply Independent Pedalboard Compression to Each Band ---
            # Note: soft_knee_db is passed to _apply_single_band_compression but is not
            # used by Pedalboard's Compressor directly, as noted in that function.
            processed_low = self._apply_single_band_compression(
                low_band, low_params["threshold_db"], low_params["ratio"],
                low_params["attack_ms"], low_params["release_ms"],
                low_params["makeup_gain_db"], low_params["soft_knee_db"], sample_rate
            )
            processed_mid = self._apply_single_band_compression(
                mid_band, mid_params["threshold_db"], mid_params["ratio"],
                mid_params["attack_ms"], mid_params["release_ms"],
                mid_params["makeup_gain_db"], mid_params["soft_knee_db"], sample_rate
            )
            processed_high = self._apply_single_band_compression(
                high_band, high_params["threshold_db"], high_params["ratio"],
                high_params["attack_ms"], high_params["release_ms"],
                high_params["makeup_gain_db"], high_params["soft_knee_db"], sample_rate
            )

            # --- Recombine Bands ---
            processed_audio[c, :] = processed_low + processed_mid + processed_high
            # Clip the recombined audio to prevent values from going out of range after summation
            processed_audio[c, :] = np.clip(processed_audio[c, :], -1.0, 1.0)


        # If original was 1D, flatten back
        if audio_data.shape[0] == 1 and audio_data.ndim == 2 and processed_audio.ndim == 2:
            return processed_audio.flatten()
        return processed_audio

    def _apply_limiter(self, audio_data, ceiling_db, lookahead_ms, release_ms, sample_rate):
        """
        Applies brickwall peak limiting using Pedalboard's Limiter.
        Note: Pedalboard's Limiter handles lookahead internally.
        """
        print(f"Applying Limiting with Pedalboard: Ceiling={ceiling_db}dB, Lookahead={lookahead_ms}ms, Release={release_ms}ms")

        # Pedalboard expects samples as float32
        audio_data = audio_data.astype(np.float32)

        # Create Pedalboard Limiter instance
        # Pedalboard's Limiter automatically handles lookahead internally.
        # The lookahead_ms parameter here is passed for consistency but is informative only.
        limiter = pedalboard.Limiter(
            threshold_db=ceiling_db,
            release_ms=release_ms
        )

        # Pedalboard works with (num_samples, num_channels) if stereo, or (num_samples,) if mono
        # Our internal representation is (num_channels, num_samples), so transpose for stereo
        if audio_data.ndim == 2: # Stereo (channels, samples)
            # Transpose, apply limiter, then transpose back
            processed_audio = limiter(audio_data.T, sample_rate=sample_rate).T
        else: # Mono (samples,)
            processed_audio = limiter(audio_data, sample_rate=sample_rate)
        
        # Ensure the output is strictly within [-1.0, 1.0] as a final safety measure
        processed_audio = np.clip(processed_audio, -1.0, 1.0)

        return processed_audio


    def _plot_waveform_to_tensor(self, audio_data, sample_rate, title="Waveform"):
        """
        Plots a single waveform (mixing stereo to mono if necessary) and converts it
        to a PyTorch tensor suitable for ComfyUI IMAGE output.
        """
        plt.figure(figsize=(10, 4)) # Adjust figure size as needed for better visualization

        # If stereo, mix to mono for plotting a single waveform
        if audio_data.ndim == 2: # Stereo (channels x samples)
            # Average the channels to get a mono representation
            mono_audio = np.mean(audio_data, axis=0)
            plot_data = mono_audio
            plot_title = f"{title} (Mono Mix)"
        else: # Mono (1D)
            plot_data = audio_data
            plot_title = f"{title} (Mono)"
        
        # Plot the single waveform
        time_axis = np.linspace(0, len(plot_data) / sample_rate, len(plot_data))
        plt.plot(time_axis, plot_data, color='blue', linewidth=0.5)
        
        plt.title(plot_title)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        
        plt.xlim(0, len(plot_data) / sample_rate) # Ensure x-axis covers full duration
        plt.ylim(-1.05, 1.05) # Assume audio is normalized or within +/-1 range for plotting
        plt.grid(True)
        plt.tight_layout()

        # Save plot to a BytesIO object
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)
        buf.seek(0)
        plt.close(plt.gcf()) # Close the plot to free memory

        # Open the image using PIL and convert to PyTorch tensor
        img = Image.open(buf)
        img = img.convert("RGB") # Convert to RGB (3 channels)

        # Convert PIL Image to NumPy array, then to PyTorch tensor
        # ComfyUI expects [batch_size, height, width, channels], values 0-1 or 0-255
        img_np = np.array(img).astype(np.float32) / 255.0 # Normalize to 0-1
        img_tensor = torch.from_numpy(img_np).unsqueeze(0) # Add batch dimension

        return img_tensor


    def apply_mastering_chain(self, audio, sample_rate, master_gain_db,
                              enable_eq, eq_high_shelf_gain_db, eq_high_shelf_freq, eq_high_shelf_q,
                              enable_low_shelf_eq, eq_low_shelf_gain_db, eq_low_shelf_freq, eq_low_shelf_q,
                              enable_param_eq1, param_eq1_gain_db, param_eq1_freq, param_eq1_q,
                              enable_param_eq2, param_eq2_gain_db, param_eq2_freq, param_eq2_q,
                              enable_param_eq3, param_eq3_gain_db, param_eq3_freq, param_eq3_q,
                              enable_param_eq4, param_eq4_gain_db, param_eq4_freq, param_eq4_q,
                              enable_comp, comp_type,
                              comp_threshold_db, comp_ratio, comp_attack_ms, comp_release_ms, comp_makeup_gain_db, comp_soft_knee_db,
                              mb_crossover_low_mid_hz, mb_crossover_mid_high_hz,
                              mb_low_threshold_db, mb_low_ratio, mb_low_attack_ms, mb_low_release_ms, mb_low_makeup_gain_db, mb_low_soft_knee_db,
                              mb_mid_threshold_db, mb_mid_ratio, mb_mid_attack_ms, mb_mid_release_ms, mb_mid_makeup_gain_db, mb_mid_soft_knee_db,
                              mb_high_threshold_db, mb_high_ratio, mb_high_attack_ms, mb_high_release_ms, mb_high_makeup_gain_db, mb_high_soft_knee_db,
                              enable_limiter, limiter_ceiling_db, limiter_lookahead_ms, limiter_release_ms):
        """
        The main function that orchestrates the mastering chain.
        Processes the input audio through the enabled DSP stages.
        """
        print(f"Starting mastering chain for audio with sample rate: {sample_rate} Hz")

        # --- Handle incoming audio format ---
        audio_tensor_input = None
        if isinstance(audio, dict):
            if 'waveform' in audio:
                audio_tensor_input = audio['waveform']
            elif 'audio' in audio:
                audio_tensor_input = audio['audio']
            elif 'samples' in audio:
                audio_tensor_input = audio['samples']
            else:
                raise ValueError("Unexpected dictionary format for 'audio' input. Could not find audio tensor.")
        else:
            audio_tensor_input = audio

        audio_np = audio_tensor_input.cpu().numpy()

        # Original input could be 1D (mono), 2D (channels, samples), or 3D (batch, channels, samples)
        # We need to normalize it to 2D (channels, samples) for internal processing
        if audio_np.ndim == 3:
            # Assuming batch_size is 1 for most ComfyUI audio, take the first item
            audio_np = audio_np[0] # Now (num_channels, num_samples)
        if audio_np.ndim == 1:
            # If it's mono (num_samples,), convert to (1, num_samples) for consistent channel handling
            audio_np = audio_np[np.newaxis, :]

        # --- Generate Before Waveform Image ---
        before_waveform_image_tensor = self._plot_waveform_to_tensor(audio_np, sample_rate, "Original Waveform")


        # --- Mastering Chain Steps ---

        # 1. Global Gain
        processed_audio = self._apply_gain(audio_np, master_gain_db)
        processed_audio = np.clip(processed_audio, -1.0, 1.0) # Clip after global gain
        if np.isnan(processed_audio).any() or np.isinf(processed_audio).any():
            print("WARNING: NaN or Inf values detected after Global Gain.")

        # 2. Equalization
        if enable_eq:
            processed_audio = self._apply_eq(processed_audio, sample_rate,
                                             eq_high_shelf_gain_db, eq_high_shelf_freq, eq_high_shelf_q,
                                             enable_low_shelf_eq, eq_low_shelf_gain_db, eq_low_shelf_freq, eq_low_shelf_q,
                                             enable_param_eq1, param_eq1_gain_db, param_eq1_freq, param_eq1_q,
                                             enable_param_eq2, param_eq2_gain_db, param_eq2_freq, param_eq2_q,
                                             enable_param_eq3, param_eq3_gain_db, param_eq3_freq, param_eq3_q,
                                             enable_param_eq4, param_eq4_gain_db, param_eq4_freq, param_eq4_q)
            processed_audio = np.clip(processed_audio, -1.0, 1.0) # Clip after EQ
            if np.isnan(processed_audio).any() or np.isinf(processed_audio).any():
                print("WARNING: NaN or Inf values detected after EQ.")
        else:
            print("EQ Bypassed.")

        # 3. Compression (Single-Band or Multiband)
        if enable_comp:
            if comp_type == "Single-Band":
                processed_audio = self._apply_single_band_compression(
                    processed_audio, comp_threshold_db, comp_ratio,
                    comp_attack_ms, comp_release_ms, comp_makeup_gain_db, comp_soft_knee_db, sample_rate
                )
            elif comp_type == "Multiband":
                # Package multiband parameters into dictionaries for easier passing
                low_params = {
                    "threshold_db": mb_low_threshold_db, "ratio": mb_low_ratio,
                    "attack_ms": mb_low_attack_ms, "release_ms": mb_low_release_ms,
                    "makeup_gain_db": mb_low_makeup_gain_db, "soft_knee_db": mb_low_soft_knee_db
                }
                mid_params = {
                    "threshold_db": mb_mid_threshold_db, "ratio": mb_mid_ratio,
                    "attack_ms": mb_mid_attack_ms, "release_ms": mb_mid_release_ms,
                    "makeup_gain_db": mb_mid_makeup_gain_db, "soft_knee_db": mb_mid_soft_knee_db
                }
                high_params = {
                    "threshold_db": mb_high_threshold_db, "ratio": mb_high_ratio,
                    "attack_ms": mb_high_attack_ms, "release_ms": mb_high_release_ms,
                    "makeup_gain_db": mb_high_makeup_gain_db, "soft_knee_db": mb_high_soft_knee_db
                }
                processed_audio = self._apply_multiband_compression(
                    processed_audio, sample_rate,
                    mb_crossover_low_mid_hz, mb_crossover_mid_high_hz,
                    low_params, mid_params, high_params
                )
            processed_audio = np.clip(processed_audio, -1.0, 1.0) # Clip after Compression
            if np.isnan(processed_audio).any() or np.isinf(processed_audio).any():
                print("WARNING: NaN or Inf values detected after Compression.")
        else:
            print("Compressor Bypassed.")

        # 4. Limiting
        if enable_limiter:
            processed_audio = self._apply_limiter(processed_audio,
                                                  limiter_ceiling_db, limiter_lookahead_ms, limiter_release_ms, sample_rate)
            processed_audio = np.clip(processed_audio, -1.0, 1.0) # Clip after Limiting (though limiter should handle this)
            if np.isnan(processed_audio).any() or np.isinf(processed_audio).any():
                print("WARNING: NaN or Inf values detected after Limiting.")
        else:
            print("Limiter Bypassed.")

        # --- End Mastering Chain Steps ---

        # Generate After Waveform Image
        after_waveform_image_tensor = self._plot_waveform_to_tensor(processed_audio, sample_rate, "Processed Waveform")

        # Convert processed NumPy array back to PyTorch tensor for ComfyUI output
        # Ensure it's 2D (channels, samples) before adding the batch dimension
        if processed_audio.ndim == 1: # If mono (N_samples,)
            processed_audio_reshaped = processed_audio[np.newaxis, :] # -> (1, N_samples)
        elif processed_audio.ndim == 2: # If stereo (N_channels, N_samples)
            processed_audio_reshaped = processed_audio
        else: # Should not happen if intermediate processing is correct, but for robustness
            raise ValueError(f"Unexpected dimension for processed_audio: {processed_audio.ndim}. Expected 1 or 2.")

        # Convert to PyTorch tensor and add the batch dimension
        output_audio_tensor = torch.from_numpy(processed_audio_reshaped).to(audio_tensor_input.device).float()
        output_audio_tensor = output_audio_tensor.unsqueeze(0) # Add batch dimension: -> (1, num_channels, num_samples)

        # Wrap the output audio tensor and sample rate in a dictionary
        output_audio_dict = {
            "waveform": output_audio_tensor,
            "sample_rate": sample_rate
        }

        print("Mastering chain completed.")
        return (output_audio_dict, before_waveform_image_tensor, after_waveform_image_tensor)

# A dictionary that maps the node class name to its instance.
# ComfyUI uses this to discover your nodes.
NODE_CLASS_MAPPINGS = {
    "MasteringChainNode": MasteringChainNode
}

# A dictionary that provides human-readable names for the nodes.
NODE_DISPLAY_NAME_MAPPINGS = {
    "MasteringChainNode": "Custom Audio Mastering Chain"
}