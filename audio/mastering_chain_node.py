# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
# ████ MD_Nodes/MasteringChainNode – Professional Mastering Chain ████▓▒░
# © 2025 MDMAchine
# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
# ░▒▓ ORIGIN & DEV:
#   • Cast into the void by: MDMAchine (sonic architect)
#   • Enhanced by: Gemini (Adapter/Fixes), Claude (v1.3 Overhaul), MDMAchine (QA)
#   • License: Apache 2.0 — No cursed floppy disks allowed

# ░▒▓ DESCRIPTION:
#   Professional-grade mastering chain for ComfyUI. Features true stereo-linked
#   multiband compression, surgical EQ control (parametric, shelf, cutoff), and
#   transparent limiting. Optimized for post-LUFS normalized audio input.

# ░▒▓ FEATURES:
#   ✓ Global gain control
#   ✓ Butterworth low-pass and high-pass cutoff filters
#   ✓ High-shelf and low-shelf filters (RBJ coefficients)
#   ✓ Four parametric EQ bands
#   ✓ Single-band compressor (Pedalboard)
#   ✓ 3-band stereo-linked multiband compressor (Pedalboard)
#   ✓ Linkwitz-Riley crossovers with configurable order
#   ✓ Lookahead limiter (Pedalboard)
#   ✓ Mono/stereo support
#   ✓ Before/after waveform visualizations
#   ✓ Optimized defaults for normalized input (-14 LUFS)
#   ✓ Robust numerical stability (NaN/Inf protection)

# ░▒▓ CHANGELOG:
#   - v1.4.3a (Guide Update - Oct 2025):
#       • UPDATED: Converted to ComfyUI_MD_Nodes Guide v1.4.3a standards.
#       • FIXED: Standardized header, imports, docstrings, tooltips (Guide 5.1, 6.4, 5.2, 8.1).
#       • ADDED: Graceful failure handling in main function (Guide 7.3).
#       • VERIFIED: Correct plot-to-tensor method used (Guide 8.2).
#       • FIXED: Added `MD: ` prefix to display name (Guide 5.4).
#   - v1.3 (Original Base - Major Overhaul):
#       • Fixed stereo phasing with stereo-linked multiband compression
#       • Added true low-pass and high-pass cutoff filters (Butterworth)
#       • Corrected shelf filter coefficient normalization (RBJ cookbook)
#       • Fixed tensor boolean evaluation error in audio extraction
#       • Optimized compression defaults for -14 LUFS normalized input
#       • Added configurable crossover filter order (4-12th order)
#       • Improved numerical stability throughout signal chain

# ░▒▓ CONFIGURATION:
#   → Primary Use: Professional audio mastering for normalized material (-14 LUFS).
#   → Secondary Use: Creative audio shaping and frequency sculpting.
#   → Advanced Use: Stereo-linked multiband dynamics for mix glue.
#   → Edge Use: Applying specific modules (e.g., only EQ or only Limiter).

# ░▒▓ WARNING:
#   This node may trigger:
#   ▓▒░ Temporal distortion and memories of ANSI art
#   ▓▒░ Unstoppable urges to tweak Q factors at 3 AM
#   ▓▒░ Spontaneous understanding of Linkwitz-Riley topology
#   Consult your nearest demoscene vet if hallucinations persist.
# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄


# =================================================================================
# == Standard Library Imports                                                    ==
# =================================================================================
import io
import traceback

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
# == ComfyUI Core Modules                                                        ==
# =================================================================================
# (None imported)

# =================================================================================
# == Local Project Imports                                                       ==
# =================================================================================
# (None imported)

# =================================================================================
# == Helper Classes & Dependencies                                               ==
# =================================================================================
# (None defined)

# =================================================================================
# == Core Node Class                                                             ==
# =================================================================================

class MasteringChainNode:
    """
    Advanced ComfyUI audio mastering chain node adhering to MD_Nodes Guide v1.4.3a.
    Features EQ, single/multiband compression (stereo-linked), and limiting.
    """

    def __init__(self):
        """
        Initializes the MasteringChainNode instance.
        """
        pass # No instance state needed currently

    @classmethod
    def INPUT_TYPES(cls):
        """
        Define all input parameters with tooltips following Guide 8.1.
        """
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "Input audio data (AUDIO dictionary format)."}),
                "sample_rate": ("INT", {
                    "default": 44100, "min": 8000, "max": 192000, "step": 1,
                    "tooltip": (
                        "SAMPLE RATE (HZ)\n"
                        "- Sampling rate of the input audio.\n"
                        "- Ensure this matches the actual audio data."
                    )
                }),
                "master_gain_db": ("FLOAT", {
                    "default": 0.0, "min": -20.0, "max": 20.0, "step": 0.1,
                    "tooltip": (
                        "MASTER GAIN (DB)\n"
                        "- Overall gain adjustment applied at the beginning.\n"
                        "- Use for initial level setting before processing."
                    )
                }),
                "enable_eq": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "ENABLE EQUALIZER\n"
                        "- True: Activates the entire EQ section (Cutoffs, Shelves, Parametric).\n"
                        "- False: Bypasses the EQ section."
                    )
                }),
                "enable_lowpass": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable the low-pass cutoff filter."
                }),
                "lowpass_freq": ("FLOAT", {
                    "default": 18000.0, "min": 1000.0, "max": 20000.0, "step": 100.0,
                    "tooltip": "Low-pass filter cutoff frequency (Hz)."
                }),
                "lowpass_order": ("INT", {
                    "default": 4, "min": 2, "max": 8, "step": 2,
                    "tooltip": "Low-pass filter order (higher = steeper slope)."
                }),
                "enable_highpass": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable the high-pass cutoff filter."
                }),
                "highpass_freq": ("FLOAT", {
                    "default": 20.0, "min": 10.0, "max": 500.0, "step": 1.0,
                    "tooltip": "High-pass filter cutoff frequency (Hz)."
                }),
                "highpass_order": ("INT", {
                    "default": 4, "min": 2, "max": 8, "step": 2,
                    "tooltip": "High-pass filter order (higher = steeper slope)."
                }),
                "eq_high_shelf_gain_db": ("FLOAT", {
                    "default": 0.0, "min": -12.0, "max": 12.0, "step": 0.1,
                    "tooltip": "High-shelf filter gain (dB)."
                }),
                "eq_high_shelf_freq": ("FLOAT", {
                    "default": 12000.0, "min": 1000.0, "max": 20000.0, "step": 100.0,
                    "tooltip": "High-shelf filter corner frequency (Hz)."
                }),
                "enable_low_shelf_eq": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable the low-shelf EQ filter."
                }),
                "eq_low_shelf_gain_db": ("FLOAT", {
                    "default": 0.0, "min": -12.0, "max": 12.0, "step": 0.1,
                    "tooltip": "Low-shelf filter gain (dB)."
                }),
                "eq_low_shelf_freq": ("FLOAT", {
                    "default": 75.0, "min": 20.0, "max": 500.0, "step": 1.0,
                    "tooltip": "Low-shelf filter corner frequency (Hz)."
                }),
                "enable_param_eq1": ("BOOLEAN", {"default": False, "tooltip": "Enable Parametric EQ Band 1."}),
                "param_eq1_gain_db": ("FLOAT", {"default": 0.0, "min": -20.0, "max": 20.0, "step": 0.1, "tooltip":"EQ1 Gain (dB)"}),
                "param_eq1_freq": ("FLOAT", {"default": 55.0, "min": 20.0, "max": 20000.0, "step": 1.0, "tooltip":"EQ1 Frequency (Hz)"}),
                "param_eq1_q": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 20.0, "step": 0.1, "tooltip":"EQ1 Q-Factor (Width)"}),
                "enable_param_eq2": ("BOOLEAN", {"default": False, "tooltip": "Enable Parametric EQ Band 2."}),
                "param_eq2_gain_db": ("FLOAT", {"default": 0.0, "min": -20.0, "max": 20.0, "step": 0.1, "tooltip":"EQ2 Gain (dB)"}),
                "param_eq2_freq": ("FLOAT", {"default": 125.0, "min": 20.0, "max": 20000.0, "step": 1.0, "tooltip":"EQ2 Frequency (Hz)"}),
                "param_eq2_q": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 20.0, "step": 0.1, "tooltip":"EQ2 Q-Factor (Width)"}),
                "enable_param_eq3": ("BOOLEAN", {"default": False, "tooltip": "Enable Parametric EQ Band 3."}),
                "param_eq3_gain_db": ("FLOAT", {"default": 0.0, "min": -20.0, "max": 20.0, "step": 0.1, "tooltip":"EQ3 Gain (dB)"}),
                "param_eq3_freq": ("FLOAT", {"default": 1250.0, "min": 20.0, "max": 20000.0, "step": 1.0, "tooltip":"EQ3 Frequency (Hz)"}),
                "param_eq3_q": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 20.0, "step": 0.1, "tooltip":"EQ3 Q-Factor (Width)"}),
                "enable_param_eq4": ("BOOLEAN", {"default": False, "tooltip": "Enable Parametric EQ Band 4."}),
                "param_eq4_gain_db": ("FLOAT", {"default": 0.0, "min": -20.0, "max": 20.0, "step": 0.1, "tooltip":"EQ4 Gain (dB)"}),
                "param_eq4_freq": ("FLOAT", {"default": 5000.0, "min": 20.0, "max": 20000.0, "step": 1.0, "tooltip":"EQ4 Frequency (Hz)"}),
                "param_eq4_q": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 20.0, "step": 0.1, "tooltip":"EQ4 Q-Factor (Width)"}),
                "enable_comp": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "ENABLE COMPRESSOR\n"
                        "- True: Activates the selected compressor type.\n"
                        "- False: Bypasses compression."
                    )
                 }),
                "comp_type": (["Single-Band", "Multiband"], {
                    "default": "Multiband",
                    "tooltip": "Select compressor type (Single-Band or 3-Band Multiband)."
                 }),
                "comp_threshold_db": ("FLOAT", {
                    "default": -8.0, "min": -60.0, "max": 0.0, "step": 0.1,
                    "tooltip": (
                        "SINGLE-BAND THRESHOLD (DB)\n"
                        "- Level above which compression starts.\n"
                        "- Default optimized for normalized input (-14 LUFS)."
                    )
                 }),
                "comp_ratio": ("FLOAT", {"default": 2.5, "min": 1.0, "max": 20.0, "step": 0.1, "tooltip":"Single-Band Ratio"}),
                "comp_attack_ms": ("FLOAT", {"default": 20.0, "min": 1.0, "max": 500.0, "step": 1.0, "tooltip":"Single-Band Attack (ms)"}),
                "comp_release_ms": ("FLOAT", {"default": 250.0, "min": 10.0, "max": 2000.0, "step": 10.0, "tooltip":"Single-Band Release (ms)"}),
                "comp_makeup_gain_db": ("FLOAT", {"default": 0.0, "min": -12.0, "max": 12.0, "step": 0.1, "tooltip":"Single-Band Makeup Gain (dB)"}),
                "mb_crossover_low_mid_hz": ("FLOAT", {
                    "default": 250.0, "min": 20.0, "max": 1000.0, "step": 10.0,
                    "tooltip":"Multiband: Low/Mid Crossover (Hz)"
                }),
                "mb_crossover_mid_high_hz": ("FLOAT", {
                    "default": 4000.0, "min": 1000.0, "max": 15000.0, "step": 100.0,
                    "tooltip":"Multiband: Mid/High Crossover (Hz)"
                }),
                "mb_crossover_order": ("INT", {
                    "default": 8, "min": 4, "max": 12, "step": 2,
                    "tooltip": (
                        "MULTIBAND CROSSOVER ORDER\n"
                        "- Steepness of Linkwitz-Riley crossover filters.\n"
                        "- Higher = steeper slope, potentially more phase shift."
                    )
                 }),
                "mb_low_threshold_db": ("FLOAT", {
                    "default": -10.0, "min": -60.0, "max": 0.0, "step": 0.1,
                    "tooltip":"Multiband Low Band: Threshold (dB)"
                 }),
                "mb_low_ratio": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 20.0, "step": 0.1, "tooltip":"Multiband Low Band: Ratio"}),
                "mb_low_attack_ms": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 500.0, "step": 1.0, "tooltip":"Multiband Low Band: Attack (ms)"}),
                "mb_low_release_ms": ("FLOAT", {"default": 300.0, "min": 10.0, "max": 2000.0, "step": 10.0, "tooltip":"Multiband Low Band: Release (ms)"}),
                "mb_low_makeup_gain_db": ("FLOAT", {"default": 0.0, "min": -12.0, "max": 12.0, "step": 0.1, "tooltip":"Multiband Low Band: Makeup Gain (dB)"}),
                "mb_mid_threshold_db": ("FLOAT", {
                    "default": -8.0, "min": -60.0, "max": 0.0, "step": 0.1,
                    "tooltip":"Multiband Mid Band: Threshold (dB)"
                 }),
                "mb_mid_ratio": ("FLOAT", {"default": 2.5, "min": 1.0, "max": 20.0, "step": 0.1, "tooltip":"Multiband Mid Band: Ratio"}),
                "mb_mid_attack_ms": ("FLOAT", {"default": 20.0, "min": 1.0, "max": 500.0, "step": 1.0, "tooltip":"Multiband Mid Band: Attack (ms)"}),
                "mb_mid_release_ms": ("FLOAT", {"default": 180.0, "min": 10.0, "max": 2000.0, "step": 10.0, "tooltip":"Multiband Mid Band: Release (ms)"}),
                "mb_mid_makeup_gain_db": ("FLOAT", {"default": 0.0, "min": -12.0, "max": 12.0, "step": 0.1, "tooltip":"Multiband Mid Band: Makeup Gain (dB)"}),
                "mb_high_threshold_db": ("FLOAT", {
                    "default": -6.0, "min": -60.0, "max": 0.0, "step": 0.1,
                    "tooltip":"Multiband High Band: Threshold (dB)"
                 }),
                "mb_high_ratio": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 20.0, "step": 0.1, "tooltip":"Multiband High Band: Ratio"}),
                "mb_high_attack_ms": ("FLOAT", {"default": 10.0, "min": 1.0, "max": 500.0, "step": 1.0, "tooltip":"Multiband High Band: Attack (ms)"}),
                "mb_high_release_ms": ("FLOAT", {"default": 120.0, "min": 10.0, "max": 2000.0, "step": 10.0, "tooltip":"Multiband High Band: Release (ms)"}),
                "mb_high_makeup_gain_db": ("FLOAT", {"default": 0.0, "min": -12.0, "max": 12.0, "step": 0.1, "tooltip":"Multiband High Band: Makeup Gain (dB)"}),
                "enable_limiter": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "ENABLE LIMITER\n"
                        "- True: Activates the final brickwall limiter.\n"
                        "- False: Bypasses limiting."
                    )
                 }),
                "limiter_ceiling_db": ("FLOAT", {
                    "default": -0.1, "min": -6.0, "max": 0.0, "step": 0.01,
                    "tooltip": "Limiter Ceiling (dB): Maximum output level."
                }),
                "limiter_release_ms": ("FLOAT", {
                    "default": 50.0, "min": 1.0, "max": 500.0, "step": 1.0,
                    "tooltip": "Limiter Release Time (ms)."
                }),
            }
        }

    RETURN_TYPES = ("AUDIO", "IMAGE", "IMAGE")
    RETURN_NAMES = ("audio", "waveform_before", "waveform_after")
    FUNCTION = "apply_mastering_chain"
    CATEGORY = "MD_Nodes/Audio Processing"

    # No IS_CHANGED method defined: Use default ComfyUI caching behavior.

    # ==================== UTILITY METHODS ====================

    def _db_to_amplitude(self, db):
        """Converts dB to linear amplitude."""
        return 10**(db / 20.0)

    def _amplitude_to_db(self, amplitude):
        """Converts linear amplitude to dB."""
        return 20 * np.log10(np.maximum(1e-9, np.abs(amplitude))) # Avoid log(0)

    def _apply_gain(self, audio_data, gain_db):
        """
        Applies gain in dB to audio data.

        Args:
            audio_data (np.array): Audio data.
            gain_db (float): Gain to apply in dB.

        Returns:
            np.array: Audio data with gain applied.
        """
        if gain_db == 0.0:
            return audio_data
        return audio_data * self._db_to_amplitude(gain_db)

    # ==================== FILTER DESIGN ====================

    def _design_lowpass_filter(self, cutoff_freq, sample_rate, order=4):
        """
        Designs a Butterworth low-pass filter.

        Args:
            cutoff_freq (float): Cutoff frequency in Hz.
            sample_rate (int): Sample rate in Hz.
            order (int): Filter order.

        Returns:
            np.array: SOS (second-order sections) filter coefficients.
        """
        nyquist = 0.5 * sample_rate
        normal_cutoff = np.clip(cutoff_freq / nyquist, 0.01, 0.99) # Clip to valid range
        b, a = signal.butter(order, normal_cutoff, btype='lowpass', analog=False)
        return signal.tf2sos(b, a)

    def _design_highpass_filter(self, cutoff_freq, sample_rate, order=4):
        """
        Designs a Butterworth high-pass filter.

        Args:
            cutoff_freq (float): Cutoff frequency in Hz.
            sample_rate (int): Sample rate in Hz.
            order (int): Filter order.

        Returns:
            np.array: SOS (second-order sections) filter coefficients.
        """
        nyquist = 0.5 * sample_rate
        normal_cutoff = np.clip(cutoff_freq / nyquist, 0.01, 0.99) # Clip to valid range
        b, a = signal.butter(order, normal_cutoff, btype='highpass', analog=False)
        return signal.tf2sos(b, a)

    def _design_peaking_filter(self, gain_db, freq, Q, sample_rate):
        """
        Designs a peaking (bell) EQ filter using RBJ cookbook formulas.

        Args:
            gain_db (float): Gain in dB.
            freq (float): Center frequency in Hz.
            Q (float): Q factor (bandwidth).
            sample_rate (int): Sample rate in Hz.

        Returns:
            np.array or None: SOS filter coefficients, or None if gain is negligible.
        """
        if abs(gain_db) < 0.001 or Q <= 0 or freq <= 0 or freq >= sample_rate / 2:
            return None # Invalid parameters or no gain needed
            
        A = 10**(gain_db / 40.0)
        omega = 2 * np.pi * freq / sample_rate
        sn = np.sin(omega)
        cs = np.cos(omega)
        alpha = sn / (2 * Q)
        a0 = 1 + alpha / A

        # Prevent division by near-zero a0
        if abs(a0) < 1e-9:
             print(f"[WARN] Peaking filter a0 near zero for F={freq}, G={gain_db}, Q={Q}. Skipping filter.")
             return None

        b0 = 1 + alpha * A
        b1 = -2 * cs
        b2 = 1 - alpha * A
        a1 = -2 * cs
        a2 = 1 - alpha / A

        b = np.array([b0 / a0, b1 / a0, b2 / a0], dtype=np.float64)
        a = np.array([1.0, a1 / a0, a2 / a0], dtype=np.float64)

        # Check for stability issues
        if not np.all(np.isfinite(b)) or not np.all(np.isfinite(a)):
             print(f"[WARN] Peaking filter coefficients invalid for F={freq}, G={gain_db}, Q={Q}. Skipping filter.")
             return None

        return signal.tf2sos(b, a)

    def _design_low_shelf_filter(self, gain_db, freq, sample_rate):
        """
        Designs a low-shelf filter using corrected RBJ cookbook formulas.

        Args:
            gain_db (float): Gain in dB.
            freq (float): Corner frequency in Hz.
            sample_rate (int): Sample rate in Hz.

        Returns:
            np.array or None: SOS filter coefficients, or None if gain is negligible.
        """
        if abs(gain_db) < 0.001 or freq <= 0 or freq >= sample_rate / 2:
            return None
            
        A = 10**(gain_db / 40.0)
        w0 = 2 * np.pi * freq / sample_rate
        cos_w0 = np.cos(w0)
        sin_w0 = np.sin(w0)
        alpha = sin_w0 / (2 * 0.707) # Using S=1/sqrt(2) for standard shelf
        a0 = (A+1) + (A-1)*cos_w0 + 2*np.sqrt(A)*alpha

        if abs(a0) < 1e-9:
             print(f"[WARN] Low-shelf filter a0 near zero for F={freq}, G={gain_db}. Skipping filter.")
             return None

        b0 = A * ((A+1) - (A-1)*cos_w0 + 2*np.sqrt(A)*alpha)
        b1 = 2 * A * ((A-1) - (A+1)*cos_w0)
        b2 = A * ((A+1) - (A-1)*cos_w0 - 2*np.sqrt(A)*alpha)
        a1 = -2 * ((A-1) + (A+1)*cos_w0)
        a2 = (A+1) + (A-1)*cos_w0 - 2*np.sqrt(A)*alpha

        b = np.array([b0 / a0, b1 / a0, b2 / a0], dtype=np.float64)
        a = np.array([1.0, a1 / a0, a2 / a0], dtype=np.float64)

        if not np.all(np.isfinite(b)) or not np.all(np.isfinite(a)):
             print(f"[WARN] Low-shelf filter coefficients invalid for F={freq}, G={gain_db}. Skipping filter.")
             return None

        return signal.tf2sos(b, a)

    def _design_high_shelf_filter(self, gain_db, freq, sample_rate):
        """
        Designs a high-shelf filter using corrected RBJ cookbook formulas.

        Args:
            gain_db (float): Gain in dB.
            freq (float): Corner frequency in Hz.
            sample_rate (int): Sample rate in Hz.

        Returns:
            np.array or None: SOS filter coefficients, or None if gain is negligible.
        """
        if abs(gain_db) < 0.001 or freq <= 0 or freq >= sample_rate / 2:
            return None
            
        A = 10**(gain_db / 40.0)
        w0 = 2 * np.pi * freq / sample_rate
        cos_w0 = np.cos(w0)
        sin_w0 = np.sin(w0)
        alpha = sin_w0 / (2 * 0.707) # Using S=1/sqrt(2) for standard shelf
        a0 = (A+1) - (A-1)*cos_w0 + 2*np.sqrt(A)*alpha

        if abs(a0) < 1e-9:
             print(f"[WARN] High-shelf filter a0 near zero for F={freq}, G={gain_db}. Skipping filter.")
             return None

        b0 = A * ((A+1) + (A-1)*cos_w0 + 2*np.sqrt(A)*alpha)
        b1 = -2 * A * ((A-1) + (A+1)*cos_w0)
        b2 = A * ((A+1) + (A-1)*cos_w0 - 2*np.sqrt(A)*alpha)
        a1 = 2 * ((A-1) - (A+1)*cos_w0)
        a2 = (A+1) - (A-1)*cos_w0 - 2*np.sqrt(A)*alpha

        b = np.array([b0 / a0, b1 / a0, b2 / a0], dtype=np.float64)
        a = np.array([1.0, a1 / a0, a2 / a0], dtype=np.float64)

        if not np.all(np.isfinite(b)) or not np.all(np.isfinite(a)):
             print(f"[WARN] High-shelf filter coefficients invalid for F={freq}, G={gain_db}. Skipping filter.")
             return None

        return signal.tf2sos(b, a)

    def _design_linkwitz_riley_crossover(self, cutoff_freq, sample_rate, order=8):
        """
        Designs Linkwitz-Riley crossover filters (low-pass and high-pass SOS pairs).

        Args:
            cutoff_freq (float): Crossover frequency in Hz.
            sample_rate (int): Sample rate in Hz.
            order (int): Filter order (must be even, >= 4).

        Returns:
            tuple: (sos_lowpass, sos_highpass) SOS filter coefficients.
        """
        if order % 2 != 0:
            order = max(4, order + 1) # Ensure even order, min 4
            print(f"[WARN] Crossover order must be even, using order {order}.")
            
        nyquist = 0.5 * sample_rate
        normal_cutoff = np.clip(cutoff_freq / nyquist, 0.01, 0.99) # Clip to valid range

        try:
            # Design N/2 order Butterworth
            b_lp, a_lp = signal.butter(order // 2, normal_cutoff, btype='lowpass')
            b_hp, a_hp = signal.butter(order // 2, normal_cutoff, btype='highpass')

            # Convert to SOS and square (cascade) for Linkwitz-Riley
            sos_lp = signal.tf2sos(b_lp, a_lp)
            sos_hp = signal.tf2sos(b_hp, a_hp)

            # Basic stability check
            if np.any(np.abs(np.roots(a_lp)) >= 1.0) or np.any(np.abs(np.roots(a_hp)) >= 1.0):
                 raise ValueError("Intermediate Butterworth filter unstable.")

            # Return cascaded SOS sections
            return np.vstack([sos_lp, sos_lp]), np.vstack([sos_hp, sos_hp])
        except ValueError as e:
            print(f"[ERROR] Crossover design failed at {cutoff_freq}Hz, Order {order}: {e}")
            raise # Re-raise error to stop processing

    def _apply_filters_to_audio(self, audio_data, sos_filters):
        """
        Applies a list of SOS filters sequentially using sosfiltfilt (zero-phase).

        Args:
            audio_data (np.array): Audio data ([channels, samples] or [samples]).
            sos_filters (list): List of SOS filter coefficients arrays.

        Returns:
            np.array: Filtered audio data, clipped and NaN protected.
        """
        if not sos_filters or all(s is None for s in sos_filters):
            return audio_data

        filtered_audio = audio_data.copy()
        
        # Apply filters using sosfiltfilt for zero phase shift
        for sos in sos_filters:
             if sos is not None:
                try:
                    filtered_audio = signal.sosfiltfilt(sos, filtered_audio, axis=-1) # Apply along sample axis
                except ValueError as e:
                     # Often happens with very short audio clips
                     print(f"[WARN] sosfiltfilt failed: {e}. Filter possibly unstable or audio too short. Skipping filter.")
                     # Continue with the audio as it was before this filter
                     continue # Skip to the next filter

        # Protect against NaN/Inf and clip
        filtered_audio = np.nan_to_num(filtered_audio, nan=0.0, posinf=1.0, neginf=-1.0)
        return np.clip(filtered_audio, -1.0, 1.0)

    # ==================== PROCESSING STAGES ====================

    def _apply_eq(self, audio_data, sample_rate, **params):
        """
        Applies the complete EQ chain (Cutoffs, Shelves, Parametrics).

        Args:
            audio_data (np.array): Audio data ([channels, samples]).
            sample_rate (int): Sample rate.
            **params: Dictionary containing all EQ parameters from INPUT_TYPES.

        Returns:
            np.array: Audio data after EQ.
        """
        print("Applying EQ Filters...")
        sos_filters = []

        # Cutoff Filters (Apply first)
        if params.get('enable_lowpass'):
            print(f"  - Low-pass Cutoff: {params['lowpass_freq']:.1f}Hz, Order={params['lowpass_order']}")
            sos_filters.append(self._design_lowpass_filter(
                params['lowpass_freq'], sample_rate, params['lowpass_order']
            ))
        if params.get('enable_highpass'):
            print(f"  - High-pass Cutoff: {params['highpass_freq']:.1f}Hz, Order={params['highpass_order']}")
            sos_filters.append(self._design_highpass_filter(
                params['highpass_freq'], sample_rate, params['highpass_order']
            ))

        # Shelving Filters
        sos = self._design_high_shelf_filter(params['eq_high_shelf_gain_db'],
                                             params['eq_high_shelf_freq'], sample_rate)
        if sos is not None:
            print(f"  - High-shelf: {params['eq_high_shelf_gain_db']:+.1f}dB @ {params['eq_high_shelf_freq']:.1f}Hz")
            sos_filters.append(sos)

        if params.get('enable_low_shelf_eq'):
            sos = self._design_low_shelf_filter(params['eq_low_shelf_gain_db'],
                                                params['eq_low_shelf_freq'], sample_rate)
            if sos is not None:
                print(f"  - Low-shelf: {params['eq_low_shelf_gain_db']:+.1f}dB @ {params['eq_low_shelf_freq']:.1f}Hz")
                sos_filters.append(sos)

        # Parametric Bands
        for i in range(1, 5):
            if params.get(f'enable_param_eq{i}'):
                gain = params[f'param_eq{i}_gain_db']
                freq = params[f'param_eq{i}_freq']
                q_val = params[f'param_eq{i}_q']
                sos = self._design_peaking_filter(gain, freq, q_val, sample_rate)
                if sos is not None:
                    print(f"  - Parametric {i}: {gain:+.1f}dB @ {freq:.1f}Hz, Q={q_val:.2f}")
                    sos_filters.append(sos)

        return self._apply_filters_to_audio(audio_data, sos_filters)

    def _apply_single_band_compression(self, audio_data, threshold_db, ratio,
                                       attack_ms, release_ms, makeup_gain_db, sample_rate):
        """
        Applies single-band compression using Pedalboard.

        Args:
            audio_data (np.array): Audio data ([channels, samples] or [samples]).
            threshold_db (float): Compressor threshold.
            ratio (float): Compressor ratio.
            attack_ms (float): Attack time.
            release_ms (float): Release time.
            makeup_gain_db (float): Makeup gain (applied *after* compression).
            sample_rate (int): Sample rate.

        Returns:
            np.array: Compressed audio data.
        """
        print(f"  Single-band Compressor: Thresh={threshold_db:.1f}dB, Ratio={ratio:.1f}:1, "
              f"Attack={attack_ms:.1f}ms, Release={release_ms:.1f}ms, Makeup={makeup_gain_db:+.1f}dB")
        
        # Pedalboard expects float32
        audio_float = audio_data.astype(np.float32)
        
        try:
            compressor = pedalboard.Compressor(
                threshold_db=threshold_db,
                ratio=ratio,
                attack_ms=attack_ms,
                release_ms=release_ms
            )
            # Pedalboard processes [samples, channels]
            if audio_float.ndim == 2 and audio_float.shape[0] < audio_float.shape[1]: # Detect [channels, samples]
                 processed = compressor(audio_float.T, sample_rate=sample_rate).T
            else: # Assume [samples] or [samples, channels]
                 processed = compressor(audio_float, sample_rate=sample_rate)

            # Apply makeup gain *after* compression
            if makeup_gain_db != 0.0:
                gain_processor = pedalboard.Gain(gain_db=makeup_gain_db)
                if processed.ndim == 2 and processed.shape[0] < processed.shape[1]:
                     processed = gain_processor(processed.T, sample_rate=sample_rate).T
                else:
                     processed = gain_processor(processed, sample_rate=sample_rate)

            return processed
        except Exception as e:
            print(f"[ERROR] Single-band compression failed: {e}")
            traceback.print_exc()
            return audio_float # Return original on error

    def _apply_multiband_compression_stereo_linked(self, audio_data, sample_rate,
                                                   crossover_low_mid, crossover_mid_high,
                                                   crossover_order, low_params, mid_params, high_params):
        """
        Applies 3-band compression with stereo-linked gain reduction using Pedalboard.

        Args:
            audio_data (np.array): Audio data ([channels, samples]).
            sample_rate (int): Sample rate.
            crossover_low_mid (float): Low/mid crossover freq (Hz).
            crossover_mid_high (float): Mid/high crossover freq (Hz).
            crossover_order (int): Linkwitz-Riley filter order.
            low_params (dict): Threshold, ratio, attack, release, makeup for low band.
            mid_params (dict): Parameters for mid band.
            high_params (dict): Parameters for high band.

        Returns:
            np.array: Compressed audio data ([channels, samples]).
        """
        print(f"Applying Stereo-Linked Multiband Compression...")
        print(f"  Crossovers: {crossover_low_mid:.1f}Hz | {crossover_mid_high:.1f}Hz (Order={crossover_order})")

        num_channels = audio_data.shape[0]
        is_stereo = num_channels == 2
        processed_audio_accumulator = np.zeros_like(audio_data, dtype=np.float32)

        try:
            # Design crossovers (returns cascaded SOS)
            sos_lm_lp, sos_lm_hp = self._design_linkwitz_riley_crossover(crossover_low_mid, sample_rate, crossover_order)
            sos_mh_lp, sos_mh_hp = self._design_linkwitz_riley_crossover(crossover_mid_high, sample_rate, crossover_order)
        except ValueError:
             # Error already printed in design function
             return audio_data # Return original if crossovers fail

        # --- Split all channels into bands using zero-phase filtering ---
        bands = {'low': [], 'mid': [], 'high': []}
        temp_mid_bands = [] # Store intermediate mid band result

        # Low band (LPF @ low_mid)
        bands['low'] = signal.sosfiltfilt(sos_lm_lp, audio_data, axis=-1)

        # High band (HPF @ mid_high)
        bands['high'] = signal.sosfiltfilt(sos_mh_hp, audio_data, axis=-1)

        # Mid band (HPF @ low_mid -> LPF @ mid_high)
        temp_mid_bands = signal.sosfiltfilt(sos_lm_hp, audio_data, axis=-1)
        bands['mid'] = signal.sosfiltfilt(sos_mh_lp, temp_mid_bands, axis=-1)

        # --- Compress each band (Stereo Linked if applicable) ---
        for band_name, params in [('low', low_params), ('mid', mid_params), ('high', high_params)]:
            print(f"  {band_name.upper()} band: Thresh={params['threshold_db']:.1f}dB, Ratio={params['ratio']:.1f}:1, "
                  f"Attack={params['attack_ms']:.1f}ms, Release={params['release_ms']:.1f}ms, Makeup={params['makeup_gain_db']:+.1f}dB")

            band_audio = bands[band_name].astype(np.float32) # Ensure float32 for Pedalboard

            # Apply compression using the single-band helper function
            # Pedalboard handles stereo linking internally if input has 2 channels
            compressed_band = self._apply_single_band_compression(
                 band_audio, params['threshold_db'], params['ratio'],
                 params['attack_ms'], params['release_ms'],
                 params['makeup_gain_db'], sample_rate
            )

            # Accumulate the processed bands
            processed_audio_accumulator += compressed_band

        # Clip final summed output
        processed_audio = np.clip(processed_audio_accumulator, -1.0, 1.0)
        processed_audio = np.nan_to_num(processed_audio, nan=0.0, posinf=1.0, neginf=-1.0) # Final safety check

        return processed_audio

    def _apply_limiter(self, audio_data, ceiling_db, release_ms, sample_rate):
        """
        Applies a brickwall limiter using Pedalboard.

        Args:
            audio_data (np.array): Audio data ([channels, samples] or [samples]).
            ceiling_db (float): Limiter threshold/ceiling.
            release_ms (float): Limiter release time.
            sample_rate (int): Sample rate.

        Returns:
            np.array: Limited audio data.
        """
        print(f"Applying Limiter: Ceiling={ceiling_db:.2f}dB, Release={release_ms:.1f}ms")
        
        audio_float = audio_data.astype(np.float32)
        
        try:
            limiter = pedalboard.Limiter(
                threshold_db=ceiling_db,
                release_ms=release_ms
            )
            # Pedalboard processes [samples, channels]
            if audio_float.ndim == 2 and audio_float.shape[0] < audio_float.shape[1]: # Detect [channels, samples]
                 processed = limiter(audio_float.T, sample_rate=sample_rate).T
            else: # Assume [samples] or [samples, channels]
                 processed = limiter(audio_float, sample_rate=sample_rate)

            # Although Pedalboard's limiter shouldn't exceed threshold, clip as safety
            return np.clip(processed, -1.0, 1.0)
        except Exception as e:
            print(f"[ERROR] Limiter failed: {e}")
            traceback.print_exc()
            return np.clip(audio_float, -1.0, 1.0) # Return clipped original on error

    def _plot_waveform_to_tensor(self, audio_data, sample_rate, title="Waveform"):
        """
        Plots waveform and returns as tensor using the reliable PIL buffer method.

        Args:
            audio_data (np.array): Audio data ([channels, samples] or [samples]).
            sample_rate (int): Sample rate.
            title (str): Plot title.

        Returns:
            torch.Tensor: Image tensor [1, height, width, 3] or placeholder on error.
        """
        if audio_data is None or audio_data.size == 0:
            print(f"[Plotting] Skipping plot '{title}': Empty audio data")
            return torch.zeros((1, 64, 64, 3), dtype=torch.float32)

        fig = None # Initialize fig to None for error handling
        try:
            plt.style.use('dark_background')
            fig, ax = plt.subplots(figsize=(10, 3)) # Reduced height slightly

            # Use mono or first channel for plotting
            if audio_data.ndim == 2:
                plot_data = audio_data[0, :] # Assume [channels, samples]
            else:
                plot_data = audio_data # Assume [samples]

            if plot_data.size == 0:
                 print(f"[Plotting] Skipping plot '{title}': Empty data after channel selection.")
                 plt.close(fig)
                 return torch.zeros((1, 64, 64, 3), dtype=torch.float32)

            time_axis = np.linspace(0, len(plot_data) / sample_rate, num=len(plot_data))
            ax.plot(time_axis, plot_data, color='#87CEEB', linewidth=0.5)

            # Calculate stats for title
            peak_val = np.max(np.abs(plot_data)) if plot_data.size > 0 else 0.0
            rms = np.sqrt(np.mean(plot_data**2)) if plot_data.size > 0 else 0.0

            ax.set_title(f"{title} | Peak: {peak_val:.3f} | RMS: {rms:.3f}", fontsize=10)
            ax.set_xlabel("Time (s)", fontsize=8)
            ax.set_ylabel("Amplitude", fontsize=8)
            ax.set_xlim(0, len(plot_data) / sample_rate)
            ax.set_ylim(-1.05, 1.05)
            ax.set_yticks([-1, -0.5, 0, 0.5, 1])
            ax.grid(True, ls=':', lw=0.5, alpha=0.3, color='gray')
            ax.tick_params(axis='both', which='major', labelsize=7)
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            plt.tight_layout()

            # Convert using reliable PIL buffer method (Guide 8.2)
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=96, facecolor=fig.get_facecolor())
            buf.seek(0)
            img = Image.open(buf).convert("RGB")
            img_np = np.array(img).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_np).unsqueeze(0)
            buf.close()
            plt.close(fig) # CRITICAL: Close figure

            return img_tensor

        except Exception as e:
            print(f"[Plotting] Error during plot generation for '{title}': {e}")
            traceback.print_exc()
            if fig:
                 plt.close(fig) # Ensure figure is closed even on error
            return torch.zeros((1, 64, 64, 3), dtype=torch.float32) # Return placeholder

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
        """
        Main execution function for the mastering chain.

        Args:
            (All inputs defined in INPUT_TYPES)

        Returns:
            tuple: (output_audio, waveform_before, waveform_after) following RETURN_TYPES.
                   On error, returns (original_audio, blank_image, blank_image).
        """
        blank_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
        before_plot = blank_image
        after_plot = blank_image

        try:
            print(f"\n{'='*60}")
            print(f"MASTERING CHAIN (MD_Nodes v1.4.3a spec) | Sample Rate: {sample_rate} Hz")
            print(f"{'='*60}\n")

            # --- Input Validation and Preparation ---
            if not isinstance(audio, dict) or 'waveform' not in audio:
                raise ValueError("Input 'audio' must be a dictionary with a 'waveform' tensor.")

            audio_tensor = audio['waveform']
            if not isinstance(audio_tensor, torch.Tensor):
                 raise TypeError("Input 'audio[waveform]' must be a torch.Tensor.")

            # Ensure tensor is on CPU and float32 for processing
            audio_np = audio_tensor.cpu().float().numpy()

            # Remove batch dimension if present, expect [channels, samples]
            if audio_np.ndim == 3 and audio_np.shape[0] == 1:
                audio_np = audio_np[0]
            elif audio_np.ndim != 2:
                 # Handle mono input [samples] -> [1, samples]
                 if audio_np.ndim == 1:
                      audio_np = audio_np[np.newaxis, :]
                 else:
                      raise ValueError(f"Input waveform has unexpected dimensions: {audio_np.shape}. Expected [channels, samples].")

            if audio_np.shape[-1] == 0: # Check for empty audio along sample dimension
                 print("[WARN] Input audio is empty. Returning original input.")
                 return (audio, before_plot, after_plot)

            print(f"Input shape (NumPy): {audio_np.shape} [channels, samples]")

            # Generate before plot (must happen after validation)
            before_plot = self._plot_waveform_to_tensor(audio_np, sample_rate, "Original Waveform")

            # --- Processing Chain ---
            processed = audio_np.copy()

            # 1. Global Gain
            if master_gain_db != 0.0:
                print(f"Applying Global Gain: {master_gain_db:+.1f} dB")
                processed = self._apply_gain(processed, master_gain_db)
                processed = np.clip(processed, -1.0, 1.0) # Clip after gain

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
                # Clipping handled within _apply_filters_to_audio

            # 3. Compression
            if enable_comp:
                print("Applying Compression...")
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
                # Clipping handled within compression functions

            # 4. Limiting
            if enable_limiter:
                processed = self._apply_limiter(processed, limiter_ceiling_db, limiter_release_ms, sample_rate)
                # Clipping handled within limiter function

            # --- Final Output ---
            print(f"Processed shape (NumPy): {processed.shape}")

            # Generate after plot
            after_plot = self._plot_waveform_to_tensor(processed, sample_rate, "Processed Waveform")

            # Convert back to [batch, channels, samples] tensor on original device
            output_tensor = torch.from_numpy(processed).to(audio_tensor.device).float().unsqueeze(0)

            output_audio = {
                "waveform": output_tensor,
                "sample_rate": sample_rate
            }

            print(f"\n{'='*60}")
            print("MASTERING CHAIN COMPLETE")
            print(f"{'='*60}\n")

            # Return tuple matching RETURN_TYPES
            return (output_audio, before_plot, after_plot)

        except Exception as e:
            # Graceful Failure (Guide 7.3)
            print(f"[ERROR] MasteringChainNode failed: {e}")
            traceback.print_exc()
            # Return original audio (passthrough) and blank images on error
            return (audio, before_plot if before_plot is not None else blank_image, blank_image)


# =================================================================================
# == Node Registration                                                           ==
# =================================================================================

NODE_CLASS_MAPPINGS = {
    "MasteringChainNode": MasteringChainNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Added MD: prefix per Guide 5.4
    "MasteringChainNode": "MD: Mastering Chain"
}