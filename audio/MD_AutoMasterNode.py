# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
# ████ MD_Nodes/AudioAutoMasterPro – Iterative Mastering Chain v6.8 ████▓▒░
# © 2025 MDMAchine
# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
# ░▒▓ ORIGIN & DEV:
#   • Cast into the void by: MDMAchine
#   • Enhanced by: Gemini, Claude
#   • License: Apache 2.0 — Sharing is caring
#   • Original source (if applicable): N/A

# ░▒▓ DESCRIPTION:
#   An all-in-one, iterative audio mastering node. It automatically analyzes
#   and processes audio using a chain of filtering, iterative spectral EQ,
#   de-essing, 3-band compression, stereo widening, and limiting to precisely
#   hit a target LUFS and spectral profile.

# ░▒▓ FEATURES:
#   ✓ Hits a user-defined target_lufs (via Pyloudnorm).
#   ✓ Iterative spectral EQ to tame harsh highs and muddy lows.
#   ✓ 3-band multiband compressor with Linkwitz-Riley crossovers.
#   ✓ Built-in de-esser, stereo widener, and brickwall limiter (via Pedalboard).
#   ✓ 5+ mastering presets (Standard, Podcast, Aggressive, Gentle, etc.).
#   ✓ Returns a detailed text log and before/after waveform images.

# ░▒▓ CHANGELOG:
#   - v6.8 (Current Release - Stability):
#       • FIXED: Robust NaN/Inf checks in filters, EQ, and analysis functions.
#       • FIXED: Filter design validation (Nyquist, divide-by-zero).
#       • FIXED: Plotting errors for empty or silent audio clips.
#   - v6.7 (Internal):
#       • FIXED: Pedalboard compressor/limiter call signature.

# ░▒▓ CONFIGURATION:
#   → Primary Use: One-click "Standard" or "Podcast" profile mastering to -14 LUFS.
#   → Secondary Use: "Aggressive" profile for loud EDM/Demoscene tracks.
#   → Edge Use: Disabling all modules (EQ, MBC, etc) to use it as a "smart" target_lufs normalizer.

# ░▒▓ WARNING:
#   This node may trigger:
#   ▓▒░ A sudden, god-like complex after turning a muddy .wav into a -14 LUFS banger.
#   ▓▒░ Compulsively checking the analysis log, whispering "just one more iteration."
#   ▓▒░ Realizing your "Podcast" preset sounds better on your music than your "Mastering" preset.
#   ▓▒░ A deep, existential dread when the iterative EQ hits `max_iterations` and just... gives up.
#   Consult your nearest demoscene vet if hallucinations persist.
# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄


# =================================================================================
# == Standard Library Imports                                                  ==
# =================================================================================
import io
import logging
import secrets
import traceback

# =================================================================================
# == Third-Party Imports                                                       ==
# =================================================================================
import torch
import numpy as np
import pyloudnorm as pln
import librosa
from scipy import signal
from scipy.signal import iirfilter, lfilter
from pedalboard import Pedalboard, Compressor, Limiter, Gain
import matplotlib as mpl
# CRITICAL: Set non-interactive backend BEFORE importing pyplot
mpl.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

# =================================================================================
# == ComfyUI Core Modules                                                      ==
# =================================================================================
# (No core modules needed for this node, but would be imported here)
# import comfy.model_management
# import nodes

# =================================================================================
# == Local Project Imports                                                     ==
# =================================================================================
# (None)

# =================================================================================
# == Helper Classes & Dependencies                                             ==
# =================================================================================
# (None)

# =================================================================================
# == Core Node Class                                                           ==
# =================================================================================

class MD_AutoMasterNode:
    """
    Audio Auto Master Pro (MD_AutoMasterNode)
    
    An all-in-one, iterative audio mastering node. It automatically analyzes
    and processes audio to hit a target LUFS and spectral profile using a
    chain of EQ, multi-band compression, de-essing, and limiting.
    """
    
    def __init__(self):
        self.analysis_log = []

    @classmethod
    def INPUT_TYPES(cls):
        """
        Define all input parameters with tooltips.
        
        Note: DO NOT use type hints in function signatures or global mappings.
        ComfyUI's dynamic loader cannot handle forward references at import time.
        """
        return {
            "required": {
                "audio": ("AUDIO", {
                    "tooltip": (
                        "AUDIO INPUT\n"
                        "- The audio data to process.\n"
                        "- Connect from a VAE Decode (Audio) or Load Audio node."
                    )
                }),
                "target_lufs": ("FLOAT", {
                    "default": -14.0, "min": -24.0, "max": -6.0, "step": 0.1,
                    "tooltip": (
                        "TARGET LOUDNESS (LUFS)\n"
                        "- The target integrated loudness (EBU R 128).\n"
                        "- -14.0 LUFS is standard for streaming.\n"
                        "- -16.0 LUFS is common for podcasts."
                    )
                }),
                "profile": (["Custom", "Standard", "Aggressive", "Podcast (Clarity)", "Gentle (Tame)", "Mastering (Transparent)"], {
                    "default": "Standard",
                    "tooltip": (
                        "MASTERING PROFILE\n"
                        "- Select a preset to auto-configure all parameters.\n"
                        "- 'Custom' uses the values set in the node."
                    )
                }),
            },
            "optional": {
                "input_gain_db": ("FLOAT", {
                    "default": 0.0, "min": -12.0, "max": 12.0, "step": 0.1,
                    "tooltip": (
                        "INPUT GAIN (dB)\n"
                        "- Pre-processing gain adjustment.\n"
                        "- Use this to boost quiet audio or tame loud audio *before* processing."
                    )
                }),
                "highpass_freq": ("FLOAT", {
                    "default": 0, "min": 0, "max": 500, "step": 1,
                    "tooltip": (
                        "HIGH-PASS FILTER (Hz)\n"
                        "- Applies a steep (zero-phase) high-pass filter.\n"
                        "- 0 = Disabled.\n"
                        "- Recommended: 20-40Hz to remove sub-sonic rumble."
                    )
                }),
                "lowpass_freq": ("FLOAT", {
                    "default": 0, "min": 0, "max": 20000, "step": 100,
                    "tooltip": (
                        "LOW-PASS FILTER (Hz)\n"
                        "- Applies a steep (zero-phase) low-pass filter.\n"
                        "- 0 = Disabled.\n"
                        "- Recommended: 18000-20000Hz to remove aliasing/hiss."
                    )
                }),
                "do_eq": ("BOOLEAN", {
                    "default": True, "label_on": "Run Iterative EQ", "label_off": "Skip EQ",
                    "tooltip": (
                        "ENABLE ITERATIVE EQ\n"
                        "- True: Runs the spectral balancing EQ.\n"
                        "- False: Skips the EQ step."
                    )
                }),
                "eq_bass_target": ("FLOAT", {
                    "default": 1.5, "min": 0.0, "max": 10.0, "step": 0.1,
                    "tooltip": (
                        "EQ BASS TARGET\n"
                        "- Target for low-end energy (arbitrary units).\n"
                        "- Lower = Tighter, less bass.\n"
                        "- Higher = Fuller, more bass."
                    )
                }),
                "eq_high_target": ("FLOAT", {
                    "default": 0.4, "min": 0.0, "max": 10.0, "step": 0.1,
                    "tooltip": (
                        "EQ HIGH TARGET\n"
                        "- Target for high-end energy (arbitrary units).\n"
                        "- Lower = Darker, less highs.\n"
                        "- Higher = Brighter, more highs/sibilance."
                    )
                }),
                "eq_adaptive": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "ADAPTIVE EQ\n"
                        "- True: Scales EQ cuts based on how far energy is from the target.\n"
                        "- False: Applies a fixed cut amount per iteration."
                    )
                }),
                "do_deess": ("BOOLEAN", {
                    "default": True, "label_on": "Run De-Esser", "label_off": "Skip De-Esser",
                    "tooltip": (
                        "ENABLE DE-ESSER\n"
                        "- True: Applies manual notch filters to reduce sibilance (5-10kHz).\n"
                        "- False: Skips the de-esser step."
                    )
                }),
                "deess_amount_db": ("FLOAT", {
                    "default": -10.0, "min": -40.0, "max": 0.0, "step": 0.5,
                    "tooltip": (
                        "DE-ESSER AMOUNT (dB)\n"
                        "- Maximum cut (in dB) applied by the manual de-esser filters."
                    )
                }),
                "do_mbc": ("BOOLEAN", {
                    "default": True, "label_on": "Run Single-Pass MBC", "label_off": "Skip MBC",
                    "tooltip": (
                        "ENABLE MULTI-BAND COMPRESSION\n"
                        "- True: Runs the 3-band compressor.\n"
                        "- False: Skips the MBC step."
                    )
                }),
                "mbc_crossover_low": ("FLOAT", {
                    "default": 300, "min": 100, "max": 1000, "step": 10,
                    "tooltip": "MBC Crossover: Low/Mid (Hz)"
                }),
                "mbc_crossover_high": ("FLOAT", {
                    "default": 3000, "min": 1000, "max": 8000, "step": 100,
                    "tooltip": "MBC Crossover: Mid/High (Hz)"
                }),
                "mbc_crossover_order": ("INT", {
                    "default": 8, "min": 4, "max": 12, "step": 2,
                    "tooltip": (
                        "MBC CROSSOVER ORDER\n"
                        "- Steepness of the Linkwitz-Riley crossover filters.\n"
                        "- Higher = Steeper, less band overlap (more phase shift).\n"
                        "- Must be an even number."
                    )
                }),
                "mbc_low_thresh_db": ("FLOAT", {
                    "default": -18.0, "min": -60.0, "max": 0.0, "step": 0.5,
                    "tooltip": "MBC LOW BAND: Threshold (dB)"
                }),
                "mbc_low_ratio": ("FLOAT", {
                    "default": 2.0, "min": 1.0, "max": 20.0, "step": 0.1,
                    "tooltip": "MBC LOW BAND: Ratio"
                }),
                "mbc_mid_thresh_db": ("FLOAT", {
                    "default": -18.0, "min": -60.0, "max": 0.0, "step": 0.5,
                    "tooltip": "MBC MID BAND: Threshold (dB)"
                }),
                "mbc_mid_ratio": ("FLOAT", {
                    "default": 2.0, "min": 1.0, "max": 20.0, "step": 0.1,
                    "tooltip": "MBC MID BAND: Ratio"
                }),
                "mbc_high_thresh_db": ("FLOAT", {
                    "default": -18.0, "min": -60.0, "max": 0.0, "step": 0.5,
                    "tooltip": "MBC HIGH BAND: Threshold (dB)"
                }),
                "mbc_high_ratio": ("FLOAT", {
                    "default": 2.0, "min": 1.0, "max": 20.0, "step": 0.1,
                    "tooltip": "MBC HIGH BAND: Ratio"
                }),
                "do_limiter": ("BOOLEAN", {
                    "default": True, "label_on": "Run Final Limiter", "label_off": "Skip Limiter",
                    "tooltip": (
                        "ENABLE FINAL LIMITER\n"
                        "- True: Applies a fast brickwall limiter at the end of the chain.\n"
                        "- False: Skips the limiter (may result in clipping)."
                    )
                }),
                "limiter_threshold_db": ("FLOAT", {
                    "default": -0.1, "min": -3.0, "max": 0.0, "step": 0.1,
                    "tooltip": (
                        "LIMITER CEILING (dB)\n"
                        "- The final brickwall ceiling for the audio.\n"
                        "- Recommended: -0.1 to -1.0 to prevent inter-sample peaks."
                    )
                }),
                "stereo_width": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": (
                        "STEREO WIDTH\n"
                        "- < 1.0: Narrows the stereo image (0.0 = Mono).\n"
                        "- 1.0: No change.\n"
                        "- > 1.0: Widens the stereo image (can cause phase issues)."
                    )
                }),
                "max_iterations_eq": ("INT", {
                    "default": 5, "min": 1, "max": 20,
                    "tooltip": (
                        "EQ MAX ITERATIONS\n"
                        "- Safety cap for the iterative EQ loop.\n"
                        "- Prevents infinite loops if targets cannot be met."
                    )
                }),
                "fast_mode": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "FAST MODE\n"
                        "- True: Skips intermediate normalization steps.\n"
                        "- False: Normalizes after EQ and before MBC.\n"
                        "- Fast mode is quicker but may be less accurate."
                    )
                }),
                "mix": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": (
                        "DRY/WET MIX\n"
                        "- Controls the blend between original (dry) and processed (wet) audio.\n"
                        "- 1.0 = 100% Processed.\n"
                        "- 0.0 = 100% Original."
                    )
                }),
            }
        }

    # @classmethod
    # def IS_CHANGED(cls, **kwargs):
        # """
        # Control cache behavior. This node performs side-effect-like audio processing
        # and analysis, so it should always re-run to provide fresh analysis.
        # """
        # # Force re-run (Approach #3 from guide)
        # return secrets.token_hex(16)

    RETURN_TYPES = ("AUDIO", "STRING", "IMAGE", "IMAGE")
    RETURN_NAMES = ("audio", "analysis_details", "waveform_before", "waveform_after")
    FUNCTION = "master_audio"
    CATEGORY = "MD_Nodes/Audio Processing"

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
        # Ensure input is numpy array for calculations
        if isinstance(audio, torch.Tensor):
            audio_np = audio.cpu().numpy()
        else:
            audio_np = audio

        # Proceed only if audio_np is actually a numpy array and not None or empty
        if isinstance(audio_np, np.ndarray) and audio_np.size > 0:
            clips = np.sum(np.abs(audio_np) > 0.99)
            peak = np.max(np.abs(audio_np))
            if clips > 0:
                self.analysis_log.append(f"⚠️ WARNING: {clips} samples clipped after {stage_name} (peak: {peak:.3f})")
        else:
            self.analysis_log.append(f"⚠️ Skipping clipping check after {stage_name}: Invalid audio data")


    def _analyze(self, audio, sr, meter_obj):
        analysis_channel = audio[:, 0] if audio.ndim > 1 else audio
        # Ensure analysis_channel is not empty before STFT
        if analysis_channel.size == 0:
            self.analysis_log.append("⚠️ Analysis skipped: Empty audio channel data")
            return {"bass": np.nan, "high": np.nan} # Return NaN or defaults

        stft = np.abs(librosa.stft(analysis_channel))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=stft.shape[0]*2-2) # Ensure n_fft calculation is correct
        
        # Add checks for frequency ranges existing
        bass_mask = freqs < 100
        high_mask = freqs > 10000
        
        bass_energy = np.mean(stft[bass_mask]) if np.any(bass_mask) else 0.0
        high_energy = np.mean(stft[high_mask]) if np.any(high_mask) else 0.0
        
        self.analysis_log.append(f"📊 Analysis: Bass={bass_energy:.2f}, Highs={high_energy:.2f}")
        return {"bass": bass_energy, "high": high_energy}


    def _apply_filters(self, audio, sr, highpass_freq, lowpass_freq):
        try:
            if highpass_freq > 0:
                # Ensure frequency is valid
                if highpass_freq >= sr / 2:
                    self.analysis_log.append(f"⚠️ Highpass freq {highpass_freq}Hz >= Nyquist ({sr/2}Hz). Skipping.")
                else:
                    sos = signal.butter(4, highpass_freq, btype='highpass', fs=sr, output='sos')
                    audio = signal.sosfiltfilt(sos, audio, axis=0)
                    self.analysis_log.append(f"🔊 Applied highpass filter @ {highpass_freq} Hz (Zero-Phase)")
            
            if lowpass_freq > 0 and lowpass_freq < sr / 2:
                sos = signal.butter(4, lowpass_freq, btype='lowpass', fs=sr, output='sos')
                audio = signal.sosfiltfilt(sos, audio, axis=0)
                self.analysis_log.append(f"🔉 Applied lowpass filter @ {lowpass_freq} Hz (Zero-Phase)")
            elif lowpass_freq >= sr / 2:
                self.analysis_log.append(f"⚠️ Lowpass freq {lowpass_freq}Hz >= Nyquist ({sr/2}Hz). Skipping.")

        except ValueError as e:
            self.analysis_log.append(f"❌ Error applying filters: {e}. Check frequencies.")
            # Optionally return original audio or raise error
            # return audio 
            raise ValueError(f"Filter design failed: {e}")
        
        return audio


    def _design_shelf_filter_coeffs(self, gain_db, freq, sample_rate, shelf_type='low'):
        if abs(gain_db) < 0.001 or freq <= 0 or freq >= sample_rate / 2: return None, None
        try:
            A = 10**(gain_db / 40.0)
            w0 = 2 * np.pi * freq / sample_rate
            cos_w0 = np.cos(w0)
            sin_w0 = np.sin(w0)
            alpha = sin_w0 / 2.0 * np.sqrt(2)
            # Add small epsilon to denominators to prevent division by zero
            epsilon = 1e-12 
            if shelf_type == 'low':
                a0 = (A+1)+(A-1)*cos_w0+2*np.sqrt(A)*alpha + epsilon
                if abs(a0) < epsilon: raise ValueError("Shelf filter denominator a0 is too close to zero.")
                b0 = A*((A+1)-(A-1)*cos_w0+2*np.sqrt(A)*alpha); b1=2*A*((A-1)-(A+1)*cos_w0); b2=A*((A+1)-(A-1)*cos_w0-2*np.sqrt(A)*alpha)
                a1=-2*((A-1)+(A+1)*cos_w0); a2=(A+1)+(A-1)*cos_w0-2*np.sqrt(A)*alpha
            elif shelf_type == 'high':
                a0 = (A+1)-(A-1)*cos_w0+2*np.sqrt(A)*alpha + epsilon
                if abs(a0) < epsilon: raise ValueError("Shelf filter denominator a0 is too close to zero.")
                b0 = A*((A+1)+(A-1)*cos_w0+2*np.sqrt(A)*alpha); b1=-2*A*((A-1)+(A+1)*cos_w0); b2=A*((A+1)+(A-1)*cos_w0-2*np.sqrt(A)*alpha)
                a1=2*((A-1)-(A+1)*cos_w0); a2=(A+1)-(A-1)*cos_w0-2*np.sqrt(A)*alpha
            else: return None, None
            b = np.array([b0/a0,b1/a0,b2/a0], dtype=np.float64)
            a = np.array([1.0,a1/a0,a2/a0], dtype=np.float64)
            # Check for NaN/Inf in coefficients
            if not np.all(np.isfinite(b)) or not np.all(np.isfinite(a)):
                raise ValueError("Shelf filter coefficients are not finite.")
            return b, a
        except (ValueError, FloatingPointError) as e:
            self.analysis_log.append(f"⚠️ Shelf filter design failed for G={gain_db}dB, F={freq}Hz: {e}")
            return None, None


    def _apply_eq(self, audio, sr, adjustments, adaptive=False):
        processed_audio = audio.copy()
        if adjustments.get("bass_cut_db"):
            cut_amount = adjustments["bass_cut_db"]
            if adaptive and "bass_scale" in adjustments: cut_amount *= adjustments["bass_scale"]
            b, a = self._design_shelf_filter_coeffs(cut_amount, 120, sr, shelf_type='low')
            if b is not None and a is not None:
                try:
                    # Apply filter and check for NaNs/Infs
                    filtered = lfilter(b, a, processed_audio, axis=0)
                    if not np.all(np.isfinite(filtered)):
                        self.analysis_log.append(f"⚠️ NaN/Inf detected after low-shelf filter. Skipping this EQ step.")
                    else:
                        processed_audio = filtered
                        self.analysis_log.append(f"🎛️ Applied EQ: Low-shelf cut of {cut_amount:.1f} dB @ 120 Hz (RBJ)")
                except Exception as e:
                    self.analysis_log.append(f"❌ Error applying low-shelf filter: {e}")
            else: self.analysis_log.append(f"⚠️ Skipped EQ: Low-shelf calc failed")

        if adjustments.get("high_cut_db"):
            cut_amount = adjustments["high_cut_db"]
            if adaptive and "high_scale" in adjustments: cut_amount *= adjustments["high_scale"]
            b, a = self._design_shelf_filter_coeffs(cut_amount, 8000, sr, shelf_type='high')
            if b is not None and a is not None:
                try:
                    # Apply filter and check for NaNs/Infs
                    filtered = lfilter(b, a, processed_audio, axis=0)
                    if not np.all(np.isfinite(filtered)):
                        self.analysis_log.append(f"⚠️ NaN/Inf detected after high-shelf filter. Skipping this EQ step.")
                    else:
                        processed_audio = filtered
                        self.analysis_log.append(f"🎛️ Applied EQ: High-shelf cut of {cut_amount:.1f} dB @ 8000 Hz (RBJ)")
                except Exception as e:
                    self.analysis_log.append(f"❌ Error applying high-shelf filter: {e}")
            else: self.analysis_log.append(f"⚠️ Skipped EQ: High-shelf calc failed")
            
        return processed_audio


    def _design_linkwitz_riley_crossover(self, cutoff_freq, sample_rate, order=8):
        if order % 2 != 0: order = order + 1
        nyquist = 0.5 * sample_rate
        normal_cutoff = np.clip(cutoff_freq / nyquist, 0.01, 0.99)
        # Check stability before returning
        try:
            b_lp, a_lp = signal.butter(order // 2, normal_cutoff, btype='lowpass')
            b_hp, a_hp = signal.butter(order // 2, normal_cutoff, btype='highpass')
            sos_lp = signal.tf2sos(b_lp, a_lp)
            sos_hp = signal.tf2sos(b_hp, a_hp)
            # Basic stability check (more advanced checks exist)
            if np.any(np.abs(np.roots(a_lp)) >= 1.0) or np.any(np.abs(np.roots(a_hp)) >= 1.0):
                raise ValueError("Crossover filter design resulted in unstable poles.")
            return [sos_lp, sos_lp], [sos_hp, sos_hp]
        except ValueError as e:
            self.analysis_log.append(f"❌ Crossover design failed at {cutoff_freq}Hz, Order {order}: {e}")
            raise # Re-raise to stop processing


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
            # Return original audio for this band to allow summing to continue
            return audio_float


    def _apply_multiband_compression_manual(self, audio_data, sample_rate, 
                                            crossover_low_mid, crossover_mid_high,
                                            crossover_order, low_params, mid_params, high_params):
        self.analysis_log.append(f"🗜️ Applying Manual MBC (Single Pass)...") 
        self.analysis_log.append(f"  Crossovers: {crossover_low_mid}Hz | {crossover_mid_high}Hz (Order={crossover_order})")
        self.analysis_log.append(f"  Params: Low(T:{low_params['threshold_db']:.1f} R:{low_params['ratio']:.1f}) | Mid(T:{mid_params['threshold_db']:.1f} R:{mid_params['ratio']:.1f}) | High(T:{high_params['threshold_db']:.1f} R:{high_params['ratio']:.1f})")
        is_stereo = audio_data.ndim > 1 and audio_data.shape[1] == 2
        if is_stereo: audio_ch = audio_data.T
        else: audio_ch = audio_data[np.newaxis, :]
        num_channels = audio_ch.shape[0]
        processed_audio_accumulator = np.zeros_like(audio_ch, dtype=np.float64) 
        
        try: # Wrap crossover design in try/except
            sos_lm_lp_list, sos_lm_hp_list = self._design_linkwitz_riley_crossover(crossover_low_mid, sample_rate, crossover_order)
            sos_mh_lp_list, sos_mh_hp_list = self._design_linkwitz_riley_crossover(crossover_mid_high, sample_rate, crossover_order)
        except ValueError as e:
            self.analysis_log.append(f"❌ Critical Error: MBC Crossover design failed. Skipping MBC. {e}")
            # Return original audio if crossovers fail
            return audio_data 
            
        for c in range(num_channels):
            ch_data = audio_ch[c, :].astype(np.float64) 
            try: # Wrap filtering per channel
                low_band = signal.sosfiltfilt(sos_lm_lp_list[0], ch_data); low_band = signal.sosfiltfilt(sos_lm_lp_list[1], low_band)
                high_band = signal.sosfiltfilt(sos_mh_hp_list[0], ch_data); high_band = signal.sosfiltfilt(sos_mh_hp_list[1], high_band)
                mid_band_tmp = signal.sosfiltfilt(sos_lm_hp_list[0], ch_data); mid_band_tmp = signal.sosfiltfilt(sos_lm_hp_list[1], mid_band_tmp)
                mid_band = signal.sosfiltfilt(sos_mh_lp_list[0], mid_band_tmp); mid_band = signal.sosfiltfilt(sos_mh_lp_list[1], mid_band)
                
                # Check for NaN/Inf after filtering
                if not np.all(np.isfinite(low_band)) or not np.all(np.isfinite(mid_band)) or not np.all(np.isfinite(high_band)):
                    self.analysis_log.append(f"⚠️ NaN/Inf detected after crossover filtering on channel {c}. Skipping MBC for this channel.")
                    processed_audio_accumulator[c, :] = ch_data # Use original data for this channel
                    continue # Skip compression and summing for this channel

            except Exception as e:
                self.analysis_log.append(f"❌ Error during crossover filtering on channel {c}: {e}. Skipping MBC for this channel.")
                processed_audio_accumulator[c, :] = ch_data
                continue

            base_attack_release = [(10.0, 200.0), (10.0, 150.0), (5.0, 100.0)] # Milliseconds
            
            comp_low = self._apply_single_band_compression(low_band, low_params['threshold_db'], low_params['ratio'], base_attack_release[0][0], base_attack_release[0][1], 0.0, sample_rate)
            comp_mid = self._apply_single_band_compression(mid_band, mid_params['threshold_db'], mid_params['ratio'], base_attack_release[1][0], base_attack_release[1][1], 0.0, sample_rate)
            comp_high = self._apply_single_band_compression(high_band, high_params['threshold_db'], high_params['ratio'], base_attack_release[2][0], base_attack_release[2][1], 0.0, sample_rate)
            
            # Sum bands, ensuring finite values
            summed_bands = comp_low.astype(np.float64) + comp_mid.astype(np.float64) + comp_high.astype(np.float64)
            if not np.all(np.isfinite(summed_bands)):
                self.analysis_log.append(f"⚠️ NaN/Inf detected after summing compressed bands on channel {c}. Using original data for channel.")
                processed_audio_accumulator[c, :] = ch_data
            else:
                processed_audio_accumulator[c, :] = summed_bands

        if is_stereo: return processed_audio_accumulator.T.astype(np.float32)
        else: return processed_audio_accumulator[0, :].astype(np.float32)
            
            
    def _apply_deesser_manual(self, audio, sr, max_cut_db):
        self.analysis_log.append(f"✂️ Manual De-Esser: Applying cuts up to {max_cut_db:.1f} dB")
        processed_audio = audio.copy()
        deess_targets = [(5500, 2.5), (7500, 3.0), (9500, 3.5)]
        for freq, Q in deess_targets:
            gain_for_filter = max_cut_db 
            if freq >= sr / 2: self.analysis_log.append(f"  - Skipping De-ess freq {freq}Hz"); continue
            try:
                A = 10**(gain_for_filter / 40.0); w0 = 2 * np.pi * freq / sr; cos_w0 = np.cos(w0); sin_w0 = np.sin(w0); safe_Q = max(Q, 0.1); alpha = sin_w0 / (2 * safe_Q)
                b0 = 1 + alpha * A; b1 = -2 * cos_w0; b2 = 1 - alpha * A; a0 = 1 + alpha / A; a1 = -2 * cos_w0; a2 = 1 - alpha / A
                # Add epsilon for safety
                epsilon = 1e-12
                a0 += epsilon
                if abs(a0) < epsilon: raise ValueError("De-ess filter denominator a0 is too close to zero.")
                b = np.array([b0 / a0, b1 / a0, b2 / a0], dtype=np.float64); a = np.array([1.0, a1 / a0, a2 / a0], dtype=np.float64)
                if not np.all(np.isfinite(b)) or not np.all(np.isfinite(a)): raise ValueError("De-ess filter coefficients are not finite.")
                
                filtered = lfilter(b, a, processed_audio, axis=0)
                if not np.all(np.isfinite(filtered)):
                    self.analysis_log.append(f"  - WARN: NaN/Inf detected after de-ess filter @ {freq}Hz. Skipping filter.")
                else:
                    processed_audio = filtered
                    self.analysis_log.append(f"  - Applied notch @ {freq} Hz (Q={Q}, Cut={gain_for_filter:.1f}dB)")
            except (ValueError, FloatingPointError) as e: 
                self.analysis_log.append(f"  - WARN: De-ess filter design/apply failed @ {freq}Hz: {e}")
        return processed_audio


    def _apply_stereo_width(self, audio, width):
        if audio.ndim < 2 or audio.shape[1] < 2: self.analysis_log.append(f"⚠️ Mono audio, skipping stereo width"); return audio
        if width == 1.0: return audio
        mid = (audio[:, 0] + audio[:, 1]) / 2.0; side = (audio[:, 0] - audio[:, 1]) / 2.0
        side *= width
        left = mid + side; right = mid - side
        processed = np.column_stack([left, right])
        # Add clipping check here if width > 1.0 can cause peaks
        if width > 1.0:
            max_peak = np.max(np.abs(processed))
            if max_peak > 1.0:
                self.analysis_log.append(f"⚠️ Stereo widening caused clipping (peak: {max_peak:.3f}). Consider reducing width or input gain.")
                processed = np.clip(processed, -1.0, 1.0) # Apply hard clip if widening causes issues

        self.analysis_log.append(f"↔️ Applied stereo width: {width:.2f}")
        return processed


    def _plot_waveform_to_tensor(self, audio_data, sample_rate, title="Waveform", max_samples=150000): 
        # Add basic check for valid audio data
        if audio_data is None or audio_data.size == 0:
            self.analysis_log.append(f"⚠️ Skipping plot '{title}': Empty audio data")
            return torch.zeros((1, 64, 64, 3), dtype=torch.float32)

        fig = None # Define fig in outer scope for error handling
        try: 
            plt.style.use('dark_background'); fig, ax = plt.subplots(figsize=(10, 3)) 
            if audio_data.ndim == 2: plot_data = audio_data[:, 0]
            else: plot_data = audio_data
            
            # Check plot_data again after potential slicing
            if plot_data.size == 0:
                self.analysis_log.append(f"⚠️ Skipping plot '{title}': Empty plot data after channel selection")
                plt.close(fig)
                return torch.zeros((1, 64, 64, 3), dtype=torch.float32)

            num_samples_original = len(plot_data)
            if num_samples_original > max_samples:
                ds_factor = num_samples_original//max_samples; plot_data = plot_data[::ds_factor]
                time_axis = np.linspace(0, num_samples_original/sample_rate, len(plot_data))
                self.analysis_log.append(f"📊 Plot downsampled {ds_factor}x")
            else: time_axis = np.linspace(0, num_samples_original/sample_rate, len(plot_data))
            
            # Final check on plot_data before plotting
            if plot_data.size == 0:
                self.analysis_log.append(f"⚠️ Skipping plot '{title}': Empty plot data after downsampling")
                plt.close(fig)
                return torch.zeros((1, 64, 64, 3), dtype=torch.float32)

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
            
            # RECOMMENDED: Convert figure to IMAGE tensor via PIL (Section 8.2)
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=96, facecolor=fig.get_facecolor())
            buf.seek(0)
            img = Image.open(buf).convert("RGB")
            img_np = np.array(img).astype(np.float32) / 255.0
            
            # Clean up
            buf.close()
            plt.close(fig) # Close figure on success
            
            return torch.from_numpy(img_np).unsqueeze(0)
        
        except Exception as e:
            self.analysis_log.append(f"⚠️ Plotting Error: {e}")
            logging.error(f"[MD_AutoMasterNode] Plotting Error: {e}")
            if fig:
                plt.close(fig) # CRITICAL: Close figure even on error
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
        
        # Initialize outputs to safe defaults in case of early exit
        waveform_before = None
        waveform_after = None

        try:
            self.analysis_log = []; self.analysis_log.append(f"{'='*50}")
            self.analysis_log.append(f"🎧 MD AutoMaster v6.8 - Processing Started")
            self.analysis_log.append(f"{'='*50}")
            
            waveform_tensor = audio['waveform']; sample_rate = audio['sample_rate']
            audio_data = waveform_tensor[0].T.cpu().numpy().astype(np.float32)
            
            if audio_data.shape[0] == 0: return (audio, "❌ Error: Empty audio input.", None, None)
            if sample_rate < 44100: return (audio, f"❌ Error: Sample rate {sample_rate}Hz < 44100 Hz.", None, None)

            meter = pln.Meter(sample_rate)
            channels = audio_data.shape[1] if audio_data.ndim > 1 else 1
            self.analysis_log.append(f"📂 Input: SR={sample_rate} Hz, Shape={audio_data.shape}, Channels={channels}")
            
            original_audio_data = audio_data.copy()
            waveform_before = self._plot_waveform_to_tensor(original_audio_data, sample_rate, "Original Waveform")

            # Apply Profile Settings
            if profile != "Custom":
                self.analysis_log.append(f"🎛️ Loading Profile: '{profile}'")
                if profile == "Standard":
                    highpass_freq, lowpass_freq = 30, 0; do_eq, eq_bass_target, eq_high_target, eq_adaptive = True, 1.5, 0.4, True
                    do_deess, deess_amount_db = True, -10.0; do_mbc = True 
                    mbc_crossover_low, mbc_crossover_high, mbc_crossover_order = 300, 3000, 8
                    mbc_low_thresh_db, mbc_low_ratio = -18.0, 2.0; mbc_mid_thresh_db, mbc_mid_ratio = -18.0, 2.0; mbc_high_thresh_db, mbc_high_ratio = -18.0, 2.0
                    do_limiter, limiter_threshold_db = True, -0.1; stereo_width = 1.0
                elif profile == "Aggressive":
                    highpass_freq, lowpass_freq = 40, 0; do_eq, eq_bass_target, eq_high_target, eq_adaptive = True, 1.2, 0.3, True
                    do_deess, deess_amount_db = True, -12.0; do_mbc = True 
                    mbc_crossover_low, mbc_crossover_high, mbc_crossover_order = 250, 2500, 10
                    mbc_low_thresh_db, mbc_low_ratio = -20.0, 3.0; mbc_mid_thresh_db, mbc_mid_ratio = -20.0, 3.0; mbc_high_thresh_db, mbc_high_ratio = -19.0, 2.5
                    do_limiter, limiter_threshold_db = True, -0.1; stereo_width = 1.1
                elif profile == "Podcast (Clarity)":
                    highpass_freq, lowpass_freq = 80, 16000; do_eq, eq_bass_target, eq_high_target, eq_adaptive = True, 2.0, 0.5, True
                    do_deess, deess_amount_db = True, -15.0; do_mbc = True 
                    mbc_crossover_low, mbc_crossover_high, mbc_crossover_order = 400, 3500, 6
                    mbc_low_thresh_db, mbc_low_ratio = -16.0, 2.5; mbc_mid_thresh_db, mbc_mid_ratio = -18.0, 3.0; mbc_high_thresh_db, mbc_high_ratio = -18.0, 2.0
                    do_limiter, limiter_threshold_db = True, -1.0; stereo_width = 0.8
                elif profile == "Gentle (Tame)":
                    highpass_freq, lowpass_freq = 20, 0; do_eq, eq_bass_target, eq_high_target, eq_adaptive = True, 2.0, 0.6, True
                    do_deess = False; do_mbc = True 
                    mbc_crossover_low, mbc_crossover_high, mbc_crossover_order = 300, 3000, 8
                    mbc_low_thresh_db, mbc_low_ratio = -16.0, 1.8; mbc_mid_thresh_db, mbc_mid_ratio = -17.0, 1.8; mbc_high_thresh_db, mbc_high_ratio = -17.0, 1.5
                    do_limiter, limiter_threshold_db = True, -0.5; stereo_width = 1.0
                elif profile == "Mastering (Transparent)":
                    highpass_freq, lowpass_freq = 20, 0; do_eq, eq_bass_target, eq_high_target, eq_adaptive = True, 1.8, 0.5, True
                    do_deess, deess_amount_db = True, -12.0; do_mbc = True 
                    mbc_crossover_low, mbc_crossover_high, mbc_crossover_order = 200, 4000, 8
                    mbc_low_thresh_db, mbc_low_ratio = -15.0, 1.5; mbc_mid_thresh_db, mbc_mid_ratio = -16.0, 1.5; mbc_high_thresh_db, mbc_high_ratio = -16.0, 1.4
                    do_limiter, limiter_threshold_db = True, -0.3; stereo_width = 1.0
            
            self.analysis_log.append(f"🎯 Targets: LUFS={target_lufs}, Bass≤{eq_bass_target}, Highs≤{eq_high_target}")
            if fast_mode: self.analysis_log.append(f"⚡ Fast Mode: Enabled")

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
                # Iterative EQ logic...
                current_analysis = self._analyze(processed_audio, sample_rate, meter)
                eq_iteration = 0; bass_ok = current_analysis["bass"] <= eq_bass_target; high_ok = current_analysis["high"] <= eq_high_target
                while (not bass_ok or not high_ok) and eq_iteration < max_iterations_eq:
                    self.analysis_log.append(f"{'─'*30}"); self.analysis_log.append(f"🔧 EQ Iteration {eq_iteration + 1}/{max_iterations_eq}")
                    eq_adjustments = {}
                    if not bass_ok:
                        if eq_adaptive: overage=np.clip((current_analysis["bass"]-eq_bass_target),0,None)/max(eq_bass_target,0.1); eq_adjustments["bass_cut_db"]=-2.0; eq_adjustments["bass_scale"]=min(overage,2.0)
                        else: eq_adjustments["bass_cut_db"] = -2.0
                    if not high_ok:
                        if eq_adaptive: overage=np.clip((current_analysis["high"]-eq_high_target),0,None)/max(eq_high_target,0.1); eq_adjustments["high_cut_db"]=-1.5; eq_adjustments["high_scale"]=min(overage,2.0)
                        else: eq_adjustments["high_cut_db"] = -1.5
                    
                    # Add safety check for adjustments dict
                    if not eq_adjustments:
                        self.analysis_log.append("  - No EQ adjustments needed this iteration.")
                        break # Exit loop if no adjustments calculated

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
                    self.analysis_log.append(f"🧱 Final Limiter: Threshold={limiter_threshold_db} dB")
                    limiter = Limiter(threshold_db=limiter_threshold_db, release_ms=5.0)
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
                    self.analysis_log.append(f"🧱 Final Limiter: Threshold={limiter_threshold_db} dB")
                    limiter = Limiter(threshold_db=limiter_threshold_db, release_ms=5.0)
                    board = Pedalboard([limiter])
                    processed_audio = board(processed_audio.astype(np.float32), sample_rate=sample_rate)
                else: self.analysis_log.append("⏭️ Skipped Final Limiter")

            # --- Final Analysis ---
            final_analysis = self._analyze(processed_audio, sample_rate, meter) 
            final_lufs = meter.integrated_loudness(processed_audio)
            final_peak = np.max(np.abs(processed_audio)) if processed_audio.size > 0 else 0.0 # Handle empty audio case
            final_peak_db = -np.inf if final_peak <= 0 else 20*np.log10(final_peak) # Avoid log10(0)

            self.analysis_log.append(f"\n{'='*50}"); self.analysis_log.append(f"🏁 FINAL RESULTS"); self.analysis_log.append(f"{'='*50}")
            self.analysis_log.append(f"  LUFS: {final_lufs:.2f} (target: {target_lufs})")
            self.analysis_log.append(f"  Peak: {final_peak:.3f} ({final_peak_db:.1f} dBFS)")
            self.analysis_log.append(f"  Bass: {final_analysis.get('bass', np.nan):.2f} (target: ≤{eq_bass_target})") # Use .get with NaN default
            self.analysis_log.append(f"  Highs: {final_analysis.get('high', np.nan):.2f} (target: ≤{eq_high_target})")
            if abs(final_lufs - target_lufs) > 0.5: self.analysis_log.append(f"⚠️ LUFS differs > 0.5 LU")
            # Adjusted peak warning threshold slightly
            if final_peak > 0.999: self.analysis_log.append(f"⚠️ Peak VERY close to 0 dBFS! Possible clipping.") 
            elif final_peak > 0.98: self.analysis_log.append(f"⚠️ Peak > -0.2 dBFS. Check for intersample peaks.")
            self.analysis_log.append(f"{'='*50}\n")

            waveform_after = self._plot_waveform_to_tensor(processed_audio, sample_rate, "Processed Waveform")

            # --- Format Output ---
            if channels == 1 and processed_audio.ndim == 2: processed_audio = processed_audio[:, 0]
            if processed_audio.ndim == 1: final_tensor = torch.from_numpy(processed_audio).unsqueeze(0)
            else: final_tensor = torch.from_numpy(processed_audio.T)
            final_tensor = final_tensor.unsqueeze(0).to(waveform_tensor.device)
            output_audio = {"waveform": final_tensor, "sample_rate": sample_rate}
            
            return (output_audio, "\n".join(self.analysis_log), waveform_before, waveform_after)

        except Exception as e:
            last_log = self.analysis_log[-1] if self.analysis_log else 'Initialization'
            error_msg = f"❌ [MD_AutoMasterNode] Error at: '{last_log}'\n{e}"
            # Log the error for debugging
            logging.error(error_msg)
            logging.debug(traceback.format_exc())
            
            # Ensure plots are None on error if they weren't created
            waveform_before = waveform_before if 'waveform_before' in locals() and waveform_before is not None else None
            
            # Return neutral, valid output (passthrough input data)
            print(f"[MD_AutoMasterNode] ⚠️ Error encountered, returning input unchanged")
            return (audio, f"{error_msg}\n{traceback.format_exc()}", waveform_before, None)

# =================================================================================
# == Node Registration                                                         ==
# =================================================================================

NODE_CLASS_MAPPINGS = {
    "MD_AutoMasterNode": MD_AutoMasterNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MD_AutoMasterNode": "MD: Audio Auto Master Pro"
}