# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
# ████ MD_Nodes/AudioAutoEQ – Spectral Equalizer v1.0.0 ████▓▒░
# © 2025 MDMAchine
# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
# ░▒▓ ORIGIN & DEV:
#   • Cast into the void by: MDMAchine
#   • Enhanced by: Gemini
#   • License: Apache 2.0 — Sharing is caring
#   • Original source (if applicable): N/A

# ░▒▓ DESCRIPTION:
#   A ComfyUI Node for automatic audio equalization. It analyzes the input
#   audio's frequency spectrum using Librosa and applies a multi-band EQ
#   profile using Pedalboard to match a selected target sound.

# ░▒▓ FEATURES:
#   ✓ Automatic spectral analysis (librosa) & EQ (pedalboard).
#   ✓ 18+ built-in EQ target profiles (Vocal, Music, De-esser, ASMR, etc.).
#   ✓ Adjustable EQ strength for blending the effect.
#   ✓ Configurable Highpass & Lowpass filters to clean up extremes.
#   ✓ Outputs a detailed text report of the spectral analysis.

# ░▒▓ CHANGELOG:
#   - v1.0.0 (Current Release - Initial Commit):
#       • ADDED: Initial node creation, spectral analysis logic.
#       • ADDED: Pedalboard EQ chain implementation with 18+ profiles.

# ░▒▓ CONFIGURATION:
#   → Primary Use: Applying one-click EQ profiles ("Podcast/Speech", "Vocal Clarity") to audio.
#   → Secondary Use: Fixing common audio problems like "De-muddy" or "Vocal De-esser".
#   → Edge Use: Using "ASMR/Whisper Enhance" on field recordings to find hidden sounds.

# ░▒▓ WARNING:
#   This node may trigger:
#   ▓▒░ A 4-hour rabbit hole A/B testing 'Warm & Smooth' vs 'Modern/Crisp'.
#   ▓▒░ Realizing the 'De-muddy' profile just exposed the 60Hz hum from your '98 Sound Blaster.
#   ▓▒░ Applying 'Lo-Fi/Vintage' and suddenly feeling the urge to wear flannel and dial up to a BBS.
#   ▓▒░ That cold sweat when Librosa's FFT reveals what your mic *really* sounds like.
#   Consult your nearest demoscene vet if hallucinations persist.
# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄


# =================================================================================
# == Standard Library Imports                                                  ==
# =================================================================================
import os
import logging
import traceback

# =================================================================================
# == Third-Party Imports                                                       ==
# =================================================================================
import numpy as np
import librosa
import torch
from pedalboard import Pedalboard, HighpassFilter, LowpassFilter, PeakFilter

# =================================================================================
# == ComfyUI Core Modules                                                      ==
# =================================================================================
# (None needed)

# =================================================================================
# == Local Project Imports                                                     ==
# =================================================================================
# (None needed)

# =================================================================================
# == Helper Classes & Dependencies                                             ==
# =================================================================================
# (None)

# =================================================================================
# == Core Node Class                                                           ==
# =================================================================================

class MD_AudioAutoEQ:
    """
    Audio Auto EQ (MD_AudioAutoEQ)

    Applies automatic equalization to audio by analyzing its spectrum
    and matching it to a selected target profile using multi-band EQ.
    """
    
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
                "target_profile": ([
                    "Flat/Neutral",
                    "Vocal Clarity", 
                    "Vocal De-esser (Tame Piercing)",
                    "Podcast/Speech",
                    "Music Master",
                    "Warm & Smooth",
                    "Bright & Airy",
                    "De-muddy",
                    "Bass Boost",
                    "Bass Reduce",
                    "Treble Boost",
                    "Treble Reduce",
                    "Lo-Fi/Vintage",
                    "Modern/Crisp",
                    "EDM/Electronic",
                    "Acoustic/Natural",
                    "Radio Voice",
                    "ASMR/Whisper Enhance"
                ], {
                    "tooltip": (
                        "TARGET PROFILE\n"
                        "- The desired EQ sound profile to apply.\n"
                        "- 'Vocal Clarity' boosts speech frequencies.\n"
                        "- 'De-muddy' cuts low-mid frequencies.\n"
                        "- 'Flat/Neutral' attempts to balance the spectrum."
                    )
                }),
                "strength": ("FLOAT", {
                    "default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": (
                        "EQ STRENGTH\n"
                        "- The intensity of the applied EQ.\n"
                        "- 0.0 = No effect (Bypassed).\n"
                        "- 1.0 = Full effect.\n"
                        "- This acts like a 'mix' or 'blend' control."
                    )
                }),
                "highpass_freq": ("FLOAT", {
                    "default": 80.0, "min": 20.0, "max": 500.0, "step": 10.0,
                    "tooltip": (
                        "HIGH-PASS FILTER (Hz)\n"
                        "- Removes frequencies *below* this value.\n"
                        "- Use to cut sub-sonic rumble or proximity effect.\n"
                        "- 'Podcast/Speech' profile often uses ~80-120Hz."
                    )
                }),
                "lowpass_freq": ("FLOAT", {
                    "default": 15000.0, "min": 8000.0, "max": 20000.0, "step": 100.0,
                    "tooltip": (
                        "LOW-PASS FILTER (Hz)\n"
                        "- Removes frequencies *above* this value.\n"
                        "- Use to cut high-frequency hiss or aliasing.\n"
                        "- 'Lo-Fi' profiles use this aggressively."
                    )
                }),
                "debug": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "ENABLE DEBUG\n"
                        "- True: Prints verbose debug information (tensor shapes, ranges) to the console.\n"
                        "- False: Runs silently."
                    )
                }),
            }
        }
    
    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "analysis_report")
    FUNCTION = "process_audio"
    CATEGORY = "MD_Nodes/Audio Processing"
    
    def analyze_spectrum(self, audio, sr):
        """
        Analyzes the frequency spectrum of a mono audio signal.
        
        Args:
            audio: Numpy array of mono audio data.
            sr: Sample rate (int).
            
        Returns:
            Tuple of (band_ratios, freqs, avg_magnitude)
        """
        # Compute STFT
        stft = librosa.stft(audio)
        magnitude = np.abs(stft)
        
        # Average magnitude across time
        avg_magnitude = np.mean(magnitude, axis=1)
        
        # Get frequency bins
        freqs = librosa.fft_frequencies(sr=sr)
        
        # Define frequency bands
        bands = {
            'sub_bass': (20, 60),
            'bass': (60, 250),
            'low_mid': (250, 500),
            'mid': (500, 2000),
            'high_mid': (2000, 4000),
            'presence': (4000, 6000),
            'brilliance': (6000, 20000)
        }
        
        # Calculate energy in each band
        band_energy = {}
        for band_name, (low, high) in bands.items():
            mask = (freqs >= low) & (freqs < high)
            # Ensure mask finds valid bins before averaging
            if np.any(mask) and avg_magnitude[mask].size > 0:
                band_energy[band_name] = np.mean(avg_magnitude[mask])
            else:
                band_energy[band_name] = 0.0
        
        # Normalize
        total_energy = sum(band_energy.values())
        if total_energy == 0:
            # Avoid division by zero for silent audio
            band_ratios = {k: 0.0 for k in band_energy}
        else:
            band_ratios = {k: v / total_energy for k, v in band_energy.items()}
        
        return band_ratios, freqs, avg_magnitude
    
    def create_eq_chain(self, band_ratios, target_profile, strength, highpass, lowpass, sr):
        """
        Creates a Pedalboard EQ chain based on analysis and target profile.
        
        Args:
            band_ratios: Dict of energy ratios per frequency band.
            target_profile: String name of the selected profile.
            strength: Float (0.0-1.0) for EQ intensity.
            highpass: Float (Hz) for the high-pass filter.
            lowpass: Float (Hz) for the low-pass filter.
            sr: Sample rate (int).
            
        Returns:
            Pedalboard object with the configured EQ chain.
        """
        board = Pedalboard()
        
        # Always add highpass and lowpass filters
        board.append(HighpassFilter(cutoff_frequency_hz=highpass))
        
        # Determine EQ adjustments based on profile
        if target_profile == "Flat/Neutral":
            # Flatten the spectrum - reduce peaks, boost valleys
            avg_ratio = 1.0 / 7  # Equal distribution across 7 bands
            
            if band_ratios['bass'] > avg_ratio * 1.3:
                board.append(PeakFilter(cutoff_frequency_hz=150, gain_db=-3 * strength, q=1.0))
            
            if band_ratios['low_mid'] > avg_ratio * 1.3:
                board.append(PeakFilter(cutoff_frequency_hz=350, gain_db=-2.5 * strength, q=1.2))
            
            if band_ratios['mid'] < avg_ratio * 0.7:
                board.append(PeakFilter(cutoff_frequency_hz=1000, gain_db=2 * strength, q=0.8))
            
            if band_ratios['presence'] > avg_ratio * 1.3:
                board.append(PeakFilter(cutoff_frequency_hz=5000, gain_db=-2 * strength, q=1.5))
        
        elif target_profile == "Vocal Clarity":
            # Enhance vocal frequencies, reduce mud and harshness
            board.append(PeakFilter(cutoff_frequency_hz=200, gain_db=-3 * strength, q=0.7))  # Reduce mud
            board.append(PeakFilter(cutoff_frequency_hz=1000, gain_db=2.5 * strength, q=1.2))  # Presence
            board.append(PeakFilter(cutoff_frequency_hz=3000, gain_db=3 * strength, q=1.5))  # Clarity
            
            if band_ratios['brilliance'] > 0.15:
                board.append(PeakFilter(cutoff_frequency_hz=8000, gain_db=-2 * strength, q=1.0))  # Reduce harshness
        
        elif target_profile == "Vocal De-esser (Tame Piercing)":
            # Aggressively reduce harsh sibilance and piercing frequencies
            board.append(PeakFilter(cutoff_frequency_hz=5000, gain_db=-4 * strength, q=2.0))   # S sounds
            board.append(PeakFilter(cutoff_frequency_hz=7000, gain_db=-5 * strength, q=2.5))   # Sharp sibilance
            board.append(PeakFilter(cutoff_frequency_hz=9000, gain_db=-3.5 * strength, q=2.0)) # High harshness
            board.append(PeakFilter(cutoff_frequency_hz=3500, gain_db=1.5 * strength, q=1.5))  # Compensate clarity
            board.append(PeakFilter(cutoff_frequency_hz=250, gain_db=-2 * strength, q=0.8))    # Clean up low mids
        
        elif target_profile == "Podcast/Speech":
            # Optimize for speech intelligibility
            board.append(PeakFilter(cutoff_frequency_hz=120, gain_db=-4 * strength, q=0.7))   # Remove rumble
            board.append(PeakFilter(cutoff_frequency_hz=250, gain_db=-2 * strength, q=0.8))   # Reduce boominess
            board.append(PeakFilter(cutoff_frequency_hz=1200, gain_db=3 * strength, q=1.0))   # Speech presence
            board.append(PeakFilter(cutoff_frequency_hz=3000, gain_db=2.5 * strength, q=1.2)) # Consonant clarity
            board.append(PeakFilter(cutoff_frequency_hz=6000, gain_db=-2 * strength, q=1.5))  # Reduce sibilance
        
        elif target_profile == "Music Master":
            # Gentle mastering curve
            if band_ratios['sub_bass'] < 0.08:
                board.append(PeakFilter(cutoff_frequency_hz=40, gain_db=1.5 * strength, q=0.7))
            
            board.append(PeakFilter(cutoff_frequency_hz=250, gain_db=-1 * strength, q=0.5))  # Slight mud reduction
            board.append(PeakFilter(cutoff_frequency_hz=3000, gain_db=1 * strength, q=1.0))  # Slight presence
            board.append(PeakFilter(cutoff_frequency_hz=10000, gain_db=1.5 * strength, q=1.2))  # Air
        
        elif target_profile == "Warm & Smooth":
            # Roll off highs, enhance low-mids
            board.append(PeakFilter(cutoff_frequency_hz=300, gain_db=2 * strength, q=0.8))    # Warmth
            board.append(PeakFilter(cutoff_frequency_hz=5000, gain_db=-2.5 * strength, q=1.0)) # Reduce brightness
            board.append(PeakFilter(cutoff_frequency_hz=8000, gain_db=-3 * strength, q=1.2))    # Smooth top end
        
        elif target_profile == "Bright & Airy":
            # Enhance high frequencies
            board.append(PeakFilter(cutoff_frequency_hz=200, gain_db=-2 * strength, q=0.7))    # Reduce low mud
            board.append(PeakFilter(cutoff_frequency_hz=4000, gain_db=2.5 * strength, q=1.2))  # Presence
            board.append(PeakFilter(cutoff_frequency_hz=8000, gain_db=3 * strength, q=1.0))    # Sparkle
            board.append(PeakFilter(cutoff_frequency_hz=12000, gain_db=2 * strength, q=1.5))   # Air
        
        elif target_profile == "De-muddy":
            # Aggressively cut muddy frequencies
            board.append(PeakFilter(cutoff_frequency_hz=200, gain_db=-4 * strength, q=0.8))
            board.append(PeakFilter(cutoff_frequency_hz=350, gain_db=-3 * strength, q=1.0))
            board.append(PeakFilter(cutoff_frequency_hz=500, gain_db=-2 * strength, q=1.2))
            board.append(PeakFilter(cutoff_frequency_hz=2000, gain_db=2 * strength, q=0.8))  # Compensate clarity
        
        elif target_profile == "Bass Boost":
            # Enhance low-end
            board.append(PeakFilter(cutoff_frequency_hz=60, gain_db=4 * strength, q=0.7))
            board.append(PeakFilter(cutoff_frequency_hz=120, gain_db=3 * strength, q=0.8))
            board.append(PeakFilter(cutoff_frequency_hz=250, gain_db=1.5 * strength, q=1.0))
        
        elif target_profile == "Bass Reduce":
            # Reduce low-end buildup
            board.append(PeakFilter(cutoff_frequency_hz=80, gain_db=-4 * strength, q=0.8))
            board.append(PeakFilter(cutoff_frequency_hz=200, gain_db=-3 * strength, q=1.0))
        
        elif target_profile == "Treble Boost":
            # Enhance high frequencies
            board.append(PeakFilter(cutoff_frequency_hz=5000, gain_db=3 * strength, q=1.0))
            board.append(PeakFilter(cutoff_frequency_hz=8000, gain_db=3.5 * strength, q=1.2))
            board.append(PeakFilter(cutoff_frequency_hz=12000, gain_db=2.5 * strength, q=1.5))
        
        elif target_profile == "Treble Reduce":
            # Reduce high-frequency harshness
            board.append(PeakFilter(cutoff_frequency_hz=5000, gain_db=-3 * strength, q=1.2))
            board.append(PeakFilter(cutoff_frequency_hz=8000, gain_db=-2.5 * strength, q=1.0))
        
        elif target_profile == "Lo-Fi/Vintage":
            # Narrow bandwidth, reduce extremes
            board.append(PeakFilter(cutoff_frequency_hz=100, gain_db=-3 * strength, q=0.5))
            board.append(PeakFilter(cutoff_frequency_hz=400, gain_db=2 * strength, q=0.8))
            board.append(PeakFilter(cutoff_frequency_hz=2000, gain_db=1.5 * strength, q=1.0))
            board.append(PeakFilter(cutoff_frequency_hz=7000, gain_db=-4 * strength, q=1.0))
        
        elif target_profile == "Modern/Crisp":
            # Clean, forward, present
            board.append(PeakFilter(cutoff_frequency_hz=200, gain_db=-2 * strength, q=0.7))
            board.append(PeakFilter(cutoff_frequency_hz=2000, gain_db=2 * strength, q=1.0))
            board.append(PeakFilter(cutoff_frequency_hz=4000, gain_db=2.5 * strength, q=1.2))
            board.append(PeakFilter(cutoff_frequency_hz=10000, gain_db=2 * strength, q=1.5))
        
        elif target_profile == "EDM/Electronic":
            # Enhanced sub-bass and highs
            board.append(PeakFilter(cutoff_frequency_hz=50, gain_db=4 * strength, q=1.0))
            board.append(PeakFilter(cutoff_frequency_hz=100, gain_db=3 * strength, q=0.8))
            board.append(PeakFilter(cutoff_frequency_hz=300, gain_db=-2 * strength, q=0.7))
            board.append(PeakFilter(cutoff_frequency_hz=8000, gain_db=3 * strength, q=1.2))
            board.append(PeakFilter(cutoff_frequency_hz=12000, gain_db=2.5 * strength, q=1.5))
        
        elif target_profile == "Acoustic/Natural":
            # Subtle, natural enhancement
            board.append(PeakFilter(cutoff_frequency_hz=250, gain_db=-1 * strength, q=0.5))
            board.append(PeakFilter(cutoff_frequency_hz=1000, gain_db=1 * strength, q=0.8))
            board.append(PeakFilter(cutoff_frequency_hz=5000, gain_db=1.5 * strength, q=1.0))
        
        elif target_profile == "Radio Voice":
            # Classic radio broadcast sound
            board.append(PeakFilter(cutoff_frequency_hz=150, gain_db=-3 * strength, q=0.7))
            board.append(PeakFilter(cutoff_frequency_hz=1000, gain_db=3 * strength, q=1.0))
            board.append(PeakFilter(cutoff_frequency_hz=3000, gain_db=2.5 * strength, q=1.5))
            board.append(PeakFilter(cutoff_frequency_hz=8000, gain_db=-2 * strength, q=1.0))
        
        elif target_profile == "ASMR/Whisper Enhance":
            # Enhance subtle sounds and air
            board.append(PeakFilter(cutoff_frequency_hz=100, gain_db=-3 * strength, q=0.5))
            board.append(PeakFilter(cutoff_frequency_hz=500, gain_db=1.5 * strength, q=1.0))
            board.append(PeakFilter(cutoff_frequency_hz=8000, gain_db=3 * strength, q=1.5))
            board.append(PeakFilter(cutoff_frequency_hz=12000, gain_db=2.5 * strength, q=1.8))
        
        board.append(LowpassFilter(cutoff_frequency_hz=lowpass))
        
        return board
    
    def process_audio(self, audio, target_profile, strength, highpass_freq, lowpass_freq, debug):
        """
        Main execution function.
        
        Args:
            audio: ComfyUI AUDIO object (dict)
            target_profile: String name of profile
            strength: Float (0.0-1.0)
            highpass_freq: Float (Hz)
            lowpass_freq: Float (Hz)
            debug: Boolean
            
        Returns:
            Tuple of (audio_output, report_string)
        """
        try:
            # Extract audio data and sample rate from AUDIO input
            waveform = audio['waveform']  # Shape: [batch, channels, samples]
            sample_rate = audio['sample_rate']
            
            # Convert to numpy for processing
            if isinstance(waveform, torch.Tensor):
                audio_np = waveform.cpu().numpy()
            else:
                audio_np = np.array(waveform)
            
            if debug:
                print(f"[AutoEQ] Input shape: {audio_np.shape}, dtype: {audio_np.dtype}")
                print(f"[AutoEQ] Input range: [{audio_np.min():.4f}, {audio_np.max():.4f}]")
            
            # Handle batch dimension - process first batch item
            if len(audio_np.shape) == 3:
                audio_np = audio_np[0]  # Take first batch [channels, samples]
            
            # Ensure shape is [channels, samples]
            if len(audio_np.shape) == 1:
                audio_np = audio_np.reshape(1, -1)
            
            if debug:
                print(f"[AutoEQ] Processing shape: {audio_np.shape}")
            
            # Get mono version for analysis
            if audio_np.shape[0] > 1:
                audio_mono = np.mean(audio_np, axis=0)
            else:
                audio_mono = audio_np[0]
            
            # Handle potentially silent audio
            if np.max(np.abs(audio_mono)) < 1e-6:
                report = "⚠️ [MD_AudioAutoEQ] Warning: Input audio is silent or near-silent. Skipping analysis and EQ."
                return (audio, report)
                
            # Analyze spectrum
            band_ratios, freqs, avg_magnitude = self.analyze_spectrum(audio_mono, sample_rate)
            
            # Create analysis report
            report = f"Spectral Analysis:\n"
            report += f"Sample Rate: {sample_rate} Hz\n"
            report += f"Channels: {audio_np.shape[0]}\n"
            report += f"Samples: {audio_np.shape[1]}\n"
            report += f"Duration: {audio_np.shape[1]/sample_rate:.2f}s\n"
            report += f"Target Profile: {target_profile}\n"
            report += f"Strength: {strength}\n\n"
            report += "Frequency Band Energy:\n"
            for band, ratio in band_ratios.items():
                report += f"  {band}: {ratio*100:.2f}%\n"
            
            # Create EQ chain
            eq_board = self.create_eq_chain(band_ratios, target_profile, strength, highpass_freq, lowpass_freq, sample_rate)
            
            # Apply EQ using Pedalboard (works on numpy arrays)
            # Pedalboard expects float32
            audio_np_float = audio_np.astype(np.float32)
            processed_audio = eq_board(audio_np_float, sample_rate)
            
            if debug:
                print(f"[AutoEQ] Output shape: {processed_audio.shape}, dtype: {processed_audio.dtype}")
                print(f"[AutoEQ] Output range: [{processed_audio.min():.4f}, {processed_audio.max():.4f}]")
            
            # Convert back to torch tensor with correct dtype and restore batch dimension
            processed_tensor = torch.from_numpy(processed_audio).float().unsqueeze(0)  # Add batch dimension [1, channels, samples]
            
            # Return in same format as input
            output_audio = {
                'waveform': processed_tensor,
                'sample_rate': sample_rate
            }
            
            return (output_audio, report)

        except Exception as e:
            # Log the error for debugging
            logging.error(f"[MD_AudioAutoEQ] Processing failed: {e}")
            logging.debug(traceback.format_exc())
            
            # Create a user-facing error message
            error_msg = f"❌ [MD_AudioAutoEQ] Error: {e}\n\n{traceback.format_exc()}"
            
            # Return neutral, valid output (passthrough input data)
            print(f"[MD_AudioAutoEQ] ⚠️ Error encountered, returning input unchanged")
            return (audio, error_msg)


# =================================================================================
# == Node Registration                                                         ==
# =================================================================================

NODE_CLASS_MAPPINGS = {
    "MD_AudioAutoEQ": MD_AudioAutoEQ
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MD_AudioAutoEQ": "MD: Audio Auto EQ"
}