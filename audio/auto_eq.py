# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
# ████ MD_Nodes/AudioAutoEQ – Spectral Equalizer ████▓▒░
# © 2025 MDMAchine
# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
# ░▒▓ ORIGIN & DEV:
#   • Cast into the void by: MDMAchine
#   • Enhanced by: Gemini
#   • License: Apache 2.0 — Sharing is caring

# ░▒▓ DESCRIPTION:
#   A ComfyUI Node for automatic audio equalization. It analyzes the input
#   audio's frequency spectrum using Librosa and applies a multi-band EQ
#   profile using Pedalboard to match a selected target sound.

# ░▒▓ FEATURES:
#   ✓ Automatic spectral analysis (librosa) & EQ (pedalboard).
#   ✓ 18+ built-in EQ target profiles (Vocal, Music, De-esser, ASMR, etc.).
#   ✓ Adjustable EQ strength for blending the effect.
#   ✓ Configurable Highpass & Lowpass filters to clean up extremes.
#   ✓ Outputs detailed text report AND Before/After waveform visuals.

# ░▒▓ CHANGELOG:
#   - v1.1.0 (Visual Update - Nov 2025):
#       • ADDED: Waveform visualization (Before/After) outputs.
#       • REFACTOR: Moved profile list to constant for readability.
#       • FIXED: Matplotlib backend safety for headless servers.
#       • FIXED: Memory leak prevention (buffer closing).
#   - v1.0.0 (Initial Release):
#       • Initial node creation, spectral analysis logic.

# ░▒▓ CONFIGURATION:
#   → Primary Use: Applying one-click EQ profiles ("Podcast/Speech", "Vocal Clarity").
#   → Secondary Use: Fixing common audio problems like "De-muddy".

# ░▒▓ WARNING:
#   This node may trigger:
#   ▓▒░ A 4-hour rabbit hole A/B testing 'Warm & Smooth' vs 'Modern/Crisp'.
#   ▓▒░ That cold sweat when Librosa's FFT reveals what your mic *really* sounds like.
#   Consult your nearest demoscene vet if hallucinations persist.
# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄


# =================================================================================
# == Standard Library Imports                                                     ==
# =================================================================================
import os
import io
import logging
import traceback

# =================================================================================
# == Third-Party Imports                                                          ==
# =================================================================================
import numpy as np
import librosa
import torch
from pedalboard import Pedalboard, HighpassFilter, LowpassFilter, PeakFilter
import matplotlib as mpl
# CRITICAL: Set non-interactive backend BEFORE importing pyplot
mpl.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

# =================================================================================
# == Constants                                                                    ==
# =================================================================================

EQ_PROFILES_LIST = [
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
]

# =================================================================================
# == Core Node Class                                                              ==
# =================================================================================

class MD_AudioAutoEQ:
    """
    Audio Auto EQ (MD_AudioAutoEQ)
    Applies automatic equalization to audio by analyzing its spectrum
    and matching it to a selected target profile using multi-band EQ.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {
                    "tooltip": "AUDIO INPUT\n- The audio data to process."
                }),
                "target_profile": (EQ_PROFILES_LIST, {
                    "tooltip": "TARGET PROFILE\n- The desired EQ sound profile to apply."
                }),
                "strength": ("FLOAT", {
                    "default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "EQ STRENGTH\n- The intensity of the applied EQ (Mix)."
                }),
                "highpass_freq": ("FLOAT", {
                    "default": 80.0, "min": 20.0, "max": 500.0, "step": 10.0,
                    "tooltip": "HIGH-PASS FILTER (Hz)\n- Removes frequencies *below* this value."
                }),
                "lowpass_freq": ("FLOAT", {
                    "default": 15000.0, "min": 8000.0, "max": 20000.0, "step": 100.0,
                    "tooltip": "LOW-PASS FILTER (Hz)\n- Removes frequencies *above* this value."
                }),
                "debug": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "ENABLE DEBUG\n- Prints verbose debug info to console."
                }),
            }
        }
    
    RETURN_TYPES = ("AUDIO", "STRING", "IMAGE", "IMAGE")
    RETURN_NAMES = ("audio", "analysis_report", "waveform_before", "waveform_after")
    FUNCTION = "process_audio"
    CATEGORY = "MD_Nodes/Audio"
    
    def analyze_spectrum(self, audio, sr):
        """Analyzes the frequency spectrum of a mono audio signal."""
        # Compute STFT
        stft = librosa.stft(audio)
        magnitude = np.abs(stft)
        avg_magnitude = np.mean(magnitude, axis=1)
        freqs = librosa.fft_frequencies(sr=sr)
        
        bands = {
            'sub_bass': (20, 60), 'bass': (60, 250), 'low_mid': (250, 500),
            'mid': (500, 2000), 'high_mid': (2000, 4000), 'presence': (4000, 6000),
            'brilliance': (6000, 20000)
        }
        
        band_energy = {}
        for band_name, (low, high) in bands.items():
            mask = (freqs >= low) & (freqs < high)
            if np.any(mask) and avg_magnitude[mask].size > 0:
                band_energy[band_name] = np.mean(avg_magnitude[mask])
            else:
                band_energy[band_name] = 0.0
        
        total_energy = sum(band_energy.values())
        if total_energy == 0:
            logging.warning("[MD_AudioAutoEQ] Zero total energy detected in spectrum analysis")
            band_ratios = {k: 0.0 for k in band_energy}
        else:
            band_ratios = {k: v / total_energy for k, v in band_energy.items()}
        
        return band_ratios, freqs, avg_magnitude
    
    def create_eq_chain(self, band_ratios, target_profile, strength, highpass, lowpass, sr):
        """Creates a Pedalboard EQ chain based on analysis and target profile."""
        board = Pedalboard()
        board.append(HighpassFilter(cutoff_frequency_hz=highpass))
        
        # EQ Profiles Logic
        if target_profile == "Flat/Neutral":
            avg_ratio = 1.0 / 7
            if band_ratios['bass'] > avg_ratio * 1.3: board.append(PeakFilter(cutoff_frequency_hz=150, gain_db=-3 * strength, q=1.0))
            if band_ratios['low_mid'] > avg_ratio * 1.3: board.append(PeakFilter(cutoff_frequency_hz=350, gain_db=-2.5 * strength, q=1.2))
            if band_ratios['mid'] < avg_ratio * 0.7: board.append(PeakFilter(cutoff_frequency_hz=1000, gain_db=2 * strength, q=0.8))
            if band_ratios['presence'] > avg_ratio * 1.3: board.append(PeakFilter(cutoff_frequency_hz=5000, gain_db=-2 * strength, q=1.5))
        
        elif target_profile == "Vocal Clarity":
            board.append(PeakFilter(cutoff_frequency_hz=200, gain_db=-3 * strength, q=0.7))
            board.append(PeakFilter(cutoff_frequency_hz=1000, gain_db=2.5 * strength, q=1.2))
            board.append(PeakFilter(cutoff_frequency_hz=3000, gain_db=3 * strength, q=1.5))
            if band_ratios['brilliance'] > 0.15: board.append(PeakFilter(cutoff_frequency_hz=8000, gain_db=-2 * strength, q=1.0))
        
        elif target_profile == "Vocal De-esser (Tame Piercing)":
            board.append(PeakFilter(cutoff_frequency_hz=5000, gain_db=-4 * strength, q=2.0))
            board.append(PeakFilter(cutoff_frequency_hz=7000, gain_db=-5 * strength, q=2.5))
            board.append(PeakFilter(cutoff_frequency_hz=9000, gain_db=-3.5 * strength, q=2.0))
            board.append(PeakFilter(cutoff_frequency_hz=3500, gain_db=1.5 * strength, q=1.5))
            board.append(PeakFilter(cutoff_frequency_hz=250, gain_db=-2 * strength, q=0.8))
        
        elif target_profile == "Podcast/Speech":
            board.append(PeakFilter(cutoff_frequency_hz=120, gain_db=-4 * strength, q=0.7))
            board.append(PeakFilter(cutoff_frequency_hz=250, gain_db=-2 * strength, q=0.8))
            board.append(PeakFilter(cutoff_frequency_hz=1200, gain_db=3 * strength, q=1.0))
            board.append(PeakFilter(cutoff_frequency_hz=3000, gain_db=2.5 * strength, q=1.2))
            board.append(PeakFilter(cutoff_frequency_hz=6000, gain_db=-2 * strength, q=1.5))
        
        elif target_profile == "Music Master":
            if band_ratios['sub_bass'] < 0.08: board.append(PeakFilter(cutoff_frequency_hz=40, gain_db=1.5 * strength, q=0.7))
            board.append(PeakFilter(cutoff_frequency_hz=250, gain_db=-1 * strength, q=0.5))
            board.append(PeakFilter(cutoff_frequency_hz=3000, gain_db=1 * strength, q=1.0))
            board.append(PeakFilter(cutoff_frequency_hz=10000, gain_db=1.5 * strength, q=1.2))
        
        elif target_profile == "Warm & Smooth":
            board.append(PeakFilter(cutoff_frequency_hz=300, gain_db=2 * strength, q=0.8))
            board.append(PeakFilter(cutoff_frequency_hz=5000, gain_db=-2.5 * strength, q=1.0))
            board.append(PeakFilter(cutoff_frequency_hz=8000, gain_db=-3 * strength, q=1.2))
        
        elif target_profile == "Bright & Airy":
            board.append(PeakFilter(cutoff_frequency_hz=200, gain_db=-2 * strength, q=0.7))
            board.append(PeakFilter(cutoff_frequency_hz=4000, gain_db=2.5 * strength, q=1.2))
            board.append(PeakFilter(cutoff_frequency_hz=8000, gain_db=3 * strength, q=1.0))
            board.append(PeakFilter(cutoff_frequency_hz=12000, gain_db=2 * strength, q=1.5))
        
        elif target_profile == "De-muddy":
            board.append(PeakFilter(cutoff_frequency_hz=200, gain_db=-4 * strength, q=0.8))
            board.append(PeakFilter(cutoff_frequency_hz=350, gain_db=-3 * strength, q=1.0))
            board.append(PeakFilter(cutoff_frequency_hz=500, gain_db=-2 * strength, q=1.2))
            board.append(PeakFilter(cutoff_frequency_hz=2000, gain_db=2 * strength, q=0.8))
        
        elif target_profile == "Bass Boost":
            board.append(PeakFilter(cutoff_frequency_hz=60, gain_db=4 * strength, q=0.7))
            board.append(PeakFilter(cutoff_frequency_hz=120, gain_db=3 * strength, q=0.8))
            board.append(PeakFilter(cutoff_frequency_hz=250, gain_db=1.5 * strength, q=1.0))
        
        elif target_profile == "Bass Reduce":
            board.append(PeakFilter(cutoff_frequency_hz=80, gain_db=-4 * strength, q=0.8))
            board.append(PeakFilter(cutoff_frequency_hz=200, gain_db=-3 * strength, q=1.0))
        
        elif target_profile == "Treble Boost":
            board.append(PeakFilter(cutoff_frequency_hz=5000, gain_db=3 * strength, q=1.0))
            board.append(PeakFilter(cutoff_frequency_hz=8000, gain_db=3.5 * strength, q=1.2))
            board.append(PeakFilter(cutoff_frequency_hz=12000, gain_db=2.5 * strength, q=1.5))
        
        elif target_profile == "Treble Reduce":
            board.append(PeakFilter(cutoff_frequency_hz=5000, gain_db=-3 * strength, q=1.2))
            board.append(PeakFilter(cutoff_frequency_hz=8000, gain_db=-2.5 * strength, q=1.0))
        
        elif target_profile == "Lo-Fi/Vintage":
            board.append(PeakFilter(cutoff_frequency_hz=100, gain_db=-3 * strength, q=0.5))
            board.append(PeakFilter(cutoff_frequency_hz=400, gain_db=2 * strength, q=0.8))
            board.append(PeakFilter(cutoff_frequency_hz=2000, gain_db=1.5 * strength, q=1.0))
            board.append(PeakFilter(cutoff_frequency_hz=7000, gain_db=-4 * strength, q=1.0))
        
        elif target_profile == "Modern/Crisp":
            board.append(PeakFilter(cutoff_frequency_hz=200, gain_db=-2 * strength, q=0.7))
            board.append(PeakFilter(cutoff_frequency_hz=2000, gain_db=2 * strength, q=1.0))
            board.append(PeakFilter(cutoff_frequency_hz=4000, gain_db=2.5 * strength, q=1.2))
            board.append(PeakFilter(cutoff_frequency_hz=10000, gain_db=2 * strength, q=1.5))
        
        elif target_profile == "EDM/Electronic":
            board.append(PeakFilter(cutoff_frequency_hz=50, gain_db=4 * strength, q=1.0))
            board.append(PeakFilter(cutoff_frequency_hz=100, gain_db=3 * strength, q=0.8))
            board.append(PeakFilter(cutoff_frequency_hz=300, gain_db=-2 * strength, q=0.7))
            board.append(PeakFilter(cutoff_frequency_hz=8000, gain_db=3 * strength, q=1.2))
            board.append(PeakFilter(cutoff_frequency_hz=12000, gain_db=2.5 * strength, q=1.5))
        
        elif target_profile == "Acoustic/Natural":
            board.append(PeakFilter(cutoff_frequency_hz=250, gain_db=-1 * strength, q=0.5))
            board.append(PeakFilter(cutoff_frequency_hz=1000, gain_db=1 * strength, q=0.8))
            board.append(PeakFilter(cutoff_frequency_hz=5000, gain_db=1.5 * strength, q=1.0))
        
        elif target_profile == "Radio Voice":
            board.append(PeakFilter(cutoff_frequency_hz=150, gain_db=-3 * strength, q=0.7))
            board.append(PeakFilter(cutoff_frequency_hz=1000, gain_db=3 * strength, q=1.0))
            board.append(PeakFilter(cutoff_frequency_hz=3000, gain_db=2.5 * strength, q=1.5))
            board.append(PeakFilter(cutoff_frequency_hz=8000, gain_db=-2 * strength, q=1.0))
        
        elif target_profile == "ASMR/Whisper Enhance":
            board.append(PeakFilter(cutoff_frequency_hz=100, gain_db=-3 * strength, q=0.5))
            board.append(PeakFilter(cutoff_frequency_hz=500, gain_db=1.5 * strength, q=1.0))
            board.append(PeakFilter(cutoff_frequency_hz=8000, gain_db=3 * strength, q=1.5))
            board.append(PeakFilter(cutoff_frequency_hz=12000, gain_db=2.5 * strength, q=1.8))
        
        board.append(LowpassFilter(cutoff_frequency_hz=lowpass))
        return board
    
    def _plot_waveform_to_tensor(self, audio_data, sample_rate, title="Waveform", max_samples=150000): 
        if audio_data is None or audio_data.size == 0:
            logging.debug("[MD_AudioAutoEQ] Skipping plot: Empty audio data")
            return torch.zeros((1, 64, 64, 3), dtype=torch.float32)

        fig = None
        buf = None
        try: 
            plt.style.use('dark_background'); fig, ax = plt.subplots(figsize=(10, 3)) 
            if audio_data.ndim == 2: plot_data = audio_data[:, 0]
            else: plot_data = audio_data
            
            if plot_data.size == 0:
                plt.close(fig)
                return torch.zeros((1, 64, 64, 3), dtype=torch.float32)

            num_samples_original = len(plot_data)
            if num_samples_original > max_samples:
                ds_factor = num_samples_original//max_samples; plot_data = plot_data[::ds_factor]
                time_axis = np.linspace(0, num_samples_original/sample_rate, len(plot_data))
            else: time_axis = np.linspace(0, num_samples_original/sample_rate, len(plot_data))
            
            if plot_data.size == 0:
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
            
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=96, facecolor=fig.get_facecolor())
            buf.seek(0)
            img = Image.open(buf).convert("RGB")
            img_np = np.array(img).astype(np.float32) / 255.0
            
            buf.close() # Prevent memory leak
            plt.close(fig)
            return torch.from_numpy(img_np).unsqueeze(0)
        
        except Exception as e:
            if fig: plt.close(fig)
            if buf: buf.close()
            return torch.zeros((1, 64, 64, 3), dtype=torch.float32)
    
    def process_audio(self, audio, target_profile, strength, highpass_freq, lowpass_freq, debug):
        """Main execution function."""
        waveform_before = None
        waveform_after = None
        
        # Input Validation
        strength = max(0.0, min(1.0, strength))
        highpass_freq = max(20.0, min(500.0, highpass_freq))
        lowpass_freq = max(8000.0, min(20000.0, lowpass_freq))

        try:
            # Extract audio data
            waveform = audio['waveform'] 
            sample_rate = audio['sample_rate']
            
            if isinstance(waveform, torch.Tensor):
                audio_np = waveform.cpu().numpy()
            else:
                audio_np = np.array(waveform)
            
            if debug:
                print(f"[AutoEQ] Input shape: {audio_np.shape}, dtype: {audio_np.dtype}")
            
            # Batch handling: Process first item
            if len(audio_np.shape) == 3:
                audio_np = audio_np[0] 
            
            if len(audio_np.shape) == 1:
                audio_np = audio_np.reshape(1, -1)

            # Generate "Before" Plot
            original_data = audio_np.T if audio_np.shape[0] > 1 else audio_np[0]
            waveform_before = self._plot_waveform_to_tensor(original_data, sample_rate, "Original Waveform")
            
            # Mono mix for spectral analysis
            if audio_np.shape[0] > 1:
                audio_mono = np.mean(audio_np, axis=0)
            else:
                audio_mono = audio_np[0]
            
            # Handle silent audio
            if np.max(np.abs(audio_mono)) < 1e-6:
                return (audio, "⚠️ Warning: Input audio is silent.", waveform_before, None)
                
            # Analyze
            band_ratios, freqs, avg_magnitude = self.analyze_spectrum(audio_mono, sample_rate)
            
            if debug:
                 print(f"[AutoEQ] Band Ratios: {band_ratios}")

            # Report
            report = f"Spectral Analysis:\nSR: {sample_rate} Hz\nProfile: {target_profile}\nStrength: {strength}\n\nBand Energy:\n"
            for band, ratio in band_ratios.items():
                report += f"  {band}: {ratio*100:.2f}%\n"
            
            # EQ
            eq_board = self.create_eq_chain(band_ratios, target_profile, strength, highpass_freq, lowpass_freq, sample_rate)
            
            if debug:
                 print(f"[AutoEQ] Applied Filters: {len(eq_board)}")

            audio_np_float = audio_np.astype(np.float32)
            processed_audio = eq_board(audio_np_float, sample_rate)
            
            # Generate "After" Plot
            processed_data_plot = processed_audio.T if processed_audio.shape[0] > 1 else processed_audio[0]
            waveform_after = self._plot_waveform_to_tensor(processed_data_plot, sample_rate, "Processed Waveform")
            
            # Pack Output
            processed_tensor = torch.from_numpy(processed_audio).float().unsqueeze(0)
            output_audio = {'waveform': processed_tensor, 'sample_rate': sample_rate}
            
            return (output_audio, report, waveform_before, waveform_after)

        except Exception as e:
            error_msg = f"❌ [MD_AudioAutoEQ] Error: {e}\n\n{traceback.format_exc()}"
            logging.error(error_msg)
            print(f"[MD_AudioAutoEQ] ⚠️ Error encountered, returning input unchanged")
            return (audio, error_msg, waveform_before, None)


# =================================================================================
# == Node Registration                                                            ==
# =================================================================================

NODE_CLASS_MAPPINGS = {
    "MD_AudioAutoEQ": MD_AudioAutoEQ
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MD_AudioAutoEQ": "MD: Audio Auto EQ"
}