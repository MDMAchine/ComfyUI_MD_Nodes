# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# █▓▒░                                                                     ░▒▓█
# █▓▒░        MD_Nodes/AudioAnalysisSuite – Broadcast Tools v2.1.3         ░▒▓█
# █▓▒░                                                                     ░▒▓█
# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
# ╠═ © 2026 MDMAchine
# ╠═ License: GNU General Public License v3.0 (GPL v3)
# ║
# ║  This program is free software: you can redistribute it and/or modify
# ║  it under the terms of the GNU General Public License as published by
# ║  the Free Software Foundation, either version 3 of the License, or
# ║  (at your option) any later version.
# ║
# ║  This program is distributed in the hope that it will be useful,
# ║  but WITHOUT ANY WARRANTY; without even the implied warranty of
# ║  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# ║  GNU General Public License for more details.
# ║
# ║  You should have received a copy of the GNU General Public License
# ║  along with this program. If not, see <https://www.gnu.org/licenses/>.
# ╠════════════════════════════════════════════════════════════════════════════
# ║ ░▒▓ DESCRIPTION:
# ║    A collection of professional audio analysis and utility nodes.
# ║    Includes ITU-R BS.1770-4 LUFS normalization, true-peak detection, 
# ║    and advanced spectral visualization.
# ║    NOTE: This is a public wrapper. Missing binaries will gracefully pass 
# ║    audio through unchanged for processing nodes.
# ╚════════════════════════════════════════════════════════════════════════════


VERSION = "v2.1.3"  # UPS v1.5.8

import io
import sys
import os
import logging
import time
import torch
import numpy as np
from scipy import signal
from PIL import Image

# =================================================================================
# == MD_Nodes Universal Binary Loader (v1.6.1)
# =================================================================================
def find_core_paths():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(current_dir, "core"),
        os.path.join(current_dir, "..", "core"),
        os.path.join(current_dir, "..", "..", "core")
    ]
    return list(dict.fromkeys([os.path.abspath(c) for c in candidates if os.path.exists(c)]))

CORE_LOCATIONS = find_core_paths()
CORE_LOADED = False
CORE_MODE = None
CORE_ERROR = None

for loc in CORE_LOCATIONS:
    if loc not in sys.path: sys.path.insert(0, loc)

try:
    import broadcast_tools_core_bin as core
    CORE_LOADED = True
    CORE_MODE = "Binary (Production)"
except ImportError as e1:
    try:
        import broadcast_tools_core as core
        CORE_LOADED = True
        CORE_MODE = "Source (Development)"
    except ImportError as e2:
        CORE_ERROR = f"Binary: {e1} | Source: {e2}"

# =================================================================================
# == Dependencies & Constants
# =================================================================================

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    from scipy.ndimage import gaussian_filter1d
    SCIPY_IMG_AVAILABLE = True
except ImportError:
    SCIPY_IMG_AVAILABLE = False

logger = logging.getLogger("MD_Nodes.Audio.Analysis")

CONST_BLANK_PLOT_SIZE = (1, 256, 512, 3)
CONST_MAX_PLOT_SAMPLES = 150000
CONST_PLOT_DPI = 120
CONST_PLOT_FIGSIZE = (10, 3)
CONST_WAVEFORM_COLOR = '#87CEEB'
CONST_PEAK_COLOR = 'orangered'
CONST_RMS_COLOR = 'mediumseagreen'
CONST_BACKGROUND_COLOR = '#1e1e1e'

CONST_DEFAULT_SAMPLE_RATE = 44100
CONST_EPSILON = 1e-12 

# =================================================================================
# == Performance Profiler
# =================================================================================

class PerformanceProfiler:
    """Standard performance profiler for MD_Nodes."""
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.timings = {}
        self.start_times = {}
    
    def start(self, operation_name):
        if not self.enabled: return
        self.start_times[operation_name] = time.perf_counter()
    
    def stop(self, operation_name):
        if not self.enabled: return
        if operation_name in self.start_times:
            elapsed = time.perf_counter() - self.start_times[operation_name]
            if operation_name not in self.timings:
                self.timings[operation_name] = []
            self.timings[operation_name].append(elapsed)
            del self.start_times[operation_name]
    
    def get_total_time(self):
        if not self.enabled or not self.timings: return 0.0
        return sum(sum(times) for times in self.timings.values())
    
    def print_report(self):
        if not self.enabled or not self.timings: return
        logging.info("\n⏱️  PERFORMANCE:")
        total = self.get_total_time()
        logging.info(f"    • Total Time: {total:.2f}s")
        for op_name, times in sorted(self.timings.items()):
            avg = sum(times) / len(times)
            if len(times) == 1:
                logging.info(f"    • {op_name}: {avg:.3f}s")
            else:
                logging.info(f"    • {op_name}: {avg:.3f}s avg ({len(times)}x)")

# =================================================================================
# == Utility Class (Wrapper Side)
# =================================================================================

class MasteringBase:
    """Shared helper methods for all analysis nodes."""
    
    def _soft_clip(self, audio_data):
        return np.clip(np.tanh(audio_data), -1.0, 1.0)

    def _unpack_audio_batch(self, audio_dict):
        if not isinstance(audio_dict, dict) or 'waveform' not in audio_dict:
            raise ValueError("Invalid audio input")
        audio_tensor = audio_dict['waveform']
        if audio_tensor.ndim == 2: 
            audio_tensor = audio_tensor.unsqueeze(0)
        return (
            audio_tensor.cpu().float().numpy(), 
            audio_dict.get('sample_rate', CONST_DEFAULT_SAMPLE_RATE), 
            audio_tensor.device
        )
    
    def _phase_correlation(self, left, right):
        if len(left) != len(right) or len(left) == 0: return 0.0
        try: return float(np.corrcoef(left, right)[0, 1])
        except Exception: return 0.0

    def _hz_to_note(self, freq):
        if freq == 0: return "N/A"
        A4 = 440
        C0 = A4 * pow(2, -4.75)
        try:
            h = round(12 * np.log2(freq / C0))
            octave = h // 12
            n = int(h % 12)
            names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
            return f"{names[n]}{octave}"
        except Exception: return "N/A"

    def _generate_lufs_report(self, mono, left, right, sr, is_stereo, lufs_data, phase_corr):
        lufs = lufs_data.get('lufs', -14.0)
        true_peak_db = lufs_data.get('true_peak_db', -1.0)
        dynamic_range = lufs_data.get('dynamic_range', 10.0)
        
        if is_stereo and CORE_LOADED:
            mid, side = core.MidSideProcessorCore.encode(left, right)
            rms_mid = np.sqrt(np.mean(mid**2))
            rms_side = np.sqrt(np.mean(side**2))
            width_ratio = rms_side / (rms_mid + CONST_EPSILON)
            width_text = f"{width_ratio:.2f} (M/S Ratio)"
            phase_text = f"{phase_corr:.2f} (+1=Mono, -1=Out of Phase)"
        else:
            width_text = "Mono"
            phase_text = "N/A"

        freqs = np.fft.rfftfreq(len(mono), 1/sr)
        magnitudes = np.abs(np.fft.rfft(mono))
        peak_freq = freqs[np.argmax(magnitudes)]
        note = self._hz_to_note(peak_freq)

        text = (
            f"🎚️ **BROADCAST MASTERING REPORT**\n{'='*60}\n"
            f"📊 **LOUDNESS**\n  • LUFS: {lufs:.1f}\n  • True Peak: {true_peak_db:.2f} dBTP\n  • Dynamics: {dynamic_range:.1f} dB\n"
            f"\n🎧 **STEREO**\n  • Width: {width_text}\n  • Phase: {phase_text}\n"
            f"\n🎼 **CONTENT**\n  • Peak Freq: {peak_freq:.1f} Hz ({note})\n  • Sample Rate: {sr} Hz"
        )
        return text

    def _plot_frequency_spectrum_shared(self, audio, sr, n_fft, scale_type, linear_limit=20000):
        if not MATPLOTLIB_AVAILABLE: return torch.zeros(CONST_BLANK_PLOT_SIZE)
        try:
            freqs, psd = signal.welch(audio, sr, nperseg=n_fft)
            psd_db = 10 * np.log10(np.maximum(psd, CONST_EPSILON))
            
            if SCIPY_IMG_AVAILABLE: psd_smooth = gaussian_filter1d(psd_db, sigma=1.0)
            else: psd_smooth = psd_db

            plt.style.use('dark_background')
            fig, ax = plt.subplots(figsize=CONST_PLOT_FIGSIZE)
            
            min_val = np.min(psd_smooth)
            max_val = np.max(psd_smooth)
            y_min = max(min_val - 10, -140) 
            y_max = min(max_val + 10, 10)
            
            nyquist_freq = sr / 2
            
            if scale_type == "Logarithmic":
                ax.semilogx(freqs, psd_smooth, color='#00ffcc', linewidth=2.0, alpha=1.0) 
                ax.fill_between(freqs, y_min, psd_smooth, color='#00ffcc', alpha=0.2)
                ax.set_xlim(20, min(20000, nyquist_freq))
            else:
                safe_linear_limit = min(linear_limit, nyquist_freq)
                ax.plot(freqs, psd_smooth, color='#00ffcc', linewidth=2.0, alpha=1.0)
                ax.fill_between(freqs, y_min, psd_smooth, color='#00ffcc', alpha=0.2)
                ax.set_xlim(0, safe_linear_limit)

            ax.set_ylim(y_min, y_max)
            ax.set_title(f"Frequency Response ({scale_type})", fontsize=10)
            ax.set_ylabel("dB")
            ax.grid(True, alpha=0.4, which="major", linestyle='-', color='#444444')
            ax.grid(True, alpha=0.2, which="minor", linestyle=':', color='#333333')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            plt.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=CONST_PLOT_DPI, facecolor=CONST_BACKGROUND_COLOR)
            buf.seek(0)
            plt.close(fig)
            img = Image.open(buf).convert("RGB")
            return torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0)
        except Exception as e: 
            logger.error(f"Plot Error: {e}")
            return torch.zeros(CONST_BLANK_PLOT_SIZE)

    def _plot_spectrogram_shared(self, audio, sr, n_fft, cmap='inferno'):
        if not MATPLOTLIB_AVAILABLE: return torch.zeros(CONST_BLANK_PLOT_SIZE)
        try:
            plt.style.use('dark_background')
            fig, ax = plt.subplots(figsize=CONST_PLOT_FIGSIZE)
            
            nyquist_freq = sr / 2
            ax.specgram(audio, NFFT=n_fft, Fs=sr, noverlap=n_fft//2, cmap=cmap)
            ax.set_title("Spectrogram", fontsize=10)
            ax.set_ylim(0, min(20000, nyquist_freq))
            plt.tight_layout()
            
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=CONST_PLOT_DPI, facecolor=CONST_BACKGROUND_COLOR)
            buf.seek(0)
            plt.close(fig)
            img = Image.open(buf).convert("RGB")
            return torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0)
        except Exception as e:
            logger.error(f"Spectrogram Error: {e}")
            return torch.zeros(CONST_BLANK_PLOT_SIZE)

# =================================================================================
# == Node 1: Broadcast Analyzer (Standard)
# =================================================================================

class AudioSpectrumAnalyzer_Enhanced(MasteringBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {
                    "tooltip": "AUDIO INPUT\n• Purpose: The audio signal to analyze."
                }),
            },
            "optional": {
                "fft_size": ("INT", {
                    "default": 4096, "min": 512, "max": 8192, "step": 512,
                    "tooltip": (
                        "FFT SIZE\n"
                        "• Purpose: Determines frequency vs time resolution.\n"
                        "• Trade-offs: Higher values give better bass detail but smear transients.\n"
                        "\n⭐ Recommended: 4096 for standard mastering analysis."
                    )
                }),
                "frequency_scale": (["Logarithmic", "Linear"], {
                    "default": "Logarithmic",
                    "tooltip": (
                        "FREQUENCY SCALE\n"
                        "• Purpose: Defines the X-Axis plotting mode.\n"
                        "• Logarithmic: Musical view (focus on bass/mids).\n"
                        "• Linear: Technical view (Uniform Hz spacing).\n"
                        "\n⭐ Recommended: Logarithmic for musical content."
                    )
                }),
                "debug_mode": (["0 - Silent", "1 - Info", "2 - Verbose"], {
                    "default": "0 - Silent",
                    "tooltip": "LOGGING VERBOSITY\n• Controls console output and profiling."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("spectrum_plot", "spectrogram_plot", "lufs_report")
    FUNCTION = "analyze_audio"
    CATEGORY = "MD_Nodes/Audio Processing"

    def analyze_audio(self, audio, fft_size=4096, frequency_scale="Logarithmic", debug_mode="0 - Silent"):
        debug_level = int(debug_mode.split(" ")[0])
        profiler = PerformanceProfiler(enabled=(debug_level >= 1))
        profiler.start("total")
        
        if not isinstance(audio, dict) or 'waveform' not in audio:
            return (torch.zeros(CONST_BLANK_PLOT_SIZE), torch.zeros(CONST_BLANK_PLOT_SIZE), "No Audio Data")
        
        audio_tensor = audio['waveform']
        sample_rate = audio.get('sample_rate', CONST_DEFAULT_SAMPLE_RATE)
        
        if audio_tensor.ndim == 3: audio_data = audio_tensor[0].cpu().numpy()
        else: audio_data = audio_tensor.cpu().numpy()

        if audio_data.ndim == 2 and audio_data.shape[0] == 2:
            left, right = audio_data[0], audio_data[1]
            mono = np.mean(audio_data, axis=0)
            is_stereo = True
        else:
            mono = audio_data.flatten()
            left, right = mono, mono
            is_stereo = False

        if mono.size == 0: 
            return (torch.zeros(CONST_BLANK_PLOT_SIZE), torch.zeros(CONST_BLANK_PLOT_SIZE), "Empty Audio")

        profiler.start("analysis")
        if CORE_LOADED:
            lufs_meter = core.LUFSMeterCore(sample_rate)
            measurements = lufs_meter.measure_lufs(audio_data)
        else:
            logger.warning("Core Missing. Using rough LUFS estimate.")
            rms = np.sqrt(np.mean(audio_data**2))
            rms_db = 20 * np.log10(max(rms, CONST_EPSILON))
            measurements = {'lufs': rms_db - 23.0, 'true_peak_db': rms_db + 3.0, 'dynamic_range': 0.0}
            
        phase_corr = self._phase_correlation(left, right) if is_stereo else 1.0
        report = self._generate_lufs_report(mono, left, right, sample_rate, is_stereo, measurements, phase_corr)
        profiler.stop("analysis")
        
        if debug_level >= 1:
            logging.debug("\n" + "=" * 60)
            logging.info("📊 [Analyzer] AUDIO REPORT")
            logging.debug("=" * 60)
            logging.info(report)

        profiler.start("plotting")
        spectrum_img = self._plot_frequency_spectrum_shared(mono, sample_rate, fft_size, frequency_scale, linear_limit=20000)
        spectrogram_img = self._plot_spectrogram_shared(mono, sample_rate, fft_size)
        profiler.stop("plotting")

        profiler.stop("total")
        if debug_level >= 1: profiler.print_report()

        return (spectrum_img, spectrogram_img, report)

# =================================================================================
# == Node 2: Spectrum Visualizer
# =================================================================================

class AudioSpectrumVisualizer(MasteringBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "AUDIO INPUT\n• Signal to visualize."}),
                "fft_size": ("INT", {
                    "default": 4096, "min": 512, "max": 8192, "step": 512,
                    "tooltip": "FFT SIZE\n• Purpose: Resolution of analysis.\n⭐ 4096 is the best balance for full mixes."
                }),
                "scale_type": (["Logarithmic", "Linear"], {
                    "default": "Logarithmic",
                    "tooltip": "SCALE TYPE\n• Purpose: X-Axis frequency scaling.\n⭐ Use Linear to spot specific high-frequency noise."
                }),
                "linear_max_freq": ("INT", {
                    "default": 5000, "min": 1000, "max": 22050, "step": 500,
                    "tooltip": "LINEAR ZOOM LIMIT\n• Purpose: Sets max frequency for Linear scale (Auto-clamped to Nyquist).\n⭐ Set to 5000 to inspect low-mids."
                }),
                "colormap": (["inferno", "viridis", "plasma", "magma", "cividis"], {
                    "default": "inferno",
                    "tooltip": "COLORMAP\n• Purpose: Color scheme for the Spectrogram."
                }),
            },
            "optional": {
                "debug_mode": (["0 - Silent", "1 - Info", "2 - Verbose"], {
                    "default": "0 - Silent",
                    "tooltip": "LOGGING VERBOSITY\n• Controls console output detail."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("frequency_plot", "spectrogram_plot", "lufs_report")
    FUNCTION = "visualize"
    CATEGORY = "MD_Nodes/Debugging & Visualization"

    def visualize(self, audio, fft_size, scale_type, linear_max_freq, colormap, debug_mode="0 - Silent"):
        debug_level = int(debug_mode.split(" ")[0])
        profiler = PerformanceProfiler(enabled=(debug_level >= 1))
        profiler.start("total")
        
        if not isinstance(audio, dict) or 'waveform' not in audio:
            return (torch.zeros(CONST_BLANK_PLOT_SIZE), torch.zeros(CONST_BLANK_PLOT_SIZE), "No Audio Data")
            
        audio_tensor = audio['waveform']
        sr = audio.get('sample_rate', CONST_DEFAULT_SAMPLE_RATE)
        
        if audio_tensor.ndim == 3: audio_data = audio_tensor[0].cpu().numpy()
        else: audio_data = audio_tensor.cpu().numpy()
        
        if audio_data.ndim == 2 and audio_data.shape[0] == 2:
            left, right = audio_data[0], audio_data[1]
            mono = np.mean(audio_data, axis=0)
            is_stereo = True
        else:
            mono = audio_data.flatten()
            left, right = mono, mono
            is_stereo = False

        if mono.size == 0: 
            return (torch.zeros(CONST_BLANK_PLOT_SIZE), torch.zeros(CONST_BLANK_PLOT_SIZE), "Empty Audio")

        profiler.start("analysis")
        if CORE_LOADED:
            lufs_meter = core.LUFSMeterCore(sr)
            measurements = lufs_meter.measure_lufs(audio_data)
        else:
            rms = np.sqrt(np.mean(audio_data**2))
            rms_db = 20 * np.log10(max(rms, CONST_EPSILON))
            measurements = {'lufs': rms_db - 23.0, 'true_peak_db': rms_db + 3.0, 'dynamic_range': 0.0}
            
        phase_corr = self._phase_correlation(left, right) if is_stereo else 1.0
        report = self._generate_lufs_report(mono, left, right, sr, is_stereo, measurements, phase_corr)
        profiler.stop("analysis")

        profiler.start("plotting")
        freq_plot = self._plot_frequency_spectrum_shared(mono, sr, fft_size, scale_type, linear_limit=linear_max_freq)
        spec_plot = self._plot_spectrogram_shared(mono, sr, fft_size, colormap)
        profiler.stop("plotting")
        
        profiler.stop("total")
        if debug_level >= 1: profiler.print_report()

        return (freq_plot, spec_plot, report)


# =================================================================================
# == Node 3: LUFS Normalizer
# =================================================================================

class MD_LUFS_Normalizer(MasteringBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "AUDIO INPUT\n• Signal to be normalized."}),
                "target_lufs": ("FLOAT", {
                    "default": -14.0, "min": -30.0, "max": 0.0, "step": 0.1, 
                    "tooltip": (
                        "TARGET LUFS\n"
                        "• Purpose: Integrated loudness target level.\n"
                        "• Standards: -14.0 (Spotify/YouTube), -23.0 (EBU R128 Broadcast).\n"
                        "\n⭐ Recommended: -14.0 for music distribution."
                    )
                }),
                "true_peak_limit_db": ("FLOAT", {
                    "default": -1.0, "min": -10.0, "max": 0.0, "step": 0.1, 
                    "tooltip": (
                        "TRUE PEAK LIMIT\n"
                        "• Purpose: Maximum allowed peak level (dBTP) to prevent clipping.\n"
                        "\n⭐ Recommended: -1.0 for safe streaming."
                    )
                }),
            },
            "optional": {
                "debug_mode": (["0 - Silent", "1 - Info", "2 - Verbose"], {"default": "0 - Silent", "tooltip": "LOGGING VERBOSITY"}),
            }
        }
    
    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "report")
    FUNCTION = "normalize_lufs"
    CATEGORY = "MD_Nodes/Audio Processing"

    def normalize_lufs(self, audio, target_lufs, true_peak_limit_db, debug_mode="0 - Silent"):
        debug_level = int(debug_mode.split(" ")[0])
        profiler = PerformanceProfiler(enabled=(debug_level >= 1))
        profiler.start("total")
        
        try:
            audio_batch, input_sr, device = self._unpack_audio_batch(audio)
            processed_batch = []
            reports = []
            
            if not CORE_LOADED:
                logger.warning(f"⚠️ Broadcast Core Missing: {CORE_ERROR}. Audio passed through unchanged.")
                return (audio, "Core Missing. No change applied.")
            
            for i, audio_item in enumerate(audio_batch):
                profiler.start(f"process_batch_{i}")
                
                lufs_meter = core.LUFSMeterCore(input_sr)
                
                # Measure first, compute exact gain delta, apply directly.
                # This bypasses any internal measurement drift in normalize_to_lufs.
                before = lufs_meter.measure_lufs(audio_item)
                measured_lufs = before['lufs']
                
                gain_db = target_lufs - measured_lufs
                gain_linear = 10.0 ** (gain_db / 20.0)
                normalized = audio_item * gain_linear
                
                # True peak safety: use oversampled detection, only clamp if exceeded.
                true_peak_limit_linear = 10.0 ** (true_peak_limit_db / 20.0)
                true_peak = lufs_meter._calculate_true_peak(normalized)
                if true_peak > true_peak_limit_linear:
                    normalized *= (true_peak_limit_linear / true_peak)
                
                after = lufs_meter.measure_lufs(normalized)
                profiler.stop(f"process_batch_{i}")
                
                if i == 0:
                    report = (
                        f"LUFS Normalization:\n"
                        f"  {measured_lufs:.1f} -> {after['lufs']:.1f} LUFS\n"
                        f"  Gain: {gain_db:+.1f} dB"
                    )
                    reports.append(report)
                    if debug_level >= 1: logger.info(report)
                
                processed_batch.append(normalized)
            
            out_tensor = torch.from_numpy(np.stack(processed_batch)).to(device).float()
            profiler.stop("total")
            if debug_level >= 1: profiler.print_report()
            
            return ({"waveform": out_tensor, "sample_rate": input_sr}, reports[0] if reports else "")
        except Exception as e:
            logger.error(f"LUFS normalization failed: {e}")
            return (audio, f"Error: {e}")

# =================================================================================
# == Node 4: Stereo Width Controller
# =================================================================================

class MD_Stereo_Width_Controller(MasteringBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "AUDIO INPUT\n• Stereo audio required. Mono will be bypassed."}),
                "width_percent": ("FLOAT", {
                    "default": 100.0, "min": 0.0, "max": 200.0, "step": 1.0, 
                    "tooltip": (
                        "STEREO WIDTH %\n"
                        "• Purpose: Adjusts the stereo image width via Mid/Side processing.\n"
                        "• Range: 0% (Mono) to 200% (Extra Wide).\n"
                        "• Trade-offs: >100% can cause phase cancellation.\n"
                        "\n⭐ Recommended: 100% (No change) or 110-120% for subtle enhancement."
                    )
                }),
            },
            "optional": {
                "debug_mode": (["0 - Silent", "1 - Info", "2 - Verbose"], {"default": "0 - Silent", "tooltip": "LOGGING VERBOSITY"}),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",) 
    FUNCTION = "adjust_width"
    CATEGORY = "MD_Nodes/Audio Processing"

    def adjust_width(self, audio, width_percent, debug_mode="0 - Silent"):
        debug_level = int(debug_mode.split(" ")[0])
        profiler = PerformanceProfiler(enabled=(debug_level >= 1))
        profiler.start("total")
        
        try:
            audio_batch, input_sr, device = self._unpack_audio_batch(audio)
            processed_batch = []
            
            if not CORE_LOADED:
                logger.warning(f"⚠️ Broadcast Core Missing: {CORE_ERROR}. Audio passed through unchanged.")
                return (audio,)
                
            for i, audio_item in enumerate(audio_batch):
                profiler.start(f"process_item_{i}")
                adjusted = core.MidSideProcessorCore.adjust_width(audio_item, width_percent)
                processed_batch.append(adjusted)
                profiler.stop(f"process_item_{i}")
            
            out_tensor = torch.from_numpy(np.stack(processed_batch)).to(device).float()
            profiler.stop("total")
            if debug_level >= 1: profiler.print_report()
            
            return ({"waveform": out_tensor, "sample_rate": input_sr},)
        except Exception as e:
            logger.error(f"Width adjustment failed: {e}")
            return (audio,)

# =================================================================================
# == Node Registration
# =================================================================================

NODE_CLASS_MAPPINGS = {
    "MD_Audio_Spectrum_Analyzer_Enhanced": AudioSpectrumAnalyzer_Enhanced,
    "MD_Audio_Spectrum_Visualizer": AudioSpectrumVisualizer,
    "MD_LUFS_Normalizer": MD_LUFS_Normalizer,
    "MD_Stereo_Width_Controller": MD_Stereo_Width_Controller,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MD_Audio_Spectrum_Analyzer_Enhanced": "MD: Audio Analyzer (Report + LUFS)",
    "MD_Audio_Spectrum_Visualizer": "MD: Audio Spectrum Visualizer (Plot)",
    "MD_LUFS_Normalizer": "MD: LUFS Normalizer",
    "MD_Stereo_Width_Controller": "MD: Stereo Width Controller",
}


# ==============================================================================
# == Unit Tests (smoke — runs without ComfyUI)
# ==============================================================================

if __name__ == "__main__":
    print("\n🧪 Smoke tests: MD_BroadcastTools")
    print("   VERSION :", VERSION)
    _pass = _fail = 0

    def _check(label, expr):
        global _pass, _fail
        if expr:
            print(f"  ✅  {label}")
            _pass += 1
        else:
            print(f"  ❌  {label}")
            _fail += 1

    _check("VERSION defined",    VERSION == "v2.1.3")
    _check("CONST CONST_MAX_PLOT_SAMPLES defined", CONST_MAX_PLOT_SAMPLES is not None)
    _check("CONST CONST_PLOT_DPI defined", CONST_PLOT_DPI is not None)
    _check("CONST CONST_WAVEFORM_COLOR defined", CONST_WAVEFORM_COLOR is not None)
    _check("NODE_CLASS_MAPPINGS defined",
           isinstance(NODE_CLASS_MAPPINGS, dict) and len(NODE_CLASS_MAPPINGS) > 0)
    _check("  class MD_Audio_Spectrum_Analyzer_Enhanced in map", "MD_Audio_Spectrum_Analyzer_Enhanced" in NODE_CLASS_MAPPINGS)
    _check("  class MD_Audio_Spectrum_Visualizer in map", "MD_Audio_Spectrum_Visualizer" in NODE_CLASS_MAPPINGS)
    _check("  class MD_LUFS_Normalizer in map", "MD_LUFS_Normalizer" in NODE_CLASS_MAPPINGS)

    print(f"\n  {_pass} passed, {_fail} failed")
    if _fail == 0:
        print("  🎉 All good.")
