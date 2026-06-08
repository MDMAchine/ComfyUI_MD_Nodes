# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# █▓▒░                                                                     ░▒▓█
# █▓▒░             MD_Nodes Wrapper: Mastering Suite (v2.1.0)              ░▒▓█
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
# ║   Professional audio mastering nodes.
# ║   NOTE: This is a public wrapper. Missing binaries will gracefully 
# ║   pass audio through unchanged to prevent workflow crashes.
# ╚════════════════════════════════════════════════════════════════════════════


VERSION = "v2.1.0"  # UPS v1.5.8

import os
import sys
import io
import time
import logging
import torch
import numpy as np
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
    return [os.path.abspath(c) for c in candidates if os.path.exists(c)]

CORE_LOCATIONS = find_core_paths()
CORE_LOADED = False
CORE_MODE = None
CORE_ERROR = None

for loc in CORE_LOCATIONS:
    if loc not in sys.path:
        sys.path.insert(0, loc)

try:
    import mastering_core_bin as core
    CORE_LOADED = True
    CORE_MODE = "Binary (Production)"
except ImportError as e1:
    try:
        import mastering_core as core
        CORE_LOADED = True
        CORE_MODE = "Source (Development)"
    except ImportError as e2:
        CORE_ERROR = f"Binary: {e1}\nSource: {e2}"

logger = logging.getLogger("MD_Nodes.Audio.MasteringChain")

# =================================================================================
# == Dependencies & Analytics
# =================================================================================
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

CONST_BLANK_PLOT_SIZE = (1, 256, 512, 3)
CONST_WAVEFORM_COLOR = '#87CEEB'
CONST_PLOT_DPI = 100
CONST_DEFAULT_SR = 44100

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
        logging.info("\n⏱️  PERFORMANCE (DSP):")
        total = self.get_total_time()
        logging.info(f"    • Total Time: {total:.4f}s")
        for op_name, times in sorted(self.timings.items()):
            avg = sum(times) / len(times)
            if len(times) == 1:
                logging.info(f"    • {op_name}: {avg:.4f}s")
            else:
                logging.info(f"    • {op_name}: {avg:.4f}s avg ({len(times)}x)")

class MasteringWrapperBase:
    """Base logic for Local Execution and Plotting."""
    
    def _unpack_audio(self, audio):
        waveform = audio['waveform']
        sr = audio.get('sample_rate', CONST_DEFAULT_SR)
        audio_np = waveform.cpu().numpy() if isinstance(waveform, torch.Tensor) else np.array(waveform)
        if len(audio_np.shape) == 2: audio_np = audio_np[np.newaxis, :, :] 
        return audio_np, sr, waveform.device

    def _pack_audio(self, processed_list, sr, device):
        out_tensor = torch.from_numpy(np.stack(processed_list)).to(device).float()
        return {"waveform": out_tensor, "sample_rate": sr}

    def _plot(self, audio_data, sr, title):
        if not MATPLOTLIB_AVAILABLE or audio_data is None: return torch.zeros(CONST_BLANK_PLOT_SIZE)
        try:
            plt.style.use('dark_background')
            fig, ax = plt.subplots(figsize=(10, 3), dpi=CONST_PLOT_DPI)
            plot_data = audio_data[0] if audio_data.ndim == 2 else audio_data
            if len(plot_data) > 150000: plot_data = plot_data[::len(plot_data)//150000]
            
            time_axis = np.linspace(0, len(plot_data)/sr, len(plot_data))
            ax.plot(time_axis, plot_data, color=CONST_WAVEFORM_COLOR, linewidth=0.5)
            ax.set_title(title, fontsize=9)
            ax.set_ylim(-1.05, 1.05)
            plt.tight_layout()
            
            buf = io.BytesIO()
            fig.savefig(buf, format='png', facecolor=fig.get_facecolor())
            buf.seek(0)
            img = Image.open(buf).convert("RGB")
            plt.close(fig)
            return torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0)
        except Exception: return torch.zeros(CONST_BLANK_PLOT_SIZE)

    def _execute_local(self, audio_np_list, sr, operation, params, gain=0.0):
        processed_list = []
        
        # Graceful Degradation
        if not CORE_LOADED:
            logger.warning(f"⚠️ Mastering Core Missing: {CORE_ERROR}. Passing audio through unchanged.")
            return audio_np_list 
            
        for item in audio_np_list:
            if operation == "chain":
                res = core.execute_full_chain(item, sr, gain, params)
            elif operation == "gain":
                res = core.process_gain(item, gain)
            elif operation == "eq":
                res = core.process_eq(item, sr, params)
            elif operation == "comp":
                res = core.process_compression(item, sr, params)
            elif operation == "limit":
                res = core.process_limiting(item, sr, params)
            processed_list.append(res)
            
        return processed_list

# =================================================================================
# == Node 1: Mastering Chain (Full)
# =================================================================================

class MasteringChainNode(MasteringWrapperBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {
                    "tooltip": (
                        "AUDIO INPUT\n"
                        "• Purpose: Raw audio waveform to be mastered.\n"
                        "• Range: [Batch, Channels, Samples].\n"
                        "• Output: Mastered audio dictionary."
                    )
                }),
                "sample_rate": ("INT", {
                    "default": 44100, "min": 8000, "max": 192000, 
                    "tooltip": "SAMPLE RATE\n• Purpose: Fallback sample rate if not provided by audio dict."
                }),
                "master_gain_db": ("FLOAT", {
                    "default": 0.0, "min": -60.0, "max": 24.0, "step": 0.1, 
                    "tooltip": (
                        "MASTER GAIN (DB)\n"
                        "• Purpose: Input gain adjustment before processing.\n"
                        "• Range: -60dB to +24dB.\n"
                        "\n⭐ Recommended: 0.0 unless audio is exceptionally quiet."
                    )
                }),
                
                # EQ Params
                "enable_lowpass": ("BOOLEAN", {
                    "default": False, 
                    "tooltip": "ENABLE LOWPASS\n• Purpose: Attenuate frequencies above the lowpass_freq."
                }),
                "lowpass_freq": ("FLOAT", {
                    "default": 18000.0, "min": 20.0, "max": 22000.0, "step": 10.0, 
                    "tooltip": "LOWPASS FREQUENCY\n• Purpose: The cutoff frequency in Hz."
                }),
                "lowpass_order": ("INT", {
                    "default": 4, "min": 1, "max": 8, 
                    "tooltip": "LOWPASS SLOPE\n• Purpose: Sharpness of the cut (Higher = steeper)."
                }),
                "enable_highpass": ("BOOLEAN", {
                    "default": False, 
                    "tooltip": "ENABLE HIGHPASS\n• Purpose: Attenuate sub-frequencies below highpass_freq."
                }),
                "highpass_freq": ("FLOAT", {
                    "default": 20.0, "min": 10.0, "max": 1000.0, "step": 1.0, 
                    "tooltip": "HIGHPASS FREQUENCY\n• Purpose: The cutoff frequency in Hz.\n⭐ Recommended: 30Hz to remove sub-mud."
                }),
                "highpass_order": ("INT", {
                    "default": 4, "min": 1, "max": 8, 
                    "tooltip": "HIGHPASS SLOPE\n• Purpose: Sharpness of the cut."
                }),
                "eq_high_shelf_gain_db": ("FLOAT", {
                    "default": 0.0, "min": -24.0, "max": 24.0, "step": 0.1, 
                    "tooltip": "HIGH SHELF GAIN\n• Purpose: Boost/cut high frequencies (dB)."
                }),
                "eq_high_shelf_freq": ("FLOAT", {
                    "default": 12000.0, "min": 1000.0, "max": 22000.0, "step": 10.0, 
                    "tooltip": "HIGH SHELF FREQUENCY\n• Purpose: Start point of the shelf (Hz)."
                }),
                "enable_low_shelf_eq": ("BOOLEAN", {
                    "default": False, 
                    "tooltip": "ENABLE LOW SHELF\n• Purpose: Toggle low frequency shelf filter."
                }),
                "eq_low_shelf_gain_db": ("FLOAT", {
                    "default": 0.0, "min": -24.0, "max": 24.0, "step": 0.1, 
                    "tooltip": "LOW SHELF GAIN\n• Purpose: Boost/cut low frequencies (dB)."
                }),
                "eq_low_shelf_freq": ("FLOAT", {
                    "default": 75.0, "min": 20.0, "max": 1000.0, "step": 1.0, 
                    "tooltip": "LOW SHELF FREQUENCY\n• Purpose: Start point of the shelf (Hz)."
                }),
                "enable_param_eq1": ("BOOLEAN", {"default": False, "tooltip": "ENABLE BAND 1\n• Purpose: Low-Mid parametric EQ band."}),
                "param_eq1_gain_db": ("FLOAT", {"default": 0.0, "min": -24.0, "max": 24.0, "step": 0.1, "tooltip": "BAND 1 GAIN (dB)"}),
                "param_eq1_freq": ("FLOAT", {"default": 55.0, "min": 20.0, "max": 22000.0, "step": 1.0, "tooltip": "BAND 1 FREQ (Hz)"}),
                "param_eq1_q": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 10.0, "step": 0.1, "tooltip": "BAND 1 Q FACTOR (Width)"}),
                "enable_param_eq2": ("BOOLEAN", {"default": False, "tooltip": "ENABLE BAND 2\n• Purpose: Mid parametric EQ band."}),
                "param_eq2_gain_db": ("FLOAT", {"default": 0.0, "min": -24.0, "max": 24.0, "step": 0.1, "tooltip": "BAND 2 GAIN (dB)"}),
                "param_eq2_freq": ("FLOAT", {"default": 125.0, "min": 20.0, "max": 22000.0, "step": 1.0, "tooltip": "BAND 2 FREQ (Hz)"}),
                "param_eq2_q": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 10.0, "step": 0.1, "tooltip": "BAND 2 Q FACTOR (Width)"}),
                "enable_param_eq3": ("BOOLEAN", {"default": False, "tooltip": "ENABLE BAND 3\n• Purpose: High-Mid parametric EQ band."}),
                "param_eq3_gain_db": ("FLOAT", {"default": 0.0, "min": -24.0, "max": 24.0, "step": 0.1, "tooltip": "BAND 3 GAIN (dB)"}),
                "param_eq3_freq": ("FLOAT", {"default": 1250.0, "min": 20.0, "max": 22000.0, "step": 1.0, "tooltip": "BAND 3 FREQ (Hz)"}),
                "param_eq3_q": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1, "tooltip": "BAND 3 Q FACTOR (Width)"}),
                "enable_param_eq4": ("BOOLEAN", {"default": False, "tooltip": "ENABLE BAND 4\n• Purpose: High parametric EQ band."}),
                "param_eq4_gain_db": ("FLOAT", {"default": 0.0, "min": -24.0, "max": 24.0, "step": 0.1, "tooltip": "BAND 4 GAIN (dB)"}),
                "param_eq4_freq": ("FLOAT", {"default": 5000.0, "min": 20.0, "max": 22000.0, "step": 1.0, "tooltip": "BAND 4 FREQ (Hz)"}),
                "param_eq4_q": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1, "tooltip": "BAND 4 Q FACTOR (Width)"}),
                
                # Comp Params
                "enable_comp": ("BOOLEAN", {
                    "default": False, 
                    "tooltip": "ENABLE COMPRESSOR\n• Purpose: Reduce dynamic range."
                }),
                "comp_type": (["Single-Band", "Multiband"], {
                    "default": "Multiband", 
                    "tooltip": (
                        "COMPRESSOR TYPE\n"
                        "• Single: Affects entire signal evenly.\n"
                        "• Multiband: Compresses Low/Mid/High independently.\n"
                        "\n⭐ Recommended: Multiband for mastering."
                    )
                }),
                "comp_threshold_db": ("FLOAT", {"default": -8.0, "min": -60.0, "max": 0.0, "step": 0.1, "tooltip": "SINGLE THRESHOLD (dB)"}),
                "comp_ratio": ("FLOAT", {"default": 2.5, "min": 1.0, "max": 20.0, "step": 0.1, "tooltip": "SINGLE RATIO"}),
                "comp_attack_ms": ("FLOAT", {"default": 20.0, "min": 0.1, "max": 1000.0, "step": 0.1, "tooltip": "SINGLE ATTACK (ms)"}),
                "comp_release_ms": ("FLOAT", {"default": 250.0, "min": 1.0, "max": 5000.0, "step": 1.0, "tooltip": "SINGLE RELEASE (ms)"}),
                "comp_makeup_gain_db": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 24.0, "step": 0.1, "tooltip": "SINGLE MAKEUP GAIN"}),
                
                "mb_crossover_low_mid_hz": ("FLOAT", {"default": 250.0, "min": 20.0, "max": 1000.0, "step": 1.0, "tooltip": "CROSSOVER: LOW-MID (Hz)"}),
                "mb_crossover_mid_high_hz": ("FLOAT", {"default": 4000.0, "min": 1000.0, "max": 15000.0, "step": 1.0, "tooltip": "CROSSOVER: MID-HIGH (Hz)"}),
                "mb_crossover_order": ("INT", {"default": 8, "min": 2, "max": 8, "tooltip": "CROSSOVER SLOPE"}),
                "mb_low_threshold_db": ("FLOAT", {"default": -10.0, "min": -60.0, "max": 0.0, "step": 0.1, "tooltip": "LOW BAND THRESHOLD"}),
                "mb_low_ratio": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 20.0, "step": 0.1, "tooltip": "LOW BAND RATIO"}),
                "mb_low_attack_ms": ("FLOAT", {"default": 30.0, "min": 0.1, "max": 1000.0, "step": 0.1, "tooltip": "LOW BAND ATTACK"}),
                "mb_low_release_ms": ("FLOAT", {"default": 300.0, "min": 1.0, "max": 5000.0, "step": 1.0, "tooltip": "LOW BAND RELEASE"}),
                "mb_low_makeup_gain_db": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 24.0, "step": 0.1, "tooltip": "LOW BAND MAKEUP"}),
                "mb_mid_threshold_db": ("FLOAT", {"default": -8.0, "min": -60.0, "max": 0.0, "step": 0.1, "tooltip": "MID BAND THRESHOLD"}),
                "mb_mid_ratio": ("FLOAT", {"default": 2.5, "min": 1.0, "max": 20.0, "step": 0.1, "tooltip": "MID BAND RATIO"}),
                "mb_mid_attack_ms": ("FLOAT", {"default": 20.0, "min": 0.1, "max": 1000.0, "step": 0.1, "tooltip": "MID BAND ATTACK"}),
                "mb_mid_release_ms": ("FLOAT", {"default": 180.0, "min": 1.0, "max": 5000.0, "step": 1.0, "tooltip": "MID BAND RELEASE"}),
                "mb_mid_makeup_gain_db": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 24.0, "step": 0.1, "tooltip": "MID BAND MAKEUP"}),
                "mb_high_threshold_db": ("FLOAT", {"default": -6.0, "min": -60.0, "max": 0.0, "step": 0.1, "tooltip": "HIGH BAND THRESHOLD"}),
                "mb_high_ratio": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 20.0, "step": 0.1, "tooltip": "HIGH BAND RATIO"}),
                "mb_high_attack_ms": ("FLOAT", {"default": 10.0, "min": 0.1, "max": 1000.0, "step": 0.1, "tooltip": "HIGH BAND ATTACK"}),
                "mb_high_release_ms": ("FLOAT", {"default": 120.0, "min": 1.0, "max": 5000.0, "step": 1.0, "tooltip": "HIGH BAND RELEASE"}),
                "mb_high_makeup_gain_db": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 24.0, "step": 0.1, "tooltip": "HIGH BAND MAKEUP"}),
                
                # Limiter Params
                "enable_limiter": ("BOOLEAN", {
                    "default": False, 
                    "tooltip": (
                        "ENABLE LIMITER\n"
                        "• Purpose: Brickwall limits audio peaks to prevent clipping.\n"
                        "\n⭐ Recommended: True (placed at the end of the chain)."
                    )
                }),
                "limiter_ceiling_db": ("FLOAT", {
                    "default": -0.1, "min": -10.0, "max": 0.0, "step": 0.1, 
                    "tooltip": "LIMITER CEILING\n• Purpose: Absolute maximum volume (dB)."
                }),
                "limiter_release_ms": ("FLOAT", {
                    "default": 50.0, "min": 1.0, "max": 2000.0, "step": 1.0, 
                    "tooltip": "LIMITER RELEASE\n• Purpose: Time to recover from limiting (ms)."
                }),
                
                # Debug & Profiling
                "debug_mode": (["0 - Silent", "1 - Info", "2 - Verbose"], {
                    "default": "0 - Silent", 
                    "tooltip": "LOGGING VERBOSITY\n• Controls console output detail level."
                }),
                "enable_profiling": ("BOOLEAN", {
                    "default": False, 
                    "tooltip": "ENABLE PROFILING\n• Enable detailed DSP operation timing."
                }),
            }
        }

    RETURN_TYPES = ("AUDIO", "IMAGE", "IMAGE")
    RETURN_NAMES = ("audio", "waveform_before", "waveform_after")
    FUNCTION = "apply_chain"
    CATEGORY = "MD_Nodes/Audio Processing"

    def apply_chain(self, audio, sample_rate, master_gain_db, **kwargs):
        debug_mode = kwargs.get("debug_mode", "0 - Silent")
        debug_level = int(debug_mode.split(" ")[0])
        profiler = PerformanceProfiler(enabled=kwargs.get("enable_profiling", False) or debug_level >= 1)
        profiler.start("total_execution")

        # 1. Unpack
        profiler.start("tensor_unpack")
        audio_batch, input_sr, device = self._unpack_audio(audio)
        sr = input_sr if input_sr else sample_rate
        profiler.stop("tensor_unpack")
        
        # 2. Process Local
        profiler.start("dsp_processing")
        processed_list = self._execute_local(audio_batch, sr, "chain", kwargs, gain=master_gain_db)
        profiler.stop("dsp_processing")
        
        # 3. Viz
        profiler.start("visualization")
        plot_before = self._plot(audio_batch[0], sr, "Before")
        plot_after = self._plot(processed_list[0], sr, "After")
        profiler.stop("visualization")
        
        profiler.stop("total_execution")
        if debug_level >= 1: profiler.print_report()
        
        return (self._pack_audio(processed_list, sr, device), plot_before, plot_after)

# =================================================================================
# == Node 2: Gain (Modular)
# =================================================================================

class MasteringGainNode(MasteringWrapperBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "AUDIO INPUT"}),
                "gain_db": ("FLOAT", {
                    "default": 0.0, "min": -60.0, "max": 24.0,
                    "tooltip": "GAIN (dB)\n• Purpose: Boost or cut volume."
                }),
            }
        }
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "apply"
    CATEGORY = "MD_Nodes/Audio Processing"

    def apply(self, audio, gain_db):
        audio_batch, sr, device = self._unpack_audio(audio)
        processed = self._execute_local(audio_batch, sr, "gain", {}, gain=gain_db)
        return (self._pack_audio(processed, sr, device),)

# =================================================================================
# == Node 3: EQ (Modular)
# =================================================================================

class MasteringEQNode(MasteringWrapperBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "AUDIO INPUT"}),
                "sample_rate": ("INT", {
                    "default": 44100, "min": 8000, "max": 192000, 
                    "tooltip": "SAMPLE RATE\n• Purpose: Fallback sample rate if not provided by audio dict."
                }),
                "enable_lowpass": ("BOOLEAN", {"default": False, "tooltip": "ENABLE LOWPASS"}),
                "lowpass_freq": ("FLOAT", {"default": 18000.0, "min": 20.0, "max": 22000.0, "tooltip": "LOWPASS FREQ"}),
                "lowpass_order": ("INT", {"default": 4, "min": 1, "max": 8, "tooltip": "LOWPASS SLOPE"}),
                "enable_highpass": ("BOOLEAN", {"default": False, "tooltip": "ENABLE HIGHPASS"}),
                "highpass_freq": ("FLOAT", {"default": 20.0, "min": 10.0, "max": 1000.0, "tooltip": "HIGHPASS FREQ"}),
                "highpass_order": ("INT", {"default": 4, "min": 1, "max": 8, "tooltip": "HIGHPASS SLOPE"}),
                "eq_high_shelf_gain_db": ("FLOAT", {"default": 0.0, "min": -24.0, "max": 24.0, "tooltip": "HIGH SHELF GAIN"}),
                "eq_high_shelf_freq": ("FLOAT", {"default": 12000.0, "min": 1000.0, "max": 22000.0, "tooltip": "HIGH SHELF FREQ"}),
                "enable_low_shelf_eq": ("BOOLEAN", {"default": False, "tooltip": "ENABLE LOW SHELF"}),
                "eq_low_shelf_gain_db": ("FLOAT", {"default": 0.0, "min": -24.0, "max": 24.0, "tooltip": "LOW SHELF GAIN"}),
                "eq_low_shelf_freq": ("FLOAT", {"default": 75.0, "min": 20.0, "max": 1000.0, "tooltip": "LOW SHELF FREQ"}),
                "enable_param_eq1": ("BOOLEAN", {"default": False, "tooltip": "ENABLE BAND 1"}),
                "param_eq1_gain_db": ("FLOAT", {"default": 0.0, "min": -24.0, "max": 24.0}),
                "param_eq1_freq": ("FLOAT", {"default": 55.0, "min": 20.0, "max": 22000.0}),
                "param_eq1_q": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 10.0}),
                "enable_param_eq2": ("BOOLEAN", {"default": False, "tooltip": "ENABLE BAND 2"}),
                "param_eq2_gain_db": ("FLOAT", {"default": 0.0, "min": -24.0, "max": 24.0}),
                "param_eq2_freq": ("FLOAT", {"default": 125.0, "min": 20.0, "max": 22000.0}),
                "param_eq2_q": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 10.0}),
                "enable_param_eq3": ("BOOLEAN", {"default": False, "tooltip": "ENABLE BAND 3"}),
                "param_eq3_gain_db": ("FLOAT", {"default": 0.0, "min": -24.0, "max": 24.0}),
                "param_eq3_freq": ("FLOAT", {"default": 1250.0, "min": 20.0, "max": 22000.0}),
                "param_eq3_q": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0}),
                "enable_param_eq4": ("BOOLEAN", {"default": False, "tooltip": "ENABLE BAND 4"}),
                "param_eq4_gain_db": ("FLOAT", {"default": 0.0, "min": -24.0, "max": 24.0}),
                "param_eq4_freq": ("FLOAT", {"default": 5000.0, "min": 20.0, "max": 22000.0}),
                "param_eq4_q": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0}),
            }
        }
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "apply"
    CATEGORY = "MD_Nodes/Audio Processing"

    def apply(self, audio, sample_rate, **kwargs):
        audio_batch, input_sr, device = self._unpack_audio(audio)
        sr = input_sr if input_sr else sample_rate
        processed = self._execute_local(audio_batch, sr, "eq", kwargs)
        return (self._pack_audio(processed, sr, device),)

# =================================================================================
# == Node 4: Compressor (Modular)
# =================================================================================

class MasteringCompressorNode(MasteringWrapperBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "AUDIO INPUT"}),
                "sample_rate": ("INT", {
                    "default": 44100, "min": 8000, "max": 192000, 
                    "tooltip": "SAMPLE RATE\n• Purpose: Fallback sample rate if not provided by audio dict."
                }),
                "enable_comp": ("BOOLEAN", {"default": True, "tooltip": "ENABLE COMPRESSOR"}),
                "comp_type": (["Single-Band", "Multiband"], {"default": "Multiband"}),
                "comp_threshold_db": ("FLOAT", {"default": -8.0, "min": -60.0, "max": 0.0}),
                "comp_ratio": ("FLOAT", {"default": 2.5, "min": 1.0, "max": 20.0}),
                "comp_attack_ms": ("FLOAT", {"default": 20.0, "min": 0.1, "max": 1000.0}),
                "comp_release_ms": ("FLOAT", {"default": 250.0, "min": 1.0, "max": 5000.0}),
                "comp_makeup_gain_db": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 24.0}),
                "mb_crossover_low_mid_hz": ("FLOAT", {"default": 250.0}),
                "mb_crossover_mid_high_hz": ("FLOAT", {"default": 4000.0}),
                "mb_crossover_order": ("INT", {"default": 8}),
                "mb_low_threshold_db": ("FLOAT", {"default": -10.0, "min": -60.0, "max": 0.0}),
                "mb_low_ratio": ("FLOAT", {"default": 3.0}),
                "mb_low_attack_ms": ("FLOAT", {"default": 30.0}),
                "mb_low_release_ms": ("FLOAT", {"default": 300.0}),
                "mb_low_makeup_gain_db": ("FLOAT", {"default": 0.0}),
                "mb_mid_threshold_db": ("FLOAT", {"default": -8.0, "min": -60.0, "max": 0.0}),
                "mb_mid_ratio": ("FLOAT", {"default": 2.5}),
                "mb_mid_attack_ms": ("FLOAT", {"default": 20.0}),
                "mb_mid_release_ms": ("FLOAT", {"default": 180.0}),
                "mb_mid_makeup_gain_db": ("FLOAT", {"default": 0.0}),
                "mb_high_threshold_db": ("FLOAT", {"default": -6.0, "min": -60.0, "max": 0.0}),
                "mb_high_ratio": ("FLOAT", {"default": 2.0}),
                "mb_high_attack_ms": ("FLOAT", {"default": 10.0}),
                "mb_high_release_ms": ("FLOAT", {"default": 120.0}),
                "mb_high_makeup_gain_db": ("FLOAT", {"default": 0.0}),
            }
        }
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "apply"
    CATEGORY = "MD_Nodes/Audio Processing"

    def apply(self, audio, sample_rate, **kwargs):
        audio_batch, input_sr, device = self._unpack_audio(audio)
        sr = input_sr if input_sr else sample_rate
        processed = self._execute_local(audio_batch, sr, "comp", kwargs)
        return (self._pack_audio(processed, sr, device),)

# =================================================================================
# == Node 5: Limiter (Modular)
# =================================================================================

class MasteringLimiterNode(MasteringWrapperBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "AUDIO INPUT"}),
                "sample_rate": ("INT", {
                    "default": 44100, "min": 8000, "max": 192000, 
                    "tooltip": "SAMPLE RATE\n• Purpose: Fallback sample rate if not provided by audio dict."
                }),
                "enable_limiter": ("BOOLEAN", {"default": True, "tooltip": "ENABLE LIMITER"}),
                "limiter_ceiling_db": ("FLOAT", {"default": -0.1, "min": -10.0, "max": 0.0, "step": 0.1, "tooltip": "LIMITER CEILING"}),
                "limiter_release_ms": ("FLOAT", {"default": 50.0, "min": 1.0, "max": 2000.0, "step": 1.0, "tooltip": "LIMITER RELEASE"}),
            }
        }
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "apply"
    CATEGORY = "MD_Nodes/Audio Processing"

    def apply(self, audio, sample_rate, **kwargs):
        audio_batch, input_sr, device = self._unpack_audio(audio)
        sr = input_sr if input_sr else sample_rate
        processed = self._execute_local(audio_batch, sr, "limit", kwargs)
        return (self._pack_audio(processed, sr, device),)

# =================================================================================
# == Registration
# =================================================================================

NODE_CLASS_MAPPINGS = {
    "MasteringChainNode": MasteringChainNode,
    "MD_Mastering_Gain": MasteringGainNode,
    "MD_Mastering_EQ": MasteringEQNode,
    "MD_Mastering_Compressor": MasteringCompressorNode,
    "MD_Mastering_Limiter": MasteringLimiterNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MasteringChainNode": "MD: Mastering Chain (Full)",
    "MD_Mastering_Gain": "MD: Mastering Gain",
    "MD_Mastering_EQ": "MD: Mastering EQ",
    "MD_Mastering_Compressor": "MD: Mastering Compressor",
    "MD_Mastering_Limiter": "MD: Mastering Limiter",
}


# ==============================================================================
# == Unit Tests (smoke — runs without ComfyUI)
# ==============================================================================

if __name__ == "__main__":
    print("\n🧪 Smoke tests: MD_MasteringChain")
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

    _check("VERSION defined",    VERSION == "v2.1.0")
    _check("CONST CONST_WAVEFORM_COLOR defined", CONST_WAVEFORM_COLOR is not None)
    _check("CONST CONST_PLOT_DPI defined", CONST_PLOT_DPI is not None)
    _check("CONST CONST_DEFAULT_SR defined", CONST_DEFAULT_SR is not None)
    _check("NODE_CLASS_MAPPINGS defined",
           isinstance(NODE_CLASS_MAPPINGS, dict) and len(NODE_CLASS_MAPPINGS) > 0)
    _check("  class MasteringChainNode in map", "MasteringChainNode" in NODE_CLASS_MAPPINGS)
    _check("  class MD_Mastering_Gain in map", "MD_Mastering_Gain" in NODE_CLASS_MAPPINGS)
    _check("  class MD_Mastering_EQ in map", "MD_Mastering_EQ" in NODE_CLASS_MAPPINGS)

    print(f"\n  {_pass} passed, {_fail} failed")
    if _fail == 0:
        print("  🎉 All good.")
