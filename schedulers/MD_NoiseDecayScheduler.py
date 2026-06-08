# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# █▓▒░                                                                     ░▒▓█
# █▓▒░       MD_Nodes/MD_NoiseDecayScheduler – Advanced Decay v1.6.0       ░▒▓█
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
# ║    Advanced mathematical noise decay scheduler wrapper. 
# ║    Generates a customizable noise decay curve object and visualizes it.
# ║    NOTE: This is a public wrapper. Missing binaries will safely degrade 
# ║    to a flat linear schedule.
# ╚════════════════════════════════════════════════════════════════════════════


VERSION = "v1.6.0"  # UPS v1.5.8

import logging
import traceback
import time
import io
import torch
import numpy as np
import os
import sys

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
    import noise_decay_core_bin as core
    CORE_LOADED = True
    CORE_MODE = "Binary (Production)"
except ImportError as e1:
    try:
        import noise_decay_core as core
        CORE_LOADED = True
        CORE_MODE = "Source (Development)"
    except ImportError as e2:
        CORE_ERROR = f"Binary: {e1}\nSource: {e2}"

# =================================================================================
# == Configuration Constants
# =================================================================================

CONST_PLOT_DPI = 120
CONST_PLOT_FIGSIZE = (10, 4)
CONST_WAVEFORM_COLOR = '#87CEEB'       
CONST_GRID_COLOR = '#555555'

logger = logging.getLogger("MD_Nodes.Schedulers.NoiseDecay")

# =================================================================================
# == Dependency Checks
# =================================================================================

try:
    import matplotlib
    matplotlib.use('Agg') 
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("Matplotlib not available. Plotting disabled.")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("PIL not available. Plotting disabled.")

# =================================================================================
# == Performance Profiler (Enterprise Standard)
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
        logging.info("\n⏱️  PERFORMANCE (Math Ops):")
        total = self.get_total_time()
        logging.info(f"    • Total Time: {total:.4f}s")
        for op_name, times in sorted(self.timings.items()):
            avg = sum(times) / len(times)
            if len(times) == 1:
                logging.info(f"    • {op_name}: {avg:.4f}s")
            else:
                logging.info(f"    • {op_name}: {avg:.4f}s avg ({len(times)}x)")

# =================================================================================
# == Core Node Class
# =================================================================================

class NoiseDecayScheduler_Custom:
    """
    MD: Noise Decay Scheduler (Advanced)
    Generates a customizable noise decay curve object and visualizes it.
    """
    CATEGORY = "MD_Nodes/Schedulers"
    FUNCTION = "generate"
    RETURN_TYPES = ("SCHEDULER", "IMAGE")
    RETURN_NAMES = ("scheduler", "plot_image")
    DESCRIPTION = "Advanced noise decay scheduler with visualization."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "algorithm_type": (
                    ["polynomial", "sigmoidal", "piecewise", "fourier", "exponential", "gaussian"],
                    {
                        "default": "polynomial",
                        "tooltip": (
                            "DECAY ALGORITHM\n"
                            "• Purpose: Defines the mathematical function shaping the curve.\n"
                            "• Options: Polynomial (Linear/Quad), Sigmoidal (S-Curve), Gaussian (Bell).\n"
                            "• Trade-offs: Non-standard shapes (Fourier) may yield unpredictable injection results.\n"
                            "\n⭐ Recommended: 'polynomial' for standard diffusion ease-in."
                        )
                    }
                ),
                "decay_exponent": ("FLOAT", {
                    "default": 2.0, "min": 0.1, "max": 10.0, "step": 0.1,
                    "tooltip": (
                        "DECAY EXPONENT (STEEPNESS)\n"
                        "• Purpose: Controls the rate/curvature of the decay drop-off.\n"
                        "• Options: 1.0 = Linear, 2.0 = Standard Quadratic.\n"
                        "• Trade-offs: Extreme values (>5.0) cause massive steps at the very beginning.\n"
                        "\n⭐ Recommended: 2.0 for standard ease-in."
                    )
                }),
                "start_value": ("FLOAT", {
                    "default": 1.0, "min": -2.0, "max": 2.0, "step": 0.01,
                    "tooltip": (
                        "START VALUE\n"
                        "• Purpose: The Y-value magnitude at the very first step (t=0).\n"
                        "• Range: Typically 0.0 to 1.0.\n"
                        "\n⭐ Recommended: 1.0 for standard decay, 0.0 for attack envelopes."
                    )
                }),
                "end_value": ("FLOAT", {
                    "default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01,
                    "tooltip": (
                        "END VALUE\n"
                        "• Purpose: The Y-value magnitude at the final step (t=max).\n"
                        "• Range: Typically 0.0 to 1.0.\n"
                        "\n⭐ Recommended: 0.0 for standard decay."
                    )
                }),
                "invert_curve": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "INVERT CURVE\n"
                        "• Purpose: Flips the curve vertically (Calculates 1 - y).\n"
                        "• Effect: Quickly turns a decay schedule into an attack schedule.\n"
                        "\n⭐ Recommended: False unless intentionally building inverse logic."
                    )
                }),
                "preview_steps": ("INT", {
                    "default": 20, "min": 2, "max": 1000,
                    "tooltip": (
                        "PREVIEW STEPS\n"
                        "• Purpose: Resolution fidelity of the generated plot image.\n"
                        "• Range: 2 to 1000 samples.\n"
                        "• Trade-offs: Higher values yield smoother curves but slightly slower node execution.\n"
                        "\n⭐ Recommended: 20-50 is sufficient for visualization."
                    )
                }),
                "use_caching": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "USE CACHING\n"
                        "• Purpose: Stores results in memory to avoid recalculating identical curves.\n"
                        "• Trade-offs: Uses trace memory, but massively speeds up queue runs.\n"
                        "\n⭐ Recommended: True."
                    )
                }),
                "enable_temporal_smoothing": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "ENABLE SMOOTHING\n"
                        "• Purpose: Applies a moving average filter to the final curve.\n"
                        "• Effect: Softens sharp transitions, highly useful in 'piecewise' mode.\n"
                        "\n⭐ Recommended: False for pure mathematical curves."
                    )
                }),
            },
            "optional": {
                "smoothing_window": ("INT", {
                    "default": 3, "min": 2, "max": 20,
                    "tooltip": (
                        "SMOOTHING WINDOW\n"
                        "• Purpose: The rolling kernel size of the moving average filter.\n"
                        "• Requirement: Active only if 'enable_temporal_smoothing' is True.\n"
                        "\n⭐ Recommended: 3-5 to retain general shape while blurring spikes."
                    )
                }),
                "custom_piecewise_points": ("STRING", {
                    "default": "1.0,0.5,0.0",
                    "tooltip": (
                        "PIECEWISE POINTS\n"
                        "• Purpose: Manual control anchors for the curve.\n"
                        "• Requirement: Active only if Algorithm = 'piecewise'.\n"
                        "• Format: Comma-separated floats (e.g., '1.0, 0.8, 0.2, 0.0')."
                    )
                }),
                "fourier_frequency": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1,
                    "tooltip": (
                        "FOURIER FREQUENCY\n"
                        "• Purpose: Number of oscillating cycles for the algorithm.\n"
                        "• Requirement: Active only if Algorithm = 'fourier'.\n"
                        "\n⭐ Recommended: 1.0 creates a single bell-like cosine wave."
                    )
                }),
                "debug_mode": (["0 - Silent", "1 - Info", "2 - Verbose"], {
                    "default": "0 - Silent",
                    "tooltip": (
                        "LOGGING VERBOSITY\n"
                        "• Purpose: Controls console output and structural profiling.\n"
                        "• Options: 0 (Production), 1 (Analytics Report), 2 (Full trace).\n"
                        "\n⭐ Recommended: 0 - Silent."
                    )
                }),
                "enable_profiling": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "ENABLE PROFILING\n"
                        "• Purpose: Measure generation time for performance tuning.\n"
                        "• Note: Automatically enabled if debug_mode >= 1."
                    )
                }),
            }
        }

    def _generate_plot(self, decay_values, title="Decay Curve"):
        """Generates visual plot tensor."""
        if not MATPLOTLIB_AVAILABLE or not PIL_AVAILABLE:
            return torch.zeros((1, 64, 64, 3), dtype=torch.float32)

        try:
            steps = list(range(len(decay_values)))
            
            plt.style.use('dark_background')
            fig, ax = plt.subplots(figsize=CONST_PLOT_FIGSIZE)
            
            ax.plot(steps, decay_values, color=CONST_WAVEFORM_COLOR, linewidth=2.0, label='Value')
            ax.fill_between(steps, decay_values, alpha=0.2, color=CONST_WAVEFORM_COLOR)
            
            # Add markers
            if len(steps) < 50:
                ax.scatter(steps, decay_values, color='#FFD700', s=15, alpha=0.6)

            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.set_xlabel("Step Index", fontsize=9)
            ax.set_ylabel("Value", fontsize=9)
            ax.grid(True, linestyle=':', alpha=0.3, color=CONST_GRID_COLOR)
            
            plt.tight_layout()

            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=CONST_PLOT_DPI, facecolor=fig.get_facecolor())
            buf.seek(0)
            plt.close(fig)

            img = Image.open(buf).convert("RGB")
            img_np = np.array(img).astype(np.float32) / 255.0
            return torch.from_numpy(img_np).unsqueeze(0)

        except Exception as e:
            logger.error(f"Plot generation error: {e}")
            return torch.zeros((1, 64, 64, 3), dtype=torch.float32)

    def generate(self, **kwargs):
        """Main execution: Creates scheduler and generates preview plot."""
        
        # 1. Config Logging
        debug_mode_raw = kwargs.get("debug_mode", "0 - Silent")
        if isinstance(debug_mode_raw, str):
            debug_level = int(debug_mode_raw.split(" ")[0])
        else:
            debug_level = int(debug_mode_raw)
            
        profiling = kwargs.get("enable_profiling", False)
        
        profiler = PerformanceProfiler(enabled=profiling or debug_level >= 1)
        profiler.start("total_execution")
        
        if debug_level >= 2: logger.setLevel(logging.DEBUG)
        elif debug_level >= 1: logger.setLevel(logging.INFO)
        else: logger.setLevel(logging.WARNING)

        # 2. Graceful Degradation Check
        if not CORE_LOADED:
            if debug_level >= 1:
                logger.warning(f"⚠️ Noise Decay Core Missing: {CORE_ERROR}. Falling back to flat dummy object.")
            # Fallback mock object and blank image
            class DummyScheduler:
                def get_decay(self, steps): return [0.0] * steps
            
            return (DummyScheduler(), torch.zeros((1, 64, 64, 3), dtype=torch.float32))

        try:
            # 3. Instantiate Scheduler (Using Core)
            profiler.start("core_init")
            scheduler_obj = core.NoiseDecayObject(**kwargs)
            profiler.stop("core_init")
            
            # 4. Generate Preview Data
            preview_steps = kwargs.get("preview_steps", 20)
            
            profiler.start("core_calc")
            preview_decay = scheduler_obj.get_decay(preview_steps)
            profiler.stop("core_calc")
            
            if debug_level >= 1:
                logger.info(f"Generated decay curve ({kwargs.get('algorithm_type')}) with {preview_steps} preview steps.")
            if debug_level >= 2:
                logger.debug(f"Preview Values: {preview_decay}")

            # 5. Generate Plot
            profiler.start("plotting")
            plot_title = f"{kwargs.get('algorithm_type').title()} Decay (Exp: {kwargs.get('decay_exponent')})"
            plot_image = self._generate_plot(preview_decay, title=plot_title)
            profiler.stop("plotting")
            
            # 6. Reporting
            profiler.stop("total_execution")
            
            if debug_level >= 1:
                logging.info("\n" + "="*60)
                logging.info("📊 [NoiseDecay] ANALYTICS REPORT")
                logging.info("="*60)
                logging.info(f"    • Algorithm: {kwargs.get('algorithm_type')}")
                logging.info(f"    • Range:     {kwargs.get('start_value')} -> {kwargs.get('end_value')}")
                logging.info(f"    • Preview:   {preview_steps} steps")
                profiler.print_report()
                logging.info("="*60)

            return (scheduler_obj, plot_image)

        except Exception as e:
            logger.error(f"Generate failed: {e}")
            logging.debug(traceback.format_exc())
            
            if CORE_LOADED:
                default_sched = core.NoiseDecayObject(algorithm_type="polynomial")
            else:
                default_sched = None 
                
            blank_plot = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return (default_sched, blank_plot)

# =================================================================================
# == Node Registration
# =================================================================================

NODE_CLASS_MAPPINGS = {
    "NoiseDecayScheduler_Custom": NoiseDecayScheduler_Custom,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "NoiseDecayScheduler_Custom": "MD: Noise Decay Scheduler (Advanced)",
}


# ==============================================================================
# == Unit Tests (smoke — runs without ComfyUI)
# ==============================================================================

if __name__ == "__main__":
    print("\n🧪 Smoke tests: MD_NoiseDecayScheduler")
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

    _check("VERSION defined",    VERSION == "v1.6.0")
    _check("CONST CONST_PLOT_DPI defined", CONST_PLOT_DPI is not None)
    _check("CONST CONST_WAVEFORM_COLOR defined", CONST_WAVEFORM_COLOR is not None)
    _check("CONST CONST_GRID_COLOR defined", CONST_GRID_COLOR is not None)
    _check("NODE_CLASS_MAPPINGS defined",
           isinstance(NODE_CLASS_MAPPINGS, dict) and len(NODE_CLASS_MAPPINGS) > 0)
    _check("  class NoiseDecayScheduler_Custom in map", "NoiseDecayScheduler_Custom" in NODE_CLASS_MAPPINGS)

    print(f"\n  {_pass} passed, {_fail} failed")
    if _fail == 0:
        print("  🎉 All good.")
