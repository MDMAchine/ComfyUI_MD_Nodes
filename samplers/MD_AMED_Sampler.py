# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# █▓▒░                                                                     ░▒▓█
# █▓▒░     MD_Nodes/MD_AMED_Sampler – Advanced Corrected Euler v1.7.3      ░▒▓█
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
# ║    Advanced sampler using a predictor-corrector Euler method with 
# ║    dynamic dampening across the sigma schedule. 
# ║    NOTE: This is a public wrapper. Missing binaries will safely degrade to standard Euler.
# ╚════════════════════════════════════════════════════════════════════════════


VERSION = "v1.7.3"  # UPS v1.5.8

import torch
import logging
import time
import io
import math
from contextlib import contextmanager
import sys
import os
import numpy as np

from comfy.samplers import KSAMPLER
import comfy.model_management

logger = logging.getLogger("MD_Nodes.Samplers.AMED")

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
    import amed_sampler_core_bin as core
    CORE_LOADED = True
    CORE_MODE = "Binary (Production)"
except ImportError as e1:
    try:
        import amed_sampler_core as core
        CORE_LOADED = True
        CORE_MODE = "Source (Development)"
    except ImportError as e2:
        CORE_ERROR = f"Binary: {e1}\nSource: {e2}"

# =================================================================================
# == Dependency Checks & Constants
# =================================================================================
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

CONST_PLOT_DPI = 100
CONST_PLOT_FIGSIZE = (10, 4)

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
# == Visualization Logic
# =================================================================================

def plot_dampening_curve(dampening_factor):
    """Generates a plot of the AMED Correction Weight vs Sigma."""
    if not MATPLOTLIB_AVAILABLE or not PIL_AVAILABLE:
        return torch.zeros((1, 64, 64, 3), dtype=torch.float32)

    try:
        sigmas = [100.0 * (0.95 ** i) for i in range(150)]
        weights = []
        
        # Graceful Degradation for Plotting
        if CORE_LOADED:
            for s in sigmas:
                w = core.calculate_correction_weight(s, dampening_factor)
                weights.append(w)
        else:
            for s in sigmas:
                base_weight = 0.5 * min(1.0, max(0.1, s / 2.0))
                w = min(0.5, max(0.0, base_weight * dampening_factor))
                weights.append(w)

        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=CONST_PLOT_FIGSIZE)
        
        ax.plot(sigmas, weights, color='#FF6B6B', linewidth=2.0, label='Correction Weight')
        ax.fill_between(sigmas, weights, alpha=0.2, color='#FF6B6B')
        ax.axhline(y=0.5, color='#555555', linestyle=':', label='Max Theoretical (0.5)')

        ax.set_xscale('log')
        ax.invert_xaxis() 
        ax.set_title(f"AMED Stabilization Profile (Factor: {dampening_factor})", fontsize=11, fontweight='bold')
        ax.set_xlabel("Sigma (Log Scale)", fontsize=9)
        ax.set_ylabel("Weight (0=Euler, 0.5=AMED)", fontsize=9)
        ax.grid(True, which="both", linestyle=':', alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)
        
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=CONST_PLOT_DPI, facecolor=fig.get_facecolor())
        buf.seek(0)
        plt.close(fig)

        img = Image.open(buf).convert("RGB")
        img_np = np.array(img).astype(np.float32) / 255.0
        return torch.from_numpy(img_np).unsqueeze(0)

    except Exception as e:
        logger.error(f"Plot generation failed: {e}")
        return torch.zeros((1, 64, 64, 3), dtype=torch.float32)

# =================================================================================
# == Sampler Implementation
# =================================================================================

class MD_AMED_Sampler_Runner:
    def __init__(self, model, sigmas, extra_args=None):
        self.model = model
        self.sigmas = sigmas
        self.extra_args = extra_args or {}
        
        self.debug_mode = self.extra_args.get("amed_debug_mode", 0)
        self.profiling = self.extra_args.get("amed_enable_profiling", False) or self.debug_mode >= 1

        if self.debug_mode >= 2: logger.setLevel(logging.DEBUG)
        elif self.debug_mode >= 1: logger.setLevel(logging.INFO)
        else: logger.setLevel(logging.WARNING)
        
        self.profiler = PerformanceProfiler(enabled=self.profiling)

    def sample(self, x):
        total_steps = len(self.sigmas) - 1
        self.profiler.start("total_sampling")
        
        # Graceful Degradation: Fallback to standard Euler if core is missing
        if not CORE_LOADED:
            if self.debug_mode >= 1:
                logger.warning(f"⚠️ AMED Core Missing: {CORE_ERROR}. Falling back to standard Euler step.")
            
            for i in range(total_steps):
                self.profiler.start("step")
                sigma_curr = self.sigmas[i]
                sigma_next = self.sigmas[i + 1]
                
                # Sanitize args for fallback
                model_args = self.extra_args.copy()
                model_args.pop("amed_dampening_factor", None)
                model_args.pop("amed_force_euler_last", None)
                model_args.pop("amed_debug_mode", None)
                model_args.pop("amed_enable_profiling", None)
                if "callback" in model_args: del model_args["callback"]
                
                sigma_curr_t = sigma_curr * torch.ones((x.shape[0],), device=x.device, dtype=x.dtype)
                denoised = self.model(x, sigma_curr_t, **model_args)
                d = (x - denoised) / sigma_curr
                dt = sigma_next - sigma_curr
                x = x + d * dt
                
                if "callback" in self.extra_args and self.extra_args["callback"]:
                    try:
                        self.extra_args["callback"]({'x': x, 'i': i, 'sigma': sigma_curr, 'sigma_hat': sigma_curr, 'denoised': x})
                    except Exception: pass
                self.profiler.stop("step")
                
            self.profiler.stop("total_sampling")
            return x

        # Optimized Core Path
        for i in range(total_steps):
            self.profiler.start("step")
            
            sigma_curr = self.sigmas[i]
            sigma_next = self.sigmas[i + 1]
            is_last_step = (i == total_steps - 1)
            
            x = core.amed_solver_step(
                self.model, x, sigma_curr, sigma_next, self.extra_args, 
                is_last_step=is_last_step, logger=logger
            )
            
            if "callback" in self.extra_args and self.extra_args["callback"]:
                try:
                    self.extra_args["callback"]({'x': x, 'i': i, 'sigma': sigma_curr, 'sigma_hat': sigma_curr, 'denoised': x})
                except Exception: pass
            
            self.profiler.stop("step")
            if self.debug_mode >= 2:
                logger.debug(f"Step {i+1} complete. Sigma: {sigma_curr.item():.4f}")

        self.profiler.stop("total_sampling")
        
        if self.debug_mode >= 1:
            logging.info("\n" + "=" * 60)
            logging.info("📊 [MD: AMED Sampler] ANALYTICS REPORT")
            logging.info("=" * 60)
            logging.info("📉  SAMPLING:")
            logging.info(f"    • Steps:        {total_steps}")
            logging.info(f"    • Dampening:    {self.extra_args.get('amed_dampening_factor', 1.0)}")
            self.profiler.print_report()
            logging.info("=" * 60)
            
        return x

def get_amed_sampler_function(model, x, sigmas, extra_args=None, callback=None, disable=None, **kwargs):
    if extra_args is None: extra_args = {}
    for k, v in kwargs.items(): extra_args[k] = v
    extra_args["callback"] = callback
    
    runner = MD_AMED_Sampler_Runner(model, sigmas, extra_args)
    return runner.sample(x)

# =================================================================================
# == Node Class
# =================================================================================

class MD_AMED_Sampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dampening_factor": ("FLOAT", {
                    "default": 1.0, 
                    "min": 0.0, 
                    "max": 2.0, 
                    "step": 0.05,
                    "tooltip": (
                        "DAMPENING FACTOR\n"
                        "• Purpose: Controls the strength of AMED correction.\n"
                        "• Range: 0.0 (Pure Euler) to 2.0 (Aggressive Predictor).\n"
                        "• Trade-offs: High values improve detail but risk mathematical instability (NaNs).\n"
                        "\n⭐ Recommended: 1.0 for standard generation. Reduce if output goes black."
                    )
                }),
                "force_euler_last_step": ("BOOLEAN", {
                    "default": True, 
                    "tooltip": (
                        "FORCE EULER LAST STEP\n"
                        "• Purpose: Switches back to simple Euler math for the very final step.\n"
                        "• Trade-offs: Prevents sharp artifacts from forming at near-zero sigmas.\n"
                        "\n⭐ Recommended: True. Highly required for audio/spectrograms."
                    )
                }),
                "debug_mode": (["0 - Silent", "1 - Info", "2 - Verbose"], {
                    "default": "0 - Silent",
                    "tooltip": (
                        "LOGGING VERBOSITY\n"
                        "• Purpose: Controls console output and analytics generation.\n"
                        "• Options: 0 (Silent), 1 (Stats/Performance), 2 (Full trace).\n"
                        "\n⭐ Recommended: 0 - Silent for production."
                    )
                }),
                "enable_profiling": ("BOOLEAN", {
                    "default": False, 
                    "tooltip": (
                        "ENABLE PROFILING\n"
                        "• Purpose: Measure timing of individual sampling steps.\n"
                        "• Note: Automatically enabled if debug_mode >= 1."
                    )
                }),
            }
        }

    RETURN_TYPES = ("SAMPLER", "IMAGE")
    RETURN_NAMES = ("sampler", "curve_plot")
    FUNCTION = "get_sampler"
    CATEGORY = "MD_Nodes/Samplers"

    def get_sampler(self, dampening_factor, force_euler_last_step, debug_mode, enable_profiling):
        if isinstance(debug_mode, str):
            debug_level = int(debug_mode.split(" ")[0])
        else:
            debug_level = debug_mode
            
        config = {
            "amed_dampening_factor": dampening_factor,
            "amed_force_euler_last": force_euler_last_step,
            "amed_debug_mode": debug_level,
            "amed_enable_profiling": enable_profiling
        }
        
        plot_img = plot_dampening_curve(dampening_factor)
        return (KSAMPLER(get_amed_sampler_function, extra_options=config), plot_img)

NODE_CLASS_MAPPINGS = {"MD_AMED_Sampler": MD_AMED_Sampler}
NODE_DISPLAY_NAME_MAPPINGS = {"MD_AMED_Sampler": "MD: AMED Solver (Corrected Euler)"}


# ==============================================================================
# == Unit Tests (smoke — runs without ComfyUI)
# ==============================================================================

if __name__ == "__main__":
    print("\n🧪 Smoke tests: MD_AMED_Sampler")
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

    _check("VERSION defined",    VERSION == "v1.7.3")
    _check("CONST CONST_PLOT_DPI defined", CONST_PLOT_DPI is not None)
    _check("NODE_CLASS_MAPPINGS defined",
           isinstance(NODE_CLASS_MAPPINGS, dict) and len(NODE_CLASS_MAPPINGS) > 0)
    _check("  class MD_AMED_Sampler in map", "MD_AMED_Sampler" in NODE_CLASS_MAPPINGS)

    print(f"\n  {_pass} passed, {_fail} failed")
    if _fail == 0:
        print("  🎉 All good.")
