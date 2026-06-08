# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# █▓▒░                                                                     ░▒▓█
# █▓▒░       MD_Nodes/MD_GITS_Scheduler – Geometric Schedule v1.3.0        ░▒▓█
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
# ║    GITS Scheduler Node for ComfyUI. Uses geometric curvature to 
# ║    optimize step distribution in the critical mid-noise regions.
# ║    NOTE: This is a public wrapper. Missing binaries will safely degrade 
# ║    to standard log-linear scheduling.
# ╚════════════════════════════════════════════════════════════════════════════


VERSION = "v1.3.0"  # UPS v1.5.8

import logging
import json
import time
import io
import math
import torch
import numpy as np
import comfy.model_management

# =================================================================================
# == MD_Nodes Universal Binary Loader (v1.6.1)
# =================================================================================

import sys
import os

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
    import gits_scheduler_core_bin as core
    CORE_LOADED = True
    CORE_MODE = "Binary (Production)"
except ImportError as e1:
    try:
        import gits_scheduler_core as core
        CORE_LOADED = True
        CORE_MODE = "Source (Development)"
    except ImportError as e2:
        CORE_ERROR = f"Binary: {e1}\nSource: {e2}"

# =================================================================================
# == Configuration Constants
# =================================================================================

CONST_EPSILON = 1e-6
CONST_FALLBACK_SIGMA_MIN = 0.002
CONST_FALLBACK_SIGMA_MAX = 80.0
CONST_PLOT_DPI = 120
CONST_PLOT_FIGSIZE = (10, 4)

# Density Analysis Thresholds
CONST_HIGH_NOISE_THRESHOLD = 0.66
CONST_LOW_NOISE_THRESHOLD = 0.33

CONST_LOGGER_NAME = "MD_Nodes.Schedulers.GITS"
logger = logging.getLogger(CONST_LOGGER_NAME)

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
    logger.warning("[MD_GITS] Matplotlib not available. Plotting disabled.")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("[MD_GITS] PIL not available. Plotting disabled.")

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
        logging.info("\n⏱️  PERFORMANCE (Generation):")
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

class MD_GITS_Scheduler:
    """
    GITS Scheduler Node for ComfyUI.
    Uses geometric curvature to optimize step distribution.
    """
    
    CURVATURE_PRESETS = {
        "Linear (0.0)": 0.0,
        "Light (0.5)": 0.5,
        "Standard (1.0)": 1.0,
        "Heavy (2.0)": 2.0,
        "Extreme (3.5)": 3.5,
        "Custom": -1.0
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {
                    "tooltip": (
                        "MODEL INPUT\n"
                        "• Purpose: Auto-detects native sigma range (max/min noise limits).\n"
                        "• Requirement: Absolute necessity for mathematically accurate scheduling.\n"
                        "• Output: Generates sigmas tuned specifically to this model."
                    )
                }),
                "steps": ("INT", {
                    "default": 20, "min": 1, "max": 10000, 
                    "tooltip": (
                        "TOTAL STEPS\n"
                        "• Purpose: Number of denoising iterations to divide the schedule into.\n"
                        "• Range: 4 (Turbo/Lightning) to 50+ (High Quality).\n"
                        "• Trade-offs: Lower steps require 'Heavy' curvature to cluster math properly.\n"
                        "\n⭐ Recommended: 8-12 with 'Heavy' preset, or 20-30 with 'Standard' preset."
                    )
                }),
                "curvature_preset": (list(cls.CURVATURE_PRESETS.keys()), {
                    "default": "Standard (1.0)",
                    "tooltip": (
                        "CURVATURE PRESET\n"
                        "• Purpose: Controls the 'Boomerang' (tanh) intensity of step clustering.\n"
                        "• Standard (1.0): Balanced for standard 20-30 steps.\n"
                        "• Heavy (2.0): Optimized specifically for 8-12 step workflows.\n"
                        "• Extreme (3.5): Use only for ultra-low step counts.\n"
                        "\n⭐ Recommended: Use 'Heavy' for Lightning/Turbo models."
                    )
                }),
                "denoise": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": (
                        "DENOISE STRENGTH\n"
                        "• Purpose: Truncates the schedule for Img2Img or refinement workflows.\n"
                        "• Range: 0.0 (None) to 1.0 (Full Generation).\n"
                        "• Trade-offs: Lowering this slices from the start, preserving low-noise steps.\n"
                        "\n⭐ Recommended: 1.0 for Txt2Img, 0.4 - 0.7 for Img2Img."
                    )
                }),
            },
            "optional": {
                "curvature_scale": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1,
                    "tooltip": (
                        "CUSTOM CURVATURE\n"
                        "• Purpose: Manual control of clustering intensity.\n"
                        "• Requirement: Active ONLY when Preset is set to 'Custom'.\n"
                        "• Effect: Higher values tightly cluster steps in the middle noise regions.\n"
                        "\n⭐ Recommended: 1.0"
                    )
                }),
                "sigma_override_max": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1000.0, "step": 0.1,
                    "tooltip": (
                        "SIGMA MAX OVERRIDE\n"
                        "• Purpose: Manually force the starting noise level.\n"
                        "• Trade-offs: Overrides native model detection. Can cause burn-in.\n"
                        "\n⭐ Recommended: 0.0 (Auto-detect)."
                    )
                }),
                "sigma_override_min": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1000.0, "step": 0.001,
                    "tooltip": (
                        "SIGMA MIN OVERRIDE\n"
                        "• Purpose: Manually force the ending noise level.\n"
                        "• Trade-offs: Setting too low risks numerical instability.\n"
                        "\n⭐ Recommended: 0.0 (Auto-detect)."
                    )
                }),
                "debug_mode": (["0 - Silent", "1 - Info", "2 - Verbose"], {
                    "default": "0 - Silent",
                    "tooltip": (
                        "LOGGING VERBOSITY\n"
                        "• Purpose: Controls console output and structural profiling.\n"
                        "• Options: 0 (Production), 1 (Analytics Report), 2 (Full trace).\n"
                        "\n⭐ Recommended: 1 - Info to visualize step density metrics."
                    )
                }),
            }
        }

    RETURN_TYPES = ("SIGMAS", "IMAGE", "STRING")
    RETURN_NAMES = ("sigmas", "plot", "schedule_info")
    FUNCTION = "get_sigmas"
    CATEGORY = "MD_Nodes/Schedulers"

    def _plot_schedule(self, sigmas, title="GITS Schedule"):
        """Wrapper-side visualization logic."""
        if not MATPLOTLIB_AVAILABLE or not PIL_AVAILABLE:
            return torch.zeros((1, 64, 64, 3), dtype=torch.float32)

        try:
            sigma_list = sigmas.cpu().tolist()
            steps = list(range(len(sigma_list)))

            plt.style.use('dark_background')
            fig, ax = plt.subplots(figsize=CONST_PLOT_FIGSIZE)
            
            ax.plot(steps, sigma_list, color='#87CEEB', linewidth=2.0, label='Sigma', zorder=3)
            ax.fill_between(steps, sigma_list, alpha=0.2, color='#87CEEB')
            
            ax.scatter(steps, sigma_list, color='#FFD700', s=15, alpha=0.6, zorder=4, label='Steps')

            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.set_xlabel("Step Index", fontsize=9)
            ax.set_ylabel("Sigma (Noise Level)", fontsize=9)
            ax.grid(True, linestyle=':', alpha=0.3)
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

    def get_sigmas(self, model, steps, curvature_preset, denoise, curvature_scale=1.0, sigma_override_max=0.0, sigma_override_min=0.0, debug_mode="0 - Silent"):
        
        # 1. Setup Logging & Profiling
        try:
            debug_level = int(debug_mode.split(" ")[0])
        except (ValueError, AttributeError):
            debug_level = 0

        if debug_level >= 2: logger.setLevel(logging.DEBUG)
        elif debug_level >= 1: logger.setLevel(logging.INFO)
        else: logger.setLevel(logging.WARNING)

        profiler = PerformanceProfiler(enabled=(debug_level >= 1))
        profiler.start("total_execution")

        # 2. Resolve Curvature
        if curvature_preset == "Custom":
            final_curvature = curvature_scale
        else:
            final_curvature = self.CURVATURE_PRESETS.get(curvature_preset, 1.0)
            
        if debug_level >= 2:
            logger.debug(f"Config: Steps={steps}, Curvature={final_curvature}, Denoise={denoise}")

        # 3. Get Model Limits
        profiler.start("model_detection")
        try:
            ms = model.get_model_object("model_sampling")
            s_min = float(ms.sigma_min)
            s_max = float(ms.sigma_max)
            if debug_level >= 2: logger.debug(f"Detected Range: {s_max:.4f} -> {s_min:.4f}")
        except Exception:
            s_min = CONST_FALLBACK_SIGMA_MIN
            s_max = CONST_FALLBACK_SIGMA_MAX
            logger.warning("[GITS] Could not detect model sigmas. Using fallbacks.")
        profiler.stop("model_detection")

        # 4. Apply Overrides
        if sigma_override_max > 0: s_max = sigma_override_max
        if sigma_override_min > 0: s_min = sigma_override_min
        if s_min >= s_max: s_min = s_max * 0.01

        # 5. Generate Full Schedule (Graceful Degradation Core Hook)
        profiler.start("generation_math")
        device = comfy.model_management.get_torch_device()
        
        if CORE_LOADED:
            sigmas = core.get_gits_sigmas(steps, s_min, s_max, final_curvature, device)
        else:
            if debug_level >= 1:
                logger.warning(f"⚠️ GITS Core Missing: {CORE_ERROR}. Falling back to standard Log-Linear schedule.")
            # Basic fallback logic preventing crashes
            t = torch.linspace(0, 1, steps + 1, dtype=torch.float32, device=device)
            log_min = math.log(s_min)
            log_max = math.log(s_max)
            sigmas = torch.exp(log_max + t * (log_min - log_max))
            sigmas[0] = s_max
            sigmas[-1] = s_min
            
        profiler.stop("generation_math")
        
        # 6. Handle Denoise (Img2Img slicing)
        if denoise < 1.0:
            total_len = len(sigmas) - 1
            needed_steps = int(total_len * denoise)
            start_idx = total_len - needed_steps
            sigmas = sigmas[start_idx:]
            if debug_level >= 2: logger.debug(f"Denoise Slice: {needed_steps} steps")
            
        # 7. Append precise zero
        if sigmas[-1] > CONST_EPSILON:
            sigmas = torch.cat([sigmas, torch.zeros(1, device=device)])

        # 8. Generate Plot
        profiler.start("visualization")
        plot_image = torch.zeros((1, 64, 64, 3))
        if MATPLOTLIB_AVAILABLE:
            preset_name = curvature_preset.split(' ')[0] if ' ' in curvature_preset else curvature_preset
            plot_title = f"GITS | Steps: {len(sigmas)-1} | Curve: {final_curvature:.1f} ({preset_name})"
            plot_image = self._plot_schedule(sigmas, title=plot_title)
        profiler.stop("visualization")

        # 9. Analysis & Info
        sigma_list = sigmas.cpu().tolist()
        density_analysis = {}
        if len(sigma_list) > 1:
            high_thresh = s_min + (s_max - s_min) * CONST_HIGH_NOISE_THRESHOLD
            low_thresh = s_min + (s_max - s_min) * CONST_LOW_NOISE_THRESHOLD
            
            h_steps = sum(1 for s in sigma_list if s >= high_thresh)
            l_steps = sum(1 for s in sigma_list if s <= low_thresh)
            total_steps = len(sigma_list) - 1
            m_steps = max(0, total_steps - h_steps - l_steps)
            
            density_analysis = {
                "High Noise (>66%)": h_steps,
                "Mid Noise": m_steps,
                "Low Noise (<33%)": l_steps
            }

        profiler.stop("total_execution")

        info = {
            "version": "1.3.0",
            "scheduler": "GITS",
            "curvature": final_curvature,
            "steps": len(sigmas) - 1,
            "range": [f"{sigmas[0]:.4f}", f"{sigmas[-1]:.4f}"],
            "analysis": density_analysis,
            "execution_time": f"{profiler.get_total_time()*1000:.2f}ms"
        }
        
        # 10. Analytics Report (Standardized)
        if debug_level >= 1:
            curve_desc = {
                0.0: "Linear distribution (no clustering)",
                0.5: "Light clustering (gentle boomerang)",
                1.0: "Standard clustering (balanced)",
                2.0: "Heavy clustering (low-step optimized)",
                3.5: "Extreme clustering (ultra-low steps)"
            }.get(final_curvature, "Custom curve")

            logging.info("\n" + "=" * 60)
            logging.info("📊 [MD_GITS_Scheduler] ANALYTICS REPORT")
            logging.info("=" * 60)
            logging.info("⚙️  CONFIGURATION:")
            logging.info(f"    • Steps:      {steps}")
            logging.info(f"    • Curvature:  {final_curvature:.2f} ({curvature_preset}) - {curve_desc}")
            logging.info(f"    • Denoise:    {denoise:.2f}")
            logging.info(f"📈  SCHEDULE:")
            logging.info(f"    • Final Steps: {len(sigmas) - 1}")
            logging.info(f"    • Sigma Range: {sigmas[0]:.4f} → {sigmas[-2]:.4f}")
            
            if density_analysis:
                logging.info(f"📊  STEP DISTRIBUTION:")
                for region, count in density_analysis.items():
                    logging.info(f"    • {region}: {count}")
                
                uniform_mid = (len(sigmas) - 1) // 3
                actual_mid = density_analysis.get("Mid Noise", 0)
                if actual_mid > uniform_mid and uniform_mid > 0:
                    boost = ((actual_mid / uniform_mid) - 1) * 100
                    logging.info(f"💡  EFFICIENCY:")
                    logging.info(f"    • Mid-range boost: +{boost:.0f}% vs uniform")

            profiler.print_report()
            logging.info("=" * 60)

        return (sigmas, plot_image, json.dumps(info, indent=2))

# =================================================================================
# == Node Registration
# =================================================================================

NODE_CLASS_MAPPINGS = {
    "MD_GITS_Scheduler": MD_GITS_Scheduler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MD_GITS_Scheduler": "MD: GITS Scheduler (Boomerang)",
}


# ==============================================================================
# == Unit Tests (smoke — runs without ComfyUI)
# ==============================================================================

if __name__ == "__main__":
    print("\n🧪 Smoke tests: MD_GITS_Scheduler")
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

    _check("VERSION defined",    VERSION == "v1.3.0")
    _check("CONST CONST_FALLBACK_SIGMA_MIN defined", CONST_FALLBACK_SIGMA_MIN is not None)
    _check("CONST CONST_FALLBACK_SIGMA_MAX defined", CONST_FALLBACK_SIGMA_MAX is not None)
    _check("CONST CONST_PLOT_DPI defined", CONST_PLOT_DPI is not None)
    _check("NODE_CLASS_MAPPINGS defined",
           isinstance(NODE_CLASS_MAPPINGS, dict) and len(NODE_CLASS_MAPPINGS) > 0)
    _check("  class MD_GITS_Scheduler in map", "MD_GITS_Scheduler" in NODE_CLASS_MAPPINGS)

    print(f"\n  {_pass} passed, {_fail} failed")
    if _fail == 0:
        print("  🎉 All good.")
