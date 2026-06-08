# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# █▓▒░                                                                     ░▒▓█
# █▓▒░          MD_Nodes Wrapper: Hybrid Scheduler Suite (v1.6.0)          ░▒▓█
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
# ║   Advanced mathematical scheduler capable of split-curves, 
# ║   adaptive slicing, and custom formulas (Bong, AYS, etc).
# ║   NOTE: This is a public wrapper. Missing binaries will safely degrade 
# ║   to standard linear scheduling.
# ╚════════════════════════════════════════════════════════════════════════════


VERSION = "v1.6.0"  # UPS v1.5.8

import json
import io
import logging
import traceback
import time
import os
import sys
import torch
import numpy as np
import comfy.model_management

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
    import hybrid_scheduler_core_bin as core
    CORE_LOADED = True
    CORE_MODE = "Binary (Production)"
except ImportError as e1:
    try:
        import hybrid_scheduler_core as core
        CORE_LOADED = True
        CORE_MODE = "Source (Development)"
    except ImportError as e2:
        CORE_ERROR = f"Binary: {e1}\nSource: {e2}"

# =================================================================================
# == Dependencies & Utilities
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

logger = logging.getLogger("MD_Nodes.Schedulers.Hybrid")

CONST_PLOT_DPI = 120
CONST_COLOR_PRIMARY = '#00FFFF'
CONST_FALLBACK_SIGMA_MIN = 0.006
CONST_FALLBACK_SIGMA_MAX = 1.000

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
        logging.info("\n⏱️  PERFORMANCE:")
        total = self.get_total_time()
        logging.info(f"    • Total Time: {total:.4f}s")
        for op_name, times in sorted(self.timings.items()):
            avg = sum(times) / len(times)
            if len(times) == 1:
                logging.info(f"    • {op_name}: {avg:.4f}s")
            else:
                logging.info(f"    • {op_name}: {avg:.4f}s avg ({len(times)}x)")

# =================================================================================
# == Base Class (UI & Shared Logic)
# =================================================================================

class HybridSchedulerWrapperBase:
    SCHEDULER_MODES = [
        "karras_rho", "simple", "linear_quadratic", "bong_tangent", "exponential", 
        "poly", "beta", "ays", "sgm_uniform", "ddim_uniform", "adaptive_linear", 
        "variance_preserving", "blended_curves", "kl_optimal"
    ]
    
    PRESETS = {
        "Custom": None,
        "ComfyUI Default": {"mode": "karras_rho", "rho": 7.0, "denoise_mode": "Subtractive (Slice)"},
        "High Detail (Recommended)": {"mode": "linear_quadratic", "split_schedule": True, "mode_b": "polynomial", "split_at_step": 30, "power": 1.5, "threshold_noise": 0.001, "linear_steps": 27, "min_steps_mode": "adaptive", "adaptive_min_percentage": 2.0},
        "Fast Draft": {"mode": "karras_rho", "rho": 2.5, "min_steps_mode": "fixed", "min_sliced_steps": 2},
        "Tiling Safe": {"mode": "blended_curves", "blend_factor": 0.3, "denoise_mode": "Hybrid (Adaptive Steps)", "low_denoise_color_fix": 0.2, "adaptive_min_percentage": 3.0},
        "Smooth Gradient": {"mode": "polynomial", "power": 1.8, "detail_preservation": 0.1},
        "Aggressive Start": {"mode": "exponential", "split_schedule": True, "mode_b": "adaptive_linear", "split_at_step": 40},
        "Ultra Quality": {"mode": "kl_optimal", "split_schedule": True, "mode_b": "variance_preserving", "split_at_step": 60, "adaptive_min_percentage": 5.0},
        "Beta Distribution": {"mode": "beta", "beta_alpha": 0.6, "beta_beta": 0.6},
        "AYS Optimal": {"mode": "ays", "adaptive_min_percentage": 2.5},
        "Composition Focus (Bong)": {"mode": "bong_tangent", "bong_preset": "composition_focus", "adaptive_min_percentage": 3.0},
        "Detail Focus (Bong)": {"mode": "bong_tangent", "bong_preset": "detail_focus", "adaptive_min_percentage": 3.0},
        "Balanced (Bong)": {"mode": "bong_tangent", "bong_preset": "balanced", "adaptive_min_percentage": 3.0},
    }

    @classmethod
    def _validate_presets(cls, input_data):
        valid_keys = set(input_data["required"].keys()) | set(input_data["optional"].keys())
        for preset_name, settings in cls.PRESETS.items():
            if not settings: continue
            for key in settings:
                if key not in valid_keys:
                    logger.warning(f"⚠️ PRESET VALIDATION WARNING: Preset '{preset_name}' contains unknown key: '{key}'")

    def _apply_preset(self, preset_name, **kwargs):
        if preset_name == "Custom" or preset_name not in self.PRESETS:
            return kwargs
        preset = self.PRESETS[preset_name].copy()
        for key, value in preset.items():
            if key not in kwargs or kwargs.get('preset') == preset_name:
                kwargs[key] = value
        return kwargs

    def _apply_bong_preset(self, preset):
        presets = {
            "composition_focus": (0.7, 1.5, 0.5),
            "balanced": (0.5, 1.2, 0.8),
            "detail_focus": (0.3, 0.8, 1.2),
        }
        if preset in presets: return presets[preset]
        return 0.5, 1.2, 0.8

    def _plot(self, sigmas, title_info):
        if not MATPLOTLIB_AVAILABLE or not PIL_AVAILABLE: return torch.zeros((1, 64, 64, 3))
        try:
            plt.style.use('dark_background')
            fig, ax = plt.subplots(figsize=(10, 5))
            y = sigmas.cpu().numpy(); x = np.arange(len(y))
            ax.plot(x, y, color=CONST_COLOR_PRIMARY, linewidth=2, label='Sigma Schedule')
            ax.fill_between(x, 0, y, color=CONST_COLOR_PRIMARY, alpha=0.1)
            
            if len(y) > 1:
                deltas = np.abs(np.diff(y))
                norm_deltas = deltas / (np.max(deltas) + 1e-6) * np.max(y) * 0.5
                ax.plot(x[:-1], norm_deltas, color='#FF00FF', linewidth=1, linestyle=':', label='Step Size (Delta)')
            
            stats = (f"Steps: {len(y)-1}\nMax: {y[0]:.2f}\nMin: {y[-1]:.4f}\n{title_info}")
            props = dict(boxstyle='round', facecolor='#222222', alpha=0.8)
            ax.text(0.02, 0.95, stats, transform=ax.transAxes, fontsize=9, verticalalignment='top', bbox=props, color='white')

            ax.set_title("Hybrid Schedule Analysis", fontweight='bold')
            ax.set_xlabel("Steps"); ax.set_ylabel("Sigma Value")
            ax.grid(True, alpha=0.2); ax.legend(loc='center right')
            
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=CONST_PLOT_DPI, facecolor='#111111')
            buf.seek(0); plt.close(fig)
            img = Image.open(buf).convert("RGB")
            return torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0)
        except Exception as e:
            logger.error(f"Plot generation failed: {e}")
            return torch.zeros((1, 64, 64, 3))

# =================================================================================
# == Node 1: Advanced Scheduler
# =================================================================================

class HybridAdaptiveSigmas_Advanced(HybridSchedulerWrapperBase):
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "model": ("MODEL", {
                    "tooltip": (
                        "DIFFUSION MODEL\n"
                        "• Purpose: Auto-detects sigma range (max/min noise levels).\n"
                        "• Output: Generates sigmas tuned specifically to this model."
                    )
                }),
                "steps": ("INT", {
                    "default": 30, "min": 1, 
                    "tooltip": (
                        "TOTAL STEPS\n"
                        "• Purpose: Number of denoising iterations to divide the schedule into.\n"
                        "\n⭐ Recommended: 20-30 for images, 15-25 for audio."
                    )
                }),
                "mode": (cls.SCHEDULER_MODES, {
                    "default": "karras_rho", 
                    "tooltip": (
                        "SCHEDULER MODE\n"
                        "• Purpose: The mathematical curve used for noise reduction.\n"
                        "• Options: Karras (Standard), Bong (Audio), AYS (Fast), etc.\n"
                        "\n⭐ Recommended: Karras for images, Bong for audio."
                    )
                }),
                "denoise": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, 
                    "tooltip": (
                        "DENOISE STRENGTH\n"
                        "• Purpose: How much of the schedule to execute.\n"
                        "• Use: <1.0 for Img2Img or Refinement."
                    )
                }),
                "denoise_mode": (["Hybrid (Adaptive Steps)", "Subtractive (Slice)", "Repaced (Full Steps)"], {
                    "tooltip": (
                        "DENOISE MODE\n"
                        "• Hybrid: Recalculates curve for partial steps (Best Quality).\n"
                        "• Subtractive: Simply slices the end of the curve.\n"
                        "• Repaced: Compresses full schedule into fewer steps."
                    )
                }),
                
                # Advanced
                "split_schedule": ("BOOLEAN", {
                    "default": False, 
                    "tooltip": (
                        "SPLIT SCHEDULE\n"
                        "• Purpose: Use two different algorithms in one generation.\n"
                        "\n⭐ Useful for advanced research and custom effects."
                    )
                }),
                "mode_b": (cls.SCHEDULER_MODES, {
                    "tooltip": "SECONDARY MODE\n• The algorithm to use after the split point."
                }),
                "split_at_step": ("INT", {
                    "default": 30, 
                    "tooltip": "SPLIT STEP\n• The exact step index where the switch occurs."
                }),
                "use_percentage_split": ("BOOLEAN", {
                    "default": False, 
                    "tooltip": "PERCENTAGE SPLIT\n• Use a percentage instead of a fixed step count."
                }),
                "split_percentage": ("FLOAT", {
                    "default": 0.25, "step": 0.01, 
                    "tooltip": "SPLIT PERCENTAGE\n• Point (0.0-1.0) to switch modes."
                }),
                
                # Parameters
                "rho": ("FLOAT", {
                    "default": 7.0, 
                    "tooltip": (
                        "RHO (STEEPNESS)\n"
                        "• Purpose: Controls Karras/Poly curve shape.\n"
                        "• Higher = More steps at lower noise levels.\n"
                        "\n⭐ Recommended: 7.0 (Standard), 3.0 (Flat)."
                    )
                }),
                "bong_pivot": ("FLOAT", {"default": 0.5, "step": 0.01, "tooltip": "BONG PIVOT\n• Division between Composition and Detail phases."}),
                "bong_preset": (["custom", "composition_focus", "balanced", "detail_focus"], {"default": "balanced", "tooltip": "BONG PRESET\n• Quick setup for Bong Tangent mode."}),
                "bong_slope_composition": ("FLOAT", {"default": 1.2, "step": 0.01, "tooltip": "BONG SLOPE (EARLY)\n• Steepness of early phase."}),
                "bong_slope_detail": ("FLOAT", {"default": 0.8, "step": 0.01, "tooltip": "BONG SLOPE (LATE)\n• Steepness of late phase."}),
                "linear_steps": ("INT", {"default": 15, "tooltip": "LINEAR STEPS\n• For Linear-Quadratic mode."}),
                "linear_steps_relative": ("BOOLEAN", {"default": False, "tooltip": "LINEAR RELATIVE\n• Use % instead of fixed count."}),
                "power": ("FLOAT", {"default": 2.0, "tooltip": "POLYNOMIAL POWER\n• Exponent for Poly mode."}),
                "beta_alpha": ("FLOAT", {"default": 0.6, "step": 0.01, "tooltip": "BETA ALPHA\n• Shape param for Beta mode."}),
                "beta_beta": ("FLOAT", {"default": 0.6, "step": 0.01, "tooltip": "BETA BETA\n• Shape param for Beta mode."}),
                "threshold_noise": ("FLOAT", {"default": 0.0025, "step": 0.0001, "tooltip": "NOISE THRESHOLD\n• Gating value for Linear-Quadratic."}),
                "blend_factor": ("FLOAT", {"default": 0.5, "step": 0.01, "tooltip": "BLEND FACTOR\n• Mix ratio for Blended Curves."}),
                
                # Min Steps
                "min_steps_mode": (["fixed", "adaptive"], {"default": "fixed", "tooltip": "MIN STEPS MODE\n• Prevent step count dropping too low."}),
                "min_sliced_steps": ("INT", {"default": 3, "tooltip": "MIN FIXED STEPS\n• Hard minimum step count."}),
                "adaptive_min_percentage": ("FLOAT", {"default": 2.0, "step": 0.1, "tooltip": "MIN PERCENTAGE\n• Minimum steps as % of total."}),
                
                # Fixes
                "detail_preservation": ("FLOAT", {"default": 0.0, "step": 0.01, "tooltip": "DETAIL PRESERVATION\n• Boosts final sigma slightly to keep texture."}),
                "low_denoise_color_fix": ("FLOAT", {"default": 0.0, "step": 0.01, "tooltip": "COLOR FIX\n• Corrects color shift at very low denoise."}),
                
                # Overrides
                "use_sigma_override": ("BOOLEAN", {"default": False, "tooltip": "ENABLE OVERRIDE\n• Ignore model sigmas and use manuals."}),
                "start_sigma_override": ("FLOAT", {"default": 1.0, "tooltip": "MANUAL START SIGMA"}),
                "end_sigma_override": ("FLOAT", {"default": 0.006, "step": 0.001, "tooltip": "MANUAL END SIGMA"}),
                "reverse_sigmas": ("BOOLEAN", {"default": False, "tooltip": "REVERSE\n• Flip schedule (Experimental)."}),
                
                "memory_efficient": ("BOOLEAN", {"default": False, "tooltip": "MEMORY EFFICIENT\n• Chunk calculation for high step counts (500+)."}),
                
                # Logging
                "debug_mode": (["0 - Silent", "1 - Info", "2 - Verbose"], {"default": "0 - Silent", "tooltip": "LOGGING VERBOSITY"}),
                "enable_profiling": ("BOOLEAN", {"default": False, "tooltip": "ENABLE PROFILING"}),
            },
            "optional": {
                "preset": (list(cls.PRESETS.keys()), {"default": "Custom", "tooltip": "PRESET\n• Load pre-configured settings."}),
            }
        }
        
        if not hasattr(cls, '_presets_validated'):
            cls._validate_presets(inputs)
            cls._presets_validated = True
            
        return inputs

    RETURN_TYPES = ("SIGMAS", "IMAGE", "STRING")
    RETURN_NAMES = ("sigmas", "plot", "info")
    FUNCTION = "execute"
    CATEGORY = "MD_Nodes/Schedulers"

    def execute(self, model, steps, mode, denoise, denoise_mode, split_schedule, mode_b, split_at_step, use_percentage_split, split_percentage, 
                rho, bong_pivot, bong_preset, bong_slope_composition, bong_slope_detail, linear_steps, linear_steps_relative, 
                power, beta_alpha, beta_beta, threshold_noise, blend_factor, min_steps_mode, min_sliced_steps, adaptive_min_percentage,
                detail_preservation, low_denoise_color_fix, use_sigma_override, start_sigma_override, end_sigma_override, 
                reverse_sigmas, memory_efficient, debug_mode, enable_profiling, preset="Custom"):
        
        d_level = int(debug_mode.split(" ")[0])
        profiler = PerformanceProfiler(enabled=enable_profiling or d_level >= 1)
        profiler.start("total_execution")
        
        device = comfy.model_management.get_torch_device()
        
        # 1. Preset & Validation
        steps = max(1, steps)
        denoise = max(0.0, min(1.0, denoise))
        
        kwargs = {k: v for k, v in locals().items() if k not in ['self', 'model', 'profiler', 'device']}
        kwargs = self._apply_preset(preset, **kwargs)
        
        mode = kwargs['mode']; bong_preset = kwargs['bong_preset']

        if mode == "bong_tangent" and bong_preset != "custom":
             p, sc, sd = self._apply_bong_preset(bong_preset)
             kwargs['bong_pivot'] = p; kwargs['bong_slope_composition'] = sc; kwargs['bong_slope_detail'] = sd

        if kwargs['use_percentage_split']:
            kwargs['split_at_step'] = max(1, min(kwargs['steps'] - 1, int(kwargs['steps'] * kwargs['split_percentage'])))
        
        if kwargs['split_schedule'] and not (0 < kwargs['split_at_step'] < kwargs['steps']):
             kwargs['split_schedule'] = False

        if kwargs['linear_steps_relative']:
            anchor = kwargs['split_at_step'] if kwargs['split_schedule'] else kwargs['steps']
            kwargs['linear_steps'] = max(1, anchor - kwargs['linear_steps'])

        try:
            ms = model.get_model_object("model_sampling")
            s_min, s_max = float(ms.sigma_min), float(ms.sigma_max)
        except Exception:
            s_min, s_max = CONST_FALLBACK_SIGMA_MIN, CONST_FALLBACK_SIGMA_MAX
            
        if kwargs['use_sigma_override']: s_max = kwargs['start_sigma_override']; s_min = kwargs['end_sigma_override']
        if s_min >= s_max: s_max = s_min + 0.1

        if kwargs['min_steps_mode'] == "adaptive":
            kwargs['min_sliced_steps'] = max(1, int(kwargs['steps'] * kwargs['adaptive_min_percentage'] / 100.0))

        # 2. Core Calculation with Graceful Degradation
        profiler.start("core_math")
        gen_params = {
            "rho": kwargs['rho'], "blend_factor": kwargs['blend_factor'], "power": kwargs['power'],
            "threshold_noise": kwargs['threshold_noise'], "linear_steps": kwargs['linear_steps'],
            "beta_alpha": kwargs['beta_alpha'], "beta_beta": kwargs['beta_beta'],
            "bong_pivot": kwargs['bong_pivot'], "bong_slope_comp": kwargs['bong_slope_composition'],
            "bong_slope_detail": kwargs['bong_slope_detail'], "memory_efficient": kwargs['memory_efficient']
        }

        if CORE_LOADED:
            if kwargs['split_schedule']:
                full_temp = core.calculate_sigmas(kwargs['steps'], mode, s_min, s_max, device, **gen_params)
                split_sigma = full_temp[kwargs['split_at_step']].item()
                sig_a = core.calculate_sigmas(kwargs['split_at_step'], mode, split_sigma, s_max, device, **gen_params)
                sig_b = core.calculate_sigmas(kwargs['steps'] - kwargs['split_at_step'], kwargs['mode_b'], s_min, split_sigma, device, **gen_params)
                full_sigmas = torch.cat((sig_a[:-1], sig_b))
            else:
                full_sigmas = core.calculate_sigmas(kwargs['steps'], mode, s_min, s_max, device, **gen_params)
        else:
            if d_level >= 1: logger.warning(f"⚠️ Hybrid Core Missing: {CORE_ERROR}. Falling back to standard linear.")
            full_sigmas = torch.linspace(s_max, s_min, kwargs['steps'] + 1, device=device)
            
        profiler.stop("core_math")

        # 3. Denoise Logic
        denoise_start = int(kwargs['steps'] * (1.0 - kwargs['denoise'])) if kwargs['denoise'] < 1.0 else 0
        start_idx = max(0, min(max(0, denoise_start), len(full_sigmas) - 1))
        
        if kwargs['denoise_mode'] == "Repaced (Full Steps)" and CORE_LOADED:
            eff_max = full_sigmas[start_idx].item()
            final_sigmas = core.calculate_sigmas(kwargs['steps'], mode, s_min, eff_max, device, **gen_params)
        elif kwargs['denoise_mode'] == "Hybrid (Adaptive Steps)" and CORE_LOADED:
            final_sigmas = full_sigmas[start_idx:]
            if (len(final_sigmas) - 1) < kwargs['min_sliced_steps']:
                final_sigmas = core.calculate_sigmas(kwargs['min_sliced_steps'], mode, final_sigmas[-1].item(), final_sigmas[0].item(), device, **gen_params)
        else:
            final_sigmas = full_sigmas[start_idx:]

        # 4. Post-Process
        final_sigmas = final_sigmas.clone()
        if kwargs['detail_preservation'] > 0:
             final_sigmas[-1] += (min(s_max/50, 0.5) - final_sigmas[-1]) * kwargs['detail_preservation']
        if kwargs['low_denoise_color_fix'] > 0 and len(final_sigmas) > 1:
             target = final_sigmas[-2] * 0.25
             final_sigmas[-1] += (target - final_sigmas[-1]) * kwargs['low_denoise_color_fix']
        if kwargs['reverse_sigmas']: final_sigmas = torch.flip(final_sigmas, [0])

        # 5. Output & Logging
        plot = self._plot(final_sigmas, f"Mode: {mode}\nSplit: {kwargs['split_schedule']}")
        info = json.dumps({"steps": len(final_sigmas)-1, "mode": mode, "params": gen_params}, indent=2)
        
        profiler.stop("total_execution")
        if d_level >= 1:
            logging.debug("\n" + "="*60)
            logging.info("📊 [HybridScheduler] ANALYTICS REPORT")
            logging.info(f"    • Mode: {mode}")
            logging.info(f"    • Steps: {len(final_sigmas)-1}")
            profiler.print_report()
            logging.debug("="*60)
        
        return (final_sigmas, plot, info)

class HybridAdaptiveSigmas_Basic(HybridSchedulerWrapperBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "DIFFUSION MODEL"}),
                "steps": ("INT", {"default": 30, "min": 1, "tooltip": "TOTAL STEPS\n• Recommended: 20-30"}),
                "mode": (cls.SCHEDULER_MODES, {"tooltip": "SCHEDULER MODE\n• Math curve type."}),
                "denoise": ("FLOAT", {"default": 1.0, "tooltip": "DENOISE STRENGTH\n• 0.0-1.0"}),
                "rho": ("FLOAT", {"default": 7.0, "tooltip": "RHO (STEEPNESS)\n• Controls curve shape."}),
            }
        }
    RETURN_TYPES = ("SIGMAS", "IMAGE")
    FUNCTION = "execute"
    CATEGORY = "MD_Nodes/Schedulers"

    def execute(self, model, steps, mode, denoise, rho):
        device = comfy.model_management.get_torch_device()
        try:
            ms = model.get_model_object("model_sampling")
            s_min, s_max = float(ms.sigma_min), float(ms.sigma_max)
        except Exception: s_min, s_max = CONST_FALLBACK_SIGMA_MIN, CONST_FALLBACK_SIGMA_MAX
            
        params = {"rho": rho, "bong_pivot": 0.5, "linear_steps": steps//2, "power": 2.0}
        
        if CORE_LOADED:
            sigmas = core.calculate_sigmas(steps, mode, s_min, s_max, device, **params)
        else:
            sigmas = torch.linspace(s_max, s_min, steps + 1, device=device)
        
        if denoise < 1.0:
            keep = int(steps * denoise)
            sigmas = sigmas[-(keep+1):]
            
        plot = self._plot(sigmas, f"Basic: {mode}")
        return (sigmas, plot)

class HybridAdaptiveSigmas_Lite(HybridSchedulerWrapperBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "DIFFUSION MODEL"}),
                "steps": ("INT", {"default": 20, "tooltip": "STEPS"}),
                "denoise": ("FLOAT", {"default": 1.0, "tooltip": "DENOISE"}),
            }
        }
    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "execute"
    CATEGORY = "MD_Nodes/Schedulers"

    def execute(self, model, steps, denoise):
        device = comfy.model_management.get_torch_device()
        try:
            ms = model.get_model_object("model_sampling")
            s_min, s_max = float(ms.sigma_min), float(ms.sigma_max)
        except Exception: s_min, s_max = 0.006, 1.0
            
        if CORE_LOADED:
            sigmas = core.calculate_sigmas(steps, "karras_rho", s_min, s_max, device, rho=7.0)
        else:
            sigmas = torch.linspace(s_max, s_min, steps + 1, device=device)
        
        if denoise < 1.0:
            keep = int(steps * denoise)
            sigmas = sigmas[-(keep+1):]
            
        return (sigmas,)

# =================================================================================
# == Sigma Utilities (Wrapper to Core)
# =================================================================================

class SigmaSmooth:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"tooltip": "INPUT SIGMAS"}),
                "smoothing_strength": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "STRENGTH\n• Amount of smoothing."
                }),
                "smoothing_type": (["gaussian", "moving_average", "exponential"], {
                    "tooltip": (
                        "ALGORITHM\n"
                        "• Gaussian: Bell curve weighted.\n"
                        "• MA: Simple average.\n"
                        "• Exponential: Decay weighted."
                    )
                }),
            },
            "optional": {
                "preserve_endpoints": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "PRESERVE START/END\n• Keep max/min noise values locked."
                }),
                "window_size": ("INT", {
                    "default": 3, "min": 2,
                    "tooltip": "WINDOW SIZE\n• Kernel size for smoothing."
                }),
            }
        }
    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "smooth"
    CATEGORY = "MD_Nodes/Schedulers/Utilities"
    
    def smooth(self, sigmas, smoothing_strength, smoothing_type, preserve_endpoints=True, window_size=3):
        if not CORE_LOADED: return (sigmas,)
        return (core.smooth_sigmas(sigmas, smoothing_strength, smoothing_type, preserve_endpoints, window_size),)

class SigmaConcatenate:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas_a": ("SIGMAS", {"tooltip": "SCHEDULE A (First)"}),
                "sigmas_b": ("SIGMAS", {"tooltip": "SCHEDULE B (Second)"}),
                "blend_mode": (["concatenate", "crossfade", "append_from_overlap"], {
                    "tooltip": (
                        "BLEND MODE\n"
                        "• Concatenate: Simple join.\n"
                        "• Crossfade: Smooth transition.\n"
                        "• Overlap: Joins where values match."
                    )
                }),
            },
            "optional": {
                "crossfade_steps": ("INT", {"default": 5, "tooltip": "FADE STEPS\n• For crossfade mode."}),
                "normalize_range": ("BOOLEAN", {"default": False, "tooltip": "NORMALIZE\n• Scale B to match A's end value."}),
            }
        }
    RETURN_TYPES = ("SIGMAS", "INT")
    RETURN_NAMES = ("sigmas", "total_steps")
    FUNCTION = "concatenate"
    CATEGORY = "MD_Nodes/Schedulers/Utilities"
    
    def concatenate(self, sigmas_a, sigmas_b, blend_mode, crossfade_steps=5, normalize_range=False):
        if not CORE_LOADED: return (sigmas_a, len(sigmas_a))
        res = core.concatenate_sigmas(sigmas_a, sigmas_b, blend_mode, crossfade_steps, normalize_range)
        return (res, len(res)-1)

# =================================================================================
# == Node Registration
# =================================================================================

NODE_CLASS_MAPPINGS = {
    "HybridAdaptiveSigmas": HybridAdaptiveSigmas_Advanced,
    "HybridAdaptiveSigmas_Basic": HybridAdaptiveSigmas_Basic,
    "HybridAdaptiveSigmas_Lite": HybridAdaptiveSigmas_Lite,
    "SigmaSmooth": SigmaSmooth,
    "SigmaConcatenate": SigmaConcatenate
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "HybridAdaptiveSigmas": "MD: Hybrid Scheduler (Advanced)",
    "HybridAdaptiveSigmas_Basic": "MD: Hybrid Scheduler (Basic)",
    "HybridAdaptiveSigmas_Lite": "MD: Hybrid Scheduler (Lite)",
    "SigmaSmooth": "MD: Sigma Smooth",
    "SigmaConcatenate": "MD: Sigma Concatenate"
}


# ==============================================================================
# == Unit Tests (smoke — runs without ComfyUI)
# ==============================================================================

if __name__ == "__main__":
    print("\n🧪 Smoke tests: MD_HybridSigmaScheduler")
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
    _check("CONST CONST_COLOR_PRIMARY defined", CONST_COLOR_PRIMARY is not None)
    _check("CONST CONST_FALLBACK_SIGMA_MIN defined", CONST_FALLBACK_SIGMA_MIN is not None)
    _check("CONST CONST_FALLBACK_SIGMA_MAX defined", CONST_FALLBACK_SIGMA_MAX is not None)
    _check("NODE_CLASS_MAPPINGS defined",
           isinstance(NODE_CLASS_MAPPINGS, dict) and len(NODE_CLASS_MAPPINGS) > 0)
    _check("  class HybridAdaptiveSigmas in map", "HybridAdaptiveSigmas" in NODE_CLASS_MAPPINGS)
    _check("  class HybridAdaptiveSigmas_Basic in map", "HybridAdaptiveSigmas_Basic" in NODE_CLASS_MAPPINGS)
    _check("  class HybridAdaptiveSigmas_Lite in map", "HybridAdaptiveSigmas_Lite" in NODE_CLASS_MAPPINGS)

    print(f"\n  {_pass} passed, {_fail} failed")
    if _fail == 0:
        print("  🎉 All good.")
