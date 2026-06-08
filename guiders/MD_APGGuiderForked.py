# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# █▓▒░                                                                     ░▒▓█
# █▓▒░ MD_Nodes/APGGuiderForked – Adaptive Projected Gradient Guider v1... ░▒▓█
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
# ║ ░▒▓ ORIGIN & DEV:
# ║    • Cast into the void by: Blepping (Original)
# ║    • Enhanced by: MDMAchine
# ║    • Original source: github.com/blepping
# ║
# ║ ░▒▓ DESCRIPTION:
# ║    A robust fork of Blepping's APG Guider utilizing the MD_Nodes Core/Wrapper architecture.
# ║    Provides surgical, step-by-step control over latent space evolution by
# ║    scheduling APG scale, CFG, and momentum based on sigma
# ╚════════════════════════════════════════════════════════════════════════════


VERSION = "v3.0.0"  # UPS v1.5.8

import os
import sys
import logging
import math
import traceback
import io
import time

import torch
import yaml
import numpy as np

import comfy.samplers

# =================================================================================
# == MD_Nodes Universal Binary Loader (v1.6.1)
# =================================================================================

def find_core_paths():
    """Locate core directory in dev and production environments."""
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
    import apg_guider_core_bin as core
    CORE_LOADED = True
    CORE_MODE = "Binary (Production)"
except ImportError as e1:
    try:
        import apg_guider_core as core
        CORE_LOADED = True
        CORE_MODE = "Source (Development)"
    except ImportError as e2:
        CORE_ERROR = f"Binary: {e1}\nSource: {e2}"

# =================================================================================
# == Configuration Constants
# =================================================================================
CONST_PLOT_DPI = 100
CONST_PLOT_FIGSIZE = (10, 5)

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

# =================================================================================
# == Helper Classes
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
        print("\n⏱️  PERFORMANCE:")
        total = self.get_total_time()
        print(f"    • Total Time: {total:.2f}s")
        for op_name, times in sorted(self.timings.items()):
            avg = sum(times) / len(times)
            if len(times) == 1:
                print(f"    • {op_name}: {avg:.3f}s")
            else:
                print(f"    • {op_name}: {avg:.3f}s avg ({len(times)}x)")

# =================================================================================
# == UI Classes
# =================================================================================

class APGGuider(comfy.samplers.CFGGuider):
    """Custom CFG guider with APG rule-based guidance. Handled by wrapper UI."""
    def __init__(self, model, *, positive, negative, rules, params):
        super().__init__(model)
        self.set_conds(positive, negative)
        self.set_cfg(1.0)
        
        # Instantiate Core APG objects
        self.apg_rules = tuple(core.APG(rule_config) for rule_config in rules)
        self.apg_verbose = params.get("verbose", False)
        
        if self.apg_verbose:
            logging.info(f"\n{'='*60}")
            logging.info(f"[APGGuider] Initialized with {len(rules)} rule(s):")
            for i, rule in enumerate(rules, 1):
                sigma_str = "∞" if rule.start_sigma == math.inf else f"{rule.start_sigma:.2f}"
                apg_status = "ACTIVE" if (rule.apg_blend != 0 and rule.apg_scale != 0) else "DISABLED"
                logging.info(f"  Rule {i}: σ≤{sigma_str}, CFG={rule.cfg:.1f}, APG={rule.apg_scale:.1f} [{apg_status}]")
            logging.info(f"{'='*60}\n")

    def apg_reset(self, *, exclude=None):
        for apg_rule in self.apg_rules:
            if apg_rule is not exclude:
                apg_rule.reset()

    def apg_get_match(self, sigma):
        for rule in self.apg_rules:
            if sigma <= rule.start_sigma:
                return rule
        raise RuntimeError(f"No APG rule matched for sigma={sigma:.4f}.")

    def outer_sample(self, *args, **kwargs):
        self.apg_reset()
        result = super().outer_sample(*args, **kwargs)
        self.apg_reset()
        return result

    def predict_noise(self, x, timestep, model_options=None, seed=None, **kwargs):
        if model_options is None:
            model_options = {}
        
        sigma = (
            timestep.max().detach().cpu().item() 
            if isinstance(timestep, torch.Tensor) 
            else float(timestep)
        )
        
        rule = self.apg_get_match(sigma)
        self.apg_reset(exclude=rule)
        
        matched = rule.apg_blend != 0 and rule.apg_scale != 0
        
        if self.apg_verbose:
            apg_status = "ACTIVE" if matched else "BYPASSED"
            logging.info(
                f"[APGGuider] σ={sigma:.4f} → Rule: CFG={rule.cfg:.2f}, "
                f"APG_scale={rule.apg_scale:.2f} [{apg_status}]"
            )
        
        if matched:
            model_options = model_options | {"disable_cfg1_optimization": True}
            if rule.pre_cfg_mode:
                pre_cfg_handlers = model_options.get("sampler_pre_cfg_function", []).copy()
                pre_cfg_handlers.append(rule.pre_cfg_function)
                model_options["sampler_pre_cfg_function"] = pre_cfg_handlers
                cfg = rule.apg_scale
            else:
                model_options["sampler_cfg_function"] = rule.cfg_function
                cfg = rule.cfg
        else:
            cfg = rule.cfg
        
        orig_cfg = self.cfg
        try:
            self.cfg = cfg
            result = super().predict_noise(
                x, timestep, 
                model_options=model_options, 
                seed=seed, 
                **kwargs
            )
        finally:
            self.cfg = orig_cfg
        
        return result


class APGGuiderNode:
    """ComfyUI wrapper node for APG (Adaptive Projected Gradient) guidance."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {
                    "tooltip": (
                        "MODEL INPUT\n"
                        "• Purpose: The diffusion model to apply APG guidance to.\n"
                        "• Requirement: Any standard diffusion model (SD1.5, SDXL, Flux).\n"
                        "• Processing: Wraps model in custom CFGGuider."
                    )
                }),
                "positive": ("CONDITIONING", {
                    "tooltip": (
                        "POSITIVE COND\n"
                        "• Purpose: Your main prompt conditioning (what you want).\n"
                        "• Usage: Standard positive prompt input."
                    )
                }),
                "negative": ("CONDITIONING", {
                    "tooltip": (
                        "NEGATIVE COND\n"
                        "• Purpose: Negative prompt conditioning (what you avoid).\n"
                        "• Trade-off: APG uses this heavily for orthogonal projection math.\n"
                        "\n⭐ Essential for APG logic."
                    )
                }),
                "disable_apg": ("BOOLEAN", {
                    "default": False, 
                    "label_on": "APG Disabled", 
                    "label_off": "APG Enabled",
                    "tooltip": (
                        "DISABLE APG\n"
                        "• Purpose: Bypass APG entirely for A/B testing.\n"
                        "• Effect: If True, acts as standard CFGGuider.\n"
                        "\n⭐ Use to compare results without rewiring."
                    )
                }),
                "apg_scale": ("FLOAT", {
                    "default": 4.5, "min": 0.0, "max": 1000.0, "step": 0.1,
                    "tooltip": (
                        "APG SCALE\n"
                        "• Purpose: Strength of orthogonal correction (how much to fix the path).\n"
                        "• Range: 0.0 (Off) to 20.0 (Extreme).\n"
                        "• Trade-offs: Higher values enforce prompt adherence but may 'burn' the image.\n"
                        "• Recommended: 3.0 - 6.0.\n"
                        "\n⭐ Start with 4.5 for balanced results."
                    )
                }),
                "cfg_before": ("FLOAT", {
                    "default": 4.0, "min": 1.0, "max": 1000.0, "step": 0.1,
                    "tooltip": (
                        "CFG (BEFORE)\n"
                        "• Purpose: Static CFG applied *during* the APG active phase (High Sigma).\n"
                        "• Range: 1.0 to 30.0.\n"
                        "• Trade-offs: Lower values allow more creativity; higher values enforce strictness.\n"
                        "\n⭐ Recommended: 4.0 - 7.0."
                    )
                }),
                "cfg_after": ("FLOAT", {
                    "default": 3.0, "min": 1.0, "max": 1000.0, "step": 0.1,
                    "tooltip": (
                        "CFG (AFTER)\n"
                        "• Purpose: Static CFG applied *after* APG deactivates (Low Sigma).\n"
                        "• Range: 1.0 to 30.0.\n"
                        "• Trade-offs: Lower values here improve fine details and texture.\n"
                        "\n⭐ Recommended: 2.0 - 3.5."
                    )
                }),
                "norm_threshold": ("FLOAT", {
                    "default": 2.5, "min": 0.0, "max": 1000.0, "step": 0.1,
                    "tooltip": (
                        "NORM THRESHOLD\n"
                        "• Purpose: Cap the guidance vector magnitude to prevent artifacts.\n"
                        "• Range: 0.0 (No Cap) to 10.0.\n"
                        "• Trade-offs: Lower values prevent burning but reduce guidance strength.\n"
                        "\n⭐ Recommended: 2.5."
                    )
                }),
                "momentum": ("FLOAT", {
                    "default": 0.75, "min": -1000.0, "max": 1000.0, "step": 0.01,
                    "tooltip": (
                        "MOMENTUM\n"
                        "• Purpose: Smooths guidance vectors across steps using a running average.\n"
                        "• Range: -1.0 to 1.0.\n"
                        "• Effect: Positive values stabilize; negative values oscillate.\n"
                        "\n⭐ Recommended: 0.5 - 0.75."
                    )
                }),
                "start_sigma": ("FLOAT", {
                    "default": -1.0, "min": -1.0, "max": 10000.0, "step": 0.01,
                    "tooltip": (
                        "START SIGMA\n"
                        "• Purpose: Noise level where APG activates.\n"
                        "• Options: -1.0 (Infinity/Always On) or specific sigma (e.g., 15.0).\n"
                        "• Logic: Activates when current sigma <= start_sigma.\n"
                        "\n⭐ Recommended: -1.0 (or match your scheduler's max sigma)."
                    )
                }),
                "end_sigma": ("FLOAT", {
                    "default": -1.0, "min": -1.0, "max": 10000.0, "step": 0.01,
                    "tooltip": (
                        "END SIGMA\n"
                        "• Purpose: Noise level where APG deactivates (switching to 'cfg_after').\n"
                        "• Options: -1.0 (Never disable) or specific sigma (e.g., 1.0).\n"
                        "• Logic: Deactivates when current sigma < end_sigma.\n"
                        "\n⭐ Recommended: 1.0 to 3.0 (let the model refine details freely)."
                    )
                }),
                "eta": ("FLOAT", {
                    "default": 0.0, "min": -1000.0, "max": 1000.0, "step": 0.1,
                    "tooltip": (
                        "ETA (DEPRECATED)\n"
                        "• Purpose: Legacy parameter from original APG, no longer used in this fork.\n"
                        "• Recommendation: Leave at 0.0."
                    )
                }),
                "dims": ("STRING", {
                    "default": "-1, -2",
                    "tooltip": (
                        "DIMS\n"
                        "• Purpose: Dimensions for normalization and projection.\n"
                        "• Format: Comma-separated integers (e.g., '-1, -2').\n"
                        "• Meaning: -1,-2 corresponds to Height/Width spatial dimensions.\n"
                        "\n⭐ Recommended: '-1, -2'."
                    )
                }),
                "predict_image": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "PREDICT IMAGE\n"
                        "• Purpose: Defines target domain for guidance.\n"
                        "• Options: True (v-prediction/x0), False (epsilon/noise).\n"
                        "• Trade-offs: True usually works better for modern models.\n"
                        "\n⭐ Recommended: True."
                    )
                }),
                "mode": (
                    ("pure_apg", "pre_cfg", "pure_alt1", "pre_alt1", "pure_alt2", "pre_alt2"),
                    {
                        "tooltip": (
                            "APG MODE\n"
                            "• Purpose: Algorithm variant selection.\n"
                            "• Options:\n"
                            "  - pure_apg: Standard orthogonal projection.\n"
                            "  - pre_cfg: Applies projection before CFG scaling.\n"
                            "  - alt1/alt2: Alternative momentum blending strategies.\n"
                            "\n⭐ Recommended: 'pure_apg'."
                        )
                    }
                ),
            },
            "optional": {
                "yaml_parameters_opt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "dynamic_prompt": False,
                    "tooltip": (
                        "YAML OVERRIDE\n"
                        "• Purpose: Define complex multi-stage schedules.\n"
                        "• Format: List of dicts with 'start_sigma', 'cfg', 'apg_scale', etc.\n"
                        "• Priority: Overrides slider inputs if valid.\n"
                        "\n⭐ Advanced users only."
                    )
                }),
                "debug_mode": (["0 - Silent", "1 - Info", "2 - Verbose"], {
                    "default": "0 - Silent",
                    "tooltip": (
                        "LOGGING VERBOSITY\n"
                        "• Purpose: Controls console output details.\n"
                        "• Options:\n"
                        "  - 0: Minimal output.\n"
                        "  - 1: Basic info + Profiler report.\n"
                        "  - 2: Step-by-step guidance logs.\n"
                        "\n⭐ Recommended: 0 for production, 1 for tuning."
                    )
                }),
                "enable_profiling": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "ENABLE PROFILING\n"
                        "• Purpose: Measure timing of rule building and plotting.\n"
                        "• Automatic: Enabled if Debug Mode >= 1.\n"
                        "\n⭐ Recommended: False."
                    )
                }),
                "verbose_debug": ("BOOLEAN", {
                    "default": False, 
                    "label_on": "Verbose On", 
                    "label_off": "Verbose Off",
                    "tooltip": "LEGACY VERBOSE\n• Deprecated. Use 'debug_mode' instead."
                }),
            },
        }

    RETURN_TYPES = ("GUIDER", "IMAGE", "STRING")
    RETURN_NAMES = ("apg_guider", "guidance_plot", "config_summary")
    FUNCTION = "go"
    CATEGORY = "MD_Nodes/Guidance"

    @classmethod
    def _build_rules_from_inputs(cls, cfg_before, cfg_after, start_sigma, end_sigma, **kwargs):
        rules = []
        main_rule_params = {
            "cfg": cfg_before,
            "apg_blend": 1.0,
            "start_sigma": start_sigma if start_sigma >= 0 else math.inf,
            **kwargs
        }
        rules.append(core.APGConfig.build(**main_rule_params))

        if end_sigma > 0:
            rules.append(core.APGConfig.build(
                cfg=cfg_after,
                start_sigma=end_sigma,
                apg_blend=0.0,
            ))
        return rules

    @classmethod
    def _plot_rules(cls, rules):
        if not MATPLOTLIB_AVAILABLE or not PIL_AVAILABLE:
            return torch.zeros((1, 64, 64, 3), dtype=torch.float32)

        try:
            plt.style.use('dark_background')
            fig, ax = plt.subplots(figsize=CONST_PLOT_FIGSIZE)
            sigmas = np.geomspace(20, 0.01, 200)
            
            cfg_values, apg_values = [], []
            for s in sigmas:
                matched_rule = None
                for rule in rules:
                    if s <= rule.start_sigma:
                        matched_rule = rule
                        break
                
                if matched_rule:
                    cfg_values.append(matched_rule.cfg)
                    apg_val = matched_rule.apg_scale if (matched_rule.apg_blend != 0 and matched_rule.apg_scale != 0) else 0
                    apg_values.append(apg_val)
                else:
                    cfg_values.append(0)
                    apg_values.append(0)

            ax.plot(sigmas, cfg_values, label='CFG Scale', color='cyan', linewidth=2)
            ax.plot(sigmas, apg_values, label='APG Scale', color='magenta', linewidth=2, linestyle='--')
            
            ax.set_xscale('log')
            ax.invert_xaxis()
            ax.set_title("Guidance Schedule (Log Sigma)", fontsize=12, fontweight='bold')
            ax.set_xlabel("Sigma (Noise Level)", fontsize=10)
            ax.set_ylabel("Scale Value", fontsize=10)
            ax.legend(loc='upper right')
            ax.grid(True, which="both", ls="-", alpha=0.2)
            
            for rule in rules:
                if rule.start_sigma < 20:
                    ax.axvline(x=rule.start_sigma, color='yellow', linestyle=':', alpha=0.5)

            plt.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=CONST_PLOT_DPI, facecolor=fig.get_facecolor())
            buf.seek(0)
            plt.close(fig)
            
            img = Image.open(buf).convert("RGB")
            img_np = np.array(img).astype(np.float32) / 255.0
            return torch.from_numpy(img_np).unsqueeze(0)

        except Exception as e:
            print(f"[APGGuider] Plotting error: {e}")
            return torch.zeros((1, 64, 64, 3), dtype=torch.float32)

    @classmethod
    def go(cls, *, model, positive, negative, disable_apg, verbose_debug, apg_scale, 
           cfg_before, cfg_after, norm_threshold, momentum, start_sigma, 
           end_sigma, dims, predict_image, mode, eta=0.0, yaml_parameters_opt=None,
           debug_mode="0 - Silent", enable_profiling=False):
        
        debug_level = int(debug_mode.split(" ")[0])
        should_profile = enable_profiling or (debug_level >= 1)
        profiler = PerformanceProfiler(enabled=should_profile)
        profiler.start("total")
        
        # Graceful Degradation: If binary/source fails to load, gracefully fall back to standard CFG
        if not CORE_LOADED:
            logging.error(f"❌ ERROR: Core not available\nMode: {CORE_MODE or 'Not Loaded'}\nError: {CORE_ERROR}")
            fallback_cfg = cfg_after if not disable_apg else 1.0
            guider = comfy.samplers.CFGGuider(model)
            guider.set_conds(positive, negative)
            guider.set_cfg(fallback_cfg)
            blank_img = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            err_msg = "❌ ERROR: APG Core missing. Using standard CFG."
            return (guider, blank_img, err_msg)
        
        try:
            if yaml_parameters_opt is None:
                yaml_parameters_opt = ""
            yaml_parameters_opt = yaml_parameters_opt.strip()
            
            is_verbose = verbose_debug or (debug_level >= 2)
            params = {"verbose": is_verbose}
            rules = ()

            profiler.start("parse_config")
            if yaml_parameters_opt:
                try:
                    loaded_params = yaml.safe_load(yaml_parameters_opt)
                    if isinstance(loaded_params, dict):
                        params.update(loaded_params)
                    elif loaded_params:
                        params["rules"] = tuple(loaded_params)
                    else:
                        raise TypeError("Invalid YAML format. Expected dict or list.")
                except yaml.YAMLError as e:
                    raise ValueError(f"Error parsing YAML parameters: {e}")

            rules = tuple(params.pop("rules", ()))

            if disable_apg:
                rules = (core.APGConfig.build(cfg=cfg_after, start_sigma=math.inf, apg_blend=0.0),)
            elif not rules:
                rules = tuple(cls._build_rules_from_inputs(
                    cfg_before=cfg_before,
                    cfg_after=cfg_after,
                    start_sigma=start_sigma,
                    end_sigma=end_sigma,
                    momentum=momentum,
                    eta=eta,
                    apg_scale=apg_scale,
                    norm_threshold=norm_threshold,
                    dims=dims,
                    predict_image=predict_image,
                    mode=mode,
                ))
            else:
                rules = tuple(core.APGConfig.build(**rule) for rule in rules)

            if not disable_apg:
                rules = tuple(sorted(rules, key=lambda r: r.start_sigma))
                if not rules or rules[-1].start_sigma < math.inf:
                    fallback_rule = core.APGConfig.build(
                        cfg=cfg_after, 
                        start_sigma=math.inf, 
                        apg_blend=0.0
                    )
                    rules = (*rules, fallback_rule)
            profiler.stop("parse_config")
            
            summary = "============================================================\n"
            summary += f"[APGGuider] Initialized with {len(rules)} rule(s):\n"
            for i, rule in enumerate(rules, 1):
                sigma_str = "∞" if rule.start_sigma == math.inf else f"{rule.start_sigma:.2f}"
                apg_status = "ACTIVE" if (rule.apg_blend != 0 and rule.apg_scale != 0) else "DISABLED"
                summary += f"  Rule {i}: σ≤{sigma_str}, CFG={rule.cfg:.1f}, APG={rule.apg_scale:.1f} [{apg_status}]\n"
            summary += "============================================================"

            profiler.start("plotting")
            plot_image = cls._plot_rules(rules)
            profiler.stop("plotting")

            guider = APGGuider(
                model,
                positive=positive,
                negative=negative,
                rules=rules,
                params=params,
            )
            
            profiler.stop("total")
            if debug_level >= 1:
                print(summary)
                profiler.print_report()
                
            return (guider, plot_image, summary)

        except Exception as e:
            logging.error(f"[APGGuider] Failed to create guider: {e}")
            logging.debug(traceback.format_exc())
            
            fallback_cfg = cfg_after if not disable_apg else 1.0
            guider = comfy.samplers.CFGGuider(model)
            guider.set_conds(positive, negative)
            guider.set_cfg(fallback_cfg)
            
            blank_img = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            err_msg = f"ERROR: {str(e)}"
            return (guider, blank_img, err_msg)

# =================================================================================
# == Node Registration
# =================================================================================

NODE_CLASS_MAPPINGS = {
    "APGGuiderForked": APGGuiderNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "APGGuiderForked": "MD: APG Guider",
}


# ==============================================================================
# == Unit Tests (smoke — runs without ComfyUI)
# ==============================================================================

if __name__ == "__main__":
    print("\n🧪 Smoke tests: MD_APGGuiderForked")
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

    _check("VERSION defined",    VERSION == "v3.0.0")
    _check("CONST CONST_PLOT_DPI defined", CONST_PLOT_DPI is not None)
    _check("NODE_CLASS_MAPPINGS defined",
           isinstance(NODE_CLASS_MAPPINGS, dict) and len(NODE_CLASS_MAPPINGS) > 0)
    _check("  class APGGuiderForked in map", "APGGuiderForked" in NODE_CLASS_MAPPINGS)

    print(f"\n  {_pass} passed, {_fail} failed")
    if _fail == 0:
        print("  🎉 All good.")
