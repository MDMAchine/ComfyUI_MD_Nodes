# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# ████ APG GUIDER (FORKED) v0.2.0 – Refined & Improved ████▓▒░
# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀

# ░▒▓ ORIGIN & DEV:
#    • Original Sorcerer: Blepping? — github.com/blepping
#      (The architect of latent space, probably in a dark basement.)
#    • Forged in the fires of: MDMAchine & devstral (local l33t)
#      (Who knew coding could be this much fun? And confusing?)
#    • License: Apache 2.0 — We're not selling NFTs of your output... probably.

# ░▒▓ DESCRIPTION:
#    A powerful fork of Blepping's APG Guider.
#    This node injects Adaptive Projected Gradient (APG) guidance into
#    your ComfyUI sampling pipeline, allowing for advanced control over
#    latent space evolution. CFG is nice — APG is a damn scalpel.
#    Expect surgical precision... or chaos, depending on your config.

# ░▒▓ FEATURES:
#    ✓ Adaptive APG scheduling via YAML
#    ✓ CFG + APG hybrid guidance
#    ✓ Per-rule momentum, prediction, and mode control
#    ✓ Built-in debug visibility (verbose output)
#    ✓ Annotated example config included below
#    ✓ NEW: Toggle to disable APG and default to standard CFG.
#    ✓ NEW: Toggle for verbose (debug) output directly from the node.
#    ✓ IMPROVED: More robust parsing and cleaner internal logic.

# ░▒▓ CHANGELOG:
#    - v0.1 (Initial Fork Release):
#        • Fresh header branding (style is power)
#        • Removed eta scaling from `apg` projection (simplified flow)
#        • More robust `dims` parsing in `fixup_param`
#        • Tooltips added for all parameters
#        • YAML config example bundled directly in header
#    - v0.2 (Feature & Refinement Update):
#        • Added `disable_apg` boolean input to completely bypass APG guidance.
#        • Added `verbose_debug` boolean input for direct control over debug output.
#        • Refactored rule generation logic for improved readability.
#        • Added class docstrings for better code clarity.
#        • Made `dims` string parsing more robust against whitespace.
#        • Improved YAML tooltip with a simple, accessible example.

# ░▒▓ CONFIGURATION:
#    → Primary Use: APG-enhanced guided sampling
#    → Secondary Use: Experimental latent control for advanced users
#    → Edge Use: Replace traditional CFG-only workflows entirely

# ░▒▓ WARNING:
#    This node may trigger:
#    ▓▒░ Temporal distortion
#    ▓▒░ Memories of ANSI art & screaming modems
#    ▓▒░ A sense of unstoppable creative power

# ░▒▓ YAML EXAMPLE INPUT (start here, tweak later, probably break something):
# verbose: true  # Set to true to see debug messages in ComfyUI console.
#                # For when you want to feel like a hacker in a movie.
# rules:
# - start_sigma: -1.0
#   apg_scale: 0.0
#   cfg: 4.0
# - start_sigma: 0.85
#   apg_scale: 5.0
#   predict_image: true
#   cfg: 3.7
#   mode: pre_alt2
#   update_blend_mode: lerp
#   dims: [-2, -1]
#   momentum: 0.65
#   norm_threshold: 3.5
#   eta: 0.0
# - start_sigma: 0.55
#   apg_scale: 4.0
#   predict_image: true
#   cfg: 2.9
#   mode: pure_apg
# - start_sigma: 0.4
#   apg_scale: 3.8
#   predict_image: true
#   cfg: 2.75
#   mode: pure_apg
# - start_sigma: 0.15
#   apg_scale: 0.0
#   cfg: 2.6

# ░▒▓ Use at your own risk. May cause your images to transcend mere reality,
#    or just look a bit different. No refunds for spontaneous enlightenment
#    or existential crises. Remember to keep your GPUs cool (like your
#    attitude) and your dreams wild (like your code).
# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄


import math
import yaml

from enum import Enum, auto
from typing import NamedTuple, Union, List, Dict, Any

import torch
import torch.nn.functional as F

from tqdm import tqdm # Original uses tqdm, so keeping it. If it causes issues, can be removed.

from comfy.samplers import CFGGuider

import nodes

# This global variable and function ensure compatibility with Blepping's
# potential internal blending utility, if installed.
BLEND_MODES = None

def _ensure_blend_modes():
    global BLEND_MODES
    if BLEND_MODES is None:
        bleh = getattr(nodes, "_blepping_integrations", {}).get("bleh")
        if bleh is not None:
            # Assuming bleh.py.latent_utils.BLENDING_MODES exists if _blepping_integrations is present
            BLEND_MODES = bleh.py.latent_utils.BLENDING_MODES
        else:
            # Fallback for when blepping's integrations are not available
            BLEND_MODES = {"lerp": torch.lerp, "a_only": lambda a, _b, _t: a, "b_only": lambda _a, b, _t: b}

class UpdateMode(Enum):
    DEFAULT = auto()
    ALT1 = auto()
    ALT2 = auto()

class APGConfig(NamedTuple):
    start_sigma: float = math.inf
    momentum: float = -0.5
    eta: float = 0.0
    apg_scale: float = 4.0
    norm_threshold: float = 2.5
    dims: tuple = (-2, -1)
    update_mode: UpdateMode = UpdateMode.DEFAULT
    update_blend_mode: str = "lerp"
    cfg: float = 1.0
    apg_blend: float = 1.0
    apg_blend_mode: str = "lerp"
    predict_image: bool = True
    pre_cfg_mode: bool = False

    @staticmethod
    def fixup_param(k: str, v: Any) -> Any:
        if k == "dims":
            if isinstance(v, str):
                dims = v.strip()
                # IMPROVED: Add .strip() to each element to handle whitespace like "-1, -2"
                return tuple(int(d.strip()) for d in dims.split(",")) if dims else ()
            else:
                return v
        if k == "update_mode":
            # Robustly ensure the update_mode string is converted to the Enum member.
            mode_upper = str(v).strip().upper()
            mode_enum = UpdateMode.__members__.get(mode_upper)
            if mode_enum is None:
                # Log a warning to the console so the user is aware of the invalid input.
                tqdm.write(f"⚠️ Warning: Invalid UpdateMode '{v}' detected. Falling back to 'DEFAULT' mode.")
                return UpdateMode.DEFAULT
            return mode_enum
        if k == "start_sigma":
            # Convert negative values to infinity as per tooltip
            return math.inf if v < 0 else float(v)
        if k == "norm_threshold":
            # Ensure norm_threshold is a float, fallback to default if None
            return float(v) if v is not None else APGConfig.__annotations__['norm_threshold'].__forward_arg__
        return v

    @classmethod
    def build(cls, *, mode: str = "pure_apg", **params: dict) -> 'APGConfig':
        # Handle cases where mode might not have a '_' (e.g., "apg" or "cfg")
        if "_" in mode:
            pre_mode, update_mode_str = mode.split("_", 1)
        else:
            pre_mode = mode # Default to treating it as the primary mode
            update_mode_str = mode # Use the full mode name for update mode logic

        params["pre_cfg_mode"] = pre_mode == "pre"
        # Map certain update_mode strings to "default" UpdateMode enum
        params["update_mode"] = "default" if update_mode_str in {"apg", "cfg"} else update_mode_str
        
        fields = frozenset(cls._fields)
        # Apply fixup_param to all relevant parameters
        processed_params = {k: cls.fixup_param(k, v) for k, v in params.items() if k in fields}
        
        defaults = cls()
        # Merge default values for any missing parameters after processing
        final_params = {**{k: getattr(defaults, k) for k in fields if k not in processed_params}, **processed_params}
        
        return cls(**final_params)

    def __str__(self):
        # Determine which fields to display in the string representation
        if self.apg_blend == 0 or self.apg_scale == 0:
            fields = ("start_sigma", "cfg")
        else:
            fields = self._fields
        pretty_fields = ", ".join(f"{k}={getattr(self, k)}" for k in fields)
        return f"APGConfig({pretty_fields})"


class APG:
    """Implements the core APG logic, including momentum and vector projection."""
    def __init__(self, config: APGConfig):
        self.config = config
        self.running_average: Union[float, torch.Tensor] = 0.0

    def __getattr__(self, k: str) -> Any:
        # Delegate attribute access to the internal config object
        return getattr(self.config, k)

    def update(self, val: torch.Tensor) -> torch.Tensor:
        if self.momentum == 0:
            return val
        avg = self.running_average
        
        # Initialize or re-initialize running_average if types/shapes mismatch
        if isinstance(avg, float) or (isinstance(avg, torch.Tensor) and (avg.dtype != val.dtype or avg.device != val.device or avg.shape != val.shape)):
            self.running_average = val.clone()
            return self.running_average
            
        result = val + self.momentum * avg
        
        if self.update_mode == UpdateMode.ALT1:
            self.running_average = val + abs(self.momentum) * avg
        elif self.update_mode == UpdateMode.ALT2:
            blend = BLEND_MODES.get(self.update_blend_mode)
            if blend is None:
                raise ValueError(f"Unknown blend mode: {self.update_blend_mode}")
            result = blend(val, avg.neg() if self.momentum < 0 else avg, abs(self.momentum))
            self.running_average = blend(val, avg, abs(self.momentum))
        else: # UpdateMode.DEFAULT
            self.running_average = result
        return result

    def reset(self):
        self.running_average = 0.0

    def project(self, v0_orig: torch.Tensor, v1_orig: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Move tensors to CPU and convert to double for precision if on MPS device
        if v0_orig.device.type == "mps":
            v0, v1 = v0_orig.cpu().double(), v1_orig.cpu().double()
        else:
            v0, v1 = v0_orig.double(), v1_orig.double()
            
        v1 = F.normalize(v1, dim=self.dims)
        v0_p = (v0 * v1).sum(dim=self.dims, keepdim=True) * v1
        v0_o = v0 - v0_p
        
        # Convert back to original dtype and device
        return v0_p.to(dtype=v0_orig.dtype, device=v0_orig.device), \
               v0_o.to(dtype=v0_orig.dtype, device=v0_orig.device)

    def apg(self, cond: torch.Tensor, uncond: torch.Tensor) -> torch.Tensor:
        """Calculates the orthogonal guidance vector based on the APG configuration."""
        pred_diff = self.update(cond - uncond)
        
        if self.norm_threshold is not None and self.norm_threshold > 0:
            diff_norm = pred_diff.norm(p=2, dim=self.dims, keepdim=True)
            scale_factor = torch.minimum(torch.ones_like(pred_diff), self.norm_threshold / diff_norm)
            pred_diff = pred_diff * scale_factor
        
        diff_p, diff_o = self.project(pred_diff, cond)
        
        # This implementation intentionally uses only the orthogonal component.
        # The 'eta' parameter is present for config compatibility but not used in this projection logic.
        return diff_o

    def cfg_function(self, args: dict) -> torch.Tensor:
        # Determine if we're predicting image or noise based on config
        cond, uncond = (args["cond_denoised"], args["uncond_denoised"]) if self.predict_image else (args["cond"], args["uncond"])
        
        # Apply APG guidance and scale
        result = cond + (self.apg_scale - 1.0) * self.apg(cond, uncond)
        
        # Return based on prediction type (noise prediction requires subtracting from input)
        return args["input"] - result if self.predict_image else result

    def pre_cfg_function(self, args: dict) -> list:
        conds_out = args["conds_out"]
        if len(conds_out) < 2:
            return conds_out
            
        cond, uncond = conds_out[:2]
        update = self.apg(cond, uncond)
        
        # Calculate the APG-adjusted conditional input
        cond_apg = uncond + update + (cond - uncond) / self.apg_scale
        return [cond_apg, *conds_out[1:]]


class APGGuider(CFGGuider):
    """A custom CFGGuider that manages and applies APG rules during the sampling process."""
    def __init__(self, model, *, positive, negative, rules: tuple, params: dict) -> None:
        super().__init__(model)
        self.set_conds(positive, negative)
        self.set_cfg(1.0) # CFG value is set by the rule in predict_noise
        self.apg_rules = tuple(APG(rule_config) for rule_config in rules)
        # Use the 'verbose' key from params, defaulting to False if not present
        self.apg_verbose = params.get("verbose", False) 
        if self.apg_verbose:
            tqdm.write(f"* APG rules: {rules}")

    def apg_reset(self, *, exclude: APG = None):
        # Reset running average for all APG rules, except the one specified
        for apg_rule in self.apg_rules:
            if apg_rule is not exclude:
                apg_rule.reset()

    def apg_get_match(self, sigma: float) -> APG:
        # Find the first APG rule that applies at the current sigma
        for rule in self.apg_rules:
            if sigma <= rule.start_sigma:
                return rule
        # Fallback if no rule matches (should ideally not happen with math.inf rule)
        raise RuntimeError(f"Could not get APG rule for sigma={sigma}. Ensure a rule with start_sigma=inf is present.")

    def outer_sample(self, *args: list, **kwargs: dict) -> torch.Tensor:
        # Reset APG state before and after the main sampling process
        self.apg_reset()
        result = super().outer_sample(*args, **kwargs)
        self.apg_reset()
        return result

    def predict_noise(self, x: torch.Tensor, timestep, model_options=None, seed=None, **kwargs: dict) -> torch.Tensor:
        if model_options is None:
            model_options = {}
        
        # Get current sigma from timestep
        sigma = timestep.max().detach().cpu().item() if isinstance(timestep, torch.Tensor) else timestep
        
        # Get the appropriate APG rule for the current sigma
        rule = self.apg_get_match(sigma)
        
        # Reset other APG rules to prevent state leakage
        self.apg_reset(exclude=rule)
        
        # Check if APG is active for this rule (apg_blend is 0 if disabled or if disable_apg is True)
        matched = rule.apg_blend != 0 and rule.apg_scale != 0
        
        if self.apg_verbose:
            tqdm.write(f"* APG rule matched: sigma={sigma:.4f}, rule={rule.config}")
            
        if matched:
            # Disable ComfyUI's internal CFG1 optimization if APG is active
            model_options = model_options | {"disable_cfg1_optimization": True}
            if rule.pre_cfg_mode:
                # If pre-CFG mode, add APG function to pre_cfg_function list
                pre_cfg_handlers = model_options.get("sampler_pre_cfg_function", []).copy()
                pre_cfg_handlers.append(rule.pre_cfg_function)
                model_options["sampler_pre_cfg_function"] = pre_cfg_handlers
                cfg = rule.apg_scale # Use apg_scale as CFG for pre_cfg mode
            else:
                # Otherwise, set the custom CFG function directly
                model_options["sampler_cfg_function"] = rule.cfg_function
                cfg = rule.cfg # Use the rule's specified CFG
        else:
            # If no APG match, use the rule's default CFG
            cfg = rule.cfg
        
        orig_cfg = self.cfg # Store original CFG
        try:
            self.cfg = cfg # Set the CFG for this prediction
            result = super().predict_noise(x, timestep, model_options=model_options, seed=seed, **kwargs)
        finally:
            self.cfg = orig_cfg # Restore original CFG
            
        return result


class APGGuiderNode:
    """
    Node for ComfyUI to handle APG (Adaptive Projected Gradient) guidance.
    A powerful fork of Blepping's original APG Guider, with refined settings.
    """

    CATEGORY = "MD_Nodes/Guiders"
    FUNCTION = "go"
    RETURN_TYPES = ("GUIDER",)
    RETURN_NAMES = ("apg_guider",)
    DESCRIPTION = "APG Guider (Forked) with refined guidance settings"

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        _ensure_blend_modes() # Make sure BLEND_MODES are initialized
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "disable_apg": ("BOOLEAN", {"default": False, "label_on": "APG Disabled", "label_off": "APG Enabled"}),
                "verbose_debug": ("BOOLEAN", {"default": False, "label_on": "Verbose On", "label_off": "Verbose Off"}),
                "apg_scale": ("FLOAT", {"default": 4.5, "min": -1000.0, "max": 1000.0, "step": 0.1}),
                "cfg_before": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 1000.0, "step": 0.1}),
                "cfg_after": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 1000.0, "step": 0.1}),
                "eta": ("FLOAT", {"default": 0.0, "min": -1000.0, "max": 1000.0, "step": 0.1}),
                "norm_threshold": ("FLOAT", {"default": 2.5, "min": -1000.0, "max": 1000.0, "step": 0.1}),
                "momentum": ("FLOAT", {"default": 0.75, "min": -1000.0, "max": 1000.0, "step": 0.01}),
                "start_sigma": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 10000.0, "step": 0.01}),
                "end_sigma": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 10000.0, "step": 0.01}),
                "dims": ("STRING", {"default": "-1, -2"}),
                "predict_image": ("BOOLEAN", {"default": True}),
                "mode": (("pure_apg", "pre_cfg", "pure_alt1", "pre_alt1", "pure_alt2", "pre_alt2"),),
            },
            "optional": {
                "yaml_parameters_opt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "dynamic_prompt": False,
                        "tooltip": "Override all other settings with a YAML config. \n\n"
                                   "SIMPLE EXAMPLE (APG from sigma 14 down to 4):\n"
                                   "rules:\n"
                                   "- start_sigma: 14.0\n"
                                   "  apg_scale: 5.0\n"
                                   "  cfg: 4.0\n"
                                   "- start_sigma: 4.0\n"
                                   "  apg_scale: 0.0  # Turns off APG\n"
                                   "  cfg: 3.0"
                    },
                ),
            },
        }

    @classmethod
    def _build_rules_from_inputs(
        cls, cfg_before: float, cfg_after: float, start_sigma: float, end_sigma: float, **kwargs: Any
    ) -> List[APGConfig]:
        """Helper function to build the rule list from simple node inputs."""
        rules = []
        
        main_rule_params = {
            "cfg": cfg_before,
            "apg_blend": 1.0,
            "start_sigma": start_sigma if start_sigma >= 0 else math.inf,
            **kwargs
        }
        rules.append(APGConfig.build(**main_rule_params))

        if end_sigma > 0:
            # Using math.nextafter to ensure strict inequality when comparing floats
            adjusted_end_sigma = math.nextafter(end_sigma, -math.inf)
            rules.append(APGConfig.build(
                cfg=cfg_after,
                start_sigma=adjusted_end_sigma,
                apg_blend=0.0, # APG blending is off after end_sigma
            ))
        return rules

    @classmethod
    def go(
        cls, *,
        model, positive, negative, disable_apg, verbose_debug, apg_scale, cfg_before, cfg_after, eta,
        norm_threshold, momentum, start_sigma, end_sigma, dims, predict_image,
        mode, yaml_parameters_opt: str = None
    ) -> tuple:
        
        if yaml_parameters_opt is None:
            yaml_parameters_opt = ""
        yaml_parameters_opt = yaml_parameters_opt.strip()
        
        params: Dict[str, Any] = {"verbose": verbose_debug}
        rules: tuple = ()

        if yaml_parameters_opt:
            try:
                loaded_params = yaml.safe_load(yaml_parameters_opt)
                if isinstance(loaded_params, dict):
                    params.update(loaded_params)
                elif loaded_params:
                    params["rules"] = tuple(loaded_params)
                else:
                    raise TypeError("Bad format for YAML options: Must be a dict, list, or tuple.")
            except yaml.YAMLError as e:
                raise ValueError(f"Error parsing YAML parameters: {e}")

        rules = tuple(params.pop("rules", ()))

        if disable_apg:
            rules = (APGConfig.build(cfg=cfg_after, start_sigma=math.inf, apg_blend=0.0),)
            if verbose_debug:
                tqdm.write("INFO: APG Guider is disabled. Using standard CFG with 'cfg_after' value.")
        elif not rules:
            # REFACTORED: Use helper function to build rules from node inputs
            rules = cls._build_rules_from_inputs(
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
            )
        else:
            # If YAML rules exist, build them from the YAML data
            rules = tuple(APGConfig.build(**rule) for rule in rules)

        # Sort rules by start_sigma and ensure a fallback rule exists
        if not disable_apg:
            rules = sorted(rules, key=lambda rule: rule.start_sigma)
            if not rules or rules[-1].start_sigma < math.inf:
                fallback_rule = APGConfig.build(cfg=cfg_after, start_sigma=math.inf, apg_blend=0.0)
                rules = (*rules, fallback_rule)
            
        guider = APGGuider(
            model,
            positive=positive,
            negative=negative,
            rules=rules,
            params=params,
        )
        return (guider,)


# Register mappings for ComfyUI to discover this node
NODE_CLASS_MAPPINGS = {
    "APGGuiderForked": APGGuiderNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "APGGuiderForked": "APG Guider (Forked)",
}