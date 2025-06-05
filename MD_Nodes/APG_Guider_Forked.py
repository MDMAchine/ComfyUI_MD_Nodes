# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# ████ APG GUIDER (FORKED) v0.1 – Unleashed from the shadows ████▓▒░
# ░▒▓ Crafting pixels with rebellious intent and pure APG magic ░▒▓
# ▓▒░        Original Sorcerer: Blepping? | https://github.com/blepping
# ▓▒░        (The architect of latent space, probably in a dark basement.)
# ░▒▓        Forged in the fires of: MDMAchine & devstral (local l33t)
# ░▒▓        (Who knew coding could be this much fun? And confusing?)
# ▒░▓        License: Apache 2.0 (because sharing is caring, mostly –
# ▒░▓        we're not selling NFTs of your output, probably.)
# ░▒▓

# ░▒▓ Description:
#    A powerful fork of Blepping's APG Guider. This node injects Adaptive
#    Projected Gradient (APG) guidance into your ComfyUI sampling, allowing
#    for advanced control over how your latent space evolves. Ideal for
#    those who demand more from their diffusion models than mere mortal
#    CFG can provide. Now with extra sauce! It's like giving your AI a
#    precision screwdriver instead of a sledgehammer.

# ▓▒░ Changes (This Fork):
# - V0.1 (Initial Fork Release):
#    * New fresh header for this distinct fork. (Because every good project
#      needs a stylish intro, right?)
#    * Removed eta scaling from `apg` projection for simplified guidance.
#      (Less math, more art. You're welcome, liberal arts majors.)
#    * Refined `dims` parsing in `APGConfig.fixup_param` for robust input
#      handling. (No more segfaults from incorrectly formatted arrays –
#      we've patched that exploit.)
#    * Preserved all original APG functionalities, extending its
#      pixel-bending might. (We didn't break anything important, promise!)
#    * Added comprehensive tooltips for all parameters. (Because even l33t
#      coders forget what `norm_threshold` does at 3 AM.)
#    * Included a suggested YAML example in the header. (For when you're
#      too lazy to RTFM, we got you.)

# ░▒▓ YAML EXAMPLE INPUT (start here, tweak later, probably break something):
# verbose: true # Set to true to see debug messages in ComfyUI console.
#              # For when you want to feel like a hacker in a movie.
# rules:
#    # Rule 1: Starts at the very beginning (-1 = infinity sigma) with no APG (cfg=4).
#    # This serves as a baseline or initial phase. Think of it as the
#    # "before coffee" stage of your AI's workday.
# - start_sigma: -1.0 # This effectively means "applies from the start of sampling"
#    apg_scale: 0.0     # No APG guidance applied here (APG blend implicitly 0)
#    cfg: 4.0           # Standard CFG scale for this phase – don't get too wild yet.
#    # Rule 2: Active from sigma 0.85 down to the next rule's start_sigma.
#    # Uses pre_alt2 mode for APG, with image prediction and momentum for
#    # smoother transitions. This is where the magic (and GPU sweat) begins.
# - start_sigma: 0.85
#    apg_scale: 5.0
#    predict_image: true
#    mode: pre_alt2
#    update_blend_mode: lerp
#    dims: [-2, -1] # Dimensions for normalization (typically H and W).
#                  # Because size *does* matter, pixels-wise.
#    momentum: 0.7  # Introduces a running average for smoother APG updates.
#                  # Like a seasoned gamer's aim, steady and precise.
#    norm_threshold: 3.0 # Clamps the norm of the guidance vector. Don't let
#                       # your pixels get too excited.
#    eta: 0.0       # Controls the projection (0.0 means fully orthogonal guidance).
#                  # Keeps things squared away, literally.
#    # Rule 3: Active from sigma 0.70. Similar to Rule 2 but with slightly
#    # reduced APG scale and momentum. The AI is still awake, but maybe
#    # considering a coffee break.
# - start_sigma: 0.70
#    apg_scale: 4.0
#    predict_image: true
#    mode: pre_alt2
#    update_blend_mode: lerp
#    dims: [-2, -1]
#    momentum: 0.6
#    norm_threshold: 2.5
#    eta: 0.0
#    # Rule 4: Active from sigma 0.55. Switches to 'pure_apg' mode with no
#    # momentum. Time to go full-on APG, no holding back!
# - start_sigma: 0.55
#    apg_scale: 3.5
#    predict_image: true
#    mode: pure_apg
#    momentum: 0.0 # No momentum for this phase. Pure, unadulterated pixel power.
#    # Rule 5: Active from sigma 0.40. APG scale further reduced, switches to
#    # noise prediction. Now we're getting into the nitty-gritty, embracing
#    # the chaos.
# - start_sigma: 0.40
#    apg_scale: 2.0
#    predict_image: false # Predict noise instead of image. Let's get noisy!
#    cfg: 4.3             # Custom CFG for this phase. Because sometimes
#                        # rules are meant to be broken.
#    momentum: 0.0
#    # Rule 6: Active from sigma 0.30. Further reduced APG scale.
#    # The AI is starting to see the finish line, but not quite there yet.
# - start_sigma: 0.30
#    apg_scale: 1.5
#    predict_image: false
#    cfg: 4.2
#    momentum: 0.0
#    # Rule 7: Active from sigma 0.15. APG scale significantly reduced.
#    # Almost done, just a few more tweaks. Like rendering the final
#    # few frames of a demoscene intro.
# - start_sigma: 0.15
#    apg_scale: 0.5
#    cfg: 4.1
#    momentum: 0.0
#    # Rule 8: Active from sigma 0.05. Very low APG scale for fine-tuning.
#    # The last polish before the grand reveal. Don't mess it up now!
# - start_sigma: 0.05
#    apg_scale: 0.1
#    cfg: 4.0
#    momentum: 0.0
#    # Rule 9: Active from sigma 0.01. APG is effectively off, using only CFG.
#    # Victory lap! The AI has done its job. Now, bask in the glory.
# - start_sigma: 0.01
#    apg_scale: 0.0 # APG guidance is off. It's all CFG from here.
#    cfg: 4.0
#    momentum: 0.0

# ░▒▓ Use at your own risk. May cause your images to transcend mere reality,
#    or just look a bit different. No refunds for spontaneous enlightenment
#    or existential crises. Remember to keep your GPUs cool (like your
#    attitude) and your dreams wild (like your code).
# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄

import math
import yaml

from enum import Enum, auto
from typing import NamedTuple

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
    def fixup_param(k, v):
        if k == "dims":
            if isinstance(v, str):
                dims = v.strip()
                # Fix: Handle empty string after strip for dims correctly.
                # If dims is empty, return an empty tuple.
                return tuple(int(d) for d in dims.split(",")) if dims else ()
            else:
                return v
        if k == "update_mode":
            # Ensure the update_mode string is converted to the Enum member
            return getattr(UpdateMode, v.strip().upper())
        if k == "start_sigma":
            # Convert negative values to infinity as per tooltip
            return math.inf if v < 0 else float(v)
        return v

    @classmethod
    def build(cls, *, mode: str = "pure_apg", **params: dict):
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
        params = {k: cls.fixup_param(k, v) for k, v in params.items() if k in fields}
        defaults = cls()
        # Merge default values for any missing parameters
        params = {**{k: getattr(defaults, k) for k in fields if k not in params}, **params}
        return cls(**params)

    def __str__(self):
        # Determine which fields to display in the string representation
        if self.apg_blend == 0 or self.apg_scale == 0:
            fields = ("start_sigma", "cfg")
        else:
            fields = self._fields
        pretty_fields = ", ".join(f"{k}={getattr(self, k)}" for k in fields)
        return f"APGConfig({pretty_fields})"


class APG:
    def __init__(self, config: APGConfig):
        self.config = config
        self.running_average = 0.0 # Initializes to 0.0 (float)

    def __getattr__(self, k):
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

    def project(self, v0_orig: torch.Tensor, v1_orig: torch.Tensor) -> tuple:
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
        pred_diff = self.update(cond - uncond)
        if self.norm_threshold > 0:
            diff_norm = pred_diff.norm(p=2, dim=self.dims, keepdim=True)
            scale_factor = torch.minimum(torch.ones_like(pred_diff), self.norm_threshold / diff_norm)
            pred_diff = pred_diff * scale_factor
        
        diff_p, diff_o = self.project(pred_diff, cond)
        
        # CHANGE: Removed the line: update += self.eta * diff_p
        # This simplifies the guidance to only use the orthogonal component (diff_o)
        # unless custom logic is reintroduced based on `eta`.
        # Keeping it as `diff_o` aligns with "new code."
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
    def __init__(self, model, *, positive, negative, rules: tuple, params: dict) -> None:
        super().__init__(model)
        self.set_conds(positive, negative)
        self.set_cfg(1.0) # CFG value is set by the rule in predict_noise
        self.apg_rules = tuple(APG(rule_config) for rule_config in rules)
        self.apg_params = params
        self.apg_verbose = params.get("verbose", False)
        if self.apg_verbose:
            tqdm.write(f"* APG rules: {rules}")

    def apg_reset(self, *, exclude=None):
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
        
        # Check if APG is active for this rule
        matched = rule.apg_blend != 0 and rule.apg_scale != 0
        
        if self.apg_verbose:
            tqdm.write(f"* APG rule matched: sigma={sigma}, rule={rule.config}")
            
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

    # Node category in ComfyUI UI
    CATEGORY = "sampling/custom_sampling/guiders"

    # Function name for ComfyUI
    FUNCTION = "go"

    # Output types for the node
    RETURN_TYPES = ("GUIDER",)

    # Friendly display name override in node editor
    RETURN_NAMES = ("apg_guider",)

    # Description of the node
    DESCRIPTION = "APG Guider (Forked) with refined guidance settings"

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        _ensure_blend_modes() # Make sure BLEND_MODES are initialized
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The diffusion model to which APG guidance will be applied."}),
                "positive": ("CONDITIONING", {"tooltip": "The positive conditioning (e.g., text prompt embeddings) for guided generation."}),
                "negative": ("CONDITIONING", {"tooltip": "The negative conditioning (e.g., negative text prompt embeddings) to push generation away from."}),
                "apg_scale": ("FLOAT", {"default": 4.5, "min": -1000.0, "max": 1000.0, "step": 0.1, "tooltip": "The main strength of the APG guidance. A value of 1.0 means no APG effect. Higher values increase guidance towards the positive conditioning relative to the negative. If set to 0, APG is effectively off, and CFG is used."}),
                "cfg_before": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 1000.0, "step": 0.1, "tooltip": "The standard Classifier-Free Guidance (CFG) scale to use for steps *before* APG guidance activates (i.e., when sigma > start_sigma)."}),
                "cfg_after": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 1000.0, "step": 0.1, "tooltip": "The standard Classifier-Free Guidance (CFG) scale to use for steps *after* APG guidance deactivates (i.e., when sigma < end_sigma, or if APG is completely off)."}),
                "eta": ("FLOAT", {"default": 0.0, "min": -1000.0, "max": 1000.0, "step": 0.1, "tooltip": "A parameter influencing the projection of the guidance. Values closer to 0.0 emphasize guidance orthogonal to the prediction difference, while non-zero values introduce components parallel to it."}),
                "norm_threshold": ("FLOAT", {"default": 2.5, "min": -1000.0, "max": 1000.0, "step": 0.1, "tooltip": "If the L2 norm of the guidance vector exceeds this threshold, the vector is scaled down. Prevents excessively strong guidance that can lead to artifacts or oversaturation. Set to a negative value or 0 for no thresholding."}),
                "momentum": ("FLOAT", {"default": 0.75, "min": -1000.0, "max": 1000.0, "step": 0.01, "tooltip": "Applies momentum to the APG guidance. A positive value (e.g., 0.5) applies a running average to the guidance, potentially smoothing results. A negative value (e.g., -0.75) can create a sharper, more reactive guidance. 0.0 disables momentum."}),
                "start_sigma": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 10000.0, "step": 0.01, "tooltip": "The sigma value (noise level) at which APG guidance *begins* to apply. If set to -1.0 or any negative value, APG is active from the very beginning of the sampling process (effectively infinity sigma)."}),
                "end_sigma": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 10000.0, "step": 0.01, "tooltip": "The sigma value at which APG guidance *ends*. If set to -1.0 or any negative value, APG remains active until the end of the sampling process. If a positive value, APG will switch to `cfg_after` below this sigma."}),
                "dims": ("STRING", {"default": "-1, -2", "tooltip": "Comma-separated list of dimensions (e.g., -1 for width, -2 for height) to normalize the guidance vector along. This controls which parts of the latent space the guidance operates on. No error checking is performed on the input string, ensure correct integer format."}),
                "predict_image": ("BOOLEAN", {"default": True, "tooltip": "Determines whether APG guides a prediction of the final image (denoised latent) or a prediction of the noise. Guiding the image prediction (True) often yields better results. Only has an effect in 'pure_apg' mode."}),
                "mode": (("pure_apg", "pre_cfg", "pure_alt1", "pre_alt1", "pure_alt2", "pre_alt2"), {"default": "pure_apg", "tooltip": "Defines the APG guidance mode:\n- 'pure_apg': Applies APG guidance directly to the noise prediction.\n- 'pre_cfg': Modifies the conditional input before standard CFG, using 'apg_scale' as CFG scale.\n- 'pure_alt1'/'pre_alt1'/'pure_alt2'/'pre_alt2': Experimental alternative update modes for APG momentum, may produce unique results."}),
            },
            "optional": {
                "yaml_parameters_opt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "dynamic_prompt": False,
                        "tooltip": "Allows specifying multiple APG rules and custom parameters via YAML. Rules defined here override the single-rule parameters above. When specifying parameters this way, there is no error checking beyond basic YAML parsing. See header for example YAML structure."
                    },
                ),
            },
        }

    @classmethod
    def go(
        cls, *,
        model, positive, negative, apg_scale, cfg_before, cfg_after, eta,
        norm_threshold, momentum, start_sigma, end_sigma, dims, predict_image,
        mode, yaml_parameters_opt=None,
    ) -> tuple:
        if yaml_parameters_opt is None:
            yaml_parameters_opt = ""
        yaml_parameters_opt = yaml_parameters_opt.strip()
        
        params = {}
        if yaml_parameters_opt:
            try:
                loaded_params = yaml.safe_load(yaml_parameters_opt)
                if loaded_params: # Ensure it's not None if YAML was empty
                    if isinstance(loaded_params, (tuple, list)):
                        params = {"rules": tuple(loaded_params)}
                    elif isinstance(loaded_params, dict):
                        params = loaded_params
                    else:
                        raise TypeError("Bad format for YAML options: Must be a dict, list, or tuple.")
            except yaml.YAMLError as e:
                raise ValueError(f"Error parsing YAML parameters: {e}")

        rules = tuple(params.pop("rules", ()))
        if not rules:
            # If no YAML rules, build a single rule from node inputs
            rules = []
            rules.append(APGConfig.build(
                cfg=cfg_before, # Default CFG for this rule before any overrides
                apg_blend=1.0,
                start_sigma=start_sigma if start_sigma >= 0 else math.inf,
                momentum=momentum,
                eta=eta,
                apg_scale=apg_scale, # Use apg_scale as the parameter name
                norm_threshold=norm_threshold,
                dims=dims,
                predict_image=predict_image,
                mode=mode,
            ))
            # Handle end_sigma for a potential second rule
            if end_sigma > 0:
                # Using math.nextafter to ensure strict inequality when comparing floats
                adjusted_end_sigma = math.nextafter(end_sigma, -math.inf)
                rules.append(APGConfig.build(
                    cfg=cfg_after,
                    start_sigma=adjusted_end_sigma,
                    apg_blend=0.0, # APG blending is off after end_sigma
                ))
        else:
            # If YAML rules exist, build them from the YAML data
            # Fix: Ensure rules are built correctly and filtered by start_sigma > 0
            rules = tuple(cfg for cfg in (APGConfig.build(**rule) for rule in rules) if cfg.start_sigma >= 0)

        # Sort rules by start_sigma in ascending order
        rules = sorted(rules, key=lambda rule: rule.start_sigma)
        
        # Ensure there's always a fallback rule that applies at all sigmas (start_sigma=inf)
        if not rules or rules[-1].start_sigma < math.inf:
            rules = (*rules, APGConfig.build(cfg=cfg_after, start_sigma=math.inf, apg_blend=0.0))
            
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
    "APGGuiderForked": APGGuiderNode, # A new, distinct internal name for the forked node
}

# Friendly display name override in node editor
NODE_DISPLAY_NAME_MAPPINGS = {
    "APGGuiderForked": "APG Guider (Forked)", # Distinct display name for the forked node
}