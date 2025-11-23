# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
# ████ MD_Nodes/APGGuiderForked – Adaptive Projected Gradient Guider ████▓▒░
# © 2025 MDMAchine
# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
# ░▒▓ ORIGIN & DEV:
#   • Cast into the void by: Blepping (Original)
#   • Enhanced by: MDMAchine, devstral, Gemini
#   • License: Apache 2.0
#   • Original source (if applicable): github.com/blepping

# ░▒▓ DESCRIPTION:
#   A robust fork of Blepping's APG (Adaptive Projected Gradient) Guider.
#   Provides surgical, step-by-step control over latent space evolution by
#   scheduling APG scale, CFG, and momentum based on sigma (noise level).

# ░▒▓ NATIVE COMPARISON (v1.4.5):
#   This node extends comfy.samplers.CFGGuider with:
#   ✓ Sigma-based rule scheduling (native CFGGuider uses fixed CFG)
#   ✓ Orthogonal projection guidance (APG algorithm, not in native)
#   ✓ Momentum-based running average (not in native)
#   ✓ Per-rule configuration via YAML (not in native)
#   Native location: comfy/samplers.py::CFGGuider
#   Assessment: Extension node - adds significant new functionality beyond native.

# ░▒▓ FEATURES:
#   ✓ Schedules APG scale, CFG, and momentum using sigma-based rules.
#   ✓ YAML-based configuration for complex, multi-stage guidance.
#   ✓ Enhanced robustness: division-by-zero protection, input validation.
#   ✓ Verbose debugging logs (via logging module) to see rule activation.

# ░▒▓ CHANGELOG:
#   - v1.4.5 (Compliance Finalization - Nov 2025):
#       • DOCS: Added Native Comparison to header.
#       • DOCS: Rewrote all tooltips to standard 4-part format.
#       • DOCS: Added type-hint replacement docstrings to all methods.
#       • FIXED: Changed _fields to immutable tuple.
#       • FIXED: Slerp float comparison safety.
#   - v1.4.3 (Optimization):
#       • FIXED: FP16 underflow protection.
#       • OPTIMIZED: Cached MPS device check.

# ░▒▓ CONFIGURATION:
#   → Primary Use: Using YAML to apply strong APG at high sigmas and low/zero APG at low sigmas.
#   → Secondary Use: Disabling APG (`apg_scale: 0.0`) to use as a simple CFG/momentum scheduler.

# ░▒▓ WARNING:
#   This node may trigger:
#   ▓▒░ A 4-hour YAML editing session just to change one sigma value.
#   ▓▒░ The unshakable, god-like feeling of *finally* understanding orthogonal projection.
#   Consult your nearest demoscene vet if hallucinations persist.
# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄


# =================================================================================
# == Standard Library Imports                                                     ==
# =================================================================================
import enum
import logging
import math
import traceback

# =================================================================================
# == Third-Party Imports                                                          ==
# =================================================================================
import torch
import torch.nn.functional as F
import yaml

# =================================================================================
# == ComfyUI Core Modules                                                         ==
# =================================================================================
import comfy.samplers
import nodes

# =================================================================================
# == Helper Classes & Dependencies                                                ==
# =================================================================================

# Global blend modes with comprehensive fallbacks
BLEND_MODES = None

def _ensure_blend_modes():
    """
    Initialize blend modes with comprehensive fallbacks.
    Checks for Blepping's integrations first, then falls back.
    """
    global BLEND_MODES
    if BLEND_MODES is None:
        bleh = getattr(nodes, "_blepping_integrations", {}).get("bleh")
        if bleh is not None:
            try:
                BLEND_MODES = bleh.py.latent_utils.BLENDING_MODES
            except AttributeError:
                bleh = None
        
        if bleh is None:
            # Comprehensive fallback blend modes
            BLEND_MODES = {
                "lerp": torch.lerp,
                "slerp": lambda a, b, t: _slerp(a, b, t),
                "add": lambda a, b, t: a + b * t,
                "multiply": lambda a, b, t: a * (1 + (b - 1) * t),
                "a_only": lambda a, b, t: a,
                "b_only": lambda a, b, t: b,
                "average": lambda a, b, t: (a + b) / 2,
            }

def _slerp(a, b, t):
    """
    Spherical linear interpolation fallback.
    
    Args:
        a (Tensor): Start tensor
        b (Tensor): End tensor
        t (float): Interpolation factor
        
    Returns:
        Tensor: Interpolated result
    """
    omega = torch.acos((a * b).sum() / (a.norm() * b.norm()))
    so = torch.sin(omega)
    if abs(so) < 1e-6: # Fixed float comparison
        return torch.lerp(a, b, t) 
    return (torch.sin((1.0 - t) * omega) / so) * a + (torch.sin(t * omega) / so) * b

def _validate_dims(dims, tensor_ndim=4):
    """
    Validate dimension tuple for tensor operations.
    
    Args:
        dims (tuple): Tuple of dimension indices
        tensor_ndim (int): Number of dimensions (default 4)
        
    Returns:
        tuple: Validated dimensions
    """
    if not dims:
        raise ValueError("dims cannot be empty. Use dimensions like (-1, -2) for spatial dims.")
    
    for dim in dims:
        if not isinstance(dim, int):
            raise TypeError(f"All dims must be integers, got {type(dim)} for value {dim}")
        # Normalize negative indices
        normalized_dim = dim if dim >= 0 else tensor_ndim + dim
        if normalized_dim < 0 or normalized_dim >= tensor_ndim:
            raise ValueError(f"Dimension {dim} is out of bounds for {tensor_ndim}D tensor")
    
    return dims

class UpdateMode(enum.Enum):
    """Momentum update modes for APG guidance."""
    DEFAULT = enum.auto()
    ALT1 = enum.auto()
    ALT2 = enum.auto()

class APGConfig:
    """
    Helper class for APG (Adaptive Projected Gradient) configuration.
    Stores the settings for a single sigma-based rule.
    """
    def __init__(self,
                 start_sigma=math.inf,
                 momentum=-0.5,
                 eta=0.0, # Deprecated
                 apg_scale=4.0,
                 norm_threshold=2.5,
                 dims=(-2, -1),
                 update_mode=None, # Will be set to UpdateMode.DEFAULT
                 update_blend_mode="lerp",
                 cfg=1.0,
                 apg_blend=1.0,
                 apg_blend_mode="lerp",
                 predict_image=True,
                 pre_cfg_mode=False):
        """
        Initialize configuration for a single guidance rule.
        
        Args:
            start_sigma (float): Sigma threshold to activate rule
            momentum (float): Momentum factor
            apg_scale (float): APG guidance strength
            norm_threshold (float): Max norm for guidance vector
            dims (tuple): Dimensions to normalize over
            cfg (float): Standard CFG scale
            ... (other params)
        """
        
        self.start_sigma = start_sigma
        self.momentum = momentum
        self.eta = eta
        self.apg_scale = apg_scale
        self.norm_threshold = norm_threshold
        self.dims = dims
        self.update_mode = update_mode if update_mode is not None else UpdateMode.DEFAULT
        self.update_blend_mode = update_blend_mode
        self.cfg = cfg
        self.apg_blend = apg_blend
        self.apg_blend_mode = apg_blend_mode
        self.predict_image = predict_image
        self.pre_cfg_mode = pre_cfg_mode

        # Fixed: Tuple for immutability
        self._fields = (
            "start_sigma", "momentum", "eta", "apg_scale", "norm_threshold",
            "dims", "update_mode", "update_blend_mode", "cfg", "apg_blend",
            "apg_blend_mode", "predict_image", "pre_cfg_mode"
        )

    @staticmethod
    def fixup_param(k, v):
        """Process and validate configuration parameters."""
        if k == "dims":
            if isinstance(v, str):
                dims_str = v.strip()
                if not dims_str:
                    raise ValueError("dims string cannot be empty")
                try:
                    parsed_dims = tuple(int(d.strip()) for d in dims_str.split(","))
                except ValueError as e:
                    raise ValueError(f"Invalid dims format '{v}'. Expected comma-separated integers like '-1, -2'. Error: {e}")
                return _validate_dims(parsed_dims)
            elif isinstance(v, (list, tuple)):
                return _validate_dims(tuple(v))
            else:
                raise TypeError(f"dims must be string or tuple, got {type(v)}")
        
        if k == "update_mode":
            mode_upper = str(v).strip().upper()
            mode_enum = UpdateMode.__members__.get(mode_upper)
            if mode_enum is None:
                return UpdateMode.DEFAULT
            return mode_enum
        
        if k == "start_sigma":
            return math.inf if float(v) < 0 else float(v)
        
        if k == "norm_threshold":
            threshold = float(v) if v is not None else 2.5
            if threshold < 0:
                return 0.0
            return threshold
        
        if k == "apg_scale":
            scale = float(v)
            if abs(scale) < 1e-6:  # Effectively zero
                return 0.0
            return scale
        
        return v

    @classmethod
    def build(cls, *, mode="pure_apg", **params):
        """Build an APGConfig from mode string and parameters."""
        if "_" in mode:
            pre_mode, update_mode_str = mode.split("_", 1)
        else:
            pre_mode = mode
            update_mode_str = mode

        params["pre_cfg_mode"] = pre_mode == "pre"
        params["update_mode"] = "default" if update_mode_str in {"apg", "cfg"} else update_mode_str
        
        fields = frozenset(cls()._fields)
        processed_params = {k: cls.fixup_param(k, v) for k, v in params.items() if k in fields}
        
        defaults = cls()
        final_params = {
            **{k: getattr(defaults, k) for k in fields if k not in processed_params},
            **processed_params
        }
        
        return cls(**final_params)

    def __str__(self):
        """Human-readable string representation."""
        if self.apg_blend == 0 or self.apg_scale == 0:
            fields = ("start_sigma", "cfg")
        else:
            fields = self._fields
        pretty_fields = ", ".join(f"{k}={getattr(self, k)}" for k in fields)
        return f"APGConfig({pretty_fields})"


class APG:
    """
    Core APG logic with momentum and orthogonal projection.
    """
    
    def __init__(self, config):
        """
        Initialize APG Logic.
        
        Args:
            config (APGConfig): The configuration for this rule.
        """
        self.config = config
        self.running_average = 0.0
        self.is_mps = None  # Cached device check

    def __getattr__(self, k):
        """Delegate attribute access to config."""
        return getattr(self.config, k)

    def update(self, val):
        """
        Update running average with momentum.
        
        Args:
            val (Tensor): New value to incorporate
            
        Returns:
            Tensor: Updated momentum state
        """
        if self.momentum == 0:
            return val
        
        avg = self.running_average
        
        # Initialize or re-initialize if needed
        if isinstance(avg, float) or (
            isinstance(avg, torch.Tensor) and 
            (avg.dtype != val.dtype or avg.device != val.device or avg.shape != val.shape)
        ):
            self.running_average = val.clone()
            return self.running_average
        
        result = val + self.momentum * avg
        
        if self.update_mode == UpdateMode.ALT1:
            self.running_average = val + abs(self.momentum) * avg
        elif self.update_mode == UpdateMode.ALT2:
            _ensure_blend_modes()
            blend = BLEND_MODES.get(self.update_blend_mode)
            if blend is None:
                available_modes = ", ".join(BLEND_MODES.keys())
                raise ValueError(f"Unknown blend mode: '{self.update_blend_mode}'")
            result = blend(val, avg.neg() if self.momentum < 0 else avg, abs(self.momentum))
            self.running_average = blend(val, avg, abs(self.momentum))
        else:  # UpdateMode.DEFAULT
            self.running_average = result
        
        return result

    def reset(self):
        """Reset momentum state."""
        self.running_average = 0.0
        self.is_mps = None 

    def project(self, v0_orig, v1_orig):
        """
        Project v0 onto v1 and return parallel and orthogonal components.
        
        Args:
            v0_orig (Tensor): Vector to project
            v1_orig (Tensor): Vector to project onto
            
        Returns:
            Tuple[Tensor, Tensor]: (Parallel Component, Orthogonal Component)
        """
        # Optimization: Lazy cache device check
        if self.is_mps is None:
            self.is_mps = v0_orig.device.type == "mps"
            if self.is_mps:
                logging.warning("[APGGuider] MPS device detected. Performance may be impacted.")

        # Move to CPU and convert to double for MPS
        if self.is_mps:
            v0, v1 = v0_orig.cpu().double(), v1_orig.cpu().double()
        else:
            v0, v1 = v0_orig.double(), v1_orig.double()
        
        # Normalize v1 along specified dimensions
        # Added eps=1e-6 to prevent FP16 underflow issues
        v1 = F.normalize(v1, dim=self.dims, eps=1e-6)
        
        # Calculate parallel component
        v0_p = (v0 * v1).sum(dim=self.dims, keepdim=True) * v1
        
        # Calculate orthogonal component
        v0_o = v0 - v0_p
        
        # Convert back to original dtype and device
        return (
            v0_p.to(dtype=v0_orig.dtype, device=v0_orig.device),
            v0_o.to(dtype=v0_orig.dtype, device=v0_orig.device)
        )

    def apg(self, cond, uncond):
        """
        Calculate APG guidance vector using orthogonal projection.
        
        Args:
            cond (Tensor): Conditional prediction
            uncond (Tensor): Unconditional prediction
            
        Returns:
            Tensor: Orthogonal guidance vector
        """
        pred_diff = self.update(cond - uncond)
        
        # Apply norm threshold if specified
        if self.norm_threshold is not None and self.norm_threshold > 0:
            diff_norm = pred_diff.norm(p=2, dim=self.dims, keepdim=True)
            # Prevent division by zero
            diff_norm = torch.clamp(diff_norm, min=1e-6) 
            scale_factor = torch.minimum(torch.ones_like(diff_norm), self.norm_threshold / diff_norm)
            pred_diff = pred_diff * scale_factor
        
        # Project into parallel and orthogonal components
        diff_p, diff_o = self.project(pred_diff, cond)
        
        # Return orthogonal component
        return diff_o

    def cfg_function(self, args):
        """Custom CFG function with APG guidance (Post-CFG mode)."""
        cond, uncond = (
            (args["cond_denoised"], args["uncond_denoised"]) 
            if self.predict_image 
            else (args["cond"], args["uncond"])
        )
        
        # Division by zero protection
        if abs(self.apg_scale - 1.0) < 1e-8:
            result = cond
        else:
            # Apply APG guidance with scaling
            result = cond + (self.apg_scale - 1.0) * self.apg(cond, uncond)
        
        # Return based on prediction type
        return args["input"] - result if self.predict_image else result

    def pre_cfg_function(self, args):
        """Pre-CFG mode: modify conditioning before CFG application."""
        conds_out = args["conds_out"]
        if len(conds_out) < 2:
            return conds_out
        
        cond, uncond = conds_out[:2]
        update = self.apg(cond, uncond)
        
        # Division by zero protection
        if abs(self.apg_scale) < 1e-6:
            cond_apg = cond
        else:
            cond_apg = uncond + update + (cond - uncond) / self.apg_scale
        
        return [cond_apg, *conds_out[1:]]


class APGGuider(comfy.samplers.CFGGuider):
    """Custom CFG guider with APG rule-based guidance."""
    
    def __init__(self, model, *, positive, negative, rules, params):
        """
        Initialize Guider.
        
        Args:
            model: ComfyUI Model
            positive: Positive Cond
            negative: Negative Cond
            rules: Tuple of APGConfig rules
            params: Dict of extra params (verbose, etc)
        """
        super().__init__(model)
        self.set_conds(positive, negative)
        self.set_cfg(1.0)
        self.apg_rules = tuple(APG(rule_config) for rule_config in rules)
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
        """Reset all APG momentum states except the specified one."""
        for apg_rule in self.apg_rules:
            if apg_rule is not exclude:
                apg_rule.reset()

    def apg_get_match(self, sigma):
        """Find the appropriate APG rule for the current sigma value."""
        for rule in self.apg_rules:
            if sigma <= rule.start_sigma:
                return rule
        
        sigma_list = [f"{r.start_sigma:.2f}" for r in self.apg_rules]
        raise RuntimeError(f"No APG rule matched for sigma={sigma:.4f}.")

    def outer_sample(self, *args, **kwargs):
        """Wrap sampling with APG state reset."""
        self.apg_reset()
        result = super().outer_sample(*args, **kwargs)
        self.apg_reset()
        return result

    def predict_noise(self, x, timestep, model_options=None, seed=None, **kwargs):
        """Predict noise with APG guidance for the current timestep."""
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


# =================================================================================
# == Core Node Class                                                             ==
# =================================================================================

class APGGuiderNode:
    """
    ComfyUI node for APG (Adaptive Projected Gradient) guidance.
    """

    @classmethod
    def INPUT_TYPES(cls):
        """Define node input parameters."""
        _ensure_blend_modes()
        return {
            "required": {
                "model": ("MODEL", {
                    "tooltip": "MODEL INPUT\n• Purpose: The diffusion model to apply APG guidance to.\n• Recommendation: Connect from a standard model loader."
                }),
                "positive": ("CONDITIONING", {
                    "tooltip": "POSITIVE COND\n• Purpose: Your main prompt conditioning.\n• Trade-off: Stronger conditioning needs better APG tuning."
                }),
                "negative": ("CONDITIONING", {
                    "tooltip": "NEGATIVE COND\n• Purpose: Negative prompt conditioning.\n• Trade-off: APG uses this heavily for orthogonal projection."
                }),
                "disable_apg": ("BOOLEAN", {
                    "default": False, 
                    "label_on": "APG Disabled", 
                    "label_off": "APG Enabled",
                    "tooltip": "DISABLE APG\n• Purpose: Bypass APG entirely for A/B testing.\n• Effect: Reverts to standard CFG behavior."
                }),
                "verbose_debug": ("BOOLEAN", {
                    "default": False, 
                    "label_on": "Verbose On", 
                    "label_off": "Verbose Off",
                    "tooltip": "VERBOSE DEBUG\n• Purpose: Print step-by-step rule matching to console.\n• Recommendation: Enable when tuning YAML rules."
                }),
                "apg_scale": ("FLOAT", {
                    "default": 4.5, "min": 0.0, "max": 1000.0, "step": 0.1,
                    "tooltip": "APG SCALE\n• Purpose: Strength of orthogonal correction.\n• Range: 0.0=Off, 3-6=Typical, 10+=Aggressive.\n• Trade-off: Too high = artifacts; Too low = standard CFG look."
                }),
                "cfg_before": ("FLOAT", {
                    "default": 4.0, "min": 1.0, "max": 1000.0, "step": 0.1,
                    "tooltip": "CFG (BEFORE)\n• Purpose: Static CFG applied *during* APG phase.\n• Recommendation: 4.0-6.0 is a good baseline."
                }),
                "cfg_after": ("FLOAT", {
                    "default": 3.0, "min": 1.0, "max": 1000.0, "step": 0.1,
                    "tooltip": "CFG (AFTER)\n• Purpose: Static CFG applied *after* APG deactivates (end_sigma).\n• Recommendation: Lower values (3.0) help refine details."
                }),
                "eta": ("FLOAT", {
                    "default": 0.0, "min": -1000.0, "max": 1000.0, "step": 0.1,
                    "tooltip": "ETA (DEPRECATED)\n• Purpose: Legacy parameter, no longer used.\n• Recommendation: Leave at 0.0."
                }),
                "norm_threshold": ("FLOAT", {
                    "default": 2.5, "min": 0.0, "max": 1000.0, "step": 0.1,
                    "tooltip": "NORM THRESHOLD\n• Purpose: Cap the guidance vector magnitude.\n• Trade-off: Prevents 'exploding' gradients but limits strength.\n• Recommendation: 2.5 - 5.0."
                }),
                "momentum": ("FLOAT", {
                    "default": 0.75, "min": -1000.0, "max": 1000.0, "step": 0.01,
                    "tooltip": "MOMENTUM\n• Purpose: Smooths guidance across steps.\n• Trade-off: Higher = smoother but laggy; Lower = reactive but jittery."
                }),
                "start_sigma": ("FLOAT", {
                    "default": -1.0, "min": -1.0, "max": 10000.0, "step": 0.01,
                    "tooltip": "START SIGMA\n• Purpose: Noise level where APG activates.\n• Note: -1.0 = Infinity (Start immediately)."
                }),
                "end_sigma": ("FLOAT", {
                    "default": -1.0, "min": -1.0, "max": 10000.0, "step": 0.01,
                    "tooltip": "END SIGMA\n• Purpose: Noise level where APG deactivates.\n• Note: -1.0 = Never disable."
                }),
                "dims": ("STRING", {
                    "default": "-1, -2",
                    "tooltip": "DIMS\n• Purpose: Dimensions for normalization.\n• Recommendation: '-1, -2' targets spatial dimensions (H, W)."
                }),
                "predict_image": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "PREDICT IMAGE\n• Purpose: Match your model's prediction type.\n• True = v-prediction (common for audio/video).\n• False = epsilon (common for SD1.5)."
                }),
                "mode": (
                    ("pure_apg", "pre_cfg", "pure_alt1", "pre_alt1", "pure_alt2", "pre_alt2"),
                    {"tooltip": "APG MODE\n• Purpose: Algorithm variant.\n• 'pure_*': Standard Post-CFG.\n• 'pre_*': Pre-CFG injection.\n• 'alt*': Alternative momentum math."}
                ),
            },
            "optional": {
                "yaml_parameters_opt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "dynamic_prompt": False,
                    "tooltip": "YAML OVERRIDE\n• Purpose: Define complex multi-stage schedules.\n• Format: List of rules sorted by sigma.\n• See README for examples."
                }),
            },
        }

    RETURN_TYPES = ("GUIDER",)
    RETURN_NAMES = ("apg_guider",)
    FUNCTION = "go"
    CATEGORY = "MD_Nodes/Sampling"

    @classmethod
    def _build_rules_from_inputs(cls, cfg_before, cfg_after, start_sigma, end_sigma, **kwargs):
        """Build rule list from simple node inputs."""
        rules = []
        
        main_rule_params = {
            "cfg": cfg_before,
            "apg_blend": 1.0,
            "start_sigma": start_sigma if start_sigma >= 0 else math.inf,
            **kwargs
        }
        rules.append(APGConfig.build(**main_rule_params))

        if end_sigma > 0:
            rules.append(APGConfig.build(
                cfg=cfg_after,
                start_sigma=end_sigma,
                apg_blend=0.0,
            ))
        
        return rules

    @classmethod
    def go(cls, *, model, positive, negative, disable_apg, verbose_debug, apg_scale, 
           cfg_before, cfg_after, eta, norm_threshold, momentum, start_sigma, 
           end_sigma, dims, predict_image, mode, yaml_parameters_opt=None):
        """
        Execute node to create APG guider.
        
        Returns:
            Tuple[APGGuider]: The configured guider instance
        """
        try:
            if yaml_parameters_opt is None:
                yaml_parameters_opt = ""
            yaml_parameters_opt = yaml_parameters_opt.strip()
            
            params = {"verbose": verbose_debug}
            rules = ()

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
                rules = (APGConfig.build(cfg=cfg_after, start_sigma=math.inf, apg_blend=0.0),)
                if verbose_debug:
                    logging.info("[APGGuider] APG is disabled. Using standard CFG guidance only.")
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
                rules = tuple(APGConfig.build(**rule) for rule in rules)

            if not disable_apg:
                rules = tuple(sorted(rules, key=lambda r: r.start_sigma))
                if not rules or rules[-1].start_sigma < math.inf:
                    fallback_rule = APGConfig.build(
                        cfg=cfg_after, 
                        start_sigma=math.inf, 
                        apg_blend=0.0
                    )
                    rules = (*rules, fallback_rule)
                    if verbose_debug:
                        logging.info("[APGGuider] Added automatic fallback rule with infinite start_sigma.")
            
            guider = APGGuider(
                model,
                positive=positive,
                negative=negative,
                rules=rules,
                params=params,
            )
            return (guider,)

        except Exception as e:
            logging.error(f"[APGGuider] Failed to create guider: {e}")
            logging.debug(traceback.format_exc())
            print(f"[APGGuider] ⚠️ Error: {e}. Falling back to standard CFG.")
            
            fallback_cfg = cfg_after if not disable_apg else 1.0
            guider = comfy.samplers.CFGGuider(model)
            guider.set_conds(positive, negative)
            guider.set_cfg(fallback_cfg)
            return (guider,)

# =================================================================================
# == Node Registration                                                            ==
# =================================================================================

NODE_CLASS_MAPPINGS = {
    "APGGuiderForked": APGGuiderNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "APGGuiderForked": "MD: APG Guider",
}