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

# ░▒▓ FEATURES:
#   ✓ Schedules APG scale, CFG, and momentum using sigma-based rules.
#   ✓ YAML-based configuration for complex, multi-stage guidance.
#   ✓ Enhanced robustness: division-by-zero protection, input validation.
#   ✓ Verbose debugging logs (via logging module) to see rule activation.

# ░▒▓ CHANGELOG:
#   - v1.4.2 (Guideline Update - Oct 2025):
#       • REFACTOR: Full compliance update to v1.4.2 guidelines.
#       • CRITICAL: Removed all type hints from function signatures (Section 6.2).
#       • REFACTOR: Replaced `NamedTuple` with a standard helper class (Section 5.2).
#       • STYLE: Standardized imports, docstrings, and error handling.
#       • STYLE: Replaced `tqdm.write` and `warnings.warn` with `logging` (Section 6.3).
#       • STYLE: Rewrote all tooltips to new standard format (Section 8.1).
#       • ROBUST: Added graceful failure fallback in main `go` function (Section 7.3).
#       • STYLE: Updated category to 'MD_Nodes/Sampling' and display name.
#   - v0.3.0 (Robustness Update):
#       • FIXED: Division-by-zero, float comparison edge cases, input validation.
#       • ADDED: MPS/eta warnings, improved error messages, blend mode fallbacks.

# ░▒▓ CONFIGURATION:
#   → Primary Use: Using YAML to apply strong APG at high sigmas and low/zero APG at low sigmas.
#   → Secondary Use: Disabling APG (`apg_scale: 0.0`) to use as a simple CFG/momentum scheduler.
#   → Edge Use: Fine-tuning momentum against APG scale for complex guidance effects.

# ░▒▓ WARNING:
#   This node may trigger:
#   ▓▒░ A 4-hour YAML editing session just to change one sigma value.
#   ▓▒░ The unshakable, god-like feeling of *finally* understanding orthogonal projection.
#   ▓▒░ Compulsive checking of `verbose: true` logs, whispering "just as I planned."
#   ▓▒░ Flashbacks to manually patching Z-buffer routines in a 4k intro.
#   Consult your nearest demoscene vet if hallucinations persist.
# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄


# =================================================================================
# == Standard Library Imports                                                    ==
# =================================================================================
import enum
import logging
import math
import traceback

# =================================================================================
# == Third-Party Imports                                                         ==
# =================================================================================
import torch
import torch.nn.functional as F
import yaml

# =================================================================================
# == ComfyUI Core Modules                                                        ==
# =================================================================================
import comfy.samplers
import nodes

# =================================================================================
# == Local Project Imports                                                       ==
# =================================================================================
# (No local project imports in this file)

# =================================================================================
# == Helper Classes & Dependencies                                               ==
# =================================================================================

# Version for compatibility checks
__version__ = "0.3.0"

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
        a (torch.Tensor): Start tensor.
        b (torch.Tensor): End tensor.
        t (float): Interpolation factor.
        
    Returns:
        torch.Tensor: Interpolated tensor.
    """
    omega = torch.acos((a * b).sum() / (a.norm() * b.norm()))
    so = torch.sin(omega)
    if so == 0:
        return torch.lerp(a, b, t) # Fallback to lerp if inputs are collinear
    return (torch.sin((1.0 - t) * omega) / so) * a + (torch.sin(t * omega) / so) * b

def _validate_dims(dims, tensor_ndim=4):
    """
    Validate dimension tuple for tensor operations.
    
    Args:
        dims (tuple): Tuple of dimension indices.
        tensor_ndim (int): The number of dimensions of the tensor (default 4).
        
    Returns:
        tuple: The validated dimension tuple.
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
        Initialize a new APG configuration rule.
        
        Args:
            start_sigma (float): The sigma level at which this rule becomes active (matches if current_sigma <= start_sigma).
            momentum (float): Momentum factor for the running average.
            eta (float): Deprecated, no effect.
            apg_scale (float): The strength of the orthogonal guidance.
            norm_threshold (float): Maximum norm for the guidance vector.
            dims (tuple): Dimensions for normalization/projection.
            update_mode (UpdateMode): The momentum update algorithm to use.
            update_blend_mode (str): Blend mode for ALT2 momentum.
            cfg (float): The standard CFG scale to apply when this rule is active.
            apg_blend (float): A factor to blend APG (0.0 = off, 1.0 = on).
            apg_blend_mode (str): Blend mode for APG (not currently used, but kept for compatibility).
            predict_image (bool): True if the model predicts the denoised image, False if it predicts noise.
            pre_cfg_mode (bool): True to apply APG *before* CFG, False to apply *after*.
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

        # Add _fields property for compatibility with original code's __str__
        self._fields = [
            "start_sigma", "momentum", "eta", "apg_scale", "norm_threshold",
            "dims", "update_mode", "update_blend_mode", "cfg", "apg_blend",
            "apg_blend_mode", "predict_image", "pre_cfg_mode"
        ]

    @staticmethod
    def fixup_param(k, v):
        """
        Process and validate configuration parameters.
        
        Args:
            k (str): The parameter key.
            v (any): The parameter value.
            
        Returns:
            any: The processed and validated parameter value.
        """
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
                valid_modes = ", ".join(UpdateMode.__members__.keys())
                logging.warning(f"[APGGuider] Invalid UpdateMode '{v}'. Valid options: {valid_modes}. Using DEFAULT.")
                return UpdateMode.DEFAULT
            return mode_enum
        
        if k == "start_sigma":
            return math.inf if float(v) < 0 else float(v)
        
        if k == "norm_threshold":
            threshold = float(v) if v is not None else 2.5
            if threshold < 0:
                logging.warning(f"[APGGuider] norm_threshold cannot be negative ({threshold}). Clamping to 0.")
                return 0.0
            return threshold
        
        if k == "apg_scale":
            scale = float(v)
            if abs(scale) < 1e-6:  # Effectively zero
                return 0.0
            return scale
        
        if k == "eta":
            eta_val = float(v)
            if abs(eta_val) > 1e-6:
                logging.warning(
                    f"[APGGuider] 'eta' parameter (value: {eta_val}) is deprecated and not used in APG projection. "
                    "It's kept for config compatibility but has no effect. Consider removing it from your config."
                )
            return eta_val
        
        return v

    @classmethod
    def build(cls, *, mode="pure_apg", **params):
        """
        Build an APGConfig from mode string and parameters.
        
        Args:
            mode (str): The mode string (e.g., "pure_apg", "pre_alt1").
            **params (dict): Dictionary of parameters.
            
        Returns:
            APGConfig: A new instance of APGConfig.
        """
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
        """
        Human-readable string representation.
        
        Returns:
            str: A string summary of the config.
        """
        if self.apg_blend == 0 or self.apg_scale == 0:
            fields = ("start_sigma", "cfg")
        else:
            fields = self._fields
        pretty_fields = ", ".join(f"{k}={getattr(self, k)}" for k in fields)
        return f"APGConfig({pretty_fields})"


class APG:
    """
    Core APG logic with momentum and orthogonal projection.
    
    Attributes:
        config (APGConfig): The configuration for this specific rule.
        running_average (float or torch.Tensor): The momentum state.
    """
    
    def __init__(self, config):
        """
        Initialize the APG logic container.
        
        Args:
            config (APGConfig): The configuration object for this rule.
        """
        self.config = config
        self.running_average = 0.0
        self._warned_mps = False

    def __getattr__(self, k):
        """Delegate attribute access to config."""
        return getattr(self.config, k)

    def update(self, val):
        """
        Update running average with momentum.
        
        Args:
            val (torch.Tensor): The new value to incorporate.
        
        Returns:
            torch.Tensor: The updated value after applying momentum.
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
                raise ValueError(
                    f"Unknown blend mode: '{self.update_blend_mode}'. "
                    f"Available modes: {available_modes}"
                )
            result = blend(val, avg.neg() if self.momentum < 0 else avg, abs(self.momentum))
            self.running_average = blend(val, avg, abs(self.momentum))
        else:  # UpdateMode.DEFAULT
            self.running_average = result
        
        return result

    def reset(self):
        """Reset momentum state."""
        self.running_average = 0.0

    def project(self, v0_orig, v1_orig):
        """
        Project v0 onto v1 and return parallel and orthogonal components.
        
        Returns:
            (v0_parallel, v0_orthogonal) where v0 = v0_parallel + v0_orthogonal
        """
        # MPS performance warning
        if v0_orig.device.type == "mps" and not self._warned_mps:
            logging.warning(
                "[APGGuider] MPS device detected. Moving tensors to CPU for projection. "
                "This may impact performance. Consider using CUDA if available."
            )
            self._warned_mps = True
        
        # Move to CPU and convert to double for MPS
        if v0_orig.device.type == "mps":
            v0, v1 = v0_orig.cpu().double(), v1_orig.cpu().double()
        else:
            v0, v1 = v0_orig.double(), v1_orig.double()
        
        # Normalize v1 along specified dimensions
        v1 = F.normalize(v1, dim=self.dims)
        
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
            cond (torch.Tensor): Conditional prediction
            uncond (torch.Tensor): Unconditional prediction
            
        Returns:
            torch.Tensor: Orthogonal guidance correction vector
        """
        pred_diff = self.update(cond - uncond)
        
        # Apply norm threshold if specified
        if self.norm_threshold is not None and self.norm_threshold > 0:
            diff_norm = pred_diff.norm(p=2, dim=self.dims, keepdim=True)
            # Prevent division by zero
            diff_norm = torch.clamp(diff_norm, min=1e-8)
            scale_factor = torch.minimum(
                torch.ones_like(diff_norm), 
                self.norm_threshold / diff_norm
            )
            pred_diff = pred_diff * scale_factor
        
        # Project into parallel and orthogonal components
        diff_p, diff_o = self.project(pred_diff, cond)
        
        # Return orthogonal component
        return diff_o

    def cfg_function(self, args):
        """
        Custom CFG function with APG guidance (Post-CFG mode).
        
        Args:
            args (dict): Dictionary containing cond, uncond, input, cond_denoised, uncond_denoised.
            
        Returns:
            torch.Tensor: Guided prediction.
        """
        # Select appropriate tensors based on prediction mode
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
        """
        Pre-CFG mode: modify conditioning before CFG application.
        
        Args:
            args (dict): Dictionary containing conds_out list.
            
        Returns:
            list: Modified conditioning list.
        """
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
        Initialize the APGGuider.
        
        Args:
            model: The ComfyUI model object.
            positive: The positive conditioning.
            negative: The negative conditioning.
            rules (tuple): A tuple of APGConfig objects.
            params (dict): Additional parameters (e.g., "verbose").
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
        """
        Reset all APG momentum states except the specified one.
        
        Args:
            exclude (APG, optional): The one APG rule *not* to reset.
        """
        for apg_rule in self.apg_rules:
            if apg_rule is not exclude:
                apg_rule.reset()

    def apg_get_match(self, sigma):
        """
        Find the appropriate APG rule for the current sigma value.
        
        Args:
            sigma (float): Current noise level.
            
        Returns:
            APG: Matching APG rule object.
        """
        for rule in self.apg_rules:
            if sigma <= rule.start_sigma:
                return rule
        
        # This should never happen if rules are properly configured
        sigma_list = [f"{r.start_sigma:.2f}" for r in self.apg_rules]
        raise RuntimeError(
            f"No APG rule matched for sigma={sigma:.4f}. "
            f"Available rules start at: {', '.join(sigma_list)}. "
            f"Ensure you have a fallback rule with start_sigma=-1 (infinity). "
            f"This error indicates a configuration issue."
        )

    def outer_sample(self, *args, **kwargs):
        """Wrap sampling with APG state reset."""
        self.apg_reset()
        result = super().outer_sample(*args, **kwargs)
        self.apg_reset()
        return result

    def predict_noise(self, x, timestep, model_options=None, seed=None, **kwargs):
        """
        Predict noise with APG guidance for the current timestep.
        
        Args:
            x (torch.Tensor): Latent tensor.
            timestep (torch.Tensor or float): Current timestep/sigma.
            model_options (dict, optional): Model configuration options.
            seed (int, optional): Random seed.
            **kwargs: Additional arguments.
            
        Returns:
            torch.Tensor: Predicted noise.
        """
        if model_options is None:
            model_options = {}
        
        # Extract sigma from timestep
        sigma = (
            timestep.max().detach().cpu().item() 
            if isinstance(timestep, torch.Tensor) 
            else float(timestep)
        )
        
        # Get appropriate rule for current sigma
        rule = self.apg_get_match(sigma)
        
        # Reset other rules to prevent state leakage
        self.apg_reset(exclude=rule)
        
        # Check if APG is active
        matched = rule.apg_blend != 0 and rule.apg_scale != 0
        
        if self.apg_verbose:
            apg_status = "ACTIVE" if matched else "BYPASSED"
            logging.info(
                f"[APGGuider] σ={sigma:.4f} → Rule: CFG={rule.cfg:.2f}, "
                f"APG_scale={rule.apg_scale:.2f} [{apg_status}]"
            )
        
        if matched:
            # Disable ComfyUI's CFG optimization when using APG
            model_options = model_options | {"disable_cfg1_optimization": True}
            
            if rule.pre_cfg_mode:
                # Pre-CFG mode: modify conditioning before CFG
                pre_cfg_handlers = model_options.get("sampler_pre_cfg_function", []).copy()
                pre_cfg_handlers.append(rule.pre_cfg_function)
                model_options["sampler_pre_cfg_function"] = pre_cfg_handlers
                cfg = rule.apg_scale
            else:
                # Post-CFG mode: apply custom CFG function
                model_options["sampler_cfg_function"] = rule.cfg_function
                cfg = rule.cfg
        else:
            # APG disabled, use standard CFG
            cfg = rule.cfg
        
        # Apply CFG and call parent predict_noise
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
    
    Enhanced fork with improved robustness, validation, and error handling.
    """

    @classmethod
    def INPUT_TYPES(cls):
        """Define node input parameters."""
        _ensure_blend_modes()
        return {
            "required": {
                "model": ("MODEL", {
                    "tooltip": (
                        "MODEL INPUT\n"
                        "- The diffusion model to apply APG guidance to.\n"
                        "- Connects from a model loader."
                    )
                }),
                "positive": ("CONDITIONING", {
                    "tooltip": (
                        "POSITIVE CONDITIONING\n"
                        "- The positive conditioning (prompt).\n"
                        "- Connects from a text encoder."
                    )
                }),
                "negative": ("CONDITIONING", {
                    "tooltip": (
                        "NEGATIVE CONDITIONING\n"
                        "- The negative conditioning (negative prompt).\n"
                        "- Connects from a text encoder."
                    )
                }),
                "disable_apg": ("BOOLEAN", {
                    "default": False, 
                    "label_on": "APG Disabled", 
                    "label_off": "APG Enabled",
                    "tooltip": (
                        "DISABLE APG\n"
                        "- If True, completely disables APG guidance.\n"
                        "- The node will only apply the scheduled 'cfg_after' value.\n"
                        "- Use this to quickly A/B test."
                    )
                }),
                "verbose_debug": ("BOOLEAN", {
                    "default": False, 
                    "label_on": "Verbose On", 
                    "label_off": "Verbose Off",
                    "tooltip": (
                        "VERBOSE DEBUG\n"
                        "- If True, prints detailed step-by-step guidance info to the console.\n"
                        "- Useful for debugging YAML rules."
                    )
                }),
                "apg_scale": ("FLOAT", {
                    "default": 4.5, 
                    "min": 0.0, 
                    "max": 1000.0, 
                    "step": 0.1,
                    "tooltip": (
                        "APG SCALE\n"
                        "- APG guidance strength. Higher = more aggressive orthogonal correction.\n"
                        "- 0.0 = disabled.\n"
                        "- Recommended: 2.0 - 8.0"
                    )
                }),
                "cfg_before": ("FLOAT", {
                    "default": 4.0, 
                    "min": 1.0, 
                    "max": 1000.0, 
                    "step": 0.1,
                    "tooltip": (
                        "CFG (BEFORE END)\n"
                        "- The standard CFG scale to use *while* APG is active.\n"
                        "- This applies *before* the 'end_sigma' threshold is met."
                    )
                }),
                "cfg_after": ("FLOAT", {
                    "default": 3.0, 
                    "min": 1.0, 
                    "max": 1000.0, 
                    "step": 0.1,
                    "tooltip": (
                        "CFG (AFTER END)\n"
                        "- The standard CFG scale to use *after* APG is disabled.\n"
                        "- This applies *after* the 'end_sigma' threshold is met."
                    )
                }),
                "eta": ("FLOAT", {
                    "default": 0.0, 
                    "min": -1000.0, 
                    "max": 1000.0, 
                    "step": 0.1,
                    "tooltip": (
                        "ETA (DEPRECATED)\n"
                        "- This parameter is deprecated and has NO effect.\n"
                        "- It is kept only for workflow compatibility.\n"
                        "- A warning will be logged if set to a non-zero value."
                    )
                }),
                "norm_threshold": ("FLOAT", {
                    "default": 2.5, 
                    "min": 0.0, 
                    "max": 1000.0, 
                    "step": 0.1,
                    "tooltip": (
                        "NORM THRESHOLD\n"
                        "- Maximum norm (length) for the guidance vector.\n"
                        "- Prevents extreme, exploding corrections.\n"
                        "- 0.0 = no limit."
                    )
                }),
                "momentum": ("FLOAT", {
                    "default": 0.75, 
                    "min": -1000.0, 
                    "max": 1000.0, 
                    "step": 0.01,
                    "tooltip": (
                        "MOMENTUM\n"
                        "- Momentum for the running average of guidance vectors.\n"
                        "- Smooths guidance over time.\n"
                        "- Negative values reverse momentum direction."
                    )
                }),
                "start_sigma": ("FLOAT", {
                    "default": -1.0, 
                    "min": -1.0, 
                    "max": 10000.0, 
                    "step": 0.01,
                    "tooltip": (
                        "START SIGMA\n"
                        "- Sigma (noise level) at which APG *activates*.\n"
                        "- -1.0 = infinity (active from the very beginning).\n"
                        "- Higher sigma = earlier in the generation."
                    )
                }),
                "end_sigma": ("FLOAT", {
                    "default": -1.0, 
                    "min": -1.0, 
                    "max": 10000.0, 
                    "step": 0.01,
                    "tooltip": (
                        "END SIGMA\n"
                        "- Sigma (noise level) at which APG *deactivates*.\n"
                        "- -1.0 = never disable.\n"
                        "- Use this to disable APG for final refinement steps (e.g., set to 1.0)."
                    )
                }),
                "dims": ("STRING", {
                    "default": "-1, -2",
                    "tooltip": (
                        "DIMS\n"
                        "- Dimensions for normalization/projection.\n"
                        "- Default '-1, -2' targets the spatial (height, width) dimensions.\n"
                        "- Format: comma-separated integers."
                    )
                }),
                "predict_image": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "PREDICT IMAGE\n"
                        "- True = model predicts the final denoised image (e.g., v-prediction).\n"
                        "- False = model predicts the noise (e.g., epsilon-prediction).\n"
                        "- Must match your model's prediction type."
                    )
                }),
                "mode": (
                    ("pure_apg", "pre_cfg", "pure_alt1", "pre_alt1", "pure_alt2", "pre_alt2"),
                    {
                        "tooltip": (
                            "APG MODE\n"
                            "- 'pure_*': APG applied *after* standard CFG.\n"
                            "- 'pre_*': APG applied *before* standard CFG.\n"
                            "- '*_apg': Default momentum update.\n"
                            "- '*_alt1'/'*_alt2': Alternative momentum algorithms."
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
                        "- Advanced: Override all settings with a YAML config.\n\n"
                        "RULES APPLY IN ORDER (sorted by start_sigma, highest first):\n"
                        "- Higher start_sigma rules match first.\n"
                        "- Use start_sigma: -1 for infinity (fallback).\n\n"
                        "EXAMPLE:\n"
                        "rules:\n"
                        "- start_sigma: 14.0  # High noise phase\n"
                        "  apg_scale: 5.0\n"
                        "  cfg: 4.0\n"
                        "- start_sigma: 4.0   # Medium noise\n"
                        "  apg_scale: 3.0\n"
                        "  cfg: 3.5\n"
                        "- start_sigma: -1    # Fallback (infinity)\n"
                        "  apg_scale: 0.0     # Disable APG\n"
                        "  cfg: 3.0\n\n"
                        "Set 'verbose: true' at top level for debug output."
                    )
                }),
            },
        }

    RETURN_TYPES = ("GUIDER",)
    RETURN_NAMES = ("apg_guider",)
    FUNCTION = "go"
    CATEGORY = "MD_Nodes/Sampling" # Changed from Guiders to match guide structure

    @classmethod
    def _build_rules_from_inputs(
        cls, 
        cfg_before, 
        cfg_after, 
        start_sigma, 
        end_sigma, 
        **kwargs
    ):
        """
        Build rule list from simple node inputs.
        
        Args:
            (various): Parameters from the node's simple inputs.
            
        Returns:
            list: A list of APGConfig objects.
        """
        rules = []
        
        # Main APG rule
        main_rule_params = {
            "cfg": cfg_before,
            "apg_blend": 1.0,
            "start_sigma": start_sigma if start_sigma >= 0 else math.inf,
            **kwargs
        }
        rules.append(APGConfig.build(**main_rule_params))

        # End rule if specified
        if end_sigma > 0:
            rules.append(APGConfig.build(
                cfg=cfg_after,
                start_sigma=end_sigma,  # Use exact value, matching uses <=
                apg_blend=0.0,
            ))
        
        return rules

    @classmethod
    def go(
        cls, *,
        model, 
        positive, 
        negative, 
        disable_apg, 
        verbose_debug, 
        apg_scale, 
        cfg_before, 
        cfg_after, 
        eta,
        norm_threshold, 
        momentum, 
        start_sigma, 
        end_sigma, 
        dims, 
        predict_image,
        mode, 
        yaml_parameters_opt=None
    ):
        """
        Execute node to create APG guider.
        
        Args:
            (All args from INPUT_TYPES)
        
        Returns:
            Tuple containing (apg_guider,)
        """
        try:
            if yaml_parameters_opt is None:
                yaml_parameters_opt = ""
            yaml_parameters_opt = yaml_parameters_opt.strip()
            
            params = {"verbose": verbose_debug}
            rules = ()

            # Parse YAML if provided
            if yaml_parameters_opt:
                try:
                    loaded_params = yaml.safe_load(yaml_parameters_opt)
                    if isinstance(loaded_params, dict):
                        params.update(loaded_params)
                    elif loaded_params:
                        params["rules"] = tuple(loaded_params)
                    else:
                        raise TypeError(
                            "Invalid YAML format. Expected a dictionary or list of rules. "
                            "See tooltip for examples."
                        )
                except yaml.YAMLError as e:
                    raise ValueError(f"Error parsing YAML parameters: {e}")

            rules = tuple(params.pop("rules", ()))

            # Build rules based on configuration
            if disable_apg:
                rules = (APGConfig.build(cfg=cfg_after, start_sigma=math.inf, apg_blend=0.0),)
                if verbose_debug:
                    logging.info("[APGGuider] APG is disabled. Using standard CFG guidance only.")
            elif not rules:
                # Build from simple inputs
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
                # Build from YAML rules
                rules = tuple(APGConfig.build(**rule) for rule in rules)

            # Sort rules by start_sigma (ascending) and ensure fallback
            if not disable_apg:
                rules = tuple(sorted(rules, key=lambda r: r.start_sigma))
                
                # Ensure there's an infinity fallback rule
                if not rules or rules[-1].start_sigma < math.inf:
                    fallback_rule = APGConfig.build(
                        cfg=cfg_after, 
                        start_sigma=math.inf, 
                        apg_blend=0.0
                    )
                    rules = (*rules, fallback_rule)
                    if verbose_debug:
                        logging.info("[APGGuider] Added automatic fallback rule with infinite start_sigma.")
            
            # Create and return guider
            guider = APGGuider(
                model,
                positive=positive,
                negative=negative,
                rules=rules,
                params=params,
            )
            # Must return a tuple
            return (guider,)

        except Exception as e:
            logging.error(f"[APGGuider] Failed to create guider: {e}")
            logging.debug(traceback.format_exc())
            print(f"[APGGuider] ⚠️ Error: {e}. Falling back to standard CFG. Check console for details.")
            
            # Graceful fallback (Section 7.3)
            # Return a standard CFGGuider as a safe, valid output
            fallback_cfg = cfg_after if not disable_apg else 1.0
            guider = comfy.samplers.CFGGuider(model)
            guider.set_conds(positive, negative)
            guider.set_cfg(fallback_cfg)
            return (guider,)


# =================================================================================
# == Node Registration                                                           ==
# =================================================================================

NODE_CLASS_MAPPINGS = {
    "APGGuiderForked": APGGuiderNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "APGGuiderForked": "MD: APG Guider",
}