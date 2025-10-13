# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# ████ APG GUIDER (FORKED) v0.3.0 – Enhanced & Bulletproof ████▓▒░
# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀

# ░▒▓ ORIGIN & DEV:
#    • Original Sorcerer: Blepping — github.com/blepping
#    • Forged in the fires of: MDMAchine & devstral
#    • Enhanced by: Code review & systematic improvement
#    • License: Apache 2.0

# ░▒▓ DESCRIPTION:
#    A powerful fork of Blepping's APG Guider with enhanced robustness,
#    better error handling, and improved user experience.
#    APG (Adaptive Projected Gradient) provides surgical precision control
#    over latent space evolution during sampling.

# ░▒▓ KEY CONCEPTS:
#    • CFG (Classifier-Free Guidance): Controls how much the model follows your prompt
#    • APG Scale: Controls the strength of orthogonal guidance correction
#    • Relationship: APG acts as a "steering correction" on top of CFG
#      - Higher APG scale = More aggressive correction of generation direction
#      - APG works by projecting guidance into orthogonal space relative to conditional
#    • Rules apply in descending sigma order (noise level decreases during sampling)

# ░▒▓ CHANGELOG:
#    - v0.3.0 (Enhancement & Robustness Update):
#        • Added eta parameter deprecation warnings
#        • Improved type hints throughout
#        • Added MPS device performance warning
#        • Fixed division-by-zero protection for apg_scale
#        • Enhanced error messages with actionable suggestions
#        • Added comprehensive BLEND_MODES fallbacks
#        • Implemented dims validation
#        • Added norm_threshold clamping
#        • Improved verbose logging format with rule enumeration
#        • Fixed float comparison edge cases in sigma matching
#        • Added input validation for critical parameters
#        • Enhanced YAML tooltip with precedence documentation
#    - v0.2.0: Added disable_apg, verbose_debug, improved parsing
#    - v0.1.0: Initial fork with refined header and tooltips

# ░▒▓ YAML EXAMPLE (rules apply in order of descending start_sigma):
# verbose: true
# rules:
# - start_sigma: 14.0    # High noise - use strong APG guidance
#   apg_scale: 5.0
#   cfg: 4.0
#   momentum: 0.75
# - start_sigma: 4.0     # Medium noise - reduce APG
#   apg_scale: 3.0
#   cfg: 3.5
# - start_sigma: 1.0     # Low noise - disable APG, pure CFG
#   apg_scale: 0.0
#   cfg: 3.0

# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄


import math
import yaml
import warnings

from enum import Enum, auto
from typing import NamedTuple, Union, List, Dict, Any, Optional, Tuple, Callable

import torch
import torch.nn.functional as F

from tqdm import tqdm

from comfy.samplers import CFGGuider

import nodes

# Version for compatibility checks
__version__ = "0.3.0"

# Global blend modes with comprehensive fallbacks
BLEND_MODES: Optional[Dict[str, Callable]] = None

def _ensure_blend_modes() -> None:
    """Initialize blend modes with comprehensive fallbacks."""
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

def _slerp(a: torch.Tensor, b: torch.Tensor, t: float) -> torch.Tensor:
    """Spherical linear interpolation fallback."""
    omega = torch.acos((a * b).sum() / (a.norm() * b.norm()))
    so = torch.sin(omega)
    return (torch.sin((1.0 - t) * omega) / so) * a + (torch.sin(t * omega) / so) * b

def _validate_dims(dims: Tuple[int, ...], tensor_ndim: int = 4) -> Tuple[int, ...]:
    """Validate dimension tuple for tensor operations."""
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

class UpdateMode(Enum):
    """Momentum update modes for APG guidance."""
    DEFAULT = auto()
    ALT1 = auto()
    ALT2 = auto()

class APGConfig(NamedTuple):
    """Configuration for a single APG rule with validated parameters."""
    start_sigma: float = math.inf
    momentum: float = -0.5
    eta: float = 0.0  # Deprecated: kept for config compatibility
    apg_scale: float = 4.0
    norm_threshold: float = 2.5
    dims: Tuple[int, ...] = (-2, -1)
    update_mode: UpdateMode = UpdateMode.DEFAULT
    update_blend_mode: str = "lerp"
    cfg: float = 1.0
    apg_blend: float = 1.0
    apg_blend_mode: str = "lerp"
    predict_image: bool = True
    pre_cfg_mode: bool = False

    @staticmethod
    def fixup_param(k: str, v: Any) -> Any:
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
                valid_modes = ", ".join(UpdateMode.__members__.keys())
                tqdm.write(f"⚠️  [APG] Invalid UpdateMode '{v}'. Valid options: {valid_modes}. Using DEFAULT.")
                return UpdateMode.DEFAULT
            return mode_enum
        
        if k == "start_sigma":
            return math.inf if v < 0 else float(v)
        
        if k == "norm_threshold":
            # Clamp to non-negative values
            threshold = float(v) if v is not None else 2.5
            if threshold < 0:
                tqdm.write(f"⚠️  [APG] norm_threshold cannot be negative ({threshold}). Clamping to 0.")
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
                warnings.warn(
                    f"eta parameter (value: {eta_val}) is deprecated and not used in APG projection. "
                    "It's kept for config compatibility but has no effect. Consider removing it from your config.",
                    DeprecationWarning,
                    stacklevel=2
                )
            return eta_val
        
        return v

    @classmethod
    def build(cls, *, mode: str = "pure_apg", **params: Dict[str, Any]) -> 'APGConfig':
        """Build an APGConfig from mode string and parameters."""
        if "_" in mode:
            pre_mode, update_mode_str = mode.split("_", 1)
        else:
            pre_mode = mode
            update_mode_str = mode

        params["pre_cfg_mode"] = pre_mode == "pre"
        params["update_mode"] = "default" if update_mode_str in {"apg", "cfg"} else update_mode_str
        
        fields = frozenset(cls._fields)
        processed_params = {k: cls.fixup_param(k, v) for k, v in params.items() if k in fields}
        
        defaults = cls()
        final_params = {
            **{k: getattr(defaults, k) for k in fields if k not in processed_params},
            **processed_params
        }
        
        return cls(**final_params)

    def __str__(self) -> str:
        """Human-readable string representation."""
        if self.apg_blend == 0 or self.apg_scale == 0:
            fields = ("start_sigma", "cfg")
        else:
            fields = self._fields
        pretty_fields = ", ".join(f"{k}={getattr(self, k)}" for k in fields)
        return f"APGConfig({pretty_fields})"


class APG:
    """Core APG logic with momentum and orthogonal projection."""
    
    def __init__(self, config: APGConfig):
        self.config = config
        self.running_average: Union[float, torch.Tensor] = 0.0
        self._warned_mps = False

    def __getattr__(self, k: str) -> Any:
        """Delegate attribute access to config."""
        return getattr(self.config, k)

    def update(self, val: torch.Tensor) -> torch.Tensor:
        """Update running average with momentum."""
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

    def reset(self) -> None:
        """Reset momentum state."""
        self.running_average = 0.0

    def project(
        self, v0_orig: torch.Tensor, v1_orig: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Project v0 onto v1 and return parallel and orthogonal components.
        
        Returns:
            (v0_parallel, v0_orthogonal) where v0 = v0_parallel + v0_orthogonal
        """
        # MPS performance warning
        if v0_orig.device.type == "mps" and not self._warned_mps:
            tqdm.write(
                "⚠️  [APG] MPS device detected. Moving tensors to CPU for projection. "
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

    def apg(self, cond: torch.Tensor, uncond: torch.Tensor) -> torch.Tensor:
        """
        Calculate APG guidance vector using orthogonal projection.
        
        Args:
            cond: Conditional prediction
            uncond: Unconditional prediction
            
        Returns:
            Orthogonal guidance correction vector
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
        
        # Return orthogonal component (eta parameter is deprecated)
        return diff_o

    def cfg_function(self, args: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Custom CFG function with APG guidance.
        
        Args:
            args: Dictionary containing cond, uncond, input, cond_denoised, uncond_denoised
            
        Returns:
            Guided prediction
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

    def pre_cfg_function(self, args: Dict[str, Any]) -> List[torch.Tensor]:
        """
        Pre-CFG mode: modify conditioning before CFG application.
        
        Args:
            args: Dictionary containing conds_out list
            
        Returns:
            Modified conditioning list
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


class APGGuider(CFGGuider):
    """Custom CFG guider with APG rule-based guidance."""
    
    def __init__(
        self, model, *, 
        positive, 
        negative, 
        rules: Tuple[APGConfig, ...], 
        params: Dict[str, Any]
    ) -> None:
        super().__init__(model)
        self.set_conds(positive, negative)
        self.set_cfg(1.0)
        self.apg_rules = tuple(APG(rule_config) for rule_config in rules)
        self.apg_verbose = params.get("verbose", False)
        
        if self.apg_verbose:
            tqdm.write(f"\n{'='*60}")
            tqdm.write(f"[APG] Initialized with {len(rules)} rule(s):")
            for i, rule in enumerate(rules, 1):
                sigma_str = "∞" if rule.start_sigma == math.inf else f"{rule.start_sigma:.2f}"
                apg_status = "ACTIVE" if (rule.apg_blend != 0 and rule.apg_scale != 0) else "DISABLED"
                tqdm.write(f"  Rule {i}: σ≤{sigma_str}, CFG={rule.cfg:.1f}, APG={rule.apg_scale:.1f} [{apg_status}]")
            tqdm.write(f"{'='*60}\n")

    def apg_reset(self, *, exclude: Optional[APG] = None) -> None:
        """Reset all APG momentum states except the specified one."""
        for apg_rule in self.apg_rules:
            if apg_rule is not exclude:
                apg_rule.reset()

    def apg_get_match(self, sigma: float) -> APG:
        """
        Find the appropriate APG rule for the current sigma value.
        
        Rules are matched in order, returning the first rule where sigma <= start_sigma.
        
        Args:
            sigma: Current noise level
            
        Returns:
            Matching APG rule
            
        Raises:
            RuntimeError: If no matching rule found (should not happen with proper fallback)
        """
        for rule in self.apg_rules:
            if sigma <= rule.start_sigma:
                return rule
        
        # This should never happen if rules are properly configured with an infinity fallback
        sigma_list = [f"{r.start_sigma:.2f}" for r in self.apg_rules]
        raise RuntimeError(
            f"No APG rule matched for sigma={sigma:.4f}. "
            f"Available rules start at: {', '.join(sigma_list)}. "
            f"Ensure you have a fallback rule with start_sigma=-1 (infinity). "
            f"This error indicates a configuration issue."
        )

    def outer_sample(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Wrap sampling with APG state reset."""
        self.apg_reset()
        result = super().outer_sample(*args, **kwargs)
        self.apg_reset()
        return result

    def predict_noise(
        self, 
        x: torch.Tensor, 
        timestep: Union[torch.Tensor, float], 
        model_options: Optional[Dict[str, Any]] = None, 
        seed: Optional[int] = None, 
        **kwargs: Any
    ) -> torch.Tensor:
        """
        Predict noise with APG guidance for the current timestep.
        
        Args:
            x: Latent tensor
            timestep: Current timestep/sigma
            model_options: Model configuration options
            seed: Random seed
            **kwargs: Additional arguments
            
        Returns:
            Predicted noise
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
            tqdm.write(
                f"[APG] σ={sigma:.4f} → Rule: CFG={rule.cfg:.2f}, "
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


class APGGuiderNode:
    """
    ComfyUI node for APG (Adaptive Projected Gradient) guidance.
    
    Enhanced fork with improved robustness, validation, and error handling.
    """

    CATEGORY = "MD_Nodes/Guiders"
    FUNCTION = "go"
    RETURN_TYPES = ("GUIDER",)
    RETURN_NAMES = ("apg_guider",)
    DESCRIPTION = "APG Guider (Forked v0.3.0) with enhanced guidance and validation"

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        """Define node input parameters."""
        _ensure_blend_modes()
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The model to apply APG guidance to"}),
                "positive": ("CONDITIONING", {"tooltip": "Positive conditioning (your prompt)"}),
                "negative": ("CONDITIONING", {"tooltip": "Negative conditioning (what to avoid)"}),
                "disable_apg": (
                    "BOOLEAN", 
                    {
                        "default": False, 
                        "label_on": "APG Disabled", 
                        "label_off": "APG Enabled",
                        "tooltip": "Disable APG guidance entirely and use only standard CFG"
                    }
                ),
                "verbose_debug": (
                    "BOOLEAN", 
                    {
                        "default": False, 
                        "label_on": "Verbose On", 
                        "label_off": "Verbose Off",
                        "tooltip": "Enable detailed logging in the console for debugging"
                    }
                ),
                "apg_scale": (
                    "FLOAT", 
                    {
                        "default": 4.5, 
                        "min": 0.0, 
                        "max": 1000.0, 
                        "step": 0.1,
                        "tooltip": "APG guidance strength. Higher = more aggressive orthogonal correction. 0 = disabled."
                    }
                ),
                "cfg_before": (
                    "FLOAT", 
                    {
                        "default": 4.0, 
                        "min": 1.0, 
                        "max": 1000.0, 
                        "step": 0.1,
                        "tooltip": "CFG scale while APG is active (before end_sigma)"
                    }
                ),
                "cfg_after": (
                    "FLOAT", 
                    {
                        "default": 3.0, 
                        "min": 1.0, 
                        "max": 1000.0, 
                        "step": 0.1,
                        "tooltip": "CFG scale after APG is disabled (after end_sigma or when APG disabled)"
                    }
                ),
                "eta": (
                    "FLOAT", 
                    {
                        "default": 0.0, 
                        "min": -1000.0, 
                        "max": 1000.0, 
                        "step": 0.1,
                        "tooltip": "DEPRECATED: Kept for compatibility but has no effect. Will show warning if non-zero."
                    }
                ),
                "norm_threshold": (
                    "FLOAT", 
                    {
                        "default": 2.5, 
                        "min": 0.0, 
                        "max": 1000.0, 
                        "step": 0.1,
                        "tooltip": "Maximum norm for guidance vector. Prevents extreme corrections. 0 = no limit."
                    }
                ),
                "momentum": (
                    "FLOAT", 
                    {
                        "default": 0.75, 
                        "min": -1000.0, 
                        "max": 1000.0, 
                        "step": 0.01,
                        "tooltip": "Momentum for running average. Negative values reverse momentum direction."
                    }
                ),
                "start_sigma": (
                    "FLOAT", 
                    {
                        "default": -1.0, 
                        "min": -1.0, 
                        "max": 10000.0, 
                        "step": 0.01,
                        "tooltip": "Sigma (noise level) at which APG activates. -1 = infinity (always active). "
                                   "Higher sigma = earlier in generation."
                    }
                ),
                "end_sigma": (
                    "FLOAT", 
                    {
                        "default": -1.0, 
                        "min": -1.0, 
                        "max": 10000.0, 
                        "step": 0.01,
                        "tooltip": "Sigma at which APG deactivates. -1 = never disable. "
                                   "Use to disable APG for final refinement steps."
                    }
                ),
                "dims": (
                    "STRING", 
                    {
                        "default": "-1, -2",
                        "tooltip": "Dimensions for normalization/projection. -1,-2 = spatial dimensions. "
                                   "Format: comma-separated integers."
                    }
                ),
                "predict_image": (
                    "BOOLEAN", 
                    {
                        "default": True,
                        "tooltip": "True = predict denoised image. False = predict noise. "
                                   "Match your model's prediction type."
                    }
                ),
                "mode": (
                    ("pure_apg", "pre_cfg", "pure_alt1", "pre_alt1", "pure_alt2", "pre_alt2"),
                    {
                        "tooltip": "APG mode:\n"
                                   "• pure_* = APG after CFG\n"
                                   "• pre_* = APG before CFG\n"
                                   "• *_apg = default momentum\n"
                                   "• *_alt1 = alternative momentum update\n"
                                   "• *_alt2 = blended momentum update"
                    }
                ),
            },
            "optional": {
                "yaml_parameters_opt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "dynamic_prompt": False,
                        "tooltip": "Advanced: Override all settings with YAML config.\n\n"
                                   "RULES APPLY IN ORDER (sorted by start_sigma, highest first):\n"
                                   "• Higher start_sigma rules match first\n"
                                   "• Use start_sigma: -1 for infinity (always-active fallback)\n\n"
                                   "SIMPLE EXAMPLE:\n"
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
                    },
                ),
            },
        }

    @classmethod
    def _build_rules_from_inputs(
        cls, 
        cfg_before: float, 
        cfg_after: float, 
        start_sigma: float, 
        end_sigma: float, 
        **kwargs: Any
    ) -> List[APGConfig]:
        """Build rule list from simple node inputs."""
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
        disable_apg: bool, 
        verbose_debug: bool, 
        apg_scale: float, 
        cfg_before: float, 
        cfg_after: float, 
        eta: float,
        norm_threshold: float, 
        momentum: float, 
        start_sigma: float, 
        end_sigma: float, 
        dims: str, 
        predict_image: bool,
        mode: str, 
        yaml_parameters_opt: Optional[str] = None
    ) -> Tuple[APGGuider]:
        """
        Execute node to create APG guider.
        
        Returns:
            Tuple containing configured APGGuider instance
        """
        if yaml_parameters_opt is None:
            yaml_parameters_opt = ""
        yaml_parameters_opt = yaml_parameters_opt.strip()
        
        params: Dict[str, Any] = {"verbose": verbose_debug}
        rules: Tuple[APGConfig, ...] = ()

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
                tqdm.write("[APG] APG is disabled. Using standard CFG guidance only.")
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
                    tqdm.write("[APG] Added automatic fallback rule with infinite start_sigma.")
        
        # Create and return guider
        guider = APGGuider(
            model,
            positive=positive,
            negative=negative,
            rules=rules,
            params=params,
        )
        return (guider,)


# ComfyUI node registration
NODE_CLASS_MAPPINGS = {
    "APGGuiderForked": APGGuiderNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "APGGuiderForked": "APG Guider (Forked v0.3.0)",
}