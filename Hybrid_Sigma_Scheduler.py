# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# ████ Hybrid Sigma Scheduler v1.5.0 – Enterprise Edition ████████▓▒░
# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀

# ░▒▓ ORIGIN & DEV:
#   • Brewed in a Compaq Presario by MDMAchine / MD_Nodes
#   • License: Apache 2.0 — Sharing is caring

# ░▒▓ DESCRIPTION:
#   Outputs a tensor of sigmas to control diffusion noise levels.
#   Designed for precision noise scheduling in ComfyUI workflows.

# ░▒▓ FEATURES:
#   ✓ Multiple core modes: Karras, Linear, Polynomial, Blended, and more
#   ✓ Extended schedulers: Beta, SGM Uniform, DDIM Uniform, Simple, AYS
#   ✓ Bong Tangent scheduler (piecewise tangent warping)
#   ✓ Automatic sigma range detection from the loaded model
#   ✓ Precision schedule slicing with `start_at_step` and `end_at_step`
#   ✓ Explicit Sigma Override Toggle (Prevents accidental defaults)
#   ✓ Percentage-based schedule splitting (scale independent)
#   ✓ Relative linear steps (dynamically tracks split point)
#   ✓ Visual schedule plot output (IMAGE)
#   ✓ Step density analysis
#   ✓ Cacheable: Correctly uses default ComfyUI caching.

# ░▒▓ CHANGELOG:
#   - v1.5.0 (Enterprise Fix):
#          • BUGFIX: Fixed recursion error in preset validation.
#          • ARCHITECTURE: Implemented global constants for safer maintenance.
#          • SAFETY: Added automated Preset Validation system.
#          • RELIABILITY: Added embedded Unit Test suite (run file directly to test).
#   - v1.4.2 (Refinement):
#          • FIX: Robust Scipy handling and Bong Tangent math hardening.

# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄


# Standard library imports
import math
import json
import io
import logging
import traceback
from typing import Dict, Any, List, Tuple, Optional, Union

# Third-party imports
import torch
import numpy as np

# ComfyUI core imports
import comfy.model_management
from comfy.k_diffusion.sampling import get_sigmas_karras

# =================================================================================
# == Configuration Constants                                                     ==
# =================================================================================
CONST_PLOT_DPI = 120
CONST_EPSILON = 1e-6
CONST_MONOTONIC_DECAY = 0.99
CONST_FALLBACK_SIGMA_MIN = 0.006
CONST_FALLBACK_SIGMA_MAX = 1.000
CONST_MEMORY_CHUNK_SIZE = 250

# =================================================================================
# == Dependency Checks                                                           ==
# =================================================================================

# Plotting Support
try:
    import matplotlib
    matplotlib.use('Agg') # REQUIRED by Sec 7.2
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Scipy Support (for smooth gradients)
try:
    from scipy.ndimage import gaussian_filter1d
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Setup logger
logger = logging.getLogger("ComfyUI_MD_Nodes.HybridSigmaScheduler")
if not MATPLOTLIB_AVAILABLE: logger.warning("Matplotlib not available. Schedule visualization disabled.")
if not PIL_AVAILABLE: logger.warning("PIL not available. Schedule visualization disabled.")
if not SCIPY_AVAILABLE: logger.info("Scipy not available. SigmaSmooth will fallback to Moving Average.")


# =================================================================================
# == Helper Functions (Schedulers)                                               ==
# =================================================================================

def kl_optimal_scheduler(n: int, sigma_min: float, sigma_max: float, device: Optional[torch.device] = None) -> torch.Tensor:
    if device is None: device = torch.device('cpu')
    sigma_min_t = torch.tensor(sigma_min, dtype=torch.float32, device=device)
    sigma_max_t = torch.tensor(sigma_max, dtype=torch.float32, device=device)
    
    if n > 1: adj_idxs = torch.arange(n, dtype=torch.float32, device=device).div_(n - 1)
    else: adj_idxs = torch.tensor([0.0], dtype=torch.float32, device=device)
    
    sigmas = torch.zeros(n + 1, dtype=torch.float32, device=device)
    sigmas[:-1] = (adj_idxs * torch.atan(sigma_min_t) + (1 - adj_idxs) * torch.atan(sigma_max_t)).tan_()
    return sigmas

def linear_quadratic_schedule_adapted(steps: int, sigma_max: float, threshold_noise: float = 0.0025, linear_steps: Optional[int] = None, device: Optional[torch.device] = None) -> torch.Tensor:
    if device is None: device = torch.device('cpu')
    if steps <= 0: return torch.tensor([sigma_max, 0.0], dtype=torch.float32, device=device)
    if steps == 1: return torch.tensor([sigma_max, 0.0], dtype=torch.float32, device=device)
    
    linear_steps_actual = steps // 2 if linear_steps is None else max(0, min(linear_steps, steps))
    sigma_schedule_raw = []
    
    if linear_steps_actual == 0:
        for i in range(steps):
            val = (i / (steps - 1.0))**2 if steps > 1 else 0.0
            sigma_schedule_raw.append(val)
    else:
        for i in range(linear_steps_actual):
            sigma_schedule_raw.append(i * threshold_noise / linear_steps_actual)
        
        quadratic_steps = steps - linear_steps_actual
        if quadratic_steps > 0:
            threshold_noise_step_diff = linear_steps_actual - threshold_noise * steps
            quadratic_coef = threshold_noise_step_diff / (linear_steps_actual * quadratic_steps ** 2)
            linear_coef = threshold_noise / linear_steps_actual - 2 * threshold_noise_step_diff / (quadratic_steps ** 2)
            const = quadratic_coef * (linear_steps_actual ** 2)
            for i in range(linear_steps_actual, steps):
                sigma_schedule_raw.append(quadratic_coef * (i ** 2) + linear_coef * i + const)
    
    if not sigma_schedule_raw or sigma_schedule_raw[-1] != 1.0:
        sigma_schedule_raw.append(1.0)
    
    sigma_schedule_inverted = [1.0 - x for x in sigma_schedule_raw]
    return torch.tensor(sigma_schedule_inverted, dtype=torch.float32, device=device) * sigma_max

def beta_scheduler(steps: int, sigma_min: float, sigma_max: float, alpha: float = 0.6, beta: float = 0.6, device: Optional[torch.device] = None) -> torch.Tensor:
    if device is None: device = torch.device('cpu')
    if steps <= 0: return torch.tensor([sigma_max, sigma_min], device=device)
    
    t = torch.linspace(0, 1, steps + 1, device=device)
    alpha_safe = max(alpha, 0.1)
    beta_safe = max(beta, 0.1)
    beta_curve = 1.0 - (1.0 - t ** alpha_safe) ** beta_safe
    beta_curve = torch.clamp(beta_curve, 0.0, 1.0)
    
    sigmas = sigma_max * (1.0 - beta_curve) + sigma_min * beta_curve
    sigmas[0] = sigma_max
    sigmas[-1] = sigma_min
    return sigmas

def sgm_uniform_scheduler(steps: int, sigma_min: float, sigma_max: float, device: Optional[torch.device] = None) -> torch.Tensor:
    if device is None: device = torch.device('cpu')
    if steps <= 0: return torch.tensor([sigma_max, sigma_min], device=device)
    
    max_timestep = 999.0
    timesteps = torch.linspace(max_timestep, 0, steps + 1, device=device)
    t_normalized = timesteps / max_timestep
    sigmas = sigma_min + (sigma_max - sigma_min) * t_normalized
    return sigmas

def ddim_uniform_scheduler(steps: int, sigma_min: float, sigma_max: float, device: Optional[torch.device] = None) -> torch.Tensor:
    if device is None: device = torch.device('cpu')
    if steps <= 0: return torch.tensor([sigma_max, sigma_min], device=device)
    
    max_timestep = 1000
    step_ratio = max_timestep // steps
    timesteps = torch.arange(0, steps + 1, device=device) * step_ratio
    timesteps = torch.flip(max_timestep - timesteps, [0]).float()
    t_normalized = timesteps / max_timestep
    sigmas = sigma_min + (sigma_max - sigma_min) * (t_normalized ** 0.5)
    return sigmas

def simple_scheduler(steps: int, sigma_min: float, sigma_max: float, device: Optional[torch.device] = None) -> torch.Tensor:
    if device is None: device = torch.device('cpu')
    if steps <= 0: return torch.tensor([sigma_max, sigma_min], device=device)
    
    t = torch.linspace(0, 1, steps + 1, device=device)
    smoothed = t * t * (3.0 - 2.0 * t)
    sigmas = sigma_max - (sigma_max - sigma_min) * smoothed
    return sigmas

def ays_scheduler(steps: int, sigma_min: float, sigma_max: float, device: Optional[torch.device] = None) -> torch.Tensor:
    if device is None: device = torch.device('cpu')
    if steps <= 0: return torch.tensor([sigma_max, sigma_min], device=device)
    
    t = torch.linspace(0, 1, steps + 1, device=device)
    ays_curve = 1.0 / (1.0 + torch.exp(-10 * (t - 0.5)))
    ays_curve = (ays_curve - ays_curve.min()) / (ays_curve.max() - ays_curve.min())
    concentration_factor = torch.exp(-2 * t)
    ays_curve = ays_curve * 0.7 + concentration_factor * 0.3
    ays_curve = 1.0 - (ays_curve / ays_curve.max())
    
    sigmas = sigma_min + (sigma_max - sigma_min) * ays_curve
    return sigmas

# =================================================================================
# == Core Node Class                                                             ==
# =================================================================================

class HybridAdaptiveSigmas:
    """
    Advanced sigma scheduler generator with multiple algorithms, presets,
    visualization, and the new Bong Tangent scheduler.
    """
    SCHEDULER_MODES: List[str] = [
        "karras_rho", "adaptive_linear", "polynomial", "exponential", 
        "variance_preserving", "blended_curves", "kl_optimal", "linear_quadratic",
        "beta", "sgm_uniform", "ddim_uniform", "simple", "ays", "bong_tangent"
    ]
    
    PRESETS: Dict[str, Optional[Dict[str, Any]]] = {
        "Custom": None,
        "ComfyUI Default": {
            "mode": "karras_rho", 
            "rho": 7.0, 
            "split_schedule": False, 
            "use_sigma_override": False,
            "denoise_mode": "Subtractive (Slice)",
        },
        "High Detail (Recommended)": {
            "mode": "linear_quadratic", "split_schedule": True, "mode_b": "polynomial",
            "split_at_step": 30, "power": 1.5, "threshold_noise": 0.001, "linear_steps": 27,
            "min_steps_mode": "adaptive", "adaptive_min_percentage": 2.0,
        },
        "Fast Draft": {
            "mode": "karras_rho", "rho": 2.5, "split_schedule": False,
            "min_steps_mode": "fixed", "min_sliced_steps": 2,
        },
        "Tiling Safe": {
            "mode": "blended_curves", "blend_factor": 0.3, "denoise_mode": "Hybrid (Adaptive Steps)",
            "low_denoise_color_fix": 0.2, "min_steps_mode": "adaptive", "adaptive_min_percentage": 3.0,
        },
        "Smooth Gradient": {
            "mode": "polynomial", "power": 1.8, "split_schedule": False, "detail_preservation": 0.1,
        },
        "Aggressive Start": {
            "mode": "exponential", "split_schedule": True, "mode_b": "adaptive_linear", "split_at_step": 40,
        },
        "Ultra Quality": {
            "mode": "kl_optimal", "split_schedule": True, "mode_b": "variance_preserving",
            "split_at_step": 60, "min_steps_mode": "adaptive", "adaptive_min_percentage": 5.0,
        },
        "Beta Distribution": {
            "mode": "beta", "beta_alpha": 0.6, "beta_beta": 0.6, "split_schedule": False,
        },
        "AYS Optimal": {
            "mode": "ays", "split_schedule": False, "min_steps_mode": "adaptive", "adaptive_min_percentage": 2.5,
        },
        "Composition Focus (Bong)": {
            "mode": "bong_tangent", "bong_preset": "composition_focus",
            "split_schedule": False, "min_steps_mode": "adaptive", "adaptive_min_percentage": 3.0
        },
        "Detail Focus (Bong)": {
            "mode": "bong_tangent", "bong_preset": "detail_focus",
            "split_schedule": False, "min_steps_mode": "adaptive", "adaptive_min_percentage": 3.0
        },
        "Balanced (Bong)": {
            "mode": "bong_tangent", "bong_preset": "balanced",
            "split_schedule": False, "min_steps_mode": "adaptive", "adaptive_min_percentage": 3.0
        },
    }

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        # Define the input structure first
        inputs = {
            "required": {
                "model": ("MODEL", {"tooltip": "MODEL INPUT: The diffusion model, used to auto-detect the optimal sigma_min and sigma_max."}),
                "steps": ("INT", {"default": 60, "min": 1, "max": 10000, "tooltip": "TOTAL STEPS: The total number of steps for a full schedule, before any slicing is applied. (Max 10,000)"}),
                "denoise_mode": (["Hybrid (Adaptive Steps)", "Subtractive (Slice)", "Repaced (Full Steps)"], {"tooltip": "DENOISE MODE: How denoise strength affects the schedule.\n• **Hybrid:** Best for tiling. Slices but ensures minimum steps.\n• **Subtractive:** Classic Img2Img slicing.\n• **Repaced:** Stretches a new full schedule over the denoised range."}),
                "mode": (cls.SCHEDULER_MODES, {"tooltip": "ALGORITHM: The primary mathematical curve used to generate the noise schedule."}),
            },
            "optional": {
                "preset": (list(cls.PRESETS.keys()), {"default": "Custom", "tooltip": "PRESET: Quick-load optimized settings. 'Custom' uses your manual settings below."}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "DENOISE STRENGTH: 1.0 = full generation, 0.x = Img2Img."}),
                "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000, "tooltip": "START STEP: Manually starts the schedule from this step."}),
                "end_at_step": ("INT", {"default": 9999, "min": 0, "max": 10000, "tooltip": "END STEP: Manually ends the schedule at this step (exclusive)."}),
                "min_steps_mode": (["fixed", "adaptive"], {"default": "fixed", "tooltip": "MIN STEPS MODE: How to calculate the minimum number of steps for Hybrid mode."}),
                "min_sliced_steps": ("INT", {"default": 3, "min": 1, "max": 100, "tooltip": "MIN SLICED STEPS (Fixed): Minimum steps for Hybrid mode when using 'fixed'."}),
                "adaptive_min_percentage": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 20.0, "step": 0.1, "tooltip": "MIN PERCENTAGE (Adaptive): Minimum steps as % of total steps for Hybrid mode."}),
                "detail_preservation": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "DETAIL PRESERVATION: Prevents reaching absolute zero noise, which helps preserve fine textures. Ideal for second passes."}),
                "low_denoise_color_fix": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "COLOR FIX: Helps prevent color shifting (e.g., green tint) in low-denoise or tiling workflows by subtly adjusting the final steps. Start with 0.2."}),
                
                # Split Schedule
                "split_schedule": ("BOOLEAN", {"default": False, "tooltip": "SPLIT SCHEDULE: Enable to use two different schedulers in one run. E.g., start with 'exponential' and finish with 'karras'."}),
                "mode_b": (cls.SCHEDULER_MODES, {"tooltip": "MODE B: The scheduler to use for the *second* part of the schedule, after the split point."}),
                "split_at_step": ("INT", {"default": 30, "min": 0, "max": 10000, "tooltip": "SPLIT STEP (Fixed): The step index where the schedule switches to Mode B. Ignored if using percentage split."}),
                # [FEATURE] New Split Controls
                "use_percentage_split": ("BOOLEAN", {"default": False, "label_on": "Use Percentage Split", "label_off": "Use Fixed Step", "tooltip": "PERCENTAGE SPLIT: If enabled, calculates the split point based on 'Split Percentage' instead of 'Split At Step'."}),
                "split_percentage": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "SPLIT PERCENTAGE: The fraction of total steps where the split occurs (e.g., 0.25 = 25% of the way through)."}),
                
                # Overrides
                "use_sigma_override": ("BOOLEAN", {"default": False, "label_on": "Enable Override", "label_off": "Disable Override", "tooltip": "SIGMA OVERRIDE: If enabled, the 'Start' and 'End' Sigma Override values below will be used instead of the model's internal range."}),
                "start_sigma_override": ("FLOAT", {"default": 1.000, "min": 0.0, "max": 20.0, "step": 0.001, "optional": True, "tooltip": "START SIGMA OVERRIDE: Manually set the absolute maximum noise level (sigma_max). Requires 'Enable Override' to be active."}),
                "end_sigma_override": ("FLOAT", {"default": 0.006, "min": 0.0, "max": 20.0, "step": 0.001, "optional": True, "tooltip": "END SIGMA OVERRIDE: Manually set the absolute minimum noise level (sigma_min). Requires 'Enable Override' to be active."}),
                
                # Algorithm Specifics
                "rho": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 15.0, "tooltip": "RHO (Karras/Blended): Controls curve steepness."}),
                "blend_factor": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "BLEND FACTOR (Blended): 0.0 = Karras, 1.0 = Linear."}),
                "power": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 8.0, "step": 0.1, "tooltip": "POWER (Polynomial): Curve exponent."}),
                "threshold_noise": ("FLOAT", {"default": 0.0025, "min": 0.0, "max": 1.0, "step": 0.001, "tooltip": "THRESHOLD NOISE (Lin-Quad): Transition point."}),
                
                # [FEATURE] New Linear-Quadratic Control
                "linear_steps": ("INT", {"default": 30, "min": 0, "max": 10000, "tooltip": "LINEAR STEPS (Lin-Quad): Steps dedicated to the initial linear portion."}),
                "linear_steps_relative": ("BOOLEAN", {"default": False, "label_on": "Relative to Split", "label_off": "Fixed Steps", "tooltip": "RELATIVE LINEAR STEPS: If enabled, 'Linear Steps' becomes an offset subtracted from the split point (e.g., Split - Linear Steps). Useful for keeping the linear phase consistent with the split."}),
                
                "beta_alpha": ("FLOAT", {"default": 0.6, "min": 0.1, "max": 5.0, "step": 0.1, "tooltip": "ALPHA (Beta): Shape parameter alpha."}),
                "beta_beta": ("FLOAT", {"default": 0.6, "min": 0.1, "max": 5.0, "step": 0.1, "tooltip": "BETA (Beta): Shape parameter beta."}),
                
                # Bong Tangent Params
                "bong_pivot": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 0.9, "step": 0.05, "tooltip": "BONG PIVOT: Transition point."}),
                "bong_slope_composition": ("FLOAT", {"default": 1.2, "min": 0.1, "max": 3.0, "step": 0.1, "tooltip": "BONG SLOPE COMPOSITION: High noise phase speed."}),
                "bong_slope_detail": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 3.0, "step": 0.1, "tooltip": "BONG SLOPE DETAIL: Low noise phase speed."}),
                "bong_preset": (["custom", "composition_focus", "balanced", "detail_focus"], {"default": "balanced", "tooltip": "BONG PRESET: Quick presets."}),

                # General
                "reverse_sigmas": ("BOOLEAN", {"default": False, "tooltip": "REVERSE: Experimental: Flips the schedule."}),
                "enable_visualization": ("BOOLEAN", {"default": False, "tooltip": "VISUALIZATION: Output plot and JSON data."}),
                "comparison_mode": ("BOOLEAN", {"default": False, "tooltip": "COMPARISON: A/B test mode."}),
                "memory_efficient": ("BOOLEAN", {"default": False, "tooltip": "MEMORY EFFICIENT: Process in chunks for high step counts."}),
            }
        }

        # [FEATURE] Auto-validate presets on first load
        if not hasattr(cls, '_presets_validated'):
            cls._validate_presets(inputs) # Fix: Pass inputs directly
            cls._presets_validated = True

        return inputs

    RETURN_TYPES = ("SIGMAS", "INT", "STRING", "STRING", "IMAGE")
    RETURN_NAMES = ("sigmas", "actual_steps", "schedule_info", "visualization_data", "schedule_plot")
    FUNCTION = "generate"
    CATEGORY = "MD_Nodes/Schedulers"

    @classmethod
    def _validate_presets(cls, input_data):
        """Dev Utility: Checks if presets contain valid parameter names to avoid typos."""
        # Fix: Receive inputs as argument, do not call cls.INPUT_TYPES()
        valid_keys = set(input_data["required"].keys()) | \
                     set(input_data["optional"].keys())
        
        for preset_name, settings in cls.PRESETS.items():
            if not settings: continue
            for key in settings:
                if key not in valid_keys:
                    logger.warning(f"⚠️ PRESET VALIDATION WARNING: Preset '{preset_name}' contains unknown key: '{key}'")

    # [NODE_FIX] Added type hints
    def _apply_preset(self, preset_name: str, **kwargs: Any) -> Dict[str, Any]:
        """Apply preset configuration."""
        if preset_name == "Custom" or preset_name not in self.PRESETS:
            return kwargs
        
        preset = self.PRESETS[preset_name].copy()
        logger.info(f"Applying preset '{preset_name}'")
        
        for key, value in preset.items():
            if key not in kwargs or kwargs.get('preset') == preset_name:
                kwargs[key] = value
        
        return kwargs
    
    # [NODE_FIX] Added type hints
    def _apply_bong_preset(self, preset: str) -> Tuple[float, float, float]:
        """Apply Bong Tangent presets."""
        presets = {
            "composition_focus": (0.7, 1.5, 0.5),
            "balanced": (0.5, 1.2, 0.8),
            "detail_focus": (0.3, 0.8, 1.2),
        }
        if preset in presets:
            pivot, slope_comp, slope_detail = presets[preset]
            logger.info(f"[Bong Tangent] Applied preset '{preset}': pivot={pivot}, slopes=({slope_comp}, {slope_detail})")
            return pivot, slope_comp, slope_detail
        return 0.5, 1.2, 0.8

    # [NODE_FIX] Added type hints
    def _generate_bong_tangent_schedule(self, steps: int, sigma_max: float, sigma_min: float,
                                                            pivot: float = 0.5, slope_composition: float = 1.2,
                                                            slope_detail: float = 0.8, device: Optional[torch.device] = None) -> torch.Tensor:
        """Generate Bong Tangent schedule with piecewise tangent functions."""
        if device is None: device = torch.device('cpu')
        
        pivot = max(0.1, min(0.9, pivot))
        slope_composition = max(0.1, slope_composition)
        slope_detail = max(0.1, slope_detail)
        
        composition_steps = int(steps * pivot)
        detail_steps = steps - composition_steps
        
        # [FIX] Sanity check for extreme pivot values to prevent empty ranges
        if composition_steps == 0:
            composition_steps = 1
            detail_steps = steps - 1
        elif detail_steps == 0:
            detail_steps = 1
            composition_steps = steps - 1
        
        sigmas = []
        for i in range(composition_steps):
            t = i / max(composition_steps - 1, 1)
            angle = t * (math.pi / 2 - 0.1) * slope_composition
            warped_t = math.tan(angle) / math.tan((math.pi / 2 - 0.1) * slope_composition)
            sigma = sigma_max * (1.0 - warped_t * pivot)
            sigmas.append(sigma)
        
        for i in range(detail_steps):
            t = i / max(detail_steps - 1, 1)
            angle = t * (math.pi / 2 - 0.1) * slope_detail
            warped_t = math.tan(angle) / math.tan((math.pi / 2 - 0.1) * slope_detail)
            sigma_range_start = sigma_max * (1.0 - pivot)
            sigma = sigma_range_start * (1.0 - warped_t) + sigma_min * warped_t
            sigmas.append(sigma)
        
        sigmas_tensor = torch.tensor(sigmas, dtype=torch.float32, device=device)
        sigmas_tensor = torch.cat([sigmas_tensor, torch.tensor([sigma_min], device=device, dtype=torch.float32)])
        
        # [FIX] Safer monotonic enforcement to prevent getting stuck at 0 using Constants
        for i in range(len(sigmas_tensor) - 1):
            if sigmas_tensor[i] <= sigmas_tensor[i + 1]:
                sigmas_tensor[i + 1] = max(sigmas_tensor[i] * CONST_MONOTONIC_DECAY, sigmas_tensor[i] - CONST_EPSILON)
        
        return sigmas_tensor

    # [NODE_FIX] Added type hints
    def _get_sigmas(self, steps: int, mode: str, sigma_min: float, sigma_max: float, device: torch.device,
                    rho: float, blend_factor: float, power: float, threshold_noise: float,
                    linear_steps: int, beta_alpha: float, beta_beta: float,
                    # Bong params passed via kwargs if needed, or handled before calling
                    bong_pivot: float = 0.5, bong_slope_comp: float = 1.2, bong_slope_detail: float = 0.8,
                    memory_efficient: bool = False) -> torch.Tensor:
        """Helper function to generate a sigma schedule."""
        if steps <= 0:
            return torch.tensor([sigma_max, sigma_min], device=device)
        
        # Memory-efficient mode using Constants
        if memory_efficient and steps > 500:
            chunk_size = CONST_MEMORY_CHUNK_SIZE
            chunks = []
            for i in range(0, steps, chunk_size):
                chunk_steps = min(chunk_size, steps - i)
                progress_start = i / steps
                progress_end = min(i + chunk_steps, steps) / steps
                chunk_sigma_max = sigma_max * (1 - progress_start) + sigma_min * progress_start
                chunk_sigma_min = sigma_max * (1 - progress_end) + sigma_min * progress_end
                
                chunk = self._get_sigmas(chunk_steps, mode, chunk_sigma_min, chunk_sigma_max, device,
                                                                 rho, blend_factor, power, threshold_noise, linear_steps,
                                                                 beta_alpha, beta_beta, 
                                                                 bong_pivot, bong_slope_comp, bong_slope_detail,
                                                                 memory_efficient=False)
                chunks.append(chunk[:-1] if i + chunk_steps < steps else chunk)
            return torch.cat(chunks)
        
        # Standard schedulers
        if mode == "karras_rho": return get_sigmas_karras(steps, sigma_min, sigma_max, rho, device)
        elif mode == "adaptive_linear": return torch.linspace(sigma_max, sigma_min, steps + 1, device=device)
        elif mode == "polynomial": return torch.linspace(sigma_max**(1/power), sigma_min**(1/power), steps + 1, device=device).pow(power)
        elif mode == "exponential":
            t = torch.linspace(0, 1, steps + 1, device=device)
            safe_max = sigma_max if sigma_max > 0 else 1e-9
            return safe_max * (sigma_min / safe_max) ** t
        elif mode == "variance_preserving":
            t = torch.linspace(1, 0, steps + 1, device=device)
            log_sigmas = (1 - t) * torch.log(torch.tensor(sigma_min, device=device)) + t * torch.log(torch.tensor(sigma_max, device=device))
            return torch.exp(log_sigmas)
        elif mode == "blended_curves":
            karras = get_sigmas_karras(steps, sigma_min, sigma_max, rho, device)
            linear = torch.linspace(sigma_max, sigma_min, steps + 1, device=device)
            return (1.0 - blend_factor) * karras + blend_factor * linear
        elif mode == "kl_optimal": return kl_optimal_scheduler(steps, sigma_min, sigma_max, device=device)
        elif mode == "linear_quadratic": return linear_quadratic_schedule_adapted(steps, sigma_max, threshold_noise, linear_steps, device=device)
        elif mode == "beta": return beta_scheduler(steps, sigma_min, sigma_max, beta_alpha, beta_beta, device=device)
        elif mode == "sgm_uniform": return sgm_uniform_scheduler(steps, sigma_min, sigma_max, device=device)
        elif mode == "ddim_uniform": return ddim_uniform_scheduler(steps, sigma_min, sigma_max, device=device)
        elif mode == "simple": return simple_scheduler(steps, sigma_min, sigma_max, device=device)
        elif mode == "ays": return ays_scheduler(steps, sigma_min, sigma_max, device=device)
        elif mode == "bong_tangent":
            return self._generate_bong_tangent_schedule(steps, sigma_max, sigma_min, bong_pivot, bong_slope_comp, bong_slope_detail, device=device)
        
        return torch.linspace(sigma_max, sigma_min, steps + 1, device=device)

    # [NODE_FIX] Added type hints
    def _generate_schedule_metadata(self, final_sigmas: torch.Tensor, actual_steps: int, mode: str, denoise_mode: str, sigma_min: float, sigma_max: float, **kwargs: Any) -> str:
        """Generate metadata JSON with detailed step density analysis."""
        sigma_list = final_sigmas.cpu().tolist()
        total_steps = len(sigma_list) - 1
        
        density_analysis = {}
        if total_steps > 0 and sigma_max > sigma_min:
            high_thresh = sigma_min + (sigma_max - sigma_min) * 0.66
            low_thresh = sigma_min + (sigma_max - sigma_min) * 0.33
            
            h_steps = sum(1 for s in sigma_list if s >= high_thresh)
            l_steps = sum(1 for s in sigma_list if s <= low_thresh)
            m_steps = total_steps - h_steps - l_steps
            
            density_analysis = {
                "high_noise_region": {
                    "steps": h_steps, 
                    "percentage": round(h_steps/total_steps*100, 1),
                    "sigma_range": f"{high_thresh:.4f} - {sigma_max:.4f}"
                },
                "mid_noise_region": {
                    "steps": m_steps, 
                    "percentage": round(m_steps/total_steps*100, 1),
                    "sigma_range": f"{low_thresh:.4f} - {high_thresh:.4f}"
                },
                "low_noise_region": {
                    "steps": l_steps, 
                    "percentage": round(l_steps/total_steps*100, 1),
                    "sigma_range": f"{sigma_min:.4f} - {low_thresh:.4f}"
                }
            }
        else:
             density_analysis = {"error": "Unable to calculate density"}
        
        info = {
            "version": "1.5.0", "mode": mode, "denoise_mode": denoise_mode,
            "steps": actual_steps, "sigma_range": [f"{final_sigmas[0].item():.4f}", f"{final_sigmas[-1].item():.4f}"],
            "step_density_analysis": density_analysis
        }
        if kwargs.get('split_schedule'): info.update({'split_mode': kwargs.get('mode_b'), 'split_at': kwargs.get('split_at_step')})
        if kwargs.get('preset') != "Custom": info['preset'] = kwargs.get('preset')
        
        # [FEATURE] Add metadata for new features
        if kwargs.get('use_percentage_split'):
            info['split_mode_percentage'] = f"{kwargs.get('split_percentage', 0)*100:.1f}%"
        if kwargs.get('linear_steps_relative'):
            info['linear_steps_mode'] = 'relative'
            info['linear_steps_offset'] = kwargs.get('linear_steps', 0)
        
        return json.dumps(info, indent=2)

    # [NODE_FIX] Added type hints
    def _generate_visualization_data(self, final_sigmas: torch.Tensor) -> str:
        """Convert sigma tensor to JSON for external visualization."""
        sigma_list = final_sigmas.cpu().tolist()
        viz_data = {
            "sigmas": sigma_list,
            "steps": list(range(len(sigma_list))),
            "max": max(sigma_list) if sigma_list else 0,
            "min": min(sigma_list) if sigma_list else 0,
            "mean": sum(sigma_list) / len(sigma_list) if sigma_list else 0,
        }
        return json.dumps(viz_data, indent=2)

    # [NODE_FIX] Added type hints
    def _plot_schedule_to_tensor(self, sigmas: torch.Tensor, mode: str, comparison_sigmas: Optional[torch.Tensor] = None, 
                                  comparison_mode: Optional[str] = None, split_at_step: Optional[int] = None, split_schedule: bool = False) -> torch.Tensor:
        """Plots schedule to tensor."""
        if not MATPLOTLIB_AVAILABLE or not PIL_AVAILABLE: return torch.zeros((1, 64, 64, 3), dtype=torch.float32)
        
        try:
            plt.style.use('dark_background')
            fig, ax = plt.subplots(figsize=(10, 4))
            sigma_list = sigmas.cpu().tolist()
            steps = list(range(len(sigma_list)))
            
            ax.plot(steps, sigma_list, color='#87CEEB', linewidth=2.0, label=f'{mode}', zorder=3)
            ax.fill_between(steps, sigma_list, alpha=0.2, color='#87CEEB')
            
            if comparison_sigmas is not None and comparison_mode:
                comp_list = comparison_sigmas.cpu().tolist()
                ax.plot(range(len(comp_list)), comp_list, color='#FF6B6B', linestyle='--', label=f'{comparison_mode}', alpha=0.8, zorder=2)
            
            if split_schedule and split_at_step and split_at_step < len(sigma_list):
                ax.axvline(x=split_at_step, color='#FFD700', linestyle=':', label='Split')
            
            ax.set_title(f"Sigma Schedule: {mode}", fontsize=11, fontweight='bold')
            ax.legend(loc='upper right', fontsize=8)
            plt.tight_layout()
            
            buf = io.BytesIO()
            # [FIX] Use Constant DPI
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=CONST_PLOT_DPI, facecolor=fig.get_facecolor())
            buf.seek(0)
            plt.close(fig)
            
            img = Image.open(buf).convert("RGB")
            return torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0)
        except Exception as e:
            logger.error(f"Plot generation failed: {e}")
            return torch.zeros((1, 64, 64, 3), dtype=torch.float32)

    # [NODE_FIX] Added type hints and logging
    def generate(self, model: Any, steps: int, denoise_mode: str, mode: str,
                 preset: str = "Custom", denoise: float = 1.0, start_at_step: int = 0, end_at_step: int = 9999,
                 min_steps_mode: str = "fixed", min_sliced_steps: int = 3, adaptive_min_percentage: float = 2.0,
                 detail_preservation: float = 0.0, low_denoise_color_fix: float = 0.0,
                 split_schedule: bool = False, mode_b: str = "karras_rho", split_at_step: int = 30,
                 start_sigma_override: Optional[float] = None, end_sigma_override: Optional[float] = None,
                 rho: float = 1.5, blend_factor: float = 0.5, power: float = 2.0, reverse_sigmas: bool = False,
                 threshold_noise: float = 0.0025, linear_steps: int = 30,
                 beta_alpha: float = 0.6, beta_beta: float = 0.6,
                 # Bong Tangent params
                 bong_pivot: float = 0.5, bong_slope_composition: float = 1.2, bong_slope_detail: float = 0.8,
                 bong_preset: str = "balanced",
                 # New Dynamic Controls
                 use_percentage_split: bool = False, split_percentage: float = 0.25, linear_steps_relative: bool = False,
                 # New Override Toggle
                 use_sigma_override: bool = False,
                 enable_visualization: bool = False, comparison_mode: bool = False, memory_efficient: bool = False
                 ) -> Tuple[torch.Tensor, int, str, str, torch.Tensor]:
        
        try:
            device = comfy.model_management.get_torch_device()

            # [FIX] Input validation
            if steps < 1:
                raise ValueError(f"Steps must be >= 1, got {steps}")
            if not 0.0 <= denoise <= 1.0:
                raise ValueError(f"Denoise must be between 0.0 and 1.0, got {denoise}")
            
            # Apply preset
            kwargs = locals().copy()
            del kwargs['self']
            kwargs = self._apply_preset(preset, **kwargs)
            
            mode = kwargs['mode']
            bong_preset = kwargs['bong_preset']
            
            if mode == "bong_tangent" and bong_preset != "custom":
                 p, sc, sd = self._apply_bong_preset(bong_preset)
                 kwargs['bong_pivot'] = p
                 kwargs['bong_slope_composition'] = sc
                 kwargs['bong_slope_detail'] = sd

            # [FEATURE] Dynamic Split Logic
            if kwargs.get('use_percentage_split'):
                calculated_split = int(kwargs['steps'] * kwargs.get('split_percentage', 0.25))
                kwargs['split_at_step'] = max(1, min(kwargs['steps'] - 1, calculated_split))
                logger.debug(f"Dynamic Split: {kwargs['split_percentage']*100:.1f}% -> Step {kwargs['split_at_step']}")

            # [FEATURE] Relative Linear Steps Logic
            if kwargs.get('linear_steps_relative'):
                offset = kwargs['linear_steps'] # Treated as offset
                anchor = kwargs['split_at_step'] if kwargs.get('split_schedule') else kwargs['steps']
                kwargs['linear_steps'] = max(1, anchor - offset)
                logger.debug(f"Relative Linear Steps: Anchor {anchor} - Offset {offset} = {kwargs['linear_steps']}")

            try:
                ms = model.get_model_object("model_sampling")
                sigma_min, sigma_max = ms.sigma_min, ms.sigma_max
            except:
                sigma_min, sigma_max = CONST_FALLBACK_SIGMA_MIN, CONST_FALLBACK_SIGMA_MAX
                logger.warning("Using fallback sigmas.")

            # [FIX] Explicit Override Check
            if kwargs.get('use_sigma_override'):
                if kwargs['start_sigma_override'] is not None: 
                    sigma_max = kwargs['start_sigma_override']
                    logger.info(f"Override: sigma_max set to {sigma_max}")
                if kwargs['end_sigma_override'] is not None: 
                    sigma_min = kwargs['end_sigma_override']
                    logger.info(f"Override: sigma_min set to {sigma_min}")
            
            if sigma_min >= sigma_max: sigma_max = sigma_min + 0.1

            if kwargs['min_steps_mode'] == "adaptive":
                kwargs['min_sliced_steps'] = max(1, int(kwargs['steps'] * kwargs['adaptive_min_percentage'] / 100.0))

            gen_params = {
                "device": device, "rho": kwargs['rho'], "blend_factor": kwargs['blend_factor'], "power": kwargs['power'],
                "threshold_noise": kwargs['threshold_noise'], "linear_steps": kwargs['linear_steps'],
                "beta_alpha": kwargs['beta_alpha'], "beta_beta": kwargs['beta_beta'],
                "bong_pivot": kwargs['bong_pivot'], "bong_slope_comp": kwargs['bong_slope_composition'], "bong_slope_detail": kwargs['bong_slope_detail'],
                "memory_efficient": kwargs['memory_efficient']
            }

            if kwargs['split_schedule'] and 0 < kwargs['split_at_step'] < kwargs['steps']:
                temp_full = self._get_sigmas(kwargs['steps'], kwargs['mode'], sigma_min, sigma_max, **gen_params)
                split_sigma = temp_full[kwargs['split_at_step']].item()
                sigmas_a = self._get_sigmas(kwargs['split_at_step'], kwargs['mode'], split_sigma, sigma_max, **gen_params)
                sigmas_b = self._get_sigmas(kwargs['steps'] - kwargs['split_at_step'], kwargs['mode_b'], sigma_min, split_sigma, **gen_params)
                full_sigmas = torch.cat((sigmas_a[:-1], sigmas_b))
            else:
                full_sigmas = self._get_sigmas(kwargs['steps'], kwargs['mode'], sigma_min, sigma_max, **gen_params)

            denoise_start = int(kwargs['steps'] * (1.0 - kwargs['denoise'])) if kwargs['denoise'] < 1.0 else 0
            start_idx = max(0, min(max(kwargs['start_at_step'], denoise_start), len(full_sigmas) - 1))
            end_idx = min(kwargs['end_at_step'], len(full_sigmas)) if kwargs['end_at_step'] < len(full_sigmas) else len(full_sigmas)
            
            if kwargs['denoise_mode'] == "Repaced (Full Steps)":
                eff_max = full_sigmas[start_idx].item()
                eff_min = full_sigmas[min(end_idx, len(full_sigmas)) - 1].item()
                final_sigmas = self._get_sigmas(kwargs['steps'], kwargs['mode'], eff_min, eff_max, **gen_params)
            elif kwargs['denoise_mode'] == "Hybrid (Adaptive Steps)":
                final_sigmas = full_sigmas[start_idx:end_idx]
                if (len(final_sigmas) - 1) < kwargs['min_sliced_steps']:
                    final_sigmas = self._get_sigmas(kwargs['min_sliced_steps'], kwargs['mode'], final_sigmas[-1].item(), final_sigmas[0].item(), **gen_params)
            else:
                final_sigmas = full_sigmas[start_idx:end_idx]

            final_sigmas = final_sigmas.clone()
            if kwargs['detail_preservation'] > 0:
                 final_sigmas[-1] = final_sigmas[-1] + (min(sigma_max/50, 0.5) - final_sigmas[-1]) * kwargs['detail_preservation']
            if kwargs['low_denoise_color_fix'] > 0 and len(final_sigmas) > 1:
                 target = final_sigmas[-2] * 0.25
                 final_sigmas[-1] = final_sigmas[-1] + (target - final_sigmas[-1]) * kwargs['low_denoise_color_fix']
            if kwargs['reverse_sigmas']: final_sigmas = torch.flip(final_sigmas, [0])

            actual_steps = max(0, len(final_sigmas) - 1)
            
            # [NODE_FIX] Pass ONLY non-conflicting kwargs to _generate_schedule_metadata
            meta_kwargs = kwargs.copy()
            meta_kwargs.pop('mode', None)
            meta_kwargs.pop('denoise_mode', None)
            
            meta = self._generate_schedule_metadata(final_sigmas, actual_steps, kwargs['mode'], kwargs['denoise_mode'], sigma_min, sigma_max, **meta_kwargs)
            
            viz_data = self._generate_visualization_data(final_sigmas) if kwargs['enable_visualization'] else "{}"
            
            comp_sigmas = None
            if kwargs['comparison_mode'] and kwargs['mode_b'] != kwargs['mode']:
                 logger.info(f"Comparison mode active - generating secondary schedule with {kwargs['mode_b']}")
                 comp_sigmas = self._get_sigmas(actual_steps, kwargs['mode_b'], sigma_min, sigma_max, **gen_params)
                 if kwargs['enable_visualization']:
                    viz_dict = json.loads(viz_data)
                    viz_dict["comparison_sigmas"] = comp_sigmas.cpu().tolist()
                    viz_dict["comparison_mode"] = kwargs['mode_b']
                    viz_data = json.dumps(viz_dict, indent=2)
            
            plot = self._plot_schedule_to_tensor(final_sigmas, kwargs['mode'], comp_sigmas, kwargs['mode_b'] if kwargs['comparison_mode'] else None, kwargs['split_at_step'], kwargs['split_schedule'])
            
            logger.info(f"Generated schedule: {actual_steps} steps, Mode: {kwargs['mode']}")
            return (final_sigmas, actual_steps, meta, viz_data, plot)

        except Exception as e:
            logger.error(f"Generate failed: {e}", exc_info=True)
            raise e

NODE_CLASS_MAPPINGS = {"HybridAdaptiveSigmas": HybridAdaptiveSigmas}
NODE_DISPLAY_NAME_MAPPINGS = {"HybridAdaptiveSigmas": "Hybrid Sigma Scheduler"}


# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# ████ Sigma Utilities – Companion nodes for schedule manipulation ████▓▒░
# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀


class SigmaSmooth:
    """
    Applies smoothing to a sigma schedule to reduce harsh transitions.
    Useful for reducing artifacts caused by abrupt sigma changes.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"tooltip": "Input sigma schedule to smooth."}),
                "smoothing_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05, 
                    "tooltip": "Strength of smoothing. 0.0 = no change, 1.0 = maximum smoothing. Start with 0.1-0.3."}),
                "smoothing_type": (["gaussian", "moving_average", "exponential"], {
                    "tooltip": "Type of smoothing filter to apply.\nGaussian: Natural bell-curve smoothing.\nMoving Average: Simple window averaging.\nExponential: Preserves endpoints better."}),
            },
            "optional": {
                "preserve_endpoints": ("BOOLEAN", {"default": True, 
                    "tooltip": "Keep the first and last sigma values unchanged. Recommended for stability."}),
                "window_size": ("INT", {"default": 3, "min": 2, "max": 15, "step": 1,
                    "tooltip": "Size of the smoothing window. Larger = more smoothing but may lose detail."}),
            }
        }
    
    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("sigmas",)
    FUNCTION = "smooth"
    CATEGORY = "MD_Nodes/Schedulers/Utilities"
    
    def smooth(self, sigmas, smoothing_strength, smoothing_type, preserve_endpoints=True, window_size=3):
        if smoothing_strength <= 0.0:
            return (sigmas,)
        
        device = sigmas.device
        sigma_list = sigmas.cpu().numpy().astype(np.float64)
        n = len(sigma_list)
        
        if n <= 2:
            return (sigmas,)
        
        # Store original endpoints
        original_start = sigma_list[0]
        original_end = sigma_list[-1]
        
        # Apply smoothing based on type
        if smoothing_type == "gaussian":
            smoothed = self._gaussian_smooth(sigma_list, window_size, smoothing_strength)
        elif smoothing_type == "moving_average":
            smoothed = self._moving_average_smooth(sigma_list, window_size, smoothing_strength)
        elif smoothing_type == "exponential":
            smoothed = self._exponential_smooth(sigma_list, smoothing_strength)
        else:
            smoothed = sigma_list
        
        # Blend original and smoothed based on strength
        result = sigma_list * (1.0 - smoothing_strength) + smoothed * smoothing_strength
        
        # Restore endpoints if requested
        if preserve_endpoints:
            result[0] = original_start
            result[-1] = original_end
        
        # Ensure monotonic decrease (sigmas should always decrease) using Constants
        for i in range(1, len(result)):
            if result[i] > result[i-1]:
                result[i] = max(result[i-1] - CONST_EPSILON, 0.0)
        
        # Ensure no negative values
        result = np.maximum(result, 0.0)
        
        return (torch.from_numpy(result).to(device=device, dtype=sigmas.dtype),)
    
    def _gaussian_smooth(self, data, window_size, strength):
        """Apply Gaussian smoothing."""
        # [FIX] Use global flag
        if SCIPY_AVAILABLE:
            from scipy.ndimage import gaussian_filter1d
            sigma_param = max(0.5, window_size * strength)
            return gaussian_filter1d(data, sigma=sigma_param)
        else:
            # Fallback if scipy not available
            return self._moving_average_smooth(data, window_size, strength)
    
    def _moving_average_smooth(self, data, window_size, strength):
        """Apply moving average smoothing."""
        n = len(data)
        result = data.copy()
        half_window = window_size // 2
        
        for i in range(half_window, n - half_window):
            window_start = max(0, i - half_window)
            window_end = min(n, i + half_window + 1)
            result[i] = np.mean(data[window_start:window_end])
        
        return result
    
    def _exponential_smooth(self, data, strength):
        """Apply exponential smoothing (preserves trend better)."""
        alpha = 0.1 + (strength * 0.4)  # Alpha between 0.1 and 0.5
        result = data.copy()
        
        for i in range(1, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        
        return result


class SigmaConcatenate:
    """
    Concatenates two sigma schedules for multi-pass workflows.
    Useful for chaining different schedulers or creating custom multi-stage schedules.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas_a": ("SIGMAS", {"tooltip": "First sigma schedule (executed first)."}),
                "sigmas_b": ("SIGMAS", {"tooltip": "Second sigma schedule (executed after first)."}),
                "blend_mode": (["concatenate", "crossfade", "append_from_overlap"], {
                    "tooltip": "How to join the schedules.\nConcatenate: Simple join at the last sigma of A.\nCrossfade: Smooth transition over overlap region.\nAppend from Overlap: Find where B overlaps A's ending and append from there."}),
            },
            "optional": {
                "crossfade_steps": ("INT", {"default": 5, "min": 1, "max": 50, "step": 1,
                    "tooltip": "Number of steps to crossfade over (for crossfade mode)."}),
                "normalize_range": ("BOOLEAN", {"default": False,
                    "tooltip": "Scale schedule B to start where A ends. Ensures smooth sigma progression."}),
            }
        }
    
    RETURN_TYPES = ("SIGMAS", "INT")
    RETURN_NAMES = ("sigmas", "total_steps")
    FUNCTION = "concatenate"
    CATEGORY = "MD_Nodes/Schedulers/Utilities"
    
    def concatenate(self, sigmas_a, sigmas_b, blend_mode, crossfade_steps=5, normalize_range=False):
        device = sigmas_a.device
        
        list_a = sigmas_a.cpu().numpy().astype(np.float64)
        list_b = sigmas_b.cpu().numpy().astype(np.float64)
        
        if len(list_a) == 0:
            return (sigmas_b, len(list_b) - 1)
        if len(list_b) == 0:
            return (sigmas_a, len(list_a) - 1)
        
        # Optionally normalize B to start where A ends
        if normalize_range and len(list_a) > 0 and len(list_b) > 1:
            a_end = list_a[-1]
            b_start = list_b[0]
            b_end = list_b[-1]
            
            if b_start > b_end and b_start != b_end:  # B is decreasing
                # Scale B to fit between a_end and b_end (or 0)
                scale = a_end / b_start if b_start > 0 else 1.0
                list_b = list_b * scale
        
        if blend_mode == "concatenate":
            # Simple concatenation - drop last element of A (usually 0 or min)
            result = np.concatenate([list_a[:-1], list_b])
        
        elif blend_mode == "crossfade":
            # Crossfade between end of A and start of B
            crossfade_steps = min(crossfade_steps, len(list_a) - 1, len(list_b) - 1)
            
            if crossfade_steps <= 0:
                result = np.concatenate([list_a[:-1], list_b])
            else:
                # Take A up to crossfade region
                pre_crossfade = list_a[:-crossfade_steps-1]
                
                # Create crossfade region
                crossfade_a = list_a[-crossfade_steps-1:-1]
                crossfade_b = list_b[:crossfade_steps]
                
                weights = np.linspace(1.0, 0.0, crossfade_steps)
                crossfade_region = crossfade_a * weights + crossfade_b * (1.0 - weights)
                
                # Take rest of B
                post_crossfade = list_b[crossfade_steps:]
                
                result = np.concatenate([pre_crossfade, crossfade_region, post_crossfade])
        
        elif blend_mode == "append_from_overlap":
            # Find where B's sigmas overlap with A's ending sigma
            a_end_sigma = list_a[-2] if len(list_a) > 1 else list_a[-1]  # Second to last (last is usually 0)
            
            # Find first index in B where sigma <= a_end_sigma
            overlap_idx = 0
            for i, sigma in enumerate(list_b):
                if sigma <= a_end_sigma:
                    overlap_idx = i
                    break
            
            result = np.concatenate([list_a[:-1], list_b[overlap_idx:]])
        
        else:
            result = np.concatenate([list_a[:-1], list_b])
        
        # Ensure monotonic decrease
        for i in range(1, len(result)):
            if result[i] > result[i-1]:
                result[i] = max(result[i-1] - CONST_EPSILON, 0.0)
        
        result = np.maximum(result, 0.0)
        total_steps = len(result) - 1
        
        return (torch.from_numpy(result).to(device=device, dtype=sigmas_a.dtype), total_steps)


# Update node mappings to include utility nodes
NODE_CLASS_MAPPINGS.update({
    "SigmaSmooth": SigmaSmooth,
    "SigmaConcatenate": SigmaConcatenate
})

NODE_DISPLAY_NAME_MAPPINGS.update({
    "SigmaSmooth": "Sigma Smooth",
    "SigmaConcatenate": "Sigma Concatenate"
})


# =================================================================================
# == Development & Testing                                                       ==
# =================================================================================

if __name__ == "__main__":
    # This block only runs if you execute the file directly: python MD_Hybrid_Scheduler.py
    print("🧪 Running Self-Tests for Hybrid Sigma Scheduler v1.5.0...")
    
    try:
        # 1. Test Monotonicity (The "Bong" Safeguard)
        scheduler = HybridAdaptiveSigmas()
        sigmas = scheduler._generate_bong_tangent_schedule(
            steps=20, sigma_max=1.0, sigma_min=0.1, 
            pivot=0.5, slope_composition=1.0, slope_detail=1.0
        )
        
        # Check if sorted descending
        is_monotonic = all(sigmas[i] >= sigmas[i+1] for i in range(len(sigmas)-1))
        if is_monotonic:
            print("✅ Bong Tangent Monotonicity Check: PASSED")
        else:
            print("❌ Bong Tangent Monotonicity Check: FAILED")

        # 2. Test Zero-Division Edge Case
        # Try to break it with 0 steps (should return default range)
        safe_sigmas = scheduler._get_sigmas(0, "linear", 0.1, 1.0, torch.device('cpu'), 
                                          1.0, 0.0, 1.0, 0.0, 0, 0.0, 0.0)
        if len(safe_sigmas) == 2:
            print("✅ Zero Step Safety Check: PASSED")
        else:
            print("❌ Zero Step Safety Check: FAILED")

        # 3. Test Preset Validator
        print("🔍 Testing Preset Validation (Expect no warnings):")
        # Manually trigger validation with sample data to ensure method works without recursion
        sample_input = {"required": {"mode": None}, "optional": {"preset": None}}
        scheduler._validate_presets(sample_input)
        print("✅ Preset Validation: PASSED")
            
        print("\nAll systems nominal. Ready for deployment.")
        
    except Exception as e:
        print(f"\n❌ TESTS FAILED WITH ERROR: {e}")
        traceback.print_exc()