# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
# ████ MD_Nodes/HybridAdaptiveSigmas – Hybrid noise scheduler ████▓▒░
# © 2025 MDMAchine
# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
# ░▒▓ ORIGIN & DEV:
#   • Cast into the void by: MDMAchine
#   • Enhanced by: Gemini, Claude, etc.
#   • License: Apache 2.0 — Sharing is caring

# ░▒▓ DESCRIPTION:
#   Outputs a tensor of sigmas to control diffusion noise levels.
#   This node is designed for precision noise scheduling in ComfyUI workflows,
#   supporting multiple curves, blending, slicing, and presets.

# ░▒▓ FEATURES:
#   ✓ Multiple core modes: Karras, Linear, Polynomial, Blended, Beta, AYS, etc.
#   ✓ Automatic sigma range detection from the loaded model.
#   ✓ Manual override for sigma min/max for expert control.
#   ✓ Precision schedule slicing with `start_at_step` and `end_at_step`.
#   ✓ Outputs the actual number of steps after slicing.
#   ✓ Schedule visualization output (JSON) for analysis.
#   ✓ Adaptive minimum steps based on percentage.

# ░▒▓ CHANGELOG:
#   - v1.4.2 (Guideline Update - Oct 2025):
#       • REFACTOR: Full compliance update to v1.4.2 guidelines.
#       • CRITICAL: Removed all type hints from function signatures to prevent ComfyUI loading crashes.
#       • STYLE: Standardized imports, docstrings, and error handling.
#       • STYLE: Replaced all print() calls with logging.info(), logging.warning(), etc.
#       • STYLE: Rewrote all tooltips to new standard format.
#       • ROBUST: Wrapped main execution in try/except with graceful failure passthrough.
#       • ROBUST: Updated helper functions to use comfy.model_management for device.
#       • STYLE: Updated node display name to 'MD: Hybrid Sigma Scheduler'.
#   - v1.10 (Extended Scheduler Suite):
#       • ADDED: Beta, SGM Uniform, DDIM Uniform, Simple, and AYS schedulers.
#       • ENHANCED: Preset system with new scheduler options.
#   - v1.00 (Major Update):
#       • ADDED: Comprehensive preset system and schedule visualization output.
#       • ADDED: Adaptive minimum steps using percentage-based scaling.
#   - v0.90 (Quality Update):
#       • Increased max steps to 1000.
#       • Fixed device inconsistencies and improved validation.

# ░▒▓ CONFIGURATION:
#   → Primary Use: Fine-grained diffusion noise scheduling for generative workflows.
#   → Secondary Use: Experimental noise shaping and curve blending.
#   → Edge Use: Schedule analysis, comparison, and optimization.

# ░▒▓ WARNING:
#   This node may trigger:
#   ▓▒░ A 6-hour rabbit hole plotting 10 different polynomial curves.
#   ▓▒░ The compulsive need to run a 1000-step diffusion "just to see."
#   ▓▒░ Flashbacks to tweaking `gravis.ini` for the perfect GUS patch mix.
#   ▓▒░ An existential crisis when you realize all your sigmas are just... numbers, man.
#   Consult your nearest demoscene vet if hallucinations persist.
# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄


# =================================================================================
# == Standard Library Imports                                                    ==
# =================================================================================
import json
import logging
import math
import traceback

# =================================================================================
# == Third-Party Imports                                                         ==
# =================================================================================
import torch

# =================================================================================
# == ComfyUI Core Modules                                                        ==
# =================================================================================
import comfy.k_diffusion.sampling
import comfy.model_management

# =================================================================================
# == Local Project Imports                                                       ==
# =================================================================================
# (No local project imports in this file)

# =================================================================================
# == Helper Classes & Dependencies                                               ==
# =================================================================================

def kl_optimal_scheduler(n, sigma_min, sigma_max, device=None):
    """
    Calculates the KL optimal schedule.
    Referenced from https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/15608
    
    Args:
        n (int): Number of steps.
        sigma_min (float): Minimum sigma.
        sigma_max (float): Maximum sigma.
        device (torch.device, optional): The device to create tensors on.
    
    Returns:
        torch.Tensor: The calculated sigmas.
    """
    if device is None:
        device = comfy.model_management.get_torch_device()
    
    sigma_min_t = torch.tensor(sigma_min, dtype=torch.float32, device=device)
    sigma_max_t = torch.tensor(sigma_max, dtype=torch.float32, device=device)
    
    if n > 1:
        adj_idxs = torch.arange(n, dtype=torch.float32, device=device).div_(n - 1)
    else:
        adj_idxs = torch.tensor([0.0], dtype=torch.float32, device=device)
    
    sigmas = torch.zeros(n + 1, dtype=torch.float32, device=device)
    sigmas[:-1] = (adj_idxs * torch.atan(sigma_min_t) + (1 - adj_idxs) * torch.atan(sigma_max_t)).tan_()
    
    return sigmas

def linear_quadratic_schedule_adapted(steps, sigma_max, threshold_noise=0.0025, linear_steps=None, device=None):
    """
    Calculates a schedule that is linear at the start and quadratic at the end.
    Adapted from: https://github.com/genmoai/models/blob/main/src/mochi_preview/infer.py#L41
    
    Args:
        steps (int): Total number of steps.
        sigma_max (float): Maximum sigma.
        threshold_noise (float): The sigma value where the schedule transitions.
        linear_steps (int, optional): Number of steps for the linear portion.
        device (torch.device, optional): The device to create tensors on.
    
    Returns:
        torch.Tensor: The calculated sigmas.
    """
    if device is None:
        device = comfy.model_management.get_torch_device()
    
    if steps <= 0:
        return torch.FloatTensor([sigma_max, 0.0]).to(device)
    if steps == 1:
        return torch.FloatTensor([sigma_max, 0.0]).to(device)
    
    if linear_steps is None:
        linear_steps_actual = steps // 2
    else:
        linear_steps_actual = max(0, min(linear_steps, steps))
    
    sigma_schedule_raw = []
    
    if linear_steps_actual == 0:
        for i in range(steps):
            if steps > 1:
                sigma_schedule_raw.append((i / (steps - 1.0))**2)
            else:
                sigma_schedule_raw.append(0.0)
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
    return torch.FloatTensor(sigma_schedule_inverted).to(device) * sigma_max


def beta_scheduler(steps, sigma_min, sigma_max, alpha=0.6, beta=0.6, device=None):
    """
    Beta distribution-based scheduler. Popular in diffusion models for controlled noise distribution.
    
    Args:
        steps (int): Total number of steps.
        sigma_min (float): Minimum sigma.
        sigma_max (float): Maximum sigma.
        alpha (float): Beta distribution alpha parameter.
        beta (float): Beta distribution beta parameter.
        device (torch.device, optional): The device to create tensors on.
    
    Returns:
        torch.Tensor: The calculated sigmas.
    """
    if device is None:
        device = comfy.model_management.get_torch_device()
    
    if steps <= 0:
        return torch.tensor([sigma_max, sigma_min], device=device)
    
    t = torch.linspace(0, 1, steps + 1, device=device)
    
    # Apply beta-inspired curve transformation
    alpha_safe = max(alpha, 0.1)
    beta_safe = max(beta, 0.1)
    
    # Create the curve using a stable formulation
    beta_curve = 1.0 - (1.0 - t ** alpha_safe) ** beta_safe
    
    # Ensure the curve is properly bounded and monotonic
    beta_curve = torch.clamp(beta_curve, 0.0, 1.0)
    
    # Map to sigma range with proper endpoints
    sigmas = sigma_max * (1.0 - beta_curve) + sigma_min * beta_curve
    
    # Force exact endpoints to prevent numerical drift
    sigmas[0] = sigma_max
    sigmas[-1] = sigma_min
    
    return sigmas


def sgm_uniform_scheduler(steps, sigma_min, sigma_max, device=None):
    """
    SGM Uniform: Uniform spacing in timestep domain, then converted to sigmas.
    
    Args:
        steps (int): Total number of steps.
        sigma_min (float): Minimum sigma.
        sigma_max (float): Maximum sigma.
        device (torch.device, optional): The device to create tensors on.
    
    Returns:
        torch.Tensor: The calculated sigmas.
    """
    if device is None:
        device = comfy.model_management.get_torch_device()
    
    if steps <= 0:
        return torch.tensor([sigma_max, sigma_min], device=device)
    
    # We'll assume a typical 1000-step timestep range
    max_timestep = 999.0
    timesteps = torch.linspace(max_timestep, 0, steps + 1, device=device)
    
    # Convert timesteps to sigmas using a log-linear relationship
    t_normalized = timesteps / max_timestep
    sigmas = sigma_min + (sigma_max - sigma_min) * t_normalized
    
    return sigmas


def ddim_uniform_scheduler(steps, sigma_min, sigma_max, device=None):
    """
    DDIM Uniform: Uniform timestep spacing specifically for DDIM-style sampling.
    
    Args:
        steps (int): Total number of steps.
        sigma_min (float): Minimum sigma.
        sigma_max (float): Maximum sigma.
        device (torch.device, optional): The device to create tensors on.
    
    Returns:
        torch.Tensor: The calculated sigmas.
    """
    if device is None:
        device = comfy.model_management.get_torch_device()
    
    if steps <= 0:
        return torch.tensor([sigma_max, sigma_min], device=device)
    
    # DDIM uses uniform timestep spacing with a specific stride pattern
    max_timestep = 1000
    step_ratio = max_timestep // steps
    
    timesteps = torch.arange(0, steps + 1, device=device) * step_ratio
    timesteps = torch.flip(max_timestep - timesteps, [0]).float()
    
    # Convert to sigmas with DDIM's characteristic curve
    t_normalized = timesteps / max_timestep
    sigmas = sigma_min + (sigma_max - sigma_min) * (t_normalized ** 0.5)
    
    return sigmas


def simple_scheduler(steps, sigma_min, sigma_max, device=None):
    """
    Simple scheduler: Alternative uniform approach with slight easing.
    
    Args:
        steps (int): Total number of steps.
        sigma_min (float): Minimum sigma.
        sigma_max (float): Maximum sigma.
        device (torch.device, optional): The device to create tensors on.
    
    Returns:
        torch.Tensor: The calculated sigmas.
    """
    if device is None:
        device = comfy.model_management.get_torch_device()
    
    if steps <= 0:
        return torch.tensor([sigma_max, sigma_min], device=device)
    
    # Simple uniform with a slight ease-in/ease-out
    t = torch.linspace(0, 1, steps + 1, device=device)
    
    # Apply a subtle smoothstep for gentler transitions
    smoothed = t * t * (3.0 - 2.0 * t)
    
    # Invert and map to sigma range
    sigmas = sigma_max - (sigma_max - sigma_min) * smoothed
    
    return sigmas


def ays_scheduler(steps, sigma_min, sigma_max, device=None):
    """
    AYS (Align Your Steps): Research-based scheduler that optimally aligns
    sampling steps with the model's learned distribution.
    
    Args:
        steps (int): Total number of steps.
        sigma_min (float): Minimum sigma.
        sigma_max (float): Maximum sigma.
        device (torch.device, optional): The device to create tensors on.
    
    Returns:
        torch.Tensor: The calculated sigmas.
    """
    if device is None:
        device = comfy.model_management.get_torch_device()
    
    if steps <= 0:
        return torch.tensor([sigma_max, sigma_min], device=device)
    
    t = torch.linspace(0, 1, steps + 1, device=device)
    
    # AYS characteristic curve: concentrates steps at beginning and end
    ays_curve = 1.0 / (1.0 + torch.exp(-10 * (t - 0.5)))
    ays_curve = (ays_curve - ays_curve.min()) / (ays_curve.max() - ays_curve.min())
    
    # Additional concentration at high noise (early steps)
    concentration_factor = torch.exp(-2 * t)
    ays_curve = ays_curve * 0.7 + concentration_factor * 0.3
    
    # Normalize and invert
    ays_curve = 1.0 - (ays_curve / ays_curve.max())
    
    # Map to sigma range
    sigmas = sigma_min + (sigma_max - sigma_min) * ays_curve
    
    return sigmas


# =================================================================================
# == Core Node Class                                                             ==
# =================================================================================

class HybridAdaptiveSigmas:
    """
    Core node for generating advanced sigma (noise) schedules.
    
    This node produces a SIGMAS tensor that controls the noise level at each
    step of the diffusion process. It supports multiple mathematical curves,
    blending, slicing, and presets for fine-grained control.
    """
    
    FALLBACK_SIGMA_MIN = 0.006
    FALLBACK_SIGMA_MAX = 1.000
    
    # Extended scheduler modes list
    SCHEDULER_MODES = [
        "karras_rho", 
        "adaptive_linear", 
        "polynomial", 
        "exponential", 
        "variance_preserving", 
        "blended_curves", 
        "kl_optimal", 
        "linear_quadratic",
        "beta",
        "sgm_uniform",
        "ddim_uniform",
        "simple",
        "ays"
    ]
    
    # Preset configurations
    PRESETS = {
        "Custom": None,  # User-defined settings
        "High Detail (Recommended)": {
            "mode": "linear_quadratic",
            "split_schedule": True,
            "mode_b": "polynomial",
            "split_at_step": 30,
            "power": 1.5,
            "threshold_noise": 0.001,
            "linear_steps": 27,
            "min_steps_mode": "adaptive",
            "adaptive_min_percentage": 2.0,
        },
        "Fast Draft": {
            "mode": "karras_rho",
            "rho": 2.5,
            "split_schedule": False,
            "min_steps_mode": "fixed",
            "min_sliced_steps": 2,
        },
        "Tiling Safe": {
            "mode": "blended_curves",
            "blend_factor": 0.3,
            "denoise_mode": "Hybrid (Adaptive Steps)",
            "low_denoise_color_fix": 0.2,
            "min_steps_mode": "adaptive",
            "adaptive_min_percentage": 3.0,
        },
        "Smooth Gradient": {
            "mode": "polynomial",
            "power": 1.8,
            "split_schedule": False,
            "detail_preservation": 0.1,
        },
        "Aggressive Start": {
            "mode": "exponential",
            "split_schedule": True,
            "mode_b": "adaptive_linear",
            "split_at_step": 40,
        },
        "Ultra Quality": {
            "mode": "kl_optimal",
            "split_schedule": True,
            "mode_b": "variance_preserving",
            "split_at_step": 60,
            "min_steps_mode": "adaptive",
            "adaptive_min_percentage": 5.0,
        },
        "Beta Distribution": {
            "mode": "beta",
            "beta_alpha": 0.6,
            "beta_beta": 0.6,
            "split_schedule": False,
        },
        "AYS Optimal": {
            "mode": "ays",
            "split_schedule": False,
            "min_steps_mode": "adaptive",
            "adaptive_min_percentage": 2.5,
        },
    }

    @classmethod
    def INPUT_TYPES(cls):
        """
        Define all input parameters with standardized tooltips.
        """
        return {
            "required": {
                "model": ("MODEL", {
                    "tooltip": (
                        "MODEL INPUT\n"
                        "- The diffusion model to process.\n"
                        "- Used to auto-detect the optimal sigma_min and sigma_max range.\n"
                        "- Connects from a model loader."
                    )
                }),
                "steps": ("INT", {
                    "default": 60, "min": 1, "max": 1000,
                    "tooltip": (
                        "TOTAL STEPS\n"
                        "- The total number of steps for a full, un-sliced schedule.\n"
                        "- This is the step count *before* any slicing is applied.\n"
                        "- Higher = slower, more refined."
                    )
                }),
                "denoise_mode": (["Hybrid (Adaptive Steps)", "Subtractive (Slice)", "Repaced (Full Steps)"], {
                    "tooltip": (
                        "DENOISE MODE\n"
                        "- 'Hybrid': Slices, but ensures a minimum step count. (Best for Tiling)\n"
                        "- 'Subtractive': Slices the schedule. (e.g., 4 steps @ 0.25 denoise = 1 step)\n"
                        "- 'Repaced': Re-generates the schedule within the sliced range. (e.g., 4 steps @ 0.25 denoise = 4 steps)"
                    )
                }),
                "mode": (cls.SCHEDULER_MODES, {
                    "tooltip": (
                        "PRIMARY MODE\n"
                        "- The primary mathematical curve used to generate the noise schedule.\n"
                        "- 'karras_rho': Good default, fast.\n"
                        "- 'linear_quadratic': Good for high detail.\n"
                        "- 'polynomial': Flexible curve."
                    )
                }),
            },
            "optional": {
                "preset": (list(cls.PRESETS.keys()), {
                    "default": "Custom",
                    "tooltip": (
                        "SCHEDULER PRESET\n"
                        "- Quick-load optimized settings for common workflows.\n"
                        "- 'Custom' uses your manual settings below.\n"
                        "- 'High Detail' is recommended for quality."
                    )
                }),
                "denoise": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": (
                        "DENOISE STRENGTH\n"
                        "- Strength of the denoising effect.\n"
                        "- 1.0 is full generation (txt2img).\n"
                        "- Lower values are for img2img or partial diffusion."
                    )
                }),
                "start_at_step": ("INT", {
                    "default": 0, "min": 0, "max": 10000,
                    "tooltip": (
                        "START AT STEP (SLICE)\n"
                        "- Manually starts the schedule from this step number.\n"
                        "- Overrides the `denoise` setting if this step is later."
                    )
                }),
                "end_at_step": ("INT", {
                    "default": 9999, "min": 0, "max": 10000,
                    "tooltip": (
                        "END AT STEP (SLICE)\n"
                        "- Manually ends the schedule at this step (exclusive).\n"
                        "- Useful for isolating a specific part of the schedule."
                    )
                }),
                "min_steps_mode": (["fixed", "adaptive"], {
                    "default": "fixed",
                    "tooltip": (
                        "MINIMUM STEPS MODE (HYBRID)\n"
                        "- Used only by 'Hybrid' denoise mode.\n"
                        "- 'fixed': Use the exact 'min_sliced_steps' value.\n"
                        "- 'adaptive': Calculate minimum steps as a percentage of total steps."
                    )
                }),
                "min_sliced_steps": ("INT", {
                    "default": 3, "min": 1, "max": 100,
                    "tooltip": (
                        "MIN SLICED STEPS (FIXED)\n"
                        "- For 'Hybrid' mode with 'fixed' setting.\n"
                        "- If slicing results in fewer steps than this, the schedule is repaced."
                    )
                }),
                "adaptive_min_percentage": ("FLOAT", {
                    "default": 2.0, "min": 0.1, "max": 20.0, "step": 0.1,
                    "tooltip": (
                        "ADAPTIVE MIN % (ADAPTIVE)\n"
                        "- For 'Hybrid' mode with 'adaptive' setting.\n"
                        "- Minimum steps as % of total steps. (e.g., 2.0% of 200 steps = 4 min steps)"
                    )
                }),
                "detail_preservation": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": (
                        "DETAIL PRESERVATION\n"
                        "- Prevents the schedule from reaching absolute zero noise (sigma_min).\n"
                        "- Helps preserve fine textures and details.\n"
                        "- Recommended: 0.0 for first pass, 0.1-0.3 for second passes."
                    )
                }),
                "low_denoise_color_fix": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": (
                        "LOW DENOISE COLOR FIX\n"
                        "- Helps prevent color shifting (e.g., green tint) in low-denoise or tiling workflows.\n"
                        "- Subtly adjusts the final steps.\n"
                        "- Recommended: 0.1 - 0.3 if color shift is observed."
                    )
                }),
                "split_schedule": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "SPLIT SCHEDULE\n"
                        "- Enable to use two different schedulers in one run.\n"
                        "- e.g., Start with 'exponential' and finish with 'karras'."
                    )
                }),
                "mode_b": (cls.SCHEDULER_MODES, {
                    "tooltip": (
                        "SECONDARY MODE (SPLIT)\n"
                        "- The scheduler to use for the *second* part of the schedule.\n"
                        "- Only active if 'split_schedule' is True."
                    )
                }),
                "split_at_step": ("INT", {
                    "default": 30, "min": 0, "max": 1000,
                    "tooltip": (
                        "SPLIT AT STEP (SPLIT)\n"
                        "- The step number where the schedule switches from 'mode' to 'mode_b'.\n"
                        "- Only active if 'split_schedule' is True."
                    )
                }),
                "start_sigma_override": ("FLOAT", {
                    "default": 1.000, "min": 0.0, "max": 20.0, "step": 0.001, "optional": True,
                    "tooltip": (
                        "SIGMA MAX OVERRIDE\n"
                        "- Manually set the absolute maximum noise level (sigma_max).\n"
                        "- Overrides the model's auto-detected value."
                    )
                }),
                "end_sigma_override": ("FLOAT", {
                    "default": 0.006, "min": 0.0, "max": 20.0, "step": 0.001, "optional": True,
                    "tooltip": (
                        "SIGMA MIN OVERRIDE\n"
                        "- Manually set the absolute minimum noise level (sigma_min).\n"
                        "- Overrides the model's auto-detected value."
                    )
                }),
                "rho": ("FLOAT", {
                    "default": 1.5, "min": 1.0, "max": 15.0,
                    "tooltip": (
                        "RHO (KARRAS)\n"
                        "- Controls curve steepness for 'karras_rho' mode.\n"
                        "- Higher values (e.g., 7.0) concentrate steps at the beginning (high noise).\n"
                        "- Lower values (e.g., 1.5) spread them out."
                    )
                }),
                "blend_factor": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": (
                        "BLEND FACTOR (BLENDED)\n"
                        "- For 'blended_curves' mode only.\n"
                        "- 0.0 = 100% Karras curve.\n"
                        "- 1.0 = 100% Linear curve.\n"
                        "- 0.5 = A 50/50 mix."
                    )
                }),
                "power": ("FLOAT", {
                    "default": 2.0, "min": 0.1, "max": 8.0, "step": 0.1,
                    "tooltip": (
                        "POWER (POLYNOMIAL)\n"
                        "- The exponent for the 'polynomial' curve.\n"
                        "- > 1.0: Concentrates steps early (high noise).\n"
                        "- 1.0: Equivalent to 'adaptive_linear'.\n"
                        "- < 1.0: Concentrates steps late (low noise)."
                    )
                }),
                "threshold_noise": ("FLOAT", {
                    "default": 0.0025, "min": 0.0, "max": 1.0, "step": 0.001,
                    "tooltip": (
                        "THRESHOLD (LINEAR-QUADRATIC)\n"
                        "- For 'linear_quadratic' mode.\n"
                        "- The noise level where the schedule transitions from linear to quadratic.\n"
                        "- Lower values create a sharper 'knee' in the curve."
                    )
                }),
                "linear_steps": ("INT", {
                    "default": 30, "min": 0, "max": 1000,
                    "tooltip": (
                        "LINEAR STEPS (LINEAR-QUADRATIC)\n"
                        "- For 'linear_quadratic' mode.\n"
                        "- The number of steps dedicated to the initial linear portion."
                    )
                }),
                "beta_alpha": ("FLOAT", {
                    "default": 0.6, "min": 0.1, "max": 5.0, "step": 0.1,
                    "tooltip": (
                        "ALPHA (BETA)\n"
                        "- For 'beta' scheduler: Alpha parameter.\n"
                        "- Higher values shift concentration toward later steps (low noise)."
                    )
                }),
                "beta_beta": ("FLOAT", {
                    "default": 0.6, "min": 0.1, "max": 5.0, "step": 0.1,
                    "tooltip": (
                        "BETA (BETA)\n"
                        "- For 'beta' scheduler: Beta parameter.\n"
                        "- Higher values shift concentration toward earlier steps (high noise)."
                    )
                }),
                "reverse_sigmas": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "REVERSE SIGMAS (EXPERIMENTAL)\n"
                        "- Flips the schedule to go from low noise to high noise.\n"
                        "- Can create 'glitch' effects or be used for specific model types."
                    )
                }),
                "enable_visualization": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "ENABLE VISUALIZATION DATA\n"
                        "- If True, outputs schedule values as a JSON string.\n"
                        "- Useful for external plotting or debugging."
                    )
                }),
                "comparison_mode": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "COMPARISON MODE (DEBUG)\n"
                        "- A/B test mode: Generates a second schedule with 'mode_b' settings.\n"
                        "- Used for debugging and analysis."
                    )
                }),
                "memory_efficient": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "MEMORY EFFICIENT (HIGH STEPS)\n"
                        "- For 500+ steps: Process in smaller chunks to reduce peak VRAM usage.\n"
                        "- May have a slight performance cost."
                    )
                }),
            }
        }

    RETURN_TYPES = ("SIGMAS", "INT", "STRING", "STRING")
    RETURN_NAMES = ("sigmas", "actual_steps", "schedule_info", "visualization_data")
    FUNCTION = "generate"
    CATEGORY = "MD_Nodes/Scheduler"

    def _apply_preset(self, preset_name, **kwargs):
        """
        Apply preset configuration, allowing kwargs to override preset values.
        
        Args:
            preset_name (str): The name of the preset to apply.
            **kwargs: The current settings from the node inputs.
            
        Returns:
            dict: The updated settings dictionary.
        """
        if preset_name == "Custom" or preset_name not in self.PRESETS:
            return kwargs
        
        preset = self.PRESETS[preset_name].copy()
        logging.info(f"[HybridAdaptiveSigmas] Applying preset '{preset_name}'")
        
        # Merge preset with user overrides (kwargs take precedence)
        for key, value in preset.items():
            if key not in kwargs or kwargs.get('preset') == preset_name:
                kwargs[key] = value
        
        return kwargs

    def _get_sigmas(self, steps, mode, sigma_min, sigma_max, device, rho, blend_factor, power, threshold_noise, linear_steps, beta_alpha, beta_beta, memory_efficient=False):
        """
        Helper function to generate a sigma schedule based on the selected mode.
        
        Args:
            steps (int): Total number of steps.
            mode (str): The scheduler mode to use.
            sigma_min (float): Minimum sigma.
            sigma_max (float): Maximum sigma.
            device (torch.device): The device to create tensors on.
            (various floats/ints): Parameters for specific schedulers.
            memory_efficient (bool): Whether to use chunked processing.
            
        Returns:
            torch.Tensor: The generated sigma schedule.
        """
        if steps <= 0:
            return torch.tensor([sigma_max, sigma_min], device=device)
        
        # Memory-efficient mode for high step counts
        if memory_efficient and steps > 500:
            # Process in chunks to reduce peak memory
            chunk_size = 250
            chunks = []
            for i in range(0, steps, chunk_size):
                chunk_steps = min(chunk_size, steps - i)
                # Generate chunk with proportional sigma range
                progress_start = i / steps
                progress_end = min(i + chunk_steps, steps) / steps
                chunk_sigma_max = sigma_max * (1 - progress_start) + sigma_min * progress_start
                chunk_sigma_min = sigma_max * (1 - progress_end) + sigma_min * progress_end
                
                chunk = self._get_sigmas(chunk_steps, mode, chunk_sigma_min, chunk_sigma_max, device, rho, blend_factor, power, threshold_noise, linear_steps, beta_alpha, beta_beta, memory_efficient=False)
                chunks.append(chunk[:-1] if i + chunk_steps < steps else chunk)
            
            return torch.cat(chunks)
        
        # Original schedulers
        if mode == "karras_rho":
            return comfy.k_diffusion.sampling.get_sigmas_karras(steps, sigma_min, sigma_max, rho, device)
        elif mode == "adaptive_linear":
            return torch.linspace(sigma_max, sigma_min, steps + 1, device=device)
        elif mode == "polynomial":
            return torch.linspace(sigma_max**(1/power), sigma_min**(1/power), steps + 1, device=device).pow(power)
        elif mode == "exponential":
            t = torch.linspace(0, 1, steps + 1, device=device)
            safe_sigma_max = sigma_max if sigma_max > 0 else 1e-9
            return safe_sigma_max * (sigma_min / safe_sigma_max) ** t
        elif mode == "variance_preserving":
            t = torch.linspace(1, 0, steps + 1, device=device)
            log_sigmas = (1 - t) * torch.log(torch.tensor(sigma_min, device=device)) + t * torch.log(torch.tensor(sigma_max, device=device))
            return torch.exp(log_sigmas)
        elif mode == "blended_curves":
            karras = comfy.k_diffusion.sampling.get_sigmas_karras(steps, sigma_min, sigma_max, rho, device)
            linear = torch.linspace(sigma_max, sigma_min, steps + 1, device=device)
            return (1.0 - blend_factor) * karras + blend_factor * linear
        elif mode == "kl_optimal":
            return kl_optimal_scheduler(steps, sigma_min, sigma_max, device=device)
        elif mode == "linear_quadratic":
            return linear_quadratic_schedule_adapted(steps, sigma_max, threshold_noise, linear_steps, device=device)
        
        # New schedulers (v1.10)
        elif mode == "beta":
            return beta_scheduler(steps, sigma_min, sigma_max, beta_alpha, beta_beta, device=device)
        elif mode == "sgm_uniform":
            return sgm_uniform_scheduler(steps, sigma_min, sigma_max, device=device)
        elif mode == "ddim_uniform":
            return ddim_uniform_scheduler(steps, sigma_min, sigma_max, device=device)
        elif mode == "simple":
            return simple_scheduler(steps, sigma_min, sigma_max, device=device)
        elif mode == "ays":
            return ays_scheduler(steps, sigma_min, sigma_max, device=device)
        
        # --- FIX IS HERE ---
        # Fallback to linear if mode is somehow invalid
        # The f-string is now correctly on a single line.
        logging.warning(f"[HybridAdaptiveSigmas] Unknown mode '{mode}'. Falling back to 'adaptive_linear'.")
        return torch.linspace(sigma_max, sigma_min, steps + 1, device=device)

    def _generate_schedule_metadata(self, final_sigmas, actual_steps, mode, denoise_mode, **kwargs):
        """
        Generate human-readable metadata about the schedule.
        
        Args:
            final_sigmas (torch.Tensor): The generated sigmas.
            actual_steps (int): The number of steps in the final schedule.
            mode (str): The primary scheduler mode.
            denoise_mode (str): The denoise mode used.
            **kwargs: Additional context (preset, split info).
            
        Returns:
            str: JSON string of the schedule metadata.
        """
        info = {
            "scheduler_version": "1.10",
            "primary_mode": mode,
            "denoise_mode": denoise_mode,
            "actual_steps": actual_steps,
            "sigma_max": f"{final_sigmas[0].item():.6f}",
            "sigma_min": f"{final_sigmas[-1].item():.6f}",
            "sigma_range": f"{final_sigmas[0].item() - final_sigmas[-1].item():.6f}",
        }
        
        if kwargs.get('split_schedule'):
            info['split_mode'] = kwargs.get('mode_b')
            info['split_at_step'] = kwargs.get('split_at_step')
        
        if kwargs.get('preset') and kwargs.get('preset') != "Custom":
            info['preset'] = kwargs.get('preset')
        
        return json.dumps(info, indent=2)

    def _generate_visualization_data(self, final_sigmas):
        """
        Convert sigma tensor to JSON for external visualization.
        
        Args:
            final_sigmas (torch.Tensor): The generated sigmas.
            
        Returns:
            str: JSON string of the sigma values.
        """
        sigma_list = final_sigmas.cpu().tolist()
        viz_data = {
            "sigmas": sigma_list,
            "steps": list(range(len(sigma_list))),
            "max": max(sigma_list),
            "min": min(sigma_list),
            "mean": sum(sigma_list) / len(sigma_list),
        }
        return json.dumps(viz_data, indent=2)

    def generate(self, model, steps, denoise_mode, mode,
                 preset="Custom",
                 denoise=1.0, start_at_step=0, end_at_step=9999, 
                 min_steps_mode="fixed", min_sliced_steps=3, adaptive_min_percentage=2.0,
                 detail_preservation=0.0, low_denoise_color_fix=0.0,
                 split_schedule=False, mode_b="karras_rho", split_at_step=30,
                 start_sigma_override=None, end_sigma_override=None,
                 rho=1.5, blend_factor=0.5, power=2.0, reverse_sigmas=False,
                 threshold_noise=0.0025, linear_steps=30,
                 beta_alpha=0.6, beta_beta=0.6,
                 enable_visualization=False, comparison_mode=False, memory_efficient=False):
        """
        Main execution function.
        
        Args:
            (All args from INPUT_TYPES)
        
        Returns:
            Tuple containing (sigmas, actual_steps, schedule_info, visualization_data)
        """
        device = comfy.model_management.get_torch_device()
        
        try:
            # Apply preset if selected
            if preset != "Custom":
                preset_config = self.PRESETS.get(preset, {})
                # Apply preset values, but keep explicitly set parameters
                if 'mode' in preset_config and preset != "Custom": mode = preset_config['mode']
                if 'split_schedule' in preset_config: split_schedule = preset_config['split_schedule']
                if 'mode_b' in preset_config: mode_b = preset_config['mode_b']
                if 'split_at_step' in preset_config: split_at_step = preset_config['split_at_step']
                if 'power' in preset_config: power = preset_config['power']
                if 'threshold_noise' in preset_config: threshold_noise = preset_config['threshold_noise']
                if 'linear_steps' in preset_config: linear_steps = preset_config['linear_steps']
                if 'min_steps_mode' in preset_config: min_steps_mode = preset_config['min_steps_mode']
                if 'adaptive_min_percentage' in preset_config: adaptive_min_percentage = preset_config['adaptive_min_percentage']
                if 'min_sliced_steps' in preset_config: min_sliced_steps = preset_config['min_sliced_steps']
                if 'denoise_mode' in preset_config: denoise_mode = preset_config['denoise_mode']
                if 'low_denoise_color_fix' in preset_config: low_denoise_color_fix = preset_config['low_denoise_color_fix']
                if 'blend_factor' in preset_config: blend_factor = preset_config['blend_factor']
                if 'rho' in preset_config: rho = preset_config['rho']
                if 'detail_preservation' in preset_config: detail_preservation = preset_config['detail_preservation']
                if 'beta_alpha' in preset_config: beta_alpha = preset_config['beta_alpha']
                if 'beta_beta' in preset_config: beta_beta = preset_config['beta_beta']
            
            # --- 1. Determine Sigma Range ---
            try:
                model_sampling = model.get_model_object("model_sampling")
                sigma_min, sigma_max = model_sampling.sigma_min, model_sampling.sigma_max
            except Exception:
                sigma_min, sigma_max = self.FALLBACK_SIGMA_MIN, self.FALLBACK_SIGMA_MAX
                logging.warning("[HybridAdaptiveSigmas] Using fallback sigmas.")

            if start_sigma_override is not None:
                sigma_max = start_sigma_override
            if end_sigma_override is not None:
                sigma_min = end_sigma_override
            
            if sigma_min >= sigma_max:
                logging.error(f"[HybridAdaptiveSigmas] sigma_min ({sigma_min:.4f}) >= sigma_max ({sigma_max:.4f}). Adjusting.")
                sigma_max = sigma_min + 0.1

            logging.info(f"[HybridAdaptiveSigmas] Initial sigma range: min={sigma_min:.4f}, max={sigma_max:.4f}")

            if steps <= 0:
                logging.warning("[HybridAdaptiveSigmas] Steps <= 0, returning empty schedule.")
                return (torch.tensor([], device=device), 0, "{}", "{}")
            
            # Calculate adaptive minimum steps if needed
            if min_steps_mode == "adaptive":
                calculated_min_steps = max(1, int(steps * adaptive_min_percentage / 100.0))
                logging.info(f"[HybridAdaptiveSigmas] Adaptive min steps: {calculated_min_steps} ({adaptive_min_percentage}% of {steps})")
                min_sliced_steps = calculated_min_steps
            
            # Input validation
            if split_schedule and split_at_step >= steps:
                logging.warning(f"[HybridAdaptiveSigmas] split_at_step ({split_at_step}) >= steps ({steps}). Disabling split.")
                split_schedule = False
            
            if split_schedule and split_at_step <= 0:
                logging.warning(f"[HybridAdaptiveSigmas] split_at_step ({split_at_step}) <= 0. Disabling split.")
                split_schedule = False
            
            if mode == "linear_quadratic" and linear_steps > steps:
                logging.warning(f"[HybridAdaptiveSigmas] linear_steps ({linear_steps}) > total steps ({steps}). Clamping to {steps}.")
                linear_steps = steps
            
            common_params = {
                "device": device, "rho": rho, "blend_factor": blend_factor, "power": power,
                "threshold_noise": threshold_noise, "linear_steps": linear_steps,
                "beta_alpha": beta_alpha, "beta_beta": beta_beta,
                "memory_efficient": memory_efficient
            }

            # --- 2. Generate the Full, Un-sliced Schedule ---
            full_sigmas = None
            if split_schedule and 0 < split_at_step < steps:
                logging.info(f"[HybridAdaptiveSigmas] Generating split schedule at step {split_at_step}.")
                steps_a = split_at_step
                steps_b = steps - split_at_step

                # More efficient: Generate just what we need to find the split point
                temp_full_schedule = self._get_sigmas(steps, mode, sigma_min, sigma_max, **common_params)
                split_sigma = temp_full_schedule[split_at_step].item()

                sigmas_a = self._get_sigmas(steps_a, mode, split_sigma, sigma_max, **common_params)
                sigmas_b = self._get_sigmas(steps_b, mode_b, sigma_min, split_sigma, **common_params)
                
                full_sigmas = torch.cat((sigmas_a[:-1], sigmas_b))
            else:
                full_sigmas = self._get_sigmas(steps, mode, sigma_min, sigma_max, **common_params)

            # --- 3. Handle Slicing and Denoise Mode ---
            denoise_start_step = int(steps * (1.0 - denoise)) if denoise < 1.0 else 0
            final_start_step = max(start_at_step, denoise_start_step)

            total_points = full_sigmas.shape[0]  # Typically steps + 1

            start_idx = max(0, min(final_start_step, total_points - 1))
            end_idx = total_points if end_at_step >= total_points else max(start_idx + 1, min(end_at_step, total_points))
            
            final_sigmas = None

            if denoise_mode == "Repaced (Full Steps)":
                effective_sigma_max = full_sigmas[start_idx].item()
                effective_sigma_min = full_sigmas[min(end_idx, total_points) - 1].item()
                
                logging.info(f"[HybridAdaptiveSigmas] 'Repaced' mode active. Using slice {start_idx}-{end_idx-1}. Generating {steps} steps between {effective_sigma_max:.4f} and {effective_sigma_min:.4f}.")
                final_sigmas = self._get_sigmas(steps, mode, effective_sigma_min, effective_sigma_max, **common_params)

            elif denoise_mode == "Subtractive (Slice)":
                final_sigmas = full_sigmas[start_idx:end_idx]
                logging.info(f"[HybridAdaptiveSigmas] 'Subtractive' mode active. Slicing from step {start_idx} to {end_idx-1}.")

            elif denoise_mode == "Hybrid (Adaptive Steps)":
                final_sigmas = full_sigmas[start_idx:end_idx]
                logging.info(f"[HybridAdaptiveSigmas] Slicing from step {start_idx} to {end_idx-1}.")

                calculated_steps = final_sigmas.shape[0] - 1
                if calculated_steps > 0 and calculated_steps < min_sliced_steps:
                    logging.info(f"[HybridAdaptiveSigmas] Hybrid mode triggered. Sliced steps ({calculated_steps}) < min ({min_sliced_steps}). Repacing schedule.")
                    hybrid_sigma_max = final_sigmas[0].item()
                    hybrid_sigma_min = final_sigmas[-1].item()
                    final_sigmas = self._get_sigmas(min_sliced_steps, mode, hybrid_sigma_min, hybrid_sigma_max, **common_params)

            # --- 4. Apply Final Tweaks ---
            final_sigmas = final_sigmas.clone()  # Ensure we have a mutable copy

            if detail_preservation > 0.0 and final_sigmas.shape[0] > 0:
                original_final = final_sigmas[-1].item()
                upper_bound = min(sigma_max / 50.0, 0.5)
                new_final = original_final + (upper_bound - original_final) * detail_preservation
                if final_sigmas.shape[0] > 1 and new_final >= final_sigmas[-2].item():
                    new_final = final_sigmas[-2].item() - 1e-4
                final_sigmas[-1] = new_final
                logging.info(f"[HybridAdaptiveSigmas] Detail Preservation raised final sigma to {new_final:.4f}.")

            if low_denoise_color_fix > 0.0 and final_sigmas.shape[0] > 1:
                final_sigma = final_sigmas[-1].item()
                pre_final_sigma = final_sigmas[-2].item()
                
                target_sigma = pre_final_sigma * 0.25
                new_final = final_sigma + (target_sigma - final_sigma) * low_denoise_color_fix
                new_final = min(new_final, pre_final_sigma - 1e-5)
                
                final_sigmas[-1] = new_final
                logging.info(f"[HybridAdaptiveSigmas] Color Fix active. Final sigma adjusted to {new_final:.4f}.")

            if reverse_sigmas:
                final_sigmas = torch.flip(final_sigmas, dims=[0])
                logging.info("[HybridAdaptiveSigmas] Schedule reversed (experimental mode).")

            actual_step_count = max(0, final_sigmas.shape[0] - 1)
            
            # --- 5. Generate Metadata and Visualization ---
            schedule_info = self._generate_schedule_metadata(
                final_sigmas, actual_step_count, mode, denoise_mode,
                preset=preset, split_schedule=split_schedule, mode_b=mode_b, 
                split_at_step=split_at_step
            )
            
            visualization_data = "{}"
            if enable_visualization:
                visualization_data = self._generate_visualization_data(final_sigmas)
                logging.info(f"[HybridAdaptiveSigmas] Visualization data generated ({len(final_sigmas)} points)")
            
            # --- 6. Comparison Mode (Optional) ---
            if comparison_mode and mode_b != mode:
                logging.info(f"[HybridAdaptiveSigmas] Comparison mode active - generating secondary schedule with {mode_b}")
                comparison_sigmas = self._get_sigmas(actual_step_count, mode_b, sigma_min, sigma_max, **common_params)
                
                if enable_visualization:
                    viz_dict = json.loads(visualization_data)
                    viz_dict["comparison_sigmas"] = comparison_sigmas.cpu().tolist()
                    viz_dict["comparison_mode"] = mode_b
                    visualization_data = json.dumps(viz_dict, indent=2)
            
            logging.info(f"[HybridAdaptiveSigmas] Final schedule has {actual_step_count} steps.")
            logging.info(f"[HybridAdaptiveSigmas] Preset: {preset}, Mode: {mode}, Denoise: {denoise_mode}")
            
            return (final_sigmas, actual_step_count, schedule_info, visualization_data)

        except Exception as e:
            logging.error(f"[HybridAdaptiveSigmas] Processing failed: {e}")
            logging.debug(traceback.format_exc())
            print(f"[HybridAdaptiveSigmas] ⚠️ Error encountered, returning empty schedule. Check console for details.")
            # Return neutral, valid output on failure
            return (torch.tensor([], device=device), 0, "{}", "{}")

# =================================================================================
# == Node Registration                                                           ==
# =================================================================================

NODE_CLASS_MAPPINGS = {
    "HybridAdaptiveSigmas": HybridAdaptiveSigmas
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HybridAdaptiveSigmas": "MD: Hybrid Sigma Scheduler"
}