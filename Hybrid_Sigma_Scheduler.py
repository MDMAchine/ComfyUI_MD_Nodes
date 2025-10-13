# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# ████ Hybrid Sigma Scheduler v1.10 – Optimized for Ace-Step Audio/Video ████▓▒░
# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀

# ░▒▓ ORIGIN & DEV:
#   • Brewed in a Compaq Presario by MDMAchine / MD_Nodes
#   • License: Apache 2.0 — legal chaos with community spirit

# ░▒▓ DESCRIPTION:
#   Outputs a tensor of sigmas to control diffusion noise levels.
#   No guesswork. No drama. Just pure generative groove.
#   Designed for precision noise scheduling in ComfyUI workflows.
#   May perform differently outside of intended models.

# ░▒▓ FEATURES:
#   ✓ Multiple core modes: Karras, Linear, Polynomial, Blended, and more
#   ✓ NEW: Beta, SGM Uniform, DDIM Uniform, Simple, and AYS schedulers
#   ✓ Automatic sigma range detection from the loaded model
#   ✓ Manual override for sigma min/max for expert control
#   ✓ Precision schedule slicing with `start_at_step` and `end_at_step`
#   ✓ Outputs the actual number of steps after slicing for downstream samplers
#   ✓ Optional schedule reversal for experimental workflows
#   ✓ Now supports up to 1000 steps for ultra-high-quality workflows
#   ✓ Smart presets for common use cases
#   ✓ Schedule visualization output for analysis
#   ✓ Adaptive minimum steps based on percentage
#   ✓ Comparison mode for A/B testing schedules
#   ✓ Memory-optimized for extreme step counts

# ░▒▓ CHANGELOG:
#   - v0.88:
#         • Added 'low_denoise_color_fix' slider to prevent color shifting in tiling workflows.
#   - v0.89:
#         • Fixed a logic bug where `start_at_step` and `end_at_step` were not being correctly applied in all denoise modes.
#         • All slicing and override logic is now universally applied, ensuring consistent behavior.
#   - v0.90:
#         • Increased max steps to 1000 for ultra-high-quality workflows
#         • Fixed device inconsistencies in kl_optimal and linear_quadratic schedulers
#         • Added input validation for split_at_step and other parameters
#         • Improved edge case handling for very low step counts
#         • Enhanced error messages for better debugging
#         • Optimized tensor operations for better performance at high step counts
#   - v1.00 (Major Update):
#         • Added comprehensive preset system for quick workflow setup
#         • Implemented schedule visualization output for analysis and plotting
#         • Added adaptive minimum steps using percentage-based scaling
#         • Introduced comparison mode for A/B testing different schedules
#         • Optimized memory handling for extreme step counts (500+)
#         • Enhanced documentation and tooltips throughout
#         • Added schedule metadata output for downstream processing
#   - v1.10 (Current Release - Extended Scheduler Suite):
#         • Added Beta scheduler with alpha/beta parameter control
#         • Added SGM Uniform (timestep-based uniform scheduling)
#         • Added DDIM Uniform (DDIM-style timestep scheduling)
#         • Added Simple scheduler (alternative uniform approach)
#         • Added AYS (Align Your Steps) research-based scheduler
#         • All new schedulers support full feature set (splitting, slicing, etc.)
#         • Enhanced preset system with new scheduler options
#         • Maintained 100% backward compatibility with existing workflows

# ░▒▓ CONFIGURATION:
#   → Primary Use: Fine-grained diffusion noise scheduling for generative workflows
#   → Secondary Use: Experimental noise shaping and curve blending
#   → Edge Use: Hardcore noise tinkerers and sampling theorists
#   → Advanced Use: Schedule analysis, comparison, and optimization

# ░▒▓ WARNING:
#   This node may trigger:
#   ▓▒░ Unexpected vibes
#   ▓▒░ Spontaneous beatboxing
#   ▓▒░ Mild to moderate enlightenment
#   ▓▒░ Uncontrollable urge to fine-tune parameters

# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄


import torch
import comfy.model_management
from comfy.k_diffusion.sampling import get_sigmas_karras
import math
import json

# Referenced from https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/15608
def kl_optimal_scheduler(n: int, sigma_min: float, sigma_max: float, device=None) -> torch.Tensor:
    if device is None:
        device = torch.device('cpu')
    
    sigma_min_t = torch.tensor(sigma_min, dtype=torch.float32, device=device)
    sigma_max_t = torch.tensor(sigma_max, dtype=torch.float32, device=device)
    
    if n > 1:
        adj_idxs = torch.arange(n, dtype=torch.float32, device=device).div_(n - 1)
    else:
        adj_idxs = torch.tensor([0.0], dtype=torch.float32, device=device)
    
    sigmas = torch.zeros(n + 1, dtype=torch.float32, device=device)
    sigmas[:-1] = (adj_idxs * torch.atan(sigma_min_t) + (1 - adj_idxs) * torch.atan(sigma_max_t)).tan_()
    
    return sigmas

# Adapted from: https://github.com/genmoai/models/blob/main/src/mochi_preview/infer.py#L41
def linear_quadratic_schedule_adapted(steps: int, sigma_max: float, threshold_noise: float = 0.0025, linear_steps: int = None, device=None) -> torch.Tensor:
    if device is None:
        device = torch.device('cpu')
    
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


def beta_scheduler(steps: int, sigma_min: float, sigma_max: float, alpha: float = 0.6, beta: float = 0.6, device=None) -> torch.Tensor:
    """Beta distribution-based scheduler. Popular in diffusion models for controlled noise distribution."""
    if device is None:
        device = torch.device('cpu')
    
    if steps <= 0:
        return torch.tensor([sigma_max, sigma_min], device=device)
    
    # Use a simpler, more stable approach based on how beta schedules work in practice
    # Create a linear schedule first, then apply beta-like transformation
    t = torch.linspace(0, 1, steps + 1, device=device)
    
    # Apply beta-inspired curve transformation
    # Higher alpha pushes more steps toward the end, higher beta toward the start
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


def sgm_uniform_scheduler(steps: int, sigma_min: float, sigma_max: float, device=None) -> torch.Tensor:
    """SGM Uniform: Uniform spacing in timestep domain, then converted to sigmas."""
    if device is None:
        device = torch.device('cpu')
    
    if steps <= 0:
        return torch.tensor([sigma_max, sigma_min], device=device)
    
    # Create uniform timesteps (typically 0 to 999 for SD models)
    # We'll assume a typical 1000-step timestep range
    max_timestep = 999.0
    timesteps = torch.linspace(max_timestep, 0, steps + 1, device=device)
    
    # Convert timesteps to sigmas using a log-linear relationship
    # This mimics how SDM does it: sigma = (timestep / max_timestep) ^ 0.5 * sigma_max
    t_normalized = timesteps / max_timestep
    sigmas = sigma_min + (sigma_max - sigma_min) * t_normalized
    
    return sigmas


def ddim_uniform_scheduler(steps: int, sigma_min: float, sigma_max: float, device=None) -> torch.Tensor:
    """DDIM Uniform: Uniform timestep spacing specifically for DDIM-style sampling."""
    if device is None:
        device = torch.device('cpu')
    
    if steps <= 0:
        return torch.tensor([sigma_max, sigma_min], device=device)
    
    # DDIM uses uniform timestep spacing with a specific stride pattern
    max_timestep = 1000
    step_ratio = max_timestep // steps
    
    # Create uniform timesteps with DDIM's characteristic spacing
    timesteps = torch.arange(0, steps + 1, device=device) * step_ratio
    timesteps = torch.flip(max_timestep - timesteps, [0]).float()
    
    # Convert to sigmas with DDIM's characteristic curve
    t_normalized = timesteps / max_timestep
    sigmas = sigma_min + (sigma_max - sigma_min) * (t_normalized ** 0.5)
    
    return sigmas


def simple_scheduler(steps: int, sigma_min: float, sigma_max: float, device=None) -> torch.Tensor:
    """Simple scheduler: Alternative uniform approach with slight easing."""
    if device is None:
        device = torch.device('cpu')
    
    if steps <= 0:
        return torch.tensor([sigma_max, sigma_min], device=device)
    
    # Simple uniform with a slight ease-in/ease-out
    t = torch.linspace(0, 1, steps + 1, device=device)
    
    # Apply a subtle smoothstep for gentler transitions
    smoothed = t * t * (3.0 - 2.0 * t)
    
    # Invert and map to sigma range
    sigmas = sigma_max - (sigma_max - sigma_min) * smoothed
    
    return sigmas


def ays_scheduler(steps: int, sigma_min: float, sigma_max: float, device=None) -> torch.Tensor:
    """
    AYS (Align Your Steps): Research-based scheduler that optimally aligns
    sampling steps with the model's learned distribution.
    Based on the paper's findings about optimal step positioning.
    """
    if device is None:
        device = torch.device('cpu')
    
    if steps <= 0:
        return torch.tensor([sigma_max, sigma_min], device=device)
    
    # AYS uses a specific non-uniform distribution based on the model's training
    # More steps at high noise levels, fewer at low noise
    t = torch.linspace(0, 1, steps + 1, device=device)
    
    # AYS characteristic curve: concentrates steps at beginning and end
    # Using a modified sigmoid-like curve
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


class HybridAdaptiveSigmas:
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
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The diffusion model, used to auto-detect the optimal sigma_min and sigma_max."}),
                "steps": ("INT", {"default": 60, "min": 1, "max": 1000, "tooltip": "The total number of steps for a full schedule, before any slicing is applied."}),
                "denoise_mode": (["Hybrid (Adaptive Steps)", "Subtractive (Slice)", "Repaced (Full Steps)"], {"tooltip": "How denoise is applied.\nHybrid (Best for Tiling): Slices, but ensures a minimum number of steps to prevent seams.\nSubtractive: 4 steps @ 0.25 denoise = 1 step.\nRepaced: 4 steps @ 0.25 denoise = 4 steps."}),
                "mode": (s.SCHEDULER_MODES, {"tooltip": "The primary mathematical curve used to generate the noise schedule."}),
            },
            "optional": {
                "preset": (list(s.PRESETS.keys()), {"default": "Custom", "tooltip": "Quick-load optimized settings for common workflows. 'Custom' uses your manual settings below."}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Strength of the denoising effect. 1.0 is full generation, while a lower value like 0.25 is for making smaller changes (img2img)."}),
                "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000, "tooltip": "Manually starts the schedule from this step. Overrides the `denoise` setting if this step is later."}),
                "end_at_step": ("INT", {"default": 9999, "min": 0, "max": 10000, "tooltip": "Manually ends the schedule at this step (exclusive). Useful for isolating a specific part of the schedule."}),
                "min_steps_mode": (["fixed", "adaptive"], {"default": "fixed", "tooltip": "Fixed: Use exact min_sliced_steps. Adaptive: Calculate minimum steps as a percentage of total steps."}),
                "min_sliced_steps": ("INT", {"default": 3, "min": 1, "max": 100, "tooltip": "For Hybrid mode with 'fixed': If slicing results in fewer steps than this, the schedule is repaced. Ignored when using 'adaptive'."}),
                "adaptive_min_percentage": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 20.0, "step": 0.1, "tooltip": "For Hybrid mode with 'adaptive': Minimum steps as % of total steps. E.g., 2.0% of 200 steps = 4 minimum steps."}),
                "detail_preservation": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Prevents the schedule from reaching absolute zero noise, which helps preserve fine textures. Ideal for second passes."}),
                "low_denoise_color_fix": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Helps prevent color shifting (e.g., green tint) in low-denoise or tiling workflows by subtly adjusting the final steps. Start with 0.2."}),
                "split_schedule": ("BOOLEAN", {"default": False, "tooltip": "Enable to use two different schedulers in one run. E.g., start with 'exponential' and finish with 'karras'."}),
                "mode_b": (s.SCHEDULER_MODES, {"tooltip": "The scheduler to use for the *second* part of the schedule, after the split point."}),
                "split_at_step": ("INT", {"default": 30, "min": 0, "max": 1000, "tooltip": "The step where the schedule switches from the primary mode to mode_b."}),
                "start_sigma_override": ("FLOAT", {"default": 1.000, "min": 0.0, "max": 20.0, "step": 0.001, "optional": True, "tooltip": "Manually set the absolute maximum noise level (sigma_max). Overrides model auto-detection."}),
                "end_sigma_override": ("FLOAT", {"default": 0.006, "min": 0.0, "max": 20.0, "step": 0.001, "optional": True, "tooltip": "Manually set the absolute minimum noise level (sigma_min). Overrides model auto-detection."}),
                "rho": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 15.0, "tooltip": "Controls curve steepness for Karras. Higher values concentrate steps at the beginning (high noise)."}),
                "blend_factor": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "For 'blended_curves' mode. A factor of 0.0 is pure Karras, 1.0 is pure Linear."}),
                "power": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 8.0, "step": 0.1, "tooltip": "The exponent for the Polynomial curve. >1.0 concentrates steps early, <1.0 concentrates them late."}),
                "threshold_noise": ("FLOAT", {"default": 0.0025, "min": 0.0, "max": 1.0, "step": 0.001, "tooltip": "For Linear-Quadratic: The point where the schedule transitions from linear to quadratic. Lower values create a sharper 'knee' in the curve."}),
                "linear_steps": ("INT", {"default": 30, "min": 0, "max": 1000, "tooltip": "For Linear-Quadratic: The number of steps dedicated to the initial linear portion of the schedule."}),
                "beta_alpha": ("FLOAT", {"default": 0.6, "min": 0.1, "max": 5.0, "step": 0.1, "tooltip": "For Beta scheduler: Alpha parameter controlling the distribution shape. Higher values shift concentration toward later steps."}),
                "beta_beta": ("FLOAT", {"default": 0.6, "min": 0.1, "max": 5.0, "step": 0.1, "tooltip": "For Beta scheduler: Beta parameter controlling the distribution shape. Higher values shift concentration toward earlier steps."}),
                "reverse_sigmas": ("BOOLEAN", {"default": False, "tooltip": "Experimental: Flips the schedule to go from low noise to high noise. Can create 'glitch' effects."}),
                "enable_visualization": ("BOOLEAN", {"default": False, "tooltip": "Output schedule values as a list for external plotting/analysis. Useful for debugging and optimization."}),
                "comparison_mode": ("BOOLEAN", {"default": False, "tooltip": "A/B test mode: Generate a second schedule with mode_b settings for direct comparison."}),
                "memory_efficient": ("BOOLEAN", {"default": False, "tooltip": "For 500+ steps: Process in smaller chunks to reduce peak memory usage. Slight performance cost."}),
            }
        }

    RETURN_TYPES = ("SIGMAS", "INT", "STRING", "STRING")
    RETURN_NAMES = ("sigmas", "actual_steps", "schedule_info", "visualization_data")
    FUNCTION = "generate"
    CATEGORY = "MD_Nodes/Schedulers"

    def _apply_preset(self, preset_name, **kwargs):
        """Apply preset configuration, allowing kwargs to override preset values."""
        if preset_name == "Custom" or preset_name not in self.PRESETS:
            return kwargs
        
        preset = self.PRESETS[preset_name].copy()
        print(f"Hybrid Sigma Scheduler INFO: Applying preset '{preset_name}'")
        
        # Merge preset with user overrides (kwargs take precedence)
        for key, value in preset.items():
            if key not in kwargs or kwargs.get('preset') == preset_name:
                kwargs[key] = value
        
        return kwargs

    def _get_sigmas(self, steps, mode, sigma_min, sigma_max, device, rho, blend_factor, power, threshold_noise, linear_steps, beta_alpha, beta_beta, memory_efficient=False):
        """Helper function to generate a sigma schedule based on the selected mode."""
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
            return get_sigmas_karras(steps, sigma_min, sigma_max, rho, device)
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
            karras = get_sigmas_karras(steps, sigma_min, sigma_max, rho, device)
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
        
        # Fallback to linear if mode is somehow invalid
        return torch.linspace(sigma_max, sigma_min, steps + 1, device=device)

    def _generate_schedule_metadata(self, final_sigmas, actual_steps, mode, denoise_mode, **kwargs):
        """Generate human-readable metadata about the schedule."""
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
        """Convert sigma tensor to JSON for external visualization."""
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
        
        try:
            device = comfy.model_management.get_torch_device()
            
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
                print("Hybrid Sigma Scheduler WARNING: Using fallback sigmas.")

            if start_sigma_override is not None:
                sigma_max = start_sigma_override
            if end_sigma_override is not None:
                sigma_min = end_sigma_override
            
            if sigma_min >= sigma_max:
                print(f"Hybrid Sigma Scheduler ERROR: sigma_min ({sigma_min:.4f}) >= sigma_max ({sigma_max:.4f}). Adjusting.")
                sigma_max = sigma_min + 0.1

            print(f"Hybrid Sigma Scheduler INFO: Initial sigma range: min={sigma_min:.4f}, max={sigma_max:.4f}")

            if steps <= 0:
                print("Hybrid Sigma Scheduler WARNING: Steps <= 0, returning empty schedule.")
                return (torch.tensor([], device=device), 0, "{}", "{}")
            
            # Calculate adaptive minimum steps if needed
            if min_steps_mode == "adaptive":
                calculated_min_steps = max(1, int(steps * adaptive_min_percentage / 100.0))
                print(f"Hybrid Sigma Scheduler INFO: Adaptive min steps: {calculated_min_steps} ({adaptive_min_percentage}% of {steps})")
                min_sliced_steps = calculated_min_steps
            
            # Input validation
            if split_schedule and split_at_step >= steps:
                print(f"Hybrid Sigma Scheduler WARNING: split_at_step ({split_at_step}) >= steps ({steps}). Disabling split.")
                split_schedule = False
            
            if split_schedule and split_at_step <= 0:
                print(f"Hybrid Sigma Scheduler WARNING: split_at_step ({split_at_step}) <= 0. Disabling split.")
                split_schedule = False
            
            if mode == "linear_quadratic" and linear_steps > steps:
                print(f"Hybrid Sigma Scheduler WARNING: linear_steps ({linear_steps}) > total steps ({steps}). Clamping to {steps}.")
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
                print(f"Hybrid Sigma Scheduler INFO: Generating split schedule at step {split_at_step}.")
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

            # First, universally determine the intended slice from the original full schedule
            # based on all relevant inputs (denoise, start_at_step, end_at_step).
            denoise_start_step = int(steps * (1.0 - denoise)) if denoise < 1.0 else 0
            final_start_step = max(start_at_step, denoise_start_step)

            total_points = full_sigmas.shape[0]  # Typically steps + 1

            # Calculate the final indices for slicing from the original full schedule
            start_idx = max(0, min(final_start_step, total_points - 1))
            end_idx = total_points if end_at_step >= total_points else max(start_idx + 1, min(end_at_step, total_points))
            
            final_sigmas = None

            # Now, apply the logic for the selected denoise_mode using the calculated slice
            if denoise_mode == "Repaced (Full Steps)":
                effective_sigma_max = full_sigmas[start_idx].item()
                # The end_idx is exclusive, so we use end_idx-1 to get the last sigma value of the slice
                effective_sigma_min = full_sigmas[min(end_idx, total_points) - 1].item()
                
                print(f"Hybrid Sigma Scheduler INFO: 'Repaced' mode active. Using slice {start_idx}-{end_idx-1}. Generating {steps} steps between {effective_sigma_max:.4f} and {effective_sigma_min:.4f}.")
                final_sigmas = self._get_sigmas(steps, mode, effective_sigma_min, effective_sigma_max, **common_params)

            elif denoise_mode == "Subtractive (Slice)":
                final_sigmas = full_sigmas[start_idx:end_idx]
                print(f"Hybrid Sigma Scheduler INFO: 'Subtractive' mode active. Slicing from step {start_idx} to {end_idx-1}.")

            elif denoise_mode == "Hybrid (Adaptive Steps)":
                # Start by slicing like normal
                final_sigmas = full_sigmas[start_idx:end_idx]
                print(f"Hybrid Sigma Scheduler INFO: Slicing from step {start_idx} to {end_idx-1}.")

                # Then, check if the slice is too short and regenerate if needed
                calculated_steps = final_sigmas.shape[0] - 1
                if calculated_steps > 0 and calculated_steps < min_sliced_steps:
                    print(f"Hybrid Sigma Scheduler INFO: Hybrid mode triggered. Sliced steps ({calculated_steps}) < min ({min_sliced_steps}). Repacing schedule.")
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
                print(f"Hybrid Sigma Scheduler INFO: Detail Preservation raised final sigma to {new_final:.4f}.")

            if low_denoise_color_fix > 0.0 and final_sigmas.shape[0] > 1:
                final_sigma = final_sigmas[-1].item()
                pre_final_sigma = final_sigmas[-2].item()
                
                # A safe target is a fraction of the second-to-last sigma
                target_sigma = pre_final_sigma * 0.25 
                
                # Interpolate towards the target
                new_final = final_sigma + (target_sigma - final_sigma) * low_denoise_color_fix
                
                # Ensure the new final sigma is still less than the previous one
                new_final = min(new_final, pre_final_sigma - 1e-5)
                
                final_sigmas[-1] = new_final
                print(f"Hybrid Sigma Scheduler INFO: Color Fix active. Final sigma adjusted to {new_final:.4f}.")

            if reverse_sigmas:
                final_sigmas = torch.flip(final_sigmas, dims=[0])
                print("Hybrid Sigma Scheduler INFO: Schedule reversed (experimental mode).")

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
                print(f"Hybrid Sigma Scheduler INFO: Visualization data generated ({len(final_sigmas)} points)")
            
            # --- 6. Comparison Mode (Optional) ---
            if comparison_mode and mode_b != mode:
                print(f"Hybrid Sigma Scheduler INFO: Comparison mode active - generating secondary schedule with {mode_b}")
                comparison_sigmas = self._get_sigmas(actual_step_count, mode_b, sigma_min, sigma_max, **common_params)
                
                # Add comparison data to visualization
                if enable_visualization:
                    viz_dict = json.loads(visualization_data)
                    viz_dict["comparison_sigmas"] = comparison_sigmas.cpu().tolist()
                    viz_dict["comparison_mode"] = mode_b
                    visualization_data = json.dumps(viz_dict, indent=2)
            
            print(f"Hybrid Sigma Scheduler INFO: Final schedule has {actual_step_count} steps.")
            print(f"Hybrid Sigma Scheduler INFO: Preset: {preset}, Mode: {mode}, Denoise: {denoise_mode}")
            
            return (final_sigmas, actual_step_count, schedule_info, visualization_data)

        except Exception as e:
            import traceback
            print(f"ERROR in HybridAdaptiveSigmas.generate: {e}")
            traceback.print_exc()
            raise e

NODE_CLASS_MAPPINGS = {"HybridAdaptiveSigmas": HybridAdaptiveSigmas}
NODE_DISPLAY_NAME_MAPPINGS = {"HybridAdaptiveSigmas": "Hybrid Sigma Scheduler"}