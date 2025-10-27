# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
# ████ MD_Nodes/PingPongSamplerNodeFBG – Advanced sampler with Feedback Guidance ████▓▒░
# © 2025 MDMAchine
# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
# ░▒▓ ORIGIN & DEV:
#   • Cast into the void by: Junmin Gong (Concept), blepping (Port), MDMAchine (Adaptation)
#   • Enhanced by: Gemini, Claude, devstral/qwen3
#   • License: Apache 2.0 — Sharing is caring
#   • Original source (if applicable): https://gist.github.com/blepping/b372ef6c5412080af136aad942d9d76c
#   • Academic sources: Song et al. (2023) [arXiv:2303.01469], Novack et al. (2025) [arXiv:2505.08175]

# ░▒▓ DESCRIPTION:
#   Advanced ancestral sampler combining PingPong's noise mixing with Feedback
#   Guidance (FBG) for dynamic, content-aware guidance scaling. Designed for
#   precision audio/video generation in ComfyUI workflows.

# ░▒▓ FEATURES:
#   ✓ Feedback Guidance (FBG) for dynamic guidance scaling.
#   ✓ Multiple blend modes (lerp, slerp, cosine, cubic, add) with numerical stability.
#   ✓ Ancestral noise types: gaussian, uniform, brownian.
#   ✓ Conditional blending based on sigma thresholds or step changes.
#   ✓ Progressive blend mode adapting through sampling.
#   ✓ Adaptive noise scaling based on denoising progress.
#   ✓ NaN/Inf detection with automatic recovery.

# ░▒▓ CHANGELOG:
#   - v1.4.2 (Guideline Update - Oct 2025):
#       • REFACTOR: Full compliance update to v1.4.2 guidelines.
#       • CRITICAL: Removed all type hints from function signatures to prevent ComfyUI loading crashes.
#       • CRITICAL: Refactored FBGConfig from NamedTuple to a standard class per guidelines.
#       • FIXED: Corrected NODE_CLASS_MAPPINGS key mismatch (was "PingPongSampler_Custom_FBG").
#       • STYLE: Standardized imports, docstrings, logging, and error handling.
#       • STYLE: Rewrote all tooltips to new standard format.
#       • ROBUST: Wrapped main sampling loop in try/except for graceful failure.
#       • ROBUST: Updated profiler to use comfy.model_management for device handling.
#   - v0.9.9-p4 (Stability Fix):
#       • FIXED: tensor_memory_optimization bug (default FALSE).
#       • FIXED: Removed in-place tensor ops & fixed early exit logic.
#       • ENHANCED: Numerical stability in blend functions, added sigma schedule validation.
#   - v0.9.9-p3 (Feature Update):
#       • ADDED: Tensor batching, adaptive noise scaling, progressive blend mode, step profiling.
#   - v0.9.9-p2 (Feature Update):
#       • ADDED: Slerp/Add blend modes, ancestral noise types, FBG scale summary.

# ░▒▓ CONFIGURATION:
#   → Primary Use: High-quality audio/video generation with Ace-Step diffusion using FBG.
#   → Secondary Use: Using YAML to define complex, multi-stage FBG/CFG schedules.
#   → Edge Use: Extreme guidance scales with `max_guidance_scale` (tested up to 2000x).

# ░▒▓ WARNING:
#   This node may trigger:
#   ▓▒░ A sudden urge to `slerp` your breakfast cereal.
#   ▓▒░ Existential dread as FBG pushes guidance scale over 9000.
#   ▓▒░ Flashbacks to hand-optimizing assembly loops for a C64 plasma effect.
#   ▓▒░ The unshakable belief that `brownian` noise holds the secrets of the universe.
#   Consult your nearest demoscene vet if hallucinations persist.
# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄


# =================================================================================
# == Standard Library Imports                                                    ==
# =================================================================================
import contextlib
import enum
import logging
import math
import time
import traceback

# =================================================================================
# == Third-Party Imports                                                         ==
# =================================================================================
import numpy as np
import torch
import yaml
from tqdm.auto import trange

# =================================================================================
# == ComfyUI Core Modules                                                        ==
# =================================================================================
import comfy.k_diffusion.sampling
import comfy.model_management
import comfy.model_patcher
import comfy.model_sampling
import comfy.samplers

# =================================================================================
# == Local Project Imports                                                       ==
# =================================================================================
# (No local project imports in this file)

# =================================================================================
# == Helper Classes & Dependencies                                               ==
# =================================================================================

def slerp(a, b, t):
    """
    Spherical linear interpolation with enhanced numerical stability.
    
    Args:
        a (torch.Tensor): Start tensor.
        b (torch.Tensor): End tensor.
        t (float or torch.Tensor): Interpolation factor (0.0 to 1.0).
    
    Returns:
        torch.Tensor: Interpolated tensor.
    """
    eps = 1e-8
    
    # Normalize vectors to lie on the unit hypersphere
    a_norm = a / (torch.norm(a, dim=-1, keepdim=True) + eps)
    b_norm = b / (torch.norm(b, dim=-1, keepdim=True) + eps)
    
    # Calculate the dot product, clamping to avoid numerical errors
    dot = torch.sum(a_norm * b_norm, dim=-1, keepdim=True).clamp(-0.9999, 0.9999)
    
    # If vectors are nearly collinear, fall back to linear interpolation
    if torch.all(torch.abs(dot) > 0.9995):
        return torch.lerp(a, b, t)
    
    # Standard slerp formula with improved numerical stability
    theta = torch.acos(torch.abs(dot)) * t
    c = b_norm - a_norm * dot
    c_norm = c / (torch.norm(c, dim=-1, keepdim=True) + eps)
    
    result = a_norm * torch.cos(theta) + c_norm * torch.sin(theta)
    
    # Scale back to original magnitude (average of a and b)
    avg_norm = (torch.norm(a, dim=-1, keepdim=True) + torch.norm(b, dim=-1, keepdim=True)) / 2
    return result * avg_norm

def cosine_interpolation(a, b, t):
    """
    Cosine interpolation for smoother transitions.
    
    Args:
        a (torch.Tensor): Start tensor.
        b (torch.Tensor): End tensor.
        t (float or torch.Tensor): Interpolation factor (0.0 to 1.0).
    
    Returns:
        torch.Tensor: Interpolated tensor.
    """
    cos_t = (1 - torch.cos(t * math.pi)) * 0.5
    return a * (1 - cos_t) + b * cos_t

def cubic_interpolation(a, b, t):
    """
    Cubic interpolation for even smoother transitions.
    
    Args:
        a (torch.Tensor): Start tensor.
        b (torch.Tensor): End tensor.
        t (float or torch.Tensor): Interpolation factor (0.0 to 1.0).
    
    Returns:
        torch.Tensor: Interpolated tensor.
    """
    cubic_t = t * t * (3.0 - 2.0 * t)
    return torch.lerp(a, b, cubic_t)

# Enhanced blend modes with improved implementations
_INTERNAL_BLEND_MODES = {
    "lerp": torch.lerp,
    "slerp": slerp,
    "cosine": cosine_interpolation,
    "cubic": cubic_interpolation,
    "add": lambda a, b, t: a * (1 - t) + b * t,
    "a_only": lambda a, _b, _t: a,
    "b_only": lambda _a, b, _t: b
}

class SamplerMode(enum.Enum):
    EULER = enum.auto()
    PINGPONG = enum.auto()

class FBGConfig:
    """
    Helper class for FBG (Feedback Guidance) configuration.
    This replaces the previous NamedTuple implementation for guideline compliance.
    """
    def __init__(self,
                 sampler_mode=SamplerMode.EULER,
                 cfg_start_sigma=1.0,
                 cfg_end_sigma=0.004,
                 fbg_start_sigma=1.0,
                 fbg_end_sigma=0.004,
                 fbg_guidance_multiplier=1.0,
                 ancestral_start_sigma=1.0,
                 ancestral_end_sigma=0.004,
                 cfg_scale=1.0,
                 max_guidance_scale=10.0,
                 max_posterior_scale=3.0,
                 initial_value=0.0,
                 initial_guidance_scale=1.0,
                 guidance_max_change=1000.0,
                 temp=0.0,
                 offset=0.0,
                 pi=0.5,
                 t_0=0.5,
                 t_1=0.4):
        self.sampler_mode = sampler_mode
        self.cfg_start_sigma = cfg_start_sigma
        self.cfg_end_sigma = cfg_end_sigma
        self.fbg_start_sigma = fbg_start_sigma
        self.fbg_end_sigma = fbg_end_sigma
        self.fbg_guidance_multiplier = fbg_guidance_multiplier
        self.ancestral_start_sigma = ancestral_start_sigma
        self.ancestral_end_sigma = ancestral_end_sigma
        self.cfg_scale = cfg_scale
        self.max_guidance_scale = max_guidance_scale
        self.max_posterior_scale = max_posterior_scale
        self.initial_value = initial_value
        self.initial_guidance_scale = initial_guidance_scale
        self.guidance_max_change = guidance_max_change
        self.temp = temp
        self.offset = offset
        self.pi = pi
        self.t_0 = t_0
        self.t_1 = t_1

    def _asdict(self):
        """Helper to mimic NamedTuple._asdict() for compatibility."""
        return self.__dict__

def batch_mse_loss(a, b, *, start_dim=1):
    """
    Optimized MSE loss with memory efficiency and numerical stability.
    
    Args:
        a (torch.Tensor): First tensor.
        b (torch.Tensor): Second tensor.
        start_dim (int): The dimension from which to start summing.
        
    Returns:
        torch.Tensor: Calculated MSE loss.
    """
    # For very large tensors, process more carefully
    if a.numel() > 1e7:  # 10M elements
        diff = a - b
        return (diff * diff).sum(dim=tuple(range(start_dim, a.ndim)))
    return torch.sum((a - b).pow(2), dim=tuple(range(start_dim, a.ndim)))

class PerformanceProfiler:
    """Enhanced profiler for tracking step timing and memory usage."""
    def __init__(self, enabled=False):
        self.enabled = enabled
        self.step_times = []
        self.memory_usage = []
        self.step_names = []
        
    @contextlib.contextmanager
    def profile_step(self, step_name="step"):
        """
        Profiles a single step, tracking time and VRAM usage.
        
        Args:
            step_name (str): The name of the step to profile.
        """
        if not self.enabled:
            yield
            return
            
        device = comfy.model_management.get_torch_device()
        start_time = time.time()
        
        if device != torch.device("cpu"):
            torch.cuda.synchronize(device)
            start_memory = torch.cuda.memory_allocated(device)
        else:
            start_memory = 0
            
        yield
        
        if device != torch.device("cpu"):
            torch.cuda.synchronize(device)
            end_memory = torch.cuda.memory_allocated(device)
        else:
            end_memory = 0
            
        end_time = time.time()
        
        self.step_times.append(end_time - start_time)
        self.memory_usage.append(end_memory - start_memory)
        self.step_names.append(step_name)
        
    def get_summary(self):
        """
        Returns a string summary of the profiling session.
        
        Returns:
            str: Formatted summary.
        """
        if not self.step_times:
            return "No profiling data available"
            
        avg_time = sum(self.step_times) / len(self.step_times)
        max_time = max(self.step_times)
        total_time = sum(self.step_times)
        
        summary = f"Performance Summary:\n"
        summary += f"  Total time: {total_time:.3f}s\n"
        summary += f"  Average step time: {avg_time:.3f}s\n"
        summary += f"  Max step time: {max_time:.3f}s\n"
        summary += f"  Steps profiled: {len(self.step_times)}\n"
        
        device = comfy.model_management.get_torch_device()
        if self.memory_usage and device != torch.device("cpu"):
            total_memory = sum(self.memory_usage)
            avg_memory = total_memory / len(self.memory_usage)
            max_memory = max(self.memory_usage)
            summary += f"  Total memory delta: {total_memory / 1024**2:.1f}MB\n"
            summary += f"  Average memory delta: {avg_memory / 1024**2:.1f}MB\n"
            summary += f"  Max memory delta: {max_memory / 1024**2:.1f}MB"
            
        return summary

class PingPongSamplerCore:
    """
    Enhanced PingPongSampler core logic with stability fixes, FBG, and optimizations.
    This class handles the actual sampling loop and is called by the KSAMPLER wrapper.
    """
    
    def __init__(
        self,
        model,
        x,
        sigmas,
        extra_args=None,
        callback=None,
        disable=None,
        noise_sampler=None,
        # PingPong-specific parameters
        start_sigma_index=0,
        end_sigma_index=-1,
        enable_clamp_output=False,
        step_random_mode="off",
        step_size=5,
        seed=42,
        blend_function=torch.lerp,
        step_blend_function=torch.lerp,
        scheduler=None,
        pingpong_options=None,
        fbg_config=None,
        debug_mode=0,
        eta=0.0,
        s_noise=1.0,
        sigma_range_preset="Custom",
        conditional_blend_mode=False,
        conditional_blend_sigma_threshold=0.5,
        conditional_blend_function=torch.lerp,
        conditional_blend_on_change=False,
        conditional_blend_change_threshold=0.1,
        clamp_noise_norm=False,
        max_noise_norm=1.0,
        log_posterior_ema_factor=0.0,
        # Enhanced parameters
        adaptive_noise_scaling=False,
        noise_scale_factor=1.0,
        progressive_blend_mode=False,
        gradient_norm_tracking=False,
        enable_profiling=False,
        checkpoint_steps=None,
        early_exit_threshold=1e-6,
        tensor_memory_optimization=False,  # DEFAULT CHANGED TO FALSE
        **kwargs
    ):
        # Initialize core attributes
        self.model_ = model
        self.x = x
        self.sigmas = sigmas
        self.extra_args = extra_args.copy() if extra_args is not None else {}
        self.callback_ = callback
        self.disable_pbar = disable
        
        # PingPong specific
        self.start_sigma_index = start_sigma_index
        self.end_sigma_index = end_sigma_index
        self.enable_clamp_output = enable_clamp_output
        self.step_random_mode = step_random_mode
        self.step_size = step_size
        self.seed = seed if seed is not None else 42
        self.blend_function = blend_function
        self.step_blend_function = step_blend_function
        self.debug_mode = debug_mode
        
        # Enhanced features
        self.adaptive_noise_scaling = adaptive_noise_scaling
        self.noise_scale_factor = noise_scale_factor
        self.progressive_blend_mode = progressive_blend_mode
        self.gradient_norm_tracking = gradient_norm_tracking
        self.checkpoint_steps = checkpoint_steps or []
        self.early_exit_threshold = early_exit_threshold
        self.tensor_memory_optimization = tensor_memory_optimization
        
        # Warn if tensor_memory_optimization is enabled
        if self.tensor_memory_optimization:
            print("[PingPongSamplerFBG] ⚠️ WARNING: tensor_memory_optimization is EXPERIMENTAL and may cause artifacts!")
        
        # Performance profiler
        self.profiler = PerformanceProfiler(enable_profiling)
        
        # Gradient tracking (disabled - doesn't work in sampling context)
        self.gradient_norms = [] if gradient_norm_tracking else None
        if gradient_norm_tracking:
            logging.info("[PingPongSamplerFBG] Note: Gradient norm tracking disabled (not applicable in sampling)")
        
        # Determine the number of total steps
        num_steps_available = len(sigmas) - 1 if len(sigmas) > 0 else 0
        logging.info(f"[PingPongSamplerFBG] Total steps based on sigmas: {num_steps_available}")
        
        # Validate sigma schedule
        self._validate_sigma_schedule()
        
        # Adaptive step size for high step counts
        if num_steps_available > 500:
            suggested_step_size = max(num_steps_available // 100, 5)
            logging.info(f"[PingPongSamplerFBG] High step count detected ({num_steps_available} steps).")
            logging.info(f"[PingPongSamplerFBG] Consider using step_size >= {suggested_step_size} for optimal noise distribution")
        
        # Ancestral operation boundaries
        if pingpong_options is None:
            pingpong_options = {}
        raw_first_ancestral_step = pingpong_options.get("first_ancestral_step", 0)
        raw_last_ancestral_step = pingpong_options.get("last_ancestral_step", num_steps_available - 1)
        self.first_ancestral_step = max(0, min(raw_first_ancestral_step, raw_last_ancestral_step))
        if num_steps_available > 0:
            self.last_ancestral_step = min(num_steps_available - 1, max(raw_first_ancestral_step, raw_last_ancestral_step))
        else:
            self.last_ancestral_step = -1

        # Sigma range presets handling
        self.sigma_range_preset = sigma_range_preset
        self.original_fbg_config = fbg_config if fbg_config is not None else FBGConfig()
        self.config = self.original_fbg_config
        
        if self.sigma_range_preset != "Custom" and num_steps_available > 0:
            self.config = self._apply_sigma_preset(num_steps_available)

        # Conditional blend functions
        self.conditional_blend_mode = conditional_blend_mode
        self.conditional_blend_sigma_threshold = conditional_blend_sigma_threshold
        self.conditional_blend_function = conditional_blend_function
        self.conditional_blend_on_change = conditional_blend_on_change
        self.conditional_blend_change_threshold = conditional_blend_change_threshold

        # Noise norm clamping
        self.clamp_noise_norm = clamp_noise_norm
        self.max_noise_norm = max_noise_norm

        # EMA for log posterior
        self.log_posterior_ema_factor = max(0.0, min(1.0, log_posterior_ema_factor))

        # Model type detection
        self.is_rf = self._detect_model_type()

        # Noise sampler setup
        self.noise_sampler = self._setup_noise_sampler(noise_sampler, kwargs.get("ancestral_noise_type", "gaussian"))

        # Build noise decay array
        self.noise_decay = self._build_noise_decay_array(num_steps_available, scheduler)

        # FBG specific initialization
        self.eta = eta
        self.s_noise = s_noise
        self.update_fbg_config_params()
        
        # Initialize FBG internal states (FIXED: no broken optimization)
        cfg = self.config
        self.minimal_log_posterior = self._calculate_minimal_log_posterior(cfg)
        self.log_posterior = x.new_full((x.shape[0],), cfg.initial_value)
        self.guidance_scale = x.new_full((x.shape[0], *(1,) * (x.ndim - 1)), cfg.initial_guidance_scale)

        if self.debug_mode >= 1:
            logging.info(f"[PingPongSamplerFBG] FBG config: {self.config._asdict()}")
            logging.info(f"[PingPongSamplerFBG] Enhanced features: adaptive_noise={self.adaptive_noise_scaling}, progressive_blend={self.progressive_blend_mode}")

    def _validate_sigma_schedule(self):
        """Validate sigma schedule is monotonically decreasing."""
        if len(self.sigmas) < 2:
            return True
        
        for i in range(len(self.sigmas) - 1):
            if self.sigmas[i] < self.sigmas[i + 1]:
                logging.warning(f"[PingPongSamplerFBG] Sigma schedule not monotonic at step {i}: "
                                f"{self.sigmas[i]:.6f} -> {self.sigmas[i+1]:.6f}")
                return False
        return True

    def _apply_sigma_preset(self, num_steps_available):
        """Apply sigma range presets with validation."""
        sorted_sigmas_desc = torch.sort(self.sigmas, descending=True).values
        config_dict = self.config._asdict()
        
        if self.sigma_range_preset == "High":
            idx = max(1, num_steps_available // 4)
            high_sigma = sorted_sigmas_desc[idx].item()
            config_dict.update({
                'cfg_start_sigma': high_sigma,
                'cfg_end_sigma': self.sigmas[-2].item(),
                'fbg_start_sigma': high_sigma,
                'fbg_end_sigma': self.sigmas[-2].item()
            })
        elif self.sigma_range_preset == "Mid":
            idx_start = num_steps_available // 4
            idx_end = 3 * num_steps_available // 4
            config_dict.update({
                'cfg_start_sigma': sorted_sigmas_desc[idx_start].item(),
                'cfg_end_sigma': sorted_sigmas_desc[idx_end].item(),
                'fbg_start_sigma': sorted_sigmas_desc[idx_start].item(),
                'fbg_end_sigma': sorted_sigmas_desc[idx_end].item()
            })
        elif self.sigma_range_preset == "Low":
            idx = 3 * num_steps_available // 4
            low_sigma = sorted_sigmas_desc[idx].item()
            config_dict.update({
                'cfg_start_sigma': self.sigmas[0].item(),
                'cfg_end_sigma': low_sigma,
                'fbg_start_sigma': self.sigmas[0].item(),
                'fbg_end_sigma': low_sigma
            })
        elif self.sigma_range_preset == "All":
            config_dict.update({
                'cfg_start_sigma': self.sigmas[0].item(),
                'cfg_end_sigma': self.sigmas[-2].item(),
                'fbg_start_sigma': self.sigmas[0].item(),
                'fbg_end_sigma': self.sigmas[-2].item()
            })
            
        logging.info(f"[PingPongSamplerFBG] Applied sigma range preset '{self.sigma_range_preset}'")
            
        return FBGConfig(**config_dict)

    def _detect_model_type(self):
        """Enhanced model type detection with error handling."""
        try:
            current_model_check = self.model_
            while hasattr(current_model_check, 'inner_model') and current_model_check.inner_model is not None:
                current_model_check = current_model_check.inner_model
            if hasattr(current_model_check, 'model_sampling') and current_model_check.model_sampling is not None:
                return isinstance(current_model_check.model_sampling, comfy.model_sampling.CONST)
        except (AttributeError, TypeError) as e:
            logging.warning(f"[PingPongSamplerFBG] Model type detection failed: {e}. Assuming non-CONST sampling.")
        return False

    def _setup_noise_sampler(self, noise_sampler, noise_type):
        """
        Setup noise sampler with enhanced noise types.
        
        Args:
            noise_sampler (callable): An existing noise sampler, if provided.
            noise_type (str): The type of noise ("gaussian", "uniform", "brownian").
            
        Returns:
            callable: A noise sampler function.
        """
        if noise_sampler is not None:
            return noise_sampler
            
        logging.info(f"[PingPongSamplerFBG] Using ancestral noise type: {noise_type}")
            
        def create_noise_sampler(noise_func):
            def sampler(sigma, sigma_next):
                base_noise = noise_func()
                if self.adaptive_noise_scaling:
                    # Scale noise based on denoising progress
                    progress = 1.0 - (sigma / self.sigmas[0]) if self.sigmas[0] > 0 else 0.0
                    scale = self.noise_scale_factor * (1.0 + progress * 0.5)
                    base_noise = base_noise * scale
                return base_noise
            return sampler
        
        if noise_type == "uniform":
            return create_noise_sampler(lambda: torch.rand_like(self.x) * 2 - 1)
        elif noise_type == "brownian":
            return create_noise_sampler(lambda: torch.randn_like(self.x).cumsum(dim=-1) / (self.x.shape[-1]**0.5))
        else:  # gaussian
            return create_noise_sampler(lambda: torch.randn_like(self.x))

    def _build_noise_decay_array(self, num_steps_available, scheduler):
        """Build noise decay array with error handling."""
        if num_steps_available <= 0:
            return torch.empty((0,), dtype=torch.float32, device=self.x.device)
            
        if scheduler is not None and hasattr(scheduler, 'get_decay'):
            try:
                arr = scheduler.get_decay(num_steps_available)
                decay = np.asarray(arr, dtype=np.float32)
                assert decay.shape == (num_steps_available,)
                return torch.tensor(decay, device=self.x.device)
            except Exception as e:
                logging.warning(f"[PingPongSamplerFBG] Scheduler error: {e}. Using zero decay.")
        
        return torch.zeros((num_steps_available,), dtype=torch.float32, device=self.x.device)

    def _calculate_minimal_log_posterior(self, cfg):
        """Calculate minimal log posterior with enhanced error handling."""
        try:
            if cfg.cfg_scale > 1 and cfg.cfg_start_sigma > 0:
                numerator = (1.0 - cfg.pi) * (cfg.max_guidance_scale - cfg.cfg_scale + 1)
                denominator = (cfg.max_guidance_scale - cfg.cfg_scale)
            else:
                numerator = (1.0 - cfg.pi) * cfg.max_guidance_scale
                denominator = (cfg.max_guidance_scale - 1.0)
                
            if denominator <= 0 or numerator <= 0:
                return float('-inf')
            return math.log(numerator / denominator)
        except (ValueError, ZeroDivisionError):
            logging.warning("[PingPongSamplerFBG] Could not calculate minimal_log_posterior. Using -inf.")
            return float('-inf')

    def get_progressive_blend_function(self, step_idx, total_steps):
        """Get blend function that changes based on sampling progress."""
        if not self.progressive_blend_mode:
            return self.blend_function
            
        progress = step_idx / max(total_steps - 1, 1)
        
        # Early steps: use linear interpolation
        if progress < 0.3:
            return torch.lerp
        # Middle steps: use cosine interpolation
        elif progress < 0.7:
            return cosine_interpolation
        # Later steps: use cubic interpolation for smoothness
        else:
            return cubic_interpolation

    def calculate_adaptive_noise_scale(self, sigma_current, step_idx, total_steps):
        """Calculate adaptive noise scaling factor."""
        if not self.adaptive_noise_scaling:
            return 1.0
            
        # Base scale factor
        base_scale = self.noise_scale_factor
        
        # Progress-based scaling
        progress = step_idx / max(total_steps - 1, 1)
        progress_scale = 1.0 + (1.0 - progress) * 0.3  # Reduce noise as we progress
        
        # Sigma-based scaling
        normalized_sigma = sigma_current / self.sigmas[0] if self.sigmas[0] > 0 else 0.0
        sigma_scale = 0.5 + 0.5 * normalized_sigma  # More noise at higher sigma
        
        return base_scale * progress_scale * sigma_scale

    def save_checkpoint(self, x_current, step_idx):
        """Save checkpoint at specific steps."""
        if step_idx not in self.checkpoint_steps:
            return
            
        logging.info(f"[PingPongSamplerFBG] Saving checkpoint at step {step_idx}")
        
        if not hasattr(self, 'checkpoints'):
            self.checkpoints = {}
        self.checkpoints[step_idx] = x_current.clone()

    def check_early_exit(self, sigma_next_item, denoised_sample, x_current):
        """Check if we should exit early based on convergence (FIXED)."""
        if sigma_next_item > self.early_exit_threshold:
            return False
            
        # Check for convergence
        if hasattr(self, '_prev_x') and self._prev_x is not None:
            change_norm = torch.norm(x_current - self._prev_x).item()
            if change_norm < self.early_exit_threshold * 0.1:
                logging.info(f"[PingPongSamplerFBG] Early exit due to convergence (change norm: {change_norm:.8f})")
                return True
        
        # Always clone for next comparison
        self._prev_x = x_current.clone()
        return False

    def _check_for_nan_inf(self, tensor, name, step_idx):
        """Check tensor for NaN or Inf values."""
        if torch.isnan(tensor).any():
            logging.warning(f"[PingPongSamplerFBG] ⚠️ WARNING: NaN detected in {name} at step {step_idx}!")
            return True
        if torch.isinf(tensor).any():
            logging.warning(f"[PingPongSamplerFBG] ⚠️ WARNING: Inf detected in {name} at step {step_idx}!")
            return True
        return False

    def _debug_step_info(self, idx, sigma_current, sigma_next, x_current):
        """Comprehensive debug info for troubleshooting."""
        if self.debug_mode >= 2:
            logging.debug(f"\n[PingPongSamplerFBG] === Step {idx} Detailed Info ===")
            logging.debug(f"  Sigma: {sigma_current.item():.6f} -> {sigma_next.item():.6f}")
            logging.debug(f"  X stats: min={x_current.min().item():.4f}, "
                          f"max={x_current.max().item():.4f}, "
                          f"mean={x_current.mean().item():.4f}, "
                          f"std={x_current.std().item():.4f}")
            self._check_for_nan_inf(x_current, "x_current", idx)

    def _stepped_seed(self, step):
        """
        Enhanced seed calculation with validation.
        
        Args:
            step (int): The current sampling step.
            
        Returns:
            int or None: The calculated seed, or None if mode is 'off'.
        """
        if self.step_random_mode == "off":
            return None
        current_step_size = max(self.step_size, 1)
        
        seed_map = {
            "block": self.seed + (step // current_step_size),
            "reset": self.seed + (step * current_step_size),
            "step": self.seed + step
        }
        
        return seed_map.get(self.step_random_mode, self.seed)

    def _get_sigma_square_tilde(self, sigmas):
        """
        Optimized sigma square tilde calculation.
        
        Args:
            sigmas (torch.Tensor): The tensor of sigmas.
            
        Returns:
            torch.Tensor: Calculated sigma square tilde values.
        """
        if len(sigmas) < 2:
            return torch.tensor([], device=sigmas.device)
        
        s_sq, sn_sq = sigmas[:-1] ** 2, sigmas[1:] ** 2
        safe_s_sq = torch.where(s_sq == 0, torch.tensor(1e-6, device=s_sq.device), s_sq)
        return ((s_sq - sn_sq) * sn_sq / safe_s_sq).flip(dims=(0,))

    def _get_offset(self, steps, sigma_square_tilde, **kwargs):
        """
        Enhanced offset calculation with better error handling.
        
        Args:
            steps (int): Total number of steps.
            sigma_square_tilde (torch.Tensor): Precomputed values.
            
        Returns:
            float: Calculated offset.
        """
        cfg = self.config
        lambda_ref = kwargs.get('lambda_ref', 3.0)
        decimals = kwargs.get('decimals', 4)
        
        t_0_clamped = max(0.0, min(1.0, cfg.t_0))
        
        if t_0_clamped >= 1.0 or (lambda_ref - 1.0) <= 0 or (1.0 - cfg.pi) <= 0:
            return 0.0
        
        try:
            log_term = math.log((1.0 - cfg.pi) * lambda_ref / (lambda_ref - 1.0))
            denominator = (1.0 - t_0_clamped) * steps
            if denominator == 0:
                return 0.0
            return round(log_term / denominator, decimals)
        except (ValueError, ZeroDivisionError):
            return 0.0

    def _get_temp(self, steps, offset, sigma_square_tilde, **kwargs):
        """
        Enhanced temperature calculation.
        
        Args:
            steps (int): Total number of steps.
            offset (float): Calculated offset.
            sigma_square_tilde (torch.Tensor): Precomputed values.
            
        Returns:
            float: Calculated temperature.
        """
        cfg = self.config
        alpha = kwargs.get('alpha', 10.0)
        decimals = kwargs.get('decimals', 4)
        
        t_1_clamped = max(0.0, min(1.0, cfg.t_1))
        t1_lower_idx = int(math.floor(t_1_clamped * (steps - 1)))
        
        if len(sigma_square_tilde) == 0 or alpha == 0:
            return 0.0
        
        t1_lower_idx = max(0, min(t1_lower_idx, len(sigma_square_tilde) - 1))
        
        try:
            if len(sigma_square_tilde) == 1:
                sst = sigma_square_tilde[0].item()
            elif t1_lower_idx == len(sigma_square_tilde) - 1:
                sst = sigma_square_tilde[t1_lower_idx].item()
            else:
                sst_t1 = sigma_square_tilde[t1_lower_idx].item()
                sst_t1_next = sigma_square_tilde[t1_lower_idx + 1].item()
                a = (t_1_clamped * (steps - 1)) - t1_lower_idx
                sst = sst_t1 * (1 - a) + sst_t1_next * a
            
            temp = (2 * sst / alpha * offset)
            return round(temp, decimals)
        except (IndexError, ValueError):
            return 0.0

    def update_fbg_config_params(self):
        """Enhanced FBG config parameter updates."""
        if self.config.t_0 == 0 and self.config.t_1 == 0:
            return
        
        steps = len(self.sigmas) - 1
        if steps <= 0:
            return
        
        sst = self._get_sigma_square_tilde(self.sigmas)
        calculated_offset = self._get_offset(steps, sst)
        calculated_temp = self._get_temp(steps, calculated_offset, sst)
        
        config_dict = self.config._asdict()
        config_dict.update({
            "offset": calculated_offset,
            "temp": calculated_temp
        })
        self.config = FBGConfig(**config_dict)

    def get_dynamic_guidance_scale(self, log_posterior_val, guidance_scale_prev, sigma_item):
        """
        FIXED: Dynamic guidance scale calculation without broken optimization.
        
        Args:
            log_posterior_val (torch.Tensor): Current log posterior.
            guidance_scale_prev (torch.Tensor): Guidance scale from previous step.
            sigma_item (float): Current sigma value.
            
        Returns:
            torch.Tensor: The new guidance scale.
        """
        config = self.config
        
        using_fbg = config.fbg_end_sigma <= sigma_item <= config.fbg_start_sigma
        using_cfg = config.cfg_scale != 1 and (config.cfg_end_sigma <= sigma_item <= config.cfg_start_sigma)
        
        # FIXED: Always use correct shape initialization
        guidance_scale = log_posterior_val.new_ones(guidance_scale_prev.shape[0])
        
        if using_fbg:
            # Enhanced FBG calculation with numerical stability
            denom = log_posterior_val.exp() - (1.0 - config.pi)
            safe_denom = torch.where(denom.abs() < 1e-6, 
                                     torch.full_like(denom, 1e-6), denom)
            fbg_component = log_posterior_val.exp() / safe_denom
            fbg_component *= config.fbg_guidance_multiplier
            guidance_scale = fbg_component.clamp(1.0, config.max_guidance_scale)
            
            if self.debug_mode >= 2:
                logging.debug(f"[PingPongSamplerFBG] FBG active (sigma {sigma_item:.3f}). "
                              f"Raw FBG component: {guidance_scale.mean().item():.4f}")
        
        if using_cfg:
            guidance_scale += config.cfg_scale - 1.0
            if self.debug_mode >= 2:
                logging.debug(f"[PingPongSamplerFBG] CFG active. Added: {config.cfg_scale - 1.0:.2f}")
        
        # Apply constraints
        guidance_scale = guidance_scale.clamp(1.0, config.max_guidance_scale)
        guidance_scale = guidance_scale.view(guidance_scale_prev.shape)
        
        # Apply max change constraint
        safe_prev = torch.where(guidance_scale_prev.abs() < 1e-6, 
                                torch.full_like(guidance_scale_prev, 1e-6), 
                                guidance_scale_prev)
        change_pct = ((guidance_scale - guidance_scale_prev) / safe_prev).clamp(
            -config.guidance_max_change, config.guidance_max_change
        )
        
        final_guidance_scale = guidance_scale_prev + guidance_scale_prev * change_pct
        final_guidance_scale = final_guidance_scale.clamp(1.0, config.max_guidance_scale)
        
        return final_guidance_scale

    def _model_denoise_with_guidance(self, x_tensor, sigma_scalar, override_cfg=None):
        """
        Enhanced model denoising with improved error handling.
        
        Args:
            x_tensor (torch.Tensor): The input latent.
            sigma_scalar (torch.Tensor): The current sigma.
            override_cfg (torch.Tensor, optional): Dynamic guidance scale.
            
        Returns:
            tuple: (denoised, cond, uncond) tensors.
        """
        sigma_tensor = sigma_scalar * x_tensor.new_ones((x_tensor.shape[0],))
        
        cond = uncond = None
        
        def post_cfg_function(args):
            nonlocal cond, uncond
            cond, uncond = args["cond_denoised"], args["uncond_denoised"]
            return args["denoised"]
        
        extra_args = self.extra_args.copy()
        orig_model_options = extra_args.get("model_options", {})
        model_options = orig_model_options.copy()
        model_options["disable_cfg1_optimization"] = True
        extra_args["model_options"] = comfy.model_patcher.set_model_options_post_cfg_function(
            model_options, post_cfg_function
        )
        
        inner_model = self.model_.inner_model
        
        try:
            if (override_cfg is None or (isinstance(override_cfg, torch.Tensor) and override_cfg.numel() < 2)) and hasattr(inner_model, "cfg"):
                orig_cfg = inner_model.cfg
                try:
                    if override_cfg is not None:
                        if isinstance(override_cfg, torch.Tensor) and override_cfg.numel() == 1:
                            inner_model.cfg = override_cfg.detach().item()
                        elif isinstance(override_cfg, torch.Tensor):
                            inner_model.cfg = override_cfg.mean().detach().item()
                        else:
                            inner_model.cfg = override_cfg
                    
                    denoised = inner_model.predict_noise(
                        x_tensor, sigma_tensor,
                        model_options=extra_args["model_options"],
                        seed=extra_args.get("seed"),
                    )
                finally:
                    inner_model.cfg = orig_cfg
            else:
                _ = self.model_(x_tensor, sigma_tensor, **extra_args)
                denoised = comfy.samplers.cfg_function(
                    inner_model.inner_model, cond, uncond, override_cfg,
                    x_tensor, sigma_tensor, model_options=orig_model_options,
                )
        except Exception as e:
            logging.warning(f"[PingPongSamplerFBG] Model denoising error: {e}. Using fallback.")
            # Fallback to basic model call
            denoised = self.model_(x_tensor, sigma_tensor, **extra_args)
            if cond is None:
                cond = uncond = denoised
        
        return denoised, cond, uncond

    def _update_log_posterior(self, prev_log_posterior, 
                              x_curr, x_next,
                              t_curr, t_next,
                              uncond, cond):
        """
        FIXED: Log posterior update without broken in-place operations.
        
        Args:
            prev_log_posterior (torch.Tensor): Log posterior from previous step.
            x_curr (torch.Tensor): Latent at current step (before sampling).
            x_next (torch.Tensor): Latent at next step (after sampling).
            t_curr (torch.Tensor): Sigma at current step.
            t_next (torch.Tensor): Sigma at next step.
            uncond (torch.Tensor): Unconditional denoised output.
            cond (torch.Tensor): Conditional denoised output.
            
        Returns:
            torch.Tensor: The updated log posterior.
        """
        
        def apply_ema_and_return(value):
            if self.log_posterior_ema_factor > 0:
                smoothed = (self.log_posterior_ema_factor * prev_log_posterior + 
                            (1 - self.log_posterior_ema_factor) * value)
                return smoothed.clamp(self.minimal_log_posterior, self.config.max_posterior_scale)
            return value.clamp(self.minimal_log_posterior, self.config.max_posterior_scale)
        
        if torch.isclose(t_curr, torch.tensor(0.0, device=t_curr.device)).all():
            return apply_ema_and_return(prev_log_posterior)
        
        t_csq = t_curr**2
        if torch.isclose(t_csq, torch.tensor(0.0, device=t_csq.device)).all():
            return apply_ema_and_return(prev_log_posterior)
        
        t_ndc = t_next**2 / t_csq
        t_cmn = t_csq - t_next**2
        sigma_square_tilde_t = t_cmn * t_ndc
        
        if torch.isclose(sigma_square_tilde_t, torch.tensor(0.0, device=sigma_square_tilde_t.device)).all():
            return apply_ema_and_return(prev_log_posterior)
        
        # FIXED: Never use in-place operations
        pred_base = t_ndc * x_curr
        uncond_pred_mean = pred_base + (t_cmn / t_csq) * uncond
        cond_pred_mean = pred_base + (t_cmn / t_csq) * cond
        
        diff = batch_mse_loss(x_next, cond_pred_mean) - batch_mse_loss(x_next, uncond_pred_mean)
        
        if torch.isclose(sigma_square_tilde_t, torch.tensor(0.0, device=sigma_square_tilde_t.device)):
            result = prev_log_posterior + self.config.offset
        else:
            result = (prev_log_posterior - 
                      self.config.temp / (2 * sigma_square_tilde_t) * diff + 
                      self.config.offset)
        
        return apply_ema_and_return(result)

    def _do_callback(self, step_idx, current_x, current_sigma, denoised_sample):
        """Enhanced callback with error handling."""
        if self.callback_:
            try:
                self.callback_({
                    "i": step_idx,
                    "x": current_x,
                    "sigma": current_sigma,
                    "sigma_hat": current_sigma,
                    "denoised": denoised_sample
                })
            except Exception as e:
                logging.warning(f"[PingPongSamplerFBG] Callback error at step {step_idx}: {e}")

    def __call__(self):
        """
        FIXED: Main sampling loop with all stability improvements and error handling.
        
        Returns:
            torch.Tensor: The final denoised latent.
        """
        try:
            if self.debug_mode >= 1:
                logging.info("[PingPongSamplerFBG] Starting sampling loop")
                guidance_scales_used = []
            
            x_current = self.x.clone()
            num_steps = len(self.sigmas) - 1
            
            if num_steps <= 0:
                if self.enable_clamp_output:
                    x_current = torch.clamp(x_current, -1.0, 1.0)
                return x_current
            
            astart = self.first_ancestral_step
            aend = self.last_ancestral_step
            actual_end_idx = min(self.end_sigma_index if self.end_sigma_index >= 0 else num_steps - 1, 
                                 num_steps - 1)
            
            # Main sampling loop with enhanced features
            for idx in trange(num_steps, disable=self.disable_pbar):
                if idx < self.start_sigma_index or idx > actual_end_idx:
                    continue
                
                with self.profiler.profile_step(f"step_{idx}"):
                    sigma_current, sigma_next = self.sigmas[idx], self.sigmas[idx + 1]
                    sigma_item = sigma_current.max().detach().item()
                    sigma_next_item = sigma_next.min().detach().item()
                    
                    # Enhanced guidance scale calculation
                    self.guidance_scale = self.get_dynamic_guidance_scale(
                        self.log_posterior, self.guidance_scale, sigma_item
                    )
                    
                    if self.debug_mode >= 1:
                        guidance_scales_used.append(self.guidance_scale.mean().item())
                    
                    # Model denoising
                    denoised_sample, cond, uncond = self._model_denoise_with_guidance(
                        x_current, sigma_current, override_cfg=self.guidance_scale
                    )
                    
                    # Check for numerical issues
                    if self._check_for_nan_inf(denoised_sample, "denoised_sample", idx):
                        logging.warning(f"[PingPongSamplerFBG] Attempting to recover from NaN/Inf at step {idx}")
                        denoised_sample = torch.where(
                            torch.isnan(denoised_sample) | torch.isinf(denoised_sample),
                            x_current,
                            denoised_sample
                        )
                    
                    # Callback
                    self._do_callback(idx, x_current, sigma_current, denoised_sample)
                    
                    # Debug output
                    if self.debug_mode >= 1:
                        gs_str = f"{self.guidance_scale.mean().item():.2f}"
                        logging.info(f"[PingPongSamplerFBG] Step {idx}: σ={sigma_item:.3f}→{sigma_next_item:.3f}, GS={gs_str}")
                    
                    # Detailed debug
                    self._debug_step_info(idx, sigma_current, sigma_next, x_current)
                    
                    # Store original for FBG update
                    x_orig = x_current.clone()
                    
                    # Sampling step logic
                    use_anc = (astart <= idx <= aend) if astart <= aend else False
                    
                    if not use_anc:
                        # Non-ancestral step with progressive blending
                        blend = sigma_next / sigma_current if sigma_current > 0 else 0.0
                        blend_func = self.get_progressive_blend_function(idx, num_steps)
                        
                        # Conditional blending
                        if (self.conditional_blend_mode and 
                            sigma_item < self.conditional_blend_sigma_threshold):
                            blend_func = self.conditional_blend_function
                        
                        # FIXED: Always use function, never in-place
                        x_current = blend_func(denoised_sample, x_current, blend)
                    else:
                        # Ancestral step
                        local_seed = self._stepped_seed(idx)
                        if local_seed is not None:
                            torch.manual_seed(local_seed)
                            if self.debug_mode >= 2:
                                logging.debug(f"[PingPongSamplerFBG] Using seed: {local_seed}")
                        
                        # Generate noise with adaptive scaling
                        noise_sample = self.noise_sampler(sigma_current, sigma_next)
                        adaptive_scale = self.calculate_adaptive_noise_scale(
                            sigma_item, idx, num_steps
                        )
                        
                        if adaptive_scale != 1.0:
                            noise_sample = noise_sample * adaptive_scale
                        
                        # Noise norm clamping
                        if self.clamp_noise_norm:
                            noise_dims = list(range(1, noise_sample.ndim))
                            noise_norm = torch.norm(noise_sample, p=2, dim=noise_dims, keepdim=True)
                            scale_factor = torch.where(
                                noise_norm > self.max_noise_norm,
                                self.max_noise_norm / (noise_norm + 1e-8),
                                torch.ones_like(noise_norm)
                            )
                            noise_sample = noise_sample * scale_factor
                        
                        # Apply noise based on model type
                        if self.is_rf:
                            blend_func = self.get_progressive_blend_function(idx, num_steps)
                            x_current = blend_func(denoised_sample, noise_sample, sigma_next)
                        else:
                            # FIXED: Never use in-place operations
                            x_current = denoised_sample + noise_sample * sigma_next
                        
                        # Conditional blending on change
                        if self.conditional_blend_on_change:
                            denoised_norm = torch.norm(denoised_sample, dim=list(range(1, denoised_sample.ndim)), keepdim=True) + 1e-8
                            noise_norm = torch.norm(noise_sample * sigma_next, dim=list(range(1, noise_sample.ndim)), keepdim=True)
                            relative_change = (noise_norm / denoised_norm).mean().item()
                            
                            if relative_change > self.conditional_blend_change_threshold:
                                blend_factor = min(1.0, (relative_change - self.conditional_blend_change_threshold) / 
                                                     self.conditional_blend_change_threshold)
                                x_current = self.conditional_blend_function(denoised_sample, x_current, blend_factor)
                    
                    # FBG log posterior update
                    self.log_posterior = self._update_log_posterior(
                        self.log_posterior, x_orig, x_current, 
                        sigma_current, sigma_next, uncond, cond
                    )
                    
                    # Save checkpoints
                    self.save_checkpoint(x_current, idx)
                    
                    # Early exit
                    if self.check_early_exit(sigma_next_item, denoised_sample, x_current):
                        break
                    
                    if self.enable_clamp_output and sigma_next_item < 1e-3:
                        x_current = torch.clamp(x_current, -1.0, 1.0)
                        break
                    
                    if sigma_next_item <= 1e-6:
                        x_current = denoised_sample
                        break
            
            # Final processing
            if self.enable_clamp_output:
                x_current = torch.clamp(x_current, -1.0, 1.0)
            
            # Enhanced debug summary
            if self.debug_mode >= 1:
                logging.info("\n" + "="*50)
                logging.info("PingPongSamplerFBG Summary")
                logging.info("="*50)
                if guidance_scales_used:
                    logging.info(f"  Guidance Scale - Min: {min(guidance_scales_used):.3f}, "
                                 f"Max: {max(guidance_scales_used):.3f}, "
                                 f"Avg: {sum(guidance_scales_used) / len(guidance_scales_used):.3f}")
                
                if hasattr(self, 'checkpoints'):
                    logging.info(f"  Checkpoints saved: {len(self.checkpoints)}")
                
                if self.profiler.enabled:
                    logging.info("\n" + self.profiler.get_summary())
                
                logging.info("="*50 + "\n")
            
            return x_current

        except Exception as e:
            # Graceful failure per Sec 7.3
            logging.error(f"[PingPongSamplerFBG] Sampling failed catastrophically: {e}")
            logging.debug(traceback.format_exc())
            print(f"[PingPongSamplerFBG] ⚠️ Error encountered, returning original latent unchanged")
            return self.x # Return original input latent

    @staticmethod
    def go(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, **kwargs):
        """
        Static method wrapper for KSAMPLER compatibility.
        
        Args:
            model: The ComfyUI model object.
            x: The latent tensor.
            sigmas: The sigma schedule.
            extra_args (dict, optional): Extra arguments for the model.
            callback (callable, optional): Step callback function.
            disable (bool, optional): Disable progress bar.
            noise_sampler (callable, optional): Custom noise sampler.
            **kwargs: All other options from the node.
            
        Returns:
            torch.Tensor: The final denoised latent.
        """
        # Extract and process FBG config
        fbg_config_kwargs_raw = kwargs.pop("fbg_config", {})
        fbg_config_kwargs = {}
        
        remap_rules = {
            "fbg_sampler_mode": "sampler_mode",
            "fbg_temp": "temp", 
            "fbg_offset": "offset",
            "log_posterior_initial_value": "initial_value",
        }
        
        for key, value in fbg_config_kwargs_raw.items():
            mapped_key = remap_rules.get(key, key)
            fbg_config_kwargs[mapped_key] = value
        
        # Handle sampler mode enum conversion
        if "sampler_mode" in fbg_config_kwargs and isinstance(fbg_config_kwargs["sampler_mode"], str):
            try:
                fbg_config_kwargs["sampler_mode"] = getattr(SamplerMode, fbg_config_kwargs["sampler_mode"].upper())
            except AttributeError:
                fbg_config_kwargs.pop("sampler_mode", None)
        elif "sampler_mode" not in fbg_config_kwargs:
            fbg_config_kwargs["sampler_mode"] = FBGConfig().sampler_mode
        
        fbg_config_instance = FBGConfig(**fbg_config_kwargs)
        
        # Extract other parameters
        pingpong_options_kwargs = kwargs.pop("pingpong_options", {})
        
        # Extract blend function names and convert to functions
        blend_function_name = kwargs.pop("blend_function_name", "lerp")
        step_blend_function_name = kwargs.pop("step_blend_function_name", "lerp")
        conditional_blend_function_name = kwargs.pop("conditional_blend_function_name", "slerp")
        
        blend_function = _INTERNAL_BLEND_MODES.get(blend_function_name, torch.lerp)
        step_blend_function = _INTERNAL_BLEND_MODES.get(step_blend_function_name, torch.lerp)
        conditional_blend_function = _INTERNAL_BLEND_MODES.get(conditional_blend_function_name, slerp)
        
        # Create enhanced sampler instance
        sampler_instance = PingPongSamplerCore(
            model=model, x=x, sigmas=sigmas,
            extra_args=extra_args, callback=callback, disable=disable,
            noise_sampler=noise_sampler,
            blend_function=blend_function,
            step_blend_function=step_blend_function,
            conditional_blend_function=conditional_blend_function,
            fbg_config=fbg_config_instance,
            pingpong_options=pingpong_options_kwargs,
            **kwargs
        )
        
        return sampler_instance()


# =================================================================================
# == Core Node Class                                                             ==
# =================================================================================

class PingPongSamplerNodeFBG:
    """
    ComfyUI node wrapper for the PingPong FBG Sampler.
    
    This node creates and configures the PingPongSamplerCore object,
    which is then returned as a standard ComfyUI SAMPLER object.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        """
        Define all input parameters with standardized tooltips.
        """
        defaults_fbg_config = FBGConfig()
        return {
            "required": {
                # --- Core PingPong Parameters ---
                "step_random_mode": (["off", "block", "reset", "step"], {
                    "default": "block",
                    "tooltip": (
                        "STEP RANDOM MODE\n"
                        "- Controls noise seeding behavior during ancestral steps.\n"
                        "- 'off': No randomization (uses main seed).\n"
                        "- 'block': Seed changes every 'step_size' steps.\n"
                        "- 'reset': Seed resets every 'step_size' steps.\n"
                        "- 'step': Seed changes every single step."
                    )
                }),
                "step_size": ("INT", {
                    "default": 4, "min": 1, "max": 100,
                    "tooltip": (
                        "STEP SIZE\n"
                        "- Works with 'step_random_mode' ('block'/'reset').\n"
                        "- Defines the number of steps in a 'block' for noise seeding."
                    )
                }),
                "seed": ("INT", {
                    "default": 80085, "min": 0, "max": 0xffffffffffffffff,
                    "tooltip": (
                        "SEED\n"
                        "- The master seed for noise generation.\n"
                        "- Used in conjunction with 'step_random_mode'."
                    )
                }),
                "first_ancestral_step": ("INT", {
                    "default": 0, "min": -1, "max": 10000,
                    "tooltip": (
                        "FIRST ANCESTRAL STEP\n"
                        "- The step index to *start* using ancestral noise.\n"
                        "- Set to -1 to disable ancestral noise."
                    )
                }),
                "last_ancestral_step": ("INT", {
                    "default": -1, "min": -1, "max": 10000,
                    "tooltip": (
                        "LAST ANCESTRAL STEP\n"
                        "- The step index to *stop* using ancestral noise.\n"
                        "- Set to -1 to use ancestral noise until the last step."
                    )
                }),
                "ancestral_noise_type": (["gaussian", "uniform", "brownian"], {
                    "default": "gaussian",
                    "tooltip": (
                        "ANCESTRAL NOISE TYPE\n"
                        "- Type of noise to add during ancestral steps.\n"
                        "- 'gaussian': Standard random noise.\n"
                        "- 'uniform': Uniformly distributed noise.\n"
                        "- 'brownian': Correlated noise (can add texture)."
                    )
                }),
                "start_sigma_index": ("INT", {
                    "default": 0, "min": 0, "max": 10000,
                    "tooltip": (
                        "START SIGMA INDEX\n"
                        "- The step index to start sampling from.\n"
                        "- Default 0 starts from the beginning."
                    )
                }),
                "end_sigma_index": ("INT", {
                    "default": -1, "min": -10000, "max": 10000,
                    "tooltip": (
                        "END SIGMA INDEX\n"
                        "- The step index to end sampling at.\n"
                        "- Default -1 ends at the last step."
                    )
                }),
                "enable_clamp_output": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "ENABLE CLAMP OUTPUT\n"
                        "- Clamps the output tensor to [-1.0, 1.0] at the end.\n"
                        "- WARNING: May cause artifacts or loss of dynamic range."
                    )
                }),
                "scheduler": ("SCHEDULER", {
                    "tooltip": (
                        "SCHEDULER\n"
                        "- Connect a custom scheduler (e.g., NoiseDecay, HybridAdaptive).\n"
                        "- Required for custom noise decay curves."
                    )
                }),
                "blend_mode": (tuple(_INTERNAL_BLEND_MODES.keys()), {
                    "default": "lerp",
                    "tooltip": (
                        "BLEND MODE\n"
                        "- Main blend function for non-ancestral steps.\n"
                        "- 'lerp': Linear interpolation (standard).\n"
                        "- 'slerp': Spherical interpolation (preserves magnitude).\n"
                        "- 'cosine'/'cubic': Smoother transitions."
                    )
                }),
                "step_blend_mode": (tuple(_INTERNAL_BLEND_MODES.keys()), {
                    "default": "lerp",
                    "tooltip": (
                        "STEP BLEND MODE\n"
                        "- Blend function used for ancestral steps (if is_rf=True).\n"
                        "- Usually 'lerp' is fine."
                    )
                }),
                
                # --- Advanced Features & Debug ---
                "debug_mode": ("INT", {
                    "default": 0, "min": 0, "max": 2,
                    "tooltip": (
                        "DEBUG MODE\n"
                        "- Controls console logging verbosity.\n"
                        "- 0: Off (Warnings only).\n"
                        "- 1: Info (Step info, GS, summaries).\n"
                        "- 2: Debug (Detailed tensor stats per step)."
                    )
                }),
                "sigma_range_preset": (["Custom", "High", "Mid", "Low", "All"], {
                    "default": "Custom",
                    "tooltip": (
                        "SIGMA RANGE PRESET\n"
                        "- Auto-configures FBG/CFG start/end sigmas.\n"
                        "- 'Custom': Use manually entered values.\n"
                        "- 'High'/'Mid'/'Low'/'All': Apply guidance over specific sigma ranges."
                    )
                }),
                "conditional_blend_mode": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "CONDITIONAL BLEND MODE\n"
                        "- If True, switches to 'conditional_blend_function' when sigma is below threshold."
                    )
                }),
                "conditional_blend_sigma_threshold": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 100.0, "step": 0.01,
                    "tooltip": (
                        "CONDITIONAL BLEND SIGMA THRESHOLD\n"
                        "- The sigma value below which to switch blend functions.\n"
                        "- Only active if 'conditional_blend_mode' is True."
                    )
                }),
                "conditional_blend_function_name": (tuple(_INTERNAL_BLEND_MODES.keys()), {
                    "default": "slerp",
                    "tooltip": (
                        "CONDITIONAL BLEND FUNCTION\n"
                        "- The blend function to use at low sigmas.\n"
                        "- 'slerp' is recommended to preserve detail."
                    )
                }),
                "conditional_blend_on_change": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "CONDITIONAL BLEND ON CHANGE\n"
                        "- Blends denoised/noisy samples if ancestral noise addition is too high.\n"
                        "- Experimental method to control noise."
                    )
                }),
                "conditional_blend_change_threshold": ("FLOAT", {
                    "default": 0.1, "min": 0.0, "max": 10.0, "step": 0.001,
                    "tooltip": (
                        "CONDITIONAL BLEND CHANGE THRESHOLD\n"
                        "- The noise-to-signal ratio threshold to trigger blending.\n"
                        "- Only active if 'conditional_blend_on_change' is True."
                    )
                }),
                "clamp_noise_norm": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "CLAMP NOISE NORM\n"
                        "- If True, clamps the L2 norm of ancestral noise.\n"
                        "- Can prevent noise from overpowering the sample."
                    )
                }),
                "max_noise_norm": ("FLOAT", {
                    "default": 1.0, "min": 0.01, "max": 100.0, "step": 0.01,
                    "tooltip": (
                        "MAX NOISE NORM\n"
                        "- The maximum L2 norm for ancestral noise.\n"
                        "- Only active if 'clamp_noise_norm' is True."
                    )
                }),
                "log_posterior_ema_factor": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": (
                        "LOG POSTERIOR EMA FACTOR\n"
                        "- Applies Exponential Moving Average (EMA) smoothing to FBG's internal state.\n"
                        "- 0.0 = No smoothing.\n"
                        "- > 0.0 = Smoother, less reactive guidance changes."
                    )
                }),
                "adaptive_noise_scaling": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "ADAPTIVE NOISE SCALING\n"
                        "- Enable adaptive noise scaling based on denoising progress.\n"
                        "- Tries to reduce noise addition in later steps."
                    )
                }),
                "noise_scale_factor": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1,
                    "tooltip": (
                        "NOISE SCALE FACTOR\n"
                        "- Base multiplier for ancestral noise.\n"
                        "- Used by 'adaptive_noise_scaling'."
                    )
                }),
                "progressive_blend_mode": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "PROGRESSIVE BLEND MODE\n"
                        "- Use different blend functions based on progress.\n"
                        "- Starts with 'lerp', moves to 'cosine', ends with 'cubic'."
                    )
                }),
                "enable_profiling": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "ENABLE PROFILING\n"
                        "- Enables internal performance profiling.\n"
                        "- Prints a summary of step timings and VRAM usage to console."
                    )
                }),
                "early_exit_threshold": ("FLOAT", {
                    "default": 1e-6, "min": 1e-10, "max": 1e-2, "step": 1e-7,
                    "tooltip": (
                        "EARLY EXIT THRESHOLD\n"
                        "- Sigma threshold for early convergence exit.\n"
                        "- Stops sampling if sigma OR change norm falls below this."
                    )
                }),
                "tensor_memory_optimization": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "TENSOR MEMORY OPTIMIZATION\n"
                        "- ⚠️ EXPERIMENTAL! Keep FALSE for stability.\n"
                        "- Attempts to free tensors, but is known to cause artifacts."
                    )
                }),
                
                # --- FBG Parameters ---
                "fbg_sampler_mode": (tuple(SamplerMode.__members__), {
                    "default": defaults_fbg_config.sampler_mode.name,
                    "tooltip": (
                        "FBG SAMPLER MODE\n"
                        "- Internal sampler type for FBG calculations.\n"
                        "- 'EULER' or 'PINGPONG'. 'EULER' is standard."
                    )
                }),
                "cfg_scale": ("FLOAT", {
                    "default": defaults_fbg_config.cfg_scale, "min": -1000.0, "max": 1000.0, "step": 0.01,
                    "tooltip": (
                        "CFG SCALE\n"
                        "- The *static* CFG scale to *add* to the dynamic FBG scale.\n"
                        "- 1.0 = No static CFG. > 1.0 = Adds static guidance."
                    )
                }),
                "cfg_start_sigma": ("FLOAT", {
                    "default": defaults_fbg_config.cfg_start_sigma, "min": 0.0, "max": 9999.0, "step": 0.001,
                    "tooltip": (
                        "CFG START SIGMA\n"
                        "- The sigma value to *start* applying static CFG scale.\n"
                        "- Ignored if 'sigma_range_preset' is not 'Custom'."
                    )
                }),
                "cfg_end_sigma": ("FLOAT", {
                    "default": defaults_fbg_config.cfg_end_sigma, "min": 0.0, "max": 9999.0, "step": 0.001,
                    "tooltip": (
                        "CFG END SIGMA\n"
                        "- The sigma value to *stop* applying static CFG scale.\n"
                        "- Ignored if 'sigma_range_preset' is not 'Custom'."
                    )
                }),
                "fbg_start_sigma": ("FLOAT", {
                    "default": defaults_fbg_config.fbg_start_sigma, "min": 0.0, "max": 9999.0, "step": 0.001,
                    "tooltip": (
                        "FBG START SIGMA\n"
                        "- The sigma value to *start* applying dynamic FBG.\n"
                        "- Ignored if 'sigma_range_preset' is not 'Custom'."
                    )
                }),
                "fbg_end_sigma": ("FLOAT", {
                    "default": defaults_fbg_config.fbg_end_sigma, "min": 0.0, "max": 9999.0, "step": 0.001,
                    "tooltip": (
                        "FBG END SIGMA\n"
                        "- The sigma value to *stop* applying dynamic FBG.\n"
                        "- Ignored if 'sigma_range_preset' is not 'Custom'."
                    )
                }),
                "max_guidance_scale": ("FLOAT", {
                    "default": defaults_fbg_config.max_guidance_scale, "min": 1.0, "max": 2000.0, "step": 0.01,
                    "tooltip": (
                        "MAX GUIDANCE SCALE\n"
                        "- The absolute maximum cap for the *total* guidance scale (FBG + CFG).\n"
                        "- Prevents numerical instability from extreme FBG values."
                    )
                }),
                "initial_guidance_scale": ("FLOAT", {
                    "default": defaults_fbg_config.initial_guidance_scale, "min": 1.0, "max": 1000.0, "step": 0.01,
                    "tooltip": (
                        "INITIAL GUIDANCE SCALE\n"
                        "- The guidance scale to use for the very first step, before FBG computes."
                    )
                }),
                "guidance_max_change": ("FLOAT", {
                    "default": defaults_fbg_config.guidance_max_change, "min": 0.0, "max": 1000.0, "step": 0.01,
                    "tooltip": (
                        "GUIDANCE MAX CHANGE\n"
                        "- A dampening factor to limit how much guidance can change in one step (as a percentage).\n"
                        "- Prevents sudden, sharp spikes in guidance."
                    )
                }),
                "pi": ("FLOAT", {
                    "default": defaults_fbg_config.pi, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": (
                        "PI (FBG Parameter)\n"
                        "- Internal FBG tuning parameter (see paper)."
                    )
                }),
                "t_0": ("FLOAT", {
                    "default": defaults_fbg_config.t_0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": (
                        "T_0 (FBG Parameter)\n"
                        "- Internal FBG tuning parameter (see paper)."
                    )
                }),
                "t_1": ("FLOAT", {
                    "default": defaults_fbg_config.t_1, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": (
                        "T_1 (FBG Parameter)\n"
                        "- Internal FBG tuning parameter (see paper)."
                    )
                }),
                "fbg_temp": ("FLOAT", {
                    "default": defaults_fbg_config.temp, "min": -1000.0, "max": 1000.0, "step": 0.0001,
                    "tooltip": (
                        "FBG TEMP\n"
                        "- FBG temperature parameter. Auto-calculated if t_0/t_1 are non-zero."
                    )
                }),
                "fbg_offset": ("FLOAT", {
                    "default": defaults_fbg_config.offset, "min": -1000.0, "max": 1000.0, "step": 0.001,
                    "tooltip": (
                        "FBG OFFSET\n"
                        "- FBG offset parameter. Auto-calculated if t_0/t_1 are non-zero."
                    )
                }),
                "log_posterior_initial_value": ("FLOAT", {
                    "default": defaults_fbg_config.initial_value, "min": -1000.0, "max": 1000.0, "step": 0.01,
                    "tooltip": (
                        "LOG POSTERIOR INITIAL VALUE\n"
                        "- The starting value for FBG's internal state."
                    )
                }),
                "max_posterior_scale": ("FLOAT", {
                    "default": 3.0, "min": 0.0, "max": 100.0, "step": 0.01,
                    "tooltip": (
                        "MAX POSTERIOR SCALE\n"
                        "- The maximum value the internal log_posterior state can reach.\n"
                        "- Acts as a cap on FBG's internal calculations."
                    )
                }),
                "fbg_guidance_multiplier": ("FLOAT", {
                    "default": defaults_fbg_config.fbg_guidance_multiplier, "min": 0.001, "max": 1000.0, "step": 0.01,
                    "tooltip": (
                        "FBG GUIDANCE MULTIPLIER\n"
                        "- A final multiplier applied to the FBG component *before* adding static CFG.\n"
                        "- Scales the FBG-derived guidance up or down."
                    )
                }),
                "fbg_eta": ("FLOAT", {
                    "default": 0.0, "min": -1000.0, "max": 1000.0, "step": 0.01,
                    "tooltip": (
                        "FBG ETA\n"
                        "- FBG eta parameter (not typically used)."
                    )
                }),
                "fbg_s_noise": ("FLOAT", {
                    "default": 1.0, "min": -1000.0, "max": 1000.0, "step": 0.01,
                    "tooltip": (
                        "FBG S_NOISE\n"
                        "- FBG s_noise parameter (not typically used)."
                    )
                }),
                # --- Deprecated / Unused ---
                "ancestral_start_sigma": ("FLOAT", {
                    "default": defaults_fbg_config.ancestral_start_sigma, "min": 0.0, "max": 9999.0, "step": 0.001,
                    "tooltip": (
                        "ANCESTRAL START SIGMA (DEPRECATED)\n"
                        "- This parameter is deprecated.\n"
                        "- Use 'first_ancestral_step' instead."
                    )
                }),
                "ancestral_end_sigma": ("FLOAT", {
                    "default": defaults_fbg_config.ancestral_end_sigma, "min": 0.0, "max": 9999.0, "step": 0.001,
                    "tooltip": (
                        "ANCESTRAL END SIGMA (DEPRECATED)\n"
                        "- This parameter is deprecated.\n"
                        "- Use 'last_ancestral_step' instead."
                    )
                }),
                "gradient_norm_tracking": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "GRADIENT NORM TRACKING (DISABLED)\n"
                        "- This feature is not applicable in a sampling context.\n"
                        "- It has no effect."
                    )
                }),
            },
            "optional": {
                "yaml_settings_str": ("STRING", {
                    "multiline": True, "default": "", "dynamic_prompt": False,
                    "tooltip": (
                        "YAML SETTINGS OVERRIDE\n"
                        "- YAML-formatted string to override any node inputs.\n"
                        "- Allows for complex configurations to be saved/loaded.\n"
                        "- Example: 'debug_mode: 1\n  cfg_scale: 2.5'"
                    )
                }),
                "checkpoint_steps_str": ("STRING", {
                    "default": "",
                    "tooltip": (
                        "CHECKPOINT STEPS\n"
                        "- Comma-separated list of step indices to save internal checkpoints.\n"
                        "- This is a debug feature and does not output anything.\n"
                        "- Example: '10,20,50'"
                    )
                }),
            }
        }
    
    RETURN_TYPES = ("SAMPLER",)
    RETURN_NAMES = ("sampler",)
    FUNCTION = "get_sampler"
    CATEGORY = "MD_Nodes/Sampling"
    
    def get_sampler(
        self,
        step_random_mode,
        step_size,
        seed,
        first_ancestral_step,
        last_ancestral_step,
        ancestral_noise_type,
        start_sigma_index,
        end_sigma_index,
        enable_clamp_output,
        scheduler,
        blend_mode,
        step_blend_mode,
        debug_mode,
        sigma_range_preset,
        conditional_blend_mode,
        conditional_blend_sigma_threshold,
        conditional_blend_function_name,
        conditional_blend_on_change,
        conditional_blend_change_threshold,
        clamp_noise_norm,
        max_noise_norm,
        log_posterior_ema_factor,
        adaptive_noise_scaling,
        noise_scale_factor,
        progressive_blend_mode,
        gradient_norm_tracking,
        enable_profiling,
        early_exit_threshold,
        tensor_memory_optimization,
        fbg_sampler_mode,
        cfg_scale,
        cfg_start_sigma,
        cfg_end_sigma,
        fbg_start_sigma,
        fbg_end_sigma,
        ancestral_start_sigma,
        ancestral_end_sigma,
        max_guidance_scale,
        log_posterior_initial_value,
        initial_guidance_scale,
        guidance_max_change,
        pi,
        t_0,
        t_1,
        fbg_temp,
        fbg_offset,
        fbg_guidance_multiplier,
        fbg_eta,
        fbg_s_noise,
        max_posterior_scale,
        yaml_settings_str="",
        checkpoint_steps_str=""
    ):
        """
        Main execution function. Creates and configures the sampler.
        
        Args:
            (All args from INPUT_TYPES)
        
        Returns:
            Tuple containing (sampler,)
        """
        
        # Parse checkpoint steps
        checkpoint_steps = []
        if checkpoint_steps_str:
            try:
                checkpoint_steps = [int(x.strip()) for x in checkpoint_steps_str.split(",") if x.strip()]
            except ValueError as e:
                print(f"[PingPongSamplerFBG] ⚠️ Warning: Could not parse checkpoint_steps_str: {e}")
        
        # Create FBG config from inputs
        try:
            fbg_sampler_mode_enum = getattr(SamplerMode, fbg_sampler_mode.upper())
        except AttributeError:
            print(f"[PingPongSamplerFBG] ⚠️ Warning: Invalid fbg_sampler_mode '{fbg_sampler_mode}'. Defaulting to EULER.")
            fbg_sampler_mode_enum = SamplerMode.EULER

        fbg_config_instance = FBGConfig(
            sampler_mode=fbg_sampler_mode_enum,
            cfg_start_sigma=cfg_start_sigma,
            cfg_end_sigma=cfg_end_sigma,
            fbg_start_sigma=fbg_start_sigma,
            fbg_end_sigma=fbg_end_sigma,
            ancestral_start_sigma=ancestral_start_sigma,
            ancestral_end_sigma=ancestral_end_sigma,
            cfg_scale=cfg_scale,
            max_guidance_scale=max_guidance_scale,
            initial_guidance_scale=initial_guidance_scale,
            guidance_max_change=guidance_max_change,
            temp=fbg_temp,
            offset=fbg_offset,
            pi=pi,
            t_0=t_0,
            t_1=t_1,
            initial_value=log_posterior_initial_value,
            fbg_guidance_multiplier=fbg_guidance_multiplier,
            max_posterior_scale=max_posterior_scale,
        )
        
        # Compile direct inputs
        direct_inputs = {
            "step_random_mode": step_random_mode,
            "step_size": step_size,
            "seed": seed,
            "pingpong_options": {
                "first_ancestral_step": first_ancestral_step,
                "last_ancestral_step": last_ancestral_step,
            },
            "ancestral_noise_type": ancestral_noise_type,
            "start_sigma_index": start_sigma_index,
            "end_sigma_index": end_sigma_index,
            "enable_clamp_output": enable_clamp_output,
            "scheduler": scheduler,
            "blend_function_name": blend_mode,
            "step_blend_function_name": step_blend_mode,
            "fbg_config": fbg_config_instance._asdict(),
            "eta": fbg_eta,
            "s_noise": fbg_s_noise,
            "max_posterior_scale": max_posterior_scale,
            "debug_mode": debug_mode,
            "sigma_range_preset": sigma_range_preset,
            "conditional_blend_mode": conditional_blend_mode,
            "conditional_blend_sigma_threshold": conditional_blend_sigma_threshold,
            "conditional_blend_function_name": conditional_blend_function_name,
            "conditional_blend_on_change": conditional_blend_on_change,
            "conditional_blend_change_threshold": conditional_blend_change_threshold,
            "clamp_noise_norm": clamp_noise_norm,
            "max_noise_norm": max_noise_norm,
            "log_posterior_ema_factor": log_posterior_ema_factor,
            "adaptive_noise_scaling": adaptive_noise_scaling,
            "noise_scale_factor": noise_scale_factor,
            "progressive_blend_mode": progressive_blend_mode,
            "gradient_norm_tracking": gradient_norm_tracking,
            "enable_profiling": enable_profiling,
            "checkpoint_steps": checkpoint_steps,
            "early_exit_threshold": early_exit_threshold,
            "tensor_memory_optimization": tensor_memory_optimization,
        }
        
        final_options = direct_inputs.copy()
        
        # Process YAML overrides
        if yaml_settings_str:
            try:
                yaml_data = yaml.safe_load(yaml_settings_str)
                if isinstance(yaml_data, dict):
                    for key, value in yaml_data.items():
                        if key == "pingpong_options" and isinstance(value, dict):
                            if key not in final_options:
                                final_options[key] = {}
                            final_options[key].update(value)
                        elif key == "fbg_config" and isinstance(value, dict):
                            if key not in final_options:
                                final_options[key] = {}
                            final_options[key].update(value)
                        elif key == "checkpoint_steps" and isinstance(value, (list, str)):
                            if isinstance(value, str):
                                try:
                                    final_options[key] = [int(x.strip()) for x in value.split(",") if x.strip()]
                                except ValueError:
                                    pass
                            else:
                                final_options[key] = value
                        else:
                            final_options[key] = value
                    if debug_mode >= 1:
                        logging.info(f"[PingPongSamplerFBG] Loaded YAML settings: {list(yaml_data.keys())}")
            except Exception as e:
                print(f"[PingPongSamplerFBG] ⚠️ WARNING: YAML error: {e}. Using direct inputs only.")
        
        # Validation
        try:
            fbg_cfg = final_options.get("fbg_config", {})
            if fbg_cfg.get("max_guidance_scale", 1.0) < 1.0:
                print("[PingPongSamplerFBG] ⚠️ Warning: max_guidance_scale < 1.0, setting to 1.0")
                fbg_cfg["max_guidance_scale"] = 1.0
            
            if fbg_cfg.get("fbg_start_sigma", 1.0) < fbg_cfg.get("fbg_end_sigma", 0.004):
                print("[PingPongSamplerFBG] ⚠️ Warning: fbg_start_sigma should be >= fbg_end_sigma")
            
            ema = final_options.get("log_posterior_ema_factor", 0.0)
            if ema < 0.0 or ema > 1.0:
                final_options["log_posterior_ema_factor"] = max(0.0, min(1.0, ema))
        except Exception as e:
            print(f"[PingPongSamplerFBG] ⚠️ Warning during validation: {e}")
        
        # Create sampler
        # (Must return a tuple with a trailing comma!)
        return (comfy.samplers.KSAMPLER(PingPongSamplerCore.go, extra_options=final_options),)


# =================================================================================
# == Node Registration                                                           ==
# =================================================================================

NODE_CLASS_MAPPINGS = {
    "PingPongSamplerNodeFBG": PingPongSamplerNodeFBG,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PingPongSamplerNodeFBG": "MD: PingPong Sampler (FBG)",
}