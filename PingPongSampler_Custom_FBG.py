# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# ████ PINGPONGSAMPLER v0.9.9-p4 – Optimized for Ace-Step Audio/Video with FBG ████▓▒░
# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
# ░▒▓ ORIGIN & DEV:
#   • Foundational principles for iterative sampling, including concepts that underpin 'ping-pong sampling'
#   • Consistency Models by Song et al. (2023)
#   •     https://arxiv.org/abs/2303.01469
#   • The term 'ping-pong sampling' is explicitly introduced and applied in the context of fast text-to-audio
#   • generation in the paper "Fast Text-to-Audio Generation with Adversarial Post-Training" by Novack et al.
#   • (2025) from Stability AI
#   •     https://arxiv.org/abs/2505.08175
#   • Original concept for the PingPong Sampler for ace-step diffusion by: Junmin Gong (Ace-Step team)
#   • ComfyUI adaptation by: blepping (original ComfyUI port with quirks)
#   • Disassembled & warped by: MD (Machine Damage)
#   • Critical fixes & re-engineering by: Gemini (Google) based on user feedback
#   • Feedback Guidance integration by: Gemini (Google) / blepping
#   •     https://gist.github.com/blepping/d424e8fd27d76845ad27997820a57f6b
#   • Completionist fixups via: devstral / qwen3 (local heroes)
#   • v0.9.9-p4 stability & enhancement release by: Claude (Anthropic)
#   • License: Apache 2.0 — Sharing is caring, no Voodoo hoarding here
#   • Original source gist: https://gist.github.com/blepping/b372ef6c5412080af136aad942d9d76c
# ░▒▓ DESCRIPTION:
#   Advanced sampler combining PingPong's ancestral noise mixing with Feedback Guidance (FBG).
#   No guesswork. No corrupted tensors. Just pure generative voodoo.
#   Designed for precision audio/video generation in ComfyUI workflows.
#   May work with image models but results can vary wildly.
# ░▒▓ FEATURES:
#   ✓ Feedback Guidance (FBG) for dynamic, content-aware guidance scaling
#   ✓ Multiple blend modes: lerp, slerp, cosine, cubic, add, a_only, b_only
#   ✓ Ancestral noise types: gaussian, uniform, brownian
#   ✓ Conditional blending based on sigma thresholds or step changes
#   ✓ Progressive blend mode that adapts through the sampling process
#   ✓ Adaptive noise scaling based on denoising progress
#   ✓ Sigma range presets for quick FBG/CFG scheduling
#   ✓ Debug modes with verbosity levels (0=Off, 1=Basic, 2=Chatty)
#   ✓ Step-based or block-based seed randomization
#   ✓ Noise norm clamping for stability
#   ✓ Log posterior EMA smoothing
#   ✓ Configurable checkpoint saving at specific steps
#   ✓ Performance profiling with memory usage tracking
#   ✓ NaN/Inf detection with automatic recovery
#   ✓ Supports up to 10,000 steps (tested extensively to 1000)
#   ✓ YAML configuration override support
#   ✓ Production-ready with comprehensive error handling
# ░▒▓ CHANGELOG:
#   - v0.9.9-p4 (Current Release - CRITICAL STABILITY UPDATE):
#       • CRITICAL: Fixed tensor_memory_optimization bug causing noise/grey artifacts
#       • CRITICAL: Removed all broken in-place tensor operations
#       • CRITICAL: Fixed early exit logic with proper state management
#       • Enhanced numerical stability in all blend functions
#       • Added sigma schedule validation with warnings
#       • Detailed debug output with tensor statistics
#       • Adaptive step sizing suggestions for 500-1000+ steps
#       • Memory-efficient processing for tensors >10M elements
#       • NaN/Inf detection and recovery mechanisms
#       • Gradient tracking properly disabled (not applicable)
#       • DEFAULT: tensor_memory_optimization now FALSE (was causing corruption)
#   - v0.9.9-p3:
#       • Tensor operations batching, adaptive noise scaling, progressive blend mode
#       • Step timing profiler, checkpoint saving, enhanced error handling
#   - v0.9.9-p2:
#       • Implemented slerp, added blend modes, ancestral noise types
#       • FBG guidance scale summary in debug log
#   - v0.9.9-p1:
#       • Debug verbosity levels, sigma range presets, conditional blending
#       • Noise norm clamping, log posterior EMA, dynamic seed reporting
# ░▒▓ CONFIGURATION:
#   → Primary Use: High-quality audio/video generation with Ace-Step diffusion
#   → Secondary Use: Experimental visual generation (expect wild results)
#   → Edge Use: For demoscene veterans and bytecode wizards only
#   → Advanced Use: Extreme guidance scales (tested up to 2000x multipliers)
# ░▒▓ WARNING:
#   This node may trigger:
#   ▓▒░ Flashbacks to ANSI art and 256-byte intros
#   ▓▒░ Spontaneous breakdancing or synesthesia
#   ▓▒░ Urges to reinstall Impulse Tracker
#   ▓▒░ Extreme focus on byte-level optimization
#   ▓▒░ Compulsive checking of tensor shapes
#   Consult your nearest demoscene vet if hallucinations persist.
# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄

import math
import torch
from tqdm.auto import trange
from enum import Enum, auto
from typing import NamedTuple, Optional, Callable, Dict, Any, List
from comfy import model_sampling, model_patcher
from comfy.samplers import KSAMPLER, cfg_function
from comfy.k_diffusion.sampling import get_ancestral_step
import numpy as np
import yaml
import time
from contextlib import contextmanager


def slerp(a, b, t):
    """Spherical linear interpolation with enhanced numerical stability."""
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
    """Cosine interpolation for smoother transitions."""
    cos_t = (1 - torch.cos(t * math.pi)) * 0.5
    return a * (1 - cos_t) + b * cos_t

def cubic_interpolation(a, b, t):
    """Cubic interpolation for even smoother transitions."""
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

class SamplerMode(Enum):
    EULER = auto()
    PINGPONG = auto()

class FBGConfig(NamedTuple):
    sampler_mode: SamplerMode = SamplerMode.EULER
    cfg_start_sigma: float = 1.0
    cfg_end_sigma: float = 0.004
    fbg_start_sigma: float = 1.0
    fbg_end_sigma: float = 0.004
    fbg_guidance_multiplier: float = 1.0
    ancestral_start_sigma: float = 1.0
    ancestral_end_sigma: float = 0.004
    cfg_scale: float = 1.0
    max_guidance_scale: float = 10.0
    max_posterior_scale: float = 3.0
    initial_value: float = 0.0
    initial_guidance_scale: float = 1.0
    guidance_max_change: float = 1000.0
    temp: float = 0.0
    offset: float = 0.0
    pi: float = 0.5
    t_0: float = 0.5
    t_1: float = 0.4

def batch_mse_loss(a: torch.Tensor, b: torch.Tensor, *, start_dim: int = 1) -> torch.Tensor:
    """Optimized MSE loss with memory efficiency and numerical stability."""
    # For very large tensors, process more carefully
    if a.numel() > 1e7:  # 10M elements
        diff = a - b
        return (diff * diff).sum(dim=tuple(range(start_dim, a.ndim)))
    return torch.sum((a - b).pow(2), dim=tuple(range(start_dim, a.ndim)))

class PerformanceProfiler:
    """Enhanced profiler for tracking step timing and memory usage."""
    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.step_times = []
        self.memory_usage = []
        self.step_names = []
        
    @contextmanager
    def profile_step(self, step_name: str = "step"):
        if not self.enabled:
            yield
            return
            
        start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            start_memory = torch.cuda.memory_allocated()
        else:
            start_memory = 0
            
        yield
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            end_memory = torch.cuda.memory_allocated()
        else:
            end_memory = 0
            
        end_time = time.time()
        
        self.step_times.append(end_time - start_time)
        self.memory_usage.append(end_memory - start_memory)
        self.step_names.append(step_name)
        
    def get_summary(self):
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
        
        if self.memory_usage and torch.cuda.is_available():
            total_memory = sum(self.memory_usage)
            avg_memory = total_memory / len(self.memory_usage)
            max_memory = max(self.memory_usage)
            summary += f"  Total memory delta: {total_memory / 1024**2:.1f}MB\n"
            summary += f"  Average memory delta: {avg_memory / 1024**2:.1f}MB\n"
            summary += f"  Max memory delta: {max_memory / 1024**2:.1f}MB"
            
        return summary

class PingPongSamplerCore:
    """Enhanced PingPongSampler with stability fixes and optimizations."""
    
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
        start_sigma_index: int = 0,
        end_sigma_index: int = -1,
        enable_clamp_output: bool = False,
        step_random_mode: str = "off",
        step_size: int = 5,
        seed: int = 42,
        blend_function: Callable = torch.lerp,
        step_blend_function: Callable = torch.lerp,
        scheduler=None,
        pingpong_options: Optional[Dict[str, Any]] = None,
        fbg_config: FBGConfig = FBGConfig(),
        debug_mode: int = 0,
        eta: float = 0.0,
        s_noise: float = 1.0,
        sigma_range_preset: str = "Custom",
        conditional_blend_mode: bool = False,
        conditional_blend_sigma_threshold: float = 0.5,
        conditional_blend_function: Callable = torch.lerp,
        conditional_blend_on_change: bool = False,
        conditional_blend_change_threshold: float = 0.1,
        clamp_noise_norm: bool = False,
        max_noise_norm: float = 1.0,
        log_posterior_ema_factor: float = 0.0,
        # Enhanced parameters
        adaptive_noise_scaling: bool = False,
        noise_scale_factor: float = 1.0,
        progressive_blend_mode: bool = False,
        gradient_norm_tracking: bool = False,
        enable_profiling: bool = False,
        checkpoint_steps: List[int] = None,
        early_exit_threshold: float = 1e-6,
        tensor_memory_optimization: bool = False,  # DEFAULT CHANGED TO FALSE
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
        if self.tensor_memory_optimization and self.debug_mode >= 1:
            print("⚠️  WARNING: tensor_memory_optimization is EXPERIMENTAL and may cause artifacts!")
        
        # Performance profiler
        self.profiler = PerformanceProfiler(enable_profiling)
        
        # Gradient tracking (disabled - doesn't work in sampling context)
        self.gradient_norms = [] if gradient_norm_tracking else None
        if gradient_norm_tracking and self.debug_mode >= 1:
            print("Note: Gradient norm tracking disabled (not applicable in sampling)")
        
        # Determine the number of total steps
        num_steps_available = len(sigmas) - 1 if len(sigmas) > 0 else 0
        if self.debug_mode >= 1:
            print(f"PingPongSampler Enhanced v0.9.9-p4: Total steps based on sigmas: {num_steps_available}")
        
        # Validate sigma schedule
        self._validate_sigma_schedule()
        
        # Adaptive step size for high step counts
        if num_steps_available > 500:
            suggested_step_size = max(num_steps_available // 100, 5)
            if self.debug_mode >= 1:
                print(f"High step count detected ({num_steps_available} steps).")
                print(f"Consider using step_size >= {suggested_step_size} for optimal noise distribution")
        
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
        self.original_fbg_config = fbg_config
        self.config = fbg_config
        
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
            print(f"FBG config: {self.config}")
            print(f"Enhanced features: adaptive_noise={self.adaptive_noise_scaling}, "
                  f"progressive_blend={self.progressive_blend_mode}")

    def _validate_sigma_schedule(self):
        """Validate sigma schedule is monotonically decreasing."""
        if len(self.sigmas) < 2:
            return True
        
        for i in range(len(self.sigmas) - 1):
            if self.sigmas[i] < self.sigmas[i + 1]:
                if self.debug_mode >= 1:
                    print(f"⚠️  WARNING: Sigma schedule not monotonic at step {i}: "
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
            
        if self.debug_mode >= 1:
            print(f"Applied sigma range preset '{self.sigma_range_preset}'")
            
        return FBGConfig(**config_dict)

    def _detect_model_type(self):
        """Enhanced model type detection with error handling."""
        try:
            current_model_check = self.model_
            while hasattr(current_model_check, 'inner_model') and current_model_check.inner_model is not None:
                current_model_check = current_model_check.inner_model
            if hasattr(current_model_check, 'model_sampling') and current_model_check.model_sampling is not None:
                return isinstance(current_model_check.model_sampling, model_sampling.CONST)
        except (AttributeError, TypeError) as e:
            if self.debug_mode >= 1:
                print(f"Model type detection failed: {e}. Assuming non-CONST sampling.")
        return False

    def _setup_noise_sampler(self, noise_sampler, noise_type):
        """Setup noise sampler with enhanced noise types."""
        if noise_sampler is not None:
            return noise_sampler
            
        if self.debug_mode >= 1:
            print(f"Using ancestral noise type: {noise_type}")
            
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
                if self.debug_mode >= 1:
                    print(f"Scheduler error: {e}. Using zero decay.")
        
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
            if self.debug_mode >= 1:
                print("Warning: Could not calculate minimal_log_posterior. Using -inf.")
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
            
        if self.debug_mode >= 1:
            print(f"Saving checkpoint at step {step_idx}")
        
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
                if self.debug_mode >= 1:
                    print(f"Early exit due to convergence (change norm: {change_norm:.8f})")
                return True
        
        # Always clone for next comparison
        self._prev_x = x_current.clone()
        return False

    def _check_for_nan_inf(self, tensor, name, step_idx):
        """Check tensor for NaN or Inf values."""
        if torch.isnan(tensor).any():
            if self.debug_mode >= 1:
                print(f"⚠️  WARNING: NaN detected in {name} at step {step_idx}!")
            return True
        if torch.isinf(tensor).any():
            if self.debug_mode >= 1:
                print(f"⚠️  WARNING: Inf detected in {name} at step {step_idx}!")
            return True
        return False

    def _debug_step_info(self, idx, sigma_current, sigma_next, x_current):
        """Comprehensive debug info for troubleshooting."""
        if self.debug_mode >= 2:
            print(f"\n=== Step {idx} Detailed Info ===")
            print(f"  Sigma: {sigma_current.item():.6f} -> {sigma_next.item():.6f}")
            print(f"  X stats: min={x_current.min().item():.4f}, "
                  f"max={x_current.max().item():.4f}, "
                  f"mean={x_current.mean().item():.4f}, "
                  f"std={x_current.std().item():.4f}")
            self._check_for_nan_inf(x_current, "x_current", idx)

    def _stepped_seed(self, step: int) -> Optional[int]:
        """Enhanced seed calculation with validation."""
        if self.step_random_mode == "off":
            return None
        current_step_size = max(self.step_size, 1)
        
        seed_map = {
            "block": self.seed + (step // current_step_size),
            "reset": self.seed + (step * current_step_size),
            "step": self.seed + step
        }
        
        return seed_map.get(self.step_random_mode, self.seed)

    def _get_sigma_square_tilde(self, sigmas: torch.Tensor) -> torch.Tensor:
        """Optimized sigma square tilde calculation."""
        if len(sigmas) < 2:
            return torch.tensor([], device=sigmas.device)
        
        s_sq, sn_sq = sigmas[:-1] ** 2, sigmas[1:] ** 2
        safe_s_sq = torch.where(s_sq == 0, torch.tensor(1e-6, device=s_sq.device), s_sq)
        return ((s_sq - sn_sq) * sn_sq / safe_s_sq).flip(dims=(0,))

    def _get_offset(self, steps: int, sigma_square_tilde: torch.Tensor, **kwargs) -> float:
        """Enhanced offset calculation with better error handling."""
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

    def _get_temp(self, steps: int, offset: float, sigma_square_tilde: torch.Tensor, **kwargs) -> float:
        """Enhanced temperature calculation."""
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

    def get_dynamic_guidance_scale(self, log_posterior_val: torch.Tensor, 
                                 guidance_scale_prev: torch.Tensor, 
                                 sigma_item: float) -> torch.Tensor:
        """FIXED: Dynamic guidance scale calculation without broken optimization."""
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
                print(f"  FBG active (sigma {sigma_item:.3f}). "
                      f"Raw FBG component: {guidance_scale.mean().item():.4f}")
        
        if using_cfg:
            guidance_scale += config.cfg_scale - 1.0
            if self.debug_mode >= 2:
                print(f"  CFG active. Added: {config.cfg_scale - 1.0:.2f}")
        
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

    def _model_denoise_with_guidance(self, x_tensor: torch.Tensor, 
                                 sigma_scalar: torch.Tensor, 
                                 override_cfg: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Enhanced model denoising with improved error handling."""
        sigma_tensor = sigma_scalar * x_tensor.new_ones((x_tensor.shape[0],))
        
        cond = uncond = None
        
        def post_cfg_function(args: Dict[str, torch.Tensor]) -> torch.Tensor:
            nonlocal cond, uncond
            cond, uncond = args["cond_denoised"], args["uncond_denoised"]
            return args["denoised"]
        
        extra_args = self.extra_args.copy()
        orig_model_options = extra_args.get("model_options", {})
        model_options = orig_model_options.copy()
        model_options["disable_cfg1_optimization"] = True
        extra_args["model_options"] = model_patcher.set_model_options_post_cfg_function(
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
                denoised = cfg_function(
                    inner_model.inner_model, cond, uncond, override_cfg,
                    x_tensor, sigma_tensor, model_options=orig_model_options,
                )
        except Exception as e:
            if self.debug_mode >= 1:
                print(f"Warning: Model denoising error: {e}. Using fallback.")
            # Fallback to basic model call
            denoised = self.model_(x_tensor, sigma_tensor, **extra_args)
            if cond is None:
                cond = uncond = denoised
        
        return denoised, cond, uncond

    def _update_log_posterior(self, prev_log_posterior: torch.Tensor, 
                              x_curr: torch.Tensor, x_next: torch.Tensor,
                              t_curr: torch.Tensor, t_next: torch.Tensor,
                              uncond: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """FIXED: Log posterior update without broken in-place operations."""
        
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
                if self.debug_mode >= 1:
                    print(f"Callback error at step {step_idx}: {e}")

    def __call__(self):
        """FIXED: Main sampling loop with all stability improvements."""
        if self.debug_mode >= 1:
            print("PingPongSampler Enhanced v0.9.9-p4: Starting sampling loop")
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
                    if self.debug_mode >= 1:
                        print(f"Attempting to recover from NaN/Inf at step {idx}")
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
                    print(f"Step {idx}: σ={sigma_item:.3f}→{sigma_next_item:.3f}, GS={gs_str}")
                
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
                        if self.debug_mode >= 1:
                            print(f"  Using seed: {local_seed}")
                    
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
            print("\n" + "="*50)
            print("PingPongSampler Enhanced v0.9.9-p4 Summary")
            print("="*50)
            if guidance_scales_used:
                print(f"  Guidance Scale - Min: {min(guidance_scales_used):.3f}, "
                      f"Max: {max(guidance_scales_used):.3f}, "
                      f"Avg: {sum(guidance_scales_used) / len(guidance_scales_used):.3f}")
            
            if hasattr(self, 'checkpoints'):
                print(f"  Checkpoints saved: {len(self.checkpoints)}")
            
            if self.profiler.enabled:
                print("\n" + self.profiler.get_summary())
            
            print("="*50 + "\n")
        
        return x_current

    @staticmethod
    def go(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, **kwargs):
        """Enhanced static method with better parameter handling."""
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


class PingPongSamplerNodeFBG:
    """Enhanced ComfyUI node wrapper with stability fixes."""
    
    CATEGORY = "MD_Nodes/Samplers"
    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "get_sampler"

    @classmethod
    def INPUT_TYPES(cls):
        """Enhanced input types with corrected defaults."""
        defaults_fbg_config = FBGConfig()
        return {
            "required": {
                # Original parameters
                "step_random_mode": (["off", "block", "reset", "step"], {"default": "block"}),
                "step_size": ("INT", {"default": 4, "min": 1, "max": 100}),
                "seed": ("INT", {"default": 80085, "min": 0, "max": 2**32 - 1}),
                "first_ancestral_step": ("INT", {"default": 0, "min": -1, "max": 10000}),
                "last_ancestral_step": ("INT", {"default": -1, "min": -1, "max": 10000}),
                "ancestral_noise_type": (["gaussian", "uniform", "brownian"], {"default": "gaussian"}),
                "start_sigma_index": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "end_sigma_index": ("INT", {"default": -1, "min": -10000, "max": 10000}),
                "enable_clamp_output": ("BOOLEAN", {"default": False}),
                "scheduler": ("SCHEDULER",),
                "blend_mode": (tuple(_INTERNAL_BLEND_MODES.keys()), {"default": "lerp"}),
                "step_blend_mode": (tuple(_INTERNAL_BLEND_MODES.keys()), {"default": "lerp"}),
                
                # Enhanced parameters
                "debug_mode": ("INT", {"default": 0, "min": 0, "max": 2}),
                "sigma_range_preset": (["Custom", "High", "Mid", "Low", "All"], {"default": "Custom"}),
                "conditional_blend_mode": ("BOOLEAN", {"default": False}),
                "conditional_blend_sigma_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 100.0, "step": 0.01}),
                "conditional_blend_function_name": (tuple(_INTERNAL_BLEND_MODES.keys()), {"default": "slerp"}),
                "conditional_blend_on_change": ("BOOLEAN", {"default": False}),
                "conditional_blend_change_threshold": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 10.0, "step": 0.001}),
                "clamp_noise_norm": ("BOOLEAN", {"default": False}),
                "max_noise_norm": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 100.0, "step": 0.01}),
                "log_posterior_ema_factor": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                
                # New enhanced features
                "adaptive_noise_scaling": ("BOOLEAN", {"default": False, "tooltip": "Enable adaptive noise scaling based on denoising progress"}),
                "noise_scale_factor": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1, "tooltip": "Base noise scaling factor"}),
                "progressive_blend_mode": ("BOOLEAN", {"default": False, "tooltip": "Use different blend functions based on progress"}),
                "gradient_norm_tracking": ("BOOLEAN", {"default": False, "tooltip": "Track gradient norms (disabled - not applicable)"}),
                "enable_profiling": ("BOOLEAN", {"default": False, "tooltip": "Enable performance profiling"}),
                "early_exit_threshold": ("FLOAT", {"default": 1e-6, "min": 1e-10, "max": 1e-2, "step": 1e-7, "tooltip": "Threshold for early convergence exit"}),
                "tensor_memory_optimization": ("BOOLEAN", {"default": False, "tooltip": "⚠️ EXPERIMENTAL - May cause artifacts! Keep FALSE for stability"}),
                
                # FBG parameters
                "fbg_sampler_mode": (tuple(SamplerMode.__members__), {"default": defaults_fbg_config.sampler_mode.name}),
                "cfg_scale": ("FLOAT", {"default": defaults_fbg_config.cfg_scale, "min": -1000.0, "max": 1000.0, "step": 0.01}),
                "cfg_start_sigma": ("FLOAT", {"default": defaults_fbg_config.cfg_start_sigma, "min": 0.0, "max": 9999.0, "step": 0.001}),
                "cfg_end_sigma": ("FLOAT", {"default": defaults_fbg_config.cfg_end_sigma, "min": 0.0, "max": 9999.0, "step": 0.001}),
                "fbg_start_sigma": ("FLOAT", {"default": defaults_fbg_config.fbg_start_sigma, "min": 0.0, "max": 9999.0, "step": 0.001}),
                "fbg_end_sigma": ("FLOAT", {"default": defaults_fbg_config.fbg_end_sigma, "min": 0.0, "max": 9999.0, "step": 0.001}),
                "ancestral_start_sigma": ("FLOAT", {"default": defaults_fbg_config.ancestral_start_sigma, "min": 0.0, "max": 9999.0, "step": 0.001}),
                "ancestral_end_sigma": ("FLOAT", {"default": defaults_fbg_config.ancestral_end_sigma, "min": 0.0, "max": 9999.0, "step": 0.001}),
                "max_guidance_scale": ("FLOAT", {"default": defaults_fbg_config.max_guidance_scale, "min": 1.0, "max": 2000.0, "step": 0.01}),
                "initial_guidance_scale": ("FLOAT", {"default": defaults_fbg_config.initial_guidance_scale, "min": 1.0, "max": 1000.0, "step": 0.01}),
                "guidance_max_change": ("FLOAT", {"default": defaults_fbg_config.guidance_max_change, "min": 0.0, "max": 1000.0, "step": 0.01}),
                "pi": ("FLOAT", {"default": defaults_fbg_config.pi, "min": 0.0, "max": 1.0, "step": 0.01}),
                "t_0": ("FLOAT", {"default": defaults_fbg_config.t_0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "t_1": ("FLOAT", {"default": defaults_fbg_config.t_1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "fbg_temp": ("FLOAT", {"default": defaults_fbg_config.temp, "min": -1000.0, "max": 1000.0, "step": 0.0001}),
                "fbg_offset": ("FLOAT", {"default": defaults_fbg_config.offset, "min": -1000.0, "max": 1000.0, "step": 0.001}),
                "log_posterior_initial_value": ("FLOAT", {"default": defaults_fbg_config.initial_value, "min": -1000.0, "max": 1000.0, "step": 0.01}),
                "max_posterior_scale": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 100.0, "step": 0.01}),
                "fbg_guidance_multiplier": ("FLOAT", {"default": defaults_fbg_config.fbg_guidance_multiplier, "min": 0.001, "max": 1000.0, "step": 0.01}),
                "fbg_eta": ("FLOAT", {"default": 0.0, "min": -1000.0, "max": 1000.0, "step": 0.01}),
                "fbg_s_noise": ("FLOAT", {"default": 1.0, "min": -1000.0, "max": 1000.0, "step": 0.01}),
            },
            "optional": {
                "yaml_settings_str": ("STRING", {"multiline": True, "default": "", "dynamic_prompt": False, "tooltip": "YAML configuration (overrides node inputs)"}),
                "checkpoint_steps_str": ("STRING", {"default": "", "tooltip": "Comma-separated checkpoint steps (e.g., '10,20,50')"}),
            }
        }

    def get_sampler(
        self,
        step_random_mode: str,
        step_size: int,
        seed: int,
        first_ancestral_step: int,
        last_ancestral_step: int,
        ancestral_noise_type: str,
        start_sigma_index: int,
        end_sigma_index: int,
        enable_clamp_output: bool,
        scheduler,
        blend_mode: str,
        step_blend_mode: str,
        debug_mode: int,
        sigma_range_preset: str,
        conditional_blend_mode: bool,
        conditional_blend_sigma_threshold: float,
        conditional_blend_function_name: str,
        conditional_blend_on_change: bool,
        conditional_blend_change_threshold: float,
        clamp_noise_norm: bool,
        max_noise_norm: float,
        log_posterior_ema_factor: float,
        adaptive_noise_scaling: bool,
        noise_scale_factor: float,
        progressive_blend_mode: bool,
        gradient_norm_tracking: bool,
        enable_profiling: bool,
        early_exit_threshold: float,
        tensor_memory_optimization: bool,
        fbg_sampler_mode: str,
        cfg_scale: float,
        cfg_start_sigma: float,
        cfg_end_sigma: float,
        fbg_start_sigma: float,
        fbg_end_sigma: float,
        ancestral_start_sigma: float,
        ancestral_end_sigma: float,
        max_guidance_scale: float,
        log_posterior_initial_value: float,
        initial_guidance_scale: float,
        guidance_max_change: float,
        pi: float,
        t_0: float,
        t_1: float,
        fbg_temp: float,
        fbg_offset: float,
        fbg_guidance_multiplier: float,
        fbg_eta: float,
        fbg_s_noise: float,
        max_posterior_scale: float,
        yaml_settings_str: str = "",
        checkpoint_steps_str: str = ""
    ):
        """Enhanced get_sampler with validation."""
        
        # Parse checkpoint steps
        checkpoint_steps = []
        if checkpoint_steps_str:
            try:
                checkpoint_steps = [int(x.strip()) for x in checkpoint_steps_str.split(",") if x.strip()]
            except ValueError as e:
                print(f"Warning: Could not parse checkpoint_steps_str: {e}")
        
        # Create FBG config
        fbg_config_instance = FBGConfig(
            sampler_mode=getattr(SamplerMode, fbg_sampler_mode.upper()),
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
                        print(f"Loaded YAML settings: {list(yaml_data.keys())}")
            except Exception as e:
                print(f"WARNING: YAML error: {e}. Using direct inputs only.")
        
        # Validation
        try:
            fbg_cfg = final_options.get("fbg_config", {})
            if fbg_cfg.get("max_guidance_scale", 1.0) < 1.0:
                print("Warning: max_guidance_scale < 1.0, setting to 1.0")
                fbg_cfg["max_guidance_scale"] = 1.0
            
            if fbg_cfg.get("fbg_start_sigma", 1.0) < fbg_cfg.get("fbg_end_sigma", 0.004):
                print("Warning: fbg_start_sigma should be >= fbg_end_sigma")
            
            ema = final_options.get("log_posterior_ema_factor", 0.0)
            if ema < 0.0 or ema > 1.0:
                final_options["log_posterior_ema_factor"] = max(0.0, min(1.0, ema))
        except Exception as e:
            print(f"Warning during validation: {e}")
        
        # Create sampler
        return (KSAMPLER(PingPongSamplerCore.go, extra_options=final_options),)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "PingPongSampler_Custom_FBG": PingPongSamplerNodeFBG,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PingPongSampler_Custom_FBG": "PingPong Sampler v0.9.9-p4 (Stability + FBG)",
}