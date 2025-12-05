# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
# ████ MD_Nodes/PingPongSamplerFBG – Ace-Step FBG + Res 2 Restarts ████▓▒░
# © 2025 MDMAchine
# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
# ░▒▓ ORIGIN & DEV:
#   • Cast into the void by: Junmin Gong (Concept), blepping (ComfyUI Port), MD (Adaptation)
#   • Enhanced by: Gemini, Claude (v0.9.9-p4), devstral/qwen3 (Fixups)
#   • License: Apache 2.0 — Sharing is caring

# ░▒▓ DESCRIPTION:
#   Advanced ancestral sampler combining PingPong's noise mixing with Feedback
#   Guidance (FBG) for dynamic, content-aware guidance scaling. Now includes
#   "Res 2" restart logic for iterative error correction during sampling.

# ░▒▓ FEATURES:
#   ✓ Feedback Guidance (FBG) for dynamic guidance scaling.
#   ✓ Res 2 Restart Logic: Iteratively "re-noises" and re-samples to correct errors.
#   ✓ Multiple blend modes (lerp, slerp, cosine, cubic, add, etc.).
#   ✓ Ancestral noise types: gaussian, uniform, brownian.
#   ✓ JS-Safe Seed Precision (Enterprise Standard).

# ░▒▓ CHANGELOG:
#   - v1.5.0 (Enterprise Standards):
#       • FIX: Robust CFGDenoiser import (Sec 5.3).
#       • COMPLIANCE: Removed type hints from signatures, added constants.
#       • NEW: Embedded unit tests and JS-safe seed clamping.

# ░▒▓ CONFIGURATION:
#   → Primary Use: High-quality audio/video generation with Ace-Step diffusion.
#   → Secondary Use: "Res 2" restarts for complex coherence.

# ░▒▓ WARNING:
#   This node may trigger:
#   ▓▒░ A sudden urge to `slerp` your breakfast cereal.
#   ▓▒░ Existential dread as FBG pushes guidance scale over 9000.
# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄

# =================================================================================
# == Standard Library Imports
# =================================================================================
import math
import yaml
import time
import logging
import traceback
import sys
from enum import Enum, auto
from typing import NamedTuple, Optional, Callable, Dict, Any, List, Union
from contextlib import contextmanager

# =================================================================================
# == Third-Party Imports
# =================================================================================
import torch
import numpy as np
from tqdm.auto import trange

# =================================================================================
# == ComfyUI Core Modules
# =================================================================================
from comfy import model_sampling, model_patcher
from comfy.samplers import KSAMPLER, cfg_function
import comfy.model_management

# Setup logger
logger = logging.getLogger("ComfyUI_MD_Nodes.PingPongSamplerFBG")

# [NODE_FIX] Robust import for CFGDenoiser (Sec 5.3)
try:
    from comfy.k_diffusion.sampling import CFGDenoiser
    logger.info("✓ CFGDenoiser imported from comfy.k_diffusion.sampling")
except ImportError:
    try:
        from comfy.samplers import CFGDenoiser
        logger.info("✓ CFGDenoiser imported from comfy.samplers")
    except ImportError:
        logger.debug("CFGDenoiser not available - using type fallback (normal for some ComfyUI versions)")
        CFGDenoiser = Any # type: ignore

# =================================================================================
# == Configuration Constants
# =================================================================================

# Seed Management
CONST_JS_MAX_SAFE_INTEGER = 9007199254740991  # 2^53 - 1 (JavaScript safe range)
CONST_SEED_MIN = 0

# Math & Stability
CONST_EPSILON = 1e-8
CONST_EARLY_EXIT_THRESHOLD = 1e-6

# =================================================================================
# == Helper Classes & Dependencies
# =================================================================================

# --- Blend Mode Functions ---
def slerp(a, b, t):
    """Spherical linear interpolation with enhanced numerical stability."""
    eps = CONST_EPSILON
    # 1. Flatten to treat as single vector
    a_flat = a.flatten(start_dim=1).float()
    b_flat = b.flatten(start_dim=1).float().to(a.device)
    
    # 2. Calculate magnitudes
    a_norm = torch.norm(a_flat, dim=-1, keepdim=True)
    b_norm = torch.norm(b_flat, dim=-1, keepdim=True)
    
    # 3. Normalize
    a_n = a_flat / (a_norm + eps)
    b_n = b_flat / (b_norm + eps)
    
    # 4. Calculate angle
    dot = (a_n * b_n).sum(dim=-1, keepdim=True).clamp(-0.9999, 0.9999)
    
    if torch.all(torch.abs(dot) > 0.9995):
        return torch.lerp(a, b, t)
        
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    
    # 5. Spherical interpolation
    t_tensor = torch.tensor(t, device=a.device, dtype=a.dtype)
    scale_a = torch.sin((1 - t_tensor) * theta) / (sin_theta + eps)
    scale_b = torch.sin(t_tensor * theta) / (sin_theta + eps)
    result_n = scale_a * a_n + scale_b * b_n
    
    # 6. Interpolate magnitude
    result_norm = torch.lerp(a_norm, b_norm, t_tensor)
    
    # 7. Apply magnitude and reshape
    result_flat = result_n * result_norm
    return result_flat.reshape_as(a)

def cosine_interpolation(a, b, t):
    """Cosine interpolation."""
    t_tensor = torch.tensor(t * math.pi, device=a.device)
    cos_t = (1.0 - torch.cos(t_tensor)) * 0.5
    return a * (1.0 - cos_t) + b * cos_t

def cubic_interpolation(a, b, t):
    """Cubic interpolation."""
    t_tensor = torch.tensor(t, device=a.device)
    cubic_t = t_tensor * t_tensor * (3.0 - 2.0 * t_tensor)
    return torch.lerp(a, b, cubic_t)

# Global dictionary of available blend modes
_INTERNAL_BLEND_MODES = {
    "lerp": torch.lerp,
    "slerp": slerp,
    "cosine": cosine_interpolation,
    "cubic": cubic_interpolation,
    "add": lambda a, b, t: a * (1 - t) + b * t,
    "a_only": lambda a, _b, _t: a,
    "b_only": lambda _a, b, _t: b
}

# --- Enums and Named Tuples ---
class SamplerMode(Enum):
    """Enum for FBG sampler modes."""
    EULER = auto()
    PINGPONG = auto()

class FBGConfig(NamedTuple):
    """Configuration for Feedback Guidance (FBG)."""
    sampler_mode: SamplerMode = SamplerMode.PINGPONG
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

# --- Helper Functions ---
def batch_mse_loss(a, b, start_dim=1):
    """Calculate Mean Squared Error."""
    a = a.float()
    b = b.float().to(a.device)
    if a.numel() > 1e7:
        diff = a - b
        return (diff * diff).sum(dim=tuple(range(start_dim, a.ndim)))
    return torch.sum((a - b).pow(2), dim=tuple(range(start_dim, a.ndim)))

def validate_seed(seed_value):
    """
    Ensure seed is within JavaScript-safe range.
    """
    try:
        int_value = int(seed_value)
    except (ValueError, TypeError):
        return CONST_SEED_MIN
    return max(CONST_SEED_MIN, min(int_value, CONST_JS_MAX_SAFE_INTEGER))

# --- Performance Profiler ---
class PerformanceProfiler:
    """Profiler for step timing, memory usage, and restart tracking."""
    def __init__(self, enabled=False):
        self.enabled = enabled
        self.step_times = []
        self.memory_usage = []
        self.step_names = []
        self.restart_count = 0
        self.restart_steps = []

    @contextmanager
    def profile_step(self, step_name="step"):
        """Context manager to profile a block."""
        if not self.enabled:
            yield
            return
        
        device = comfy.model_management.get_torch_device()
        start_time = time.time()
        start_mem = 0
        if device.type != 'cpu':
            torch.cuda.synchronize(device)
            start_mem = torch.cuda.memory_allocated(device)
            
        try:
            yield
        finally:
            if device.type != 'cpu':
                torch.cuda.synchronize(device)
                end_mem = torch.cuda.memory_allocated(device)
                self.memory_usage.append(end_mem - start_mem)
            else:
                 self.memory_usage.append(0)
            
            self.step_times.append(time.time() - start_time)
            self.step_names.append(step_name)

    def log_restart(self, step_index, sigma_curr, sigma_next):
        """Log a restart event."""
        if not self.enabled: return
        self.restart_count += 1
        s_c = float(sigma_curr.item()) if isinstance(sigma_curr, torch.Tensor) else float(sigma_curr)
        s_n = float(sigma_next.item()) if isinstance(sigma_next, torch.Tensor) else float(sigma_next)
        self.restart_steps.append({'step': step_index, 'sigma_curr': s_c, 'sigma_next': s_n})

    def get_summary(self):
        """Generate summary string."""
        if not self.step_times: return "No profiling data available"
        
        total_time = sum(self.step_times)
        avg_time = total_time / len(self.step_times)
        
        summary = "[PingPongSamplerFBG] Performance Summary:\n"
        summary += f"  Total time: {total_time:.3f}s\n"
        summary += f"  Average step time: {avg_time:.3f}s\n"
        
        if self.restart_count > 0:
            summary += f"  Restarts executed: {self.restart_count}\n"
            
        device = comfy.model_management.get_torch_device()
        if self.memory_usage and device.type != 'cpu':
             avg_mem = sum(self.memory_usage) / len(self.memory_usage)
             summary += f"  Avg memory delta: {avg_mem / 1024**2:.1f}MB"
             
        return summary

# =================================================================================
# == Core Sampler Logic
# =================================================================================

class PingPongSamplerCore:
    """Core implementation of the PingPong Sampler with FBG and Res 2 restarts."""

    def __init__(
        self,
        model,
        x,
        sigmas,
        extra_args=None,
        callback=None,
        disable=None,
        noise_sampler=None,
        # PingPong specific
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
        sigma_range_preset="Custom",
        conditional_blend_mode=False,
        conditional_blend_sigma_threshold=0.5,
        conditional_blend_function=slerp,
        conditional_blend_on_change=False,
        conditional_blend_change_threshold=0.1,
        clamp_noise_norm=False,
        max_noise_norm=1.0,
        log_posterior_ema_factor=0.0,
        # Enhanced
        adaptive_noise_scaling=False,
        noise_scale_factor=1.0,
        progressive_blend_mode=False,
        enable_profiling=False,
        checkpoint_steps=None,
        early_exit_threshold=CONST_EARLY_EXIT_THRESHOLD,
        tensor_memory_optimization=False,
        ancestral_noise_type="gaussian",
        # Res 2
        enable_restarts=False,
        restart_mode="balanced",
        restart_noise_scale=0.5,
        restart_s_noise=1.0,
        restart_steps="",
        **kwargs
    ):
        # Core attributes
        self.model_ = model
        self.x = x
        self.sigmas = sigmas
        self.extra_args = extra_args.copy() if extra_args is not None else {}
        self.callback_ = callback
        self.disable_pbar = disable
        
        # Debug & Logging
        self.debug_mode = debug_mode
        if self.debug_mode >= 2: logger.setLevel(logging.DEBUG)
        elif self.debug_mode >= 1: logger.setLevel(logging.INFO)
        else: logger.setLevel(logging.WARNING)

        # Core parameters
        self.start_sigma_index = start_sigma_index
        self.end_sigma_index = end_sigma_index
        self.enable_clamp_output = enable_clamp_output
        self.step_random_mode = step_random_mode
        self.step_size = step_size
        
        # Enterprise Standard: Validate Seed
        self.seed = validate_seed(seed) if seed is not None else 0
        
        self.blend_function = blend_function
        self.step_blend_function = step_blend_function

        # Advanced features
        self.adaptive_noise_scaling = adaptive_noise_scaling
        self.noise_scale_factor = noise_scale_factor
        self.progressive_blend_mode = progressive_blend_mode
        self.checkpoint_steps = checkpoint_steps or []
        self.early_exit_threshold = early_exit_threshold
        self.tensor_memory_optimization = tensor_memory_optimization

        self.profiler = PerformanceProfiler(enable_profiling)

        if self.tensor_memory_optimization:
            logger.warning("tensor_memory_optimization is EXPERIMENTAL!")

        # Steps & validation
        num_steps_available = len(sigmas) - 1 if len(sigmas) > 0 else 0
        logger.info(f"Total steps based on sigmas: {num_steps_available}")
        self._validate_sigma_schedule()

        if num_steps_available > 500:
            suggested = max(num_steps_available // 100, 5)
            logger.info(f"High step count ({num_steps_available}). Consider step_size >= {suggested}")

        # Ancestral Step Handling
        if pingpong_options is None: pingpong_options = {}
        raw_first = pingpong_options.get("first_ancestral_step", kwargs.get("first_ancestral_step", 0))
        raw_last = pingpong_options.get("last_ancestral_step", kwargs.get("last_ancestral_step", num_steps_available - 1))
        
        self.first_ancestral_step = max(0, min(raw_first, raw_last))
        if num_steps_available > 0:
            self.last_ancestral_step = min(num_steps_available - 1, max(raw_first, raw_last))
        else:
            self.last_ancestral_step = -1

        # Sigma range presets
        self.sigma_range_preset = sigma_range_preset
        self.original_fbg_config = fbg_config if fbg_config is not None else FBGConfig()
        self.config = self.original_fbg_config
        if self.sigma_range_preset != "Custom" and num_steps_available > 0:
            self.config = self._apply_sigma_preset(num_steps_available)

        # Conditional blend
        self.conditional_blend_mode = conditional_blend_mode
        self.conditional_blend_sigma_threshold = conditional_blend_sigma_threshold
        self.conditional_blend_function = conditional_blend_function
        self.conditional_blend_on_change = conditional_blend_on_change
        self.conditional_blend_change_threshold = conditional_blend_change_threshold

        # Noise clamping
        self.clamp_noise_norm = clamp_noise_norm
        self.max_noise_norm = max(0.01, max_noise_norm)

        # EMA
        self.log_posterior_ema_factor = max(0.0, min(1.0, log_posterior_ema_factor))

        # Model type
        self.is_rf = self._detect_model_type()

        # Noise sampler
        self.noise_sampler = self._setup_noise_sampler(noise_sampler, ancestral_noise_type)

        # Noise decay
        self.noise_decay = self._build_noise_decay_array(num_steps_available, scheduler)

        # FBG Params update
        self.update_fbg_config_params()

        # Initialize FBG state
        cfg = self.config
        self.minimal_log_posterior = self._calculate_minimal_log_posterior(cfg)
        self.log_posterior = x.new_full((x.shape[0],), cfg.initial_value)
        self.guidance_scale = x.new_full((x.shape[0], *(1,) * (x.ndim - 1)), cfg.initial_guidance_scale)

        # Res 2 Initialization
        self.enable_restarts = enable_restarts
        self.restart_mode = restart_mode
        self.restart_noise_scale = restart_noise_scale
        self.restart_s_noise = restart_s_noise
        
        self.custom_restart_schedule = None
        if enable_restarts and restart_steps and restart_steps.strip():
            try:
                self.custom_restart_schedule = [
                    int(x.strip()) for x in restart_steps.split(',') if x.strip()
                ]
                if self.debug_mode >= 1:
                    logger.info(f"Custom restart schedule: {self.custom_restart_schedule}")
            except ValueError as e:
                logger.warning(f"Invalid restart_steps format: {e}. Using auto-generated schedule")
                self.custom_restart_schedule = None

        if self.debug_mode >= 1 and enable_restarts:
            logger.info(f"Res 2 Restarts: ENABLED (Mode: {restart_mode}, Scale: {restart_noise_scale})")

        # Internal state
        self._prev_x = None
        self.checkpoints = {}

    def _validate_sigma_schedule(self):
        if len(self.sigmas) < 2: return True
        for i in range(len(self.sigmas) - 1):
            if self.sigmas[i] < self.sigmas[i + 1]:
                logger.warning(f"Sigma not monotonic at {i}: {self.sigmas[i]:.6f} -> {self.sigmas[i+1]:.6f}")
                return False
        return True

    def _apply_sigma_preset(self, num_steps):
        try:
             sorted_desc = torch.sort(self.sigmas, descending=True).values
             if len(sorted_desc) < 2: return self.config
             
             sigma_at_0 = sorted_desc[0].item()
             sigma_near_end = sorted_desc[-2].item()
             
             d = self.config._asdict()
             preset_name = self.sigma_range_preset
             
             if preset_name == "High":
                idx = max(1, num_steps // 4)
                idx = min(idx, len(sorted_desc) - 2)
                threshold = sorted_desc[idx].item()
                d.update({'cfg_start_sigma': sigma_at_0, 'cfg_end_sigma': threshold, 'fbg_start_sigma': sigma_at_0, 'fbg_end_sigma': threshold})
             elif preset_name == "Mid":
                idx_start = max(1, num_steps // 4)
                idx_end = max(idx_start + 1, 3 * num_steps // 4)
                idx_start = min(idx_start, len(sorted_desc) - 2)
                idx_end = min(idx_end, len(sorted_desc) - 2)
                if idx_start >= idx_end: idx_start = idx_end - 1
                start_sigma = sorted_desc[idx_start].item()
                end_sigma = sorted_desc[idx_end].item()
                d.update({'cfg_start_sigma': start_sigma, 'cfg_end_sigma': end_sigma, 'fbg_start_sigma': start_sigma, 'fbg_end_sigma': end_sigma})
             elif preset_name == "Low":
                idx = max(1, 3 * num_steps // 4)
                idx = min(idx, len(sorted_desc) - 2)
                threshold = sorted_desc[idx].item()
                d.update({'cfg_start_sigma': threshold, 'cfg_end_sigma': sigma_near_end, 'fbg_start_sigma': threshold, 'fbg_end_sigma': sigma_near_end})
             elif preset_name == "All":
                  d.update({'cfg_start_sigma': sigma_at_0, 'cfg_end_sigma': sigma_near_end, 'fbg_start_sigma': sigma_at_0, 'fbg_end_sigma': sigma_near_end})
             
             logger.info(f"Applied sigma preset '{self.sigma_range_preset}'")
             return FBGConfig(**d)
        except Exception as e:
             logger.error(f"Error applying sigma preset: {e}")
             return self.config

    def _detect_model_type(self):
        try:
            curr = self.model_
            while hasattr(curr, 'inner_model') and curr.inner_model is not None:
                curr = curr.inner_model
            if hasattr(curr, 'model_sampling') and curr.model_sampling is not None:
                return isinstance(curr.model_sampling, comfy.model_sampling.CONST)
        except: pass
        return False

    def _setup_noise_sampler(self, noise_sampler, noise_type):
        if noise_sampler is not None: return noise_sampler
        logger.info(f"Using ancestral noise type: {noise_type}")
        
        def create_noise_sampler(noise_func):
            def sampler(sigma, sigma_next):
                base_noise = noise_func()
                if self.adaptive_noise_scaling:
                    progress = 1.0 - (sigma / self.sigmas[0]) if self.sigmas[0] > 0 else 0.0
                    scale = self.noise_scale_factor * (1.0 + progress * 0.5)
                    base_noise = base_noise * scale
                return base_noise
            return sampler
        
        if noise_type == "uniform": return create_noise_sampler(lambda: torch.rand_like(self.x) * 2 - 1)
        elif noise_type == "brownian": return create_noise_sampler(lambda: torch.randn_like(self.x).cumsum(dim=-1) / (self.x.shape[-1]**0.5))
        return create_noise_sampler(lambda: torch.randn_like(self.x))

    def _build_noise_decay_array(self, num_steps, scheduler):
        if num_steps <= 0: return torch.empty((0,), dtype=torch.float32, device=self.x.device)
        if scheduler is not None and hasattr(scheduler, 'get_decay'):
            try:
                arr = scheduler.get_decay(num_steps)
                decay = np.asarray(arr, dtype=np.float32)
                return torch.tensor(decay, device=self.x.device)
            except Exception as e:
                logger.warning(f"Scheduler error: {e}. Using zero decay.")
        return torch.zeros((num_steps,), dtype=torch.float32, device=self.x.device)

    def _calculate_minimal_log_posterior(self, cfg):
        try:
            if cfg.cfg_scale > 1 and cfg.cfg_start_sigma > 0:
                num = (1.0 - cfg.pi) * (cfg.max_guidance_scale - cfg.cfg_scale + 1)
                den = (cfg.max_guidance_scale - cfg.cfg_scale)
            else:
                num = (1.0 - cfg.pi) * cfg.max_guidance_scale
                den = (cfg.max_guidance_scale - 1.0)
            if den <= 0 or num <= 0: return float('-inf')
            return math.log(num / den)
        except:
            logger.warning("Could not calculate minimal_log_posterior. Using -inf.")
            return float('-inf')

    def get_progressive_blend_function(self, step_idx, total_steps):
        if not self.progressive_blend_mode: return self.blend_function
        progress = step_idx / max(total_steps - 1, 1)
        if progress < 0.3: return torch.lerp
        elif progress < 0.7: return cosine_interpolation
        else: return cubic_interpolation

    def save_checkpoint(self, x_current, step_idx):
        if step_idx not in self.checkpoint_steps: return
        logger.info(f"Saving checkpoint at step {step_idx}")
        self.checkpoints[step_idx] = x_current.clone()

    def check_early_exit(self, sigma_next_item, denoised, x_current):
        if sigma_next_item > self.early_exit_threshold: return False
        if hasattr(self, '_prev_x') and self._prev_x is not None:
            change = torch.norm(x_current - self._prev_x).item()
            if change < self.early_exit_threshold * 0.1:
                logger.info(f"Early exit due to convergence (change: {change:.8f})")
                return True
        self._prev_x = x_current.clone()
        return False

    def _check_for_nan_inf(self, tensor, name, step_idx):
        if tensor is None: return False
        if torch.isnan(tensor).any():
            logger.warning(f"NaN detected in {name} at step {step_idx}!")
            return True
        if torch.isinf(tensor).any():
            logger.warning(f"Inf detected in {name} at step {step_idx}!")
            return True
        return False

    def _debug_step_info(self, idx, sigma_curr, sigma_next, x_current):
        if self.debug_mode >= 2:
            logger.debug(f"--- Step {idx} Detailed Info ---")
            logger.debug(f"  Sigma: {sigma_curr.item():.6f} -> {sigma_next.item():.6f}")
            self._check_for_nan_inf(x_current, "x_current", idx)

    def _stepped_seed(self, step):
        if self.step_random_mode == "off": return None
        sz = max(self.step_size, 1)
        # Ensure base seed is valid
        base_seed = validate_seed(self.seed)
        
        seed_map = {
            "block": base_seed + (step // sz), 
            "reset": base_seed + (step * sz), 
            "step": base_seed + step
        }
        
        raw_seed = seed_map.get(self.step_random_mode, base_seed)
        return validate_seed(raw_seed)

    def _get_sigma_square_tilde(self, sigmas):
        if len(sigmas) < 2: return torch.tensor([], device=sigmas.device)
        s_sq, sn_sq = sigmas[:-1] ** 2, sigmas[1:] ** 2
        safe_s_sq = torch.where(s_sq == 0, torch.tensor(1e-6, device=s_sq.device), s_sq)
        return ((s_sq - sn_sq) * sn_sq / safe_s_sq).flip(dims=(0,))

    def _get_offset(self, steps, sst, **kwargs):
        cfg = self.config
        lambda_ref = kwargs.get('lambda_ref', 3.0)
        decimals = kwargs.get('decimals', 4)
        t0 = max(0.0, min(1.0, cfg.t_0))
        if t0 >= 1.0 or (lambda_ref - 1.0) <= 0 or (1.0 - cfg.pi) <= 0: return 0.0
        try:
            log_term = math.log((1.0 - cfg.pi) * lambda_ref / (lambda_ref - 1.0))
            den = (1.0 - t0) * steps
            return round(log_term / den, decimals) if den != 0 else 0.0
        except: return 0.0

    def _get_temp(self, steps, offset, sst, **kwargs):
        cfg = self.config
        alpha = kwargs.get('alpha', 10.0)
        decimals = kwargs.get('decimals', 4)
        t1 = max(0.0, min(1.0, cfg.t_1))
        idx = int(math.floor(t1 * (steps - 1)))
        if len(sst) == 0 or alpha == 0: return 0.0
        idx = max(0, min(idx, len(sst) - 1))
        try:
            if len(sst) == 1: val = sst[0].item()
            elif idx == len(sst) - 1: val = sst[idx].item()
            else:
                v1, v2 = sst[idx].item(), sst[idx + 1].item()
                a = (t1 * (steps - 1)) - idx
                val = v1 * (1 - a) + v2 * a
            return round((2 * val / alpha * offset), decimals)
        except: return 0.0

    def update_fbg_config_params(self):
        if self.config.t_0 == 0 and self.config.t_1 == 0: return
        steps = len(self.sigmas) - 1
        if steps <= 0: return
        sst = self._get_sigma_square_tilde(self.sigmas)
        offset = self._get_offset(steps, sst)
        temp = self._get_temp(steps, offset, sst)
        d = self.config._asdict()
        d.update({"offset": offset, "temp": temp})
        self.config = FBGConfig(**d)

    def get_dynamic_guidance_scale(self, log_post, gs_prev, sigma_item):
        cfg = self.config
        using_fbg = cfg.fbg_end_sigma <= sigma_item <= cfg.fbg_start_sigma
        using_cfg = cfg.cfg_scale != 1 and (cfg.cfg_end_sigma <= sigma_item <= cfg.cfg_start_sigma)
        gs = log_post.new_ones(gs_prev.shape[0])
        if using_fbg:
            denom = log_post.exp() - (1.0 - cfg.pi)
            safe_denom = torch.where(denom.abs() < 1e-6, torch.full_like(denom, 1e-6), denom)
            fbg = log_post.exp() / safe_denom * cfg.fbg_guidance_multiplier
            gs = fbg.clamp(1.0, cfg.max_guidance_scale)
            if self.debug_mode >= 2:
                logger.debug(f"FBG active (σ {sigma_item:.3f}). Raw: {gs.mean().item():.4f}")
        if using_cfg:
            gs += cfg.cfg_scale - 1.0
        gs = gs.clamp(1.0, cfg.max_guidance_scale).view(gs_prev.shape)
        safe_prev = torch.where(gs_prev.abs() < 1e-6, torch.full_like(gs_prev, 1e-6), gs_prev)
        change = ((gs - gs_prev) / safe_prev).clamp(-cfg.guidance_max_change, cfg.guidance_max_change)
        return (gs_prev + gs_prev * change).clamp(1.0, cfg.max_guidance_scale)

    def _model_denoise_with_guidance(self, x, sigma, override_cfg=None):
        sigma_t = sigma * x.new_ones((x.shape[0],))
        cond = uncond = None
        def post_cfg(args):
            nonlocal cond, uncond
            cond, uncond = args.get("cond_denoised"), args.get("uncond_denoised")
            return args["denoised"]
        
        extra = self.extra_args.copy()
        mo = extra.get("model_options", {}).copy()
        mo["disable_cfg1_optimization"] = True
        extra["model_options"] = comfy.model_patcher.set_model_options_post_cfg_function(mo, post_cfg)
        
        inner = self.model_.inner_model
        try:
            # Try setting CFG directly on model if supported
            if (override_cfg is None or override_cfg.numel() < 2) and hasattr(inner, "cfg"):
                orig = inner.cfg
                try:
                    if override_cfg is not None:
                        inner.cfg = override_cfg.item() if override_cfg.numel() == 1 else override_cfg.mean().item()
                    denoised = inner.predict_noise(x, sigma_t, model_options=extra["model_options"], seed=extra.get("seed"))
                finally:
                    inner.cfg = orig
            else:
                # Standard CFG execution
                denoised = self.model_(x, sigma_t, **extra)
                
        except Exception as e:
            logger.warning(f"Denoise error: {e}. Fallback.")
            denoised = self.model_(x, sigma_t, **extra)
            
        if cond is None: cond = uncond = denoised
        return denoised, cond, uncond

    def _update_log_posterior(self, prev_lp, x_curr, x_next, t_curr, t_next, uncond, cond):
        """RESTORED v1.4.2: EXACT FBG math (NO epsilon corruption)."""
        def apply_ema(val):
            if self.log_posterior_ema_factor > 0:
                smoothed = self.log_posterior_ema_factor * prev_lp + (1 - self.log_posterior_ema_factor) * val
                return smoothed.clamp(self.minimal_log_posterior, self.config.max_posterior_scale)
            return val.clamp(self.minimal_log_posterior, self.config.max_posterior_scale)
        
        if torch.isclose(t_curr, torch.tensor(0.0, device=t_curr.device)).all(): return apply_ema(prev_lp)
        
        t_csq = t_curr**2
        if torch.isclose(t_csq, torch.tensor(0.0, device=t_csq.device)).all(): return apply_ema(prev_lp)
        
        t_ndc = t_next**2 / t_csq
        t_cmn = t_csq - t_next**2
        sst_t = t_cmn * t_ndc
        
        if torch.isclose(sst_t, torch.tensor(0.0, device=sst_t.device)).all(): return apply_ema(prev_lp)
        
        pred_base = t_ndc * x_curr
        uncond_mean = pred_base + (t_cmn / t_csq) * uncond
        cond_mean = pred_base + (t_cmn / t_csq) * cond
        diff = batch_mse_loss(x_next, cond_mean) - batch_mse_loss(x_next, uncond_mean)
        
        if torch.isclose(sst_t, torch.tensor(0.0, device=sst_t.device)):
            result = prev_lp + self.config.offset
        else:
            result = prev_lp - self.config.temp / (2 * sst_t) * diff + self.config.offset
        
        return apply_ema(result)

    def _generate_restart_schedule(self, sigmas):
        """Auto-generate restart schedule based on sigma values."""
        n = len(sigmas) - 1
        schedule = []
        mode = self.restart_mode
        
        if mode == 'aggressive':
            schedule = [i for i in range(1, n, 2)]
        elif mode == 'balanced':
            for i in range(n):
                if i == 0: continue
                sigma_ratio = sigmas[i] / sigmas[0]
                if sigma_ratio > 0.7: # High noise
                    if i % 2 == 0: schedule.append(i)
                elif sigma_ratio > 0.3: # Mid noise
                    if i % 3 == 0: schedule.append(i)
                else: # Low noise
                    if i % 5 == 0: schedule.append(i)
        elif mode == 'conservative':
            midpoint = n // 2
            schedule = [i for i in range(2, midpoint, 3)]
        elif mode == 'detail_focus':
            midpoint = n // 2
            for i in range(midpoint, n):
                if i % 2 == 0: schedule.append(i)
        elif mode == 'composition_focus':
            midpoint = n // 2
            for i in range(1, midpoint):
                if i % 2 == 0: schedule.append(i)
                
        return schedule

    def _generate_restart_noise(self, x, step_index):
        """Generate noise specifically for restart steps."""
        step_seed = self._stepped_seed(step_index)
        if step_seed is None: step_seed = self.seed
        
        restart_seed_offset = 982451653
        restart_seed = step_seed + restart_seed_offset
        restart_seed = validate_seed(restart_seed) # Ensure offset doesn't break JS cap
        
        generator = torch.Generator(device=x.device)
        generator.manual_seed(restart_seed)
        
        return torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=generator)

    def _execute_restart_step(self, model, x, sigma_curr, sigma_next, step_index, extra_args):
        """Execute a single Res 2 restart step."""
        sigma_restart = sigma_next + (sigma_curr - sigma_next) * self.restart_noise_scale
        
        if self.debug_mode >= 2:
            logger.debug(f"\n[Res2] RESTART at step {step_index}: σ_curr={sigma_curr:.4f} -> σ_next={sigma_next:.4f} -> σ_restart={sigma_restart:.4f}")

        # 1. Re-noise
        noise = self._generate_restart_noise(x, step_index)
        noise_amount = torch.sqrt(torch.clamp(sigma_restart**2 - sigma_next**2, min=0.0) + 1e-8)
        x_renoised = x + noise * noise_amount * self.restart_s_noise
        
        # 2. Denoise
        denoised_restart, _, _ = self._model_denoise_with_guidance(x_renoised, sigma_restart, self.guidance_scale)
        
        # 3. Euler step back
        d_restart = (x_renoised - denoised_restart) / sigma_restart
        dt_restart = sigma_next - sigma_restart
        x_corrected = x_renoised + d_restart * dt_restart
        
        return x_corrected, self.guidance_scale.mean().item()

    def _do_callback(self, step, x, sigma, denoised):
        """Execute callback if provided."""
        if self.callback_:
            try:
                self.callback_({
                    "i": step,
                    "x": x,
                    "sigma": sigma,
                    "sigma_hat": sigma,
                    "denoised": denoised
                })
            except Exception as e:
                logger.warning(f"Callback error at step {step}: {e}")

    def __call__(self):
        """Main sampling loop with ENHANCED console output + RES 2 restart logic."""
        try:
            gs_used = [] if self.debug_mode >= 1 else None
            if self.debug_mode >= 1: logger.info("Starting sampling loop")
            
            x = self.x.clone()
            num_steps = len(self.sigmas) - 1
            if num_steps <= 0: return x
            
            astart, aend = self.first_ancestral_step, self.last_ancestral_step
            actual_end = min(self.end_sigma_index if self.end_sigma_index >= 0 else num_steps - 1, num_steps - 1)
            
            # Generate restart schedule
            if self.enable_restarts:
                if self.custom_restart_schedule is not None:
                    restart_schedule = self.custom_restart_schedule
                else:
                    restart_schedule = self._generate_restart_schedule(self.sigmas)
                if self.debug_mode >= 1: logger.info(f"Restart schedule: {restart_schedule}")
            else:
                restart_schedule = []

            # Main Loop
            for idx in trange(num_steps, disable=self.disable_pbar):
                if idx < self.start_sigma_index or idx > actual_end: continue
                
                with self.profiler.profile_step(f"step_{idx}"):
                    s_curr, s_next = self.sigmas[idx], self.sigmas[idx + 1]
                    
                    # FBG Guidance
                    self.guidance_scale = self.get_dynamic_guidance_scale(self.log_posterior, self.guidance_scale, s_curr.max().item())
                    if gs_used is not None: gs_used.append(self.guidance_scale.mean().item())
                    
                    # Denoise
                    denoised, cond, uncond = self._model_denoise_with_guidance(x, s_curr, self.guidance_scale)
                    
                    # NaN Recovery
                    if self._check_for_nan_inf(denoised, "denoised", idx):
                        logger.warning(f"Recovering from NaN/Inf at step {idx}")
                        denoised = torch.where(torch.isnan(denoised) | torch.isinf(denoised), x, denoised)
                    
                    x_orig = x.clone()
                    self._do_callback(idx, x, s_curr, denoised)
                    self._debug_step_info(idx, s_curr, s_next, x)
                    
                    # Ancestral check
                    is_anc = (astart <= idx <= aend) if astart <= aend else False
                    
                    if not is_anc:
                        blend = s_next / s_curr if s_curr > 0 else 0.0
                        bf = self.get_progressive_blend_function(idx, num_steps)
                        if self.conditional_blend_mode and s_curr.max().item() < self.conditional_blend_sigma_threshold:
                            bf = self.conditional_blend_function
                        x = bf(denoised, x, blend)
                    else:
                        seed = self._stepped_seed(idx)
                        if seed is not None: torch.manual_seed(seed)
                        
                        noise = self.noise_sampler(s_curr, s_next)
                        
                        if self.clamp_noise_norm:
                            nn = torch.norm(noise, p=2, dim=list(range(1, noise.ndim)), keepdim=True)
                            sf = torch.where(nn > self.max_noise_norm, self.max_noise_norm / (nn + CONST_EPSILON), torch.ones_like(nn))
                            noise = noise * sf
                        
                        if self.is_rf:
                            bf = self.get_progressive_blend_function(idx, num_steps)
                            x = bf(denoised, noise, s_next)
                        else:
                            x = denoised + noise * s_next
                        
                        if self.conditional_blend_on_change:
                            dn = torch.norm(denoised, dim=list(range(1, denoised.ndim)), keepdim=True) + CONST_EPSILON
                            nn = torch.norm(noise * s_next, dim=list(range(1, noise.ndim)), keepdim=True)
                            rel = (nn / dn).mean().item()
                            if rel > self.conditional_blend_change_threshold:
                                bf = min(1.0, (rel - self.conditional_blend_change_threshold) / self.conditional_blend_change_threshold)
                                x = self.conditional_blend_function(denoised, x, bf)
                    
                    # RES 2 Restart
                    if self.enable_restarts and idx in restart_schedule and s_next > 0:
                        x, _ = self._execute_restart_step(self.model_, x, s_curr, s_next, idx, self.extra_args)
                        if self.profiler.enabled: self.profiler.log_restart(idx, s_curr, s_next)
                    
                    # FBG update
                    self.log_posterior = self._update_log_posterior(self.log_posterior, x_orig, x, s_curr, s_next, uncond, cond)
                    
                    self.save_checkpoint(x, idx)
                    if self.check_early_exit(s_next.min().item(), denoised, x): break
                    if self.enable_clamp_output and s_next.min().item() < 1e-3:
                        x = torch.clamp(x, -1.0, 1.0)
                        break
                    if s_next.min().item() <= 1e-6:
                        x = denoised
                        break
                    
                    if self.debug_mode >= 1:
                        logger.info(f"Step {idx}: GS={self.guidance_scale.mean().item():.2f}")
            
            if self.enable_clamp_output: x = torch.clamp(x, -1.0, 1.0)
            
            if self.debug_mode >= 1:
                logger.info("=" * 60)
                logger.info("PingPongSamplerFBG - Sampling Summary")
                logger.info("=" * 60)
                if gs_used:
                    logger.info(f"Guidance Scale: Min={min(gs_used):.3f}, Max={max(gs_used):.3f}, Avg={sum(gs_used)/len(gs_used):.3f}")
                if self.enable_restarts and restart_schedule:
                    logger.info(f"Restarts: {len(restart_schedule)}/{num_steps} executed (Mode: {self.restart_mode})")
                logger.info("=" * 60)
            
            if self.profiler.enabled:
                logger.info("\n" + self.profiler.get_summary())
            
            return x
        except Exception as e:
            logger.error(f"Crash in sampling loop: {e}")
            logger.debug(traceback.format_exc())
            return self.x

    @staticmethod
    def go(
        model,
        x,
        sigmas,
        extra_args=None,
        callback=None,
        disable=None,
        noise_sampler=None,
        **kwargs
    ):
        """Entry point for ComfyUI's KSAMPLER."""
        
        # Handle FBG config
        fbg_config_raw = kwargs.pop("fbg_config", {})
        fbg_config_kwargs = {}
        
        if isinstance(fbg_config_raw, FBGConfig):
            fbg_config_kwargs = fbg_config_raw._asdict()
        elif isinstance(fbg_config_raw, dict):
            remap = {
                "fbg_sampler_mode": "sampler_mode",
                "fbg_temp": "temp",
                "fbg_offset": "offset",
                "log_posterior_initial_value": "initial_value"
            }
            for k, v in fbg_config_raw.items():
                fbg_config_kwargs[remap.get(k, k)] = v
        
        # Handle sampler mode
        if "sampler_mode" in fbg_config_kwargs and isinstance(fbg_config_kwargs["sampler_mode"], str):
            try:
                fbg_config_kwargs["sampler_mode"] = getattr(SamplerMode, fbg_config_kwargs["sampler_mode"].upper())
            except:
                fbg_config_kwargs.pop("sampler_mode", None)
        elif "sampler_mode" not in fbg_config_kwargs:
            fbg_config_kwargs["sampler_mode"] = FBGConfig().sampler_mode
        
        fbg_config_instance = FBGConfig(**fbg_config_kwargs)
        
        # Handle blend functions
        pingpong_options = kwargs.pop("pingpong_options", {})
        blend_fn = _INTERNAL_BLEND_MODES.get(kwargs.pop("blend_function_name", "lerp"), torch.lerp)
        step_blend_fn = _INTERNAL_BLEND_MODES.get(kwargs.pop("step_blend_function_name", "lerp"), torch.lerp)
        cond_blend_fn = _INTERNAL_BLEND_MODES.get(kwargs.pop("conditional_blend_function_name", "slerp"), slerp)
        
        # Create sampler instance
        sampler = PingPongSamplerCore(
            model=model,
            x=x,
            sigmas=sigmas,
            extra_args=extra_args,
            callback=callback,
            disable=disable,
            noise_sampler=noise_sampler,
            blend_function=blend_fn,
            step_blend_function=step_blend_fn,
            conditional_blend_function=cond_blend_fn,
            fbg_config=fbg_config_instance,
            pingpong_options=pingpong_options,
            **kwargs
        )
        
        return sampler()

# =================================================================================
# == Node Wrapper Class
# =================================================================================

class PingPongSamplerNodeFBG:
    """ComfyUI Node wrapper with Res 2 restart support (v1.5.0)."""
    
    @classmethod
    def INPUT_TYPES(cls):
        d = FBGConfig()
        return {
            "required": {
                "step_random_mode": (["off", "block", "reset", "step"], {"default": "block", "tooltip": "STEP RANDOMIZATION: How the seed changes per step."}),
                "step_size": ("INT", {"default": 4, "min": 1, "max": 100, "tooltip": "STEP SIZE: Steps between seed changes."}),
                "seed": ("INT", {
                    "default": 80085, 
                    "min": CONST_SEED_MIN, 
                    "max": CONST_JS_MAX_SAFE_INTEGER, 
                    "tooltip": "SEED: Base random seed (JS-Safe)."
                }),
                "first_ancestral_step": ("INT", {"default": 0, "tooltip": "FIRST ANCESTRAL STEP: Start of noise injection."}),
                "last_ancestral_step": ("INT", {"default": -1, "tooltip": "LAST ANCESTRAL STEP: End of noise injection."}),
                "ancestral_noise_type": (["gaussian", "uniform", "brownian"], {"default": "gaussian", "tooltip": "ANCESTRAL NOISE TYPE: Distribution of injected noise."}),
                "start_sigma_index": ("INT", {"default": 0, "tooltip": "START STEP: Sampling start index."}),
                "end_sigma_index": ("INT", {"default": -1, "tooltip": "END STEP: Sampling end index."}),
                "enable_clamp_output": ("BOOLEAN", {"default": False, "tooltip": "CLAMP OUTPUT: Limit output to [-1, 1]."}),
                "scheduler": ("SCHEDULER", {"tooltip": "SCHEDULER: Input sigma schedule."}),
                "blend_mode": (list(_INTERNAL_BLEND_MODES.keys()), {"default": "lerp", "tooltip": "BLEND MODE: Primary interpolation function."}),
                "step_blend_mode": (list(_INTERNAL_BLEND_MODES.keys()), {"default": "lerp", "tooltip": "STEP BLEND MODE: Secondary interpolation function."}),
                
                "fbg_sampler_mode": (["EULER", "PINGPONG"], {"default": "EULER", "tooltip": "FBG MODE: Internal step type."}),
                "cfg_scale": ("FLOAT", {"default": d.cfg_scale, "min": -1000, "max": 1000, "step": 0.01, "tooltip": "CFG SCALE: Base guidance."}),
                "cfg_start_sigma": ("FLOAT", {"default": d.cfg_start_sigma, "tooltip": "CFG START SIGMA: Start of CFG range."}),
                "cfg_end_sigma": ("FLOAT", {"default": d.cfg_end_sigma, "tooltip": "CFG END SIGMA: End of CFG range."}),
                "fbg_start_sigma": ("FLOAT", {"default": d.fbg_start_sigma, "tooltip": "FBG START SIGMA: Start of FBG range."}),
                "fbg_end_sigma": ("FLOAT", {"default": d.fbg_end_sigma, "tooltip": "FBG END SIGMA: End of FBG range."}),
                "max_guidance_scale": ("FLOAT", {"default": d.max_guidance_scale, "tooltip": "MAX GUIDANCE: Ceiling for combined scale."}),
                "initial_guidance_scale": ("FLOAT", {"default": d.initial_guidance_scale, "tooltip": "INITIAL GUIDANCE: Starting scale."}),
                "fbg_guidance_multiplier": ("FLOAT", {"default": d.fbg_guidance_multiplier, "tooltip": "FBG MULTIPLIER: Scaling factor for FBG component."}),
                "guidance_max_change": ("FLOAT", {"default": d.guidance_max_change, "tooltip": "MAX CHANGE: Max % change per step."}),
                "pi": ("FLOAT", {"default": d.pi, "step": 0.01, "tooltip": "PI: Posterior factor."}),
                "t_0": ("FLOAT", {"default": d.t_0, "step": 0.01, "tooltip": "T0: Offset parameter."}),
                "t_1": ("FLOAT", {"default": d.t_1, "step": 0.01, "tooltip": "T1: Temp parameter."}),
                "fbg_temp": ("FLOAT", {"default": d.temp, "tooltip": "TEMP: FBG temperature."}),
                "fbg_offset": ("FLOAT", {"default": d.offset, "tooltip": "OFFSET: FBG offset."}),
                "fbg_eta": ("FLOAT", {"default": 0.0, "tooltip": "ETA: Deprecated."}),
                "fbg_s_noise": ("FLOAT", {"default": 1.0, "tooltip": "S_NOISE: Deprecated."}),
                "max_posterior_scale": ("FLOAT", {"default": d.max_posterior_scale, "tooltip": "MAX POSTERIOR: Ceiling for log posterior."}),
                "log_posterior_initial_value": ("FLOAT", {"default": d.initial_value, "tooltip": "INITIAL POSTERIOR: Starting value."}),
                
                "log_posterior_ema_factor": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "EMA FACTOR: Smoothing for posterior."}),
                "adaptive_noise_scaling": ("BOOLEAN", {"default": False, "tooltip": "ADAPTIVE NOISE: Scale noise by progress."}),
                "noise_scale_factor": ("FLOAT", {"default": 1.0, "tooltip": "NOISE SCALE: Base factor."}),
                "progressive_blend_mode": ("BOOLEAN", {"default": False, "tooltip": "PROGRESSIVE BLEND: Switch modes over time."}),
                
                "conditional_blend_mode": ("BOOLEAN", {"default": False, "tooltip": "CONDITIONAL BLEND: Enable threshold blending."}),
                "conditional_blend_sigma_threshold": ("FLOAT", {"default": 0.5, "tooltip": "BLEND THRESHOLD: Sigma trigger."}),
                "conditional_blend_function_name": (list(_INTERNAL_BLEND_MODES.keys()), {"default": "slerp", "tooltip": "BLEND FUNC: Conditional function."}),
                "conditional_blend_on_change": ("BOOLEAN", {"default": False, "tooltip": "BLEND ON CHANGE: Trigger on delta."}),
                "conditional_blend_change_threshold": ("FLOAT", {"default": 0.1, "tooltip": "CHANGE THRESHOLD: Delta trigger."}),
                
                "clamp_noise_norm": ("BOOLEAN", {"default": False, "tooltip": "CLAMP NOISE: Limit noise magnitude."}),
                "max_noise_norm": ("FLOAT", {"default": 1.0, "tooltip": "MAX NOISE: Limit value."}),
                "gradient_norm_tracking": ("BOOLEAN", {"default": False, "tooltip": "GRADIENT TRACKING: Disabled."}),
                
                "enable_profiling": ("BOOLEAN", {"default": False, "tooltip": "PROFILING: Log timing stats."}),
                "debug_mode": ("INT", {"default": 0, "min": 0, "max": 2, "tooltip": "DEBUG MODE: 0=Off, 1=Basic, 2=Detailed."}),
                "tensor_memory_optimization": ("BOOLEAN", {"default": False, "tooltip": "MEMORY OPT: Experimental."}),
                "early_exit_threshold": ("FLOAT", {"default": CONST_EARLY_EXIT_THRESHOLD, "tooltip": "EARLY EXIT: Convergence threshold."}),
                "ancestral_start_sigma": ("FLOAT", {"default": 1.0, "tooltip": "ANC START: Unused range."}),
                "ancestral_end_sigma": ("FLOAT", {"default": 0.004, "tooltip": "ANC END: Unused range."}),
                "sigma_range_preset": (["Custom", "High", "Mid", "Low", "All"], {"default": "Custom", "tooltip": "PRESET: Auto-set FBG/CFG ranges."}),
                
                # Res 2
                "enable_restarts": ("BOOLEAN", {"default": False, "tooltip": "ENABLE RESTARTS: Turn on Res 2 logic."}),
            },
            "optional": {
                "yaml_settings_str": ("STRING", {"multiline": True, "default": "", "dynamicPrompt": False, "tooltip": "YAML OVERRIDE: Custom config."}),
                "checkpoint_steps_str": ("STRING", {"default": "", "tooltip": "CHECKPOINTS: Steps to save state."}),
                
                # Res 2 Optionals
                "restart_mode": (["balanced", "aggressive", "conservative", "detail_focus", "composition_focus"], {"default": "balanced", "tooltip": "RESTART MODE: Frequency pattern."}),
                "restart_noise_scale": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "RESTART SCALE: Noise amount for restart."}),
                "restart_s_noise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05, "tooltip": "RESTART S_NOISE: Multiplier."}),
                "restart_steps": ("STRING", {"default": "", "multiline": False, "tooltip": "CUSTOM RESTARTS: Specific steps (e.g. '1,5,10')."}),
            }
        }

    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "get_sampler"
    CATEGORY = "MD_Nodes/Samplers"
    
    def get_sampler(self, **kwargs):
        """Creates the KSAMPLER instance."""
        
        # Parse YAML overrides
        yaml_str = kwargs.pop("yaml_settings_str", "")
        if yaml_str:
            try:
                overrides = yaml.safe_load(yaml_str)
                if isinstance(overrides, dict):
                    for k, v in overrides.items():
                        if k == "fbg_config" and isinstance(v, dict):
                            for fk, fv in v.items():
                                kwargs[fk] = fv
                        else:
                            kwargs[k] = v
            except Exception as e:
                logger.warning(f"YAML parse error: {e}")
        
        # Parse checkpoint steps
        cp_str = kwargs.pop("checkpoint_steps_str", "")
        if cp_str:
            try:
                kwargs["checkpoint_steps"] = [int(x.strip()) for x in cp_str.split(",") if x.strip()]
            except:
                pass
        
        # Parse sampler mode
        mode_str = kwargs.pop("fbg_sampler_mode", "EULER")
        try:
            mode = getattr(SamplerMode, mode_str.upper())
        except:
            mode = SamplerMode.EULER
        
        # Build FBG config
        d = FBGConfig()
        fbg_cfg = FBGConfig(
            sampler_mode=mode,
            cfg_start_sigma=kwargs.pop("cfg_start_sigma", d.cfg_start_sigma),
            cfg_end_sigma=kwargs.pop("cfg_end_sigma", d.cfg_end_sigma),
            fbg_start_sigma=kwargs.pop("fbg_start_sigma", d.fbg_start_sigma),
            fbg_end_sigma=kwargs.pop("fbg_end_sigma", d.fbg_end_sigma),
            fbg_guidance_multiplier=kwargs.pop("fbg_guidance_multiplier", d.fbg_guidance_multiplier),
            ancestral_start_sigma=kwargs.pop("ancestral_start_sigma", d.ancestral_start_sigma),
            ancestral_end_sigma=kwargs.pop("ancestral_end_sigma", d.ancestral_end_sigma),
            cfg_scale=kwargs.pop("cfg_scale", d.cfg_scale),
            max_guidance_scale=kwargs.pop("max_guidance_scale", d.max_guidance_scale),
            max_posterior_scale=kwargs.pop("max_posterior_scale", d.max_posterior_scale),
            initial_value=kwargs.pop("log_posterior_initial_value", d.initial_value),
            initial_guidance_scale=kwargs.pop("initial_guidance_scale", d.initial_guidance_scale),
            guidance_max_change=kwargs.pop("guidance_max_change", d.guidance_max_change),
            temp=kwargs.pop("fbg_temp", d.temp),
            offset=kwargs.pop("fbg_offset", d.offset),
            pi=kwargs.pop("pi", d.pi),
            t_0=kwargs.pop("t_0", d.t_0),
            t_1=kwargs.pop("t_1", d.t_1)
        )
        
        # Remove deprecated params
        kwargs.pop("fbg_eta", None)
        kwargs.pop("fbg_s_noise", None)
        
        # Rename blend modes
        kwargs['blend_function_name'] = kwargs.pop("blend_mode", "lerp")
        kwargs['step_blend_function_name'] = kwargs.pop("step_blend_mode", "lerp")
        # conditional_blend_function_name stays as-is
        
        kwargs['fbg_config'] = fbg_cfg
        
        return (KSAMPLER(PingPongSamplerCore.go, extra_options=kwargs),)

NODE_CLASS_MAPPINGS = {"PingPongSamplerNodeFBG": PingPongSamplerNodeFBG}
NODE_DISPLAY_NAME_MAPPINGS = {"PingPongSamplerNodeFBG": "MD: PingPong Sampler (FBG v1.5.0)"}

# =================================================================================
# == Development & Testing
# =================================================================================

if __name__ == "__main__":
    """
    Embedded unit tests for standalone validation.
    Run with: python PingPongSampler_Custom_FBG.py
    """
    print("🧪 Running Self-Tests for PingPongSamplerFBG...")
    
    test_passed = 0
    test_failed = 0
    
    try:
        # Test 1: Constants validation
        assert CONST_JS_MAX_SAFE_INTEGER == 9007199254740991, "JS safe integer mismatch"
        assert CONST_SEED_MIN == 0, "Seed minimum mismatch"
        print("✅ Constants Check: PASSED")
        test_passed += 1
    except AssertionError as e:
        print(f"❌ Constants Check: FAILED - {e}")
        test_failed += 1
    
    try:
        # Test 2: Seed validation within range
        test_seed = 123456
        validated = validate_seed(test_seed)
        assert validated == test_seed, f"Seed validation failed: {validated} != {test_seed}"
        print("✅ Seed Validation (In Range): PASSED")
        test_passed += 1
    except AssertionError as e:
        print(f"❌ Seed Validation (In Range): FAILED - {e}")
        test_failed += 1
    
    try:
        # Test 3: Seed clamping above JS-safe max
        large_seed = CONST_JS_MAX_SAFE_INTEGER + 1000
        validated = validate_seed(large_seed)
        assert validated == CONST_JS_MAX_SAFE_INTEGER, "Large seed not clamped correctly"
        print("✅ Seed Clamping (Above Max): PASSED")
        test_passed += 1
    except AssertionError as e:
        print(f"❌ Seed Clamping (Above Max): FAILED - {e}")
        test_failed += 1

    # Test Summary
    print(f"\n{'='*60}")
    print(f"Test Results: {test_passed} passed, {test_failed} failed")
    print(f"{'='*60}")
    
    if test_failed == 0:
        print("🎉 All tests passed!")
    else:
        print("⚠️ Some tests failed. Review output above.")