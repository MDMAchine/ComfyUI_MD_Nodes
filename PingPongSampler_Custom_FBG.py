# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
# ████ MD_Nodes/PingPongSamplerNodeFBG – Advanced sampler with Feedback Guidance ████▓▒░
# © 2025 MDMAchine
# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
# ░▒▓ CHANGELOG:
#   - v1.4.5g (Math Restoration - Nov 2025):
#       • CRITICAL: Restored v1.4.2's EXACT FBG math (no epsilon corruption in log posterior)
#       • CRITICAL: Restored v1.4.2's adaptive noise scaling (inside noise sampler)
#       • CRITICAL: Restored v1.4.2's _get_sigma_square_tilde with safety checks
#       • ENHANCED: Added comprehensive debug summary box with GS min/max/avg stats
#       • ENHANCED: Added detailed PerformanceProfiler with max time & memory breakdown
#       • ENHANCED: Added NaN/Inf detection with recovery
#       • ENHANCED: Added detailed step info for debug_mode=2
#       • MAINTAINED: Clean parameter handling from v1.4.5e (no kwargs.get bugs)
#   - v1.4.5e: Fixed critical kwargs.get() bug
#   - v1.4.2: Working audio quality baseline

# =================================================================================
# == Standard Library Imports                                                     ==
# =================================================================================
import contextlib
import enum
import logging
import math
import time
import traceback
import sys

# =================================================================================
# == Third-Party Imports                                                          ==
# =================================================================================
import numpy as np
import torch
import yaml
from tqdm.auto import trange

# =================================================================================
# == ComfyUI Core Modules                                                         ==
# =================================================================================
import comfy.k_diffusion.sampling
import comfy.model_management
import comfy.model_patcher
import comfy.model_sampling
import comfy.samplers

# =================================================================================
# == Logging Setup                                                                ==
# =================================================================================
logger = logging.getLogger("PingPongSamplerFBG")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('[%(name)s] %(message)s'))
    logger.addHandler(handler)

# =================================================================================
# == Helper Functions & Classes                                                   ==
# =================================================================================

def slerp(a, b, t):
    """
    Spherical linear interpolation with CORRECT magnitude interpolation.
    Fixes 'static' audio issues by properly scaling the result based on 't'.
    """
    eps = 1e-8
    
    # 1. Flatten to treat the whole latent as a single vector (safer for audio tensors)
    a_flat = a.flatten(start_dim=1)
    b_flat = b.flatten(start_dim=1)
    
    # 2. Calculate magnitudes (norms) per batch item
    a_norm = torch.norm(a_flat, dim=-1, keepdim=True)
    b_norm = torch.norm(b_flat, dim=-1, keepdim=True)
    
    # 3. Normalize vectors
    a_n = a_flat / (a_norm + eps)
    b_n = b_flat / (b_norm + eps)
    
    # 4. Calculate angle (theta)
    dot = (a_n * b_n).sum(dim=-1, keepdim=True).clamp(-0.9999, 0.9999)
    
    # Fallback to lerp if vectors are parallel (dot close to 1)
    if torch.all(torch.abs(dot) > 0.9995):
        return torch.lerp(a, b, t)
        
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    
    # 5. Spherical interpolation of direction
    scale_a = torch.sin((1 - t) * theta) / (sin_theta + eps)
    scale_b = torch.sin(t * theta) / (sin_theta + eps)
    result_n = scale_a * a_n + scale_b * b_n
    
    # 6. CRITICAL FIX: Interpolate magnitude based on 't' (was previously just averaging)
    result_norm = torch.lerp(a_norm, b_norm, t)
    
    # 7. Apply magnitude and reshape back to original
    result_flat = result_n * result_norm
    return result_flat.reshape_as(a)

def cosine_interpolation(a, b, t):
    cos_t = (1 - torch.cos(t * math.pi)) * 0.5
    return a * (1 - cos_t) + b * cos_t

def cubic_interpolation(a, b, t):
    cubic_t = t * t * (3.0 - 2.0 * t)
    return torch.lerp(a, b, cubic_t)

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
    """Configuration container for Feedback Guidance."""
    def __init__(self, sampler_mode=SamplerMode.EULER, cfg_start_sigma=1.0, cfg_end_sigma=0.004,
                 fbg_start_sigma=1.0, fbg_end_sigma=0.004, fbg_guidance_multiplier=1.0,
                 ancestral_start_sigma=1.0, ancestral_end_sigma=0.004, cfg_scale=1.0,
                 max_guidance_scale=10.0, max_posterior_scale=3.0, initial_value=0.0,
                 initial_guidance_scale=1.0, guidance_max_change=1000.0, temp=0.0,
                 offset=0.0, pi=0.5, t_0=0.5, t_1=0.4):
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
        return self.__dict__

def batch_mse_loss(a, b, *, start_dim=1):
    if a.numel() > 1e7:
        diff = a - b
        return (diff * diff).sum(dim=tuple(range(start_dim, a.ndim)))
    return torch.sum((a - b).pow(2), dim=tuple(range(start_dim, a.ndim)))

class PerformanceProfiler:
    """ENHANCED: Detailed profiler with max/min/avg time & memory stats."""
    def __init__(self, enabled=False):
        self.enabled = enabled
        self.step_times = []
        self.memory_usage = []
        self.step_names = []
        
    @contextlib.contextmanager
    def profile_step(self, step_name="step"):
        if not self.enabled:
            yield
            return
        device = comfy.model_management.get_torch_device()
        start_time = time.time()
        if device != torch.device("cpu"):
            torch.cuda.synchronize(device)
            start_mem = torch.cuda.memory_allocated(device)
        else: start_mem = 0
        yield
        if device != torch.device("cpu"):
            torch.cuda.synchronize(device)
            end_mem = torch.cuda.memory_allocated(device)
        else: end_mem = 0
        self.step_times.append(time.time() - start_time)
        self.memory_usage.append(end_mem - start_mem)
        self.step_names.append(step_name)
        
    def get_summary(self):
        """ENHANCED: Returns detailed summary with max/min/avg stats."""
        if not self.step_times:
            return "No profiling data available"
        
        total_time = sum(self.step_times)
        avg_time = total_time / len(self.step_times)
        max_time = max(self.step_times)
        min_time = min(self.step_times)
        
        summary = "[PingPongSamplerFBG] Performance Summary:\n"
        summary += f"  Total time: {total_time:.3f}s\n"
        summary += f"  Average step time: {avg_time:.3f}s\n"
        summary += f"  Max step time: {max_time:.3f}s\n"
        summary += f"  Min step time: {min_time:.3f}s\n"
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

# =================================================================================
# == Core Logic: PingPongSamplerCore                                             ==
# =================================================================================

class PingPongSamplerCore:
    """PingPongSampler with v1.4.2 FBG math + enhanced console output."""
    
    def __init__(self, model, x, sigmas, extra_args=None, callback=None, disable=None, 
                 noise_sampler=None, start_sigma_index=0, end_sigma_index=-1, 
                 enable_clamp_output=False, step_random_mode="off", step_size=5, seed=42,
                 blend_function=torch.lerp, step_blend_function=torch.lerp, scheduler=None,
                 pingpong_options=None, fbg_config=None, debug_mode=0, eta=0.0, s_noise=1.0,
                 sigma_range_preset="Custom", conditional_blend_mode=False,
                 conditional_blend_sigma_threshold=0.5, conditional_blend_function=torch.lerp,
                 conditional_blend_on_change=False, conditional_blend_change_threshold=0.1,
                 clamp_noise_norm=False, max_noise_norm=1.0, log_posterior_ema_factor=0.0,
                 adaptive_noise_scaling=False, noise_scale_factor=1.0,
                 progressive_blend_mode=False, gradient_norm_tracking=False,
                 enable_profiling=False, checkpoint_steps=None, early_exit_threshold=1e-6,
                 tensor_memory_optimization=False, **kwargs):
        
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
        self.seed = seed if seed is not None else 42
        self.blend_function = blend_function
        self.step_blend_function = step_blend_function
        
        # Advanced features
        self.adaptive_noise_scaling = adaptive_noise_scaling
        self.noise_scale_factor = noise_scale_factor
        self.progressive_blend_mode = progressive_blend_mode
        self.gradient_norm_tracking = gradient_norm_tracking
        self.checkpoint_steps = checkpoint_steps or []
        self.early_exit_threshold = early_exit_threshold
        self.tensor_memory_optimization = tensor_memory_optimization
        
        if self.tensor_memory_optimization:
            print("[PingPongSamplerFBG] ⚠️ WARNING: tensor_memory_optimization is EXPERIMENTAL!")
        
        # ENHANCED: Profiler
        self.profiler = PerformanceProfiler(enable_profiling)
        
        if gradient_norm_tracking:
            logger.info("Note: Gradient norm tracking disabled (not applicable in sampling)")
        
        # Steps & validation
        num_steps_available = len(sigmas) - 1 if len(sigmas) > 0 else 0
        logger.info(f"Total steps based on sigmas: {num_steps_available}")
        self._validate_sigma_schedule()
        
        if num_steps_available > 500:
            suggested = max(num_steps_available // 100, 5)
            logger.info(f"High step count ({num_steps_available}). Consider step_size >= {suggested}")
        
        # --- FIXED: Robust Ancestral Step Handling (The Bridge for your YAML) ---
        if pingpong_options is None: pingpong_options = {}
        
        # Logic: Check dictionary first, then fall back to flat kwargs (which YAML uses)
        raw_first = pingpong_options.get("first_ancestral_step", kwargs.get("first_ancestral_step", 0))
        raw_last = pingpong_options.get("last_ancestral_step", kwargs.get("last_ancestral_step", num_steps_available - 1))
        
        self.first_ancestral_step = max(0, min(raw_first, raw_last))
        if num_steps_available > 0:
            self.last_ancestral_step = min(num_steps_available - 1, max(raw_first, raw_last))
        else:
            self.last_ancestral_step = -1
        # -----------------------------------------------------------------------
        
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
        self.max_noise_norm = max_noise_norm
        
        # EMA
        self.log_posterior_ema_factor = max(0.0, min(1.0, log_posterior_ema_factor))
        
        # Model type
        self.is_rf = self._detect_model_type()
        
        # RESTORED v1.4.2: Noise sampler with adaptive scaling inside
        self.noise_sampler = self._setup_noise_sampler(noise_sampler, kwargs.get("ancestral_noise_type", "gaussian"))
        
        # Noise decay
        self.noise_decay = self._build_noise_decay_array(num_steps_available, scheduler)
        
        # FBG
        self.eta = eta
        self.s_noise = s_noise
        self.update_fbg_config_params()
        
        # Initialize FBG state
        cfg = self.config
        self.minimal_log_posterior = self._calculate_minimal_log_posterior(cfg)
        self.log_posterior = x.new_full((x.shape[0],), cfg.initial_value)
        self.guidance_scale = x.new_full((x.shape[0], *(1,) * (x.ndim - 1)), cfg.initial_guidance_scale)
        
        if self.debug_mode >= 1:
            logger.info(f"FBG config: {self.config._asdict()}")
            logger.info(f"Enhanced features: adaptive_noise={self.adaptive_noise_scaling}, progressive_blend={self.progressive_blend_mode}")

    @staticmethod
    def go(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, **kwargs):
        """Entry point for ComfyUI's KSAMPLER."""
        fbg_config_raw = kwargs.pop("fbg_config", {})
        fbg_config_kwargs = {}
        
        # Handle both dict and FBGConfig object
        if isinstance(fbg_config_raw, FBGConfig):
            fbg_config_kwargs = fbg_config_raw._asdict()
        elif isinstance(fbg_config_raw, dict):
            remap = {"fbg_sampler_mode": "sampler_mode", "fbg_temp": "temp", 
                     "fbg_offset": "offset", "log_posterior_initial_value": "initial_value"}
            for k, v in fbg_config_raw.items():
                fbg_config_kwargs[remap.get(k, k)] = v
        
        if "sampler_mode" in fbg_config_kwargs and isinstance(fbg_config_kwargs["sampler_mode"], str):
            try: 
                fbg_config_kwargs["sampler_mode"] = getattr(SamplerMode, fbg_config_kwargs["sampler_mode"].upper())
            except: 
                fbg_config_kwargs.pop("sampler_mode", None)
        elif "sampler_mode" not in fbg_config_kwargs:
            fbg_config_kwargs["sampler_mode"] = FBGConfig().sampler_mode
        
        fbg_config_instance = FBGConfig(**fbg_config_kwargs)
        
        pingpong_options = kwargs.pop("pingpong_options", {})
        
        # FIXED: Pop these before passing to __init__ to avoid duplicate keyword args
        blend_fn = _INTERNAL_BLEND_MODES.get(kwargs.pop("blend_function_name", "lerp"), torch.lerp)
        step_blend_fn = _INTERNAL_BLEND_MODES.get(kwargs.pop("step_blend_function_name", "lerp"), torch.lerp)
        cond_blend_fn = _INTERNAL_BLEND_MODES.get(kwargs.pop("conditional_blend_function_name", "slerp"), slerp)
        
        sampler = PingPongSamplerCore(
            model=model, x=x, sigmas=sigmas, extra_args=extra_args, callback=callback,
            disable=disable, noise_sampler=noise_sampler, blend_function=blend_fn,
            step_blend_function=step_blend_fn, conditional_blend_function=cond_blend_fn,
            fbg_config=fbg_config_instance, pingpong_options=pingpong_options, **kwargs
        )
        return sampler()

    def _validate_sigma_schedule(self):
        if len(self.sigmas) < 2: return True
        for i in range(len(self.sigmas) - 1):
            if self.sigmas[i] < self.sigmas[i + 1]:
                logger.warning(f"Sigma not monotonic at {i}: {self.sigmas[i]:.6f} -> {self.sigmas[i+1]:.6f}")
                return False
        return True

    def _apply_sigma_preset(self, num_steps):
        sorted_desc = torch.sort(self.sigmas, descending=True).values
        d = self.config._asdict()
        if self.sigma_range_preset == "High":
            idx = max(1, num_steps // 4)
            high = sorted_desc[idx].item()
            d.update({'cfg_start_sigma': high, 'cfg_end_sigma': self.sigmas[-2].item(),
                      'fbg_start_sigma': high, 'fbg_end_sigma': self.sigmas[-2].item()})
        elif self.sigma_range_preset == "Mid":
            start, end = num_steps // 4, 3 * num_steps // 4
            d.update({'cfg_start_sigma': sorted_desc[start].item(), 'cfg_end_sigma': sorted_desc[end].item(),
                      'fbg_start_sigma': sorted_desc[start].item(), 'fbg_end_sigma': sorted_desc[end].item()})
        elif self.sigma_range_preset == "Low":
            idx = 3 * num_steps // 4
            low = sorted_desc[idx].item()
            d.update({'cfg_start_sigma': self.sigmas[0].item(), 'cfg_end_sigma': low,
                      'fbg_start_sigma': self.sigmas[0].item(), 'fbg_end_sigma': low})
        elif self.sigma_range_preset == "All":
            d.update({'cfg_start_sigma': self.sigmas[0].item(), 'cfg_end_sigma': self.sigmas[-2].item(),
                      'fbg_start_sigma': self.sigmas[0].item(), 'fbg_end_sigma': self.sigmas[-2].item()})
        logger.info(f"Applied sigma preset '{self.sigma_range_preset}'")
        return FBGConfig(**d)

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
        """RESTORED v1.4.2: Noise sampler with adaptive scaling INSIDE."""
        if noise_sampler is not None: return noise_sampler
        logger.info(f"Using ancestral noise type: {noise_type}")
        
        def create_noise_sampler(noise_func):
            def sampler(sigma, sigma_next):
                base_noise = noise_func()
                # RESTORED v1.4.2: Adaptive scaling inside noise sampler
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
                assert decay.shape == (num_steps,)
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
        if not hasattr(self, 'checkpoints'): self.checkpoints = {}
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
        """ENHANCED: Check for NaN/Inf."""
        if torch.isnan(tensor).any():
            logger.warning(f"NaN detected in {name} at step {step_idx}!")
            return True
        if torch.isinf(tensor).any():
            logger.warning(f"Inf detected in {name} at step {step_idx}!")
            return True
        return False

    def _debug_step_info(self, idx, sigma_curr, sigma_next, x_current):
        """ENHANCED: Detailed debug for troubleshooting."""
        if self.debug_mode >= 2:
            logger.debug(f"\n=== Step {idx} Detailed Info ===")
            logger.debug(f"  Sigma: {sigma_curr.item():.6f} -> {sigma_next.item():.6f}")
            logger.debug(f"  X stats: min={x_current.min().item():.4f}, max={x_current.max().item():.4f}, "
                        f"mean={x_current.mean().item():.4f}, std={x_current.std().item():.4f}")
            self._check_for_nan_inf(x_current, "x_current", idx)

    def _stepped_seed(self, step):
        if self.step_random_mode == "off": return None
        sz = max(self.step_size, 1)
        return {"block": self.seed + (step // sz), "reset": self.seed + (step * sz), "step": self.seed + step}.get(self.step_random_mode, self.seed)

    def _get_sigma_square_tilde(self, sigmas):
        """RESTORED v1.4.2: With safety check for division by zero."""
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
            if self.debug_mode >= 2:
                logger.debug(f"CFG active. Added: {cfg.cfg_scale - 1.0:.2f}")
        gs = gs.clamp(1.0, cfg.max_guidance_scale).view(gs_prev.shape)
        safe_prev = torch.where(gs_prev.abs() < 1e-6, torch.full_like(gs_prev, 1e-6), gs_prev)
        change = ((gs - gs_prev) / safe_prev).clamp(-cfg.guidance_max_change, cfg.guidance_max_change)
        return (gs_prev + gs_prev * change).clamp(1.0, cfg.max_guidance_scale)

    def _model_denoise_with_guidance(self, x, sigma, override_cfg=None):
        sigma_t = sigma * x.new_ones((x.shape[0],))
        cond = uncond = None
        def post_cfg(args):
            nonlocal cond, uncond
            cond, uncond = args["cond_denoised"], args["uncond_denoised"]
            return args["denoised"]
        extra = self.extra_args.copy()
        mo = extra.get("model_options", {}).copy()
        mo["disable_cfg1_optimization"] = True
        extra["model_options"] = comfy.model_patcher.set_model_options_post_cfg_function(mo, post_cfg)
        inner = self.model_.inner_model
        try:
            if (override_cfg is None or override_cfg.numel() < 2) and hasattr(inner, "cfg"):
                orig = inner.cfg
                try:
                    if override_cfg is not None:
                        inner.cfg = override_cfg.item() if override_cfg.numel() == 1 else override_cfg.mean().item()
                    denoised = inner.predict_noise(x, sigma_t, model_options=extra["model_options"], seed=extra.get("seed"))
                finally:
                    inner.cfg = orig
            else:
                _ = self.model_(x, sigma_t, **extra)
                denoised = comfy.samplers.cfg_function(inner.inner_model, cond, uncond, override_cfg, x, sigma_t, model_options=mo)
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
        
        if torch.isclose(t_curr, torch.tensor(0.0, device=t_curr.device)).all():
            return apply_ema(prev_lp)
        
        # RESTORED v1.4.2: NO epsilon added to t_csq!
        t_csq = t_curr**2
        if torch.isclose(t_csq, torch.tensor(0.0, device=t_csq.device)).all():
            return apply_ema(prev_lp)
        
        t_ndc = t_next**2 / t_csq
        t_cmn = t_csq - t_next**2
        sst_t = t_cmn * t_ndc
        
        if torch.isclose(sst_t, torch.tensor(0.0, device=sst_t.device)).all():
            return apply_ema(prev_lp)
        
        pred_base = t_ndc * x_curr
        uncond_mean = pred_base + (t_cmn / t_csq) * uncond
        cond_mean = pred_base + (t_cmn / t_csq) * cond
        diff = batch_mse_loss(x_next, cond_mean) - batch_mse_loss(x_next, uncond_mean)
        
        # RESTORED v1.4.2: NO epsilon in division!
        if torch.isclose(sst_t, torch.tensor(0.0, device=sst_t.device)):
            result = prev_lp + self.config.offset
        else:
            result = prev_lp - self.config.temp / (2 * sst_t) * diff + self.config.offset
        
        return apply_ema(result)

    def _do_callback(self, step, x, sigma, denoised):
        if self.callback_:
            try:
                self.callback_({"i": step, "x": x, "sigma": sigma, "sigma_hat": sigma, "denoised": denoised})
            except Exception as e:
                logger.warning(f"Callback error at step {step}: {e}")

    def __call__(self):
        """Main sampling loop with ENHANCED console output."""
        try:
            # ENHANCED: Track guidance scales for summary
            gs_used = [] if self.debug_mode >= 1 else None
            if self.debug_mode >= 1: logger.info("Starting sampling loop")
            
            x = self.x.clone()
            num_steps = len(self.sigmas) - 1
            if num_steps <= 0: return x
            
            astart, aend = self.first_ancestral_step, self.last_ancestral_step
            actual_end = min(self.end_sigma_index if self.end_sigma_index >= 0 else num_steps - 1, num_steps - 1)
            
            for idx in trange(num_steps, disable=self.disable_pbar):
                if idx < self.start_sigma_index or idx > actual_end: continue
                
                with self.profiler.profile_step(f"step_{idx}"):
                    s_curr, s_next = self.sigmas[idx], self.sigmas[idx + 1]
                    
                    # FBG guidance
                    self.guidance_scale = self.get_dynamic_guidance_scale(self.log_posterior, self.guidance_scale, s_curr.max().item())
                    if gs_used is not None: gs_used.append(self.guidance_scale.mean().item())
                    
                    # Denoise
                    denoised, cond, uncond = self._model_denoise_with_guidance(x, s_curr, self.guidance_scale)
                    
                    # ENHANCED: NaN/Inf recovery
                    if self._check_for_nan_inf(denoised, "denoised", idx):
                        logger.warning(f"Recovering from NaN/Inf at step {idx}")
                        denoised = torch.where(torch.isnan(denoised) | torch.isinf(denoised), x, denoised)
                    
                    x_orig = x.clone()
                    self._do_callback(idx, x, s_curr, denoised)
                    
                    # ENHANCED: Debug step info
                    self._debug_step_info(idx, s_curr, s_next, x)
                    
                    # Sampling step
                    is_anc = (astart <= idx <= aend) if astart <= aend else False
                    
                    if not is_anc:
                        blend = s_next / s_curr if s_curr > 0 else 0.0
                        bf = self.get_progressive_blend_function(idx, num_steps)
                        if self.conditional_blend_mode and s_curr.max().item() < self.conditional_blend_sigma_threshold:
                            bf = self.conditional_blend_function
                        x = bf(denoised, x, blend)
                    else:
                        seed = self._stepped_seed(idx)
                        if seed is not None:
                            torch.manual_seed(seed)
                            if self.debug_mode >= 2: logger.debug(f"Using seed: {seed}")
                        
                        noise = self.noise_sampler(s_curr, s_next)
                        
                        # Noise clamping
                        if self.clamp_noise_norm:
                            nn = torch.norm(noise, p=2, dim=list(range(1, noise.ndim)), keepdim=True)
                            sf = torch.where(nn > self.max_noise_norm, self.max_noise_norm / (nn + 1e-8), torch.ones_like(nn))
                            noise = noise * sf
                        
                        if self.is_rf:
                            bf = self.get_progressive_blend_function(idx, num_steps)
                            x = bf(denoised, noise, s_next)
                        else:
                            x = denoised + noise * s_next
                        
                        # Conditional blend on change
                        if self.conditional_blend_on_change:
                            dn = torch.norm(denoised, dim=list(range(1, denoised.ndim)), keepdim=True) + 1e-8
                            nn = torch.norm(noise * s_next, dim=list(range(1, noise.ndim)), keepdim=True)
                            rel = (nn / dn).mean().item()
                            if rel > self.conditional_blend_change_threshold:
                                bf = min(1.0, (rel - self.conditional_blend_change_threshold) / self.conditional_blend_change_threshold)
                                x = self.conditional_blend_function(denoised, x, bf)
                    
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
                    
                    # Logging
                    if self.debug_mode >= 1:
                        logger.info(f"Step {idx}: GS={self.guidance_scale.mean().item():.2f}")
            
            if self.enable_clamp_output: x = torch.clamp(x, -1.0, 1.0)
            
            # ENHANCED: Beautiful summary box
            if self.debug_mode >= 1:
                print("\n" + "="*60)
                print("PingPongSamplerFBG - Sampling Summary")
                print("="*60)
                if gs_used:
                    print(f"Guidance Scale Statistics:")
                    print(f"  Min:     {min(gs_used):.3f}")
                    print(f"  Max:     {max(gs_used):.3f}")
                    print(f"  Average: {sum(gs_used) / len(gs_used):.3f}")
                if hasattr(self, 'checkpoints'):
                    print(f"Checkpoints saved: {len(self.checkpoints)}")
                print("="*60)
            
            # ENHANCED: Always print profiler summary for visibility
            if self.profiler.enabled:
                print("\n" + self.profiler.get_summary() + "\n")
            
            return x
        except Exception as e:
            logger.error(f"Crash: {e}")
            logger.debug(traceback.format_exc())
            return self.x

# =================================================================================
# == Node Wrapper Class                                                           ==
# =================================================================================

class PingPongSamplerNodeFBG:
    """ComfyUI node wrapper."""
    @classmethod
    def INPUT_TYPES(cls):
        d = FBGConfig()
        return {
            "required": {
                "step_random_mode": (["off", "block", "reset", "step"], {"default": "block"}),
                "step_size": ("INT", {"default": 4, "min": 1, "max": 100}),
                "seed": ("INT", {"default": 80085, "min": 0, "max": 0xffffffffffffffff}),
                "first_ancestral_step": ("INT", {"default": 0}),
                "last_ancestral_step": ("INT", {"default": -1}),
                "ancestral_noise_type": (["gaussian", "uniform", "brownian"], {"default": "gaussian"}),
                "start_sigma_index": ("INT", {"default": 0}),
                "end_sigma_index": ("INT", {"default": -1}),
                "enable_clamp_output": ("BOOLEAN", {"default": False}),
                "scheduler": ("SCHEDULER",),
                "blend_mode": (tuple(_INTERNAL_BLEND_MODES.keys()), {"default": "lerp"}),
                "step_blend_mode": (tuple(_INTERNAL_BLEND_MODES.keys()), {"default": "lerp"}),
                "fbg_sampler_mode": (["EULER", "PINGPONG"], {"default": "EULER"}),
                "cfg_scale": ("FLOAT", {"default": d.cfg_scale, "min": -1000, "max": 1000, "step": 0.01}),
                "cfg_start_sigma": ("FLOAT", {"default": d.cfg_start_sigma}),
                "cfg_end_sigma": ("FLOAT", {"default": d.cfg_end_sigma}),
                "fbg_start_sigma": ("FLOAT", {"default": d.fbg_start_sigma}),
                "fbg_end_sigma": ("FLOAT", {"default": d.fbg_end_sigma}),
                "max_guidance_scale": ("FLOAT", {"default": d.max_guidance_scale}),
                "initial_guidance_scale": ("FLOAT", {"default": d.initial_guidance_scale}),
                "fbg_guidance_multiplier": ("FLOAT", {"default": d.fbg_guidance_multiplier}),
                "guidance_max_change": ("FLOAT", {"default": d.guidance_max_change}),
                "pi": ("FLOAT", {"default": d.pi, "step": 0.01}),
                "t_0": ("FLOAT", {"default": d.t_0, "step": 0.01}),
                "t_1": ("FLOAT", {"default": d.t_1, "step": 0.01}),
                "fbg_temp": ("FLOAT", {"default": d.temp}),
                "fbg_offset": ("FLOAT", {"default": d.offset}),
                "fbg_eta": ("FLOAT", {"default": 0.0}),
                "fbg_s_noise": ("FLOAT", {"default": 1.0}),
                "max_posterior_scale": ("FLOAT", {"default": d.max_posterior_scale}),
                "log_posterior_initial_value": ("FLOAT", {"default": d.initial_value}),
                "log_posterior_ema_factor": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "adaptive_noise_scaling": ("BOOLEAN", {"default": False}),
                "noise_scale_factor": ("FLOAT", {"default": 1.0}),
                "progressive_blend_mode": ("BOOLEAN", {"default": False}),
                "conditional_blend_mode": ("BOOLEAN", {"default": False}),
                "conditional_blend_sigma_threshold": ("FLOAT", {"default": 0.5}),
                "conditional_blend_function_name": (tuple(_INTERNAL_BLEND_MODES.keys()), {"default": "slerp"}),
                "conditional_blend_on_change": ("BOOLEAN", {"default": False}),
                "conditional_blend_change_threshold": ("FLOAT", {"default": 0.1}),
                "clamp_noise_norm": ("BOOLEAN", {"default": False}),
                "max_noise_norm": ("FLOAT", {"default": 1.0}),
                "gradient_norm_tracking": ("BOOLEAN", {"default": False}),
                "enable_profiling": ("BOOLEAN", {"default": False}),
                "debug_mode": ("INT", {"default": 0, "min": 0, "max": 2}),
                "tensor_memory_optimization": ("BOOLEAN", {"default": False}),
                "early_exit_threshold": ("FLOAT", {"default": 1e-6}),
                "ancestral_start_sigma": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 9999.0, "step": 0.001}),
                "ancestral_end_sigma": ("FLOAT", {"default": 0.004, "min": 0.0, "max": 9999.0, "step": 0.001}),
                "sigma_range_preset": (["Custom", "High", "Mid", "Low", "All"], {"default": "Custom"}),
            },
            "optional": {
                "yaml_settings_str": ("STRING", {"multiline": True}),
                "checkpoint_steps_str": ("STRING", {"default": ""}),
            }
        }
    
    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "get_sampler"
    CATEGORY = "MD_Nodes/Sampling"
    
    def get_sampler(self, **kwargs):
        yaml_str = kwargs.get("yaml_settings_str", "")
        if yaml_str:
            try:
                overrides = yaml.safe_load(yaml_str)
                if isinstance(overrides, dict):
                    for k, v in overrides.items():
                        if k == "fbg_config" and isinstance(v, dict):
                            for fk, fv in v.items(): kwargs[fk] = fv
                        else: kwargs[k] = v
            except Exception as e:
                logger.warning(f"YAML error: {e}")
        
        cp_str = kwargs.get("checkpoint_steps_str", "")
        if cp_str:
            try:
                kwargs["checkpoint_steps"] = [int(x.strip()) for x in cp_str.split(",") if x.strip()]
            except: pass
        
        mode_str = kwargs.get("fbg_sampler_mode", "EULER")
        try: mode = getattr(SamplerMode, mode_str.upper())
        except: mode = SamplerMode.EULER
        
        fbg_cfg = FBGConfig(
            sampler_mode=mode,
            cfg_start_sigma=kwargs.get("cfg_start_sigma", 1.0),
            cfg_end_sigma=kwargs.get("cfg_end_sigma", 0.004),
            fbg_start_sigma=kwargs.get("fbg_start_sigma", 1.0),
            fbg_end_sigma=kwargs.get("fbg_end_sigma", 0.004),
            fbg_guidance_multiplier=kwargs.get("fbg_guidance_multiplier", 1.0),
            ancestral_start_sigma=1.0, ancestral_end_sigma=0.004,
            cfg_scale=kwargs.get("cfg_scale", 1.0),
            max_guidance_scale=kwargs.get("max_guidance_scale", 10.0),
            max_posterior_scale=kwargs.get("max_posterior_scale", 3.0),
            initial_value=kwargs.get("log_posterior_initial_value", 0.0),
            initial_guidance_scale=kwargs.get("initial_guidance_scale", 1.0),
            guidance_max_change=kwargs.get("guidance_max_change", 1000.0),
            temp=kwargs.get("fbg_temp", 0.0),
            offset=kwargs.get("fbg_offset", 0.0),
            pi=kwargs.get("pi", 0.5),
            t_0=kwargs.get("t_0", 0.5),
            t_1=kwargs.get("t_1", 0.4)
        )
        
        # FIXED: Rename blend_mode -> blend_function_name for go() method
        kwargs['blend_function_name'] = kwargs.get('blend_mode', 'lerp')
        kwargs['step_blend_function_name'] = kwargs.get('step_blend_mode', 'lerp')
        # conditional_blend_function_name is already correct
        
        kwargs['fbg_config'] = fbg_cfg
        
        return (comfy.samplers.KSAMPLER(PingPongSamplerCore.go, extra_options=kwargs),)

NODE_CLASS_MAPPINGS = {"PingPongSamplerNodeFBG": PingPongSamplerNodeFBG}
NODE_DISPLAY_NAME_MAPPINGS = {"PingPongSamplerNodeFBG": "MD: PingPong Sampler (FBG)"}