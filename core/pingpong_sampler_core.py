# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# █▓▒░                                                                     ░▒▓█
# █▓▒░         MD_Nodes Core: PingPong Sampler FBG (IP-Protected)          ░▒▓█
# █▓▒░                                                                     ░▒▓█
# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
# ╠═ © 2026 MDMAchine
# ╠═ License: GNU General Public License v3.0 (GPL v3)
# ║
# ║  This program is free software: you can redistribute it and/or modify
# ║  it under the terms of the GNU General Public License as published by
# ║  the Free Software Foundation, either version 3 of the License, or
# ║  (at your option) any later version.
# ║
# ║  This program is distributed in the hope that it will be useful,
# ║  but WITHOUT ANY WARRANTY; without even the implied warranty of
# ║  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# ║  GNU General Public License for more details.
# ║
# ║  You should have received a copy of the GNU General Public License
# ║  along with this program. If not, see <https://www.gnu.org/licenses/>.
# ╠════════════════════════════════════════════════════════════════════════════
# ║ ░▒▓ DESCRIPTION:
# ║   Core mathematical engine for Forward-Backward Guidance (FBG).
# ║   Handles tensor restarts, spherical linear interpolation, and noise scaling.
# ╚════════════════════════════════════════════════════════════════════════════


VERSION = "v3.0.0"  # UPS v1.5.8

import math
import time
import logging
import traceback
import torch
import numpy as np
from enum import Enum, auto
from typing import NamedTuple, Optional, Dict, Any, List, Callable
from contextlib import contextmanager

# =================================================================================
# == Constants (Bit-Exact Parity)
# =================================================================================

CONST_JS_MAX_SAFE_INTEGER = 9007199254740991
CONST_SEED_MIN = 0
CONST_EPSILON = 1e-8
CONST_EARLY_EXIT_THRESHOLD = 1e-6

logger = logging.getLogger("MD_Nodes.Core.PingPong")

# =================================================================================
# == Helper Classes & Enums
# =================================================================================

class SamplerMode(Enum):
    EULER = auto()
    PINGPONG = auto()

class FBGConfig(NamedTuple):
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

# =================================================================================
# == Math Functions (Blend & Interpolation)
# =================================================================================

def slerp(a, b, t):
    """Spherical linear interpolation with enhanced numerical stability."""
    eps = CONST_EPSILON
    a_flat = a.flatten(start_dim=1).float()
    b_flat = b.flatten(start_dim=1).float().to(a.device)
    
    a_norm = torch.norm(a_flat, dim=-1, keepdim=True)
    b_norm = torch.norm(b_flat, dim=-1, keepdim=True)
    
    a_n = a_flat / (a_norm + eps)
    b_n = b_flat / (b_norm + eps)
    
    dot = (a_n * b_n).sum(dim=-1, keepdim=True).clamp(-0.9999, 0.9999)
    
    if torch.all(torch.abs(dot) > 0.9995):
        return torch.lerp(a, b, t)
        
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    
    t_tensor = torch.tensor(t, device=a.device, dtype=a.dtype)
    scale_a = torch.sin((1 - t_tensor) * theta) / (sin_theta + eps)
    scale_b = torch.sin(t_tensor * theta) / (sin_theta + eps)
    result_n = scale_a * a_n + scale_b * b_n
    
    result_norm = torch.lerp(a_norm, b_norm, t_tensor)
    result_flat = result_n * result_norm
    return result_flat.reshape_as(a)

def cosine_interpolation(a, b, t):
    t_tensor = torch.tensor(t * math.pi, device=a.device)
    cos_t = (1.0 - torch.cos(t_tensor)) * 0.5
    return a * (1.0 - cos_t) + b * cos_t

def cubic_interpolation(a, b, t):
    t_tensor = torch.tensor(t, device=a.device)
    cubic_t = t_tensor * t_tensor * (3.0 - 2.0 * t_tensor)
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

def batch_mse_loss(a, b, start_dim=1):
    a = a.float()
    b = b.float().to(a.device)
    if a.numel() > 1e7:
        diff = a - b
        return (diff * diff).sum(dim=tuple(range(start_dim, a.ndim)))
    return torch.sum((a - b).pow(2), dim=tuple(range(start_dim, a.ndim)))

def validate_seed(seed_value):
    try:
        int_value = int(seed_value)
    except (ValueError, TypeError):
        return CONST_SEED_MIN
    return max(CONST_SEED_MIN, min(int_value, CONST_JS_MAX_SAFE_INTEGER))

# =================================================================================
# == Specialized Performance Profiler
# =================================================================================

class PerformanceProfiler:
    """Profiler for step timing, memory usage, and restart tracking."""
    def __init__(self, enabled=False):
        self.enabled = enabled
        self.step_times = []
        self.memory_usage = []
        self.restart_count = 0
        self.restart_steps = []

    @contextmanager
    def profile_step(self, step_name="step"):
        if not self.enabled:
            yield
            return
        
        start_time = time.time()
        start_mem = 0
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            start_mem = torch.cuda.memory_allocated()
            
        try:
            yield
        finally:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                end_mem = torch.cuda.memory_allocated()
                self.memory_usage.append(end_mem - start_mem)
            else:
                 self.memory_usage.append(0)
            self.step_times.append(time.time() - start_time)

    def log_restart(self, step_index, sigma_curr, sigma_next):
        if not self.enabled: return
        self.restart_count += 1
        s_c = float(sigma_curr.item()) if isinstance(sigma_curr, torch.Tensor) else float(sigma_curr)
        s_n = float(sigma_next.item()) if isinstance(sigma_next, torch.Tensor) else float(sigma_next)
        self.restart_steps.append({'step': step_index, 'sigma_curr': s_c, 'sigma_next': s_n})

    def get_summary(self):
        if not self.step_times: return "No profiling data available"
        total_time = sum(self.step_times)
        avg_time = total_time / len(self.step_times)
        summary = "[PingPongSamplerFBG] Performance Summary:\n"
        summary += f"  Total time: {total_time:.3f}s\n"
        summary += f"  Average step time: {avg_time:.3f}s\n"
        if self.restart_count > 0:
            summary += f"  Restarts executed: {self.restart_count}\n"
        if self.memory_usage and torch.cuda.is_available():
             avg_mem = sum(self.memory_usage) / len(self.memory_usage)
             summary += f"  Avg memory delta: {avg_mem / 1024**2:.1f}MB"
        return summary

# =================================================================================
# == Core Logic Class
# =================================================================================

class PingPongSamplerCore:
    def __init__(self, x, sigmas, denoise_impl: Callable, is_rf: bool, extra_args=None, callback=None, disable=None, noise_sampler=None,
                 start_sigma_index=0, end_sigma_index=-1, enable_clamp_output=False, step_random_mode="off",
                 step_size=5, seed=42, blend_function=torch.lerp, step_blend_function=torch.lerp,
                 scheduler=None, pingpong_options=None, fbg_config=None, debug_mode=0, sigma_range_preset="Custom",
                 conditional_blend_mode=False, conditional_blend_sigma_threshold=0.5,
                 conditional_blend_function=slerp, conditional_blend_on_change=False,
                 conditional_blend_change_threshold=0.1, clamp_noise_norm=False, max_noise_norm=1.0,
                 log_posterior_ema_factor=0.0, adaptive_noise_scaling=False, noise_scale_factor=1.0,
                 progressive_blend_mode=False, enable_profiling=False, checkpoint_steps=None,
                 early_exit_threshold=CONST_EARLY_EXIT_THRESHOLD, tensor_memory_optimization=False,
                 ancestral_noise_type="gaussian", enable_restarts=False, restart_mode="balanced",
                 restart_noise_scale=0.5, restart_s_noise=1.0, restart_steps="", **kwargs):
        
        # Dependency Injection
        self.denoise_impl = denoise_impl
        self.is_rf = is_rf
        
        self.x = x
        self.sigmas = sigmas
        self.extra_args = extra_args.copy() if extra_args is not None else {}
        self.callback_ = callback
        self.disable_pbar = disable
        
        self.debug_mode = debug_mode
        if self.debug_mode >= 2: logger.setLevel(logging.DEBUG)
        elif self.debug_mode >= 1: logger.setLevel(logging.INFO)
        else: logger.setLevel(logging.WARNING)

        self.start_sigma_index = start_sigma_index
        self.end_sigma_index = end_sigma_index
        self.enable_clamp_output = enable_clamp_output
        self.step_random_mode = step_random_mode
        self.step_size = step_size
        self.seed = validate_seed(seed) if seed is not None else 0
        self.blend_function = blend_function
        self.step_blend_function = step_blend_function
        self.adaptive_noise_scaling = adaptive_noise_scaling
        self.noise_scale_factor = noise_scale_factor
        self.progressive_blend_mode = progressive_blend_mode
        self.checkpoint_steps = checkpoint_steps or []
        self.early_exit_threshold = early_exit_threshold
        self.tensor_memory_optimization = tensor_memory_optimization
        self.profiler = PerformanceProfiler(enable_profiling)
        
        num_steps_available = len(sigmas) - 1 if len(sigmas) > 0 else 0
        if pingpong_options is None: pingpong_options = {}
        raw_first = pingpong_options.get("first_ancestral_step", kwargs.get("first_ancestral_step", 0))
        raw_last = pingpong_options.get("last_ancestral_step", kwargs.get("last_ancestral_step", num_steps_available - 1))
        self.first_ancestral_step = max(0, min(raw_first, raw_last))
        self.last_ancestral_step = min(num_steps_available - 1, max(raw_first, raw_last)) if num_steps_available > 0 else -1
        
        self.sigma_range_preset = sigma_range_preset
        self.original_fbg_config = fbg_config if fbg_config is not None else FBGConfig()
        self.config = self.original_fbg_config
        if self.sigma_range_preset != "Custom" and num_steps_available > 0:
            self.config = self._apply_sigma_preset(num_steps_available)
            
        self.conditional_blend_mode = conditional_blend_mode
        self.conditional_blend_sigma_threshold = conditional_blend_sigma_threshold
        self.conditional_blend_function = conditional_blend_function
        self.conditional_blend_on_change = conditional_blend_on_change
        self.conditional_blend_change_threshold = conditional_blend_change_threshold
        self.clamp_noise_norm = clamp_noise_norm
        self.max_noise_norm = max(0.01, max_noise_norm)
        self.log_posterior_ema_factor = max(0.0, min(1.0, log_posterior_ema_factor))
        self.noise_sampler = self._setup_noise_sampler(noise_sampler, ancestral_noise_type)
        self.update_fbg_config_params()
        
        cfg = self.config
        self.minimal_log_posterior = self._calculate_minimal_log_posterior(cfg)
        self.log_posterior = x.new_full((x.shape[0],), cfg.initial_value)
        self.guidance_scale = x.new_full((x.shape[0], *(1,) * (x.ndim - 1)), cfg.initial_guidance_scale)
        
        self.enable_restarts = enable_restarts
        self.restart_mode = restart_mode
        self.restart_noise_scale = restart_noise_scale
        self.restart_s_noise = restart_s_noise
        self.custom_restart_schedule = None
        if enable_restarts and restart_steps and restart_steps.strip():
            try:
                self.custom_restart_schedule = [int(x.strip()) for x in restart_steps.split(',') if x.strip()]
            except Exception: pass
            
        self._prev_x = None
        self.checkpoints = {}

    def _apply_sigma_preset(self, num_steps):
        try:
             sorted_desc = torch.sort(self.sigmas, descending=True).values
             if len(sorted_desc) < 2: return self.config
             sigma_at_0 = sorted_desc[0].item()
             sigma_near_end = sorted_desc[-2].item()
             d = self.config._asdict()
             preset = self.sigma_range_preset
             
             if preset == "High":
                idx = min(max(1, num_steps // 4), len(sorted_desc) - 2)
                threshold = sorted_desc[idx].item()
                d.update({'cfg_start_sigma': sigma_at_0, 'cfg_end_sigma': threshold, 'fbg_start_sigma': sigma_at_0, 'fbg_end_sigma': threshold})
             elif preset == "Mid":
                idx_start = min(max(1, num_steps // 4), len(sorted_desc) - 2)
                idx_end = min(max(idx_start + 1, 3 * num_steps // 4), len(sorted_desc) - 2)
                d.update({'cfg_start_sigma': sorted_desc[idx_start].item(), 'cfg_end_sigma': sorted_desc[idx_end].item(), 'fbg_start_sigma': sorted_desc[idx_start].item(), 'fbg_end_sigma': sorted_desc[idx_end].item()})
             elif preset == "Low":
                idx = min(max(1, 3 * num_steps // 4), len(sorted_desc) - 2)
                threshold = sorted_desc[idx].item()
                d.update({'cfg_start_sigma': threshold, 'cfg_end_sigma': sigma_near_end, 'fbg_start_sigma': threshold, 'fbg_end_sigma': sigma_near_end})
             elif preset == "All":
                  d.update({'cfg_start_sigma': sigma_at_0, 'cfg_end_sigma': sigma_near_end, 'fbg_start_sigma': sigma_at_0, 'fbg_end_sigma': sigma_near_end})
             return FBGConfig(**d)
        except Exception: return self.config

    def _setup_noise_sampler(self, noise_sampler, noise_type):
        if noise_sampler is not None: return noise_sampler
        def create_sampler(func):
            def s(sigma, sigma_next):
                bn = func()
                if self.adaptive_noise_scaling:
                    prog = 1.0 - (sigma / self.sigmas[0]) if self.sigmas[0] > 0 else 0.0
                    bn = bn * self.noise_scale_factor * (1.0 + prog * 0.5)
                return bn
            return s
        if noise_type == "uniform": return create_sampler(lambda: torch.rand_like(self.x) * 2 - 1)
        if noise_type == "brownian": return create_sampler(lambda: torch.randn_like(self.x).cumsum(dim=-1) / (self.x.shape[-1]**0.5))
        return create_sampler(lambda: torch.randn_like(self.x))

    def _calculate_minimal_log_posterior(self, cfg):
        try:
            den = (cfg.max_guidance_scale - cfg.cfg_scale) if cfg.cfg_scale > 1 else (cfg.max_guidance_scale - 1.0)
            if den <= 0: return float('-inf')
            num = (1.0 - cfg.pi) * (cfg.max_guidance_scale - cfg.cfg_scale + 1) if cfg.cfg_scale > 1 else (1.0 - cfg.pi) * cfg.max_guidance_scale
            return math.log(num / den)
        except Exception: return float('-inf')

    def get_progressive_blend_function(self, step_idx, total_steps):
        if not self.progressive_blend_mode: return self.blend_function
        p = step_idx / max(total_steps - 1, 1)
        if p < 0.3: return torch.lerp
        elif p < 0.7: return cosine_interpolation
        return cubic_interpolation

    def _check_for_nan_inf(self, tensor, name, step_idx):
        if tensor is None: return False
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            return True
        return False

    def _stepped_seed(self, step):
        if self.step_random_mode == "off": return None
        sz = max(self.step_size, 1)
        base = validate_seed(self.seed)
        seed_map = {"block": base + (step // sz), "reset": base + (step * sz), "step": base + step}
        return validate_seed(seed_map.get(self.step_random_mode, base))

    def _get_sigma_square_tilde(self, sigmas):
        if len(sigmas) < 2: return torch.tensor([], device=sigmas.device)
        # ---> CAST TO FP32 HERE
        sigmas_f = sigmas.float()
        s_sq, sn_sq = sigmas_f[:-1] ** 2, sigmas_f[1:] ** 2
        safe_s_sq = torch.where(s_sq == 0, torch.tensor(1e-8, device=s_sq.device), s_sq)
        return ((s_sq - sn_sq) * sn_sq / safe_s_sq).flip(dims=(0,))

    def _get_offset(self, steps, sst):
        cfg = self.config
        t0 = max(0.0, min(1.0, cfg.t_0))
        if t0 >= 1.0: return 0.0
        try:
            return round(math.log((1.0 - cfg.pi) * 3.0 / 2.0) / ((1.0 - t0) * steps), 4)
        except Exception: return 0.0

    def _get_temp(self, steps, offset, sst):
        cfg = self.config
        t1 = max(0.0, min(1.0, cfg.t_1))
        idx = max(0, min(int(math.floor(t1 * (steps - 1))), len(sst) - 1))
        if len(sst) == 0: return 0.0
        try:
            val = sst[idx].item()
            return round((2 * val / 10.0 * offset), 4)
        except Exception: return 0.0

    def update_fbg_config_params(self):
        if self.config.t_0 == 0 and self.config.t_1 == 0: return
        steps = len(self.sigmas) - 1
        if steps <= 0: return
        sst = self._get_sigma_square_tilde(self.sigmas)
        d = self.config._asdict()
        d.update({"offset": self._get_offset(steps, sst), "temp": self._get_temp(steps, 0, sst)})
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
        if using_cfg: gs += cfg.cfg_scale - 1.0
        gs = gs.clamp(1.0, cfg.max_guidance_scale).view(gs_prev.shape)
        safe_prev = torch.where(gs_prev.abs() < 1e-6, torch.full_like(gs_prev, 1e-6), gs_prev)
        change = ((gs - gs_prev) / safe_prev).clamp(-cfg.guidance_max_change, cfg.guidance_max_change)
        return (gs_prev + gs_prev * change).clamp(1.0, cfg.max_guidance_scale)

    def _update_log_posterior(self, prev_lp, x_curr, x_next, t_curr, t_next, uncond, cond):
        if cond is None or uncond is None: return prev_lp
        def apply_ema(val):
            if self.log_posterior_ema_factor > 0:
                return (self.log_posterior_ema_factor * prev_lp + (1 - self.log_posterior_ema_factor) * val).clamp(self.minimal_log_posterior, self.config.max_posterior_scale)
            return val.clamp(self.minimal_log_posterior, self.config.max_posterior_scale)
        
        # ---> CAST TO FP32 AND CLAMP
        t_curr_f = t_curr.float()
        t_next_f = t_next.float()
        
        t_csq = t_curr_f**2
        t_csq_safe = torch.clamp(t_csq, min=1e-8)
        
        if torch.isclose(t_csq_safe, torch.tensor(0.0, device=t_csq_safe.device)).all(): return apply_ema(prev_lp)
        
        t_ndc = t_next_f**2 / t_csq_safe
        t_cmn = t_csq - t_next_f**2
        sst_t = t_cmn * t_ndc
        
        pred_base = t_ndc * x_curr
        diff = batch_mse_loss(x_next, pred_base + (t_cmn / t_csq_safe) * cond) - batch_mse_loss(x_next, pred_base + (t_cmn / t_csq_safe) * uncond)
        
        if torch.isclose(sst_t, torch.tensor(0.0, device=sst_t.device)): result = prev_lp + self.config.offset
        else: result = prev_lp - self.config.temp / (2 * sst_t) * diff + self.config.offset
        return apply_ema(result)

    def _generate_restart_schedule(self, sigmas):
        n = len(sigmas) - 1
        mode = self.restart_mode
        if mode == 'aggressive': return [i for i in range(1, n, 2)]
        if mode == 'conservative': return [i for i in range(2, n // 2, 3)]
        if mode == 'detail_focus': return [i for i in range(n // 2, n) if i % 2 == 0]
        if mode == 'composition_focus': return [i for i in range(1, n // 2) if i % 2 == 0]
        sch = []
        for i in range(1, n):
            if i % 3 == 0:
                sch.append(i)
        return sch

    def _execute_restart_step(self, x, sigma_curr, sigma_next, step_index):
        sigma_restart = sigma_next + (sigma_curr - sigma_next) * self.restart_noise_scale
        seed = validate_seed(self.seed + step_index + 982451653)
        gen = torch.Generator(device=x.device).manual_seed(seed)
        noise = torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=gen)
        
        # ---> USE STANDARD MATH.SQRT WITH FLOAT CAST
        s_res_f = float(sigma_restart.item() if hasattr(sigma_restart, 'item') else sigma_restart)
        s_next_f = float(sigma_next.item() if hasattr(sigma_next, 'item') else sigma_next)
        noise_amt = math.sqrt(max(0.0, s_res_f**2 - s_next_f**2) + 1e-8)
        
        x_renoised = x + noise * noise_amt * self.restart_s_noise
        
        denoised, _, _ = self.denoise_impl(x_renoised, sigma_restart, self.guidance_scale, self.extra_args)
        
        d_restart = (x_renoised - denoised) / sigma_restart
        return x_renoised + d_restart * (sigma_next - sigma_restart)

    def execute_sampling(self):
        try:
            x = self.x.clone()
            num_steps = len(self.sigmas) - 1
            if num_steps <= 0: return x
            
            astart, aend = self.first_ancestral_step, self.last_ancestral_step
            actual_end = min(self.end_sigma_index if self.end_sigma_index >= 0 else num_steps - 1, num_steps - 1)
            
            restart_schedule = []
            if self.enable_restarts:
                if self.custom_restart_schedule is not None:
                    restart_schedule = self.custom_restart_schedule
                else:
                    restart_schedule = self._generate_restart_schedule(self.sigmas)

            for idx in range(num_steps):
                if idx < self.start_sigma_index or idx > actual_end: continue
                
                with self.profiler.profile_step(f"step_{idx}"):
                    s_curr, s_next = self.sigmas[idx], self.sigmas[idx + 1]
                    
                    self.guidance_scale = self.get_dynamic_guidance_scale(self.log_posterior, self.guidance_scale, s_curr.max().item())
                    
                    denoised, cond, uncond = self.denoise_impl(x, s_curr, self.guidance_scale, self.extra_args)
                    
                    if self._check_for_nan_inf(denoised, "denoised", idx):
                        denoised = torch.where(torch.isnan(denoised) | torch.isinf(denoised), x, denoised)
                    
                    x_orig = x.clone()
                    if self.callback_: self.callback_({'i': idx, 'x': x, 'sigma': s_curr, 'sigma_hat': s_curr, 'denoised': denoised})
                    
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
                    
                    if self.enable_restarts and idx in restart_schedule and s_next > 0:
                        x = self._execute_restart_step(x, s_curr, s_next, idx)
                        if self.profiler.enabled: self.profiler.log_restart(idx, s_curr, s_next)
                    
                    self.log_posterior = self._update_log_posterior(self.log_posterior, x_orig, x, s_curr, s_next, uncond, cond)
                    
                    if idx in self.checkpoint_steps: self.checkpoints[idx] = x.clone()
                    if s_next.min().item() > self.early_exit_threshold and hasattr(self, '_prev_x') and self._prev_x is not None:
                         if torch.norm(x - self._prev_x).item() < self.early_exit_threshold * 0.1: break
                    self._prev_x = x.clone()
                    
                    if self.enable_clamp_output and s_next.min().item() < 1e-3:
                        x = torch.clamp(x, -1.0, 1.0)
                        break
                    if s_next.min().item() <= 1e-6:
                        x = denoised
                        break
            
            if self.enable_clamp_output: x = torch.clamp(x, -1.0, 1.0)
            
            if self.debug_mode >= 1:
                logger.info(self.profiler.get_summary())
                
            return x
        except Exception as e:
            logger.error(f"Sampling error: {e}")
            return self.x


# ==============================================================================
# == Unit Tests (smoke — runs without ComfyUI)
# ==============================================================================

if __name__ == "__main__":
    print("\n🧪 Smoke tests: pingpong_sampler_core")
    print("   VERSION :", VERSION)
    _pass = _fail = 0

    def _check(label, expr):
        global _pass, _fail
        if expr:
            print(f"  ✅  {label}")
            _pass += 1
        else:
            print(f"  ❌  {label}")
            _fail += 1

    _check("VERSION defined",    VERSION == "v3.0.0")
    _check("CONST CONST_JS_MAX_SAFE_INTEGER defined", CONST_JS_MAX_SAFE_INTEGER is not None)
    _check("CONST CONST_SEED_MIN defined", CONST_SEED_MIN is not None)
    _check("fn slerp is callable", callable(slerp))
    _check("fn cosine_interpolation is callable", callable(cosine_interpolation))
    _check("fn cubic_interpolation is callable", callable(cubic_interpolation))

    print(f"\n  {_pass} passed, {_fail} failed")
    if _fail == 0:
        print("  🎉 All good.")
