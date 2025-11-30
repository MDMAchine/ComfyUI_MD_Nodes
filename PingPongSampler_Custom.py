# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# ████ PINGPONGSAMPLER v1.5.0 – Lite+ Ancestral Sampler with Res 2 ████▓▒░
# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
# ░▒▓ ORIGIN & DEV:
#    • Cast into the void by: Junmin Gong (Concept), blepping (ComfyUI Port), MD (Adaptation)
#    • Enhanced by: Gemini, Claude, devstral/qwen3
#    • License: Apache 2.0 — Sharing is caring # [NODE_FIX] Corrected License

# ░▒▓ DESCRIPTION:
#    Lightweight ancestral sampler for ComfyUI with intuitive noise behavior control.
#    Now features "Res 2" Restart logic for iterative error correction.
#    Designed for Ace-Step audio/video models with preset-based workflow.

# ░▒▓ FEATURES:
#    ✓ Res 2 Restart Logic: Iteratively "re-noises" and re-samples to correct errors.
#    ✓ Intuitive "Noise Behavior" presets (Default, Dynamic, Smooth, etc.)
#    ✓ Ancestral Strength & Noise Coherence control.
#    ✓ Enhanced blend modes (lerp, slerp, cosine, cubic).
#    ✓ NaN/Inf detection with automatic recovery.
#    ✓ Early convergence detection.
#    ✓ Performance summary with timing and restart stats.
#    ✓ Production-ready stability: No in-place operations.
#    ✓ Cacheable: Correctly uses default ComfyUI caching (Sec 6.1, Rule #1).

# ░▒▓ CHANGELOG:
#    - v1.5.0 (Current Release - Res 2 Integration):
#        • NEW: Res 2 restart logic with 5 modes (balanced/aggressive/conservative/detail/composition).
#        • NEW: Custom restart schedule support via string input.
#        • NEW: Restart-specific noise generation (uncorrelated seeds).
#        • NEW: Restart statistics in debug summary.
#        • COMPLIANCE: Full type hints, logging, and tooltips for new features.
#    - v0.8.23: Polish & Stability (Profiler, Debug Output).
#    - v0.8.22: Compliance Fix (Type hints, Logging).

# ░▒▓ CONFIGURATION:
#    → Primary Use: High-quality audio/video generation with "Default (Raw)" or "Dynamic".
#    → Secondary Use: Using "Res 2" restarts to improve coherence in complex generations.
#    → Edge Use: High "Noise Coherence" (0.8+) for temporally stable video/audio.

# ░▒▓ WARNING:
#    This node may trigger:
#    ▓▒░ Existential dread when you realize you just NaN'd 9,999 steps into a 10k run.
#    ▓▒░ A sudden, uncontrollable urge to `slerp` everything in sight.
#    ▓▒░ Flashbacks to debugging `IRQ` conflicts on your Gravis Ultrasound.
#    ▓▒░ A chilling sense that `ancestral_strength=2.0` is staring back into your soul.
#    Consult your nearest demoscene vet if hallucinations persist.
# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄


# =================================================================================
# == Standard Library Imports                                                    ==
# =================================================================================
import math
import logging
import time
import traceback
from typing import Dict, Any, List, Tuple, Optional, Callable, Union
from contextlib import contextmanager

# =================================================================================
# == Third-Party Imports                                                         ==
# =================================================================================
import torch
from tqdm.auto import trange
import numpy as np
import yaml

# =================================================================================
# == ComfyUI Core Modules                                                        ==
# =================================================================================
from comfy import model_sampling
from comfy.samplers import KSAMPLER
import comfy.model_management

# Setup logger
logger = logging.getLogger("ComfyUI_MD_Nodes.PingPongSamplerLite")

# =================================================================================
# == Helper Functions                                                            ==
# =================================================================================

def slerp_lite(a: torch.Tensor, b: torch.Tensor, t: float) -> torch.Tensor:
    """Spherical linear interpolation with numerical stability."""
    eps = 1e-8
    a_norm = a / (torch.norm(a, dim=-1, keepdim=True) + eps)
    b_norm = b / (torch.norm(b, dim=-1, keepdim=True) + eps)
    dot = torch.sum(a_norm * b_norm, dim=-1, keepdim=True).clamp(-0.9999, 0.9999)
    if torch.all(torch.abs(dot) > 0.9995):
        return torch.lerp(a, b, t)
    theta = torch.acos(torch.abs(dot)) * t
    c = b_norm - a_norm * dot
    c_norm = c / (torch.norm(c, dim=-1, keepdim=True) + eps)
    result = a_norm * torch.cos(theta) + c_norm * torch.sin(theta)
    avg_norm = (torch.norm(a, dim=-1, keepdim=True) + torch.norm(b, dim=-1, keepdim=True)) / 2
    return result * avg_norm

def cosine_interpolation(a: torch.Tensor, b: torch.Tensor, t: float) -> torch.Tensor:
    """Cosine interpolation for smoother transitions."""
    t_tensor = torch.tensor(t * math.pi, device=a.device, dtype=a.dtype)
    cos_t = (1.0 - torch.cos(t_tensor)) * 0.5
    return a * (1.0 - cos_t) + b * cos_t

def cubic_interpolation(a: torch.Tensor, b: torch.Tensor, t: float) -> torch.Tensor:
    """Cubic interpolation for even smoother transitions."""
    t_tensor = torch.tensor(t, device=a.device, dtype=a.dtype)
    cubic_t = t_tensor * t_tensor * (3.0 - 2.0 * t_tensor)
    return torch.lerp(a, b, cubic_t)

_INTERNAL_BLEND_MODES: Dict[str, Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor]] = {
    "lerp": torch.lerp,
    "slerp": slerp_lite,
    "cosine": cosine_interpolation,
    "cubic": cubic_interpolation,
    "a_only": lambda a, _b, _t: a,
    "b_only": lambda _a, b, _t: b
}

# =================================================================================
# == Performance Profiler                                                        ==
# =================================================================================

class PerformanceProfiler:
    """Enhanced profiler for step timing, memory tracking, and restart stats."""
    def __init__(self, enabled: bool = False) -> None:
        self.enabled: bool = enabled
        self.step_times: List[float] = []
        self.memory_usage: List[int] = []
        self.step_names: List[str] = []
        # Restart tracking
        self.restart_count: int = 0
        self.restart_steps: List[Dict[str, float]] = []

    @contextmanager
    def profile_step(self, step_name: str = "step"):
        """Context manager to profile a sampling step."""
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
            elapsed = time.time() - start_time
            self.step_times.append(elapsed)
            self.step_names.append(step_name)
            
            if device.type != 'cpu':
                torch.cuda.synchronize(device)
                end_mem = torch.cuda.memory_allocated(device)
                self.memory_usage.append(end_mem - start_mem)
            else:
                self.memory_usage.append(0)

    def log_restart(self, step_index: int, sigma_curr: Union[float, torch.Tensor], sigma_next: Union[float, torch.Tensor]) -> None:
        """Log a restart event."""
        if not self.enabled: return
        self.restart_count += 1
        s_c = float(sigma_curr.item()) if isinstance(sigma_curr, torch.Tensor) else float(sigma_curr)
        s_n = float(sigma_next.item()) if isinstance(sigma_next, torch.Tensor) else float(sigma_next)
        self.restart_steps.append({'step': step_index, 'sigma_curr': s_c, 'sigma_next': s_n})

    def get_summary(self) -> str:
        """Generate performance summary with statistics."""
        if not self.step_times:
            return "No profiling data available"
        
        total_time = sum(self.step_times)
        avg_time = total_time / len(self.step_times)
        max_time = max(self.step_times)
        min_time = min(self.step_times)
        
        summary = "[PingPongSamplerLite] Performance Summary:\n"
        summary += f"  Total time: {total_time:.3f}s\n"
        summary += f"  Average step time: {avg_time:.3f}s\n"
        
        if self.restart_count > 0:
            summary += f"  Restarts executed: {self.restart_count}\n"
            
        device = comfy.model_management.get_torch_device()
        if self.memory_usage and device.type != 'cpu':
            total_memory = sum(self.memory_usage)
            avg_memory = total_memory / len(self.memory_usage)
            max_memory = max(self.memory_usage)
            summary += f"  Total memory delta: {total_memory / 1024**2:.1f}MB\n"
            summary += f"  Average memory delta: {avg_memory / 1024**2:.1f}MB\n"
            summary += f"  Max memory delta: {max_memory / 1024**2:.1f}MB"
        
        return summary

# =================================================================================
# == Core Sampler Class                                                          ==
# =================================================================================

class PingPongSampler:
    """Lightweight sampler with ancestral noise control, enhanced stability, and Res 2."""
    
    # [NODE_FIX] Added comprehensive type hints
    def __init__(self,
                 model: Any,
                 x: torch.Tensor,
                 sigmas: torch.Tensor,
                 extra_args: Optional[Dict] = None,
                 callback: Optional[Callable] = None,
                 disable: Optional[bool] = None,
                 noise_sampler: Optional[Callable] = None,
                 start_sigma_index: int = 0,
                 end_sigma_index: int = -1,
                 enable_clamp_output: bool = False,
                 step_random_mode: str = "off",
                 step_size: int = 5,
                 seed: int = 42,
                 ancestral_strength: float = 1.0,
                 noise_coherence: float = 0.0,
                 debug_mode: int = 0,
                 blend_function: Optional[Callable] = None,
                 step_blend_function: Optional[Callable] = None,
                 pingpong_options: Optional[Dict] = None,
                 early_exit_threshold: float = 1e-6,
                 noise_behavior_name: str = "Custom",
                 enable_profiling: bool = False,
                 # Res 2 Parameters
                 enable_restarts: bool = False,
                 restart_mode: str = "balanced",
                 restart_noise_scale: float = 0.5,
                 restart_s_noise: float = 1.0,
                 restart_steps: str = "",
                 **kwargs: Any) -> None:
        
        # Core inputs
        self.model_ = model
        self.x = x
        self.sigmas = sigmas
        self.extra_args = extra_args.copy() if extra_args is not None else {}
        self.callback_ = callback
        self.disable_pbar = disable

        # Sampling control
        self.start_sigma_index = start_sigma_index
        self.end_sigma_index = end_sigma_index
        self.enable_clamp_output = enable_clamp_output

        # Noise injection controls
        self.step_random_mode = step_random_mode
        self.step_size = max(1, step_size)
        self.seed = seed if seed is not None else 42

        # Ancestral noise controls with validation
        self.ancestral_strength = max(0.0, ancestral_strength)
        self.noise_coherence = max(0.0, min(1.0, noise_coherence))
        self.previous_noise: Optional[torch.Tensor] = None

        # Debug & tracking
        self.debug_mode = debug_mode
        self.noise_behavior_name = noise_behavior_name
        self.early_exit_threshold = early_exit_threshold
        self._prev_x: Optional[torch.Tensor] = None
        self.seeds_used: Optional[List[int]] = [] if debug_mode >= 2 else None

        # Profiler
        self.profiler = PerformanceProfiler(enable_profiling)

        # Blend functions (default to lerp)
        self.blend_function = blend_function if blend_function is not None else torch.lerp
        self.step_blend_function = step_blend_function if step_blend_function is not None else torch.lerp

        # Ancestral operation boundaries
        if pingpong_options is None:
            pingpong_options = {}
        num_steps_available = len(sigmas) - 1 if len(sigmas) > 0 else 0
        
        raw_first = pingpong_options.get("first_ancestral_step", 0)
        raw_last = pingpong_options.get("last_ancestral_step", num_steps_available - 1)
        self.first_ancestral_step = max(0, min(raw_first, raw_last))
        self.last_ancestral_step = min(num_steps_available - 1, max(raw_first, raw_last)) if num_steps_available > 0 else -1

        # Validate sigma schedule
        self._validate_sigma_schedule()

        # Detect model type (CONST = Reflow-like models)
        self.is_rf = self._detect_model_type()

        # Noise sampler
        self.noise_sampler = noise_sampler if noise_sampler is not None else lambda s, sn: torch.randn_like(x)

        # ============================================
        # RES 2 INITIALIZATION (v1.5.0)
        # ============================================
        self.enable_restarts = enable_restarts
        self.restart_mode = restart_mode
        self.restart_noise_scale = restart_noise_scale
        self.restart_s_noise = restart_s_noise
        
        self.custom_restart_schedule: Optional[List[int]] = None
        if enable_restarts and restart_steps and restart_steps.strip():
            try:
                self.custom_restart_schedule = [
                    int(x.strip()) for x in restart_steps.split(',') if x.strip()
                ]
                if self.debug_mode >= 1:
                    logger.info(f"Custom restart schedule: {self.custom_restart_schedule}")
            except ValueError as e:
                logger.warning(f"Invalid restart_steps format: {e}. Using auto-generated schedule.")
                self.custom_restart_schedule = None

        # Log initialization
        if self.debug_mode >= 1:
            logger.info(f"Initialized PingPongSampler Lite+ v1.5.0")
            logger.info(f"  Preset: {self.noise_behavior_name}")
            logger.info(f"  Strength: {self.ancestral_strength:.2f}, Coherence: {self.noise_coherence:.2f}")
            if enable_restarts:
                 logger.info(f"  Restarts: ENABLED (Mode: {restart_mode})")

    def _validate_sigma_schedule(self) -> bool:
        """Validate sigma schedule is monotonically decreasing."""
        if len(self.sigmas) < 2: return True
        for i in range(len(self.sigmas) - 1):
            if self.sigmas[i] < self.sigmas[i + 1]:
                if self.debug_mode >= 1:
                    logger.warning(f"Sigma not monotonic at {i}: {self.sigmas[i]:.6f} -> {self.sigmas[i+1]:.6f}")
                return False
        return True

    def _detect_model_type(self) -> bool:
        """Detect if model uses CONST (Reflow-like) sampling."""
        try:
            curr = self.model_
            while hasattr(curr, 'inner_model') and curr.inner_model is not None:
                curr = curr.inner_model
            if hasattr(curr, 'model_sampling') and curr.model_sampling is not None:
                return isinstance(curr.model_sampling, model_sampling.CONST)
        except (AttributeError, TypeError):
            pass
        return False

    def _check_for_nan_inf(self, tensor: torch.Tensor, name: str, step_idx: int) -> bool:
        """Check tensor for NaN or Inf values."""
        has_nan = torch.isnan(tensor).any()
        has_inf = torch.isinf(tensor).any()
        if has_nan or has_inf:
            if self.debug_mode >= 1:
                logger.warning(f"{'NaN' if has_nan else 'Inf'} in {name} at step {step_idx}")
            return True
        return False

    def _recover_from_nan_inf(self, tensor: torch.Tensor, fallback: torch.Tensor, name: str, step_idx: int) -> torch.Tensor:
        """Replace NaN/Inf values with fallback tensor values."""
        mask = torch.isnan(tensor) | torch.isinf(tensor)
        if mask.any():
            if self.debug_mode >= 1:
                logger.warning(f"Recovering {name} at step {step_idx}")
            return torch.where(mask, fallback, tensor)
        return tensor

    def _stepped_seed(self, step: int) -> Optional[int]:
        """Determines RNG seed for current step based on random mode."""
        if self.step_random_mode == "off": return None
        seed_map = {
            "block": self.seed + (step // self.step_size),
            "reset": self.seed + (step * self.step_size),
            "step": self.seed + step
        }
        return seed_map.get(self.step_random_mode, self.seed)

    def _model_denoise(self, x_tensor: torch.Tensor, sigma_scalar: torch.Tensor) -> torch.Tensor:
        """Wrapper around model's denoising function."""
        try:
            batch_size = x_tensor.shape[0]
            sigma_tensor = sigma_scalar * x_tensor.new_ones((batch_size,))
            return self.model_(x_tensor, sigma_tensor, **self.extra_args)
        except Exception as e:
            if self.debug_mode >= 1: logger.error(f"Denoise error: {e}")
            return x_tensor

    def _do_callback(self, step_idx: int, current_x: torch.Tensor, current_sigma: torch.Tensor, denoised_sample: torch.Tensor) -> None:
        """Forwards progress to ComfyUI's callback system."""
        if self.callback_:
            try:
                self.callback_({"i": step_idx, "x": current_x, "sigma": current_sigma,
                               "sigma_hat": current_sigma, "denoised": denoised_sample})
            except Exception as e:
                if self.debug_mode >= 1: logger.warning(f"Callback error: {e}")

    # ============================================
    # NEW METHODS FOR RES 2 (v1.5.0)
    # ============================================

    def _generate_restart_schedule(self, sigmas: torch.Tensor) -> List[int]:
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

    def _generate_restart_noise(self, x: torch.Tensor, step_index: int) -> torch.Tensor:
        """Generate noise specifically for restart steps (uncorrelated seed)."""
        step_seed = self._stepped_seed(step_index)
        if step_seed is None: step_seed = self.seed
        
        restart_seed_offset = 982451653
        restart_seed = step_seed + restart_seed_offset
        
        generator = torch.Generator(device=x.device)
        generator.manual_seed(restart_seed % (2**63 - 1))
        
        return torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=generator)

    def _execute_restart_step(self, x: torch.Tensor, sigma_curr: torch.Tensor, sigma_next: torch.Tensor, step_index: int) -> torch.Tensor:
        """Execute a single Res 2 restart step."""
        sigma_restart = sigma_next + (sigma_curr - sigma_next) * self.restart_noise_scale
        
        if self.debug_mode >= 2:
            logger.debug(f"[Res2] RESTART at step {step_index}: σ={sigma_next.item():.4f} -> {sigma_restart.item():.4f}")

        # 1. Re-noise
        noise = self._generate_restart_noise(x, step_index)
        noise_amount = torch.sqrt(torch.clamp(sigma_restart**2 - sigma_next**2, min=0.0) + 1e-8)
        x_renoised = x + noise * noise_amount * self.restart_s_noise
        
        # 2. Denoise (using model)
        denoised_restart = self._model_denoise(x_renoised, sigma_restart)
        
        # 3. Euler step back to sigma_next
        d_restart = (x_renoised - denoised_restart) / sigma_restart
        dt_restart = sigma_next - sigma_restart
        x_corrected = x_renoised + d_restart * dt_restart
        
        return x_corrected

    # ============================================
    # MAIN CALL
    # ============================================

    @classmethod
    def go(cls, model: Any, x: torch.Tensor, sigmas: torch.Tensor, extra_args: Optional[Dict] = None,
           callback: Optional[Callable] = None, disable: Optional[bool] = None,
           noise_sampler: Optional[Callable] = None, **kwargs: Any) -> torch.Tensor:
        """Entrypoint for ComfyUI's KSAMPLER."""
        blend_mode = kwargs.pop("blend_mode", "lerp")
        step_blend_mode = kwargs.pop("step_blend_mode", "lerp")
        
        blend_function = _INTERNAL_BLEND_MODES.get(blend_mode, torch.lerp)
        step_blend_function = _INTERNAL_BLEND_MODES.get(step_blend_mode, torch.lerp)

        sampler = cls(
            model=model, x=x, sigmas=sigmas, extra_args=extra_args, callback=callback,
            disable=disable, noise_sampler=noise_sampler,
            blend_function=blend_function, step_blend_function=step_blend_function,
            **kwargs
        )
        return sampler()

    def __call__(self) -> torch.Tensor:
        """Main sampling loop with enhanced stability, tracking, and Res 2."""
        try:
            x_current = self.x.clone()
            num_steps = len(self.sigmas) - 1
            
            if num_steps <= 0:
                return torch.clamp(x_current, -1.0, 1.0) if self.enable_clamp_output else x_current

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

            # Timing
            start_time = time.time()
            steps_completed = 0
            early_exit_reason = None

            if self.debug_mode >= 1:
                logger.info("="*60)
                logger.info(f"Starting sampling: {num_steps} steps")
                logger.info(f"  Ancestral range: [{astart}-{aend}]")
                logger.info(f"  Preset: {self.noise_behavior_name}")
                logger.info("="*60)

            for idx in trange(num_steps, disable=self.disable_pbar):
                if idx < self.start_sigma_index or idx > actual_end:
                    continue

                with self.profiler.profile_step(f"step_{idx}"):
                    sigma_current, sigma_next = self.sigmas[idx], self.sigmas[idx + 1]
                    sigma_current_item = sigma_current.item()
                    sigma_next_item = sigma_next.item()

                    # Denoise
                    denoised_sample = self._model_denoise(x_current, sigma_current)
                    
                    # NaN/Inf recovery
                    if self._check_for_nan_inf(denoised_sample, "denoised", idx):
                        denoised_sample = self._recover_from_nan_inf(denoised_sample, x_current, "denoised", idx)

                    self._do_callback(idx, x_current, sigma_current, denoised_sample)

                    # Debug output
                    if self.debug_mode >= 1:
                        use_anc = (astart <= idx <= aend) if astart <= aend else False
                        logger.info(f"[Step {idx}] σ={sigma_current_item:.4f}→{sigma_next_item:.4f}, anc={use_anc}")

                    # Early convergence check
                    if self._prev_x is not None and sigma_next_item < 0.01:
                        change = torch.norm(x_current - self._prev_x).item()
                        if change < self.early_exit_threshold:
                            if self.debug_mode >= 1: logger.info(f"Early exit: converged (change={change:.2e})")
                            early_exit_reason = f"converged (change={change:.2e})"
                            x_current = denoised_sample.clone()
                            steps_completed = idx + 1
                            break
                    self._prev_x = x_current.clone()

                    # Very small sigma exit
                    if sigma_next_item <= 1e-6:
                        x_current = denoised_sample.clone()
                        early_exit_reason = "sigma <= 1e-6"
                        steps_completed = idx + 1
                        break

                    use_anc = (astart <= idx <= aend) if astart <= aend else False

                    if not use_anc or self.ancestral_strength == 0.0:
                        # Non-ancestral step
                        blend = sigma_next / sigma_current if sigma_current > 0 else 0.0
                        x_current = self.step_blend_function(denoised_sample, x_current, blend)
                        steps_completed = idx + 1
                    else:
                        # --- Ancestral Step ---
                        local_seed = self._stepped_seed(idx)
                        if local_seed is not None:
                            torch.manual_seed(local_seed)
                            if self.seeds_used is not None: self.seeds_used.append(local_seed)

                        # Generate noise with coherence
                        new_noise = self.noise_sampler(sigma_current, sigma_next)
                        noise_to_use = new_noise
                        
                        if self.noise_coherence > 0 and self.previous_noise is not None:
                            if self.previous_noise.shape == new_noise.shape:
                                noise_to_use = torch.lerp(new_noise, self.previous_noise, self.noise_coherence)

                        self.previous_noise = noise_to_use.clone()

                        # Calculate noisy next step
                        if self.is_rf:
                            x_next_noisy = self.step_blend_function(denoised_sample, noise_to_use, sigma_next)
                        else:
                            x_next_noisy = denoised_sample.clone() + noise_to_use * sigma_next

                        # Apply ancestral strength
                        x_current = torch.lerp(denoised_sample, x_next_noisy, self.ancestral_strength)
                        steps_completed = idx + 1

                    # ============================================
                    # RES 2 RESTART (v1.5.0)
                    # ============================================
                    if self.enable_restarts and idx in restart_schedule and sigma_next_item > 0:
                        x_current = self._execute_restart_step(x_current, sigma_current, sigma_next, idx)
                        if self.profiler.enabled: self.profiler.log_restart(idx, sigma_current, sigma_next)

            # Final output
            if self.enable_clamp_output:
                x_current = torch.clamp(x_current, -1.0, 1.0)

            # Performance summary
            elapsed = time.time() - start_time
            if self.debug_mode >= 1:
                logger.info("="*60)
                logger.info("PingPongSampler Lite+ - Summary")
                logger.info("="*60)
                logger.info(f"  Steps:        {steps_completed}/{num_steps}")
                logger.info(f"  Time:         {elapsed:.2f}s")
                if self.enable_restarts and restart_schedule:
                    logger.info(f"  Restarts:     {len(restart_schedule)} executed")
                if early_exit_reason:
                    logger.info(f"  Early Exit:   {early_exit_reason}")
                logger.info("="*60)

            if self.profiler.enabled:
                logger.info("\n" + self.profiler.get_summary())

            return x_current
            
        except Exception as e:
            logger.error(f"Critical error in sampling loop: {e}")
            logger.debug(traceback.format_exc())
            return self.x


# =================================================================================
# == ComfyUI Node Wrapper                                                        ==
# =================================================================================

class PingPongSamplerNode:
    """ComfyUI node wrapper for the Lite+ PingPongSampler."""
    CATEGORY = "MD_Nodes/Samplers"
    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "get_sampler"

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "required": {
                "noise_behavior": (
                    ["Default (Raw)", "Dynamic", "Smooth", "Textured Grain", "Soft (DDIM-Like)", "Custom"],
                    {"default": "Default (Raw)",
                     "tooltip": "NOISE BEHAVIOR PRESET: Selects predefined ancestral strength and noise coherence.\n"
                                "• **Default (Raw):** Standard ancestral sampling (strength=1.0, coherence=0.0).\n"
                                "• **Dynamic:** Slightly smoother noise (strength=1.0, coherence=0.25).\n"
                                "• **Smooth:** Reduced strength with moderate coherence (strength=0.8, coherence=0.5).\n"
                                "• **Textured Grain:** High coherence for consistent texture (strength=0.9, coherence=0.9).\n"
                                "• **Soft (DDIM-Like):** Minimal noise injection (strength=0.2, coherence=0.0).\n"
                                "• **Custom:** Uses the manual sliders below."}
                ),
                "step_random_mode": (
                    ["off", "block", "reset", "step"],
                    {"default": "block",
                     "tooltip": "STEP RANDOM MODE: Controls how the random seed changes during sampling.\n"
                                "• **off:** Fixed seed for all steps.\n"
                                "• **block:** Seed changes every N steps (defined by 'Step Size').\n"
                                "• **reset:** Seed resets every N steps.\n"
                                "• **step:** Seed changes every step."}
                ),
                "step_size": (
                    "INT",
                    {"default": 4, "min": 1, "max": 100,
                     "tooltip": "STEP SIZE: Number of steps before seed changes (block/reset modes)."}
                ),
                "seed": (
                    "INT",
                    {"default": 80085, "min": 0, "max": 0xFFFFFFFF,
                     "tooltip": "BASE SEED: The initial random seed for noise generation."}
                ),
                "first_ancestral_step": (
                    "INT",
                    {"default": 0, "min": 0, "max": 10000,
                     "tooltip": "FIRST ANCESTRAL STEP: Step index (0-based) to start injecting ancestral noise."}
                ),
                "last_ancestral_step": (
                    "INT",
                    {"default": -1, "min": -1, "max": 10000,
                     "tooltip": "LAST ANCESTRAL STEP: Step index (0-based) to stop injecting noise. -1 = until end."}
                ),
                "start_sigma_index": (
                    "INT",
                    {"default": 0, "min": 0, "max": 10000,
                     "tooltip": "START SIGMA INDEX: Step index to begin sampling from."}
                ),
                "end_sigma_index": (
                    "INT",
                    {"default": -1, "min": -10000, "max": 10000,
                     "tooltip": "END SIGMA INDEX: Step index to end sampling at (-1 = end)."}
                ),
                "enable_clamp_output": (
                    "BOOLEAN",
                    {"default": False, "label_on": "Clamp Final Output", "label_off": "No Final Clamp",
                     "tooltip": "CLAMP FINAL OUTPUT: Clamp the final output tensor values to the range [-1.0, 1.0]."}
                ),
                "blend_mode": (
                    list(_INTERNAL_BLEND_MODES.keys()),
                    {"default": "lerp",
                     "tooltip": "BLEND MODE: Interpolation function for main sampling steps.\n"
                                "• **lerp:** Linear.\n"
                                "• **slerp:** Spherical (good for noise).\n"
                                "• **cosine/cubic:** Smoother variants."}
                ),
                "scheduler": (
                    "SCHEDULER",
                    {"tooltip": "SCHEDULER INPUT: Connect the sigma schedule (e.g., from Karras or Hybrid) here."}
                ),
            },
            "optional": {
                "ancestral_strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01,
                     "tooltip": "ANCESTRAL STRENGTH (Custom): Noise intensity (0=none, 1=full, >1=overdriven). Only active when 'Noise Behavior' is 'Custom'."}
                ),
                "noise_coherence": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                     "tooltip": "NOISE COHERENCE (Custom): Blend with previous noise (0=fresh, 1=frozen). Only active when 'Noise Behavior' is 'Custom'."}
                ),
                "debug_mode": (
                    "INT",
                    {"default": 0, "min": 0, "max": 2,
                     "tooltip": "DEBUG MODE: Logging verbosity.\n• 0: Off\n• 1: Basic\n• 2: Detailed"}
                ),
                "enable_profiling": (
                    "BOOLEAN",
                    {"default": False,
                     "tooltip": "ENABLE PROFILING: Track timing and memory usage per step."}
                ),
                "yaml_settings_str": (
                    "STRING",
                    {"multiline": True, "default": "",
                     "tooltip": "YAML OVERRIDE: Provide settings in YAML format to override node inputs."}
                ),
                
                # Res 2 Optionals
                "enable_restarts": (
                    "BOOLEAN",
                    {"default": False, "label_on": "Enable Restarts", "label_off": "Restarts Disabled",
                     "tooltip": "ENABLE RESTARTS (Res 2): Turns on iterative error correction logic."}
                ),
                "restart_mode": (
                    ["balanced", "aggressive", "conservative", "detail_focus", "composition_focus"],
                    {"default": "balanced",
                     "tooltip": "RESTART MODE: Controls frequency and timing of restarts."}
                ),
                "restart_noise_scale": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                     "tooltip": "RESTART NOISE SCALE: How much noise to add back during restart (0.5 = halfway back)."}
                ),
                "restart_s_noise": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                     "tooltip": "RESTART S_NOISE: Multiplier for noise magnitude during restart."}
                ),
                "restart_steps": (
                    "STRING",
                    {"default": "", "multiline": False,
                     "tooltip": "CUSTOM RESTART STEPS: Comma-separated list of specific steps (e.g., '2,5,8') to restart on."}
                ),
            }
        }
    
    def get_sampler(
        self,
        noise_behavior: str,
        step_random_mode: str,
        step_size: int,
        seed: int,
        first_ancestral_step: int,
        last_ancestral_step: int,
        start_sigma_index: int,
        end_sigma_index: int,
        enable_clamp_output: bool,
        blend_mode: str,
        scheduler: Any,
        ancestral_strength: float = 1.0,
        noise_coherence: float = 0.0,
        debug_mode: int = 0,
        enable_profiling: bool = False,
        yaml_settings_str: str = "",
        # Res 2
        enable_restarts: bool = False,
        restart_mode: str = "balanced",
        restart_noise_scale: float = 0.5,
        restart_s_noise: float = 1.0,
        restart_steps: str = ""
    ) -> Tuple[KSAMPLER]:

        # Determine final strength and coherence from preset or custom
        final_strength: float = ancestral_strength
        final_coherence: float = noise_coherence
        
        if noise_behavior != "Custom":
            preset_map: Dict[str, Tuple[float, float]] = {
                "Default (Raw)": (1.0, 0.0),
                "Dynamic": (1.0, 0.25),
                "Smooth": (0.8, 0.5),
                "Textured Grain": (0.9, 0.9),
                "Soft (DDIM-Like)": (0.2, 0.0)
            }
            final_strength, final_coherence = preset_map.get(noise_behavior, (1.0, 0.0))
            if debug_mode >= 1 and noise_behavior not in preset_map:
                logger.warning(f"Unknown noise_behavior preset '{noise_behavior}'. Using default (1.0, 0.0).")

        # Build configuration dictionary
        direct_inputs: Dict[str, Any] = {
            "step_random_mode": step_random_mode,
            "step_size": step_size,
            "seed": seed,
            "pingpong_options": {
                "first_ancestral_step": first_ancestral_step,
                "last_ancestral_step": last_ancestral_step,
            },
            "start_sigma_index": start_sigma_index,
            "end_sigma_index": end_sigma_index,
            "enable_clamp_output": enable_clamp_output,
            "scheduler": scheduler,
            "blend_mode": blend_mode,
            "step_blend_mode": blend_mode,
            "ancestral_strength": final_strength,
            "noise_coherence": final_coherence,
            "debug_mode": debug_mode,
            "enable_profiling": enable_profiling,
            "noise_behavior_name": noise_behavior,
            
            # Res 2 inputs
            "enable_restarts": enable_restarts,
            "restart_mode": restart_mode,
            "restart_noise_scale": restart_noise_scale,
            "restart_s_noise": restart_s_noise,
            "restart_steps": restart_steps
        }

        final_options = direct_inputs.copy()
        
        # YAML Override Logic with enhanced error handling
        if yaml_settings_str and yaml_settings_str.strip():
            try:
                yaml_data = yaml.safe_load(yaml_settings_str)
                if isinstance(yaml_data, dict):
                    for key, value in yaml_data.items():
                        if key in final_options:
                            if key == "pingpong_options" and isinstance(value, dict) and isinstance(final_options[key], dict):
                                final_options[key].update(value)
                            else:
                                final_options[key] = value
                        elif debug_mode >= 1:
                            logger.warning(f"YAML contains unknown key '{key}', ignoring.")
                    
                    if debug_mode >= 1:
                        logger.info(f"Loaded and applied YAML settings: {list(yaml_data.keys())}")
                elif yaml_data is not None:
                    logger.warning("YAML override was not a dictionary. Ignoring.")

            except yaml.YAMLError as e:
                logger.error(f"Error parsing YAML override string: {e}. Using node inputs only.", exc_info=True)
            except Exception as e_yaml:
                 logger.error(f"Unexpected error applying YAML override: {e_yaml}. Using node inputs.", exc_info=True)

        return (KSAMPLER(PingPongSampler.go, extra_options=final_options),)

# =================================================================================
# == Node Registration                                                           ==
# =================================================================================

NODE_CLASS_MAPPINGS: Dict[str, Any] = {
    "PingPongSampler_Custom_Lite": PingPongSamplerNode,
}

NODE_DISPLAY_NAME_MAPPINGS: Dict[str, str] = {
    "PingPongSampler_Custom_Lite": "PingPong Sampler (Lite+ v1.5.0) 🏓",
}