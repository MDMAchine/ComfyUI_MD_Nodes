# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
# ████ MD_Nodes/PingPongSamplerNode – Lite+ Ancestral Sampler v0.8.22 ████▓▒░
# © 2025 MDMAchine
# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
# ░▒▓ ORIGIN & DEV:
#   • Cast into the void by: Junmin Gong (Concept), blepping (ComfyUI Port), MD (Adaptation)
#   • Enhanced by: Gemini, Claude, devstral/qwen3
#   • License: Apache 2.0 — Sharing is caring, no Voodoo hoarding here
#   • Original source: https://gist.github.com/blepping/b372ef6c5412080af136aad942d9d76c
#   • Academic sources: Song et al. (2023) [arXiv:2303.01469], Novack et al. (2025) [arXiv:2505.08175]

# ░▒▓ DESCRIPTION:
#   Lightweight ancestral sampler for ComfyUI with intuitive noise behavior control.
#   No complexity. No tensor corruption. Just clean ancestral sampling.
#   Designed for Ace-Step audio/video models with preset-based workflow.

# ░▒▓ FEATURES:
#   ✓ Intuitive "Noise Behavior" presets (Default, Dynamic, Smooth, etc.)
#   ✓ Ancestral Strength control (0.0 to 2.0+)
#   ✓ Noise Coherence for temporal smoothing
#   ✓ Enhanced blend modes with numerical stability (lerp, slerp, cosine, cubic)
#   ✓ Step-based or block-based seed randomization
#   ✓ Simple debug mode (0=Off, 1=Basic, 2=Detailed)
#   ✓ NaN/Inf detection with automatic recovery
#   ✓ Early convergence detection
#   ✓ Performance summary with timing stats
#   ✓ YAML configuration override support

# ░▒▓ CHANGELOG:
#   - v0.8.22 (Enhancement Update - Nov 2025):
#       • ADDED: Performance summary with timing stats (debug_mode >= 1)
#       • ADDED: NaN/Inf recovery (replaces corrupted values with previous state)
#       • ADDED: Early convergence detection (optional exit when stable)
#       • ADDED: Noise statistics logging (debug_mode >= 2)
#       • ADDED: Seed tracking for reproducibility debugging
#       • ADDED: Preset name in summary output
#       • FIXED: Potential duplicate kwarg bug in go() method
#       • ENHANCED: Cleaner debug output formatting
#   - v0.8.21 (Stability Update):
#       • CRITICAL: Ported tensor safety fixes from v0.9.9-p4
#       • ENHANCED: Added stable blend modes (slerp, cosine, cubic)
#   - v0.8.20 (Lite+ with Behavior Control):
#       • ADDED: "Noise Behavior" presets, "Ancestral Strength", "Noise Coherence"

# ░▒▓ CONFIGURATION:
#   → Primary Use: High-quality audio/video generation with Ace-Step using "Default (Raw)" or "Dynamic"
#   → Secondary Use: "Soft (DDIM-Like)" preset for smoother, less noisy results
#   → Edge Use: High "Noise Coherence" (0.8+) for temporally stable video/audio

# ░▒▓ WARNING:
#   This node may trigger:
#   ▓▒░ Existential dread when you realize you just NaN'd 9,999 steps into a 10k run.
#   ▓▒░ A sudden, uncontrollable urge to `slerp` everything in sight.
#   ▓▒░ Flashbacks to debugging `IRQ` conflicts on your Gravis Ultrasound.
#   ▓▒░ A chilling sense that `ancestral_strength=2.0` is staring back into your soul.
#   Consult your nearest demoscene vet if hallucinations persist.
# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄


# =================================================================================
# == Standard Library Imports                                                    ==
# =================================================================================
import math
import logging
import time
import traceback

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

# =================================================================================
# == Helper Functions                                                            ==
# =================================================================================

def slerp_lite(a, b, t):
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

def cosine_interpolation(a, b, t):
    """Cosine interpolation for smoother transitions."""
    cos_t = (1 - torch.cos(t * math.pi)) * 0.5
    return a * (1 - cos_t) + b * cos_t

def cubic_interpolation(a, b, t):
    """Cubic interpolation for even smoother transitions."""
    cubic_t = t * t * (3.0 - 2.0 * t)
    return torch.lerp(a, b, cubic_t)

_INTERNAL_BLEND_MODES = {
    "lerp": torch.lerp,
    "slerp": slerp_lite,
    "cosine": cosine_interpolation,
    "cubic": cubic_interpolation,
    "a_only": lambda a, _b, _t: a,
    "b_only": lambda _a, b, _t: b
}

# =================================================================================
# == Core Sampler Class                                                          ==
# =================================================================================

class PingPongSampler:
    """Lightweight sampler with ancestral noise control and enhanced stability."""
    
    def __init__(self, model, x, sigmas, extra_args=None, callback=None, disable=None,
                 noise_sampler=None, start_sigma_index=0, end_sigma_index=-1,
                 enable_clamp_output=False, step_random_mode="off", step_size=5, seed=42,
                 ancestral_strength=1.0, noise_coherence=0.0, debug_mode=0,
                 blend_function=None, step_blend_function=None, pingpong_options=None,
                 early_exit_threshold=1e-6, noise_behavior_name="Custom", **kwargs):
        
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
        self.previous_noise = None

        # Debug & tracking
        self.debug_mode = debug_mode
        self.noise_behavior_name = noise_behavior_name
        self.early_exit_threshold = early_exit_threshold
        self._prev_x = None
        self.seeds_used = [] if debug_mode >= 2 else None

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

        # Log unused kwargs
        if kwargs and self.debug_mode >= 1:
            print(f"[PingPongSampler Lite+] Unused options: {list(kwargs.keys())}")

    def _validate_sigma_schedule(self):
        """Validate sigma schedule is monotonically decreasing."""
        if len(self.sigmas) < 2:
            return True
        for i in range(len(self.sigmas) - 1):
            if self.sigmas[i] < self.sigmas[i + 1]:
                if self.debug_mode >= 1:
                    print(f"[PingPongSampler Lite+] Sigma not monotonic at {i}: {self.sigmas[i]:.6f} -> {self.sigmas[i+1]:.6f}")
                return False
        return True

    def _detect_model_type(self):
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

    def _check_for_nan_inf(self, tensor, name, step_idx):
        """Check tensor for NaN or Inf values."""
        has_nan = torch.isnan(tensor).any()
        has_inf = torch.isinf(tensor).any()
        if has_nan or has_inf:
            if self.debug_mode >= 1:
                print(f"[PingPongSampler Lite+] {'NaN' if has_nan else 'Inf'} in {name} at step {step_idx}")
            return True
        return False

    def _recover_from_nan_inf(self, tensor, fallback, name, step_idx):
        """Replace NaN/Inf values with fallback tensor values."""
        mask = torch.isnan(tensor) | torch.isinf(tensor)
        if mask.any():
            if self.debug_mode >= 1:
                print(f"[PingPongSampler Lite+] Recovering {name} at step {step_idx}")
            return torch.where(mask, fallback, tensor)
        return tensor

    def _stepped_seed(self, step):
        """Determines RNG seed for current step based on random mode."""
        if self.step_random_mode == "off":
            return None
        seed_map = {
            "block": self.seed + (step // self.step_size),
            "reset": self.seed + (step * self.step_size),
            "step": self.seed + step
        }
        return seed_map.get(self.step_random_mode, self.seed)

    def _model_denoise(self, x_tensor, sigma_scalar):
        """Wrapper around model's denoising function."""
        try:
            batch_size = x_tensor.shape[0]
            sigma_tensor = sigma_scalar * x_tensor.new_ones((batch_size,))
            return self.model_(x_tensor, sigma_tensor, **self.extra_args)
        except Exception as e:
            if self.debug_mode >= 1:
                print(f"[PingPongSampler Lite+] Denoise error: {e}")
            return x_tensor

    def _do_callback(self, step_idx, current_x, current_sigma, denoised_sample):
        """Forwards progress to ComfyUI's callback system."""
        if self.callback_:
            try:
                self.callback_({"i": step_idx, "x": current_x, "sigma": current_sigma,
                               "sigma_hat": current_sigma, "denoised": denoised_sample})
            except Exception as e:
                if self.debug_mode >= 1:
                    print(f"[PingPongSampler Lite+] Callback error: {e}")

    @classmethod
    def go(cls, model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, **kwargs):
        """Entrypoint for ComfyUI's KSAMPLER."""
        # Pop blend modes to avoid duplicate kwargs
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

    def __call__(self):
        """Main sampling loop with enhanced stability and tracking."""
        x_current = self.x.clone()
        num_steps = len(self.sigmas) - 1
        
        if num_steps <= 0:
            return torch.clamp(x_current, -1.0, 1.0) if self.enable_clamp_output else x_current

        astart, aend = self.first_ancestral_step, self.last_ancestral_step
        actual_end = min(self.end_sigma_index if self.end_sigma_index >= 0 else num_steps - 1, num_steps - 1)

        # Timing
        start_time = time.time()
        steps_completed = 0
        early_exit_reason = None

        if self.debug_mode >= 1:
            print(f"\n[PingPongSampler Lite+] Starting: {num_steps} steps, ancestral [{astart}-{aend}]")
            print(f"[PingPongSampler Lite+] Preset: {self.noise_behavior_name}, Strength: {self.ancestral_strength:.2f}, Coherence: {self.noise_coherence:.2f}")

        for idx in trange(num_steps, disable=self.disable_pbar):
            if idx < self.start_sigma_index or idx > actual_end:
                continue

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
                print(f"[Step {idx}] σ={sigma_current_item:.4f}→{sigma_next_item:.4f}, anc={use_anc}")
            
            if self.debug_mode >= 2:
                print(f"  X: min={x_current.min().item():.4f}, max={x_current.max().item():.4f}, "
                      f"mean={x_current.mean().item():.4f}, std={x_current.std().item():.4f}")

            # Early convergence check
            if self._prev_x is not None and sigma_next_item < 0.01:
                change = torch.norm(x_current - self._prev_x).item()
                if change < self.early_exit_threshold:
                    if self.debug_mode >= 1:
                        print(f"[PingPongSampler Lite+] Early exit: converged (change={change:.2e})")
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
                continue

            # --- Ancestral Step ---
            local_seed = self._stepped_seed(idx)
            if local_seed is not None:
                torch.manual_seed(local_seed)
                if self.seeds_used is not None:
                    self.seeds_used.append(local_seed)
                if self.debug_mode >= 2:
                    print(f"  Seed: {local_seed}")

            # Generate noise with coherence
            new_noise = self.noise_sampler(sigma_current, sigma_next)
            noise_to_use = new_noise
            
            if self.noise_coherence > 0 and self.previous_noise is not None:
                if self.previous_noise.shape == new_noise.shape:
                    noise_to_use = torch.lerp(new_noise, self.previous_noise, self.noise_coherence)

            self.previous_noise = noise_to_use.clone()

            # Noise stats (debug_mode >= 2)
            if self.debug_mode >= 2:
                print(f"  Noise: min={noise_to_use.min().item():.4f}, max={noise_to_use.max().item():.4f}, "
                      f"std={noise_to_use.std().item():.4f}")

            # Calculate noisy next step
            if self.is_rf:
                x_next_noisy = self.step_blend_function(denoised_sample, noise_to_use, sigma_next)
            else:
                x_next_noisy = denoised_sample.clone() + noise_to_use * sigma_next

            # Apply ancestral strength
            x_current = torch.lerp(denoised_sample, x_next_noisy, self.ancestral_strength)
            steps_completed = idx + 1

        # Final output
        if self.enable_clamp_output:
            x_current = torch.clamp(x_current, -1.0, 1.0)

        # Performance summary
        elapsed = time.time() - start_time
        if self.debug_mode >= 1:
            print(f"\n{'='*55}")
            print(f"  PingPongSampler Lite+ - Summary")
            print(f"{'='*55}")
            print(f"  Preset:      {self.noise_behavior_name}")
            print(f"  Steps:       {steps_completed}/{num_steps}")
            print(f"  Time:        {elapsed:.2f}s ({elapsed/max(steps_completed,1)*1000:.1f}ms/step)")
            print(f"  Ancestral:   [{astart}-{aend}]")
            print(f"  Strength:    {self.ancestral_strength:.2f}")
            print(f"  Coherence:   {self.noise_coherence:.2f}")
            if early_exit_reason:
                print(f"  Early Exit:  {early_exit_reason}")
            if self.seeds_used:
                print(f"  Seeds Used:  {len(self.seeds_used)}")
            print(f"  Output:      [{x_current.min().item():.4f}, {x_current.max().item():.4f}]")
            print(f"{'='*55}\n")

        return x_current


# =================================================================================
# == ComfyUI Node Wrapper                                                        ==
# =================================================================================

class PingPongSamplerNode:
    """ComfyUI node wrapper for the Lite+ PingPongSampler."""
    CATEGORY = "MD_Nodes/Sampling"
    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "get_sampler"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "noise_behavior": (
                    ["Default (Raw)", "Dynamic", "Smooth", "Textured Grain", "Soft (DDIM-Like)", "Custom"],
                    {"default": "Default (Raw)", 
                     "tooltip": (
                         "NOISE BEHAVIOR PRESET\n"
                         "- 'Default (Raw)': Pure ancestral noise (strength=1.0)\n"
                         "- 'Dynamic': Slight coherence for smoother results\n"
                         "- 'Smooth': Reduced strength with moderate coherence\n"
                         "- 'Textured Grain': High coherence for consistent texture\n"
                         "- 'Soft (DDIM-Like)': Minimal noise (strength=0.2)\n"
                         "- 'Custom': Use manual sliders below"
                     )}
                ),
                "step_random_mode": (
                    ["off", "block", "reset", "step"], 
                    {"default": "block",
                     "tooltip": (
                         "STEP RANDOM MODE\n"
                         "- 'off': Fixed seed for all steps\n"
                         "- 'block': Seed changes every N steps\n"
                         "- 'reset': Seed resets every N steps\n"
                         "- 'step': Seed changes every step"
                     )}
                ),
                "step_size": ("INT", {"default": 4, "min": 1, "max": 100,
                    "tooltip": "Number of steps before seed changes (block/reset modes)"}),
                "seed": ("INT", {"default": 80085, "min": 0, "max": 2**32 - 1,
                    "tooltip": "Base random seed for noise generation"}),
                "first_ancestral_step": ("INT", {"default": 0, "min": -1, "max": 10000,
                    "tooltip": "First step index to inject ancestral noise (-1 to disable)"}),
                "last_ancestral_step": ("INT", {"default": -1, "min": -1, "max": 10000,
                    "tooltip": "Last step index for ancestral noise (-1 = until end)"}),
                "start_sigma_index": ("INT", {"default": 0, "min": 0, "max": 10000,
                    "tooltip": "Step index to begin sampling from"}),
                "end_sigma_index": ("INT", {"default": -1, "min": -10000, "max": 10000,
                    "tooltip": "Step index to stop sampling at (-1 = end)"}),
                "enable_clamp_output": ("BOOLEAN", {"default": False,
                    "tooltip": "Clamp final output to [-1.0, 1.0] range"}),
                "blend_mode": (tuple(_INTERNAL_BLEND_MODES.keys()), {"default": "lerp",
                    "tooltip": (
                        "BLEND MODE\n"
                        "- 'lerp': Standard linear interpolation\n"
                        "- 'slerp': Spherical interpolation\n"
                        "- 'cosine'/'cubic': Smoother transitions"
                    )}),
                "scheduler": ("SCHEDULER", {"tooltip": "Sigma schedule from upstream node"}),
            },
            "optional": {
                "ancestral_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01,
                    "tooltip": "(Custom only) Noise intensity: 0=none, 1=full, >1=overdriven"}),
                "noise_coherence": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "(Custom only) Blend with previous noise: 0=fresh, 1=frozen"}),
                "debug_mode": ("INT", {"default": 0, "min": 0, "max": 2,
                    "tooltip": "Console verbosity: 0=Off, 1=Basic, 2=Detailed"}),
                "early_exit_threshold": ("FLOAT", {"default": 1e-6, "min": 1e-10, "max": 1e-2, "step": 1e-7,
                    "tooltip": "Convergence threshold for early exit (0 to disable)"}),
                "yaml_settings_str": ("STRING", {"multiline": True, "default": "",
                    "tooltip": "YAML override for any node input (takes priority)"}),
            }
        }

    def get_sampler(self, noise_behavior, step_random_mode, step_size, seed,
                    first_ancestral_step, last_ancestral_step, start_sigma_index, end_sigma_index,
                    enable_clamp_output, blend_mode, scheduler, 
                    ancestral_strength=1.0, noise_coherence=0.0, debug_mode=0,
                    early_exit_threshold=1e-6, yaml_settings_str=""):
        try:
            # Determine strength and coherence from preset
            strength, coherence = ancestral_strength, noise_coherence
            
            if noise_behavior != "Custom":
                preset_map = {
                    "Default (Raw)": (1.0, 0.0),
                    "Dynamic": (1.0, 0.25),
                    "Smooth": (0.8, 0.5),
                    "Textured Grain": (0.9, 0.9),
                    "Soft (DDIM-Like)": (0.2, 0.0)
                }
                strength, coherence = preset_map.get(noise_behavior, (1.0, 0.0))

            # Build configuration
            options = {
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
                "ancestral_strength": strength,
                "noise_coherence": coherence,
                "debug_mode": debug_mode,
                "early_exit_threshold": early_exit_threshold,
                "noise_behavior_name": noise_behavior,
            }

            # YAML override
            if yaml_settings_str:
                try:
                    yaml_data = yaml.safe_load(yaml_settings_str)
                    if isinstance(yaml_data, dict):
                        for key, value in yaml_data.items():
                            if key == "pingpong_options" and isinstance(value, dict):
                                options.setdefault("pingpong_options", {}).update(value)
                            else:
                                options[key] = value
                        if debug_mode >= 1:
                            print(f"[PingPongSampler Lite+] YAML loaded: {list(yaml_data.keys())}")
                except Exception as e:
                    print(f"[PingPongSampler Lite+] YAML error: {e}")

            return (KSAMPLER(PingPongSampler.go, extra_options=options),)
        
        except Exception as e:
            logging.error(f"[PingPongSampler Lite+] Error: {e}")
            logging.debug(traceback.format_exc())
            print(f"[PingPongSampler Lite+] Error creating sampler: {e}")
            return (KSAMPLER(lambda model, x, sigmas, **kw: x, extra_options={}),)


# =================================================================================
# == Node Registration                                                           ==
# =================================================================================

NODE_CLASS_MAPPINGS = {
    "PingPongSampler_Custom_Lite": PingPongSamplerNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PingPongSampler_Custom_Lite": "MD: PingPong Sampler (Lite+)",
}