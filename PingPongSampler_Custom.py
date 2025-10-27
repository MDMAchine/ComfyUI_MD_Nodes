# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
# ████ MD_Nodes/PingPongSamplerNode – Lite+ Ancestral Sampler v0.8.21 ████▓▒░
# © 2025 MDMAchine
# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
# ░▒▓ ORIGIN & DEV:
#   • Cast into the void by: Junmin Gong (Concept), blepping (ComfyUI Port), MD (Adaptation)
#   • Enhanced by: Gemini (Fixes/v0.8.20), Claude (v0.8.21 Port), devstral/qwen3 (Fixups)
#   • License: Apache 2.0 — Sharing is caring, no Voodoo hoarding here
#   • Original source (if applicable): https://gist.github.com/blepping/b372ef6c5412080af136aad942d9d76c
#   • Academic sources: Song et al. (2023) [arXiv:2303.01469], Novack et al. (2025) [arXiv:2505.08175]

# ░▒▓ DESCRIPTION:
#   Lightweight ancestral sampler for ComfyUI with intuitive noise behavior control.
#   No complexity. No tensor corruption. Just clean ancestral sampling.
#   Designed for Ace-Step audio/video models with preset-based workflow.
#   May work with other models but audio/video is the sweet spot.

# ░▒▓ FEATURES:
#   ✓ Intuitive "Noise Behavior" presets (Default, Dynamic, Smooth, etc.)
#   ✓ Ancestral Strength control (0.0 to 2.0+)
#   ✓ Noise Coherence for temporal smoothing
#   ✓ Enhanced blend modes with numerical stability (lerp, slerp, cosine, cubic)
#   ✓ Step-based or block-based seed randomization
#   ✓ Simple debug mode (0=Off, 1=Basic, 2=Detailed)
#   ✓ NaN/Inf detection with automatic recovery
#   ✓ Parameter validation to prevent crashes
#   ✓ Safe tensor operations (no corruption bugs)
#   ✓ Sigma schedule validation
#   ✓ YAML configuration override support
#   ✓ Supports up to 10,000 steps
#   ✓ Backwards compatible with v0.8.20 workflows

# ░▒▓ CHANGELOG:
#   - v0.8.21 (Current Release - STABILITY UPDATE):
#       • CRITICAL: Ported tensor safety fixes from v0.9.9-p4 & added NaN/Inf detection/recovery.
#       • ENHANCED: Added stable blend modes (slerp, cosine, cubic) & parameter/sigma validation.
#       • ADDED: Simple debug mode (0/1/2) & safer tensor operations (cloning).
#   - v0.8.20 ("Lite+ with Behavior Control"):
#       • ADDED: "Noise Behavior" presets, "Ancestral Strength", and "Noise Coherence" controls.
#   - v0.8.15 ("Targeted Ace-Step Optimization"):
#       • ADDED: Full optimization for Ace-Step models, pure raw noise injection.

# ░▒▓ CONFIGURATION:
#   → Primary Use: High-quality audio/video generation with Ace-Step diffusion using "Default (Raw)" or "Dynamic" behavior.
#   → Secondary Use: "Soft (DDIM-Like)" preset for smoother, less noisy results.
#   → Edge Use: High "Noise Coherence" (0.8+) for temporally stable video/audio generation.

# ░▒▓ WARNING:
#   This node may trigger:
#   ▓▒░ Existential dread when you realize you just NaN'd 9,999 steps into a 10k run.
#   ▓▒░ A sudden, uncontrollable urge to `slerp` everything in sight.
#   ▓▒░ Flashbacks to debugging `IRQ` conflicts trying to get your Gravis Ultrasound and modem to coexist.
#   ▓▒░ A chilling sense that `ancestral_strength=2.0` is staring back into your soul.
#   Consult your nearest demoscene vet if hallucinations persist.
# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄


# =================================================================================
# == Standard Library Imports                                                  ==
# =================================================================================
import math
import logging
import secrets
import os
import traceback

# =================================================================================
# == Third-Party Imports                                                       ==
# =================================================================================
import torch
from tqdm.auto import trange
import numpy as np
import yaml

# =================================================================================
# == ComfyUI Core Modules                                                      ==
# =================================================================================
from comfy import model_sampling
from comfy.samplers import KSAMPLER

# =================================================================================
# == Local Project Imports                                                     ==
# =================================================================================
# (None)

# =================================================================================
# == Helper Classes & Dependencies                                             ==
# =================================================================================

# --- Enhanced Blend Modes with Numerical Stability ---

def slerp_lite(a, b, t):
    """Spherical linear interpolation with numerical stability for lite version."""
    eps = 1e-8
    
    # Normalize to unit sphere
    a_norm = a / (torch.norm(a, dim=-1, keepdim=True) + eps)
    b_norm = b / (torch.norm(b, dim=-1, keepdim=True) + eps)
    
    # Calculate dot product with clamping
    dot = torch.sum(a_norm * b_norm, dim=-1, keepdim=True).clamp(-0.9999, 0.9999)
    
    # Fall back to lerp for nearly parallel vectors
    if torch.all(torch.abs(dot) > 0.9995):
        return torch.lerp(a, b, t)
    
    # Slerp formula
    theta = torch.acos(torch.abs(dot)) * t
    c = b_norm - a_norm * dot
    c_norm = c / (torch.norm(c, dim=-1, keepdim=True) + eps)
    
    result = a_norm * torch.cos(theta) + c_norm * torch.sin(theta)
    
    # Scale to average magnitude
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
# == Core Node Class                                                           ==
# =================================================================================

class PingPongSampler:
    """
    Lightweight sampler with ancestral noise control and enhanced stability.
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
        **kwargs
    ):
        # Core inputs
        self.model_ = model
        self.x = x
        self.sigmas = sigmas
        self.extra_args = extra_args.copy() if extra_args is not None else {}
        self.callback_ = callback
        self.disable_pbar = disable

        # Sampling control
        self.start_sigma_index = kwargs.pop("start_sigma_index", 0)
        self.end_sigma_index = kwargs.pop("end_sigma_index", -1)
        self.enable_clamp_output = kwargs.pop("enable_clamp_output", False)

        # Noise injection controls
        self.step_random_mode = kwargs.pop("step_random_mode", "off")
        self.step_size = kwargs.pop("step_size", 5)
        self.seed = kwargs.pop("seed", 42)

        # Ancestral noise controls with validation
        self.ancestral_strength = max(0.0, kwargs.pop("ancestral_strength", 1.0))
        self.noise_coherence = max(0.0, min(1.0, kwargs.pop("noise_coherence", 0.0)))
        self.previous_noise = None

        # Debug mode
        self.debug_mode = kwargs.pop("debug_mode", 0)

        # Blend functions
        self.blend_function = kwargs.pop("blend_function", torch.lerp)
        self.step_blend_function = kwargs.pop("step_blend_function", torch.lerp)

        # Validate parameters
        if self.ancestral_strength < 0:
            if self.debug_mode >= 1:
                print("Warning: ancestral_strength < 0, clamping to 0")
            self.ancestral_strength = 0.0
        
        if self.noise_coherence < 0 or self.noise_coherence > 1:
            if self.debug_mode >= 1:
                print(f"Warning: noise_coherence {self.noise_coherence} out of range, clamping to [0,1]")
            self.noise_coherence = max(0.0, min(1.0, self.noise_coherence))

        # Ancestral operation boundaries
        pingpong_options = kwargs.pop("pingpong_options", {})
        num_steps_available = len(sigmas) - 1 if len(sigmas) > 0 else 0
        
        if self.debug_mode >= 1:
            # Removed version number
            print(f"PingPongSampler Lite+: {num_steps_available} steps available")
        
        raw_first_ancestral_step = pingpong_options.get("first_ancestral_step", 0)
        raw_last_ancestral_step = pingpong_options.get("last_ancestral_step", num_steps_available - 1)
        self.first_ancestral_step = max(0, min(raw_first_ancestral_step, raw_last_ancestral_step))
        
        if num_steps_available > 0:
            self.last_ancestral_step = min(num_steps_available - 1, max(raw_first_ancestral_step, raw_last_ancestral_step))
        else:
            self.last_ancestral_step = -1

        # Validate sigma schedule
        self._validate_sigma_schedule()

        # Detect model type
        self.is_rf = False
        current_model_check = model
        try:
            while hasattr(current_model_check, 'inner_model') and current_model_check.inner_model is not None:
                current_model_check = current_model_check.inner_model
            if hasattr(current_model_check, 'model_sampling') and current_model_check.model_sampling is not None:
                self.is_rf = isinstance(current_model_check.model_sampling, model_sampling.CONST)
        except (AttributeError, TypeError) as e:
            if self.debug_mode >= 1:
                print(f"Model type detection failed: {e}. Assuming non-CONST sampling.")
            self.is_rf = False

        # Noise sampler
        self.noise_sampler = noise_sampler
        if self.noise_sampler is None:
            def default_noise_sampler(sigma, sigma_next):
                return torch.randn_like(x)
            self.noise_sampler = default_noise_sampler

        if kwargs and self.debug_mode >= 1:
            print(f"PingPongSampler initialized with unused options: {kwargs}")

    def _validate_sigma_schedule(self):
        """Validate that sigma schedule is monotonically decreasing."""
        if len(self.sigmas) < 2:
            return True
        
        for i in range(len(self.sigmas) - 1):
            if self.sigmas[i] < self.sigmas[i + 1]:
                if self.debug_mode >= 1:
                    print(f"⚠️  WARNING: Sigma schedule not monotonic at step {i}: "
                          f"{self.sigmas[i]:.6f} -> {self.sigmas[i+1]:.6f}")
                return False
        return True

    def _check_for_nan_inf(self, tensor, name, step_idx):
        """Check tensor for NaN or Inf values."""
        has_nan = torch.isnan(tensor).any()
        has_inf = torch.isinf(tensor).any()
        
        if has_nan or has_inf:
            if self.debug_mode >= 1:
                print(f"⚠️  WARNING: {'NaN' if has_nan else 'Inf'} detected in {name} at step {step_idx}!")
            return True
        return False

    def _stepped_seed(self, step):
        """
        Determines RNG seed for current step based on random mode.
        REMOVED type hint from `step`
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

    @classmethod
    def go(cls, model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, **kwargs):
        """Entrypoint for ComfyUI's KSAMPLER."""
        # Resolve blend functions
        blend_mode = kwargs.pop("blend_mode", "lerp")
        step_blend_mode = kwargs.pop("step_blend_mode", "lerp")
        resolved_blend_function = _INTERNAL_BLEND_MODES.get(blend_mode, torch.lerp)
        resolved_step_blend_function = _INTERNAL_BLEND_MODES.get(step_blend_mode, torch.lerp)

        sampler_instance = cls(
            model=model, x=x, sigmas=sigmas, extra_args=extra_args, callback=callback,
            disable=disable, noise_sampler=noise_sampler,
            blend_function=resolved_blend_function,
            step_blend_function=resolved_step_blend_function,
            **kwargs
        )
        return sampler_instance()

    def _model_denoise(self, x_tensor, sigma_scalar, **model_call_kwargs):
        """Wrapper around model's denoising function with error handling."""
        try:
            batch_size = x_tensor.shape[0]
            sigma_tensor = sigma_scalar * x_tensor.new_ones((batch_size,))
            final_extra_args = {**self.extra_args, **model_call_kwargs}
            return self.model_(x_tensor, sigma_tensor, **final_extra_args)
        except Exception as e:
            if self.debug_mode >= 1:
                print(f"Model denoising error: {e}. Returning input as fallback.")
            return x_tensor

    def _do_callback(self, step_idx, current_x, current_sigma, denoised_sample):
        """Forwards progress to ComfyUI's callback system."""
        if self.callback_:
            try:
                self.callback_({
                    "i": step_idx, "x": current_x, "sigma": current_sigma,
                    "sigma_hat": current_sigma, "denoised": denoised_sample
                })
            except Exception as e:
                if self.debug_mode >= 1:
                    print(f"Callback error at step {step_idx}: {e}")

    def __call__(self):
        """Main sampling loop with enhanced stability."""
        x_current = self.x.clone()
        num_steps = len(self.sigmas) - 1
        
        if num_steps <= 0:
            return torch.clamp(x_current, -1.0, 1.0) if self.enable_clamp_output else x_current

        astart = self.first_ancestral_step
        aend = self.last_ancestral_step
        actual_iteration_end_idx = self.end_sigma_index if self.end_sigma_index >= 0 else num_steps - 1
        actual_iteration_end_idx = min(actual_iteration_end_idx, num_steps - 1)

        if self.debug_mode >= 1:
            print(f"Starting sampling: {num_steps} steps, ancestral range: {astart}-{aend}")

        for idx in trange(num_steps, disable=self.disable_pbar):
            if idx < self.start_sigma_index or idx > actual_iteration_end_idx:
                continue

            sigma_current, sigma_next = self.sigmas[idx], self.sigmas[idx + 1]
            sigma_current_item = sigma_current.item()
            sigma_next_item = sigma_next.item()

            # Denoise with NaN/Inf check
            denoised_sample = self._model_denoise(x_current, sigma_current)
            
            if self._check_for_nan_inf(denoised_sample, "denoised_sample", idx):
                if self.debug_mode >= 1:
                    print(f"Recovering from NaN/Inf at step {idx} by keeping previous state")
                continue  # Skip this corrupted step

            self._do_callback(idx, x_current, sigma_current, denoised_sample)

            # Debug output
            if self.debug_mode >= 1:
                use_anc_debug = (astart <= idx <= aend) if astart <= aend else False
                print(f"Step {idx}: σ={sigma_current_item:.3f}→{sigma_next_item:.3f}, "
                      f"ancestral={use_anc_debug}, strength={self.ancestral_strength:.2f}")
            
            if self.debug_mode >= 2:
                print(f"  X stats: min={x_current.min().item():.4f}, "
                      f"max={x_current.max().item():.4f}, "
                      f"mean={x_current.mean().item():.4f}")

            # Early exit for very small sigma
            if sigma_next_item <= 1e-6:
                x_current = denoised_sample.clone()
                break

            use_anc = (astart <= idx <= aend) if astart <= aend else False

            if not use_anc or self.ancestral_strength == 0.0:
                # Non-ancestral step or zero strength
                blend = sigma_next / sigma_current if sigma_current > 0 else 0.0
                x_current = self.step_blend_function(denoised_sample, x_current, blend)
                continue

            # --- Ancestral Step with Enhanced Safety ---
            local_seed = self._stepped_seed(idx)
            if local_seed is not None:
                torch.manual_seed(local_seed)
                if self.debug_mode >= 2:
                    print(f"  Using seed: {local_seed}")

            # Generate new noise with coherence
            new_noise = self.noise_sampler(sigma_current, sigma_next)
            noise_to_use = new_noise
            
            if self.noise_coherence > 0 and self.previous_noise is not None:
                if self.previous_noise.shape == new_noise.shape:
                    noise_to_use = torch.lerp(new_noise, self.previous_noise, self.noise_coherence)

            self.previous_noise = noise_to_use.clone()

            # Calculate fully noisy next step (SAFE operations)
            if self.is_rf:
                x_next_with_full_noise = self.step_blend_function(denoised_sample, noise_to_use, sigma_next)
            else:
                # Explicit tensor operations for safety
                noise_contribution = noise_to_use * sigma_next
                x_next_with_full_noise = denoised_sample.clone() + noise_contribution

            # Apply ancestral strength
            x_current = torch.lerp(denoised_sample, x_next_with_full_noise, self.ancestral_strength)

        # Final output
        if self.enable_clamp_output:
            x_current = torch.clamp(x_current, -1.0, 1.0)

        if self.debug_mode >= 1:
            print(f"Sampling complete. Final range: [{x_current.min().item():.4f}, {x_current.max().item():.4f}]")

        return x_current

class PingPongSamplerNode:
    """ComfyUI node wrapper for the Lite+ PingPongSampler."""
    CATEGORY = "MD_Nodes/Sampling"
    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "get_sampler"

    @classmethod
    def INPUT_TYPES(cls):
        """
        Define all input parameters with tooltips.
        
        Note: DO NOT use type hints in function signatures or global mappings.
        ComfyUI's dynamic loader cannot handle forward references at import time.
        """
        return {
            "required": {
                "noise_behavior": (
                    ["Default (Raw)", "Dynamic", "Smooth", "Textured Grain", "Soft (DDIM-Like)", "Custom"],
                    {"default": "Default (Raw)", 
                     "tooltip": (
                         "NOISE BEHAVIOR PRESET\n"
                         "- Selects a preset for ancestral noise strength and coherence.\n"
                         "- 'Default (Raw)' is pure ancestral noise (strength=1.0).\n"
                         "- 'Soft (DDIM-Like)' is weak noise (strength=0.2).\n"
                         "- 'Custom' uses the manual sliders below."
                     )}
                ),
                "step_random_mode": (
                    ["off", "block", "reset", "step"], 
                    {"default": "block",
                     "tooltip": (
                         "STEP RANDOM MODE\n"
                         "- Controls how the seed changes during sampling.\n"
                         "- 'off': Fixed seed for all steps.\n"
                         "- 'block': Seed changes every N steps (see step_size).\n"
                         "- 'step': Seed changes every single step."
                     )}
                ),
                "step_size": (
                    "INT", 
                    {"default": 4, "min": 1, "max": 100,
                     "tooltip": (
                         "STEP SIZE\n"
                         "- Number of steps to wait before changing the seed.\n"
                         "- Only used if 'step_random_mode' is 'block' or 'reset'."
                     )}
                ),
                "seed": (
                    "INT", 
                    {"default": 80085, "min": 0, "max": 2**32 - 1,
                     "tooltip": (
                         "SEED\n"
                         "- The base random seed for noise generation.\n"
                         "- This seed is modified by the 'step_random_mode'."
                     )}
                ),
                "first_ancestral_step": (
                    "INT", 
                    {"default": 0, "min": -1, "max": 10000,
                     "tooltip": (
                         "FIRST ANCESTRAL STEP\n"
                         "- The first step index (0-based) to begin injecting ancestral noise.\n"
                         "- Steps before this will be non-ancestral."
                     )}
                ),
                "last_ancestral_step": (
                    "INT", 
                    {"default": -1, "min": -1, "max": 10000,
                     "tooltip": (
                         "LAST ANCESTRAL STEP\n"
                         "- The last step index (0-based) to inject ancestral noise.\n"
                         "- -1 = inject noise until the very last step.\n"
                         "- Set to 0 to only inject noise on the first step."
                     )}
                ),
                "start_sigma_index": (
                    "INT", 
                    {"default": 0, "min": 0, "max": 10000,
                     "tooltip": (
                         "START SIGMA INDEX\n"
                         "- The step index (0-based) to *begin* the sampling process from.\n"
                         "- Allows skipping the initial high-noise steps."
                     )}
                ),
                "end_sigma_index": (
                    "INT", 
                    {"default": -1, "min": -10000, "max": 10000,
                     "tooltip": (
                         "END SIGMA INDEX\n"
                         "- The step index (0-based) to *stop* the sampling process at.\n"
                         "- -1 = run until the end of the sigma schedule."
                     )}
                ),
                "enable_clamp_output": (
                    "BOOLEAN", 
                    {"default": False,
                     "tooltip": (
                         "CLAMP FINAL OUTPUT\n"
                         "- True: Clamps the final output tensor to the [-1.0, 1.0] range.\n"
                         "- False: Leaves the output as-is.\n"
                         "- Recommended: False (off) for audio/video latents."
                     )}
                ),
                "blend_mode": (
                    tuple(_INTERNAL_BLEND_MODES.keys()),
                    {"default": "lerp",
                     "tooltip": (
                         "BLEND MODE\n"
                         "- The mathematical function used to blend states between steps.\n"
                         "- 'lerp': Standard linear interpolation.\n"
                         "- 'slerp': Spherical interpolation (good for latents).\n"
                         "- 'cosine' / 'cubic': Smoother transitions."
                     )}
                ),
                "scheduler": (
                    "SCHEDULER",
                    {"tooltip": (
                        "SCHEDULER\n"
                        "- The sigma schedule object from an upstream node (e.g., HybridAdaptiveSigmas)."
                    )}
                ),
            },
            "optional": {
                "ancestral_strength": (
                    "FLOAT", 
                    {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01, 
                     "tooltip": (
                         "CUSTOM: ANCESTRAL STRENGTH\n"
                         "- (Custom Mode Only) Controls the intensity of injected noise.\n"
                         "- 0.0 = No noise (pure DDIM).\n"
                         "- 1.0 = Full ancestral noise.\n"
                         "- > 1.0 = Over-driven noise."
                     )}
                ),
                "noise_coherence": (
                    "FLOAT", 
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, 
                     "tooltip": (
                         "CUSTOM: NOISE COHERENCE\n"
                         "- (Custom Mode Only) Blends new noise with the previous step's noise.\n"
                         "- 0.0 = New random noise every step.\n"
                         "- 1.0 = Re-use the same noise (frozen).\n"
                         "- High values improve temporal stability in video/audio."
                     )}
                ),
                "debug_mode": (
                    "INT",
                    {"default": 0, "min": 0, "max": 2,
                     "tooltip": (
                         "DEBUG MODE\n"
                         "- Controls console print verbosity.\n"
                         "- 0: Off\n- 1: Basic (step info, warnings)\n- 2: Detailed (tensor stats, seeds)"
                     )}
                ),
                "yaml_settings_str": (
                    "STRING", 
                    {"multiline": True, "default": "", 
                     "tooltip": (
                         "YAML OVERRIDE\n"
                         "- YAML-formatted string to override any node input.\n"
                         "- This takes *priority* over all UI settings.\n"
                         "- Example: `ancestral_strength: 1.2`"
                     )}
                ),
            }
        }

    def get_sampler(
        self, noise_behavior, step_random_mode, step_size, seed,
        first_ancestral_step, last_ancestral_step, start_sigma_index, end_sigma_index,
        enable_clamp_output, blend_mode, scheduler, 
        ancestral_strength=1.0, noise_coherence=0.0, debug_mode=0, yaml_settings_str=""
    ):
        """
        Main execution function.
        
        Args:
            (All args match INPUT_TYPES)
            
        Returns:
            Tuple containing the KSAMPLER object
        """
        try:
            # Determine final strength and coherence from preset or custom
            strength = ancestral_strength
            coherence = noise_coherence
            
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
            direct_inputs = {
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
                "step_blend_mode": blend_mode,  # Use same blend for steps
                "ancestral_strength": strength,
                "noise_coherence": coherence,
                "debug_mode": debug_mode,
            }

            final_options = direct_inputs.copy()
            
            # YAML override
            if yaml_settings_str:
                try:
                    yaml_data = yaml.safe_load(yaml_settings_str)
                    if isinstance(yaml_data, dict):
                        for key, value in yaml_data.items():
                            if key == "pingpong_options" and isinstance(value, dict):
                                if key not in final_options:
                                    final_options[key] = {}
                                final_options[key].update(value)
                            else:
                                final_options[key] = value
                        if debug_mode >= 1:
                            print(f"Loaded YAML settings: {list(yaml_data.keys())}")
                except Exception as e:
                    print(f"WARNING: PingPongSamplerNode YAML parsing error: {e}. Using node inputs.")

            # Return the KSAMPLER tuple
            return (KSAMPLER(PingPongSampler.go, extra_options=final_options),)
        
        except Exception as e:
            logging.error(f"[PingPongSamplerNode] Error creating sampler: {e}")
            logging.debug(traceback.format_exc())
            print(f"[PingPongSamplerNode] ⚠️ Error creating sampler, returning dummy sampler: {e}")
            # Return a dummy/passthrough sampler on failure
            return (KSAMPLER(lambda model, x, sigmas, **kwargs: x, extra_options={}),)

# =================================================================================
# == Node Registration                                                         ==
# =================================================================================

NODE_CLASS_MAPPINGS = {
    "PingPongSampler_Custom_Lite": PingPongSamplerNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PingPongSampler_Custom_Lite": "MD: PingPong Sampler (Lite+)",
}