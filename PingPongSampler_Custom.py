# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# ████ PINGPONGSAMPLER v0.8.20 "LITE+" – Optimized for Ace-Step with Behavior Control! ████▓▒░
# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀

# ░▒▓ ORIGIN & DEV:
#   • Foundational principles for iterative sampling, including concepts that underpin 'ping-pong sampling'
#   • Consistency Models by Song et al. (2023)
#   •     https://arxiv.org/abs/2303.01469
#   • The term 'ping-pong sampling' is explicitly introduced and applied in the context of fast text-to-audio
#   • generationin the paper "Fast Text-to-Audio Generation with Adversarial Post-Training" by Novack et al.
#   • (2025) from Stability AI
#   •     https://arxiv.org/abs/2505.08175
#   • original concept for the PingPong Sampler for ace-step diffusion by: Junmin Gong (Ace-Step team)
#   • ComfyUI adaptation by: blepping (original ComfyUI port with quirks)
#   • Disassembled & warped by: MD (Machine Damage)
#   • Critical fixes & re-engineering by: Gemini (Google) based on user feedback
#   • v0.8.20 "Lite+" enhancements by: Gemini (Google) based on user feedback
#   • Completionist fixups via: devstral / qwen3 (local heroes)
#   • License: Apache 2.0 — Sharing is caring, no Voodoo hoarding here
#   • Original source gist: https://gist.github.com/blepping/b372ef6c5412080af136aad942d9d76c

# ░▒▓ DESCRIPTION:
#   A "lite" but powerful sampler node for ComfyUI, tuned for Ace-Step
#   audio and video diffusion models. This version enhances the original raw signal
#   processing with intuitive "Noise Behavior" presets, allowing users to control
#   the texture and evolution of the ancestral noise without adding UI complexity.
#   It introduces Ancestral Strength and Noise Coherence to dial in effects
#   from chaotic and energetic to smooth and textured.

# ░▒▓ CHANGELOG HIGHLIGHTS:
#   - v0.8.15 “Targeted Ace-Step Optimization”:
#       • Full optimization declared for Ace-Step audio/video models
#       • Pure, raw noise injection for clean results.
#   - v0.8.20 “Lite+ with Behavior Control”:
#       • NEW: Added "Noise Behavior" presets (Default, Dynamic, Smooth, etc.) for intuitive control.
#       • NEW: Added "Ancestral Strength" to control the magnitude of injected noise (0.0 to 1.0+).
#       • NEW: Added "Noise Coherence" to control the evolution of noise over time, blending previous noise with new noise.
#       • All new features are fully backwards-compatible. "Default (Raw)" behavior is identical to v0.8.15.
#       • UI remains simple, with advanced controls available under a "Custom" preset.

# ░▒▓ CONFIGURATION:
#   → Primary Use: High-quality audio/video generation with Ace-Step diffusion
#   → Secondary Use: Experimental visual generation (expect wild results)
#   → Edge Use: For demoscene veterans and bytecode wizards only

# ░▒▓ WARNING:
#   This node may trigger:
#   ▓▒░ Flashbacks to ANSI art and 256-byte intros
#   ▓▒░ Spontaneous breakdancing or synesthesia
#   ▓▒░ Urges to reinstall Impulse Tracker
#   ▓▒░ Extreme focus on byte-level optimization
#   Consult your nearest demoscene vet if hallucinations persist.

# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄


import torch
from tqdm.auto import trange
from comfy import model_sampling
from comfy.samplers import KSAMPLER
import numpy as np
import yaml # YAML for handling optional string inputs

# --- Internal Blend Modes ---
_INTERNAL_BLEND_MODES = {
    "lerp": torch.lerp,
    "a_only": lambda a, _b, _t: a, # Returns the first input (a)
    "b_only": lambda _a, b, _t: b  # Returns the second input (b)
}

class PingPongSampler:
    """
    A custom sampler implementing a "ping-pong" ancestral noise mixing strategy.
    This "Lite+" version adds controls for noise strength and coherence.
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
        # All parameters are now passed via kwargs, to avoid duplicate arguments.
        **kwargs
    ):
        # Core inputs, directly from KSAMPLER
        self.model_ = model
        self.x = x
        self.sigmas = sigmas
        self.extra_args = extra_args.copy() if extra_args is not None else {}
        self.callback_ = callback
        self.disable_pbar = disable

        # Sampling control parameters
        self.start_sigma_index = kwargs.pop("start_sigma_index", 0)
        self.end_sigma_index = kwargs.pop("end_sigma_index", -1)
        self.enable_clamp_output = kwargs.pop("enable_clamp_output", False)

        # Noise injection and random seed controls
        self.step_random_mode = kwargs.pop("step_random_mode", "off")
        self.step_size = kwargs.pop("step_size", 5)
        self.seed = kwargs.pop("seed", 42)

        # --- NEW v0.8.20: Ancestral Noise Controls ---
        self.ancestral_strength = kwargs.pop("ancestral_strength", 1.0)
        self.noise_coherence = kwargs.pop("noise_coherence", 0.0)
        self.previous_noise = None # State for storing noise between steps for coherence

        # Blend functions
        self.blend_function = kwargs.pop("blend_function", torch.lerp)
        self.step_blend_function = kwargs.pop("step_blend_function", torch.lerp)

        # Ancestral (ping-pong) operation boundaries
        pingpong_options = kwargs.pop("pingpong_options", {})
        num_steps_available = len(sigmas) - 1 if len(sigmas) > 0 else 0
        raw_first_ancestral_step = pingpong_options.get("first_ancestral_step", 0)
        raw_last_ancestral_step = pingpong_options.get("last_ancestral_step", num_steps_available - 1)
        self.first_ancestral_step = max(0, min(raw_first_ancestral_step, raw_last_ancestral_step))
        if num_steps_available > 0:
            self.last_ancestral_step = min(num_steps_available - 1, max(raw_first_ancestral_step, raw_last_ancestral_step))
        else:
            self.last_ancestral_step = -1

        # Detect if model uses ComfyUI CONST sampling
        self.is_rf = False
        current_model_check = model
        try:
            while hasattr(current_model_check, 'inner_model') and current_model_check.inner_model is not None:
                current_model_check = current_model_check.inner_model
            if hasattr(current_model_check, 'model_sampling') and current_model_check.model_sampling is not None:
                self.is_rf = isinstance(current_model_check.model_sampling, model_sampling.CONST)
        except AttributeError:
            print("PingPongSampler Warning: Could not definitively determine model_sampling type, assuming not CONST.")
            self.is_rf = False

        # Default noise sampler returns raw, unconditioned unit variance noise.
        self.noise_sampler = noise_sampler
        if self.noise_sampler is None:
            def default_noise_sampler(sigma, sigma_next):
                return torch.randn_like(x)
            self.noise_sampler = default_noise_sampler

        # Note: self.noise_decay from scheduler is calculated but not used in this simplified version.
        scheduler = kwargs.pop("scheduler", None)
        if kwargs:
            print(f"PingPongSampler initialized with unused extra_options: {kwargs}")

    def _stepped_seed(self, step: int):
        """Determines the RNG seed for the current step based on the selected random mode."""
        if self.step_random_mode == "off":
            return None
        current_step_size = self.step_size if self.step_size > 0 else 1
        if self.step_random_mode == "block":
            return self.seed + (step // current_step_size)
        elif self.step_random_mode == "reset":
            return self.seed + (step * current_step_size)
        elif self.step_random_mode == "step":
            return self.seed + step
        else:
            return self.seed

    @classmethod
    def go(cls, model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, **kwargs):
        """Entrypoint for ComfyUI's KSAMPLER to initiate sampling."""
        # Resolve blend functions from strings
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
        """Wrapper around the underlying diffusion model's denoising function."""
        batch_size = x_tensor.shape[0]
        sigma_tensor = sigma_scalar * x_tensor.new_ones((batch_size,))
        final_extra_args = {**self.extra_args, **model_call_kwargs}
        return self.model_(x_tensor, sigma_tensor, **final_extra_args)

    def _do_callback(self, step_idx, current_x, current_sigma, denoised_sample):
        """Forwards progress information to ComfyUI's callback system."""
        if self.callback_:
            self.callback_({
                "i": step_idx, "x": current_x, "sigma": current_sigma,
                "sigma_hat": current_sigma, "denoised": denoised_sample
            })

    def __call__(self):
        """Executes the main "ping-pong" sampling loop."""
        x_current = self.x.clone()
        num_steps = len(self.sigmas) - 1
        if num_steps <= 0:
            return torch.clamp(x_current, -1.0, 1.0) if self.enable_clamp_output else x_current

        astart = self.first_ancestral_step
        aend = self.last_ancestral_step
        actual_iteration_end_idx = self.end_sigma_index if self.end_sigma_index >= 0 else num_steps - 1
        actual_iteration_end_idx = min(actual_iteration_end_idx, num_steps - 1)

        for idx in trange(num_steps, disable=self.disable_pbar):
            if idx < self.start_sigma_index or idx > actual_iteration_end_idx:
                continue

            sigma_current, sigma_next = self.sigmas[idx], self.sigmas[idx + 1]
            denoised_sample = self._model_denoise(x_current, sigma_current)
            self._do_callback(idx, x_current, sigma_current, denoised_sample)

            if sigma_next <= 1e-6:
                x_current = denoised_sample
                break

            use_anc = (astart <= idx <= aend) if astart <= aend else False

            if not use_anc or self.ancestral_strength == 0.0:
                # Non-Ancestral Step (or strength is zero)
                blend = sigma_next / sigma_current if sigma_current > 0 else 0.0
                x_current = self.step_blend_function(denoised_sample, x_current, blend)
                continue

            # --- Ancestral Step with Strength and Coherence ---
            local_seed = self._stepped_seed(idx)
            if local_seed is not None:
                torch.manual_seed(local_seed)

            # 1. Generate new noise and apply coherence
            new_noise = self.noise_sampler(sigma_current, sigma_next)
            noise_to_use = new_noise
            if self.noise_coherence > 0 and self.previous_noise is not None:
                if self.previous_noise.shape == new_noise.shape:
                    noise_to_use = torch.lerp(new_noise, self.previous_noise, self.noise_coherence)

            self.previous_noise = noise_to_use.clone() # Update state for next step

            # 2. Calculate the fully noise-injected next step
            if self.is_rf:
                x_next_with_full_noise = self.step_blend_function(denoised_sample, noise_to_use, sigma_next)
            else:
                x_next_with_full_noise = denoised_sample + noise_to_use * sigma_next

            # 3. Apply ancestral strength by blending between denoised and fully noisy next step
            x_current = torch.lerp(denoised_sample, x_next_with_full_noise, self.ancestral_strength)

        return torch.clamp(x_current, -1.0, 1.0) if self.enable_clamp_output else x_current

class PingPongSamplerNode:
    """ComfyUI node wrapper for the enhanced 'Lite+' PingPongSampler."""
    CATEGORY = "sampling/custom_sampling/samplers"
    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "get_sampler"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "noise_behavior": (
                    ["Default (Raw)", "Dynamic", "Smooth", "Textured Grain", "Soft (DDIM-Like)", "Custom"],
                    {"default": "Default (Raw)", "tooltip": "Select a preset for ancestral noise behavior. Overrides custom sliders unless set to 'Custom'."}
                ),
                "step_random_mode": (["off", "block", "reset", "step"], {"default": "block"}),
                "step_size": ("INT", {"default": 4, "min": 1, "max": 100}),
                "seed": ("INT", {"default": 80085, "min": 0, "max": 2**32 - 1}),
                "first_ancestral_step": ("INT", {"default": 0, "min": -1, "max": 10000}),
                "last_ancestral_step": ("INT", {"default": -1, "min": -1, "max": 10000}),
                "start_sigma_index": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "end_sigma_index": ("INT", {"default": -1, "min": -10000, "max": 10000}),
                "enable_clamp_output": ("BOOLEAN", {"default": False}),
                "scheduler": ("SCHEDULER",),
            },
            "optional": {
                "ancestral_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01, "tooltip": "Magnitude of injected noise. Only active if Noise Behavior is 'Custom'."}),
                "noise_coherence": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "How much of the previous step's noise to re-use. Only active if Noise Behavior is 'Custom'."}),
                "yaml_settings_str": ("STRING", {"multiline": True, "default": "", "tooltip": "YAML string to override node inputs. YAML takes priority."}),
            }
        }

    def get_sampler(
        self, noise_behavior, step_random_mode, step_size, seed,
        first_ancestral_step, last_ancestral_step, start_sigma_index, end_sigma_index,
        enable_clamp_output, scheduler, ancestral_strength, noise_coherence, yaml_settings_str=""
    ):
        # Determine final strength and coherence based on the preset
        strength = ancestral_strength
        coherence = noise_coherence
        if noise_behavior != "Custom":
            if noise_behavior == "Default (Raw)":
                strength, coherence = 1.0, 0.0
            elif noise_behavior == "Dynamic":
                strength, coherence = 1.0, 0.25
            elif noise_behavior == "Smooth":
                strength, coherence = 0.8, 0.5
            elif noise_behavior == "Textured Grain":
                strength, coherence = 0.9, 0.9
            elif noise_behavior == "Soft (DDIM-Like)":
                strength, coherence = 0.2, 0.0

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
            "blend_mode": "lerp", # Using default blend modes
            "step_blend_mode": "lerp",
            "ancestral_strength": strength,
            "noise_coherence": coherence,
        }

        final_options = direct_inputs.copy()
        if yaml_settings_str:
            try:
                yaml_data = yaml.safe_load(yaml_settings_str)
                if isinstance(yaml_data, dict):
                    # Simple key-value override from YAML
                    final_options.update(yaml_data)
            except Exception as e:
                print(f"WARNING: PingPongSamplerNode YAML parsing error: {e}. Using direct node inputs.")

        return (KSAMPLER(PingPongSampler.go, extra_options=final_options),)

NODE_CLASS_MAPPINGS = {
    "PingPongSampler_Custom_Lite": PingPongSamplerNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "PingPongSampler_Custom_Lite": "PingPong Sampler (Lite+ V0.8.20)",
}