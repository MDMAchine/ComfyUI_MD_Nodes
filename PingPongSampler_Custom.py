# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# ████ PINGPONGSAMPLER v0.8.15 – Optimized for Ace-Step Audio/Video! ████▓▒░
# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀

# ░▒▓ ORIGIN & DEV:
#   • Foundational principles for iterative sampling, including concepts that underpin 'ping-pong sampling'
#   • Consistency Models by Song et al. (2023)
#   •      https://arxiv.org/abs/2303.01469
#   • The term 'ping-pong sampling' is explicitly introduced and applied in the context of fast text-to-audio
#   • generationin the paper "Fast Text-to-Audio Generation with Adversarial Post-Training" by Novack et al.
#   • (2025) from Stability AI
#   •      https://arxiv.org/abs/2505.08175
#   • original concept for the PingPong Sampler for ace-step diffusion by: Junmin Gong (Ace-Step team)
#   • ComfyUI adaptation by: blepping (original ComfyUI port with quirks)
#   • Disassembled & warped by: MD (Machine Damage)
#   • Critical fixes & re-engineering by: Gemini (Google) based on user feedback
#   • Completionist fixups via: devstral / qwen3 (local heroes)
#   • License: Apache 2.0 — Sharing is caring, no Voodoo hoarding here
#   • Original source gist: https://gist.github.com/blepping/b372ef6c5412080af136aad942d9d76c

# ░▒▓ DESCRIPTION:
#   A sampler node for ComfyUI, tuned specifically for Ace-Step
#   audio and video diffusion models. Embraces noise amplitude like
#   a tracker module loves a breakbeat. Maintains ancestral noise
#   sync with the original blepping sampler while ditching parameters
#   that muddy the raw signal. May work with image models but results
#   can vary wildly — consider it your party-ready audio/video workhorse,
#   not your Instagram selfie filter.
#   Warning: May induce flashbacks to 256-byte intros or
#   compulsive byte-optimization urges. Use responsibly.

# ░▒▓ CHANGELOG HIGHLIGHTS:
#   - v0.5 “The Wobbly One”:
#       • Noise decay & stepped randomness introduced
#       • Removed wave shape selection (square reigns)
#   - v0.8 “Consolidated Weirdness”:
#       • Scheduler & tempo sync integration
#       • Step mode per-frame seed variation
#       • Output clamping, sigma skipping, and is_rf detection added
#   - v0.8.1 “Modem Memories”:
#       • Audio_type presets & dynamic noise control debuted
#   - v0.8.11 “Strict Ace-Step Mode”:
#       • Removed s_noise and dynamic_noise for ancestral steps
#       • Pure raw noise injection, no fluff
#   - v0.8.13 “YAML is Boss”:
#       • YAML overrides take priority over node inputs
#   - v0.8.15 “Targeted Ace-Step Optimization”:
#       • Full optimization declared for Ace-Step audio/video models
#       • Use with non-Ace-Step image models with caution

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
    Steps through provided noise levels (sigmas) and alternates between denoising
    and injecting controlled noise according to scheduler or manual settings.
    This version synchronizes noise application with the original `blepping` sampler.
    """

    CATEGORY = "sampling/custom_sampling" # This is for internal categorization, not node UI
    FUNCTION = "go" # This is the main function that KSAMPLER will call
    RETURN_TYPES = ("SAMPLER",) # This is not directly used for the sampler class itself, but for a node wrapper

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
        self.model_ = model # Underlying diffusion model function
        self.x = x # Initial noisy tensor to sample from
        self.sigmas = sigmas # Array of noise levels to step through
        self.extra_args = extra_args.copy() if extra_args is not None else {} # Additional arguments for model (e.g., conditioning)
        self.callback_ = callback # External callback to report progress (for ComfyUI's progress bar)
        self.disable_pbar = disable # Disable progress bar if truthy (renamed for clarity)

        # Sampling control parameters, extracted from kwargs
        self.start_sigma_index = kwargs.pop("start_sigma_index", 0) # Sigma array index to start sampling from
        self.end_sigma_index = kwargs.pop("end_sigma_index", -1) # Sigma array index to end sampling early (-1 means all steps)
        self.enable_clamp_output = kwargs.pop("enable_clamp_output", False) # Clamp output to [-1,1]

        # Noise injection and random seed controls
        self.step_random_mode = kwargs.pop("step_random_mode", "off") # How to vary RNG seed per step ("off", "block", "reset", "step")
        self.step_size = kwargs.pop("step_size", 5) # Block size for block-based mode or multiplier for reset mode
        self.seed = kwargs.pop("seed", 42) # Base seed for reproducibility

        # Blend functions (already resolved to functions in .go())
        self.blend_function = kwargs.pop("blend_function", torch.lerp)
        self.step_blend_function = kwargs.pop("step_blend_function", torch.lerp)

        # Ancestral (ping-pong) operation boundaries
        pingpong_options = kwargs.pop("pingpong_options", None)
        if pingpong_options is None:
            pingpong_options= {}
        
        # Determine the number of total steps available from the sigmas schedule
        num_steps_available = len(sigmas) - 1 if len(sigmas) > 0 else 0

        # Get raw ancestral step inputs, with sensible defaults if not provided
        raw_first_ancestral_step = pingpong_options.get("first_ancestral_step", 0)
        # Default last_ancestral_step to cover full range if not specified
        raw_last_ancestral_step = pingpong_options.get("last_ancestral_step", num_steps_available - 1)
        
        # Ensure first is always <= last for the ancestral range to be valid
        # And clamp both to be within the actual number of available steps [0, num_steps_available-1]
        self.first_ancestral_step = max(0, min(raw_first_ancestral_step, raw_last_ancestral_step))
        if num_steps_available > 0:
            self.last_ancestral_step = min(num_steps_available - 1, max(raw_first_ancestral_step, raw_last_ancestral_step))
        else:
            self.last_ancestral_step = -1 # No steps available, so no ancestral steps will run


        # Detect if model uses ComfyUI CONST sampling (e.g., for Reflow-like noise injection behavior)
        self.is_rf = False
        current_model_check = model # Use a temporary variable for traversing model wrappers
        try:
            # Common pattern for ComfyUI to wrap the actual model inside `inner_model`
            while hasattr(current_model_check, 'inner_model') and current_model_check.inner_model is not None:
                current_model_check = current_model_check.inner_model
            # Check if the innermost model's sampling type is CONST
            if hasattr(current_model_check, 'model_sampling') and current_model_check.model_sampling is not None:
                self.is_rf = isinstance(current_model_check.model_sampling, model_sampling.CONST)
        except AttributeError:
            # If any attribute is missing along the chain, assume not CONST to be safe
            print("PingPongSampler Warning: Could not definitively determine model_sampling type, assuming not CONST.")
            self.is_rf = False


        # Default noise sampler: NOW RETURNS RAW, UNCONDITIONED UNIT VARIANCE NOISE
        # All scaling will happen explicitly in the __call__ method.
        self.noise_sampler = noise_sampler # Use external noise sampler if provided
        if self.noise_sampler is None:
            def default_noise_sampler(sigma, sigma_next): # Signature kept for compatibility, but sigma/sigma_next are unused here
                return torch.randn_like(x) # Returns raw random noise with the same shape as self.x
            self.noise_sampler = default_noise_sampler

        # Build noise decay array, either from external scheduler or use zeros (no decay)
        scheduler = kwargs.pop("scheduler", None) # Get scheduler from kwargs
        if num_steps_available > 0:
            if scheduler is not None and hasattr(scheduler, 'get_decay'):
                try:
                    # Attempt to get noise decay values from the external scheduler
                    arr = scheduler.get_decay(num_steps_available)
                    decay = np.asarray(arr, dtype=np.float32)
                    assert decay.shape == (num_steps_available,), (
                        f"Expected {num_steps_available} values from scheduler.get_decay, got {decay.shape}"
                    )
                    self.noise_decay = torch.tensor(decay, device=x.device)
                except Exception as e:
                    # Fallback to zeros if scheduler fails (e.g., invalid input or method missing)
                    print(f"PingPongSampler Warning: Could not get decay from scheduler: {e}. Using zeros for noise decay.")
                    self.noise_decay = torch.zeros((num_steps_available,), dtype=torch.float32, device=x.device)
            else:
                # No external scheduler provided or it lacks 'get_decay': no noise decay (all zeros)
                self.noise_decay = torch.zeros((num_steps_available,), dtype=torch.float32, device=x.device)
        else:
            # No steps, so noise_decay should be empty or not accessed
            self.noise_decay = torch.empty((0,), dtype=torch.float32, device=x.device)
        
        # Report any unhandled keyword arguments received during initialization
        if kwargs:
            print(f"PingPongSampler initialized with unused extra_options: {kwargs}")


    def _stepped_seed(self, step: int):
        """
        Determines the RNG seed for the current step based on the selected random mode.
        This provides controlled variations in noise generation across sampling steps.
        """
        if self.step_random_mode == "off":
            return None # No specific seed override, rely on global RNG state if applicable
        
        # Ensure step_size is positive to avoid division by zero
        current_step_size = self.step_size if self.step_size > 0 else 1

        if self.step_random_mode == "block":
            # Seed changes only after a certain number of steps (a "block")
            return self.seed + (step // current_step_size)
        elif self.step_random_mode == "reset":
            # Seed changes by a multiple of step_size, creating large jumps
            return self.seed + (step * current_step_size)
        elif self.step_random_mode == "step":
            # Seed changes incrementally with each step
            return self.seed + step
        else:
            # Unknown mode: fallback to base seed and warn (should not happen with ComfyUI dropdown)
            print(f"PingPongSampler Warning: Unknown step_random_mode '{self.step_random_mode}'. Using base seed.")
            return self.seed

    @classmethod
    def go(cls, # This is the entry point function that ComfyUI's KSAMPLER will call
        model, # The latent tensor representing the current state (noisy or denoised)
        x, # The latent tensor representing the current state (noisy or denoised)
        sigmas, # A tensor of sigma values defining the noise schedule
        extra_args=None, # Additional arguments to pass to the model's forward pass
        callback=None, # A callback function for ComfyUI's progress updates
        disable=None, # Flag to disable the progress bar
        noise_sampler=None, # An optional custom noise sampler function
        # All custom options from PingPongSamplerNode.INPUT_TYPES arrive here as kwargs
        **kwargs
    ):
        """
        Entrypoint for ComfyUI's KSAMPLER to initiate sampling with PingPongSampler.
        This method constructs an instance of PingPongSampler and then executes its main sampling loop.
        """
        # Extract explicit parameters from kwargs that are expected by __init__
        # And also resolve blend functions from strings
        blend_mode = kwargs.pop("blend_mode", "lerp")
        step_blend_mode = kwargs.pop("step_blend_mode", "lerp")
        
        resolved_blend_function = _INTERNAL_BLEND_MODES.get(blend_mode, torch.lerp)
        resolved_step_blend_function = _INTERNAL_BLEND_MODES.get(step_blend_mode, torch.lerp)

        # Other parameters that need to be popped for clarity and explicit passing to __init__
        # These are handled correctly by the final_options merge logic, but need to be popped
        # to prevent them from being passed as redundant kwargs to __init__.
        pingpong_options = kwargs.pop("pingpong_options", None)
        start_sigma_index = kwargs.pop("start_sigma_index", 0)
        end_sigma_index = kwargs.pop("end_sigma_index", -1)
        enable_clamp_output = kwargs.pop("enable_clamp_output", False)
        step_random_mode = kwargs.pop("step_random_mode", "off")
        step_size = kwargs.pop("step_size", 5)
        seed = kwargs.pop("seed", 42)
        scheduler = kwargs.pop("scheduler", None)

        # --- ADD THESE LINES TO POP THE REDUNDANT KEYS ---
        _ = kwargs.pop("verbose", False) # Pop verbose if it exists at top level
        _ = kwargs.pop("first_ancestral_step", -1) # Pop if it exists at top level
        _ = kwargs.pop("last_ancestral_step", -1) # Pop if it exists at top level
        # --- END ADDED LINES ---

        # yaml_settings_str is consumed by get_sampler, but ensure it's popped if it somehow reaches here
        _ = kwargs.pop("yaml_settings_str", "") 

        # Create an instance of the PingPongSampler class.
        # Pass all the required arguments explicitly.
        sampler_instance = cls(
            model=model,
            x=x,
            sigmas=sigmas,
            extra_args=extra_args,
            callback=callback,
            disable=disable,
            noise_sampler=noise_sampler, # This can be None, handled in __init__
            # Pass all the extracted parameters explicitly:
            pingpong_options=pingpong_options,
            start_sigma_index=start_sigma_index,
            end_sigma_index=end_sigma_index,
            enable_clamp_output=enable_clamp_output,
            step_random_mode=step_random_mode,
            step_size=step_size,
            seed=seed,
            scheduler=scheduler, # Pass scheduler here
            blend_function=resolved_blend_function, # Pass resolved functions here
            step_blend_function=resolved_step_blend_function, # Pass resolved functions here
            **kwargs # Pass any remaining kwargs to __init__ (should be empty now for standard operation)
        )
        # Call the __call__ method of the created sampler instance to perform the actual sampling process.
        return sampler_instance()

    def _model_denoise(self, x_tensor, sigma_scalar, **model_call_kwargs):
        """
        Wrapper around the underlying diffusion model's denoising function.
        It expands the scalar sigma to a tensor suitable for batch processing
        and combines any additional arguments before calling the model.
        """
        batch_size = x_tensor.shape[0]
        # Create a sigma tensor of the same batch size as x_tensor
        sigma_tensor = sigma_scalar * x_tensor.new_ones((batch_size,))
        # Merge extra_args from __init__ with any specific to this call
        final_extra_args = {**self.extra_args, **model_call_kwargs}
        # Call the actual diffusion model to predict the denoised latent
        return self.model_(x_tensor, sigma_tensor, **final_extra_args)

    def _do_callback(self, step_idx, current_x, current_sigma, denoised_sample):
        """
        Forwards progress information and intermediate results to ComfyUI's callback system.
        This allows the UI to display progress and intermediate images/latents.
        """
        if self.callback_:
            # Provide dictionary conforming to ComfyUI's expected callback format
            self.callback_({
                "i": step_idx, # Current step index
                "x": current_x, # Current latent state (noisy)
                "sigma": current_sigma, # Current noise level (sigma)
                "sigma_hat": current_sigma, # Denoised sigma (often same as current_sigma for simple samplers)
                "denoised": denoised_sample # The model's prediction of the denoised latent
            })

    def __call__(self): # This is the main sampling loop execution
        """
        Executes the main "ping-pong" sampling loop, iterating through sigma steps
        and applying denoising, ancestral noise injection.
        """
        x_current = self.x.clone() # Create a mutable copy of the initial latent to work on
        num_steps = len(self.sigmas) - 1 # Total number of sampling steps (transitions between sigmas)

        # Handle edge case: no steps to take if only one sigma value or less
        if num_steps <= 0:
            if self.enable_clamp_output:
                x_current = torch.clamp(x_current, -1.0, 1.0) # Apply final clamp if enabled
            return x_current
            
        # Retrieve pre-calculated and clamped ancestral step boundaries from __init__
        astart = self.first_ancestral_step
        aend = self.last_ancestral_step
        
        # Determine the actual end index for the sigma iteration, respecting user input
        # If end_sigma_index is -1, iterate through all steps up to num_steps-1
        actual_iteration_end_idx = self.end_sigma_index if self.end_sigma_index >= 0 else num_steps - 1
        # Ensure the end index does not exceed the actual number of available steps
        actual_iteration_end_idx = min(actual_iteration_end_idx, num_steps - 1)


        # Main sampling loop: iterate through each step (from 0 to num_steps-1)
        for idx in trange(num_steps, disable=self.disable_pbar):
            # Skip steps outside the user-defined start and end sigma indices
            if idx < self.start_sigma_index or idx > actual_iteration_end_idx:
                continue

            # Get the current and next sigma values from the schedule
            sigma_current, sigma_next = self.sigmas[idx], self.sigmas[idx + 1]
            
            # Call the diffusion model to get the denoised prediction for the current step
            denoised_sample = self._model_denoise(x_current, sigma_current)
            
            # Report progress to ComfyUI
            self._do_callback(idx, x_current, sigma_current, denoised_sample)

            # Clamping Logic: Apply clamping only at very low sigma_next (near the end of sampling)
            # This prevents premature clamping of intermediate noisy latents, preserving detail.
            if self.enable_clamp_output and sigma_next < 1e-3: # Threshold 1e-3 can be adjusted
                x_current = torch.clamp(x_current, -1.0, 1.0)
                break # Exit loop after final clamp, as sampling is effectively complete

            # If sigma_next is practically zero, we've reached the final denoised state
            if sigma_next <= 1e-6:
                x_current = denoised_sample # The final result is the denoised prediction
                break # Exit loop, no more noise to add or steps to take


            # Determine whether ancestral noise injection should be used for this specific step
            # `astart` and `aend` are already 0-based and clamped within the valid step range
            use_anc = (astart <= idx <= aend) if astart <= aend else False

            if not use_anc:
                # Non-Ancestral Step (DDIM-like interpolation): Reverting to original's step_blend_function
                # This path does NOT use `noise_sample` or `s_noise` or `dynamic_noise`.
                blend = sigma_next / sigma_current if sigma_current > 0 else 0.0 # Calculate interpolation factor
                x_current = self.step_blend_function(denoised_sample, x_current, blend) # Interpolate using step_blend_function
                continue # Move to the next step

            # --- Ancestral Step Logic: Injecting controlled noise (Reverted to Original `blepping` Behavior) ---
            # Set the RNG seed for reproducibility/variability if a stepped random mode is enabled
            local_seed = self._stepped_seed(idx)
            if local_seed is not None:
                torch.manual_seed(local_seed)

            # Generate new unit variance noise (always raw from default_noise_sampler)
            noise_sample = self.noise_sampler(sigma_current, sigma_next) 

            # IMPORTANT: Based on user feedback, Ace-Step appears to be highly sensitive to the
            # exact magnitude of ancestral noise. 's_noise' and 'dynamic_noise' parameters
            # caused artifacts when applied. Therefore, for ancestral steps, we apply the
            # raw, unscaled noise_sample. The necessary scaling is inherently handled by
            # `sigma_next` in the original blepping formulas or the blend function.
            final_noise_to_add = noise_sample

            # Apply the ancestral step update (ORIGINAL BLEPPING LOGIC)
            if self.is_rf: # If it's a CONST model (like some reflow/consistency models, e.g. from Stable Audio)
                x_current = self.step_blend_function(denoised_sample, final_noise_to_add, sigma_next)
            else: # For other models (e.g., EDM, standard DDPM)
                x_current = denoised_sample + final_noise_to_add * sigma_next

        # Final clamping for the very last step if it wasn't already handled by the `sigma_next < 1e-3` check in the loop.
        if self.enable_clamp_output:
             x_current = torch.clamp(x_current, -1.0, 1.0)

        return x_current


class PingPongSamplerNode:
    """
    ComfyUI node wrapper to register PingPongSampler as a custom sampler.
    This class defines the inputs and outputs that appear in the ComfyUI user interface.
    """
    CATEGORY = "sampling/custom_sampling/samplers" # Standard ComfyUI category for samplers
    RETURN_TYPES = ("SAMPLER",) # This node outputs a SAMPLER object
    FUNCTION = "get_sampler" # The class method that ComfyUI will call to get the sampler object

    @classmethod
    def INPUT_TYPES(cls):
        """
        Defines the input parameters that will be exposed in the ComfyUI node editor.
        These inputs allow users to customize the behavior of the PingPongSampler.
        """
        return {
            "required": {
                # Removed s_noise parameter
                
                # step_random_mode: How the RNG seed varies per sampling step.
                "step_random_mode":    (["off", "block", "reset", "step"], {"default": "block", "tooltip": "Controls how the RNG seed varies per sampling step.\n- 'off': Seed is constant. Predictable, but where's the fun in that?\n- 'block': Seed changes every 'step_size' frames. Like a glitch in the matrix, but intentional.\n- 'reset': Seed is reset based on 'step_size' multiplied by the frame index, offering more varied randomness.\n- 'step': Seed changes incrementally by the frame index at each step, providing subtle variations."}),
                
                # step_size: Interval for 'block' and 'reset' random modes.
                "step_size":           ("INT",    {"default": 4,    "min": 1,        "max": 100, "tooltip": "Used by 'block' and 'reset' step random modes to define the block/reset interval for the seed."}),
                
                # seed: Base random seed.
                "seed":                ("INT",    {"default": 80085,  "min": 0,        "max": 2**32 - 1, "tooltip": "Base random seed. The cosmic initializer. Change it for new universes, keep it for deja vu."}),
                
                # first_ancestral_step: Start index for ancestral noise.
                "first_ancestral_step": ("INT",    {"default": 0, "min": -1, "max": 10000, "tooltip": "The sampler step index (0-based) to begin ancestral\nnoise injection (ping-pong behavior). Use -1 to effectively disable ancestral noise if last_ancestral_step is also -1."}),
                
                # last_ancestral_step: End index for ancestral noise.
                "last_ancestral_step":  ("INT",    {"default": -1, "min": -1, "max": 10000, "tooltip": "The sampler step index (0-based) to end ancestral\nnoise injection (ping-pong behavior). Use -1 to extend ancestral noise to the end of the sampling process."}),
                
                # start_sigma_index: Sigma array index to begin sampling.
                "start_sigma_index":    ("INT",    {"default": 0,    "min": 0,        "max": 10000, "tooltip": "The index in the sigma array (denoising schedule) to begin sampling from. Allows skipping initial high-noise steps, potentially speeding up generation or changing visual character."}),
                
                # end_sigma_index: Sigma array index to end sampling.
                "end_sigma_index":      ("INT",    {"default": -1,   "min": -10000,  "max": 10000, "tooltip": "The index in the sigma array to end sampling at. -1 means sample all steps. To the bitter end, or a graceful exit? You decide."}),
                
                # enable_clamp_output: Clamp final output.
                "enable_clamp_output":  ("BOOLEAN", {"default": False, "tooltip": "If true, clamps the final output latent to the range [-1.0, 1.0]. Useful for preventing extreme values that might lead to artifacts during decoding."}),
                
                # scheduler: External noise decay scheduler.
                "scheduler":           ("SCHEDULER", {"tooltip": "Connect a ComfyUI scheduler node (e.g., KSamplerScheduler) to define the noise decay curve for the sampler. Essential for proper sampling progression. It's the tempo track for your pixels!"}),
                
                # Removed dynamic_noise parameters
                
                # blend_mode: Blend mode for ancestral steps.
                "blend_mode":          (tuple(_INTERNAL_BLEND_MODES.keys()), {"default": "lerp", "tooltip": "Blend mode to use for blending noise in ancestral steps. Defaults to 'lerp' (linear interpolation). Choose your flavor: 'lerp' (smooth blend), 'a_only' (take noise as is), 'b_only' (take other input as is). Fancy, right?"}),
                
                # step_blend_mode: Blend mode for non-ancestral steps.
                "step_blend_mode":     (tuple(_INTERNAL_BLEND_MODES.keys()), {"default": "lerp", "tooltip": "Blend mode to use for non-ancestral steps (regular denoising progression). Changing this from 'lerp' is generally not recommended unless you're feeling particularly chaotic. Like trying to render Doom on a 386SX with 2MB RAM."}),
            },
            "optional": {
                # yaml_settings_str: Optional YAML string for configuration.
                "yaml_settings_str": ("STRING", {"multiline": True, "default": "", "dynamic_prompt": False, "tooltip": "YAML string to configure sampler parameters. Parameters provided directly via the ComfyUI node's inputs will now be **OVERRIDDEN** by any corresponding values set in the YAML string. If the YAML is empty, node inputs are used. YAML is the boss now, respect its authority!"}),
            }
        }

    # This method is called by ComfyUI to get the sampler object
    def get_sampler(
        self,
        step_random_mode: str,
        step_size: int,
        seed: int,
        first_ancestral_step: int,
        last_ancestral_step: int,
        start_sigma_index: int,
        end_sigma_index: int,
        enable_clamp_output: bool,
        scheduler=None, # Scheduler is optional (can be None)
        blend_mode: str = "lerp", # Parameter: blend mode for ancestral steps
        step_blend_mode: str = "lerp", # Parameter: blend mode for non-ancestral steps
        yaml_settings_str: str = "" # Optional YAML string input
    ):
        """
        This method gathers all input parameters from the ComfyUI node,
        merges them with any provided YAML settings (prioritizing YAML),
        and returns a KSAMPLER object configured with the PingPongSampler.
        """
        # Create a dictionary with direct node inputs.
        direct_inputs = {
            "step_random_mode": step_random_mode,
            "step_size": step_size,
            "seed": seed,
            "pingpong_options": { # Nested dictionary for ancestral step control
                "first_ancestral_step": first_ancestral_step,
                "last_ancestral_step": last_ancestral_step,
            },
            "start_sigma_index": start_sigma_index,
            "end_sigma_index": end_sigma_index,
            "enable_clamp_output": enable_clamp_output,
            "scheduler": scheduler,
            "blend_mode": blend_mode, # Pass the string mode here
            "step_blend_mode": step_blend_mode, # Pass the string mode here
            "yaml_settings_str": yaml_settings_str, # Pass this through for `go` to pop it
        }

        # Initialize final_options with direct node inputs first.
        # This acts as the baseline/default if YAML is empty or doesn't specify a key.
        final_options = direct_inputs.copy()

        # Attempt to load YAML data.
        yaml_data = {}
        if yaml_settings_str:
            try:
                temp_yaml_data = yaml.safe_load(yaml_settings_str)
                if isinstance(temp_yaml_data, dict):
                    yaml_data = temp_yaml_data
                    # Convert string representations of booleans from YAML to actual booleans
                    for k, v in yaml_data.items():
                        if isinstance(v, str):
                            if v.lower() == 'true':
                                yaml_data[k] = True
                            elif v.lower() == 'false':
                                yaml_data[k] = False
            except yaml.YAMLError as e:
                print(f"WARNING: PingPongSamplerNode YAML parsing error: {e}. Using direct node inputs only.")
            except Exception as e:
                print(f"WARNING: PingPongSamplerNode unexpected error during YAML parsing: {e}. Using direct node inputs only.")

        # --- MERGE YAML DATA ON TOP OF DIRECT INPUTS ---
        # This is the key change: YAML data now overrides direct inputs.
        for key, value in yaml_data.items():
            if key == "pingpong_options" and isinstance(value, dict) and key in final_options and isinstance(final_options[key], dict):
                # If 'pingpong_options' exists in both YAML and direct inputs and are both dicts,
                # merge their contents (YAML keys now overwrite direct input keys).
                final_options[key].update(value)
            else:
                # For all other parameters, YAML value simply overwrites or adds the value.
                final_options[key] = value

        # The KSAMPLER wrapper takes the sampling function (PingPongSampler.go)
        # and an `extra_options` dictionary, whose contents are passed as **kwargs
        # to the sampling function.
        return (KSAMPLER(PingPongSampler.go, extra_options=final_options),)

# Dictionary mapping class names to identifiers used by ComfyUI
NODE_CLASS_MAPPINGS = {
    "PingPongSampler_Custom": PingPongSamplerNode,
}

# Dictionary mapping class names to display names in the ComfyUI UI
NODE_DISPLAY_NAME_MAPPINGS = {
    "PingPongSampler_Custom": "PingPong Sampler (Custom V0.8.15)", # Updated version number!
}
