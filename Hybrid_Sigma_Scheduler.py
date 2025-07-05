# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# ████ Hybrid Sigma Scheduler v0.69.420.2 – Optimized for Ace-Step Audio/Video ████▓▒░
# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀

# ░▒▓ ORIGIN & DEV:
#   • Brewed in a Compaq Presario by MDMAchine / MD_Nodes
#   • License: Apache 2.0 — legal chaos with community spirit

# ░▒▓ DESCRIPTION:
#   Outputs a tensor of sigmas to control diffusion noise levels.
#   No guesswork. No drama. Just pure generative groove.
#   Designed for precision noise scheduling in ComfyUI workflows.
#   May perform differently outside of intended models.

# ░▒▓ FEATURES:
#   ✓ Two core modes: Karras Fury & Linear Chill
#   ✓ Bonus: Curve blending for smooth vibe control
#   ✓ Flexible sigma adjustment with start and end controls
#   ✓ Supports advanced noise profiles: distro, uniform, collatz, pyramid, gaussian, highres_pyramid_bislerp
#   ✓ Pro tips baked in for dialing in your noise precisely

# ░▒▓ CHANGELOG:
#   - v0.1 (Initial Brew):
#       • Core sigma scheduling implemented
#       • Basic Karras and Linear modes functional
#   - v0.5:
#       • Added blended curve mode for mixing noise profiles
#       • Improved stability and noise curve precision
#   - v0.6:
#       • Introduced kl_optimal and linear_quadratic modes for advanced sampling control
#       • Added pro tips for noise tuning
#   - v0.69.420.1 (Current Release):
#       • Refined blending algorithm for ultimate vibe control
#       • Enhanced compatibility with ComfyUI internal sampling functions
#   - v0.70 (Proposed Update):
#       • Added 'model' input for dynamic sigma_min/max retrieval from the loaded model.
#       • Introduced 'start_at_step' and 'end_at_step' for precise slicing of the generated sigma schedule.
#       • Clarified 'denoise' parameter role as a passthrough.

# ░▒▓ CONFIGURATION:
#   → Primary Use: Fine-grained diffusion noise scheduling for generative workflows
#   → Secondary Use: Experimental noise shaping and curve blending
#   → Edge Use: Hardcore noise tinkerers and sampling theorists

# ░▒▓ WARNING:
#   This node may trigger:
#   ▓▒░ Unexpected vibes
#   ▓▒░ Spontaneous beatboxing
#   ▓▒░ Mild to moderate enlightenment

# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄


import torch
import comfy.model_management
# IMPORTANT: get_sigmas_karras from comfy.k_diffusion.sampling does NOT take a 'device' argument directly.
# It uses .new_zeros() which inherits device from the input tensor.
from comfy.k_diffusion.sampling import get_sigmas_karras
import math

# Referenced from https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/15608
def kl_optimal_scheduler(n: int, sigma_min: float, sigma_max: float) -> torch.Tensor:
    # Ensure inputs are torch tensors for operations
    sigma_min_t = torch.tensor(sigma_min, dtype=torch.float32)
    sigma_max_t = torch.tensor(sigma_max, dtype=torch.float32)

    adj_idxs = torch.arange(n, dtype=torch.float32).div_(n - 1)
    sigmas = adj_idxs.new_zeros(n + 1)
    # Using torch.atan and torch.tan_ for tensor operations
    sigmas[:-1] = (adj_idxs * torch.atan(sigma_min_t) + (1 - adj_idxs) * torch.atan(sigma_max_t)).tan_()
    return sigmas

# Adapted from: https://github.com/genmoai/models/blob/main/src/mochi_preview/infer.py#L41
def linear_quadratic_schedule_adapted(steps: int, sigma_max: float, threshold_noise: float = 0.025, linear_steps: int = None) -> torch.Tensor:
    if steps <= 0:
        return torch.FloatTensor([sigma_max, 0.0]) # Return a minimal valid schedule
    
    if steps == 1:
        return torch.FloatTensor([sigma_max, 0.0]) # Directly scaled

    # Determine actual linear_steps, clamping it within [0, steps]
    if linear_steps is None:
        linear_steps_actual = steps // 2
    else:
        linear_steps_actual = max(0, min(linear_steps, steps))

    sigma_schedule_raw = []

    if linear_steps_actual == 0:
        # Purely quadratic schedule if no linear steps are specified
        # Raw values should go from 0 up to 1.0 (before inversion)
        for i in range(steps): # Indices from 0 to steps-1
            # This implements a quadratic ramp from 0.0 to 1.0
            # (i / (steps - 1.0))^2, ensuring steps-1.0 is not zero for steps > 1
            if steps > 1:
                sigma_schedule_raw.append((i / (steps - 1.0))**2)
            else: # Should not happen due to steps == 1 check above, but for safety
                sigma_schedule_raw.append(0.0)
    else:
        # Linear part (from 0 to threshold_noise)
        for i in range(linear_steps_actual):
            sigma_schedule_raw.append(i * threshold_noise / linear_steps_actual)

        # Quadratic part (from threshold_noise to 1.0)
        quadratic_steps = steps - linear_steps_actual
        if quadratic_steps > 0:
            # Original calculations from the source
            threshold_noise_step_diff = linear_steps_actual - threshold_noise * steps
            quadratic_coef = threshold_noise_step_diff / (linear_steps_actual * quadratic_steps ** 2)
            linear_coef = threshold_noise / linear_steps_actual - 2 * threshold_noise_step_diff / (quadratic_steps ** 2)
            const = quadratic_coef * (linear_steps_actual ** 2)
            
            for i in range(linear_steps_actual, steps):
                sigma_schedule_raw.append(quadratic_coef * (i ** 2) + linear_coef * i + const)
    
    # Ensure the final element is 1.0 before inversion and scaling, matching original behavior
    # This also ensures the output tensor has `steps + 1` elements.
    if not sigma_schedule_raw or sigma_schedule_raw[-1] != 1.0:
        sigma_schedule_raw.append(1.0)
    
    # Invert the schedule (1.0 - x) and scale by sigma_max
    sigma_schedule_inverted = [1.0 - x for x in sigma_schedule_raw]
    
    return torch.FloatTensor(sigma_schedule_inverted) * sigma_max


class HybridAdaptiveSigmas:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                # --- ADDED: Model input to derive sigma_min/max ---
                "model": ("MODEL", {"tooltip": "The diffusion model. Used to get inherent sigma_min and sigma_max for scheduling."}),
                # ----------------------------------------------------

                # How many sigma steps you want. Think of this like "resolution" for noise decay.
                "steps": ("INT", {
                    "default": 60, "min": 5, "max": 200,
                    "tooltip": "The number of steps from chaos to clarity. More steps = more precision (and more CPU tears). Note: Higher 'rho' values (in Karras Fury mode) often benefit from a higher number of steps."
                }),

                # Choose your decay curve style or blending.
                "mode": (["karras_rho", "adaptive_linear", "blended_curves", "kl_optimal", "linear_quadratic"], {
                    "tooltip": "🔥 karras_rho = rich, curvy chaos.\\n🧊 adaptive_linear = budget-friendly flatline (generally effective with around 60 steps).\\n💖 blended_curves = mix Karras and Linear.\\n✨ kl_optimal = Based on Theorem 3.1 of 'Align Your Steps' paper, designed for improved sample quality. (Note: This mode uses the model's inherent sigma range, ignoring start_sigma/end_sigma.)\\n📈 linear_quadratic = A schedule with an initial linear phase and a subsequent quadratic phase."
                }),

                # --- ADDED: Parameters for slicing the output sigma schedule ---
                "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000, "tooltip": "Start denoising from this step (inclusive, relative to total steps before slicing)."}),
                "end_at_step": ("INT", {"default": 9999, "min": 0, "max": 10000, "tooltip": "End denoising at this step (exclusive, relative to total steps before slicing). A high value like 9999 means to the very end."}),
                # ---------------------------------------------------------------

                # Only used if you're running with Karras Fury or Blended mode.
                "rho": ("FLOAT", {
                    "default": 1.5, "min": 1.0, "max": 15.0,
                    "tooltip": "Rho controls the Karras curve sharpness. Low = gentle slopes. High = noise rollercoaster. 1.1-2.5 often yields cohesive results. Higher RHOs typically require more steps. Rho beyond 4.5 may lead to scattered output, especially with certain noise types."
                }),

                # New: Blend factor for blended_curves mode
                "blend_factor": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Only for 'blended_curves' mode. Think of it as your DJ slider: 0.0 is pure Karras beats, 1.0 is all Linear chill. Find your perfect mix!"
                }),
            },
            "optional": {
                # --- MOVED AND UPDATED: Denoise parameter for downstream samplers ---
                "denoise": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0,
                    "tooltip": "Denoising strength. Passed downstream to samplers; doesn't affect sigmas directly generated by this node."
                }),
                # -------------------------------------------------------------------

                # --- UPDATED: These are now explicit overrides for model's sigma_min/max ---
                "start_sigma_override": ("FLOAT", {
                    "default": 1.000, "min": 0.0, "max": 20.0, "step": 0.001, "optional": True,
                    "tooltip": "Optional override for the schedule's maximum noise (sigma_max). If provided, this value takes precedence over the model's inherent sigma_max. E.g., 1.0 for SDXL, 14.6 for SD1.5."
                }),
                "end_sigma_override": ("FLOAT", {
                    "default": 0.006, "min": 0.0, "max": 20.0, "step": 0.001, "optional": True,
                    "tooltip": "Optional override for the schedule's minimum noise (sigma_min). If provided, this value takes precedence over the model's inherent sigma_min. E.g., 0.006 for SDXL, 0.02 for SD1.5."
                }),
                # -----------------------------------------------------------------------------

                "threshold_noise": ("FLOAT", {
                    "default": 0.025, "min": 0.0, "max": 1.0, "step": 0.001,
                    "tooltip": "Only for 'linear_quadratic' mode. It's the pivot point, where your noise decay decides to stop being a straight-shooter and gets a bit more... quadratic. Think of it as where the curve 'breaks' from a straight line."
                }),
                "linear_steps": ("INT", {
                    "default": None, "min": 0, "max": 200,
                    "tooltip": "Only for 'linear_quadratic' mode. How many steps should be 'boringly' linear before the curve gets interesting? If left empty, it'll pick half the steps for straight talk."
                }),
            }
        }

    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("sigmas",)
    FUNCTION = "generate"
    CATEGORY = "MD_Nodes/Schedulers" # Updated Category

    def generate(self, model, steps, mode, rho, blend_factor, start_at_step, end_at_step, # ADDED: start_at_step, end_at_step
                 denoise=1.0, # MOVED: Now optional
                 start_sigma_override=None, end_sigma_override=None, # UPDATED NAMES
                 threshold_noise=None, linear_steps=None):
        try:
            device = comfy.model_management.get_torch_device()
            sigmas = None

            actual_sigma_min = None
            actual_sigma_max = None

            # --- UPDATED LOGIC: Determine actual sigma_min/max ---
            # 1. Attempt to get model's inherent sigma range
            try:
                model_sampling_obj = model.get_model_object("model_sampling")
                actual_sigma_min = model_sampling_obj.sigma_min
                actual_sigma_max = model_sampling_obj.sigma_max
                print(f"DEBUG: Model inherent sigma_min={actual_sigma_min}, sigma_max={actual_sigma_max}.")

                # Heuristic: Check for "bad" model values (e.g., identical or very small range)
                if actual_sigma_min is not None and actual_sigma_max is not None:
                    if abs(actual_sigma_max - actual_sigma_min) < 1e-5 or actual_sigma_max < 0.1:
                        print(f"WARNING: Model reported a very small or unusual sigma range ({actual_sigma_min} to {actual_sigma_max}). This might be incorrect metadata. Will check for manual overrides.")
                        actual_sigma_min = None # Reset to force fallback/override
                        actual_sigma_max = None # Reset to force fallback/override

            except AttributeError as e:
                print(f"WARNING: Could not access model.get_model_object('model_sampling').sigma_min/max ({e}). Will check for manual overrides or fall back to common defaults.")
            except Exception as e: # Catch other potential issues during model access
                print(f"WARNING: An unexpected error occurred while trying to get model's sigma_min/max: {e}. Will check for manual overrides or fall back to common defaults.")
            
            # 2. Apply user overrides if provided (start_sigma_override -> sigma_max, end_sigma_override -> sigma_min)
            if start_sigma_override is not None:
                actual_sigma_max = start_sigma_override
                print(f"DEBUG: User provided start_sigma_override={start_sigma_override}. Overriding sigma_max.")
            
            if end_sigma_override is not None:
                actual_sigma_min = end_sigma_override
                print(f"DEBUG: User provided end_sigma_override={end_sigma_override}. Overriding sigma_min.")

            # 3. Apply final fallback if still None (meaning model didn't provide good values AND no user override)
            if actual_sigma_min is None or actual_sigma_max is None:
                # Fallback to common stable diffusion defaults if no valid range found from model or user override
                actual_sigma_min = 0.02
                actual_sigma_max = 14.6120
                print(f"WARNING: No valid sigma range found (either from model or user override). Falling back to common defaults: sigma_min={actual_sigma_min}, sigma_max={actual_sigma_max}. Results may vary. Consider providing overrides manually or using a model with correct metadata.")
            
            # Final check to ensure actual_sigma_max is greater than actual_sigma_min
            if actual_sigma_min >= actual_sigma_max:
                print(f"ERROR: Calculated sigma_min ({actual_sigma_min}) is greater than or equal to sigma_max ({actual_sigma_max}). Adjusting to a small valid range to prevent crash.")
                actual_sigma_max = actual_sigma_min + 0.1 # Ensure a small positive range
                if actual_sigma_max > 15.0: # Cap if it becomes too large from adjustment
                    actual_sigma_max = 15.0
                    actual_sigma_min = actual_sigma_max - 0.1
                
            print(f"INFO: Final sigma range used: sigma_min={actual_sigma_min}, sigma_max={actual_sigma_max}")
            # -----------------------------------------------------

            # Now use actual_sigma_min and actual_sigma_max in all modes
            if mode == "karras_rho":
                # Removed 'device=device' from get_sigmas_karras as it's not a valid parameter for this function.
                sigmas = get_sigmas_karras(
                    steps,
                    sigma_min=actual_sigma_min,
                    sigma_max=actual_sigma_max,
                    rho=rho
                )
                sigmas = sigmas.to(device) # Ensure it's on the correct device after generation

            elif mode == "adaptive_linear":
                current_sigma = actual_sigma_max
                adaptive_sigmas = [current_sigma]
                step_size = (actual_sigma_max - actual_sigma_min) / steps
                for _ in range(steps):
                    current_sigma -= step_size
                    current_sigma = max(current_sigma, actual_sigma_min)
                    adaptive_sigmas.append(current_sigma)
                sigmas = torch.tensor(adaptive_sigmas, device=device)

            elif mode == "blended_curves":
                # Removed 'device=device' from get_sigmas_karras here as well.
                karras_sigmas = get_sigmas_karras(
                    steps,
                    sigma_min=actual_sigma_min,
                    sigma_max=actual_sigma_max,
                    rho=rho
                )
                karras_sigmas = karras_sigmas.to(device) # Ensure on device

                linear_current_sigma = actual_sigma_max
                linear_adaptive_sigmas = [linear_current_sigma]
                linear_step_size = (actual_sigma_max - actual_sigma_min) / steps
                for _ in range(steps):
                    linear_current_sigma -= linear_step_size
                    linear_current_sigma = max(linear_current_sigma, actual_sigma_min)
                    linear_adaptive_sigmas.append(linear_current_sigma)
                linear_sigmas = torch.tensor(linear_adaptive_sigmas, device=device)

                if karras_sigmas.shape != linear_sigmas.shape:
                    print("Warning: Karras and Linear sigma tensors have different shapes. Blending might not work as expected.")
                # Ensure blending happens on tensors of the same device
                sigmas = (1.0 - blend_factor) * karras_sigmas + blend_factor * linear_sigmas

            elif mode == "kl_optimal":
                sigmas = kl_optimal_scheduler(
                    n=steps,
                    sigma_min=actual_sigma_min,
                    sigma_max=actual_sigma_max
                )
                sigmas = sigmas.to(device)
                print(f"DEBUG: kl_optimal sigmas generated: {sigmas}")
                print(f"DEBUG: kl_optimal sigmas has NaNs: {torch.isnan(sigmas).any()}")
                print(f"DEBUG: kl_optimal sigmas has Infs: {torch.isinf(sigmas).any()}")
            
            elif mode == "linear_quadratic":
                actual_threshold_noise = threshold_noise if threshold_noise is not None else 0.025
                actual_linear_steps = linear_steps if linear_steps is not None else steps // 2

                sigmas = linear_quadratic_schedule_adapted(
                    steps=steps,
                    sigma_max=actual_sigma_max, # Use the determined actual_sigma_max
                    threshold_noise=actual_threshold_noise,
                    linear_steps=actual_linear_steps
                )
                sigmas = sigmas.to(device)
                print(f"DEBUG: linear_quadratic sigmas generated: {sigmas}")
                print(f"DEBUG: linear_quadratic sigmas has NaNs: {torch.isnan(sigmas).any()}")
                print(f"DEBUG: linear_quadratic sigmas has Infs: {torch.isinf(sigmas).any()}")

            # --- NEW LOGIC TO SLICE SIGMAS BASED ON start_at_step AND end_at_step ---
            if sigmas is not None:
                # Clamp start_at_step and end_at_step to be within valid bounds
                # If steps is 20, sigmas will have 21 elements (0 to 20).
                # Slicing is typically [start_index : end_index (exclusive)]
                actual_start_idx = max(0, min(start_at_step, sigmas.shape[0] -1))
                # end_at_step is exclusive, so it can go up to sigmas.shape[0]
                actual_end_idx = max(actual_start_idx + 1, min(end_at_step, sigmas.shape[0])) # Ensure end_idx is at least start_idx + 1 if possible

                # If the user sets end_at_step to 9999, it should go to the end of the original sigmas.
                if end_at_step == 9999: # Special case for "go to end" default
                    actual_end_idx = sigmas.shape[0]

                sliced_sigmas = sigmas[actual_start_idx:actual_end_idx]

                print(f"DEBUG: Original sigmas shape: {sigmas.shape}")
                print(f"DEBUG: Slicing sigmas from index {actual_start_idx} to {actual_end_idx} (exclusive of end index)")
                print(f"DEBUG: Sliced sigmas shape: {sliced_sigmas.shape}")
                return (sliced_sigmas,) # Return the sliced sigmas
            # -------------------------------------------------------------------------

            return (sigmas,) # Fallback if sigmas was somehow not generated (shouldn't happen with current logic)

        except Exception as e:
            import traceback
            print(f"ERROR in HybridAdaptiveSigmas.generate: {e}")
            traceback.print_exc()
            raise e


# Register this beast into ComfyUI’s family tree
NODE_CLASS_MAPPINGS = {
    "HybridAdaptiveSigmas": HybridAdaptiveSigmas
}

# The name that shows up in your comfy dropdown like a seductive whisper
NODE_DISPLAY_NAME_MAPPINGS = {
    "HybridAdaptiveSigmas": "Hybrid Sigma Scheduler"
}