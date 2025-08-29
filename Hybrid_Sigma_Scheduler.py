# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# ████ Hybrid Sigma Scheduler v0.71 – Optimized for Ace-Step Audio/Video ████▓▒░
# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀

# ░▒▓ ORIGIN & DEV:
#    • Brewed in a Compaq Presario by MDMAchine / MD_Nodes
#    • License: Apache 2.0 — legal chaos with community spirit

# ░▒▓ DESCRIPTION:
#    Outputs a tensor of sigmas to control diffusion noise levels.
#    No guesswork. No drama. Just pure generative groove.
#    Designed for precision noise scheduling in ComfyUI workflows.
#    May perform differently outside of intended models.

# ░▒▓ FEATURES:
#    ✓ Multiple core modes: Karras, Linear, Polynomial, Blended, and more
#    ✓ Automatic sigma range detection from the loaded model
#    ✓ Manual override for sigma min/max for expert control
#    ✓ Precision schedule slicing with `start_at_step` and `end_at_step`
#    ✓ Outputs the actual number of steps after slicing for downstream samplers
#    ✓ Optional schedule reversal for experimental workflows

# ░▒▓ CHANGELOG:
#    - v0.1 (Initial Brew): Core sigma scheduling implemented.
#    - v0.5: Added blended curve mode.
#    - v0.6: Introduced kl_optimal and linear_quadratic modes.
#    - v0.69.420.1: Refined blending algorithm.
#    - v0.70:
#         • Added 'model' input for dynamic sigma_min/max retrieval.
#         • Introduced 'start_at_step' and 'end_at_step' for slicing.
#         • Clarified 'denoise' parameter role as a passthrough.
#    - v0.71 (Current Release):
#         • Added 'polynomial' scheduler mode with a 'power' control.
#         • Added 'actual_steps' integer output for sliced schedule length.
#         • Added 'reverse_sigmas' toggle for experimental workflows.
#         • Refactored code for clarity and maintainability (DRY principle).
#         • Improved console logging for easier debugging.

# ░▒▓ CONFIGURATION:
#    → Primary Use: Fine-grained diffusion noise scheduling for generative workflows
#    → Secondary Use: Experimental noise shaping and curve blending
#    → Edge Use: Hardcore noise tinkerers and sampling theorists

# ░▒▓ WARNING:
#    This node may trigger:
#    ▓▒░ Unexpected vibes
#    ▓▒░ Spontaneous beatboxing
#    ▓▒░ Mild to moderate enlightenment

# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄


import torch
import comfy.model_management
from comfy.k_diffusion.sampling import get_sigmas_karras
import math

# Referenced from https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/15608
def kl_optimal_scheduler(n: int, sigma_min: float, sigma_max: float) -> torch.Tensor:
    sigma_min_t = torch.tensor(sigma_min, dtype=torch.float32)
    sigma_max_t = torch.tensor(sigma_max, dtype=torch.float32)
    adj_idxs = torch.arange(n, dtype=torch.float32).div_(n - 1)
    sigmas = adj_idxs.new_zeros(n + 1)
    sigmas[:-1] = (adj_idxs * torch.atan(sigma_min_t) + (1 - adj_idxs) * torch.atan(sigma_max_t)).tan_()
    return sigmas

# Adapted from: https://github.com/genmoai/models/blob/main/src/mochi_preview/infer.py#L41
def linear_quadratic_schedule_adapted(steps: int, sigma_max: float, threshold_noise: float = 0.025, linear_steps: int = None) -> torch.Tensor:
    if steps <= 0: return torch.FloatTensor([sigma_max, 0.0])
    if steps == 1: return torch.FloatTensor([sigma_max, 0.0])
    if linear_steps is None: linear_steps_actual = steps // 2
    else: linear_steps_actual = max(0, min(linear_steps, steps))
    sigma_schedule_raw = []
    if linear_steps_actual == 0:
        for i in range(steps):
            if steps > 1: sigma_schedule_raw.append((i / (steps - 1.0))**2)
            else: sigma_schedule_raw.append(0.0)
    else:
        for i in range(linear_steps_actual):
            sigma_schedule_raw.append(i * threshold_noise / linear_steps_actual)
        quadratic_steps = steps - linear_steps_actual
        if quadratic_steps > 0:
            threshold_noise_step_diff = linear_steps_actual - threshold_noise * steps
            quadratic_coef = threshold_noise_step_diff / (linear_steps_actual * quadratic_steps ** 2)
            linear_coef = threshold_noise / linear_steps_actual - 2 * threshold_noise_step_diff / (quadratic_steps ** 2)
            const = quadratic_coef * (linear_steps_actual ** 2)
            for i in range(linear_steps_actual, steps):
                sigma_schedule_raw.append(quadratic_coef * (i ** 2) + linear_coef * i + const)
    if not sigma_schedule_raw or sigma_schedule_raw[-1] != 1.0: sigma_schedule_raw.append(1.0)
    sigma_schedule_inverted = [1.0 - x for x in sigma_schedule_raw]
    return torch.FloatTensor(sigma_schedule_inverted) * sigma_max


class HybridAdaptiveSigmas:
    # --- Fallback constants for models without sigma metadata ---
    FALLBACK_SIGMA_MIN = 0.029167 # SD1.5 default
    FALLBACK_SIGMA_MAX = 14.61467  # SD1.5 default

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The diffusion model, used to get inherent sigma_min and sigma_max."}),
                "steps": ("INT", {
                    "default": 60, "min": 5, "max": 200,
                    "tooltip": "The number of steps from chaos to clarity. More steps = more precision."
                }),
                "mode": (["karras_rho", "adaptive_linear", "blended_curves", "kl_optimal", "linear_quadratic", "polynomial"], {
                    "tooltip": "🔥 karras_rho: Curvy chaos.\n🧊 adaptive_linear: Simple straight line.\n💖 blended_curves: Mix Karras & Linear.\n✨ kl_optimal: 'Align Your Steps' paper schedule.\n📈 linear_quadratic: Two-phase schedule.\n📐 polynomial: Power-based curve."
                }),
                "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000, "tooltip": "Start denoising from this step (inclusive)."}),
                "end_at_step": ("INT", {"default": 9999, "min": 0, "max": 10000, "tooltip": "End denoising at this step (exclusive). 9999 means to the very end."}),
            },
            "optional": {
                "denoise": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0,
                    "tooltip": "Denoising strength. Passed downstream to samplers; doesn't affect sigmas directly."
                }),
                "start_sigma_override": ("FLOAT", {
                    "default": 1.000, "min": 0.0, "max": 20.0, "step": 0.001, "optional": True,
                    "tooltip": "Optional override for the schedule's maximum noise (sigma_max). E.g., 14.6 for SD1.5."
                }),
                "end_sigma_override": ("FLOAT", {
                    "default": 0.006, "min": 0.0, "max": 20.0, "step": 0.001, "optional": True,
                    "tooltip": "Optional override for the schedule's minimum noise (sigma_min). E.g., 0.02 for SD1.5."
                }),
                "rho": ("FLOAT", {
                    "default": 1.5, "min": 1.0, "max": 15.0,
                    "tooltip": "For 'karras_rho' mode. Controls curve sharpness. Low = gentle, High = steep."
                }),
                "blend_factor": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "For 'blended_curves' mode. 0.0 = pure Karras, 1.0 = pure Linear."
                }),
                "power": ("FLOAT", {
                    "default": 2.0, "min": 0.1, "max": 8.0, "step": 0.1,
                    "tooltip": "For 'polynomial' mode. Controls the curve exponent. 1.0 is linear."
                }),
                "threshold_noise": ("FLOAT", {
                    "default": 0.025, "min": 0.0, "max": 1.0, "step": 0.001,
                    "tooltip": "For 'linear_quadratic' mode. The pivot point where the curve transitions from linear to quadratic."
                }),
                "linear_steps": ("INT", {
                    "default": None, "min": 0, "max": 200,
                    "tooltip": "For 'linear_quadratic' mode. Number of linear steps before the quadratic part begins."
                }),
                "reverse_sigmas": ("BOOLEAN", {"default": False, "tooltip": "Reverse the schedule to go from low noise to high noise."}),
            }
        }

    RETURN_TYPES = ("SIGMAS", "INT")
    RETURN_NAMES = ("sigmas", "actual_steps")
    FUNCTION = "generate"
    CATEGORY = "MD_Nodes/Schedulers"

    def _generate_linear_sigmas(self, steps, sigma_min, sigma_max, device):
        """Helper to generate a linear sigma schedule efficiently."""
        return torch.linspace(sigma_max, sigma_min, steps + 1).to(device)

    def generate(self, model, steps, mode, start_at_step, end_at_step,
                 denoise=1.0, start_sigma_override=None, end_sigma_override=None,
                 rho=1.5, blend_factor=0.5, power=2.0,
                 threshold_noise=None, linear_steps=None, reverse_sigmas=False):
        try:
            device = comfy.model_management.get_torch_device()
            actual_sigma_min, actual_sigma_max = None, None

            # --- Determine actual sigma_min/max ---
            try:
                model_sampling_obj = model.get_model_object("model_sampling")
                actual_sigma_min = model_sampling_obj.sigma_min
                actual_sigma_max = model_sampling_obj.sigma_max
                if abs(actual_sigma_max - actual_sigma_min) < 1e-5 or actual_sigma_max < 0.1:
                    print(f"Hybrid Sigma Scheduler WARNING: Model reported unusual sigma range ({actual_sigma_min} to {actual_sigma_max}). Checking for overrides.")
                    actual_sigma_min, actual_sigma_max = None, None
            except Exception as e:
                print(f"Hybrid Sigma Scheduler WARNING: Could not read sigmas from model ({e}). Checking for overrides or using fallback.")
            
            if start_sigma_override is not None:
                actual_sigma_max = start_sigma_override
            if end_sigma_override is not None:
                actual_sigma_min = end_sigma_override

            if actual_sigma_min is None or actual_sigma_max is None:
                actual_sigma_min = self.FALLBACK_SIGMA_MIN
                actual_sigma_max = self.FALLBACK_SIGMA_MAX
                print(f"Hybrid Sigma Scheduler WARNING: Using fallback sigmas: min={actual_sigma_min}, max={actual_sigma_max}.")
            
            if actual_sigma_min >= actual_sigma_max:
                print(f"Hybrid Sigma Scheduler ERROR: sigma_min ({actual_sigma_min}) >= sigma_max ({actual_sigma_max}). Adjusting to prevent crash.")
                actual_sigma_max = actual_sigma_min + 0.1
                if actual_sigma_max > 20.0:
                    actual_sigma_max = 20.0
                    actual_sigma_min = 19.9

            print(f"Hybrid Sigma Scheduler INFO: Final sigma range: min={actual_sigma_min:.4f}, max={actual_sigma_max:.4f}")

            # --- Generate Sigmas Based on Mode ---
            sigmas = None
            if mode == "karras_rho":
                sigmas = get_sigmas_karras(steps, actual_sigma_min, actual_sigma_max, rho, device)
            elif mode == "adaptive_linear":
                sigmas = self._generate_linear_sigmas(steps, actual_sigma_min, actual_sigma_max, device)
            elif mode == "polynomial":
                sigmas = torch.linspace(actual_sigma_max**(1/power), actual_sigma_min**(1/power), steps + 1).pow(power).to(device)
            elif mode == "blended_curves":
                karras_sigmas = get_sigmas_karras(steps, actual_sigma_min, actual_sigma_max, rho, device)
                linear_sigmas = self._generate_linear_sigmas(steps, actual_sigma_min, actual_sigma_max, device)
                sigmas = (1.0 - blend_factor) * karras_sigmas + blend_factor * linear_sigmas
            elif mode == "kl_optimal":
                sigmas = kl_optimal_scheduler(steps, actual_sigma_min, actual_sigma_max).to(device)
            elif mode == "linear_quadratic":
                actual_threshold = threshold_noise if threshold_noise is not None else 0.025
                actual_linear_steps = linear_steps if linear_steps is not None else steps // 2
                sigmas = linear_quadratic_schedule_adapted(steps, actual_sigma_max, actual_threshold, actual_linear_steps).to(device)

            # --- Slice Sigmas Based on Start/End Step ---
            if sigmas is not None:
                total_sigmas = sigmas.shape[0]
                actual_start_idx = max(0, min(start_at_step, total_sigmas - 1))
                actual_end_idx = total_sigmas if end_at_step >= total_sigmas or end_at_step > 9000 else max(actual_start_idx + 1, min(end_at_step, total_sigmas))
                
                sliced_sigmas = sigmas[actual_start_idx:actual_end_idx]
                
                # Reverse if requested
                if reverse_sigmas:
                    sliced_sigmas = torch.flip(sliced_sigmas, dims=[0])

                # A schedule for N steps has N+1 sigma values
                actual_step_count = sliced_sigmas.shape[0] - 1
                if actual_step_count < 1:
                    print(f"Hybrid Sigma Scheduler WARNING: Slicing resulted in < 1 step ({actual_step_count}). This may cause errors.")

                print(f"Hybrid Sigma Scheduler INFO: Original steps: {steps}, Sliced to {actual_step_count} steps (from index {actual_start_idx} to {actual_end_idx}).")
                return (sliced_sigmas, actual_step_count)

            # Fallback return
            return (torch.tensor([actual_sigma_max, actual_sigma_min]), 1)

        except Exception as e:
            import traceback
            print(f"ERROR in HybridAdaptiveSigmas.generate: {e}")
            traceback.print_exc()
            raise e

# --- Node Registration ---
NODE_CLASS_MAPPINGS = { "HybridAdaptiveSigmas": HybridAdaptiveSigmas }
NODE_DISPLAY_NAME_MAPPINGS = { "HybridAdaptiveSigmas": "Hybrid Sigma Scheduler" }