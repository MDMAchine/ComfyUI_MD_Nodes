# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# ████▓▒░ Hybrid Sigma Scheduler v0.69.420 🍆💦 ░▒▓████
# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
# ┌────────────────────────────────────────────────────────────────┐
# │  "Because standard schedulers are like decaf coffee."         │
# │  - Now with TWO distinct modes: ⚡ Karras Fury & 🧠 Linear Chill│
# └────────────────────────────────────────────────────────────────┘
# ░▒▓ Originally baked in a Compaq Presario by MDMAchine / MD_Nodes ▓▒░
#
# > What it does:
#     Outputs a tensor of sigmas to control diffusion noise levels—
#     without the guesswork, bad trips, or your ex’s drama.
#
# > What the hell is a sigma?
#     It's the noise standard deviation used in samplers.
#     Bigger sigma = more chaos. Smaller sigma = more clarity.
#
# > Modes:
#     🔥 karras_rho:
#         Uses comfy’s sexy `get_sigmas_karras()` function.
#         Controlled by the `rho` value to get that *chef’s kiss* curve.
#     🧊 adaptive_linear:
#         Straight-line decay from start to end like a sitcom marriage.
#
# > Pro Tip:
#     If your sound is fuzzier than a dial-up handshake, try lowering start_sigma or raising end_sigma.
#
# > LEGAL:
#     May cause unexpected vibes, spontaneous jamming, and mild enlightenment.
#     Combine with PingPongSampler at your own risk (or reward).
# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄

import torch
import comfy.model_management
from comfy.k_diffusion.sampling import get_sigmas_karras


class HybridAdaptiveSigmas:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                # Required for Karras functions and device context.
                "model": ("MODEL",),

                # How many sigma steps you want. Think of this like "resolution" for noise decay.
                "steps": ("INT", {
                    "default": 60, "min": 5, "max": 200,
                    "tooltip": "The number of steps from chaos to clarity. More steps = more precision (and more CPU tears)."
                }),

                # Choose your decay curve style.
                "mode": (["karras_rho", "adaptive_linear"], {
                    "tooltip": "🔥 karras_rho = rich, curvy chaos.\n🧊 adaptive_linear = budget-friendly flatline."
                }),

                # Passed downstream to samplers; doesn't affect sigmas directly.
                "denoise": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0,
                    "tooltip": "Denoising strength. 1.0 = max effect, 0.0 = placebo."
                }),

                # Top of the mountain, aka initial chaos.
                "start_sigma": ("FLOAT", {
                    "default": 1.0,
                    "tooltip": "Where your noise journey begins. Bigger = noisier."
                }),

                # Bottom of the well, aka enlightenment.
                "end_sigma": ("FLOAT", {
                    "default": 0.0,
                    "tooltip": "Where the noise stops. Zero = total inner peace (or radio silence)."
                }),

                # Only used if you're running with Karras Fury mode.
                "rho": ("FLOAT", {
                    "default": 2.5, "min": 1.0, "max": 15.0,
                    "tooltip": "Rho controls the Karras curve sharpness. Low = gentle slopes. High = noise rollercoaster."
                }),

                # 🛑 COMMENTED OUT FOR NOW – awaiting enlightenment or caffeine.
                # "tolerance": ("FLOAT", {"default": 1e-3, "min": 1e-6, "max": 0.1, "step": 0.0001}),
                # "min_step_factor": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 0.5}),
                # "max_step_factor": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 5.0}),
                # "euler_scale_factor": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01}),
                # "fixed_noise_epsilon": ("FLOAT", {"default": 1e-6, "min": 1e-7, "max": 1e-4, "step": 1e-7}),
            }
        }

    # Comfy expects this return type to plug into other diffusion nodes.
    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("sigmas",)
    FUNCTION = "generate"
    CATEGORY = "MD_Nodes/Schedulers"

    def generate(self, model, steps, mode, denoise, start_sigma, end_sigma, rho):
        try:
            # Get current CUDA/CPU device (so your tensor doesn't get homesick)
            device = comfy.model_management.get_torch_device()

            # Initialize sigmas to None, in case the vibes are off
            sigmas = None

            if mode == "karras_rho":
                # Use the comfy magic Karras function
                sigmas = get_sigmas_karras(
                    steps,
                    sigma_min=end_sigma,
                    sigma_max=start_sigma,
                    rho=rho,
                    device=device
                )

            elif mode == "adaptive_linear":
                # Flat and simple like your ex’s mixtape
                current_sigma = start_sigma
                adaptive_sigmas = [current_sigma]

                # Uniform decay – no twists, no surprises
                step_size = (start_sigma - end_sigma) / steps

                for _ in range(steps - 1):
                    current_sigma -= step_size
                    current_sigma = max(current_sigma, end_sigma)
                    adaptive_sigmas.append(current_sigma)

                # Wrap it up in a tensor like a noise burrito
                sigmas = torch.tensor(adaptive_sigmas, device=device)

            return (sigmas,)  # Comfy likes its tuples tupled.

        except Exception as e:
            # Oops! Something exploded. Call IT (or fix your math).
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
