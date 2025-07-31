# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# ████ NOISE DECAY SCHEDULER v0.4.1 – Released to the Wild ████▓▒░
# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀

# ░▒▓ ORIGIN & DEV:
#   • Crafted in the fumes of dial-up and hot solder smell
#   • Inspired by: Blepping? | https://github.com/blepping
#   • Originally bungled by: MDMAchine
#   • Made not-suck by: devstral (local l33t)
#   • License: Apache 2.0 — Because we’re polite anarchists

# ░▒▓ DESCRIPTION:
#   A ComfyUI scheduler that outputs a cosine-based decay curve,
#   raised to your almighty `decay_power`. Perfect for:
#     - pingpongsampler_custom
#     - Escaping aesthetic purgatory
#     - Those who say “vibe” unironically
#   Example usage & details: https://gist.github.com/MDMAchine/7edf8244c7cead4082f4168cdd8b2b23

# ░▒▓ CHANGELOG HIGHLIGHTS:
#   - v0.1 “OG Release – Jank Included”:
#       • Required `sigmas` and `decay_power`
#       • Output as raw tensor in dict (1995-era vibes)
#   - v0.2 “Object-Oriented Enlightenment”:
#       • Dropped `sigmas` input; samplers handle it now
#       • Added `.get_decay(num_steps)` method
#       • Output: structured `SCHEDULER` object
#   - v0.3 “ComfyUI-ification Complete”:
#       • Renamed to `NoiseDecayScheduler_Custom_V03`
#       • Torch removed—100% NumPy implementation
#       • Lazy evaluation—no decay math until requested
#   - v0.4.1 “Refinement & Stability Pass”:
#       • Added input validation and error handling
#       • Implemented memoization for repeated calculations
#       • Improved documentation and UI clarity
#       • Added type hints for better code maintainability

# ░▒▓ CONFIGURATION:
#   → Primary Use: Fine-tuning decay curves in custom schedulers
#   → Secondary Use: Enhancing sampler behaviors with smooth decay
#   → Edge Use: For those chasing that perfect vibe decay

# ░▒▓ WARNING:
#   This scheduler may trigger:
#   ▓▒░ Creative hallucinations
#   ▓▒░ Recursive memes
#   ▓▒░ Sudden comprehension of “vibe decay”
#   Remember: clean packets and chaotic intentions required.

# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄

import numpy as np
from typing import Tuple


class NoiseDecayScheduler_Custom_V04:
    """
    Enhanced noise decay scheduler for ComfyUI with validation and memoization.

    - INPUT: decay_power (controls curvature of cosine decay)
    - OUTPUT: scheduler object implementing get_decay(num_steps)

    Example usage:
        >>> from ComfyUI import LoadImage, PingPongSampler
        >>> img_node = LoadImage("image.png")
        >>> sampler_node = PingPongSampler()
        >>> scheduler_node = NoiseDecayScheduler_Custom_V04(decay_power=4.2)
        >>> sampler_node.scheduler = scheduler_node.scheduler
        >>> noise = sampler_node.generate_noise(img_node.image, steps=512)
    """

    # Node configuration constants
    CATEGORY = "schedulers/custom"
    FUNCTION = "generate"
    RETURN_TYPES = ("SCHEDULER",)
    RETURN_NAMES = ("scheduler",)
    DESCRIPTION = "Custom noise decay scheduler with cosine curve for PingPongSampler"

    @classmethod
    def INPUT_TYPES(cls):
        """
        Defines inputs for the node UI.
        Only a single `decay_power` parameter to shape the decay curve.
        """
        return {
            "required": {
                "decay_power": (
                    "FLOAT", {
                        "default": 3.0,
                        "min": 0.1,
                        "max": 10.0,
                        "step": 0.1,
                        "tooltip": (
                            "Controls the curvature of the noise decay curve.\n"
                            "Higher values (e.g., 5.0-10.0) result in a steeper decay,\n"
                            "making noise reduce more quickly for sharper results.\n"
                            "Lower values (e.g., 0.1-2.0) create a gentler decay,\n"
                            "prolonging noise influence for softer outputs."
                        ),
                    }
                ),
            },
        }

    class NoiseDecayScheduler:
        """
        Scheduler implementation with memoization and validation.
        """

        def __init__(self, decay_power: float):
            self._validate_decay_power(decay_power)
            self.decay_power = decay_power
            self._memo = None

        def _validate_decay_power(self, value: float) -> None:
            """Internal validation for decay_power parameter."""
            if not 0.1 <= value <= 10.0:
                raise ValueError(
                    f"decay_power {value} must be between 0.1 and 10.0"
                )

        def get_decay(self, num_steps: int) -> np.ndarray:
            """
            Compute and return a decay array of length `num_steps` with memoization.

            Formula:
                decay[i] = cos(linspace(0, π/2)[i]) ** decay_power
            """
            if self._memo is None or len(self._memo) != num_steps:
                lin = np.linspace(0, np.pi / 2, num_steps)
                self._memo = np.cos(lin) ** self.decay_power

            return self._memo.copy()

    def generate(self, decay_power: float) -> Tuple[NoiseDecayScheduler]:
        """
        Node function: instantiate and return the NoiseDecayScheduler.

        Args:
            decay_power: exponent controlling decay steepness

        Returns:
            Tuple containing the scheduler object
        """
        return NoiseDecayScheduler(decay_power)

    # Register node mappings for ComfyUI
    NODE_CLASS_MAPPINGS = {
        "NoiseDecayScheduler_Custom": NoiseDecayScheduler_Custom_V04,
    }


# Register node display names
NODE_DISPLAY_NAME_MAPPINGS = {
    "NoiseDecayScheduler_Custom": "Noise Decay Scheduler (Enhanced)",
}


# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
