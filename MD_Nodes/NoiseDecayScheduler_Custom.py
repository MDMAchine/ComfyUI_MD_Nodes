# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# ████ NOISE DECAY SCHEDULER v0.3.1 – released to the wild ████▓▒░
# ░▒▓ Crafted in the fumes of dial-up and hot solder smell ░▒▓
# ▓▒░        Inspired by: Blepping? | https://github.com/blepping
# ░▒▓        Originally bungled by: MDMAchine
# ▒░▓        Made not-suck by: devstral (local l33t)
# ░▒▓        License: Apache 2.0 like anyone actually reads that
# ▓▒░
# ░▒▓ Description:
#    This ComfyUI scheduler node outputs a cosine-based decay curve
#    raised to your almighty `decay_power`. Meant for use with
#    pingpongsampler_custom or anyone stuck in aesthetic purgatory.
#    **** - https://gist.github.com/MDMAchine/7edf8244c7cead4082f4168cdd8b2b23
#
# ▓▒░ Changes:
# - V0.1 (OG release, barely functional):
#    * Required both `sigmas` and `decay_power`
#    * Returned a raw tensor in a dict – like it’s 1995 and we’re
#      scared of objects
#
# - V0.2 (object-oriented enlightenment):
#    * Killed the `sigmas` input – sampler does the legwork now
#    * Added a scheduler class with `.get_decay(num_steps)` method
#    * Output changed from tensor dump to actual `SCHEDULER` object
#
# - V0.3 (ComfyUI assimilation complete):
#    * Renamed to `NoiseDecayScheduler_Custom_V03` (internally) for version hygiene
#    * Dropped Torch like it’s hot – now 100% NumPy-powered
#    * Lazy compute decay – won’t lift a finger until sampler calls
#
# - V0.3.1 (Refinement & Clarity Pass):
#    * Consolidated node property definitions for cleaner code.
#    * Explicitly added `RETURN_NAMES` for clearer UI output labeling.
#    * Updated internal `NODE_DISPLAY_NAME_MAPPINGS` for consistency.
#    * Added comprehensive tooltip for `decay_power`.
#    * No functional changes to the decay logic, just structural cleanup.
#
# ░▒▓ Use at your own risk. May cause creative hallucinations,
#    recursive memes, or spontaneous understanding of noise decay.
#    Remember to feed your scheduler fresh packets daily.
# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄

import numpy as np  # Numerical operations for decay curve

class NoiseDecayScheduler_Custom_V03:
    """
    Node wrapper for ComfyUI to output a noise-decay scheduler.

    - INPUT: decay_power (controls curvature of cosine decay)
    - OUTPUT: scheduler object implementing get_decay(num_steps)
    """

    # Node category in ComfyUI UI
    CATEGORY = "schedulers/custom"

    # Function name for ComfyUI
    FUNCTION = "generate"

    # Output types for the node
    RETURN_TYPES = ("SCHEDULER",)

    # Friendly display name override in node editor
    RETURN_NAMES = ("scheduler",)

    # Description of the node
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
                        "default": 3.0,     # default exponent for cosine decay
                        "min": 0.1,         # minimum allowed exponent
                        "max": 10.0,        # maximum allowed exponent
                        "step": 0.1,        # UI slider step size
                        "tooltip": "Controls the curvature of the noise decay curve. Higher values (e.g., 5.0-10.0) result in a steeper decay, making noise reduce more quickly, which can lead to sharper, more detailed results. Lower values (e.g., 0.1-2.0) create a gentler decay, prolonging the influence of noise and potentially leading to softer or more experimental outputs. This parameter directly influences how 'ancestral' noise is faded out during sampling, especially useful with the PingPongSampler."
                    }
                ),
            }
        }

    class NoiseDecayObject:
        """
        Scheduler implementation that lazily computes a cosine decay curve.
        """
        def __init__(self, decay_power: float):
            # Store the exponent for shaping the cosine curve
            self.decay_power = decay_power

        def get_decay(self, num_steps: int) -> np.ndarray:
            """
            Compute and return a decay array of length `num_steps`.

            Uses:
              decay[i] = cos( linspace(0, π/2)[i] ) ** decay_power
            """
            # Create linearly spaced angles from 0 to π/2
            lin = np.linspace(0, np.pi / 2, num_steps)
            # Raise cosine of each angle to the decay_power
            return np.cos(lin) ** self.decay_power

    def generate(self, decay_power: float):
        """
        Node function: instantiate and return the NoiseDecayObject.

        Args:
          decay_power: exponent controlling decay steepness

        Returns:
          Tuple containing the scheduler object
        """
        scheduler_obj = self.NoiseDecayObject(decay_power)
        return (scheduler_obj,)


# Register mappings for ComfyUI to discover this node
NODE_CLASS_MAPPINGS = {
    "NoiseDecayScheduler_Custom": NoiseDecayScheduler_Custom_V03,
}

# Friendly display name override in node editor
# This is usually optional if the class name is descriptive enough,
# but it's good practice to keep it for explicit control.
NODE_DISPLAY_NAME_MAPPINGS = {
    "NoiseDecayScheduler_Custom": "Noise Decay Scheduler (Custom)",
}