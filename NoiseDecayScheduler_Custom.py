# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# ████ ADVANCED NOISE DECAY SCHEDULER v0.5.2 – Final? Build ████▓▒░
# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
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
#   - v0.3.1 “Refinement & Clarity Pass”:
#       • Consolidated property definitions
#       • Added `RETURN_NAMES` for UI clarity
#       • Mapped `NODE_DISPLAY_NAME_MAPPINGS` for consistency
#       • Documented `decay_power` with useful tooltip
#   - v0.5.1 "Advanced Features & Refactor":
#       • Implemented multiple decay algorithms: polynomial, sigmoidal, piecewise, and fourier.
#       • Added performance features: caching and temporal smoothing.
#       • Refactored code for better readability and maintainability.
#   - v0.5.2 "Final Fixes":
#       • Restored original class name to `NoiseDecayScheduler_Custom` for graph compatibility.
#       • Fixed critical syntax error in `INPUT_TYPES` to ensure node loads correctly.

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
from typing import List, Tuple, Optional, Any
import hashlib

class NoiseDecayScheduler_Custom:
    """
    Advanced Noise Decay Scheduler with customizable algorithms.
    This node serves as the main entry point for ComfyUI.
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
    DESCRIPTION = "Advanced noise decay scheduler with multiple algorithms and performance features."

    @classmethod
    def INPUT_TYPES(cls):
        """
        Defines inputs for the node UI.
        - algorithm_type: Selects the decay algorithm.
        - decay_exponent: Controls the steepness for polynomial and sigmoidal decay.
        - use_caching: Toggles caching of decay curves.
        - enable_temporal_smoothing: Applies a moving average filter.
        - custom_piecewise_points: A string of comma-separated floats for piecewise decay.
        - fourier_frequency: Controls the frequency for Fourier-based decay.
        """
        return {
            "required": {
                "algorithm_type": (
                    ["polynomial", "sigmoidal", "piecewise", "fourier"],
                    {
                        "default": "polynomial",
                        "tooltip": "Select the decay algorithm: polynomial, sigmoidal, piecewise, fourier."
                    }
                ),
                "decay_exponent": (
                    "FLOAT", {
                        "default": 2.0,
                        "min": 0.1,
                        "max": 10.0,
                        "step": 0.1,
                        "tooltip": "Exponent for polynomial decay or sigmoidal curve steepness."
                    }
                ),
                "use_caching": (
                    "BOOLEAN", {
                        "default": True,
                        "tooltip": "Enable caching of computed decay values to avoid redundant calculations."
                    }
                ),
                "enable_temporal_smoothing": (
                    "BOOLEAN", {
                        "default": False,
                        "tooltip": "Apply temporal filtering to smooth noise transitions."
                    }
                ),
            },
            "optional": {
                "custom_piecewise_points": (
                    "STRING", {
                        "default": "0.0,0.5,1.0",
                        "tooltip": "Custom piecewise decay points (comma-separated values). Only used with 'piecewise' algorithm."
                    }
                ),
                "fourier_frequency": (
                    "FLOAT", {
                        "default": 1.0,
                        "min": 0.1,
                        "max": 10.0,
                        "step": 0.1,
                        "tooltip": "Frequency for Fourier-based decay. Only used with 'fourier' algorithm."
                    }
                ),
            }
        }

    class NoiseDecayObject:
        """
        The actual scheduler implementation, which is returned by the node.
        This class must contain a 'get_decay' method.
        """
        def __init__(self,
                     algorithm_type: str,
                     decay_exponent: float,
                     use_caching: bool,
                     enable_temporal_smoothing: bool,
                     custom_piecewise_points: str,
                     fourier_frequency: float):

            self.algorithm_type = algorithm_type
            self.decay_exponent = decay_exponent
            self.use_caching = use_caching
            self.enable_temporal_smoothing = enable_temporal_smoothing
            self.fourier_frequency = fourier_frequency
            self._cache = {}

            # Parse custom piecewise points
            try:
                self.piecewise_points = [float(x.strip()) for x in custom_piecewise_points.split(",")]
            except (ValueError, TypeError):
                self.piecewise_points = [0.0, 0.5, 1.0]

        def _generate_cache_key(self, num_steps: int) -> str:
            """Generate a unique cache key based on all relevant parameters."""
            params_str = f"{self.algorithm_type}_{self.decay_exponent}_{self.fourier_frequency}_{self.enable_temporal_smoothing}_{num_steps}"
            if self.algorithm_type == "piecewise":
                params_str += f"_{','.join(map(str, self.piecewise_points))}"
            return hashlib.md5(params_str.encode()).hexdigest()

        def _apply_temporal_smoothing(self, decay_array: np.ndarray, window_size: int = 3) -> np.ndarray:
            """Apply a simple moving average filter for smoothing."""
            if len(decay_array) <= window_size:
                return decay_array
            
            # Simple moving average smoothing
            smoothed = np.convolve(decay_array, np.ones(window_size)/window_size, mode='same')
            return smoothed

        def _compute_polynomial_decay(self, num_steps: int) -> np.ndarray:
            """Compute polynomial decay."""
            normalized_steps = np.linspace(0.0, 1.0, num_steps)
            decay_values = np.power(1.0 - normalized_steps, self.decay_exponent)
            return decay_values

        def _compute_sigmoidal_decay(self, num_steps: int) -> np.ndarray:
            """Compute sigmoidal decay."""
            normalized_steps = np.linspace(0.0, 1.0, num_steps)
            # Adjust the input range for the sigmoid function for a clearer curve
            x = (normalized_steps - 0.5) * self.decay_exponent * 4
            sigmoid = 1 / (1 + np.exp(x))
            # Normalize to go from a high value to a low value
            return 1.0 - sigmoid

        def _compute_piecewise_decay(self, num_steps: int) -> np.ndarray:
            """Compute piecewise decay with custom points."""
            normalized_steps = np.linspace(0.0, 1.0, num_steps)
            if len(self.piecewise_points) >= 2:
                x_points = np.linspace(0, 1, len(self.piecewise_points))
                y_points = np.array(self.piecewise_points)
                decay_values = np.interp(normalized_steps, x_points, y_points)
            else:
                decay_values = self._compute_polynomial_decay(num_steps)
            return decay_values

        def _compute_fourier_decay(self, num_steps: int) -> np.ndarray:
            """Compute Fourier-based decay."""
            normalized_steps = np.linspace(0.0, 1.0, num_steps)
            # Create a sine wave and normalize it to go from 1.0 to 0.0
            decay_values = np.cos(self.fourier_frequency * np.pi * normalized_steps)
            decay_values = (decay_values + 1) / 2
            return decay_values

        def get_decay(self, num_steps: int) -> np.ndarray:
            """
            ComfyUI standard method to get the decay schedule.
            This method computes the decay curve, applies smoothing and caching.
            """
            if self.use_caching:
                cache_key = self._generate_cache_key(num_steps)
                if cache_key in self._cache:
                    return self._cache[cache_key]

            if self.algorithm_type == "polynomial":
                decay_values = self._compute_polynomial_decay(num_steps)
            elif self.algorithm_type == "sigmoidal":
                decay_values = self._compute_sigmoidal_decay(num_steps)
            elif self.algorithm_type == "piecewise":
                decay_values = self._compute_piecewise_decay(num_steps)
            elif self.algorithm_type == "fourier":
                decay_values = self._compute_fourier_decay(num_steps)
            else:
                # Fallback to polynomial decay if algorithm is not recognized
                decay_values = self._compute_polynomial_decay(num_steps)

            if self.enable_temporal_smoothing:
                decay_values = self._apply_temporal_smoothing(decay_values)

            if self.use_caching:
                self._cache[cache_key] = decay_values

            return decay_values

    def generate(self,
                 algorithm_type: str,
                 decay_exponent: float,
                 use_caching: bool,
                 enable_temporal_smoothing: bool,
                 custom_piecewise_points: str,
                 fourier_frequency: float):
        """
        Main node function that instantiates and returns the scheduler object.
        """
        scheduler_obj = self.NoiseDecayObject(
            algorithm_type=algorithm_type,
            decay_exponent=decay_exponent,
            use_caching=use_caching,
            enable_temporal_smoothing=enable_temporal_smoothing,
            custom_piecewise_points=custom_piecewise_points,
            fourier_frequency=fourier_frequency
        )
        return (scheduler_obj,)


# Register mappings for ComfyUI to discover this node
NODE_CLASS_MAPPINGS = {
    "NoiseDecayScheduler_Custom": NoiseDecayScheduler_Custom,
}

# Friendly display name override in node editor
NODE_DISPLAY_NAME_MAPPINGS = {
    "NoiseDecayScheduler_Custom": "Noise Decay Scheduler (Advanced)",
}
