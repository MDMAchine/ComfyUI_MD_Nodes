# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# ████ ADVANCED NOISE DECAY SCHEDULER v0.6.0 – Enhanced Build ████▓▒░
# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
# ░▒▓ ORIGIN & DEV:
#   • Crafted in the fumes of dial-up and hot solder smell
#   • Inspired by: Blepping? | https://github.com/blepping
#   • Originally bungled by: MDMAchine
#   • Made not-suck by: devstral (local l33t)
#   • License: Apache 2.0 — Because we’re polite anarchists
#
# ░▒▓ DESCRIPTION:
#   A ComfyUI scheduler that outputs a customizable decay curve.
#   Perfect for custom samplers, aesthetic exploration, and fine-tuning
#   the generative process. Now with more algorithms and curve manipulation!
#
# ░▒▓ CHANGELOG HIGHLIGHTS (v0.6.0):
#   - NEW ALGORITHMS: Added 'exponential' and 'gaussian' decay types.
#   - ENHANCED CONTROL: Added 'start_value', 'end_value', and 'invert_curve'
#     for complete control over the curve's range and shape.
#   - CONFIGURABLE SMOOTHING: 'smoothing_window' is now a user parameter.
#
# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄

import numpy as np
from typing import List, Tuple, Optional, Any
import hashlib

class NoiseDecayScheduler_Custom:
    CATEGORY = "schedulers/custom"
    FUNCTION = "generate"
    RETURN_TYPES = ("SCHEDULER",)
    RETURN_NAMES = ("scheduler",)
    DESCRIPTION = "Advanced noise decay scheduler with multiple algorithms and performance features."

    @classmethod
    def INPUT_TYPES(cls):
        """
        Defines inputs for the node UI.
        """
        return {
            "required": {
                "algorithm_type": (
                    # ✨ NEW: Added exponential and gaussian algorithms
                    ["polynomial", "sigmoidal", "piecewise", "fourier", "exponential", "gaussian"],
                    {"default": "polynomial"}
                ),
                "decay_exponent": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 10.0, "step": 0.1, "tooltip": "Exponent/steepness for polynomial, sigmoidal, or exponential decay."}),
                "start_value": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.01, "tooltip": "The starting value of the decay curve."}), # ✨ NEW
                "end_value": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01, "tooltip": "The final value of the decay curve."}), # ✨ NEW
                "invert_curve": ("BOOLEAN", {"default": False, "tooltip": "Invert the curve's shape (e.g., for noise injection)."}), # ✨ NEW
                "use_caching": ("BOOLEAN", {"default": True, "tooltip": "Enable caching of computed decay values to avoid redundant calculations."}),
                "enable_temporal_smoothing": ("BOOLEAN", {"default": False, "tooltip": "Apply temporal filtering to smooth noise transitions."}),
            },
            "optional": {
                "smoothing_window": ("INT", {"default": 3, "min": 2, "max": 20, "step": 1, "tooltip": "Window size for temporal smoothing."}), # ✨ NEW
                "custom_piecewise_points": ("STRING", {"default": "1.0,0.5,0.0", "tooltip": "Comma-separated values for 'piecewise' algorithm."}),
                "fourier_frequency": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1, "tooltip": "Frequency for 'fourier' algorithm."}),
            }
        }

    class NoiseDecayObject:
        """
        The actual scheduler implementation, which is returned by the node.
        """
        def __init__(self, **kwargs):
            # Use kwargs to easily handle all parameters
            self.algorithm_type = kwargs.get("algorithm_type", "polynomial")
            self.decay_exponent = kwargs.get("decay_exponent", 2.0)
            self.start_value = kwargs.get("start_value", 1.0)
            self.end_value = kwargs.get("end_value", 0.0)
            self.invert_curve = kwargs.get("invert_curve", False)
            self.use_caching = kwargs.get("use_caching", True)
            self.enable_temporal_smoothing = kwargs.get("enable_temporal_smoothing", False)
            self.smoothing_window = kwargs.get("smoothing_window", 3)
            self.custom_piecewise_points = kwargs.get("custom_piecewise_points", "1.0,0.5,0.0")
            self.fourier_frequency = kwargs.get("fourier_frequency", 1.0)
            self._cache = {}

            try:
                self.piecewise_points = [float(x.strip()) for x in self.custom_piecewise_points.split(",")]
            except (ValueError, TypeError):
                self.piecewise_points = [1.0, 0.5, 0.0]

        def _generate_cache_key(self, num_steps: int) -> str:
            """Generate a unique cache key based on all relevant parameters."""
            params = (
                self.algorithm_type, self.decay_exponent, self.start_value, self.end_value,
                self.invert_curve, self.enable_temporal_smoothing, self.smoothing_window,
                self.fourier_frequency, num_steps, ','.join(map(str, self.piecewise_points))
            )
            params_str = '_'.join(map(str, params))
            return hashlib.md5(params_str.encode()).hexdigest()

        def _apply_temporal_smoothing(self, decay_array: np.ndarray) -> np.ndarray:
            """Apply a simple moving average filter with a configurable window."""
            if len(decay_array) < self.smoothing_window or self.smoothing_window < 2:
                return decay_array
            
            # Use a convolution for an efficient moving average
            return np.convolve(decay_array, np.ones(self.smoothing_window) / self.smoothing_window, mode='same')

        # --- Base Curve Computations (Normalized from 1.0 to 0.0) ---

        def _compute_polynomial_decay(self, num_steps: int) -> np.ndarray:
            """Compute polynomial decay."""
            x = np.linspace(0.0, 1.0, num_steps)
            return (1.0 - x) ** self.decay_exponent

        def _compute_sigmoidal_decay(self, num_steps: int) -> np.ndarray:
            """Compute sigmoidal decay."""
            x = np.linspace(-1.0, 1.0, num_steps) * (self.decay_exponent / 2) * 2.5
            sigmoid = 1 / (1 + np.exp(-x))
            # Rescale to ensure it precisely spans 1.0 to 0.0
            scaled = (sigmoid - sigmoid.min()) / (sigmoid.max() - sigmoid.min())
            return 1.0 - scaled

        def _compute_piecewise_decay(self, num_steps: int) -> np.ndarray:
            """Compute piecewise decay with custom points."""
            if len(self.piecewise_points) < 2:
                return self._compute_polynomial_decay(num_steps)
            
            x_points = np.linspace(0, 1, len(self.piecewise_points))
            y_points = np.array(self.piecewise_points)
            
            # The base curve should be normalized 1->0 before final scaling
            base_curve = np.interp(np.linspace(0, 1, num_steps), x_points, y_points)
            min_val, max_val = base_curve.min(), base_curve.max()
            if max_val - min_val > 1e-6: # Avoid division by zero
                return (base_curve - min_val) / (max_val - min_val)
            return base_curve

        def _compute_fourier_decay(self, num_steps: int) -> np.ndarray:
            """Compute Fourier-based (cosine) decay."""
            x = np.linspace(0.0, 1.0, num_steps)
            return (np.cos(self.fourier_frequency * np.pi * x) + 1) / 2

        # ✨ NEW: Exponential decay algorithm
        def _compute_exponential_decay(self, num_steps: int) -> np.ndarray:
            """Compute exponential decay."""
            x = np.linspace(0.0, 1.0, num_steps)
            return np.exp(-self.decay_exponent * x)

        # ✨ NEW: Gaussian (inverted bell curve) decay algorithm
        def _compute_gaussian_decay(self, num_steps: int) -> np.ndarray:
            """Compute inverted Gaussian decay."""
            x = np.linspace(-1.0, 1.0, num_steps)
            # The exponent controls the 'width' of the bell
            sigma = 1.0 / (self.decay_exponent * 0.5)
            # This creates a curve that is low at the ends and 1.0 in the middle
            bell_curve = np.exp(-(x**2) / (2 * sigma**2))
            # We invert it to create a decay curve
            return 1.0 - bell_curve

        def get_decay(self, num_steps: int) -> np.ndarray:
            """
            ComfyUI standard method to get the decay schedule.
            """
            if self.use_caching:
                cache_key = self._generate_cache_key(num_steps)
                if cache_key in self._cache:
                    return self._cache[cache_key]

            # 1. Select and compute the base decay curve (normalized 1 to 0)
            decay_algorithms = {
                "polynomial": self._compute_polynomial_decay,
                "sigmoidal": self._compute_sigmoidal_decay,
                "piecewise": self._compute_piecewise_decay,
                "fourier": self._compute_fourier_decay,
                "exponential": self._compute_exponential_decay, # ✨ NEW
                "gaussian": self._compute_gaussian_decay,       # ✨ NEW
            }
            compute_func = decay_algorithms.get(self.algorithm_type, self._compute_polynomial_decay)
            decay_values = compute_func(num_steps)

            # 2. Apply smoothing if enabled
            if self.enable_temporal_smoothing:
                decay_values = self._apply_temporal_smoothing(decay_values)

            # ✨ NEW: 3. Apply inversion if toggled
            if self.invert_curve:
                decay_values = 1.0 - decay_values

            # ✨ NEW: 4. Rescale to the user-defined start and end values
            decay_values = decay_values * (self.start_value - self.end_value) + self.end_value

            if self.use_caching:
                self._cache[cache_key] = decay_values

            return decay_values

    def generate(self, **kwargs):
        """
        Main node function that instantiates and returns the scheduler object.
        """
        # Pass all inputs as kwargs for cleaner code
        scheduler_obj = self.NoiseDecayObject(**kwargs)
        return (scheduler_obj,)


NODE_CLASS_MAPPINGS = {
    "NoiseDecayScheduler_Custom": NoiseDecayScheduler_Custom,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "NoiseDecayScheduler_Custom": "Noise Decay Scheduler (Advanced)",
}