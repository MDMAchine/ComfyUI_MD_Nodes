# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
# ████ MD_Nodes/NoiseDecayScheduler_Custom – Advanced Noise Decay scheduler v0.6.0 ████▓▒░
# © 2025 MDMAchine
# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
# ░▒▓ ORIGIN & DEV:
#   • Cast into the void by: MDMAchine
#   • Enhanced by: devstral (local l33t)
#   • License: Apache 2.0 — Because we’re polite anarchists
#   • Original source (if applicable): [Inspired by Blepping - github.com/blepping]

# ░▒▓ DESCRIPTION:
#   A ComfyUI scheduler node that outputs a customizable decay curve object.
#   Perfect for custom samplers, aesthetic exploration, and fine-tuning
#   the generative process with multiple algorithms and curve manipulation options.

# ░▒▓ FEATURES:
#   ✓ Six decay algorithms: polynomial, sigmoidal, piecewise, fourier, exponential, gaussian.
#   ✓ Full curve control: `start_value`, `end_value`, and `invert_curve`.
#   ✓ Configurable `smoothing_window` for temporal smoothing.
#   ✓ Built-in caching to prevent re-calculation of schedules.
#   ✓ Supports custom piecewise points and fourier frequency.

# ░▒▓ CHANGELOG:
#   - v0.6.0 (Current Release - Feature Update):
#       • ADDED: 'exponential' and 'gaussian' decay algorithms.
#       • ADDED: 'start_value', 'end_value', and 'invert_curve' inputs.
#       • ADDED: 'smoothing_window' as a configurable user input.

# ░▒▓ CONFIGURATION:
#   → Primary Use: Generating a 'polynomial' decay curve from 1.0 down to 0.0 for a custom sampler.
#   → Secondary Use: Using 'invert_curve' to create a noise *injection* schedule that starts at 0.0 and ends at 1.0.
#   → Edge Use: Creating a 'fourier' (cosine) curve with temporal smoothing for an oscillating aesthetic effect.

# ░▒▓ WARNING:
#   This node may trigger:
#   ▓▒░ An irresistible urge to apply 'gaussian' decay to everything in your life.
#   ▓▒░ Flashbacks to plotting `y = 1 / (1 + e^-x)` on a TI-83 calculator.
#   ▓▒░ Severe temporal distortion from staring at 'fourier' waves.
#   ▓▒░ Heated, 3AM debates on the merits of 'sigmoidal' vs 'polynomial' curves.
#   Consult your nearest demoscene vet if hallucinations persist.
# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄


# =================================================================================
# == Standard Library Imports                                                  ==
# =================================================================================
import hashlib
import logging
import traceback

# =================================================================================
# == Third-Party Imports                                                       ==
# =================================================================================
import numpy as np
# Note: No torch needed directly in this node

# =================================================================================
# == ComfyUI Core Modules                                                      ==
# =================================================================================
# (None needed)

# =================================================================================
# == Local Project Imports                                                     ==
# =================================================================================
# (None needed)

# =================================================================================
# == Helper Classes & Dependencies                                             ==
# =================================================================================
# (None)

# =================================================================================
# == Core Node Class                                                           ==
# =================================================================================

class NoiseDecayScheduler_Custom:
    """
    MD: Noise Decay Scheduler (Advanced)

    Generates a customizable noise decay curve object based on various algorithms
    (polynomial, sigmoidal, etc.) for use with compatible custom samplers.
    """
    CATEGORY = "MD_Nodes/Scheduler"
    FUNCTION = "generate"
    RETURN_TYPES = ("SCHEDULER",)
    RETURN_NAMES = ("scheduler",)
    DESCRIPTION = "Advanced noise decay scheduler with multiple algorithms and performance features."

    @classmethod
    def INPUT_TYPES(cls):
        """
        Define all input parameters with tooltips.

        Note: DO NOT use type hints in function signatures or global mappings.
        ComfyUI's dynamic loader cannot handle forward references at import time.
        """
        return {
            "required": {
                "algorithm_type": (
                    ["polynomial", "sigmoidal", "piecewise", "fourier", "exponential", "gaussian"],
                    {"default": "polynomial",
                     "tooltip": (
                         "DECAY ALGORITHM\n"
                         "- Determines the shape of the decay curve.\n"
                         "- 'polynomial': Smooth curve based on exponent.\n"
                         "- 'sigmoidal': S-shaped curve.\n"
                         "- 'piecewise': Linear interpolation between custom points.\n"
                         "- 'fourier': Cosine wave.\n"
                         "- 'exponential': Exponential decay.\n"
                         "- 'gaussian': Inverted bell curve."
                     )}
                ),
                "decay_exponent": ("FLOAT", {
                    "default": 2.0, "min": 0.1, "max": 10.0, "step": 0.1,
                    "tooltip": (
                        "DECAY EXPONENT / STEEPNESS\n"
                        "- Controls the curve shape for 'polynomial', 'sigmoidal', 'exponential', and 'gaussian'.\n"
                        "- Higher values generally create steeper curves."
                    )}
                ),
                "start_value": ("FLOAT", {
                    "default": 1.0, "min": -2.0, "max": 2.0, "step": 0.01,
                    "tooltip": (
                        "START VALUE\n"
                        "- The value of the curve at the first step (step 0)."
                    )}
                ),
                "end_value": ("FLOAT", {
                    "default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01,
                    "tooltip": (
                        "END VALUE\n"
                        "- The value of the curve at the last step."
                    )}
                ),
                "invert_curve": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "INVERT CURVE\n"
                        "- True: Flips the curve vertically (e.g., 1->0 becomes 0->1).\n"
                        "- Useful for creating noise *injection* schedules instead of decay."
                    )}
                ),
                "use_caching": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "ENABLE CACHING\n"
                        "- True: Caches computed decay schedules based on parameters.\n"
                        "- False: Recomputes the schedule every time `get_decay` is called.\n"
                        "- Recommended: True (enabled)."
                    )}
                ),
                "enable_temporal_smoothing": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "ENABLE TEMPORAL SMOOTHING\n"
                        "- True: Applies a moving average filter to the curve.\n"
                        "- False: Uses the raw curve values.\n"
                        "- Can help smooth transitions in generated sequences."
                    )}
                ),
            },
            "optional": {
                "smoothing_window": ("INT", {
                    "default": 3, "min": 2, "max": 20, "step": 1,
                    "tooltip": (
                        "SMOOTHING WINDOW SIZE\n"
                        "- Number of steps to average over for temporal smoothing.\n"
                        "- Only used if 'enable_temporal_smoothing' is True."
                    )}
                ),
                "custom_piecewise_points": ("STRING", {
                    "default": "1.0,0.5,0.0",
                    "tooltip": (
                        "CUSTOM PIECEWISE POINTS\n"
                        "- Comma-separated list of values for the 'piecewise' algorithm.\n"
                        "- Example: '1.0, 0.8, 0.2, 0.0'.\n"
                        "- Points are evenly spaced along the timeline."
                    )}
                ),
                "fourier_frequency": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1,
                    "tooltip": (
                        "FOURIER FREQUENCY\n"
                        "- Controls the frequency of the cosine wave for the 'fourier' algorithm.\n"
                        "- 1.0 = One half-cycle.\n"
                        "- 2.0 = One full cycle."
                    )}
                ),
            }
        }

    class NoiseDecayObject:
        """
        The actual scheduler implementation, which is returned by the node.
        This object contains the configuration and the `get_decay` method.
        """
        def __init__(self, **kwargs):
            """
            Initialize the scheduler object with parameters from the node.
            """
            self.algorithm_type = kwargs.get("algorithm_type", "polynomial")
            self.decay_exponent = kwargs.get("decay_exponent", 2.0)
            self.start_value = kwargs.get("start_value", 1.0)
            self.end_value = kwargs.get("end_value", 0.0)
            self.invert_curve = kwargs.get("invert_curve", False)
            self.use_caching = kwargs.get("use_caching", True)
            self.enable_temporal_smoothing = kwargs.get("enable_temporal_smoothing", False)
            self.smoothing_window = kwargs.get("smoothing_window", 3)
            self.custom_piecewise_points_str = kwargs.get("custom_piecewise_points", "1.0,0.5,0.0")
            self.fourier_frequency = kwargs.get("fourier_frequency", 1.0)
            self._cache = {}

            # Safely parse piecewise points
            try:
                self.piecewise_points = [float(x.strip()) for x in self.custom_piecewise_points_str.split(",")]
                if not self.piecewise_points: # Handle empty string case
                    raise ValueError("Piecewise points string is empty.")
            except (ValueError, TypeError) as e:
                logging.warning(f"[NoiseDecayScheduler] Invalid piecewise points '{self.custom_piecewise_points_str}', falling back to default [1.0, 0.5, 0.0]. Error: {e}")
                self.piecewise_points = [1.0, 0.5, 0.0]

        def _generate_cache_key(self, num_steps):
            """
            Generate a unique cache key based on all relevant parameters.

            Args:
                num_steps: The number of steps for the schedule.

            Returns:
                A unique string key for caching.
            """
            params = (
                self.algorithm_type, self.decay_exponent, self.start_value, self.end_value,
                self.invert_curve, self.enable_temporal_smoothing, self.smoothing_window,
                self.fourier_frequency, num_steps, ','.join(map(str, self.piecewise_points))
            )
            params_str = '_'.join(map(str, params))
            return hashlib.md5(params_str.encode()).hexdigest()

        def _apply_temporal_smoothing(self, decay_array):
            """
            Apply a simple moving average filter with a configurable window.

            Args:
                decay_array: Numpy array of decay values.

            Returns:
                Smoothed numpy array.
            """
            if len(decay_array) < self.smoothing_window or self.smoothing_window < 2:
                return decay_array

            # Use a convolution for an efficient moving average
            return np.convolve(decay_array, np.ones(self.smoothing_window) / self.smoothing_window, mode='same')

        # --- Base Curve Computations (Normalized from 1.0 to 0.0) ---

        def _compute_polynomial_decay(self, num_steps):
            """Compute polynomial decay (1->0)."""
            x = np.linspace(0.0, 1.0, num_steps)
            return (1.0 - x) ** self.decay_exponent

        def _compute_sigmoidal_decay(self, num_steps):
            """Compute sigmoidal decay (1->0)."""
            x = np.linspace(-1.0, 1.0, num_steps) * (self.decay_exponent / 2) * 2.5
            sigmoid = 1 / (1 + np.exp(-x))
            # Rescale to ensure it precisely spans 1.0 to 0.0
            min_val, max_val = sigmoid.min(), sigmoid.max()
            if max_val - min_val > 1e-6:
                scaled = (sigmoid - min_val) / (max_val - min_val)
                return 1.0 - scaled
            return np.ones(num_steps) * 0.5 # Fallback for flat sigmoid

        def _compute_piecewise_decay(self, num_steps):
            """Compute piecewise decay (normalized 1->0 based on min/max of points)."""
            if len(self.piecewise_points) < 2:
                logging.warning("[NoiseDecayScheduler] Piecewise needs at least 2 points, falling back to polynomial.")
                return self._compute_polynomial_decay(num_steps)

            x_points = np.linspace(0, 1, len(self.piecewise_points))
            y_points = np.array(self.piecewise_points)

            # Interpolate first
            base_curve = np.interp(np.linspace(0, 1, num_steps), x_points, y_points)

            # Normalize the interpolated curve to span 0 to 1 before final scaling
            min_val, max_val = base_curve.min(), base_curve.max()
            if max_val - min_val > 1e-6: # Avoid division by zero
                normalized_curve = (base_curve - min_val) / (max_val - min_val)
                # If the original points were descending, the normalized curve should be descending (1->0)
                # If they were ascending, the normalized curve should also be ascending (0->1)
                # We want the base curve to represent a 1->0 decay shape *before* inversion/scaling.
                # So, if the original points were generally ascending, flip the normalized curve.
                if y_points[-1] > y_points[0]:
                     return 1.0 - normalized_curve
                else:
                     return normalized_curve
            return np.ones(num_steps) * 0.5 # Fallback if points are identical


        def _compute_fourier_decay(self, num_steps):
            """Compute Fourier-based (cosine) decay (1->0)."""
            x = np.linspace(0.0, 1.0, num_steps)
            # Ensure it maps 1 -> 0 over the specified frequency
            return (np.cos(self.fourier_frequency * np.pi * x) + 1) / 2

        def _compute_exponential_decay(self, num_steps):
            """Compute exponential decay (1->0)."""
            x = np.linspace(0.0, 1.0, num_steps)
            decay = np.exp(-self.decay_exponent * x)
            # Normalize to ensure it precisely spans 1.0 to 0.0
            min_val, max_val = decay.min(), decay.max()
            if max_val - min_val > 1e-6:
                 return (decay - min_val) / (max_val - min_val)
            return np.ones(num_steps) # Fallback

        def _compute_gaussian_decay(self, num_steps):
            """Compute inverted Gaussian decay (1->0)."""
            x = np.linspace(-1.0, 1.0, num_steps)
            # The exponent controls the 'width' of the bell
            sigma = 1.0 / max(1e-6, (self.decay_exponent * 0.5)) # Avoid division by zero
            # This creates a curve that is low at the ends and 1.0 in the middle
            bell_curve = np.exp(-(x**2) / (2 * sigma**2))
            # We invert it to create a decay curve (1 at ends, low in middle)
            # and normalize to ensure it spans 1->0
            inverted_bell = 1.0 - bell_curve
            min_val, max_val = inverted_bell.min(), inverted_bell.max()
            if max_val - min_val > 1e-6:
                return (inverted_bell - min_val) / (max_val - min_val)
            return np.zeros(num_steps) # Fallback

        def get_decay(self, num_steps):
            """
            ComfyUI standard method to get the decay schedule as a NumPy array.

            Args:
                num_steps: The total number of steps requested for the schedule.

            Returns:
                A NumPy array containing the decay values for each step.
            """
            if num_steps <= 0:
                return np.array([])
                
            cache_key = None
            if self.use_caching:
                cache_key = self._generate_cache_key(num_steps)
                if cache_key in self._cache:
                    return self._cache[cache_key].copy() # Return a copy

            try:
                # 1. Select and compute the base decay curve (normalized 1 to 0)
                decay_algorithms = {
                    "polynomial": self._compute_polynomial_decay,
                    "sigmoidal": self._compute_sigmoidal_decay,
                    "piecewise": self._compute_piecewise_decay,
                    "fourier": self._compute_fourier_decay,
                    "exponential": self._compute_exponential_decay,
                    "gaussian": self._compute_gaussian_decay,
                }
                compute_func = decay_algorithms.get(self.algorithm_type, self._compute_polynomial_decay)
                decay_values = compute_func(num_steps)

                # 2. Apply smoothing if enabled
                if self.enable_temporal_smoothing:
                    decay_values = self._apply_temporal_smoothing(decay_values)

                # 3. Apply inversion if toggled
                if self.invert_curve:
                    decay_values = 1.0 - decay_values

                # 4. Rescale to the user-defined start and end values
                decay_values = decay_values * (self.start_value - self.end_value) + self.end_value

                # Ensure finite values
                if not np.all(np.isfinite(decay_values)):
                     raise ValueError("Computed decay values contain NaN or Inf.")

                if self.use_caching and cache_key is not None:
                    self._cache[cache_key] = decay_values

                return decay_values.copy() # Return a copy

            except Exception as e:
                 logging.error(f"[NoiseDecayScheduler] Error during get_decay: {e}")
                 logging.debug(traceback.format_exc())
                 # Return a simple linear fallback on error
                 return np.linspace(self.start_value, self.end_value, num_steps)

    def generate(self, **kwargs):
        """
        Main node execution function. Instantiates and returns the scheduler object.
        """
        try:
            # Pass all inputs as kwargs for cleaner code
            scheduler_obj = self.NoiseDecayObject(**kwargs)
            # Always return a tuple matching RETURN_TYPES
            return (scheduler_obj,)
        except Exception as e:
            logging.error(f"[NoiseDecayScheduler] Error creating scheduler object: {e}")
            logging.debug(traceback.format_exc())
            print(f"[NoiseDecayScheduler] ⚠️ Error creating scheduler object: {e}. Returning default.")
            # Return a default polynomial scheduler on error
            default_scheduler = self.NoiseDecayObject(algorithm_type="polynomial")
            return (default_scheduler,)


# =================================================================================
# == Node Registration                                                         ==
# =================================================================================

NODE_CLASS_MAPPINGS = {
    "NoiseDecayScheduler_Custom": NoiseDecayScheduler_Custom,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "NoiseDecayScheduler_Custom": "MD: Noise Decay Scheduler (Advanced)",
}