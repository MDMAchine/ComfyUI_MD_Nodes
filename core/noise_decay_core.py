# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# █▓▒░                                                                     ░▒▓█
# █▓▒░         MD_Nodes Core: Noise Decay Scheduler (IP-Protected)         ░▒▓█
# █▓▒░                                                                     ░▒▓█
# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
# ╠═ © 2026 MDMAchine
# ╠═ License: GNU General Public License v3.0 (GPL v3)
# ║
# ║  This program is free software: you can redistribute it and/or modify
# ║  it under the terms of the GNU General Public License as published by
# ║  the Free Software Foundation, either version 3 of the License, or
# ║  (at your option) any later version.
# ║
# ║  This program is distributed in the hope that it will be useful,
# ║  but WITHOUT ANY WARRANTY; without even the implied warranty of
# ║  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# ║  GNU General Public License for more details.
# ║
# ║  You should have received a copy of the GNU General Public License
# ║  along with this program. If not, see <https://www.gnu.org/licenses/>.
# ╚════════════════════════════════════════════════════════════════════════════


VERSION = "v3.0.0"  # UPS v1.5.8

import hashlib
import numpy as np
import pickle
import zlib
import base64
import logging

# =================================================================================
# == Constants (Bit-Exact Parity)
# =================================================================================

CONST_EPSILON = 1e-6

logger = logging.getLogger("MD_Nodes.Core.NoiseDecay")

# =================================================================================
# == Core Logic Class
# =================================================================================

class NoiseDecayObject:
    """
    The actual scheduler implementation (Math Core).
    """
    def __init__(self, **kwargs):
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

        # Parse piecewise (Parity: Include logging warning)
        try:
            self.piecewise_points = [float(x.strip()) for x in self.custom_piecewise_points_str.split(",")]
            if not self.piecewise_points: raise ValueError("Empty")
        except Exception as e:
            logger.warning(f"Invalid piecewise points, using default. Error: {e}")
            self.piecewise_points = [1.0, 0.5, 0.0]

    def _generate_cache_key(self, num_steps):
        params = (
            self.algorithm_type, self.decay_exponent, self.start_value, self.end_value,
            self.invert_curve, self.enable_temporal_smoothing, self.smoothing_window,
            self.fourier_frequency, num_steps, ','.join(map(str, self.piecewise_points))
        )
        return hashlib.md5('_'.join(map(str, params)).encode()).hexdigest()

    def _apply_temporal_smoothing(self, decay_array):
        if len(decay_array) < self.smoothing_window or self.smoothing_window < 2:
            return decay_array
        return np.convolve(decay_array, np.ones(self.smoothing_window) / self.smoothing_window, mode='same')

    # --- Base Curve Computations ---
    def _compute_polynomial_decay(self, num_steps):
        x = np.linspace(0.0, 1.0, num_steps)
        return (1.0 - x) ** self.decay_exponent

    def _compute_sigmoidal_decay(self, num_steps):
        x = np.linspace(-1.0, 1.0, num_steps) * (self.decay_exponent / 2) * 2.5
        sigmoid = 1 / (1 + np.exp(-x))
        min_v, max_v = sigmoid.min(), sigmoid.max()
        if max_v - min_v > CONST_EPSILON:
            scaled = (sigmoid - min_v) / (max_v - min_v)
            return 1.0 - scaled
        return np.ones(num_steps) * 0.5

    def _compute_piecewise_decay(self, num_steps):
        if len(self.piecewise_points) < 2:
            return self._compute_polynomial_decay(num_steps)
        x_pts = np.linspace(0, 1, len(self.piecewise_points))
        y_pts = np.array(self.piecewise_points)
        base = np.interp(np.linspace(0, 1, num_steps), x_pts, y_pts)
        min_v, max_v = base.min(), base.max()
        if max_v - min_v > CONST_EPSILON:
            norm = (base - min_v) / (max_v - min_v)
            # Respect original direction roughly
            return 1.0 - norm if y_pts[-1] > y_pts[0] else norm
        return np.ones(num_steps) * 0.5

    def _compute_fourier_decay(self, num_steps):
        x = np.linspace(0.0, 1.0, num_steps)
        return (np.cos(self.fourier_frequency * np.pi * x) + 1) / 2

    def _compute_exponential_decay(self, num_steps):
        x = np.linspace(0.0, 1.0, num_steps)
        decay = np.exp(-self.decay_exponent * x)
        min_v, max_v = decay.min(), decay.max()
        if max_v - min_v > CONST_EPSILON:
            return (decay - min_v) / (max_v - min_v)
        return np.ones(num_steps)

    def _compute_gaussian_decay(self, num_steps):
        x = np.linspace(-1.0, 1.0, num_steps)
        sigma = 1.0 / max(CONST_EPSILON, (self.decay_exponent * 0.5))
        bell = np.exp(-(x**2) / (2 * sigma**2))
        inv_bell = 1.0 - bell
        min_v, max_v = inv_bell.min(), inv_bell.max()
        if max_v - min_v > CONST_EPSILON:
            return (inv_bell - min_v) / (max_v - min_v)
        return np.zeros(num_steps)

    def get_decay(self, num_steps):
        if num_steps <= 0: return np.array([])
        
        cache_key = None
        if self.use_caching:
            cache_key = self._generate_cache_key(num_steps)
            if cache_key in self._cache:
                return self._cache[cache_key].copy()

        try:
            algos = {
                "polynomial": self._compute_polynomial_decay,
                "sigmoidal": self._compute_sigmoidal_decay,
                "piecewise": self._compute_piecewise_decay,
                "fourier": self._compute_fourier_decay,
                "exponential": self._compute_exponential_decay,
                "gaussian": self._compute_gaussian_decay,
            }
            func = algos.get(self.algorithm_type, self._compute_polynomial_decay)
            decay = func(num_steps)

            if self.enable_temporal_smoothing:
                decay = self._apply_temporal_smoothing(decay)

            if self.invert_curve:
                decay = 1.0 - decay

            # Rescale
            decay = decay * (self.start_value - self.end_value) + self.end_value

            if not np.all(np.isfinite(decay)):
                raise ValueError("NaN/Inf in decay values")

            if self.use_caching and cache_key:
                self._cache[cache_key] = decay

            return decay.copy()

        except Exception as e:
            # Fallback linear
            logger.error(f"Calculation failed: {e}")
            return np.linspace(self.start_value, self.end_value, num_steps)

# =================================================================================
# == API Serialization
# =================================================================================

def serialize_for_api(decay_values):
    """Encodes result for API transmission."""
    # Convert numpy array to list
    payload = {"decay": decay_values.tolist()}
    return base64.b64encode(zlib.compress(pickle.dumps(payload))).decode('utf-8')

def deserialize_from_api(b64_data):
    """Decodes API result."""
    data = pickle.loads(zlib.decompress(base64.b64decode(b64_data)))
    if 'decay' in data:
        data['decay'] = np.array(data['decay'])
    return data


# ==============================================================================
# == Unit Tests (smoke — runs without ComfyUI)
# ==============================================================================

if __name__ == "__main__":
    print("\n🧪 Smoke tests: noise_decay_core")
    print("   VERSION :", VERSION)
    _pass = _fail = 0

    def _check(label, expr):
        global _pass, _fail
        if expr:
            print(f"  ✅  {label}")
            _pass += 1
        else:
            print(f"  ❌  {label}")
            _fail += 1

    _check("VERSION defined",    VERSION == "v3.0.0")
    _check("fn serialize_for_api is callable", callable(serialize_for_api))
    _check("fn deserialize_from_api is callable", callable(deserialize_from_api))

    print(f"\n  {_pass} passed, {_fail} failed")
    if _fail == 0:
        print("  🎉 All good.")
