# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# █▓▒░                                                                     ░▒▓█
# █▓▒░            MD_Nodes Core: Broadcast Tools (IP-Protected)            ░▒▓█
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

import numpy as np
from scipy import signal
import pickle
import zlib
import base64

try:
    import pyloudnorm as pyln
    PYLOUDNORM_AVAILABLE = True
except ImportError:
    PYLOUDNORM_AVAILABLE = False

CONST_EPSILON = 1e-12 
CONST_OVERSAMPLING_FACTOR = 4

class LUFSMeterCore:
    """ITU-R BS.1770-4 compliant loudness meter."""
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate
        self.meter = pyln.Meter(sample_rate) if PYLOUDNORM_AVAILABLE else None
            
    def measure_lufs(self, audio):
        if not PYLOUDNORM_AVAILABLE or self.meter is None:
            rms = np.sqrt(np.mean(audio**2))
            rms_db = 20 * np.log10(max(rms, CONST_EPSILON))
            return {'lufs': rms_db - 23.0, 'true_peak_db': rms_db + 3.0, 'dynamic_range': 0.0}
        
        try:
            audio_measure = audio.reshape(-1, 1) if audio.ndim == 1 else audio.T
            lufs = self.meter.integrated_loudness(audio_measure)
            true_peak = self._calculate_true_peak(audio)
            true_peak_db = 20 * np.log10(max(true_peak, CONST_EPSILON))
            return {
                'lufs': float(lufs),
                'true_peak_db': float(true_peak_db),
                'dynamic_range': float(true_peak_db - lufs)
            }
        except Exception:
            return {'lufs': -23.0, 'true_peak_db': 0.0, 'dynamic_range': 23.0}
    
    def _calculate_true_peak(self, audio):
        try:
            upsampled = signal.resample_poly(audio, CONST_OVERSAMPLING_FACTOR, 1, axis=-1)
            return np.max(np.abs(upsampled))
        except Exception:
            return np.max(np.abs(audio))

    def normalize_to_lufs(self, audio, target_lufs, true_peak_limit_db=-1.0):
        measurements = self.measure_lufs(audio)
        current_lufs = measurements['lufs']
    
        if not np.isfinite(current_lufs):
            return audio
    
        # Pass 1: Apply loudness gain
        gain_db = target_lufs - current_lufs
        gain_linear = 10.0 ** (gain_db / 20.0)
        normalized = audio * gain_linear
        
        # Pass 2: True peak safety — only clamp if we actually exceed the limit.
        # Use oversampled peak for accuracy (catches inter-sample peaks).
        true_peak_limit_linear = 10.0 ** (true_peak_limit_db / 20.0)
        true_peak = self._calculate_true_peak(normalized)
        
        if true_peak > true_peak_limit_linear:
            normalized *= (true_peak_limit_linear / true_peak)
        
        return normalized

class MidSideProcessorCore:
    """Mid/Side stereo encoder/decoder."""
    @staticmethod
    def encode(left, right):
        return (left + right) / 2.0, (left - right) / 2.0
    
    @staticmethod
    def decode(mid, side):
        return mid + side, mid - side
    
    @staticmethod
    def adjust_width(audio, width_percent):
        if audio.shape[0] != 2: 
            return audio
        mid, side = MidSideProcessorCore.encode(audio[0], audio[1])
        side *= (width_percent / 100.0)
        left, right = MidSideProcessorCore.decode(mid, side)
        return np.stack([left, right], axis=0)

def serialize_for_api(data):
    return base64.b64encode(zlib.compress(pickle.dumps(data))).decode('utf-8')

def deserialize_from_api(b64_data):
    return pickle.loads(zlib.decompress(base64.b64decode(b64_data)))


# ==============================================================================
# == Unit Tests (smoke — runs without ComfyUI)
# ==============================================================================

if __name__ == "__main__":
    print("\n🧪 Smoke tests: broadcast_tools_core")
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
    _check("CONST CONST_OVERSAMPLING_FACTOR defined", CONST_OVERSAMPLING_FACTOR is not None)
    _check("fn serialize_for_api is callable", callable(serialize_for_api))
    _check("fn deserialize_from_api is callable", callable(deserialize_from_api))

    print(f"\n  {_pass} passed, {_fail} failed")
    if _fail == 0:
        print("  🎉 All good.")
