# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# █▓▒░                                                                     ░▒▓█
# █▓▒░           gits_scheduler_core.py - Core Algorithm v1.3.0            ░▒▓█
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
# ╠════════════════════════════════════════════════════════════════════════════
# ║ CORE RESPONSIBILITIES:
# ║   • GITS "Boomerang" (tanh) S-Curve Schedule Generation
# ║   • Mathematical mapping of t-warped tensors
# ║   • Sigma Grid construction and interpolation
# ╚════════════════════════════════════════════════════════════════════════════


VERSION = "v1.3.0"  # UPS v1.5.8

import math
import torch
import pickle
import zlib
import base64

# =================================================================================
# == Constants (Bit-Exact Parity)
# =================================================================================

CONST_EPSILON = 1e-6

# =================================================================================
# == Core Logic
# =================================================================================

def get_gits_sigmas(steps, sigma_min, sigma_max, curvature_scale=1.0, device=None):
    """
    Calculates the GITS 'Boomerang' schedule using tanh warping.
    """
    if device is None:
        device = torch.device("cpu")
        
    if steps < 1:
        return torch.tensor([sigma_max, sigma_min], dtype=torch.float32, device=device)

    # 1. Base grid (Linear)
    # Range [0, 1]
    t = torch.linspace(0, 1, steps + 1, dtype=torch.float32, device=device)
    
    # 2. Apply "Boomerang" Curvature transform
    if curvature_scale > 0:
        # Cosine warping to cluster steps in the middle (high curvature region)
        # Using a specialized S-curve derived from GITS theory
        m = curvature_scale
        # Formula parity check: 0.5 * (1 + tanh(m * (2t - 1)))
        t_warped = 0.5 * (1 + torch.tanh(m * (2 * t - 1)))
        
        # Normalize back to 0-1 exactly
        t_warped = (t_warped - t_warped[0]) / (t_warped[-1] - t_warped[0])
    else:
        t_warped = t

    # 3. Map to Sigmas using Log-Linear interpolation
    log_min = math.log(sigma_min)
    log_max = math.log(sigma_max)
    
    # Invert direction: t=0 -> sigma_max, t=1 -> sigma_min
    log_sigmas = log_max + t_warped * (log_min - log_max)
    
    sigmas = torch.exp(log_sigmas)
    
    # 4. Force exact endpoints for precision
    sigmas[0] = sigma_max
    sigmas[-1] = sigma_min
    
    return sigmas

# =================================================================================
# == API Serialization
# =================================================================================

def serialize_for_api(sigmas):
    """Encodes result for API transmission."""
    payload = {"sigmas": sigmas.cpu().numpy().tolist()}
    return base64.b64encode(zlib.compress(pickle.dumps(payload))).decode('utf-8')

def deserialize_from_api(b64_data):
    """Decodes API result."""
    data = pickle.loads(zlib.decompress(base64.b64decode(b64_data)))
    if 'sigmas' in data:
        data['sigmas'] = torch.tensor(data['sigmas'])
    return data


# ==============================================================================
# == Unit Tests (smoke — runs without ComfyUI)
# ==============================================================================

if __name__ == "__main__":
    print("\n🧪 Smoke tests: gits_scheduler_core")
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

    _check("VERSION defined",    VERSION == "v1.3.0")
    _check("fn get_gits_sigmas is callable", callable(get_gits_sigmas))
    _check("fn serialize_for_api is callable", callable(serialize_for_api))
    _check("fn deserialize_from_api is callable", callable(deserialize_from_api))

    print(f"\n  {_pass} passed, {_fail} failed")
    if _fail == 0:
        print("  🎉 All good.")
