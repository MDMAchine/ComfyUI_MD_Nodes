# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# █▓▒░                                                                     ░▒▓█
# █▓▒░            amed_sampler_core.py - Core Algorithm v1.7.3             ░▒▓█
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
# ║   • AMED Dynamic Dampening Calculation
# ║   • Two-stage Gradient Evaluation
# ║   • NaN/Inf Guard logic
# ║   • API Serialization utilities
# ╚════════════════════════════════════════════════════════════════════════════


VERSION = "v1.7.3"  # UPS v1.5.8

import torch
import math
import pickle
import zlib
import base64

# =================================================================================
# == Constants (Bit-Exact Parity)
# =================================================================================

CONST_EPSILON = 1e-5
CONST_JS_MAX_SAFE_INTEGER = 9007199254740991
CONST_SEED_MIN = 0

# =================================================================================
# == Core Math Logic
# =================================================================================

def calculate_correction_weight(sigma_val, dampening_factor):
    """
    Calculates the AMED correction weight based on current sigma.
    Pure math function shared by solver and visualization.
    """
    # Dampening Curve: 0.5 at sigma=2.0+, fades to 0.05 at sigma=0.1
    base_weight = 0.5 * min(1.0, max(0.1, sigma_val / 2.0))
    correction_weight = base_weight * dampening_factor
    
    # Clamp to safe range [0.0, 0.5]
    return min(0.5, max(0.0, correction_weight))

def amed_solver_step(model_func, x, sigma_curr, sigma_next, extra_args, is_last_step=False, logger=None):
    """
    Performs a single AMED step with Dynamic Dampening.
    
    Args:
        model_func: Callable (x, sigma, **kwargs) -> denoised
        x: Latent tensor
        sigma_curr: Current noise level
        sigma_next: Target noise level
        extra_args: Dict of model arguments (will be copied)
        is_last_step: Bool
        logger: Optional logger for warnings
    """
    
    # Sanitize extra_args for model call
    model_args = extra_args.copy()
    
    # Extract AMED specific config (removed from args passed to model)
    dampening_factor = model_args.pop("amed_dampening_factor", 1.0)
    force_euler = model_args.pop("amed_force_euler_last", True)
    debug_mode = model_args.pop("amed_debug_mode", 0)
    
    # Remove metadata keys
    model_args.pop("amed_enable_profiling", None)
    if "callback" in model_args: del model_args["callback"]

    # 1. First Evaluation (Current Gradient)
    sigma_curr_t = sigma_curr * torch.ones((x.shape[0],), device=x.device, dtype=x.dtype)
    denoised_curr = model_func(x, sigma_curr_t, **model_args)
    
    # Finalization Check
    if sigma_next <= CONST_EPSILON or (is_last_step and force_euler):
        return denoised_curr

    # d_curr = (x - x0) / sigma
    d_curr = (x - denoised_curr) / sigma_curr
    
    # 2. Predictor Step (Euler Guess)
    dt = sigma_next - sigma_curr
    x_pred = x + d_curr * dt
    
    # 3. Dynamic Dampening Calculation
    s_val = sigma_next.item() if isinstance(sigma_next, torch.Tensor) else float(sigma_next)
    correction_weight = calculate_correction_weight(s_val, dampening_factor)

    # Optimization: If weight is negligible, skip 2nd eval (Pure Euler)
    if correction_weight < 0.01:
        return x_pred

    # 4. Second Evaluation (Future Gradient at Prediction)
    sigma_next_t = sigma_next * torch.ones((x.shape[0],), device=x.device, dtype=x.dtype)
    denoised_pred = model_func(x_pred, sigma_next_t, **model_args)
    d_pred = (x_pred - denoised_pred) / sigma_next
    
    # 5. AMED Correction (Weighted Mean)
    # d_final = (1 - w) * d_curr + w * d_pred
    d_final = (1.0 - correction_weight) * d_curr + correction_weight * d_pred
    
    # 6. Final Update
    x_next = x + d_final * dt
    
    # NaN/Inf Guard
    if torch.isnan(x_next).any() or torch.isinf(x_next).any():
        if debug_mode >= 1 and logger:
            logger.warning(f"AMED Instability at sigma={sigma_curr.item():.4f}. Reverting to Euler.")
        return x_pred
    
    return x_next

# =================================================================================
# == API Serialization
# =================================================================================

def serialize_for_api(data):
    """Encodes result for API transmission."""
    return base64.b64encode(zlib.compress(pickle.dumps(data))).decode('utf-8')

def deserialize_from_api(b64_data):
    """Decodes API result."""
    return pickle.loads(zlib.decompress(base64.b64decode(b64_data)))


# ==============================================================================
# == Unit Tests (smoke — runs without ComfyUI)
# ==============================================================================

if __name__ == "__main__":
    print("\n🧪 Smoke tests: amed_sampler_core")
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

    _check("VERSION defined",    VERSION == "v1.7.3")
    _check("CONST CONST_JS_MAX_SAFE_INTEGER defined", CONST_JS_MAX_SAFE_INTEGER is not None)
    _check("CONST CONST_SEED_MIN defined", CONST_SEED_MIN is not None)
    _check("fn calculate_correction_weight is callable", callable(calculate_correction_weight))
    _check("fn amed_solver_step is callable", callable(amed_solver_step))
    _check("fn serialize_for_api is callable", callable(serialize_for_api))

    print(f"\n  {_pass} passed, {_fail} failed")
    if _fail == 0:
        print("  🎉 All good.")
