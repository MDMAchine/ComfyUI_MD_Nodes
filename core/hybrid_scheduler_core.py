# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# █▓▒░                                                                     ░▒▓█
# █▓▒░           MD_Nodes Core: Hybrid Scheduler                           ░▒▓█
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

import math
import torch
import numpy as np
import pickle
import zlib
import base64

# =================================================================================
# == Constants (Bit-Exact Parity)
# =================================================================================

CONST_EPSILON_TINY = 1e-9
CONST_EPSILON_STANDARD = 1e-6
CONST_MONOTONIC_DECAY = 0.99
CONST_MEMORY_CHUNK_SIZE = 250

# =================================================================================
# == Dependency Management
# =================================================================================

try:
    from scipy.ndimage import gaussian_filter1d
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# =================================================================================
# == Standalone Math Functions (No ComfyUI Deps)
# =================================================================================

def _get_sigmas_karras_standalone(n, sigma_min, sigma_max, rho=7.0, device=None):
    if device is None: device = torch.device('cpu')
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    ramp = torch.linspace(0, 1, n, device=device)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return torch.cat([sigmas, torch.tensor([0.0], device=device)])

def _kl_optimal_scheduler(n, sigma_min, sigma_max, device=None):
    if device is None: device = torch.device('cpu')
    sigma_min_t = torch.tensor(sigma_min, dtype=torch.float32, device=device)
    sigma_max_t = torch.tensor(sigma_max, dtype=torch.float32, device=device)
    
    if n > 1: adj_idxs = torch.arange(n, dtype=torch.float32, device=device).div_(n - 1)
    else: adj_idxs = torch.tensor([0.0], dtype=torch.float32, device=device)
    
    sigmas = torch.zeros(n + 1, dtype=torch.float32, device=device)
    sigmas[:-1] = (adj_idxs * torch.atan(sigma_min_t) + (1 - adj_idxs) * torch.atan(sigma_max_t)).tan_()
    return sigmas

def _linear_quadratic_schedule_adapted(steps, sigma_max, threshold_noise=0.0025, linear_steps=None, device=None):
    if device is None: device = torch.device('cpu')
    if steps <= 0: return torch.tensor([sigma_max, 0.0], dtype=torch.float32, device=device)
    if steps == 1: return torch.tensor([sigma_max, 0.0], dtype=torch.float32, device=device)
    
    linear_steps_actual = steps // 2 if linear_steps is None else max(0, min(linear_steps, steps))
    sigma_schedule_raw = []
    
    if linear_steps_actual == 0:
        for i in range(steps):
            val = (i / (steps - 1.0))**2 if steps > 1 else 0.0
            sigma_schedule_raw.append(val)
    else:
        for i in range(linear_steps_actual):
            sigma_schedule_raw.append(i * threshold_noise / linear_steps_actual)
        
        quadratic_steps = steps - linear_steps_actual
        if quadratic_steps > 0:
            threshold_noise_step_diff = linear_steps_actual - threshold_noise * steps
            quadratic_coef = threshold_noise_step_diff / (linear_steps_actual * quadratic_steps ** 2)
            linear_coef = threshold_noise / linear_steps_actual - 2 * threshold_noise_step_diff / (quadratic_steps ** 2)
            const = quadratic_coef * (linear_steps_actual ** 2)
            for i in range(linear_steps_actual, steps):
                sigma_schedule_raw.append(quadratic_coef * (i ** 2) + linear_coef * i + const)
    
    if not sigma_schedule_raw or sigma_schedule_raw[-1] != 1.0:
        sigma_schedule_raw.append(1.0)
    
    sigma_schedule_inverted = [1.0 - x for x in sigma_schedule_raw]
    return torch.tensor(sigma_schedule_inverted, dtype=torch.float32, device=device) * sigma_max

def _basic_tangent_schedule(steps, s_max, s_min, device):
    """
    Simple tangent-based sigma schedule.
    Standard textbook approach using basic tan() interpolation.
    """
    if device is None: 
        device = torch.device('cpu')
    if steps < 2: 
        return torch.linspace(s_max, s_min, steps + 1, device=device)
    
    # Simple tangent interpolation (no novel two-phase warping)
    t = torch.linspace(0, 1, steps, device=device)
    
    # Basic tangent curve (standard diffusion literature)
    # Using tan() to create non-linear progression
    angle = t * (math.pi / 4)  # Simple quarter-circle
    tan_values = torch.tan(angle)
    
    # Normalize to [0, 1] range
    tan_normalized = (tan_values - tan_values.min()) / (tan_values.max() - tan_values.min())
    
    # Map to sigma range
    sigmas = s_max * (1.0 - tan_normalized) + s_min * tan_normalized
    
    # Add final zero sigma
    return torch.cat([sigmas, torch.tensor([s_min], device=device)])

def _beta_scheduler(steps, sigma_min, sigma_max, alpha=0.6, beta=0.6, device=None):
    if device is None: device = torch.device('cpu')
    t = torch.linspace(0, 1, steps + 1, device=device)
    alpha = max(alpha, 0.1); beta = max(beta, 0.1)
    beta_curve = 1.0 - (1.0 - t ** alpha) ** beta
    beta_curve = torch.clamp(beta_curve, 0.0, 1.0)
    sigmas = sigma_max * (1.0 - beta_curve) + sigma_min * beta_curve
    sigmas[0] = sigma_max; sigmas[-1] = sigma_min
    return sigmas

def _ays_scheduler(steps, sigma_min, sigma_max, device=None):
    if device is None: device = torch.device('cpu')
    t = torch.linspace(0, 1, steps + 1, device=device)
    ays_curve = 1.0 / (1.0 + torch.exp(-10 * (t - 0.5)))
    ays_curve = (ays_curve - ays_curve.min()) / (ays_curve.max() - ays_curve.min())
    concentration_factor = torch.exp(-2 * t)
    ays_curve = ays_curve * 0.7 + concentration_factor * 0.3
    ays_curve = 1.0 - (ays_curve / ays_curve.max())
    return sigma_min + (sigma_max - sigma_min) * ays_curve

def _ddim_uniform_scheduler(steps, sigma_min, sigma_max, device=None):
    if device is None: device = torch.device('cpu')
    max_timestep = 1000
    step_ratio = max_timestep // steps
    timesteps = torch.arange(0, steps + 1, device=device) * step_ratio
    timesteps = torch.flip(max_timestep - timesteps, [0]).float()
    t_normalized = timesteps / max_timestep
    sigmas = sigma_min + (sigma_max - sigma_min) * (t_normalized ** 0.5)
    return sigmas

def _sgm_uniform_scheduler(steps, sigma_min, sigma_max, device=None):
    if device is None: device = torch.device('cpu')
    max_timestep = 999
    timesteps = torch.linspace(max_timestep, 0, steps + 1, device=device)
    t_normalized = timesteps / max_timestep
    sigmas = sigma_min + (sigma_max - sigma_min) * t_normalized
    return sigmas

def _simple_scheduler(steps, sigma_min, sigma_max, device):
    if device is None: device = torch.device('cpu')
    t = torch.linspace(0, 1, steps + 1, device=device)
    smoothed = t * t * (3.0 - 2.0 * t)
    return sigma_max - (sigma_max - sigma_min) * smoothed

# =================================================================================
# == Core Calculation Engine
# =================================================================================

def calculate_sigmas(steps, mode, s_min, s_max, device, **kwargs):
    if mode == "polynomial": mode = "poly"
    if steps <= 0: return torch.tensor([s_max, s_min], device=device)

    # Memory Chunking
    memory_efficient = kwargs.get('memory_efficient', False)
    if memory_efficient and steps > 500:
        chunk_size = CONST_MEMORY_CHUNK_SIZE
        chunks = []
        for i in range(0, steps, chunk_size):
            chunk_steps = min(chunk_size, steps - i)
            progress_start = i / steps
            progress_end = min(i + chunk_steps, steps) / steps
            chunk_s_max = s_max * (1 - progress_start) + s_min * progress_start
            chunk_s_min = s_max * (1 - progress_end) + s_min * progress_end
            
            sub_kwargs = kwargs.copy()
            sub_kwargs['memory_efficient'] = False
            
            chunk = calculate_sigmas(chunk_steps, mode, chunk_s_min, chunk_s_max, device, **sub_kwargs)
            chunks.append(chunk[:-1] if i + chunk_steps < steps else chunk)
        return torch.cat(chunks)

    # Dispatch
    if mode == "karras_rho": return _get_sigmas_karras_standalone(steps, s_min, s_max, kwargs.get('rho', 7.0), device)
    if mode == "simple": return _simple_scheduler(steps, s_min, s_max, device)
    if mode == "linear_quadratic": return _linear_quadratic_schedule_adapted(steps, s_max, kwargs.get('threshold_noise', 0.0025), kwargs.get('linear_steps'), device)
    if mode == "bong_tangent": return _basic_tangent_schedule(steps, s_max, s_min, device)
    if mode == "poly": return torch.linspace(s_max**(1/kwargs.get('power', 2.0)), s_min**(1/kwargs.get('power', 2.0)), steps + 1, device=device).pow(kwargs.get('power', 2.0))
    if mode == "beta": return _beta_scheduler(steps, s_min, s_max, kwargs.get('beta_alpha', 0.6), kwargs.get('beta_beta', 0.6), device)
    if mode == "ays": return _ays_scheduler(steps, s_min, s_max, device)
    if mode == "ddim_uniform": return _ddim_uniform_scheduler(steps, s_min, s_max, device)
    if mode == "sgm_uniform": return _sgm_uniform_scheduler(steps, s_min, s_max, device)
    if mode == "adaptive_linear": return torch.linspace(s_max, s_min, steps + 1, device=device)
    if mode == "kl_optimal": return _kl_optimal_scheduler(steps, s_min, s_max, device)
    if mode == "exponential":
        t = torch.linspace(0, 1, steps + 1, device=device)
        safe_max = s_max if s_max > 0 else CONST_EPSILON_TINY
        return safe_max * (s_min / safe_max) ** t
    if mode == "variance_preserving":
        t = torch.linspace(1, 0, steps + 1, device=device)
        log_sigmas = (1 - t) * torch.log(torch.tensor(s_min, device=device)) + t * torch.log(torch.tensor(s_max, device=device))
        return torch.exp(log_sigmas)
    if mode == "blended_curves":
        karras = _get_sigmas_karras_standalone(steps, s_min, s_max, kwargs.get('rho', 7.0), device)
        linear = torch.linspace(s_max, s_min, steps + 1, device=device)
        blend = kwargs.get('blend_factor', 0.5)
        return (1.0 - blend) * karras + blend * linear
    
    return torch.linspace(s_max, s_min, steps + 1, device=device)

# =================================================================================
# == Sigma Utility Functions
# =================================================================================

def smooth_sigmas(sigmas, smoothing_strength, smoothing_type, preserve_endpoints=True, window_size=3):
    if len(sigmas) == 0 or smoothing_strength <= 0.0: return sigmas
    device = sigmas.device
    sigma_list = sigmas.cpu().numpy().astype(np.float64)
    n = len(sigma_list)
    if n <= 2: return sigmas
    
    original_start = sigma_list[0]
    original_end = sigma_list[-1]
    
    if smoothing_type == "gaussian" and SCIPY_AVAILABLE:
        sigma_param = max(0.5, window_size * smoothing_strength)
        smoothed = gaussian_filter1d(sigma_list, sigma=sigma_param)
    elif smoothing_type == "exponential":
        alpha = 0.1 + (smoothing_strength * 0.4)
        smoothed = sigma_list.copy()
        for i in range(1, len(sigma_list)):
            smoothed[i] = alpha * sigma_list[i] + (1 - alpha) * smoothed[i-1]
    else:
        smoothed = sigma_list.copy()
        half_window = window_size // 2
        for i in range(half_window, n - half_window):
            smoothed[i] = np.mean(sigma_list[max(0, i-half_window):min(n, i+half_window+1)])
    
    result = sigma_list * (1.0 - smoothing_strength) + smoothed * smoothing_strength
    if preserve_endpoints:
        result[0] = original_start; result[-1] = original_end
        
    return torch.from_numpy(result).to(device=device, dtype=sigmas.dtype)

def concatenate_sigmas(sigmas_a, sigmas_b, blend_mode, crossfade_steps=5, normalize_range=False):
    device = sigmas_a.device
    list_a = sigmas_a.cpu().numpy().astype(np.float64)
    list_b = sigmas_b.cpu().numpy().astype(np.float64)
    
    if len(list_a) == 0: return sigmas_b
    if len(list_b) == 0: return sigmas_a
    
    if normalize_range:
        a_end = list_a[-1]
        b_start = list_b[0]
        if b_start > 0: list_b = list_b * (a_end / b_start)
            
    if blend_mode == "concatenate":
        result = np.concatenate([list_a[:-1], list_b])
    elif blend_mode == "crossfade":
        crossfade_steps = min(crossfade_steps, len(list_a)-1, len(list_b)-1)
        if crossfade_steps <= 0: result = np.concatenate([list_a[:-1], list_b])
        else:
            pre = list_a[:-crossfade_steps-1]
            mid_a = list_a[-crossfade_steps-1:-1]
            mid_b = list_b[:crossfade_steps]
            post = list_b[crossfade_steps:]
            weights = np.linspace(1.0, 0.0, crossfade_steps)
            mid = mid_a * weights + mid_b * (1.0 - weights)
            result = np.concatenate([pre, mid, post])
    else: # overlap
        a_end = list_a[-2] if len(list_a) > 1 else list_a[-1]
        overlap_idx = 0
        for i, s in enumerate(list_b):
            if s <= a_end:
                overlap_idx = i
                break
        result = np.concatenate([list_a[:-1], list_b[overlap_idx:]])
        
    return torch.from_numpy(result).to(device=device, dtype=sigmas_a.dtype)

# =================================================================================
# == API Serialization
# =================================================================================
def serialize_for_api(sigmas, info):
    payload = {"sigmas": sigmas.cpu().numpy().tolist(), "info": info}
    return base64.b64encode(zlib.compress(pickle.dumps(payload))).decode('utf-8')

def deserialize_from_api(b64_data):
    data = pickle.loads(zlib.decompress(base64.b64decode(b64_data)))
    if 'sigmas' in data: data['sigmas'] = torch.tensor(data['sigmas'])
    return data


# ==============================================================================
# == Unit Tests (smoke — runs without ComfyUI)
# ==============================================================================

if __name__ == "__main__":
    print("\n🧪 Smoke tests: hybrid_scheduler_core")
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
    _check("CONST CONST_MONOTONIC_DECAY defined", CONST_MONOTONIC_DECAY is not None)
    _check("CONST CONST_MEMORY_CHUNK_SIZE defined", CONST_MEMORY_CHUNK_SIZE is not None)
    _check("fn calculate_sigmas is callable", callable(calculate_sigmas))
    _check("fn smooth_sigmas is callable", callable(smooth_sigmas))
    _check("fn concatenate_sigmas is callable", callable(concatenate_sigmas))

    print(f"\n  {_pass} passed, {_fail} failed")
    if _fail == 0:
        print("  🎉 All good.")
