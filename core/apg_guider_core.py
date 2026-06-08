# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# █▓▒░                                                                     ░▒▓█
# █▓▒░             apg_guider_core.py - Core Algorithm v1.6.2              ░▒▓█
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
# ║   • Orthogonal projection math (APG)
# ║   • Momentum and blend mode calculations
# ║   • Sigma rule configuration definitions
# ║   • Stateless processing (pure tensor math)
# ║   • A robust fork of Blepping's APG Guider utilizing the MD_Nodes Core/Wrapper architecture.
# ╚════════════════════════════════════════════════════════════════════════════


VERSION = "v1.6.2"  # UPS v1.5.8

import enum
import math
import logging
import torch
import torch.nn.functional as F

# =================================================================================
# == Constants & Pure Math Helpers
# =================================================================================
CONST_EPSILON = 1e-6

def _slerp(a, b, t):
    """Spherical linear interpolation."""
    omega = torch.acos((a * b).sum() / (a.norm() * b.norm()))
    so = torch.sin(omega)
    if abs(so) < CONST_EPSILON:
        return torch.lerp(a, b, t) 
    return (torch.sin((1.0 - t) * omega) / so) * a + (torch.sin(t * omega) / so) * b

def _validate_dims(dims, tensor_ndim=4):
    """Validate dimension tuple for tensor operations."""
    if not dims:
        raise ValueError("dims cannot be empty. Use dimensions like (-1, -2).")
    
    for dim in dims:
        if not isinstance(dim, int):
            raise TypeError(f"All dims must be integers, got {type(dim)} for value {dim}")
        normalized_dim = dim if dim >= 0 else tensor_ndim + dim
        if normalized_dim < 0 or normalized_dim >= tensor_ndim:
            raise ValueError(f"Dimension {dim} is out of bounds for {tensor_ndim}D tensor")
    
    return dims

# Pure tensor blend modes
BLEND_MODES = {
    "lerp": torch.lerp,
    "slerp": lambda a, b, t: _slerp(a, b, t),
    "add": lambda a, b, t: a + b * t,
    "multiply": lambda a, b, t: a * (1 + (b - 1) * t),
    "a_only": lambda a, b, t: a,
    "b_only": lambda a, b, t: b,
    "average": lambda a, b, t: (a + b) / 2,
}

class UpdateMode(enum.Enum):
    """Momentum update modes for APG guidance."""
    DEFAULT = enum.auto()
    ALT1 = enum.auto()
    ALT2 = enum.auto()

# =================================================================================
# == Configuration & APG Logic
# =================================================================================

class APGConfig:
    """Stores the settings for a single sigma-based rule."""
    def __init__(self,
                 start_sigma=math.inf,
                 momentum=-0.5,
                 eta=0.0, 
                 apg_scale=4.0,
                 norm_threshold=2.5,
                 dims=(-2, -1),
                 update_mode=None,
                 update_blend_mode="lerp",
                 cfg=1.0,
                 apg_blend=1.0,
                 apg_blend_mode="lerp",
                 predict_image=True,
                 pre_cfg_mode=False):
        
        self.start_sigma = start_sigma
        self.momentum = momentum
        self.eta = eta
        self.apg_scale = apg_scale
        self.norm_threshold = norm_threshold
        self.dims = dims
        self.update_mode = update_mode if update_mode is not None else UpdateMode.DEFAULT
        self.update_blend_mode = update_blend_mode
        self.cfg = cfg
        self.apg_blend = apg_blend
        self.apg_blend_mode = apg_blend_mode
        self.predict_image = predict_image
        self.pre_cfg_mode = pre_cfg_mode

        self._fields = (
            "start_sigma", "momentum", "eta", "apg_scale", "norm_threshold",
            "dims", "update_mode", "update_blend_mode", "cfg", "apg_blend",
            "apg_blend_mode", "predict_image", "pre_cfg_mode"
        )

    @staticmethod
    def fixup_param(k, v):
        if k == "dims":
            if isinstance(v, str):
                dims_str = v.strip()
                if not dims_str:
                    raise ValueError("dims string cannot be empty")
                try:
                    parsed_dims = tuple(int(d.strip()) for d in dims_str.split(","))
                except ValueError as e:
                    raise ValueError(f"Invalid dims format '{v}'. Expected comma-separated integers. Error: {e}")
                return _validate_dims(parsed_dims)
            elif isinstance(v, (list, tuple)):
                return _validate_dims(tuple(v))
            else:
                raise TypeError(f"dims must be string or tuple, got {type(v)}")
        
        if k == "update_mode":
            mode_upper = str(v).strip().upper()
            mode_enum = UpdateMode.__members__.get(mode_upper)
            if mode_enum is None:
                return UpdateMode.DEFAULT
            return mode_enum
        
        if k == "start_sigma":
            return math.inf if float(v) < 0 else float(v)
        
        if k == "norm_threshold":
            threshold = float(v) if v is not None else 2.5
            return 0.0 if threshold < 0 else threshold
        
        if k == "apg_scale":
            scale = float(v)
            return 0.0 if abs(scale) < CONST_EPSILON else scale
        
        return v

    @classmethod
    def build(cls, *, mode="pure_apg", **params):
        is_pre = False
        up_mode = mode
        
        if "_" in mode:
            parts = mode.split("_", 1)
            is_pre = (parts[0] == "pre")
            up_mode = parts[1]
            
        if up_mode in ["apg", "cfg"]:
            up_mode = "default"
            
        params["pre_cfg_mode"] = is_pre
        params["update_mode"] = up_mode
        
        valid_keys = set(cls()._fields)
        init_kwargs = {}
        
        for key in valid_keys:
            if key in params:
                init_kwargs[key] = cls.fixup_param(key, params[key])
            else:
                init_kwargs[key] = getattr(cls(), key)
                
        return cls(**init_kwargs)

    def __str__(self):
        if self.apg_blend == 0 or self.apg_scale == 0:
            fields = ("start_sigma", "cfg")
        else:
            fields = self._fields
        pretty_fields = ", ".join(f"{k}={getattr(self, k)}" for k in fields)
        return f"APGConfig({pretty_fields})"


class APG:
    """Core APG logic with momentum and orthogonal projection."""
    def __init__(self, config):
        self.config = config
        self.running_average = 0.0
        self.is_mps = None

    def __getattr__(self, k):
        return getattr(self.config, k)

    def update(self, val):
        if self.momentum == 0.0:
            return val
            
        current_avg = self.running_average
        needs_reset = False
        
        if isinstance(current_avg, float):
            needs_reset = True
        elif isinstance(current_avg, torch.Tensor):
            if current_avg.dtype != val.dtype or current_avg.device != val.device or current_avg.shape != val.shape:
                needs_reset = True
                
        if needs_reset:
            self.running_average = val.clone()
            return self.running_average
            
        step_result = val + (self.momentum * current_avg)
        
        if self.update_mode == UpdateMode.ALT1:
            self.running_average = val + (abs(self.momentum) * current_avg)
        elif self.update_mode == UpdateMode.ALT2:
            blend_func = BLEND_MODES.get(self.update_blend_mode)
            if not blend_func:
                raise ValueError("Invalid blend mode")
                
            mix_val = current_avg.neg() if self.momentum < 0 else current_avg
            step_result = blend_func(val, mix_val, abs(self.momentum))
            self.running_average = blend_func(val, current_avg, abs(self.momentum))
        else:
            self.running_average = step_result
            
        return step_result

    def reset(self):
        self.running_average = 0.0
        self.is_mps = None 

    def project(self, v0_orig, v1_orig):
        if self.is_mps is None:
            self.is_mps = v0_orig.device.type == "mps"
            if self.is_mps:
                logging.warning("[APG Core] MPS device detected. Performance may be impacted.")

        if self.is_mps:
            v0, v1 = v0_orig.cpu().double(), v1_orig.cpu().double()
        else:
            v0, v1 = v0_orig.double(), v1_orig.double()
        
        v1 = F.normalize(v1, dim=self.dims, eps=CONST_EPSILON)
        v0_p = (v0 * v1).sum(dim=self.dims, keepdim=True) * v1
        v0_o = v0 - v0_p
        
        return (
            v0_p.to(dtype=v0_orig.dtype, device=v0_orig.device),
            v0_o.to(dtype=v0_orig.dtype, device=v0_orig.device)
        )

    def apg(self, cond, uncond):
        delta = cond - uncond
        projected_update = self.update(delta)
        
        if self.norm_threshold and self.norm_threshold > 0.0:
            magnitude = projected_update.norm(p=2, dim=self.dims, keepdim=True)
            safe_magnitude = torch.clamp(magnitude, min=CONST_EPSILON)
            ratio = self.norm_threshold / safe_magnitude
            
            if torch.any(ratio < 1.0):
                projected_update = projected_update * torch.clamp(ratio, max=1.0)
                
        _, orthogonal_component = self.project(projected_update, cond)
        return orthogonal_component

    def cfg_function(self, args):
        is_img = self.predict_image
        c = args["cond_denoised"] if is_img else args["cond"]
        uc = args["uncond_denoised"] if is_img else args["uncond"]
        
        if abs(self.apg_scale - 1.0) < 1e-6:
            out = c
        else:
            shift = self.apg(c, uc)
            out = c + ((self.apg_scale - 1.0) * shift)
            
        if is_img:
            return args["input"] - out
        return out

    def pre_cfg_function(self, args):
        items = args.get("conds_out", [])
        if len(items) < 2:
            return items
            
        c_tensor = items[0]
        uc_tensor = items[1]
        
        apg_shift = self.apg(c_tensor, uc_tensor)
        
        if abs(self.apg_scale) < 1e-6:
            modified_c = c_tensor
        else:
            base_diff = (c_tensor - uc_tensor) / self.apg_scale
            modified_c = uc_tensor + apg_shift + base_diff
            
        result_list = [modified_c]
        for i in range(1, len(items)):
            result_list.append(items[i])
        return result_list


# ==============================================================================
# == Unit Tests (smoke — runs without ComfyUI)
# ==============================================================================

if __name__ == "__main__":
    print("\n🧪 Smoke tests: apg_guider_core")
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

    _check("VERSION defined",    VERSION == "v1.6.2")

    print(f"\n  {_pass} passed, {_fail} failed")
    if _fail == 0:
        print("  🎉 All good.")
