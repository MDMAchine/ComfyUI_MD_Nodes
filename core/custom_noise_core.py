# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# █▓▒░                                                                     ░▒▓█
# █▓▒░            custom_noise_core.py - Core Algorithm v1.6.1             ░▒▓█
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
# ║   • 17 Custom Noise Algorithms (Collatz, Voronoi, Wavelets, etc.)
# ║   • 5-Stage Tensor Blending Math (Add, Multiply, Screen, Overlay, etc.)
# ║   • Tensor Normalization
# ║   • Stateless processing (pure PyTorch tensor math)
# ╚════════════════════════════════════════════════════════════════════════════


VERSION = "v1.6.1"  # UPS v1.5.8

import math
import torch
import torch.nn.functional as F
import logging

# Constants
CONST_JS_MAX_SAFE_INTEGER = 9007199254740991
CONST_SEED_MIN = 0

def validate_seed(seed_value):
    try:
        val = int(seed_value)
    except (ValueError, TypeError):
        return CONST_SEED_MIN
    return max(CONST_SEED_MIN, min(val, CONST_JS_MAX_SAFE_INTEGER))

# =================================================================================
# == Noise Generation Algorithms
# =================================================================================

def generate_perlin_noise(shape, scale=1.0, seed=0, device="cpu", independent_channels=False):
    b, c, h, w = shape
    if independent_channels:
        channels = []
        for i in range(c):
            gen = torch.Generator(device=device).manual_seed(seed + i * 100)
            freq_h = max(2, int(h / (16 * scale)))
            freq_w = max(2, int(w / (16 * scale)))
            grid = torch.randn((b, 1, freq_h, freq_w), device=device, generator=gen)
            noise = F.interpolate(grid, size=(h, w), mode='bicubic', align_corners=False)
            channels.append(noise)
        noise = torch.cat(channels, dim=1)
    else:
        gen = torch.Generator(device=device).manual_seed(seed)
        freq_h = max(2, int(h / (16 * scale)))
        freq_w = max(2, int(w / (16 * scale)))
        grid = torch.randn((b, c, freq_h, freq_w), device=device, generator=gen)
        noise = F.interpolate(grid, size=(h, w), mode='bicubic', align_corners=False)
    
    return (noise - noise.mean()) / (noise.std() + 1e-6)

def generate_voronoi_noise(shape, scale=1.0, metric="euclidean", seed=0, device="cpu", independent_channels=False):
    b, c, h, w = shape
    num_points = int(60 * scale)
    
    def get_layer(s):
        gen = torch.Generator(device=device).manual_seed(s)
        points_x = torch.rand((b, num_points), device=device, generator=gen) * w
        points_y = torch.rand((b, num_points), device=device, generator=gen) * h
        y_coords = torch.arange(h, device=device).float().view(1, h, 1)
        x_coords = torch.arange(w, device=device).float().view(1, 1, w)
        
        noise_batch = []
        for i in range(b):
            px = points_x[i].view(num_points, 1, 1)
            py = points_y[i].view(num_points, 1, 1)
            
            if metric == "manhattan":
                dist = torch.abs(x_coords - px) + torch.abs(y_coords - py)
            else:
                dist = (x_coords - px)**2 + (y_coords - py)**2 
            
            min_dist, _ = torch.min(dist, dim=0)
            noise_batch.append(torch.sqrt(min_dist) if metric != "manhattan" else min_dist)
        return torch.stack(noise_batch).unsqueeze(1)

    if independent_channels:
        layers = [get_layer(seed + i*50) for i in range(c)]
        noise = torch.cat(layers, dim=1)
    else:
        base = get_layer(seed)
        noise = base.repeat(1, c, 1, 1)

    return (noise - noise.mean()) / (noise.std() + 1e-6)

def generate_collatz_noise(shape, scale=1.0, seed=0, device="cpu", independent_channels=False):
    b, c, h, w = shape
    
    def get_layer(offset_val):
        y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
        val = (x + y * w).float() * (0.01 * scale) + offset_val
        val = val.long().clamp(min=1)
        
        steps = torch.zeros_like(val).float()
        current = val
        
        for _ in range(40): 
            mask_even = (current % 2 == 0)
            mask_odd = ~mask_even & (current > 1)
            current[mask_even] = current[mask_even] // 2
            current[mask_odd] = 3 * current[mask_odd] + 1
            steps[current > 1] += 1
        return steps.unsqueeze(0).unsqueeze(0)

    offset = seed % 1000
    if independent_channels:
        layers = [get_layer(offset + i*100) for i in range(c)]
        noise = torch.cat(layers, dim=1).repeat(b, 1, 1, 1)
    else:
        base = get_layer(offset)
        noise = base.repeat(b, c, 1, 1)
        
    return (noise - noise.mean()) / (noise.std() + 1e-6)

def generate_laplacian_noise(shape, scale, seed, device, independent_channels=False):
    generator = torch.Generator(device=device).manual_seed(seed)
    b, c, h, w = shape
    
    if independent_channels:
        noise = torch.zeros(shape, device=device)
        for ch in range(c):
            gen_ch = torch.Generator(device=device).manual_seed(seed + ch + 1000)
            u1 = torch.rand((b, 1, h, w), device=device, generator=gen_ch)
            noise[:, ch:ch+1, :, :] = -torch.sign(u1 - 0.5) * torch.log(1 - 2 * torch.abs(u1 - 0.5) + 1e-7)
    else:
        u1 = torch.rand(shape, device=device, generator=generator)
        noise = -torch.sign(u1 - 0.5) * torch.log(1 - 2 * torch.abs(u1 - 0.5) + 1e-7)
    return noise * scale

def generate_student_t_noise(shape, df=3.0, seed=0, device="cpu", independent_channels=False):
    generator = torch.Generator(device=device).manual_seed(seed)
    b, c, h, w = shape
    
    if independent_channels:
        noise = torch.zeros(shape, device=device)
        for ch in range(c):
            gen_ch = torch.Generator(device=device).manual_seed(seed + ch + 2000)
            z = torch.randn((b, 1, h, w), device=device, generator=gen_ch)
            chi_sq_samples = torch.randn((int(df), b, 1, h, w), device=device, generator=gen_ch) ** 2
            chi_sq = torch.sum(chi_sq_samples, dim=0)
            noise[:, ch:ch+1, :, :] = z / torch.sqrt(chi_sq / df + 1e-7)
    else:
        z = torch.randn(shape, device=device, generator=generator)
        chi_sq_samples = torch.randn((int(df),) + shape, device=device, generator=generator) ** 2
        chi_sq = torch.sum(chi_sq_samples, dim=0)
        noise = z / torch.sqrt(chi_sq / df + 1e-7)
    return noise * 0.7

def _generate_pink_single_channel(shape, device, generator):
    b, c, h, w = shape
    result = torch.zeros(shape, device=device)
    num_octaves = min(8, int(math.log2(min(h, w))))
    total_weight = 0.0
    
    for octave in range(num_octaves):
        scale = 2 ** octave
        if h // scale < 2 or w // scale < 2:
            break
        small_h = max(2, h // scale)
        small_w = max(2, w // scale)
        small_noise = torch.randn((b, c, small_h, small_w), device=device, generator=generator)
        upsampled = F.interpolate(small_noise, size=(h, w), mode='bilinear', align_corners=False)
        weight = 1.0 / (octave + 1)
        result += upsampled * weight
        total_weight += weight
    
    if total_weight > 0:
        result = result / total_weight
        result = result / (torch.std(result) + 1e-7)
    return result

def generate_pink_noise(shape, seed, device, independent_channels=False):
    generator = torch.Generator(device=device).manual_seed(seed)
    b, c, h, w = shape
    if independent_channels:
        noise = torch.zeros(shape, device=device)
        for ch in range(c):
            gen_ch = torch.Generator(device=device).manual_seed(seed + ch + 3000)
            noise[:, ch:ch+1, :, :] = _generate_pink_single_channel((b, 1, h, w), device, gen_ch)
    else:
        noise = _generate_pink_single_channel(shape, device, generator)
    return noise

def _generate_haar_single_channel(shape, scale, device, generator):
    b, c, h, w = shape
    decomp_levels = min(3, int(math.log2(min(h, w))) - 1)
    if decomp_levels < 1:
        return torch.randn(shape, device=device, generator=generator) * scale
    
    result = torch.zeros(shape, device=device)
    for level in range(decomp_levels):
        scale_factor = 2 ** (level + 1)
        coeff_h = max(1, h // scale_factor)
        coeff_w = max(1, w // scale_factor)
        coeffs = torch.randn((b, c, coeff_h, coeff_w), device=device, generator=generator)
        upsampled = F.interpolate(coeffs, size=(h, w), mode='nearest')
        weight = scale / (level + 1)
        result += upsampled * weight
    
    fine = torch.randn(shape, device=device, generator=generator)
    result += fine * (scale * 0.3)
    return result

def generate_haar_wavelet_noise(shape, scale, seed, device, independent_channels=False):
    generator = torch.Generator(device=device).manual_seed(seed)
    b, c, h, w = shape
    if independent_channels:
        noise = torch.zeros(shape, device=device)
        for ch in range(c):
            gen_ch = torch.Generator(device=device).manual_seed(seed + ch + 4000)
            noise[:, ch:ch+1, :, :] = _generate_haar_single_channel((b, 1, h, w), scale, device, gen_ch)
    else:
        noise = _generate_haar_single_channel(shape, scale, device, generator)
    return noise

def _generate_daubechies_single_channel(shape, scale, device, generator):
    b, c, h, w = shape
    decomp_levels = min(4, int(math.log2(min(h, w))) - 1)
    if decomp_levels < 1:
        return torch.randn(shape, device=device, generator=generator) * scale
    
    result = torch.zeros(shape, device=device)
    for level in range(decomp_levels):
        scale_factor = 2 ** (level + 1)
        coeff_h = max(2, h // scale_factor)
        coeff_w = max(2, w // scale_factor)
        coeffs = torch.randn((b, c, coeff_h, coeff_w), device=device, generator=generator)
        upsampled = F.interpolate(coeffs, size=(h, w), mode='bicubic', align_corners=False)
        weight = (scale / scale_factor) * 0.8
        result += upsampled * weight
    
    detail = torch.randn(shape, device=device, generator=generator)
    result += detail * (scale * 0.2)
    return result

def generate_daubechies_wavelet_noise(shape, scale, seed, device, independent_channels=False):
    generator = torch.Generator(device=device).manual_seed(seed)
    b, c, h, w = shape
    if independent_channels:
        noise = torch.zeros(shape, device=device)
        for ch in range(c):
            gen_ch = torch.Generator(device=device).manual_seed(seed + ch + 5000)
            noise[:, ch:ch+1, :, :] = _generate_daubechies_single_channel((b, 1, h, w), scale, device, gen_ch)
    else:
        noise = _generate_daubechies_single_channel(shape, scale, device, generator)
    return noise

# =================================================================================
# == Core Dispatcher
# =================================================================================

def get_noise_tensor(shape, algo, scale, seed, device, independent_channels):
    generator = torch.Generator(device=device).manual_seed(seed)
    if len(shape) != 4:
        return torch.randn(shape, device=device, generator=generator)

    b, c, h, w = shape
    
    if algo == "Gaussian":
        return torch.randn(shape, device=device, generator=generator)
    elif algo == "Uniform":
        return (torch.rand(shape, device=device, generator=generator) - 0.5) * 3.46
    elif algo == "Laplacian":
        return generate_laplacian_noise(shape, scale, seed, device, independent_channels)
    elif algo == "Student-t":
        return generate_student_t_noise(shape, 3.0, seed, device, independent_channels)
    elif algo == "Perlin":
        return generate_perlin_noise(shape, scale, seed, device, independent_channels)
    elif algo == "Voronoi (Euclidean)":
        return generate_voronoi_noise(shape, scale, "euclidean", seed, device, independent_channels)
    elif algo == "Voronoi (Manhattan)":
        return generate_voronoi_noise(shape, scale, "manhattan", seed, device, independent_channels)
    elif algo == "Collatz (Orbit)":
        return generate_collatz_noise(shape, scale, seed, device, independent_channels)
    elif algo == "Wavelet (Haar)":
        return generate_haar_wavelet_noise(shape, scale, seed, device, independent_channels)
    elif algo == "Wavelet (Daubechies)":
        return generate_daubechies_wavelet_noise(shape, scale, seed, device, independent_channels)
    elif "Pyramid" in algo:
        small_shape = (b, c, max(1, h // 2), max(1, w // 2))
        small = torch.randn(small_shape, device=device, generator=generator)
        if "Bislerp" in algo:
            up_bi = F.interpolate(small, size=(h, w), mode='bilinear', align_corners=False)
            up_cu = F.interpolate(small, size=(h, w), mode='bicubic', align_corners=False)
            upscaled = up_bi * 0.5 + up_cu * 0.5
        else:
            upscaled = F.interpolate(small, size=(h, w), mode='bilinear', align_corners=False)
        detail = torch.randn(shape, device=device, generator=generator)
        return upscaled * 0.6 + detail * 0.4
    elif algo == "Distro (Power Normal)":
        base = torch.randn(shape, device=device, generator=generator)
        return torch.sign(base) * torch.pow(torch.abs(base), max(0.1, scale))
    elif algo == "Pink Noise":
        return generate_pink_noise(shape, seed, device, independent_channels)
    
    return torch.randn(shape, device=device, generator=generator)

# =================================================================================
# == Core Processing & Blending
# =================================================================================

def blend_tensors(n1, n2, mode, factor):
    if mode == "Add":
        return n1 * (1 - factor) + n2 * factor
    elif mode == "Average":
        return n1 * (1 - factor) + n2 * factor
    elif mode == "Multiply":
        multiplied = n1 * n2
        return n1 * (1 - factor) + multiplied * factor
    elif mode == "Screen":
        screened = 1 - (1 - n1) * (1 - n2)
        return n1 * (1 - factor) + screened * factor
    elif mode == "Overlay":
        multiplied = 2 * n1 * n2
        screened = 1 - 2 * (1 - n1) * (1 - n2)
        overlay = torch.where(n1 < 0, multiplied, screened)
        return n1 * (1 - factor) + overlay * factor
    elif mode == "Difference":
        return torch.abs(n1 - n2 * factor)
    elif mode == "Exclusion":
        excluded = n1 + n2 - 2 * n1 * n2
        return n1 * (1 - factor) + excluded * factor
    elif mode == "Max":
        return torch.max(n1, n2 * factor)
    elif mode == "Min":
        return torch.min(n1, n2 * factor)
    return n1 * (1 - factor) + n2 * factor

def normalize_tensor(tensor, mode):
    if mode == "Clamp (-1 to 1)":
        return torch.clamp(tensor, -1.0, 1.0)
    elif mode == "Auto-Norm (Std Dev)":
        return (tensor - tensor.mean()) / (tensor.std() + 1e-6)
    return tensor


# ==============================================================================
# == Unit Tests (smoke — runs without ComfyUI)
# ==============================================================================

if __name__ == "__main__":
    print("\n🧪 Smoke tests: custom_noise_core")
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

    _check("VERSION defined",    VERSION == "v1.6.1")
    _check("CONST CONST_JS_MAX_SAFE_INTEGER defined", CONST_JS_MAX_SAFE_INTEGER is not None)
    _check("CONST CONST_SEED_MIN defined", CONST_SEED_MIN is not None)
    _check("fn validate_seed is callable", callable(validate_seed))
    _check("fn generate_perlin_noise is callable", callable(generate_perlin_noise))
    _check("fn generate_voronoi_noise is callable", callable(generate_voronoi_noise))

    print(f"\n  {_pass} passed, {_fail} failed")
    if _fail == 0:
        print("  🎉 All good.")
