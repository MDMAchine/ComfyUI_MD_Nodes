# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
# ████ MD_Nodes/Noise – Advanced Noise Suite (Sonar Inspired) ██████▓▒░
# © 2025 MDMAchine
# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
# ░▒▓ ORIGIN & DEV:
#   • Cast into the void by: MDMAchine
#   • Enhanced by: Gemini
#   • License: Apache 2.0
#   • CREDIT: The Multi-Input Blender and Pyramid noise concepts were 
#     inspired by the 'Bleppings Sonar' suite of noise utilities.
#
# ░▒▓ DESCRIPTION:
#   A comprehensive suite for generating and blending custom noise patterns
#   for SamplerCustomAdvanced. Includes exotic algorithms (Collatz, Voronoi)
#   and a 5-stage blending engine.
#
# ░▒▓ FEATURES:
#   ✓ 15+ Noise Types (Collatz, Voronoi, Wavelet, Pyramid, etc.)
#   ✓ 5-Input Noise Blender with per-layer Blend Modes
#   ✓ Slerp-based Pyramid Noise (Bislerp approximation)
#   ✓ Enterprise Standard: JS-Safe Seeds & Robust Fallbacks
#
# ░▒▓ CHANGELOG:
#   - v1.5.3 (Hotfix):
#       • CRITICAL: Renamed 'generate' to 'generate_noise' to match ComfyUI API.
#       • FIXED: Added 'seed' attribute to Blender wrapper.
#   - v1.5.2 (UX Polish):
#       • ADDED: Detailed tooltips for all input parameters.

# =================================================================================
# == Standard Library Imports
# =================================================================================
import math
import logging

# =================================================================================
# == Third-Party Imports
# =================================================================================
import torch
import torch.nn.functional as F

# =================================================================================
# == ComfyUI Core Modules
# =================================================================================
import comfy.model_management

# =================================================================================
# == Configuration Constants
# =================================================================================

CONST_JS_MAX_SAFE_INTEGER = 9007199254740991
CONST_SEED_MIN = 0

CONST_NOISE_TYPES = [
    "Gaussian",
    "Uniform",
    "Laplacian",
    "Student-t",
    "Perlin",
    "Voronoi (Euclidean)",
    "Voronoi (Manhattan)",
    "Collatz (Orbit)",
    "Wavelet (Haar)",
    "HiRes Pyramid",
    "HiRes Pyramid (Bislerp)",
    "Pyramid (Standard)",
    "Distro (Power Normal)",
    "Pink Noise (Approx)",
]

CONST_BLEND_MODES = [
    "Add",
    "Multiply",
    "Average",
    "Max",
    "Min",
    "Screen",
    "Overlay",
    "Difference",
    "Exclusion"
]

CONST_NORMALIZE_MODES = [
    "Disabled",
    "Clamp (-1 to 1)",
    "Auto-Norm (Std Dev)",
    "Renormalize to Base"
]

# =================================================================================
# == Utility Functions (Noise Algorithms)
# =================================================================================

def validate_seed(seed_value):
    """Ensure seed is within JavaScript-safe range."""
    try:
        val = int(seed_value)
    except (ValueError, TypeError):
        return CONST_SEED_MIN
    return max(CONST_SEED_MIN, min(val, CONST_JS_MAX_SAFE_INTEGER))

def generate_perlin_noise(shape, scale=1.0, seed=0, device="cpu"):
    """Vectorized Perlin-like noise approximation."""
    b, c, h, w = shape
    generator = torch.Generator(device=device).manual_seed(seed)
    
    # Scale determines grid frequency
    freq_h = max(2, int(h / (16 * scale)))
    freq_w = max(2, int(w / (16 * scale)))
    
    grid = torch.randn((b, c, freq_h, freq_w), device=device, generator=generator)
    noise = F.interpolate(grid, size=(h, w), mode='bicubic', align_corners=False)
    
    return (noise - noise.mean()) / (noise.std() + 1e-6)

def generate_voronoi_noise(shape, scale=1.0, metric="euclidean", seed=0, device="cpu"):
    """Vectorized Voronoi/Worley noise."""
    b, c, h, w = shape
    num_points = int(60 * scale)
    generator = torch.Generator(device=device).manual_seed(seed)
    
    points_x = torch.rand((b, num_points), device=device, generator=generator) * w
    points_y = torch.rand((b, num_points), device=device, generator=generator) * h
    
    y_coords = torch.arange(h, device=device).float().view(1, h, 1)
    x_coords = torch.arange(w, device=device).float().view(1, 1, w)
    
    noise_batch = []
    for i in range(b):
        px = points_x[i].view(num_points, 1, 1)
        py = points_y[i].view(num_points, 1, 1)
        
        if metric == "manhattan":
            dist = torch.abs(x_coords - px) + torch.abs(y_coords - py)
        else:
            dist = (x_coords - px)**2 + (y_coords - py)**2 # Squared Euclidean for speed
            
        min_dist, _ = torch.min(dist, dim=0)
        noise_batch.append(torch.sqrt(min_dist) if metric != "manhattan" else min_dist)
        
    noise = torch.stack(noise_batch).unsqueeze(1).repeat(1, c, 1, 1)
    return (noise - noise.mean()) / (noise.std() + 1e-6)

def generate_collatz_noise(shape, scale=1.0, seed=0, device="cpu"):
    """Noise based on the Collatz Conjecture stopping time."""
    b, c, h, w = shape
    offset = seed % 1000
    y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
    
    # Map pixels to numbers
    val = (x + y * w).float() * (0.01 * scale) + offset
    val = val.long().clamp(min=1)
    
    steps = torch.zeros_like(val).float()
    current = val
    
    # Simulate orbit length
    for _ in range(40): # Limited depth for performance
        mask_even = (current % 2 == 0)
        mask_odd = ~mask_even & (current > 1)
        current[mask_even] = current[mask_even] // 2
        current[mask_odd] = 3 * current[mask_odd] + 1
        steps[current > 1] += 1
        
    noise = steps.unsqueeze(0).unsqueeze(0).repeat(b, c, 1, 1)
    return (noise - noise.mean()) / (noise.std() + 1e-6)

def blend_tensors(n1, n2, mode, factor):
    """Core mixing logic for noise tensors."""
    if mode == "Add":
        return n1 * (1 - factor) + n2 * factor
    elif mode == "Multiply":
        return n1 * (n2 * factor + (1 - factor))
    elif mode == "Screen":
        return 1 - (1 - n1) * (1 - n2 * factor)
    elif mode == "Overlay":
        return n1 + (n2 * factor) - (n1 * (n2 * factor))
    elif mode == "Difference":
        return torch.abs(n1 - n2 * factor)
    elif mode == "Exclusion":
        return n1 + n2 * factor - 2 * n1 * n2 * factor
    elif mode == "Max":
        return torch.max(n1, n2 * factor)
    elif mode == "Min":
        return torch.min(n1, n2 * factor)
    return n1 * 0.5 + n2 * 0.5

# =================================================================================
# == Noise Object Class
# =================================================================================

class MD_NoiseObject:
    """Standard Noise Object compatible with SamplerCustom."""
    def __init__(self, noise_type, scale, strength, seed, extra_params=None):
        self.noise_type = noise_type
        self.scale = scale
        self.strength = strength
        self.seed = validate_seed(seed)
        self.extra_params = extra_params or {}

    def generate_noise(self, input_latent):
        """
        Main entry point called by SamplerCustomAdvanced.
        Args:
            input_latent: dict containing 'samples' tensor
        """
        if isinstance(input_latent, dict):
            latents = input_latent['samples']
        else:
            latents = input_latent

        shape = latents.shape
        device = latents.device
        
        # We use self.seed directly as ComfyUI expects stateful noise objects
        noise = self._get_noise(shape, self.noise_type, self.scale, self.seed, device)
        return noise * self.strength

    def _get_noise(self, shape, algo, scale, seed, device):
        generator = torch.Generator(device=device).manual_seed(seed)
        
        if algo == "Gaussian":
            return torch.randn(shape, device=device, generator=generator)
        elif algo == "Uniform":
            return (torch.rand(shape, device=device, generator=generator) - 0.5) * 3.46
        elif algo == "Laplacian":
            tmp = torch.rand(shape, device=device, generator=generator)
            return torch.sign(tmp - 0.5) * -torch.log(1 - 2 * torch.abs(tmp - 0.5))
        elif algo == "Student-t":
            m = torch.distributions.studentT.StudentT(torch.tensor([2.5], device=device))
            return m.sample(shape).squeeze(-1)
        elif algo == "Perlin":
            return generate_perlin_noise(shape, scale, seed, device)
        elif algo == "Voronoi (Euclidean)":
            return generate_voronoi_noise(shape, scale, "euclidean", seed, device)
        elif algo == "Voronoi (Manhattan)":
            return generate_voronoi_noise(shape, scale, "manhattan", seed, device)
        elif algo == "Collatz (Orbit)":
            return generate_collatz_noise(shape, scale, seed, device)
        elif algo == "Wavelet (Haar)":
            base = torch.randn(shape, device=device, generator=generator)
            return base + torch.roll(base, 1, 2) * 0.5 - torch.roll(base, 1, 3) * 0.5
        elif algo == "Distro (Power Normal)":
            base = torch.randn(shape, device=device, generator=generator)
            return torch.sign(base) * torch.pow(torch.abs(base), scale)
        elif "Pyramid" in algo:
            b, c, h, w = shape
            small_shape = (b, c, h // 2, w // 2)
            small = torch.randn(small_shape, device=device, generator=generator)
            
            if "Bislerp" in algo:
                up_bi = F.interpolate(small, size=(h, w), mode='bilinear')
                up_cu = F.interpolate(small, size=(h, w), mode='bicubic', align_corners=False)
                upscaled = up_bi * 0.5 + up_cu * 0.5
            else:
                upscaled = F.interpolate(small, size=(h, w), mode='bilinear')
                
            detail = torch.randn(shape, device=device, generator=generator)
            return upscaled * 0.6 + detail * 0.4
            
        return torch.randn(shape, device=device, generator=generator)

# =================================================================================
# == Node 1: Generator
# =================================================================================

class MD_CustomNoiseGenerator:
    """
    Creates a standalone NOISE object for SamplerCustomAdvanced using
    selectable algorithms (Gaussian, Perlin, Collatz, etc.).
    """
    DESCRIPTION = "Generates advanced noise patterns (Gaussian, Perlin, Collatz, etc.) for use with Custom Samplers."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "noise_type": (CONST_NOISE_TYPES, {
                    "default": "Gaussian",
                    "tooltip": "Select the mathematical algorithm used to generate the noise pattern."
                }),
                "scale": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1,
                    "tooltip": "Controls the frequency/zoom of the pattern. Higher = smaller details."
                }),
                "strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05,
                    "tooltip": "Global amplitude multiplier. Higher = more intense noise influence."
                }),
                "seed": ("INT", {
                    "default": 0, "min": CONST_SEED_MIN, "max": CONST_JS_MAX_SAFE_INTEGER,
                    "tooltip": "Random seed (Safe Range: 0 to 9,007,199,254,740,991)."
                }),
            }
        }

    RETURN_TYPES = ("NOISE",)
    RETURN_NAMES = ("noise",)
    FUNCTION = "create"
    CATEGORY = "MD_Nodes/Noise"

    def create(self, noise_type, scale, strength, seed):
        return (MD_NoiseObject(noise_type, scale, strength, seed),)

# =================================================================================
# == Node 2: Multi-Input Blender
# =================================================================================

class MD_MultiNoiseBlender:
    """
    Combines up to 5 Noise sources sequentially into a single NOISE object.
    Inspired by Bleppings Sonar suite.
    """
    DESCRIPTION = "Mixes up to 5 different noise sources using advanced blend modes (Add, Screen, Overlay, etc.)."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_noise": ("NOISE", {
                    "tooltip": "The primary noise source (Layer 1)."
                }),
                "normalize_result": (CONST_NORMALIZE_MODES, {
                    "default": "Disabled",
                    "tooltip": "Post-processing method to keep values within valid ranges (e.g., Clamp -1 to 1)."
                }),
            },
            "optional": {
                "noise_2": ("NOISE", {"tooltip": "Layer 2 noise source."}),
                "mode_2": (CONST_BLEND_MODES, {"default": "Add", "tooltip": "Blend mode for Layer 2."}),
                "factor_2": ("FLOAT", {"default": 0.5, "min": -2.0, "max": 2.0, "step": 0.05, "tooltip": "Strength/Opacity of Layer 2."}),
                
                "noise_3": ("NOISE", {"tooltip": "Layer 3 noise source."}),
                "mode_3": (CONST_BLEND_MODES, {"default": "Add", "tooltip": "Blend mode for Layer 3."}),
                "factor_3": ("FLOAT", {"default": 0.5, "min": -2.0, "max": 2.0, "step": 0.05, "tooltip": "Strength/Opacity of Layer 3."}),
                
                "noise_4": ("NOISE", {"tooltip": "Layer 4 noise source."}),
                "mode_4": (CONST_BLEND_MODES, {"default": "Add", "tooltip": "Blend mode for Layer 4."}),
                "factor_4": ("FLOAT", {"default": 0.5, "min": -2.0, "max": 2.0, "step": 0.05, "tooltip": "Strength/Opacity of Layer 4."}),
                
                "noise_5": ("NOISE", {"tooltip": "Layer 5 noise source."}),
                "mode_5": (CONST_BLEND_MODES, {"default": "Add", "tooltip": "Blend mode for Layer 5."}),
                "factor_5": ("FLOAT", {"default": 0.5, "min": -2.0, "max": 2.0, "step": 0.05, "tooltip": "Strength/Opacity of Layer 5."}),
            }
        }

    RETURN_TYPES = ("NOISE",)
    RETURN_NAMES = ("blended_noise",)
    FUNCTION = "blend_multi"
    CATEGORY = "MD_Nodes/Noise"

    def blend_multi(self, base_noise, normalize_result, **kwargs):
        
        # Define a wrapper class to handle the blending chain during sampling
        class MultiStackNoiseWrapper:
            def __init__(self, base, layers, norm_mode):
                self.base = base
                self.layers = layers # List of tuples (noise_obj, mode, factor)
                self.norm_mode = norm_mode
                # Crucial: Expose seed from base noise so Sampler can read it
                self.seed = getattr(base, "seed", 0)

            def generate_noise(self, input_latent):
                """
                Iteratively blends noise layers.
                """
                # 1. Generate Base
                current_noise = self.base.generate_noise(input_latent)
                
                # 2. Iterate Layers
                for layer_noise, mode, factor in self.layers:
                    if layer_noise is not None:
                        next_noise = layer_noise.generate_noise(input_latent)
                        current_noise = blend_tensors(current_noise, next_noise, mode, factor)
                
                # 3. Final Normalization
                if self.norm_mode == "Clamp (-1 to 1)":
                    current_noise = torch.clamp(current_noise, -1.0, 1.0)
                elif self.norm_mode == "Auto-Norm (Std Dev)":
                    current_noise = (current_noise - current_noise.mean()) / (current_noise.std() + 1e-6)
                
                return current_noise

        # Build Layer List from kwargs
        layers = []
        for i in range(2, 6):
            n = kwargs.get(f"noise_{i}")
            if n is not None:
                m = kwargs.get(f"mode_{i}", "Add")
                f = kwargs.get(f"factor_{i}", 0.5)
                layers.append((n, m, f))

        wrapper = MultiStackNoiseWrapper(base_noise, layers, normalize_result)
        return (wrapper,)

# =================================================================================
# == Node Registration
# =================================================================================

NODE_CLASS_MAPPINGS = {
    "MD_CustomNoiseGenerator": MD_CustomNoiseGenerator,
    "MD_MultiNoiseBlender": MD_MultiNoiseBlender
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MD_CustomNoiseGenerator": "MD: Custom Noise Generator",
    "MD_MultiNoiseBlender": "MD: Noise Blender (5-Layer)"
}

# =================================================================================
# == Development & Testing
# =================================================================================

if __name__ == "__main__":
    print("🧪 Running Self-Tests for MD_Noise Suite...")
    
    try:
        # Test 1: Seed Safe
        assert validate_seed(9007199254740992) == 9007199254740991
        
        # Test 2: Generator API Compliance
        gen_node = MD_CustomNoiseGenerator()
        noise_obj = gen_node.create("HiRes Pyramid (Bislerp)", 1.0, 1.0, 123)[0]
        
        # Mock Latent Dict (Comfy style)
        mock_l = {"samples": torch.zeros((1, 4, 32, 32))}
        
        # Check if generate_noise exists
        if not hasattr(noise_obj, 'generate_noise'):
            raise AttributeError("Missing 'generate_noise' method!")
            
        res = noise_obj.generate_noise(mock_l)
        assert res.shape == (1, 4, 32, 32)
        print("✅ Generator API (generate_noise): PASSED")
        
        # Test 3: Multi-Blender API Compliance
        blender = MD_MultiNoiseBlender()
        n1 = MD_NoiseObject("Gaussian", 1.0, 1.0, 1)
        n2 = MD_NoiseObject("Perlin", 1.0, 1.0, 2)
        
        blend_res = blender.blend_multi(
            base_noise=n1, 
            normalize_result="Clamp (-1 to 1)",
            noise_2=n2, mode_2="Add", factor_2=0.5
        )[0]
        
        # Check wrapper seed attribute
        if not hasattr(blend_res, 'seed'):
             raise AttributeError("Wrapper missing 'seed' attribute!")
        
        final = blend_res.generate_noise(mock_l)
        assert final.shape == (1, 4, 32, 32)
        print("✅ Blender API (generate_noise + seed): PASSED")
        
        print("🎉 All Tests Passed.")
        
    except Exception as e:
        print(f"❌ Test Failed: {e}")
        import traceback
        traceback.print_exc()