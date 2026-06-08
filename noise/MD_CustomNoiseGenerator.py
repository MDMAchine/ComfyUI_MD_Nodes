# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# █▓▒░                                                                     ░▒▓█
# █▓▒░            MD_Nodes/Noise – Advanced Noise Suite v2.1.0             ░▒▓█
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
# ║ ░▒▓ ORIGIN & DEV:
# ║   • Cast into the void by: MDMAchine
# ║   • Enhanced by: Gemini
# ║   • CREDIT: The Multi-Input Blender and Pyramid noise concepts were 
# ║     inspired by the 'Bleppings Sonar' suite of noise utilities.
# ╚════════════════════════════════════════════════════════════════════════════


VERSION = "v2.1.0"  # UPS v1.5.8

import os
import sys
import math
import logging
import io

import torch
import torch.nn.functional as F
import numpy as np
import comfy.model_management

# =================================================================================
# == MD_Nodes Universal Binary Loader (v1.6.1)
# =================================================================================

def find_core_paths():
    """Locate core directory in dev and production environments."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(current_dir, "core"),
        os.path.join(current_dir, "..", "core"),
        os.path.join(current_dir, "..", "..", "core")
    ]
    return [os.path.abspath(c) for c in candidates if os.path.exists(c)]

CORE_LOCATIONS = find_core_paths()
CORE_LOADED = False
CORE_MODE = None
CORE_ERROR = None

for loc in CORE_LOCATIONS:
    if loc not in sys.path:
        sys.path.insert(0, loc)

try:
    import custom_noise_core_bin as core
    CORE_LOADED = True
    CORE_MODE = "Binary (Production)"
except ImportError as e1:
    try:
        import custom_noise_core as core
        CORE_LOADED = True
        CORE_MODE = "Source (Development)"
    except ImportError as e2:
        CORE_ERROR = f"Binary: {e1}\nSource: {e2}"

# =================================================================================
# == Configuration Constants
# =================================================================================

CONST_JS_MAX_SAFE_INTEGER = 9007199254740991
CONST_SEED_MIN = 0

CONST_NOISE_TYPES = [
    "Gaussian", "Uniform", "Laplacian", "Student-t",
    "Perlin", "Voronoi (Euclidean)", "Voronoi (Manhattan)",
    "Collatz (Orbit)", 
    "Wavelet (Haar)", "Wavelet (Daubechies)",
    "HiRes Pyramid", "HiRes Pyramid (Bislerp)", "Pyramid (Standard)",
    "Distro (Power Normal)", "Pink Noise",
]

CONST_BLEND_MODES = [
    "Add", "Multiply", "Average", "Max", "Min",
    "Screen", "Overlay", "Difference", "Exclusion"
]

CONST_NORMALIZE_MODES = [
    "Disabled", "Clamp (-1 to 1)", "Auto-Norm (Std Dev)", "Renormalize to Base"
]

def validate_seed(seed_value):
    try:
        val = int(seed_value)
    except (ValueError, TypeError):
        return CONST_SEED_MIN
    return max(CONST_SEED_MIN, min(val, CONST_JS_MAX_SAFE_INTEGER))

# =================================================================================
# == Noise Object Definitions
# =================================================================================

class MD_NoiseObject:
    """Standard Noise Object compatible with SamplerCustom."""
    def __init__(self, noise_type, scale, strength, seed, extra_params=None):
        self.noise_type = noise_type
        self.scale = scale
        self.strength = strength
        self.seed = validate_seed(seed)
        self.extra_params = extra_params or {}
        self.independent_channels = self.extra_params.get("independent_channels", False)

    def generate_noise(self, input_latent):
        if isinstance(input_latent, dict):
            latents = input_latent['samples']
        else:
            latents = input_latent

        original_shape = latents.shape
        device = latents.device
        
        if len(original_shape) == 5:
            b, f, c, h, w = original_shape
            processing_shape = (b * f, c, h, w)
            is_video = True
        else:
            processing_shape = original_shape
            is_video = False

        # Graceful Degradation: Fallback to Gaussian if Core is missing
        if CORE_LOADED:
            noise = core.get_noise_tensor(
                processing_shape, self.noise_type, self.scale, self.seed, device, self.independent_channels
            )
        else:
            logging.warning(f"❌ Core missing. Falling back to Gaussian noise. Mode: {CORE_MODE or 'Not Loaded'}")
            generator = torch.Generator(device=device).manual_seed(self.seed)
            noise = torch.randn(processing_shape, device=device, generator=generator)
        
        if is_video:
            noise = noise.view(original_shape)

        return noise * self.strength

# =================================================================================
# == Node 1: Generator
# =================================================================================

class MD_CustomNoiseGenerator:
    """Creates a standalone NOISE object and renders a preview."""
    DESCRIPTION = "Generates advanced noise patterns with visual preview."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "noise_type": (CONST_NOISE_TYPES, {
                    "default": "Gaussian",
                    "tooltip": (
                        "NOISE TYPE\n"
                        "• Purpose: Select the mathematical algorithm used to generate the pattern.\n"
                        "• Options: 17 types including Voronoi, Wavelets, and Heavy-tail distributions.\n"
                        "• Trade-offs: Complex types (e.g., Wavelet, Collatz) take slightly longer to generate.\n"
                        "\n⭐ Recommended: Perlin or Wavelet for creative texture."
                    )
                }),
                "scale": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1,
                    "tooltip": (
                        "NOISE SCALE\n"
                        "• Purpose: Controls the frequency/zoom of the pattern.\n"
                        "• Range: 0.1 (Macro/Zoomed in) to 10.0 (Micro/Zoomed out).\n"
                        "\n⭐ Recommended: 1.0"
                    )
                }),
                "strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05,
                    "tooltip": (
                        "NOISE STRENGTH\n"
                        "• Purpose: Global amplitude multiplier for the output tensor.\n"
                        "• Effect: Determines how aggressively the noise replaces standard Gaussian generation.\n"
                        "\n⭐ Recommended: 1.0"
                    )
                }),
                "channel_mode": (["Shared (Monochrome)", "Independent (Color)"], {
                    "default": "Shared (Monochrome)",
                    "tooltip": (
                        "CHANNEL MODE\n"
                        "• Purpose: Determines if noise patterns vary across latent channels.\n"
                        "• Shared: Same pattern on all channels (Good for structural composition).\n"
                        "• Independent: Different pattern per channel (Good for color/texture variation).\n"
                        "\n⭐ Recommended: Shared (Monochrome)"
                    )
                }),
                "seed": ("INT", {
                    "default": 0, "min": CONST_SEED_MIN, "max": CONST_JS_MAX_SAFE_INTEGER,
                    "tooltip": (
                        "RANDOM SEED\n"
                        "• Purpose: Deterministic starting point for noise generation.\n"
                        "• Note: Capped at 9 Quadrillion to prevent UI rounding errors.\n"
                        "\n⭐ Recommended: Use a fixed value for reproducible outputs."
                    )
                }),
                "preview_width": ("INT", {
                    "default": 512, "min": 64, "max": 2048, 
                    "tooltip": "PREVIEW WIDTH\n• Purpose: Resolution for the preview image output."
                }),
                "preview_height": ("INT", {
                    "default": 512, "min": 64, "max": 2048,
                    "tooltip": "PREVIEW HEIGHT\n• Purpose: Resolution for the preview image output."
                }),
            }
        }

    RETURN_TYPES = ("NOISE", "IMAGE")
    RETURN_NAMES = ("noise", "preview_image")
    FUNCTION = "create"
    CATEGORY = "MD_Nodes/Noise"

    def create(self, noise_type, scale, strength, channel_mode, seed, preview_width, preview_height):
        extra = {"independent_channels": (channel_mode == "Independent (Color)")}
        noise_obj = MD_NoiseObject(noise_type, scale, strength, seed, extra)
        
        try:
            device = comfy.model_management.get_torch_device()
            dummy_latent = torch.zeros((1, 4, preview_height // 8, preview_width // 8), device=device)
            
            # Generate noise WITH scale/strength applied
            raw_noise = noise_obj.generate_noise(dummy_latent)
            
            # Visualize: First 3 channels (RGB)
            viz_tensor = raw_noise[0, :3, :, :].clone()
            viz_tensor = F.interpolate(viz_tensor.unsqueeze(0), size=(preview_height, preview_width), mode='nearest').squeeze(0)
            
            min_v, max_v = viz_tensor.min(), viz_tensor.max()
            if max_v - min_v > 1e-6:
                viz_tensor = (viz_tensor - min_v) / (max_v - min_v)
            else:
                viz_tensor = torch.zeros_like(viz_tensor)
            
            # Add text overlay showing scale/strength
            try:
                from PIL import Image, ImageDraw, ImageFont
                img_np = (viz_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
                img_pil = Image.fromarray(img_np, mode='RGB')
                draw = ImageDraw.Draw(img_pil)
                
                try:
                    font = ImageFont.truetype("arial.ttf", 16)
                except Exception:
                    font = ImageFont.load_default()
                
                text = f"Scale: {scale:.2f} | Strength: {strength:.2f}"
                
                try:
                    bbox = draw.textbbox((0, 0), text, font=font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                except Exception:
                    text_width = len(text) * 8
                    text_height = 16
                
                padding = 4
                draw.rectangle(
                    [(10 - padding, 10 - padding), 
                     (10 + text_width + padding, 10 + text_height + padding)],
                    fill=(0, 0, 0, 200)
                )
                
                draw.text((10, 10), text, fill=(255, 255, 0), font=font)
                type_text = f"Type: {noise_type}"
                draw.text((10, 10 + text_height + 8), type_text, fill=(0, 255, 255), font=font)
                
                img_np = np.array(img_pil).astype(np.float32) / 255.0
                image_out = torch.from_numpy(img_np).unsqueeze(0)
            except Exception as e:
                logging.info(f"[Preview] Could not add text overlay: {e}")
                image_out = viz_tensor.permute(1, 2, 0).unsqueeze(0).cpu()
            
        except Exception as e:
            logging.error(f"Preview generation failed: {e}")
            image_out = torch.zeros((1, preview_height, preview_width, 3))

        return (noise_obj, image_out)

# =================================================================================
# == Node 2: Multi-Input Blender
# =================================================================================

class MD_MultiNoiseBlender:
    """Combines up to 5 Noise sources sequentially into a single NOISE object."""
    DESCRIPTION = "Mixes up to 5 different noise sources using advanced blend modes."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_noise": ("NOISE", {"tooltip": "BASE NOISE\n• Purpose: The primary noise source (Layer 1)."}),
                "normalize_result": (CONST_NORMALIZE_MODES, {
                    "default": "Disabled",
                    "tooltip": (
                        "NORMALIZE RESULT\n"
                        "• Purpose: Post-processing method to keep mixed values within valid ranges.\n"
                        "• Options:\n"
                        "  - Clamp: Cuts off extremes.\n"
                        "  - Auto-Norm: Restores standard deviation.\n"
                        "\n⭐ Recommended: Auto-Norm (Std Dev) for complex blends."
                    )
                }),
                "preview_width": ("INT", {"default": 512, "min": 64, "max": 2048, "tooltip": "PREVIEW WIDTH\n• Width for preview output."}),
                "preview_height": ("INT", {"default": 512, "min": 64, "max": 2048, "tooltip": "PREVIEW HEIGHT\n• Height for preview output."}),
            },
            "optional": {
                "noise_2": ("NOISE", {"tooltip": "LAYER 2\n• Optional secondary noise source."}),
                "mode_2": (CONST_BLEND_MODES, {"default": "Add"}),
                "factor_2": ("FLOAT", {"default": 0.5, "min": -2.0, "max": 2.0, "step": 0.05}),
                
                "noise_3": ("NOISE", {"tooltip": "LAYER 3\n• Optional third noise source."}),
                "mode_3": (CONST_BLEND_MODES, {"default": "Add"}),
                "factor_3": ("FLOAT", {"default": 0.5, "min": -2.0, "max": 2.0, "step": 0.05}),
                
                "noise_4": ("NOISE", {"tooltip": "LAYER 4\n• Optional fourth noise source."}),
                "mode_4": (CONST_BLEND_MODES, {"default": "Add"}),
                "factor_4": ("FLOAT", {"default": 0.5, "min": -2.0, "max": 2.0, "step": 0.05}),
                
                "noise_5": ("NOISE", {"tooltip": "LAYER 5\n• Optional fifth noise source."}),
                "mode_5": (CONST_BLEND_MODES, {"default": "Add"}),
                "factor_5": ("FLOAT", {"default": 0.5, "min": -2.0, "max": 2.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("NOISE", "IMAGE")
    RETURN_NAMES = ("blended_noise", "preview_image")
    FUNCTION = "blend_multi"
    CATEGORY = "MD_Nodes/Noise"

    def blend_multi(self, base_noise, normalize_result, preview_width, preview_height, **kwargs):
        
        class MultiStackNoiseWrapper:
            def __init__(self, base, layers, norm_mode):
                self.base = base
                self.layers = layers
                self.norm_mode = norm_mode
                self.seed = getattr(base, "seed", 0)

            def generate_noise(self, input_latent):
                current_noise = self.base.generate_noise(input_latent)
                
                for layer_noise, mode, factor in self.layers:
                    if layer_noise is not None:
                        next_noise = layer_noise.generate_noise(input_latent)
                        # Graceful Degradation: Fallback to basic 'Add' if Core is missing
                        if CORE_LOADED:
                            current_noise = core.blend_tensors(current_noise, next_noise, mode, factor)
                        else:
                            current_noise = current_noise * (1 - factor) + next_noise * factor
                
                if CORE_LOADED:
                    current_noise = core.normalize_tensor(current_noise, self.norm_mode)
                else:
                    if self.norm_mode == "Clamp (-1 to 1)":
                        current_noise = torch.clamp(current_noise, -1.0, 1.0)
                
                return current_noise

        layers = []
        for i in range(2, 6):
            n = kwargs.get(f"noise_{i}")
            if n is not None:
                layers.append((n, kwargs.get(f"mode_{i}", "Add"), kwargs.get(f"factor_{i}", 0.5)))

        wrapper = MultiStackNoiseWrapper(base_noise, layers, normalize_result)
        
        # --- Generate Preview ---
        image_out = torch.zeros((1, preview_height, preview_width, 3)) 
        try:
            device = comfy.model_management.get_torch_device()
            dummy_latent = torch.zeros((1, 4, preview_height // 8, preview_width // 8), device=device)
            final_noise = wrapper.generate_noise(dummy_latent)
            
            viz = final_noise[0, :3, :, :].clone()
            
            if viz.shape[0] == 1:
                viz = viz.repeat(3, 1, 1)
            elif viz.shape[0] > 3:
                viz = viz[:3, :, :]
                
            viz = F.interpolate(viz.unsqueeze(0), size=(preview_height, preview_width), mode='nearest').squeeze(0)
            
            min_v, max_v = viz.min(), viz.max()
            if max_v - min_v > 1e-6:
                viz = (viz - min_v) / (max_v - min_v)
            else:
                viz = torch.zeros_like(viz)
                
            try:
                from PIL import Image, ImageDraw, ImageFont
                img_np = (viz.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
                img_pil = Image.fromarray(img_np, mode='RGB')
                draw = ImageDraw.Draw(img_pil)
                
                try:
                    font = ImageFont.truetype("arial.ttf", 16)
                except Exception:
                    font = ImageFont.load_default()
                
                num_layers = 1 + len([l for l in layers if l[0] is not None])
                text = f"Layers: {num_layers} | Norm: {normalize_result}"
                
                try:
                    bbox = draw.textbbox((0, 0), text, font=font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                except Exception:
                    text_width = len(text) * 8
                    text_height = 16
                
                padding = 4
                draw.rectangle(
                    [(10 - padding, 10 - padding), 
                     (10 + text_width + padding, 10 + text_height + padding)],
                    fill=(0, 0, 0, 200)
                )
                
                draw.text((10, 10), text, fill=(255, 255, 0), font=font)
                if num_layers > 1:
                    layer_text = f"Base + {num_layers - 1} blend(s)"
                    draw.text((10, 10 + text_height + 8), layer_text, fill=(0, 255, 255), font=font)
                
                img_np = np.array(img_pil).astype(np.float32) / 255.0
                image_out = torch.from_numpy(img_np).unsqueeze(0)
            except Exception as text_e:
                logging.info(f"[Blender Preview] Could not add text overlay: {text_e}")
                image_out = viz.permute(1, 2, 0).unsqueeze(0).cpu()
            
        except Exception as e:
            logging.error(f"Blender preview failed: {e}")
            import traceback
            traceback.print_exc()

        return (wrapper, image_out)

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


# ==============================================================================
# == Unit Tests (smoke — runs without ComfyUI)
# ==============================================================================

if __name__ == "__main__":
    print("\n🧪 Smoke tests: MD_CustomNoiseGenerator")
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

    _check("VERSION defined",    VERSION == "v2.1.0")
    _check("CONST CONST_JS_MAX_SAFE_INTEGER defined", CONST_JS_MAX_SAFE_INTEGER is not None)
    _check("CONST CONST_SEED_MIN defined", CONST_SEED_MIN is not None)
    _check("NODE_CLASS_MAPPINGS defined",
           isinstance(NODE_CLASS_MAPPINGS, dict) and len(NODE_CLASS_MAPPINGS) > 0)
    _check("  class MD_CustomNoiseGenerator in map", "MD_CustomNoiseGenerator" in NODE_CLASS_MAPPINGS)
    _check("  class MD_MultiNoiseBlender in map", "MD_MultiNoiseBlender" in NODE_CLASS_MAPPINGS)

    print(f"\n  {_pass} passed, {_fail} failed")
    if _fail == 0:
        print("  🎉 All good.")
