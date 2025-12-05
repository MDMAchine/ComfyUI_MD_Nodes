# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
# ████ MD_Nodes/Utilities – Empty Latent Ratio Generator ███████████▓▒░
# © 2025 MDMAchine
# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
# ░▒▓ ORIGIN & DEV:
#   • Cast into the void by: MDMAchine
#   • Enhanced by: Gemini
#   • License: Apache 2.0
#
# ░▒▓ DESCRIPTION:
#   Generates empty latents based on aspect ratio and megapixel targets.
#   Designed for SDXL/SD3/Flux workflows where total pixel count matters.
#
# ░▒▓ FEATURES:
#   ✓ Presets for 1:1, 16:9, 21:9, etc.
#   ✓ Landscape/Portrait orientation toggle
#   ✓ Megapixel targeting (SD1.5, SDXL, 1080p, 4K)
#   ✓ Output dimension integers for workflow piping
#
# ░▒▓ CHANGELOG:
#   - v1.0.0 (Initial):
#       • ADDED: Core ratio logic and orientation switching
#       • ADDED: Dimension scaling and flexible megapixel inputs

# =================================================================================
# == Standard Library Imports
# =================================================================================
import math
import logging

# =================================================================================
# == Third-Party Imports
# =================================================================================
import torch

# =================================================================================
# == ComfyUI Core Modules
# =================================================================================
import comfy.model_management

# =================================================================================
# == Configuration Constants
# =================================================================================

# Dimensions must be divisible by this number (standard VAE requirement)
CONST_DIVISIBLE_BY = 8

# Base unit for Megapixel calculations (1024*1024)
CONST_BASE_MP_UNIT = 1048576

# Aspect Ratio Dictionary (Width / Height)
CONST_RATIOS = {
    "1:1 (Square)": 1.0,
    "5:4": 1.25,
    "4:3": 1.3333333333,
    "3:2": 1.5,
    "16:9 (Standard)": 1.7777777778,
    "2:1 (Cinema)": 2.0,
    "21:9 (Ultrawide)": 2.3333333333,
    "32:9 (Super Ultrawide)": 3.5555555556
}

# Megapixel Presets
CONST_MP_PRESETS = {
    "SD1.5 (512x512) - 0.26 MP": 0.262144,
    "SDXL (1024x1024) - 1.0 MP": 1.0,
    "SDXL High (1.5 MP)": 1.5,
    "1080p (2.0 MP)": 1.9775,
    "4K (8.3 MP)": 8.2944,
    "Custom (Use Slider)": -1.0
}

# =================================================================================
# == Core Node Class
# =================================================================================

class MD_EmptyLatentRatioSelector:
    """
    Generates empty latents based on Aspect Ratio and Target Megapixels.
    Calculates precise W/H dimensions to match total pixel count area.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ratio_preset": (list(CONST_RATIOS.keys()), {
                    "default": "16:9 (Standard)",
                    "tooltip": "Select the desired Aspect Ratio."
                }),
                "orientation": (["Landscape (Horizontal)", "Portrait (Vertical)"], {
                    "default": "Landscape (Horizontal)",
                    "tooltip": "Swaps Width and Height calculations."
                }),
                "base_mp_preset": (list(CONST_MP_PRESETS.keys()), {
                    "default": "SDXL (1024x1024) - 1.0 MP",
                    "tooltip": "Target total pixel count (Resolution class)."
                }),
                "manual_mp_size": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 16.0,
                    "step": 0.1,
                    "tooltip": "Used if preset is set to 'Custom'. Represents Millions of Pixels (1.0 = 1024x1024)."
                }),
                "dimension_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 4.0,
                    "step": 0.05,
                    "tooltip": "Multiplies final dimensions. 0.5 = half size, 2.0 = double size."
                }),
                "batch_size": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 64,
                    "tooltip": "Number of latent samples to generate."
                }),
            }
        }

    RETURN_TYPES = ("LATENT", "INT", "INT")
    RETURN_NAMES = ("latent", "width", "height")
    FUNCTION = "generate"
    CATEGORY = "MD_Nodes/Utilities"

    def generate(self, ratio_preset, orientation, base_mp_preset, manual_mp_size, dimension_scale, batch_size):
        """
        Calculates dimensions and returns empty latent.
        Args:
            ratio_preset (str): Key for ratio dict.
            orientation (str): Landscape or Portrait.
            base_mp_preset (str): Key for MP dict.
            manual_mp_size (float): Fallback MP value.
            dimension_scale (float): Multiplier.
            batch_size (int): Batch count.
        Returns:
            tuple: (LATENT, width, height)
        """
        try:
            # 1. Determine Target Megapixels
            if base_mp_preset == "Custom (Use Slider)":
                target_mp = manual_mp_size
            else:
                target_mp = CONST_MP_PRESETS[base_mp_preset]
            
            # Convert MP to total pixels (1.0 MP = 1024*1024 pixels roughly)
            target_area = target_mp * CONST_BASE_MP_UNIT

            # 2. Determine Ratio
            ratio_value = CONST_RATIOS[ratio_preset]

            # 3. Handle Orientation (Flip ratio if Portrait)
            is_portrait = "Portrait" in orientation
            if is_portrait:
                ratio_value = 1.0 / ratio_value

            # 4. Calculate Width and Height from Area and Ratio
            # Area = W * H
            # Ratio = W / H  => W = H * Ratio
            # Area = (H * Ratio) * H => Area = H^2 * Ratio => H = sqrt(Area / Ratio)
            
            height_float = math.sqrt(target_area / ratio_value)
            width_float = height_float * ratio_value

            # 5. Apply Dimension Scaling (Multiply/Divide)
            width_float *= dimension_scale
            height_float *= dimension_scale

            # 6. Round to nearest multiple of 8 (VAE requirement)
            width = int(round(width_float / CONST_DIVISIBLE_BY) * CONST_DIVISIBLE_BY)
            height = int(round(height_float / CONST_DIVISIBLE_BY) * CONST_DIVISIBLE_BY)

            # Ensure minimum dimensions to prevent crashes
            width = max(64, width)
            height = max(64, height)

            # 7. Generate Latent
            # Shape: [batch, 4, height // 8, width // 8]
            latent_tensor = torch.zeros([batch_size, 4, height // 8, width // 8])

            logging.info(f"[MD_EmptyLatent] Generated: {width}x{height} (Ratio: {ratio_preset}, MP: {target_mp:.2f})")

            return ({"samples": latent_tensor}, width, height)

        except Exception as e:
            logging.error(f"[MD_EmptyLatent] Critical Error: {e}")
            # Fallback safe latent
            fallback = torch.zeros([batch_size, 4, 64, 64])
            return ({"samples": fallback}, 512, 512)

# =================================================================================
# == Node Registration
# =================================================================================

NODE_CLASS_MAPPINGS = {
    "MD_EmptyLatentRatioSelector": MD_EmptyLatentRatioSelector
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MD_EmptyLatentRatioSelector": "MD: Empty Latent Ratio Select"
}

# =================================================================================
# == Development & Testing
# =================================================================================

if __name__ == "__main__":
    print("🧪 Running Self-Tests for MD_EmptyLatentRatioSelector...")
    
    test_passed = 0
    test_failed = 0
    
    node = MD_EmptyLatentRatioSelector()
    
    try:
        # Test 1: Standard 16:9 Landscape Calculation
        # 1MP = 1048576 pixels. 16:9 ratio.
        # H = sqrt(1048576 / 1.777) = sqrt(589824) = 768
        # W = 768 * 1.777 = 1365.33 -> 1368 (mod 8)
        result = node.generate(
            "16:9 (Standard)", 
            "Landscape (Horizontal)", 
            "SDXL (1024x1024) - 1.0 MP", 
            1.0, 
            1.0, 
            1
        )
        w, h = result[1], result[2]
        
        # Allow small rounding tolerance due to int conversion
        assert 1360 <= w <= 1376, f"Width calculation off: got {w}"
        assert 760 <= h <= 776, f"Height calculation off: got {h}"
        print("✅ 16:9 Landscape Calc: PASSED")
        test_passed += 1
    except AssertionError as e:
        print(f"❌ 16:9 Landscape Calc: FAILED - {e}")
        test_failed += 1
        
    try:
        # Test 2: Portrait Flip
        # Should roughly swap W and H from previous test
        result = node.generate(
            "16:9 (Standard)", 
            "Portrait (Vertical)", 
            "SDXL (1024x1024) - 1.0 MP", 
            1.0, 
            1.0, 
            1
        )
        w, h = result[1], result[2]
        
        assert 760 <= w <= 776, f"Portrait Width off: got {w}"
        assert 1360 <= h <= 1376, f"Portrait Height off: got {h}"
        print("✅ Portrait Flip: PASSED")
        test_passed += 1
    except AssertionError as e:
        print(f"❌ Portrait Flip: FAILED - {e}")
        test_failed += 1

    try:
        # Test 3: Dimension Scaling
        # 1:1 at 1MP is 1024x1024. Scale 0.5 should be 512x512.
        result = node.generate(
            "1:1 (Square)", 
            "Landscape (Horizontal)", 
            "SDXL (1024x1024) - 1.0 MP", 
            1.0, 
            0.5, 
            1
        )
        w, h = result[1], result[2]
        assert w == 512 and h == 512, f"Scaling failed: {w}x{h}"
        print("✅ Dimension Scaling: PASSED")
        test_passed += 1
    except AssertionError as e:
        print(f"❌ Dimension Scaling: FAILED - {e}")
        test_failed += 1

    print(f"\n{'='*60}")
    print(f"Test Results: {test_passed} passed, {test_failed} failed")
    print(f"{'='*60}")