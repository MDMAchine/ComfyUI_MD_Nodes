# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# █▓▒░                                                                     ░▒▓█
# █▓▒░      MD_Nodes/Utilities – Empty Latent Ratio Generator v1.6.0       ░▒▓█
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
# ║
# ║ ░▒▓ DESCRIPTION:
# ║   Generates empty latents based on aspect ratio and megapixel targets.
# ║   Designed for SDXL/SD3/Flux workflows where total pixel count matters.
# ║   NOTE: As a mathematical utility, this runs entirely in the public wrapper domain.
# ║
# ║ ░▒▓ FEATURES:
# ║   ✓ Presets for 1:1, 16:9, 21:9, etc.
# ║   ✓ Landscape/Portrait orientation toggle
# ║   ✓ Megapixel targeting (SD1.5, SDXL, 1080p, 4K)
# ║   ✓ Output dimension integers for workflow piping
# ║
# ║ ░▒▓ CHANGELOG:
# ║   - v1.6.0 (Enterprise Standards - Feb 2026):
# ║       • REFACTOR: Tooltips strictly updated to 5-part v1.5.4 standard.
# ║       • VERIFIED: PerformanceProfiler matches v1.5.3 exact specifications.
# ║   - v1.5.4 (Prior Update):
# ║       • ADDED: Comprehensive tooltips to all inputs.
# ║       • ADDED: PerformanceProfiler and logging.
# ╚════════════════════════════════════════════════════════════════════════════

# =================================================================================
# == Standard Library Imports
# =================================================================================
VERSION = "v1.6.0"  # UPS v1.5.8


import math
import logging
import time

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
# == Helper Classes (Enterprise Standards)
# =================================================================================

class PerformanceProfiler:
    """Standard performance profiler for MD_Nodes."""
    
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.timings = {}
        self.start_times = {}
    
    def start(self, operation_name):
        if not self.enabled: return
        self.start_times[operation_name] = time.perf_counter()
    
    def stop(self, operation_name):
        if not self.enabled: return
        if operation_name in self.start_times:
            elapsed = time.perf_counter() - self.start_times[operation_name]
            if operation_name not in self.timings:
                self.timings[operation_name] = []
            self.timings[operation_name].append(elapsed)
            del self.start_times[operation_name]
    
    def get_total_time(self):
        if not self.enabled or not self.timings: return 0.0
        return sum(sum(times) for times in self.timings.values())
    
    def print_report(self):
        if not self.enabled or not self.timings: return
        logging.info("\n⏱️  PERFORMANCE:")
        total = self.get_total_time()
        logging.info(f"    • Total Time: {total:.4f}s")
        for op_name, times in sorted(self.timings.items()):
            avg = sum(times) / len(times)
            if len(times) == 1:
                logging.info(f"    • {op_name}: {avg:.4f}s")
            else:
                logging.info(f"    • {op_name}: {avg:.4f}s avg ({len(times)}x)")

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
                    "tooltip": (
                        "ASPECT RATIO\n"
                        "• Purpose: Defines the width-to-height relationship.\n"
                        "• Options: 1:1, 16:9, 21:9, 32:9, etc.\n"
                        "• Trade-offs: Extreme ultrawide ratios may cause duplicate subjects if model isn't trained for it.\n"
                        "\n⭐ Recommended: 16:9 for standard outputs, 1:1 for social media."
                    )
                }),
                "orientation": (["Landscape (Horizontal)", "Portrait (Vertical)"], {
                    "default": "Landscape (Horizontal)",
                    "tooltip": (
                        "ORIENTATION\n"
                        "• Purpose: Rotates the selected aspect ratio.\n"
                        "• Options: Landscape (Width > Height) or Portrait (Height > Width).\n"
                        "• Trade-offs: Ensure orientation matches prompt intent (e.g., Portrait for character art).\n"
                        "\n⭐ Recommended: Landscape for environments, Portrait for characters."
                    )
                }),
                "base_mp_preset": (list(CONST_MP_PRESETS.keys()), {
                    "default": "SDXL (1024x1024) - 1.0 MP",
                    "tooltip": (
                        "MEGAPIXEL TARGET\n"
                        "• Purpose: Sets the total resolution/quality class of the generation.\n"
                        "• Options: SD1.5 (0.26 MP), SDXL (1.0 MP), 1080p (2.0 MP), 4K (8.3 MP), or Custom.\n"
                        "• Trade-offs: Higher MP targets exponentially increase VRAM usage and generation time.\n"
                        "\n⭐ Recommended: Match this exactly to your underlying model's training resolution (e.g., SDXL = 1.0 MP)."
                    )
                }),
                "manual_mp_size": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 16.0, "step": 0.1,
                    "tooltip": (
                        "CUSTOM MEGAPIXELS\n"
                        "• Purpose: Sets a specific resolution target when 'base_mp_preset' is set to 'Custom'.\n"
                        "• Range: 0.1 to 16.0 MP.\n"
                        "• Trade-offs: Pushing models beyond their trained MP limit can cause compositional artifacting.\n"
                        "\n⭐ Recommended: 1.0 (Standard SDXL) or 2.0 (FLUX)."
                    )
                }),
                "dimension_scale": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 4.0, "step": 0.05,
                    "tooltip": (
                        "DIMENSION SCALE\n"
                        "• Purpose: Linear multiplier for the final calculated dimensions.\n"
                        "• Options: 0.5 (Half resolution), 1.0 (Default), 2.0 (Double resolution).\n"
                        "• Trade-offs: Quickly scales up/down testing renders without changing the MP target logic.\n"
                        "\n⭐ Recommended: 1.0 for final outputs, 0.5 for rapid testing."
                    )
                }),
                "batch_size": ("INT", {
                    "default": 1, "min": 1, "max": 64,
                    "tooltip": (
                        "BATCH SIZE\n"
                        "• Purpose: Number of latent samples to generate in a single batch.\n"
                        "• Range: 1 to 64.\n"
                        "• Trade-offs: VRAM usage scales linearly with batch size.\n"
                        "\n⭐ Recommended: 1 to 4 depending on GPU VRAM."
                    )
                }),
                "debug_mode": (["0 - Silent", "1 - Info"], {
                    "default": "0 - Silent",
                    "tooltip": (
                        "LOGGING VERBOSITY\n"
                        "• Purpose: Controls console output and enables analytical reporting.\n"
                        "• Options: 0 (Production), 1 (Shows calculated WxH and timing).\n"
                        "\n⭐ Recommended: 0 - Silent."
                    )
                }),
                "enable_profiling": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "ENABLE PROFILING\n"
                        "• Purpose: Measure timing of mathematical operations and tensor allocations.\n"
                        "• Note: Automatically enabled if debug_mode is 1 - Info.\n"
                        "\n⭐ Recommended: False."
                    )
                }),
            }
        }

    RETURN_TYPES = ("LATENT", "INT", "INT")
    RETURN_NAMES = ("latent", "width", "height")
    FUNCTION = "generate"
    CATEGORY = "MD_Nodes/Utility"

    def generate(self, ratio_preset, orientation, base_mp_preset, manual_mp_size, dimension_scale, batch_size, **kwargs):
        """
        Calculates dimensions and returns an empty latent tensor.
        """
        debug_mode = kwargs.get("debug_mode", "0 - Silent")
        debug_level = int(debug_mode.split(" ")[0])
        enable_profiling = kwargs.get("enable_profiling", False)
        
        profiler = PerformanceProfiler(enabled=(debug_level >= 1 or enable_profiling))
        profiler.start("total")

        try:
            profiler.start("calculation")
            
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
            
            profiler.stop("calculation")

            # 7. Generate Latent
            profiler.start("tensor_alloc")
            # Shape: [batch, 4, height // 8, width // 8]
            latent_tensor = torch.zeros([batch_size, 4, height // 8, width // 8])
            profiler.stop("tensor_alloc")

            profiler.stop("total")

            if debug_level >= 1:
                logging.info("\n" + "=" * 60)
                logging.info("📊 [MD_EmptyLatent] ANALYTICS REPORT")
                logging.info("=" * 60)
                logging.info(f"📏 Output: {width}x{height} (Ratio: {ratio_preset})")
                logging.info(f"🎯 Target MP: {target_mp:.2f}")
                profiler.print_report()
                logging.info("=" * 60)

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
    logging.info("🧪 Running Self-Tests for MD_EmptyLatentRatioSelector...")
    
    test_passed = 0
    test_failed = 0
    
    node = MD_EmptyLatentRatioSelector()
    
    try:
        # Test 1: Standard 16:9 Landscape Calculation
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
        logging.info("✅ 16:9 Landscape Calc: PASSED")
        test_passed += 1
    except AssertionError as e:
        logging.error(f"❌ 16:9 Landscape Calc: FAILED - {e}")
        test_failed += 1
        
    try:
        # Test 2: Portrait Flip
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
        logging.info("✅ Portrait Flip: PASSED")
        test_passed += 1
    except AssertionError as e:
        logging.error(f"❌ Portrait Flip: FAILED - {e}")
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
        logging.info("✅ Dimension Scaling: PASSED")
        test_passed += 1
    except AssertionError as e:
        logging.error(f"❌ Dimension Scaling: FAILED - {e}")
        test_failed += 1

    logging.info(f"\n{'='*60}")
    logging.info(f"Test Results: {test_passed} passed, {test_failed} failed")
    logging.info(f"{'='*60}")