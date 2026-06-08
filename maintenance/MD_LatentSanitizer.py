# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# █▓▒░                                                                     ░▒▓█
# █▓▒░     MD_Nodes/MD_LatentSanitizer – Universal Latent Fixer v1.3.1     ░▒▓█
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
# ║ ░▒▓ ORIGIN: Audio/Video/Image VAE Instability Research
# ║ ░▒▓ DESCRIPTION:
# ║    The "Universal Fixer" for generative media artifacts.
# ║    Works on AUDIO (crackle/drift), VIDEO (flickering/NaNs), and IMAGES (grid lines).
# ║    It sanitizes latents (clamping/NaNs) AND patches the VAE to prevent artifacts.
# ║    NOTE: As a system-level dictionary and tensor scrubber, this runs entirely 
# ║    in the public wrapper domain.
# ║
# ║ ░▒▓ ARCHITECTURE:
# ║    ┌── Latent Scrub:  Clamps outliers and fixes NaNs (Volume Explosion/Flicker Fix)
# ║    ├── VAE Patcher:   Forces internal sample rate config (Pitch/Drift Fix)
# ║    ├── Tiling Guard:  Disables VAE slicing (Boundary Crackle/Grid Line Fix)
# ║    └── Logger:        Outputs a string summary for UI display
# ║
# ║ ░▒▓ CORE FEATURES:
# ║    ✔ Universal Clamping: Fixes exploding tensors in 4D (Batch, Channel, H, W).
# ║    ✔ Anti-Crackle/Anti-Grid: Disables VAE tiling to prevent boundary artifacts.
# ║    ✔ Drift Correction: Patches VAE sample rates (Audio only).
# ║    ✔ Log Output: Outputs a string report for UI display.
# ║    ✔ Enterprise Logging: Detailed reports with PerformanceProfiler.
# ║
# ║ ░▒▓ USE CASES:
# ║    → AUDIO: Removing "white noise", "crackle", or pitch drift (44.1k vs 48k).
# ║    → VIDEO: Fixing "flickering" frames caused by exploding latents (AnimateDiff).
# ║    → IMAGE: Removing "grid lines" in high-res upscales (Tiling fix).
# ║
# ║ ░▒▓ TECHNICAL SPECS:
# ║    - Compatible: ComfyUI 0.2.0+, PyTorch 2.0+
# ║    - Dependencies: torch
# ║    - Performance: negligible overhead (<0.01s typically)
# ║    - Testing: Embedded unit tests included
# ║
# ║ ░▒▓ CHANGELOG:
# ║    v1.3.1 (2026-02-24) - Enterprise Standards Update
# ║    ├── VERIFIED: Tooltips meet v1.5.4 standard.
# ║    ├── VERIFIED: PerformanceProfiler integration.
# ║    v1.3.0 (2026-02-14) - Universal Update
# ║    ├── ADDED: Explicit support for Image/Video dimensions (4D tensors).
# ║    ├── ADDED: 'log_info' STRING output for UI feedback.
# ║    ├── REBRANDED: "Universal Latent Sanitizer" (formerly Audio-only).
# ║    └── FIXED: Sample Rate now reads from VAE if "Unchanged" is selected.
# ╚════════════════════════════════════════════════════════════════════════════

# =================================================================================
# == Standard Library Imports
# =================================================================================
VERSION = "v1.3.1"  # UPS v1.5.8


import logging
import time
import sys

# =================================================================================
# == Third-Party Imports
# =================================================================================
import torch

# =================================================================================
# == Configuration Constants
# =================================================================================
CONST_LOG_CATEGORY = "MD_Nodes.Maintenance.LatentSanitizer"
CONST_DEFAULT_CLAMP = 4.5
CONST_SAFE_TILING_SIZE = 1000000  # Large enough to force single-pass

# =================================================================================
# == PerformanceProfiler Class (MD_Nodes Standard)
# =================================================================================

class PerformanceProfiler:
    """Standard performance profiler for MD_Nodes."""
    
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.timings = {}
        self.start_times = {}
    
    def start(self, operation_name):
        if not self.enabled:
            return
        self.start_times[operation_name] = time.perf_counter()
    
    def stop(self, operation_name):
        if not self.enabled:
            return
        if operation_name in self.start_times:
            elapsed = time.perf_counter() - self.start_times[operation_name]
            if operation_name not in self.timings:
                self.timings[operation_name] = []
            self.timings[operation_name].append(elapsed)
            del self.start_times[operation_name]
    
    def get_total_time(self):
        if not self.enabled or not self.timings:
            return 0.0
        return sum(sum(times) for times in self.timings.values())
    
    def print_report(self):
        if not self.enabled or not self.timings:
            return
        logging.info("\n⏱️   PERFORMANCE:")
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

class MD_LatentSanitizer:
    """
    MD: Universal Latent Sanitizer (Audio/Video/Image)
    
    A comprehensive maintenance node that clamps latent outliers, fixes NaNs,
    and patches VAE settings to prevent decoding artifacts like static and crackle.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT", {
                    "tooltip": (
                        "LATENT INPUT\n"
                        "• Purpose: The latent chunks from your KSampler.\n"
                        "• Requirement: Connect this BEFORE VAE Decode."
                    )
                }),
                "vae": ("VAE", {
                    "tooltip": (
                        "VAE INPUT (REQUIRED)\n"
                        "• Purpose: The VAE used for decoding.\n"
                        "• Requirement: Must be passed through this node to apply rate/tiling fixes."
                    )
                }),
                "fix_nans": ("BOOLEAN", {
                    "default": True, 
                    "label_on": "Enabled", 
                    "label_off": "Disabled",
                    "tooltip": (
                        "FIX NANS / INFINITIES\n"
                        "• Purpose: Replaces broken numerical values with 0.0.\n"
                        "• Effect: Prevents 'Black Hole' audio/video and silent outputs.\n"
                        "• Trade-offs: None (broken values are useless anyway).\n"
                        "\n⭐ Recommended: Enabled"
                    )
                }),
                "clamp_sigma": ("FLOAT", {
                    "default": CONST_DEFAULT_CLAMP, 
                    "min": 1.0, 
                    "max": 20.0, 
                    "step": 0.1,
                    "tooltip": (
                        "HARD CLAMP LIMIT (SIGMA)\n"
                        "• Purpose: Limits the maximum loudness/variance of the latent data.\n"
                        "• Audio Use: Prevents VAE 'blowout' (digital clipping/crackle).\n"
                        "• Video Use: Prevents 'flickering' frames.\n"
                        "• Range: 3.0 (Strict) to 6.0 (Loose).\n"
                        "\n⭐ Recommended: 3.0 to 4.5 if hearing crackle."
                    )
                }),
                "vae_sample_rate": (["Unchanged", "44100", "48000", "32000", "24000", "16000"], {
                    "default": "Unchanged",
                    "tooltip": (
                        "FORCE VAE SAMPLE RATE (Audio Only)\n"
                        "• Purpose: Patches the VAE metadata to expected Hz to fix 'Drift'.\n"
                        "• Effect: Prevents pitch shifting or high-pitched aliasing/whine.\n"
                        "• Note: Ignored for Image/Video VAEs.\n"
                        "\n⭐ Recommended: Unchanged (unless correcting a known issue)."
                    )
                }),
                "disable_tiling": ("BOOLEAN", {
                    "default": True,
                    "label_on": "Force Disable (Anti-Crackle)", 
                    "label_off": "Allow Tiling (Default)",
                    "tooltip": (
                        "DISABLE VAE TILING\n"
                        "• Purpose: Forces the VAE to decode the whole file at once.\n"
                        "• Audio: Eliminates 'seam crackle' every few seconds.\n"
                        "• Image: Eliminates 'grid lines' in high-res upscales.\n"
                        "• Trade-offs: Uses significantly more VRAM when enabled.\n"
                        "\n⭐ Recommended: Enabled (fixes boundary artifacts)."
                    )
                }),
                "debug_mode": (["0 - Silent", "1 - Info", "2 - Verbose"], {
                    "default": "1 - Info",
                    "tooltip": (
                        "LOGGING VERBOSITY\n"
                        "• Purpose: Controls console output detail level.\n"
                        "• Options: 0 (Production), 1 (Analytics Report), 2 (Full trace).\n"
                        "\n⭐ Recommended: 1 - Info."
                    )
                }),
                "enable_profiling": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "PERFORMANCE PROFILING\n"
                        "• Purpose: Enable detailed operation timing.\n"
                        "• Note: Automatically enabled when debug_mode >= 1."
                    )
                }),
            }
        }

    RETURN_TYPES = ("LATENT", "VAE", "INT", "STRING")
    RETURN_NAMES = ("sanitized_latents", "patched_vae", "sample_rate_int", "log_info")
    FUNCTION = "sanitize"
    CATEGORY = "MD_Nodes/Maintenance"

    def sanitize(self, samples, vae, fix_nans, clamp_sigma, vae_sample_rate, disable_tiling, debug_mode="1 - Info", enable_profiling=False):
        """
        Executes the sanitization process.
        """
        # 1. Setup Logging & Profiling
        logger = logging.getLogger(CONST_LOG_CATEGORY)
        
        debug_level = 0
        if isinstance(debug_mode, str):
            debug_level = int(debug_mode.split(" ")[0])
            
        profiler = PerformanceProfiler(enabled=(debug_level >= 1 or enable_profiling))
        profiler.start("total_execution")
        
        if debug_level >= 2: logger.setLevel(logging.DEBUG)
        elif debug_level >= 1: logger.setLevel(logging.INFO)
        else: logger.setLevel(logging.WARNING)

        try:
            # 2. Latent Processing (Clone & Scrub)
            profiler.start("latent_processing")
            
            new_samples = samples.copy()
            latent_tensor = new_samples["samples"].clone()
            
            log_messages = []
            
            orig_min = float(latent_tensor.min())
            orig_max = float(latent_tensor.max())
            orig_mean = float(latent_tensor.mean())
            
            # A. NaN/Inf Fix
            nan_fixed = False
            if fix_nans:
                if torch.isnan(latent_tensor).any() or torch.isinf(latent_tensor).any():
                    latent_tensor = torch.nan_to_num(latent_tensor, nan=0.0, posinf=clamp_sigma, neginf=-clamp_sigma)
                    nan_fixed = True
                    log_messages.append("⚠️ Fixed NaNs/Infs")

            # B. Hard Clamping (Universal)
            clamped_count = 0
            current_abs_max = torch.max(torch.abs(latent_tensor))
            
            if current_abs_max > clamp_sigma:
                outliers = torch.sum(torch.abs(latent_tensor) > clamp_sigma).item()
                clamped_count = outliers
                
                latent_tensor = torch.clamp(latent_tensor, min=-clamp_sigma, max=clamp_sigma)
                log_messages.append(f"✂️ Clamped {outliers} outliers > {clamp_sigma}")
            
            new_samples["samples"] = latent_tensor
            profiler.stop("latent_processing")

            # 3. VAE Patching
            profiler.start("vae_patching")
            patched_vae = vae
            
            # C. Tiling Control
            tiling_status = "Default"
            if disable_tiling:
                if hasattr(patched_vae, "tile_sample_min_size"):
                    patched_vae.tile_sample_min_size = CONST_SAFE_TILING_SIZE
                
                if not hasattr(patched_vae, "model_options"):
                    patched_vae.model_options = {}
                patched_vae.model_options["tiling"] = False
                
                tiling_status = "Disabled (Anti-Crackle)"
            else:
                tiling_status = "Enabled (Standard)"

            # D. Sample Rate Patch
            out_rate = 44100 
            rate_status = "Unchanged"
            
            if hasattr(patched_vae, "first_stage_model") and hasattr(patched_vae.first_stage_model, "sample_rate"):
                 out_rate = patched_vae.first_stage_model.sample_rate
            elif hasattr(patched_vae, "config") and isinstance(patched_vae.config, dict):
                 out_rate = patched_vae.config.get("sample_rate", 44100)
            
            if vae_sample_rate != "Unchanged":
                target_rate = int(vae_sample_rate)
                out_rate = target_rate
                
                patch_count = 0
                
                if hasattr(patched_vae, "config") and isinstance(patched_vae.config, dict):
                    patched_vae.config["sample_rate"] = target_rate
                    patch_count += 1
                
                if hasattr(patched_vae, "first_stage_model"):
                    if hasattr(patched_vae.first_stage_model, "sample_rate"):
                        patched_vae.first_stage_model.sample_rate = target_rate
                        patch_count += 1
                
                rate_status = f"Patched to {target_rate}Hz"
            
            profiler.stop("vae_patching")
            profiler.stop("total_execution")
            
            # 4. Construct Log String
            log_info = f"--- MD Sanitizer Report ---\n"
            log_info += f"Stats: Range [{orig_min:.2f}, {orig_max:.2f}], Mean {orig_mean:.4f}\n"
            if nan_fixed: log_info += f"⚠️ FIXED NaNs/Infs found in tensor.\n"
            if clamped_count > 0: log_info += f"✂️ CLAMPED {clamped_count} outliers (Limit: {clamp_sigma})\n"
            else: log_info += f"✅ No clamping needed.\n"
            log_info += f"VAE: Tiling={tiling_status}\n"
            log_info += f"Rate: {rate_status} (Output: {out_rate}Hz)"

            # 5. Analytics Report
            if debug_level >= 1:
                logging.info("\n" + "=" * 60)
                logging.info(f"🛡️ [MD_LatentSanitizer] ANALYTICS REPORT")
                logging.info("=" * 60)
                logging.info("📊  LATENT STATS:")
                logging.info(f"    • Range:        {orig_min:.2f} to {orig_max:.2f}")
                logging.info(f"    • Mean:         {orig_mean:.4f}")
                logging.info(f"    • NaNs Fixed:   {nan_fixed}")
                logging.info(f"    • Clamped:      {clamped_count > 0} (Limit: {clamp_sigma})")
                
                logging.info("🔧  VAE PATCHING:")
                logging.info(f"    • Tiling:       {tiling_status}")
                logging.info(f"    • Sample Rate:  {rate_status}")
                
                profiler.print_report()
                
                if log_messages:
                    logging.info(f"\n📝  NOTES:")
                    for msg in log_messages:
                        logging.info(f"    • {msg}")
                logging.info("=" * 60 + "\n")

            return (new_samples, patched_vae, out_rate, log_info)

        except Exception as e:
            logger.error(f"Failed to sanitize latents: {e}")
            if debug_level >= 1:
                import traceback
                traceback.print_exc()
            return (samples, vae, 44100, f"Error: {str(e)}")

# =================================================================================
# == Node Registration
# =================================================================================

NODE_CLASS_MAPPINGS = {
    "MD_LatentSanitizer": MD_LatentSanitizer
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MD_LatentSanitizer": "MD: Universal Latent Sanitizer (Audio/Video/Image)"
}

# =================================================================================
# == Embedded Unit Tests
# =================================================================================

if __name__ == "__main__":
    logging.info("🧪 Running Self-Tests for MD_LatentSanitizer...")
    
    test_passed = 0
    test_failed = 0
    
    # Mock Objects
    class MockVAE:
        def __init__(self):
            self.model_options = {}
            self.config = {"sample_rate": 44100}

    try:
        prof = PerformanceProfiler()
        prof.start("test_op")
        time.sleep(0.01)
        prof.stop("test_op")
        assert prof.get_total_time() > 0
        logging.info("✅ Profiler Logic: PASSED")
        test_passed += 1
    except Exception as e:
        logging.error(f"❌ Profiler Logic: FAILED - {e}")
        test_failed += 1

    try:
        mock_tensor = torch.tensor([10.0, -10.0, 0.0, float('nan'), float('inf')])
        mock_samples = {"samples": mock_tensor}
        
        node = MD_LatentSanitizer()
        mock_vae = MockVAE()
        
        result, _, _, log_str = node.sanitize(
            mock_samples, mock_vae, fix_nans=True, clamp_sigma=5.0, 
            vae_sample_rate="Unchanged", disable_tiling=True, debug_mode="0 - Silent"
        )
        
        res_tensor = result["samples"]
        
        assert not torch.isnan(res_tensor).any(), "NaNs should be gone"
        assert not torch.isinf(res_tensor).any(), "Infs should be gone"
        assert res_tensor.max() <= 5.0, "Max value should be clamped"
        assert res_tensor.min() >= -5.0, "Min value should be clamped"
        assert "FIXED NaNs" in log_str, "Log string missing NaN info"
        
        logging.info("✅ Latent Clamping & NaN Fix: PASSED")
        test_passed += 1
    except Exception as e:
        logging.error(f"❌ Latent Clamping: FAILED - {e}")
        test_failed += 1

    try:
        node = MD_LatentSanitizer()
        mock_vae = MockVAE()
        
        _, patched_vae, rate, _ = node.sanitize(
            {"samples": torch.zeros(1)}, mock_vae, fix_nans=True, clamp_sigma=5.0, 
            vae_sample_rate="48000", disable_tiling=True, debug_mode="0 - Silent"
        )
        
        assert patched_vae.config["sample_rate"] == 48000, "Sample rate patch failed"
        assert rate == 48000, "Returned int rate failed"
        assert patched_vae.model_options["tiling"] == False, "Tiling disable failed"
        
        logging.info("✅ VAE Patching: PASSED")
        test_passed += 1
    except Exception as e:
        logging.error(f"❌ VAE Patching: FAILED - {e}")
        test_failed += 1

    logging.info(f"\n{'='*60}")
    logging.info(f"Test Results: {test_passed} passed, {test_failed} failed")
    logging.info(f"{'='*60}")
    
    if test_failed == 0:
        logging.info("🎉 All Enterprise Standards Met")