# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# █▓▒░                                                                     ░▒▓█
# █▓▒░ ▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂... ░▒▓█
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
# ║ ░▒▓ ORIGIN: Community Adaptation - Caching Implementation
# ║ ░▒▓ DESCRIPTION:
# ║    Enterprise-grade save and load nodes for conditioning data. Allows users
# ║    to "bake" heavy text encoder outputs (like unchanging negative prompts)
# ║    to disk and reload them instantly, bypassing the encoder entirely.
# ║    NOTE: As an I/O utility, this node runs entirely in the public wrapper domain.
# ║
# ║ ░▒▓ CORE FEATURES:
# ║    ✔ Fast IS_CHANGED: Uses file modification time instead of SHA256 hashing.
# ║    ✔ Safe Tensor Casting: Native CPU mapping for memory efficiency.
# ║    ✔ Auto-Registration: Seamlessly integrates with ComfyUI's folder paths.
# ║    ✔ Production Ready: Input validation, error handling, structured logging.
# ║
# ║ ░▒▓ USE CASES:
# ║    → Bypassing Text Encoders: Save a static negative prompt and load it instantly.
# ║    → Workflow Optimization: Drastically reduce startup time in Ace-Step workflows.
# ║    → Cross-Workflow Sharing: Move complex conditioning payloads between graphs.
# ║
# ║ ░▒▓ TECHNICAL SPECS:
# ║    - Compatible: ComfyUI Core, PyTorch 2.0+
# ║    - Dependencies: None
# ║    - Performance: Instant file tracking, memory-safe tensor loading.
# ║
# ║ ░▒▓ CHANGELOG:
# ║    v1.0.1 (2026-02-24) - Enterprise Standards Update
# ║    ├── VERIFIED: Tooltips meet v1.5.4 standard.
# ║    ├── VERIFIED: PerformanceProfiler integration.
# ╚════════════════════════════════════════════════════════════════════════════

# =================================================================================
# == Standard Library Imports
# =================================================================================
VERSION = "v3.0.0"  # UPS v1.5.8


import os
import logging
import hashlib
import time

# =================================================================================
# == Third-Party Imports
# =================================================================================
import torch

# =================================================================================
# == ComfyUI Core Modules
# =================================================================================
import folder_paths
import comfy.model_management

# =================================================================================
# == Configuration Constants (No Magic Numbers!)
# =================================================================================
CONST_DIR_NAME = "conditionings"
CONST_EXT = ".bin"
CONST_JS_MAX_SAFE_INTEGER = 9007199254740991
CONST_SEED_MIN = 0

# =================================================================================
# == Directory Registration & Setup
# =================================================================================
# Ensure the conditionings directory exists in the models folder
_target_dir = os.path.join(folder_paths.models_dir, CONST_DIR_NAME)
os.makedirs(_target_dir, exist_ok=True)

if CONST_DIR_NAME not in folder_paths.folder_names_and_paths:
    folder_paths.folder_names_and_paths[CONST_DIR_NAME] = ([_target_dir], set([CONST_EXT]))
else:
    current_paths, current_exts = folder_paths.folder_names_and_paths[CONST_DIR_NAME]
    if _target_dir not in current_paths:
        current_paths.append(_target_dir)
    folder_paths.folder_names_and_paths[CONST_DIR_NAME] = (current_paths, set([CONST_EXT]))

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
        logging.info("\n⏱️  PERFORMANCE (I/O):")
        total = self.get_total_time()
        logging.info(f"    • Total Time: {total:.2f}s")
        for op_name, times in sorted(self.timings.items()):
            avg = sum(times) / len(times)
            if len(times) == 1:
                logging.info(f"    • {op_name}: {avg:.3f}s")
            else:
                logging.info(f"    • {op_name}: {avg:.3f}s avg ({len(times)}x)")

# =================================================================================
# == Core Node Classes
# =================================================================================

class MD_SaveConditioning:
    """Saves conditioning data directly to the models/conditionings directory."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING", {
                    "tooltip": (
                        "CONDITIONING DATA\n"
                        "• Purpose: The encoded text/audio data to cache.\n"
                        "• Options: Any standard CONDITIONING stream.\n"
                        "\n⭐ Recommended: Connect your static negative prompt here."
                    )
                }),
                "filename": ("STRING", {
                    "default": "cached_conditioning",
                    "tooltip": (
                        "FILENAME PREFIX\n"
                        "• Purpose: The name of the saved .bin file.\n"
                        "• Options: Any valid text string (without extension).\n"
                        "\n⭐ Recommended: Use recognizable names like 'acestep_neg'."
                    )
                }),
                "force_update": ("INT", {
                    "default": 0,
                    "min": CONST_SEED_MIN,
                    "max": CONST_JS_MAX_SAFE_INTEGER,
                    "tooltip": (
                        "FORCE UPDATE TRIGGER\n"
                        "• Purpose: Change this value to force ComfyUI to re-save the file.\n"
                        "• Range: 0 to 9,007,199,254,740,991 (JS-safe range).\n"
                        "\n⭐ Recommended: Set widget to 'randomize' if you always want it to save."
                    )
                }),
                "debug_mode": (["0 - Silent", "1 - Info", "2 - Verbose"], {
                    "default": "1 - Info",
                    "tooltip": (
                        "LOGGING VERBOSITY\n"
                        "• Purpose: Controls console output detail level.\n"
                        "• Options: 0 (Silent), 1 (Analytics Report), 2 (Full trace).\n"
                        "\n⭐ Recommended: 1 - Info for confirmation."
                    )
                })
            }
        }

    RETURN_TYPES = ("CONDITIONING", "STRING")
    RETURN_NAMES = ("conditioning", "filepath")
    FUNCTION = "execute"
    OUTPUT_NODE = True
    CATEGORY = "MD_Nodes/Utility"

    def execute(self, conditioning, filename, force_update, debug_mode):
        debug_level = int(debug_mode.split(" ")[0])
        profiler = PerformanceProfiler(enabled=(debug_level >= 1))
        profiler.start("total_execution")
        
        try:
            save_dir = folder_paths.folder_names_and_paths[CONST_DIR_NAME][0][0]
            safe_filename = "".join(c for c in filename if c.isalnum() or c in ('_', '-'))
            if not safe_filename:
                safe_filename = "conditioning_cache"
                
            full_path = os.path.join(save_dir, f"{safe_filename}{CONST_EXT}")
            
            profiler.start("disk_write")
            torch.save(conditioning, full_path)
            profiler.stop("disk_write")

            if debug_level >= 2:
                for idx, cond_item in enumerate(conditioning):
                    logging.info(f"  → Batch {idx}: Tensor Shape {cond_item[0].shape}")
            
        except Exception as e:
            logging.error(f"[MD_SaveConditioning] Error saving file: {e}")
            full_path = "ERROR"

        profiler.stop("total_execution")
        
        if debug_level >= 1:
            logging.info("\n" + "=" * 60)
            logging.info("📊 [MD_SaveConditioning] ANALYTICS REPORT")
            logging.info("=" * 60)
            logging.info("💾  CACHE:")
            logging.info(f"    • Saved to: {full_path}")
            profiler.print_report()
            logging.info("=" * 60)

        return (conditioning, full_path)


class MD_LoadConditioning:
    """Loads conditioning data from the models/conditionings directory."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning_file": (folder_paths.get_filename_list(CONST_DIR_NAME), {
                    "tooltip": (
                        "CONDITIONING FILE\n"
                        "• Purpose: The cached .bin file to load.\n"
                        "• Requirement: Any file saved by the Save Conditioning node.\n"
                        "\n⭐ Recommended: Ensure the model architecture matches the cache."
                    )
                }),
                "debug_mode": (["0 - Silent", "1 - Info"], {
                    "default": "0 - Silent",
                    "tooltip": (
                        "LOGGING VERBOSITY\n"
                        "• Purpose: Controls console output detail level.\n"
                        "• Options: 0 (Silent), 1 (Analytics Report).\n"
                        "\n⭐ Recommended: 0 - Silent."
                    )
                })
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "execute"
    CATEGORY = "MD_Nodes/Utility"

    @classmethod
    def IS_CHANGED(cls, conditioning_file, **kwargs):
        """Optimized cache check using file modification time instead of hashing."""
        try:
            file_path = folder_paths.get_full_path(CONST_DIR_NAME, conditioning_file)
            if file_path and os.path.exists(file_path):
                mtime = os.path.getmtime(file_path)
                return str(mtime)
        except Exception:
            pass
        return float("NaN")

    @classmethod
    def VALIDATE_INPUTS(cls, conditioning_file, **kwargs):
        file_path = folder_paths.get_full_path(CONST_DIR_NAME, conditioning_file)
        if not file_path or not os.path.exists(file_path):
            return f"Invalid conditioning file: {conditioning_file}"
        return True

    def execute(self, conditioning_file, debug_mode):
        debug_level = int(debug_mode.split(" ")[0])
        profiler = PerformanceProfiler(enabled=(debug_level >= 1))
        profiler.start("total_execution")
        
        conditioning_list = None
        file_path = folder_paths.get_full_path(CONST_DIR_NAME, conditioning_file)
        
        try:
            profiler.start("disk_read")
            # Load mapping strictly to CPU to prevent VRAM spikes during load
            conditioning_list = torch.load(file_path, map_location="cpu")
            profiler.stop("disk_read")
            
            profiler.start("tensor_processing")
            # Ensure complex nested dictionary structures are safely cast
            for idx in range(len(conditioning_list)):
                if isinstance(conditioning_list[idx][0], torch.Tensor):
                    conditioning_list[idx][0] = conditioning_list[idx][0].cpu()
                
                if len(conditioning_list[idx]) > 1 and isinstance(conditioning_list[idx][1], dict):
                    for key, value in conditioning_list[idx][1].items():
                        if isinstance(value, torch.Tensor):
                            conditioning_list[idx][1][key] = value.cpu()
                            
                if hasattr(conditioning_list[idx][0], "addit_embeds") and isinstance(conditioning_list[idx][0].addit_embeds, dict):
                    for key, value in conditioning_list[idx][0].addit_embeds.items():
                        if isinstance(value, torch.Tensor):
                            conditioning_list[idx][0].addit_embeds[key] = value.cpu()
            profiler.stop("tensor_processing")
            
        except Exception as e:
            logging.error(f"[MD_LoadConditioning] Failed to load cache: {e}")
            raise e

        profiler.stop("total_execution")
        
        if debug_level >= 1:
            logging.info("\n" + "=" * 60)
            logging.info("📊 [MD_LoadConditioning] ANALYTICS REPORT")
            logging.info("=" * 60)
            logging.info("📂  CACHE:")
            logging.info(f"    • Loaded: {conditioning_file}")
            profiler.print_report()
            logging.info("=" * 60)

        return (conditioning_list,)

# =================================================================================
# == Node Registration
# =================================================================================

NODE_CLASS_MAPPINGS = {
    "MD_SaveConditioning": MD_SaveConditioning,
    "MD_LoadConditioning": MD_LoadConditioning,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MD_SaveConditioning": "MD: Save Conditioning 💾",
    "MD_LoadConditioning": "MD: Load Conditioning 📂",
}


# ==============================================================================
# == Unit Tests (smoke — runs without ComfyUI)
# ==============================================================================

if __name__ == "__main__":
    print("\n🧪 Smoke tests: MD_ConditioningCacheNodes")
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
    _check("CONST CONST_DIR_NAME defined", CONST_DIR_NAME is not None)
    _check("CONST CONST_EXT defined", CONST_EXT is not None)
    _check("CONST CONST_JS_MAX_SAFE_INTEGER defined", CONST_JS_MAX_SAFE_INTEGER is not None)
    _check("CONST CONST_SEED_MIN defined", CONST_SEED_MIN is not None)
    _check("NODE_CLASS_MAPPINGS defined",
           isinstance(NODE_CLASS_MAPPINGS, dict) and len(NODE_CLASS_MAPPINGS) > 0)
    _check("  class MD_SaveConditioning in map", "MD_SaveConditioning" in NODE_CLASS_MAPPINGS)
    _check("  class MD_LoadConditioning in map", "MD_LoadConditioning" in NODE_CLASS_MAPPINGS)

    print(f"\n  {_pass} passed, {_fail} failed")
    if _fail == 0:
        print("  🎉 All good.")
