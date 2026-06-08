# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# █▓▒░                                                                     ░▒▓█
# █▓▒░            MD_Nodes/Utilities – MD_YAML_Generator v1.3.2            ░▒▓█
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
# ║ ░▒▓ DESCRIPTION:
# ║   A central "Command Center" for managing, merging, and saving 
# ║   complex YAML configurations. It bridges the gap between static
# ║   presets and dynamic runtime adjustments.
# ║   NOTE: As a file I/O utility, this runs entirely in the public wrapper domain.
# ║
# ║ ░▒▓ FEATURES:
# ║   ✔ Central Command: Manage huge sampler configs via Presets.
# ║   ✔ Subfolder Support: Recursive scanning (e.g., 'sampling/legacy.yaml').
# ║   ✔ Smart Header Reader: Extracts info for Console AND Output String.
# ║   ✔ Diff Logging: Console shows exactly what values changed (Old -> New).
# ║   ✔ Auto-Save: Instantly save successful experiments as new presets.
# ║
# ║ ░▒▓ CHANGELOG:
# ║   - v1.3.2 (Enterprise Standards - Feb 2026):
# ║       • VERIFIED: Tooltips meet v1.5.4 standard.
# ║   - v1.3.1 (UX Update):
# ║       • ADDED: 'preset_info_string' output.
# ║   - v1.3.0 (Feature Update):
# ║       • ADDED: Header Extraction logic.
# ║
# ║ ░▒▓ WARNING:
# ║   ▓▒░ Editing YAML manually requires indentation discipline.
# ║   ▓▒░ 2 spaces, not tabs. Or else.
# ╚════════════════════════════════════════════════════════════════════════════

# =================================================================================
# == Standard Library Imports
# =================================================================================
VERSION = "v1.3.2"  # UPS v1.5.8


import os
import logging
import time
from typing import Dict, Any, List

# =================================================================================
# == Third-Party Imports
# =================================================================================
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    logging.error("[MD_YAML_Generator] PyYAML not found. Node will fail.")

# =================================================================================
# == Configuration Constants
# =================================================================================
CONST_PRESET_DIR_NAME = "yaml_presets"
CONST_HEADER_MARKER = "# ░▒▓" 
CONST_HEADER_STOP = "# ▄▄▄"

# =================================================================================
# == Setup & Logging
# =================================================================================
logger = logging.getLogger("ComfyUI_MD_Nodes.YAML_Manager")

NODE_DIR = os.path.dirname(os.path.realpath(__file__))
PRESET_DIR = os.path.join(NODE_DIR, CONST_PRESET_DIR_NAME)

if not os.path.exists(PRESET_DIR):
    try:
        os.makedirs(PRESET_DIR)
        logger.info(f"✅ [MD_YAML] Created preset directory at: {PRESET_DIR}")
    except OSError as e:
        logger.error(f"❌ [MD_YAML] Could not create directory {PRESET_DIR}: {e}")

# =================================================================================
# == Core Node Class
# =================================================================================
class MD_YAML_Generator:
    """
    A power-user tool to load base YAML configurations, apply manual overrides,
    validate the syntax, and optionally save the result as a new preset.
    """
    
    def __init__(self):
        self.preset_dir = PRESET_DIR

    @classmethod
    def INPUT_TYPES(cls):
        # Scan for existing YAML files RECURSIVELY
        file_list = ["None"]
        if os.path.exists(PRESET_DIR):
            try:
                for root, dirs, files in os.walk(PRESET_DIR):
                    for file in files:
                        if file.lower().endswith(('.yaml', '.yml')):
                            abs_path = os.path.join(root, file)
                            rel_path = os.path.relpath(abs_path, PRESET_DIR)
                            rel_path = rel_path.replace("\\", "/") 
                            file_list.append(rel_path)
                file_list.sort()
            except Exception as e:
                logger.warning(f"⚠️ [MD_YAML] Error scanning presets: {e}")

        return {
            "required": {
                "base_preset": (file_list, {
                    "default": "None",
                    "tooltip": (
                        "BASE PRESET\n"
                        "• Purpose: The foundational YAML file to load.\n"
                        "• Requirement: Files must be stored in the 'yaml_presets' folder."
                    )
                }),
                "merge_strategy": (["Override (Smart Merge)", "Append Only", "Pure Manual"], {
                    "default": "Override (Smart Merge)",
                    "tooltip": (
                        "MERGE STRATEGY\n"
                        "• Purpose: How manual edits interact with the base preset.\n"
                        "• Override: Replaces base values with your manual edits.\n"
                        "• Append Only: Only adds new keys, ignores existing ones.\n"
                        "• Pure Manual: Ignores the base preset entirely.\n"
                        "\n⭐ Recommended: Override (Smart Merge)."
                    )
                }),
                "force_update": ("BOOLEAN", {
                    "default": False, 
                    "label_on": "Always Re-Run", 
                    "label_off": "Only on Change",
                    "tooltip": (
                        "FORCE UPDATE\n"
                        "• Purpose: Bypasses caching to re-read the file every generation.\n"
                        "\n⭐ Recommended: Only on Change (Faster)."
                    )
                }),
            },
            "optional": {
                "manual_edits": ("STRING", {
                    "multiline": True, 
                    "default": "# Type overrides here (e.g. cfg_scale: 5.0)\n# These will replace values in the Base Preset.",
                    "placeholder": "restart_mode: aggressive\ncfg_scale: 8.0",
                    "tooltip": (
                        "MANUAL EDITS\n"
                        "• Purpose: YAML string to merge with the base preset.\n"
                        "• Warning: Must use exact YAML syntax (2 spaces, no tabs)."
                    )
                }),
                "save_new_preset_as": ("STRING", {
                    "default": "", 
                    "placeholder": "subfolder/my_config.yaml",
                    "tooltip": (
                        "SAVE AS NEW PRESET\n"
                        "• Purpose: Saves the final merged result to disk.\n"
                        "• Requirement: Leave blank unless you want to create a new file."
                    )
                }),
                "debug_log": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "DEBUG LOG\n"
                        "• Purpose: Prints exactly which values were changed during the merge."
                    )
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("yaml_config_str", "preset_info_string")
    FUNCTION = "generate_yaml"
    CATEGORY = "MD_Nodes/Utility"
    
    # ──────────────────────────────────────────────────────────────────────────────
    # EXECUTION CONTROL LOGIC
    # ──────────────────────────────────────────────────────────────────────────────
    @classmethod
    def IS_CHANGED(cls, base_preset, merge_strategy, manual_edits, save_new_preset_as, force_update, **kwargs):
        if force_update:
            return float("nan")  
        if save_new_preset_as.strip():
            return float("nan")  
        return hash((base_preset, merge_strategy, manual_edits))

    def generate_yaml(self, base_preset, merge_strategy, force_update, manual_edits="", save_new_preset_as="", debug_log=False):
        if not YAML_AVAILABLE:
            return ("", "PyYAML Missing")

        final_dict = {}
        info_str = "No preset selected."
        
        # 1. LOAD BASE PRESET
        if base_preset != "None" and merge_strategy != "Pure Manual":
            path = os.path.join(self.preset_dir, base_preset)
            if debug_log: logger.info(f"📂 [MD_YAML] Loading Base Preset: {base_preset}")
            
            info_str = self._extract_preset_info(path)
            if debug_log: 
                print(f"\n📘 [MD_YAML] Loaded: {base_preset}\n{info_str}")

            try:
                with open(path, 'r', encoding='utf-8') as f:
                    loaded = yaml.safe_load(f)
                    if isinstance(loaded, dict): 
                        final_dict = loaded
                    else: 
                        logger.warning(f"⚠️ [MD_YAML] Preset is empty.")
            except Exception as e:
                logger.error(f"❌ [MD_YAML] Load Error: {e}")
                info_str = f"Error loading preset: {e}"

        # 2. PARSE MANUAL EDITS
        manual_dict = {}
        if manual_edits and manual_edits.strip():
            try:
                manual_dict = yaml.safe_load(manual_edits)
                if not isinstance(manual_dict, dict): 
                    logger.warning("⚠️ [MD_YAML] Manual edits must be a dictionary.")
                    manual_dict = {}
            except yaml.YAMLError as e:
                raise ValueError(f"❌ [MD_YAML] Syntax Error in Manual Edits:\n{e}")

        # 3. MERGE & LOG DIFFS
        if merge_strategy == "Pure Manual":
            final_dict = manual_dict
            info_str = "Mode: Pure Manual (No Preset)"
        elif merge_strategy == "Override (Smart Merge)":
            if manual_dict:
                self._deep_update_verbose(final_dict, manual_dict, debug_log)
        elif merge_strategy == "Append Only":
            if manual_dict:
                for k, v in manual_dict.items():
                    if k not in final_dict:
                        final_dict[k] = v
                        if debug_log: logger.info(f"➕ [MD_YAML] Appended: {k}")

        # 4. VALIDATE & DUMP
        if not isinstance(final_dict, dict):
             raise ValueError("[MD_YAML] Invalid result configuration. Must be a dictionary.")

        try:
            final_yaml_str = yaml.dump(final_dict, sort_keys=False, default_flow_style=False)
        except Exception as e:
             raise ValueError(f"[MD_YAML] Dump Error: {e}")

        # 5. SAVE
        if save_new_preset_as and save_new_preset_as.strip():
            self._save_preset(save_new_preset_as, final_yaml_str)

        if debug_log:
            logger.info("✅ [MD_YAML] Generation Complete.")

        return (final_yaml_str, info_str)

    def _extract_preset_info(self, path):
        """Reads the raw file to extract Header Description as a string."""
        info_lines = []
        try:
            with open(path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            capturing = False
            for line in lines:
                stripped = line.strip()
                if not stripped.startswith("#"):
                    break
                
                if "DESCRIPTION:" in stripped or "RECOMMENDED USAGE:" in stripped:
                    capturing = True
                
                if capturing:
                    clean_line = stripped.lstrip("#").strip()
                    if "▄▄▄" not in clean_line and "▀▀▀" not in clean_line:
                        info_lines.append(clean_line)
                
                if "▄▄▄" in stripped: 
                    break
            
            if not info_lines:
                return "(No Description Found)"
            return "\n".join(info_lines)
                
        except Exception as e:
            return f"(Error reading header: {e})"

    def _deep_update_verbose(self, base, update, debug=False, path=""):
        """Recursive merge with logging of changes."""
        for key, value in update.items():
            current_path = f"{path}.{key}" if path else key
            
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                self._deep_update_verbose(base[key], value, debug, current_path)
            else:
                if key in base:
                    old_val = base[key]
                    if old_val != value:
                        if debug: logger.info(f"✏️ [Override] {current_path}: {old_val} -> {value}")
                else:
                    if debug: logger.info(f"➕ [New Key] {current_path}: {value}")
                
                base[key] = value

    def _save_preset(self, filename, content):
        """Saves current config as a new file in the preset directory."""
        name = filename.strip()
        if not name.lower().endswith(('.yaml', '.yml')): name += '.yaml'
        path = os.path.join(self.preset_dir, name)
        save_dir = os.path.dirname(path)
        try:
            if not os.path.exists(save_dir): os.makedirs(save_dir)
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"💾 [MD_YAML] Preset Saved: {name}")
        except Exception as e:
            logger.error(f"❌ [MD_YAML] Save Failed: {e}")

# =================================================================================
# == Node Registration
# =================================================================================
NODE_CLASS_MAPPINGS = { "MD_YAML_Generator": MD_YAML_Generator }
NODE_DISPLAY_NAME_MAPPINGS = { "MD_YAML_Generator": "MD: YAML Configuration Tool" }

# =================================================================================
# == Development & Testing
# =================================================================================
if __name__ == "__main__":
    print("🧪 Running MD_YAML_Generator Tests...")
    
    test_passed = 0
    test_failed = 0
    
    try:
        node = MD_YAML_Generator()
        print("   • Test 1: Deep Update...", end=" ")
        base = {"a": 1, "b": {"x": 10}}
        update = {"a": 2, "b": {"x": 20}}
        node._deep_update_verbose(base, update, debug=True)
        assert base["a"] == 2 and base["b"]["x"] == 20, "Deep update failed"
        print("PASSED ✅")
        test_passed += 1
        
    except Exception as e:
        print(f"❌ Tests Failed: {e}")
        test_failed += 1
    
    print(f"\n{'='*60}")
    print(f"Test Results: {test_passed} passed, {test_failed} failed")
    
    if test_failed == 0:
        print("🎉 All tests passed!")