# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# █▓▒░                                                                     ░▒▓█
# █▓▒░          MD_Nodes/MD_YAML_Utils – Config Architect v1.3.1           ░▒▓█
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
# ║   A Swiss-Army Knife for YAML configurations.
# ║   It allows you to dynamically Modify, Merge, and Read YAML strings.
# ║   NOTE: As a dictionary/string utility, this runs entirely in the public wrapper domain.
# ║
# ║ ░▒▓ FEATURES:
# ║   ✔ Dynamic Patching: Inject values via Dot Notation (supports lists).
# ║   ✔ Template Substitute: Simple {placeholder} replacement mode.
# ║   ✔ Smart Types: Auto-converts "[1,2]" to lists, "true" to bools.
# ║   ✔ Diff Logging: Logs exact changes for debugging.
# ║   ✔ Deep Merge: Smartly combines nested configurations.
# ║
# ║ ░▒▓ CHANGELOG:
# ║   - v1.3.1 (Enterprise Standards - Feb 2026):
# ║       • VERIFIED: Tooltips meet v1.5.4 standard.
# ║   - v1.3.0 (Enterprise Standard):
# ║       • Refactored for MD Guidelines 1.5.1.
# ║       • Added strict unit tests and global constants.
# ║
# ║ ░▒▓ CONFIGURATION:
# ║   → Primary Use: Dynamic injection of values (seeds, filenames) into YAML.
# ║   → Secondary Use: Merging two partial configs.
# ╚════════════════════════════════════════════════════════════════════════════

# =================================================================================
# == Standard Library Imports
# =================================================================================
VERSION = "v1.3.1"  # UPS v1.5.8


import logging

# =================================================================================
# == Third-Party Imports
# =================================================================================
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    logging.error("[MD_YAML_Utils] PyYAML not found. Node will fail.")

# =================================================================================
# == Configuration Constants
# =================================================================================
CONST_ANY_TYPE = "*"

# =================================================================================
# == Helper Logic
# =================================================================================

class AnyType(str):
    """Wildcard type for ComfyUI connections."""
    def __ne__(self, __value: object) -> bool:
        return False

any_type = AnyType(CONST_ANY_TYPE)

def deep_merge(dict_a, dict_b):
    """Recursive merge of two dictionaries."""
    result = dict_a.copy()
    for key, value in dict_b.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result

def infer_value_type(value):
    """Smartly converts string inputs to their real Python type."""
    if not isinstance(value, str):
        return value
    if not YAML_AVAILABLE:
        return value
    try:
        return yaml.safe_load(value)
    except Exception:
        return value

def get_nested(data, key_path):
    """Get value using dot notation, supporting list indices."""
    if not key_path:
        return None
    keys = key_path.split('.')
    current = data
    for k in keys:
        if isinstance(current, dict) and k in current:
            current = current[k]
        elif isinstance(current, list) and k.isdigit():
            idx = int(k)
            if 0 <= idx < len(current):
                current = current[idx]
            else:
                return None 
        else:
            return None
    return current

def set_nested(data, key_path, value):
    """Set value using dot notation, supporting list indices."""
    if not key_path:
        return data, "ERROR: No key provided"
        
    keys = key_path.split('.')
    current = data
    
    for i, k in enumerate(keys[:-1]):
        if isinstance(current, list) and k.isdigit():
            idx = int(k)
            if 0 <= idx < len(current):
                current = current[idx]
            else:
                return data, f"ERROR: Index {idx} out of bounds"
                
        elif isinstance(current, dict):
            if k not in current:
                current[k] = {} 
            current = current[k]
        else:
            return data, f"ERROR: Path blocked at '{k}'"

    final_key = keys[-1]
    
    if isinstance(current, list) and final_key.isdigit():
        idx = int(final_key)
        if 0 <= idx < len(current):
            old_value = current[idx]
            current[idx] = value
            return data, old_value
        else:
            return data, f"ERROR: Index {idx} out of bounds"
            
    elif isinstance(current, dict):
        old_value = current.get(final_key, "NEW")
        current[final_key] = value
        return data, old_value
        
    return data, "ERROR: Invalid Target Structure"

def count_keys_recursive(data):
    """Counts total keys in a nested structure."""
    count = 0
    if isinstance(data, dict):
        for k, v in data.items():
            count += 1
            count += count_keys_recursive(v)
    elif isinstance(data, list):
        for item in data:
            count += count_keys_recursive(item)
    return count

# =================================================================================
# == Core Node Class
# =================================================================================

class MD_YAML_Utils:
    """
    The Machine Interface for YAML manipulation.
    Allows runtime modification of configuration data.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "yaml_input": ("STRING", {
                    "multiline": True, 
                    "default": "", 
                    "tooltip": (
                        "BASE YAML INPUT\n"
                        "• Purpose: The primary YAML string to be processed.\n"
                        "• Requirement: Must be a valid YAML structure (except for Template mode)."
                    )
                }),
                "mode": (["Merge (A + B)", "Patch Key", "Template Substitute", "Get Key", "Validate"], {
                    "default": "Patch Key", 
                    "tooltip": (
                        "EXECUTION MODE\n"
                        "• Merge: Combines Base YAML with Secondary YAML.\n"
                        "• Patch: Injects value at target_key (supports 'list.0.item').\n"
                        "• Template: Replaces {target_key} placeholders in text.\n"
                        "• Validate: Checks syntax & outputs structure stats.\n"
                        "\n⭐ Recommended: Patch Key for dynamic workflows."
                    )
                }),
            },
            "optional": {
                "yaml_secondary": ("STRING", {
                    "multiline": True, 
                    "default": "", 
                    "tooltip": (
                        "SECONDARY YAML\n"
                        "• Purpose: The data to merge into the Base YAML.\n"
                        "• Note: Only used in 'Merge (A + B)' mode."
                    )
                }),
                "target_key": ("STRING", {
                    "default": "", 
                    "tooltip": (
                        "TARGET KEY / PATH\n"
                        "• Purpose: Dot-notation path to modify (e.g. 'models.0.name' or 'seed').\n"
                        "• Template Mode: The exact string to replace (without brackets)."
                    )
                }),
                "value_input": (any_type, {
                    "tooltip": (
                        "VALUE TO INJECT\n"
                        "• Purpose: The data to insert at the target key.\n"
                        "• Note: Automatically converts strings like '[1,2]' to real Python lists."
                    )
                }),
            }
        }

    RETURN_TYPES = ("STRING", any_type, "BOOLEAN")
    RETURN_NAMES = ("yaml_output", "value_output", "is_valid")
    FUNCTION = "process_yaml"
    CATEGORY = "MD_Nodes/Utility"

    def process_yaml(self, yaml_input, mode, yaml_secondary="", target_key="", value_input=None):
        if not YAML_AVAILABLE:
            return (yaml_input, None, False)

        try:
            if mode != "Template Substitute":
                data = yaml.safe_load(yaml_input) if yaml_input.strip() else {}
                if not isinstance(data, (dict, list)): 
                    data = {} 
            is_valid = True
        except Exception as e:
            logging.error(f"[MD_YAML_Utils] Invalid Base YAML: {e}")
            return (yaml_input, None, False)

        result_yaml = yaml_input
        result_value = None

        if mode == "Merge (A + B)":
            try:
                data_b = yaml.safe_load(yaml_secondary) if yaml_secondary.strip() else {}
                if isinstance(data, dict) and isinstance(data_b, dict):
                    merged = deep_merge(data, data_b)
                    result_yaml = yaml.dump(merged, sort_keys=False)
                    logging.info(f"[MD_YAML_Utils] Merged configs successfully.")
                else:
                    logging.warning("[MD_YAML_Utils] Merge requires two dictionary structures.")
            except Exception as e:
                logging.error(f"[MD_YAML_Utils] Merge Failed: {e}")

        elif mode == "Patch Key":
            if target_key and value_input is not None:
                real_val = infer_value_type(value_input)
                if isinstance(data, (dict, list)):
                    updated_data, old_val = set_nested(data, target_key, real_val)
                    if str(old_val).startswith("ERROR"):
                        logging.warning(f"[MD_YAML_Utils] Patch Failed: {old_val}")
                    else:
                        result_yaml = yaml.dump(updated_data, sort_keys=False)
                        result_value = real_val
                        logging.info(f"[MD_YAML_Utils] Patch '{target_key}': {old_val} -> {real_val}")

        elif mode == "Template Substitute":
            if target_key and value_input is not None:
                placeholder = f"{{{target_key}}}"
                if placeholder in yaml_input:
                    val_str = str(value_input)
                    result_yaml = yaml_input.replace(placeholder, val_str)
                    logging.info(f"[MD_YAML_Utils] Replaced {placeholder} with '{val_str}'")
                    try:
                        yaml.safe_load(result_yaml)
                    except Exception:
                        logging.warning("[MD_YAML_Utils] Template result resulted in invalid YAML.")
                        is_valid = False
                else:
                    logging.warning(f"[MD_YAML_Utils] Placeholder {placeholder} not found.")

        elif mode == "Get Key":
            val = get_nested(data, target_key)
            result_value = val
            logging.info(f"[MD_YAML_Utils] Extracted '{target_key}': {val}")

        elif mode == "Validate":
            try:
                parsed = yaml.safe_load(yaml_input)
                count = count_keys_recursive(parsed)
                logging.info(f"[MD_YAML_Utils] ✓ Valid YAML structure. Total Keys/Items: {count}")
                is_valid = True
            except Exception as e:
                logging.error(f"[MD_YAML_Utils] ✗ Invalid YAML: {e}")
                is_valid = False

        return (result_yaml, result_value, is_valid)

# =================================================================================
# == Node Registration
# =================================================================================

NODE_CLASS_MAPPINGS = {
    "MD_YAML_Utils": MD_YAML_Utils
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MD_YAML_Utils": "MD: YAML Utils (Architect)"
}

# =================================================================================
# == Development & Testing
# =================================================================================

if __name__ == "__main__":
    logging.info("🧪 Running Self-Tests for MD_YAML_Utils...")
    
    test_passed = 0
    test_failed = 0
    
    try:
        if not YAML_AVAILABLE:
            raise ImportError("PyYAML not found for testing.")
            
        node = MD_YAML_Utils()
        
        base_list = "models:\n  - name: A\n  - name: B"
        res, _, _ = node.process_yaml(base_list, "Patch Key", target_key="models.1.name", value_input="C")
        assert "- name: C" in res, "Failed to patch list index"
        logging.info("✅ Array Indexing: PASSED")
        test_passed += 1
        
        base_valid = "a: 1\nb: 2"
        _, _, valid = node.process_yaml(base_valid, "Validate")
        assert valid is True, "Validation failed valid input"
        logging.info("✅ Validation: PASSED")
        test_passed += 1
        
        yaml_a = "a: 1\nb:\n  c: 3"
        yaml_b = "b:\n  d: 4"
        res, _, _ = node.process_yaml(yaml_a, "Merge (A + B)", yaml_secondary=yaml_b)
        assert "c: 3" in res and "d: 4" in res, "Deep merge failed"
        logging.info("✅ Deep Merge: PASSED")
        test_passed += 1

    except Exception as e:
        logging.error(f"❌ Test Failed: {e}")
        test_failed += 1
    
    logging.info(f"\n{'='*60}")
    logging.info(f"Test Results: {test_passed} passed, {test_failed} failed")
    
    if test_failed == 0:
        logging.info("🎉 All tests passed!")