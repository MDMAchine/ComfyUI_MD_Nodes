# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# █▓▒░                                                                     ░▒▓█
# █▓▒░          MD_Nodes/DynamicLoRAStacker – Style Butler v1.0.1          ░▒▓█
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
# ║ ░▒▓ ORIGIN: Original Implementation - Quality of Life Enhancement
# ║ ░▒▓ DESCRIPTION:
# ║    The Style Butler - YAML-driven LoRA stacking automation.
# ║    Loads multiple LoRAs with individual strengths from preset styles.
# ║    Automatically applies prompt prefix/suffix for complete style transformation.
# ║    NOTE: As an automation/routing utility, this runs entirely in the wrapper.
# ║
# ║ ░▒▓ CORE FEATURES:
# ║    ✔ YAML-Driven Presets: Load styles from lora_styles.yaml
# ║    ✔ Multi-LoRA Stacking: Automatically loads and applies up to 10 LoRAs
# ║    ✔ Prompt Enhancement: Auto-adds style-specific prefix and suffix safely
# ║    ✔ Fallback Handling: Never crashes if a LoRA is missing from your drive
# ║
# ║ ░▒▓ CHANGELOG:
# ║    v1.0.1 (2026-02-24) - Enterprise Standards Update
# ║    ├── FIX: Corrected prompt concatenation spacing issues (no squished words).
# ║    ├── FIX: Hardened `try/except` block around LoRA loading to prevent Comfy crashes.
# ║    └── VERIFIED: Tooltips meet strict v1.5.4 standards.
# ╚════════════════════════════════════════════════════════════════════════════


VERSION = "v1.0.1"  # UPS v1.5.8

import os
import logging
import folder_paths

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    logging.warning("[MD_DynamicLoRAStacker] PyYAML not available - install with: pip install pyyaml")

try:
    import comfy.sd
    LORA_LOADER_AVAILABLE = True
except ImportError:
    LORA_LOADER_AVAILABLE = False

# =================================================================================
# == Configuration Constants
# =================================================================================
CONST_STYLES_FILENAME = "lora_styles.yaml"
CONST_MAX_LORAS_PER_STYLE = 10

CONST_STRENGTH_MIN = 0.0
CONST_STRENGTH_MAX = 2.0
CONST_STRENGTH_DEFAULT = 1.0
CONST_MULTIPLIER_MIN = 0.0
CONST_MULTIPLIER_MAX = 3.0
CONST_MULTIPLIER_DEFAULT = 1.0
CONST_MULTIPLIER_STEP = 0.05

CONST_FALLBACK_STYLE_NAME = "None"
CONST_EMPTY_PROMPT_PREFIX = ""
CONST_EMPTY_PROMPT_SUFFIX = ""

# =================================================================================
# == Core Node Class
# =================================================================================
class MD_DynamicLoRAStacker:
    """Automates LoRA loading and prompt enhancement based on YAML presets."""
    
    @classmethod
    def INPUT_TYPES(cls):
        style_list = cls._load_style_names()
        
        return {
            "required": {
                "model": ("MODEL", {
                    "tooltip": (
                        "MODEL INPUT\n"
                        "• Purpose: The base diffusion model.\n"
                        "• Action: Selected LoRAs will be patched into this model."
                    )
                }),
                "clip": ("CLIP", {
                    "tooltip": (
                        "CLIP INPUT\n"
                        "• Purpose: The text encoder.\n"
                        "• Action: Selected LoRAs will be patched into this encoder."
                    )
                }),
                "style_preset": (style_list, {
                    "default": style_list[0] if style_list else CONST_FALLBACK_STYLE_NAME,
                    "tooltip": (
                        "STYLE PRESET\n"
                        "• Purpose: Loads a pre-configured LoRA stack from lora_styles.yaml.\n"
                        "• Action: Applies multiple LoRAs, weights, and prompt tags instantly.\n"
                        "\n⭐ Tip: Edit the YAML file to create your own custom studio presets."
                    )
                }),
                "global_strength_multiplier": ("FLOAT", {
                    "default": CONST_MULTIPLIER_DEFAULT,
                    "min": CONST_MULTIPLIER_MIN,
                    "max": CONST_MULTIPLIER_MAX,
                    "step": CONST_MULTIPLIER_STEP,
                    "tooltip": (
                        "GLOBAL STRENGTH MULTIPLIER\n"
                        "• Purpose: Scales the intensity of the entire preset stack.\n"
                        "• Note: 1.0 uses YAML values exactly. 0.5 cuts all strengths in half.\n"
                        "\n⭐ Recommended: 1.0."
                    )
                }),
            },
            "optional": {
                "base_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": (
                        "BASE PROMPT\n"
                        "• Purpose: Your main subject prompt.\n"
                        "• Action: The node will automatically wrap this with the preset's prefix/suffix."
                    )
                }),
                "override_prefix": ("STRING", {
                    "default": "",
                    "tooltip": (
                        "OVERRIDE PREFIX\n"
                        "• Purpose: Replaces the preset's default prefix with your own."
                    )
                }),
                "override_suffix": ("STRING", {
                    "default": "",
                    "tooltip": (
                        "OVERRIDE SUFFIX\n"
                        "• Purpose: Replaces the preset's default suffix with your own."
                    )
                }),
            }
        }
    
    RETURN_TYPES = ("MODEL", "CLIP", "STRING", "STRING")
    RETURN_NAMES = ("model", "clip", "enhanced_prompt", "style_info")
    FUNCTION = "apply_style"
    CATEGORY = "MD_Nodes/LoRa"
    
    @classmethod
    def _load_style_names(cls):
        if not YAML_AVAILABLE: return [CONST_FALLBACK_STYLE_NAME]
        
        yaml_path = cls._get_yaml_path()
        if not os.path.exists(yaml_path):
            logging.warning(f"[MD_DynamicLoRAStacker] YAML not found: {yaml_path}")
            return [CONST_FALLBACK_STYLE_NAME]
        
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                styles_data = yaml.safe_load(f)
            if styles_data and isinstance(styles_data, dict):
                return [CONST_FALLBACK_STYLE_NAME] + list(styles_data.keys())
            return [CONST_FALLBACK_STYLE_NAME]
        except Exception as e:
            logging.error(f"[MD_DynamicLoRAStacker] Error loading YAML: {e}")
            return [CONST_FALLBACK_STYLE_NAME]
    
    @classmethod
    def _get_yaml_path(cls):
        node_dir = os.path.dirname(os.path.realpath(__file__))
        return os.path.join(node_dir, CONST_STYLES_FILENAME)
    
    def _load_style_config(self, style_name):
        if style_name == CONST_FALLBACK_STYLE_NAME: return None
        yaml_path = self._get_yaml_path()
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                styles_data = yaml.safe_load(f)
            if style_name in styles_data: return styles_data[style_name]
            logging.error(f"[MD_DynamicLoRAStacker] Style '{style_name}' not found in YAML")
            return None
        except Exception as e:
            logging.error(f"[MD_DynamicLoRAStacker] Error loading config: {e}")
            return None
    
    def _apply_lora(self, model, clip, lora_name, strength_model, strength_clip):
        if not LORA_LOADER_AVAILABLE: return (model, clip)
        
        lora_path = folder_paths.get_full_path("loras", lora_name)
        if lora_path is None or not os.path.exists(lora_path):
            logging.warning(f"[MD_DynamicLoRAStacker] ⚠️ LoRA not found: {lora_name} - Skipping")
            return (model, clip)
        
        try:
            import comfy.utils
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            model_lora, clip_lora = comfy.sd.load_lora_for_models(
                model, clip, lora, strength_model, strength_clip
            )
            return (model_lora, clip_lora)
        except Exception as e:
            logging.error(f"[MD_DynamicLoRAStacker] Error loading LoRA {lora_name}: {e}")
            return (model, clip)
    
    def apply_style(self, model, clip, style_preset, global_strength_multiplier, 
                    base_prompt="", override_prefix="", override_suffix=""):
        try:
            if style_preset == CONST_FALLBACK_STYLE_NAME:
                return (model, clip, base_prompt, "No style applied - 'None' selected")
            
            style_config = self._load_style_config(style_preset)
            if style_config is None:
                return (model, clip, base_prompt, f"⚠️ Failed to load style: {style_preset}")
            
            loras_list = style_config.get("loras", [])
            yaml_prefix = style_config.get("prompt_prefix", CONST_EMPTY_PROMPT_PREFIX)
            yaml_suffix = style_config.get("prompt_suffix", CONST_EMPTY_PROMPT_SUFFIX)
            description = style_config.get("description", "No description")
            
            final_prefix = override_prefix if override_prefix else yaml_prefix
            final_suffix = override_suffix if override_suffix else yaml_suffix
            
            logging.info("\n" + "="*70)
            logging.info(f"🎨 MD_DynamicLoRAStacker - Applying Style: {style_preset}")
            logging.info(f"📝 Description: {description}")
            logging.info("="*70)
            
            lora_count = 0
            skipped_count = 0
            
            for lora_entry in loras_list[:CONST_MAX_LORAS_PER_STYLE]:
                lora_name = lora_entry.get("name")
                strength_model = lora_entry.get("strength_model", CONST_STRENGTH_DEFAULT)
                strength_clip = lora_entry.get("strength_clip", CONST_STRENGTH_DEFAULT)
                if not lora_name: continue
                
                adjusted_model = max(CONST_STRENGTH_MIN, min(CONST_STRENGTH_MAX, strength_model * global_strength_multiplier))
                adjusted_clip = max(CONST_STRENGTH_MIN, min(CONST_STRENGTH_MAX, strength_clip * global_strength_multiplier))
                
                model_before = model
                model, clip = self._apply_lora(model, clip, lora_name, adjusted_model, adjusted_clip)
                
                if model is model_before:
                    logging.warning(f"  ⚠️  SKIPPED: {lora_name}")
                    skipped_count += 1
                else:
                    logging.info(f"  ✅ LOADED: {lora_name}")
                    logging.info(f"      Model: {adjusted_model:.2f} | CLIP: {adjusted_clip:.2f}")
                    lora_count += 1
            
            # Formatting the prompt correctly
            parts = []
            if final_prefix.strip(): parts.append(final_prefix.strip())
            if base_prompt.strip(): parts.append(base_prompt.strip())
            if final_suffix.strip(): parts.append(final_suffix.strip())
            
            enhanced_prompt = ", ".join(parts).replace(" ,", ",")
            
            style_info = (
                f"Style: {style_preset}\n"
                f"LoRAs Loaded: {lora_count}\n"
                f"LoRAs Skipped: {skipped_count}\n"
                f"Global Multiplier: {global_strength_multiplier:.2f}\n"
                f"Prefix: {final_prefix[:50]}{'...' if len(final_prefix) > 50 else ''}\n"
                f"Suffix: {final_suffix[:50]}{'...' if len(final_suffix) > 50 else ''}"
            )
            
            logging.info("-"*70)
            logging.info(f"📊 Summary:")
            logging.info(f"  ✅ LoRAs Loaded: {lora_count}")
            if skipped_count > 0: print(f"  ⚠️  LoRAs Skipped: {skipped_count}")
            logging.info(f"  🔢 Global Multiplier: {global_strength_multiplier:.2f}")
            logging.info("="*70 + "\n")
            
            return (model, clip, enhanced_prompt, style_info)
            
        except Exception as e:
            logging.error(f"[MD_DynamicLoRAStacker] Error applying style: {e}")
            return (model, clip, base_prompt, f"❌ Error: {str(e)}")

# =================================================================================
# == Node Registration
# =================================================================================
NODE_CLASS_MAPPINGS = { "MD_DynamicLoRAStacker": MD_DynamicLoRAStacker }
NODE_DISPLAY_NAME_MAPPINGS = { "MD_DynamicLoRAStacker": "MD: Dynamic LoRA Stacker (Style Butler)" }


# ==============================================================================
# == Unit Tests (smoke — runs without ComfyUI)
# ==============================================================================

if __name__ == "__main__":
    print("\n🧪 Smoke tests: MD_DynamicLoRAStacker")
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

    _check("VERSION defined",    VERSION == "v1.0.1")
    _check("CONST CONST_STYLES_FILENAME defined", CONST_STYLES_FILENAME is not None)
    _check("CONST CONST_MAX_LORAS_PER_STYLE defined", CONST_MAX_LORAS_PER_STYLE is not None)
    _check("CONST CONST_STRENGTH_MIN defined", CONST_STRENGTH_MIN is not None)
    _check("CONST CONST_STRENGTH_MAX defined", CONST_STRENGTH_MAX is not None)
    _check("CONST CONST_STRENGTH_DEFAULT defined", CONST_STRENGTH_DEFAULT is not None)
    _check("NODE_CLASS_MAPPINGS defined",
           isinstance(NODE_CLASS_MAPPINGS, dict) and len(NODE_CLASS_MAPPINGS) > 0)
    _check("  class MD_DynamicLoRAStacker in map", "MD_DynamicLoRAStacker" in NODE_CLASS_MAPPINGS)

    print(f"\n  {_pass} passed, {_fail} failed")
    if _fail == 0:
        print("  🎉 All good.")
