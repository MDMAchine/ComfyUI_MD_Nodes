# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
# ████ MD_Nodes/SmartFilenameBuilder – Dynamic Filename Toolkit v1.1.0 ████▓▒░
# © 2025 MDMAchine
# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
# ░▒▓ ORIGIN & DEV:
#   • Cast into the void by: MDMAchine
#   • Enhanced by: Gemini, Claude
#   • License: Apache 2.0 — Sharing is caring

# ░▒▓ DESCRIPTION:
#   A suite of nodes for creating dynamic, complex, and clean filenames.
#   Includes a preset-based builder, a simple token replacer, and a
#   persistent, file-based counter for robust file organization.

# ░▒▓ FEATURES:
#   ✓ `SmartFilenameBuilder`: Preset-driven (Vocal, Master) & custom filename generation.
#   ✓ `FilenameTokenReplacer`: Simple `{token}` substitution for custom templates.
#   ✓ `FilenameCounterNode`: Persistent, context-aware, auto-incrementing counter.
#   ✓ Handles date/time formatting (`%Y-%m-%d`), component toggles, and sanitization.

# ░▒▓ CHANGELOG:
#   - v1.1.0 (Stability & Format Fixes):
#       • FIXED: Added missing `json` import (prevented Counter crash).
#       • FIXED: Changed logger reference from undefined `logger` to `logging`.
#       • CHANGED: Seed formatting now uses underscore separator (`Seed_1234`).
#       • CHANGED: Counter storage moved to input directory for better persistence.
#   - v1.0.0 (Initial Release):
#       • Initial suite release.

# ░▒▓ CONFIGURATION:
#   → Primary Use: `SmartFilenameBuilder` to create complex, clean filenames with presets.
#   → Secondary Use: `FilenameCounterNode` to add persistent, zero-padded counters (`#0001`).

# ░▒▓ WARNING:
#   This node may trigger:
#   ▓▒░ Obsessively tweaking separators, padding, and date formats.
#   ▓▒░ A deep, spiritual satisfaction from seeing `#0001` increment to `#0002`.
#   Consult your nearest demoscene vet if hallucinations persist.
# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄


# =================================================================================
# == Standard Library Imports                                                     ==
# =================================================================================
import os
import re
import json  # CRITICAL FIX: Added missing import
from datetime import datetime
import logging
import traceback
import secrets

# =================================================================================
# == ComfyUI Core Modules                                                         ==
# =================================================================================
import folder_paths

# =================================================================================
# == Core Node Class: SmartFilenameBuilder                                        ==
# =================================================================================

class SmartFilenameBuilder:
    """
    MD: Smart Filename Builder
    Generates complex and clean filenames using presets or custom configurations.
    """

    # Class variable for presets
    PRESETS = {
        "Custom": {},
        "Instrumental": {
            "mode_tag": "(Instrumental)", "include_steps": True, "include_schedule": True,
            "include_seed": False, "include_genre": True,
        },
        "Vocal": {
            "mode_tag": "(Vocal)", "include_steps": True, "include_schedule": True,
            "include_seed": False, "include_genre": True,
        },
        "Master": {
            "mode_tag": "(Master)", "include_steps": True, "include_schedule": False,
            "include_seed": False, "include_genre": False,
        },
        "Raw Output": {
            "mode_tag": "(Raw)", "include_steps": True, "include_schedule": True,
            "include_seed": True, "include_genre": False,
        },
        "AB Test": {
            "mode_tag": "(ABMode - MD LoRa)", "include_steps": True, "include_schedule": True,
            "include_seed": False, "include_genre": False,
        },
    }

    @classmethod
    def INPUT_TYPES(cls):
        """Define node inputs with detailed tooltips."""
        return {
            "required": {
                "preset": (list(cls.PRESETS.keys()), {
                    "default": "Custom",
                    "tooltip": "FILENAME PRESET\n- Select a preset to automatically configure options."
                }),
                "base_template": ("STRING", {
                    "default": "MD_Nodes_Workflow %Y-%m-%d", "multiline": False,
                    "tooltip": "BASE TEMPLATE\n- The starting part. Supports strftime: %Y, %m, %d, %H, %M."
                }),
                "project_path": ("STRING", {
                    "default": "Ace-Step/313/", "multiline": False,
                    "tooltip": "PROJECT SUBDIRECTORY\n- Defines subfolder structure within output."
                }),
            },
            "optional": {
                "mode_tag": ("STRING", {"default": "(Custom Mode)", "tooltip": "MODE TAG (Custom Only)"}),
                "steps": ("INT", {"default": 0, "min": 0, "tooltip": "SAMPLING STEPS"}),
                "schedule_info": ("STRING", {"default": "", "tooltip": "SCHEDULER INFO"}),
                "seed": ("INT", {"default": 0, "min": 0, "tooltip": "SEED VALUE"}),
                "genre": ("STRING", {"default": "", "tooltip": "GENRE TAGS"}),
                "custom_tag_1": ("STRING", {"default": "", "tooltip": "CUSTOM TAG 1"}),
                "custom_tag_2": ("STRING", {"default": "", "tooltip": "CUSTOM TAG 2"}),
                "counter_start": ("INT", {"default": 0, "min": 0, "max": 99999, "tooltip": "COUNTER VALUE"}),
                
                # Component Toggles
                "include_steps": ("BOOLEAN", {"default": True, "tooltip": "Include 'XXX Steps'?"}),
                "include_schedule": ("BOOLEAN", {"default": True, "tooltip": "Include Scheduler info?"}),
                "include_seed": ("BOOLEAN", {"default": False, "tooltip": "Include 'Seed_XXXX'?"}),
                "include_genre": ("BOOLEAN", {"default": False, "tooltip": "Include cleaned genre tags?"}),
                "include_counter": ("BOOLEAN", {"default": True, "tooltip": "Include '#XXXX' counter?"}),
                
                # Formatting
                "separator": ("STRING", {"default": " - ", "tooltip": "SEPARATOR string."}),
                "genre_max_length": ("INT", {"default": 40, "min": 10, "max": 100, "tooltip": "GENRE MAX LENGTH"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("full_path_prefix", "filename_preview")
    FUNCTION = "build_filename"
    CATEGORY = "MD_Nodes/Utility"

    def build_filename(self, preset, base_template, project_path,
                       mode_tag="(Custom Mode)", steps=0, schedule_info="",
                       seed=0, genre="", custom_tag_1="", custom_tag_2="",
                       counter_start=0,
                       include_steps=True, include_schedule=True,
                       include_seed=False, include_genre=False, include_counter=True,
                       separator=" - ", genre_max_length=40):
        try:
            # --- Apply Preset Overrides ---
            preset_config = {}
            if preset != "Custom" and preset in self.PRESETS:
                preset_config = self.PRESETS[preset]
                if mode_tag == "(Custom Mode)" and "mode_tag" in preset_config:
                    mode_tag = preset_config["mode_tag"]
                include_steps = preset_config.get("include_steps", include_steps)
                include_schedule = preset_config.get("include_schedule", include_schedule)
                include_seed = preset_config.get("include_seed", include_seed)
                include_genre = preset_config.get("include_genre", include_genre)

            # --- Process Base Template (Date/Time) ---
            try:
                filename_base = datetime.now().strftime(base_template)
            except ValueError as date_err:
                 logging.warning(f"[SmartFilenameBuilder] Invalid date format: {date_err}")
                 filename_base = base_template

            filename_base = self._sanitize_filename(filename_base)

            # --- Build Component Parts List ---
            parts = []

            if include_counter and counter_start > 0:
                parts.append(f"#{counter_start:04d}")

            if include_steps and steps > 0:
                parts.append(f"{steps}S")

            if include_schedule and schedule_info:
                clean_schedule = self._sanitize_filename(schedule_info)
                if clean_schedule: parts.append(clean_schedule)

            if include_seed and seed > 0:
                # UPDATED: Added underscore for readability
                parts.append(f"Seed_{seed}") 

            if include_genre and genre:
                clean_genre = self._clean_genre(genre, genre_max_length)
                if clean_genre: parts.append(clean_genre)

            if custom_tag_1:
                clean_tag1 = self._sanitize_filename(custom_tag_1)
                if clean_tag1: parts.append(clean_tag1)
            if custom_tag_2:
                clean_tag2 = self._sanitize_filename(custom_tag_2)
                if clean_tag2: parts.append(clean_tag2)

            if mode_tag:
                clean_mode = self._sanitize_filename(mode_tag, allow_parentheses=True)
                if clean_mode: parts.append(clean_mode)

            # --- Combine Filename ---
            filename_final = filename_base
            if parts:
                filename_final += separator + separator.join(parts)

            # Final cleanup
            safe_separator = re.escape(separator)
            filename_final = re.sub(f'({safe_separator})+', separator, filename_final)
            filename_final = filename_final.strip(separator.strip())

            # --- Construct Full Path Prefix ---
            path_segments = [self._sanitize_filename(seg) for seg in project_path.replace('\\', '/').split('/') if seg]
            clean_project_path = "/".join(path_segments) + "/" if path_segments else ""
            full_path_prefix = os.path.join(clean_project_path, filename_final).replace('\\', '/')

            preview = (f"Directory: [output]/{clean_project_path}\n"
                       f"Filename: {filename_final}\n\n"
                       f"➡️ Full Prefix: {full_path_prefix}")

            return (full_path_prefix, preview)

        except Exception as e:
            logging.error(f"[SmartFilenameBuilder] Error: {e}", exc_info=True)
            return ("error/error_filename", f"ERROR: {e}")


    def _sanitize_filename(self, text, allow_parentheses=False):
        if not isinstance(text, str) or not text: return ""
        text = text.strip()
        if allow_parentheses:
            invalid_chars = r'[<>:"/\\|?*\x00-\x1f]'
        else:
            invalid_chars = r'[<>:"/\\|?*\(\)\[\]\{\}\x00-\x1f]'
        text = re.sub(invalid_chars, '', text)
        text = re.sub(r'\s+', '_', text)
        text = re.sub(r'[-_]{2,}', '_', text)
        text = text.strip('._- ')
        return text if text else "sanitized_empty"

    def _clean_genre(self, genre, max_length):
        if not isinstance(genre, str) or not genre: return ""
        text = re.sub(r'[,;]+', ' ', genre)
        text = re.sub(r'[^\w\s-]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'[-]{2,}', '-', text).strip('- ')
        if len(text) > max_length:
            cut_point = -1
            for char in [' ', '-']:
                 found = text.rfind(char, 0, max_length)
                 if found > cut_point: cut_point = found
            if cut_point != -1: text = text[:cut_point]
            else: text = text[:max_length]
            text = text.strip(' -')
        return text


# =================================================================================
# == Core Node Class: FilenameTokenReplacer                                       ==
# =================================================================================

class FilenameTokenReplacer:
    """
    MD: Filename Token Replacer
    Replaces predefined tokens in a template string.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "template": ("STRING", {
                    "default": "{project}/{date} - {steps}S - {mode}", "multiline": True,
                    "tooltip": "Use tokens like {date}, {project}, {mode}, {steps}, {seed}."
                }),
            },
            "optional": {
                "project": ("STRING", {"default": "MyProject"}),
                "mode": ("STRING", {"default": "Image"}),
                "steps": ("INT", {"default": 0, "min": 0}),
                "seed": ("INT", {"default": 0, "min": 0}),
                "genre": ("STRING", {"default": ""}),
                "custom1": ("STRING", {"default": ""}),
                "custom2": ("STRING", {"default": ""}),
                "date_format": ("STRING", {"default": "%Y-%m-%d"}),
                "time_format": ("STRING", {"default": "%H-%M-%S"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("result_string", "preview")
    FUNCTION = "replace_tokens"
    CATEGORY = "MD_Nodes/Utility"

    def replace_tokens(self, template, project="", mode="", steps=0, seed=0,
                       genre="", custom1="", custom2="",
                       date_format="%Y-%m-%d", time_format="%H-%M-%S"):
        try:
            now = datetime.now()
            tokens = {
                "project": self._sanitize_text(project),
                "mode": self._sanitize_text(mode),
                "steps": str(steps) if steps > 0 else "",
                "seed": str(seed) if seed > 0 else "",
                "genre": self._clean_text(genre, 40),
                "custom1": self._sanitize_text(custom1),
                "custom2": self._sanitize_text(custom2),
                "date": self._safe_strftime(now, date_format, "DATE_ERR"),
                "time": self._safe_strftime(now, time_format, "TIME_ERR"),
                "year": self._safe_strftime(now, "%Y", "YYYY"),
                "month": self._safe_strftime(now, "%m", "MM"),
                "day": self._safe_strftime(now, "%d", "DD"),
                "hour": self._safe_strftime(now, "%H", "HH"),
                "minute": self._safe_strftime(now, "%M", "MM"),
                "second": self._safe_strftime(now, "%S", "SS"),
            }

            result = template
            for key, value in tokens.items():
                result = re.sub(r'\{' + re.escape(key) + r'\}', str(value), result, flags=re.IGNORECASE)

            result = re.sub(r'\{[a-zA-Z0-9_]+\}', '', result)
            result = re.sub(r'\s*-\s*-?\s*', ' - ', result)
            result = re.sub(r'\s*/\s*/?\s*', '/', result)
            result = re.sub(r'\s*_\s*_?\s*', '_', result)
            result = re.sub(r'[_\-\s]{2,}', '_', result)
            result = result.strip(' _-/')
            result = self._sanitize_filepath(result)

            preview = "Token Replacements:\n" + "\n".join([
                f"  {{{k}}}: {v}" for k, v in tokens.items() if v or k in ["steps", "seed"]
            ]) + f"\n\n➡️ Result:\n  {result}"

            return (result, preview)

        except Exception as e:
            logging.error(f"[FilenameTokenReplacer] Error: {e}", exc_info=True)
            return ("error_replacing_tokens", f"ERROR: {e}")

    def _safe_strftime(self, dt_obj, fmt, fallback=""):
        try: return dt_obj.strftime(fmt)
        except ValueError: return fallback

    def _sanitize_text(self, text):
        if not isinstance(text, str): text = str(text)
        text = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '', text)
        text = re.sub(r'\s+', '_', text).strip('_')
        return text[:100]

    def _clean_text(self, text, max_length):
        if not isinstance(text, str): text = str(text)
        text = re.sub(r'[^\w\s-]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        if len(text) > max_length:
            text = text[:max_length].rsplit(' ', 1)[0]
        return text.replace(' ', '_')

    def _sanitize_filepath(self, path_str):
        parts = path_str.replace('\\', '/').split('/')
        sanitized_parts = [self._sanitize_text(part) for part in parts if part and part != '.']
        return "/".join(sanitized_parts)


# =================================================================================
# == Core Node Class: FilenameCounterNode                                         ==
# =================================================================================

class FilenameCounterNode:
    """
    MD: Filename Counter
    Provides a persistent, auto-incrementing counter stored in a JSON file.
    """

    # Store in Input folder for better persistence than temp
    COUNTERS_FILE = os.path.join(folder_paths.get_input_directory(), "md_filename_counters.json")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "context_key": ("STRING", {"default": "default_counter", "tooltip": "Unique name for this counter."}),
                "start_value": ("INT", {"default": 1, "min": 0, "max": 999999}),
                "increment": ("INT", {"default": 1, "min": 1, "max": 100}),
                "padding": ("INT", {"default": 4, "min": 0, "max": 8}),
            },
            "optional": {
                "reset_counter": ("BOOLEAN", {"default": False, "tooltip": "Reset context_key to start_value?"}),
                "prefix": ("STRING", {"default": "#", "tooltip": "Optional prefix (e.g., '#')."}),
                "suffix": ("STRING", {"default": "", "tooltip": "Optional suffix."}),
                "trigger": ("BOOLEAN", {"default": True, "label_on": "INCREMENT", "label_off": "READ ONLY"}),
            }
        }

    RETURN_TYPES = ("STRING", "INT", "STRING")
    RETURN_NAMES = ("formatted_counter", "current_value", "info")
    FUNCTION = "get_counter"
    CATEGORY = "MD_Nodes/Utility"
    OUTPUT_NODE = False

    def __init__(self):
        self.counters = self._load_counters()

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return secrets.token_hex(16)

    def get_counter(self, context_key, start_value=1, increment=1, padding=4,
                    reset_counter=False, prefix="#", suffix="", trigger=True):
        info = f"Context: {context_key}\n"
        current_value = -1

        try:
            safe_context_key = re.sub(r'[^\w\-]+', '_', context_key)
            if not safe_context_key: safe_context_key = "default_counter"

            self.counters = self._load_counters()

            if reset_counter:
                self.counters[safe_context_key] = max(0, start_value)
                self._save_counters()
                info += f"Action: RESET to {self.counters[safe_context_key]}\n"

            if safe_context_key not in self.counters:
                self.counters[safe_context_key] = max(0, start_value)
                info += f"Action: Initialized to {self.counters[safe_context_key]}\n"

            current_value = self.counters[safe_context_key]
            
            if padding > 0: formatted_num = f"{current_value:0{padding}d}"
            else: formatted_num = str(current_value)
            
            formatted_string = f"{prefix}{formatted_num}{suffix}"

            next_value = current_value
            if trigger:
                 next_value = current_value + max(1, increment)
                 self.counters[safe_context_key] = next_value
                 self._save_counters()
                 info += f"Action: Incremented by {increment}\n"
            else:
                 info += "Action: Read Only (Trigger OFF)\n"

            info += f"Output Value: {current_value}\nNext Value: {next_value}"
            return (formatted_string, current_value, info)

        except Exception as e:
            logging.error(f"[FilenameCounter] Error: {e}", exc_info=True)
            return (f"{prefix}ERROR{suffix}", current_value, f"ERROR: {e}")

    def _load_counters(self):
        counters = {}
        try:
            if os.path.exists(self.COUNTERS_FILE):
                with open(self.COUNTERS_FILE, 'r', encoding='utf-8') as f:
                    try:
                        counters = json.load(f)
                    except json.JSONDecodeError:
                         counters = {}
        except Exception as e:
            logging.error(f"[FilenameCounter] Load Error: {e}")
        return counters

    def _save_counters(self):
        try:
            os.makedirs(os.path.dirname(self.COUNTERS_FILE), exist_ok=True)
            with open(self.COUNTERS_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.counters, f, indent=4)
        except Exception as e:
            logging.error(f"[FilenameCounter] Save Error: {e}")


# =================================================================================
# == Node Registration                                                            ==
# =================================================================================

NODE_CLASS_MAPPINGS = {
    "SmartFilenameBuilder": SmartFilenameBuilder,
    "FilenameTokenReplacer": FilenameTokenReplacer,
    "FilenameCounterNode": FilenameCounterNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SmartFilenameBuilder": "MD: Smart Filename Builder",
    "FilenameTokenReplacer": "MD: Filename Token Replacer",
    "FilenameCounterNode": "MD: Filename Counter",
}