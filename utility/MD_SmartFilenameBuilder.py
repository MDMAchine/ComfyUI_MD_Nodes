# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# █▓▒░                                                                     ░▒▓█
# █▓▒░   MD_Nodes/SmartFilenameBuilder – Dynamic Filename Toolkit v1.6.0   ░▒▓█
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
# ║   • Enhanced by: Gemini, Claude
# ║
# ║ ░▒▓ DESCRIPTION:
# ║   A suite of nodes for creating dynamic, complex, and clean filenames.
# ║   Includes a preset-based builder, a simple token replacer, and a
# ║   persistent, file-based counter for robust file organization.
# ║   NOTE: As an I/O string utility, this runs entirely in the public wrapper domain.
# ║
# ║ ░▒▓ FEATURES:
# ║   ✓ `SmartFilenameBuilder`: Preset-driven & custom filename generation.
# ║   ✓ `FilenameTokenReplacer`: Simple `{token}` substitution.
# ║   ✓ `FilenameCounterNode`: Persistent, context-aware, auto-incrementing counter.
# ║
# ║ ░▒▓ CHANGELOG:
# ║   - v1.6.0 (Enterprise Standards - Feb 2026):
# ║       • REFACTOR: Tooltips strictly updated to 5-part v1.5.4 standard.
# ║       • VERIFIED: PerformanceProfiler matches v1.5.3 exact specifications.
# ║   - v1.5.4 (Prior Update):
# ║       • ADDED: PerformanceProfiler and debug logging to SmartFilenameBuilder.
# ║       • ADDED: JS-safe seed capping.
# ╚════════════════════════════════════════════════════════════════════════════

# =================================================================================
# == Standard Library Imports                                                    ==
# =================================================================================
VERSION = "v1.6.0"  # UPS v1.5.8


import os
import re
import json
from datetime import datetime
import logging
import traceback
import secrets
import time

# =================================================================================
# == ComfyUI Core Modules                                                        ==
# =================================================================================
import folder_paths

# =================================================================================
# == Global Constants                                                            ==
# =================================================================================
CONST_JS_MAX_SAFE_INTEGER = 9007199254740991

# =================================================================================
# == Helper Classes (Enterprise Standards)                                       ==
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
        print("\n⏱️  PERFORMANCE (String Parse):")
        total = self.get_total_time()
        print(f"    • Total Time: {total:.4f}s")
        for op_name, times in sorted(self.timings.items()):
            avg = sum(times) / len(times)
            if len(times) == 1:
                print(f"    • {op_name}: {avg:.4f}s")
            else:
                print(f"    • {op_name}: {avg:.4f}s avg ({len(times)}x)")

# =================================================================================
# == Core Node Class: SmartFilenameBuilder                                       ==
# =================================================================================

class SmartFilenameBuilder:
    """Generates complex and clean filenames using presets or custom configurations."""

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
        return {
            "required": {
                "preset": (list(cls.PRESETS.keys()), {
                    "default": "Custom",
                    "tooltip": (
                        "FILENAME PRESET\n"
                        "• Purpose: Automatically configure toggles and tags for standard MD workflows.\n"
                        "• Options: Custom, Instrumental, Vocal, Master, etc.\n"
                        "• Trade-offs: Presets override manual boolean toggles.\n"
                        "\n⭐ Recommended: 'Custom' for full manual control."
                    )
                }),
                "base_template": ("STRING", {
                    "default": "MD_Nodes_Workflow %Y-%m-%d", "multiline": False,
                    "tooltip": (
                        "BASE TEMPLATE\n"
                        "• Purpose: The primary starting text of the filename.\n"
                        "• Format: Supports Python strftime codes (%Y=Year, %m=Month, %d=Day).\n"
                        "\n⭐ Recommended: Keep date at start for easy OS sorting."
                    )
                }),
                "project_path": ("STRING", {
                    "default": "Ace-Step/313/", "multiline": False,
                    "tooltip": (
                        "PROJECT SUBDIRECTORY\n"
                        "• Purpose: Defines nested subfolders within the ComfyUI output directory.\n"
                        "• Format: 'Folder/Subfolder/' (Auto-sanitized).\n"
                        "\n⭐ Recommended: Organize by Project/BPM/Version."
                    )
                }),
            },
            "optional": {
                "mode_tag": ("STRING", {
                    "default": "(Custom Mode)", 
                    "tooltip": (
                        "MODE TAG (CUSTOM)\n"
                        "• Purpose: A text tag identifying the workflow mode.\n"
                        "• Requirement: Only active when Preset is set to 'Custom'.\n"
                        "• Note: Parentheses are preserved during sanitization."
                    )
                }),
                "steps": ("INT", {
                    "default": 0, "min": 0, 
                    "tooltip": "SAMPLING STEPS\n• Purpose: Appends step count (e.g., '20S')."
                }),
                "schedule_info": ("STRING", {
                    "default": "", 
                    "tooltip": "SCHEDULER INFO\n• Purpose: Appends scheduler name/sigma info."
                }),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": CONST_JS_MAX_SAFE_INTEGER,
                    "tooltip": "SEED VALUE\n• Purpose: Appends JS-safe seed (e.g., 'Seed_12345')."
                }),
                "genre": ("STRING", {
                    "default": "", 
                    "tooltip": "GENRE TAGS\n• Purpose: Appends sanitized genre string."
                }),
                "custom_tag_1": ("STRING", {
                    "default": "", 
                    "tooltip": "CUSTOM TAG 1\n• Purpose: Extra user-defined tag (auto-sanitized)."
                }),
                "custom_tag_2": ("STRING", {
                    "default": "", 
                    "tooltip": "CUSTOM TAG 2\n• Purpose: Extra user-defined tag (auto-sanitized)."
                }),
                "counter_start": ("INT", {
                    "default": 0, "min": 0, "max": 99999, 
                    "tooltip": "COUNTER VALUE\n• Purpose: Current index for numbering (e.g. #0001).\n• Usage: Connect FilenameCounterNode here."
                }),
                
                # Component Toggles
                "include_steps": ("BOOLEAN", {
                    "default": True, 
                    "tooltip": "INCLUDE STEPS\n• Toggle appending step count."
                }),
                "include_schedule": ("BOOLEAN", {
                    "default": True, 
                    "tooltip": "INCLUDE SCHEDULE\n• Toggle appending scheduler info."
                }),
                "include_seed": ("BOOLEAN", {
                    "default": False, 
                    "tooltip": "INCLUDE SEED\n• Toggle appending seed value."
                }),
                "include_genre": ("BOOLEAN", {
                    "default": False, 
                    "tooltip": "INCLUDE GENRE\n• Toggle appending genre tags."
                }),
                "include_counter": ("BOOLEAN", {
                    "default": True, 
                    "tooltip": "INCLUDE COUNTER\n• Toggle appending numbered index."
                }),
                
                # Formatting
                "separator": ("STRING", {
                    "default": " - ", 
                    "tooltip": (
                        "SEPARATOR\n"
                        "• Purpose: String used to join filename parts.\n"
                        "• Options: ' - ', '_', ' ', etc.\n"
                        "\n⭐ Recommended: ' - ' is most readable."
                    )
                }),
                "genre_max_length": ("INT", {
                    "default": 40, "min": 10, "max": 100, 
                    "tooltip": "GENRE MAX LENGTH\n• Purpose: Truncate long genre strings to keep filenames manageable."
                }),
                "debug_mode": (["0 - Silent", "1 - Info"], {
                    "default": "0 - Silent",
                    "tooltip": "LOGGING VERBOSITY\n• Controls console output and string parse profiling."
                }),
                "enable_profiling": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "ENABLE PROFILING\n• Auto-on if debug >= 1."
                }),
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
                       separator=" - ", genre_max_length=40, **kwargs):
        
        debug_mode = kwargs.get("debug_mode", "0 - Silent")
        debug_level = int(debug_mode.split(" ")[0])
        enable_profiling = kwargs.get("enable_profiling", False)
        
        profiler = PerformanceProfiler(enabled=(debug_level >= 1 or enable_profiling))
        profiler.start("total_execution")

        try:
            profiler.start("config_setup")
            preset_config = {}
            if preset != "Custom" and preset in self.PRESETS:
                preset_config = self.PRESETS[preset]
                if mode_tag == "(Custom Mode)" and "mode_tag" in preset_config:
                    mode_tag = preset_config["mode_tag"]
                include_steps = preset_config.get("include_steps", include_steps)
                include_schedule = preset_config.get("include_schedule", include_schedule)
                include_seed = preset_config.get("include_seed", include_seed)
                include_genre = preset_config.get("include_genre", include_genre)
            profiler.stop("config_setup")

            profiler.start("date_formatting")
            try:
                filename_base = datetime.now().strftime(base_template)
            except ValueError as date_err:
                 logging.warning(f"[SmartFilenameBuilder] Invalid date format: {date_err}")
                 filename_base = base_template

            filename_base = self._sanitize_filename(filename_base)
            profiler.stop("date_formatting")

            profiler.start("build_components")
            parts = []

            if include_counter and counter_start > 0:
                parts.append(f"#{counter_start:04d}")

            if include_steps and steps > 0:
                parts.append(f"{steps}S")

            if include_schedule and schedule_info:
                clean_schedule = self._sanitize_filename(schedule_info)
                if clean_schedule: parts.append(clean_schedule)

            if include_seed and seed > 0:
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
            profiler.stop("build_components")

            profiler.start("assembly")
            filename_final = filename_base
            if parts:
                filename_final += separator + separator.join(parts)

            safe_separator = re.escape(separator)
            filename_final = re.sub(f'({safe_separator})+', separator, filename_final)
            filename_final = filename_final.strip(separator.strip())

            path_segments = [self._sanitize_filename(seg) for seg in project_path.replace('\\', '/').split('/') if seg]
            clean_project_path = "/".join(path_segments) + "/" if path_segments else ""
            full_path_prefix = os.path.join(clean_project_path, filename_final).replace('\\', '/')

            preview = (f"Directory: [output]/{clean_project_path}\n"
                       f"Filename: {filename_final}\n\n"
                       f"➡️ Full Prefix: {full_path_prefix}")
            profiler.stop("assembly")
            
            profiler.stop("total_execution")
            
            if debug_level >= 1:
                print("\n" + "=" * 60)
                print("📊 [SmartFilenameBuilder] ANALYTICS REPORT")
                print("=" * 60)
                print(f"📁 Path: {full_path_prefix}")
                profiler.print_report()
                print("=" * 60)

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
    """Replaces predefined tokens in a template string."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "template": ("STRING", {
                    "default": "{project}/{date} - {steps}S - {mode}", "multiline": True,
                    "tooltip": (
                        "TEMPLATE STRING\n"
                        "• Purpose: Text pattern with tokens to be replaced.\n"
                        "• Tokens: {date}, {project}, {mode}, {steps}, {seed}, {genre}.\n"
                        "\n⭐ Example: '{project}/Session_{date}_{mode}'"
                    )
                }),
            },
            "optional": {
                "project": ("STRING", {
                    "default": "MyProject",
                    "tooltip": "Token replacement for {project}."
                }),
                "mode": ("STRING", {
                    "default": "Image",
                    "tooltip": "Token replacement for {mode}."
                }),
                "steps": ("INT", {
                    "default": 0, "min": 0,
                    "tooltip": "Token replacement for {steps}."
                }),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": CONST_JS_MAX_SAFE_INTEGER,
                    "tooltip": "Token replacement for {seed}."
                }),
                "genre": ("STRING", {
                    "default": "",
                    "tooltip": "Token replacement for {genre} (auto-cleaned)."
                }),
                "custom1": ("STRING", {
                    "default": "",
                    "tooltip": "Token replacement for {custom1}."
                }),
                "custom2": ("STRING", {
                    "default": "",
                    "tooltip": "Token replacement for {custom2}."
                }),
                "date_format": ("STRING", {
                    "default": "%Y-%m-%d",
                    "tooltip": "Format for {date} token (Python strftime)."
                }),
                "time_format": ("STRING", {
                    "default": "%H-%M-%S",
                    "tooltip": "Format for {time} token (Python strftime)."
                }),
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
    """Provides a persistent, auto-incrementing counter stored in a JSON file."""

    COUNTERS_FILE = os.path.join(folder_paths.get_input_directory(), "md_filename_counters.json")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "context_key": ("STRING", {
                    "default": "default_counter", 
                    "tooltip": (
                        "CONTEXT KEY\n"
                        "• Purpose: Unique name to identify this counter in the JSON save file.\n"
                        "• Usage: Use different keys for different projects to keep counts separate.\n"
                        "\n⭐ e.g., 'Project_Alpha', 'Daily_Renders'"
                    )
                }),
                "start_value": ("INT", {
                    "default": 1, "min": 0, "max": 999999,
                    "tooltip": "START VALUE\n• Purpose: Initial value if key doesn't exist or is reset."
                }),
                "increment": ("INT", {
                    "default": 1, "min": 1, "max": 100,
                    "tooltip": "INCREMENT\n• Purpose: Amount to increase counter by on each run."
                }),
                "padding": ("INT", {
                    "default": 4, "min": 0, "max": 8,
                    "tooltip": "ZERO PADDING\n• Purpose: Number of digits to pad (e.g. 4 padding -> 0001)."
                }),
            },
            "optional": {
                "reset_counter": ("BOOLEAN", {
                    "default": False, 
                    "tooltip": "RESET COUNTER\n• Purpose: If True, forces counter back to 'start_value'."
                }),
                "prefix": ("STRING", {
                    "default": "#", 
                    "tooltip": "PREFIX\n• Purpose: String added before the number (e.g., '#0001')."
                }),
                "suffix": ("STRING", {
                    "default": "", 
                    "tooltip": "SUFFIX\n• Purpose: String added after the number."
                }),
                "trigger": ("BOOLEAN", {
                    "default": True, "label_on": "INCREMENT", "label_off": "READ ONLY",
                    "tooltip": (
                        "TRIGGER\n"
                        "• Purpose: Controls if the counter saves and advances.\n"
                        "• ON (Increment): Updates value and saves to disk.\n"
                        "• OFF (Read Only): Returns current value without changing it."
                    )
                }),
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
# == Node Registration                                                           ==
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


# ==============================================================================
# == Unit Tests (smoke — runs without ComfyUI)
# ==============================================================================

if __name__ == "__main__":
    print("\n🧪 Smoke tests: MD_SmartFilenameBuilder")
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

    _check("VERSION defined",    VERSION == "v1.6.0")
    _check("CONST CONST_JS_MAX_SAFE_INTEGER defined", CONST_JS_MAX_SAFE_INTEGER is not None)
    _check("NODE_CLASS_MAPPINGS defined",
           isinstance(NODE_CLASS_MAPPINGS, dict) and len(NODE_CLASS_MAPPINGS) > 0)
    _check("  class SmartFilenameBuilder in map", "SmartFilenameBuilder" in NODE_CLASS_MAPPINGS)
    _check("  class FilenameTokenReplacer in map", "FilenameTokenReplacer" in NODE_CLASS_MAPPINGS)
    _check("  class FilenameCounterNode in map", "FilenameCounterNode" in NODE_CLASS_MAPPINGS)

    print(f"\n  {_pass} passed, {_fail} failed")
    if _fail == 0:
        print("  🎉 All good.")
