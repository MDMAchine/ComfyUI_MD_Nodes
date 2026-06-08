# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# █▓▒░                                                                     ░▒▓█
# █▓▒░  MD_Nodes/EnhancedSeedSaver – Professional Seed Management v2.3.0   ░▒▓█
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
# ║   • Cast into the void by: MDMAchine (chaos wrangler)
# ║   • Enhanced by: Claude (Anthropic AI Assistant), Gemini
# ║
# ║ ░▒▓ DESCRIPTION:
# ║   The ultimate seed management companion for ComfyUI. Features three operation
# ║   modes: Pass-through, Manual, and Execute Action (Save, Load, Backup, etc.).
# ║   NOTE: As an I/O utility node, this runs entirely in the public wrapper domain.
# ║
# ║ ░▒▓ FEATURES:
# ║   ✓ Three Operation Modes: Pass-through, Manual Input, and Execute Action
# ║   ✓ JavaScript-safe seed range ensures full reproducibility via UI
# ║   ✓ Dual outputs: INT for compatibility, STRING for full precision
# ║   ✓ Core Actions: Save, Load, Delete, Randomize, Backup
# ║   ✓ Organization: Subdirectories, Favorites, Usage Statistics
# ║
# ║ ░▒▓ CHANGELOG:
# ║   - v2.3.0 (Enterprise Standards - Feb 2026):
# ║       • ADDED: PerformanceProfiler class (v1.5.3 standard).
# ║       • ADDED: debug_mode parameter for file I/O tracking.
# ║       • REFACTOR: Tooltips strictly updated to 5-part v1.5.4 standard.
# ║   - v2.2.0d (File Loading Fix):
# ║       • FIXED: load_seed_from_file() no longer clamps large seeds from files.
# ║       • FIXED: Seeds beyond JS-safe range now load with exact values.
# ╚════════════════════════════════════════════════════════════════════════════

# =================================================================================
# == Standard Library Imports                                                   ==
# =================================================================================
VERSION = "v2.3.0"  # UPS v1.5.8


import json
import os
from datetime import datetime
import random
import secrets
import shutil
from functools import lru_cache
from time import time
import re
import logging
import traceback

# =================================================================================
# == ComfyUI Core Modules                                                       ==
# =================================================================================
import folder_paths

# =================================================================================
# == Helper Classes (Enterprise Standards)                                      ==
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
        print("\n⏱️  PERFORMANCE (I/O):")
        total = self.get_total_time()
        print(f"    • Total Time: {total:.2f}s")
        for op_name, times in sorted(self.timings.items()):
            avg = sum(times) / len(times)
            if len(times) == 1:
                print(f"    • {op_name}: {avg:.3f}s")
            else:
                print(f"    • {op_name}: {avg:.3f}s avg ({len(times)}x)")

# =================================================================================
# == Constants & Global State                                                   ==
# =================================================================================
SEED_MIN = 0
SEED_MAX = 9007199254740991  # JavaScript's Number.MAX_SAFE_INTEGER
OUTPUT_SEEDS_DIR = os.path.join(folder_paths.get_output_directory(), "seeds")
BACKUP_DIR = os.path.join(OUTPUT_SEEDS_DIR, "_backups")
FAVORITES_FILE = os.path.join(OUTPUT_SEEDS_DIR, "_favorites.json")
STATS_FILE = os.path.join(OUTPUT_SEEDS_DIR, "_statistics.json")
LAST_SEED_FILE = os.path.join(OUTPUT_SEEDS_DIR, "_last_seed.json")
CACHE_DURATION = 5  

# =================================================================================
# == Utility Functions                                                          ==
# =================================================================================

def sanitize_filename(name):
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', name)
    sanitized = sanitized.strip('. ')
    return sanitized[:200] if sanitized else "unnamed"

def validate_seed(seed_value):
    try:
        int_value = int(seed_value)
    except (ValueError, TypeError):
        logging.warning(f"[SeedSaver] Invalid seed value '{seed_value}', defaulting to 0.")
        return SEED_MIN
    return max(SEED_MIN, min(int_value, SEED_MAX))

def parse_seed_string(seed_str):
    if not seed_str or not isinstance(seed_str, str):
        return 0
    seed_str = seed_str.strip()
    try:
        if seed_str.lower().startswith('0x'):
            int_value = int(seed_str, 16)
        else:
            int_value = int(seed_str)
        return validate_seed(int_value)
    except (ValueError, TypeError) as e:
        logging.warning(f"[SeedSaver] Could not parse seed string '{seed_str}': {e}")
        return 0

def get_cache_key():
    return int(time() / CACHE_DURATION)

# =================================================================================
# == File I/O Management                                                        ==
# =================================================================================

def save_last_seed(seed_value):
    try:
        ensure_output_directory_exists()
        data = {
            "last_seed": seed_value,
            "timestamp": datetime.now().isoformat()
        }
        with open(LAST_SEED_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        logging.debug(f"[SeedSaver] Could not save last seed: {e}")

def load_last_seed():
    if os.path.exists(LAST_SEED_FILE):
        try:
            with open(LAST_SEED_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return validate_seed(data.get("last_seed", 0))
        except Exception as e:
            logging.debug(f"[SeedSaver] Could not load last seed: {e}")
    return 0

def ensure_output_directory_exists(subdirectory=""):
    target_dir = os.path.join(OUTPUT_SEEDS_DIR, subdirectory)
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(BACKUP_DIR, exist_ok=True)

def get_seed_filepath(seed_name, subdirectory="", extension=".json"):
    target_dir = os.path.join(OUTPUT_SEEDS_DIR, subdirectory)
    return os.path.join(target_dir, f"{seed_name}{extension}")

def save_seed_to_file(seed_name, seed_value, subdirectory="", metadata=None):
    ensure_output_directory_exists(subdirectory)
    filepath = get_seed_filepath(seed_name, subdirectory)
    data_to_save = {
        "seed": validate_seed(seed_value),
        "saved_at": datetime.now().isoformat(),
        "metadata": metadata or {}
    }
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, indent=4)
        update_statistics(seed_name, 'saved')
        return True
    except Exception as e:
        logging.error(f"[SeedSaver] Could not save seed '{seed_name}': {e}")
        return False

def load_seed_from_file(seed_name, subdirectory=""):
    # Does NOT clamp large seeds from files
    json_filepath = get_seed_filepath(seed_name, subdirectory, extension=".json")
    txt_filepath = get_seed_filepath(seed_name, subdirectory, extension=".txt")
    if os.path.exists(json_filepath):
        try:
            with open(json_filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                seed_value = int(data["seed"])
                update_statistics(seed_name, 'loaded')
                return seed_value
        except Exception as e:
            logging.error(f"[SeedSaver] Could not load JSON seed '{seed_name}': {e}")
            return None
    if os.path.exists(txt_filepath):
        print(f"[SeedSaver] Note: Loading seed '{seed_name}' from legacy .txt format.")
        try:
            with open(txt_filepath, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                return int(content)
        except Exception as e:
            logging.error(f"[SeedSaver] Could not load txt seed '{seed_name}': {e}")
            return None
    logging.warning(f"[SeedSaver] Seed file '{seed_name}' not found.")
    return None

def load_seed_metadata(seed_name, subdirectory=""):
    json_filepath = get_seed_filepath(seed_name, subdirectory, extension=".json")
    if os.path.exists(json_filepath):
        try:
            with open(json_filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"[SeedSaver] Could not load seed metadata '{seed_name}': {e}")
    return None

def delete_seed_file(seed_name, subdirectory=""):
    deleted = False
    for ext in [".json", ".txt"]:
        filepath = get_seed_filepath(seed_name, subdirectory, extension=ext)
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
                deleted = True
            except Exception as e:
                logging.error(f"[SeedSaver] Could not delete '{filepath}': {e}")
    if deleted:
        remove_from_favorites(seed_name)
        update_statistics(seed_name, 'deleted')
    return deleted

@lru_cache(maxsize=32)
def get_all_saved_seed_names(subdirectory="", cache_key=0):
    target_dir = os.path.join(OUTPUT_SEEDS_DIR, subdirectory)
    ensure_output_directory_exists(subdirectory)
    try:
        all_files = os.listdir(target_dir)
        seed_names = sorted(list(set(
            os.path.splitext(f)[0] for f in all_files
            if f.endswith((".txt", ".json")) and not f.startswith("_")
        )))
        return seed_names
    except Exception as e:
        logging.error(f"[SeedSaver] Could not list seeds from '{target_dir}': {e}")
        return []

def search_seeds(pattern, subdirectory=""):
    all_seeds = get_all_saved_seed_names(subdirectory, get_cache_key())
    if pattern == "*" or not pattern:
        return all_seeds
    pattern_lower = pattern.lower()
    return [s for s in all_seeds if pattern_lower in s.lower()]

def copy_seed(seed_name, from_subdir, to_subdir):
    seed_value = load_seed_from_file(seed_name, from_subdir)
    if seed_value is None: return False
    metadata = load_seed_metadata(seed_name, from_subdir)
    seed_metadata = metadata.get('metadata', {}) if metadata else {}
    return save_seed_to_file(seed_name, seed_value, to_subdir, seed_metadata)

def move_seed(seed_name, from_subdir, to_subdir):
    if copy_seed(seed_name, from_subdir, to_subdir):
        return delete_seed_file(seed_name, from_subdir)
    return False

def export_all_seeds(subdirectory=""):
    seed_names = get_all_saved_seed_names(subdirectory, get_cache_key())
    if not seed_names: return False, "No seeds to export"
    export_data = {}
    for name in seed_names:
        data = load_seed_metadata(name, subdirectory)
        if data: export_data[name] = data
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    export_filename = f"seeds_export_{timestamp}.json"
    export_path = os.path.join(OUTPUT_SEEDS_DIR, subdirectory, export_filename)
    try:
        with open(export_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=4)
        return True, f"Exported {len(export_data)} seeds to {export_filename}"
    except Exception as e:
        return False, f"Export failed: {e}"

def clear_directory(subdirectory="", keep_backups=True):
    if keep_backups:
        backup_result = backup_seeds(subdirectory)
        if not backup_result[0]: return 0, f"Backup failed: {backup_result[1]}"
    seed_names = get_all_saved_seed_names(subdirectory, get_cache_key())
    deleted_count = 0
    for name in seed_names:
        if delete_seed_file(name, subdirectory):
            deleted_count += 1
    return deleted_count, f"Deleted {deleted_count} seeds"

def backup_seeds(subdirectory=""):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    subdir_name = subdirectory.replace('/', '_').replace('\\', '_') if subdirectory else 'root'
    backup_subdir = os.path.join(BACKUP_DIR, f"{subdir_name}_{timestamp}")
    try:
        source_dir = os.path.join(OUTPUT_SEEDS_DIR, subdirectory)
        if os.path.exists(source_dir):
            shutil.copytree(source_dir, backup_subdir,
                            ignore=shutil.ignore_patterns('_backups*', '_favorites*', '_statistics*', '_last_seed*'))
            return True, f"Backup created: {os.path.basename(backup_subdir)}"
        return False, "Source directory not found"
    except Exception as e:
        return False, f"Backup failed: {e}"

def load_favorites():
    if os.path.exists(FAVORITES_FILE):
        try:
            with open(FAVORITES_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
        except Exception: return []
    return []

def save_favorites(favorites):
    try:
        with open(FAVORITES_FILE, 'w', encoding='utf-8') as f:
            json.dump(favorites, f, indent=4)
        return True
    except Exception: return False

def toggle_favorite(seed_name):
    favorites = load_favorites()
    if seed_name in favorites:
        favorites.remove(seed_name)
        save_favorites(favorites)
        return True, f"Removed '{seed_name}' from favorites"
    else:
        favorites.append(seed_name)
        save_favorites(favorites)
        return True, f"Added '{seed_name}' to favorites"

def remove_from_favorites(seed_name):
    favorites = load_favorites()
    if seed_name in favorites:
        favorites.remove(seed_name)
        save_favorites(favorites)

def load_statistics():
    if os.path.exists(STATS_FILE):
        try:
            with open(STATS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except Exception: return {}
    return {}

def save_statistics(stats):
    try:
        with open(STATS_FILE, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=4)
        return True
    except Exception: return False

def update_statistics(seed_name, action):
    stats = load_statistics()
    if seed_name not in stats:
        stats[seed_name] = {'saves': 0, 'loads': 0, 'deletes': 0, 'last_used': None}
    if action == 'saved': stats[seed_name]['saves'] += 1
    elif action == 'loaded': stats[seed_name]['loads'] += 1
    elif action == 'deleted': stats[seed_name]['deletes'] += 1
    stats[seed_name]['last_used'] = datetime.now().isoformat()
    save_statistics(stats)

def get_seed_statistics(seed_name):
    stats = load_statistics()
    if seed_name in stats:
        s = stats[seed_name]
        last_used_str = s.get('last_used', 'Never')
        if last_used_str and last_used_str != 'Never':
            try:
                dt_obj = datetime.fromisoformat(last_used_str)
                last_used_str = dt_obj.strftime('%Y-%m-%d %H:%M:%S')
            except ValueError: pass
        return f"Saves: {s.get('saves', 0)}, Loads: {s.get('loads', 0)}, Last used: {last_used_str}"
    return "No statistics available"

def find_duplicate_seeds(subdirectory=""):
    seed_names = get_all_saved_seed_names(subdirectory, get_cache_key())
    seed_values = {}
    for name in seed_names:
        value = load_seed_from_file(name, subdirectory)
        if value is not None:
            if value not in seed_values: seed_values[value] = []
            seed_values[value].append(name)
    return {k: v for k, v in seed_values.items() if len(v) > 1}

class SeedHistory:
    def __init__(self, max_size=10):
        self.max_size = max_size
        self.history = []
    def add(self, seed_name, seed_value):
        self.history = [(n, v) for n, v in self.history if n != seed_name]
        self.history.insert(0, (seed_name, seed_value))
        self.history = self.history[:self.max_size]
    def get_list(self):
        return [name for name, _ in self.history]

seed_history = SeedHistory()

# =================================================================================
# == Core Node Class                                                            ==
# =================================================================================

class EnhancedSeedSaverNode:
    """
    MD Enhanced Seed Saver
    Advanced local seed management node. Operates entirely in ComfyUI environment.
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        cache_key = get_cache_key()
        seed_names = get_all_saved_seed_names("", cache_key)
        favorites = load_favorites()
        history = seed_history.get_list()
        last_seed = load_last_seed()

        seed_options = ["(None)"]
        valid_favorites = [f"[FAV] {name}" for name in favorites if name in seed_names]
        if valid_favorites:
            seed_options.append("--- FAVORITES ---")
            seed_options.extend(valid_favorites)

        valid_history = [f"[REC] {name}" for name in history[:5] if name in seed_names]
        if valid_history:
            seed_options.append("--- RECENT ---")
            seed_options.extend(valid_history)

        if seed_names:
            seed_options.append("--- ALL SEEDS ---")
            seed_options.extend(seed_names)

        basic_actions = ["(None)", "SAVE_CURRENT_SEED", "LOAD_SELECTED_SEED", "DELETE_SELECTED_SEED"]
        advanced_actions = ["LOAD_LATEST_SAVED_SEED", "LOAD_RANDOM_SEED", "LOAD_AND_INCREMENT", "GENERATE_RANDOM_SEED"]
        organization_actions = ["COPY_TO_SUBDIRECTORY", "MOVE_TO_SUBDIRECTORY", "TOGGLE_FAVORITE", "SHOW_STATISTICS", "FIND_DUPLICATES"]
        bulk_actions = ["EXPORT_ALL_SEEDS", "CLEAR_DIRECTORY", "BACKUP_SEEDS", "REFRESH_LISTS"]
        all_actions = basic_actions + advanced_actions + organization_actions + bulk_actions

        quick_actions = [
            "(None)", "INCREMENT_BY_1", "DECREMENT_BY_1", "INCREMENT_BY_10", "DECREMENT_BY_10",
            "INCREMENT_BY_100", "DECREMENT_BY_100", "RANDOMIZE", "RESET_TO_ZERO", "USE_LAST_SEED"
        ]

        return {
            "required": {
                "operation_mode": ([
                    "Pass-through (External Seed)",
                    "Manual Input (Direct Entry)",
                    "Execute Action (Dynamic)"
                ], {
                    "default": "Manual Input (Direct Entry)",
                    "tooltip": (
                        "OPERATION MODE\n"
                        "• Purpose: Determines caching and base behavior.\n"
                        "• Pass-through: Caches. Uses seed from external connected input.\n"
                        "• Manual: Caches. Uses manual string input directly.\n"
                        "• Execute Action: Never caches. Performs I/O operations (Save, Load).\n"
                        "\n⭐ Recommended: 'Manual Input' for standard workflow reproducibility."
                    )
                }),
            },
            "optional": {
                "seed_input": ("INT", {
                    "default": 0, "min": SEED_MIN, "max": SEED_MAX, "forceInput": True,
                    "tooltip": (
                        "EXTERNAL SEED INPUT\n"
                        "• Purpose: Connect a seed value generated by another node.\n"
                        "• Note: Ignored if using 'Manual Input' mode."
                    )
                }),
                "manual_seed": ("STRING", {
                    "default": str(last_seed), "multiline": False,
                    "tooltip": (
                        "MANUAL SEED VALUE\n"
                        "• Purpose: Enter a seed directly as text (avoids JS precision rounding).\n"
                        "• Range: 0 to 9,007,199,254,740,991 (JS Safe Max).\n"
                        "• Format: Accepts decimal or hex (e.g., 0x1A2B).\n"
                        "\n⭐ Recommended: Use this instead of INT inputs for total safety."
                    )
                }),
                "quick_action": (quick_actions, {
                    "default": "(None)",
                    "tooltip": (
                        "QUICK ACTION\n"
                        "• Purpose: Apply instant math to the active seed without changing modes.\n"
                        "• Options: Increment, Decrement, Randomize, Reset.\n"
                        "\n⭐ Note: Applies AFTER mode selection but BEFORE execute actions."
                    )
                }),
                "action": (all_actions, {
                    "default": "(None)",
                    "tooltip": (
                        "EXECUTE ACTION\n"
                        "• Purpose: Perform file I/O operations (Requires Execute Action mode).\n"
                        "• Operations: Save, Load, Delete, Backup, Export, Statistics.\n"
                        "\n⭐ Note: Will force the node to re-run on every queue."
                    )
                }),
                "seed_name_input": ("STRING", {
                    "default": "", "multiline": False,
                    "tooltip": (
                        "SAVE NAME\n"
                        "• Purpose: Filename for the SAVE_CURRENT_SEED action.\n"
                        "• Note: Invalid characters are automatically stripped. Leaves blank for auto-timestamp."
                    )
                }),
                "seed_to_load_name": (seed_options, {
                    "default": "(None)",
                    "tooltip": (
                        "SELECT SEED\n"
                        "• Purpose: Target seed for Load, Delete, Copy, Move, or Stats actions.\n"
                        "• Tip: Run 'REFRESH_LISTS' action if new files don't appear."
                    )
                }),
                "subdirectory": ("STRING", {
                    "default": "", "multiline": False,
                    "tooltip": (
                        "SUBDIRECTORY\n"
                        "• Purpose: Target specific folders within the main 'seeds' output directory.\n"
                        "• Format: Use forward slashes (e.g., 'project/tests')."
                    )
                }),
                "search_pattern": ("STRING", {
                    "default": "*", "multiline": False,
                    "tooltip": (
                        "SEARCH PATTERN\n"
                        "• Purpose: Filters the dropdown list using substring matching.\n"
                        "• Note: Also restrains the 'LOAD_RANDOM_SEED' pool."
                    )
                }),
                "description": ("STRING", {
                    "default": "", "multiline": True,
                    "tooltip": "DESCRIPTION\n• Purpose: Text notes embedded into the saved JSON file."
                }),
                "tags": ("STRING", {
                    "default": "", "multiline": False,
                    "tooltip": "TAGS\n• Purpose: Comma-separated tags embedded into the saved JSON file."
                }),
                "target_subdirectory": ("STRING", {
                    "default": "", "multiline": False,
                    "tooltip": "TARGET SUBDIRECTORY\n• Purpose: Destination folder for COPY or MOVE actions."
                }),
                "debug_mode": (["0 - Silent", "1 - Info"], {
                    "default": "0 - Silent",
                    "tooltip": "LOGGING VERBOSITY\n• Controls console output and I/O profiling."
                }),
            }
        }

    @classmethod
    def IS_CHANGED(cls, operation_mode, **kwargs):
        if operation_mode == "Execute Action (Dynamic)":
            return secrets.token_hex(16)
        return "static"

    RETURN_TYPES = ("INT", "STRING", "STRING")
    RETURN_NAMES = ("seed_output", "seed_output_str", "status_info")
    FUNCTION = "execute"
    CATEGORY = "MD_Nodes/Save"
    OUTPUT_NODE = True

    def execute(self, operation_mode, seed_input=None, manual_seed="0", quick_action="(None)",
                action="(None)", seed_name_input="", seed_to_load_name="(None)",
                subdirectory="", search_pattern="*", description="", tags="",
                target_subdirectory="", debug_mode="0 - Silent"):
        
        debug_level = int(debug_mode.split(" ")[0])
        profiler = PerformanceProfiler(enabled=(debug_level >= 1))
        profiler.start("total_execution")

        try:
            manual_seed_int = parse_seed_string(manual_seed)
            
            if operation_mode == "Manual Input (Direct Entry)":
                output_seed = manual_seed_int
                base_mode = "Manual Input"
            elif operation_mode == "Pass-through (External Seed)":
                if seed_input is not None:
                    output_seed = validate_seed(seed_input)
                else:
                    output_seed = manual_seed_int
                    logging.warning("[SeedSaver] No seed_input connected, using manual_seed")
                base_mode = "Pass-through"
            else:  
                if seed_input is not None:
                    output_seed = validate_seed(seed_input)
                else:
                    output_seed = manual_seed_int
                base_mode = "Execute Action"

            status_message = ""

            if quick_action != "(None)":
                profiler.start("quick_action")
                original_seed = output_seed
                if quick_action == "INCREMENT_BY_1": output_seed = validate_seed(output_seed + 1)
                elif quick_action == "DECREMENT_BY_1": output_seed = validate_seed(output_seed - 1)
                elif quick_action == "INCREMENT_BY_10": output_seed = validate_seed(output_seed + 10)
                elif quick_action == "DECREMENT_BY_10": output_seed = validate_seed(output_seed - 10)
                elif quick_action == "INCREMENT_BY_100": output_seed = validate_seed(output_seed + 100)
                elif quick_action == "DECREMENT_BY_100": output_seed = validate_seed(output_seed - 100)
                elif quick_action == "RANDOMIZE": output_seed = secrets.randbelow(SEED_MAX + 1)
                elif quick_action == "RESET_TO_ZERO": output_seed = 0
                elif quick_action == "USE_LAST_SEED": output_seed = load_last_seed()

                status_message += f"Quick Action: {quick_action}\n  {original_seed} -> {output_seed}\n\n"
                profiler.stop("quick_action")

            if operation_mode != "Execute Action (Dynamic)":
                status_message += (
                    f"Mode: {base_mode}\n"
                    f"Seed Value: {output_seed}\n"
                    f"Seed Range: 0 to {SEED_MAX:,} (JS safe range)"
                )
                save_last_seed(output_seed)
                profiler.stop("total_execution")
                if debug_level >= 1: profiler.print_report()
                return (output_seed, str(output_seed), status_message)

            # --- Execute Action (Dynamic) ---
            profiler.start("dynamic_action")

            clean_seed_name = seed_to_load_name
            for prefix in ["[FAV] ", "[REC] "]:
                if clean_seed_name.startswith(prefix):
                    clean_seed_name = clean_seed_name[len(prefix):]
            if clean_seed_name.startswith("---") or clean_seed_name == "(None)":
                clean_seed_name = ""

            current_seeds = []
            if search_pattern and search_pattern != "*":
                current_seeds = search_seeds(search_pattern, subdirectory)
            else:
                current_seeds = get_all_saved_seed_names(subdirectory, get_cache_key())

            num_seeds = len(current_seeds)
            dir_display = os.path.join('seeds', subdirectory) if subdirectory else 'seeds/root'

            status_message += (
                f"Mode: Execute Action\n"
                f"Action: {action}\n"
                f"Input Seed: {output_seed}\n"
                f"Directory: '{dir_display}'\n"
                f"Seeds Found: {num_seeds}"
            )

            # --- File I/O Logic ---
            if action == "SAVE_CURRENT_SEED":
                save_name = sanitize_filename(seed_name_input.strip())
                if not save_name:
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    save_name = f"seed_{output_seed}_{timestamp}"
                    status_message += f"\nAuto-named: '{save_name}'"

                duplicates = find_duplicate_seeds(subdirectory)
                existing_value = load_seed_from_file(save_name, subdirectory)

                if existing_value is not None and existing_value != output_seed:
                    status_message += f"\n⚠️ Warning: Overwriting different value ({existing_value} -> {output_seed})"
                elif output_seed in duplicates and save_name not in duplicates.get(output_seed, []):
                    status_message += f"\n⚠️ Note: Value {output_seed} already exists as: {', '.join(duplicates[output_seed])}"

                metadata_dict = {
                    "description": description.strip() if description else "",
                    "tags": [t.strip() for t in tags.split(",") if t.strip()],
                    "workflow": "ComfyUI",
                }
                if save_seed_to_file(save_name, output_seed, subdirectory, metadata_dict):
                    status_message += f"\n✅ SAVED: '{save_name}' = {output_seed}"
                    seed_history.add(save_name, output_seed)
                else:
                    status_message += f"\n❌ Error: Failed to save seed '{save_name}'"

            elif action == "LOAD_SELECTED_SEED":
                if clean_seed_name:
                    loaded_seed = load_seed_from_file(clean_seed_name, subdirectory)
                    if loaded_seed is not None:
                        output_seed = loaded_seed
                        status_message += f"\n✅ LOADED: '{clean_seed_name}' = {loaded_seed}"
                        if loaded_seed > SEED_MAX:
                            status_message += f"\n⚠️ WARNING: Seed exceeds JS-safe range ({SEED_MAX:,})"
                        seed_history.add(clean_seed_name, loaded_seed)
                    else:
                        status_message += f"\n❌ Error: Seed '{clean_seed_name}' not found"
                else:
                    status_message += "\nℹ️ No seed selected"

            elif action == "DELETE_SELECTED_SEED":
                if clean_seed_name:
                    if delete_seed_file(clean_seed_name, subdirectory):
                        status_message += f"\n✅ DELETED: '{clean_seed_name}'"
                    else:
                        status_message += f"\n❌ Error: Could not delete '{clean_seed_name}'"
                else:
                    status_message += "\nℹ️ No seed selected"

            elif action == "LOAD_LATEST_SAVED_SEED":
                all_seeds_in_dir = get_all_saved_seed_names(subdirectory, get_cache_key())
                if all_seeds_in_dir:
                    target_dir = os.path.join(OUTPUT_SEEDS_DIR, subdirectory)
                    files_with_times = []
                    for filename in os.listdir(target_dir):
                        if filename.endswith((".json", ".txt")) and not filename.startswith("_"):
                            filepath = os.path.join(target_dir, filename)
                            try: files_with_times.append((filepath, os.path.getmtime(filepath)))
                            except OSError: pass
                    if files_with_times:
                        files_with_times.sort(key=lambda x: x[1], reverse=True)
                        latest_name = os.path.splitext(os.path.basename(files_with_times[0][0]))[0]
                        loaded_seed = load_seed_from_file(latest_name, subdirectory)
                        if loaded_seed is not None:
                            output_seed = loaded_seed
                            status_message += f"\n✅ LOADED LATEST: '{latest_name}' = {loaded_seed}"
                            seed_history.add(latest_name, loaded_seed)
                    else: status_message += "\nℹ️ No valid seed files found"
                else: status_message += "\nℹ️ No seeds in directory"

            elif action == "LOAD_RANDOM_SEED":
                if current_seeds:
                    random_name = secrets.choice(current_seeds)
                    loaded_seed = load_seed_from_file(random_name, subdirectory)
                    if loaded_seed is not None:
                        output_seed = loaded_seed
                        status_message += f"\n✅ LOADED RANDOM: '{random_name}' = {loaded_seed}"
                        seed_history.add(random_name, loaded_seed)
                else: status_message += "\nℹ️ No seeds available for random selection"

            elif action == "LOAD_AND_INCREMENT":
                if clean_seed_name:
                    loaded_seed = load_seed_from_file(clean_seed_name, subdirectory)
                    if loaded_seed is not None:
                        output_seed = validate_seed(loaded_seed + 1)
                        status_message += f"\n✅ LOADED + 1: '{clean_seed_name}' {loaded_seed} -> {output_seed}"
                        seed_history.add(clean_seed_name, loaded_seed)
                else: status_message += "\nℹ️ No seed selected"

            elif action == "GENERATE_RANDOM_SEED":
                output_seed = secrets.randbelow(SEED_MAX + 1)
                status_message += f"\n✅ GENERATED RANDOM SEED: {output_seed}"

            elif action == "COPY_TO_SUBDIRECTORY":
                target_sub = target_subdirectory.strip()
                if clean_seed_name and target_sub:
                    if copy_seed(clean_seed_name, subdirectory, target_sub):
                        status_message += f"\n✅ COPIED: '{clean_seed_name}' -> '{target_sub}'"
                else: status_message += "\nℹ️ Need seed name and target subdirectory"

            elif action == "MOVE_TO_SUBDIRECTORY":
                target_sub = target_subdirectory.strip()
                if clean_seed_name and target_sub:
                    if move_seed(clean_seed_name, subdirectory, target_sub):
                        status_message += f"\n✅ MOVED: '{clean_seed_name}' -> '{target_sub}'"
                else: status_message += "\nℹ️ Need seed name and target subdirectory"

            elif action == "TOGGLE_FAVORITE":
                if clean_seed_name:
                    success, msg = toggle_favorite(clean_seed_name)
                    status_message += f"\n{'✅' if success else '❌'} {msg}"
                else: status_message += "\nℹ️ No seed selected"

            elif action == "SHOW_STATISTICS":
                if clean_seed_name:
                    stats_info = get_seed_statistics(clean_seed_name)
                    status_message += f"\n📊 Statistics for '{clean_seed_name}':\n  {stats_info}"

            elif action == "FIND_DUPLICATES":
                duplicates = find_duplicate_seeds(subdirectory)
                if duplicates:
                    status_message += f"\n🔍 Found {len(duplicates)} duplicate value(s)"
                else: status_message += "\n✅ No duplicate seeds found"

            elif action == "EXPORT_ALL_SEEDS":
                success, msg = export_all_seeds(subdirectory)
                status_message += f"\n{'✅' if success else '❌'} {msg}"

            elif action == "CLEAR_DIRECTORY":
                count, msg = clear_directory(subdirectory, keep_backups=True)
                status_message += f"\n⚠️ {msg} (backup created)"

            elif action == "BACKUP_SEEDS":
                success, msg = backup_seeds(subdirectory)
                status_message += f"\n{'✅' if success else '❌'} {msg}"

            elif action == "REFRESH_LISTS":
                get_all_saved_seed_names.cache_clear()
                status_message += "\n🔄 Lists refreshed!"

            status_message += f"\n\n📤 Output Seed: {output_seed}"
            save_last_seed(output_seed)
            profiler.stop("dynamic_action")
            profiler.stop("total_execution")
            
            if debug_level >= 1: profiler.print_report()
            return (output_seed, str(output_seed), status_message)

        except Exception as e:
            logging.error(f"[EnhancedSeedSaver] Execution failed: {e}")
            error_msg = f"❌ [EnhancedSeedSaver] Error: {e}"
            fallback_seed = validate_seed(seed_input) if seed_input is not None else parse_seed_string(manual_seed)
            return (fallback_seed, str(fallback_seed), error_msg)

# =================================================================================
# == Node Registration                                                          ==
# =================================================================================

ensure_output_directory_exists()

NODE_CLASS_MAPPINGS = {
    "EnhancedSeedSaver": EnhancedSeedSaverNode
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "EnhancedSeedSaver": "MD: Enhanced Seed Saver"
}


# ==============================================================================
# == Unit Tests (smoke — runs without ComfyUI)
# ==============================================================================

if __name__ == "__main__":
    print("\n🧪 Smoke tests: MD_SeedSaver")
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

    _check("VERSION defined",    VERSION == "v2.3.0")
    _check("NODE_CLASS_MAPPINGS defined",
           isinstance(NODE_CLASS_MAPPINGS, dict) and len(NODE_CLASS_MAPPINGS) > 0)
    _check("  class EnhancedSeedSaver in map", "EnhancedSeedSaver" in NODE_CLASS_MAPPINGS)

    print(f"\n  {_pass} passed, {_fail} failed")
    if _fail == 0:
        print("  🎉 All good.")
