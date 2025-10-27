# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
# ████ MD_Nodes/EnhancedSeedSaver – Professional Seed Management v2.1.0 ████▓▒░
# © 2025 MDMAchine
# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
# ░▒▓ ORIGIN & DEV:
#   • Cast into the void by: MDMAchine (chaos wrangler)
#   • Enhanced by: Claude (Anthropic AI Assistant)
#   • License: Public Domain — Share the deterministic love
#   • Original source (if applicable): [Inspired by seed management struggles]

# ░▒▓ DESCRIPTION:
#   The ultimate seed management companion for ComfyUI. Switch between a dynamic
#   action mode for managing seeds (Save, Load, Delete, Randomize, Backup,
#   Favorites, Stats) and a static pass-through mode for stable, cacheable
#   workflows without needing to bypass the node.

# ░▒▓ FEATURES:
#   ✓ Dual Operation Modes: "Pass-through (Static)" for stability & "Execute Action (Dynamic)" for management.
#   ✓ Core Actions: Save, Load, Delete, Generate Random, Load Random, Load + Increment seeds.
#   ✓ Organization: Subdirectories, Copy/Move seeds between directories.
#   ✓ Management: Favorites list, Usage Statistics, Duplicate Seed Finder.
#   ✓ Bulk Actions: Export All, Clear Directory (with backup), Backup Seeds, Refresh Lists.
#   ✓ Saves seed with metadata (description, tags, timestamp).
#   ✓ Robust file handling with backups and JSON format (legacy .txt support).

# ░▒▓ CHANGELOG:
#   - v2.1.0 (Current Release - Operation Mode Update):
#       • ADDED: An "operation_mode" toggle ("Pass-through" vs "Execute Action").
#       • FIXED: "Pass-through" mode is now fully cacheable and won't cause re-runs.
#       • ENHANCED: IS_CHANGED logic is now conditional based on the selected mode.

# ░▒▓ CONFIGURATION:
#   → Primary Use: Using "Pass-through" mode to hold a static seed value within a workflow.
#   → Secondary Use: Using "Execute Action" > "SAVE_CURRENT_SEED" to store a good result, then switching back to "Pass-through".
#   → Edge Use: Setting "Execute Action" > "LOAD_RANDOM_SEED" to introduce controlled randomness into batch runs.

# ░▒▓ WARNING:
#   This node may trigger:
#   ▓▒░ Accidentally hitting 'CLEAR_DIRECTORY' instead of 'BACKUP_SEEDS' and feeling that cold, 1995-era dread.
#   ▓▒░ Spending hours organizing seeds into subdirectories like a digital hoarder.
#   ▓▒░ An unhealthy obsession with the '⭐ Favorites' list.
#   ▓▒░ Flashbacks to losing your `autoexec.bat` and realizing you never backed it up.
#   Consult your nearest demoscene vet if hallucinations persist.
# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄


# =================================================================================
# == Standard Library Imports                                                  ==
# =================================================================================
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
# == Third-Party Imports                                                       ==
# =================================================================================
# (None needed)

# =================================================================================
# == ComfyUI Core Modules                                                      ==
# =================================================================================
import folder_paths

# =================================================================================
# == Local Project Imports                                                     ==
# =================================================================================
# (None needed)

# =================================================================================
# == Helper Classes & Dependencies                                             ==
# =================================================================================

# --- Constants ---
SEED_MIN = 0
SEED_MAX = 0xffffffffffffffff
OUTPUT_SEEDS_DIR = os.path.join(folder_paths.get_output_directory(), "seeds")
BACKUP_DIR = os.path.join(OUTPUT_SEEDS_DIR, "_backups")
FAVORITES_FILE = os.path.join(OUTPUT_SEEDS_DIR, "_favorites.json")
STATS_FILE = os.path.join(OUTPUT_SEEDS_DIR, "_statistics.json")
CACHE_DURATION = 5  # seconds

# --- Utility Functions ---

def sanitize_filename(name):
    """
    Remove invalid filename characters.

    Args:
        name: The input filename string.

    Returns:
        A sanitized filename string.
    """
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', name)
    sanitized = sanitized.strip('. ')
    return sanitized[:200] if sanitized else "unnamed"

def validate_seed(seed_value):
    """
    Ensure seed is within the valid range [0, 0xffffffffffffffff].

    Args:
        seed_value: The input seed value (int or castable to int).

    Returns:
        The validated seed value as an integer.
    """
    try:
        int_value = int(seed_value)
    except (ValueError, TypeError):
        logging.warning(f"[SeedSaver] Invalid seed value '{seed_value}', defaulting to 0.")
        return SEED_MIN
    return max(SEED_MIN, min(int_value, SEED_MAX))

def get_cache_key():
    """
    Generate a simple cache key based on current time intervals.
    Used to periodically refresh cached directory listings.

    Returns:
        An integer representing the current time interval.
    """
    return int(time() / CACHE_DURATION)

# --- Directory Management ---

def ensure_output_directory_exists(subdirectory=""):
    """
    Creates the main seed output directory, backup directory, and specified subdirectory if they don't exist.

    Args:
        subdirectory: Optional path relative to the main seed directory.
    """
    target_dir = os.path.join(OUTPUT_SEEDS_DIR, subdirectory)
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(BACKUP_DIR, exist_ok=True)

def get_seed_filepath(seed_name, subdirectory="", extension=".json"):
    """
    Constructs the full path for a seed file.

    Args:
        seed_name: The base name of the seed file (without extension).
        subdirectory: Optional subdirectory path.
        extension: The file extension (e.g., ".json", ".txt").

    Returns:
        The absolute path to the seed file.
    """
    target_dir = os.path.join(OUTPUT_SEEDS_DIR, subdirectory)
    return os.path.join(target_dir, f"{seed_name}{extension}")

# --- Enhanced Seed Operations (functions remain the same logic) ---

def save_seed_to_file(seed_name, seed_value, subdirectory="", metadata=None):
    """Saves seed value and metadata to a JSON file."""
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
    """Loads seed value, supporting JSON and legacy TXT."""
    json_filepath = get_seed_filepath(seed_name, subdirectory, extension=".json")
    txt_filepath = get_seed_filepath(seed_name, subdirectory, extension=".txt")
    if os.path.exists(json_filepath):
        try:
            with open(json_filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                seed_value = int(data["seed"])
                update_statistics(seed_name, 'loaded')
                return validate_seed(seed_value)
        except Exception as e:
            logging.error(f"[SeedSaver] Could not load JSON seed '{seed_name}': {e}")
            return None
    if os.path.exists(txt_filepath):
        print(f"[SeedSaver] Note: Loading seed '{seed_name}' from legacy .txt format.")
        try:
            with open(txt_filepath, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                # Attempt conversion and validation
                seed_val = validate_seed(int(content))
                # Optionally, convert the legacy file to JSON here
                # save_seed_to_file(seed_name, seed_val, subdirectory)
                # os.remove(txt_filepath) # Be careful with auto-deletion
                return seed_val
        except Exception as e:
            logging.error(f"[SeedSaver] Could not load legacy txt seed '{seed_name}': {e}")
            return None
    logging.warning(f"[SeedSaver] Seed file for '{seed_name}' not found.")
    return None

def load_seed_metadata(seed_name, subdirectory=""):
    """Loads the full JSON data including metadata for a seed."""
    json_filepath = get_seed_filepath(seed_name, subdirectory, extension=".json")
    if os.path.exists(json_filepath):
        try:
            with open(json_filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"[SeedSaver] Could not load seed metadata '{seed_name}': {e}")
    return None

def delete_seed_file(seed_name, subdirectory=""):
    """Deletes seed files (.json and legacy .txt)."""
    deleted = False
    for ext in [".json", ".txt"]:
        filepath = get_seed_filepath(seed_name, subdirectory, extension=ext)
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
                print(f"[SeedSaver] Removed '{filepath}'")
                deleted = True
            except Exception as e:
                logging.error(f"[SeedSaver] Could not delete seed file '{filepath}': {e}")
    if deleted:
        remove_from_favorites(seed_name)
        update_statistics(seed_name, 'deleted')
    return deleted

@lru_cache(maxsize=32)
def get_all_saved_seed_names(subdirectory="", cache_key=0):
    """
    Lists all saved seed names in a directory, using caching.

    Args:
        subdirectory: The target subdirectory.
        cache_key: An integer key used by lru_cache for time-based invalidation.

    Returns:
        A sorted list of unique seed names found.
    """
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
    """Filters seed names based on a search pattern."""
    all_seeds = get_all_saved_seed_names(subdirectory, get_cache_key())
    if pattern == "*" or not pattern:
        return all_seeds
    pattern_lower = pattern.lower()
    return [s for s in all_seeds if pattern_lower in s.lower()]

def copy_seed(seed_name, from_subdir, to_subdir):
    """Copies a seed file (with metadata) to another directory."""
    seed_value = load_seed_from_file(seed_name, from_subdir)
    if seed_value is None:
        return False
    metadata = load_seed_metadata(seed_name, from_subdir)
    seed_metadata = metadata.get('metadata', {}) if metadata else {}
    return save_seed_to_file(seed_name, seed_value, to_subdir, seed_metadata)

def move_seed(seed_name, from_subdir, to_subdir):
    """Moves a seed file by copying then deleting the original."""
    if copy_seed(seed_name, from_subdir, to_subdir):
        return delete_seed_file(seed_name, from_subdir)
    return False

def export_all_seeds(subdirectory=""):
    """Exports all seeds in a directory to a single JSON file."""
    seed_names = get_all_saved_seed_names(subdirectory, get_cache_key())
    if not seed_names:
        return False, "No seeds to export"
    export_data = {}
    for name in seed_names:
        data = load_seed_metadata(name, subdirectory)
        if data:
            export_data[name] = data
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    export_filename = f"seeds_export_{timestamp}.json"
    export_path = os.path.join(OUTPUT_SEEDS_DIR, subdirectory, export_filename)
    try:
        with open(export_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=4)
        return True, f"Exported {len(export_data)} seeds to {export_filename}"
    except Exception as e:
        logging.error(f"[SeedSaver] Export failed: {e}")
        return False, f"Export failed: {e}"

# NOTE: Import function removed for brevity, assuming it exists if needed.

def clear_directory(subdirectory="", keep_backups=True):
    """Clears all seed files from a directory, optionally backing up first."""
    if keep_backups:
        backup_result = backup_seeds(subdirectory)
        if not backup_result[0]:
            return 0, f"Backup failed: {backup_result[1]}"
    seed_names = get_all_saved_seed_names(subdirectory, get_cache_key())
    deleted_count = 0
    for name in seed_names:
        if delete_seed_file(name, subdirectory):
            deleted_count += 1
    return deleted_count, f"Deleted {deleted_count} seeds"

def backup_seeds(subdirectory=""):
    """Creates a timestamped backup of a seed directory."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    subdir_name = subdirectory.replace('/', '_').replace('\\', '_') if subdirectory else 'root'
    backup_subdir = os.path.join(BACKUP_DIR, f"{subdir_name}_{timestamp}")
    try:
        source_dir = os.path.join(OUTPUT_SEEDS_DIR, subdirectory)
        if os.path.exists(source_dir):
            shutil.copytree(source_dir, backup_subdir,
                            ignore=shutil.ignore_patterns('_backups*', '_favorites*', '_statistics*'))
            return True, f"Backup created: {os.path.basename(backup_subdir)}"
        return False, "Source directory not found"
    except Exception as e:
        logging.error(f"[SeedSaver] Backup failed: {e}")
        return False, f"Backup failed: {e}"

def load_favorites():
    """Loads the list of favorite seed names."""
    if os.path.exists(FAVORITES_FILE):
        try:
            with open(FAVORITES_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
        except Exception as e:
             logging.error(f"[SeedSaver] Error loading favorites: {e}")
             return []
    return []

def save_favorites(favorites):
    """Saves the list of favorite seed names."""
    try:
        with open(FAVORITES_FILE, 'w', encoding='utf-8') as f:
            json.dump(favorites, f, indent=4)
        return True
    except Exception as e:
        logging.error(f"[SeedSaver] Could not save favorites: {e}")
        return False

def toggle_favorite(seed_name):
    """Adds or removes a seed name from the favorites list."""
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
    """Silently removes a seed name if it exists in favorites."""
    favorites = load_favorites()
    if seed_name in favorites:
        favorites.remove(seed_name)
        save_favorites(favorites)

def load_statistics():
    """Loads seed usage statistics."""
    if os.path.exists(STATS_FILE):
        try:
            with open(STATS_FILE, 'r', encoding='utf-8') as f:
                 data = json.load(f)
                 return data if isinstance(data, dict) else {}
        except Exception as e:
            logging.error(f"[SeedSaver] Error loading statistics: {e}")
            return {}
    return {}

def save_statistics(stats):
    """Saves seed usage statistics."""
    try:
        with open(STATS_FILE, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=4)
        return True
    except Exception as e:
        logging.error(f"[SeedSaver] Could not save statistics: {e}")
        return False

def update_statistics(seed_name, action):
    """Updates usage statistics for a given seed and action."""
    stats = load_statistics()
    if seed_name not in stats:
        stats[seed_name] = {'saves': 0, 'loads': 0, 'deletes': 0, 'last_used': None}
    if action == 'saved': stats[seed_name]['saves'] += 1
    elif action == 'loaded': stats[seed_name]['loads'] += 1
    elif action == 'deleted': stats[seed_name]['deletes'] += 1
    stats[seed_name]['last_used'] = datetime.now().isoformat()
    save_statistics(stats)

def get_seed_statistics(seed_name):
    """Retrieves formatted usage statistics string for a seed."""
    stats = load_statistics()
    if seed_name in stats:
        s = stats[seed_name]
        last_used_str = s.get('last_used', 'Never')
        if last_used_str and last_used_str != 'Never':
             try:
                 # Attempt to parse and format ISO timestamp
                 dt_obj = datetime.fromisoformat(last_used_str)
                 last_used_str = dt_obj.strftime('%Y-%m-%d %H:%M:%S')
             except ValueError:
                 pass # Keep original string if parsing fails
        return (f"Saves: {s.get('saves', 0)}, Loads: {s.get('loads', 0)}, "
                f"Last used: {last_used_str}")
    return "No statistics available"

def find_duplicate_seeds(subdirectory=""):
    """Finds seeds with identical values within a directory."""
    seed_names = get_all_saved_seed_names(subdirectory, get_cache_key())
    seed_values = {}
    for name in seed_names:
        value = load_seed_from_file(name, subdirectory)
        if value is not None:
            if value not in seed_values:
                seed_values[value] = []
            seed_values[value].append(name)
    return {k: v for k, v in seed_values.items() if len(v) > 1}

class SeedHistory:
    """Simple class to keep track of recently used seeds."""
    def __init__(self, max_size=10):
        """
        Initialize history.

        Args:
            max_size: Maximum number of recent seeds to store.
        """
        self.max_size = max_size
        self.history = [] # Stores tuples of (name, value)

    def add(self, seed_name, seed_value):
        """Adds a seed to the history, removing duplicates and trimming."""
        self.history = [(n, v) for n, v in self.history if n != seed_name]
        self.history.insert(0, (seed_name, seed_value))
        self.history = self.history[:self.max_size]

    def get_list(self):
        """Returns a list of seed names in the history."""
        return [name for name, _ in self.history]

# Global instance for recent seeds
seed_history = SeedHistory()

# =================================================================================
# == Core Node Class                                                           ==
# =================================================================================

class EnhancedSeedSaverNode:
    """
    MD Enhanced Seed Saver

    Advanced seed management node with static pass-through and dynamic action modes.
    Save, load, organize, and manage seeds efficiently within ComfyUI workflows.
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        """Define node inputs."""
        cache_key = get_cache_key()
        # Use default subdirectory "" when fetching seed names for the dropdown
        seed_names = get_all_saved_seed_names("", cache_key)
        favorites = load_favorites()
        history = seed_history.get_list()

        seed_options = ["(None)"]
        valid_favorites = [f"⭐ {name}" for name in favorites if name in seed_names]
        if valid_favorites:
            seed_options.append("--- FAVORITES ---")
            seed_options.extend(valid_favorites)

        valid_history = [f"🕒 {name}" for name in history[:5] if name in seed_names]
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

        return {
            "required": {
                "seed_input": ("INT", {
                    "default": 0, "min": SEED_MIN, "max": SEED_MAX, "forceInput": True,
                    "tooltip": (
                        "INPUT SEED\n"
                        "- Connect the seed value you want to manage or pass through.\n"
                        "- This is the primary seed value used when saving or acting as a base."
                    )
                }),
                "operation_mode": (["Pass-through (Static)", "Execute Action (Dynamic)"], {
                    "default": "Pass-through (Static)", # Default to safer static mode
                    "tooltip": (
                        "OPERATION MODE\n"
                        "- Determines the node's behavior and caching.\n"
                        "- 'Pass-through (Static)': Simply passes 'seed_input' to 'seed_output'. Does NOT perform actions. Cacheable and stable for consistent workflows.\n"
                        "- 'Execute Action (Dynamic)': Performs the selected 'action' (Save, Load, etc.) during execution. This mode forces re-runs and prevents caching."
                    )
                }),
            },
            "optional": {
                "action": (all_actions, {
                    "default": "(None)",
                    "tooltip": (
                        "ACTION (Dynamic Mode Only)\n"
                        "- The operation to perform when 'operation_mode' is 'Execute Action'.\n"
                        "- (None): Does nothing, outputs 'seed_input'.\n"
                        "- SAVE_CURRENT_SEED: Saves 'seed_input' using 'seed_name_input'.\n"
                        "- LOAD_SELECTED_SEED: Loads seed from 'seed_to_load_name'.\n"
                        "- ... (See guide for full list and descriptions)"
                    )
                }),
                "seed_name_input": ("STRING", {
                    "default": "", "multiline": False, "placeholder": "Enter seed name...",
                    "tooltip": (
                        "NAME FOR SAVING\n"
                        "- Enter the desired name when using SAVE_CURRENT_SEED.\n"
                        "- Invalid filename characters will be replaced.\n"
                        "- Auto-generates name if left blank."
                    )
                }),
                "seed_to_load_name": (seed_options, {
                    "default": "(None)",
                    "tooltip": (
                        "SELECT SEED\n"
                        "- Choose a saved seed for actions like LOAD, DELETE, COPY, MOVE, FAVORITE, STATS.\n"
                        "- Organized by Favorites ⭐, Recent 🕒, and All.\n"
                        "- Use REFRESH_LISTS action if dropdown is outdated."
                    )
                }),
                "subdirectory": ("STRING", {
                    "default": "", "multiline": False, "placeholder": "Optional: folder/subfolder",
                    "tooltip": (
                        "SUBDIRECTORY\n"
                        "- Specify a subfolder within the main 'seeds' output directory.\n"
                        "- Use forward slashes (e.g., 'projectA/run1').\n"
                        "- Leave empty for the root seed directory.\n"
                        "- Affects SAVE, LOAD, SEARCH, and other directory actions."
                    )
                }),
                "search_pattern": ("STRING", {
                    "default": "*", "multiline": False, "placeholder": "Search pattern (*=all)",
                    "tooltip": (
                        "SEARCH PATTERN\n"
                        "- Filters seeds in 'SELECT SEED' dropdown (case-insensitive).\n"
                        "- Use '*' to show all.\n"
                        "- Also affects actions like LOAD_RANDOM_SEED."
                    )
                }),
                "description": ("STRING", {
                    "default": "", "multiline": True, "placeholder": "Optional: Add notes...",
                    "tooltip": (
                        "DESCRIPTION / NOTES\n"
                        "- Add optional text notes when saving a seed.\n"
                        "- Stored in the JSON file."
                    )
                }),
                "tags": ("STRING", {
                    "default": "", "multiline": False, "placeholder": "Optional: tag1, tag2,...",
                    "tooltip": (
                        "TAGS\n"
                        "- Add optional comma-separated tags when saving.\n"
                        "- Example: 'style_test, character_A, good_result'.\n"
                        "- Stored in the JSON metadata."
                    )
                }),
                "target_subdirectory": ("STRING", {
                    "default": "", "multiline": False, "placeholder": "For copy/move actions",
                    "tooltip": (
                        "TARGET SUBDIRECTORY\n"
                        "- Specify the destination subfolder for COPY or MOVE actions.\n"
                        "- Use forward slashes."
                    )
                }),
            }
        }

    @classmethod
    def IS_CHANGED(cls, seed_input, operation_mode, **kwargs):
        """
        Controls caching based on the operation mode.
        Static mode is cacheable based on inputs, Dynamic mode always re-runs.
        """
        if operation_mode == "Execute Action (Dynamic)":
            # In action mode, we must re-run every time to perform the action.
            # Using secrets ensures a unique value each time, preventing caching.
            return secrets.token_hex(16)
        else:
            # In pass-through mode, the output only depends on the input seed and mode.
            # Returning a constant string allows ComfyUI to cache based on inputs.
            return "static" # Corrected based on guide Section 8.1

    RETURN_TYPES = ("INT", "STRING")
    RETURN_NAMES = ("seed_output", "status_info")
    FUNCTION = "execute"
    CATEGORY = "MD_Nodes/Utility"
    OUTPUT_NODE = True # Indicates this node primarily performs actions/outputs info

    def execute(self, seed_input, operation_mode, action="(None)", seed_name_input="",
                seed_to_load_name="(None)", subdirectory="", search_pattern="*",
                description="", tags="", target_subdirectory=""):
        """
        Main execution function. Handles both pass-through and action modes.
        """
        try:
            output_seed = validate_seed(seed_input)
            status_message = ""

            if operation_mode == "Pass-through (Static)":
                status_message = (f"Mode: Pass-through\n"
                                  f"Input Seed: {seed_input}\n"
                                  f"Output Seed: {output_seed}")
                # Ensure the correct tuple format is returned
                return (output_seed, status_message)

            # --- The rest of the logic is for "Execute Action (Dynamic)" mode ---

            # Clean up selected seed name from dropdown prefixes
            clean_seed_name = seed_to_load_name
            for prefix in ["⭐ ", "🕒 "]:
                if clean_seed_name.startswith(prefix):
                    clean_seed_name = clean_seed_name[len(prefix):]
            # Ignore category headers or (None)
            if clean_seed_name.startswith("---") or clean_seed_name == "(None)":
                clean_seed_name = "" # Treat as no selection

            # Get list of seeds based on search pattern and subdirectory
            current_seeds = []
            if search_pattern and search_pattern != "*":
                current_seeds = search_seeds(search_pattern, subdirectory)
            else:
                current_seeds = get_all_saved_seed_names(subdirectory, get_cache_key())

            num_seeds = len(current_seeds)

            # Initial status message for dynamic mode
            status_message = (
                f"Mode: Execute Action\n"
                f"Action: {action}\n"
                f"Input Seed: {seed_input}\n"
                f"Directory: '{os.path.join('seeds', subdirectory) if subdirectory else 'seeds/root'}'\n"
                f"Seeds Found (matching '{search_pattern}'): {num_seeds}"
            )

            print(f"[SeedSaver] Action='{action}', Input={seed_input}, Subdir='{subdirectory}'")

            # --- Action Handling Logic ---
            if action == "SAVE_CURRENT_SEED":
                save_name = sanitize_filename(seed_name_input.strip())
                if not save_name:
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    save_name = f"seed_{seed_input}_{timestamp}"
                    status_message += f"\nAuto-named: '{save_name}'"

                duplicates = find_duplicate_seeds(subdirectory)
                existing_value = load_seed_from_file(save_name, subdirectory)

                if existing_value is not None and existing_value != output_seed:
                    status_message += f"\n⚠️ Warning: Overwriting different value ({existing_value} -> {output_seed})"
                elif output_seed in duplicates and save_name not in duplicates[output_seed]:
                     status_message += f"\n⚠️ Note: Value {output_seed} already exists as: {', '.join(duplicates[output_seed])}"

                metadata = {
                    "description": description.strip() if description else "",
                    "tags": [t.strip() for t in tags.split(",") if t.strip()],
                    "workflow": "ComfyUI", # Example metadata
                }
                if save_seed_to_file(save_name, output_seed, subdirectory, metadata):
                    status_message += f"\n✅ SAVED: '{save_name}' = {output_seed}"
                    seed_history.add(save_name, output_seed)
                    if metadata["description"]: status_message += f"\n  Description: {metadata['description'][:50]}..."
                    if metadata["tags"]: status_message += f"\n  Tags: {', '.join(metadata['tags'])}"
                else:
                    status_message += f"\n❌ Error: Failed to save seed '{save_name}'"

            elif action == "LOAD_SELECTED_SEED":
                if clean_seed_name:
                    loaded_seed = load_seed_from_file(clean_seed_name, subdirectory)
                    if loaded_seed is not None:
                        output_seed = loaded_seed
                        status_message += f"\n✅ LOADED: '{clean_seed_name}' = {loaded_seed}"
                        seed_history.add(clean_seed_name, loaded_seed)
                        data = load_seed_metadata(clean_seed_name, subdirectory)
                        if data and data.get('metadata'):
                            meta = data['metadata']
                            if meta.get('description'): status_message += f"\n  Description: {meta['description'][:50]}..."
                            if meta.get('tags'): status_message += f"\n  Tags: {', '.join(meta['tags'])}"
                    else:
                        status_message += f"\n❌ Error: Seed '{clean_seed_name}' not found"
                else:
                    status_message += "\nℹ️ No seed selected to load"

            elif action == "DELETE_SELECTED_SEED":
                if clean_seed_name:
                    if delete_seed_file(clean_seed_name, subdirectory):
                        status_message += f"\n✅ DELETED: '{clean_seed_name}'"
                    else:
                        status_message += f"\n❌ Error: Could not delete '{clean_seed_name}' (Not found?)"
                else:
                    status_message += "\nℹ️ No seed selected for deletion"

            elif action == "LOAD_LATEST_SAVED_SEED":
                all_seeds_in_dir = get_all_saved_seed_names(subdirectory, get_cache_key()) # Needs full list
                if all_seeds_in_dir:
                    target_dir = os.path.join(OUTPUT_SEEDS_DIR, subdirectory)
                    files_with_times = []
                    for filename in os.listdir(target_dir):
                        if filename.endswith((".json", ".txt")) and not filename.startswith("_"):
                            filepath = os.path.join(target_dir, filename)
                            try: files_with_times.append((filepath, os.path.getmtime(filepath)))
                            except OSError: pass # Ignore files that might disappear
                    if files_with_times:
                        files_with_times.sort(key=lambda x: x[1], reverse=True)
                        latest_path = files_with_times[0][0]
                        latest_name = os.path.splitext(os.path.basename(latest_path))[0]
                        loaded_seed = load_seed_from_file(latest_name, subdirectory)
                        if loaded_seed is not None:
                            output_seed = loaded_seed
                            status_message += f"\n✅ LOADED LATEST: '{latest_name}' = {loaded_seed}"
                            seed_history.add(latest_name, loaded_seed)
                        else:
                            status_message += f"\n❌ Error loading latest seed file: {latest_name}"
                    else:
                         status_message += "\nℹ️ No valid seed files found to determine latest"
                else:
                    status_message += "\nℹ️ No seeds in directory"

            elif action == "LOAD_RANDOM_SEED":
                if current_seeds: # Uses the potentially filtered list
                    random_name = secrets.choice(current_seeds)
                    loaded_seed = load_seed_from_file(random_name, subdirectory)
                    if loaded_seed is not None:
                        output_seed = loaded_seed
                        status_message += f"\n✅ LOADED RANDOM (from {len(current_seeds)}): '{random_name}' = {loaded_seed}"
                        seed_history.add(random_name, loaded_seed)
                    else:
                        status_message += f"\n❌ Error loading randomly selected seed: {random_name}"
                else:
                    status_message += "\nℹ️ No seeds available for random selection (check search pattern?)"

            elif action == "LOAD_AND_INCREMENT":
                if clean_seed_name:
                    loaded_seed = load_seed_from_file(clean_seed_name, subdirectory)
                    if loaded_seed is not None:
                        output_seed = validate_seed(loaded_seed + 1)
                        status_message += f"\n✅ LOADED + 1: '{clean_seed_name}' {loaded_seed} -> {output_seed}"
                        seed_history.add(clean_seed_name, loaded_seed) # Add original loaded name to history
                    else:
                        status_message += f"\n❌ Error: Seed '{clean_seed_name}' not found for increment"
                else:
                    status_message += "\nℹ️ No seed selected for increment"

            elif action == "GENERATE_RANDOM_SEED":
                output_seed = secrets.randbelow(SEED_MAX + 1) # Ensure max value is inclusive
                status_message += f"\n✅ GENERATED RANDOM SEED: {output_seed}"

            elif action == "COPY_TO_SUBDIRECTORY":
                target_sub = target_subdirectory.strip()
                if clean_seed_name and target_sub:
                    if copy_seed(clean_seed_name, subdirectory, target_sub):
                        status_message += f"\n✅ COPIED: '{clean_seed_name}' from '{subdirectory or 'root'}' -> '{target_sub}'"
                    else: status_message += f"\n❌ Error: Copy failed (seed exists? source missing?)"
                else: status_message += "\nℹ️ Need seed name and valid target subdirectory"

            elif action == "MOVE_TO_SUBDIRECTORY":
                target_sub = target_subdirectory.strip()
                if clean_seed_name and target_sub:
                    if move_seed(clean_seed_name, subdirectory, target_sub):
                        status_message += f"\n✅ MOVED: '{clean_seed_name}' from '{subdirectory or 'root'}' -> '{target_sub}'"
                    else: status_message += f"\n❌ Error: Move failed (seed exists? permissions? source missing?)"
                else: status_message += "\nℹ️ Need seed name and valid target subdirectory"

            elif action == "TOGGLE_FAVORITE":
                if clean_seed_name:
                    success, msg = toggle_favorite(clean_seed_name)
                    status_message += f"\n{'✅' if success else '❌'} {msg}"
                else: status_message += "\nℹ️ No seed selected for favorites"

            elif action == "SHOW_STATISTICS":
                if clean_seed_name:
                    stats_info = get_seed_statistics(clean_seed_name)
                    status_message += f"\n📊 Statistics for '{clean_seed_name}':\n  {stats_info}"
                else: status_message += "\nℹ️ No seed selected for statistics"

            elif action == "FIND_DUPLICATES":
                duplicates = find_duplicate_seeds(subdirectory)
                if duplicates:
                    status_message += f"\n🔍 Found {len(duplicates)} duplicate value(s):"
                    for value, names in list(duplicates.items())[:5]: # Show first 5
                        status_message += f"\n  Value {value}: {', '.join(names)}"
                    if len(duplicates) > 5: status_message += f"\n  ... and {len(duplicates) - 5} more"
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
                status_message += "\n🔄 Lists refreshed! Dropdown will update on next queue."

            # Final output info for dynamic mode
            status_message += f"\n\n📤 Output Seed: {output_seed}"
            if action not in ["(None)", "LOAD_SELECTED_SEED", "LOAD_LATEST_SAVED_SEED", "LOAD_RANDOM_SEED", "LOAD_AND_INCREMENT", "GENERATE_RANDOM_SEED", "SHOW_STATISTICS", "FIND_DUPLICATES"]:
                 status_message += "\n💡 Tip: Use REFRESH_LISTS action if dropdowns seem outdated after changes."

            print(f"[SeedSaver] Output seed = {output_seed}")
            return (output_seed, status_message)

        except Exception as e:
            logging.error(f"[EnhancedSeedSaver] Execution failed: {e}")
            logging.debug(traceback.format_exc())
            error_msg = f"❌ [EnhancedSeedSaver] Error: {e}\n{traceback.format_exc()}"
            print(f"[EnhancedSeedSaver] ⚠️ Error encountered, passing through input seed.")
            # Graceful failure: return input seed and error message
            return (validate_seed(seed_input), error_msg)


# =================================================================================
# == Node Registration                                                         ==
# =================================================================================

# Ensure directories exist on ComfyUI load/script execution
ensure_output_directory_exists()

NODE_CLASS_MAPPINGS = {
    "EnhancedSeedSaver": EnhancedSeedSaverNode
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "EnhancedSeedSaver": "MD: Enhanced Seed Saver" # Added MD: prefix, removed emoji
}

# --- Initial Load Messages ---
# Removed version number from print statement
print(f"MD Enhanced Seed Saver node loaded. Seed storage directory: {OUTPUT_SEEDS_DIR}")
try:
    initial_seed_names = get_all_saved_seed_names() # Check root directory on load
    favorites = load_favorites()
    if initial_seed_names:
        print("Currently saved seeds (in root directory):")
        for name in initial_seed_names[:10]: # Limit initial printout
            prefix = "⭐ " if name in favorites else "  " # Keep favorite indicator here is okay
            print(f"{prefix}- {name}")
        if len(initial_seed_names) > 10: print(f"  ... and {len(initial_seed_names) - 10} more")
    else: print("No seeds currently saved in the root directory.")
    if favorites: print(f"Favorite seeds found: {len(favorites)}")
except Exception as e:
    print(f"[EnhancedSeedSaver] Warning: Could not list initial seeds on load: {e}")
print("-" * 30)