# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# ████ MD ENHANCED SEED SAVER v2.1.0 – Professional Seed Management ████▓▒░
# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀

# ░▒▓ ORIGIN & DEV:
#   • Forged in the digital foundry by: MDMAchine (chaos wrangler)
#   • Enhanced and Optimized by: Claude (Anthropic AI Assistant)
#   • Original concept inspired by: The eternal struggle of seed management
#   • Built for: ComfyUI workflow architects and generation artists
#   • License: Public Domain — Share the deterministic love

# ░▒▓ CHANGELOG v2.1.0 (Operation Mode Update):
#   ✓ ADDED: An "operation_mode" toggle to switch between "Pass-through (Static)" and "Execute Action (Dynamic)".
#   ✓ FIXED: In "Pass-through" mode, the node is fully cacheable and will not cause re-runs.
#   ✓ ENHANCED: IS_CHANGED logic is now conditional based on the selected mode for perfect stability.

# ░▒▓ DESCRIPTION:
#   The ultimate seed management companion for ComfyUI. Switch between a dynamic action mode
#   for managing seeds and a static pass-through mode for stable, cacheable workflows without
#   needing to bypass the node.

import json
import os
import folder_paths
from datetime import datetime
import random
import secrets
import shutil
from functools import lru_cache
from time import time
from typing import Optional, List, Tuple, Dict, Any
import re

# Constants
SEED_MIN = 0
SEED_MAX = 0xffffffffffffffff
OUTPUT_SEEDS_DIR = os.path.join(folder_paths.get_output_directory(), "seeds")
BACKUP_DIR = os.path.join(OUTPUT_SEEDS_DIR, "_backups")
FAVORITES_FILE = os.path.join(OUTPUT_SEEDS_DIR, "_favorites.json")
STATS_FILE = os.path.join(OUTPUT_SEEDS_DIR, "_statistics.json")
CACHE_DURATION = 5  # seconds

# --- Utility Functions ---

def sanitize_filename(name: str) -> str:
    """Remove invalid filename characters"""
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', name)
    sanitized = sanitized.strip('. ')
    return sanitized[:200] if sanitized else "unnamed"

def validate_seed(seed_value: int) -> int:
    """Ensure seed is within valid range"""
    return max(SEED_MIN, min(int(seed_value), SEED_MAX))

def get_cache_key() -> int:
    """Generate cache key based on current time"""
    return int(time() / CACHE_DURATION)

# --- Directory Management ---

def ensure_output_directory_exists(subdirectory: str = "") -> None:
    target_dir = os.path.join(OUTPUT_SEEDS_DIR, subdirectory)
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(BACKUP_DIR, exist_ok=True)

def get_seed_filepath(seed_name: str, subdirectory: str = "", extension: str = ".json") -> str:
    target_dir = os.path.join(OUTPUT_SEEDS_DIR, subdirectory)
    return os.path.join(target_dir, f"{seed_name}{extension}")

# --- Enhanced Seed Operations (functions remain the same) ---
# ... All helper functions (save_seed_to_file, load_seed_from_file, etc.) are unchanged ...
def save_seed_to_file(seed_name: str, seed_value: int, subdirectory: str = "",
                      metadata: Optional[Dict[str, Any]] = None) -> bool:
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
        print(f"ERROR: Could not save seed '{seed_name}': {e}")
        return False
def load_seed_from_file(seed_name: str, subdirectory: str = "") -> Optional[int]:
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
            print(f"ERROR: Could not load JSON seed '{seed_name}': {e}")
            return None
    if os.path.exists(txt_filepath):
        print(f"Note: Loading seed '{seed_name}' from legacy .txt format.")
        try:
            with open(txt_filepath, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                return validate_seed(int(content))
        except Exception as e:
            print(f"ERROR: Could not load legacy txt seed '{seed_name}': {e}")
            return None
    print(f"WARNING: Seed file for '{seed_name}' not found.")
    return None
def load_seed_metadata(seed_name: str, subdirectory: str = "") -> Optional[Dict[str, Any]]:
    json_filepath = get_seed_filepath(seed_name, subdirectory, extension=".json")
    if os.path.exists(json_filepath):
        try:
            with open(json_filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"ERROR: Could not load seed metadata '{seed_name}': {e}")
    return None
def delete_seed_file(seed_name: str, subdirectory: str = "") -> bool:
    deleted = False
    for ext in [".json", ".txt"]:
        filepath = get_seed_filepath(seed_name, subdirectory, extension=ext)
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
                print(f"MD Seed Saver: Removed '{filepath}'")
                deleted = True
            except Exception as e:
                print(f"ERROR: Could not delete seed file '{filepath}': {e}")
    if deleted:
        remove_from_favorites(seed_name)
        update_statistics(seed_name, 'deleted')
    return deleted
@lru_cache(maxsize=32)
def get_all_saved_seed_names(subdirectory: str = "", cache_key: int = 0) -> List[str]:
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
        print(f"ERROR: Could not list seeds from '{target_dir}': {e}")
        return []
def search_seeds(pattern: str, subdirectory: str = "") -> List[str]:
    all_seeds = get_all_saved_seed_names(subdirectory, get_cache_key())
    if pattern == "*" or not pattern:
        return all_seeds
    pattern_lower = pattern.lower()
    return [s for s in all_seeds if pattern_lower in s.lower()]
def copy_seed(seed_name: str, from_subdir: str, to_subdir: str) -> bool:
    seed_value = load_seed_from_file(seed_name, from_subdir)
    if seed_value is None:
        return False
    metadata = load_seed_metadata(seed_name, from_subdir)
    seed_metadata = metadata.get('metadata', {}) if metadata else {}
    return save_seed_to_file(seed_name, seed_value, to_subdir, seed_metadata)
def move_seed(seed_name: str, from_subdir: str, to_subdir: str) -> bool:
    if copy_seed(seed_name, from_subdir, to_subdir):
        return delete_seed_file(seed_name, from_subdir)
    return False
def export_all_seeds(subdirectory: str = "") -> Tuple[bool, str]:
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
        return False, f"Export failed: {e}"
def import_seeds(import_file: str, subdirectory: str = "") -> Tuple[int, List[str]]:
    try:
        with open(import_file, 'r', encoding='utf-8') as f:
            import_data = json.load(f)
        imported = 0
        errors = []
        for name, data in import_data.items():
            try:
                seed_value = data.get('seed', data) if isinstance(data, dict) else data
                metadata = data.get('metadata', {}) if isinstance(data, dict) else {}
                if save_seed_to_file(name, seed_value, subdirectory, metadata):
                    imported += 1
                else:
                    errors.append(f"Failed to save: {name}")
            except Exception as e:
                errors.append(f"{name}: {str(e)}")
        return imported, errors
    except Exception as e:
        return 0, [f"Import failed: {e}"]
def clear_directory(subdirectory: str = "", keep_backups: bool = True) -> Tuple[int, str]:
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
def backup_seeds(subdirectory: str = "") -> Tuple[bool, str]:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    subdir_name = subdirectory.replace('/', '_') if subdirectory else 'root'
    backup_subdir = os.path.join(BACKUP_DIR, f"{subdir_name}_{timestamp}")
    try:
        source_dir = os.path.join(OUTPUT_SEEDS_DIR, subdirectory)
        if os.path.exists(source_dir):
            shutil.copytree(source_dir, backup_subdir, 
                            ignore=shutil.ignore_patterns('_backups*', '_favorites*', '_statistics*'))
            return True, f"Backup created: {subdir_name}_{timestamp}"
        return False, "Source directory not found"
    except Exception as e:
        return False, f"Backup failed: {e}"
def load_favorites() -> List[str]:
    if os.path.exists(FAVORITES_FILE):
        try:
            with open(FAVORITES_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except: return []
    return []
def save_favorites(favorites: List[str]) -> bool:
    try:
        with open(FAVORITES_FILE, 'w', encoding='utf-8') as f:
            json.dump(favorites, f, indent=4)
        return True
    except Exception as e:
        print(f"ERROR: Could not save favorites: {e}")
        return False
def toggle_favorite(seed_name: str) -> Tuple[bool, str]:
    favorites = load_favorites()
    if seed_name in favorites:
        favorites.remove(seed_name)
        save_favorites(favorites)
        return True, f"Removed '{seed_name}' from favorites"
    else:
        favorites.append(seed_name)
        save_favorites(favorites)
        return True, f"Added '{seed_name}' to favorites"
def remove_from_favorites(seed_name: str) -> None:
    favorites = load_favorites()
    if seed_name in favorites:
        favorites.remove(seed_name)
        save_favorites(favorites)
def load_statistics() -> Dict[str, Dict[str, Any]]:
    if os.path.exists(STATS_FILE):
        try:
            with open(STATS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except: return {}
    return {}
def save_statistics(stats: Dict[str, Dict[str, Any]]) -> bool:
    try:
        with open(STATS_FILE, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=4)
        return True
    except Exception as e:
        print(f"ERROR: Could not save statistics: {e}")
        return False
def update_statistics(seed_name: str, action: str) -> None:
    stats = load_statistics()
    if seed_name not in stats:
        stats[seed_name] = {'saves': 0, 'loads': 0, 'deletes': 0, 'last_used': None}
    if action == 'saved': stats[seed_name]['saves'] += 1
    elif action == 'loaded': stats[seed_name]['loads'] += 1
    elif action == 'deleted': stats[seed_name]['deletes'] += 1
    stats[seed_name]['last_used'] = datetime.now().isoformat()
    save_statistics(stats)
def get_seed_statistics(seed_name: str) -> str:
    stats = load_statistics()
    if seed_name in stats:
        s = stats[seed_name]
        return (f"Saves: {s['saves']}, Loads: {s['loads']}, "
                f"Last used: {s.get('last_used', 'Never')[:19]}")
    return "No statistics available"
def find_duplicate_seeds(subdirectory: str = "") -> Dict[int, List[str]]:
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
    def __init__(self, max_size: int = 10):
        self.max_size = max_size
        self.history: List[Tuple[str, int]] = []
    def add(self, seed_name: str, seed_value: int):
        self.history = [(n, v) for n, v in self.history if n != seed_name]
        self.history.insert(0, (seed_name, seed_value))
        self.history = self.history[:self.max_size]
    def get_list(self) -> List[str]:
        return [name for name, _ in self.history]

seed_history = SeedHistory()

class EnhancedSeedSaverNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        cache_key = get_cache_key()
        seed_names = get_all_saved_seed_names("", cache_key)
        favorites = load_favorites()
        history = seed_history.get_list()
        
        seed_options = ["(None)"]
        if favorites:
            seed_options.append("--- FAVORITES ---")
            seed_options.extend([f"⭐ {name}" for name in favorites if name in seed_names])
        if history:
            seed_options.append("--- RECENT ---")
            seed_options.extend([f"🕒 {name}" for name in history[:5] if name in seed_names])
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
                "seed_input": ("INT", { "default": 0, "min": SEED_MIN, "max": SEED_MAX, "forceInput": True }),
                "operation_mode": (["Pass-through (Static)", "Execute Action (Dynamic)"],),
            },
            "optional": {
                "action": (all_actions,),
                "seed_name_input": ("STRING", { "default": "", "multiline": False, "placeholder": "Enter seed name..." }),
                "seed_to_load_name": (seed_options, {"default": "(None)"}),
                "subdirectory": ("STRING", { "default": "", "multiline": False, "placeholder": "Optional: folder/subfolder" }),
                "search_pattern": ("STRING", { "default": "*", "multiline": False, "placeholder": "Search pattern" }),
                "description": ("STRING", { "default": "", "multiline": True, "placeholder": "Optional: Add notes or description" }),
                "tags": ("STRING", { "default": "", "multiline": False, "placeholder": "Optional: tag1, tag2, tag3" }),
                "target_subdirectory": ("STRING", { "default": "", "multiline": False, "placeholder": "For copy/move operations" }),
            }
        }

    RETURN_TYPES = ("INT", "STRING")
    RETURN_NAMES = ("seed_output", "status_info")
    FUNCTION = "execute"
    CATEGORY = "MD_Nodes/Utility"
    OUTPUT_NODE = True

    def execute(self, seed_input: int, operation_mode: str, action: str, seed_name_input: str, 
                seed_to_load_name: str, subdirectory: str, search_pattern: str,
                description: str, tags: str, target_subdirectory: str):
        
        output_seed = validate_seed(seed_input)
        status_message = ""

        if operation_mode == "Pass-through (Static)":
            status_message = (f"Mode: Pass-through\n"
                              f"Input Seed: {seed_input}\n"
                              f"Output Seed: {output_seed}")
            return (output_seed, status_message)

        # --- The rest of the logic is for "Execute Action (Dynamic)" mode ---
        
        clean_seed_name = seed_to_load_name
        for prefix in ["⭐ ", "🕒 ", "--- ", " ---"]:
            if clean_seed_name.startswith(prefix):
                clean_seed_name = clean_seed_name[len(prefix):]
        if clean_seed_name.startswith("FAVORITES") or clean_seed_name.startswith("RECENT") or clean_seed_name.startswith("ALL SEEDS"):
            clean_seed_name = "(None)"
        
        if search_pattern and search_pattern != "*":
            current_seeds = search_seeds(search_pattern, subdirectory)
        else:
            current_seeds = get_all_saved_seed_names(subdirectory, get_cache_key())
        
        num_seeds = len(current_seeds)
        
        status_message = (
            f"Mode: Execute Action\n"
            f"Input Seed: {seed_input}\n"
            f"Directory: '{os.path.join('seeds', subdirectory) if subdirectory else 'seeds/root'}'\n"
            f"Seeds in Directory: {num_seeds}"
        )
        
        # ... All action handling logic (if action == "SAVE_CURRENT_SEED": etc.) is unchanged ...
        print(f"MD Seed Saver: Action='{action}', Input={seed_input}, Subdir='{subdirectory}'")
        if action == "SAVE_CURRENT_SEED":
            save_name = sanitize_filename(seed_name_input.strip())
            if not save_name:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                save_name = f"seed_{seed_input}_{timestamp}"
                status_message += f"\nAuto-named: '{save_name}'"
            duplicates = find_duplicate_seeds(subdirectory)
            existing_value = load_seed_from_file(save_name, subdirectory)
            if existing_value is not None and existing_value != output_seed:
                status_message += f"\n⚠️ Warning: Overwriting different value ({existing_value} → {output_seed})"
            elif output_seed in duplicates and save_name not in duplicates[output_seed]:
                status_message += f"\n⚠️ Note: This value already exists as: {', '.join(duplicates[output_seed])}"
            metadata = {
                "description": description.strip() if description else "",
                "tags": [t.strip() for t in tags.split(",") if t.strip()],
                "workflow": "ComfyUI",
            }
            if save_seed_to_file(save_name, output_seed, subdirectory, metadata):
                status_message += f"\n✅ SAVED: '{save_name}' = {output_seed}"
                seed_history.add(save_name, output_seed)
                if metadata["description"]: status_message += f"\n   Description: {metadata['description'][:50]}..."
                if metadata["tags"]: status_message += f"\n   Tags: {', '.join(metadata['tags'])}"
            else:
                status_message += f"\n❌ Error: Failed to save seed '{save_name}'"
        elif action == "LOAD_SELECTED_SEED":
            if clean_seed_name and clean_seed_name != "(None)":
                loaded_seed = load_seed_from_file(clean_seed_name, subdirectory)
                if loaded_seed is not None:
                    output_seed = loaded_seed
                    status_message += f"\n✅ LOADED: '{clean_seed_name}' = {loaded_seed}"
                    seed_history.add(clean_seed_name, loaded_seed)
                    data = load_seed_metadata(clean_seed_name, subdirectory)
                    if data and data.get('metadata'):
                        meta = data['metadata']
                        if meta.get('description'): status_message += f"\n   Description: {meta['description'][:50]}..."
                        if meta.get('tags'): status_message += f"\n   Tags: {', '.join(meta['tags'])}"
                else:
                    status_message += f"\n❌ Error: Seed '{clean_seed_name}' not found"
            else:
                status_message += "\nℹ️ No seed selected to load"
        elif action == "DELETE_SELECTED_SEED":
            if clean_seed_name and clean_seed_name != "(None)":
                if delete_seed_file(clean_seed_name, subdirectory):
                    status_message += f"\n✅ DELETED: '{clean_seed_name}'"
                else:
                    status_message += f"\n❌ Error: Could not delete '{clean_seed_name}'"
            else:
                status_message += "\nℹ️ No seed selected for deletion"
        elif action == "LOAD_LATEST_SAVED_SEED":
            if current_seeds:
                target_dir = os.path.join(OUTPUT_SEEDS_DIR, subdirectory)
                files_with_times = []
                for filename in os.listdir(target_dir):
                    if filename.endswith((".json", ".txt")) and not filename.startswith("_"):
                        filepath = os.path.join(target_dir, filename)
                        try: files_with_times.append((filepath, os.path.getmtime(filepath)))
                        except: pass
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
                status_message += "\nℹ️ No seeds in directory"
        elif action == "LOAD_RANDOM_SEED":
            if current_seeds:
                random_name = secrets.choice(current_seeds)
                loaded_seed = load_seed_from_file(random_name, subdirectory)
                if loaded_seed is not None:
                    output_seed = loaded_seed
                    status_message += f"\n✅ LOADED RANDOM: '{random_name}' = {loaded_seed}"
                    seed_history.add(random_name, loaded_seed)
            else:
                status_message += "\nℹ️ No seeds available for random selection"
        elif action == "LOAD_AND_INCREMENT":
            if clean_seed_name and clean_seed_name != "(None)":
                loaded_seed = load_seed_from_file(clean_seed_name, subdirectory)
                if loaded_seed is not None:
                    output_seed = validate_seed(loaded_seed + 1)
                    status_message += f"\n✅ LOADED + 1: '{clean_seed_name}' {loaded_seed} → {output_seed}"
                    seed_history.add(clean_seed_name, loaded_seed)
            else:
                status_message += "\nℹ️ No seed selected for increment"
        elif action == "GENERATE_RANDOM_SEED":
            output_seed = secrets.randbelow(SEED_MAX)
            status_message += f"\n✅ GENERATED RANDOM SEED: {output_seed}"
        elif action == "COPY_TO_SUBDIRECTORY":
            if clean_seed_name and clean_seed_name != "(None)" and target_subdirectory:
                if copy_seed(clean_seed_name, subdirectory, target_subdirectory):
                    status_message += f"\n✅ COPIED: '{clean_seed_name}' → '{target_subdirectory}'"
                else: status_message += f"\n❌ Error: Copy failed"
            else: status_message += "\nℹ️ Need seed name and target subdirectory"
        elif action == "MOVE_TO_SUBDIRECTORY":
            if clean_seed_name and clean_seed_name != "(None)" and target_subdirectory:
                if move_seed(clean_seed_name, subdirectory, target_subdirectory):
                    status_message += f"\n✅ MOVED: '{clean_seed_name}' → '{target_subdirectory}'"
                else: status_message += f"\n❌ Error: Move failed"
            else: status_message += "\nℹ️ Need seed name and target subdirectory"
        elif action == "TOGGLE_FAVORITE":
            if clean_seed_name and clean_seed_name != "(None)":
                success, msg = toggle_favorite(clean_seed_name)
                status_message += f"\n{'✅' if success else '❌'} {msg}"
            else: status_message += "\nℹ️ No seed selected for favorites"
        elif action == "SHOW_STATISTICS":
            if clean_seed_name and clean_seed_name != "(None)":
                stats_info = get_seed_statistics(clean_seed_name)
                status_message += f"\n📊 Statistics for '{clean_seed_name}':\n   {stats_info}"
            else: status_message += "\nℹ️ No seed selected for statistics"
        elif action == "FIND_DUPLICATES":
            duplicates = find_duplicate_seeds(subdirectory)
            if duplicates:
                status_message += f"\n🔍 Found {len(duplicates)} duplicate value(s):"
                for value, names in list(duplicates.items())[:5]:
                    status_message += f"\n   Value {value}: {', '.join(names)}"
                if len(duplicates) > 5: status_message += f"\n   ... and {len(duplicates) - 5} more"
            else: status_message += "\n✅ No duplicate seeds found"
        elif action == "EXPORT_ALL_SEEDS":
            success, msg = export_all_seeds(subdirectory)
            status_message += f"\n{'✅' if success else '❌'} {msg}"
        elif action == "CLEAR_DIRECTORY":
            count, msg = clear_directory(subdirectory, keep_backups=True)
            status_message += f"\n⚠️ {msg}"
        elif action == "BACKUP_SEEDS":
            success, msg = backup_seeds(subdirectory)
            status_message += f"\n{'✅' if success else '❌'} {msg}"
        elif action == "REFRESH_LISTS":
            get_all_saved_seed_names.cache_clear()
            status_message += "\n🔄 Lists refreshed! Dropdown will update on next queue."

        status_message += f"\n\n📤 Output Seed: {output_seed}"
        if action in ["SAVE_CURRENT_SEED", "DELETE_SELECTED_SEED", "CLEAR_DIRECTORY", "REFRESH_LISTS"]:
            status_message += "\n💡 Tip: Use REFRESH_LISTS to update dropdowns"
        
        print(f"MD Seed Saver: Output seed = {output_seed}")
        return (output_seed, status_message)

    @classmethod
    def IS_CHANGED(cls, seed_input, operation_mode, **kwargs):
        if operation_mode == "Execute Action (Dynamic)":
            # In action mode, we must re-run every time.
            return secrets.token_hex(16)
        else:
            # In pass-through mode, we only depend on the input seed.
            # This makes the node cacheable and prevents re-runs.
            return seed_input

ensure_output_directory_exists()

NODE_CLASS_MAPPINGS = { "EnhancedSeedSaver": EnhancedSeedSaverNode }
NODE_DISPLAY_NAME_MAPPINGS = { "EnhancedSeedSaver": "MD Enhanced Seed Saver" }

print(f"MD Enhanced Seed Saver node loaded. Seed storage directory: {OUTPUT_SEEDS_DIR}")
# ... Initial print statements are unchanged ...
initial_seed_names = get_all_saved_seed_names()
favorites = load_favorites()
if initial_seed_names:
    print("Currently saved seeds (in root directory):")
    for name in initial_seed_names[:10]:
        prefix = "⭐ " if name in favorites else "  "
        print(f"{prefix}- {name}")
    if len(initial_seed_names) > 10: print(f"  ... and {len(initial_seed_names) - 10} more")
else: print("No seeds currently saved in the root directory.")
if favorites: print(f"Favorite seeds: {len(favorites)}")
print("-" * 30)