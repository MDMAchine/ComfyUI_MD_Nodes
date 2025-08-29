import json
import os
import folder_paths
from datetime import datetime
import random

# Define the directory for saving seeds.
OUTPUT_SEEDS_DIR = os.path.join(folder_paths.get_output_directory(), "seeds")

# --- Enhanced File Operations for Seed Storage ---

def ensure_output_directory_exists(subdirectory=""):
    """
    Ensures the custom output directory for seeds, including any subdirectories, exists.
    """
    target_dir = os.path.join(OUTPUT_SEEDS_DIR, subdirectory)
    os.makedirs(target_dir, exist_ok=True)

def get_seed_filepath(seed_name, subdirectory="", extension=".json"):
    """
    Returns the full file path for a given seed name, inside an optional subdirectory.
    Defaults to .json extension.
    """
    target_dir = os.path.join(OUTPUT_SEEDS_DIR, subdirectory)
    return os.path.join(target_dir, f"{seed_name}{extension}")

def save_seed_to_file(seed_name, seed_value, subdirectory=""):
    """
    Saves a seed and metadata to a .json file.
    """
    ensure_output_directory_exists(subdirectory)
    filepath = get_seed_filepath(seed_name, subdirectory)
    
    data_to_save = {
        "seed": seed_value,
        "saved_at": datetime.now().isoformat()
    }

    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, indent=4)
        return True
    except Exception as e:
        print(f"ERROR: Could not save seed '{seed_name}' to file: {e}")
        return False

def load_seed_from_file(seed_name, subdirectory=""):
    """
    Loads a seed from a .json file, with fallback to .txt for old files.
    Returns the integer seed value or None if not found/error.
    """
    json_filepath = get_seed_filepath(seed_name, subdirectory, extension=".json")
    txt_filepath = get_seed_filepath(seed_name, subdirectory, extension=".txt")
    
    # Prioritize loading from new .json format
    if os.path.exists(json_filepath):
        try:
            with open(json_filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return int(data["seed"])
        except Exception as e:
            print(f"ERROR: Could not load JSON seed '{seed_name}': {e}")
            return None

    # Fallback for old .txt format
    if os.path.exists(txt_filepath):
        print(f"Note: Loading seed '{seed_name}' from legacy .txt format.")
        try:
            with open(txt_filepath, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                return int(content)
        except Exception as e:
            print(f"ERROR: Could not load legacy .txt seed '{seed_name}': {e}")
            return None
            
    print(f"WARNING: Seed file for '{seed_name}' not found in '{os.path.join(OUTPUT_SEEDS_DIR, subdirectory)}'.")
    return None


def delete_seed_file(seed_name, subdirectory=""):
    """
    Deletes a seed file, attempting to remove both .json and legacy .txt files.
    """
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
    return deleted

def get_all_saved_seed_names(subdirectory=""):
    """
    Lists all saved seed names from a directory (without extension),
    finding both .json and .txt files.
    """
    target_dir = os.path.join(OUTPUT_SEEDS_DIR, subdirectory)
    ensure_output_directory_exists(subdirectory)
    
    try:
        all_files_in_dir = os.listdir(target_dir)
        seed_names = sorted(list(set(
            os.path.splitext(f)[0] for f in all_files_in_dir if f.endswith((".txt", ".json"))
        )))
        return seed_names
    except Exception as e:
        print(f"ERROR: Could not list seeds from directory '{target_dir}': {e}")
        return []

# Ensure the root output directory exists when the module loads
ensure_output_directory_exists()

# --- ComfyUI Node Definition ---

class SeedSaverNode:
    """
    An enhanced ComfyUI node to manage generation seeds using individual files,
    with support for subdirectories and random loading.
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        seed_names_for_display = get_all_saved_seed_names(subdirectory="")
        seed_load_options = ["(None)"] + seed_names_for_display

        return {
            "required": {
                "seed_input": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "forceInput": True}),
            },
            "optional": {
                "action": (["(None)", "SAVE_CURRENT_SEED", "LOAD_SELECTED_SEED", "DELETE_SELECTED_SEED", "LOAD_LATEST_SAVED_SEED", "LOAD_RANDOM_SEED"],),
                "seed_name_input": ("STRING", {"default": "", "multiline": False}),
                "seed_to_load_name": (seed_load_options, {"default": "(None)"}),
                "subdirectory": ("STRING", {"default": "", "multiline": False}),
            }
        }

    RETURN_TYPES = ("INT", "STRING")
    RETURN_NAMES = ("seed_output", "status_info")
    FUNCTION = "execute"
    CATEGORY = "MD_Nodes/Utility"
    OUTPUT_NODE = True

    def execute(self, seed_input, action, seed_name_input, seed_to_load_name, subdirectory):
        output_seed = seed_input
        
        current_saved_seed_names_in_subdir = get_all_saved_seed_names(subdirectory)
        num_saved_seeds = len(current_saved_seed_names_in_subdir)

        status_message = (f"Input Seed: {seed_input}\n"
                          f"Directory: '{os.path.join('seeds', subdirectory)}'\n"
                          f"Saved Seeds in Dir: {num_saved_seeds}")
        
        print(f"MD Seed Saver: Processing. Input seed: {seed_input}, Subdirectory: '{subdirectory}'")

        if action == "SAVE_CURRENT_SEED":
            save_name = seed_name_input.strip()
            if not save_name:
                timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
                save_name = f"seed_{seed_input}_{timestamp}"
                status_message += f"\nAuto-named: '{save_name}'"

            if save_name in current_saved_seed_names_in_subdir:
                 status_message += f"\nWarning: Overwriting existing seed '{save_name}'."
                 print(f"MD Seed Saver: Overwriting existing seed file for '{save_name}'")

            if save_seed_to_file(save_name, seed_input, subdirectory):
                status_message += f"\nAction: SAVED. Name: '{save_name}', Value: {seed_input}"
                print(f"MD Seed Saver: Successfully saved seed '{save_name}' with value {seed_input}")
                status_message += f"\nNote: Refresh browser to update dropdown list."
            else:
                status_message += f"\nError: Failed to save seed '{save_name}'."

        elif action == "LOAD_SELECTED_SEED":
            if seed_to_load_name and seed_to_load_name != "(None)":
                # FIX: Always load from the root directory ("") for this action, as the dropdown is populated from there.
                loaded_seed = load_seed_from_file(seed_to_load_name, subdirectory="")
                if loaded_seed is not None:
                    output_seed = loaded_seed
                    status_message += f"\nAction: LOADED from root. Name: '{seed_to_load_name}', Value: {loaded_seed}"
                    print(f"MD Seed Saver: Successfully loaded seed '{seed_to_load_name}' from root with value {loaded_seed}")
                else:
                    status_message += f"\nError: Seed '{seed_to_load_name}' not found in root. Outputting input seed."
            else:
                status_message += f"\nInfo: No seed selected to load."

        elif action == "DELETE_SELECTED_SEED":
            if seed_to_load_name and seed_to_load_name != "(None)":
                # FIX: Always delete from the root directory ("") for this action, matching the dropdown source.
                if delete_seed_file(seed_to_load_name, subdirectory=""):
                    status_message += f"\nAction: DELETED from root. Name: '{seed_to_load_name}'"
                    print(f"MD Seed Saver: Successfully deleted seed '{seed_to_load_name}' from root.")
                    status_message += f"\nNote: Refresh browser to update dropdown list."
                else:
                    status_message += f"\nError: Seed '{seed_to_load_name}' not found in root for deletion."
            else:
                status_message += f"\nInfo: No seed selected for deletion."
        
        elif action == "LOAD_LATEST_SAVED_SEED":
            if current_saved_seed_names_in_subdir:
                files_with_times = []
                target_dir = os.path.join(OUTPUT_SEEDS_DIR, subdirectory)
                for filename in os.listdir(target_dir):
                    if filename.endswith((".json", ".txt")):
                        filepath = os.path.join(target_dir, filename)
                        try:
                            files_with_times.append((filepath, os.path.getmtime(filepath)))
                        except Exception as e:
                            print(f"WARNING: Could not get mod time for {filename}: {e}")
                
                if files_with_times:
                    files_with_times.sort(key=lambda x: x[1], reverse=True)
                    latest_filepath = files_with_times[0][0]
                    latest_seed_name = os.path.splitext(os.path.basename(latest_filepath))[0]
                    
                    loaded_seed = load_seed_from_file(latest_seed_name, subdirectory)
                    if loaded_seed is not None:
                        output_seed = loaded_seed
                        status_message += f"\nAction: LOADED LATEST. Name: '{latest_seed_name}', Value: {loaded_seed}"
                        print(f"MD Seed Saver: Loaded latest seed '{latest_seed_name}'")
                else:
                    status_message += f"\nInfo: No valid seed files found to determine the latest."
            else:
                status_message += f"\nInfo: No seeds in directory to load latest from."

        elif action == "LOAD_RANDOM_SEED":
            if current_saved_seed_names_in_subdir:
                random_seed_name = random.choice(current_saved_seed_names_in_subdir)
                loaded_seed = load_seed_from_file(random_seed_name, subdirectory)
                if loaded_seed is not None:
                    output_seed = loaded_seed
                    status_message += f"\nAction: LOADED RANDOM. Name: '{random_seed_name}', Value: {loaded_seed}"
                    print(f"MD Seed Saver: Loaded random seed '{random_seed_name}'")
                else:
                    status_message += f"\nError: Failed to load chosen random seed '{random_seed_name}'."
            else:
                status_message += f"\nInfo: No saved seeds to choose from for random load."

        status_message += f"\nFinal Output Seed: {output_seed}"
        print(f"MD Seed Saver: Final output seed: {output_seed}")

        return (output_seed, status_message)

    @classmethod
    def IS_CHANGED(s, seed_input, action, seed_name_input, seed_to_load_name, subdirectory):
        return (seed_input, action, seed_name_input, seed_to_load_name, subdirectory)

# --- ComfyUI Node Registration ---

NODE_CLASS_MAPPINGS = {
    "SeedSaver": SeedSaverNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SeedSaver": "MD Seed Saver"
}

# Initial print of saved seeds to the console when the node is loaded
print(f"MD Seed Saver node loaded. Seed storage directory: {OUTPUT_SEEDS_DIR}")
initial_seed_names = get_all_saved_seed_names()
if initial_seed_names:
    print("Currently saved seeds (in root directory):")
    for name in initial_seed_names:
        print(f"  - {name}")
else:
    print("No seeds currently saved in the root directory.")
print("-" * 30)