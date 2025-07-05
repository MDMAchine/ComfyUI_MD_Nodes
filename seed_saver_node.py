# C:\Stable Diffusion\ComfyUI\custom_nodes\ComfyUI_MD_Nodes\seed_saver_node.py

import json
import os
import folder_paths # Still used for get_output_directory()
from datetime import datetime

# Define the directory for saving seeds.
# This will be inside ComfyUI's output directory, in a 'seeds' subfolder.
OUTPUT_SEEDS_DIR = os.path.join(folder_paths.get_output_directory(), "seeds")

# --- File Operations for Seed Storage ---

def ensure_output_directory_exists():
    """
    Ensures the custom output directory for seeds exists.
    """
    os.makedirs(OUTPUT_SEEDS_DIR, exist_ok=True)

def get_seed_filepath(seed_name):
    """
    Returns the full file path for a given seed name (appending .txt).
    """
    return os.path.join(OUTPUT_SEEDS_DIR, f"{seed_name}.txt")

def save_seed_to_file(seed_name, seed_value):
    """
    Saves a single seed to a .txt file.
    """
    ensure_output_directory_exists()
    filepath = get_seed_filepath(seed_name)
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(str(seed_value))
        return True
    except Exception as e:
        print(f"ERROR: Could not save seed '{seed_name}' to file: {e}")
        return False

def load_seed_from_file(seed_name):
    """
    Loads a single seed from a .txt file.
    Returns the integer seed value or None if not found/error.
    """
    filepath = get_seed_filepath(seed_name)
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            return int(content)
    except Exception as e:
        print(f"ERROR: Could not load seed '{seed_name}' from file: {e}")
        return None

def delete_seed_file(seed_name):
    """
    Deletes a seed .txt file.
    """
    filepath = get_seed_filepath(seed_name)
    if os.path.exists(filepath):
        try:
            os.remove(filepath)
            return True
        except Exception as e:
            print(f"ERROR: Could not delete seed file '{filepath}': {e}")
            return False
    return False

def get_all_saved_seed_names():
    """
    Lists all currently saved seed names (without .txt extension).
    """
    ensure_output_directory_exists()
    try:
        all_files_in_dir = os.listdir(OUTPUT_SEEDS_DIR)
        seed_names = sorted([os.path.splitext(f)[0] for f in all_files_in_dir if f.endswith(".txt")])
        return seed_names
    except FileNotFoundError:
        return []
    except Exception as e:
        print(f"ERROR: Could not list seeds from directory '{OUTPUT_SEEDS_DIR}': {e}")
        return []


# Ensure the output directory exists when the module loads
ensure_output_directory_exists()

# --- ComfyUI Node Definition ---

class SeedSaverNode:
    """
    An enhanced pure Python ComfyUI node to manage generation seeds using individual files.
    This version uses a static dropdown for loading/deleting seeds that updates
    only upon browser refresh, aligning with typical ComfyUI behavior for file lists.
    Provides a string output for real-time status and seed values.
    Includes QoL features: saved seed count, load latest seed, and startup console list.
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        # Read files directly from the directory to populate the dropdown.
        # This list will be static until ComfyUI is restarted or the node is re-added.
        seed_names_for_display = get_all_saved_seed_names()

        # Add a default option for "None" or "Select a seed"
        seed_load_options = ["(None)"] + seed_names_for_display

        return {
            "required": {
                "seed_input": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "forceInput": True}),
            },
            "optional": {
                "action": (["(None)", "SAVE_CURRENT_SEED", "LOAD_SELECTED_SEED", "DELETE_SELECTED_SEED", "LOAD_LATEST_SAVED_SEED"],), # Added LOAD_LATEST_SAVED_SEED
                "seed_name_input": ("STRING", {"default": "", "multiline": False}), # Name for saving new seeds
                "seed_to_load_name": (seed_load_options, {"default": "(None)"}), # Static dropdown for loading/deleting
            }
        }

    # Define the output types and names of the node.
    RETURN_TYPES = ("INT", "STRING")
    RETURN_NAMES = ("seed_output", "status_info")

    # The name of the function that will be executed when the node runs.
    FUNCTION = "execute"

    # The category under which the node will appear in the ComfyUI menu.
    CATEGORY = "MD_Nodes/Utility"

    # This node primarily outputs a seed, so it acts as an intermediate processor.
    OUTPUT_NODE = True

    def execute(self, seed_input, action, seed_name_input, seed_to_load_name):
        """
        The main execution method for the node.
        Handles saving, loading, and deleting seeds based on the 'action' input.
        """
        output_seed = seed_input
        # Reload seed names for counting and latest logic
        current_saved_seed_names = get_all_saved_seed_names()
        num_saved_seeds = len(current_saved_seed_names)

        status_message = (f"Input Seed: {seed_input}\n"
                          f"Saved Seeds Count: {num_saved_seeds}")
        
        print(f"MD Seed Saver: Processing. Input seed: {seed_input}")

        if action == "SAVE_CURRENT_SEED":
            save_name = seed_name_input.strip()
            if not save_name:
                timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
                save_name = f"seed_{seed_input}_{timestamp}"
                status_message += f"\nNo name provided. Generating default: '{save_name}'"
                print(f"MD Seed Saver: No name provided. Generating default name: '{save_name}'")

            original_save_name = save_name
            counter = 0
            while os.path.exists(get_seed_filepath(save_name)):
                counter += 1
                save_name = f"{original_save_name}_{counter}"
                status_message += f"\nName '{original_save_name}' exists. Trying '{save_name}'"
                print(f"MD Seed Saver: File for '{original_save_name}' already exists. Trying '{save_name}'")

            if save_seed_to_file(save_name, seed_input):
                status_message += f"\nAction: SAVED. Name: '{save_name}', Value: {seed_input}"
                print(f"MD Seed Saver: Successfully saved seed '{save_name}' with value {seed_input}")
                status_message += f"\nNote: Refresh browser to update dropdown list."
            else:
                status_message += f"\nError: Failed to save seed '{save_name}'."
                print(f"MD Seed Saver: Failed to save seed '{save_name}'.")

        elif action == "LOAD_SELECTED_SEED":
            if seed_to_load_name and seed_to_load_name != "(None)":
                loaded_seed = load_seed_from_file(seed_to_load_name)
                if loaded_seed is not None:
                    output_seed = loaded_seed
                    status_message += f"\nAction: LOADED. Name: '{seed_to_load_name}', Value: {loaded_seed}"
                    print(f"MD Seed Saver: Successfully loaded seed '{seed_to_load_name}' with value {loaded_seed}")
                else:
                    status_message += f"\nError: Seed '{seed_to_load_name}' not found or could not be loaded. Outputting input seed."
                    print(f"MD Seed Saver: Seed '{seed_to_load_name}' not found or could not be loaded. Outputting input seed.")
            else:
                status_message += f"\nInfo: No seed selected to load. Outputting input seed."
                print("MD Seed Saver: No seed selected to load. Outputting input seed.")

        elif action == "DELETE_SELECTED_SEED":
            if seed_to_load_name and seed_to_load_name != "(None)":
                if delete_seed_file(seed_to_load_name):
                    status_message += f"\nAction: DELETED. Name: '{seed_to_load_name}'"
                    print(f"MD Seed Saver: Successfully deleted seed '{seed_to_load_name}'.")
                    status_message += f"\nNote: Refresh browser to update dropdown list."
                else:
                    status_message += f"\nError: Seed '{seed_to_load_name}' not found for deletion or could not be deleted."
                    print(f"MD Seed Saver: Seed '{seed_to_load_name}' not found or could not be deleted. No action taken.")
            else:
                status_message += f"\nInfo: No seed selected for deletion. No action taken."
                print("MD Seed Saver: No seed selected for deletion. No action taken.")
        
        elif action == "LOAD_LATEST_SAVED_SEED":
            # Get all .txt files with their modification times
            seed_files_with_times = []
            for filename in os.listdir(OUTPUT_SEEDS_DIR):
                if filename.endswith(".txt"):
                    filepath = os.path.join(OUTPUT_SEEDS_DIR, filename)
                    try:
                        mod_time = os.path.getmtime(filepath)
                        seed_files_with_times.append((filename, mod_time))
                    except Exception as e:
                        print(f"WARNING: Could not get modification time for {filename}: {e}")

            if seed_files_with_times:
                # Sort by modification time (most recent first)
                seed_files_with_times.sort(key=lambda x: x[1], reverse=True)
                latest_filename = seed_files_with_times[0][0]
                latest_seed_name = os.path.splitext(latest_filename)[0] # Name without .txt

                loaded_seed = load_seed_from_file(latest_seed_name)
                if loaded_seed is not None:
                    output_seed = loaded_seed
                    status_message += f"\nAction: LOADED LATEST. Name: '{latest_seed_name}', Value: {loaded_seed}"
                    print(f"MD Seed Saver: Successfully loaded latest seed '{latest_seed_name}' with value {loaded_seed}")
                else:
                    status_message += f"\nError: Failed to load latest seed '{latest_seed_name}'. Outputting input seed."
                    print(f"MD Seed Saver: Failed to load latest seed '{latest_seed_name}'.")
            else:
                status_message += f"\nInfo: No saved seeds found to load the latest. Outputting input seed."
                print("MD Seed Saver: No saved seeds found to load the latest.")


        # Final status and output seed
        status_message += f"\nFinal Output Seed: {output_seed}"
        print(f"MD Seed Saver: Final output seed: {output_seed}")

        return (output_seed, status_message)

    @classmethod
    def IS_CHANGED(s, seed_input, action, seed_name_input, seed_to_load_name):
        """
        Tells ComfyUI when to re-execute the node.
        It should re-execute if any relevant input changes.
        """
        return (seed_input, action, seed_name_input, seed_to_load_name)


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
    print("Currently saved seeds:")
    for name in initial_seed_names:
        print(f"  - {name}")
else:
    print("No seeds currently saved.")
print("-" * 30) # Separator for clarity
