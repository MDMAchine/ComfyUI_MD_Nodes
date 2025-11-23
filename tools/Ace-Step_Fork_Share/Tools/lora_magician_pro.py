# lora_magician_pro.py

import os
import shutil
import sys
import datetime
import subprocess  # <-- Import subprocess to call external scripts
import argparse   # <-- Import argparse for the new section
import json     # <-- Import json for saving/loading training config
import multiprocessing # For num_workers calculation

# --- ANSI Escape Codes for Colors ---
RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
BLUE = "\033[94m"
WHITE = "\033[97m"

# Global variable for the log file object
log_file = None

# --- Constants for folder names ---
EMPTY_FOLDER_NAME = "empty"
TOO_SHORT_FOLDER_NAME = "too_short_prompts"
TOO_LONG_FOLDER_NAME = "too_long_prompts"
BACKUP_FOLDER_NAME = "_original_backups"

# --- Default for num_workers calculation ---
DEFAULT_NUM_WORKERS = max(1, min(8, multiprocessing.cpu_count() - 2))

def print_colored(text, color=RESET, bold=False, to_log=True):
    """Helper function to print colored text to console and optionally log."""
    prefix = color
    if bold:
        prefix += BOLD
    formatted_text = f"{prefix}{text}{RESET}"
    print(formatted_text)
    if to_log and log_file:
        # Remove ANSI codes for logging to keep it clean
        clean_text = text.replace(RESET, '').replace(BOLD, '').replace(GREEN, '').replace(YELLOW, '') \
                         .replace(RED, '').replace(CYAN, '').replace(MAGENTA, '').replace(BLUE, '') \
                         .replace(WHITE, '')
        # Ensure the log file is still open before writing
        try:
            if not log_file.closed:
                log_file.write(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {clean_text}\n")
        except ValueError: # Handle potential "I/O operation on closed file" if log_file state is somehow inconsistent
             pass # Silently fail on log write if file is closed, don't crash the program

def process_lora_files(directory=".", common_tags_str=None, blacklist_tags_str=None,
                        min_tags_per_line=None, max_tags_per_line=None, create_backups=False,
                        audio_extensions_str=None):
    """
    Processes text files and associated audio files for LoRA training.
    Args:
        directory (str): The directory to search for files.
        common_tags_str (str, optional): Comma-separated tags to add to _prompt.txt.
        blacklist_tags_str (str, optional): Comma-separated tags to remove from _prompt.txt.
        min_tags_per_line (int, optional): Minimum number of tags required per line in _prompt.txt.
        max_tags_per_line (int, optional): Maximum number of tags allowed per line in _prompt.txt.
        create_backups (bool): If True, creates a backup of original files before modification.
        audio_extensions_str (str, optional): Comma-separated audio file extensions (e.g., "mp3,wav,flac").
    """
    global log_file # Declare global to modify the global log_file variable
    # --- Setup Logging ---
    log_filename = f"lora_processing_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    try:
        log_file = open(os.path.join(directory, log_filename), 'w', encoding='utf-8')
        print_colored(f"\n📝 All actions will be logged to: {log_filename}", CYAN, to_log=False) # Don't log this message to file yet
    except Exception as e:
        print_colored(f"🚨 Error creating log file: {e}. Proceeding without detailed logging.", RED, bold=True, to_log=False)
        log_file = None # Ensure log_file is None if creation fails
    # Define and create necessary subfolders using constants
    empty_folder = os.path.join(directory, EMPTY_FOLDER_NAME)
    too_short_folder = os.path.join(directory, TOO_SHORT_FOLDER_NAME)
    too_long_folder = os.path.join(directory, TOO_LONG_FOLDER_NAME)
    backup_folder = os.path.join(directory, BACKUP_FOLDER_NAME)
    folders_to_create = [empty_folder, too_short_folder, too_long_folder]
    if create_backups:
        folders_to_create.append(backup_folder)
    # Pre-calculate the list of special folder paths for the file filtering check
    special_folder_paths = [empty_folder, too_short_folder, too_long_folder, backup_folder]
    
    for folder in folders_to_create:
        try:
            os.makedirs(folder, exist_ok=True)
            print_colored(f"🚀 Ensuring folder exists: '{os.path.basename(folder)}'", GREEN, to_log=True)
        except OSError as e:
            print_colored(f"🚨 Oh no! Could not create '{os.path.basename(folder)}' folder: {e}", RED, bold=True, to_log=True)
            print_colored("Please check directory permissions and try again.", RED, to_log=True)
            if log_file and not log_file.closed:
                try:
                    log_file.close()
                except:
                    pass
                log_file = None # Close log and disable further logging
            return
    # Parse common tags and blacklist tags
    common_tags = [tag.strip().lower() for tag in common_tags_str.split(',')] if common_tags_str else []
    blacklist_tags = [tag.strip().lower() for tag in blacklist_tags_str.split(',')] if blacklist_tags_str else []
    audio_extensions = [f".{ext.strip().lower()}" for ext in audio_extensions_str.split(',')] if audio_extensions_str else []
    if common_tags:
        print_colored(f"\n✨ Common magical tags to sprinkle: {', '.join(common_tags)}", YELLOW, to_log=True)
    else:
        print_colored("\nNo common tags specified for _prompt.txt files.", CYAN, to_log=True)
    if blacklist_tags:
        print_colored(f"🚫 Tags to banish from _prompt.txt files: {', '.join(blacklist_tags)}", RED, to_log=True)
    else:
        print_colored("No tags specified for blacklisting.", CYAN, to_log=True)
    if min_tags_per_line is not None and max_tags_per_line is not None:
        print_colored(f"📏 Prompt line length filter: {min_tags_per_line} to {max_tags_per_line} tags per line.", BLUE, to_log=True)
    elif min_tags_per_line is not None:
        print_colored(f"📏 Prompt line length filter: Minimum {min_tags_per_line} tags per line.", BLUE, to_log=True)
    elif max_tags_per_line is not None:
        print_colored(f"📏 Prompt line length filter: Maximum {max_tags_per_line} tags per line.", BLUE, to_log=True)
    else:
        print_colored("📏 No prompt line length filtering applied.", BLUE, to_log=True)
    if audio_extensions:
        print_colored(f"🎧 Audio file extensions to check: {', '.join(audio_extensions)}", BLUE, to_log=True)
    else:
        print_colored("🎧 No audio extensions provided. Audio files will not be moved based on text file status.", YELLOW, to_log=True)
    print_colored(f"\n🔍 Searching for treasures (files) in: '{directory}'...", BLUE, to_log=True)
    processed_prompt_count = 0
    processed_lyrics_count = 0
    moved_to_empty_count = 0
    moved_to_too_short_count = 0
    moved_to_too_long_count = 0
    moved_audio_count = 0
    # Statistics for _prompt.txt tag counts
    all_prompt_tag_counts = []
    # --- FIRST PASS: Process Text Files ---
    print_colored("\n--- PHASE 1: Processing Text Files (Prompts & Lyrics) ---", MAGENTA, bold=True, to_log=True)
    try:
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            # Ensure we are only processing actual files and not directories or our special folders
            # Use the pre-calculated list of special folder paths
            if os.path.isfile(filepath) and not any(filepath.startswith(sp) for sp in special_folder_paths):
                # --- Handle _prompt.txt files ---
                if filename.endswith("_prompt.txt"):
                    print_colored(f"\n--- 📝 Polishing prompt file: {filename} ---", MAGENTA, bold=True, to_log=True)
                    try:
                        if create_backups:
                            backup_filepath = os.path.join(backup_folder, f"{filename}.{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.bak")
                            shutil.copy2(filepath, backup_filepath)
                            print_colored(f"  💾 Created backup: {os.path.basename(backup_filepath)}", WHITE, to_log=True)
                        with open(filepath, 'r', encoding='utf-8') as f:
                            original_lines = f.readlines()
                        seen_final_lines = set()
                        final_processed_lines = []
                        lines_removed_count = 0
                        current_file_tag_counts = [] # For this specific prompt file
                        for line in original_lines:
                            stripped_line = line.strip()
                            if not stripped_line:
                                lines_removed_count += 1
                                continue # Skip empty lines
                            # Split, strip, lowercase, and filter out empty strings
                            current_tags = [tag.strip().lower() for tag in stripped_line.split(',') if tag.strip()]
                            # Apply blacklist: remove any blacklisted tags
                            filtered_tags = [tag for tag in current_tags if tag not in blacklist_tags]
                            # Use a set for efficient merging and uniqueness of tags within a line
                            merged_tags = set(filtered_tags)
                            # Add common tags to the set if they are not already present
                            for c_tag in common_tags:
                                if c_tag:
                                    merged_tags.add(c_tag)
                            # --- OPTIMIZATION: Simplified and more efficient tag sorting and filtering ---
                            # Reconstruct the line from the merged, unique, and sorted tags.
                            # clean_sorted_tags = [tag for tag in sorted(list(merged_tags)) if tag] # OLD
                            clean_sorted_tags = sorted(tag for tag in merged_tags if tag) # NEW
                            # --- END OF OPTIMIZATION ---
                            # Apply line length filtering check for this line
                            tag_count = len(clean_sorted_tags)
                            current_file_tag_counts.append(tag_count) # Add to current file's tag counts
                            if clean_sorted_tags:
                                # Check if this line itself passes the length filter
                                line_passes_length_filter = True
                                if min_tags_per_line is not None and tag_count < min_tags_per_line:
                                    line_passes_length_filter = False
                                if max_tags_per_line is not None and tag_count > max_tags_per_line:
                                    line_passes_length_filter = False
                                if line_passes_length_filter:
                                    new_line_content = ", ".join(clean_sorted_tags) + "\n"
                                    if new_line_content not in seen_final_lines:
                                        final_processed_lines.append(new_line_content)
                                        seen_final_lines.add(new_line_content)
                                    else:
                                        lines_removed_count += 1 # Count duplicates removed
                                else:
                                    # This line failed the length filter, so it's effectively removed from the file
                                    lines_removed_count += 1
                                    print_colored(f"  ➡️ Line removed due to length filter ({tag_count} tags): '{stripped_line}'", YELLOW, to_log=True)
                            else:
                                lines_removed_count += 1 # Count lines that became empty after processing
                        # Add tag counts for this file to overall statistics
                        all_prompt_tag_counts.extend(current_file_tag_counts)
                        # Determine final action for the file based on its content and overall length filtering
                        if not final_processed_lines:
                            print_colored(f"  🗑️ File '{filename}' became completely empty or was empty. Moving to '{empty_folder}'.", RED, to_log=True)
                            shutil.move(filepath, os.path.join(empty_folder, filename))
                            moved_to_empty_count += 1
                        elif min_tags_per_line is not None and any(count < min_tags_per_line for count in current_file_tag_counts):
                             # If any line was too short, move the whole file to too_short_folder
                            print_colored(f"  📏 File '{filename}' contains lines too short. Moving to '{too_short_folder}'.", YELLOW, to_log=True)
                            shutil.move(filepath, os.path.join(too_short_folder, filename))
                            moved_to_too_short_count += 1
                        elif max_tags_per_line is not None and any(count > max_tags_per_line for count in current_file_tag_counts):
                            # If any line was too long, move the whole file to too_long_folder
                            print_colored(f"  📏 File '{filename}' contains lines too long. Moving to '{too_long_folder}'.", YELLOW, to_log=True)
                            shutil.move(filepath, os.path.join(too_long_folder, filename))
                            moved_to_too_long_count += 1
                        else:
                            # Write the unique and tagged lines back to the original file
                            with open(filepath, 'w', encoding='utf-8') as f:
                                f.writelines(final_processed_lines)
                            print_colored(f"  ✅ File '{filename}' updated with {len(final_processed_lines)} unique, sparkling line(s).", GREEN, to_log=True)
                            processed_prompt_count += 1
                        if lines_removed_count > 0:
                            print_colored(f"  ✨ Removed {lines_removed_count} duplicate or empty/filtered lines from '{filename}'.", YELLOW, to_log=True)
                    except Exception as e:
                        print_colored(f"  🚨 Error processing '{filename}': {e}", RED, bold=True, to_log=True)
                # --- Handle _lyrics.txt files ---
                elif filename.endswith("_lyrics.txt"):
                    print_colored(f"\n--- 🎶 Checking lyrics file: {filename} ---", CYAN, bold=True, to_log=True)
                    try:
                        if create_backups:
                            backup_filepath = os.path.join(backup_folder, f"{filename}.{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.bak")
                            shutil.copy2(filepath, backup_filepath)
                            print_colored(f"  💾 Created backup: {os.path.basename(backup_filepath)}", WHITE, to_log=True)
                        with open(filepath, 'r', encoding='utf-8') as f:
                            content = f.read().strip()
                        if not content:
                            print_colored(f"  🗑️ File '{filename}' is empty! Moving to '{empty_folder}'.", RED, to_log=True)
                            shutil.move(filepath, os.path.join(empty_folder, filename))
                            moved_to_empty_count += 1
                        else:
                            print_colored(f"  👍 File '{filename}' is full of rhythm! No action needed.", GREEN, to_log=True)
                            processed_lyrics_count += 1
                    except Exception as e:
                        print_colored(f"  🚨 Error processing '{filename}': {e}", RED, bold=True, to_log=True)
                else:
                    print_colored(f"  ➡️ Skipping file (not a prompt or lyrics file): {filename}", BLUE, to_log=True)
    except FileNotFoundError:
        print_colored(f"\n🚨 Directory not found: '{directory}'. Please check the path and try again.", RED, bold=True, to_log=True)
        if log_file and not log_file.closed:
            try:
                log_file.close()
            except:
                pass
            log_file = None
        return
    except Exception as e:
        print_colored(f"\n🚨 An unexpected error occurred during file listing: {e}", RED, bold=True, to_log=True)
        if log_file and not log_file.closed:
            try:
                log_file.close()
            except:
                pass
            log_file = None
        return
    # --- SECOND PASS: Check and Move Audio Files ---
    if audio_extensions:
        print_colored("\n--- PHASE 2: Checking and Moving Audio Files ---", MAGENTA, bold=True, to_log=True)
        try:
            for filename in os.listdir(directory):
                filepath = os.path.join(directory, filename)
                if os.path.isfile(filepath):
                    base_name, ext = os.path.splitext(filename)
                    if ext.lower() in audio_extensions:
                        print_colored(f"\n--- 🎧 Checking audio file: {filename} ---", CYAN, bold=True, to_log=True)
                        # Construct expected text file paths
                        prompt_filepath = os.path.join(directory, f"{base_name}_prompt.txt")
                        lyrics_filepath = os.path.join(directory, f"{base_name}_lyrics.txt")
                        # Check if prompt AND lyrics files are missing from the current directory
                        # This implies they were moved by the first processing pass
                        prompt_missing = not os.path.exists(prompt_filepath)
                        lyrics_missing = not os.path.exists(lyrics_filepath)
                        # Move audio file ONLY if BOTH prompt AND lyrics are missing
                        if prompt_missing and lyrics_missing:
                            print_colored(f"  🗑️ Audio file '{filename}' is missing *both* its _prompt.txt and _lyrics.txt. Moving to '{empty_folder}'.", RED, to_log=True)
                            shutil.move(filepath, os.path.join(empty_folder, filename))
                            moved_audio_count += 1
                        else:
                            print_colored(f"  👍 Audio file '{filename}' has at least one valid text file. No action needed.", GREEN, to_log=True)
        except Exception as e:
            print_colored(f"\n🚨 An unexpected error occurred during audio file processing: {e}", RED, bold=True, to_log=True)
    print_colored("\n--- 🎉 Processing Complete! Here's the Grand Summary! 🎉 ---", MAGENTA, bold=True, to_log=True)
    print_colored(f"  📝 _prompt.txt files polished: {processed_prompt_count}", GREEN, to_log=True)
    print_colored(f"  🎶 _lyrics.txt files checked: {processed_lyrics_count}", GREEN, to_log=True)
    print_colored(f"  🗑️ Files moved to '{empty_folder}': {moved_to_empty_count}", YELLOW, to_log=True)
    print_colored(f"  📏 Files moved to '{too_short_folder}': {moved_to_too_short_count}", YELLOW, to_log=True)
    print_colored(f"  📏 Files moved to '{too_long_folder}': {moved_to_too_long_count}", YELLOW, to_log=True)
    print_colored(f"  🎧 Audio files moved to '{empty_folder}': {moved_audio_count}", YELLOW, to_log=True)
    if create_backups:
        print_colored(f"  💾 Original file backups created in '{backup_folder}'.", WHITE, to_log=True)
    print_colored("----------------------------------------------------------", MAGENTA, bold=True, to_log=True)
    # Prompt Tag Statistics
    if all_prompt_tag_counts:
        min_tags = min(all_prompt_tag_counts)
        max_tags = max(all_prompt_tag_counts)
        avg_tags = sum(all_prompt_tag_counts) / len(all_prompt_tag_counts)
        print_colored("\n📊 _prompt.txt Tag Count Statistics:", BLUE, bold=True, to_log=True)
        print_colored(f"  Minimum tags per line: {min_tags}", BLUE, to_log=True)
        print_colored(f"  Maximum tags per line: {max_tags}", BLUE, to_log=True)
        print_colored(f"  Average tags per line: {avg_tags:.2f}", BLUE, to_log=True)
        print_colored("----------------------------------------------------------", MAGENTA, bold=True, to_log=True)
    else:
        print_colored("\n📊 No _prompt.txt files processed to generate tag count statistics.", BLUE, to_log=True)
    print_colored("Thanks for using the LoRA File Magician! Happy training! ✨", CYAN, bold=True, to_log=True)
    # Close the log file at the very end
    if log_file and not log_file.closed:
        try:
            log_file.close()
        except:
            pass

# --- NEW SECTION: Audio Generation Integration ---
def find_generation_script():
    """Attempts to find the generate_prompts_lyrics.py script based on expected relative path."""
    # Get the directory where this script (lora_magician_pro.py) is located
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    print_colored(f"🔍 This script is located in: {current_script_dir}", BLUE, to_log=True)

    # Calculate the path to the ACE-Step directory (sibling to Tools) and the script within it
    # Assuming structure: .../Ace-Step_Fork/Tools/lora_magician_pro.py
    #                     .../Ace-Step_Fork/ACE-Step/generate_prompts_lyrics.py
    ace_step_dir = os.path.join(os.path.dirname(current_script_dir), "ACE-Step")
    gen_script_path = os.path.join(ace_step_dir, "generate_prompts_lyrics.py")

    print_colored(f"🔍 Looking for generation script at: {gen_script_path}", BLUE, to_log=True)

    if os.path.exists(gen_script_path):
        print_colored(f"✅ Found generation script at: {gen_script_path}", GREEN, to_log=True)
        return gen_script_path
    else:
        print_colored(f"⚠️ Generation script not found at expected path: {gen_script_path}", YELLOW, to_log=True)
        return None

def generate_audio_prompts_cli():
    """Handles the command-line interface for the audio prompt generation."""
    print_colored("\n--- 🎵 PHASE 0: Audio Prompt & Lyrics Generation ---", MAGENTA, bold=True, to_log=True)

    # Ask user if they want to generate prompts/lyrics
    gen_choice = input(f"{YELLOW}Do you want to generate prompts and lyrics from audio files first? (yes/no): {RESET}").strip().lower()
    if gen_choice != 'yes':
        print_colored("Okay, skipping audio prompt generation.", CYAN, to_log=True)
        return

    # --- Find the generation script ---
    gen_script_path = find_generation_script()

    # If not found automatically, ask the user for the path
    if not gen_script_path:
        print_colored("Please provide the full path to 'generate_prompts_lyrics.py':", YELLOW, to_log=True)
        user_provided_path = input(f"{YELLOW}Path to generate_prompts_lyrics.py: {RESET}").strip()
        if user_provided_path:
            # Normalize the path for the current OS
            user_provided_path = os.path.normpath(user_provided_path)
            if os.path.exists(user_provided_path) and os.path.isfile(user_provided_path):
                gen_script_path = user_provided_path
                print_colored(f"✅ Using generation script provided by user: {gen_script_path}", GREEN, to_log=True)
            else:
                print_colored(f"🚨 The path '{user_provided_path}' is invalid or the file does not exist.", RED, bold=True, to_log=True)
                return # Exit the generation function if path is invalid
        else:
            print_colored("No path provided. Cannot run audio generation.", RED, bold=True, to_log=True)
            return # Exit the generation function if no path is given

    # Create a parser for the generation script arguments (for help text reference)
    gen_parser = argparse.ArgumentParser(description="Generate prompts and lyrics from audio files.", add_help=False)
    # Add arguments specific to the generation script (for reference)
    gen_parser.add_argument("--data_dir", type=str, default=".", help="Directory containing audio files")
    gen_parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    gen_parser.add_argument("--lyrics", action="store_true", help="Include lyrics transcription")
    gen_parser.add_argument("--model_type", type=str, default="qwen", choices=["qwen", "ollama"],
                           help="Model to use for processing")
    gen_parser.add_argument("--qwen_model", type=str, default="Qwen/Qwen2.5-Omni-7B-GPTQ-Int4",
                           help="Qwen model path or HuggingFace ID")
    gen_parser.add_argument("--ollama_model", type=str, default="qwen2.5",
                           help="Ollama model name")
    gen_parser.add_argument("--recursive", action="store_true",
                           help="Process subdirectories recursively")
    gen_parser.add_argument("--no_auto_pull", action="store_true",
                           help="Disable automatic pulling of Ollama models")
    gen_parser.add_argument("--no_auto_install", action="store_true",
                           help="Disable automatic installation of ollama package")


    # Collect arguments from user
    print_colored("\n--- 🎛️  Audio Generation Configuration ---", CYAN, bold=True, to_log=True)

    # Data directory
    data_dir = input(f"{YELLOW}Enter the directory containing audio files (default: '.'): {RESET}").strip()
    if not data_dir:
        data_dir = "."

    # Overwrite
    overwrite_choice = input(f"{YELLOW}Overwrite existing _prompt.txt/_lyrics.txt files? (yes/no, default: no): {RESET}").strip().lower()
    overwrite = overwrite_choice == 'yes'

    # Lyrics
    lyrics_choice = input(f"{YELLOW}Generate lyrics transcription? (yes/no, default: yes): {RESET}").strip().lower()
    lyrics = lyrics_choice != 'no' # Default to True

    # Model type
    while True:
        model_type = input(f"{YELLOW}Choose model type (qwen/ollama, default: qwen): {RESET}").strip().lower()
        if not model_type:
            model_type = "qwen"
        if model_type in ["qwen", "ollama"]:
            break
        else:
            print_colored("Invalid model type. Please enter 'qwen' or 'ollama'.", RED, bold=True)

    # Model-specific options
    qwen_model = "Qwen/Qwen2.5-Omni-7B-GPTQ-Int4"
    ollama_model = "qwen2.5"
    if model_type == "qwen":
        qwen_model_input = input(f"{YELLOW}Enter Qwen model path/ID (default: 'Qwen/Qwen2.5-Omni-7B-GPTQ-Int4'): {RESET}").strip()
        if qwen_model_input:
            qwen_model = qwen_model_input
    else: # ollama
        ollama_model_input = input(f"{YELLOW}Enter Ollama model name (default: 'qwen2.5'): {RESET}").strip()
        if ollama_model_input:
            ollama_model = ollama_model_input

    # Recursive
    recursive_choice = input(f"{YELLOW}Process subdirectories recursively? (yes/no, default: no): {RESET}").strip().lower()
    recursive = recursive_choice == 'yes'

    # Auto-pull
    auto_pull_choice = input(f"{YELLOW}Allow automatic pulling of Ollama models? (yes/no, default: yes): {RESET}").strip().lower()
    no_auto_pull = auto_pull_choice == 'no'

    # Auto-install
    auto_install_choice = input(f"{YELLOW}Allow automatic installation of Ollama Python package? (yes/no, default: yes): {RESET}").strip().lower()
    no_auto_install = auto_install_choice == 'no'

    # Build the command
    cmd = [sys.executable, gen_script_path]
    cmd.extend(["--data_dir", data_dir])
    if overwrite:
        cmd.append("--overwrite")
    if lyrics:
        cmd.append("--lyrics")
    cmd.extend(["--model_type", model_type])
    if model_type == "qwen":
        cmd.extend(["--qwen_model", qwen_model])
    else: # ollama
        cmd.extend(["--ollama_model", ollama_model])
    if recursive:
        cmd.append("--recursive")
    if no_auto_pull:
        cmd.append("--no_auto_pull")
    if no_auto_install:
        cmd.append("--no_auto_install")

    print_colored(f"\n🚀 Running audio generation command:", GREEN, to_log=True)
    print_colored(f"   {' '.join(cmd)}", WHITE, to_log=True)

    # Execute the command
    try:
        # Use subprocess.run to capture output in real-time
        # shell=True on Windows can sometimes help with path resolution, but can be a security risk with untrusted input.
        # Given we are building the command ourselves, it should be safe.
        # However, shell=False is generally preferred.
        result = subprocess.run(cmd, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        print_colored(f"\n✅ Audio generation completed successfully!", GREEN, bold=True, to_log=True)
        # Optionally print the output (might be verbose)
        # print(result.stdout)
    except subprocess.CalledProcessError as e:
        print_colored(f"\n🚨 Audio generation failed with return code {e.returncode}", RED, bold=True, to_log=True)
        print_colored(f"   Output: {e.output}", RED, to_log=True)
        print_colored("   Please check the error above and ensure your generation script is working correctly.", RED, to_log=True)
        # Decide whether to continue or exit
        cont_choice = input(f"{YELLOW}Do you want to continue with the rest of the processing? (yes/no): {RESET}").strip().lower()
        if cont_choice != 'yes':
            print_colored("Exiting LoRA Magician Pro.", RED, to_log=True)
            sys.exit(1)
    except FileNotFoundError:
        print_colored(f"\n🚨 Python executable or script not found. Cannot run generation.", RED, bold=True, to_log=True)
        sys.exit(1)
    except Exception as e:
        print_colored(f"\n🚨 An unexpected error occurred while running the generation script: {e}", RED, bold=True, to_log=True)
        sys.exit(1)

# --- NEW SECTION: Training Integration ---
def find_training_script():
    """Attempts to find the trainer_new.py script based on expected relative path."""
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    print_colored(f"🔍 Looking for trainer script relative to: {current_script_dir}", BLUE, to_log=True)

    # Assuming structure: .../Ace-Step_Fork/Tools/lora_magician_pro.py
    #                     .../Ace-Step_Fork/ACE-Step/trainer_new.py
    ace_step_dir = os.path.join(os.path.dirname(current_script_dir), "ACE-Step")
    train_script_path = os.path.join(ace_step_dir, "trainer_new.py")

    print_colored(f"🔍 Looking for trainer script at: {train_script_path}", BLUE, to_log=True)

    if os.path.exists(train_script_path):
        print_colored(f"✅ Found trainer script at: {train_script_path}", GREEN, to_log=True)
        return train_script_path
    else:
        print_colored(f"⚠️ Trainer script not found at expected path: {train_script_path}", YELLOW, to_log=True)
        return None

def load_training_config(config_path):
    """Load training configuration from a JSON file."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print_colored(f"✅ Loaded training configuration from: {config_path}", GREEN, to_log=True)
        return config
    except Exception as e:
        print_colored(f"⚠️ Could not load training config from {config_path}: {e}. Using defaults.", YELLOW, to_log=True)
        return {}

def save_training_config(config, config_path):
    """Save training configuration to a JSON file."""
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        print_colored(f"✅ Saved training configuration to: {config_path}", GREEN, to_log=True)
    except Exception as e:
        print_colored(f"⚠️ Could not save training config to {config_path}: {e}", YELLOW, to_log=True)

def configure_training_cli(working_dir):
    """Handles the command-line interface for configuring and launching training."""
    print_colored("\n--- 🏋️ PHASE 3: Training Configuration & Launch ---", MAGENTA, bold=True, to_log=True)

    train_choice = input(f"{YELLOW}Do you want to configure and launch the LoRA training now? (yes/no): {RESET}").strip().lower()
    if train_choice != 'yes':
        print_colored("Okay, skipping training launch.", CYAN, to_log=True)
        return

    train_script_path = find_training_script()
    if not train_script_path:
        print_colored("Please provide the full path to 'trainer_new.py':", YELLOW, to_log=True)
        user_provided_path = input(f"{YELLOW}Path to trainer_new.py: {RESET}").strip()
        if user_provided_path:
            user_provided_path = os.path.normpath(user_provided_path)
            if os.path.exists(user_provided_path) and os.path.isfile(user_provided_path):
                train_script_path = user_provided_path
                print_colored(f"✅ Using trainer script provided by user: {train_script_path}", GREEN, to_log=True)
            else:
                print_colored(f"🚨 The path '{user_provided_path}' is invalid or the file does not exist.", RED, bold=True, to_log=True)
                return
        else:
            print_colored("No path provided. Cannot launch training.", RED, bold=True, to_log=True)
            return

    # --- Training Configuration ---
    print_colored("\n--- 🎛️  Training Configuration ---", CYAN, bold=True, to_log=True)

    # Load previous config if exists
    config_file_path = os.path.join(working_dir, "lora_training_config.json")
    config = load_training_config(config_file_path)

    # --- Essential Arguments ---
    # dataset_path (often the same as working_dir after processing)
    dataset_path = config.get("dataset_path", working_dir) # Default to working dir
    dataset_path_input = input(f"{YELLOW}Dataset path (HDF5 files, default: '{dataset_path}'): {RESET}").strip()
    if dataset_path_input:
        dataset_path = dataset_path_input

    # checkpoint_dir (base model)
    checkpoint_dir = config.get("checkpoint_dir", "")
    checkpoint_dir_input = input(f"{YELLOW}Base model checkpoint directory (required): {RESET}").strip()
    if checkpoint_dir_input:
        checkpoint_dir = checkpoint_dir_input
    if not checkpoint_dir:
         print_colored("🚨 Base model checkpoint directory is required for training.", RED, bold=True, to_log=True)
         return

    # lora_config_path
    lora_config_path = config.get("lora_config_path", "./config/lora_config_transformer_only.json")
    lora_config_path_input = input(f"{YELLOW}LoRA config path (default: '{lora_config_path}'): {RESET}").strip()
    if lora_config_path_input:
        lora_config_path = lora_config_path_input

    # exp_name
    exp_name = config.get("exp_name", "ace_step_lora_run")
    exp_name_input = input(f"{YELLOW}Experiment name (default: '{exp_name}'): {RESET}").strip()
    if exp_name_input:
        exp_name = exp_name_input

    # output_dir
    output_dir = config.get("output_dir", "./runs")
    output_dir_input = input(f"{YELLOW}Output directory for logs/checkpoints (default: '{output_dir}'): {RESET}").strip()
    if output_dir_input:
        output_dir = output_dir_input

    # --- Optimizer & Hyperparameters ---
    print_colored("\n--- Optimizer & Hyperparameters ---", CYAN, to_log=True)
    optimizer = config.get("optimizer", "adamw")
    while True:
        optimizer_input = input(f"{YELLOW}Optimizer (adamw/prodigy, default: '{optimizer}'): {RESET}").strip().lower()
        if not optimizer_input:
            break
        if optimizer_input in ["adamw", "prodigy"]:
            optimizer = optimizer_input
            break
        else:
            print_colored("Invalid optimizer. Please enter 'adamw' or 'prodigy'.", RED, bold=True)

    learning_rate = config.get("learning_rate", 1e-4 if optimizer == "adamw" else 1.0)
    try:
        lr_input = input(f"{YELLOW}Learning rate (default: {learning_rate} for {optimizer}): {RESET}").strip()
        if lr_input:
            learning_rate = float(lr_input)
    except ValueError:
        print_colored("Invalid learning rate. Using default.", RED, to_log=True)

    # Prodigy specific
    prodigy_slice_p = config.get("prodigy_slice_p", 11)
    if optimizer == "prodigy":
        try:
            slice_p_input = input(f"{YELLOW}Prodigy slice_p (default: {prodigy_slice_p}): {RESET}").strip()
            if slice_p_input:
                prodigy_slice_p = int(slice_p_input)
        except ValueError:
            print_colored("Invalid slice_p. Using default.", RED, to_log=True)

    warmup_steps = config.get("warmup_steps", 100 if optimizer == "adamw" else 0) # Default 0 for Prodigy
    try:
        warmup_input = input(f"{YELLOW}Warmup steps (default: {warmup_steps} for {optimizer}): {RESET}").strip()
        if warmup_input:
            warmup_steps = int(warmup_input)
    except ValueError:
        print_colored("Invalid warmup steps. Using default.", RED, to_log=True)

    max_steps = config.get("max_steps", 5000)
    try:
        max_steps_input = input(f"{YELLOW}Max training steps (default: {max_steps}): {RESET}").strip()
        if max_steps_input:
            max_steps = int(max_steps_input)
    except ValueError:
        print_colored("Invalid max steps. Using default.", RED, to_log=True)

    # --- Data & Performance ---
    print_colored("\n--- Data & Performance ---", CYAN, to_log=True)
    batch_size = config.get("batch_size", 1)
    try:
        batch_input = input(f"{YELLOW}Batch size (default: {batch_size}): {RESET}").strip()
        if batch_input:
            batch_size = int(batch_input)
    except ValueError:
        print_colored("Invalid batch size. Using default.", RED, to_log=True)

    # Use the pre-calculated DEFAULT_NUM_WORKERS constant
    num_workers = config.get("num_workers", DEFAULT_NUM_WORKERS)
    try:
        workers_input = input(f"{YELLOW}Number of data loader workers (default: {DEFAULT_NUM_WORKERS}, 0 disables multiprocessing): {RESET}").strip()
        if workers_input:
            num_workers = int(workers_input)
    except ValueError:
        print_colored("Invalid num_workers. Using default.", RED, to_log=True)

    # --- Advanced Options (Optional) ---
    print_colored("\n--- Advanced Options ---", CYAN, to_log=True)
    show_advanced = input(f"{YELLOW}Configure advanced options (tag dropout, gradient clipping, etc.)? (yes/no, default: no): {RESET}").strip().lower()
    tag_dropout = config.get("tag_dropout", 0.5)
    speaker_dropout = config.get("speaker_dropout", 0.05)
    lyrics_dropout = config.get("lyrics_dropout", 0.05)
    accumulate_grad_batches = config.get("accumulate_grad_batches", 2)
    gradient_clip_val = config.get("gradient_clip_val", 1.0)
    gradient_clip_algorithm = config.get("gradient_clip_algorithm", "norm")
    save_every_n_train_steps = config.get("save_every_n_train_steps", 500)

    if show_advanced == 'yes':
        try:
            tag_dropout_input = input(f"{YELLOW}Tag dropout (default: {tag_dropout}): {RESET}").strip()
            if tag_dropout_input: tag_dropout = float(tag_dropout_input)

            speaker_dropout_input = input(f"{YELLOW}Speaker dropout (default: {speaker_dropout}): {RESET}").strip()
            if speaker_dropout_input: speaker_dropout = float(speaker_dropout_input)

            lyrics_dropout_input = input(f"{YELLOW}Lyrics dropout (default: {lyrics_dropout}): {RESET}").strip()
            if lyrics_dropout_input: lyrics_dropout = float(lyrics_dropout_input)

            accumulate_input = input(f"{YELLOW}Accumulate gradient batches (default: {accumulate_grad_batches}): {RESET}").strip()
            if accumulate_input: accumulate_grad_batches = int(accumulate_input)

            clip_val_input = input(f"{YELLOW}Gradient clip value (default: {gradient_clip_val}): {RESET}").strip()
            if clip_val_input: gradient_clip_val = float(clip_val_input)

            clip_alg_input = input(f"{YELLOW}Gradient clip algorithm (norm/value, default: '{gradient_clip_algorithm}'): {RESET}").strip().lower()
            if clip_alg_input in ['norm', 'value']: gradient_clip_algorithm = clip_alg_input

            save_steps_input = input(f"{YELLOW}Save checkpoint every N steps (default: {save_every_n_train_steps}): {RESET}").strip()
            if save_steps_input: save_every_n_train_steps = int(save_steps_input)

        except ValueError as e:
            print_colored(f"Invalid advanced option value: {e}. Using defaults for advanced options.", RED, to_log=True)


    # --- Resume Options ---
    print_colored("\n--- Resume Options ---", CYAN, to_log=True)
    last_lora_path = config.get("last_lora_path", "")
    last_lora_path_input = input(f"{YELLOW}Path to initial LoRA weights (.safetensors) to load (optional): {RESET}").strip()
    if last_lora_path_input:
        last_lora_path = last_lora_path_input

    resume_full_ckpt_path = config.get("resume_full_ckpt_path", "")
    resume_full_ckpt_path_input = input(f"{YELLOW}Path to full trainer checkpoint (.ckpt) to resume from (optional): {RESET}").strip()
    if resume_full_ckpt_path_input:
        resume_full_ckpt_path = resume_full_ckpt_path_input

    wandb_run_id = config.get("wandb_run_id", "")
    wandb_run_id_input = input(f"{YELLOW}W&B Run ID to resume (optional): {RESET}").strip()
    if wandb_run_id_input:
        wandb_run_id = wandb_run_id_input

    # --- Build the command ---
    cmd = [sys.executable, train_script_path]
    # Essential
    cmd.extend(["--dataset_path", dataset_path])
    cmd.extend(["--checkpoint_dir", checkpoint_dir])
    cmd.extend(["--lora_config_path", lora_config_path])
    cmd.extend(["--exp_name", exp_name])
    cmd.extend(["--output_dir", output_dir])
    # Optimizer & Hyperparams
    cmd.extend(["--optimizer", optimizer])
    cmd.extend(["--learning_rate", str(learning_rate)])
    cmd.extend(["--warmup_steps", str(warmup_steps)])
    cmd.extend(["--max_steps", str(max_steps)])
    # Data & Performance
    cmd.extend(["--batch_size", str(batch_size)])
    cmd.extend(["--num_workers", str(num_workers)]) # Explicitly pass num_workers
    # Advanced
    cmd.extend(["--tag_dropout", str(tag_dropout)])
    cmd.extend(["--speaker_dropout", str(speaker_dropout)])
    cmd.extend(["--lyrics_dropout", str(lyrics_dropout)])
    cmd.extend(["--accumulate_grad_batches", str(accumulate_grad_batches)])
    cmd.extend(["--gradient_clip_val", str(gradient_clip_val)])
    cmd.extend(["--gradient_clip_algorithm", gradient_clip_algorithm])
    cmd.extend(["--save_every_n_train_steps", str(save_every_n_train_steps)])
    # Prodigy specific
    if optimizer == "prodigy":
        cmd.extend(["--prodigy_slice_p", str(prodigy_slice_p)])
    # Resume
    if last_lora_path:
        cmd.extend(["--last_lora_path", last_lora_path])
    if resume_full_ckpt_path:
        cmd.extend(["--resume_full_ckpt_path", resume_full_ckpt_path])
    if wandb_run_id:
        cmd.extend(["--wandb_run_id", wandb_run_id])

    # Save config for next time
    config_to_save = {
        "dataset_path": dataset_path,
        "checkpoint_dir": checkpoint_dir,
        "lora_config_path": lora_config_path,
        "exp_name": exp_name,
        "output_dir": output_dir,
        "optimizer": optimizer,
        "learning_rate": learning_rate,
        "warmup_steps": warmup_steps,
        "max_steps": max_steps,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "tag_dropout": tag_dropout,
        "speaker_dropout": speaker_dropout,
        "lyrics_dropout": lyrics_dropout,
        "accumulate_grad_batches": accumulate_grad_batches,
        "gradient_clip_val": gradient_clip_val,
        "gradient_clip_algorithm": gradient_clip_algorithm,
        "save_every_n_train_steps": save_every_n_train_steps,
        "prodigy_slice_p": prodigy_slice_p if optimizer == "prodigy" else 11, # Save even if not used
        "last_lora_path": last_lora_path,
        "resume_full_ckpt_path": resume_full_ckpt_path,
        "wandb_run_id": wandb_run_id,
    }
    save_training_config(config_to_save, config_file_path)

    print_colored(f"\n🚀 Running training command:", GREEN, to_log=True)
    print_colored(f"   {' '.join(cmd)}", WHITE, to_log=True)

    launch_choice = input(f"{YELLOW}\nLaunch training with these settings? (yes/no): {RESET}").strip().lower()
    if launch_choice != 'yes':
        print_colored("Training launch cancelled.", CYAN, to_log=True)
        return

    try:
        # Launch training in the foreground. User can Ctrl+C to stop.
        print_colored(f"\n🚀 Starting training... (Press Ctrl+C in this terminal to stop)", GREEN, bold=True, to_log=True)
        result = subprocess.run(cmd, check=True) # Don't capture output to see live logs
        print_colored(f"\n🎉 Training completed successfully!", GREEN, bold=True, to_log=True)
    except subprocess.CalledProcessError as e:
        print_colored(f"\n🚨 Training failed with return code {e.returncode}", RED, bold=True, to_log=True)
        # Note: stdout/stderr not captured, so user sees live logs in terminal
        print_colored("   Please check the terminal output and logs in the output directory.", RED, to_log=True)
    except FileNotFoundError:
        print_colored(f"\n🚨 Python executable or trainer script not found.", RED, bold=True, to_log=True)
    except KeyboardInterrupt:
        print_colored(f"\n🛑 Training launch was cancelled by user (Ctrl+C).", YELLOW, bold=True, to_log=True)
    except Exception as e:
        print_colored(f"\n🚨 An unexpected error occurred while launching training: {e}", RED, bold=True, to_log=True)

def main():
    """Main function to orchestrate the workflow."""
    print_colored("\n" + "="*50, MAGENTA)
    print_colored("    Welcome to the LoRA File Magician Pro!    ", MAGENTA, bold=True)
    print_colored("    Let's clean up your training data!        ", MAGENTA, bold=True)
    print_colored("="*50 + "\n", MAGENTA)

    # --- PHASE 0: Audio Generation (New) ---
    generate_audio_prompts_cli()

    # --- PHASE 1 & 2: Existing Processing ---
    # --- Prompt for working directory ---
    while True:
        working_dir = input(f"{YELLOW}Enter the working directory (e.g., './my_lora_data' or just '.' for current): {RESET}").strip()
        if not working_dir:
            working_dir = "." # Default to current directory if input is empty
            print_colored("Using current directory ('.').", CYAN)
        if os.path.isdir(working_dir):
            print_colored(f"Great! We'll work in: {BOLD}{working_dir}{RESET}", GREEN)
            break
        else:
            print_colored(f"Oops! '{working_dir}' is not a valid directory. Please try again.", RED, bold=True)

    # --- Prompt for common tags ---
    add_tags_choice = input(f"{YELLOW}Do you want to add common tags to all _prompt.txt files? (yes/no): {RESET}").strip().lower()
    common_tags_input = None
    if add_tags_choice == 'yes':
        common_tags_input = input(f"{YELLOW}Enter common tags, comma-separated (e.g., 'activate, mdmachine, banger'): {RESET}").strip()
        if not common_tags_input:
            print_colored("No tags entered. Skipping common tag insertion.", CYAN)
            common_tags_input = None
    else:
        print_colored("Okay, no common tags will be added.", CYAN)

    # --- Prompt for blacklist tags ---
    blacklist_choice = input(f"{YELLOW}Do you want to remove specific 'blacklist' tags from _prompt.txt files? (yes/no): {RESET}").strip().lower()
    blacklist_tags_input = None
    if blacklist_choice == 'yes':
        blacklist_tags_input = input(f"{YELLOW}Enter tags to remove, comma-separated (e.g., 'bad_quality, ugly, watermark'): {RESET}").strip()
        if not blacklist_tags_input:
            print_colored("No tags entered for blacklisting. Skipping tag removal.", CYAN)
            blacklist_tags_input = None
    else:
        print_colored("Okay, no tags will be blacklisted.", CYAN)

    # --- Prompt for line length filtering ---
    filter_length_choice = input(f"{YELLOW}Do you want to filter _prompt.txt files by number of tags per line? (yes/no): {RESET}").strip().lower()
    min_tags = None
    max_tags = None
    if filter_length_choice == 'yes':
        while True:
            try:
                min_input = input(f"{YELLOW}Enter minimum tags per line (leave empty for no minimum): {RESET}").strip()
                if min_input:
                    min_tags = int(min_input)
                    if min_tags < 0: raise ValueError
                break
            except ValueError:
                print_colored("Invalid input. Please enter a positive whole number or leave empty.", RED, bold=True)
        while True:
            try:
                max_input = input(f"{YELLOW}Enter maximum tags per line (leave empty for no maximum): {RESET}").strip()
                if max_input:
                    max_tags = int(max_input)
                    if max_tags < 0: raise ValueError
                break
            except ValueError:
                print_colored("Invalid input. Please enter a positive whole number or leave empty.", RED, bold=True)
        if min_tags is None and max_tags is None:
            print_colored("No min or max tags specified. Skipping line length filtering.", CYAN)
    else:
        print_colored("Okay, no line length filtering will be applied.", CYAN)

    # --- Prompt for backup ---
    backup_choice = input(f"{YELLOW}Do you want to create backups of original files before modification? (yes/no): {RESET}").strip().lower()
    create_backups_input = (backup_choice == 'yes')
    if create_backups_input:
        print_colored("Great! Backups will be created.", GREEN)
    else:
        print_colored("No backups will be created. Proceeding with caution!", YELLOW)

    # --- Prompt for audio extensions ---
    # --- ENHANCEMENT: Clearer prompt about file extensions ---
    # OLD: audio_ext_input = input(f"{YELLOW}Enter common audio file extensions, comma-separated (e.g., 'mp3,wav,flac,ogg'): {RESET}").strip()
    audio_ext_input = input(f"{YELLOW}Enter common audio file extensions (without dots), comma-separated (e.g., 'mp3,wav,flac,ogg'): {RESET}").strip()
    # --- END OF ENHANCEMENT ---
    if not audio_ext_input:
        print_colored("No audio extensions entered. Audio files will not be moved based on text file status.", CYAN)
        audio_ext_input = None

    # --- Run the processing function with user inputs ---
    process_lora_files(
        directory=working_dir,
        common_tags_str=common_tags_input,
        blacklist_tags_str=blacklist_tags_input,
        min_tags_per_line=min_tags,
        max_tags_per_line=max_tags,
        create_backups=create_backups_input,
        audio_extensions_str=audio_ext_input
    )

    # --- PHASE 3: Training Configuration & Launch (New) ---
    configure_training_cli(working_dir)

if __name__ == "__main__":
    main() # Call the main function
