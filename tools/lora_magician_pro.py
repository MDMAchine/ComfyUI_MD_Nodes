import os
import shutil
import sys
import datetime # For timestamps in logs and backups

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
        log_file.write(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {clean_text}\n")

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

    # Define and create necessary subfolders
    empty_folder = os.path.join(directory, "empty")
    too_short_folder = os.path.join(directory, "too_short_prompts")
    too_long_folder = os.path.join(directory, "too_long_prompts")
    backup_folder = os.path.join(directory, "_original_backups")

    folders_to_create = [empty_folder, too_short_folder, too_long_folder]
    if create_backups:
        folders_to_create.append(backup_folder)

    for folder in folders_to_create:
        try:
            os.makedirs(folder, exist_ok=True)
            print_colored(f"🚀 Ensuring folder exists: '{os.path.basename(folder)}'", GREEN, to_log=True)
        except OSError as e:
            print_colored(f"🚨 Oh no! Could not create '{os.path.basename(folder)}' folder: {e}", RED, bold=True, to_log=True)
            print_colored("Please check directory permissions and try again.", RED, to_log=True)
            if log_file: log_file.close(); log_file = None # Close log and disable further logging
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
            if os.path.isfile(filepath) and \
               not any(filepath.startswith(f) for f in [empty_folder, too_short_folder, too_long_folder, backup_folder]):

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

                            # Reconstruct the line from the merged, unique, and sorted tags.
                            clean_sorted_tags = [tag for tag in sorted(list(merged_tags)) if tag]

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
        if log_file: log_file.close(); log_file = None
        return
    except Exception as e:
        print_colored(f"\n🚨 An unexpected error occurred during file listing: {e}", RED, bold=True, to_log=True)
        if log_file: log_file.close(); log_file = None
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
    if log_file:
        log_file.close()

if __name__ == "__main__":
    print_colored("\n" + "="*50, MAGENTA)
    print_colored("    Welcome to the LoRA File Magician!    ", MAGENTA, bold=True)
    print_colored("    Let's clean up your training data!    ", MAGENTA, bold=True)
    print_colored("="*50 + "\n", MAGENTA)

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
    audio_ext_input = input(f"{YELLOW}Enter common audio file extensions, comma-separated (e.g., 'mp3,wav,flac,ogg'): {RESET}").strip()
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
