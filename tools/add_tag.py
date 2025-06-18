"""
## ████ Prompt Tagging Tool v1.0.0

---

### ░▒▓ ORIGIN & DEV:
- **Cast into the void by:** MDMAchine
- **License:** Apache 2.0 _(No cursed floppy disks allowed)_

---

### ░▒▓ DESCRIPTION:
A **Command-Line Utility** designed with precision. Built for **batch adding custom tags to text prompt files**, it channels the chaos of demoscene nights. _May work with other text-based files, but don't expect mercy or miracles._

---

### ░▒▓ FEATURES:
- ✓ Processes multiple text files
- ✓ Appends tags efficiently
- ✓ Handles existing tags to avoid duplicates
- ✓ Provides clear execution feedback
- ✓ Debug-ready (logs optional)

---

### ░▒▓ CHANGELOG:

#### v1.0 – Initial Release:
- Core logic implemented
- Basic stability ensured
- Ground-up optimization passes

#### vX.X – Optimization Drop:
- [Insert performance boosts or fixes]
- UI/UX tweaks
- Parameter matrix expanded

---

### ░▒▓ CONFIGURATION:
- **Primary Use:** Append tags to text files (e.g., `_prompt.txt`) for data management or organization.
- **Secondary Use:** Automate modification of text-based datasets for various purposes.
- **Edge Use:** _Process large, unverified text datasets._

---

### ░▒▓ WARNING:
This tool may trigger:
- ▓▒░ Temporal distortion
- ▓▒░ Memories of ANSI art & screaming modems
- ▓▒░ A sense of unstoppable creative power
"""

import os
import argparse

def add_tags_to_prompts(data_directory: str, tags_to_add: list[str]):
    """
    Adds one or more custom tags to all _prompt.txt files within the specified directory.

    Args:
        data_directory (str): The path to the directory containing the _prompt.txt files.
        tags_to_add (list[str]): A list of tags to append to the content of each _prompt.txt file.
                                   Tags are added only if they are not already present.
    """
    print(f"\n--- Starting Tag Addition ---")
    print(f"Processing directory: '{data_directory}'")
    print(f"Tags to append: {tags_to_add}")

    total_files_updated = 0
    total_tags_appended = 0

    if not os.path.exists(data_directory):
        print(f"Error: The specified data directory '{data_directory}' does not exist. Please check the path.")
        return

    # Walk through the directory to find all _prompt.txt files
    for root, _, files in os.walk(data_directory):
        for file_name in files:
            if file_name.endswith("_prompt.txt"):
                file_path = os.path.join(root, file_name)
                tags_added_to_current_file = 0

                try:
                    # Read current content
                    with open(file_path, 'r', encoding='utf-8') as f:
                        current_content = f.read().strip()

                    # Create a set of existing tags (case-insensitive) for quick lookup
                    # Ensure we handle empty files or files with only whitespace
                    existing_tags_set = set(
                        tag.strip().lower() for tag in current_content.split(',') if tag.strip()
                    )

                    new_content = current_content
                    for tag in tags_to_add:
                        clean_tag = tag.strip()
                        if clean_tag and clean_tag.lower() not in existing_tags_set:
                            # Append the new tag, adding a comma if content already exists
                            new_content = f"{new_content}, {clean_tag}" if new_content else clean_tag
                            existing_tags_set.add(clean_tag.lower()) # Add to set to avoid adding again
                            tags_added_to_current_file += 1
                            total_tags_appended += 1

                    # Write back only if changes were made
                    if tags_added_to_current_file > 0:
                        with open(file_path, 'w', encoding='utf-8', newline='\n') as f:
                            f.write(new_content)
                        print(f"  Updated: '{file_path}' (added {tags_added_to_current_file} new tags)")
                        total_files_updated += 1
                    else:
                        print(f"  Skipped: '{file_path}' (all tags already present or no valid tags to add)")

                except Exception as e:
                    print(f"  Error processing '{file_path}': {e}")

    print(f"\n--- Tag Addition Summary ---")
    print(f"Total files where tags were appended: {total_files_updated}")
    print(f"Total individual new tags added across all files: {total_tags_appended}")
    if total_files_updated == 0 and total_tags_appended == 0 and os.path.exists(data_directory) and not any(f.endswith("_prompt.txt") for r,d,f_list in os.walk(data_directory) for f in f_list):
        print("Note: No '_prompt.txt' files were found in the specified directory.")
    print(f"---------------------------\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
        Add one or more custom tags to all _prompt.txt files in a specified directory.
        This is useful for adding dataset-specific identifiers (e.g., 'mdmachine')
        to your training prompts.
        """
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="The absolute path to the directory containing your _prompt.txt files. "
             "Example: 'C:\\Users\\Admin\\Desktop\\MDMAchine\\normalized'"
    )
    parser.add_argument(
        "--tags",
        type=str,
        required=True,
        help="A comma-separated list of tags to add. "
             "Example: 'mdmachine, my_unique_style, synthwave'"
    )

    args = parser.parse_args()

    # Split the comma-separated string of tags into a list, removing extra whitespace
    # and filtering out any empty strings that might result from extra commas (e.g., "tag1,,tag2")
    tags_list = [tag.strip() for tag in args.tags.split(',') if tag.strip()]

    if not tags_list:
        print("Error: No valid tags were provided after parsing. Please ensure you provide a comma-separated list of tags (e.g., 'tag1,tag2').")
    else:
        add_tags_to_prompts(args.data_dir, tags_list)