"""
## ████ ComfyUI MP3 Workflow Extractor v1.0.0

---

### ░▒▓ ORIGIN & DEV:
- **Cast into the void by:** MDMAchine
- **License:** Apache 2.0 _(No cursed floppy disks allowed)_

---

### ░▒▓ DESCRIPTION:
A **Command-Line Utility** designed with precision. Built for **extracting ComfyUI workflow JSON from MP3 metadata**, it channels the chaos of demoscene nights. _May work with various MP3 files containing metadata, but don't expect mercy or miracles._

---

### ░▒▓ FEATURES:
- ✓ Extracts JSON from MP3 metadata
- ✓ User-friendly interactive prompts for file paths
- ✓ Optional direct saving to a JSON file
- ✓ Provides colorful terminal output and feedback
- ✓ Error handling for file access and JSON parsing

---

### ░▒▓ CHANGELOG:

#### v1.0 – Initial Release:
- Core logic implemented for metadata extraction
- Basic stability ensured
- Interactive user interface with input prompts

#### vX.X – Optimization Drop:
- [Insert performance boosts or fixes]
- UI/UX tweaks
- Parameter matrix expanded

---

### ░▒▓ CONFIGURATION:
- **Primary Use:** Extract ComfyUI workflows from MP3 audio file metadata.
- **Secondary Use:** Assist in sharing ComfyUI workflows embedded in audio files.
- **Edge Use:** _Extract workflows from large batches of unverified audio files._

---

### ░▒▓ WARNING:
This tool may trigger:
- ▓▒░ Temporal distortion
- ▓▒░ Memories of ANSI art & screaming modems
- ▓▒░ A sense of unstoppable creative power
"""

# pip install mutagen colorama

import os
import json
from mutagen.id3 import ID3, TXXX, ID3NoHeaderError
from colorama import Fore, Style, init

# Initialize Colorama for cross-platform colored terminal output
init(autoreset=True)

def extract_mp3_workflow_json(mp3_filepath, json_key="prompt"):
    """
    Extracts the ComfyUI workflow JSON from an MP3 file's ID3 metadata.
    Looks for the JSON in a TXXX (User Defined Text Information) frame.
    """
    try:
        audio = ID3(mp3_filepath)
        workflow_json_str = None

        for frame in audio.getall('TXXX'):
            if frame.desc == json_key:
                workflow_json_str = frame.text[0]
                break

        if workflow_json_str:
            try:
                workflow_data = json.loads(workflow_json_str)
                return workflow_data
            except json.JSONDecodeError:
                print(f"{Fore.RED}Error:{Style.RESET_ALL} Metadata found under '{json_key}' but it is not valid JSON in {mp3_filepath}")
                return None
        else:
            print(f"{Fore.YELLOW}Warning:{Style.RESET_ALL} No '{json_key}' metadata found under this key in {mp3_filepath}")
            return None

    except ID3NoHeaderError:
        print(f"{Fore.RED}Error:{Style.RESET_ALL} No ID3 tags found in {mp3_filepath}. This file may not contain workflow metadata.")
        return None
    except Exception as e:
        print(f"{Fore.RED}Error:{Style.RESET_ALL} Failed to read MP3 metadata from {mp3_filepath}: {e}")
        return None

def main():
    print(f"\n{Fore.CYAN}{'='*50}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}ComfyUI Workflow Extractor for MP3 Audio Files{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*50}{Style.RESET_ALL}\n")

    while True:
        audio_file_path = input(f"{Fore.GREEN}Enter the full path to your MP3 audio file (e.g., C:\\audio\\my_song.mp3): {Style.RESET_ALL}").strip()
        
        if not os.path.exists(audio_file_path):
            print(f"{Fore.RED}Error:{Style.RESET_ALL} File not found at '{audio_file_path}'. Please check the path and try again.")
            continue
        
        ext = os.path.splitext(audio_file_path)[1].lower()
        if ext != ".mp3":
            print(f"{Fore.RED}Error:{Style.RESET_ALL} Unsupported audio format: {ext}.\nThis tool now only supports {Fore.CYAN}.mp3{Style.RESET_ALL} files.")
            continue
        break # Exit loop if file exists and is MP3

    output_choice = input(f"{Fore.GREEN}Do you want to save the extracted workflow to a JSON file? (yes/no): {Style.RESET_ALL}").strip().lower()
    output_file_path = None

    if output_choice == 'yes':
        while True:
            output_file_path = input(f"{Fore.GREEN}Enter the full path where you want to save the JSON file (e.g., C:\\workflows\\extracted_workflow.json): {Style.RESET_ALL}").strip()
            if not output_file_path.lower().endswith('.json'):
                print(f"{Fore.YELLOW}Warning:{Style.RESET_ALL} It's recommended to save with a '.json' extension for ComfyUI. Appending '.json'.")
                output_file_path += ".json"
            
            # Check if directory exists, create if not
            output_dir = os.path.dirname(output_file_path)
            if output_dir and not os.path.exists(output_dir):
                try:
                    os.makedirs(output_dir)
                    print(f"{Fore.BLUE}Info:{Style.RESET_ALL} Created directory: {output_dir}")
                except OSError as e:
                    print(f"{Fore.RED}Error:{Style.RESET_ALL} Could not create directory '{output_dir}': {e}. Please enter a valid path.")
                    continue
            break # Exit loop if path is good or created

    metadata_key = "prompt" # Default key as identified

    workflow = None

    print(f"\n{Fore.YELLOW}Attempting to extract workflow from {audio_file_path} using key '{metadata_key}'...{Style.RESET_ALL}")

    # Directly call MP3 extraction since format check is done above
    workflow = extract_mp3_workflow_json(audio_file_path, metadata_key)
    
    if workflow:
        if output_file_path:
            try:
                with open(output_file_path, "w") as f:
                    json.dump(workflow, f, indent=4)
                print(f"\n{Fore.GREEN}Success!{Style.RESET_ALL} Workflow successfully extracted and saved to {Fore.MAGENTA}{output_file_path}{Style.RESET_ALL}")
                print(f"You can now drag and drop this JSON file into ComfyUI!")
            except Exception as e:
                print(f"{Fore.RED}Error:{Style.RESET_ALL} Failed to save JSON to {output_file_path}: {e}")
        else:
            print(f"\n{Fore.GREEN}--- Extracted ComfyUI Workflow JSON ---{Style.RESET_ALL}")
            print(json.dumps(workflow, indent=4))
            print(f"{Fore.GREEN}---------------------------------------{Style.RESET_ALL}")
            print(f"\n{Fore.YELLOW}Remember to save this output to a .json file manually if you want to use it with ComfyUI.{Style.RESET_ALL}")
    else:
        print(f"\n{Fore.RED}Failed to extract workflow.{Style.RESET_ALL} Please ensure the MP3 file contains the workflow metadata under the '{metadata_key}' key.")

if __name__ == "__main__":
    main()