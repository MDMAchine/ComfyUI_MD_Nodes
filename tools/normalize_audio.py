"""
## ████ Audio Normalizer & Converter v1.0.0

---

### ░▒▓ ORIGIN & DEV:
- **Cast into the void by:** MDMAchine
- **License:** Apache 2.0 _(No cursed floppy disks allowed)_

---

### ░▒▓ DESCRIPTION:
A **Command-Line Utility** designed with precision. Built for **batch processing audio files for normalization, conversion, and renaming**, it channels the chaos of demoscene nights. _May work with various audio formats, but don't expect mercy or miracles._

---

### ░▒▓ FEATURES:
- ✓ LUFS loudness normalization
- ✓ Supports multiple output audio formats (MP3, WAV, FLAC, AAC, OGG, OPUS) 
- ✓ Custom output file naming (prefix/suffix) 
- ✓ Dry run mode for testing without file modification 
- ✓ Cross-platform compatibility 

---

### ░▒▓ CHANGELOG:

#### v1.0 – Initial Release:
- Core logic implemented with LUFS normalization 
- Basic stability ensured
- Ground-up optimization passes

#### vX.X – Optimization Drop:
- [Insert performance boosts or fixes]
- UI/UX tweaks
- Parameter matrix expanded

---

### ░▒▓ CONFIGURATION:
- **Primary Use:** Normalize audio to a target LUFS, convert formats, and rename files in bulk. 
- **Secondary Use:** Prepare audio datasets for consistent loudness and format.
- **Edge Use:** _Process large, unverified audio libraries with custom settings._

---

### ░▒▓ WARNING:
This tool may trigger:
- ▓▒░ Temporal distortion
- ▓▒░ Memories of ANSI art & screaming modems
- ▓▒░ A sense of unstoppable creative power
"""

import os
import subprocess
import argparse
import datetime
from tqdm import tqdm
import platform

# --- ANSI Color Codes (for terminal output) ---
class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'

# Attempt to enable ANSI escape codes for Windows cmd.exe
if platform.system() == "Windows":
    try:
        import ctypes
        kernel32 = ctypes.WinDLL('kernel32')
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7) # ENABLE_VIRTUAL_TERMINAL_PROCESSING
    except Exception:
        # Fallback if enabling fails
        for attr in dir(Colors):
            if not attr.startswith('__'):
                setattr(Colors, attr, '')

# --- Script Header and Revisions ---
"""
Audio Processing Script: Normalize, Convert, and Rename

Revisions:
    - 2024-05-20: Initial version with LUFS normalization.
    - 2024-05-21: Added MP3 V0 conversion and metadata preservation.
    - 2025-06-13: Added extensive comments, detailed help output, and revision history.
    - 2025-06-13: Integrated --help command functionality with usage examples.
    - 2025-06-13: Implemented flexible output naming (prefix/suffix), optional normalization,
                  optional MP3 V0 conversion, selectable output format, and dry-run mode.
    - 2025-06-13: Enhanced terminal output with colors, emojis, progress bar, and summary.
    - 2025-06-13: Added interactive startup menu for user-friendly configuration.

Description:
This script automates the processing of audio files. It can:
1. Normalize audio files to a target LUFS (Loudness Units Full Scale) level.
2. Convert audio files to a specified output format (defaulting to high-quality MP3 V0).
3. Preserve metadata (like artist, title, album) from the original files.
4. Offer flexible control over output filenames (custom prefixes/suffixes).
5. Provide a dry-run mode to preview actions without making changes.

The script processes all common audio file types found in a specified input folder and
saves the processed files to a designated output folder.
"""

def print_header(title):
    """Prints a styled header."""
    print(f"\n{Colors.BLUE}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}✨ {title.upper()} ✨{Colors.RESET}")
    print(f"{Colors.BLUE}{'='*70}{Colors.RESET}")

def print_subheader(title):
    """Prints a styled sub-header."""
    print(f"\n{Colors.MAGENTA}{'-'*len(title)}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.MAGENTA}{title}{Colors.RESET}")
    print(f"{Colors.MAGENTA}{'-'*len(title)}{Colors.RESET}")

def print_footer(message):
    """Prints a styled footer."""
    print(f"\n{Colors.BLUE}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.GREEN}{message}{Colors.RESET}")
    print(f"{Colors.BLUE}{'='*70}{Colors.RESET}\n")

def process_audio_folder(input_folder, output_folder, target_lufs=-14.0,
                         apply_loudnorm=True, output_prefix="normalized_",
                         output_suffix="", output_format="mp3", dry_run=False):
    """
    Processes audio files based on provided arguments: normalizes, converts, and renames.
    """

    print_header("Audio Processing Start")

    # --- Input Validation ---
    if not os.path.isdir(input_folder):
        print(f"{Colors.RED}❌ Error:{Colors.RESET} Input folder '{Colors.YELLOW}{input_folder}{Colors.RESET}' does not exist. Please check the path and try again.")
        return

    # --- Output Folder Creation ---
    if dry_run:
        print(f"{Colors.YELLOW}🚧 [DRY RUN]{Colors.RESET} Would create output folder '{Colors.CYAN}{output_folder}{Colors.RESET}' if it doesn't exist.")
    else:
        os.makedirs(output_folder, exist_ok=True)
        print(f"📁 Output folder '{Colors.CYAN}{output_folder}{Colors.RESET}' ensured.")

    # --- Supported Audio Extensions (FFmpeg can generally read these) ---
    supported_extensions = ('.mp3', '.wav', '.flac', '.aac', '.m4a', '.ogg', '.opus', '.wma', '.aiff', '.aif')

    # --- Collect files to process ---
    files_to_process = []
    for filename in os.listdir(input_folder):
        if any(filename.lower().endswith(ext) for ext in supported_extensions):
            files_to_process.append(filename)

    if not files_to_process:
        print(f"{Colors.YELLOW}⚠️ No supported audio files found in '{Colors.CYAN}{input_folder}{Colors.RESET}'. Exiting.{Colors.RESET}")
        return

    # --- Process Files in Folder with Progress Bar ---
    processed_count = 0
    failed_count = 0
    skipped_count = 0 # For dry run
    total_files = len(files_to_process)

    print(f"\n🚀 Starting processing of {Colors.BOLD}{total_files}{Colors.RESET} files...")

    # tqdm wraps the iterable to show a progress bar
    for i, filename in enumerate(tqdm(files_to_process, desc="Overall Progress")):
        input_filepath = os.path.join(input_folder, filename)
        base_filename = os.path.splitext(filename)[0]

        # --- Construct Output Filename ---
        output_filename = f"{output_prefix}{base_filename}{output_suffix}.{output_format}"
        output_filepath = os.path.join(output_folder, output_filename)

        print(f"\n{Colors.BOLD}--- File {i+1}/{total_files}: {filename} ---{Colors.RESET}")
        print(f"  📂 {Colors.BLUE}Input:{Colors.RESET} {input_filepath}")
        print(f"  💾 {Colors.BLUE}Output:{Colors.RESET} {output_filepath}")

        if dry_run:
            print(f"  {Colors.YELLOW}🚧 [DRY RUN]{Colors.RESET} This file would be processed.")
            if apply_loudnorm:
                print(f"  🔊 {Colors.YELLOW}[DRY RUN]{Colors.RESET} Would apply LUFS normalization to {Colors.BOLD}{target_lufs} LUFS{Colors.RESET}.")
            else:
                print(f"  🔇 {Colors.YELLOW}[DRY RUN]{Colors.RESET} Would SKIP LUFS normalization.")
            print(f"  🎵 {Colors.YELLOW}[DRY RUN]{Colors.RESET} Would convert to '{Colors.BOLD}{output_format.upper()}{Colors.RESET}' format.")
            print(f"  🏷️ {Colors.YELLOW}[DRY RUN]{Colors.RESET} Would preserve metadata.")
            skipped_count += 1
            continue # Skip actual processing in dry-run mode

        try:
            # --- Build FFmpeg Command ---
            cmd_normalize_convert = [
                'ffmpeg',
                '-i', input_filepath,
                '-map_metadata', '0',  # Preserve all metadata from the input file
                '-y',                  # Overwrite output files without asking
            ]

            # Add loudness normalization if requested
            if apply_loudnorm:
                print(f"  Applying {Colors.GREEN}LUFS normalization{Colors.RESET} to {Colors.BOLD}{target_lufs} LUFS{Colors.RESET}...")
                cmd_normalize_convert.extend(['-filter_complex', f'loudnorm=I={target_lufs}:LRA=7:tp=-1'])
            else:
                print(f"  {Colors.YELLOW}Skipping LUFS normalization.{Colors.RESET}")

            # Add specific encoding parameters based on output_format
            if output_format.lower() == 'mp3':
                print(f"  Converting to {Colors.GREEN}MP3 V0 (highest quality VBR){Colors.RESET}...")
                cmd_normalize_convert.extend([
                    '-c:a', 'libmp3lame',
                    '-q:a', '0',
                    '-ar', '48000',
                    '-ac', '2',
                ])
            elif output_format.lower() == 'wav':
                print(f"  Converting to {Colors.GREEN}WAV (PCM 16-bit){Colors.RESET}...")
                cmd_normalize_convert.extend([
                    '-c:a', 'pcm_s16le',
                    '-ar', '48000',
                    '-ac', '2',
                ])
            elif output_format.lower() == 'flac':
                print(f"  Converting to {Colors.GREEN}FLAC (lossless){Colors.RESET}...")
                cmd_normalize_convert.extend([
                    '-c:a', 'flac',
                    '-compression_level', '8', # Max compression for FLAC
                    '-ar', '48000',
                    '-ac', '2',
                ])
            elif output_format.lower() == 'aac':
                print(f"  Converting to {Colors.GREEN}AAC (high quality){Colors.RESET}...")
                cmd_normalize_convert.extend([
                    '-c:a', 'aac',
                    '-b:a', '256k', # Bitrate for AAC (good quality)
                    '-ar', '48000',
                    '-ac', '2',
                ])
            elif output_format.lower() == 'ogg':
                print(f"  Converting to {Colors.GREEN}Ogg Vorbis (high quality){Colors.RESET}...")
                cmd_normalize_convert.extend([
                    '-c:a', 'libvorbis',
                    '-q:a', '6', # Quality 0-10, 6 is good balance
                    '-ar', '48000',
                    '-ac', '2',
                ])
            elif output_format.lower() == 'opus':
                print(f"  Converting to {Colors.GREEN}Opus (high quality){Colors.RESET}...")
                cmd_normalize_convert.extend([
                    '-c:a', 'libopus',
                    '-b:a', '128k', # Bitrate for Opus, highly efficient
                    '-ar', '48000',
                    '-ac', '2',
                ])
            else:
                print(f"  {Colors.YELLOW}⚠️ Warning:{Colors.RESET} Output format '{output_format}' is not explicitly handled for quality. FFmpeg will use default for this format.")
                cmd_normalize_convert.extend([
                    '-ar', '48000',
                    '-ac', '2',
                ])

            cmd_normalize_convert.append(output_filepath) # Add the output file path at the end

            # Execute FFmpeg Command
            subprocess.run(cmd_normalize_convert, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) # Suppress FFmpeg's own verbose output for cleaner script output
            print(f"  ✅ {Colors.GREEN}Successfully processed{Colors.RESET} -> '{Colors.CYAN}{os.path.basename(output_filepath)}{Colors.RESET}'")
            processed_count += 1

        except subprocess.CalledProcessError as e:
            print(f"  ❌ {Colors.RED}Error processing '{filename}':{Colors.RESET} FFmpeg exited with code {e.returncode}")
            # If you want to see FFmpeg's detailed error, uncomment lines below.
            # print(f"  FFmpeg stderr: {e.stderr.decode('utf-8', errors='ignore')}")
            failed_count += 1
        except FileNotFoundError:
            print(f"\n{Colors.RED}❌ FATAL ERROR:{Colors.RESET} FFmpeg not found!")
            print(f"  Please ensure FFmpeg is installed and its executable is accessible in your system's PATH.")
            print(f"  You can download FFmpeg from {Colors.UNDERLINE}https://ffmpeg.org/download.html{Colors.RESET}")
            return # Exit the function as FFmpeg is essential.
        except Exception as e:
            print(f"  ❌ {Colors.RED}An unexpected error occurred with '{filename}':{Colors.RESET} {e}")
            failed_count += 1

    # --- Summary ---
    print(f"\n{Colors.BOLD}{Colors.MAGENTA}--- Processing Summary ---{Colors.RESET}")
    print(f"  Files Found: {Colors.CYAN}{total_files}{Colors.RESET}")
    if dry_run:
        print(f"  {Colors.YELLOW}Files to be processed (Dry Run): {skipped_count}{Colors.RESET}")
    else:
        print(f"  {Colors.GREEN}Successfully Processed: {processed_count}{Colors.RESET}")
        print(f"  {Colors.RED}Failed to Process: {failed_count}{Colors.RESET}")

    if dry_run:
        print_footer(f"🚧 DRY RUN COMPLETE. No files were modified or created. 🚧")
    else:
        if failed_count == 0:
            print_footer(f"🎉 ALL FILES PROCESSED SUCCESSFULLY! 🎉")
        elif processed_count > 0:
            print_footer(f"✅ PROCESSING COMPLETE WITH SOME ISSUES. Check errors above. ✅")
        else:
            print_footer(f"💔 PROCESSING FAILED FOR ALL FILES. Please check errors. 💔")


def interactive_mode():
    """Guides the user through setting options interactively."""
    print_header("Interactive Setup")

    input_folder = ""
    while not os.path.isdir(input_folder):
        input_folder = input(f"{Colors.BLUE}▶️ Enter the INPUT folder path (e.g., C:\\MyAudio): {Colors.RESET}").strip()
        if not os.path.isdir(input_folder):
            print(f"{Colors.RED}❌ Invalid path. Folder does not exist. Please try again.{Colors.RESET}")

    output_folder = ""
    while not output_folder:
        output_folder = input(f"{Colors.BLUE}▶️ Enter the OUTPUT folder path (e.g., C:\\ProcessedAudio): {Colors.RESET}").strip()
        if not output_folder:
            print(f"{Colors.RED}❌ Output path cannot be empty. Please try again.{Colors.RESET}")

    print_subheader("Normalization Settings")
    target_lufs = -14.0
    try:
        lufs_input = input(f"{Colors.BLUE}▶️ Enter target LUFS (e.g., -14.0 for streaming, default: {target_lufs}): {Colors.RESET}").strip()
        if lufs_input:
            target_lufs = float(lufs_input)
    except ValueError:
        print(f"{Colors.YELLOW}⚠️ Invalid LUFS value. Using default: {target_lufs} LUFS.{Colors.RESET}")

    apply_loudnorm_choice = input(f"{Colors.BLUE}▶️ Apply loudness normalization? (yes/no, default: yes): {Colors.RESET}").strip().lower()
    apply_loudnorm = (apply_loudnorm_choice == 'yes' or apply_loudnorm_choice == '')

    print_subheader("Output Format Settings")
    output_formats = ['mp3', 'wav', 'flac', 'aac', 'ogg', 'opus']
    output_format = "mp3"
    print(f"  Available formats: {Colors.CYAN}{', '.join(output_formats)}{Colors.RESET}")
    format_input = input(f"{Colors.BLUE}▶️ Enter desired output format (default: {output_format}): {Colors.RESET}").strip().lower()
    if format_input and format_input in output_formats:
        output_format = format_input
    elif format_input:
        print(f"{Colors.YELLOW}⚠️ Invalid format '{format_input}'. Using default: {output_format}.{Colors.RESET}")


    print_subheader("Output Filename Settings")
    output_prefix = input(f"{Colors.BLUE}▶️ Enter output filename prefix (leave empty for no prefix, default: 'normalized_'): {Colors.RESET}").strip()
    if output_prefix == "": # Allow explicit empty string
        pass
    elif not output_prefix: # If user just pressed enter, use default
        output_prefix = "normalized_"

    output_suffix = input(f"{Colors.BLUE}▶️ Enter output filename suffix (before extension, e.g., '_final', default: no suffix): {Colors.RESET}").strip()

    print_subheader("Operation Mode")
    dry_run_choice = input(f"{Colors.BLUE}▶️ Perform a dry run (simulate only)? (yes/no, default: no): {Colors.RESET}").strip().lower()
    dry_run = (dry_run_choice == 'yes')

    print_header("Configuration Summary")
    print(f"  {Colors.BOLD}Input Folder:{Colors.RESET} {Colors.CYAN}{input_folder}{Colors.RESET}")
    print(f"  {Colors.BOLD}Output Folder:{Colors.RESET} {Colors.CYAN}{output_folder}{Colors.RESET}")
    print(f"  {Colors.BOLD}Normalization:{Colors.RESET} {'Yes' if apply_loudnorm else 'No'}")
    if apply_loudnorm:
        print(f"    {Colors.BOLD}Target LUFS:{Colors.RESET} {target_lufs}")
    print(f"  {Colors.BOLD}Output Format:{Colors.RESET} {output_format.upper()}")
    print(f"  {Colors.BOLD}Output Prefix:{Colors.RESET} '{output_prefix}'")
    print(f"  {Colors.BOLD}Output Suffix:{Colors.RESET} '{output_suffix}'")
    print(f"  {Colors.BOLD}Mode:{Colors.RESET} {'DRY RUN (No files will be changed)' if dry_run else 'LIVE RUN (Files will be processed)'}")

    confirm = input(f"\n{Colors.YELLOW}Confirm and start processing with these settings? (yes/no): {Colors.RESET}").strip().lower()
    if confirm == 'yes':
        process_audio_folder(input_folder, output_folder, target_lufs,
                             apply_loudnorm, output_prefix, output_suffix,
                             output_format, dry_run)
    else:
        print(f"{Colors.RED}❌ Processing cancelled by user.{Colors.RESET}")
        print_footer("Operation Cancelled")

# --- Main Execution Block ---
if __name__ == "__main__":
    # Check if any command-line arguments (besides the script name itself) are provided
    if len(os.sys.argv) > 1:
        # If arguments are provided, use argparse (for power users)
        parser = argparse.ArgumentParser(
            description="""
✨ Audio Processing Script: Normalize, Convert, and Rename ✨

Normalize audio files to a target LUFS level, convert them to a specified format (default MP3 V0),
and preserve metadata. Offers flexible naming and processing options.
Uses FFmpeg. Ensure FFmpeg is installed and available in your system's PATH.

Usage Examples:
  # Normalize all audio files in 'input_dir' to -14 LUFS, convert to MP3 V0,
  # and save to 'output_dir' with 'normalized_' prefix (default behavior):
  python your_script_name.py "input_dir" "output_dir"

  # Normalize to -16 LUFS, convert to MP3 V0, save to 'output_dir', no prefix/suffix:
  python your_script_name.py "input_dir" "output_dir" --target_lufs -16.0 --output_prefix "" --output_suffix ""

  # Convert all audio files in 'input_dir' to WAV format (no normalization),
  # save to 'output_dir' with 'converted_' prefix:
  python your_script_name.py "input_dir" "output_dir" --no_normalize --output_format wav --output_prefix "converted_"

  # Perform a dry run to see what would happen (no files created/modified):
  python your_script_name.py "input_dir" "output_dir" --dry_run --output_prefix "preview_"

  # Normalize to -18 LUFS, keep output as FLAC, add '_loud' suffix:
  python your_script_name.py "input_dir" "output_dir" --target_lufs -18.0 --output_format flac --output_suffix "_loud"
""",
            formatter_class=argparse.RawTextHelpFormatter
        )

        parser.add_argument("input_folder", help="Path to the folder containing audio files to be processed.")
        parser.add_argument("output_folder", help="Path to the folder where processed audio files will be saved.")
        parser.add_argument("--target_lufs", type=float, default=-14.0, help="""Target Integrated Loudness (LUFS) level for normalization. (Default: -14.0)""")
        parser.add_argument("--no_normalize", action="store_true", help="Skip the LUFS normalization step.")
        parser.add_argument("--output_format", type=str, default="mp3", choices=['mp3', 'wav', 'flac', 'aac', 'ogg', 'opus'], help="""Desired output audio format. (Default: mp3)""")
        parser.add_argument("--output_prefix", type=str, default="normalized_", help="""Custom prefix for output filenames. Use "" for no prefix. (Default: "normalized_")""")
        parser.add_argument("--output_suffix", type=str, default="", help="""Custom suffix for output filenames, placed before the extension. (Default: "")""")
        parser.add_argument("--dry_run", action="store_true", help="""Simulate the process without actually creating or modifying any files.""")

        args = parser.parse_args()

        process_audio_folder(
            args.input_folder,
            args.output_folder,
            target_lufs=args.target_lufs,
            apply_loudnorm=not args.no_normalize,
            output_prefix=args.output_prefix,
            output_suffix=args.output_suffix,
            output_format=args.output_format,
            dry_run=args.dry_run
        )
    else:
        # If no arguments, start the interactive mode
        interactive_mode()