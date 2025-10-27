# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
# ████ MD_Nodes/SmartFilenameBuilder – Dynamic Filename Toolkit v1.0.0 ████▓▒░
# © 2025 MDMAchine
# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
# ░▒▓ ORIGIN & DEV:
#   • Cast into the void by: MDMAchine
#   • Enhanced by: Gemini, Claude
#   • License: Apache 2.0 — Sharing is caring
#   • Original source (if applicable): N/A

# ░▒▓ DESCRIPTION:
#   A suite of nodes for creating dynamic, complex, and clean filenames.
#   Includes a preset-based builder, a simple token replacer, and a
#   persistent, file-based counter for robust file organization.

# ░▒▓ FEATURES:
#   ✓ `SmartFilenameBuilder`: Preset-driven (Vocal, Master) & custom filename generation.
#   ✓ `FilenameTokenReplacer`: Simple `{token}` substitution for custom templates.
#   ✓ `FilenameCounterNode`: Persistent, context-aware, auto-incrementing counter.
#   ✓ Handles date/time formatting (`%Y-%m-%d`), component toggles, and sanitization.

# ░▒▓ CHANGELOG:
#   - v1.0.0 (Current Release - Initial Commit):
#       • ADDED: `SmartFilenameBuilder` with presets and parameter toggles.
#       • ADDED: `FilenameTokenReplacer` for simple template-based replacement.
#       • ADDED: `FilenameCounterNode` with persistent file-based storage.

# ░▒▓ CONFIGURATION:
#   → Primary Use: `SmartFilenameBuilder` to create complex, clean filenames with presets.
#   → Secondary Use: `FilenameCounterNode` to add persistent, zero-padded counters (`#0001`).
#   → Edge Use: `FilenameTokenReplacer` for highly custom, non-standard filename structures.

# ░▒▓ WARNING:
#   This node may trigger:
#   ▓▒░ Obsessively tweaking separators, padding, and date formats instead of generating.
#   ▓▒░ A sudden, violent rage when your final path exceeds Windows' 256-character limit.
#   ▓▒░ Flashbacks to `MYSONG_1.MOD`, `MYSONG_2.MOD`, `MYSONG_F.MOD`, `MYSONG_FNL.MOD`.
#   ▓▒░ A deep, spiritual satisfaction from seeing `#0001` perfectly increment to `#0002`.
#   Consult your nearest demoscene vet if hallucinations persist.
# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄


# =================================================================================
# == Standard Library Imports                                                  ==
# =================================================================================
import os
import re
from datetime import datetime
import logging
import traceback
import secrets

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
# (Helper functions are defined as private methods within classes below)

# =================================================================================
# == Core Node Class: SmartFilenameBuilder                                     ==
# =================================================================================

class SmartFilenameBuilder:
    """
    MD: Smart Filename Builder

    Generates complex and clean filenames using presets or custom configurations.
    Includes options for date/time formatting, component toggles (steps, seed, etc.),
    and automatic sanitization. Ideal for structured output organization,
    especially in audio workflows.
    """

    # Class variable for presets
    PRESETS = {
        "Custom": {},
        "Instrumental": {
            "mode_tag": "(Instrumental)", "include_steps": True, "include_schedule": True,
            "include_seed": False, "include_genre": True,
        },
        "Vocal": {
            "mode_tag": "(Vocal)", "include_steps": True, "include_schedule": True,
            "include_seed": False, "include_genre": True,
        },
        "Master": {
            "mode_tag": "(Master)", "include_steps": True, "include_schedule": False,
            "include_seed": False, "include_genre": False,
        },
        "Raw Output": {
            "mode_tag": "(Raw)", "include_steps": True, "include_schedule": True,
            "include_seed": True, "include_genre": False,
        },
        "AB Test": {
            "mode_tag": "(ABMode - MD LoRa)", "include_steps": True, "include_schedule": True,
            "include_seed": False, "include_genre": False,
        },
    }

    @classmethod
    def INPUT_TYPES(cls):
        """Define node inputs with detailed tooltips."""
        return {
            "required": {
                "preset": (list(cls.PRESETS.keys()), {
                    "default": "Custom",
                    "tooltip": (
                        "FILENAME PRESET\n"
                        "- Select a preset to automatically configure component inclusion and mode tag.\n"
                        "- 'Custom' allows manual configuration using the options below."
                    )
                }),
                "base_template": ("STRING", {
                    "default": "MD_Nodes_Workflow %Y-%m-%d", "multiline": False,
                    "tooltip": (
                        "BASE TEMPLATE\n"
                        "- The starting part of the filename.\n"
                        "- Supports standard date/time codes (strftime format).\n"
                        "- Examples: %Y=Year, %m=Month, %d=Day, %H=Hour(24), %M=Minute, %S=Second.\n"
                        "- Common format: 'MyProject_%Y%m%d'"
                    )
                }),
                "project_path": ("STRING", {
                    "default": "Ace-Step/313/", "multiline": False,
                    "tooltip": (
                        "PROJECT SUBDIRECTORY PATH\n"
                        "- Defines the subfolder structure within ComfyUI's main 'output' directory.\n"
                        "- Use forward slashes '/' for separators.\n"
                        "- Example: 'MyAudioProject/Experiment1/'\n"
                        "- Leave blank to save directly in the 'output' folder."
                    )
                }),
            },
            "optional": {
                "mode_tag": ("STRING", {
                    "default": "(Custom Mode)", "multiline": False,
                    "tooltip": (
                        "MODE TAG (Custom Only)\n"
                        "- A custom identifier added to the filename (e.g., 'TestRun', 'LoRA_Test').\n"
                        "- Ignored if a preset other than 'Custom' is selected (preset tag overrides)."
                    )
                }),
                "steps": ("INT", {"default": 0, "min": 0, "tooltip": "SAMPLING STEPS\n- The number of steps used in generation (if applicable).\n- Set to 0 to omit."}),
                "schedule_info": ("STRING", {
                    "default": "", "multiline": False,
                    "tooltip": (
                        "SCHEDULER INFO\n"
                        "- A short identifier for the scheduler used (e.g., 'karras', 'simple', 'lq17-poly5').\n"
                        "- Keep it concise for filename readability."
                    )
                }),
                "seed": ("INT", {"default": 0, "min": 0, "tooltip": "SEED VALUE\n- The seed used for generation.\n- Set to 0 to omit."}),
                "genre": ("STRING", {
                    "default": "", "multiline": False,
                    "tooltip": (
                        "GENRE TAGS\n"
                        "- Main genre keywords (e.g., 'Synthwave, Retro').\n"
                        "- Will be automatically cleaned, sanitized, and truncated."
                    )
                }),
                "custom_tag_1": ("STRING", {"default": "", "multiline": False, "tooltip": "CUSTOM TAG 1\n- An additional custom string to include in the filename."}),
                "custom_tag_2": ("STRING", {"default": "", "multiline": False, "tooltip": "CUSTOM TAG 2\n- Another additional custom string to include."}),
                "counter_start": ("INT", { # Note: This refers to the external counter node
                    "default": 0, "min": 0, "max": 99999,
                    "tooltip": (
                         "COUNTER VALUE (from Counter Node)\n"
                         "- Connect the 'raw_count' output from a 'MD: Filename Counter' node here.\n"
                         "- Set to 0 or leave disconnected to disable counter inclusion."
                    )
                }),
                # --- Component Toggles (Used in Custom mode, overridden by presets) ---
                "include_steps": ("BOOLEAN", {"default": True, "tooltip": "INCLUDE STEPS\n- Add 'XXX Steps' to the filename?"}),
                "include_schedule": ("BOOLEAN", {"default": True, "tooltip": "INCLUDE SCHEDULER\n- Add scheduler info to the filename?"}),
                "include_seed": ("BOOLEAN", {"default": False, "tooltip": "INCLUDE SEED\n- Add 'Seed-XXXX' to the filename?"}),
                "include_genre": ("BOOLEAN", {"default": False, "tooltip": "INCLUDE GENRE\n- Add cleaned genre tags to the filename?"}),
                "include_counter": ("BOOLEAN", {"default": True, "tooltip": "INCLUDE COUNTER\n- Add '#XXXX' counter (requires 'counter_start' > 0)?"}), # Changed default to True
                # --- Formatting Options ---
                "separator": ("STRING", {
                    "default": " - ", "multiline": False,
                    "tooltip": (
                        "SEPARATOR\n"
                        "- The string used to join the different filename components.\n"
                        "- Common examples: ' - ', '_', '--'."
                    )
                }),
                "genre_max_length": ("INT", {
                    "default": 40, "min": 10, "max": 100,
                    "tooltip": (
                        "GENRE MAX LENGTH\n"
                        "- Maximum number of characters allowed for the cleaned genre string in the filename."
                    )
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("full_path_prefix", "filename_preview") # Renamed for clarity
    FUNCTION = "build_filename"
    CATEGORY = "MD_Nodes/Utility" # Corrected category

    def __init__(self):
        # Counter state removed - handled by FilenameCounterNode now
        pass

    def build_filename(self, preset, base_template, project_path,
                       mode_tag="(Custom Mode)", steps=0, schedule_info="",
                       seed=0, genre="", custom_tag_1="", custom_tag_2="",
                       counter_start=0, # Changed default to 0 to match tooltip
                       include_steps=True, include_schedule=True,
                       include_seed=False, include_genre=False, include_counter=True, # Changed default
                       separator=" - ", genre_max_length=40):
        """
        Constructs the filename string based on inputs and selected preset/options.

        Args:
            (All args match INPUT_TYPES definition)

        Returns:
            Tuple: (full_path_prefix, filename_preview)
                   Returns default strings on critical error.
        """
        try:
            # --- Apply Preset Overrides ---
            preset_config = {}
            if preset != "Custom" and preset in self.PRESETS:
                preset_config = self.PRESETS[preset]
                # Only override mode_tag if the user hasn't changed it from the default
                if mode_tag == "(Custom Mode)" and "mode_tag" in preset_config:
                    mode_tag = preset_config["mode_tag"]
                # Apply preset toggles directly
                include_steps = preset_config.get("include_steps", include_steps)
                include_schedule = preset_config.get("include_schedule", include_schedule)
                include_seed = preset_config.get("include_seed", include_seed)
                include_genre = preset_config.get("include_genre", include_genre)
                # Note: include_counter is generally kept manual or set by default True

            # --- Process Base Template (Date/Time) ---
            try:
                # Use current time to format the base template
                filename_base = datetime.now().strftime(base_template)
            except ValueError as date_err:
                 logging.warning(f"[SmartFilenameBuilder] Invalid date format code in base_template '{base_template}'. Error: {date_err}")
                 filename_base = base_template # Use raw template on error

            filename_base = self._sanitize_filename(filename_base) # Sanitize after date formatting


            # --- Build Component Parts List ---
            parts = []

            # Add counter if enabled and value > 0
            if include_counter and counter_start > 0:
                # Use 4-digit padding by default for counter
                parts.append(f"#{counter_start:04d}")

            # Add steps info if enabled and steps > 0
            if include_steps and steps > 0:
                parts.append(f"{steps}S") # Shortened tag

            # Add schedule info if enabled and provided
            if include_schedule and schedule_info:
                clean_schedule = self._sanitize_filename(schedule_info)
                if clean_schedule:
                    parts.append(clean_schedule)

            # Add seed if enabled and seed > 0
            if include_seed and seed > 0:
                parts.append(f"Seed{seed}") # Shortened tag

            # Add genre (cleaned and truncated) if enabled and provided
            if include_genre and genre:
                clean_genre = self._clean_genre(genre, genre_max_length)
                if clean_genre:
                    parts.append(clean_genre)

            # Add custom tags if provided
            if custom_tag_1:
                clean_tag1 = self._sanitize_filename(custom_tag_1)
                if clean_tag1: parts.append(clean_tag1)
            if custom_tag_2:
                clean_tag2 = self._sanitize_filename(custom_tag_2)
                if clean_tag2: parts.append(clean_tag2)

            # Add mode tag if provided (might be from preset or custom)
            if mode_tag:
                clean_mode = self._sanitize_filename(mode_tag, allow_parentheses=True)
                if clean_mode: parts.append(clean_mode)

            # --- Combine Filename ---
            # Start with the processed base template
            filename_final = filename_base
            # Join parts if any exist
            if parts:
                filename_final += separator + separator.join(parts)

            # Final cleanup: remove potential double separators, leading/trailing separators
            safe_separator = re.escape(separator)
            filename_final = re.sub(f'({safe_separator})+', separator, filename_final) # Replace multiple separators with one
            filename_final = filename_final.strip(separator.strip()) # Strip leading/trailing separators (strip spaces from separator first)

            # --- Construct Full Path Prefix ---
            # Sanitize project path segments individually
            path_segments = [self._sanitize_filename(seg) for seg in project_path.replace('\\', '/').split('/') if seg]
            clean_project_path = "/".join(path_segments) + "/" if path_segments else ""

            # Combine path and filename
            # NOTE: This is a *prefix*, the saver node will add the final extension (e.g., .wav, .png)
            full_path_prefix = os.path.join(clean_project_path, filename_final).replace('\\', '/')


            # --- Create Preview String ---
            preview = (f"Directory: [output]/{clean_project_path}\n"
                       f"Filename: {filename_final}\n\n"
                       f"➡️ Full Prefix: {full_path_prefix}")

            # Return tuple matching RETURN_TYPES
            return (full_path_prefix, preview)

        except Exception as e:
            logging.error(f"[SmartFilenameBuilder] Error building filename: {e}", exc_info=True)
            error_preview = f"ERROR: {e}\n{traceback.format_exc()}"
            # Return safe defaults on error
            return ("error/error_filename", error_preview)


    def _sanitize_filename(self, text, allow_parentheses=False):
        """
        Remove or replace invalid filename characters.

        Args:
            text: The input string.
            allow_parentheses: If True, keeps '()' characters.

        Returns:
            Sanitized string safe for use in filenames.
        """
        if not isinstance(text, str) or not text:
            return ""

        # Remove leading/trailing whitespace first
        text = text.strip()

        # Define invalid characters (OS-agnostic basic set + control chars)
        # Allows hyphen, underscore
        if allow_parentheses:
            # Keep parentheses but remove others
            invalid_chars = r'[<>:"/\\|?*\x00-\x1f]'
        else:
            # Remove all potentially problematic chars including parentheses, brackets
            invalid_chars = r'[<>:"/\\|?*\(\)\[\]\{\}\x00-\x1f]'

        # Remove invalid characters
        text = re.sub(invalid_chars, '', text)

        # Replace sequences of whitespace with a single underscore
        text = re.sub(r'\s+', '_', text)

        # Replace multiple consecutive underscores/hyphens with a single one
        text = re.sub(r'[-_]{2,}', '_', text)

        # Remove leading/trailing underscores, hyphens, dots
        text = text.strip('._- ')

        # Limit overall length (conservative limit)
        max_len = 200
        if len(text) > max_len:
            text = text[:max_len].rsplit('_', 1)[0] # Cut at last underscore if possible
            text = text.strip('._- ') # Clean up again

        return text if text else "sanitized_empty" # Return placeholder if everything removed


    def _clean_genre(self, genre, max_length):
        """
        Clean and truncate genre string specifically for filename use.

        Args:
            genre: The input genre string.
            max_length: Maximum allowed length.

        Returns:
            Cleaned and truncated genre string.
        """
        if not isinstance(genre, str) or not genre:
            return ""

        # Replace common separators (comma, semicolon) with space
        text = re.sub(r'[,;]+', ' ', genre)
        # Keep only alphanumeric, space, hyphen
        text = re.sub(r'[^\w\s-]', '', text)
        # Collapse multiple spaces/hyphens
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'[-]{2,}', '-', text).strip('- ')


        # Truncate intelligently
        if len(text) > max_length:
            # Find the last space or hyphen within the limit
            cut_point = -1
            for char in [' ', '-']:
                 found = text.rfind(char, 0, max_length)
                 if found > cut_point:
                      cut_point = found
            # If no separator found, just cut hard
            if cut_point != -1:
                 text = text[:cut_point]
            else:
                 text = text[:max_length]
            text = text.strip(' -') # Clean up trailing separator

        return text


# =================================================================================
# == Core Node Class: FilenameTokenReplacer                                    ==
# =================================================================================

class FilenameTokenReplacer:
    """
    MD: Filename Token Replacer

    Replaces predefined tokens (like {date}, {steps}, {custom1}) within a
    template string to generate a filename or path prefix. Useful for creating
    custom filename structures based on workflow parameters.
    """

    @classmethod
    def INPUT_TYPES(cls):
        """Define node inputs with detailed tooltips."""
        return {
            "required": {
                "template": ("STRING", {
                    "default": "{project}/{date} - {steps}S - {mode}", "multiline": True,
                    "tooltip": (
                        "FILENAME TEMPLATE\n"
                        "- Define the structure using placeholders in curly braces {}.\n"
                        "- Available Tokens:\n"
                        "  {date}, {time}, {year}, {month}, {day}, {hour}, {minute}, {second}\n"
                        "  {project}, {mode}, {steps}, {seed}, {genre}\n"
                        "  {custom1}, {custom2}\n"
                        "- Example: 'output/{project}/{date}_{time}_{seed}.png'"
                    )
                }),
            },
            "optional": {
                # --- Token Value Inputs ---
                "project": ("STRING", {"default": "MyProject", "tooltip": "Value for {project} token."}),
                "mode": ("STRING", {"default": "Image", "tooltip": "Value for {mode} token."}),
                "steps": ("INT", {"default": 0, "min": 0, "tooltip": "Value for {steps} token (omitted if 0)."}),
                "seed": ("INT", {"default": 0, "min": 0, "tooltip": "Value for {seed} token (omitted if 0)."}),
                "genre": ("STRING", {"default": "", "tooltip": "Value for {genre} token (cleaned)."}),
                "custom1": ("STRING", {"default": "", "tooltip": "Value for {custom1} token."}),
                "custom2": ("STRING", {"default": "", "tooltip": "Value for {custom2} token."}),
                # --- Date/Time Formatting ---
                "date_format": ("STRING", {
                    "default": "%Y-%m-%d",
                    "tooltip": (
                        "DATE FORMAT\n"
                        "- Controls the output of the {date} token.\n"
                        "- Uses standard Python strftime codes.\n"
                        "- Examples: %Y (Year), %m (Month 01-12), %d (Day 01-31), %y (Year short)."
                    )
                }),
                "time_format": ("STRING", {
                    "default": "%H-%M-%S",
                    "tooltip": (
                        "TIME FORMAT\n"
                        "- Controls the output of the {time} token.\n"
                        "- Uses standard Python strftime codes.\n"
                        "- Examples: %H (Hour 00-23), %M (Minute 00-59), %S (Second 00-59), %I (Hour 01-12), %p (AM/PM)."
                    )
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("result_string", "preview") # Renamed for clarity
    FUNCTION = "replace_tokens"
    CATEGORY = "MD_Nodes/Utility" # Corrected category

    def replace_tokens(self, template, project="", mode="", steps=0, seed=0,
                       genre="", custom1="", custom2="",
                       date_format="%Y-%m-%d", time_format="%H-%M-%S"):
        """
        Replaces tokens in the template string with provided values.

        Args:
            (All args match INPUT_TYPES definition)

        Returns:
            Tuple: (result_string, preview_string)
                   Returns default strings on critical error.
        """
        try:
            now = datetime.now()

            # --- Build Token Dictionary ---
            # Ensure keys match the template placeholders exactly
            tokens = {
                "project": self._sanitize_text(project), # Sanitize path components
                "mode": self._sanitize_text(mode),
                "steps": str(steps) if steps > 0 else "",
                "seed": str(seed) if seed > 0 else "",
                "genre": self._clean_text(genre, 40), # Use specific cleaning for genre
                "custom1": self._sanitize_text(custom1),
                "custom2": self._sanitize_text(custom2),
                # Date/Time formatting with error handling
                "date": self._safe_strftime(now, date_format, "DATE_ERR"),
                "time": self._safe_strftime(now, time_format, "TIME_ERR"),
                "year": self._safe_strftime(now, "%Y", "YYYY"),
                "month": self._safe_strftime(now, "%m", "MM"),
                "day": self._safe_strftime(now, "%d", "DD"),
                "hour": self._safe_strftime(now, "%H", "HH"),
                "minute": self._safe_strftime(now, "%M", "MM"),
                "second": self._safe_strftime(now, "%S", "SS"),
            }

            # --- Perform Replacement ---
            result = template
            # Use regex to find {token_name} patterns
            for key, value in tokens.items():
                # Replace token, ensuring value is a string
                result = re.sub(r'\{' + re.escape(key) + r'\}', str(value), result, flags=re.IGNORECASE)

            # --- Clean Up Result ---
            # Remove tokens that had no value (or were explicitly empty)
            result = re.sub(r'\{[a-zA-Z0-9_]+\}', '', result) # Remove any remaining unreplaced tokens
            # Clean up separators: remove separators next to empty sections
            result = re.sub(r'\s*-\s*-?\s*', ' - ', result) # Handle patterns like " - - ", " - ", "- - " -> " - "
            result = re.sub(r'\s*/\s*/?\s*', '/', result)   # Handle path separators like " / / ", " /", "/ / " -> "/"
            result = re.sub(r'\s*_\s*_?\s*', '_', result)   # Handle underscores
            # Collapse multiple spaces/separators
            result = re.sub(r'[_\-\s]{2,}', '_', result) # Replace multiple separators/spaces with single underscore
            # Remove leading/trailing separators/spaces/underscores
            result = result.strip(' _-/')
            # Final path sanitization (handle potential issues from combined tokens)
            result = self._sanitize_filepath(result)


            # --- Create Preview ---
            preview = "Token Replacements:\n" + "\n".join([
                f"  {{{k}}}: {v}" for k, v in tokens.items() if v or k in ["steps", "seed"] # Show even if 0
            ]) + f"\n\n➡️ Result:\n  {result}"

            # Return tuple matching RETURN_TYPES
            return (result, preview)

        except Exception as e:
            logging.error(f"[FilenameTokenReplacer] Error replacing tokens: {e}", exc_info=True)
            error_preview = f"ERROR: {e}\n{traceback.format_exc()}"
            # Return safe defaults
            return ("error_replacing_tokens", error_preview)


    def _safe_strftime(self, dt_obj, fmt, fallback=""):
        """Attempts strftime, returns fallback on error."""
        try:
            return dt_obj.strftime(fmt)
        except ValueError as e:
            logging.warning(f"[FilenameTokenReplacer] Invalid format '{fmt}': {e}")
            return fallback

    def _sanitize_text(self, text):
        """Basic sanitization for general text tokens."""
        if not isinstance(text, str): text = str(text)
        text = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '', text) # Remove illegal chars
        text = re.sub(r'\s+', '_', text).strip('_') # Replace whitespace with underscore
        return text[:100] # Limit length

    def _clean_text(self, text, max_length):
        """More specific cleaning (like for genre)."""
        if not isinstance(text, str): text = str(text)
        text = re.sub(r'[^\w\s-]', '', text) # Allow word chars, space, hyphen
        text = re.sub(r'\s+', ' ', text).strip()
        if len(text) > max_length:
            text = text[:max_length].rsplit(' ', 1)[0] # Truncate at word boundary
        return text.replace(' ', '_') # Replace spaces with underscores for filename

    def _sanitize_filepath(self, path_str):
        """Sanitizes a potentially full path string."""
        # Split into components, sanitize each, rejoin
        parts = path_str.replace('\\', '/').split('/')
        sanitized_parts = [self._sanitize_filename(part) for part in parts if part and part != '.'] # Sanitize each part, remove empty/dot parts
        # Join with OS-specific separator for preview, but return with '/' for ComfyUI consistency
        # result = os.path.join(*sanitized_parts) # This might use backslashes on Windows
        result = "/".join(sanitized_parts)
        return result


# =================================================================================
# == Core Node Class: FilenameCounterNode                                      ==
# =================================================================================

class FilenameCounterNode:
    """
    MD: Filename Counter

    Provides a persistent, auto-incrementing counter stored in a file.
    Useful for generating sequential numbers (#0001, #0002, ...) in filenames
    across multiple ComfyUI runs. Supports different contexts/projects.
    """

    # Class variable for the path to the counter storage file
    COUNTERS_FILE = os.path.join(folder_paths.get_temp_directory(), "md_filename_counters.json") # Changed to JSON

    @classmethod
    def INPUT_TYPES(cls):
        """Define node inputs with detailed tooltips."""
        return {
            "required": {
                "context_key": ("STRING", {
                    "default": "default_counter", # More descriptive default
                    "tooltip": (
                        "CONTEXT KEY\n"
                        "- A unique name for this specific counter.\n"
                        "- Allows multiple independent counters (e.g., 'project_a_images', 'project_b_audio').\n"
                        "- Used as the key in the persistent storage file."
                    )
                }),
                "start_value": ("INT", {
                    "default": 1, "min": 0, "max": 999999, # Increased max
                    "tooltip": (
                        "START VALUE\n"
                        "- The number the counter will start (or reset) to.\n"
                        "- Set to 0 if you want the first output to be #0000 (with padding 4)."
                    )
                }),
                "increment": ("INT", {
                    "default": 1, "min": 1, "max": 100,
                    "tooltip": (
                        "INCREMENT STEP\n"
                        "- The amount to add to the counter after each execution.\n"
                        "- Usually kept at 1 for sequential numbering."
                    )
                }),
                "padding": ("INT", {
                    "default": 4, "min": 0, "max": 8, # Allow 0 padding
                    "tooltip": (
                        "ZERO PADDING\n"
                        "- Total number of digits for the output number, padded with leading zeros.\n"
                        "- 4 -> 0001, 0002, ...\n"
                        "- 0 -> 1, 2, ... (no padding)."
                    )
                }),
            },
            "optional": {
                "reset_counter": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "RESET COUNTER\n"
                        "- True: Forces this specific counter ('context_key') back to 'start_value' during this execution.\n"
                        "- False: Uses the current stored value and increments it."
                    )
                }),
                "prefix": ("STRING", {"default": "#", "tooltip": "Optional prefix string added before the counter number (e.g., '#', 'File_')."}),
                "suffix": ("STRING", {"default": "", "tooltip": "Optional suffix string added after the counter number (e.g., '_final')."}),
                 # Optional trigger for conditional execution
                "trigger": ("BOOLEAN", {"default": True, "label_on": "INCREMENT ENABLED", "label_off": "INCREMENT DISABLED",
                                        "tooltip": "If False, the node outputs the current value but does NOT increment the counter."})
            }
        }

    RETURN_TYPES = ("STRING", "INT", "STRING")
    RETURN_NAMES = ("formatted_counter", "current_value", "info") # Renamed for clarity
    FUNCTION = "get_counter"
    CATEGORY = "MD_Nodes/Utility" # Corrected category
    OUTPUT_NODE = False # This node primarily outputs data for use elsewhere

    def __init__(self):
        """Initializes by loading counters from the file."""
        self.counters = self._load_counters()

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """
        Forces re-execution because the node's output depends on its internal state (the counter).
        """
        return secrets.token_hex(16)

    def get_counter(self, context_key, start_value=1, increment=1, padding=4,
                    reset_counter=False, prefix="#", suffix="", trigger=True):
        """
        Retrieves, formats, and increments the counter for the given context.

        Args:
            context_key: Unique identifier for the counter.
            start_value: Value to use if counter doesn't exist or is reset.
            increment: Amount to increment the counter by (if triggered).
            padding: Number of digits for zero-padding (0 for none).
            reset_counter: If True, resets the counter to start_value.
            prefix: String to prepend to the formatted number.
            suffix: String to append to the formatted number.
            trigger: If False, returns the current value without incrementing.

        Returns:
            Tuple: (formatted_string, current_raw_value, info_string)
        """
        info = f"Context: {context_key}\n"
        current_value = -1 # Default error value

        try:
            # Sanitize context key
            safe_context_key = re.sub(r'[^\w\-]+', '_', context_key) # Allow word chars and hyphen
            if not safe_context_key: safe_context_key = "default_counter"

            # Load counters again in case another instance modified the file
            self.counters = self._load_counters()

            # Reset if requested
            if reset_counter:
                self.counters[safe_context_key] = max(0, start_value) # Ensure start is non-negative
                self._save_counters()
                info += f"Action: RESET to {self.counters[safe_context_key]}\n"
                logger.info(f"[FilenameCounter] Counter '{safe_context_key}' reset to {self.counters[safe_context_key]}.")

            # Initialize counter if it doesn't exist
            if safe_context_key not in self.counters:
                self.counters[safe_context_key] = max(0, start_value)
                info += f"Action: Initialized to {self.counters[safe_context_key]}\n"

            # Get current value
            current_value = self.counters[safe_context_key]

            # Format the current value
            if padding > 0:
                formatted_num = f"{current_value:0{padding}d}"
            else:
                formatted_num = str(current_value) # No padding
            formatted_string = f"{prefix}{formatted_num}{suffix}"

            # Increment for next time only if triggered
            next_value = current_value # Default next value if not triggered
            if trigger:
                 next_value = current_value + max(1, increment) # Ensure increment is at least 1
                 self.counters[safe_context_key] = next_value
                 self._save_counters()
                 info += f"Action: Incremented by {increment}\n"
            else:
                 info += "Action: Read Only (Trigger OFF)\n"


            info += f"Output Value: {current_value}\n"
            info += f"Next Value (Stored): {next_value}"

            return (formatted_string, current_value, info)

        except Exception as e:
            logging.error(f"[FilenameCounter] Error processing counter '{context_key}': {e}", exc_info=True)
            error_info = f"ERROR: Could not process counter '{context_key}'.\n{e}\n{traceback.format_exc()}"
            # Return safe defaults on error
            return (f"{prefix}ERROR{suffix}", current_value, error_info)


    def _load_counters(self):
        """Load counters from persistent JSON storage."""
        counters = {}
        try:
            if os.path.exists(self.COUNTERS_FILE):
                with open(self.COUNTERS_FILE, 'r', encoding='utf-8') as f:
                    try:
                        counters = json.load(f)
                        if not isinstance(counters, dict):
                             logging.warning(f"[FilenameCounter] Counter file '{self.COUNTERS_FILE}' is not a valid JSON dictionary. Resetting.")
                             counters = {}
                    except json.JSONDecodeError:
                         logging.warning(f"[FilenameCounter] Counter file '{self.COUNTERS_FILE}' is corrupted. Resetting.")
                         counters = {} # Reset if file is corrupted
        except IOError as e:
            logger.error(f"[FilenameCounter] Error loading counters file: {e}")
        except Exception as e:
            logger.error(f"[FilenameCounter] Unexpected error loading counters: {e}")
        return counters

    def _save_counters(self):
        """Save counters to persistent JSON storage."""
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(self.COUNTERS_FILE), exist_ok=True)
            # Write the entire dictionary as JSON
            with open(self.COUNTERS_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.counters, f, indent=4)
        except IOError as e:
            logger.error(f"[FilenameCounter] Error saving counters file: {e}")
        except Exception as e:
            logger.error(f"[FilenameCounter] Unexpected error saving counters: {e}")


# =================================================================================
# == Node Registration                                                         ==
# =================================================================================

NODE_CLASS_MAPPINGS = {
    "SmartFilenameBuilder": SmartFilenameBuilder,
    "FilenameTokenReplacer": FilenameTokenReplacer,
    "FilenameCounterNode": FilenameCounterNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SmartFilenameBuilder": "MD: Smart Filename Builder",   # Added MD: prefix
    "FilenameTokenReplacer": "MD: Filename Token Replacer", # Added MD: prefix
    "FilenameCounterNode": "MD: Filename Counter",        # Added MD: prefix
}