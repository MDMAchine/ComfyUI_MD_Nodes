# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
# ████ MD_Nodes/AdvancedTextNode – Text input with wildcards & transforms ████▓▒░
# © 2025 MDMAchine
# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
# ░▒▓ ORIGIN & DEV:
#   • Cast into the void by: MDMAchine
#   • Enhanced by: Gemini, Claude
#   • License: Apache 2.0 — Sharing is caring

# ░▒▓ DESCRIPTION:
#   A versatile text input node with seed-controlled wildcard support, text
#   transformations (case, whitespace), and a companion Text File Loader node.
#   Ideal for dynamic prompts, large text blocks (YAML/JSON), and external data.

# ░▒▓ FEATURES:
#   ✓ Large multiline text input.
#   ✓ Seed-controlled wildcards: {option1|option2} or __option1|option2__.
#   ✓ Nested wildcard support.
#   ✓ Wildcard seed selection: Use {seed1|seed2|seed3} to randomly pick a seed value.
#   ✓ Text transformations: lowercase, uppercase, whitespace control.
#   ✓ Multiple outputs: processed text, original text, seed used, selected seed INT.
#   ✓ Companion 'Text File Loader' node for external file import.

# ░▒▓ CHANGELOG:
#   - v1.6.1 (Randomness Fix - Nov 2025):
#     • FIXED: Replaced manual modulo math with `rng.choice()` to prevent seed clustering.
#     • IMPROVED: Better distribution of values when using small seed lists.
#   - v1.6.0 (Seed List Feature - Nov 2025):
#     • ADDED: New 'seed_list' input parameter for wildcard seed selection.
#     • ADDED: New 'selected_seed' INT output - randomly selects seed from wildcard list.
#     • ADDED: New 'seed_offset' parameter to reduce repetition in batch workflows.

# ░▒▓ CONFIGURATION:
#   → Primary Use: Dynamic prompt generation using {wildcards|options} with a fixed seed.
#   → Secondary Use: A simple text box for holding complex YAML or JSON configs.
#   → Edge Use: Loading entire workflow templates from external .txt files via the loader.

# ░▒▓ WARNING:
#   This node may trigger:
#   ▓▒░ Obsessively nesting wildcards {like a {russian {doll|matryoshka}|madman}|until you hit a stack overflow}.
#   ▓▒░ Spending 4 hours 'seed surfing' just to find the one combo that doesn't say 'green'.
#   ▓▒░ A sudden, uncontrollable urge to create a .NFO file for your workflow.
#   Consult your nearest demoscene vet if hallucinations persist.
# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄


# =================================================================================
# == Standard Library Imports                                                     ==
# =================================================================================
import logging
import os
import random
import re
import secrets
import traceback

# =================================================================================
# == Third-Party Imports                                                          ==
# =================================================================================
# (No third-party imports needed)

# =================================================================================
# == ComfyUI Core Modules                                                         ==
# =================================================================================
# (No core ComfyUI imports needed directly in this file)

# =================================================================================
# == Local Project Imports                                                        ==
# =================================================================================
# (No local project imports in this file)

# =================================================================================
# == Helper Classes & Dependencies                                                ==
# =================================================================================
# (No helper classes needed)

# =================================================================================
# == Core Node Class: AdvancedTextNode                                            ==
# =================================================================================

class AdvancedTextNode:
    """
    A versatile text input node with seed-controlled wildcard processing,
    text transformations, and multiple output options including text length.
    """

    @classmethod
    def INPUT_TYPES(cls):
        """Define all input parameters with standardized tooltips."""
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "dynamicPrompts": False, # Keep dynamicPrompts as ComfyUI uses it
                    "tooltip": (
                        "TEXT INPUT\n"
                        "- Main text input area.\n"
                        "- Enter prompts, YAML, JSON, or any text content.\n"
                        "- Use wildcards like {option1|option2} or __option1|option2__ for random variations if enabled."
                    )
                }),
            },
            "optional": {
                "seed": ("INT", {
                    "default": 0,
                    "min": -1, # Allow -1 for dynamic seed
                    "max": 0xffffffffffffffff,
                    "step": 1,
                    "tooltip": (
                        "SEED\n"
                        "- Controls randomization for wildcard selection.\n"
                        "- Same seed = same wildcard choices every time.\n"
                        "- Change seed for different variations.\n"
                        "- Set to -1 to generate a new random seed on each run (forces re-execution)."
                    )
                }),
                "seed_list": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": (
                        "SEED LIST (WILDCARD)\n"
                        "- Provide a list of seeds using wildcard syntax.\n"
                        "- Example: {1234|5678|9012} or __42|777|1337__\n"
                        "- The node will randomly select ONE seed from the list.\n"
                        "- Selected seed is output as 'selected_seed' INT.\n"
                        "- Use the main 'seed' input to control which seed is chosen.\n"
                        "- Leave empty to use only the main 'seed' parameter."
                    )
                }),
                "seed_offset": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "step": 1,
                    "tooltip": (
                        "SEED OFFSET\n"
                        "- Adds an offset to the seed before selecting from seed_list.\n"
                        "- This helps reduce repetition in batch workflows.\n"
                        "- Example: seed=42, offset=0 → uses 42 for selection\n"
                        "- Example: seed=42, offset=1 → uses 43 for selection\n"
                        "- In batch mode, increment this to get different variations.\n"
                        "- Has no effect if seed_list is empty."
                    )
                }),
                "wildcard_mode": ("BOOLEAN", {
                    "default": False,
                    "label_on": "enabled",
                    "label_off": "disabled",
                    "tooltip": (
                        "WILDCARD MODE\n"
                        "- Enable processing of {option1|option2} or __option1|option2__ patterns.\n"
                        "- ON: Randomly selects one option based on the seed.\n"
                        "- OFF: Text passes through unchanged."
                    )
                }),
                "strip_whitespace": ("BOOLEAN", {
                    "default": False,
                    "label_on": "enabled",
                    "label_off": "disabled",
                    "tooltip": (
                        "STRIP WHITESPACE\n"
                        "- Removes leading and trailing whitespace from the *entire* text block.\n"
                        "- Useful for cleaning up pasted content."
                    )
                }),
                "lowercase": ("BOOLEAN", {
                    "default": False,
                    "label_on": "enabled",
                    "label_off": "disabled",
                    "tooltip": (
                        "FORCE LOWERCASE\n"
                        "- Converts the entire output text to lowercase.\n"
                        "- Note: This overrides 'Force Uppercase' if both are enabled."
                    )
                }),
                "uppercase": ("BOOLEAN", {
                    "default": False,
                    "label_on": "enabled",
                    "label_off": "disabled",
                    "tooltip": (
                        "FORCE UPPERCASE\n"
                        "- Converts the entire output text to UPPERCASE.\n"
                        "- Note: 'Force Lowercase' takes priority if both are enabled."
                    )
                }),
                "remove_extra_spaces": ("BOOLEAN", {
                    "default": True,
                    "label_on": "enabled",
                    "label_off": "disabled",
                    "tooltip": (
                        "REMOVE EXTRA SPACES\n"
                        "- Collapses multiple spaces into single spaces.\n"
                        "- Trims leading/trailing spaces from each line.\n"
                        "- Helps normalize formatting."
                    )
                }),
                "wildcard_syntax": (["curly_braces", "double_underscore"], {
                    "default": "curly_braces",
                    "tooltip": (
                        "WILDCARD SYNTAX\n"
                        "- Choose the pattern style for wildcards:\n"
                        "- 'curly_braces': {option1|option2}\n"
                        "- 'double_underscore': __option1|option2__"
                    )
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "INT")
    RETURN_NAMES = ("processed_text", "original_text", "seed_used", "selected_seed")
    FUNCTION = "process_text"
    CATEGORY = "MD_Nodes/Text"

    @classmethod
    def IS_CHANGED(cls, text, seed=0, seed_list="", seed_offset=0, wildcard_mode=False, strip_whitespace=False,
                       lowercase=False, uppercase=False, remove_extra_spaces=True,
                       wildcard_syntax="curly_braces"):
        """
        Determine if the node should re-run based on input changes.
        If seed is -1, treat it as dynamic (always re-run).
        """
        # Note: ComfyUI typically handles dynamic behavior via negative seeds or similar patterns internally.
        # This implementation explicitly forces re-run for seed = -1 by returning a unique random hex string.
        # This is a valid ComfyUI pattern for forcing execution.
        if seed == -1:
            return secrets.token_hex(16) 

        # Otherwise, changes in any processing parameter should trigger re-run
        # Returning a tuple/list of all relevant inputs is standard practice
        # ComfyUI hashes this to detect changes.
        return (text, seed, seed_list, seed_offset, wildcard_mode, strip_whitespace, lowercase, uppercase,
                remove_extra_spaces, wildcard_syntax)

    def _process_wildcards_recursive(self, text, pattern, rng):
        """
        Recursively processes nested wildcards. Helper function.

        Args:
            text: The text containing wildcards.
            pattern: The compiled regex pattern for the syntax.
            rng: The seeded random number generator.

        Returns:
            str: Text with wildcards processed.
        """
        iteration = 0
        max_iterations = 100 # Safety break for potential infinite loops

        # CRITICAL FIX: The loop must continue until no more *innermost* matches are found.
        # By repeatedly searching and replacing only the innermost pattern, we ensure
        # that nested wildcards are resolved from inside out.
        while iteration < max_iterations:
            match = pattern.search(text)
            if not match:
                break # No more wildcards of this type found

            options_str = match.group(1)
            options = [opt.strip() for opt in options_str.split('|')]
            
            # Validate options to prevent crash on empty brackets {}
            if not options:
                 options = [""] 

            # Use standard rng.choice() for statistically better distribution.
            chosen_option = rng.choice(options)

            # Replace only the innermost match found in this iteration
            text = text[:match.start()] + chosen_option + text[match.end():]
            iteration += 1

        if iteration == max_iterations:
             logging.warning("[AdvancedTextNode] Max wildcard processing iterations reached. Possible runaway recursion?")

        return text


    def process_wildcards_curly(self, text, seed):
        """
        Process wildcards in {option1|option2|option3} format, supporting nesting.
        """
        rng = random.Random(seed)
        # Non-greedy pattern to find the *innermost* wildcards first
        # This pattern matches any { } that contains only text and | symbols, but no nested { or }
        pattern = re.compile(r'\{([^{}]+?)\}')
        return self._process_wildcards_recursive(text, pattern, rng)

    def process_wildcards_underscore(self, text, seed):
        """
        Process wildcards in __option1|option2|option3__ format, supporting nesting.
        """
        rng = random.Random(seed)
        # Non-greedy pattern to find the *innermost* wildcards first
        pattern = re.compile(r'__([^_]+?)__')
        return self._process_wildcards_recursive(text, pattern, rng)

    def process_text(self, text, seed=0, seed_list="", seed_offset=0, wildcard_mode=False, strip_whitespace=False,
                       lowercase=False, uppercase=False, remove_extra_spaces=True,
                       wildcard_syntax="curly_braces"):
        """
        Main execution function. Processes text based on input parameters.

        Args:
            text: Main text input string.
            seed: Seed for random selection.
            seed_list: String containing wildcard list of seeds to choose from.
            seed_offset: Offset added to seed for seed_list selection.
            wildcard_mode: Boolean to enable/disable wildcard processing.
            strip_whitespace: Boolean to strip overall whitespace.
            lowercase: Boolean to force lowercase.
            uppercase: Boolean to force uppercase.
            remove_extra_spaces: Boolean to collapse spaces.
            wildcard_syntax: String defining the wildcard pattern type.

        Returns:
            tuple: (processed_text, original_text, seed_used, selected_seed)
        """
        original_text = text
        processed_text = text
        seed_used = seed
        selected_seed = 0  # Default output seed

        try:
            # Handle dynamic seed generation (-1 means generate random)
            if seed == -1:
                seed_used = random.randint(0, 0xffffffffffffffff)
                print(f"[AdvancedTextNode] 🎲 Using dynamically generated seed: {seed_used}")
            else:
                 logging.debug(f"[AdvancedTextNode] Using provided seed: {seed_used}")

            # Process seed_list if provided
            if seed_list and seed_list.strip():
                # Apply offset to create variation in seed selection
                selection_seed = (seed_used + seed_offset) & 0xffffffffffffffff  # Keep within bounds
                logging.debug(f"[AdvancedTextNode] Processing seed_list with syntax '{wildcard_syntax}' and selection_seed {selection_seed}")
                
                # Process the seed_list string using wildcard processing
                if wildcard_syntax == "curly_braces":
                    selected_seed_str = self.process_wildcards_curly(seed_list, selection_seed)
                elif wildcard_syntax == "double_underscore":
                    selected_seed_str = self.process_wildcards_underscore(seed_list, selection_seed)
                else:
                    selected_seed_str = seed_list
                
                # Convert the selected seed string to integer
                try:
                    selected_seed = int(selected_seed_str.strip())
                    print(f"[AdvancedTextNode] 🌱 Selected seed from list: {selected_seed}")
                except ValueError:
                    logging.warning(f"[AdvancedTextNode] Could not convert '{selected_seed_str}' to integer. Using 0.")
                    selected_seed = 0
            else:
                # If no seed_list provided, output the seed_used as selected_seed
                selected_seed = seed_used
                logging.debug(f"[AdvancedTextNode] No seed_list provided, using seed_used as selected_seed: {selected_seed}")

            if wildcard_mode:
                logging.debug(f"[AdvancedTextNode] Processing wildcards with syntax '{wildcard_syntax}' and seed {seed_used}")
                if wildcard_syntax == "curly_braces":
                    processed_text = self.process_wildcards_curly(processed_text, seed_used)
                elif wildcard_syntax == "double_underscore":
                    processed_text = self.process_wildcards_underscore(processed_text, seed_used)
                logging.debug("[AdvancedTextNode] Wildcard processing complete.")

            if strip_whitespace:
                processed_text = processed_text.strip()
                logging.debug("[AdvancedTextNode] Stripped leading/trailing whitespace.")

            if remove_extra_spaces:
                processed_text = re.sub(r' +', ' ', processed_text) # Collapse multiple spaces
                processed_text = '\n'.join(line.strip() for line in processed_text.split('\n')) # Trim lines
                logging.debug("[AdvancedTextNode] Removed extra spaces and trimmed lines.")

            # Lowercase takes precedence
            if lowercase:
                processed_text = processed_text.lower()
                logging.debug("[AdvancedTextNode] Converted text to lowercase.")
            elif uppercase: # Only apply if lowercase is false
                processed_text = processed_text.upper()
                logging.debug("[AdvancedTextNode] Converted text to uppercase.")

            # Return all outputs
            return (processed_text, original_text, seed_used, selected_seed)

        except Exception as e:
            logging.error(f"[AdvancedTextNode] Error processing text: {e}")
            logging.debug(traceback.format_exc())
            print(f"[AdvancedTextNode] ⚠️ Error: {e}. Returning original text unchanged.")
            # Graceful failure: Return original text, seed, and 0 for selected_seed
            return (original_text, original_text, seed_used, 0)

# =================================================================================
# == Core Node Class: TextFileLoader                                            ==
# =================================================================================

class TextFileLoader:
    """
    Companion node to load text content from external files.
    """

    @classmethod
    def INPUT_TYPES(cls):
        """Define input parameters for the file loader."""
        return {
            "required": {
                "file_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": (
                        "FILE PATH\n"
                        "- Full path to the text file (.txt, .yaml, .json, etc.).\n"
                        "- Supports relative paths from ComfyUI root or absolute paths.\n"
                        "- Example (relative): `input/my_prompts.txt`\n"
                        "- Example (absolute): `C:/data/config.yaml`"
                    )
                }),
            },
            "optional": {
                "encoding": (["utf-8", "ascii", "latin-1"], {
                    "default": "utf-8",
                    "tooltip": (
                        "FILE ENCODING\n"
                        "- Character encoding of the file.\n"
                        "- 'utf-8': Recommended universal standard.\n"
                        "- 'ascii': Basic English characters only.\n"
                        "- 'latin-1': Western European characters.\n"
                        "- If you see garbled text, try a different encoding."
                    )
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "load_file"
    CATEGORY = "MD_Nodes/Text"

    @classmethod
    def IS_CHANGED(cls, file_path, encoding="utf-8"):
        """
        Check if the file content has changed based on modification time.
        Forces re-run if the file is modified or path changes.
        """
        try:
            # Normalize path for consistent hashing
            norm_path = os.path.normpath(file_path)
            if not os.path.exists(norm_path):
                 m_time = -1 # Indicate file not found
            else:
                 m_time = os.path.getmtime(norm_path)
        except Exception as e:
            # Handle potential errors like invalid paths during check
            logging.warning(f"[TextFileLoader] IS_CHANGED check failed for path '{file_path}': {e}")
            m_time = -2 # Indicate an error occurred during check

        # Return tuple that ComfyUI hashes
        return (norm_path, encoding, m_time)

    def load_file(self, file_path, encoding="utf-8"):
        """
        Load text content from the specified file.

        Args:
            file_path: The path to the text file (str).
            encoding: The file encoding to use (str).

        Returns:
            tuple: (text_content,) or (error_message,) on failure.
        """
        print(f"[TextFileLoader] 📂 Attempting to load file: {file_path}")
        try:
            # Basic path validation
            if not file_path or not isinstance(file_path, str):
                 error_msg = "[TextFileLoader] Error: Invalid file path provided."
                 logging.error(error_msg)
                 return (error_msg,) # Must return a tuple

            if not os.path.exists(file_path):
                 error_msg = f"[TextFileLoader] Error: File not found at path: {file_path}"
                 logging.error(error_msg)
                 return (error_msg,) # Must return a tuple

            if not os.path.isfile(file_path):
                 error_msg = f"[TextFileLoader] Error: Path exists but is not a file: {file_path}"
                 logging.error(error_msg)
                 return (error_msg,) # Must return a tuple

            with open(file_path, 'r', encoding=encoding) as f:
                text = f.read()
            print(f"[TextFileLoader] ✅ Successfully loaded file: {file_path}")
            return (text,) # Must return a tuple

        except Exception as e:
            error_msg = f"[TextFileLoader] Error loading file '{file_path}': {e}"
            logging.error(error_msg)
            logging.debug(traceback.format_exc())
            print(f"[TextFileLoader] ⚠️ Error: {e}. Check console/logs.")
            # Graceful failure: Return error message in tuple
            return (error_msg,)

# =================================================================================
# == Node Registration                                                            ==
# =================================================================================

NODE_CLASS_MAPPINGS = {
    "AdvancedTextNode": AdvancedTextNode,
    "TextFileLoader": TextFileLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AdvancedTextNode": "MD: Advanced Text Input",
    "TextFileLoader": "MD: Text File Loader",
}