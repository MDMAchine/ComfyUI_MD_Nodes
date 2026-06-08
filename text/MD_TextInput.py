# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# █▓▒░                                                                     ░▒▓█
# █▓▒░    MD_Nodes/AdvancedTextNode – Text input with wildcards v1.9.0     ░▒▓█
# █▓▒░                                                                     ░▒▓█
# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
# ╠═ © 2026 MDMAchine
# ╠═ License: GNU General Public License v3.0 (GPL v3)
# ║
# ║  This program is free software: you can redistribute it and/or modify
# ║  it under the terms of the GNU General Public License as published by
# ║  the Free Software Foundation, either version 3 of the License, or
# ║  (at your option) any later version.
# ║
# ║  This program is distributed in the hope that it will be useful,
# ║  but WITHOUT ANY WARRANTY; without even the implied warranty of
# ║  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# ║  GNU General Public License for more details.
# ║
# ║  You should have received a copy of the GNU General Public License
# ║  along with this program. If not, see <https://www.gnu.org/licenses/>.
# ╠════════════════════════════════════════════════════════════════════════════
# ║ ░▒▓ ORIGIN & DEV:
# ║   • Cast into the void by: MDMAchine
# ║   • Enhanced by: Gemini, Claude
# ║
# ║ ░▒▓ DESCRIPTION:
# ║   A versatile text input node with seed-controlled wildcard support, text
# ║   transformations (case, whitespace), and a companion Text File Loader node.
# ║   NOTE: As a string utility node, this runs entirely in the public wrapper domain.
# ║
# ║ ░▒▓ FEATURES:
# ║   ✓ Large multiline text input.
# ║   ✓ Seed-controlled wildcards: {option1|option2} or __option1|option2__.
# ║   ✓ Nested wildcard support with robust error handling.
# ║   ✓ Text transformations: lowercase, uppercase, whitespace control.
# ║   ✓ Multiple seed modes: fixed, random, increment (auto-increment for batch).
# ║   ✓ Companion 'Text File Loader' node for external file import.
# ║
# ║ ░▒▓ CHANGELOG:
# ║   - v1.9.0 (Enterprise Standards - Feb 2026):
# ║       • ADDED: PerformanceProfiler class (v1.5.3 standard).
# ║       • ADDED: debug_mode parameter.
# ║       • REFACTOR: Tooltips strictly updated to 5-part v1.5.4 standard.
# ║   - v1.8.0 (Seed Precision Fix - Nov 2025):
# ║       • CRITICAL FIX: Capped seed range to JavaScript's MAX_SAFE_INTEGER.
# ╚════════════════════════════════════════════════════════════════════════════

# =================================================================================
# == Standard Library Imports                                                     ==
# =================================================================================
VERSION = "v1.9.0"  # UPS v1.5.8


import logging
import os
import random
import re
import secrets
import traceback
import time

# =================================================================================
# == Seed Constants (JavaScript Safe Range)                                      ==
# =================================================================================
SEED_MIN = 0
SEED_MAX = 9007199254740991  # JS-safe range for full reproducibility

# =================================================================================
# == Helper Classes (Enterprise Standards)                                      ==
# =================================================================================

class PerformanceProfiler:
    """Standard performance profiler for MD_Nodes."""
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.timings = {}
        self.start_times = {}
    
    def start(self, operation_name):
        if not self.enabled: return
        self.start_times[operation_name] = time.perf_counter()
    
    def stop(self, operation_name):
        if not self.enabled: return
        if operation_name in self.start_times:
            elapsed = time.perf_counter() - self.start_times[operation_name]
            if operation_name not in self.timings:
                self.timings[operation_name] = []
            self.timings[operation_name].append(elapsed)
            del self.start_times[operation_name]
    
    def get_total_time(self):
        if not self.enabled or not self.timings: return 0.0
        return sum(sum(times) for times in self.timings.values())
    
    def print_report(self):
        if not self.enabled or not self.timings: return
        logging.info("\n⏱️  PERFORMANCE (Text Parse):")
        total = self.get_total_time()
        logging.info(f"    • Total Time: {total:.4f}s")
        for op_name, times in sorted(self.timings.items()):
            avg = sum(times) / len(times)
            if len(times) == 1:
                logging.info(f"    • {op_name}: {avg:.4f}s")
            else:
                logging.info(f"    • {op_name}: {avg:.4f}s avg ({len(times)}x)")

# =================================================================================
# == Core Node Class: AdvancedTextNode                                            ==
# =================================================================================

class AdvancedTextNode:
    """
    A versatile text input node with seed-controlled wildcard processing,
    text transformations, and multiple output options including text length.
    """
    
    _PATTERN_CURLY = re.compile(r'\{([^{}]+?)\}')
    _PATTERN_UNDERSCORE = re.compile(r'__([^_]+?)__')

    @staticmethod
    def _validate_seed(seed_value):
        try:
            int_value = int(seed_value)
        except (ValueError, TypeError):
            logging.warning(f"[AdvancedTextNode] Invalid seed value: {seed_value}. Using {SEED_MIN}.")
            return SEED_MIN
        
        if int_value > SEED_MAX:
            logging.warning(f"[AdvancedTextNode] Seed {int_value} exceeds JS-safe range. Clamping to {SEED_MAX}.")
            return SEED_MAX
        elif int_value < SEED_MIN:
            logging.warning(f"[AdvancedTextNode] Seed {int_value} below minimum. Clamping to {SEED_MIN}.")
            return SEED_MIN
        
        return int_value

    @staticmethod
    def _generate_random_seed():
        return secrets.randbelow(SEED_MAX + 1)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "dynamicPrompts": False,
                    "tooltip": (
                        "TEXT INPUT\n"
                        "• Purpose: Main text input area for prompts, YAML, JSON, etc.\n"
                        "• Usage: Type raw text or use wildcards like {option1|option2}.\n"
                        "• Note: Multi-line compatible."
                    )
                }),
            },
            "optional": {
                "seed": ("INT", {
                    "default": 0,
                    "min": SEED_MIN,
                    "max": SEED_MAX,
                    "step": 1,
                    "tooltip": (
                        "RANDOM SEED\n"
                        "• Purpose: Controls randomization for wildcard selection.\n"
                        "• Range: 0 to 9,007,199,254,740,991 (JS Safe Max).\n"
                        "• Note: Ignored if 'seed_mode' is set to 'random'.\n"
                        "\n⭐ Recommended: Fixed value for reproducible generation."
                    )
                }),
                "seed_mode": (["fixed", "random", "increment"], {
                    "default": "fixed",
                    "tooltip": (
                        "SEED MODE\n"
                        "• Purpose: Defines how the seed updates per execution.\n"
                        "• Options:\n"
                        "  - fixed: Uses the exact seed value provided.\n"
                        "  - random: Generates a new seed every run.\n"
                        "  - increment: Auto-adds 1 to the seed (great for batches).\n"
                        "\n⭐ Recommended: 'fixed' for testing, 'increment' for batching."
                    )
                }),
                "seed_list": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": (
                        "SEED LIST (WILDCARD)\n"
                        "• Purpose: Randomly pick ONE seed from a wildcard string.\n"
                        "• Example: {1234|5678|9012} or __42|777|1337__\n"
                        "• Output: Outputs chosen value to the 'selected_seed' INT.\n"
                        "\n⭐ Recommended: Leave empty unless injecting specific seeds."
                    )
                }),
                "seed_offset": ("INT", {
                    "default": 0,
                    "min": SEED_MIN,
                    "max": SEED_MAX,
                    "step": 1,
                    "tooltip": (
                        "SEED OFFSET\n"
                        "• Purpose: Adds an integer offset to the seed before processing seed_list.\n"
                        "• Effect: Helps reduce repetition in batch workflows.\n"
                        "\n⭐ Recommended: 0."
                    )
                }),
                "wildcard_mode": ("BOOLEAN", {
                    "default": False,
                    "label_on": "enabled",
                    "label_off": "disabled",
                    "tooltip": (
                        "WILDCARD MODE\n"
                        "• Purpose: Enable processing of {opt1|opt2} or __opt1|opt2__ patterns.\n"
                        "• Effect: If disabled, text passes through exactly as written.\n"
                        "\n⭐ Recommended: True if writing dynamic prompts."
                    )
                }),
                "strip_whitespace": ("BOOLEAN", {
                    "default": False,
                    "label_on": "enabled",
                    "label_off": "disabled",
                    "tooltip": (
                        "STRIP WHITESPACE\n"
                        "• Purpose: Removes leading/trailing whitespace from the ENTIRE block.\n"
                        "\n⭐ Recommended: False (unless cleaning bad JSON/YAML)."
                    )
                }),
                "lowercase": ("BOOLEAN", {
                    "default": False,
                    "label_on": "enabled",
                    "label_off": "disabled",
                    "tooltip": (
                        "FORCE LOWERCASE\n"
                        "• Purpose: Converts entire output text to lowercase.\n"
                        "• Note: Overrides 'Force Uppercase' if both are true.\n"
                        "\n⭐ Recommended: False."
                    )
                }),
                "uppercase": ("BOOLEAN", {
                    "default": False,
                    "label_on": "enabled",
                    "label_off": "disabled",
                    "tooltip": (
                        "FORCE UPPERCASE\n"
                        "• Purpose: Converts entire output text to UPPERCASE.\n"
                        "\n⭐ Recommended: False."
                    )
                }),
                "remove_extra_spaces": ("BOOLEAN", {
                    "default": True,
                    "label_on": "enabled",
                    "label_off": "disabled",
                    "tooltip": (
                        "REMOVE EXTRA SPACES\n"
                        "• Purpose: Collapses multiple spaces into single spaces.\n"
                        "\n⭐ Recommended: True (keeps prompts clean)."
                    )
                }),
                "wildcard_syntax": (["curly_braces", "double_underscore"], {
                    "default": "curly_braces",
                    "tooltip": (
                        "WILDCARD SYNTAX\n"
                        "• Purpose: Choose the pattern style for wildcard parsing.\n"
                        "• Options: {curly_braces} or __double_underscore__.\n"
                        "\n⭐ Recommended: curly_braces."
                    )
                }),
                "debug_mode": (["0 - Silent", "1 - Info"], {
                    "default": "0 - Silent",
                    "tooltip": "LOGGING VERBOSITY\n• Controls console output and parser profiling."
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("processed_text", "original_text", "seed_used", "selected_seed", "text_length", "wildcard_count")
    FUNCTION = "process_text"
    CATEGORY = "MD_Nodes/Text"

    @classmethod
    def IS_CHANGED(cls, text, seed=0, seed_mode="fixed", seed_list="", seed_offset=0, wildcard_mode=False, strip_whitespace=False,
                        lowercase=False, uppercase=False, remove_extra_spaces=True,
                        wildcard_syntax="curly_braces", debug_mode="0 - Silent"):
        if seed_mode in ["random", "increment"]:
            return secrets.token_hex(16)
        return (text, seed, seed_mode, seed_list, seed_offset, wildcard_mode, strip_whitespace, lowercase, uppercase,
                remove_extra_spaces, wildcard_syntax)

    def _process_wildcards_recursive(self, text, pattern, rng):
        iteration = 0
        wildcard_count = 0
        max_iterations = 100 

        while iteration < max_iterations:
            match = pattern.search(text)
            if not match:
                break 

            options_str = match.group(1)
            options = [opt.strip() for opt in options_str.split('|') if opt.strip()]
            
            if not options:
                logging.warning(f"[AdvancedTextNode] Empty wildcard detected: {match.group(0)}. Replacing with empty string.")
                options = [""] 

            chosen_option = rng.choice(options)
            text = text[:match.start()] + chosen_option + text[match.end():]
            iteration += 1
            wildcard_count += 1

        if iteration == max_iterations:
             logging.warning("[AdvancedTextNode] Max wildcard processing iterations reached. Possible runaway recursion?")

        return text, wildcard_count

    def process_wildcards_curly(self, text, seed):
        rng = random.Random(seed)
        return self._process_wildcards_recursive(text, self._PATTERN_CURLY, rng)

    def process_wildcards_underscore(self, text, seed):
        rng = random.Random(seed)
        return self._process_wildcards_recursive(text, self._PATTERN_UNDERSCORE, rng)

    def process_text(self, text, seed=0, seed_mode="fixed", seed_list="", seed_offset=0, wildcard_mode=False, strip_whitespace=False,
                        lowercase=False, uppercase=False, remove_extra_spaces=True,
                        wildcard_syntax="curly_braces", debug_mode="0 - Silent"):
        
        debug_level = int(debug_mode.split(" ")[0])
        profiler = PerformanceProfiler(enabled=(debug_level >= 1))
        profiler.start("total_processing")

        original_text = text
        processed_text = text
        seed_used = seed
        selected_seed = 0 
        wildcard_count = 0 

        try:
            # Handle seed_mode
            if seed_mode == "random":
                seed_used = self._generate_random_seed()
                if debug_level >= 1: print(f"[AdvancedTextNode] 🎲 Random mode: Generated new seed: {seed_used}")
            elif seed_mode == "increment":
                seed_used = self._validate_seed((seed + 1) % (SEED_MAX + 1))
                if debug_level >= 1: print(f"[AdvancedTextNode] ➕ Increment mode: Using seed: {seed_used}")
            else:
                seed_used = self._validate_seed(seed)

            # Process seed_list if provided
            if seed_list and seed_list.strip():
                profiler.start("parse_seed_list")
                selection_seed = self._validate_seed((seed_used + seed_offset) % (SEED_MAX + 1))
                
                seed_list_wc_count = 0
                if wildcard_syntax == "curly_braces":
                    selected_seed_str, seed_list_wc_count = self.process_wildcards_curly(seed_list, selection_seed)
                elif wildcard_syntax == "double_underscore":
                    selected_seed_str, seed_list_wc_count = self.process_wildcards_underscore(seed_list, selection_seed)
                else:
                    selected_seed_str = seed_list
                
                wildcard_count += seed_list_wc_count
                
                try:
                    selected_seed_raw = int(selected_seed_str.strip())
                    selected_seed = self._validate_seed(selected_seed_raw)
                    if selected_seed != selected_seed_raw and debug_level >= 1:
                        logging.warning(f"[AdvancedTextNode] ⚠️ Seed from list ({selected_seed_raw}) clamped to JS-safe: {selected_seed}")
                except ValueError:
                    logging.warning(f"[AdvancedTextNode] Could not convert '{selected_seed_str}' to integer. Using {SEED_MIN}.")
                    selected_seed = SEED_MIN
                profiler.stop("parse_seed_list")
            else:
                selected_seed = seed_used

            if wildcard_mode:
                profiler.start("parse_wildcards")
                text_wc_count = 0
                if wildcard_syntax == "curly_braces":
                    processed_text, text_wc_count = self.process_wildcards_curly(processed_text, seed_used)
                elif wildcard_syntax == "double_underscore":
                    processed_text, text_wc_count = self.process_wildcards_underscore(processed_text, seed_used)
                wildcard_count += text_wc_count
                profiler.stop("parse_wildcards")

            profiler.start("string_formatting")
            if strip_whitespace:
                processed_text = processed_text.strip()

            if remove_extra_spaces:
                processed_text = re.sub(r' +', ' ', processed_text) 
                processed_text = '\n'.join(line.strip() for line in processed_text.split('\n'))

            if lowercase:
                processed_text = processed_text.lower()
            elif uppercase: 
                processed_text = processed_text.upper()
            profiler.stop("string_formatting")

            text_length = len(processed_text)

            profiler.stop("total_processing")
            if debug_level >= 1:
                logging.info("\n" + "=" * 60)
                logging.info("📊 [AdvancedTextNode] ANALYTICS REPORT")
                logging.info("=" * 60)
                logging.info("📝  TEXT STATS:")
                logging.info(f"    • Text Length:  {text_length} chars")
                logging.info(f"    • Wildcards:    {wildcard_count} processed")
                profiler.print_report()
                logging.info("=" * 60)

            return (processed_text, original_text, seed_used, selected_seed, text_length, wildcard_count)

        except Exception as e:
            logging.error(f"[AdvancedTextNode] Error processing text: {e}")
            logging.debug(traceback.format_exc())
            return (original_text, original_text, seed_used, 0, len(original_text), 0)

# =================================================================================
# == Core Node Class: TextFileLoader                                            ==
# =================================================================================

class TextFileLoader:
    """Companion node to load text content from external files."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": (
                        "FILE PATH\n"
                        "• Purpose: Full path to the text file (.txt, .yaml, .json, etc.).\n"
                        "• Note: Supports relative paths from ComfyUI root or absolute paths.\n"
                        "• Example: `C:/data/config.yaml`"
                    )
                }),
            },
            "optional": {
                "encoding": (["utf-8", "ascii", "latin-1"], {
                    "default": "utf-8",
                    "tooltip": (
                        "FILE ENCODING\n"
                        "• Purpose: Character encoding of the file.\n"
                        "• Tip: If you see garbled text, try a different encoding.\n"
                        "\n⭐ Recommended: utf-8"
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
        try:
            norm_path = os.path.normpath(file_path)
            if not os.path.exists(norm_path):
                 m_time = -1 
            else:
                 m_time = os.path.getmtime(norm_path)
        except Exception as e:
            logging.warning(f"[TextFileLoader] IS_CHANGED check failed for path '{file_path}': {e}")
            m_time = -2 
        return (norm_path, encoding, m_time)

    def load_file(self, file_path, encoding="utf-8"):
        logging.info(f"[TextFileLoader] 📂 Attempting to load file: {file_path}")
        try:
            if not file_path or not isinstance(file_path, str):
                 error_msg = "[TextFileLoader] Error: Invalid file path provided."
                 logging.error(error_msg)
                 return (error_msg,) 

            if not os.path.exists(file_path):
                 error_msg = f"[TextFileLoader] Error: File not found at path: {file_path}"
                 logging.error(error_msg)
                 return (error_msg,) 

            if not os.path.isfile(file_path):
                 error_msg = f"[TextFileLoader] Error: Path exists but is not a file: {file_path}"
                 logging.error(error_msg)
                 return (error_msg,) 

            with open(file_path, 'r', encoding=encoding) as f:
                text = f.read()
            logging.info(f"[TextFileLoader] ✅ Successfully loaded file: {file_path}")
            return (text,) 

        except Exception as e:
            error_msg = f"[TextFileLoader] Error loading file '{file_path}': {e}"
            logging.error(error_msg)
            logging.debug(traceback.format_exc())
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


# ==============================================================================
# == Unit Tests (smoke — runs without ComfyUI)
# ==============================================================================

if __name__ == "__main__":
    print("\n🧪 Smoke tests: MD_TextInput")
    print("   VERSION :", VERSION)
    _pass = _fail = 0

    def _check(label, expr):
        global _pass, _fail
        if expr:
            print(f"  ✅  {label}")
            _pass += 1
        else:
            print(f"  ❌  {label}")
            _fail += 1

    _check("VERSION defined",    VERSION == "v1.9.0")
    _check("NODE_CLASS_MAPPINGS defined",
           isinstance(NODE_CLASS_MAPPINGS, dict) and len(NODE_CLASS_MAPPINGS) > 0)
    _check("  class AdvancedTextNode in map", "AdvancedTextNode" in NODE_CLASS_MAPPINGS)
    _check("  class TextFileLoader in map", "TextFileLoader" in NODE_CLASS_MAPPINGS)

    print(f"\n  {_pass} passed, {_fail} failed")
    if _fail == 0:
        print("  🎉 All good.")
