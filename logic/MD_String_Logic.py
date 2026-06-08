# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# █▓▒░                                                                     ░▒▓█
# █▓▒░          MD_Nodes/MD_String_Logic – Workflow Router v1.2.2          ░▒▓█
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
# ║   • Enhanced by: Gemini
# ║
# ║ ░▒▓ DESCRIPTION:
# ║   A logic gate for text strings with Wildcard and Regex support.
# ║   It checks if a string matches specific patterns using Boolean logic
# ║   (AND/OR/NOR) and routing capabilities. Crucial for adaptive workflows.
# ║   NOTE: As a text parsing utility, this runs entirely in the public wrapper.
# ║
# ║ ░▒▓ FEATURES:
# ║   ✔ Wildcard Engine: Supports {A|B} syntax in Inputs and Patterns.
# ║   ✔ Regex Support: Advanced pattern matching (e.g. \d+ for numbers).
# ║   ✔ Multi-Condition: Check multiple comma-separated patterns (AND/OR).
# ║   ✔ Transformations: Outputs Uppercase/Lowercase for filename formatting.
# ║   ✔ Enterprise Standard: Robust error handling and seed control.
# ║
# ║ ░▒▓ CHANGELOG:
# ║   - v1.2.2 (2026-04-16) - Public Release Cleanup:
# ║       • FIX: Completed truncated __main__ self-test block.
# ║       • FIX: Converted production print() calls to logger.
# ║       • FIX: Corrected invalid regex escape sequence in tooltip.
# ║   - v1.2.1 (2026-02-24) - Enterprise Standards Update:
# ║       VERIFIED: Tooltips meet v1.5.4 standard.
# ║   - v1.2.0 (The Logic Update):
# ║       • NEW: 'Regex Match' operation.
# ║       • NEW: 'match_mode' (Any/All/None) for multiple patterns.
# ║       • ADDED: Uppercase/Lowercase string outputs.
# ╚════════════════════════════════════════════════════════════════════════════


VERSION = "v1.2.2"  # UPS v1.5.8

import logging
import random
import re

logger = logging.getLogger(__name__)

# =================================================================================
# == Configuration Constants
# =================================================================================
CONST_JS_MAX_SAFE_INTEGER = 9007199254740991
CONST_SEED_MIN = 0

# =================================================================================
# == Helper Classes
# =================================================================================
class WildcardExpander:
    """
    Expands {option1|option2} patterns with seeded randomness.
    Matches the logic in WildcardPromptBuilder / SceneGenius.
    """
    WILDCARD_PATTERN = re.compile(r'\{([^{}]+)\}')

    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def expand(self, template):
        if not template or not isinstance(template, str):
            return ""
        
        def replace_wildcard(match):
            options = [opt.strip() for opt in match.group(1).split('|')]
            return self.rng.choice(options)

        result = template
        for _ in range(100):
            if not self.WILDCARD_PATTERN.search(result):
                break
            result = self.WILDCARD_PATTERN.sub(replace_wildcard, result)
        
        return result

# =================================================================================
# == Node Class
# =================================================================================
class MD_String_Logic:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_string": ("STRING", {
                    "default": "", "multiline": True, 
                    "tooltip": (
                        "TEXT TO ANALYZE\n"
                        "• Purpose: The primary source text you want to check.\n"
                        "• Support: Resolves {A|B} wildcard syntax."
                    )
                }),
                "match_string": ("STRING", {
                    "default": "", 
                    "tooltip": (
                        "PATTERN(S)\n"
                        "• Purpose: The specific word or regex you are looking for.\n"
                        "• Note: Separate multiple target words with commas."
                    )
                }),
                "operation": (["Contains", "Equals", "Starts With", "Ends With", "Not Contains", "Regex Match"], {
                    "default": "Contains",
                    "tooltip": (
                        "LOGIC OPERATION\n"
                        "• Purpose: How to evaluate the match.\n"
                        r"• Note: Regex Match allows advanced patterns (e.g. ^\d+)."
                    )
                }),
                "match_mode": (["Any (OR)", "All (AND)", "None (NOR)"], {
                    "default": "Any (OR)",
                    "tooltip": (
                        "MULTI-PATTERN LOGIC\n"
                        "• Purpose: Determines routing if multiple comma-separated patterns are provided.\n"
                        "• Any (OR): True if at least ONE matches.\n"
                        "• All (AND): True only if ALL match."
                    )
                }),
                "case_sensitive": ("BOOLEAN", {
                    "default": False, 
                    "tooltip": "CASE SENSITIVE\n• If False, 'Drum' is equal to 'drum'."
                }),
                "seed": ("INT", {
                    "default": 0, "min": CONST_SEED_MIN, "max": CONST_JS_MAX_SAFE_INTEGER,
                    "tooltip": "SEED\n• Determines the random choice if wildcards {A|B} are used in the text."
                }),
            }
        }

    RETURN_TYPES = ("BOOLEAN", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("result_bool", "processed_string", "uppercase", "lowercase")
    FUNCTION = "evaluate"
    CATEGORY = "MD_Nodes/Logic"

    def _check_single(self, s_in, s_match, operation, case_sensitive):
        if not case_sensitive and operation != "Regex Match":
            s_in = s_in.lower()
            s_match = s_match.lower()
        
        if operation == "Contains": return s_match in s_in
        elif operation == "Not Contains": return s_match not in s_in
        elif operation == "Equals": return s_in == s_match
        elif operation == "Starts With": return s_in.startswith(s_match)
        elif operation == "Ends With": return s_in.endswith(s_match)
        elif operation == "Regex Match":
            try:
                flags = re.IGNORECASE if not case_sensitive else 0
                return bool(re.search(s_match, s_in, flags))
            except re.error as e:
                logger.warning(f"[MD_String_Logic] Invalid Regex: {e}")
                return False
        return False

    def evaluate(self, input_string, match_string, operation, match_mode, case_sensitive, seed):
        expander = WildcardExpander(seed)
        final_input = expander.expand(input_string)
        final_match_raw = expander.expand(match_string)
        
        if operation == "Regex Match":
            patterns = [final_match_raw]
        else:
            patterns = [p.strip() for p in final_match_raw.split(',')]

        results = [self._check_single(final_input, p, operation, case_sensitive) for p in patterns]
        
        final_result = False
        if match_mode == "Any (OR)": final_result = any(results)
        elif match_mode == "All (AND)": final_result = all(results)
        elif match_mode == "None (NOR)": final_result = not any(results)

        uppercase = final_input.upper()
        lowercase = final_input.lower()

        status = "MATCH" if final_result else "NO MATCH"
        if final_input or final_match_raw:
            logger.debug(f"[MD_String_Logic] '{final_match_raw}' in '{final_input}'? -> {status}")
        
        return (final_result, final_input, uppercase, lowercase)

# =================================================================================
# == Node Registration
# =================================================================================

NODE_CLASS_MAPPINGS = {
    "MD_String_Logic": MD_String_Logic
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MD_String_Logic": "MD: String Logic (Router)"
}

# =================================================================================
# == Embedded Unit Tests
# =================================================================================

if __name__ == "__main__":
    print("🧪 Running Self-Tests for String Logic v1.2.2...")
    try:
        node = MD_String_Logic()
        
        res, _, _, _ = node.evaluate("Drum and Bass", "drum", "Contains", "Any (OR)", False, 0)
        assert res is True
        print("✅ Contains Logic: PASSED")
        
        res_wc, text, _, _ = node.evaluate("Techno", "{Techno|House}", "Equals", "Any (OR)", False, 123)
        assert "{" not in text
        print("✅ Wildcard Logic: PASSED")

        res_and, _, _, _ = node.evaluate("Drum and Bass", "Drum, Bass", "Contains", "All (AND)", False, 0)
        assert res_and is True
        print("✅ AND Logic: PASSED")

        res_nor, _, _, _ = node.evaluate("Ambient", "Drum, Bass", "Contains", "None (NOR)", False, 0)
        assert res_nor is True
        print("✅ NOR Logic: PASSED")

        res_re, _, up, low = node.evaluate("Track_001", r"\d+", "Regex Match", "Any (OR)", False, 0)
        assert res_re is True
        assert up == "TRACK_001"
        assert low == "track_001"
        print("✅ Regex + Case Transform: PASSED")

        res_nc, _, _, _ = node.evaluate("Ambient", "Drum", "Not Contains", "Any (OR)", False, 0)
        assert res_nc is True
        print("✅ Not Contains: PASSED")

    except AssertionError as e:
        print(f"❌ Test Failed (Assertion): {e}")
    except Exception as e:
        print(f"❌ Test Failed (Exception): {e}")
    else:
        print("\n🎉 All tests passed!")
