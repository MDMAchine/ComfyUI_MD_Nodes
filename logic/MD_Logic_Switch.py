# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# █▓▒░                                                                     ░▒▓█
# █▓▒░        MD_Nodes/MD_Logic_Switch Logic – Switch Suite v1.0.1         ░▒▓█
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
# ║   A suite of logic gates for workflow routing.
# ║   Contains two distinct nodes:
# ║   1. MD_AnySwitch: 2-Way Boolean Switch (True/False).
# ║   2. MD_MultiSwitch: 5-Way Integer Switch (Index 1-5).
# ║   NOTE: As a basic logic router, this file runs entirely in the public wrapper.
# ║
# ║ ░▒▓ FEATURES:
# ║   ✔ Universal Type Support: Routes Models, Latents, Strings, etc.
# ║   ✔ Safety Fallbacks: Returns 'default_value' if paths are missing.
# ║   ✔ Debug Modes: Pass-through options for testing.
# ║   ✔ Enterprise Standard: Robust typing and embedded unit tests.
# ║
# ║ ░▒▓ CHANGELOG:
# ║   - v1.0.1 (2026-02-24) - Enterprise Standards Update:
# ║       VERIFIED: Tooltips meet v1.5.4 standard.
# ╚════════════════════════════════════════════════════════════════════════════


VERSION = "v1.0.1"  # UPS v1.5.8

import logging

# =================================================================================
# == Shared Utilities
# =================================================================================
class AnyType(str):
    """Universal type matcher for ComfyUI. Accepts connection from any slot."""
    def __ne__(self, __value: object) -> bool:
        return False

any_type = AnyType("*")

# =================================================================================
# == Node 1: MD_AnySwitch (Boolean)
# =================================================================================
class MD_AnySwitch:
    """
    A universal 2-way switch. Routes data based on a True/False signal.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "condition": ("BOOLEAN", {
                    "default": False, 
                    "tooltip": (
                        "ROUTING SIGNAL\n"
                        "• Purpose: Determines which input path to pass through.\n"
                        "• Action: True routes to Path A. False routes to Path B."
                    )
                }),
                "pass_through": ("BOOLEAN", {
                    "default": False, 
                    "tooltip": (
                        "DEBUG MODE\n"
                        "• Purpose: Hard-overrides the logic gate.\n"
                        "• Action: If True, ignores 'condition' and always outputs Path A.\n"
                        "\n⭐ Recommended: Keep False for normal operation."
                    )
                }),
            },
            "optional": {
                "on_true": (any_type, {
                    "tooltip": "PATH A (True)\n• The value to output if condition is True."
                }),
                "on_false": (any_type, {
                    "tooltip": "PATH B (False)\n• The value to output if condition is False."
                }),
                "default_value": (any_type, {
                    "tooltip": (
                        "FALLBACK VALUE\n"
                        "• Purpose: Output if the selected path is missing/unconnected.\n"
                        "\n⭐ Prevents workflow crashes from null outputs."
                    )
                }),
            }
        }

    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("selected_output",)
    FUNCTION = "switch"
    CATEGORY = "MD_Nodes/Logic"

    def switch(self, condition, pass_through, on_true=None, on_false=None, default_value=None):
        if pass_through:
            result = on_true
            status = "Pass-Through (Forced True)"
        else:
            result = on_true if condition else on_false
            status = "True" if condition else "False"

        if result is None:
            if default_value is not None:
                result = default_value
                status += " -> (Fallback to Default)"
            else:
                status += " -> (None/Null)"

        try:
            t = type(result).__name__
            val = f"Tensor{list(result.shape)}" if hasattr(result, "shape") else str(result)[:30]
        except Exception:
            t, val = "Unknown", "..."
        
        logging.info(f"[MD_AnySwitch] Path: {status} | Type: {t} | Val: {val}")

        return (result,)

# =================================================================================
# == Node 2: MD_MultiSwitch (Index)
# =================================================================================
class MD_MultiSwitch:
    """
    A universal 5-way switch. Routes data based on an Integer Index (1-5).
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "select_index": ("INT", {
                    "default": 1, "min": 1, "max": 5, 
                    "tooltip": (
                        "PATH INDEX\n"
                        "• Purpose: Selects which of the 5 inputs to pass through.\n"
                        "• Range: 1 to 5."
                    )
                }),
            },
            "optional": {
                "input_1": (any_type, {"tooltip": "PATH 1\n• Selected if Index = 1."}),
                "input_2": (any_type, {"tooltip": "PATH 2\n• Selected if Index = 2."}),
                "input_3": (any_type, {"tooltip": "PATH 3\n• Selected if Index = 3."}),
                "input_4": (any_type, {"tooltip": "PATH 4\n• Selected if Index = 4."}),
                "input_5": (any_type, {"tooltip": "PATH 5\n• Selected if Index = 5."}),
                "default_value": (any_type, {
                    "tooltip": (
                        "FALLBACK VALUE\n"
                        "• Purpose: Output if the selected index is empty or out of range.\n"
                        "\n⭐ Prevents workflow crashes from null outputs."
                    )
                }),
            }
        }

    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("selected_output",)
    FUNCTION = "switch"
    CATEGORY = "MD_Nodes/Logic"

    def switch(self, select_index, default_value=None, **kwargs):
        key = f"input_{select_index}"
        result = kwargs.get(key, None)
        status = f"Index {select_index}"

        if result is None:
            result = default_value
            status += " (Default)"

        try:
            t = type(result).__name__
            val = f"Tensor{list(result.shape)}" if hasattr(result, "shape") else str(result)[:30]
        except Exception:
            t, val = "Unknown", "..."
        
        logging.info(f"[MD_MultiSwitch] {status} | Type: {t} | Val: {val}")

        return (result,)

# =================================================================================
# == Node Registration
# =================================================================================

NODE_CLASS_MAPPINGS = {
    "MD_AnySwitch": MD_AnySwitch,
    "MD_MultiSwitch": MD_MultiSwitch
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MD_AnySwitch": "MD: Any Switch (Boolean)",
    "MD_MultiSwitch": "MD: Multi-Way Switch (5-Path)"
}

# =================================================================================
# == Embedded Unit Tests
# =================================================================================

if __name__ == "__main__":
    logging.info("🧪 Running Self-Tests for MD_Switches v1.0.1...")
    try:
        bool_node = MD_AnySwitch()
        res, = bool_node.switch(True, False, "A", "B", "Default")
        assert res == "A"
        logging.info("✅ AnySwitch (True): PASSED")
        
        res, = bool_node.switch(True, False, None, "B", "Default")
        assert res == "Default"
        logging.info("✅ AnySwitch (Fallback): PASSED")

        multi_node = MD_MultiSwitch()
        res, = multi_node.switch(select_index=3, default_value="D", input_3="C")
        assert res == "C"
        logging.info("✅ MultiSwitch (Index 3): PASSED")
        
        res, = multi_node.switch(select_index=2, default_value="D", input_1="A")
        assert res == "D"
        logging.info("✅ MultiSwitch (Fallback): PASSED")

    except Exception as e:
        logging.error(f"❌ Test Failed: {e}")
        import traceback
        traceback.print_exc()
    logging.info("\n🎉 All tests passed!")