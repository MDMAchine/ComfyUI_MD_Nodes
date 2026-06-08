# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# █▓▒░                                                                     ░▒▓█
# █▓▒░         MD_Nodes/BasicMath – Basic Arithmetic Suite v1.1.0          ░▒▓█
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
# ╚════════════════════════════════════════════════════════════════════════════

# =================================================================================
# == Standard Library Imports
# =================================================================================
VERSION = "v1.1.0"  # UPS v1.5.8


import math

# =================================================================================
# == Configuration Constants
# =================================================================================
import logging
CONST_FLOAT_MIN = -1.797e+308
CONST_FLOAT_MAX = 1.797e+308
CONST_INT_MIN = -9223372036854775808
CONST_INT_MAX = 9223372036854775807

# =================================================================================
# == Node Classes
# =================================================================================

class MD_Math_Add:
    """
    Performs basic addition (A + B).
    Returns both INT and FLOAT for workflow flexibility.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "a": ("FLOAT", {
                    "default": 0.0,
                    "min": CONST_FLOAT_MIN,
                    "max": CONST_FLOAT_MAX,
                    "step": 0.01,
                    "tooltip": (
                        "VALUE A (PRIMARY)\n"
                        "• Purpose: The first number to add.\n"
                        "• Range: Full float range.\n"
                        "\n⭐ Connect any INT or FLOAT output here."
                    )
                }),
                "b": ("FLOAT", {
                    "default": 0.0,
                    "min": CONST_FLOAT_MIN,
                    "max": CONST_FLOAT_MAX,
                    "step": 0.01,
                    "tooltip": (
                        "VALUE B (ADDEND)\n"
                        "• Purpose: The value added to A.\n"
                        "• Range: Full float range.\n"
                        "\n⭐ Standard addition operation."
                    )
                }),
            },
        }

    RETURN_TYPES = ("INT", "FLOAT")
    RETURN_NAMES = ("int_result", "float_result")
    FUNCTION = "op_add"
    CATEGORY = "MD_Nodes/Math"

    def op_add(self, a, b):
        result = a + b
        return (int(result), float(result))


class MD_Math_Subtract:
    """
    Performs basic subtraction (A - B).
    Returns both INT and FLOAT for workflow flexibility.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "a": ("FLOAT", {
                    "default": 0.0,
                    "min": CONST_FLOAT_MIN,
                    "max": CONST_FLOAT_MAX,
                    "step": 0.01,
                    "tooltip": (
                        "VALUE A (PRIMARY)\n"
                        "• Purpose: The starting number.\n"
                        "• Range: Full float range.\n"
                        "\n⭐ The number being subtracted FROM."
                    )
                }),
                "b": ("FLOAT", {
                    "default": 0.0,
                    "min": CONST_FLOAT_MIN,
                    "max": CONST_FLOAT_MAX,
                    "step": 0.01,
                    "tooltip": (
                        "VALUE B (SUBTRAHEND)\n"
                        "• Purpose: The amount to remove from A.\n"
                        "• Range: Full float range.\n"
                        "\n⭐ Logic: Result = A - B"
                    )
                }),
            },
        }

    RETURN_TYPES = ("INT", "FLOAT")
    RETURN_NAMES = ("int_result", "float_result")
    FUNCTION = "op_sub"
    CATEGORY = "MD_Nodes/Math"

    def op_sub(self, a, b):
        result = a - b
        return (int(result), float(result))


class MD_Math_Multiply:
    """
    Performs basic multiplication (A * B).
    Returns both INT and FLOAT for workflow flexibility.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "a": ("FLOAT", {
                    "default": 1.0,
                    "min": CONST_FLOAT_MIN,
                    "max": CONST_FLOAT_MAX,
                    "step": 0.01,
                    "tooltip": (
                        "VALUE A (PRIMARY)\n"
                        "• Purpose: The first factor.\n"
                        "• Range: Full float range.\n"
                        "\n⭐ Base value for scaling."
                    )
                }),
                "b": ("FLOAT", {
                    "default": 1.0,
                    "min": CONST_FLOAT_MIN,
                    "max": CONST_FLOAT_MAX,
                    "step": 0.01,
                    "tooltip": (
                        "VALUE B (MULTIPLIER)\n"
                        "• Purpose: The scaling factor.\n"
                        "• Range: Full float range.\n"
                        "\n⭐ Recommended: 2.0 (Double), 0.5 (Half)."
                    )
                }),
            },
        }

    RETURN_TYPES = ("INT", "FLOAT")
    RETURN_NAMES = ("int_result", "float_result")
    FUNCTION = "op_mul"
    CATEGORY = "MD_Nodes/Math"

    def op_mul(self, a, b):
        result = a * b
        return (int(result), float(result))


class MD_Math_Divide:
    """
    Performs basic division (A / B).
    Includes safety check for division by zero.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "a": ("FLOAT", {
                    "default": 1.0,
                    "min": CONST_FLOAT_MIN,
                    "max": CONST_FLOAT_MAX,
                    "step": 0.01,
                    "tooltip": (
                        "VALUE A (NUMERATOR)\n"
                        "• Purpose: The value to be divided.\n"
                        "• Range: Full float range.\n"
                        "\n⭐ The top number in the fraction."
                    )
                }),
                "b": ("FLOAT", {
                    "default": 1.0,
                    "min": CONST_FLOAT_MIN,
                    "max": CONST_FLOAT_MAX,
                    "step": 0.01,
                    "tooltip": (
                        "VALUE B (DENOMINATOR)\n"
                        "• Purpose: What to divide by.\n"
                        "• Trade-offs: If 0 is provided, returns 0.0 to prevent crash.\n"
                        "\n⭐ Logic: Result = A / B"
                    )
                }),
            },
        }

    RETURN_TYPES = ("INT", "FLOAT")
    RETURN_NAMES = ("int_result", "float_result")
    FUNCTION = "op_div"
    CATEGORY = "MD_Nodes/Math"

    def op_div(self, a, b):
        if b == 0:
            logging.warning("⚠️ [MD_Nodes] Warning: Division by zero detected. Returning 0.")
            return (0, 0.0)
        
        result = a / b
        return (int(result), float(result))


# =================================================================================
# == Node Registration
# =================================================================================

NODE_CLASS_MAPPINGS = {
    "MD_Math_Add": MD_Math_Add,
    "MD_Math_Subtract": MD_Math_Subtract,
    "MD_Math_Multiply": MD_Math_Multiply,
    "MD_Math_Divide": MD_Math_Divide,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MD_Math_Add": "MD: Math Add (Int/Float)",
    "MD_Math_Subtract": "MD: Math Subtract (Int/Float)",
    "MD_Math_Multiply": "MD: Math Multiply (Int/Float)",
    "MD_Math_Divide": "MD: Math Divide (Int/Float)",
}

# =================================================================================
# == Development & Testing
# =================================================================================

if __name__ == "__main__":
    logging.info("🧪 Running Self-Tests for MD_BasicMath...")
    
    test_passed = 0
    test_failed = 0
    
    try:
        node_add = MD_Math_Add()
        res_add = node_add.op_add(5.5, 4.5)
        assert res_add == (10, 10.0), f"Add failed: {res_add}"
        logging.info("✅ Addition Check: PASSED")
        test_passed += 1
    except AssertionError as e:
        logging.error(f"❌ Addition Check: FAILED - {e}")
        test_failed += 1

    try:
        node_sub = MD_Math_Subtract()
        res_sub = node_sub.op_sub(10.0, 4.5)
        assert res_sub == (5, 5.5), f"Sub failed: {res_sub}"
        logging.info("✅ Subtraction Check: PASSED")
        test_passed += 1
    except AssertionError as e:
        logging.error(f"❌ Subtraction Check: FAILED - {e}")
        test_failed += 1

    try:
        node_mul = MD_Math_Multiply()
        res_mul = node_mul.op_mul(2.0, 3.5)
        assert res_mul == (7, 7.0), f"Mul failed: {res_mul}"
        logging.info("✅ Multiplication Check: PASSED")
        test_passed += 1
    except AssertionError as e:
        logging.error(f"❌ Multiplication Check: FAILED - {e}")
        test_failed += 1

    try:
        node_div = MD_Math_Divide()
        res_div = node_div.op_div(10.0, 2.0)
        assert res_div == (5, 5.0), f"Div failed: {res_div}"
        logging.info("✅ Division Check: PASSED")
        test_passed += 1
    except AssertionError as e:
        logging.error(f"❌ Division Check: FAILED - {e}")
        test_failed += 1

    try:
        res_div_zero = node_div.op_div(10.0, 0.0)
        assert res_div_zero == (0, 0.0), f"DivZero failed: {res_div_zero}"
        logging.info("✅ Division Zero Safety: PASSED")
        test_passed += 1
    except AssertionError as e:
        logging.error(f"❌ Division Zero Safety: FAILED - {e}")
        test_failed += 1

    logging.info(f"\n{'='*60}")
    logging.info(f"Test Results: {test_passed} passed, {test_failed} failed")
    logging.info(f"{'='*60}")