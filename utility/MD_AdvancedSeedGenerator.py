# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# █▓▒░                                                                     ░▒▓█
# █▓▒░ MD_Nodes/MD_AdvancedSeedGenerator – Advanced Seed Management v1.... ░▒▓█
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
VERSION = "v3.0.0"  # UPS v1.5.8


import secrets
import time
import logging
import hashlib

# =================================================================================
# == Configuration Constants (No Magic Numbers!)
# =================================================================================

# Seed Management
CONST_JS_MAX_SAFE_INTEGER = 9007199254740991  # 2^53 - 1 (JavaScript safe range)
CONST_SEED_MIN = 0

# =================================================================================
# == PerformanceProfiler Class (Standard Implementation)
# =================================================================================

class PerformanceProfiler:
    """Standard performance profiler for MD_Nodes."""
    
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.timings = {}
        self.start_times = {}
    
    def start(self, operation_name):
        if not self.enabled:
            return
        self.start_times[operation_name] = time.perf_counter()
    
    def stop(self, operation_name):
        if not self.enabled:
            return
        if operation_name in self.start_times:
            elapsed = time.perf_counter() - self.start_times[operation_name]
            if operation_name not in self.timings:
                self.timings[operation_name] = []
            self.timings[operation_name].append(elapsed)
            del self.start_times[operation_name]
    
    def get_total_time(self):
        if not self.enabled or not self.timings:
            return 0.0
        return sum(sum(times) for times in self.timings.values())
    
    def print_report(self):
        if not self.enabled or not self.timings:
            return
        logging.info("\n⏱️  PERFORMANCE:")
        total = self.get_total_time()
        logging.info(f"    • Total Time: {total:.4f}s")
        for op_name, times in sorted(self.timings.items()):
            avg = sum(times) / len(times)
            if len(times) == 1:
                logging.info(f"    • {op_name}: {avg:.4f}s")
            else:
                logging.info(f"    • {op_name}: {avg:.4f}s avg ({len(times)}x)")

# =================================================================================
# == Utility Functions
# =================================================================================

def validate_seed(seed_value):
    """Ensure seed is within JavaScript-safe range."""
    try:
        int_value = int(seed_value)
    except (ValueError, TypeError):
        return CONST_SEED_MIN
    return max(CONST_SEED_MIN, min(int_value, CONST_JS_MAX_SAFE_INTEGER))

# =================================================================================
# == Core Node Class
# =================================================================================

class MD_AdvancedSeedGenerator:
    """
    Advanced Seed Generator with Offset, Increment, and System Time modes.
    Ensures all outputs are within JavaScript's safe integer range.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": ("INT", {
                    "default": 0,
                    "min": CONST_SEED_MIN,
                    "max": CONST_JS_MAX_SAFE_INTEGER,
                    "control_after_generate": False, # Disable native widget injection
                    "tooltip": (
                        "BASE SEED\n"
                        "• Purpose: The starting point for seed generation.\n"
                        "• Range: 0 to 9,007,199,254,740,991 (JS-Safe).\n"
                        "\n⭐ Recommended: Keep at 0 or your favorite seed, use 'Fixed' mode to stick to it."
                    )
                }),
                "action": (["Fixed", "Randomize", "Increment", "Decrement", "System Time"], {
                    "default": "Fixed",
                    "tooltip": (
                        "GENERATION ACTION\n"
                        "• Purpose: Controls how the final seed is calculated.\n"
                        "• Fixed: Output = Seed + Offset\n"
                        "• Randomize: Output = Crypto-Random Number\n"
                        "• Increment: Output = Seed + Offset + 1 (Wraps)\n"
                        "• Decrement: Output = Seed + Offset - 1 (Wraps)\n"
                        "• System Time: Output = Current Nanosecond Timestamp\n"
                        "\n⭐ Recommended: 'Fixed' for reproducibility, 'Randomize' for exploration."
                    )
                }),
                "seed_offset": ("INT", {
                    "default": 0,
                    "min": -CONST_JS_MAX_SAFE_INTEGER,
                    "max": CONST_JS_MAX_SAFE_INTEGER,
                    "tooltip": (
                        "SEED OFFSET\n"
                        "• Purpose: Add/Subtract from the base seed without changing the base input.\n"
                        "• Note: Applied in all modes except 'System Time'.\n"
                        "\n⭐ Recommended: Use for variations (e.g. set Base to 100, Offset to 1, 2, 3...)"
                    )
                }),
                "debug_mode": (["0 - Silent", "1 - Info", "2 - Verbose"], {
                    "default": "0 - Silent",
                    "tooltip": (
                        "LOGGING VERBOSITY\n"
                        "• Purpose: Controls console output detail level.\n"
                        "• Options: 0 (Production), 1 (Analytics Report), 2 (Full trace).\n"
                        "\n⭐ Recommended: 1 - Info when optimizing workflows."
                    )
                })
            }
        }
    
    RETURN_TYPES = ("INT", "STRING")
    RETURN_NAMES = ("seed", "seed_str")
    FUNCTION = "execute"
    CATEGORY = "MD_Nodes/Utility"

    @classmethod
    def IS_CHANGED(cls, seed, action, seed_offset, debug_mode):
        """
        Control execution flow with proper caching behavior.
        - Fixed/Increment/Decrement: Only change if inputs change.
        - Randomize/System Time: ALWAYS change (force re-run).
        """
        action_normalized = action.lower() if isinstance(action, str) else str(action).lower()
        
        if action_normalized in ["randomize", "system time"]:
            # Return a random value to force ComfyUI to treat this as changed every time
            return secrets.token_hex(8)
        
        # For deterministic modes, return a hash of the inputs
        input_string = f"{seed}_{action_normalized}_{seed_offset}"
        input_hash = hashlib.md5(input_string.encode()).hexdigest()
        return input_hash

    def execute(self, seed, action, seed_offset=0, debug_mode="0 - Silent"):
        debug_level = int(debug_mode.split(" ")[0]) if isinstance(debug_mode, str) else 0
        profiler = PerformanceProfiler(enabled=(debug_level >= 1))
        profiler.start("total_execution")
        
        final_seed = 0
        
        try:
            base = validate_seed(seed)
            offset = int(seed_offset)
            
            profiler.start("calculation")
            
            action_normalized = action.lower() if isinstance(action, str) else str(action).lower()
            
            if action_normalized == "fixed":
                raw_val = base + offset
                final_seed = validate_seed(raw_val)
                
            elif action_normalized == "randomize":
                final_seed = secrets.randbelow(CONST_JS_MAX_SAFE_INTEGER + 1)
                
            elif action_normalized == "increment":
                raw_val = base + offset + 1
                if raw_val > CONST_JS_MAX_SAFE_INTEGER:
                    raw_val = CONST_SEED_MIN 
                final_seed = validate_seed(raw_val)
                
            elif action_normalized == "decrement":
                raw_val = base + offset - 1
                if raw_val < CONST_SEED_MIN:
                    raw_val = CONST_JS_MAX_SAFE_INTEGER 
                final_seed = validate_seed(raw_val)
                
            elif action_normalized == "system time":
                now_ns = time.time_ns()
                final_seed = now_ns % CONST_JS_MAX_SAFE_INTEGER
            
            else:
                final_seed = base
                
            profiler.stop("calculation")
            profiler.stop("total_execution")
            
            if debug_level >= 1:
                logging.info("\n" + "=" * 60)
                logging.info("🎲 [MD_AdvancedSeedGenerator] ANALYTICS REPORT")
                logging.info("=" * 60)
                logging.info("🔢  DATA:")
                logging.info(f"    • Mode:         {action}")
                logging.info(f"    • Base Input:   {base}")
                logging.info(f"    • Offset:       {offset}")
                logging.info(f"    • FINAL SEED:   {final_seed}")
                profiler.print_report()
                logging.info("=" * 60)
                
            return (final_seed, str(final_seed))

        except Exception as e:
            logging.error(f"[MD_AdvancedSeedGenerator] Error: {e}")
            return (validate_seed(seed), str(validate_seed(seed)))

# =================================================================================
# == Node Registration
# =================================================================================

NODE_CLASS_MAPPINGS = {
    "MD_AdvancedSeedGenerator": MD_AdvancedSeedGenerator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MD_AdvancedSeedGenerator": "MD: Advanced Seed Generator",
}

# =================================================================================
# == Development & Testing (Enterprise Standard)
# =================================================================================

if __name__ == "__main__":
    logging.info("🧪 Running Self-Tests for MD_AdvancedSeedGenerator...")
    
    test_passed = 0
    test_failed = 0
    
    node = MD_AdvancedSeedGenerator()
    
    try:
        assert CONST_JS_MAX_SAFE_INTEGER == 9007199254740991
        logging.info("✅ Constants Check: PASSED")
        test_passed += 1
    except AssertionError:
        logging.error(f"❌ Constants Check: FAILED (Expected 9007199254740991, got {CONST_JS_MAX_SAFE_INTEGER})")
        test_failed += 1

    try:
        unsafe_seed = 99999999999999999999
        clamped = validate_seed(unsafe_seed)
        assert clamped == CONST_JS_MAX_SAFE_INTEGER
        logging.info("✅ Seed Clamping (Upper Bound): PASSED")
        test_passed += 1
    except AssertionError:
        logging.error(f"❌ Seed Clamping: FAILED (Expected {CONST_JS_MAX_SAFE_INTEGER}, got {clamped})")
        test_failed += 1

    try:
        base = 100
        offset = 1
        result1, _ = node.execute(seed=base, action="fixed", seed_offset=offset)
        result2, _ = node.execute(seed=base, action="Fixed", seed_offset=offset)
        assert result1 == result2 == 101  
        logging.info("✅ Case Insensitive Matching: PASSED")
        test_passed += 1
    except AssertionError:
        logging.error(f"❌ Case Insensitive Matching: FAILED (Results: {result1}, {result2})")
        test_failed += 1

    try:
        hash1 = node.IS_CHANGED(seed=123, action="fixed", seed_offset=0, debug_mode="0 - Silent")
        hash2 = node.IS_CHANGED(seed=123, action="Fixed", seed_offset=0, debug_mode="0 - Silent")
        assert hash1 == hash2  
        
        hash3 = node.IS_CHANGED(seed=124, action="fixed", seed_offset=0, debug_mode="0 - Silent")
        assert hash1 != hash3
        
        logging.info("✅ IS_CHANGED Caching: PASSED")
        test_passed += 1
    except AssertionError:
        logging.error("❌ IS_CHANGED Caching: FAILED")
        test_failed += 1

    try:
        base = 100
        offset = 1
        result, _ = node.execute(seed=base, action="Increment", seed_offset=offset)
        assert result == 102
        logging.info("✅ Increment Logic: PASSED")
        test_passed += 1
    except AssertionError:
        logging.error(f"❌ Increment Logic: FAILED (Expected 102, got {result})")
        test_failed += 1

    try:
        base = 500
        offset = -50
        result, _ = node.execute(seed=base, action="Fixed", seed_offset=offset)
        assert result == 450
        logging.info("✅ Offset Logic: PASSED")
        test_passed += 1
    except AssertionError:
        logging.error(f"❌ Offset Logic: FAILED (Expected 450, got {result})")
        test_failed += 1

    try:
        result1, _ = node.execute(seed=0, action="Randomize")
        result2, _ = node.execute(seed=0, action="Randomize")
        assert result1 != result2
        assert result1 <= CONST_JS_MAX_SAFE_INTEGER
        logging.info("✅ Randomize Logic: PASSED")
        test_passed += 1
    except AssertionError:
        logging.error("❌ Randomize Logic: FAILED (Values matched or out of range)")
        test_failed += 1

    logging.info(f"\n{'='*60}")
    logging.info(f"Test Results: {test_passed} passed, {test_failed} failed")
    logging.info(f"{'='*60}")
    
    if test_failed == 0:
        logging.info("🎉 All tests passed! Node is ready for production.")