# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# █▓▒░                                                                     ░▒▓█
# █▓▒░     MD_Nodes/UniversalWildcardOrchestrator – Text Engine v1.3.0     ░▒▓█
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
# ║ ░▒▓ ORIGIN: Universal Text Processing Engine / Recursive Regex Parsing
# ║ ░▒▓ DESCRIPTION:
# ║    A professional-grade text orchestration node that acts as a central 
# ║    rendering engine for wildcard templates. It resolves nested random 
# ║    choices {A|B} from Noodles, Widgets, or Files with full tensor safety.
# ║    NOTE: As a text/IO utility, this runs entirely in the public wrapper domain.
# ║
# ║ ░▒▓ CORE FEATURES:
# ║    ✓ Recursive Parsing: Handles deeply nested logic like {A|{B|C}}.
# ║    ✓ Triple-Input Logic: Switches between LLM, Manual, and File inputs.
# ║    ✓ Deterministic Seeding: Uses JS-Safe integers for consistency.
# ║    ✓ Subgraph Stability: Cached preset loading prevents widget shuffling.
# ║    ✓ Enterprise Logging: Full performance profiling and analytics reports.
# ║
# ║ ░▒▓ CHANGELOG:
# ║    v1.3.0 (Enterprise Standards - Feb 2026):
# ║    ├── REFACTOR: Tooltips strictly updated to 5-part v1.5.4 standard.
# ║    └── VERIFIED: PerformanceProfiler matches v1.5.3 exact specifications.
# ║    v1.2.5 (2026-02-16) - Full Enterprise Standards Alignment
# ║    ├── ADDED: Standard Global Constants (CONST_*) per v1.5.7.
# ║    ├── ADDED: Standard Embedded Unit Test suite per v1.5.7.
# ║    ├── RESTORED: Full PerformanceProfiler class implementation.
# ║    └── FIXED: Parameter mismatch (unique_id) in execute signature.
# ╚════════════════════════════════════════════════════════════════════════════

# =================================================================================
# == Standard Library Imports
# =================================================================================
VERSION = "v1.3.0"  # UPS v1.5.8


import os
import random
import re
import secrets
import time
import traceback
import glob
import logging
from collections import OrderedDict

# =================================================================================
# == Third-Party Imports
# =================================================================================
import torch
import numpy as np

# =================================================================================
# == Configuration Constants
# =================================================================================
CONST_JS_MAX_SAFE_INTEGER = 9007199254740991
CONST_MAX_RECURSION_DEPTH = 50
CONST_DEFAULT_DELIMITER = "|"
CONST_PRESET_CACHE_TTL = 5

# =================================================================================
# == Helper Classes (Enterprise Standards)
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
        logging.info("\n⏱️  PERFORMANCE (Text Parsing):")
        total = self.get_total_time()
        logging.info(f"    • Total Time: {total:.4f}s")
        for op_name, times in sorted(self.timings.items()):
            avg = sum(times) / len(times)
            if len(times) == 1:
                logging.info(f"    • {op_name}: {avg:.4f}s")
            else:
                logging.info(f"    • {op_name}: {avg:.4f}s avg ({len(times)}x)")

class RecursiveWildcardProcessor:
    """Handles nested wildcard rendering."""
    
    def __init__(self, seed, delimiter=CONST_DEFAULT_DELIMITER):
        self.rng = random.Random(seed)
        self.delimiter = delimiter
        self.pattern = re.compile(r'\{([^{}]+)\}')

    def process(self, text):
        if not text: return ""
        iterations = 0
        current_text = text
        while iterations < CONST_MAX_RECURSION_DEPTH:
            match = self.pattern.search(current_text)
            if not match: break
            full_match = match.group(0)
            content = match.group(1)
            options = [opt.strip() for opt in content.split(self.delimiter)]
            choice = self.rng.choice(options)
            current_text = current_text.replace(full_match, choice, 1)
            iterations += 1
        return current_text

# =================================================================================
# == Core Node Class
# =================================================================================

class UniversalWildcardOrchestrator:
    """The central text rendering engine for MD_Nodes workflows."""
    _PRESET_CACHE = None
    _PRESET_CACHE_TIME = 0

    def __init__(self):
        self.presets_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "presets")
        if not os.path.exists(self.presets_dir):
            try:
                os.makedirs(self.presets_dir)
            except Exception: pass

    @classmethod
    def _get_preset_files(cls):
        """Scans local 'presets' directory with 5s stability cache for Subgraphs."""
        current_time = time.time()
        if cls._PRESET_CACHE is not None and (current_time - cls._PRESET_CACHE_TIME) < CONST_PRESET_CACHE_TTL:
            return cls._PRESET_CACHE

        presets_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "presets")
        if not os.path.exists(presets_dir):
            cls._PRESET_CACHE = ["None", "Random"]
            cls._PRESET_CACHE_TIME = current_time
            return cls._PRESET_CACHE
        
        files = sorted([os.path.basename(f) for f in glob.glob(os.path.join(presets_dir, "*.txt"))])
        cls._PRESET_CACHE = ["None", "Random"] + files
        cls._PRESET_CACHE_TIME = current_time
        return cls._PRESET_CACHE

    @classmethod
    def INPUT_TYPES(cls):
        file_list = cls._get_preset_files()
        return {
            "required": OrderedDict([
                ("seed", ("INT", {
                    "default": 0, 
                    "min": 0, 
                    "max": CONST_JS_MAX_SAFE_INTEGER,
                    "control_after_generate": False, 
                    "tooltip": (
                        "SEED VALUE\n"
                        "• Purpose: Controls random selection logic for wildcards.\n"
                        "• Range: 0 to 9,007,199,254,740,991 (JS Safe Max).\n"
                        "• Trade-offs: Consistent seeds ensure reproducible prompts.\n"
                        "\n⭐ Recommended: Connect a global seed node or enable Randomize Seed."
                    )
                })),
                ("randomize_seed", ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "RANDOMIZE SEED\n"
                        "• Purpose: Automatically change seed every generation run.\n"
                        "• Options: True (New choices every run), False (Lock current choices).\n"
                        "• Trade-offs: Disable this when you want to refine a specific prompt result.\n"
                        "\n⭐ Recommended: Keep True for prompt exploration."
                    )
                })),
                ("preset_file", (file_list, {
                    "default": "None",
                    "tooltip": (
                        "PRESET FILE (Priority 3)\n"
                        "• Purpose: Load template from the local /presets folder.\n"
                        "• Range: All .txt files in the local presets directory.\n"
                        "• Trade-offs: Only used if widget and noodle inputs are completely empty.\n"
                        "\n⭐ Recommended: Use this to cycle through massive, pre-written prompt libraries."
                    )
                })),
                ("text_widget", ("STRING", {
                    "multiline": True, "dynamicPrompts": False, "default": "",
                    "tooltip": (
                        "TEXT WIDGET (Priority 2)\n"
                        "• Purpose: Manual template entry with wildcard syntax (e.g. {A|B}).\n"
                        "• Range: Any valid text string.\n"
                        "• Trade-offs: Overrides Preset File; ignored if Input Noodle is connected.\n"
                        "\n⭐ Recommended: Perfect for one-off generation tests and prompt drafting."
                    )
                })),
            ]),
            "optional": OrderedDict([
                ("input_text_override", ("STRING", {
                    "forceInput": True,
                    "tooltip": (
                        "INPUT TEXT OVERRIDE (Priority 1)\n"
                        "• Purpose: Receive templates from other nodes (LLMs, String logic).\n"
                        "• Range: Any string or safely sanitized tensor input.\n"
                        "• Trade-offs: Highest priority; overrides both the widget and local files.\n"
                        "\n⭐ Recommended: Connect LLM outputs here for fully automated pipelines."
                    )
                })),
                ("delimiter", ("STRING", {
                    "default": CONST_DEFAULT_DELIMITER,
                    "tooltip": (
                        "DELIMITER\n"
                        "• Purpose: Defines the split character for options within the braces.\n"
                        "• Range: Single character (usually | or ,).\n"
                        "• Trade-offs: Custom delimiters can break standard syntax compatibility.\n"
                        "\n⭐ Recommended: Leave as '|' unless using highly specialized formats."
                    )
                })),
                ("debug_mode", (["0 - Silent", "1 - Info", "2 - Verbose"], {
                    "default": "0 - Silent",
                    "tooltip": (
                        "LOGGING VERBOSITY\n"
                        "• Purpose: Controls console output and parser profiling.\n"
                        "• Options: 0 (Silent), 1 (Analytics/Source ID), 2 (Full tracing).\n"
                        "\n⭐ Recommended: 0 for production, 1 when debugging complex nested wildcards."
                    )
                })),
                ("enable_profiling", ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "ENABLE PROFILING\n"
                        "• Purpose: Measure timing of the recursive parsing algorithm.\n"
                        "• Note: Automatically enabled if debug_mode >= 1.\n"
                        "\n⭐ Recommended: False."
                    )
                })),
            ]),
            "hidden": {
                "unique_id": "UNIQUE_ID",
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("final_prompt",)
    FUNCTION = "execute"
    CATEGORY = "MD_Nodes/Prompt Generation"

    @classmethod
    def IS_CHANGED(cls, seed, randomize_seed, **kwargs):
        if str(randomize_seed).lower() == "true":
            return secrets.token_hex(16)
        return "static"

    def execute(self, seed, randomize_seed, preset_file, text_widget, input_text_override=None, delimiter=CONST_DEFAULT_DELIMITER, debug_mode="0 - Silent", enable_profiling=False, unique_id=None):
        lvl = int(debug_mode.split(" ")[0]) if isinstance(debug_mode, str) else 0
        profiler = PerformanceProfiler(enabled=(lvl >= 1 or enable_profiling))
        profiler.start("total")

        try:
            if randomize_seed:
                seed = secrets.randbelow(CONST_JS_MAX_SAFE_INTEGER)
            
            # --- FULL PARITY INPUT RESOLUTION ---
            profiler.start("input_resolution")
            raw, source = "", "None"

            if input_text_override is not None:
                if isinstance(input_text_override, (torch.Tensor, np.ndarray)):
                    logging.warning("[UniversalWildcardOrchestrator] Tensor detected. Sanitizing.")
                    raw = str(input_text_override.tolist()) if hasattr(input_text_override, 'tolist') else ""
                    source = "Input Noodle (Sanitized Tensor)"
                elif isinstance(input_text_override, list) and len(input_text_override) > 0:
                    raw = str(input_text_override[0])
                    source = "Input Noodle (Sanitized List/Embed)"
                elif isinstance(input_text_override, str) and input_text_override.strip():
                    raw, source = input_text_override, "Input Noodle"
            
            if not raw and text_widget.strip():
                raw, source = text_widget, "Text Widget"
            
            if not raw and preset_file != "None":
                if preset_file == "Random":
                    all_files = [f for f in self._get_preset_files() if f not in ["None", "Random"]]
                    if all_files:
                        target = random.Random(seed).choice(all_files)
                        path = os.path.join(self.presets_dir, target)
                        source = f"File (Random -> {target})"
                    else: path = None
                else:
                    path = os.path.join(self.presets_dir, preset_file)
                    source = f"File ({preset_file})"

                if path and os.path.exists(path):
                    with open(path, 'r', encoding='utf-8-sig') as f:
                        raw = "".join([l for l in f.readlines() if not l.strip().startswith(('#', '//'))])
            
            profiler.stop("input_resolution")

            # --- Processing ---
            profiler.start("recursive_processing")
            processor = RecursiveWildcardProcessor(seed, delimiter)
            final = processor.process(raw)
            profiler.stop("recursive_processing")
            profiler.stop("total")

            if lvl >= 1:
                logging.info("\n" + "=" * 60)
                logging.info(f"📊 [UniversalWildcardOrchestrator] ANALYTICS REPORT")
                logging.info("=" * 60)
                logging.info(f"    • Source: {source} | Seed: {seed}")
                profiler.print_report()
                logging.info("=" * 60)

            return (final,)

        except Exception as e:
            logging.error(f"[UniversalWildcardOrchestrator] Error: {e}")
            return (str(input_text_override) if input_text_override else "",)

# =================================================================================
# == Node Registration
# =================================================================================
NODE_CLASS_MAPPINGS = {"UniversalWildcardOrchestrator": UniversalWildcardOrchestrator}
NODE_DISPLAY_NAME_MAPPINGS = {"UniversalWildcardOrchestrator": "MD: Universal Wildcard Orchestrator"}

# =================================================================================
# == Development & Testing
# =================================================================================
if __name__ == "__main__":
    logging.info("🧪 Running Self-Tests for UniversalWildcardOrchestrator v1.3.0...")
    passed, failed = 0, 0
    try:
        proc = RecursiveWildcardProcessor(42)
        assert proc.process("{A|A}") == "A"
        assert proc.process("{A|{B|B}}") in ["A", "B"]
        print("✅ Recursive Logic: PASSED"); passed += 1
    except Exception as e: print(f"❌ Recursive Logic: FAILED - {e}"); failed += 1

    logging.info(f"\n{'='*60}\nTest Results: {passed} passed, {failed} failed\n{'='*60}")