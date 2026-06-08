# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# █▓▒░                                                                     ░▒▓█
# █▓▒░   MD_Nodes/FSampler – Fast Sampler / Epsilon Extrapolator v1.6.1    ░▒▓█
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
# ║   • Math Concept: Finite Difference Extrapolation (Linear/Richardson)
# ╚════════════════════════════════════════════════════════════════════════════


VERSION = "v1.6.1"  # UPS v1.5.8

import os
import sys
import logging
import comfy.model_management

# =================================================================================
# == MD_Nodes Universal Binary Loader (v1.6.1)
# =================================================================================

def find_core_paths():
    """Locate core directory in dev and production environments."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(current_dir, "core"),
        os.path.join(current_dir, "..", "core"),
        os.path.join(current_dir, "..", "..", "core")
    ]
    return [os.path.abspath(c) for c in candidates if os.path.exists(c)]

CORE_LOCATIONS = find_core_paths()
CORE_LOADED = False
CORE_MODE = None
CORE_ERROR = None

for loc in CORE_LOCATIONS:
    if loc not in sys.path:
        sys.path.insert(0, loc)

try:
    import fsampler_core_bin as core
    CORE_LOADED = True
    CORE_MODE = "Binary (Production)"
except ImportError as e1:
    try:
        import fsampler_core as core
        CORE_LOADED = True
        CORE_MODE = "Source (Development)"
    except ImportError as e2:
        CORE_ERROR = f"Binary: {e1}\nSource: {e2}"

# =================================================================================
# == Node Wrapper Class
# =================================================================================

class FSampler:
    """
    The ComfyUI Node definition for FSampler.
    Attaches the core extrapolation hook to the diffusion model.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {
                    "tooltip": (
                        "MODEL INPUT\n"
                        "• Purpose: The diffusion model to patch.\n"
                        "• Output: Returns the patched model ready for sampling."
                    )
                }),
                "extrapolation_mode": (["linear", "quadratic"], {
                    "default": "linear",
                    "tooltip": (
                        "EXTRAPOLATION ALGORITHM\n"
                        "• Linear: Faster, uses 2 steps history. Best for standard workflows.\n"
                        "• Quadratic: Smoother, uses 3 steps history. Best for non-linear schedules.\n"
                        "\n⭐ Recommended: linear"
                    )
                }),
                "skip_strategy": (["conservative", "aggressive"], {
                    "default": "conservative",
                    "tooltip": (
                        "SKIP STRATEGY\n"
                        "• Conservative: Run N-1 steps, Skip 1. Safest for Images (15-50 steps).\n"
                        "• Aggressive: Run 1 step, Skip N-1. Best for Audio/Video (1000+ steps).\n"
                        "\n⭐ Recommended: conservative (Images), aggressive (Audio/Video)"
                    )
                }),
                "skip_interval": ("INT", {
                    "default": 2, "min": 2, "max": 100,
                    "tooltip": (
                        "SKIP INTERVAL\n"
                        "• Conservative Mode: e.g., 3 = Run 2, Skip 1.\n"
                        "• Aggressive Mode: e.g., 5 = Run 1, Skip 4.\n"
                        "• Warning: High intervals on low step counts will cause artifacts.\n"
                        "\n⭐ Recommended: 2"
                    )
                }),
                "start_percentage": ("FLOAT", {
                    "default": 0.1, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": (
                        "START PROTECTION %\n"
                        "• Purpose: Wait until this % of generation is done before skipping.\n"
                        "• Why: Crucial for allowing the model to set the initial structure.\n"
                        "\n⭐ Recommended: 0.1 (Wait 10%)"
                    )
                }),
                "end_percentage": ("FLOAT", {
                    "default": 0.9, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": (
                        "END PROTECTION %\n"
                        "• Purpose: Stop skipping after this % of generation.\n"
                        "• Why: Ensures the model renders high-fidelity fine details at the end.\n"
                        "\n⭐ Recommended: 0.9 (Stop at 90%)"
                    )
                }),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("patched_model",)
    FUNCTION = "patch_model"
    CATEGORY = "MD_Nodes/Optimization"

    def patch_model(self, model, extrapolation_mode, skip_strategy, skip_interval, start_percentage, end_percentage):
        
        # Graceful Degradation: Core Check
        if not CORE_LOADED:
            logging.error(f"❌ ERROR: FSampler Core missing.\nMode: {CORE_MODE or 'Not Loaded'}\nError: {CORE_ERROR}")
            return (model,) # Return unpatched model
        
        # 1. Clone the model to avoid affecting other nodes
        m = model.clone()
        
        # 2. History depth config
        history_depth = 3 if extrapolation_mode == "quadratic" else 2
        
        # 3. Create Hook from Core
        hook = core.FSamplerHook(
            mode=extrapolation_mode,
            history_depth=history_depth,
            skip_interval=skip_interval,
            start_percent=start_percentage,
            end_percent=end_percentage,
            skip_strategy=skip_strategy
        )
        
        # 4. Attach Hook
        m.set_model_unet_function_wrapper(hook)
        
        logging.info(f"[FSampler] Attached: Mode={extrapolation_mode}, Strategy={skip_strategy}, Interval={skip_interval}")
        return (m,)

# =================================================================================
# == Node Registration
# =================================================================================

NODE_CLASS_MAPPINGS = {
    "FSampler": FSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FSampler": "MD: FSampler (Speed Patcher)",
}


# ==============================================================================
# == Unit Tests (smoke — runs without ComfyUI)
# ==============================================================================

if __name__ == "__main__":
    print("\n🧪 Smoke tests: MD_FSampler")
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

    _check("VERSION defined",    VERSION == "v1.6.1")
    _check("NODE_CLASS_MAPPINGS defined",
           isinstance(NODE_CLASS_MAPPINGS, dict) and len(NODE_CLASS_MAPPINGS) > 0)
    _check("  class FSampler in map", "FSampler" in NODE_CLASS_MAPPINGS)

    print(f"\n  {_pass} passed, {_fail} failed")
    if _fail == 0:
        print("  🎉 All good.")
