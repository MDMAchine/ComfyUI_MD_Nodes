# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# ▐▐▐▐ MD_Nodes/ACE_SigmaDenoisePatcher – Audio-to-Audio Slicer v1.0 ▐▐▐▐▐▐▐▗▒░
# © 2026 MDMAchine (A&E Concepts)
# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# ==============================================================================
# Part of ComfyUI_MD_Nodes by MDMAchine (A&E Concepts)
# Repository: https://github.com/MDMAchine/ComfyUI_MD_Nodes
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
# ==============================================================================


VERSION = "v1.0.0"  # UPS v1.5.8

import torch
import logging

class MD_ACE_SigmaDenoisePatcher:
    """
    Takes any existing SIGMAS schedule and slices it to create an 
    Audio-to-Audio (Img2Img) denoise effect for advanced custom samplers.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"tooltip": "The original full sigma schedule from any scheduler node."}),
                "denoise": ("FLOAT", {
                    "default": 0.70, 
                    "min": 0.01, 
                    "max": 1.0, 
                    "step": 0.01,
                    "tooltip": "Audio-to-Audio Injection Strength.\n• 1.0 = Ignore input, generate from scratch.\n• 0.70 = Keep MIDI structure, paint new sounds."
                }),
            }
        }
    
    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("sliced_sigmas",)
    FUNCTION = "patch_sigmas"
    CATEGORY = "MD_Nodes/ACE_Engine/Schedulers"

    def patch_sigmas(self, sigmas, denoise):
        # A standard sigmas tensor has a length of (steps + 1) because it ends in 0.0
        total_steps = len(sigmas) - 1
        
        if total_steps <= 0:
            return (sigmas,)
            
        # Calculate how many steps to skip from the beginning.
        # If denoise is 0.70, we want to KEEP 70% of the steps, so we SKIP the first 30%.
        skip_steps = int(total_steps * (1.0 - denoise))
        
        if skip_steps > 0:
            sliced_sigmas = sigmas[skip_steps:]
            logging.info(f"✂️ [SigmaPatcher] Sliced off {skip_steps}/{total_steps} steps to achieve {denoise:.2f} Denoise.")
            return (sliced_sigmas,)
        
        return (sigmas,)

# =================================================================================
# == Node Registration
# =================================================================================

NODE_CLASS_MAPPINGS = {
    "MD_ACE_SigmaDenoisePatcher": MD_ACE_SigmaDenoisePatcher,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MD_ACE_SigmaDenoisePatcher": "MD: ACE Sigma Denoise Patcher ✂️",
}


# ==============================================================================
# == Unit Tests (smoke — runs without ComfyUI)
# ==============================================================================

if __name__ == "__main__":
    print("\n🧪 Smoke tests: MD_ACE_SigmaDenoisePatcher")
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

    _check("VERSION defined",    VERSION == "v1.0.0")
    _check("NODE_CLASS_MAPPINGS defined",
           isinstance(NODE_CLASS_MAPPINGS, dict) and len(NODE_CLASS_MAPPINGS) > 0)
    _check("  class MD_ACE_SigmaDenoisePatcher in map", "MD_ACE_SigmaDenoisePatcher" in NODE_CLASS_MAPPINGS)

    print(f"\n  {_pass} passed, {_fail} failed")
    if _fail == 0:
        print("  🎉 All good.")
