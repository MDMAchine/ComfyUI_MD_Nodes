# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# ▐▐▐▐ MD_Nodes/AceStep_Inpaint – Audio Generative Fill v1.0.0 ▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐▐
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

class MD_ACE_LatentInpaintMask:
    """
    Applies a time-based Generative Fill mask to ACE-Step Audio Latents.
    ACE1.5 encodes 48kHz audio at a 1920 hop size = exactly 25 latent frames per second.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {"tooltip": "The encoded VAE latent of your source audio."}),
                "start_seconds": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 600.0, "step": 0.1, "tooltip": "When the generative fill should BEGIN."}),
                "end_seconds": ("FLOAT", {"default": 20.0, "min": 0.0, "max": 600.0, "step": 0.1, "tooltip": "When the generative fill should END."}),
                "clear_source_audio": ("BOOLEAN", {"default": True, "tooltip": "Mutes the original audio inside the mask so it doesn't bleed into the new generation."}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("inpainting_latent",)
    FUNCTION = "apply_mask"
    CATEGORY = "MD_Nodes/AceStep"

    def apply_mask(self, latent, start_seconds, end_seconds, clear_source_audio):
        # 1. Deep copy so we don't permanently mutate the upstream cache
        samples = latent["samples"].clone()
        batch_size, channels, length = samples.shape

        # 2. ACE 1.5 Latent Math
        fps = 25.0
        start_frame = int(start_seconds * fps)
        end_frame = int(end_seconds * fps)

        # 3. Clamp to safe bounds
        start_frame = max(0, min(start_frame, length))
        end_frame = max(0, min(end_frame, length))

        # 4. Generate the PyTorch Noise Mask
        # 1.0 = AI generates completely new audio
        # 0.0 = AI preserves original audio
        mask = torch.zeros((batch_size, 1, length), device=samples.device, dtype=samples.dtype)

        if start_frame < end_frame:
            mask[:, :, start_frame:end_frame] = 1.0

            # 5. Silence the masked region in the base latent
            if clear_source_audio:
                samples[:, :, start_frame:end_frame] = 0.0

        out_latent = latent.copy()
        out_latent["samples"] = samples
        out_latent["noise_mask"] = mask

        print(f"🖌️ [ACE Inpaint] Mask created from {start_seconds}s to {end_seconds}s (Frames {start_frame}-{end_frame})")

        return (out_latent,)

NODE_CLASS_MAPPINGS = {"MD_ACE_LatentInpaintMask": MD_ACE_LatentInpaintMask}
NODE_DISPLAY_NAME_MAPPINGS = {"MD_ACE_LatentInpaintMask": "MD: AceStep Audio Generative Fill 🖌️"}


# ==============================================================================
# == Unit Tests (smoke — runs without ComfyUI)
# ==============================================================================

if __name__ == "__main__":
    print("\n🧪 Smoke tests: MD_AceStepInpaint")
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
    _check("  class MD_ACE_LatentInpaintMask in map", "MD_ACE_LatentInpaintMask" in NODE_CLASS_MAPPINGS)

    print(f"\n  {_pass} passed, {_fail} failed")
    if _fail == 0:
        print("  🎉 All good.")
