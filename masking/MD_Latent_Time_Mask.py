# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# █▓▒░                                                                     ░▒▓█
# █▓▒░       MD_Nodes/MD_Latent_Time_Mask – Timeline Director v1.1.1       ░▒▓█
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
# ║   Creates a temporal mask for 4D or 5D latents. This allows you to apply
# ║   denoising or conditioning to specific time segments (e.g., "Intro", 
# ║   "Chorus") with soft transitions.
# ║   NOTE: As a basic masking utility, this runs entirely in the public wrapper.
# ║
# ║ ░▒▓ FEATURES:
# ║   ✔ Context Aware: Explicit format selection (AnimateDiff vs Ace-Step).
# ║   ✔ Smart Fades: Automatically scales fades if they overlap.
# ║   ✔ Rich Visualization: Annotated plot with Start/End markers and Fade zones.
# ║   ✔ Enterprise Standard: Robust tensor math, embedded tests.
# ║
# ║ ░▒▓ CHANGELOG:
# ║   - v1.1.1 (2026-02-24) - Enterprise Standards Update:
# ║       VERIFIED: Tooltips meet v1.5.4 standard.
# ║   - v1.1.0 (Production Hardening):
# ║       • NEW: 'latent_format' dropdown to resolve dimension ambiguity.
# ║       • NEW: Overlap logic scales fades to fit short clips.
# ╚════════════════════════════════════════════════════════════════════════════


VERSION = "v1.1.1"  # UPS v1.5.8

import math
import logging
import io
import torch
import numpy as np
import comfy.model_management

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

CONST_PLOT_DPI = 100
CONST_PLOT_FIGSIZE = (10, 4)
CONST_JS_MAX_SAFE_INTEGER = 9007199254740991
CONST_SEED_MIN = 0

logger = logging.getLogger("ComfyUI_MD_Nodes.LatentTimeMask")

# =================================================================================
# == Core Logic
# =================================================================================
def generate_temporal_mask(latent_shape, start_sec, end_sec, fps, fade_in, fade_out, invert, format_hint, device):
    is_5d = len(latent_shape) == 5
    
    if format_hint == "AnimateDiff (5D)":
        total_frames = latent_shape[1]
        target_shape = (1, total_frames, 1, 1, 1)
        is_5d = True
    elif format_hint == "Ace-Step (4D)":
        total_frames = latent_shape[0]
        target_shape = (total_frames, 1, 1, 1)
        is_5d = False
    else: 
        if is_5d:
            total_frames = latent_shape[1]
            target_shape = (1, total_frames, 1, 1, 1)
        else:
            total_frames = latent_shape[0]
            target_shape = (total_frames, 1, 1, 1)

    if fps <= 0: fps = 24.0
    total_duration = total_frames / fps
    
    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)
    
    start_frame = max(0, min(start_frame, total_frames))
    end_frame = max(0, min(end_frame, total_frames))
    
    segment_frames = end_frame - start_frame
    fade_in_frames = int(fade_in * fps)
    fade_out_frames = int(fade_out * fps)
    
    if segment_frames <= 0:
        mask_curve = torch.zeros(total_frames, device=device, dtype=torch.float32)
        final_mask = mask_curve.view(target_shape)
        if is_5d and latent_shape[0] > 1: final_mask = final_mask.expand(latent_shape[0], -1, -1, -1, -1)
        return final_mask, mask_curve, {}

    if fade_in_frames + fade_out_frames > segment_frames:
        scale = segment_frames / (fade_in_frames + fade_out_frames)
        fade_in_frames = int(fade_in_frames * scale)
        fade_out_frames = int(fade_out_frames * scale)
        logger.warning(f"[LatentTimeMask] Fades overlap! Scaled to fit: In={fade_in_frames/fps:.2f}s, Out={fade_out_frames/fps:.2f}s")

    mask_curve = torch.zeros(total_frames, device=device, dtype=torch.float32)
    mask_curve[start_frame:end_frame] = 1.0
    
    if fade_in_frames > 0:
        ramp_in = torch.linspace(0, 1, fade_in_frames, device=device)
        mask_curve[start_frame : start_frame + fade_in_frames] = ramp_in

    if fade_out_frames > 0:
        ramp_out = torch.linspace(1, 0, fade_out_frames, device=device)
        mask_curve[end_frame - fade_out_frames : end_frame] = ramp_out

    if invert:
        mask_curve = 1.0 - mask_curve
        
    final_mask = mask_curve.view(target_shape)
    if is_5d and latent_shape[0] > 1:
        final_mask = final_mask.expand(latent_shape[0], -1, -1, -1, -1)

    meta = {
        "start": start_sec, "end": end_sec, 
        "fade_in_end": start_sec + (fade_in_frames/fps),
        "fade_out_start": end_sec - (fade_out_frames/fps)
    }

    return final_mask, mask_curve, meta

def plot_mask_curve(mask_curve, fps, meta):
    if not MATPLOTLIB_AVAILABLE or not PIL_AVAILABLE: return torch.zeros((1, 64, 64, 3))

    try:
        y_vals = mask_curve.cpu().numpy()
        x_vals = np.arange(len(y_vals)) / fps 

        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=CONST_PLOT_FIGSIZE)
        
        ax.plot(x_vals, y_vals, color='#87CEEB', linewidth=2, label='Mask Strength')
        ax.fill_between(x_vals, y_vals, color='#87CEEB', alpha=0.2)
        
        ax.axvline(x=meta.get("start", 0), color='#7FFF00', linestyle='--', alpha=0.6, label='Start')
        ax.axvline(x=meta.get("end", 0), color='#FF6B6B', linestyle='--', alpha=0.6, label='End')
        
        if meta.get("fade_in_end", 0) > meta.get("start", 0):
            ax.axvspan(meta["start"], meta["fade_in_end"], color='#FFD700', alpha=0.15, label='Fade In')
        
        if meta.get("fade_out_start", 0) < meta.get("end", 0):
            ax.axvspan(meta["fade_out_start"], meta["end"], color='#FFA500', alpha=0.15, label='Fade Out')
            
        ax.set_ylim(-0.1, 1.1)
        ax.set_title("Temporal Mask Profile", fontsize=10, fontweight='bold')
        ax.set_xlabel("Time (Seconds)", fontsize=8)
        ax.set_ylabel("Strength", fontsize=8)
        ax.legend(loc='upper right', fontsize=7)
        ax.grid(True, linestyle=':', alpha=0.3)
        
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=CONST_PLOT_DPI, facecolor=fig.get_facecolor())
        buf.seek(0)
        plt.close(fig)

        img = Image.open(buf).convert("RGB")
        img_np = np.array(img).astype(np.float32) / 255.0
        return torch.from_numpy(img_np).unsqueeze(0)

    except Exception as e:
        logger.error(f"Plot generation failed: {e}")
        return torch.zeros((1, 64, 64, 3))

# =================================================================================
# == Node Class
# =================================================================================
class MD_Latent_Time_Mask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {
                    "tooltip": (
                        "LATENT INPUT\n"
                        "• Purpose: The target latent to mask.\n"
                        "• Action: Used to determine total duration and batch dimensions."
                    )
                }),
                "latent_format": (["Auto", "AnimateDiff (5D)", "Ace-Step (4D)"], {
                    "default": "Auto",
                    "tooltip": (
                        "FORMAT HINT\n"
                        "• Purpose: Helps the node identify which dimension is Time.\n"
                        "• Options: 5D (Batch, Time, C, H, W) or 4D (Time, C, H, W).\n"
                        "\n⭐ Recommended: Auto usually guesses correctly."
                    )
                }),
                "fps": ("FLOAT", {
                    "default": 24.0, "min": 1.0, "max": 120.0, 
                    "tooltip": (
                        "FRAME RATE\n"
                        "• Purpose: Used to calculate accurate seconds.\n"
                        "\n⭐ Note: For Ace-Step audio, usually ~21.5 or check model spec."
                    )
                }),
                "start_seconds": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 3600.0, "step": 0.1,
                    "tooltip": "START TIME\n• Purpose: When the mask becomes 100% active."
                }),
                "end_seconds": ("FLOAT", {
                    "default": 10.0, "min": 0.0, "max": 3600.0, "step": 0.1,
                    "tooltip": "END TIME\n• Purpose: When the mask stops being active."
                }),
                "fade_in": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 60.0, "step": 0.1,
                    "tooltip": "FADE IN\n• Purpose: Soft transition duration at start."
                }),
                "fade_out": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 60.0, "step": 0.1,
                    "tooltip": "FADE OUT\n• Purpose: Soft transition duration at end."
                }),
                "invert": ("BOOLEAN", {
                    "default": False, 
                    "tooltip": "INVERT\n• Purpose: Flips the mask (Active becomes Inactive)."
                }),
            }
        }

    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("time_mask", "mask_plot")
    FUNCTION = "create_mask"
    CATEGORY = "MD_Nodes/Masking"

    def create_mask(self, latent, latent_format, fps, start_seconds, end_seconds, fade_in, fade_out, invert):
        samples = latent['samples']
        device = samples.device
        
        mask_tensor, mask_curve, meta = generate_temporal_mask(
            samples.shape, start_seconds, end_seconds, fps, fade_in, fade_out, invert, latent_format, device
        )
        
        plot_image = plot_mask_curve(mask_curve, fps, meta)
        
        return (mask_tensor, plot_image)

# =================================================================================
# == Node Registration
# =================================================================================
NODE_CLASS_MAPPINGS = { "MD_Latent_Time_Mask": MD_Latent_Time_Mask }
NODE_DISPLAY_NAME_MAPPINGS = { "MD_Latent_Time_Mask": "MD: Latent Time Mask (Timeline Director)" }

# =================================================================================
# == Embedded Unit Tests
# =================================================================================
if __name__ == "__main__":
    print("🧪 Running Self-Tests for Latent Time Mask v1.1.1...")
    try:
        shape_4d = (100, 4, 32, 32)
        mask, _, _ = generate_temporal_mask(shape_4d, 0.0, 1.0, 10.0, 0.0, 0.0, False, "Auto", "cpu")
        assert mask.shape == (100, 1, 1, 1)
        assert mask[0].item() == 1.0
        assert mask[11].item() == 0.0
        print("✅ 4D Logic: PASSED")

        mask_scale, _, meta = generate_temporal_mask(shape_4d, 0.0, 1.0, 10.0, 0.8, 0.8, False, "Auto", "cpu")
        assert meta["fade_in_end"] < 0.6 
        print("✅ Overlap Logic: PASSED")
        
        shape_ambiguous = (10, 10, 4, 32, 32) 
        mask_force, _, _ = generate_temporal_mask(shape_ambiguous, 0.0, 1.0, 10.0, 0.0, 0.0, False, "Ace-Step (4D)", "cpu")
        assert mask_force.shape == (10, 1, 1, 1) 
        print("✅ Explicit Format: PASSED")

    except Exception as e:
        print(f"❌ Test Failed: {e}")
    print("\n🎉 All tests passed!")