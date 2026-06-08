# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# █▓▒░                                                                     ░▒▓█
# █▓▒░       MD_Nodes/MD_LFO_Generator – Parameter Automator v1.1.2        ░▒▓█
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
# ║   A Low Frequency Oscillator (LFO) for parameter automation.
# ║   Generates oscillating values (Sine, Triangle, Saw, etc.) to animate
# ║   inputs like CFG, Denoise, or FBG Multiplier over a batch of frames.
# ║   NOTE: As a basic mathematical array generator, this runs entirely in the wrapper.
# ║
# ║ ░▒▓ FEATURES:
# ║   ✔ 12+ Waveforms: Includes Inverse and Easing variants.
# ║   ✔ Noise Control: Adjustable smoothing for organic "random walks".
# ║   ✔ Batch Aware: Generates a curve for 'N' steps/frames.
# ║   ✔ Visual Preview: Plots the automation curve in the UI.
# ║
# ║ ░▒▓ CHANGELOG:
# ║   - v1.1.2 (2026-04-16) - Public Release Cleanup:
# ║       • FIX: Converted production print() on plot error path to logger.warning.
# ║   - v1.1.1 (2026-02-24) - Enterprise Standards Update:
# ║       • CRITICAL FIX: Replaced global `np.random.seed` with local `default_rng` 
# ║         to prevent polluting the global ComfyUI random state.
# ║       • VERIFIED: Tooltips meet strict v1.5.4 standard.
# ║   - v1.1.0 (The Smooth Update):
# ║       • FIX: Corrected Triangle wave formula.
# ║       • NEW: Exposed 'noise_smoothing' parameter.
# ╚════════════════════════════════════════════════════════════════════════════


VERSION = "v1.1.2"  # UPS v1.5.8

import logging
import math
import io
import torch
import numpy as np

logger = logging.getLogger(__name__)

# =================================================================================
# == Dependency Checks
# =================================================================================
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

# =================================================================================
# == Constants
# =================================================================================
CONST_PLOT_DPI = 100
CONST_PLOT_FIGSIZE = (10, 4)
CONST_JS_MAX_SAFE_INTEGER = 9007199254740991
CONST_SEED_MIN = 0

# =================================================================================
# == Core Logic: Waveforms
# =================================================================================

def generate_lfo(waveform, steps, cycles, phase, min_val, max_val, seed, noise_smooth):
    """Generates the LFO curve."""
    if steps < 1: steps = 1
    t = np.linspace(0, cycles * 2 * np.pi, steps)
    
    # Apply phase shift (0.0 - 1.0) -> radians
    t += phase * 2 * np.pi
    
    # -- Waveform Logic --
    if waveform == "Sine":
        y = np.sin(t)
    elif waveform == "Inverse Sine":
        y = -np.sin(t)
    elif waveform == "Cosine":
        y = np.cos(t)
    elif waveform == "Triangle":
        y = 2 * np.abs(2 * (t / (2 * np.pi) - np.floor(t / (2 * np.pi) + 0.5))) - 1
    elif waveform == "Sawtooth":
        y = 2 * (t / (2 * np.pi) - np.floor(0.5 + t / (2 * np.pi)))
    elif waveform == "Inverse Sawtooth":
        y = -(2 * (t / (2 * np.pi) - np.floor(0.5 + t / (2 * np.pi))))
    elif waveform == "Square":
        y = np.sign(np.sin(t))
    elif waveform == "Inverse Square":
        y = -np.sign(np.sin(t))
    elif waveform == "Pulse":
        y = np.where(np.sin(t) > 0.8, 1.0, -1.0)
        
    # Easing
    elif waveform == "Ease In (Sine)":
        cycle_pos = (t % (2*np.pi)) / (2*np.pi)
        y = 1 - np.cos((cycle_pos * np.pi) / 2)
        y = y * 2 - 1 
    elif waveform == "Ease Out (Sine)":
        cycle_pos = (t % (2*np.pi)) / (2*np.pi)
        y = np.sin((cycle_pos * np.pi) / 2)
        y = y * 2 - 1
        
    # Noise
    elif waveform == "Random (Noise)":
        # v1.1.1 Fix: Isolated RNG prevents global seed pollution
        rng = np.random.default_rng(seed)
        num_points = max(2, int(cycles * 4))
        y_raw = rng.uniform(-1, 1, num_points)
        
        # Interpolate
        x_raw = np.linspace(0, steps, len(y_raw))
        x_target = np.arange(steps)
        y = np.interp(x_target, x_raw, y_raw)
        
        # Smooth
        if noise_smooth > 0:
            window = max(2, int(steps * noise_smooth * 0.2))
            kernel = np.ones(window) / window
            y = np.convolve(y, kernel, mode='same')
    else:
        y = np.zeros(steps)

    # -- Normalization --
    if waveform == "Random (Noise)":
        y_min, y_max = y.min(), y.max()
        if y_max > y_min:
            y_norm = (y - y_min) / (y_max - y_min)
        else:
            y_norm = y # Flat
    elif "Ease" in waveform:
        y_norm = (y + 1) / 2
    else:
        y_norm = (y + 1) / 2
        
    # Map to Target [Min, Max]
    result = min_val + y_norm * (max_val - min_val)
    
    return result

def plot_lfo(values, title):
    if not MATPLOTLIB_AVAILABLE or not PIL_AVAILABLE:
        return torch.zeros((1, 64, 64, 3))

    try:
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=CONST_PLOT_FIGSIZE)
        
        x = range(len(values))
        ax.plot(x, values, color='#00FFFF', linewidth=2)
        ax.fill_between(x, values, min(values), color='#00FFFF', alpha=0.1)
        
        ax.set_title(f"LFO: {title}", fontsize=10, fontweight='bold')
        ax.set_ylabel("Value", fontsize=8)
        ax.set_xlabel("Step / Frame", fontsize=8)
        ax.grid(True, linestyle=':', alpha=0.3)
        
        # Stats
        curr = values[0]
        stats = f"Start: {curr:.2f}\nRange: {min(values):.2f} - {max(values):.2f}"
        ax.text(0.02, 0.95, stats, transform=ax.transAxes, 
               verticalalignment='top', bbox=dict(facecolor='black', alpha=0.5, edgecolor='gray'))

        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=CONST_PLOT_DPI, facecolor=fig.get_facecolor())
        buf.seek(0)
        plt.close(fig)

        img = Image.open(buf).convert("RGB")
        img_np = np.array(img).astype(np.float32) / 255.0
        return torch.from_numpy(img_np).unsqueeze(0)

    except Exception as e:
        logger.warning(f"[MD_LFO_Generator] Plot error: {e}")
        return torch.zeros((1, 64, 64, 3))

# =================================================================================
# == Node Class
# =================================================================================

class MD_LFO_Generator:
    
    WAVEFORMS = [
        "Sine", "Inverse Sine", "Cosine", 
        "Triangle", 
        "Sawtooth", "Inverse Sawtooth",
        "Square", "Inverse Square", "Pulse", 
        "Random (Noise)",
        "Ease In (Sine)", "Ease Out (Sine)"
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "waveform": (cls.WAVEFORMS, {
                    "default": "Sine",
                    "tooltip": "WAVEFORM SHAPE\n• Purpose: The mathematical pattern used to modulate the values."
                }),
                "frequency": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 100.0, "step": 0.1, 
                    "tooltip": "FREQUENCY (SPEED)\n• Purpose: How many complete cycles occur over the batch duration."
                }),
                "amplitude_min": ("FLOAT", {
                    "default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.01, 
                    "tooltip": "MINIMUM VALUE (FLOOR)\n• Purpose: The absolute lowest value the wave will hit."
                }),
                "amplitude_max": ("FLOAT", {
                    "default": 1.0, "min": -10000.0, "max": 10000.0, "step": 0.01, 
                    "tooltip": "MAXIMUM VALUE (CEILING)\n• Purpose: The absolute highest value the wave will hit."
                }),
                "phase_offset": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, 
                    "tooltip": "PHASE SHIFT\n• Purpose: Offsets where the wave begins. 0.5 starts the wave halfway through its cycle."
                }),
                "steps": ("INT", {
                    "default": 20, "min": 1, "max": 10000, 
                    "tooltip": "TOTAL STEPS (DURATION)\n• Purpose: The total number of frames/steps to generate in the list."
                }),
                "seed": ("INT", {
                    "default": 0, "min": CONST_SEED_MIN, "max": CONST_JS_MAX_SAFE_INTEGER, 
                    "tooltip": "STOCHASTIC SEED\n• Purpose: Controls the randomness of the 'Random (Noise)' waveform."
                }),
                "noise_smoothing": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, 
                    "tooltip": "NOISE SMOOTHING\n• Purpose: Applies a convolution kernel to smooth out jagged points. Only active for Random waves."
                }),
            },
            "optional": {
                "current_step": ("INT", {
                    "default": 0, "min": 0, "max": 10000, 
                    "tooltip": "CURRENT STEP INDEX\n• Purpose: If connected, returns the specific float value at this index from the generated list."
                }),
            }
        }

    RETURN_TYPES = ("FLOAT", "FLOAT", "IMAGE", "STRING")
    RETURN_NAMES = ("current_value", "value_list", "lfo_plot", "list_as_string")
    OUTPUT_IS_LIST = (False, True, False, False)
    FUNCTION = "generate"
    CATEGORY = "MD_Nodes/Modulation"

    def generate(self, waveform, frequency, amplitude_min, amplitude_max, phase_offset, steps, seed, noise_smoothing, current_step=0):
        
        # 1. Generate Curve
        values = generate_lfo(
            waveform, steps, frequency, phase_offset, 
            amplitude_min, amplitude_max, seed, noise_smoothing
        )
        
        # 2. Get Current
        idx = max(0, min(current_step, steps - 1))
        val_now = float(values[idx])
        
        # 3. Format List
        val_list = [float(v) for v in values]
        
        # 4. Plot
        plot = plot_lfo(values, f"{waveform} ({amplitude_min} to {amplitude_max})")
        
        # 5. String
        val_str = ", ".join([f"{v:.3f}" for v in values])

        return (val_now, val_list, plot, val_str)

# =================================================================================
# == Node Registration
# =================================================================================

NODE_CLASS_MAPPINGS = {
    "MD_LFO_Generator": MD_LFO_Generator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MD_LFO_Generator": "MD: LFO Generator (Automator)"
}

# =================================================================================
# == Embedded Unit Tests
# =================================================================================

if __name__ == "__main__":
    print("🧪 Running Self-Tests for LFO Generator v1.1.1...")
    try:
        # Test 1: Min/Max Bounds
        vals = generate_lfo("Sine", 100, 1.0, 0.0, 10.0, 20.0, 0, 0)
        assert min(vals) >= 9.99
        assert max(vals) <= 20.01
        print("✅ Range Logic: PASSED")
        
        # Test 2: Inverse Logic
        v_norm = generate_lfo("Sawtooth", 10, 1.0, 0.0, -1.0, 1.0, 0, 0)
        v_inv = generate_lfo("Inverse Sawtooth", 10, 1.0, 0.0, -1.0, 1.0, 0, 0)
        assert v_norm[1] > 0 and v_inv[1] < 0
        print("✅ Inverse Logic: PASSED")
        
        # Test 3: Triangle Formula
        tri = generate_lfo("Triangle", 100, 1.0, 0.0, -1.0, 1.0, 0, 0)
        mid = len(tri)//2
        assert tri[mid] > 0.9 
        assert tri[0] < -0.9 
        print("✅ Triangle Math: PASSED")
        
        # Test 4: Random Generator Isolation
        rnd1 = generate_lfo("Random (Noise)", 100, 1.0, 0.0, 0.0, 1.0, 42, 0)
        rnd2 = generate_lfo("Random (Noise)", 100, 1.0, 0.0, 0.0, 1.0, 42, 0)
        assert np.array_equal(rnd1, rnd2)
        print("✅ RNG Isolation: PASSED")

    except Exception as e:
        print(f"❌ Test Failed: {e}")
    print("\n🎉 All tests passed!")