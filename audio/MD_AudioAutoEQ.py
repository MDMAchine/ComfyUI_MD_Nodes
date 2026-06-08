# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# █▓▒░                                                                     ░▒▓█
# █▓▒░                MD: Audio Auto EQ (Adaptive) – v3.3.0                ░▒▓█
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
# ║ ░▒▓ DESCRIPTION:
# ║    Intelligent spectral balancing with adaptive content analysis.
# ║    NOTE: This is a public wrapper. Missing binaries will gracefully pass 
# ║    audio through unchanged to prevent workflow crashes.
# ╚════════════════════════════════════════════════════════════════════════════


VERSION = "v3.3.0"  # UPS v1.5.8

import os, sys, io, time, logging, traceback, torch
import numpy as np
from PIL import Image

# =================================================================================
# == MD_Nodes Universal Binary Loader (v1.6.1)
# =================================================================================
def find_core_paths():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(current_dir, "core"),
        os.path.join(current_dir, "..", "core"),
        os.path.join(current_dir, "..", "..", "core")
    ]
    return list(dict.fromkeys([os.path.abspath(c) for c in candidates if os.path.exists(c)]))

CORE_LOCATIONS = find_core_paths()
CORE_LOADED, CORE_MODE, core, CORE_ERROR = False, "NONE", None, ""
for loc in CORE_LOCATIONS:
    if loc not in sys.path: sys.path.insert(0, loc)

try:
    import audio_autoeq_core_bin as core
    CORE_LOADED, CORE_MODE = True, "BINARY"
except ImportError as e1:
    try:
        import audio_autoeq_core as core
        CORE_LOADED, CORE_MODE = True, "SOURCE"
    except ImportError as e2: 
        CORE_ERROR = f"Binary: {e1} | Source: {e2}"

# =================================================================================
# == Dependency Checks
# =================================================================================
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError: MATPLOTLIB_AVAILABLE = False

# =================================================================================
# == Performance Profiler
# =================================================================================
logger = logging.getLogger("MD_Nodes.Audio.AutoEQ")

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
        print("\n⏱️  PERFORMANCE (DSP):")
        total = self.get_total_time()
        print(f"    • Total Time: {total:.4f}s")
        for op_name, times in sorted(self.timings.items()):
            avg = sum(times) / len(times)
            if len(times) == 1:
                print(f"    • {op_name}: {avg:.4f}s")
            else:
                print(f"    • {op_name}: {avg:.4f}s avg ({len(times)}x)")

# =================================================================================
# == Wrapper Node Class
# =================================================================================
class MD_AudioAutoEQ_Adaptive:
    """Intelligent spectral balancing with adaptive content analysis."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {
                    "tooltip": (
                        "AUDIO INPUT\n"
                        "• Purpose: The raw audio waveform to be analyzed and equalized.\n"
                        "• Supports: Mono and Stereo."
                    )
                }),
                "target_profile": ([
                    "── 📢 VOICE ──", "Podcast/Speech", "Vocal Clarity", "Radio Voice",
                    "Vocal De-esser", "Voice Warmth", "Voice Presence",
                    "── 🎵 MUSIC ──", "Music Master", "EDM/Electronic", "Rock/Metal", 
                    "Hip-Hop/Trap", "Acoustic", "Classical",
                    "── 🔧 CORRECTIVE ──", "De-muddy", "Bass Boost", "Bass Reduce",
                    "Treble Boost", "Treble Reduce", "Harshness Tamer",
                    "── 🎨 CREATIVE ──", "Warm & Smooth", "Bright & Airy", "Modern/Crisp",
                    "── ⚪ UTILITY ──", "Flat/Neutral",
                ], {
                    "default": "Music Master", 
                    "tooltip": (
                        "TARGET PROFILE\n"
                        "• Purpose: Defines the ideal spectral balance for the engine to aim for.\n"
                        "• Music Master: Standard balanced sound.\n"
                        "• Podcast: Focused on vocal clarity.\n"
                        "\n⭐ Recommended: Use 'Music Master' for most generative music."
                    )
                }),
                "strength": ("FLOAT", {
                    "default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05, 
                    "tooltip": (
                        "EQ STRENGTH\n"
                        "• Purpose: How aggressively the math attempts to match the target profile.\n"
                        "• Range: 0.0 (Bypass) to 1.0 (Maximum correction).\n"
                        "\n⭐ Recommended: 0.6 - 0.8 for a natural sound."
                    )
                }),
                "highpass_freq": ("FLOAT", {
                    "default": 20.0, "min": 20.0, "max": 500.0, "step": 10.0, 
                    "tooltip": (
                        "HIGHPASS FILTER (Hz)\n"
                        "• Purpose: Slices off frequencies BELOW this number.\n"
                        "\n⭐ Recommended: 30Hz cleans sub-mud without losing bass power."
                    )
                }),
                "lowpass_freq": ("FLOAT", {
                    "default": 20000.0, "min": 8000.0, "max": 20000.0, "step": 1000.0, 
                    "tooltip": (
                        "LOWPASS FILTER (Hz)\n"
                        "• Purpose: Slices off frequencies ABOVE this number.\n"
                        "\n⭐ Recommended: Keep at 20000Hz for maximum transparency."
                    )
                }),
                "adaptive_mode": (["full", "hybrid", "preset"], {
                    "default": "full", 
                    "tooltip": (
                        "ADAPTIVE MODE\n"
                        "• Full: Uses real-time FFT analysis to dynamically shape the EQ.\n"
                        "• Hybrid: Blends real-time analysis with a fixed EQ curve.\n"
                        "• Preset: Applies a static EQ curve (ignores audio content).\n"
                        "\n⭐ Recommended: 'Full' for true Auto-EQ."
                    )
                }),
                "debug_mode": (["0 - Silent", "1 - Info", "2 - Verbose"], {
                    "default": "0 - Silent", 
                    "tooltip": "LOGGING VERBOSITY\n• Controls console detail levels and performance profiling."
                }),
                "enable_profiling": ("BOOLEAN", {
                    "default": False, 
                    "tooltip": "ENABLE PROFILING\n• Forces PerformanceProfiler timing regardless of debug_mode."
                }),
            }
        }

    # CORRECT
    RETURN_TYPES = ("AUDIO", "STRING", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("audio", "report_string", "image_1", "image_2", "image_3") # Adjust these names to match what your node actually returns
    FUNCTION, CATEGORY = "execute", "MD_Nodes/Audio Processing"

    def execute(self, audio, target_profile, strength, highpass_freq, lowpass_freq, adaptive_mode, **kwargs):
        debug_mode = kwargs.get("debug_mode", "0 - Silent")
        if isinstance(debug_mode, bool): debug_mode = "1 - Info" if debug_mode else "0 - Silent"
        debug_level = int(debug_mode.split(" ")[0])
        profiler = PerformanceProfiler(enabled=kwargs.get("enable_profiling", False) or debug_level >= 1)
        profiler.start("total_execution")

        sr, waveform = audio['sample_rate'], audio['waveform']
        audio_np = waveform.cpu().numpy()[0] if waveform.ndim == 3 else waveform.cpu().numpy()
        if audio_np.ndim == 1: audio_np = audio_np.reshape(1, -1)
        
        # --- Graceful Degradation ---
        if not CORE_LOADED:
            error_str = f"❌ ERROR: AutoEQ Core missing.\nMode: {CORE_MODE}\nError: {CORE_ERROR}"
            logging.error(f"[MD_AutoEQ] {error_str}")
            blank_img = torch.zeros((1, 64, 64, 3))
            profiler.stop("total_execution")
            return (audio, error_str, blank_img, blank_img, blank_img)

        try:
            profiler.start("dsp_processing")
            proc, eq_adj, r_b, r_a = core.execute_autoeq_pipeline(
                audio_np, sr, target_profile, strength, highpass_freq, lowpass_freq, adaptive_mode
            )
            profiler.stop("dsp_processing")

            # --- Visuals ---
            profiler.start("plotting")
            wf_b = self._plot_waveform(audio_np.T, sr, "Original")
            wf_a = self._plot_waveform(proc.T, sr, "Processed")
            eq_v = self._plot_eq_curve(eq_adj)
            profiler.stop("plotting")
            
            # --- Reporting ---
            report = f"═══ Auto-EQ ({CORE_MODE}) ═══\nProfile: {target_profile}\nStrength: {strength:.2f}\n\n──── BEFORE EQ ────\n"
            for b, r in r_b.items(): report += f"  {b:12s}: {r*100:5.2f}%\n"
            report += "\n──── AFTER EQ ────\n"
            for b, r in r_a.items():
                delta = (r - r_b[b]) * 100
                arrow = "↑" if delta > 0 else "↓" if delta < 0 else "→"
                report += f"  {b:12s}: {r*100:5.2f}% ({arrow} {abs(delta):.2f}%)\n"
            
            out = {'waveform': torch.from_numpy(proc).float().unsqueeze(0), 'sample_rate': sr}
            
            profiler.stop("total_execution")
            if debug_level >= 1:
                print(f"\n📊 [AutoEQ] DSP REPORT")
                profiler.print_report()

            return (out, report, wf_b, wf_a, eq_v)

        except Exception as e:
            logging.error(f"[MD_AutoEQ] Process Error: {e}", exc_info=True)
            blank_img = torch.zeros((1, 64, 64, 3))
            return (audio, f"❌ Error: {str(e)}", blank_img, blank_img, blank_img)

    def _plot_waveform(self, data, sr, title):
        if not MATPLOTLIB_AVAILABLE: return torch.zeros((1, 64, 64, 3))
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(10, 3), dpi=96)
        p_data = data[:, 0] if data.ndim == 2 else data
        peak, rms = np.max(np.abs(p_data)), np.sqrt(np.mean(p_data**2))
        ax.plot(np.linspace(0, len(p_data)/sr, len(p_data)), p_data, color='#87CEEB', linewidth=0.5)
        ax.set_title(f"{title} | Peak: {peak:.3f} | RMS: {rms:.3f}", fontsize=10)
        ax.set_ylim(-1.05, 1.05); plt.tight_layout()
        buf = io.BytesIO(); fig.savefig(buf, format='png', facecolor=fig.get_facecolor()); buf.seek(0); plt.close(fig)
        return torch.from_numpy(np.array(Image.open(buf).convert("RGB")).astype(np.float32)/255.0).unsqueeze(0)

    def _plot_eq_curve(self, eq_adj):
        if not MATPLOTLIB_AVAILABLE or not eq_adj: return torch.zeros((1, 64, 64, 3))
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(10, 4), dpi=120)
        bands = list(eq_adj.keys()); gains = [eq_adj[b] for b in bands]
        ax.bar(range(len(bands)), gains, color=['red' if g < 0 else 'green' for g in gains], alpha=0.7, edgecolor='white')
        ax.set_xticks(range(len(bands))); ax.set_xticklabels(bands, rotation=45, ha='right', fontsize=8)
        ax.set_title('EQ Curve (Adaptive)', fontsize=10); ax.axhline(0, color='white', ls='--', alpha=0.5); plt.tight_layout()
        buf = io.BytesIO(); fig.savefig(buf, format='png'); buf.seek(0); plt.close(fig)
        return torch.from_numpy(np.array(Image.open(buf).convert("RGB")).astype(np.float32)/255.0).unsqueeze(0)

NODE_CLASS_MAPPINGS = {"MD_AudioAutoEQ_Adaptive": MD_AudioAutoEQ_Adaptive}
NODE_DISPLAY_NAME_MAPPINGS = {"MD_AudioAutoEQ_Adaptive": "MD: Audio Auto EQ (Adaptive)"}


# ==============================================================================
# == Unit Tests (smoke — runs without ComfyUI)
# ==============================================================================

if __name__ == "__main__":
    print("\n🧪 Smoke tests: MD_AudioAutoEQ")
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

    _check("VERSION defined",    VERSION == "v3.3.0")
    _check("NODE_CLASS_MAPPINGS defined",
           isinstance(NODE_CLASS_MAPPINGS, dict) and len(NODE_CLASS_MAPPINGS) > 0)
    _check("  class MD_AudioAutoEQ_Adaptive in map", "MD_AudioAutoEQ_Adaptive" in NODE_CLASS_MAPPINGS)

    print(f"\n  {_pass} passed, {_fail} failed")
    if _fail == 0:
        print("  🎉 All good.")
