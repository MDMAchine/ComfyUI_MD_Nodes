# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# █▓▒░                                                                     ░▒▓█
# █▓▒░  MD_Nodes/ACELatentVisualizer – Latent Tensor Visualization v1.6.1  ░▒▓█
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
# ║ ░▒▓ ORIGIN: Latent Space Analysis / Matplotlib Visualization
# ║ ░▒▓ DESCRIPTION:
# ║    Advanced inspection tool for latent tensors utilizing the Core/Wrapper architecture.
# ║    Offers multiple visualization modes to debug conditioning, analyze model
# ║    outputs, or explore latent data creatively.
# ╚════════════════════════════════════════════════════════════════════════════


VERSION = "v1.6.1"  # UPS v1.5.8

import os
import sys
import io
import logging
import re
import traceback
import time

import torch
import numpy as np
from PIL import Image

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
    import latent_visualizer_core_bin as core
    CORE_LOADED = True
    CORE_MODE = "Binary (Production)"
except ImportError as e1:
    try:
        import latent_visualizer_core as core
        CORE_LOADED = True
        CORE_MODE = "Source (Development)"
    except ImportError as e2:
        CORE_ERROR = f"Binary: {e1}\nSource: {e2}"

# =================================================================================
# == UI Dependencies & Constants
# =================================================================================

try:
    import matplotlib as mpl
    mpl.use('Agg')  # Set backend before pyplot import
    import matplotlib.pyplot as plt
    CONST_MATPLOTLIB_AVAILABLE = True
except ImportError:
    CONST_MATPLOTLIB_AVAILABLE = False
    logging.warning("[ACELatentVisualizer] Matplotlib not found. Visualization disabled.")

# Plotting Standards
CONST_PLOT_DPI = 100
CONST_DEFAULT_LINEWIDTH = 0.75
CONST_GRID_LINEWIDTH = 0.3
CONST_GRID_ALPHA = 0.5
CONST_COLORBAR_FRACTION = 0.046
CONST_COLORBAR_PAD = 0.04

# Colors (MD Standard)
CONST_COLOR_BG = "#0D0D1A"
CONST_COLOR_WAVEFORM = "#87CEEB"  # MD Sky Blue
CONST_COLOR_SPECTRUM = "#FF00A2"
CONST_COLOR_RGB_R = "#FF3333"
CONST_COLOR_RGB_G = "#00FF8C"
CONST_COLOR_RGB_B = "#3399FF"
CONST_COLOR_LABEL = "#A0A0B0"
CONST_COLOR_GRID = "#303040"

# =================================================================================
# == Helper Classes
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
        print("\n⏱️  PERFORMANCE:")
        total = self.get_total_time()
        print(f"    • Total Time: {total:.2f}s")
        for op_name, times in sorted(self.timings.items()):
            avg = sum(times) / len(times)
            if len(times) == 1:
                print(f"    • {op_name}: {avg:.3f}s")
            else:
                print(f"    • {op_name}: {avg:.3f}s avg ({len(times)}x)")

# =================================================================================
# == Core Node Class
# =================================================================================

class ACELatentVisualizer:
    """
    MD: ACE Latent Visualizer
    Provides multiple visualization modes for inspecting latent tensors.
    Utilizes Core algorithms for data extraction and Matplotlib for UI rendering.
    """
    CATEGORY = "MD_Nodes/Debugging & Visualization"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "visualize"

    @classmethod
    def INPUT_TYPES(cls):
        """Define all input parameters with comprehensive tooltips."""
        return {
            "required": {
                "latent": ("LATENT", {
                    "tooltip": (
                        "LATENT INPUT\n"
                        "• Purpose: The latent tensor dictionary to visualize.\n"
                        "• Accepts: Standard ComfyUI Latent format.\n"
                        "• Output: Generates an image plot based on this data."
                    )
                }),
                "mode": (["waveform", "spectrum", "rgb_split", "heatmap", "histogram",
                          "statistics", "multi_channel", "phase", "difference"], {
                    "default": "waveform",
                    "tooltip": (
                        "VISUALIZATION MODE\n"
                        "• Purpose: Select the type of analysis to perform.\n"
                        "• Options:\n"
                        "  - waveform: Amplitude over spatial dimension.\n"
                        "  - spectrum: Frequency magnitude (FFT).\n"
                        "  - rgb_split: Overlay first 3 channels.\n"
                        "  - heatmap: 2D spatial representation.\n"
                        "  - histogram: Value distribution.\n"
                        "  - statistics: Mean/Std across channels.\n"
                        "  - difference: Compare two latents.\n"
                        "\n⭐ Recommended: 'waveform' for basic checks."
                    )
                }),
                "channel": ("INT", {
                    "default": 0, "min": 0,
                    "tooltip": (
                        "CHANNEL INDEX\n"
                        "• Purpose: Target specific latent channel for 1D modes.\n"
                        "• Range: 0 to (Max Channels - 1).\n"
                        "\n⭐ Recommended: 0"
                    )
                }),
                "batch_index": ("INT", {
                    "default": 0, "min": 0,
                    "tooltip": (
                        "BATCH INDEX\n"
                        "• Purpose: Target specific image/audio in a batched latent.\n"
                        "• Range: 0 to (Batch Size - 1).\n"
                        "\n⭐ Recommended: 0"
                    )
                }),
                "normalize": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "NORMALIZE SIGNAL\n"
                        "• Purpose: Rescale amplitudes to [0, 1] range for easier viewing.\n"
                        "• Trade-offs: Makes tiny signals visible, but hides true absolute magnitude.\n"
                        "\n⭐ Recommended: True"
                    )
                }),
                "width": ("INT", {
                    "default": 512, "min": 64, "max": 2048, "step": 64, 
                    "tooltip": (
                        "PLOT WIDTH\n"
                        "• Purpose: Output image width in pixels.\n"
                        "• Range: 64 to 2048.\n"
                        "\n⭐ Recommended: 512"
                    )
                }),
                "height": ("INT", {
                    "default": 256, "min": 64, "max": 2048, "step": 64, 
                    "tooltip": (
                        "PLOT HEIGHT\n"
                        "• Purpose: Output image height in pixels.\n"
                        "• Range: 64 to 2048.\n"
                        "\n⭐ Recommended: 256"
                    )
                }),
                "grid": ("BOOLEAN", {
                    "default": True, 
                    "tooltip": (
                        "DISPLAY GRID\n"
                        "• Purpose: Show background grid lines for visual reference.\n"
                        "\n⭐ Recommended: True"
                    )
                }),

                # --- Color Controls ---
                "bg_color": ("STRING", {"default": CONST_COLOR_BG, "tooltip": "BACKGROUND COLOR\n• Valid Hex Code (e.g., #0D0D1A)"}),
                "waveform_color": ("STRING", {"default": CONST_COLOR_WAVEFORM, "tooltip": "WAVEFORM COLOR\n• Valid Hex Code (e.g., #87CEEB)"}),
                "spectrum_color": ("STRING", {"default": CONST_COLOR_SPECTRUM, "tooltip": "SPECTRUM COLOR\n• Valid Hex Code (e.g., #FF00A2)"}),
                "rgb_r_color": ("STRING", {"default": CONST_COLOR_RGB_R, "tooltip": "RED CHANNEL COLOR\n• Valid Hex Code"}),
                "rgb_g_color": ("STRING", {"default": CONST_COLOR_RGB_G, "tooltip": "GREEN CHANNEL COLOR\n• Valid Hex Code"}),
                "rgb_b_color": ("STRING", {"default": CONST_COLOR_RGB_B, "tooltip": "BLUE CHANNEL COLOR\n• Valid Hex Code"}),
                "axis_label_color": ("STRING", {"default": CONST_COLOR_LABEL, "tooltip": "AXIS LABEL COLOR\n• Valid Hex Code"}),
                "grid_color": ("STRING", {"default": CONST_COLOR_GRID, "tooltip": "GRID COLOR\n• Valid Hex Code"}),

                # --- Advanced Options ---
                "all_modes": ("BOOLEAN", {
                    "default": False, 
                    "tooltip": (
                        "RENDER ALL MODES\n"
                        "• Purpose: Stacks all applicable visualization modes into one large image.\n"
                        "• Trade-offs: Slower generation, very tall output image.\n"
                        "\n⭐ Recommended: False"
                    )
                }),
                "log_scale": ("BOOLEAN", {
                    "default": False, 
                    "tooltip": (
                        "LOGARITHMIC SCALE\n"
                        "• Purpose: Use Log (dB) scale for Spectrum and Histogram Y-axis.\n"
                        "\n⭐ Recommended: False (unless analyzing audio)"
                    )
                }),
                "show_stats": ("BOOLEAN", {
                    "default": False, 
                    "tooltip": (
                        "SHOW STATISTICS\n"
                        "• Purpose: Overlay Mean, Std, Min, Max values on the plot.\n"
                        "\n⭐ Recommended: True for debugging"
                    )
                }),
                "detect_peaks": ("BOOLEAN", {
                    "default": False, 
                    "tooltip": (
                        "DETECT PEAKS\n"
                        "• Purpose: Marks local maxima on Waveform/Spectrum plots.\n"
                        "• Requirement: Requires Scipy installed in python env.\n"
                        "\n⭐ Recommended: False"
                    )
                }),
                "peak_threshold": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": (
                        "PEAK THRESHOLD\n"
                        "• Purpose: Minimum relative height (0-1) for a point to be considered a peak.\n"
                        "\n⭐ Recommended: 0.5"
                    )
                }),
                "line_style": (["solid", "dashed", "dotted", "dashdot"], {
                    "default": "solid",
                    "tooltip": "LINE STYLE\n• Purpose: Visual style of the plot lines."
                }),
                "line_alpha": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 1.0, "step": 0.1,
                    "tooltip": "LINE OPACITY\n• Purpose: Transparency of the plot lines."
                }),
                "multi_channel_count": ("INT", {
                    "default": 3, "min": 1, "max": 16,
                    "tooltip": "MULTI-CHANNEL COUNT\n• Purpose: Number of channels to render in 'multi_channel' mode."
                }),
                "colormap": (["viridis", "plasma", "inferno", "magma", "cividis", "twilight",
                              "turbo", "hot", "cool", "spring", "winter"], {
                    "default": "viridis",
                    "tooltip": "COLORMAP\n• Purpose: Gradient used for Heatmap mode."
                }),
            },
            "optional": {
                "latent_compare": ("LATENT", {
                    "tooltip": (
                        "LATENT COMPARE\n"
                        "• Purpose: Second latent tensor required for 'difference' mode.\n"
                        "• Requirement: Must match dimensions of primary latent."
                    )
                }),
            }
        }

    @staticmethod
    def validate_color(color_str, default="#FFFFFF"):
        if not isinstance(color_str, str): return default
        color_str = color_str.strip()
        if re.match(r'^#(?:[0-9a-fA-F]{3}){1,2}$', color_str):
             if len(color_str) == 4:
                  return f"#{color_str[1]*2}{color_str[2]*2}{color_str[3]*2}"
             return color_str
        return default

    def _setup_axis_style(self, ax, bg_color, label_color, grid, grid_color):
        try:
            ax.set_facecolor(bg_color)
            ax.tick_params(axis='x', colors=label_color, labelsize=6)
            ax.tick_params(axis='y', colors=label_color, labelsize=6)
            ax.xaxis.label.set_color(label_color)
            ax.yaxis.label.set_color(label_color)
            ax.title.set_color(label_color)

            for spine in ax.spines.values():
                 spine.set_edgecolor(label_color)
                 spine.set_linewidth(0.5)
                 spine.set_alpha(0.7)

            if grid:
                ax.grid(True, which='both', linestyle=':',
                        linewidth=CONST_GRID_LINEWIDTH, alpha=CONST_GRID_ALPHA, color=grid_color)
            else:
                 ax.set_xticks([])
                 ax.set_yticks([])
                 ax.set_xlabel("")
                 ax.set_ylabel("")
        except Exception as e:
            logging.error(f"[ACELatentVisualizer] Axis style error: {e}")

    def _add_statistics_overlay(self, ax, data, color):
        if len(data) == 0: return
        try:
            stats_text = (f"μ={np.mean(data):.3f}\n"
                          f"σ={np.std(data):.3f}\n"
                          f"min={np.min(data):.3f}\n"
                          f"max={np.max(data):.3f}")
            ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                    verticalalignment='top', horizontalalignment='right',
                    fontsize=6, color=color,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.6))
        except Exception: pass

    def _get_line_style(self, style_name):
        styles = {"solid": "-", "dashed": "--", "dotted": ":", "dashdot": "-."}
        return styles.get(style_name, "-")

    # --- Plot Rendering Functions ---

    def _plot_waveform(self, ax, signal_data, channel, color, style, alpha, normalize,
                       show_stats, detect_peaks, peak_threshold, label_color):
        if signal_data.size == 0:
             ax.text(0.5, 0.5, "No Waveform Data", ha='center', va='center', transform=ax.transAxes, color=label_color)
             return

        ax.plot(signal_data, linewidth=CONST_DEFAULT_LINEWIDTH, color=color, linestyle=style, alpha=alpha)
        ax.set_title(f"Latent Waveform (Ch {channel})", fontsize=8)
        ax.set_ylabel("Amplitude", fontsize=6)
        
        y_min = 0 if normalize else np.min(signal_data) - 0.1 * np.ptp(signal_data)
        y_max = 1 if normalize else np.max(signal_data) + 0.1 * np.ptp(signal_data)
        ax.set_ylim(y_min, y_max)
        ax.set_xlim(0, len(signal_data) - 1 if len(signal_data) > 1 else 1)

        if show_stats: self._add_statistics_overlay(ax, signal_data, label_color)
        if detect_peaks and CORE_LOADED: 
            px, py = core.detect_peaks(signal_data, peak_threshold)
            if px.size > 0:
                ax.plot(px, py, 'x', color=color, markersize=5, markeredgewidth=1.0, label=f'{len(px)} peaks')
                ax.legend(loc="upper left", fontsize=6, frameon=False, labelcolor=label_color)

    def _plot_spectrum(self, ax, raw_signal, channel, color, log_scale, show_stats,
                       detect_peaks, peak_threshold, label_color):
        if raw_signal.size < 2 or not CORE_LOADED: return
        try:
            freqs, plot_data = core.compute_spectrum(raw_signal, log_scale)
            y_label = "Magnitude (dB)" if log_scale else "Magnitude"

            ax.plot(freqs, plot_data, color=color, linewidth=CONST_DEFAULT_LINEWIDTH)
            ax.set_ylabel(y_label, fontsize=6)
            ax.set_title(f"Latent Spectrum (Ch {channel})", fontsize=8)
            ax.set_xlim(0, freqs.max())

            if show_stats: self._add_statistics_overlay(ax, plot_data, label_color)
            if detect_peaks: 
                px, py = core.detect_peaks(plot_data, peak_threshold)
                if px.size > 0:
                    ax.plot(freqs[px], py, 'x', color=color, markersize=5)
        except Exception: pass

    def _plot_rgb_split(self, ax, data_chw, rgb_colors, normalize, style, alpha, label_color):
        if data_chw.shape[0] < 3 or not CORE_LOADED: return
        labels = ["R", "G", "B"]
        
        for i in range(3):
            signal_data = core.extract_1d_signal(data_chw[i], normalize)
            if signal_data.size > 0:
                 ax.plot(signal_data, linewidth=CONST_DEFAULT_LINEWIDTH, color=rgb_colors[i],
                         label=labels[i], linestyle=style, alpha=alpha)
        
        ax.set_title("Latent RGB Channel Split", fontsize=8)
        if normalize: ax.set_ylim(0, 1)
        ax.legend(loc="upper right", fontsize=6, frameon=False, labelcolor=label_color)

    def _plot_heatmap(self, ax, data_chw, channel, colormap):
        try:
            data_2d = data_chw[channel].detach().cpu().numpy()
            im = ax.imshow(data_2d, cmap=colormap, aspect='auto', interpolation='nearest')
            ax.set_title(f"Latent Heatmap (Ch {channel})", fontsize=8)
            plt.colorbar(im, ax=ax, fraction=CONST_COLORBAR_FRACTION, pad=CONST_COLORBAR_PAD)
        except Exception: pass

    def _plot_histogram(self, ax, data_chw, channel, color, log_scale, show_stats, label_color):
        try:
            values = data_chw[channel].detach().cpu().numpy().flatten()
            ax.hist(values, bins=50, color=color, alpha=0.75, edgecolor='black', linewidth=0.3, log=log_scale)
            ax.set_title(f"Latent Histogram (Ch {channel})", fontsize=8)
            if show_stats: self._add_statistics_overlay(ax, values, label_color)
        except Exception: pass

    def _plot_statistics(self, ax, data_chw, label_color, bg_color):
        if not CORE_LOADED: return
        try:
            means, stds, mins, maxs, x_ticks = core.compute_statistics(data_chw)
            
            ax.plot(x_ticks, means, 'o-', color=CONST_COLOR_WAVEFORM, markersize=3, linewidth=1)
            ax.fill_between(x_ticks, means - stds, means + stds, alpha=0.3, color=CONST_COLOR_WAVEFORM)
            ax.set_title("Channel Statistics", fontsize=8)
            ax.set_xticks(x_ticks)
        except Exception: pass

    def _plot_multi_channel(self, ax, data_chw, count, normalize, style, alpha, label_color):
        if not CORE_LOADED: return
        try:
            num = min(count, data_chw.shape[0], 16)
            colors = plt.cm.rainbow(np.linspace(0, 1, num))
            for i in range(num):
                sig = core.extract_1d_signal(data_chw[i], normalize)
                ax.plot(sig, linewidth=CONST_DEFAULT_LINEWIDTH, color=colors[i], linestyle=style, alpha=alpha)
            ax.set_title(f"Multi-Channel Overlay ({num} Ch)", fontsize=8)
        except Exception: pass

    def _plot_phase(self, ax, raw_signal, channel, color, show_stats, label_color):
        if raw_signal.size < 2 or not CORE_LOADED: return
        try:
            freqs, phase = core.compute_phase(raw_signal)
            ax.plot(freqs, phase, color=color, linewidth=CONST_DEFAULT_LINEWIDTH)
            ax.set_title(f"Phase Spectrum (Ch {channel})", fontsize=8)
            ax.set_xlim(0, freqs.max())
            if show_stats: self._add_statistics_overlay(ax, phase, label_color)
        except Exception: pass

    def _plot_difference(self, ax, data1, data2, channel, color, normalize, show_stats, label_color, style, alpha):
        if not CORE_LOADED: return
        try:
            s1 = core.extract_1d_signal(data1[channel], normalize)
            s2 = core.extract_1d_signal(data2[channel], normalize)
            
            diff = core.compute_difference(s1, s2)
            
            ax.plot(diff, linewidth=CONST_DEFAULT_LINEWIDTH, color=color, linestyle=style, alpha=alpha)
            ax.axhline(y=0, color=label_color, linestyle='--', alpha=0.7, linewidth=0.5)
            ax.set_title(f"Difference (Ch {channel})", fontsize=8)
            if show_stats: self._add_statistics_overlay(ax, diff, label_color)
        except Exception: pass

    # --- Main Execution ---

    def visualize(self, latent, mode, channel, batch_index, normalize, width, height, grid,
                  bg_color, waveform_color, spectrum_color, rgb_r_color, rgb_g_color, rgb_b_color,
                  axis_label_color, grid_color, all_modes, log_scale, show_stats, detect_peaks,
                  peak_threshold, line_style, line_alpha, multi_channel_count, colormap,
                  latent_compare=None):
        
        # Graceful Degradation: Matplotlib Check
        if not CONST_MATPLOTLIB_AVAILABLE:
            logging.error("[ACELatentVisualizer] Matplotlib missing. Returning blank image.")
            return (torch.zeros((1, height, width, 3)),)
            
        # Graceful Degradation: Core Check
        if not CORE_LOADED:
            logging.warning(f"[ACELatentVisualizer] Core Missing. Some math-heavy modes disabled.\nMode: {CORE_MODE or 'Not Loaded'}\nError: {CORE_ERROR}")

        profiler = PerformanceProfiler(enabled=True)
        profiler.start("total")

        fig, buf = None, None
        try:
            if not isinstance(latent, dict) or "samples" not in latent:
                raise ValueError("Invalid LATENT input.")
            
            t = latent["samples"]
            b, c, h, w_lat = t.shape
            batch_idx = min(max(0, batch_index), b - 1)
            ch_idx = min(max(0, channel), c - 1)
            x_batch = t[batch_idx]

            modes = []
            if all_modes:
                modes = ["waveform", "spectrum", "rgb_split", "heatmap", "histogram", 
                         "statistics", "multi_channel", "phase", "difference"]
            else:
                modes = [mode]

            if c < 3 and "rgb_split" in modes: modes.remove("rgb_split")
            if latent_compare is None and "difference" in modes: modes.remove("difference")
            if not modes: modes = ["waveform"]

            num_plots = len(modes)
            fig, axes = plt.subplots(num_plots, 1, figsize=(width / CONST_PLOT_DPI, height / CONST_PLOT_DPI),
                                     dpi=CONST_PLOT_DPI, facecolor=bg_color)
            if num_plots == 1: axes = [axes]

            lc_batch = None
            if latent_compare and "samples" in latent_compare:
                 lc_batch = latent_compare["samples"][min(batch_idx, latent_compare["samples"].shape[0]-1)]

            ls = self._get_line_style(line_style)
            
            profiler.start("rendering")
            for idx, m in enumerate(modes):
                ax = axes[idx]
                self._setup_axis_style(ax, bg_color, axis_label_color, grid, grid_color)
                
                # Check Core requirement for specific modes
                if not CORE_LOADED and m in ["spectrum", "phase", "statistics", "multi_channel", "difference", "rgb_split"]:
                    ax.text(0.5, 0.5, f"Mode '{m}' Requires Core", ha='center', va='center', transform=ax.transAxes, color=axis_label_color)
                    continue

                if m == "waveform":
                    sig = core.extract_1d_signal(x_batch[ch_idx], normalize) if CORE_LOADED else x_batch[ch_idx].detach().cpu().numpy().flatten()
                    self._plot_waveform(ax, sig, ch_idx, waveform_color, ls, line_alpha, normalize,
                                        show_stats, detect_peaks, peak_threshold, axis_label_color)
                elif m == "spectrum":
                    raw = x_batch[ch_idx].detach().cpu().numpy().flatten()
                    self._plot_spectrum(ax, raw, ch_idx, spectrum_color, log_scale, show_stats,
                                        detect_peaks, peak_threshold, axis_label_color)
                elif m == "rgb_split":
                    self._plot_rgb_split(ax, x_batch, [rgb_r_color, rgb_g_color, rgb_b_color],
                                         normalize, ls, line_alpha, axis_label_color)
                elif m == "heatmap":
                    self._plot_heatmap(ax, x_batch, ch_idx, colormap)
                elif m == "histogram":
                    self._plot_histogram(ax, x_batch, ch_idx, waveform_color, log_scale, show_stats, axis_label_color)
                elif m == "statistics":
                    self._plot_statistics(ax, x_batch, axis_label_color, bg_color)
                elif m == "multi_channel":
                    self._plot_multi_channel(ax, x_batch, multi_channel_count, normalize, ls, line_alpha, axis_label_color)
                elif m == "phase":
                    raw = x_batch[ch_idx].detach().cpu().numpy().flatten()
                    self._plot_phase(ax, raw, ch_idx, spectrum_color, show_stats, axis_label_color)
                elif m == "difference" and lc_batch is not None:
                    self._plot_difference(ax, x_batch, lc_batch, ch_idx, waveform_color, normalize,
                                          show_stats, axis_label_color, ls, line_alpha)

            plt.tight_layout(pad=0.5, h_pad=0.8)
            profiler.stop("rendering")
            
            profiler.start("image_conversion")
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1, facecolor=bg_color, dpi=CONST_PLOT_DPI)
            buf.seek(0)
            
            img = Image.open(buf).convert("RGB")
            img_np = np.array(img).astype(np.float32) / 255.0
            profiler.stop("image_conversion")
            
            profiler.stop("total")
            
            return (torch.from_numpy(img_np).unsqueeze(0),)

        except Exception as e:
            logging.error(f"[ACELatentVisualizer] Error: {e}")
            traceback.print_exc()
            return (torch.zeros((1, height, width, 3)),)
        
        finally:
            if buf: buf.close()
            if fig: plt.close(fig)
            plt.clf()

# =================================================================================
# == Node Registration
# =================================================================================

NODE_CLASS_MAPPINGS = {
    "ACE_LatentVisualizer": ACELatentVisualizer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ACE_LatentVisualizer": "MD: Latent Visualizer",
}


# ==============================================================================
# == Unit Tests (smoke — runs without ComfyUI)
# ==============================================================================

if __name__ == "__main__":
    print("\n🧪 Smoke tests: MD_LatentVisualizer")
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
    _check("CONST CONST_PLOT_DPI defined", CONST_PLOT_DPI is not None)
    _check("CONST CONST_DEFAULT_LINEWIDTH defined", CONST_DEFAULT_LINEWIDTH is not None)
    _check("CONST CONST_GRID_LINEWIDTH defined", CONST_GRID_LINEWIDTH is not None)
    _check("CONST CONST_GRID_ALPHA defined", CONST_GRID_ALPHA is not None)
    _check("CONST CONST_COLORBAR_FRACTION defined", CONST_COLORBAR_FRACTION is not None)
    _check("NODE_CLASS_MAPPINGS defined",
           isinstance(NODE_CLASS_MAPPINGS, dict) and len(NODE_CLASS_MAPPINGS) > 0)
    _check("  class ACE_LatentVisualizer in map", "ACE_LatentVisualizer" in NODE_CLASS_MAPPINGS)

    print(f"\n  {_pass} passed, {_fail} failed")
    if _fail == 0:
        print("  🎉 All good.")
