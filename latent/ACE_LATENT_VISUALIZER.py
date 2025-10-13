# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# ████ ACE LATENT VISUALIZER v0.4.0 – Optimized for Ace-Step Audio/Video ████▓▒░
# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀

# ░▒▓ ORIGIN & DEV:
#   • Cast into the void by: MDMAchine (The latent abyss explorer)
#   • Original mind behind ACE LATENT VISUALIZER
#   • Initial ComfyUI adaptation by: devstral (local l33t)
#   • Enhanced & color calibrated by: Gemini (Pixel artist extraordinaire)
#   • v0.4.0 Enhancement Architecture by: Claude (Anthropic AI Assistant)
#   • Critical inspirations from: c0ffymachyne (signal processing wizardry)
#   • License: Apache 2.0 — Sharing digital secrets responsibly

# ░▒▓ DESCRIPTION:
#   The ultimate latent tensor cartographer for ComfyUI with Ace-Step precision.
#   Offers multi-mode insight into latent tensors — waveform, spectrum, RGB splits,
#   heatmaps, histograms, phase plots, and statistical analysis. Turn abstract data
#   blobs into visible art. Ideal for debugging, pattern spotting, comparative analysis,
#   or admiring your AI's hidden guts. Now with batch selection, peak detection,
#   and advanced visualization modes that make latent space exploration feel like
#   commanding a research vessel through the neural cosmos.

# ░▒▓ FEATURES:
#   ✓ Nine visualization modes: waveform, spectrum, rgb_split, heatmap, histogram, statistics, multi_channel, phase, difference
#   ✓ Batch selection support - visualize any batch item independently
#   ✓ Customizable channel selection and normalization
#   ✓ Peak detection with configurable threshold - spot anomalies instantly
#   ✓ Phase spectrum analysis for frequency insights
#   ✓ Difference mode - compare two latents side-by-side
#   ✓ Multi-channel overlay - see up to 16 channels simultaneously
#   ✓ Statistical analysis overlay - mean, std, min, max on demand
#   ✓ Logarithmic scale options for spectrum and histogram modes
#   ✓ Supports stacked multi-mode plotting (view all modes at once)
#   ✓ Adjustable resolution, gridlines, and color themes
#   ✓ Line style customization: solid, dashed, dotted, dashdot
#   ✓ Transparency/alpha controls for cleaner overlays
#   ✓ 11 colormap options for heatmap visualization
#   ✓ Robust color validation with intelligent fallbacks
#   ✓ Memory leak prevention with proper resource cleanup
#   ✓ Lightweight Matplotlib backend for crisp visuals
#   ✓ Comprehensive error handling with graceful degradation
#   ✓ Type hints throughout for better IDE support

# ░▒▓ CHANGELOG:
#   - v0.4.0 (Major Enhancement Release):
#       • NEW MODES: Heatmap, Histogram, Statistics, Multi-channel overlay, Difference, Phase
#       • Batch selection support - visualize any batch item independently
#       • Peak detection with configurable threshold for anomaly spotting
#       • Logarithmic scale options for spectrum and histogram
#       • Statistics overlay option (mean, std, min, max) on any plot
#       • Line style customization (solid, dashed, dotted, dashdot)
#       • Transparency/alpha controls for overlay clarity
#       • 11 colormap options for heatmap mode (viridis, plasma, inferno, etc.)
#       • Color validation with intelligent fallback system
#       • Memory leak fixes - proper buffer cleanup with try/finally
#       • Type hints added throughout for better code clarity
#       • Helper methods for cleaner, more maintainable code
#       • Extracted magic numbers into named constants
#       • Comprehensive error handling with detailed logging
#       • Better resource management and cleanup
#   - v0.3.1 (Refinement & Usability):
#       • Enhanced color calibration and UI tooltips
#       • Stability improvements and bug fixes
#       • Better default color choices for dark backgrounds
#   - v0.2 (Feature Expansion):
#       • Added spectrum mode with decibel scaling
#       • RGB channel splitting introduced
#       • Improved plotting customization
#       • Grid overlay controls
#   - v0.1 (Initial Release):
#       • Basic waveform mode implemented
#       • Single channel visualization support
#       • Foundation for multi-mode visualization

# ░▒▓ CONFIGURATION:
#   → Primary Use: Latent tensor visual inspection and debugging workflows
#   → Secondary Use: Artistic exploration of AI latent spaces for creative projects
#   → Edge Use: Data-driven generative art prototyping and pattern discovery
#   → Advanced Use: Comparative analysis between model architectures or training stages
#   → Research Use: Publication-ready visualizations of neural network internals

# ░▒▓ WARNING:
#   This node may trigger:
#   ▓▒░ Obsessive pattern analysis and late-night debugging sessions
#   ▓▒░ Sudden urges to tweak latent spaces endlessly seeking perfection
#   ▓▒░ Creative inspiration overload from abstract data beauty
#   ▓▒░ Compulsive channel hopping through your latent dimensions
#   ▓▒░ Existential questions about what neural networks really "see"
#   ▓▒░ Temporal distortion from staring at heatmaps too long
#   ▓▒░ Uncontrollable desire to visualize everything in your workflow

# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import io
from PIL import Image
import numpy as np
from scipy import signal
from typing import Dict, List, Tuple, Optional, Any

# Set Matplotlib backend to 'Agg' for headless rendering
plt.switch_backend('Agg')

# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS - All magic numbers extracted for easy tuning
# ═══════════════════════════════════════════════════════════════════════════
DEFAULT_DPI = 100
MIN_SIGNAL_VARIANCE = 1e-6
LOG_EPSILON = 1e-8  # For log calculations to avoid log(0)
FLAT_SIGNAL_OFFSET = 0.5  # Middle value for flat signals
DEFAULT_LINEWIDTH = 0.75
GRID_LINEWIDTH = 0.3
GRID_ALPHA = 0.5
COLORBAR_FRACTION = 0.046
COLORBAR_PAD = 0.04
PEAK_MIN_DISTANCE = 10  # Minimum distance between detected peaks


class ACE_LatentVisualizer_v04:
    """
    ACE_LatentVisualizer_v04: Enhanced latent data visualization with advanced features!
    
    Now with batch selection, multiple visualization modes, statistical analysis,
    peak detection, and much more!
    """

    CATEGORY = "MD_Nodes/Visualization"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "visualize"

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        """Define all input parameters with enhanced options."""
        return {
            "required": {
                "latent": ("LATENT",),
                "mode": (["waveform", "spectrum", "rgb_split", "heatmap", "histogram", 
                         "statistics", "multi_channel", "phase", "difference"], 
                        {"default": "waveform"}),
                "channel": ("INT", {"default": 0, "min": 0, 
                           "tooltip": "Primary channel to visualize"}),
                "batch_index": ("INT", {"default": 0, "min": 0,
                               "tooltip": "Which batch item to visualize"}),
                "normalize": ("BOOLEAN", {"default": True, 
                             "tooltip": "Auto-adjust amplitude for better visibility"}),
                "width": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64}),
                "height": ("INT", {"default": 256, "min": 64, "max": 2048, "step": 64}),
                "grid": ("BOOLEAN", {"default": True, "tooltip": "Show grid lines"}),
                
                # Color Controls
                "bg_color": ("STRING", {"default": "#0D0D1A"}),
                "waveform_color": ("STRING", {"default": "#00E6E6"}),
                "spectrum_color": ("STRING", {"default": "#FF00A2"}),
                "rgb_r_color": ("STRING", {"default": "#FF3333"}),
                "rgb_g_color": ("STRING", {"default": "#00FF8C"}),
                "rgb_b_color": ("STRING", {"default": "#3399FF"}),
                "axis_label_color": ("STRING", {"default": "#A0A0B0"}),
                "grid_color": ("STRING", {"default": "#303040"}),
                
                # Advanced Options
                "all_modes": ("BOOLEAN", {"default": False, 
                             "tooltip": "Show all visualization modes (ignores 'mode' selection)"}),
                "log_scale": ("BOOLEAN", {"default": False,
                            "tooltip": "Use logarithmic Y-axis (for spectrum/histogram)"}),
                "show_stats": ("BOOLEAN", {"default": False,
                             "tooltip": "Overlay statistics on plots"}),
                "detect_peaks": ("BOOLEAN", {"default": False,
                               "tooltip": "Detect and mark peaks in waveform/spectrum"}),
                "peak_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                                  "tooltip": "Threshold for peak detection (0-1)"}),
                "line_style": (["solid", "dashed", "dotted", "dashdot"], {"default": "solid"}),
                "line_alpha": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 1.0, "step": 0.1,
                              "tooltip": "Line transparency (1.0 = opaque)"}),
                "multi_channel_count": ("INT", {"default": 3, "min": 1, "max": 16,
                                       "tooltip": "Number of channels to overlay in multi_channel mode"}),
                "colormap": (["viridis", "plasma", "inferno", "magma", "cividis", "twilight", 
                             "turbo", "hot", "cool", "spring", "winter"], {"default": "viridis"}),
            },
            "optional": {
                "latent_compare": ("LATENT", {"tooltip": "Second latent for difference mode"}),
            }
        }

    @staticmethod
    def validate_color(color_str: str, default: str = "#FFFFFF") -> str:
        """Validate hex color format and return valid color or default."""
        if not isinstance(color_str, str):
            return default
        
        color_str = color_str.strip()
        if not color_str.startswith('#'):
            return default
        
        if len(color_str) not in [4, 7]:  # #RGB or #RRGGBB
            return default
        
        try:
            int(color_str[1:], 16)
            return color_str
        except ValueError:
            return default

    def _setup_axis_style(self, ax: plt.Axes, bg_color: str, label_color: str, 
                          grid: bool, grid_color: str) -> None:
        """Centralized axis styling configuration."""
        ax.set_facecolor(bg_color)
        ax.tick_params(axis='x', colors=label_color, labelsize=6)
        ax.tick_params(axis='y', colors=label_color, labelsize=6)
        ax.xaxis.label.set_color(label_color)
        ax.yaxis.label.set_color(label_color)
        ax.title.set_color(label_color)
        
        if grid:
            ax.grid(True, which='both', linestyle='--', 
                   linewidth=GRID_LINEWIDTH, alpha=GRID_ALPHA, color=grid_color)
        else:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("")
            ax.set_ylabel("")

    def _get_1d_signal(self, data: torch.Tensor, normalize: bool) -> np.ndarray:
        """Extract and prepare 1D signal from tensor data."""
        signal = data.detach().cpu().numpy().flatten()
        
        if np.allclose(signal, 0) or np.ptp(signal) < MIN_SIGNAL_VARIANCE:
            return np.zeros_like(signal) + FLAT_SIGNAL_OFFSET
        
        if normalize:
            min_val, max_val = signal.min(), signal.max()
            if max_val - min_val > LOG_EPSILON:
                signal = (signal - min_val) / (max_val - min_val)
            else:
                signal = np.zeros_like(signal) + FLAT_SIGNAL_OFFSET
        
        return signal

    def _add_statistics_overlay(self, ax: plt.Axes, data: np.ndarray, 
                                color: str) -> None:
        """Add statistics text overlay to plot."""
        stats_text = (f"μ={np.mean(data):.3f}\n"
                     f"σ={np.std(data):.3f}\n"
                     f"min={np.min(data):.3f}\n"
                     f"max={np.max(data):.3f}")
        
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
               verticalalignment='top', horizontalalignment='right',
               fontsize=6, color=color, 
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))

    def _detect_and_mark_peaks(self, ax: plt.Axes, signal: np.ndarray, 
                               threshold: float, color: str) -> None:
        """Detect peaks and mark them on the plot."""
        # Normalize threshold to signal range
        if len(signal) == 0:
            return
        
        peak_height = threshold * (np.max(signal) - np.min(signal)) + np.min(signal)
        peaks, properties = signal.find_peaks(signal, height=peak_height, 
                                             distance=PEAK_MIN_DISTANCE)
        
        if len(peaks) > 0:
            ax.plot(peaks, signal[peaks], 'x', color=color, markersize=6, 
                   markeredgewidth=1.5, label=f'{len(peaks)} peaks')

    def _get_line_style(self, style_name: str) -> str:
        """Convert style name to matplotlib linestyle."""
        styles = {
            "solid": "-",
            "dashed": "--",
            "dotted": ":",
            "dashdot": "-."
        }
        return styles.get(style_name, "-")

    def _plot_waveform(self, ax: plt.Axes, signal: np.ndarray, channel: int,
                       color: str, style: str, alpha: float, normalize: bool,
                       show_stats: bool, detect_peaks: bool, peak_threshold: float,
                       label_color: str) -> None:
        """Plot waveform visualization."""
        ax.plot(signal, linewidth=DEFAULT_LINEWIDTH, color=color, 
               linestyle=style, alpha=alpha)
        ax.set_title(f"Latent Waveform (Ch {channel})", fontsize=8)
        ax.set_ylabel("Amplitude", fontsize=6)
        ax.set_xlabel("Latent Pixel Index", fontsize=6)
        ax.set_ylim(0 if normalize else None, 1 if normalize else None)
        
        if show_stats:
            self._add_statistics_overlay(ax, signal, label_color)
        
        if detect_peaks:
            self._detect_and_mark_peaks(ax, signal, peak_threshold, color)
            ax.legend(loc="upper left", fontsize=6, frameon=False, labelcolor=label_color)

    def _plot_spectrum(self, ax: plt.Axes, raw_signal: np.ndarray, channel: int,
                       color: str, log_scale: bool, show_stats: bool, 
                       detect_peaks: bool, peak_threshold: float, label_color: str) -> None:
        """Plot frequency spectrum visualization."""
        if len(raw_signal) == 0:
            ax.set_title("Latent Spectrum (Empty)", fontsize=8)
            ax.text(0.5, 0.5, "No data", ha='center', va='center', 
                   transform=ax.transAxes, color=label_color)
            return
        
        spectrum = np.fft.rfft(raw_signal)
        freq = np.fft.rfftfreq(len(raw_signal))
        magnitude = np.abs(spectrum)
        
        if log_scale:
            magnitude_db = 20 * np.log10(magnitude + LOG_EPSILON)
            ax.plot(freq, magnitude_db, color=color)
            ax.set_ylabel("Magnitude (dB)", fontsize=6)
        else:
            ax.plot(freq, magnitude, color=color)
            ax.set_ylabel("Magnitude", fontsize=6)
        
        ax.set_title(f"Latent Spectrum (Ch {channel})", fontsize=8)
        ax.set_xlabel("Normalized Frequency", fontsize=6)
        ax.set_xbound(0, freq.max())
        
        if show_stats:
            self._add_statistics_overlay(ax, magnitude, label_color)
        
        if detect_peaks:
            self._detect_and_mark_peaks(ax, magnitude, peak_threshold, color)
            ax.legend(loc="upper left", fontsize=6, frameon=False, labelcolor=label_color)

    def _plot_rgb_split(self, ax: plt.Axes, data: torch.Tensor, 
                        rgb_colors: List[str], normalize: bool, 
                        style: str, alpha: float, label_color: str) -> None:
        """Plot RGB channel split visualization."""
        c = data.shape[0]
        labels = ["R (Ch 0)", "G (Ch 1)", "B (Ch 2)"]
        
        for i in range(min(3, c)):
            signal = self._get_1d_signal(data[i], normalize)
            ax.plot(signal, linewidth=DEFAULT_LINEWIDTH, color=rgb_colors[i], 
                   label=labels[i], linestyle=style, alpha=alpha)
        
        ax.set_title("Latent RGB Channel Split", fontsize=8)
        ax.set_ylabel("Amplitude", fontsize=6)
        ax.set_xlabel("Latent Pixel Index", fontsize=6)
        ax.set_ylim(0 if normalize else None, 1 if normalize else None)
        ax.legend(loc="upper right", fontsize=6, frameon=False, labelcolor=label_color)

    def _plot_heatmap(self, ax: plt.Axes, data: torch.Tensor, channel: int,
                      colormap: str) -> None:
        """Plot 2D heatmap of spatial structure."""
        data_2d = data[channel].detach().cpu().numpy()
        im = ax.imshow(data_2d, cmap=colormap, aspect='auto', interpolation='nearest')
        ax.set_title(f"Latent Heatmap (Ch {channel})", fontsize=8)
        ax.set_xlabel("Width", fontsize=6)
        ax.set_ylabel("Height", fontsize=6)
        plt.colorbar(im, ax=ax, fraction=COLORBAR_FRACTION, pad=COLORBAR_PAD)

    def _plot_histogram(self, ax: plt.Axes, data: torch.Tensor, channel: int,
                        color: str, log_scale: bool, show_stats: bool, 
                        label_color: str) -> None:
        """Plot histogram of value distribution."""
        values = data[channel].detach().cpu().numpy().flatten()
        
        ax.hist(values, bins=50, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.set_title(f"Latent Histogram (Ch {channel})", fontsize=8)
        ax.set_xlabel("Value", fontsize=6)
        ax.set_ylabel("Frequency", fontsize=6)
        
        if log_scale:
            ax.set_yscale('log')
        
        if show_stats:
            self._add_statistics_overlay(ax, values, label_color)

    def _plot_statistics(self, ax: plt.Axes, data: torch.Tensor, 
                         label_color: str, bg_color: str) -> None:
        """Plot comprehensive statistics visualization."""
        c = data.shape[0]
        channels = range(min(c, 16))  # Limit to first 16 channels
        
        means = [data[i].mean().item() for i in channels]
        stds = [data[i].std().item() for i in channels]
        mins = [data[i].min().item() for i in channels]
        maxs = [data[i].max().item() for i in channels]
        
        x = list(channels)
        ax.plot(x, means, 'o-', label='Mean', color='#00E6E6')
        ax.fill_between(x, 
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        alpha=0.3, color='#00E6E6')
        ax.plot(x, mins, 's--', label='Min', color='#FF3333', markersize=3)
        ax.plot(x, maxs, '^--', label='Max', color='#00FF8C', markersize=3)
        
        ax.set_title("Channel Statistics Overview", fontsize=8)
        ax.set_xlabel("Channel", fontsize=6)
        ax.set_ylabel("Value", fontsize=6)
        ax.legend(loc="best", fontsize=6, frameon=False, labelcolor=label_color)
        ax.grid(True, alpha=0.3)

    def _plot_multi_channel(self, ax: plt.Axes, data: torch.Tensor, 
                           num_channels: int, normalize: bool, 
                           style: str, alpha: float, label_color: str) -> None:
        """Plot multiple channels overlaid."""
        c = data.shape[0]
        colors = plt.cm.rainbow(np.linspace(0, 1, min(num_channels, c)))
        
        for i in range(min(num_channels, c)):
            signal = self._get_1d_signal(data[i], normalize)
            ax.plot(signal, linewidth=DEFAULT_LINEWIDTH, 
                   color=colors[i], label=f"Ch {i}", 
                   linestyle=style, alpha=alpha)
        
        ax.set_title(f"Multi-Channel Overlay ({min(num_channels, c)} channels)", fontsize=8)
        ax.set_ylabel("Amplitude", fontsize=6)
        ax.set_xlabel("Latent Pixel Index", fontsize=6)
        
        if min(num_channels, c) <= 8:  # Only show legend if not too crowded
            ax.legend(loc="upper right", fontsize=5, frameon=False, 
                     labelcolor=label_color, ncol=2)

    def _plot_phase(self, ax: plt.Axes, raw_signal: np.ndarray, channel: int,
                    color: str, show_stats: bool, label_color: str) -> None:
        """Plot phase spectrum visualization."""
        if len(raw_signal) == 0:
            ax.set_title("Phase Spectrum (Empty)", fontsize=8)
            ax.text(0.5, 0.5, "No data", ha='center', va='center',
                   transform=ax.transAxes, color=label_color)
            return
        
        spectrum = np.fft.rfft(raw_signal)
        freq = np.fft.rfftfreq(len(raw_signal))
        phase = np.angle(spectrum)
        
        ax.plot(freq, phase, color=color)
        ax.set_title(f"Phase Spectrum (Ch {channel})", fontsize=8)
        ax.set_ylabel("Phase (radians)", fontsize=6)
        ax.set_xlabel("Normalized Frequency", fontsize=6)
        ax.set_xbound(0, freq.max())
        
        if show_stats:
            self._add_statistics_overlay(ax, phase, label_color)

    def _plot_difference(self, ax: plt.Axes, data1: torch.Tensor, 
                        data2: torch.Tensor, channel: int, 
                        color: str, normalize: bool, show_stats: bool,
                        label_color: str) -> None:
        """Plot difference between two latents."""
        signal1 = self._get_1d_signal(data1[channel], normalize)
        signal2 = self._get_1d_signal(data2[channel], normalize)
        
        diff = signal1 - signal2
        
        ax.plot(diff, linewidth=DEFAULT_LINEWIDTH, color=color)
        ax.axhline(y=0, color=label_color, linestyle='--', alpha=0.5, linewidth=0.5)
        ax.set_title(f"Latent Difference (Ch {channel})", fontsize=8)
        ax.set_ylabel("Difference", fontsize=6)
        ax.set_xlabel("Latent Pixel Index", fontsize=6)
        
        if show_stats:
            self._add_statistics_overlay(ax, diff, label_color)

    def visualize(self, latent: Dict[str, torch.Tensor], mode: str, channel: int, 
                  batch_index: int, normalize: bool, width: int, height: int, 
                  grid: bool, bg_color: str, waveform_color: str, spectrum_color: str,
                  rgb_r_color: str, rgb_g_color: str, rgb_b_color: str,
                  axis_label_color: str, grid_color: str, all_modes: bool,
                  log_scale: bool, show_stats: bool, detect_peaks: bool,
                  peak_threshold: float, line_style: str, line_alpha: float,
                  multi_channel_count: int, colormap: str, 
                  latent_compare: Optional[Dict[str, torch.Tensor]] = None) -> Tuple[torch.Tensor]:
        """
        Main visualization function with all enhanced features.
        """
        # Validate colors
        bg_color = self.validate_color(bg_color, "#0D0D1A")
        waveform_color = self.validate_color(waveform_color, "#00E6E6")
        spectrum_color = self.validate_color(spectrum_color, "#FF00A2")
        rgb_r_color = self.validate_color(rgb_r_color, "#FF3333")
        rgb_g_color = self.validate_color(rgb_g_color, "#00FF8C")
        rgb_b_color = self.validate_color(rgb_b_color, "#3399FF")
        axis_label_color = self.validate_color(axis_label_color, "#A0A0B0")
        grid_color = self.validate_color(grid_color, "#303040")
        
        # Extract latent tensor
        x = latent["samples"]
        b, c, h, w = x.shape
        
        # Validate batch index
        batch_idx = min(max(0, batch_index), b - 1)
        
        # Handle empty latent
        if c == 0:
            print("[ACE_LatentVisualizer] Error: Latent has 0 channels.")
            blank = torch.zeros((1, height, width, 3), dtype=torch.float32)
            return (blank,)
        
        # Clamp channel index
        channel = min(max(0, channel), c - 1)
        
        # Get selected batch
        x_batch = x[batch_idx]
        
        # Determine modes to show
        if all_modes:
            modes_to_show = ["waveform", "spectrum", "rgb_split", "heatmap", 
                           "histogram", "statistics", "multi_channel", "phase"]
            if latent_compare is not None:
                modes_to_show.append("difference")
        else:
            modes_to_show = [mode]
        
        # Filter invalid modes
        if c < 3 and "rgb_split" in modes_to_show:
            modes_to_show.remove("rgb_split")
            if mode == "rgb_split":
                print("[ACE_LatentVisualizer] RGB Split requires ≥3 channels. Using waveform.")
                modes_to_show = ["waveform"]
        
        if mode == "difference" and latent_compare is None:
            print("[ACE_LatentVisualizer] Difference mode requires latent_compare input.")
            modes_to_show = ["waveform"]
        
        if not modes_to_show:
            modes_to_show = ["waveform"]
        
        num_plots = len(modes_to_show)
        
        # Create figure
        fig, axes = plt.subplots(num_plots, 1, 
                                 figsize=(width / DEFAULT_DPI, height / DEFAULT_DPI), 
                                 dpi=DEFAULT_DPI)
        fig.patch.set_facecolor(bg_color)
        
        if num_plots == 1:
            axes = [axes]
        
        # Get line style
        ls = self._get_line_style(line_style)
        
        # Plot each mode
        for idx, m in enumerate(modes_to_show):
            ax = axes[idx]
            self._setup_axis_style(ax, bg_color, axis_label_color, grid, grid_color)
            
            try:
                if m == "waveform":
                    signal = self._get_1d_signal(x_batch[channel], normalize)
                    self._plot_waveform(ax, signal, channel, waveform_color, ls, 
                                      line_alpha, normalize, show_stats, detect_peaks,
                                      peak_threshold, axis_label_color)
                
                elif m == "spectrum":
                    raw_signal = x_batch[channel].detach().cpu().numpy().flatten()
                    self._plot_spectrum(ax, raw_signal, channel, spectrum_color,
                                      log_scale, show_stats, detect_peaks, 
                                      peak_threshold, axis_label_color)
                
                elif m == "rgb_split":
                    self._plot_rgb_split(ax, x_batch, [rgb_r_color, rgb_g_color, rgb_b_color],
                                       normalize, ls, line_alpha, axis_label_color)
                
                elif m == "heatmap":
                    self._plot_heatmap(ax, x_batch, channel, colormap)
                
                elif m == "histogram":
                    self._plot_histogram(ax, x_batch, channel, waveform_color,
                                       log_scale, show_stats, axis_label_color)
                
                elif m == "statistics":
                    self._plot_statistics(ax, x_batch, axis_label_color, bg_color)
                
                elif m == "multi_channel":
                    self._plot_multi_channel(ax, x_batch, multi_channel_count,
                                           normalize, ls, line_alpha, axis_label_color)
                
                elif m == "phase":
                    raw_signal = x_batch[channel].detach().cpu().numpy().flatten()
                    self._plot_phase(ax, raw_signal, channel, spectrum_color,
                                   show_stats, axis_label_color)
                
                elif m == "difference":
                    if latent_compare is not None:
                        x_compare = latent_compare["samples"]
                        x_compare_batch = x_compare[min(batch_idx, x_compare.shape[0] - 1)]
                        self._plot_difference(ax, x_batch, x_compare_batch, channel,
                                           waveform_color, normalize, show_stats,
                                           axis_label_color)
            
            except Exception as e:
                print(f"[ACE_LatentVisualizer] Error plotting {m}: {e}")
                ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center',
                       transform=ax.transAxes, color=axis_label_color)
        
        plt.tight_layout(pad=0.5)
        
        # Save to buffer with proper cleanup
        buf = io.BytesIO()
        try:
            plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            buf.seek(0)
            
            # Convert to image
            image = Image.open(buf).convert("RGB")
            image = image.resize((width, height), Image.LANCZOS)
            
            # Convert to tensor
            image_tensor = torch.from_numpy(np.array(image)).float() / 255.0
            image_tensor = image_tensor.unsqueeze(0)
            
            return (image_tensor,)
        
        finally:
            buf.close()  # Ensure buffer is always closed


# Node registration
NODE_CLASS_MAPPINGS = {
    "ACE_LatentVisualizer": ACE_LatentVisualizer_v04,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ACE_LatentVisualizer": "ACE Latent Visualizer v0.4 (Enhanced)"
}