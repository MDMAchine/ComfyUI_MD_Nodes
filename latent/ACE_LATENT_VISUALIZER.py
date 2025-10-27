# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
# ████ MD_Nodes/ACELatentVisualizer – Latent Tensor Visualization v0.4.0 ████▓▒░
# © 2025 MDMAchine
# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
# ░▒▓ ORIGIN & DEV:
#   • Cast into the void by: MDMAchine (The latent abyss explorer)
#   • Enhanced by: devstral, Gemini, Claude, c0ffymachyne
#   • License: Apache 2.0 — Sharing digital secrets responsibly
#   • Original source (if applicable): N/A

# ░▒▓ DESCRIPTION:
#   Advanced latent tensor visualization node for ComfyUI, optimized for Ace-Step.
#   Offers multiple modes (waveform, spectrum, heatmap, histogram, etc.) to inspect
#   latent data, aiding debugging, analysis, and creative exploration.

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
#   ✓ 11 colormap options (viridis, plasma, inferno, etc.) for heatmap visualization
#   ✓ Memory leak prevention with proper resource cleanup

# ░▒▓ CHANGELOG:
#   - v0.4.0 (Current Release - Major Enhancement):
#       • NEW MODES: Heatmap, Histogram, Statistics, Multi-channel, Difference, Phase
#       • ADDED: Batch selection, Peak detection, Log scales, 11 colormaps
#       • ADDED: Line style & alpha customization, Stats overlay
#       • FIXED: Memory leaks with proper buffer cleanup and resource management
#       • ADDED: Comprehensive error handling and input validation
#   - v0.3.1 (Refinement):
#       • ENHANCED: Color calibration, UI tooltips, and stability fixes
#   - v0.2 (Expansion):
#       • ADDED: Spectrum mode (dB) and RGB channel splitting
#   - v0.1 (Initial Release):
#       • ADDED: Basic waveform mode

# ░▒▓ CONFIGURATION:
#   → Primary Use: Latent tensor visual inspection and debugging workflows.
#   → Secondary Use: Comparative analysis between model architectures or training stages.
#   → Edge Use: Artistic exploration of AI latent spaces for creative projects.

# ░▒▓ WARNING:
#   This node may trigger:
#   ▓▒░ Staring at heatmaps until you see faces in the static, like an Rorschach test for l33t haxxors.
#   ▓▒░ A crippling addiction to 'difference' mode, checking every node for 0.001% variance.
#   ▓▒░ Flashbacks to staring at ProTracker's sample editor, wondering why your latent looks like an 8-bit kick drum.
#   ▓▒░ The unshakeable belief that you can *see* the 'soul' of the AI in the phase spectrum.
#   Consult your nearest demoscene vet if hallucinations persist.
# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄


# =================================================================================
# == Standard Library Imports                                                  ==
# =================================================================================
import io
import logging
import re
import traceback
import secrets # Though IS_CHANGED is not used, keep for potential future needs

# =================================================================================
# == Third-Party Imports                                                       ==
# =================================================================================
import torch
import numpy as np
from PIL import Image
from scipy import signal # Used for peak detection

# Matplotlib setup: Set backend BEFORE importing pyplot
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# =================================================================================
# == ComfyUI Core Modules                                                      ==
# =================================================================================
# (None needed directly, interaction is via LATENT type)

# =================================================================================
# == Local Project Imports                                                     ==
# =================================================================================
# (None needed)

# =================================================================================
# == Helper Classes & Dependencies                                             ==
# =================================================================================

# --- Constants ---
DEFAULT_DPI = 100
MIN_SIGNAL_VARIANCE = 1e-6 # Threshold to consider a signal effectively flat
LOG_EPSILON = 1e-8         # Small value to avoid log(0)
FLAT_SIGNAL_OFFSET = 0.5   # Value to plot if signal is flat (after potential normalization)
DEFAULT_LINEWIDTH = 0.75   # Default line width for plots
GRID_LINEWIDTH = 0.3       # Grid line width
GRID_ALPHA = 0.5           # Grid line transparency
COLORBAR_FRACTION = 0.046  # Colorbar size relative to axes
COLORBAR_PAD = 0.04        # Colorbar padding relative to axes
PEAK_MIN_DISTANCE = 10     # Minimum sample distance between detected peaks

# =================================================================================
# == Core Node Class                                                           ==
# =================================================================================

class ACELatentVisualizer: # Renamed class, removed version suffix
    """
    MD: ACE Latent Visualizer

    Provides multiple visualization modes (waveform, spectrum, heatmap, etc.)
    for inspecting latent tensors, aiding in debugging and analysis. Includes
    options for batch/channel selection, normalization, peak detection, and styling.
    """
    CATEGORY = "MD_Nodes/Visualization" # Updated category prefix
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "visualize"

    @classmethod
    def INPUT_TYPES(cls):
        """Define all input parameters with tooltips."""
        return {
            "required": {
                "latent": ("LATENT", {"tooltip": "LATENT INPUT\n- The latent tensor dictionary to visualize."}),
                "mode": (["waveform", "spectrum", "rgb_split", "heatmap", "histogram",
                          "statistics", "multi_channel", "phase", "difference"],
                         {"default": "waveform",
                          "tooltip": (
                              "VISUALIZATION MODE\n"
                              "- Select the type of plot to generate.\n"
                              "- 'waveform': Amplitude over spatial dimension.\n"
                              "- 'spectrum': Frequency magnitude (FFT).\n"
                              "- 'rgb_split': Overlay first 3 channels (requires C>=3).\n"
                              "- 'heatmap': 2D representation of channel data.\n"
                              "- 'histogram': Distribution of values in the channel.\n"
                              "- 'statistics': Mean, Std, Min, Max across channels.\n"
                              "- 'multi_channel': Overlay multiple channels (up to 16).\n"
                              "- 'phase': Phase spectrum (FFT).\n"
                              "- 'difference': Compares primary latent with 'latent_compare'."
                          )}),
                "channel": ("INT", {"default": 0, "min": 0,
                                   "tooltip": (
                                       "CHANNEL INDEX\n"
                                       "- The primary channel index (0-based) to focus on for single-channel modes (waveform, spectrum, heatmap, histogram, phase, difference).\n"
                                       "- Will be clamped to the available channel range."
                                   )}),
                "batch_index": ("INT", {"default": 0, "min": 0,
                                      "tooltip": (
                                          "BATCH INDEX\n"
                                          "- The batch item index (0-based) to visualize if the latent contains multiple items.\n"
                                          "- Will be clamped to the available batch range."
                                      )}),
                "normalize": ("BOOLEAN", {"default": True,
                                        "tooltip": (
                                            "NORMALIZE AMPLITUDE\n"
                                            "- True: Rescales the signal amplitude to fit the full [0, 1] vertical range for better visibility (waveform, rgb_split, multi_channel).\n"
                                            "- False: Uses the original value range."
                                        )}),
                "width": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64, "tooltip": "Output image width in pixels."}),
                "height": ("INT", {"default": 256, "min": 64, "max": 2048, "step": 64, "tooltip": "Output image height in pixels."}),
                "grid": ("BOOLEAN", {"default": True, "tooltip": "Display grid lines on the plot."}),

                # --- Color Controls ---
                "bg_color": ("STRING", {"default": "#0D0D1A", "tooltip": "Background color (hex)."}),
                "waveform_color": ("STRING", {"default": "#00E6E6", "tooltip": "Color for waveform, histogram, difference (hex)."}),
                "spectrum_color": ("STRING", {"default": "#FF00A2", "tooltip": "Color for spectrum and phase plots (hex)."}),
                "rgb_r_color": ("STRING", {"default": "#FF3333", "tooltip": "Color for Red channel (Ch 0) in rgb_split (hex)."}),
                "rgb_g_color": ("STRING", {"default": "#00FF8C", "tooltip": "Color for Green channel (Ch 1) in rgb_split (hex)."}),
                "rgb_b_color": ("STRING", {"default": "#3399FF", "tooltip": "Color for Blue channel (Ch 2) in rgb_split (hex)."}),
                "axis_label_color": ("STRING", {"default": "#A0A0B0", "tooltip": "Color for axis labels, ticks, and titles (hex)."}),
                "grid_color": ("STRING", {"default": "#303040", "tooltip": "Color for grid lines (hex)."}),

                # --- Advanced Options ---
                "all_modes": ("BOOLEAN", {"default": False,
                                        "tooltip": (
                                            "SHOW ALL MODES\n"
                                            "- True: Renders all applicable visualization modes stacked vertically, ignoring the 'mode' selection.\n"
                                            "- False: Renders only the selected 'mode'."
                                        )}),
                "log_scale": ("BOOLEAN", {"default": False,
                                        "tooltip": (
                                            "LOGARITHMIC SCALE (Y-AXIS)\n"
                                            "- True: Uses a log scale for the Y-axis.\n"
                                            "- Effective for 'spectrum' and 'histogram' modes to see smaller values."
                                        )}),
                "show_stats": ("BOOLEAN", {"default": False,
                                         "tooltip": (
                                             "OVERLAY STATISTICS\n"
                                             "- True: Displays Mean, Std Dev, Min, Max values as text overlay on relevant plots (waveform, spectrum, histogram, phase, difference).\n"
                                             "- False: No statistics overlay."
                                         )}),
                "detect_peaks": ("BOOLEAN", {"default": False,
                                           "tooltip": (
                                               "DETECT PEAKS\n"
                                               "- True: Finds and marks significant peaks on 'waveform' and 'spectrum' plots.\n"
                                               "- False: No peak detection."
                                           )}),
                "peak_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                                             "tooltip": (
                                                 "PEAK DETECTION THRESHOLD\n"
                                                 "- Minimum relative height (0.0-1.0) for a point to be considered a peak.\n"
                                                 "- Only used if 'detect_peaks' is True.\n"
                                                 "- 0.5 means peaks must be above the midpoint between min and max."
                                             )}),
                "line_style": (["solid", "dashed", "dotted", "dashdot"], {"default": "solid",
                                                                           "tooltip": "Line style for waveform, rgb_split, multi_channel plots."}),
                "line_alpha": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 1.0, "step": 0.1,
                                         "tooltip": "Line transparency (alpha) for waveform plots (1.0 = opaque)."}),
                "multi_channel_count": ("INT", {"default": 3, "min": 1, "max": 16,
                                                "tooltip": (
                                                    "MULTI-CHANNEL COUNT\n"
                                                    "- Number of channels (starting from 0) to overlay in 'multi_channel' mode.\n"
                                                    "- Limited to a maximum of 16 for clarity."
                                                )}),
                "colormap": (["viridis", "plasma", "inferno", "magma", "cividis", "twilight",
                              "turbo", "hot", "cool", "spring", "winter"], {"default": "viridis",
                                                                            "tooltip": "Colormap to use for 'heatmap' mode."}),
            },
            "optional": {
                "latent_compare": ("LATENT", {"tooltip": (
                                                "LATENT COMPARE INPUT\n"
                                                "- Optional second latent tensor for 'difference' mode.\n"
                                                "- The difference (latent - latent_compare) will be plotted."
                                              )}),
            }
        }

    @staticmethod
    def validate_color(color_str, default="#FFFFFF"):
        """
        Validates a hex color string (#RGB or #RRGGBB).

        Args:
            color_str: The input color string.
            default: The default color to return if validation fails.

        Returns:
            The validated hex color string or the default string.
        """
        if not isinstance(color_str, str):
            return default

        color_str = color_str.strip()
        # Basic check for hex format
        if re.match(r'^#(?:[0-9a-fA-F]{3}){1,2}$', color_str):
             # Expand #RGB to #RRGGBB if necessary (matplotlib prefers 6 digits)
             if len(color_str) == 4:
                  return f"#{color_str[1]*2}{color_str[2]*2}{color_str[3]*2}"
             return color_str
        else:
             logging.warning(f"[ACELatentVisualizer] Invalid color '{color_str}', using default '{default}'.")
             return default


    def _setup_axis_style(self, ax, bg_color, label_color, grid, grid_color):
        """
        Applies consistent styling to a Matplotlib axis.

        Args:
            ax: The Matplotlib Axes object.
            bg_color: Background color string.
            label_color: Axis label/tick color string.
            grid: Boolean, whether to show the grid.
            grid_color: Grid color string.
        """
        try:
            ax.set_facecolor(bg_color)
            ax.tick_params(axis='x', colors=label_color, labelsize=6)
            ax.tick_params(axis='y', colors=label_color, labelsize=6)
            ax.xaxis.label.set_color(label_color)
            ax.yaxis.label.set_color(label_color)
            ax.title.set_color(label_color)

            # Ensure spines (plot borders) are visible and colored
            for spine in ax.spines.values():
                 spine.set_edgecolor(label_color)
                 spine.set_linewidth(0.5)
                 spine.set_alpha(0.7)


            if grid:
                ax.grid(True, which='both', linestyle=':', # Changed to dotted for less clutter
                       linewidth=GRID_LINEWIDTH, alpha=GRID_ALPHA, color=grid_color)
            else:
                 # When grid is off, remove ticks and labels for a cleaner look
                 ax.set_xticks([])
                 ax.set_yticks([])
                 ax.set_xlabel("")
                 ax.set_ylabel("")
                 # Optionally hide spines too when grid is off
                 # for spine in ax.spines.values(): spine.set_visible(False)
        except Exception as e:
            logging.error(f"[ACELatentVisualizer] Error setting up axis style: {e}")


    def _get_1d_signal(self, data, normalize):
        """
        Extracts, flattens, and optionally normalizes 1D signal data from a tensor slice.

        Args:
            data: A torch.Tensor slice representing one channel (e.g., shape [H, W]).
            normalize: Boolean flag for normalization.

        Returns:
            A 1D NumPy array, normalized to [0, 1] if requested, or centered if flat.
        """
        if not isinstance(data, torch.Tensor):
             logging.warning("_get_1d_signal expected Tensor, got {type(data)}. Returning empty.")
             return np.array([])

        signal_np = data.detach().cpu().numpy().flatten()

        # Handle empty or near-flat signals
        if signal_np.size == 0:
             return np.array([])
        if np.ptp(signal_np) < MIN_SIGNAL_VARIANCE: # Check peak-to-peak difference
            logging.debug("Signal is flat or near-flat.")
            # Return a flat line at the offset value if not normalizing, else return flat zeros/ones after norm
            return np.full_like(signal_np, FLAT_SIGNAL_OFFSET) if not normalize else (np.zeros_like(signal_np) if signal_np.mean() < 0.5 else np.ones_like(signal_np))


        if normalize:
            min_val, max_val = np.min(signal_np), np.max(signal_np)
            range_val = max_val - min_val
            if range_val > LOG_EPSILON: # Avoid division by zero/tiny numbers
                signal_np = (signal_np - min_val) / range_val
            else:
                 # If range is tiny after variance check (should be rare), return centered flat line
                 signal_np = np.full_like(signal_np, FLAT_SIGNAL_OFFSET)

        return signal_np

    def _add_statistics_overlay(self, ax, data, color):
        """
        Adds Mean, Std Dev, Min, Max text overlay to the plot axes.

        Args:
            ax: The Matplotlib Axes object.
            data: The 1D NumPy array containing the data.
            color: Text color string.
        """
        if data.size == 0: return # Don't add stats for empty data

        try:
            stats_text = (f"μ={np.mean(data):.3f}\n"
                          f"σ={np.std(data):.3f}\n"
                          f"min={np.min(data):.3f}\n"
                          f"max={np.max(data):.3f}")

            # Place text at top-right corner
            ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                    verticalalignment='top', horizontalalignment='right',
                    fontsize=6, color=color,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.6)) # Added padding
        except Exception as e:
            logging.error(f"Failed to add statistics overlay: {e}")


    def _detect_and_mark_peaks(self, ax, signal_data, threshold, color):
        """
        Detects peaks in a 1D signal using scipy.signal.find_peaks and marks them.

        Args:
            ax: The Matplotlib Axes object.
            signal_data: The 1D NumPy array signal.
            threshold: Relative peak height threshold (0.0-1.0).
            color: Color string for the peak markers.
        """
        if signal_data.size < 3: return # find_peaks needs at least 3 points

        try:
            # Calculate height threshold based on signal range
            min_val, max_val = np.min(signal_data), np.max(signal_data)
            peak_height_abs = threshold * (max_val - min_val) + min_val

            peaks, _ = signal.find_peaks(signal_data, height=peak_height_abs,
                                         distance=PEAK_MIN_DISTANCE)

            if len(peaks) > 0:
                ax.plot(peaks, signal_data[peaks], 'x', color=color, markersize=5, # Slightly smaller marker
                       markeredgewidth=1.0, label=f'{len(peaks)} peaks')
                logging.debug(f"Detected {len(peaks)} peaks above threshold {threshold:.2f}")

        except Exception as e:
            logging.error(f"Peak detection failed: {e}")

    def _get_line_style(self, style_name):
        """Converts style name string to Matplotlib linestyle string."""
        styles = {
            "solid": "-",
            "dashed": "--",
            "dotted": ":",
            "dashdot": "-."
        }
        return styles.get(style_name, "-") # Default to solid

    def _plot_waveform(self, ax, signal_data, channel, color, style, alpha, normalize,
                       show_stats, detect_peaks, peak_threshold, label_color):
        """Helper function to plot the waveform mode."""
        if signal_data.size == 0:
             ax.text(0.5, 0.5, "No Waveform Data", ha='center', va='center', transform=ax.transAxes, color=label_color)
             ax.set_title(f"Latent Waveform (Ch {channel}) - Empty", fontsize=8)
             return

        ax.plot(signal_data, linewidth=DEFAULT_LINEWIDTH, color=color,
                linestyle=style, alpha=alpha)
        ax.set_title(f"Latent Waveform (Ch {channel})", fontsize=8)
        ax.set_ylabel("Amplitude", fontsize=6)
        ax.set_xlabel("Spatial Index", fontsize=6)
        # Set Y limits based on normalization
        ax.set_ylim(0 if normalize else np.min(signal_data) - 0.1 * np.ptp(signal_data),
                    1 if normalize else np.max(signal_data) + 0.1 * np.ptp(signal_data))
        ax.set_xlim(0, len(signal_data) - 1 if len(signal_data) > 1 else 1)

        if show_stats:
            self._add_statistics_overlay(ax, signal_data, label_color)

        if detect_peaks:
            self._detect_and_mark_peaks(ax, signal_data, peak_threshold, color)
            # Only add legend if peaks were detected and plotted
            if ax.get_legend_handles_labels()[0]: # Check if legend items exist
                 ax.legend(loc="upper left", fontsize=6, frameon=False, labelcolor=label_color)


    def _plot_spectrum(self, ax, raw_signal, channel, color, log_scale, show_stats,
                       detect_peaks, peak_threshold, label_color):
        """Helper function to plot the spectrum mode."""
        if raw_signal.size < 2: # FFT needs at least 2 points
            ax.text(0.5, 0.5, "Not Enough Data for FFT", ha='center', va='center', transform=ax.transAxes, color=label_color)
            ax.set_title(f"Latent Spectrum (Ch {channel}) - Error", fontsize=8)
            return

        try:
            spectrum = np.fft.rfft(raw_signal)
            freqs = np.fft.rfftfreq(len(raw_signal))
            magnitude = np.abs(spectrum)

            plot_data = magnitude
            y_label = "Magnitude"
            if log_scale:
                # Calculate dB, handling potential zeros safely
                magnitude_db = 20 * np.log10(np.maximum(magnitude, LOG_EPSILON))
                plot_data = magnitude_db
                y_label = "Magnitude (dB)"

            ax.plot(freqs, plot_data, color=color, linewidth=DEFAULT_LINEWIDTH)
            ax.set_ylabel(y_label, fontsize=6)
            ax.set_title(f"Latent Spectrum (Ch {channel})", fontsize=8)
            ax.set_xlabel("Normalized Frequency", fontsize=6)
            ax.set_xlim(0, freqs.max()) # Use xbound for setting limits if needed

            if show_stats:
                self._add_statistics_overlay(ax, plot_data, label_color) # Stats on dB or linear magnitude

            if detect_peaks:
                self._detect_and_mark_peaks(ax, plot_data, peak_threshold, color) # Peaks on dB or linear
                if ax.get_legend_handles_labels()[0]:
                     ax.legend(loc="upper right", fontsize=6, frameon=False, labelcolor=label_color)

        except Exception as e:
             logging.error(f"Spectrum plot failed: {e}")
             ax.text(0.5, 0.5, "Spectrum Error", ha='center', va='center', transform=ax.transAxes, color=label_color)
             ax.set_title(f"Latent Spectrum (Ch {channel}) - Error", fontsize=8)


    def _plot_rgb_split(self, ax, data_chw, rgb_colors, normalize, style, alpha, label_color):
        """Helper function to plot the rgb_split mode."""
        num_channels = data_chw.shape[0]
        if num_channels < 3:
             ax.text(0.5, 0.5, "Requires >= 3 Channels", ha='center', va='center', transform=ax.transAxes, color=label_color)
             ax.set_title("Latent RGB Split - N/A", fontsize=8)
             return

        labels = ["R (Ch 0)", "G (Ch 1)", "B (Ch 2)"]
        min_val_overall, max_val_overall = 0.0, 1.0 # Default for normalized view
        if not normalize:
             # Find overall min/max across first 3 channels for consistent Y-axis
             all_signals = [self._get_1d_signal(data_chw[i], False) for i in range(3)]
             valid_signals = [s for s in all_signals if s.size > 0]
             if valid_signals:
                  min_val_overall = min(np.min(s) for s in valid_signals)
                  max_val_overall = max(np.max(s) for s in valid_signals)
             else:
                  min_val_overall, max_val_overall = -1.0, 1.0 # Fallback


        for i in range(3):
            signal_data = self._get_1d_signal(data_chw[i], normalize)
            if signal_data.size > 0:
                 ax.plot(signal_data, linewidth=DEFAULT_LINEWIDTH, color=rgb_colors[i],
                        label=labels[i], linestyle=style, alpha=alpha)

        ax.set_title("Latent RGB Channel Split", fontsize=8)
        ax.set_ylabel("Amplitude", fontsize=6)
        ax.set_xlabel("Spatial Index", fontsize=6)
        # Set Y limits based on normalization or calculated range
        y_min = 0 if normalize else min_val_overall - 0.1 * (max_val_overall - min_val_overall)
        y_max = 1 if normalize else max_val_overall + 0.1 * (max_val_overall - min_val_overall)
        ax.set_ylim(y_min, y_max)
        # Ensure X limits are sensible even for empty signals
        data_len = signal_data.size if 'signal_data' in locals() and signal_data.size > 0 else 1
        ax.set_xlim(0, data_len - 1 if data_len > 1 else 1)

        ax.legend(loc="upper right", fontsize=6, frameon=False, labelcolor=label_color)

    def _plot_heatmap(self, ax, data_chw, channel, colormap):
        """Helper function to plot the heatmap mode."""
        try:
            data_2d = data_chw[channel].detach().cpu().numpy()
            if data_2d.size == 0: raise ValueError("Channel data is empty.")

            im = ax.imshow(data_2d, cmap=colormap, aspect='auto', interpolation='nearest')
            ax.set_title(f"Latent Heatmap (Ch {channel})", fontsize=8)
            ax.set_xlabel("Width Index", fontsize=6)
            ax.set_ylabel("Height Index", fontsize=6)
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, fraction=COLORBAR_FRACTION, pad=COLORBAR_PAD)
            cbar.ax.tick_params(labelsize=5, colors=ax.xaxis.label.get_color()) # Style colorbar ticks

        except Exception as e:
             logging.error(f"Heatmap plot failed: {e}")
             ax.text(0.5, 0.5, "Heatmap Error", ha='center', va='center', transform=ax.transAxes, color=ax.xaxis.label.get_color())
             ax.set_title(f"Latent Heatmap (Ch {channel}) - Error", fontsize=8)


    def _plot_histogram(self, ax, data_chw, channel, color, log_scale, show_stats, label_color):
        """Helper function to plot the histogram mode."""
        try:
            values = data_chw[channel].detach().cpu().numpy().flatten()
            if values.size == 0: raise ValueError("Channel data is empty.")

            ax.hist(values, bins=50, color=color, alpha=0.75, edgecolor='black', linewidth=0.3, log=log_scale) # Use log param in hist
            ax.set_title(f"Latent Histogram (Ch {channel})", fontsize=8)
            ax.set_xlabel("Value", fontsize=6)
            ax.set_ylabel("Frequency" + (" (log scale)" if log_scale else ""), fontsize=6)

            if show_stats:
                self._add_statistics_overlay(ax, values, label_color)

        except Exception as e:
            logging.error(f"Histogram plot failed: {e}")
            ax.text(0.5, 0.5, "Histogram Error", ha='center', va='center', transform=ax.transAxes, color=label_color)
            ax.set_title(f"Latent Histogram (Ch {channel}) - Error", fontsize=8)


    def _plot_statistics(self, ax, data_chw, label_color, bg_color):
        """Helper function to plot the statistics mode."""
        try:
            num_channels = data_chw.shape[0]
            if num_channels == 0 or data_chw.numel() == 0: raise ValueError("Input tensor has no channels or data.")

            channels_to_plot = range(min(num_channels, 16)) # Limit plot to first 16 channels

            means, stds, mins, maxs = [], [], [], []
            for i in channels_to_plot:
                channel_data = data_chw[i].flatten()
                if channel_data.numel() > 0:
                     means.append(channel_data.mean().item())
                     stds.append(channel_data.std().item())
                     mins.append(channel_data.min().item())
                     maxs.append(channel_data.max().item())
                else: # Append NaN or default if channel is empty (shouldn't happen with check above)
                     means.append(np.nan); stds.append(np.nan); mins.append(np.nan); maxs.append(np.nan)


            x_ticks = list(channels_to_plot)
            ax.plot(x_ticks, means, 'o-', label='Mean', color='#00E6E6', markersize=3, linewidth=1)
            # Convert lists to numpy arrays for element-wise operations
            means_np, stds_np = np.array(means), np.array(stds)
            ax.fill_between(x_ticks, means_np - stds_np, means_np + stds_np, alpha=0.3, color='#00E6E6', label='Mean ± Std Dev')
            ax.plot(x_ticks, mins, 's--', label='Min', color='#FF3333', markersize=3, linewidth=0.8)
            ax.plot(x_ticks, maxs, '^--', label='Max', color='#00FF8C', markersize=3, linewidth=0.8)

            ax.set_title(f"Channel Statistics (First {len(channels_to_plot)} Ch.)", fontsize=8)
            ax.set_xlabel("Channel Index", fontsize=6)
            ax.set_ylabel("Value", fontsize=6)
            ax.legend(loc="best", fontsize=5, frameon=False, labelcolor=label_color) # Smaller font
            ax.grid(True, alpha=GRID_ALPHA, linestyle=':', linewidth=GRID_LINEWIDTH) # Use constants
            # Ensure x-axis shows integer channel numbers
            if len(x_ticks) > 1:
                ax.set_xticks(x_ticks)
                ax.set_xticklabels([str(i) for i in x_ticks])
            elif len(x_ticks) == 1:
                 ax.set_xticks([x_ticks[0]])
                 ax.set_xticklabels([str(x_ticks[0])])


        except Exception as e:
            logging.error(f"Statistics plot failed: {e}")
            ax.text(0.5, 0.5, "Statistics Error", ha='center', va='center', transform=ax.transAxes, color=label_color)
            ax.set_title("Channel Statistics - Error", fontsize=8)


    def _plot_multi_channel(self, ax, data_chw, num_channels_to_plot, normalize, style, alpha, label_color):
        """Helper function to plot the multi_channel mode."""
        try:
            num_available_channels = data_chw.shape[0]
            if num_available_channels == 0: raise ValueError("Input tensor has no channels.")

            channels_to_plot = range(min(num_channels_to_plot, num_available_channels, 16)) # Limit actual plotted channels
            colors = plt.cm.rainbow(np.linspace(0, 1, len(channels_to_plot)))

            min_val_overall, max_val_overall = 0.0, 1.0 # Default for normalized
            if not normalize:
                 all_signals = [self._get_1d_signal(data_chw[i], False) for i in channels_to_plot]
                 valid_signals = [s for s in all_signals if s.size > 0]
                 if valid_signals:
                      min_val_overall = min(np.min(s) for s in valid_signals)
                      max_val_overall = max(np.max(s) for s in valid_signals)
                 else:
                      min_val_overall, max_val_overall = -1.0, 1.0 # Fallback

            data_len = 0
            for i, ch_idx in enumerate(channels_to_plot):
                signal_data = self._get_1d_signal(data_chw[ch_idx], normalize)
                if signal_data.size > 0:
                     data_len = max(data_len, signal_data.size)
                     ax.plot(signal_data, linewidth=DEFAULT_LINEWIDTH,
                            color=colors[i], label=f"Ch {ch_idx}",
                            linestyle=style, alpha=alpha)

            ax.set_title(f"Multi-Channel Overlay ({len(channels_to_plot)} Ch.)", fontsize=8)
            ax.set_ylabel("Amplitude", fontsize=6)
            ax.set_xlabel("Spatial Index", fontsize=6)
            y_min = 0 if normalize else min_val_overall - 0.1 * (max_val_overall - min_val_overall)
            y_max = 1 if normalize else max_val_overall + 0.1 * (max_val_overall - min_val_overall)
            ax.set_ylim(y_min, y_max)
            ax.set_xlim(0, data_len - 1 if data_len > 1 else 1)


            if len(channels_to_plot) <= 8: # Only show legend if not too cluttered
                ax.legend(loc="upper right", fontsize=5, frameon=False,
                          labelcolor=label_color, ncol=2 if len(channels_to_plot) > 4 else 1)

        except Exception as e:
            logging.error(f"Multi-channel plot failed: {e}")
            ax.text(0.5, 0.5, "Multi-Channel Error", ha='center', va='center', transform=ax.transAxes, color=label_color)
            ax.set_title("Multi-Channel Overlay - Error", fontsize=8)


    def _plot_phase(self, ax, raw_signal, channel, color, show_stats, label_color):
        """Helper function to plot the phase spectrum mode."""
        if raw_signal.size < 2:
            ax.text(0.5, 0.5, "Not Enough Data for FFT", ha='center', va='center', transform=ax.transAxes, color=label_color)
            ax.set_title(f"Phase Spectrum (Ch {channel}) - Error", fontsize=8)
            return

        try:
            spectrum = np.fft.rfft(raw_signal)
            freqs = np.fft.rfftfreq(len(raw_signal))
            # Unwrap phase for continuity (optional but often clearer)
            phase = np.unwrap(np.angle(spectrum))

            ax.plot(freqs, phase, color=color, linewidth=DEFAULT_LINEWIDTH)
            ax.set_title(f"Phase Spectrum (Ch {channel})", fontsize=8)
            ax.set_ylabel("Phase (radians)", fontsize=6)
            ax.set_xlabel("Normalized Frequency", fontsize=6)
            ax.set_xlim(0, freqs.max())

            if show_stats:
                # Stats on phase might be less intuitive, but show range etc.
                self._add_statistics_overlay(ax, phase, label_color)

        except Exception as e:
            logging.error(f"Phase plot failed: {e}")
            ax.text(0.5, 0.5, "Phase Error", ha='center', va='center', transform=ax.transAxes, color=label_color)
            ax.set_title(f"Phase Spectrum (Ch {channel}) - Error", fontsize=8)


    def _plot_difference(self, ax, data1_chw, data2_chw, channel, color, normalize, show_stats, label_color, line_style, line_alpha):
        """Helper function to plot the difference mode."""
        try:
            if channel >= data1_chw.shape[0] or channel >= data2_chw.shape[0]:
                 raise IndexError("Channel index out of bounds for one or both latents.")

            # Get signals, ensure they are normalized *identically* if normalize=True
            # If not normalizing, they use their original scales.
            signal1 = self._get_1d_signal(data1_chw[channel], normalize)
            signal2 = self._get_1d_signal(data2_chw[channel], normalize)

            # Ensure signals have the same length for subtraction
            len1, len2 = len(signal1), len(signal2)
            if len1 == 0 or len2 == 0: raise ValueError("One or both signals are empty.")

            if len1 != len2:
                # Pad the shorter signal with zeros (or handle differently if needed)
                logging.warning(f"Difference mode: Latents have different lengths ({len1} vs {len2}). Padding shorter.")
                max_len = max(len1, len2)
                if len1 < max_len: signal1 = np.pad(signal1, (0, max_len - len1))
                if len2 < max_len: signal2 = np.pad(signal2, (0, max_len - len2))

            diff = signal1 - signal2

            ax.plot(diff, linewidth=DEFAULT_LINEWIDTH, color=color, linestyle=line_style, alpha=line_alpha)
            # Add a zero line for reference
            ax.axhline(y=0, color=label_color, linestyle='--', alpha=0.7, linewidth=0.5)
            ax.set_title(f"Latent Difference (Ch {channel})", fontsize=8)
            ax.set_ylabel("Difference", fontsize=6)
            ax.set_xlabel("Spatial Index", fontsize=6)
            # Auto-adjust Y limits based on difference range
            min_diff, max_diff = np.min(diff), np.max(diff)
            ax.set_ylim(min_diff - 0.1 * abs(min_diff), max_diff + 0.1 * abs(max_diff))
            ax.set_xlim(0, len(diff) - 1 if len(diff) > 1 else 1)


            if show_stats:
                self._add_statistics_overlay(ax, diff, label_color)

        except Exception as e:
             logging.error(f"Difference plot failed: {e}")
             ax.text(0.5, 0.5, "Difference Error", ha='center', va='center', transform=ax.transAxes, color=label_color)
             ax.set_title(f"Latent Difference (Ch {channel}) - Error", fontsize=8)


    def visualize(self, latent, mode, channel, batch_index, normalize, width, height, grid,
                  bg_color, waveform_color, spectrum_color, rgb_r_color, rgb_g_color, rgb_b_color,
                  axis_label_color, grid_color, all_modes, log_scale, show_stats, detect_peaks,
                  peak_threshold, line_style, line_alpha, multi_channel_count, colormap,
                  latent_compare=None):
        """
        Main visualization function. Generates plots based on input parameters.

        Args:
            (All args match INPUT_TYPES definition)

        Returns:
            A tuple containing a single IMAGE tensor `(image_tensor,)`.
            Returns a placeholder tensor on failure.
        """
        fig = None # Define fig in outer scope for robust cleanup
        buf = None # Define buf in outer scope

        try:
            # --- Input Validation and Preparation ---
            if not isinstance(latent, dict) or "samples" not in latent:
                 raise ValueError("Invalid LATENT input: Must be a dictionary with a 'samples' key.")
            latent_tensor = latent["samples"]
            if not isinstance(latent_tensor, torch.Tensor):
                 raise ValueError("Latent 'samples' must be a torch.Tensor.")
            if latent_tensor.numel() == 0:
                 raise ValueError("Latent 'samples' tensor is empty.")

            b, c, h, w_latent = latent_tensor.shape # Use w_latent to avoid conflict with width param

            # Validate batch index
            batch_idx = min(max(0, batch_index), b - 1)
            if batch_idx != batch_index:
                 logging.warning(f"Batch index clamped: {batch_index} -> {batch_idx} (Batch size is {b})")

            # Validate channel index
            if c == 0: # Should be caught by numel check, but double-check
                 raise ValueError("Latent has 0 channels.")
            channel = min(max(0, channel), c - 1)
            if channel != int(channel): # Check passed value, not just clamped
                 logging.warning(f"Channel index clamped: {int(channel)} -> {channel} (Channel count is {c})")


            # Get selected batch data [C, H, W]
            x_batch_chw = latent_tensor[batch_idx]

            # --- Color Validation ---
            bg_color = self.validate_color(bg_color, "#0D0D1A")
            waveform_color = self.validate_color(waveform_color, "#00E6E6")
            spectrum_color = self.validate_color(spectrum_color, "#FF00A2")
            rgb_r_color = self.validate_color(rgb_r_color, "#FF3333")
            rgb_g_color = self.validate_color(rgb_g_color, "#00FF8C")
            rgb_b_color = self.validate_color(rgb_b_color, "#3399FF")
            axis_label_color = self.validate_color(axis_label_color, "#A0A0B0")
            grid_color = self.validate_color(grid_color, "#303040")

            # --- Determine Modes and Setup Figure ---
            all_possible_modes = ["waveform", "spectrum", "rgb_split", "heatmap", "histogram",
                                  "statistics", "multi_channel", "phase", "difference"]
            modes_to_show = []
            if all_modes:
                modes_to_show = list(all_possible_modes) # Copy list
            else:
                 if mode in all_possible_modes:
                      modes_to_show = [mode]
                 else:
                      logging.warning(f"Invalid mode '{mode}', defaulting to 'waveform'.")
                      modes_to_show = ["waveform"]


            # Filter modes based on constraints
            if c < 3 and "rgb_split" in modes_to_show:
                modes_to_show.remove("rgb_split")
                logging.info("Removed 'rgb_split' mode (requires >= 3 channels).")
            if latent_compare is None and "difference" in modes_to_show:
                modes_to_show.remove("difference")
                logging.info("Removed 'difference' mode (requires 'latent_compare' input).")
            # Ensure multi_channel has enough channels
            if "multi_channel" in modes_to_show and c < 1:
                 modes_to_show.remove("multi_channel") # Should be caught earlier, but safe check


            # Handle case where all modes got filtered out
            if not modes_to_show:
                 logging.warning("No valid modes to display, defaulting to 'waveform'.")
                 modes_to_show = ["waveform"]

            num_plots = len(modes_to_show)

            # Create figure and axes
            fig, axes = plt.subplots(num_plots, 1,
                                     figsize=(width / DEFAULT_DPI, height / DEFAULT_DPI),
                                     dpi=DEFAULT_DPI, facecolor=bg_color)
            if num_plots == 1:
                axes = [axes] # Ensure axes is always a list


            # --- Plotting ---
            ls = self._get_line_style(line_style)
            latent_compare_batch_chw = None
            if latent_compare is not None and "samples" in latent_compare:
                 lc_tensor = latent_compare["samples"]
                 if isinstance(lc_tensor, torch.Tensor) and lc_tensor.dim() == 4:
                      lc_batch_idx = min(batch_idx, lc_tensor.shape[0] - 1) # Use same batch index if possible
                      latent_compare_batch_chw = lc_tensor[lc_batch_idx]


            for idx, current_mode in enumerate(modes_to_show):
                ax = axes[idx]
                # Apply base styling first
                self._setup_axis_style(ax, bg_color, axis_label_color, grid, grid_color)

                # Call specific plot function
                try:
                    if current_mode == "waveform":
                        signal_data = self._get_1d_signal(x_batch_chw[channel], normalize)
                        self._plot_waveform(ax, signal_data, channel, waveform_color, ls,
                                            line_alpha, normalize, show_stats, detect_peaks,
                                            peak_threshold, axis_label_color)
                    elif current_mode == "spectrum":
                        raw_signal = x_batch_chw[channel].detach().cpu().numpy().flatten()
                        self._plot_spectrum(ax, raw_signal, channel, spectrum_color,
                                            log_scale, show_stats, detect_peaks,
                                            peak_threshold, axis_label_color)
                    elif current_mode == "rgb_split":
                        self._plot_rgb_split(ax, x_batch_chw, [rgb_r_color, rgb_g_color, rgb_b_color],
                                             normalize, ls, line_alpha, axis_label_color)
                    elif current_mode == "heatmap":
                        self._plot_heatmap(ax, x_batch_chw, channel, colormap)
                    elif current_mode == "histogram":
                        self._plot_histogram(ax, x_batch_chw, channel, waveform_color,
                                             log_scale, show_stats, axis_label_color)
                    elif current_mode == "statistics":
                         self._plot_statistics(ax, x_batch_chw, axis_label_color, bg_color)
                    elif current_mode == "multi_channel":
                        self._plot_multi_channel(ax, x_batch_chw, multi_channel_count,
                                                 normalize, ls, line_alpha, axis_label_color)
                    elif current_mode == "phase":
                        raw_signal = x_batch_chw[channel].detach().cpu().numpy().flatten()
                        self._plot_phase(ax, raw_signal, channel, spectrum_color,
                                         show_stats, axis_label_color)
                    elif current_mode == "difference" and latent_compare_batch_chw is not None:
                         self._plot_difference(ax, x_batch_chw, latent_compare_batch_chw, channel,
                                               waveform_color, normalize, show_stats, axis_label_color, ls, line_alpha)

                except Exception as plot_err:
                    logging.error(f"[ACELatentVisualizer] Error plotting mode '{current_mode}': {plot_err}", exc_info=True)
                    # Display error message on the specific subplot
                    ax.text(0.5, 0.5, f"Plotting Error:\n{plot_err}", ha='center', va='center',
                            transform=ax.transAxes, color=axis_label_color, fontsize=6, wrap=True)
                    ax.set_title(f"Mode: {current_mode} - ERROR", fontsize=8)


            # Adjust layout and save to buffer
            plt.tight_layout(pad=0.5, h_pad=0.8) # Add vertical padding between subplots

            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1, facecolor=fig.get_facecolor(), dpi=DEFAULT_DPI)
            buf.seek(0)

            # Convert buffer to PIL Image -> NumPy array -> Torch Tensor
            image = Image.open(buf).convert("RGB")
            # No need to resize here, figsize and dpi control output size
            image_np = np.array(image).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_np).unsqueeze(0) # Add batch dimension [1, H, W, 3]

            # Return the tensor within a tuple
            return (image_tensor,)

        except Exception as e:
            logging.error(f"[ACELatentVisualizer] Visualization failed: {e}", exc_info=True)
            print(f"ERROR: [ACELatentVisualizer] {e}") # Print error for visibility
            # Return a placeholder red tensor on any failure
            placeholder = torch.zeros((1, height, width, 3), dtype=torch.float32)
            placeholder[:, :, :, 0] = 1.0 # Red channel
            return (placeholder,)

        finally:
            # --- CRITICAL Cleanup ---
            if buf:
                 try: buf.close()
                 except Exception: pass
            if fig:
                 try: plt.close(fig) # Ensure figure is always closed
                 except Exception: pass
            # Try to clear matplotlib's internal state if possible
            plt.clf()
            plt.cla()


# =================================================================================
# == Node Registration                                                         ==
# =================================================================================

NODE_CLASS_MAPPINGS = {
    "ACE_LatentVisualizer": ACELatentVisualizer, # Use updated class name
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ACE_LatentVisualizer": "MD: ACE Latent Visualizer", # Added MD: prefix, removed version/emoji
}