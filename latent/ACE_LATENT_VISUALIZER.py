# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# ████ ACE LATENT VISUALIZER v0.3.1 – Optimized for Ace-Step Audio/Video ████▓▒░
# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀

# ░▒▓ ORIGIN & DEV:
#   • Cast into the void by: MDMAchine (The latent abyss explorer)
#   • Original mind behind ACE LATENT VISUALIZER
#   • Initial ComfyUI adaptation by: devstral (local l33t)
#   • Enhanced & color calibrated by: Gemini (Pixel artist extraordinaire)
#   • Critical inspirations from: c0ffymachyne (signal processing wizardry)
#   • License: Apache 2.0 — Sharing digital secrets responsibly

# ░▒▓ DESCRIPTION:
#   A visualization node for ComfyUI designed with Ace-Step precision.
#   Offers multi-mode insight into latent tensors — waveform, spectrum,
#   and RGB channel splits — turning abstract data blobs into visible art.
#   Ideal for debugging, pattern spotting, or admiring your AI’s hidden guts.

# ░▒▓ FEATURES:
#   ✓ Multi-mode visualization: waveform, spectrum, rgb_split
#   ✓ Customizable channel selection and normalization
#   ✓ Supports stacked multi-mode plotting
#   ✓ Adjustable resolution, gridlines, and color themes
#   ✓ Lightweight Matplotlib backend for crisp visuals

# ░▒▓ CHANGELOG:
#   - v0.1 (Initial Release):
#       • Basic waveform mode implemented
#       • Single channel visualization support
#   - v0.2 (Feature Expansion):
#       • Added spectrum mode with decibel scaling
#       • RGB channel splitting introduced
#       • Improved plotting customization
#   - v0.3.1 (Refinement & Usability):
#       • Enhanced color calibration and UI tooltips
#       • Stability improvements and bug fixes

# ░▒▓ CONFIGURATION:
#   → Primary Use: Latent tensor visual inspection and debugging
#   → Secondary Use: Artistic exploration of AI latent spaces
#   → Edge Use: Data-driven generative art prototyping

# ░▒▓ WARNING:
#   This node may trigger:
#   ▓▒░ Obsessive pattern analysis
#   ▓▒░ Sudden urges to tweak latent spaces endlessly
#   ▓▒░ Creative inspiration overload

# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄


import torch
import matplotlib.pyplot as plt
import io # Our trusty digital post office for sending data around in memory
from PIL import Image # Pillow: our digital artist's toolkit for image manipulation
import numpy as np # NumPy: the mathematical muscle behind our data transformations

# Set Matplotlib backend to 'Agg'. This means Matplotlib won't try to pop up a window
# on your server. It's like telling it: "Hey, just draw it on paper, don't show me in person!"
plt.switch_backend('Agg')

class ACE_LatentVisualizer_v03:
    """
    ACE_LatentVisualizer_v03: Your personal latent data cartographer!
    Maps the hidden territories of your latent tensors.
    """

    # CATEGORY: This is where you'll find our node in the ComfyUI menu.
    # Look under 'latent/visualization' - we're hiding where the cool kids hang out!
    CATEGORY = "latent/visualization"
    
    # RETURN_TYPES: What kind of digital treasure does this node spit out?
    # Just one glorious 'IMAGE' tensor, ready for your 'Preview Image' node!
    RETURN_TYPES = ("IMAGE",)
    
    # FUNCTION: This tells ComfyUI which Python method to call when the node runs.
    # Our main event, the 'visualize' function!
    FUNCTION = "visualize"

    @classmethod
    def INPUT_TYPES(cls):
        """
        INPUT_TYPES: The ingredients list for our visualization recipe.
        Each item is a knob or button you can tweak on the ComfyUI node!
        """
        return {
            "required": { # These are the must-haves; the node won't run without them!
                "latent": ("LATENT",), # The raw, untamed latent data. Our canvas!
                "mode": (["waveform", "spectrum", "rgb_split"], {"default": "waveform"}), # How do you want to see it?
                "channel": ("INT", {"default": 0, "min": 0, "tooltip": "Which channel's secrets do you want to uncover? (for 'waveform' and 'spectrum')."}),
                "normalize": ("BOOLEAN", {"default": True, "tooltip": "Auto-adjust amplitude for better visibility? (Like hitting 'auto-level' in audio software!)"}),
                "width": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64, "tooltip": "How wide do you want your masterpiece to be (in pixels)?"}),
                "height": ("INT", {"default": 256, "min": 64, "max": 2048, "step": 64, "tooltip": "How tall should your visual art be?"}),
                "grid": ("BOOLEAN", {"default": True, "tooltip": "Want a grid for easy measurement? (Like graph paper, but digital!)"}),
                
                # --- Color Controls: Now with more rainbow power! ---
                "bg_color": ("STRING", {"default": "#0D0D1A", "tooltip": "The background color for your plots. Dark mode friendly!"}), # Deep blue/black
                "waveform_color": ("STRING", {"default": "#00E6E6", "tooltip": "The vibrant hue of your waveform line. (Think electric blue!)"}), # Electric Cyan
                "spectrum_color": ("STRING", {"default": "#FF00A2", "tooltip": "The dazzling color of your frequency spectrum. (A shocking pink!)"}), # Vibrant Magenta
                "rgb_r_color": ("STRING", {"default": "#FF3333", "tooltip": "Red channel's chosen shade. (A bold, yet subtle red)"}), # Softer Red
                "rgb_g_color": ("STRING", {"default": "#00FF8C", "tooltip": "Green channel's fresh look. (An energetic emerald)"}), # Emerald Green
                "rgb_b_color": ("STRING", {"default": "#3399FF", "tooltip": "Blue channel's cool vibe. (A tranquil sky blue)"}), # Sky Blue
                "axis_label_color": ("STRING", {"default": "#A0A0B0", "tooltip": "Color for axis labels and ticks. (Readable on dark backgrounds!)"}), # Muted light gray/blue
                "grid_color": ("STRING", {"default": "#303040", "tooltip": "Color for grid lines. (Subtle, so it doesn't steal the show)"}), # Darker blue-gray
                # --- End Color Controls ---
                
                "all_modes": ("BOOLEAN", {"default": True, "tooltip": "If True, we'll plot all modes. It's like a latent data buffet!"}),
            }
        }

    def visualize(self, latent, mode, channel, normalize, width, height, grid, bg_color, waveform_color, spectrum_color, rgb_r_color, rgb_g_color, rgb_b_color, axis_label_color, grid_color, all_modes):
        """
        visualize: The heart of our node! This function takes your latent data
        and transforms it into beautiful, insightful images.
        """
        # Extract latent tensor: [Batch, Channels, Height, Width]
        x = latent["samples"]
        b, c, h, w = x.shape # Unpack the dimensions like a digital detective!

        # First, a quick sanity check: does our latent even have channels?
        if c == 0:
            print("[ACE_LatentVisualizer] Error: Latent tensor has 0 channels. Can't visualize what's not there! Returning blank image.")
            # If there's no data, we'll return a blank black image to avoid crashing.
            # ComfyUI likes its images in (Batch, Height, Width, Channels) format.
            blank_image = torch.zeros((1, height, width, 3), dtype=torch.float32)
            return (blank_image,)

        # Clamp the requested channel index to valid range.
        # We don't want to go out of bounds, that's where the dragons are!
        if channel >= c:
            print(f"[ACE_LatentVisualizer] Warning: Selected channel {channel} exceeds max {c-1}. Using channel {c - 1}. Better safe than sorry!")
            channel = c - 1
        
        # Ensure channel is non-negative. Negative channels are a no-go!
        if channel < 0:
            channel = 0
            print(f"[ACE_LatentVisualizer] Warning: Channel index cannot be negative. Using channel 0. Defaulting to the first available.")

        # Check if latent has enough channels for RGB split visualization.
        # RGB needs at least 3 channels (R, G, B), duh!
        rgb_valid = c >= 3

        # Determine which visualization modes to show.
        # If 'all_modes' is true, we're having a visualization party!
        modes_to_show = ["waveform", "spectrum", "rgb_split"] if all_modes else [mode]
        
        # Filter out 'rgb_split' if we don't have enough channels.
        # No point trying to draw a tri-color plot with only one color crayon!
        if not rgb_valid and "rgb_split" in modes_to_show:
            modes_to_show.remove("rgb_split")
            print(f"[ACE_LatentVisualizer] Warning: RGB Split mode requires at least 3 channels. Skipping RGB Split. Your latent is colorblind for now!")

        # If the user selected an invalid single mode (e.g., 'rgb_split' on a 1-channel latent)
        # and it got removed, we'll gracefully default to 'waveform' if possible.
        if not modes_to_show and c > 0:
            modes_to_show = ["waveform"]
            print("[ACE_LatentVisualizer] Warning: Selected mode invalid/skipped. Defaulting to 'waveform'. Still got some lines for ya!")
        elif not modes_to_show: # Still no modes to show and no channels? Houston, we have no data!
             print("[ACE_LatentVisualizer] Error: No valid visualization modes available after filtering. Just a blank screen for now...")
             blank_image = torch.zeros((1, height, width, 3), dtype=torch.float32)
             return (blank_image,)


        num_plots = len(modes_to_show) # How many glorious subplots will we create?

        # Create the Matplotlib figure and axes.
        # We use `figsize` and `dpi` to control the *pixel resolution* of the output image.
        # Think of DPI as "dots per inch" - more dots means more detail!
        fig, axes = plt.subplots(num_plots, 1, figsize=(width / 100, height / 100), dpi=100)
        fig.patch.set_facecolor(bg_color)  # Set the overall canvas background color.

        # Matplotlib can be a bit quirky: if there's only one plot, 'axes' isn't a list.
        # We force it into a list for consistent handling. No favoritism here!
        if num_plots == 1:
            axes = [axes]

        def get_1d_signal_for_plot(data, normalize_signal):
            """
            This helper function preps our latent channel data for plotting.
            It flattens it and handles normalization, ensuring a nice visual.
            """
            signal = data.detach().cpu().numpy().flatten() # Grab the data, move to CPU, flatten it like a pancake.

            # If the signal is all zeros or almost constant, we add a little offset
            # to make sure you can actually *see* a line. Otherwise, it's just a blank space!
            if np.allclose(signal, 0) or np.ptp(signal) < 1e-6:
                return np.zeros_like(signal) + 0.5 # A flat line at 0.5 (middle of our 0-1 range)

            if normalize_signal:
                min_val, max_val = signal.min(), signal.max()
                if max_val - min_val > 1e-8: # Avoid division by zero!
                    signal = (signal - min_val) / (max_val - min_val) # Scale to [0, 1] range
                else: # Fallback for extremely flat signals
                    signal = np.zeros_like(signal) + 0.5
            return signal

        plot_index = 0
        for m in modes_to_show: # Loop through each mode we want to plot
            ax = axes[plot_index] # Get the current subplot's axes
            plot_index += 1 # Move to the next subplot for the next iteration

            # Set background color for each subplot. Consistency is key!
            ax.set_facecolor(bg_color)
            
            # Make sure our axis labels and ticks are readable on *any* background.
            ax.tick_params(axis='x', colors=axis_label_color, labelsize=6)
            ax.tick_params(axis='y', colors=axis_label_color, labelsize=6)
            ax.xaxis.label.set_color(axis_label_color) # X-axis label color
            ax.yaxis.label.set_color(axis_label_color) # Y-axis label color
            ax.title.set_color(axis_label_color) # Plot title color

            if m == "waveform":
                # It's waveform time! Plotting our latent channel like a heartbeat.
                signal = get_1d_signal_for_plot(x[0, channel], normalize)
                ax.plot(signal, linewidth=0.75, color=waveform_color) # Use the custom waveform color!
                ax.set_title(f"Latent Waveform (Ch {channel})", fontsize=8)
                ax.set_ylabel("Amplitude", fontsize=6)
                ax.set_xlabel("Latent Pixel Index", fontsize=6)
                # Keep the y-axis boundaries fixed if normalized, for consistent view.
                ax.set_ylim(0 if normalize else None, 1 if normalize else None)
                # Add a grid, if requested, with custom color. It's like digital graph paper!
                ax.grid(grid, which='both', linestyle='--', linewidth=0.3, alpha=0.5, color=grid_color)

            elif m == "spectrum":
                # Behold, the spectrum! Unveiling the hidden frequencies of your latent data.
                raw_signal = x[0, channel].detach().cpu().numpy().flatten()
                if len(raw_signal) == 0: # Handle empty signals gracefully
                    ax.set_title("Latent Spectrum (Empty)", fontsize=8)
                    ax.text(0.5, 0.5, "No data", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color=axis_label_color)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    continue # Skip to the next plot if no data

                spectrum = np.fft.rfft(raw_signal)  # Real FFT for positive frequencies only
                freq = np.fft.rfftfreq(len(raw_signal)) # Calculate corresponding frequencies
                # Convert magnitude to decibels (dB) because bigger numbers look cooler!
                magnitude_db = 20 * np.log10(np.abs(spectrum) + 1e-8)

                ax.plot(freq, magnitude_db, color=spectrum_color) # Use the custom spectrum color!
                ax.set_title(f"Latent Spectrum (Ch {channel}, dB)", fontsize=8)
                ax.set_ylabel("Magnitude (dB)", fontsize=6)
                ax.set_xlabel("Normalized Frequency", fontsize=6)
                ax.set_xbound(0, freq.max()) # Make sure the X-axis shows the full frequency range
                ax.grid(grid, which='both', linestyle='--', linewidth=0.3, alpha=0.5, color=grid_color)

            elif m == "rgb_split":
                # The RGB split! See how your first three channels behave like color components.
                colors = [rgb_r_color, rgb_g_color, rgb_b_color] # Our custom RGB colors!
                labels = ["R (Ch 0)", "G (Ch 1)", "B (Ch 2)"]
                for i in range(min(3, c)): # Loop through the first 3 channels (or fewer if not available)
                    signal = get_1d_signal_for_plot(x[0, i], normalize)
                    ax.plot(signal, linewidth=0.75, color=colors[i], label=labels[i])
                ax.set_title("Latent RGB Channel Split", fontsize=8)
                ax.set_ylabel("Amplitude", fontsize=6)
                ax.set_xlabel("Latent Pixel Index", fontsize=6)
                ax.set_ylim(0 if normalize else None, 1 if normalize else None)
                # A legend helps us tell Red from Green from Blue!
                ax.legend(loc="upper right", fontsize=6, frameon=False, labelcolor=axis_label_color)
                ax.grid(grid, which='both', linestyle='--', linewidth=0.3, alpha=0.5, color=grid_color)
            
            # If the user turned off the grid, let's also hide the axis ticks and labels
            # for a super clean, minimalist look. Less clutter, more latent!
            if not grid:
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xlabel("")
                ax.set_ylabel("")

        # Adjust layout to prevent titles/labels from overlapping (like polite party guests!)
        # `pad=0.5` adds a little breathing room around the plots.
        plt.tight_layout(pad=0.5)
        
        buf = io.BytesIO() # Our in-memory digital canvas
        # Save the plot to the buffer as a PNG. `bbox_inches='tight'` and `pad_inches=0`
        # ensure no extra whitespace, giving us a perfectly cropped image.
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)  # CLOSE THE FIGURE! This is super important to prevent memory leaks!
                        # Forgetting this is like leaving the party lights on forever!
        buf.seek(0) # Rewind the buffer to the beginning, ready to be read.

        # Load our masterpiece from the buffer with PIL, convert to RGB.
        image = Image.open(buf).convert("RGB")
        
        # Resize to the requested output dimensions. PIL's LANCZOS filter is like
        # a master upscaler/downscaler for pixel perfection!
        image = image.resize((width, height), Image.LANCZOS)
        
        # Convert our PIL Image into a PyTorch tensor.
        # ComfyUI expects images in (Batch, Height, Width, Channels) format,
        # with pixel values normalized to [0,1].
        image_tensor = torch.from_numpy(np.array(image)).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0) # Add the batch dimension (we only have one image)

        # Return our magnificent latent visualization image!
        # (ComfyUI always expects node outputs as a tuple of tensors)
        return (image_tensor,)


# NODE_CLASS_MAPPINGS: This is how ComfyUI knows about our amazing node!
# It maps the internal class name to a friendly string.
NODE_CLASS_MAPPINGS = {
    "ACE_LatentVisualizer": ACE_LatentVisualizer_v03,
}

# NODE_DISPLAY_NAME_MAPPINGS: This is the name you'll see in the "Add Node" menu.
NODE_DISPLAY_NAME_MAPPINGS = {
    "ACE_LatentVisualizer": "ACE Latent Visualizer (Pixel Party!)"
}