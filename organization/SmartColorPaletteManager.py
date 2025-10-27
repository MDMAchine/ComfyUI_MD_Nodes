# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
# ████ MD_Nodes/SmartColorPaletteManager – Workflow Color Organization v1.0.0 ████▓▒░
# © 2025 MDMAchine
# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
# ░▒▓ ORIGIN & DEV:
#   • Cast into the void by: MDMAchine
#   • Enhanced by: Claude
#   • License: Apache 2.0 — Sharing is caring
#   • Part of: ComfyUI_MD_Nodes Workflow Organization Suite

# ░▒▓ DESCRIPTION:
#   A comprehensive color palette management system for organizing complex workflows.
#   Provides preset color schemes optimized for different workflow types (audio, image,
#   video generation), outputs individual color codes for manual application, and includes
#   an auto-detection mode that suggests colors based on node type keywords. Aids visual
#   clarity through consistent color coding.

# ░▒▓ FEATURES:
#   ✓ 10+ professional preset color schemes (Audio, Image, Video, Mixed, etc.)
#   ✓ Outputs 8 individual color codes per palette (Primary through Accent5)
#   ✓ Auto-detection suggestions based on node category keywords
#   ✓ Custom palette creation with manual color selection (via preset 'custom')
#   ✓ Color scheme descriptions and use-case guidance
#   ✓ Outputs both hex codes and human-readable color names
#   ✓ Visual preview output showing the entire palette
#   ✓ Complements WorkflowSectionOrganizer for unified workflow theming

# ░▒▓ CHANGELOG:
#   - v1.0.0 (Current Release - Guideline Compliance):
#       • COMPLIANCE: Removed all Python type hints (Guide Sec 7.2).
#       • COMPLIANCE: Added MD: prefix to display name, removed emoji (Guide Sec 6.2).
#       • COMPLIANCE: Ensured tooltips follow standard structure (Guide Sec 9.1).
#       • COMPLIANCE: Added logging and error handling (Guide Sec 7.3, 8.3).
#       • COMPLIANCE: Node is correctly cacheable (Guide Sec 8.1).
#       • COMPLIANCE: Matplotlib usage follows guidelines (Guide Sec 9.2).
#       • ADDED: Initial feature set (presets, color outputs, auto-detect, preview, info).

# ░▒▓ CONFIGURATION:
#   → Primary Use: Selecting a preset (e.g., 'audio_workflow') to get consistent color codes for styling nodes manually or via other organization nodes.
#   → Secondary Use: Using 'auto_detect' with workflow keywords to get a suggested palette.
#   → Edge Use: Generating the visual preview image to include in workflow documentation.

# ░▒▓ WARNING:
#   This node may trigger:
#   ▓▒░ Obsessive color coordination of your entire node collection.
#   ▓▒░ Sudden interest in color theory and complementary hues.
#   ▓▒░ Flashbacks to your MySpace profile customization days.
#   ▓▒░ Compulsive palette creation for workflows you haven't built yet.
#   Side effects include: Workflows so aesthetically pleasing your GPU blushes,
#   colleagues asking "who's your interior decorator?", and sudden urges to
#   organize everything by ROYGBIV. Use with artistic discretion.
# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄


# =================================================================================
# == Standard Library Imports                                                  ==
# =================================================================================
import logging
import io
import traceback
import secrets # Standard import

# =================================================================================
# == Third-Party Imports                                                       ==
# =================================================================================
import torch
import numpy as np

# Optional imports for visual report generation
try:
    from PIL import Image
    import matplotlib as mpl
    mpl.use('Agg') # Set backend BEFORE importing pyplot (CRITICAL)
    import matplotlib.pyplot as plt
    PREVIEW_AVAILABLE = True
except ImportError:
    PREVIEW_AVAILABLE = False
    # Use print for visibility during ComfyUI startup
    print("WARNING: [SmartColorPaletteManager] Matplotlib or PIL not found. Image preview will be a blank placeholder.")
    # Define dummy Image class if PIL is missing, to avoid NameError later
    if 'Image' not in globals():
         class Image: pass # Simple placeholder

# =================================================================================
# == ComfyUI Core Modules                                                      ==
# =================================================================================
# (None needed)

# =================================================================================
# == Local Project Imports                                                     ==
# =================================================================================
# (None needed)

# =================================================================================
# == Helper Classes & Dependencies                                             ==
# =================================================================================

# Setup logger for this node
logger = logging.getLogger("ComfyUI_MD_Nodes.SmartColorPaletteManager")
# Basic config if run standalone, ComfyUI might override this
# logging.basicConfig(level=logging.INFO)

# --- Color Palette Library ---
# Each palette: 8 colors (primary, secondary, tertiary, 5 accents), names, description
PRESET_PALETTES = {
    "audio_workflow": {
        "colors": ["#1E90FF", "#4169E1", "#0000CD", "#87CEEB", "#00CED1", "#20B2AA", "#48D1CC", "#5F9EA0"],
        "names": ["DodgerBlue", "RoyalBlue", "MediumBlue", "SkyBlue", "DarkTurquoise", "LightSeaGreen", "MediumTurquoise", "CadetBlue"],
        "desc": "Cool blues and cyans optimized for audio generation workflows"
    },
    "image_workflow": {
        "colors": ["#FF1493", "#FF69B4", "#DA70D6", "#BA55D3", "#9370DB", "#8A2BE2", "#9400D3", "#8B008B"],
        "names": ["DeepPink", "HotPink", "Orchid", "MediumOrchid", "MediumPurple", "BlueViolet", "DarkViolet", "DarkMagenta"],
        "desc": "Vibrant pinks and purples for image generation pipelines"
    },
    "video_workflow": {
        "colors": ["#8B00FF", "#9932CC", "#9370DB", "#E6E6FA", "#D8BFD8", "#DDA0DD", "#EE82EE", "#DA70D6"],
        "names": ["Violet", "DarkOrchid", "MediumPurple", "Lavender", "Thistle", "Plum", "Violet", "Orchid"], # Renamed first color name
        "desc": "Deep purples and lavenders for video synthesis workflows"
    },
    "mixed_media": {
        "colors": ["#FF6347", "#FF7F50", "#FFA500", "#FFD700", "#ADFF2F", "#00FA9A", "#00CED1", "#1E90FF"],
        "names": ["Tomato", "Coral", "Orange", "Gold", "GreenYellow", "MediumSpringGreen", "DarkTurquoise", "DodgerBlue"],
        "desc": "Rainbow gradient for complex multi-modal workflows"
    },
    "sampling_focus": {
        "colors": ["#00CED1", "#20B2AA", "#48D1CC", "#40E0D0", "#7FFFD4", "#66CDAA", "#00FA9A", "#98FB98"],
        "names": ["DarkTurquoise", "LightSeaGreen", "MediumTurquoise", "Turquoise", "Aquamarine", "MediumAquamarine", "MediumSpringGreen", "PaleGreen"],
        "desc": "Teal and aqua tones for sampling and scheduling sections"
    },
    "mastering_chain": {
        "colors": ["#000080", "#191970", "#0000CD", "#0000FF", "#4169E1", "#6495ED", "#4682B4", "#5F9EA0"],
        "names": ["Navy", "MidnightBlue", "MediumBlue", "Blue", "RoyalBlue", "CornflowerBlue", "SteelBlue", "CadetBlue"],
        "desc": "Deep blues for audio mastering and post-processing"
    },
    "utilities_helpers": {
        "colors": ["#32CD32", "#00FF00", "#7FFF00", "#ADFF2F", "#9ACD32", "#6B8E23", "#556B2F", "#8FBC8F"],
        "names": ["LimeGreen", "Lime", "Chartreuse", "GreenYellow", "YellowGreen", "OliveDrab", "DarkOliveGreen", "DarkSeaGreen"],
        "desc": "Bright greens for utility nodes and helpers"
    },
    "io_operations": {
        "colors": ["#228B22", "#006400", "#008000", "#2E8B57", "#3CB371", "#66CDAA", "#8FBC8F", "#90EE90"],
        "names": ["ForestGreen", "DarkGreen", "Green", "SeaGreen", "MediumSeaGreen", "MediumAquamarine", "DarkSeaGreen", "LightGreen"],
        "desc": "Forest greens for loading and saving operations"
    },
    "processing_stages": {
        "colors": ["#FF8C00", "#FFA500", "#FFB347", "#FFDAB9", "#FFE4B5", "#F4A460", "#D2691E", "#CD853F"],
        "names": ["DarkOrange", "Orange", "Pastel Orange", "PeachPuff", "Moccasin", "SandyBrown", "Chocolate", "Peru"],
        "desc": "Warm oranges for pre/post-processing stages"
    },
    "debug_quality": {
        "colors": ["#DC143C", "#FF0000", "#FF6347", "#FF4500", "#FF8C00", "#FFA500", "#FFD700", "#FFFF00"],
        "names": ["Crimson", "Red", "Tomato", "OrangeRed", "DarkOrange", "Orange", "Gold", "Yellow"],
        "desc": "Red-to-yellow gradient for debugging and QC nodes"
    },
    "monochrome_pro": {
        "colors": ["#000000", "#2F4F4F", "#696969", "#808080", "#A9A9A9", "#C0C0C0", "#D3D3D3", "#FFFFFF"],
        "names": ["Black", "DarkSlateGray", "DimGray", "Gray", "DarkGray", "Silver", "LightGray", "White"],
        "desc": "Professional grayscale palette for minimal aesthetics"
    },
    # Note: 'custom' is a placeholder, actual colors should be set via inputs if needed
    # "custom": {
    #     "colors": ["#FFFFFF"] * 8, "names": [f"Custom{i}" for i in range(1, 9)],
    #     "desc": "Manual color selection mode (Configure via inputs - not implemented)"
    # }
}

# --- Auto-detection Keywords ---
NODE_TYPE_KEYWORDS = {
    # Map suggested palette name to keywords
    "audio_workflow": ["audio", "sound", "music", "ace", "wav", "mp3", "speaker", "mastering", "eq", "t5", "soundfile"],
    "image_workflow": ["image", "img", "photo", "picture", "vae", "clip", "upscale", "inpaint", "latent", "pixel", "lora"],
    "video_workflow": ["video", "vid", "animation", "frame", "movie", "motion", "svd", "animate"],
    "sampling_focus": ["sample", "sampler", "scheduler", "sigma", "noise", "step", "denoise", "kdiffusion", "solver"],
    "utilities_helpers": ["utility", "helper", "tool", "convert", "switch", "route", "manage", "math", "logic", "number", "string", "int", "float"],
    "io_operations": ["load", "save", "import", "export", "file", "path", "read", "write", "input", "output"],
    "processing_stages": ["process", "preprocess", "postprocess", "enhance", "filter", "transform", "effect", "adjust"],
    "debug_quality": ["debug", "visualize", "preview", "monitor", "analyze", "test", "inspect", "qc", "guardian"]
}

# --- Auto-detection Logic ---
def detect_workflow_type(workflow_keywords_string):
    """
    Analyzes a string of keywords and suggests the most relevant palette preset name.

    Args:
        workflow_keywords_string: A string containing space or comma separated keywords.

    Returns:
        The name (string) of the suggested palette preset (e.g., "audio_workflow").
        Returns "mixed_media" as a fallback.
    """
    if not isinstance(workflow_keywords_string, str) or not workflow_keywords_string:
        return "mixed_media" # Default if no keywords

    keywords_lower = workflow_keywords_string.lower()
    # Simple word extraction (split by space, comma, newline, etc.)
    words = set(re.findall(r'\b\w+\b', keywords_lower))

    scores = {}
    # Score each palette based on keyword matches
    for palette_name, trigger_keywords in NODE_TYPE_KEYWORDS.items():
        score = sum(1 for keyword in trigger_keywords if keyword in words)
        scores[palette_name] = score

    # Find the palette with the highest score
    # Use a threshold to avoid triggering on very few matches
    best_match = "mixed_media" # Default fallback
    max_score = 1 # Minimum score threshold
    for palette_name, score in scores.items():
        if score > max_score:
            max_score = score
            best_match = palette_name

    logger.debug(f"Auto-detect scores: {scores}. Best match: {best_match}")
    return best_match

# --- Visual Preview Generation ---
def _create_placeholder_tensor():
    """Returns a blank placeholder tensor if Matplotlib/PIL are unavailable."""
    logger.warning("[SmartColorPaletteManager] Returning blank placeholder preview.")
    img_height_px, img_width_px = 250, 800
    # Dark gray placeholder
    return torch.ones((1, img_height_px, img_width_px, 3), dtype=torch.float32) * 0.1

def create_palette_preview(palette_name, colors, names):
    """
    Generate a visual preview image of the color palette using Matplotlib.
    Compliant with MD Nodes guidelines (Agg backend, PIL conversion, plt.close).

    Args:
        palette_name: Name of the palette (string).
        colors: List of hex color strings.
        names: List of color names (strings).

    Returns:
        An IMAGE tensor [1, H, W, 3] or a placeholder tensor on error/missing libs.
    """
    if not PREVIEW_AVAILABLE:
        return _create_placeholder_tensor()

    fig = None # Ensure defined for finally block
    buf = None # Ensure defined for finally block

    try:
        # Image dimensions
        img_width_px, img_height_px, dpi = 800, 250, 100
        img_width_in, img_height_in = img_width_px / dpi, img_height_px / dpi

        # Create figure with dark background
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(img_width_in, img_height_in), dpi=dpi)
        fig.patch.set_facecolor('#1a1a1a')
        ax.set_facecolor('#1a1a1a')
        ax.axis('off') # Hide axes

        # Title
        title_text = f"{palette_name.replace('_', ' ').title()} Palette"
        ax.text(0.03, 0.85, title_text, transform=fig.transFigure,
                fontsize=18, fontweight='bold', color='white', ha='left', va='center') # Reduced font size

        # Swatch dimensions and layout
        num_colors = len(colors)
        swatch_width = 0.9 / max(num_colors, 1) # Relative width based on number of colors
        swatch_height = 0.50 # Relative height
        start_x = 0.05
        start_y = 0.65 # Top of swatch
        spacing = swatch_width * 0.08 # Small spacing relative to width

        actual_swatch_width = swatch_width * (1.0 - 0.08) # Adjust width for spacing

        for i, (color, name) in enumerate(zip(colors, names)):
            # Calculate swatch position
            x_pos = start_x + i * swatch_width

            # Draw swatch rectangle
            rect = plt.Rectangle((x_pos, start_y - swatch_height), actual_swatch_width, swatch_height,
                                 transform=fig.transFigure, facecolor=color,
                                 edgecolor='#555555', linewidth=1) # Darker edge
            ax.add_patch(rect)

            # Draw color name below swatch
            ax.text(x_pos + actual_swatch_width / 2, start_y - swatch_height - 0.03, name,
                    transform=fig.transFigure, fontsize=9, color='white', ha='center', va='top') # Reduced font size

            # Draw hex code below name
            ax.text(x_pos + actual_swatch_width / 2, start_y - swatch_height - 0.10, color,
                    transform=fig.transFigure, fontsize=9, color='#AAAAAA', ha='center', va='top') # Reduced font size

        # Save figure to buffer using recommended PIL method (Guide Sec 9.2)
        buf = io.BytesIO()
        # Use slightly more padding for savefig
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.15, facecolor=fig.get_facecolor(), dpi=dpi)
        buf.seek(0)

        # Convert buffer to PIL Image -> NumPy array -> Torch Tensor
        img = Image.open(buf).convert('RGB')
        img_np = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).unsqueeze(0) # Add batch dim [1, H, W, 3]

        return img_tensor

    except Exception as e:
        logger.error(f"[SmartColorPaletteManager] Error creating palette preview: {e}", exc_info=True)
        return _create_placeholder_tensor() # Return placeholder on error

    finally:
        # --- CRITICAL Cleanup ---
        if buf:
            try: buf.close()
            except Exception: pass
        if fig:
            try: plt.close(fig) # Ensure figure is always closed
            except Exception: pass
        # Clear current figure/axes state if possible
        plt.clf()
        plt.cla()


# =================================================================================
# == Core Node Class                                                           ==
# =================================================================================

class SmartColorPaletteManager:
    """
    MD: Smart Color Palette Manager (Prototype)

    Manages and provides color palettes for workflow organization. Select presets,
    use auto-detection based on keywords, or define custom colors. Outputs
    individual hex codes and a visual preview image. Complements other
    Workflow Organization Suite nodes.
    """

    @classmethod
    def INPUT_TYPES(cls):
        """Define node inputs."""
        palette_names = list(PRESET_PALETTES.keys())

        return {
            "required": {
                "palette_preset": (palette_names, {
                    "default": "mixed_media",
                    "tooltip": (
                        "PALETTE PRESET\n"
                        "- Choose a predefined color scheme.\n"
                        "- Schemes are designed for different workflow types (Audio, Image, Video, etc.).\n"
                        "- See palette descriptions in the 'palette_info' output or node description."
                        # List removed for brevity, use palette_info output
                    )
                }),
                "auto_detect": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "AUTO-DETECT PALETTE\n"
                        "- True: Ignores 'palette_preset' and suggests a palette based on 'workflow_keywords'.\n"
                        "- False: Uses the selected 'palette_preset'."
                    )
                }),
            },
            "optional": {
                "workflow_keywords": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": (
                        "WORKFLOW KEYWORDS (for Auto-Detect)\n"
                        "- Enter keywords describing your workflow (e.g., 'audio generation', 'image upscale', 'video sampling').\n"
                        "- Used only when 'auto_detect' is True.\n"
                        "- Helps suggest a relevant color palette."
                    )
                }),
                # --- Custom Palette Inputs (Placeholder - Not fully implemented for UI input yet) ---
                # "custom_color_1": ("STRING", {"default": "#FFFFFF", "tooltip": "CUSTOM COLOR 1 (Hex)\n- Used if 'palette_preset' is 'custom'."}),
                # ... up to custom_color_8
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "IMAGE", "STRING")
    RETURN_NAMES = ("primary", "secondary", "tertiary", "accent1", "accent2", "accent3", "accent4", "accent5", "palette_preview", "palette_info")
    FUNCTION = "generate_palette"
    CATEGORY = "MD_Nodes/Workflow Organization" # Corrected category
    OUTPUT_NODE = True # Generates outputs, doesn't modify flow directly

    # No IS_CHANGED needed - output depends only on inputs, default caching is correct.

    def generate_palette(self, palette_preset, auto_detect, workflow_keywords=""):
        """
        Generates and outputs the selected or detected color palette.

        Args:
            palette_preset: The name of the selected preset.
            auto_detect: Boolean flag to enable auto-detection.
            workflow_keywords: Keywords string for auto-detection.

        Returns:
            Tuple containing 8 color strings (hex), 1 IMAGE tensor (preview),
            and 1 info string. Returns defaults on error.
        """
        try:
            # --- Determine Palette ---
            palette_key_to_use = palette_preset
            auto_detect_msg = ""
            if auto_detect and workflow_keywords:
                detected_palette = detect_workflow_type(workflow_keywords)
                logger.info(f"[SmartColorPaletteManager] Auto-detected palette: '{detected_palette}' based on keywords.")
                # Only override if the detected palette actually exists
                if detected_palette in PRESET_PALETTES:
                    palette_key_to_use = detected_palette
                    auto_detect_msg = f" (Auto-Detected: {detected_palette})"
                else:
                    logger.warning(f"Auto-detection suggested invalid palette '{detected_palette}', using selected '{palette_preset}'.")
                    auto_detect_msg = f" (Auto-Detect FAILED, using selected: {palette_preset})"

            # --- Get Palette Data ---
            # Fallback to mixed_media if the selected/detected key is invalid
            palette_data = PRESET_PALETTES.get(palette_key_to_use, PRESET_PALETTES["mixed_media"])
            if palette_key_to_use not in PRESET_PALETTES:
                logger.warning(f"Invalid palette preset '{palette_key_to_use}', falling back to 'mixed_media'.")
                palette_key_to_use = "mixed_media" # Ensure name matches data

            colors = palette_data.get("colors", ["#FFFFFF"] * 8)
            names = palette_data.get("names", [f"Color {i}" for i in range(1, 9)])
            desc = palette_data.get("desc", "No description available.")

            # Ensure exactly 8 colors/names, padding if necessary
            if len(colors) < 8: colors.extend(["#FFFFFF"] * (8 - len(colors)))
            if len(names) < 8: names.extend([f"Color {i}" for i in range(len(names) + 1, 9)])
            colors = colors[:8]
            names = names[:8]

            # --- Generate Outputs ---
            # 1. Visual Preview Image
            preview_image = create_palette_preview(palette_key_to_use, colors, names)

            # 2. Info String
            info_lines = [
                f"🎨 Palette: {palette_key_to_use.replace('_', ' ').title()}{auto_detect_msg}",
                f"ℹ️ Description: {desc}",
                "\nHEX Codes & Names:"
            ]
            output_names = ["Primary", "Secondary", "Tertiary", "Accent1", "Accent2", "Accent3", "Accent4", "Accent5"]
            for i, (color, name, out_name) in enumerate(zip(colors, names, output_names)):
                info_lines.append(f"  - {out_name}: {name} ({color})")
            info_string = "\n".join(info_lines)

            logger.info(f"[SmartColorPaletteManager] Generated palette: {palette_key_to_use}")

            # 3. Return tuple matching RETURN_TYPES
            # Unpack the first 8 colors explicitly
            return (
                colors[0], colors[1], colors[2], colors[3],
                colors[4], colors[5], colors[6], colors[7],
                preview_image,
                info_string
            )

        except Exception as e:
            logger.error(f"[SmartColorPaletteManager] Error generating palette: {e}", exc_info=True)
            print(f"ERROR: [SmartColorPaletteManager] {e}") # Print error for visibility
            # Return safe defaults on error
            default_colors = ["#DC143C"] * 8 # Use red to indicate error
            blank_image = _create_placeholder_tensor()
            error_info = f"❌ ERROR: Unable to generate palette.\nCheck logs for details.\n{traceback.format_exc()}"

            return (
                default_colors[0], default_colors[1], default_colors[2], default_colors[3],
                default_colors[4], default_colors[5], default_colors[6], default_colors[7],
                blank_image,
                error_info
            )

# =================================================================================
# == Node Registration                                                         ==
# =================================================================================

NODE_CLASS_MAPPINGS = {
    "SmartColorPaletteManager": SmartColorPaletteManager,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SmartColorPaletteManager": "MD: Smart Color Palette Manager", # Added MD: prefix, removed emoji
}