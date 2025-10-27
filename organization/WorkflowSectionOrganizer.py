# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
# ████ MD_Nodes/WorkflowSectionOrganizer – Visual Chapter Markers v1.0.0 ████▓▒░
# © 2025 MDMAchine
# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
# ░▒▓ ORIGIN & DEV:
#   • Cast into the void by: MDMAchine
#   • Enhanced by: Claude
#   • License: Apache 2.0 — Sharing is caring
#   • Part of: ComfyUI_MD_Nodes Workflow Organization Suite

# ░▒▓ DESCRIPTION:
#   A universal passthrough node (*) that acts as a visual "bookmark" or "chapter
#   divider" in complex workflows. Accepts any data type, passes it through
#   unchanged, and displays as a configurable banner. Helps organize large graphs
#   into logical sections with automatic color suggestions based on section type.

# ░▒▓ FEATURES:
#   ✓ Universal passthrough—accepts ANY ComfyUI data type (*).
#   ✓ Completely transparent—no data modification.
#   ✓ Auto-color suggestions based on section type (Audio, Sampling, Utilities, etc.).
#   ✓ Custom section labels with optional emoji support.
#   ✓ Visual banner style for clear section breaks (frontend dependent).
#   ✓ Color theme presets for common workflow stages.
#   ✓ Optional description field for detailed notes within the node.
#   ✓ Zero performance impact (pure passthrough).

# ░▒▓ CHANGELOG:
#   - v1.0.0 (Current Release - Guideline Compliance):
#       • COMPLIANCE: Removed all Python type hints (Guide Sec 7.2).
#       • COMPLIANCE: Added MD: prefix to display name, removed emoji (Guide Sec 6.2).
#       • COMPLIANCE: Ensured tooltips follow standard structure (Guide Sec 9.1).
#       • COMPLIANCE: Added logging and error handling (Guide Sec 7.3, 8.3).
#       • COMPLIANCE: Node is correctly cacheable (Guide Sec 8.1).
#       • ADDED: Initial feature set (universal passthrough, auto-color, types, labels, description, themes).

# ░▒▓ CONFIGURATION:
#   → Primary Use: Visually marking major sections (e.g., 'Data Loading', 'Main Processing', 'Output') in complex workflows by connecting data flow through it.
#   → Secondary Use: Color-coding different processing stages based on 'section_type' for quick identification.
#   → Edge Use: Creating a visual "table of contents" at the start or end of a graph by daisy-chaining organizers.

# ░▒▓ WARNING:
#   This node may trigger:
#   ▓▒░ Sudden onset of workflow organization OCD.
#   ▓▒░ An uncontrollable desire to add chapter markers to literally everything.
#   ▓▒░ Flashbacks to your old three-ring binder divider tabs, now with more neon.
#   ▓▒░ Compulsive emoji selection for 'perfect' visual hierarchy in section labels.
#   Known side effects: Colleagues complimenting your workflow clarity, reduced debugging time, and spontaneous promotion to "workflow architect." Organize responsibly.
# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄


# =================================================================================
# == Standard Library Imports                                                  ==
# =================================================================================
import logging
import traceback
import secrets # Standard import

# =================================================================================
# == Third-Party Imports                                                       ==
# =================================================================================
# (None needed)

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
logger = logging.getLogger("ComfyUI_MD_Nodes.WorkflowSectionOrganizer")

# --- Color Suggestion System ---
# Maps section types to suggested colors, emojis, and descriptions
COLOR_THEME_MAP = {
    # Audio-related sections
    "audio_generation": {"color": "skyblue", "emoji": "🎵", "desc": "Audio creation and synthesis"},
    "audio_processing": {"color": "blue", "emoji": "🎚️", "desc": "Audio effects and mixing"},
    "audio_mastering": {"color": "navy", "emoji": "🎼", "desc": "Final audio mastering stage"},

    # Image generation sections
    "image_generation": {"color": "pink", "emoji": "🖼️", "desc": "Image synthesis and creation"},
    "image_processing": {"color": "magenta", "emoji": "🎨", "desc": "Image editing and effects"},
    "image_upscaling": {"color": "violet", "emoji": "🔍", "desc": "Image enhancement and upscaling"},

    # Video generation sections
    "video_generation": {"color": "purple", "emoji": "🎬", "desc": "Video synthesis and creation"},
    "video_processing": {"color": "indigo", "emoji": "🎞️", "desc": "Video editing and effects"},
    "animation": {"color": "lavender", "emoji": "🎭", "desc": "Animation and motion"},

    # Sampling and generation core
    "sampling": {"color": "cyan", "emoji": "🎲", "desc": "Diffusion sampling process"},
    "scheduling": {"color": "teal", "emoji": "📊", "desc": "Sigma/noise scheduling"},
    "conditioning": {"color": "turquoise", "emoji": "🧬", "desc": "Prompt conditioning/encoding"},

    # Utilities and I/O
    "loading": {"color": "green", "emoji": "📥", "desc": "Model, data, or file loading"},
    "saving": {"color": "olive", "emoji": "💾", "desc": "Output saving or exporting"},
    "utilities": {"color": "lime", "emoji": "🔧", "desc": "Helper utilities and workflow tools"},

    # Processing stages
    "preprocessing": {"color": "orange", "emoji": "⚙️", "desc": "Input data preparation"},
    "processing": {"color": "amber", "emoji": "⚡", "desc": "Main computational processing stage"},
    "postprocessing": {"color": "coral", "emoji": "✨", "desc": "Output refinement and final adjustments"},

    # Special sections
    "debugging": {"color": "red", "emoji": "🐛", "desc": "Debugging, analysis, and visualization"},
    "control": {"color": "gray", "emoji": "🎛️", "desc": "Workflow control logic (switches, gates)"},
    "custom": {"color": "white", "emoji": "📌", "desc": "User-defined custom section"},
}

# =================================================================================
# == Core Node Class                                                           ==
# =================================================================================

class WorkflowSectionOrganizer:
    """
    MD: Workflow Section Organizer (Prototype)

    A universal passthrough node (*) designed to act as a visual separator or
    'chapter marker' within complex ComfyUI workflows. It accepts any data type,
    passes it through completely unchanged, and displays as a customizable banner.
    Helps to visually structure large graphs into logical processing stages.
    """

    @classmethod
    def INPUT_TYPES(cls):
        """
        Define node inputs. NO TYPE HINTS HERE.
        """
        section_types = list(COLOR_THEME_MAP.keys())

        return {
            "required": {
                "section_type": (section_types, {
                    "default": "custom",
                    "tooltip": (
                        "SECTION TYPE\n"
                        "- Select the category that best describes this part of the workflow.\n"
                        "- This primarily influences the 'auto' color suggestion.\n\n"
                        "CATEGORIES:\n"
                        "- Audio (Generation, Processing, Mastering)\n"
                        "- Image (Generation, Processing, Upscaling)\n"
                        "- Video (Generation, Processing, Animation)\n"
                        "- Core (Sampling, Scheduling, Conditioning)\n"
                        "- IO (Loading, Saving, Utilities)\n"
                        "- Stages (Preprocessing, Processing, Postprocessing)\n"
                        "- Other (Debugging, Control, Custom)"
                    )
                }),
                "section_label": ("STRING", {
                    "multiline": False, "default": "Workflow Section", # Changed default
                    "tooltip": (
                        "SECTION LABEL\n"
                        "- The primary text displayed on the node's banner.\n"
                        "- Use concise, descriptive names (e.g., 'VAE Loading', 'Sampling Loop', 'Audio Output').\n"
                        "- Emojis are supported for visual flair ✨."
                    )
                }),
                "color_override": (["auto"] + sorted(list(set(theme["color"] for theme in COLOR_THEME_MAP.values()))), { # Dynamically get unique colors
                    "default": "auto",
                    "tooltip": (
                        "COLOR OVERRIDE\n"
                        "- Manually select a color for the node's banner.\n"
                        "- 'auto': Automatically uses the color suggested by the 'section_type'.\n"
                        "- Choose a specific color to enforce a custom theme."
                    )
                }),
            },
            "optional": {
                "description": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": (
                        "DESCRIPTION (Optional)\n"
                        "- Add detailed notes or explanations about this section.\n"
                        "- Visible inside the node properties panel (if implemented by frontend).\n"
                        "- Does not appear directly on the banner."
                    )
                }),
                "passthrough": ("*", {
                    "forceInput": True, # Ensure it's always visible for connection
                    "tooltip": (
                        "PASS-THROUGH (*)\n"
                        "- Connect ANY data type from the previous node here.\n"
                        "- The data will pass through to the output completely unchanged.\n"
                        "- **Crucial:** A connection here ensures this node executes in the correct sequence."
                    )
                }),
            }
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("passthrough",)
    FUNCTION = "organize_section"
    CATEGORY = "MD_Nodes/Workflow Organization" # Corrected category
    OUTPUT_NODE = False # This is primarily a passthrough node

    # No IS_CHANGED needed - this node is deterministic. Its visual state depends
    # on inputs, and its output is identical to its 'passthrough' input.
    # Default ComfyUI caching based on inputs (including 'passthrough') is correct.

    def organize_section(self, section_type, section_label, color_override,
                         description="", passthrough=None):
        """
        Passes through the input data unchanged. The node's primary function
        is visual organization within the ComfyUI graph editor.

        Args:
            section_type: Category string for color suggestion.
            section_label: Text displayed on the node.
            color_override: Manual color choice ('auto' or specific color name).
            description: Optional detailed notes.
            passthrough: The data received from the input connection.

        Returns:
            A tuple containing only the unchanged passthrough data `(passthrough,)`.
        """
        try:
            # Determine the effective color (mainly for logging/potential future use)
            theme = COLOR_THEME_MAP.get(section_type, COLOR_THEME_MAP["custom"])
            suggested_color = theme.get("color", "white") # Safe get
            final_color = suggested_color if color_override == "auto" else color_override

            # Log execution details (useful for debugging workflow structure)
            log_message = (f"[WorkflowSectionOrganizer] Executed Section: '{section_label}' | "
                           f"Type: {section_type} | Color: {final_color}")
            if description:
                # Log only the beginning of the description to avoid spamming logs
                log_message += f" | Desc: '{description[:50]}...'"
            logger.debug(log_message) # Use debug level for less console noise

            # The core functionality: return the input data exactly as received.
            # Must return a tuple matching RETURN_TYPES.
            return (passthrough,)

        except Exception as e:
            # Log any unexpected errors, but still attempt passthrough
            logger.error(f"[WorkflowSectionOrganizer] Error during execution (should be no-op): {e}", exc_info=True)
            print(f"ERROR: [WorkflowSectionOrganizer] {e}") # Print error for visibility
            # Return the passthrough data even if logging fails
            return (passthrough,)

# =================================================================================
# == Node Registration                                                         ==
# =================================================================================

NODE_CLASS_MAPPINGS = {
    "WorkflowSectionOrganizer": WorkflowSectionOrganizer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WorkflowSectionOrganizer": "MD: Workflow Section Organizer", # Added MD: prefix, removed emoji
}