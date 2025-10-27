# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
# ████ MD_Nodes/EnhancedAnnotationNode – Workflow Documentation v1.0.0 ████▓▒░
# © 2025 MDMAchine
# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
# ░▒▓ ORIGIN & DEV:
#   • Cast into the void by: MDMAchine
#   • Enhanced by: Claude
#   • License: Apache 2.0 — Sharing is caring
#   • Part of: ComfyUI_MD_Nodes Workflow Organization Suite

# ░▒▓ DESCRIPTION:
#   An enhanced annotation/comment node for documenting workflows in ComfyUI.
#   Provides customizable fonts, colors, styles, and markdown rendering for
#   creating professional documentation, section headers, and notes.
#   This node is purely visual and does not affect workflow execution.

# ░▒▓ FEATURES:
#   ✓ Multi-line text input with unlimited length
#   ✓ Adjustable font size (tiny to huge)
#   ✓ Multiple visual styles (sticky note, banner, callout, minimal, header)
#   ✓ Full color customization (background and text presets)
#   ✓ Markdown support for formatted text (frontend dependent)
#   ✓ Emoji support for visual flair
#   ✓ Optional title field for section headers
#   ✓ Width control for different use cases
#   ✓ Does not execute—zero performance impact
#   ✓ Correctly uses default ComfyUI caching (no IS_CHANGED).

# ░▒▓ CHANGELOG:
#   - v1.0.0 (Current Release - Guideline Compliance):
#       • COMPLIANCE: Removed all Python type hints (Guide Sec 7.2).
#       • COMPLIANCE: Added MD: prefix to display name, removed emoji (Guide Sec 6.2).
#       • COMPLIANCE: Ensured tooltips follow standard structure (Guide Sec 9.1).
#       • COMPLIANCE: Ensured logger usage (Guide Sec 7.3).
#       • COMPLIANCE: Correctly omits IS_CHANGED for cacheability (Guide Sec 8.1).
#       • ADDED: Initial feature set (styles, fonts, colors, markdown, title, width).

# ░▒▓ CONFIGURATION:
#   → Primary Use: Document workflow sections, add explanatory notes and reminders.
#   → Secondary Use: Create visually distinct section headers or dividers using 'banner' or 'header' styles.
#   → Edge Use: Storing workflow metadata, code snippets, or even ASCII art.

# ░▒▓ WARNING:
#   This node may trigger:
#   ▓▒░ Obsessive note-taking and over-documentation.
#   ▓▒░ A sudden urge to color-code your entire workflow like a beautiful rainbow.
#   ▓▒░ Flashbacks to your old Trapper Keeper from 1993, meticulously organizing everything.
#   ▓▒░ Compulsive emoji usage in technical documentation (go easy!).
#   Side effects include: legible workflows, helpful comments, and teammates actually understanding your graph. Use responsibly.
# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄


# =================================================================================
# == Standard Library Imports                                                  ==
# =================================================================================
import logging
import traceback
import secrets # Standard import, though not used here

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
logger = logging.getLogger("ComfyUI_MD_Nodes.EnhancedAnnotationNode")
# Basic config if run standalone, ComfyUI might override this
# logging.basicConfig(level=logging.INFO)

# =================================================================================
# == Core Node Class                                                           ==
# =================================================================================

class EnhancedAnnotationNode:
    """
    MD: Enhanced Annotation

    A highly customizable node for adding visual documentation (notes, headers,
    reminders) directly into the ComfyUI workflow graph. Supports various styles,
    colors, fonts, and markdown. It does not perform any computations and acts
    purely as a visual aid, with no impact on generation performance.
    """

    @classmethod
    def INPUT_TYPES(cls):
        """
        Define node inputs with detailed tooltips.
        NO TYPE HINTS ALLOWED HERE.
        """
        # Define color options consistently
        bg_colors = ["yellow", "blue", "green", "pink", "purple", "orange",
                     "red", "cyan", "gray", "white", "black", "mint",
                     "lavender", "peach", "lime", "sky"]
        text_colors = ["black", "white", "gray", "dark_gray"]

        return {
            "required": {
                "title": ("STRING", {
                    "multiline": False, "default": "",
                    "tooltip": (
                        "TITLE (Optional)\n"
                        "- A short header displayed prominently at the top.\n"
                        "- Leave blank for no title.\n"
                        "- Useful for section labels (e.g., 'Audio Generation', 'Image Upscaling')."
                    )
                }),
                "content": ("STRING", {
                    "multiline": True, "default": "Add notes here...\n- Supports Markdown!\n- Supports Emojis! ✨",
                    "tooltip": (
                        "CONTENT\n"
                        "- The main body of your annotation text.\n"
                        "- Supports multiple lines.\n"
                        "- Supports basic Markdown (bold, italics, lists - requires frontend support).\n"
                        "- Supports Emojis for visual cues."
                    )
                }),
                "style": (["sticky_note", "banner", "callout", "minimal", "header"], {
                    "default": "sticky_note",
                    "tooltip": (
                        "VISUAL STYLE\n"
                        "- Controls the overall appearance of the node.\n"
                        "- sticky_note: Classic yellow note look.\n"
                        "- banner: Wide, suitable for section titles across the graph.\n"
                        "- callout: Box with border for emphasis.\n"
                        "- minimal: Plain text, blends with background.\n"
                        "- header: Large, bold text, focused on the title."
                    )
                }),
                "font_size": (["tiny", "small", "normal", "large", "x-large", "huge"], {
                    "default": "normal",
                    "tooltip": (
                        "FONT SIZE\n"
                        "- Adjusts the text size for readability.\n"
                        "- tiny (8pt), small (10pt), normal (12pt), large (16pt), x-large (24pt), huge (36pt).\n"
                        "- Choose based on the amount of text and desired emphasis."
                    )
                }),
                "width": (["compact", "normal", "wide", "extra-wide"], {
                    "default": "normal",
                    "tooltip": (
                        "NODE WIDTH\n"
                        "- Controls the horizontal size of the annotation node.\n"
                        "- compact (~200px): For small notes in tight spaces.\n"
                        "- normal (~300px): Default width.\n"
                        "- wide (~450px): Good for detailed explanations.\n"
                        "- extra-wide (~600px): Suitable for banners spanning sections."
                    )
                }),
                "bg_color": (bg_colors, {
                    "default": "yellow",
                    "tooltip": (
                        "BACKGROUND COLOR\n"
                        "- Sets the background color of the annotation.\n"
                        "- Use different colors to visually categorize sections or highlight important notes."
                    )
                }),
                "text_color": (text_colors, {
                    "default": "black",
                    "tooltip": (
                        "TEXT COLOR\n"
                        "- Sets the color of the title and content text.\n"
                        "- Choose a color that contrasts well with the selected background color for readability."
                    )
                }),
            }
            # No optional inputs for this node
        }

    # This node has no outputs, so RETURN_TYPES and RETURN_NAMES are empty tuples
    RETURN_TYPES = ()
    RETURN_NAMES = ()
    OUTPUT_NODE = True # Indicates this is a terminal node in terms of data flow
    FUNCTION = "annotate"
    CATEGORY = "MD_Nodes/Workflow Organization" # Corrected category

    # No IS_CHANGED method: This node should use default ComfyUI caching.
    # Its state is determined entirely by its inputs. If inputs change,
    # ComfyUI will re-run it (which does nothing computationally but signals
    # the frontend that the node's visual representation might need an update).

    def annotate(self, title, content, style, font_size, width, bg_color, text_color):
        """
        Execution function for the annotation node.
        This function intentionally performs no operations. Its sole purpose is
        to exist for ComfyUI's execution graph. The node's value is purely visual.

        Args:
            (All args match INPUT_TYPES definition)

        Returns:
            An empty tuple `()`, matching RETURN_TYPES.
        """
        try:
            # Log execution for debugging purposes if needed
            # Use f-string for safer formatting if title might contain special chars
            logger.debug(f"EnhancedAnnotationNode executed (visual only). Title='{str(title)[:50]}...' Style='{style}'")

            # Intentionally do nothing else.
            pass

        except Exception as e:
            # Log any unexpected errors during this minimal execution
            logger.error(f"[EnhancedAnnotationNode] Unexpected error during execution (should be no-op): {e}", exc_info=True)
            # Still need to return the correct type even on error
            pass # Fall through to the return statement

        # Return an empty tuple as defined in RETURN_TYPES
        return ()

# =================================================================================
# == Node Registration                                                         ==
# =================================================================================

NODE_CLASS_MAPPINGS = {
    "EnhancedAnnotationNode": EnhancedAnnotationNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EnhancedAnnotationNode": "MD: Enhanced Annotation", # Added MD: prefix, removed emoji
}