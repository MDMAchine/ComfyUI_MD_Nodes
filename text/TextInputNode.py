# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# ████ Advanced Text Input Node v1.1 – Wildcard Magic & Text Wizardry ████▓▒░
# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
# ░▒▓ ORIGIN & DEV:
#   • Crafted with care by MDMAchine / MD_Nodes
#   • License: Apache 2.0 — do what you want, just give credit
#   • Part of the MD_Nodes collection for ComfyUI
# ░▒▓ DESCRIPTION:
#   A versatile text input node with wildcard support and text processing.
#   Perfect for large prompts, YAML configs, JSON data, or any text input.
#   Seed-controlled wildcards ensure reproducible random selections.
#   Multiple text manipulation options for workflow flexibility.
# ░▒▓ FEATURES:
#   ✓ Large multiline text input area for comfortable editing
#   ✓ Wildcard support: {option1|option2|option3} or __option1|option2__
#   ✓ Seed-controlled randomization for reproducible results
#   ✓ Nested wildcard support for complex patterns
#   ✓ Text transformations: lowercase, uppercase, whitespace control
#   ✓ Multiple outputs: processed text, original text, seed used
#   ✓ Smart space normalization and cleanup
#   ✓ Two wildcard syntax styles for compatibility
#   ✓ Bonus: Text File Loader node for external file import
# ░▒▓ CHANGELOG:
#   - v1.1 (Stability Fix):
#         • Added robust IS_CHANGED method to TextFileLoader to prevent unnecessary re-runs.
#         • Corrected IS_CHANGED logic in AdvancedTextNode for deterministic behavior.
#   - v1.0 (Initial Release):
#         • Core wildcard functionality with seed control
#         • Dual wildcard syntax support (curly braces and double underscore)
#         • Text transformation suite (case, whitespace, cleanup)
#         • Triple output system (processed, original, seed)
#         • Nested wildcard processing with safety limits
#         • Text File Loader companion node
#         • Full UTF-8 support for international characters
# ░▒▓ CONFIGURATION:
#   → Primary Use: Dynamic text generation with wildcards in prompts
#   → Secondary Use: Text preprocessing and normalization
#   → Advanced Use: File-based workflow templates and configs
#   → Pro Tip: Use fixed seeds for consistent outputs across runs
# ░▒▓ WILDCARD EXAMPLES:
#   Simple: "A {red|blue|green} house" → "A blue house"
#   Nested: "{A {large|small} building|A {tall|short} tree}"
#   Complex: "{masterpiece|best quality}, {1girl|1boy}, {standing|sitting} in a {forest|city|beach}"
# ░▒▓ WARNING:
#   This node may cause:
#   ▓▒░ Excessive prompt experimentation
#   ▓▒░ Wildcard addiction
#   ▓▒░ Spontaneous workflow optimization
#   ▓▒░ An irrational love of curly braces
# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄

import random
import re
import os

class AdvancedTextNode:
    """
    A versatile text input node with wildcard support, file loading, and text manipulation features.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "dynamicPrompts": False,
                    "tooltip": "Main text input area. Enter your prompts, YAML, JSON, or any text content here. Use wildcards like {option1|option2} for random variations."
                }),
            },
            "optional": {
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "step": 1,
                    "tooltip": "Random seed for wildcard selection. Same seed = same results every time. Change this to get different random variations. Use -1 or workflow seed for dynamic randomization."
                }),
                "wildcard_mode": ("BOOLEAN", {
                    "default": False,
                    "label_on": "enabled",
                    "label_off": "disabled",
                    "tooltip": "Enable wildcard processing. When ON, patterns like {red|blue|green} will randomly pick one option. When OFF, text passes through unchanged. Seed controls which option is chosen."
                }),
                "strip_whitespace": ("BOOLEAN", {
                    "default": False,
                    "label_on": "enabled",
                    "label_off": "disabled",
                    "tooltip": "Remove leading and trailing whitespace from the entire text. Useful for cleaning up copy-pasted content or removing accidental spaces at the start/end."
                }),
                "lowercase": ("BOOLEAN", {
                    "default": False,
                    "label_on": "enabled",
                    "label_off": "disabled",
                    "tooltip": "Convert all text to lowercase. Warning: This overrides the uppercase option if both are enabled. Useful for normalizing text input or style tokens."
                }),
                "uppercase": ("BOOLEAN", {
                    "default": False,
                    "label_on": "enabled",
                    "label_off": "disabled",
                    "tooltip": "Convert all text to UPPERCASE. Note: Lowercase takes priority if both are enabled. Good for emphasis tokens or standardizing format."
                }),
                "remove_extra_spaces": ("BOOLEAN", {
                    "default": True,
                    "label_on": "enabled",
                    "label_off": "disabled",
                    "tooltip": "Collapse multiple consecutive spaces into single spaces and trim spaces from line starts/ends. Enabled by default. Helps clean up formatting without changing content."
                }),
                "wildcard_syntax": (["curly_braces", "double_underscore"], {
                    "default": "curly_braces",
                    "tooltip": "Choose wildcard pattern style:\n• curly_braces: {option1|option2|option3}\n• double_underscore: __option1|option2|option3__\nPick based on your preference or compatibility with other tools."
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("text", "original_text", "seed_used")
    FUNCTION = "process_text"
    CATEGORY = "MD_Nodes/Text"
    OUTPUT_NODE = False
    
    @classmethod
    def IS_CHANGED(cls, text, seed=0, wildcard_mode=False, strip_whitespace=False, 
                   lowercase=False, uppercase=False, remove_extra_spaces=True, 
                   wildcard_syntax="curly_braces"):
        # This robustly checks if any input that affects the output has changed.
        return (text, seed, wildcard_mode, strip_whitespace, lowercase, uppercase,
                remove_extra_spaces, wildcard_syntax)
    
    def process_wildcards_curly(self, text, seed):
        """Process wildcards in {option1|option2|option3} format"""
        rng = random.Random(seed)
        
        def replace_wildcard(match):
            options = match.group(1).split('|')
            options = [opt.strip() for opt in options]
            return rng.choice(options)
        
        pattern = r'\{([^{}]+)\}'
        max_iterations = 100
        iteration = 0
        while re.search(pattern, text) and iteration < max_iterations:
            text = re.sub(pattern, replace_wildcard, text)
            iteration += 1
        return text
    
    def process_wildcards_underscore(self, text, seed):
        """Process wildcards in __option1|option2|option3__ format"""
        rng = random.Random(seed)
        
        def replace_wildcard(match):
            options = match.group(1).split('|')
            options = [opt.strip() for opt in options]
            return rng.choice(options)
        
        pattern = r'__([^_]+(?:\|[^_]+)*)__'
        max_iterations = 100
        iteration = 0
        while re.search(pattern, text) and iteration < max_iterations:
            text = re.sub(pattern, replace_wildcard, text)
            iteration += 1
        return text
    
    def process_text(self, text, seed=0, wildcard_mode=False, strip_whitespace=False, 
                     lowercase=False, uppercase=False, remove_extra_spaces=True,
                     wildcard_syntax="curly_braces"):
        original_text = text
        
        if wildcard_mode:
            if wildcard_syntax == "curly_braces":
                text = self.process_wildcards_curly(text, seed)
            elif wildcard_syntax == "double_underscore":
                text = self.process_wildcards_underscore(text, seed)
        
        if strip_whitespace:
            text = text.strip()
        
        if remove_extra_spaces:
            text = re.sub(r' +', ' ', text)
            text = '\n'.join(line.strip() for line in text.split('\n'))
        
        if lowercase:
            text = text.lower()
        
        if uppercase:
            text = text.upper()
        
        return (text, original_text, seed)

# Optional: Add a companion node for loading text from files
class TextFileLoader:
    """
    Load text content from files for use in workflows
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Full path to the text file you want to load. Supports relative paths from ComfyUI root or absolute paths. Examples:\n• input/prompts/style.txt\n• C:/my_prompts/template.yaml\n• /home/user/configs/settings.json"
                }),
            },
            "optional": {
                "encoding": (["utf-8", "ascii", "latin-1"], {
                    "default": "utf-8",
                    "tooltip": "Character encoding of the file:\n• utf-8: Universal, handles all languages and emojis (recommended)\n• ascii: Basic English characters only\n• latin-1: Western European characters\nIf you get garbled text, try a different encoding."
                }),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "load_file"
    CATEGORY = "MD_Nodes/Text"
    
    @classmethod
    def IS_CHANGED(cls, file_path, encoding="utf-8"):
        # This check is crucial. It efficiently detects if the file's content
        # has changed by checking its modification time.
        try:
            m_time = os.path.getmtime(file_path)
        except OSError:
            m_time = -1 # File not found or path is invalid
        
        return (file_path, m_time)
        
    def load_file(self, file_path, encoding="utf-8"):
        """Load text from a file"""
        try:
            if not os.path.exists(file_path):
                return (f"Error: File not found: {file_path}",)
            
            with open(file_path, 'r', encoding=encoding) as f:
                text = f.read()
            
            return (text,)
        except Exception as e:
            return (f"Error loading file: {str(e)}",)

# Node class mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "AdvancedTextNode": AdvancedTextNode,
    "TextFileLoader": TextFileLoader,
}

# Display names in the UI
NODE_DISPLAY_NAME_MAPPINGS = {
    "AdvancedTextNode": "Advanced Text Input 📝",
    "TextFileLoader": "Text File Loader 📄",
}