# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
# ████ MD_Nodes/SceneGeniusAutocreator – LLM-powered workflow automation ████▓▒░
# © 2025 MDMAchine
# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
# ░▒▓ ORIGIN & DEV:
#   • Cast into the void by: MDMAchine
#   • Enhanced by: Gemini, Claude
#   • License: Apache 2.0 — No cursed floppy disks allowed

# ░▒▓ DESCRIPTION:
#   Leverages local LLMs (Ollama/LM Studio) to automate Ace-Step diffusion
#   content generation. Generates genres, adaptive lyrics, durations, and tuned
#   YAML configurations for APG/Sampler/NoiseDecay based on a core concept.

# ░▒▓ FEATURES:
#   ✓ Automates generation of genre tags, lyrics/instrumental tag, duration.
#   ✓ Generates tailored YAML configurations for APG, PingPong Sampler, Noise Decay.
#   ✓ Supports Ollama and LM Studio backends with model selection and URL config.
#   ✓ Quality presets (Draft to Max) or manual step control.
#   ✓ Fine-tuning controls: tag boosting, vocal weight, excluded tags, force lyrics.
#   ✓ Seed control with optional randomization and widget validation.
#   ✓ Robust error handling, API retries, URL validation, and fallback mechanisms.

# ░▒▓ CHANGELOG:
#   - v1.4.2 (Guideline Update - Oct 2025):
#       • REFACTOR: Full compliance update to v1.4.2 guidelines.
#       • CRITICAL: Removed all type hints from function signatures (Section 6.2).
#       • STYLE: Standardized imports, docstrings, and error handling.
#       • STYLE: Replaced all print() calls with logging (Section 6.3).
#       • STYLE: Rewrote all tooltips to new standard format (Section 8.1).
#       • ROBUST: Wrapped main execution in try/except with fallback outputs (Section 7.3).
#       • CACHE: Added IS_CHANGED method with widget validation for randomize_seed (Section 9.1).
#       • STYLE: Updated node display name.
#   - v1.0.1 (Final Polish):
#       • FIXED: Included previously omitted methods & applied final code review suggestions.
#       • FIXED: Separated model list cache timestamps.
#       • ADDED: Centralized try/except block with fallback outputs.
#   - v1.0.0 (Professional Refactor):
#       • REFACTORED: Comprehensive code review, added initial docstrings.
#       • SECURITY: Added URL validation for API endpoints.
#       • PERFORMANCE: Implemented lazy-loading for model lists.

# ░▒▓ CONFIGURATION:
#   → Primary Use: Generating all creative inputs and technical YAML configs for an Ace-Step workflow from a single concept.
#   → Secondary Use: Using override inputs to manually set specific outputs.
#   → Edge Use: Running in `test_mode` to quickly test workflow connections without actual LLM calls.

# ░▒▓ WARNING:
#   This node may trigger:
#   ▓▒░ Arguing with your LLM about whether "Symphonic Glitch Hop" is a real genre.
#   ▓▒░ The cold realization that the AI writes better YAML than you do.
#   ▓▒░ Existential dread when the LLM decides your dark techno concept needs polka lyrics.
#   ▓▒░ Flashbacks to wrestling with dial-up scripts just to connect to the local BBS.
#   Consult your nearest demoscene vet if hallucinations persist.
# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄


# =================================================================================
# == Standard Library Imports                                                    ==
# =================================================================================
import logging
import re
import time
import random
import secrets
import traceback
from urllib.parse import urlparse

# =================================================================================
# == Third-Party Imports                                                         ==
# =================================================================================
import requests
import yaml

# =================================================================================
# == ComfyUI Core Modules                                                        ==
# =================================================================================
# (No core ComfyUI imports needed directly in this file)

# =================================================================================
# == Local Project Imports                                                       ==
# =================================================================================
# (No local project imports in this file)

# =================================================================================
# == Helper Classes & Dependencies                                               ==
# =================================================================================

class CustomDumper(yaml.Dumper):
    """Custom YAML Dumper for better formatting of short lists."""
    def represent_sequence(self, tag, data, flow_style=None):
        """Represent sequences with flow style for short lists."""
        if len(data) <= 2 and all(isinstance(x, (int, float)) for x in data):
            return super().represent_sequence(tag, data, flow_style=True)
        return super().represent_sequence(tag, data, flow_style=flow_style)

# =================================================================================
# == Core Node Class                                                             ==
# =================================================================================

class SceneGeniusAutocreator:
    """
    A multi-stage ComfyUI node leveraging local LLMs for dynamic creative content
    generation (genres, lyrics, duration) and technical configuration (YAMLs).
    """

    # --- Constants ---
    QUALITY_PRESETS = { "Draft": 60, "Low": 120, "Basic": 180, "Standard": 220, "Medium": 360, "High": 420, "Very High": 500, "Enhanced": 580, "Ultra": 720, "Max": 1000 }
    DEFAULT_OLLAMA_MODEL = "llama3:8b-instruct-q8_0"
    DEFAULT_LM_STUDIO_MODEL = "local-model"
    DEFAULT_OLLAMA_URL = "http://localhost:11434"
    DEFAULT_LM_STUDIO_URL = "http://localhost:1234"

    MAX_RETRY_ATTEMPTS = 3
    API_TIMEOUT_SECONDS = 300
    MODEL_LIST_CACHE_SECONDS = 300 # 5 minutes

    APG_MIN_CFG = 2.0
    APG_MAX_CFG = 8.0
    NOISE_DECAY_MIN = 0.0
    NOISE_DECAY_MAX = 10.0

    # --- Caching ---
    _ollama_models_cache = None
    _lm_studio_models_cache = None
    _ollama_cache_timestamp = 0.0
    _lm_studio_cache_timestamp = 0.0

    # --- LLM PROMPTS ---
    PROMPT_GENRE_GENERATION_BASE = """You are a highly creative AI assistant specializing in music production and audio design. Your task is to generate exactly {tag_count} descriptive elements based on the concept: "{initial_concept_prompt}".

**CRITICAL REQUIREMENTS:**
- The FIRST tag MUST be an actual music genre (e.g., "trap", "ambient", "house", "dnb", "techno", "hip-hop", "downtempo", "breakbeat")
- The remaining tags should be descriptive production techniques and musical elements
- Include relatable terms like "stutter beat", "chopped vocals", "piano section", "analog warmth", "filtered bass"
- Focus on elements that paint a clear sonic picture

{excluded_tags_instruction}
The output MUST ONLY be the comma-separated tags with NO extra text or explanations.
Output:
"""
    PROMPT_LYRICS_GENERATION_BASE = """You are a creative writer and lyricist. Your task is to write lyrics for a creative project.

**Project Details:**
- **Initial Concept:** "{initial_concept_prompt}"
- **Generated Genres:** "{genre_tags}"

**Decision Logic:**
- If `force_instrumental` is `True`, you MUST output exactly: `[instrumental]`
- If `force_lyrics_generation` is `True`, you MUST generate lyrics.
- Otherwise, you may decide if an instrumental piece is more appropriate.

**CRITICAL LYRIC GENERATION RULES:**
1.  **Structure:** If generating lyrics, use this structure:
    [Verse 1]
    (4 lines)
    [Chorus]
    (4 lines)
    [Verse 2]
    (4 lines)
    [Bridge]
    (4 lines)
    [Chorus]
    (4 lines)
    [Outro]
    (2-4 lines)
2.  Each section tag MUST be on its own line
3.  Output ONLY the lyrics or `[instrumental]`

{force_lyrics_instruction}{force_instrumental_instruction}
Generate your output.
Output:
"""
    PROMPT_DURATION_GENERATION_BASE = """Based on the concept "{initial_concept_prompt}", genres "{genre_tags}", and content "{lyrics_or_script}", provide a duration in seconds between {min_total_seconds:.1f} and {max_total_seconds:.1f}.

Output ONLY the number.
"""
    PROMPT_NOISE_DECAY_GENERATION_BASE = """Based on the concept "{initial_concept_prompt}", provide a noise decay strength between {min_noise_decay:.1f} and {max_noise_decay:.1f}.

Output ONLY the number.
"""
    PROMPT_APG_STYLE_CHOICE = """Based on the concept "{initial_concept_prompt}", is the target primarily Audio or Image? Respond with ONLY the word "Audio" or "Image"."""
    PROMPT_APG_AUDIO = """Based on concept "{initial_concept_prompt}" and genres "{genre_tags}", provide a single number for APG guidance scale (2.0 to 6.0). A balanced value is 4.5.

Output ONLY the number:
"""
    PROMPT_APG_IMAGE = """Based on concept "{initial_concept_prompt}", provide two values:
1. APG Scale (2.0-8.0)
2. Momentum (-0.95 to 0.95, negative for sharp, positive for smooth)

Output ONLY two numbers separated by a comma (e.g., 5.2,-0.4):
"""
    PROMPT_SAMPLER_FBG = """Based on concept "{initial_concept_prompt}" and genres "{genre_tags}", provide three values for FBG sampler:
1. t_0 (0.7-0.9, higher = more sensitive)
2. cfg_scale (500-800, higher = more aggressive)
3. max_guidance_scale (800-1200, guidance ceiling)

Output ONLY three numbers separated by commas (e.g., 0.85,650,1000):
"""
    PROMPT_SAMPLER_ORIGINAL = """Based on concept "{initial_concept_prompt}", choose the best noise_behavior preset: "Default (Raw)", "Dynamic", "Smooth", "Textured Grain", or "Soft (DDIM-Like)".

Output ONLY the preset name:
"""
    PROMPT_NOISE_DECAY_ALGORITHM = """Based on the concept "{initial_concept_prompt}", choose the best algorithm: polynomial, sigmoidal, gaussian, fourier, or exponential.

Output ONLY the algorithm name:
"""
    PROMPT_NOISE_DECAY_EXPONENT = """For {algo_choice} algorithm with concept "{initial_concept_prompt}", provide decay_exponent (1.0-8.0, higher = steeper).

Output ONLY the number:
"""

    def __init__(self):
        """Initialize default YAML configurations."""
        self.default_apg_yaml_audio = """
verbose: true
rules:
  - start_sigma: -1.0
    apg_scale: 0.0
    cfg: 7.0
  - start_sigma: 0.8
    apg_scale: 4.5
    predict_image: true
    cfg: 5.0
    mode: pure_apg
    update_blend_mode: lerp
    dims: [-1]
    momentum: 0.6
    norm_threshold: 2.5
  - start_sigma: 0.4
    apg_scale: 2.0
    predict_image: true
    cfg: 3.0
    mode: pure_apg
    dims: [-1]
    momentum: 0.6
  - start_sigma: 0.2
    apg_scale: 0.0
    cfg: 1.0
"""
        self.default_apg_yaml_image = """
verbose: true
rules:
  - start_sigma: -1.0
    apg_scale: 0.0
    cfg: 7.0
  - start_sigma: 14.0
    apg_scale: 4.5
    predict_image: true
    cfg: 5.0
    mode: pure_apg
    dims: [-1, -2]
    momentum: 0.0
    norm_threshold: 2.5
    eta: 0.0
  - start_sigma: 1.5
    apg_scale: 0.0
    cfg: 3.0
"""
        self.default_sampler_yaml_fbg = """
step_random_mode: "block"
step_size: 5
first_ancestral_step: 0
last_ancestral_step: -1
ancestral_noise_type: "gaussian"
start_sigma_index: 0
end_sigma_index: -1
enable_clamp_output: false
blend_mode: "cubic"
step_blend_mode: "cubic"
conditional_blend_mode: false
conditional_blend_sigma_threshold: 0.3
conditional_blend_function_name: "lerp"
conditional_blend_on_change: false
conditional_blend_change_threshold: 0.2
progressive_blend_mode: false
clamp_noise_norm: false
max_noise_norm: 1.5
adaptive_noise_scaling: false
noise_scale_factor: 1.0
debug_mode: 0 # Changed from 2 to 0 for default
enable_profiling: false # Changed from true
checkpoint_steps: ""
log_posterior_ema_factor: 0.3
fbg_eta: 0.5
fbg_s_noise: 0.8
tensor_memory_optimization: false
fbg_config:
  t_0: 0.85
  t_1: 0.3
  temp: 0.0011
  offset: -0.05
  cfg_scale: 650
  initial_guidance_scale: 250
  pi: 0.35
  sampler_mode: "PINGPONG"
  cfg_start_sigma: 1.0
  cfg_end_sigma: 0.004
  fbg_start_sigma: 1.0
  fbg_end_sigma: 0.004
  ancestral_start_sigma: 1.0
  ancestral_end_sigma: 0.004
  max_guidance_scale: 1000.0
  max_posterior_scale: 3.0
  initial_value: 0.0
  fbg_guidance_multiplier: 1.2
  guidance_max_change: 1000.0
"""
        self.default_sampler_yaml_original = """
noise_behavior: Custom
ancestral_strength: 0.9
noise_coherence: 0.75
step_random_mode: block
step_size: 2
seed: 4242
first_ancestral_step: 0
last_ancestral_step: -3
start_sigma_index: 0
end_sigma_index: -1
enable_clamp_output: false
"""
        self.default_noise_decay_yaml = """
algorithm_type: polynomial
decay_exponent: 5.0
start_value: 1.0
end_value: 0.0
invert_curve: false
enable_temporal_smoothing: true
smoothing_window: 3
"""

    @classmethod
    def INPUT_TYPES(cls):
        """Define all input parameters with standardized tooltips."""
        return {
            "required": {
                "llm_backend": (["ollama", "lm_studio"], {
                    "default": "ollama",
                    "tooltip": (
                        "LLM BACKEND\n"
                        "- Select the local LLM server backend to use.\n"
                        "- 'ollama': Connects to Ollama server.\n"
                        "- 'lm_studio': Connects to LM Studio (OpenAI compatible API)."
                    )
                }),
                "ollama_api_base_url": ("STRING", {
                    "default": cls.DEFAULT_OLLAMA_URL,
                    "tooltip": (
                        "OLLAMA API URL\n"
                        "- The base URL for your running Ollama server API.\n"
                        "- Default: http://localhost:11434"
                    )
                }),
                "ollama_model_name": (cls._get_ollama_models_lazy(), {
                    "default": cls.DEFAULT_OLLAMA_MODEL,
                    "tooltip": (
                        "OLLAMA MODEL\n"
                        "- Select the specific model hosted by your Ollama server.\n"
                        "- List populated dynamically (may require restart if server not running)."
                    )
                }),
                "lm_studio_api_base_url": ("STRING", {
                    "default": cls.DEFAULT_LM_STUDIO_URL,
                    "tooltip": (
                        "LM STUDIO API URL\n"
                        "- The base URL for your running LM Studio server.\n"
                        "- Uses an OpenAI-compatible API endpoint.\n"
                        "- Default: http://localhost:1234"
                    )
                }),
                "lm_studio_model_name": (cls._get_lm_studio_models_lazy(), {
                    "default": cls.DEFAULT_LM_STUDIO_MODEL,
                    "tooltip": (
                        "LM STUDIO MODEL\n"
                        "- Select the specific model loaded in your LM Studio server.\n"
                        "- List populated dynamically (may require restart if server not running)."
                    )
                }),
                "initial_concept_prompt": ("STRING", {
                    "multiline": True,
                    "default": "A dystopian cyberpunk future with a retro-futuristic soundscape.",
                    "tooltip": (
                        "INITIAL CONCEPT\n"
                        "- The core creative idea that drives all generation stages.\n"
                        "- Be descriptive for best results."
                    )
                }),
                "tag_count": ("INT", {
                    "default": 4, "min": 1, "max": 10,
                    "tooltip": (
                        "TAG COUNT\n"
                        "- The number of descriptive genre/style tags the LLM should generate.\n"
                        "- The first tag is always a core genre."
                    )
                }),
                "excluded_tags": ("STRING", {
                    "default": "",
                    "tooltip": (
                        "EXCLUDED TAGS\n"
                        "- A comma-separated list of tags to prevent the LLM from generating.\n"
                        "- Example: 'ambient, slow'"
                    )
                }),
                "force_lyrics_generation": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "FORCE LYRICS\n"
                        "- If True, forces the LLM to generate lyrics.\n"
                        "- Overrides the LLM's decision to output '[instrumental]' (unless Force Instrumental is True)."
                    )
                }),
                "force_instrumental": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "FORCE INSTRUMENTAL\n"
                        "- If True, forces the output to be exactly '[instrumental]'.\n"
                        "- Takes precedence over 'Force Lyrics'."
                    )
                }),
                "min_total_seconds": ("FLOAT", {
                    "default": 60.0, "min": 0.0, "max": 3600.0,
                    "tooltip": (
                        "MIN DURATION (SECONDS)\n"
                        "- The minimum possible duration for the generated content.\n"
                        "- The LLM's output will be clamped to this minimum."
                    )
                }),
                "max_total_seconds": ("FLOAT", {
                    "default": 300.0, "min": 0.0, "max": 3600.0,
                    "tooltip": (
                        "MAX DURATION (SECONDS)\n"
                        "- The maximum possible duration for the generated content.\n"
                        "- The LLM's output will be clamped to this maximum."
                    )
                }),
                "quality_preset": (list(cls.QUALITY_PRESETS.keys()) + ["Manual"], {
                    "default": "Standard",
                    "tooltip": (
                        "QUALITY PRESET (STEPS)\n"
                        "- Select a quality level to determine the number of sampling steps.\n"
                        "- 'Manual': Use the 'Manual Steps' input.\n"
                        "- Higher presets = more steps = slower generation."
                    )
                }),
                "manual_steps": ("INT", {
                    "default": 500, "min": 1, "max": 8192, "step": 1,
                    "tooltip": (
                        "MANUAL STEPS\n"
                        "- Set the total number of sampling steps manually.\n"
                        "- Only used when 'Quality Preset' is set to 'Manual'."
                    )
                }),
                "base_tag_boost": ("FLOAT", {
                    "default": 2.0, "min": 0.0, "max": 10.0, "step": 0.1,
                    "tooltip": (
                        "BASE TAG BOOST\n"
                        "- Sets the conditioning boost strength for the base genre tags.\n"
                        "- Connects to AceT5ConditioningScheduled node."
                    )
                }),
                "vocal_tag_boost": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1,
                    "tooltip": (
                        "VOCAL TAG BOOST\n"
                        "- Sets the conditioning boost strength for the extracted vocal tags.\n"
                        "- Connects to AceT5ConditioningScheduled node."
                    )
                }),
                "vocal_weight": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 0.95, "step": 0.01,
                    "tooltip": (
                        "VOCAL WEIGHT\n"
                        "- Sets the weight for blending vocal characteristics during conditioning.\n"
                        "- Connects to AceT5ConditioningScheduled node."
                    )
                }),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xffffffffffffffff,
                    "tooltip": (
                        "SEED\n"
                        "- The seed for all random operations (if 'Randomize Seed' is off).\n"
                        "- Use the same seed for reproducible results."
                    )
                }),
                "randomize_seed": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "RANDOMIZE SEED\n"
                        "- If True, generates a new random seed on each run, ignoring the 'Seed' input.\n"
                        "- Recommended: Set to True for exploration, False for reproducibility."
                    )
                }),
                "sampler_version": (["FBG Integrated PingPong Sampler", "Original PingPong Sampler"], {
                    "default": "FBG Integrated PingPong Sampler",
                    "tooltip": (
                        "SAMPLER VERSION\n"
                        "- Selects which PingPong sampler's YAML configuration to generate.\n"
                        "- Choose based on which sampler node you are using downstream."
                    )
                }),
                "generate_noise_decay_yaml": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "GENERATE NOISE DECAY YAML\n"
                        "- If True, the LLM generates a config for the Noise Decay Scheduler.\n"
                        "- If False, a default YAML configuration is used."
                    )
                }),
            },
            "optional": {
                "genre_input": ("STRING", {
                    "default": "", "multiline": True,
                    "tooltip": (
                        "MANUAL GENRE INPUT\n"
                        "- Manually provide genre tags here.\n"
                        "- If 'Genre Mixed Mode' is off, this overrides the LLM entirely."
                    )
                }),
                "genre_mixed_mode": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "GENRE MIXED MODE\n"
                        "- If True, your 'Manual Genre Input' will be prepended to the tags generated by the LLM."
                    )
                }),
                "prompt_vocal_tags_generation": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": (
                        "OVERRIDE: VOCAL TAGS\n"
                        "- Advanced: Manually provide vocal tags (e.g., 'ethereal vocal chops').\n"
                        "- Overrides the automatic extraction from generated genre tags."
                    )
                }),
                "prompt_lyrics_decision_and_generation": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": (
                        "OVERRIDE: LYRICS/SCRIPT\n"
                        "- Advanced: Manually provide the full lyrics or '[instrumental]'.\n"
                        "- Overrides the LLM's entire lyrics generation stage."
                    )
                }),
                "prompt_duration_generation": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": (
                        "OVERRIDE: DURATION\n"
                        "- Advanced: Manually provide the duration in seconds (e.g., '120.0').\n"
                        "- Overrides the LLM's duration generation stage."
                    )
                }),
                "prompt_noise_decay_generation": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": (
                        "OVERRIDE: NOISE DECAY STRENGTH\n"
                        "- Advanced: Manually provide the noise decay strength (e.g., '4.5').\n"
                        "- Overrides the LLM's noise decay strength generation."
                    )
                }),
                "prompt_apg_yaml_generation": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": (
                        "OVERRIDE: APG YAML\n"
                        "- Advanced: Manually provide the complete APG YAML configuration.\n"
                        "- Overrides the LLM's APG YAML generation."
                    )
                }),
                "prompt_sampler_yaml_generation": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": (
                        "OVERRIDE: SAMPLER YAML\n"
                        "- Advanced: Manually provide the complete Sampler YAML configuration.\n"
                        "- Overrides the LLM's Sampler YAML generation."
                    )
                }),
                "test_mode": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "TEST MODE (DEV)\n"
                        "- If True, bypasses all LLM calls and returns dummy data.\n"
                        "- Use for quickly testing workflow connections without waiting for LLM."
                    )
                })
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "FLOAT", "INT", "INT", "INT", "FLOAT", "FLOAT", "STRING", "STRING", "STRING", "STRING", "INT",)
    RETURN_NAMES = ("GENRE_TAGS", "LYRICS_OR_SCRIPT", "TOTAL_SECONDS", "TOTAL_STEPS", "BASE_TAG_BOOST", "VOCAL_TAG_BOOST", "VOCAL_WEIGHT", "NOISE_DECAY_STRENGTH", "VOCAL_TAGS", "APG_YAML_PARAMS", "SAMPLER_YAML_PARAMS", "NOISE_DECAY_YAML_PARAMS", "SEED",)
    FUNCTION = "execute"
    CATEGORY = "MD_Nodes/Workflow Automation" # Changed Category

    @classmethod
    def IS_CHANGED(cls, randomize_seed=False, **kwargs):
        """
        Control cache behavior, making node dynamic if seed is randomized.
        Includes widget validation for `randomize_seed`.

        Args:
            randomize_seed (bool or str): Flag to randomize seed. Handles widget corruption.
            **kwargs: All other node inputs (ignored for hash).

        Returns:
            str: Random token if dynamic, "static" if cacheable.
        """
        # Normalize widget value (handle cache corruption)
        # Check specifically for boolean True, or the string "True" case-insensitively
        is_randomize_true = False
        if isinstance(randomize_seed, bool):
            is_randomize_true = randomize_seed
        elif isinstance(randomize_seed, str):
            is_randomize_true = randomize_seed.lower() == "true" # Handle string "True"

        if is_randomize_true:
            # Force re-run if randomization enabled
            return secrets.token_hex(16)

        # Allow caching otherwise (depends on other inputs changing)
        # Note: API calls make this effectively dynamic anyway, but this handles the seed part explicitly.
        return "static"


    @classmethod
    def _is_ollama_cache_valid(cls):
        """Checks if the Ollama model list cache is still valid."""
        return (time.time() - cls._ollama_cache_timestamp) < cls.MODEL_LIST_CACHE_SECONDS

    @classmethod
    def _is_lm_studio_cache_valid(cls):
        """Checks if the LM Studio model list cache is still valid."""
        return (time.time() - cls._lm_studio_cache_timestamp) < cls.MODEL_LIST_CACHE_SECONDS

    @classmethod
    def _get_ollama_models_lazy(cls):
        """
        Lazy-loads Ollama models, using a cache to avoid blocking UI.

        Returns:
            list: List of available Ollama model names.
        """
        if cls._ollama_models_cache and cls._is_ollama_cache_valid():
            return cls._ollama_models_cache
        try:
            response = requests.get(f"{cls.DEFAULT_OLLAMA_URL}/api/tags", timeout=5)
            response.raise_for_status()
            models_data = response.json()
            models = [model["name"] for model in models_data.get("models", [])]
            cls._ollama_models_cache = models if models else [cls.DEFAULT_OLLAMA_MODEL]
            cls._ollama_cache_timestamp = time.time()
            return cls._ollama_models_cache
        except requests.RequestException:
            logging.warning("[SceneGenius] Ollama server not found at default URL. Using default model name.")
            return [f"{cls.DEFAULT_OLLAMA_MODEL} (Ollama not found)"]

    @classmethod
    def _get_lm_studio_models_lazy(cls):
        """
        Lazy-loads LM Studio models, using a cache to avoid blocking UI.

        Returns:
            list: List of available LM Studio model names.
        """
        if cls._lm_studio_models_cache and cls._is_lm_studio_cache_valid():
            return cls._lm_studio_models_cache
        try:
            response = requests.get(f"{cls.DEFAULT_LM_STUDIO_URL}/v1/models", timeout=5)
            response.raise_for_status()
            models_data = response.json()
            models = [model["id"] for model in models_data.get("data", [])]
            cls._lm_studio_models_cache = models if models else [cls.DEFAULT_LM_STUDIO_MODEL]
            cls._lm_studio_cache_timestamp = time.time()
            return cls._lm_studio_models_cache
        except requests.RequestException:
            logging.warning("[SceneGenius] LM Studio server not found at default URL. Using default model name.")
            return [f"{cls.DEFAULT_LM_STUDIO_MODEL} (LM Studio not found)"]

    def _validate_url(self, url):
        """
        Validates that a URL has a valid scheme and network location.

        Args:
            url (str): The URL to validate.

        Returns:
            bool: True if the URL is valid, False otherwise.
        """
        try:
            result = urlparse(url)
            return all([result.scheme in ['http', 'https'], result.netloc])
        except (ValueError, AttributeError):
            return False

    def _clean_llm_output(self, raw_output):
        """
        Cleans raw LLM output by removing thought tags and extracting content from code blocks.

        Args:
            raw_output (str): The raw string output from the LLM.

        Returns:
            str: The cleaned string.
        """
        cleaned = re.sub(r'<think>.*?</think>', '', raw_output, flags=re.DOTALL | re.IGNORECASE)

        backticks = "```"
        code_block_pattern = rf'{backticks}(?:yaml|output|python)?\s*(.*?)\s*{backticks}'

        if match := re.search(code_block_pattern, cleaned, re.DOTALL):
            cleaned = match.group(1)

        return '\n'.join(line.rstrip() for line in cleaned.strip().splitlines() if line.strip())

    def _call_llm_api(self, backend, api_url, model_name, prompt):
        """
        Unified and validated API call method for different LLM backends.

        Args:
            backend (str): The LLM backend ('ollama' or 'lm_studio').
            api_url (str): The base URL for the API.
            model_name (str): The name of the model to use.
            prompt (str): The prompt to send to the LLM.

        Returns:
            str: The cleaned response from the LLM.

        Raises:
            ValueError: If the API URL is invalid or the backend is unsupported.
            RuntimeError: If the API call fails after multiple retries.
        """
        if not self._validate_url(api_url):
            error_message = f"Invalid or disallowed API URL provided: {api_url}"
            logging.error(f"[SceneGenius] {error_message}")
            raise ValueError(error_message)

        if backend == "ollama":
            return self._call_ollama_api(api_url, model_name, prompt)
        elif backend == "lm_studio":
            return self._call_lm_studio_api(api_url, model_name, prompt)
        else:
            raise ValueError(f"Unsupported LLM backend: {backend}")

    def _call_lm_studio_api(self, api_url, model_name, prompt):
        """
        Calls the LM Studio API with retry logic.

        Args:
            api_url (str): The LM Studio API base URL.
            model_name (str): The model name.
            prompt (str): The prompt.

        Returns:
            str: The cleaned LLM response.
        """
        logging.debug(f"[SceneGenius] Calling LM Studio API: {api_url} with model {model_name}")
        for attempt in range(self.MAX_RETRY_ATTEMPTS):
            try:
                payload = { "model": model_name, "messages": [{"role": "user", "content": prompt}], "temperature": 0.7, "max_tokens": 2048, "stream": False }
                headers = {"Content-Type": "application/json"}
                response = requests.post(f"{api_url}/v1/chat/completions", json=payload, headers=headers, timeout=self.API_TIMEOUT_SECONDS)
                response.raise_for_status()
                response_data = response.json()
                content = response_data["choices"][0]["message"]["content"].strip()
                logging.debug("[SceneGenius] LM Studio API call successful.")
                return self._clean_llm_output(content)
            except (requests.RequestException, KeyError, IndexError) as e:
                logging.warning(f"[SceneGenius] LM Studio API call failed (Attempt {attempt + 1}/{self.MAX_RETRY_ATTEMPTS}): {e}. Retrying...")
                time.sleep(2 ** attempt)
        raise RuntimeError("LM Studio API call failed after multiple retries.")

    def _call_ollama_api(self, api_url, model_name, prompt):
        """
        Calls the Ollama API with retry logic.

        Args:
            api_url (str): The Ollama API base URL.
            model_name (str): The model name.
            prompt (str): The prompt.

        Returns:
            str: The cleaned LLM response.
        """
        logging.debug(f"[SceneGenius] Calling Ollama API: {api_url} with model {model_name}")
        payload = {"model": model_name, "prompt": prompt, "stream": False, "keep_alive": 0}
        for attempt in range(self.MAX_RETRY_ATTEMPTS):
            try:
                response = requests.post(f"{api_url}/api/generate", json=payload, timeout=self.API_TIMEOUT_SECONDS)
                response.raise_for_status()
                raw_response = response.json().get("response", "").strip()
                logging.debug("[SceneGenius] Ollama API call successful.")
                return self._clean_llm_output(raw_response)
            except (requests.RequestException, KeyError) as e:
                logging.warning(f"[SceneGenius] Ollama API call failed (Attempt {attempt + 1}/{self.MAX_RETRY_ATTEMPTS}): {e}. Retrying...")
                time.sleep(2 ** attempt)
        raise RuntimeError("Ollama API call failed after multiple retries.")

    def _extract_vocal_tags(self, genre_tags):
        """
        Separates vocal-related tags from genre tags using whole-word matching.

        Args:
            genre_tags (str): Comma-separated string of genre tags.

        Returns:
            tuple: (vocal_tags_str, non_vocal_tags_str)
        """
        vocal_keywords = [
            'vocal', 'vocals', 'voice', 'singing', 'lyrics', 'choir', 'chant',
            'opera', 'ad-lib', 'male', 'female', 'chorus', 'yell', 'spoken',
            'rap', 'mc', 'acapella', 'harmony', 'beatbox', 'scat', 'whisper',
            'scream', 'growl'
        ]

        vocal_keywords.sort(key=len, reverse=True) # Prevent substring issues
        vocal_pattern = re.compile(r'\b(' + '|'.join(vocal_keywords) + r')\b', re.IGNORECASE)

        tags_list = [tag.strip() for tag in genre_tags.split(',')]
        vocal_tags, non_vocal_tags = [], []

        for tag in tags_list:
            if vocal_pattern.search(tag):
                vocal_tags.append(tag)
            else:
                non_vocal_tags.append(tag)

        return ", ".join(vocal_tags), ", ".join(non_vocal_tags)

    def _generate_genres(self, llm_backend, api_url, model_name, initial_concept_prompt, tag_count, excluded_tags, genre_input, genre_mixed_mode):
        """
        Generates genre tags, handling user overrides and mixed mode.

        Args:
            (various): Parameters from the execute method.

        Returns:
            str: Comma-separated genre tags.
        """
        logging.info("[SceneGenius] Stage 1: Generating genre tags...")
        user_input = genre_input.strip() if genre_input else ""

        if user_input and not genre_mixed_mode:
            logging.info(f"[SceneGenius] Using user-provided genre input exclusively: '{user_input}'")
            return user_input

        try:
            prompt = self.PROMPT_GENRE_GENERATION_BASE.format(
                tag_count=tag_count,
                initial_concept_prompt=initial_concept_prompt,
                excluded_tags_instruction=f"Avoid generating any of these tags: {excluded_tags}." if excluded_tags else ""
            )
            llm_tags = self._call_llm_api(llm_backend, api_url, model_name, prompt)

            if user_input and genre_mixed_mode:
                logging.info(f"[SceneGenius] Mixed mode: Prepending user input '{user_input}' to LLM tags.")
                return f"{user_input}, {llm_tags}"
            else:
                logging.info(f"[SceneGenius] Generated genre tags: {llm_tags}")
                return llm_tags
        except (ValueError, RuntimeError) as e:
            logging.error(f"[SceneGenius] Genre generation failed: {e}. Falling back...")
            return user_input if user_input else "ambient" # Fallback

    def _generate_lyrics(self, llm_backend, api_url, model_name, initial_concept_prompt, genre_tags, force_lyrics, force_instrumental, override):
        """
        Generates lyrics or an [instrumental] tag.

        Args:
            (various): Parameters from the execute method.
            override (str): Manual override for lyrics.

        Returns:
            str: Lyrics string or '[instrumental]'.
        """
        logging.info("[SceneGenius] Stage 2: Generating lyrics/script...")
        if override and override.strip():
             logging.info("[SceneGenius] Using provided lyrics override.")
             return override
        if force_instrumental:
             logging.info("[SceneGenius] Forced instrumental.")
             return "[instrumental]"
        try:
            prompt = self.PROMPT_LYRICS_GENERATION_BASE.format(
                initial_concept_prompt=initial_concept_prompt,
                genre_tags=genre_tags,
                force_lyrics_instruction="The user has set `force_lyrics_generation` to True." if force_lyrics else "",
                force_instrumental_instruction="" # Force instrumental handled above
            )
            lyrics = self._call_llm_api(llm_backend, api_url, model_name, prompt)
            logging.info(f"[SceneGenius] Generated lyrics/script output: {'[instrumental]' if lyrics.strip().lower() == '[instrumental]' else 'Lyrics generated'}")
            return lyrics
        except (ValueError, RuntimeError) as e:
            logging.error(f"[SceneGenius] Lyrics generation failed: {e}. Setting to instrumental.")
            return "[instrumental]" # Fallback

    def _generate_duration(self, llm_backend, api_url, model_name, initial_concept_prompt, genre_tags, lyrics, min_sec, max_sec, override):
        """
        Generates a duration in seconds for the content.

        Args:
            (various): Parameters from the execute method.
            override (str): Manual override for duration.

        Returns:
            float: Duration in seconds, clamped within min/max bounds.
        """
        logging.info("[SceneGenius] Stage 3: Generating duration...")
        if override and override.strip():
            try:
                total_seconds = float(override)
                logging.info(f"[SceneGenius] Using user-provided duration: {total_seconds}s")
                return max(min_sec, min(total_seconds, max_sec))
            except ValueError:
                logging.warning(f"[SceneGenius] Could not parse duration override '{override}'. Using LLM.")
        try:
            prompt = self.PROMPT_DURATION_GENERATION_BASE.format(min_total_seconds=min_sec, max_total_seconds=max_sec, initial_concept_prompt=initial_concept_prompt, genre_tags=genre_tags, lyrics_or_script=lyrics)
            response = self._call_llm_api(llm_backend, api_url, model_name, prompt)
            match = re.search(r'(\d+\.?\d*)', response)
            total_seconds = float(match.group(1)) if match else min_sec
            clamped_seconds = max(min_sec, min(total_seconds, max_sec))
            logging.info(f"[SceneGenius] Generated duration: {clamped_seconds:.1f}s")
            return clamped_seconds
        except (ValueError, RuntimeError, AttributeError) as e:
            logging.warning(f"[SceneGenius] Duration generation failed: {e}. Using min duration: {min_sec}s.")
            return min_sec # Fallback

    def _generate_noise_decay_strength(self, llm_backend, api_url, model_name, initial_concept_prompt, genre_tags, lyrics, total_seconds, override):
        """
        Generates a noise decay strength value.

        Args:
            (various): Parameters from the execute method.
            override (str): Manual override for noise decay strength.

        Returns:
            float: Noise decay strength, clamped within bounds.
        """
        logging.info("[SceneGenius] Stage 4: Generating noise decay strength...")
        if override and override.strip():
            try:
                strength = float(override)
                logging.info(f"[SceneGenius] Using user-provided noise decay strength: {strength}")
                return max(self.NOISE_DECAY_MIN, min(strength, self.NOISE_DECAY_MAX))
            except ValueError:
                logging.warning(f"[SceneGenius] Could not parse noise decay override '{override}'. Using LLM.")
        try:
            prompt = self.PROMPT_NOISE_DECAY_GENERATION_BASE.format(min_noise_decay=self.NOISE_DECAY_MIN, max_noise_decay=self.NOISE_DECAY_MAX, initial_concept_prompt=initial_concept_prompt, genre_tags=genre_tags, lyrics_or_script=lyrics, total_seconds=total_seconds)
            response = self._call_llm_api(llm_backend, api_url, model_name, prompt)
            match = re.search(r'(\d+\.?\d*)', response)
            strength = float(match.group(1)) if match else 5.0
            clamped_strength = max(self.NOISE_DECAY_MIN, min(strength, self.NOISE_DECAY_MAX))
            logging.info(f"[SceneGenius] Generated noise decay strength: {clamped_strength:.1f}")
            return clamped_strength
        except (ValueError, RuntimeError, AttributeError) as e:
            logging.warning(f"[SceneGenius] Noise decay generation failed: {e}. Using default: 5.0.")
            return 5.0 # Fallback

    def _configure_apg_yaml(self, llm_backend, api_url, model_name, initial_concept_prompt, genre_tags, override):
        """
        Generates APG YAML parameters.

        Args:
            (various): Parameters from the execute method.
            override (str): Manual override for APG YAML.

        Returns:
            str: APG YAML configuration string.
        """
        logging.info("[SceneGenius] Stage 5: Generating APG YAML Parameters.")
        if override and override.strip():
            logging.info("[SceneGenius] Using provided APG YAML override.")
            return override
        try:
            style_prompt = self.PROMPT_APG_STYLE_CHOICE.format(initial_concept_prompt=initial_concept_prompt, genre_tags=genre_tags)
            style_choice = self._call_llm_api(llm_backend, api_url, model_name, style_prompt)
            logging.info(f"[SceneGenius] APG target type determined as: {style_choice}")

            if "audio" in style_choice.lower():
                yaml_config = yaml.safe_load(self.default_apg_yaml_audio)
                prompt = self.PROMPT_APG_AUDIO.format(initial_concept_prompt=initial_concept_prompt, genre_tags=genre_tags)
                raw_response = self._call_llm_api(llm_backend, api_url, model_name, prompt)
                if match := re.search(r'(\d+\.?\d*)', raw_response):
                    apg_scale = max(2.0, min(6.0, float(match.group(1))))
                    logging.info(f"[SceneGenius] Generated APG Audio scale: {apg_scale:.2f}")
                    if 'rules' in yaml_config and len(yaml_config['rules']) >= 3:
                        yaml_config['rules'][1]['apg_scale'] = apg_scale
                        yaml_config['rules'][2]['apg_scale'] = max(1.0, apg_scale - 1.5)
                else:
                    logging.warning("[SceneGenius] Could not parse APG audio scale from LLM. Using default.")
            else: # Assume Image
                yaml_config = yaml.safe_load(self.default_apg_yaml_image)
                prompt = self.PROMPT_APG_IMAGE.format(initial_concept_prompt=initial_concept_prompt)
                raw_response = self._call_llm_api(llm_backend, api_url, model_name, prompt)
                values = [v.strip() for v in raw_response.split(',')]
                if len(values) >= 2:
                    try:
                        apg_scale = max(self.APG_MIN_CFG, min(self.APG_MAX_CFG, float(values[0])))
                        momentum = max(-0.95, min(0.95, float(values[1])))
                        logging.info(f"[SceneGenius] Generated APG Image scale: {apg_scale:.2f}, momentum: {momentum:.2f}")
                        if 'rules' in yaml_config and len(yaml_config['rules']) > 1:
                            yaml_config['rules'][1]['apg_scale'] = apg_scale
                            yaml_config['rules'][1]['momentum'] = momentum
                    except ValueError:
                         logging.warning("[SceneGenius] Could not parse APG image values from LLM. Using default.")
                else:
                    logging.warning("[SceneGenius] Insufficient APG image values from LLM. Using default.")

            return yaml.dump(yaml_config, Dumper=CustomDumper, indent=2, sort_keys=False)
        except (ValueError, RuntimeError, yaml.YAMLError) as e:
            logging.warning(f"[SceneGenius] APG YAML generation failed: {e}. Using default audio YAML.")
            return self.default_apg_yaml_audio # Fallback

    def _configure_sampler_yaml(self, llm_backend, api_url, model_name, initial_concept_prompt, genre_tags, sampler_version, override):
        """
        Generates Sampler YAML parameters.

        Args:
            (various): Parameters from the execute method.
            override (str): Manual override for Sampler YAML.

        Returns:
            str: Sampler YAML configuration string.
        """
        logging.info(f"[SceneGenius] Stage 6: Generating Sampler YAML for '{sampler_version}'.")
        if override and override.strip():
            logging.info("[SceneGenius] Using provided Sampler YAML override.")
            return override
        try:
            if sampler_version == "FBG Integrated PingPong Sampler":
                prompt = self.PROMPT_SAMPLER_FBG.format(initial_concept_prompt=initial_concept_prompt, genre_tags=genre_tags)
                raw_response = self._call_llm_api(llm_backend, api_url, model_name, prompt)
                values = [v.strip() for v in raw_response.split(',')]
                if len(values) >= 3:
                    try:
                        t_0 = max(0.7, min(0.9, float(values[0])))
                        cfg_scale = max(500, min(800, int(float(values[1]))))
                        max_guidance_scale = max(800, min(1200, int(float(values[2]))))
                        logging.info(f"[SceneGenius] Generated FBG params: t_0={t_0:.2f}, cfg_scale={cfg_scale}, max_guidance={max_guidance_scale}")

                        yaml_config = yaml.safe_load(self.default_sampler_yaml_fbg)
                        yaml_config['fbg_config'].update({
                            't_0': t_0, 't_1': max(0.2, t_0 - 0.4), 'cfg_scale': cfg_scale,
                            'initial_guidance_scale': int(cfg_scale * 0.4),
                            'max_guidance_scale': max_guidance_scale
                        })
                        return yaml.dump(yaml_config, Dumper=CustomDumper, indent=2, sort_keys=False)
                    except ValueError:
                         logging.warning("[SceneGenius] Could not parse FBG values from LLM. Using default.")
                         return self.default_sampler_yaml_fbg
                else:
                    logging.warning("[SceneGenius] Insufficient FBG values from LLM. Using default.")
                    return self.default_sampler_yaml_fbg
            elif sampler_version == "Original PingPong Sampler":
                prompt = self.PROMPT_SAMPLER_ORIGINAL.format(initial_concept_prompt=initial_concept_prompt)
                behavior_choice = self._call_llm_api(llm_backend, api_url, model_name, prompt).strip()
                logging.info(f"[SceneGenius] Generated Original Sampler behavior: {behavior_choice}")
                valid_behaviors = ["Default (Raw)", "Dynamic", "Smooth", "Textured Grain", "Soft (DDIM-Like)"]
                if behavior_choice not in valid_behaviors:
                    logging.warning(f"[SceneGenius] Invalid behavior '{behavior_choice}'. Using 'Custom'.")
                    behavior_choice = "Custom"
                yaml_config = yaml.safe_load(self.default_sampler_yaml_original)
                yaml_config['noise_behavior'] = behavior_choice
                return yaml.dump(yaml_config, Dumper=CustomDumper, indent=2, sort_keys=False)
            else:
                 logging.error(f"[SceneGenius] Unknown sampler version '{sampler_version}'. Using FBG default.")
                 return self.default_sampler_yaml_fbg
        except (ValueError, RuntimeError, yaml.YAMLError) as e:
            logging.warning(f"[SceneGenius] Sampler YAML generation failed: {e}. Using default.")

        # Fallback based on selected version
        return self.default_sampler_yaml_fbg if sampler_version == "FBG Integrated PingPong Sampler" else self.default_sampler_yaml_original

    def _configure_noise_decay_yaml(self, llm_backend, api_url, model_name, initial_concept_prompt):
        """
        Generates Noise Decay Scheduler YAML parameters.

        Args:
            (various): Parameters from the execute method.

        Returns:
            str: Noise Decay YAML configuration string.
        """
        logging.info("[SceneGenius] Stage 7: Generating Noise Decay Scheduler YAML.")
        try:
            prompt = self.PROMPT_NOISE_DECAY_ALGORITHM.format(initial_concept_prompt=initial_concept_prompt)
            algo_choice = self._call_llm_api(llm_backend, api_url, model_name, prompt).strip().lower()
            logging.info(f"[SceneGenius] Generated Noise Decay algorithm: {algo_choice}")
            valid_algorithms = ["polynomial", "sigmoidal", "gaussian", "fourier", "exponential"]
            if algo_choice not in valid_algorithms:
                logging.warning(f"[SceneGenius] Invalid algorithm '{algo_choice}'. Using 'polynomial'.")
                algo_choice = "polynomial"

            yaml_config = yaml.safe_load(self.default_noise_decay_yaml)
            yaml_config['algorithm_type'] = algo_choice

            if algo_choice in ["polynomial", "sigmoidal", "exponential", "gaussian"]:
                param_prompt = self.PROMPT_NOISE_DECAY_EXPONENT.format(algo_choice=algo_choice, initial_concept_prompt=initial_concept_prompt)
                raw_response = self._call_llm_api(llm_backend, api_url, model_name, param_prompt)
                if match := re.search(r'(\d+\.?\d*)', raw_response):
                    try:
                        exponent = max(1.0, min(8.0, float(match.group(1))))
                        yaml_config['decay_exponent'] = exponent
                        logging.info(f"[SceneGenius] Generated Decay exponent: {exponent:.1f}")
                    except ValueError:
                         logging.warning("[SceneGenius] Could not parse decay exponent from LLM. Using default.")
                else:
                     logging.warning("[SceneGenius] Could not find decay exponent value in LLM response. Using default.")

            return yaml.dump(yaml_config, Dumper=CustomDumper, indent=2, sort_keys=False)
        except (ValueError, RuntimeError, yaml.YAMLError) as e:
            logging.warning(f"[SceneGenius] Noise Decay YAML generation failed: {e}. Using default.")
            return self.default_noise_decay_yaml # Fallback

    def _get_fallback_outputs(self, seed, total_steps):
        """
        Returns safe default outputs when a critical generation stage fails.

        Args:
            seed (int): The current seed.
            total_steps (int): The determined total steps.

        Returns:
            tuple: A tuple containing all return values with safe defaults.
        """
        logging.warning("[SceneGenius] Returning fallback outputs due to a critical failure.")
        return ("ambient", "[instrumental]", 120.0, total_steps, 2, 1, 0.5, 5.0, "", self.default_apg_yaml_audio, self.default_sampler_yaml_fbg, self.default_noise_decay_yaml, seed)

    def execute(self, **kwargs):
        """
        Main execution method for the node. Orchestrates LLM calls and processing.

        Args:
            **kwargs: All input parameters defined in INPUT_TYPES.

        Returns:
            tuple: A tuple containing all output values defined in RETURN_TYPES.
        """
        logging.info("[SceneGenius] --- Starting SceneGeniusAutocreator ---")

        # --- Unpack kwargs with safe defaults ---
        llm_backend = kwargs.get("llm_backend", "ollama")
        ollama_api_base_url = kwargs.get("ollama_api_base_url", self.DEFAULT_OLLAMA_URL)
        ollama_model_name = kwargs.get("ollama_model_name", self.DEFAULT_OLLAMA_MODEL)
        lm_studio_api_base_url = kwargs.get("lm_studio_api_base_url", self.DEFAULT_LM_STUDIO_URL)
        lm_studio_model_name = kwargs.get("lm_studio_model_name", self.DEFAULT_LM_STUDIO_MODEL)
        initial_concept_prompt = kwargs.get("initial_concept_prompt", "A serene landscape")
        tag_count = kwargs.get("tag_count", 4)
        excluded_tags = kwargs.get("excluded_tags", "")
        force_lyrics_generation = kwargs.get("force_lyrics_generation", False)
        force_instrumental = kwargs.get("force_instrumental", False)
        min_total_seconds = kwargs.get("min_total_seconds", 60.0)
        max_total_seconds = kwargs.get("max_total_seconds", 300.0)
        quality_preset = kwargs.get("quality_preset", "Standard")
        manual_steps = kwargs.get("manual_steps", 500)
        base_tag_boost = kwargs.get("base_tag_boost", 2.0)
        vocal_tag_boost = kwargs.get("vocal_tag_boost", 1.0)
        vocal_weight = kwargs.get("vocal_weight", 0.5)
        seed = kwargs.get("seed", 0)
        randomize_seed = kwargs.get("randomize_seed", True) # Keep default True
        sampler_version = kwargs.get("sampler_version", "FBG Integrated PingPong Sampler")
        generate_noise_decay_yaml = kwargs.get("generate_noise_decay_yaml", False)
        genre_input = kwargs.get('genre_input', "")
        genre_mixed_mode = kwargs.get('genre_mixed_mode', False)
        prompt_overrides = {k: v for k, v in kwargs.items() if k.startswith('prompt_')}
        test_mode = kwargs.get('test_mode', False)

        # --- Seed and Step Logic (with widget validation) ---
        is_randomize_true = False
        if isinstance(randomize_seed, bool):
            is_randomize_true = randomize_seed
        elif isinstance(randomize_seed, str):
            # Handle potential string "True" or "False" from cache corruption
            is_randomize_true = randomize_seed.lower() == "true"
        else:
            # Handle unexpected types by defaulting to False (use provided seed)
             logging.warning(f"[SceneGenius] Unexpected type for 'randomize_seed': {type(randomize_seed)}. Using provided seed.")

        if is_randomize_true:
            seed = random.randint(0, 0xffffffffffffffff)
            logging.info(f"[SceneGenius] Generated random seed: {seed}")
        else:
            logging.info(f"[SceneGenius] Using provided seed: {seed}")

        total_steps = manual_steps if quality_preset == "Manual" else self.QUALITY_PRESETS.get(quality_preset, 220)
        logging.info(f"[SceneGenius] Quality preset '{quality_preset}' set to {total_steps} steps")

        # --- Backend Setup ---
        api_url, model_name = (ollama_api_base_url, ollama_model_name) if llm_backend == "ollama" else (lm_studio_api_base_url, lm_studio_model_name)
        logging.info(f"[SceneGenius] Using backend: {llm_backend}, Model: {model_name}, URL: {api_url}")

        if test_mode:
            logging.info("[SceneGenius] Test mode enabled. Returning dummy output.")
            return ("electro, synthwave", "[Verse 1]\n...", 180.0, total_steps, int(base_tag_boost), int(vocal_tag_boost), vocal_weight, 3.5, "ethereal vocals", self.default_apg_yaml_audio, self.default_sampler_yaml_fbg, self.default_noise_decay_yaml, seed)

        # --- Main Generation Pipeline (Wrapped for Fallback) ---
        try:
            raw_genre_tags = self._generate_genres(llm_backend, api_url, model_name, initial_concept_prompt, tag_count, excluded_tags, genre_input, genre_mixed_mode)
            lyrics_or_script = self._generate_lyrics(llm_backend, api_url, model_name, initial_concept_prompt, raw_genre_tags, force_lyrics_generation, force_instrumental, prompt_overrides.get("prompt_lyrics_decision_and_generation", ""))

            is_instrumental = False
            if force_instrumental:
                logging.info("[SceneGenius] Instrumental mode: Forced by user flag.")
                is_instrumental = True
            elif lyrics_or_script.strip().lower() == '[instrumental]':
                logging.info("[SceneGenius] Instrumental mode: Detected from LLM output.")
                is_instrumental = True

            vocal_override = prompt_overrides.get("prompt_vocal_tags_generation", "")
            if is_instrumental:
                final_vocal_tags, remaining_genre_tags = "", raw_genre_tags
                logging.info("[SceneGenius] Instrumental mode active, clearing vocal tags.")
            elif vocal_override and vocal_override.strip():
                final_vocal_tags = vocal_override
                _, remaining_genre_tags = self._extract_vocal_tags(raw_genre_tags) # Still need non-vocal
                logging.info(f"[SceneGenius] Using user-provided vocal override: '{final_vocal_tags}'")
            else:
                final_vocal_tags, remaining_genre_tags = self._extract_vocal_tags(raw_genre_tags)
                logging.info(f"[SceneGenius] Extracted vocal tags: '{final_vocal_tags}'. Remaining genres: '{remaining_genre_tags}'")

            total_seconds = self._generate_duration(llm_backend, api_url, model_name, initial_concept_prompt, raw_genre_tags, lyrics_or_script, min_total_seconds, max_total_seconds, prompt_overrides.get("prompt_duration_generation", ""))
            noise_decay_strength = self._generate_noise_decay_strength(llm_backend, api_url, model_name, initial_concept_prompt, raw_genre_tags, lyrics_or_script, total_seconds, prompt_overrides.get("prompt_noise_decay_generation", ""))
            apg_yaml_params = self._configure_apg_yaml(llm_backend, api_url, model_name, initial_concept_prompt, raw_genre_tags, prompt_overrides.get("prompt_apg_yaml_generation", ""))
            sampler_yaml_params = self._configure_sampler_yaml(llm_backend, api_url, model_name, initial_concept_prompt, raw_genre_tags, sampler_version, prompt_overrides.get("prompt_sampler_yaml_generation", ""))
            noise_decay_yaml_params = self._configure_noise_decay_yaml(llm_backend, api_url, model_name, initial_concept_prompt) if generate_noise_decay_yaml else self.default_noise_decay_yaml

            logging.info("[SceneGenius] --- SceneGeniusAutocreator execution complete ---")

            # Ensure boost values are integers as per RETURN_TYPES
            base_tag_boost_int = int(round(base_tag_boost))
            vocal_tag_boost_int = int(round(vocal_tag_boost))

            # Must return tuple matching RETURN_TYPES
            return (remaining_genre_tags, lyrics_or_script, total_seconds, total_steps, base_tag_boost_int, vocal_tag_boost_int, vocal_weight, noise_decay_strength, final_vocal_tags, apg_yaml_params, sampler_yaml_params, noise_decay_yaml_params, seed)

        except Exception as e:
            logging.error(f"[SceneGenius] A critical failure occurred during generation: {e}")
            logging.debug(traceback.format_exc())
            print(f"[SceneGenius] ⚠️ Error encountered, returning fallback outputs. Check console/logs.")
            # Return neutral, valid fallback output (Passthrough Pattern)
            return self._get_fallback_outputs(seed, total_steps)

# =================================================================================
# == Node Registration                                                           ==
# =================================================================================

NODE_CLASS_MAPPINGS = {
    "SceneGeniusAutocreator": SceneGeniusAutocreator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SceneGeniusAutocreator": "MD: Scene Genius Autocreator"
}