# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# ████ SCENEGENIUS AUTOCREATOR v0.1.6 – Optimized for Ace-Step Audio/Video ████▓▒░
# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀

# ░▒▓ ORIGIN & DEV:
#   • Cast into the void by: MDMAchine (OG SysOp)
#   • Original mind behind SceneGenius Autocreator
#   • Initial ComfyUI adaptation by: Gemini (Google)
#   • Enhanced & refined by: MDMAchine & Gemini
#   • Critical optimizations & bugfixes: Gemini
#   • FBG Sampler integration & version selection by: Gemini (Google)
#   • Removed version numbers from sampler selection by: Gemini (Google)
#   • **FIXED: FBG Sampler default YAML on LLM failure.**
#   • Final polish: MDMAchine
#   • License: Apache 2.0 — No cursed floppy disks allowed

# ░▒▓ DESCRIPTION:
#   A multi-stage AI creative weapon node for ComfyUI.
#   Designed to automate Ace-Step diffusion content generation,
#   channeling the chaotic spirit of the demoscene and BBS era.
#   Produces authentic genres, adaptive lyrics, precise durations,
#   and finely tuned APG + Sampler configs with ease.
#   Now supports selection between original PingPong Sampler and FBG-integrated version,
#   without needing version number updates in the Scene Genius node itself.
#   May work with other models, but don't expect mercy or miracles.

# ░▒▓ FEATURES:
#   ✓ Local LLM integration via Ollama
#   ✓ Multi-stage generation pipeline (Genre → Lyrics/Script → APG/Sampler YAML)
#   ✓ Context-aware creativity with chat memory persistence
#   ✓ Intelligent duration calculation & adaptive output control
#   ✓ Dynamic noise decay adjustment for retro or polished looks
#   ✓ Robust error handling & fallback defaults
#   ✓ YAML override for advanced custom configurations
#   ✓ QoL LLM core parameter management (temperature, tokens, GPU layers)
#   ✓ Fixed multi-line YAML dims bug for clean outputs
#   ✓ Selectable PingPong Sampler version (Original vs. FBG Integrated, version-agnostic)
#   ✓ **Fixed FBG Sampler default YAML fallback.**
#   ✓ **Improved prompt formatting for LLM stability.**
#   ✓ **Added retry logic for Ollama API calls.**
#   ✓ **Enhanced YAML Dumper to prevent recursion errors.**
#   ✓ **Consistent YAML list formatting for 'dims' across APG and Sampler outputs.**
#   ✓ **Added validation for Sampler YAML overrides (e.g., FBG config presence).**
#   ✓ **FIXED: `AttributeError: 'SceneGeniusAutocreator' object has no attribute 'FBG_PROMPT_SECTION'` by reordering prompt definitions.**
#   ✓ **FIXED: `TypeError: FBGConfig.__new__() got an unexpected keyword argument 'log_posterior_ema_factor'` by clarifying Sampler YAML structure in prompt.**

# ░▒▓ CHANGELOG:
#   - v0.1.0 (Genesis Build):
#       • Concept & node skeleton established
#       • Local LLM research & integration
#       • Basic YAML schema & parsing implemented
#       • APG + Sampler parameter hookup
#   - v0.1.1 (Production Ready):
#       • Full multi-stage generation pipeline automated
#       • Enhanced context memory & genre/script intelligence
#       • Dynamic noise & sampler tuning enabled
#       • Robust error handling added
#       • Final polishing of YAML output and bug fixes
#   - v0.1.2 (Sampler Select):
#       • Added option to select between original PingPong Sampler (v0.8.15) and
#         FBG-integrated PingPong Sampler (v0.9.9) for YAML generation.
#       • Dynamic LLM prompting for Sampler YAML based on selected version.
#       • Updated default Sampler YAML to match FBG version's defaults.
#   - v0.1.3 (Version Agnostic Sampler Select):
#       • Removed explicit version numbers from the 'sampler_version' input to simplify
#         future maintenance and allow seamless updates to PingPong Sampler nodes.
#   - v0.1.4 (FBG Default Fix):
#       • Corrected the `default_sampler_yaml_fbg` to precisely match the user's
#         desired FBG YAML, ensuring proper fallback on LLM generation failure.
#   - v0.1.5 (Stability & Refinement):
#       • **FIXED: `SyntaxError: unterminated string literal` in prompt definitions.**
#       • **FIXED: `RecursionError` in `CustomDumper` by correctly calling superclass method.**
#       • Implemented exponential backoff retry logic for Ollama API calls to improve robustness against transient network/API issues.**
#       • Refactored prompt string formatting to handle conditional elements dynamically, preventing syntax errors.**
#       • Added validation to `prompt_sampler_yaml_generation` override to ensure consistency with selected `sampler_version` (e.g., checking for `fbg_config`).**
#       • Ensured consistent inline list formatting for `dims: [-2, -1]` in both APG and Sampler YAML outputs.**
#       • Improved logging for Ollama API calls and YAML parsing warnings.**
#       • **FIXED: `AttributeError: 'SceneGeniusAutocreator' object has no attribute 'FBG_PROMPT_SECTION'` by reordering prompt definitions.**
#   - v0.1.6 (Sampler YAML Structure Fix):
#       • **FIXED: `TypeError: FBGConfig.__new__() got an unexpected keyword argument 'log_posterior_ema_factor'` by explicitly instructing LLM on correct Sampler YAML structure (top-level vs. nested).**
#       • Updated `default_sampler_yaml_fbg` to match user's provided 'proper sampler yaml' for `guidance_max_change`, `fbg_offset`, `t_0`, and `t_1`.**

# ░▒▓ CONFIGURATION:
#   → Primary Use: Automated content generation for Ace-Step diffusion projects
#   → Secondary Use: Experimental demoscene-themed creative output
#   → Edge Use: Manual YAML override & parameter hacking for power users

# ░▒▓ WARNING:
#   This node may trigger:
#   ▓▒░ Temporal distortion
#   ▓▒░ Flashbacks to ANSI art & screaming modems
#   ▓▒░ Unstoppable urges to re-install Impulse Tracker
#   ▓▒░ Spontaneous creative anarchy

import logging
import yaml
import requests
import re # Import the re module for regular expressions
import time # Import time for retry logic
from typing import Dict, Any, Optional

# Set up logging for ComfyUI
logger = logging.getLogger(__name__)

# Assume 'comfy_utils' is a module that provides ComfyUI-specific utilities
# and that it has a function to get available Ollama models.
# This is a placeholder for actual ComfyUI integration.
try:
    import folder_paths # Standard ComfyUI import for paths
    # Attempt to import custom comfy_utils, if it exists
    try:
        from . import comfy_utils as s_actual
        # If import succeeds, use the actual comfy_utils
        s = s_actual
    except ImportError:
        logger.warning("Could not import comfy_utils. Mocking _get_ollama_models_list.")
        class MockComfyUtils:
            def _get_ollama_models_list(self):
                # Placeholder for actual model listing
                return ["llama3:8b-instruct-q8_0", "mistral:7b-instruct-v0.2"]
        s = MockComfyUtils()
except ImportError:
    logger.warning("Not running in ComfyUI environment. Mocking ComfyUI imports and _get_ollama_models_list.")
    class MockComfyUtils:
        def _get_ollama_models_list(self):
            return ["llama3:8b-instruct-q8_0", "mistral:7b-instruct-v0.2"]
    s = MockComfyUtils()
    # Mock folder_paths if not in ComfyUI environment
    class MockFolderPaths:
        def get_full_path(self, filename, subdir):
            return filename
    folder_paths = MockFolderPaths()


# Custom Dumper to preserve list formatting for specific cases
class CustomDumper(yaml.Dumper):
    def represent_sequence(self, tag, data, flow_style=None):
        # Preserve flow style for small numerical lists (e.g., dims: [-2, -1])
        if len(data) <= 2 and all(isinstance(x, (int, float)) for x in data):
            # Corrected: Call the parent class's represent_sequence to avoid infinite recursion
            return super().represent_sequence(tag, data, flow_style=True)
        # For all other lists, let the default behavior of yaml.Dumper handle it.
        # This means calling the parent's represent_sequence without forcing flow_style.
        # Corrected: Call the parent class's represent_sequence to avoid infinite recursion
        return super().represent_sequence(tag, data, flow_style)

class SceneGeniusAutocreator:
    """
    A multi-stage ComfyUI node leveraging local LLMs for dynamic creative content generation.
    Supports APG and Sampler YAML generation with robust error handling, logging, and extensibility.
    """

    def __init__(self):
        """Initialize default configurations once at class instantiation."""
        # Default APG YAML
        self.default_apg_yaml = """
verbose: true

rules:
  - start_sigma: -1
    apg_scale: 0.0
    cfg: 5.0

  - start_sigma: 0.85
    apg_scale: 5.0
    predict_image: true
    cfg: 3.7
    mode: pre_alt2
    update_blend_mode: lerp
    dims: [-2, -1]
    momentum: 0.65
    norm_threshold: 3.5
    eta: 0.0

  - start_sigma: 0.55
    apg_scale: 4.0
    predict_image: true
    cfg: 2.9
    mode: pure_apg
  - start_sigma: 0.4
    apg_scale: 3.8
    predict_image: true
    cfg: 2.75
    mode: pure_apg

  - start_sigma: 0.15
    apg_scale: 0.0
    cfg: 2.6
"""

        # Default Sampler YAML for original PingPong Sampler
        self.default_sampler_yaml_original = """
verbose: true
step_random_mode: block
step_size: 5
first_ancestral_step: 25
last_ancestral_step: 40
start_sigma_index: 0
end_sigma_index: -1
enable_clamp_output: false
blend_mode: lerp
step_blend_mode: lerp
"""

        # Updated Default Sampler YAML for FBG Integrated PingPong Sampler to match user's example
        self.default_sampler_yaml_fbg = """
step_random_mode: block
step_size: 5
first_ancestral_step: 0
last_ancestral_step: -1
enable_clamp_output: false
start_sigma_index: 0
end_sigma_index: -1
blend_mode: slerp
step_blend_mode: geometric_mean
debug_mode: 2
sigma_range_preset: Custom
conditional_blend_mode: true
conditional_blend_sigma_threshold: 0.4
conditional_blend_function_name: slerp
conditional_blend_on_change: true
conditional_blend_change_threshold: 0.15
clamp_noise_norm: false
max_noise_norm: 0.2
log_posterior_ema_factor: 0.3
fbg_config:
  fbg_sampler_mode: PINGPONG
  cfg_scale: 250.0
  cfg_start_sigma: 1.0
  cfg_end_sigma: 0.004
  fbg_start_sigma: 1.0
  fbg_end_sigma: 0.004
  fbg_guidance_multiplier: 1.0
  ancestral_start_sigma: 1.0
  ancestral_end_sigma: 0.004
  max_guidance_scale: 350.0
  max_posterior_scale: 1.0
  initial_value: 0.0
  initial_guidance_scale: 250.0
  guidance_max_change: 10.5 # Updated from 1.5
  fbg_temp: 0.010
  fbg_offset: -0.035 # Updated from -0.030
  pi: 0.35
  t_0: 0.60 # Updated from 0.90
  t_1: 0.30 # Updated from 0.60
fbg_eta: 0.5
fbg_s_noise: 0.7
"""

        # Base prompt for APG and Sampler generation (shared)
        self.APG_MIN_CFG = 2.5
        self.APG_MAX_CFG = 4.0

        # --- CORRECTED APG PROMPT ---
        self.DEFAULT_APG_PROMPT = """
You are an expert diffusion model parameter specialist. Your task is to generate APG YAML.

**Creative Context:**
- Initial Concept: "{initial_concept_prompt}"
- Generated Genres: "{genre_tags}"
- Lyrics/Script: "{lyrics_or_script}"
- Total Duration: {total_seconds} seconds
- Noise Decay: {noise_decay_strength}

**CRITICAL RULES FOR APG YAML GENERATION:**
1.  **Output Format:** The output MUST be a valid YAML string.
2.  **Top-Level Structure:** The YAML MUST contain only `verbose: true` at the top level, followed by a `rules` block.
3.  **Rules Block Structure:** The `rules` block MUST be a list of mappings (dictionaries).
4.  **Allowed Rule Parameters:** Each rule in the `rules` list MUST ONLY contain the following keys:
    * `start_sigma` (float): Must be strictly decreasing across rules.
    * `apg_scale` (float)
    * `cfg` (float): Must be between {min_cfg} and {max_cfg}.
    * `predict_image` (boolean): Set to `true` for active APG phases.
    * `mode` (string): e.g., `pre_alt2`, `pure_apg`.
    * `update_blend_mode` (string): e.g., `lerp`.
    * `dims` (list of integers): e.g., `[-2, -1]`. Should be formatted inline.
    * `momentum` (float)
    * `norm_threshold` (float)
    * `eta` (float)
5.  **Specific Rule Values:**
    * `apg_scale: 0.0` for the first and last rule.
    * `predict_image: true` for active APG phases.
6.  **No Extra Parameters:** DO NOT include any other top-level parameters (e.g., `apg_version`, `model_name`, `seed`, `total_duration`, `noise_decay`, `steps`, `cfg`, `apg_scale` at the top level). DO NOT include `phase`, `duration`, `end_sigma`, `strength` within the rules.
7.  **No Extra Text:** Output ONLY the YAML. Do NOT include any conversational filler, explanations, or markdown delimiters (e.g., ```yaml).

**Example of DESIRED APG YAML Structure:**
```yaml
verbose: true
rules:
  - start_sigma: -1
    apg_scale: 0.0
    cfg: 5.0

  - start_sigma: 0.85
    apg_scale: 5.0
    predict_image: true
    cfg: 3.7
    mode: pre_alt2
    update_blend_mode: lerp
    dims: [-2, -1]
    momentum: 0.65
    norm_threshold: 3.5
    eta: 0.0

  - start_sigma: 0.55
    apg_scale: 4.0
    predict_image: true
    cfg: 2.9
    mode: pure_apg
  - start_sigma: 0.4
    apg_scale: 3.8
    predict_image: true
    cfg: 2.75
    mode: pure_apg

  - start_sigma: 0.15
    apg_scale: 0.0
    cfg: 2.6
```
Output:
"""
        # --- MOVED: FBG_PROMPT_SECTION is now defined BEFORE DEFAULT_SAMPLER_PROMPT_BASE ---
        self.FBG_PROMPT_SECTION = """
---

**Feedback Guidance (FBG) Parameters (for FBG Integrated PingPong Sampler only):**
These parameters are for the `fbg_config` dictionary and top-level FBG inputs.

* **`fbg_config` (dictionary):** This nested dictionary holds the core FBG configuration.
    * **`fbg_sampler_mode` (string):** FBG's internal base sampler mode. `EULER` is standard, `PINGPONG` can also be used and has shown good results in unique instances. **Default: `EULER`**
    * **`cfg_scale` (float):** **The base Classifier-Free Guidance (CFG) scale that FBG dynamically modifies.** This is the starting point for FBG's adjustments. **Optimal performance is often seen with higher settings, from `2.0` up to `250.0`, with values "in the middle" like `8.0` through `12.0` being effective, and very high values like `250.0` for specific creative outcomes.**
    * **`cfg_start_sigma` (float):** The noise level (sigma) at which regular CFG (and thus FBG's influence over it) begins. **Default: `1.0`** (to cover typical model ranges).
    * **`cfg_end_sigma` (float):** The noise level (sigma) at which regular CFG ends. **Default: `0.004`** (to cover typical model ranges).
    * **`fbg_start_sigma` (float):** The noise level (sigma) at which FBG actively calculates and applies its dynamic scale. **Default: `1.0`** (to cover typical model ranges).
    * **`fbg_end_sigma` (float):** The noise level (sigma) at which FBG ceases its dynamic scale calculation. **Default: `0.004`**.
    * **`fbg_guidance_multiplier` (float):):** Multiplier for the FBG guidance component. **Default: `1.0`**.
    * **`ancestral_start_sigma` (float):** FBG internal: First sigma for ancestral sampling (for FBG's base sampler). **Default: `1.0`**. (`fbg_eta` must be >0 for this to have effect.)
    * **`ancestral_end_sigma` (float):** FBG internal: Last sigma for ancestral sampling (for FBG's base sampler). **Default: `0.004`**.
    * **`max_guidance_scale` (float):** Upper limit for the total guidance scale after FBG and CFG. **A good starting point is `22.0`, but can be set much higher (e.g., `350.0`) for intense guidance.**
    * **`max_posterior_scale` (float):** Limits the maximum posterior scale value. **Values from `1.0` to `7.0` can make an impact, with `1.0` to `4.0` often being ideal.**
    * **`initial_value` (float):** Initial value for FBG's internal log posterior estimate. **Default: `0.0`**.
    * **`initial_guidance_scale` (float):** Initial value for FBG's internal guidance scale. **Default: `1.0`, but can be set to match `cfg_scale` (e.g., `250.0`) for a strong starting point.**
    * **`guidance_max_change` (float):** Limits the fractional percentage change of FBG guidance scale per step. A value like `0.5` means max 50% change. **A value of `1.0` or `1.5` has shown good results, allowing for significant dynamic changes.**
    * **`fbg_temp` (float):** Temperature for FBG log posterior update. **Only applies if `t_0` and `t_1` are both `0.0` or are overridden.** **Default: `0.0`**.
    * **`fbg_offset` (float):** Offset for FBG log posterior update. **Only applies if `t_0` and `t_1` are both `0.0` or are overridden.** **Default: `0.0`**.
    * **`pi` (float):** ($\pi$) from FBG paper. Higher (e.g., `0.95-0.999`) for well-learned models. Lower (e.g., `0.2-0.8`) for general T2I. **A value of `0.25-0.35` has shown good results.**
    * **`t_0` (float):** Normalized diffusion time (0-1) where FBG guidance reaches reference. If `0.0`, `fbg_temp` and `fbg_offset` are used directly. **A value of `0.5` to `0.90` has shown good results.**
    * **`t_1` (float):** Normalized diffusion time (0-1) where FBG guidance reaches maximum. If `0.0`, `fbg_temp` and `fbg_offset` are used directly. **A value of `0.4` to `0.60` has shown good results.**
* **`log_posterior_ema_factor` (float):** Exponential Moving Average (EMA) factor for smoothing log posterior updates.
    * **Purpose:** Controls the smoothness of the log posterior estimate, affecting the stability of FBG guidance.
    * **Decision-Making:** A value between `0.0` (no smoothing) and `1.0` (maximum smoothing). `0.3` is a good starting point.
* **`fbg_eta` (float):** FBG internal parameter: Noise amount for ancestral sampling within FBG's step. **Default: `0.0`**. A value like `0.5` can introduce more dynamism.
* **`fbg_s_noise` (float):** FBG internal parameter: Scale for noise within FBG's step. **Default: `1.0`**. A value like `0.7` can introduce more dynamism.
"""

        # --- CORRECTED SAMPLER PROMPT BASE ---
        self.DEFAULT_SAMPLER_PROMPT_BASE = """
You are an expert diffusion model sampling and parameter configuration specialist. Your task is to generate YAML-formatted parameters for the Ping-Pong Sampler.

**Your understanding of the creative context:**
* **Initial Concept:** "{initial_concept_prompt}"
* **Generated Genres:** "{genre_tags}"
* **Lyrics/Script Status:** "{lyrics_or_script}"
* **Total Duration (seconds):** "{total_seconds}"
* **Generated APG Parameters (Crucial context for overall guidance strategy):** "{apg_yaml_params}"

**Your task: Generate a VALID YAML string containing parameters for the Ping-Pong Sampler. Be acutely aware of the impact of each parameter.**

**CRITICAL GUIDANCE FOR STABILITY AND QUALITY (Read Carefully!):**
For the Original PingPong Sampler, certain parameters (`start_sigma_index`, `end_sigma_index`, `blend_mode`, `step_blend_mode`) are **extremely sensitive and have a high likelihood of producing noisy, corrupted, or unfinished outputs if changed from their default values.** These parameters control fundamental aspects of the sampler's internal blending and sigma schedule traversal. **Therefore, for the Original PingPong Sampler, you MUST prioritize using the default values for these specific parameters unless there is an absolute, well-understood technical necessity for alteration (which is exceedingly rare and likely beyond typical creative adjustments).** When using the FBG Integrated PingPong Sampler, some of these parameters offer more flexibility as detailed in the FBG section. Focus your creative adjustments primarily on `step_random_mode`, `step_size`, `first_ancestral_step`, and `last_ancestral_step` for the Original Sampler, and leverage the FBG parameters for the FBG Sampler.

**Key Parameters & How to Think About Them (Common to both Sampler Versions, unless specified):**

* **Top-Level Parameters (MUST be directly under the root of the YAML):**
    * `verbose` (boolean)
    * `step_random_mode` (string)
    * `step_size` (integer)
    * `first_ancestral_step` (integer)
    * `last_ancestral_step` (integer)
    * `enable_clamp_output` (boolean)
    * `start_sigma_index` (integer)
    * `end_sigma_index` (integer)
    * `blend_mode` (string)
    * `step_blend_mode` (string)
    * `debug_mode` (integer)
    * `sigma_range_preset` (string)
    * `conditional_blend_mode` (boolean)
    * `conditional_blend_sigma_threshold` (float)
    * `conditional_blend_function_name` (string)
    * `conditional_blend_on_change` (boolean)
    * `conditional_blend_change_threshold` (float)
    * `clamp_noise_norm` (boolean)
    * `max_noise_norm` (float)
    * `log_posterior_ema_factor` (float)
    * `fbg_eta` (float)
    * `fbg_s_noise` (float)

* **Nested Parameter (`fbg_config` - MUST be a dictionary nested under the root):**
    * `fbg_config` (dictionary): This dictionary MUST contain ONLY the following keys:
        * `fbg_sampler_mode` (string)
        * `cfg_scale` (float)
        * `cfg_start_sigma` (float)
        * `cfg_end_sigma` (float)
        * `fbg_start_sigma` (float)
        * `fbg_end_sigma` (float)
        * `fbg_guidance_multiplier` (float)
        * `ancestral_start_sigma` (float)
        * `ancestral_end_sigma` (float)
        * `max_guidance_scale` (float)
        * `max_posterior_scale` (float)
        * `initial_value` (float)
        * `initial_guidance_scale` (float)
        * `guidance_max_change` (float)
        * `fbg_temp` (float)
        * `fbg_offset` (float)
        * `pi` (float)
        * `t_0` (float)
        * `t_1` (float)

**CRITICAL INSTRUCTION:** DO NOT put `log_posterior_ema_factor` or `fbg_eta` or `fbg_s_noise` inside the `fbg_config` block. They are top-level parameters.

{fbg_section}

**No Extra Text:** Output ONLY the YAML. Do NOT include any conversational filler, explanations, or markdown delimiters (e.g., ```yaml).
Output:
"""

        # Corrected: PROMPT_GENRE_GENERATION_BASE no longer contains embedded f-string logic
        self.PROMPT_GENRE_GENERATION_BASE = """
You are a highly creative AI assistant specializing in music and audio aesthetics.
Your task is to generate a single, comma-separated string of exactly {tag_count} descriptive genre tags.

These tags should {blend_description} the following concept: "{initial_concept_prompt}".

**CRITICAL GUIDANCE FOR TAG GENERATION:**
* **Focus on Real Music Genres:** All tags **must be actual, recognized music genres** (e.g., "electronic," "jazz," "hip-hop," "classical," "rock," "ambient," "synthwave," "lo-fi," "orchestral"). Do not invent genres or use non-music related descriptors as primary tags.
* **Promote Diversity & Avoid Repetition:** Strive for **good variety** among the generated tags. Avoid selecting tags that are overly redundant or extremely similar (e.g., don't just list "deep house, tech house, progressive house" if a broader range is possible). Explore different facets of the initial concept and genre associations to provide a rich and diverse set of descriptors.
* **Descriptive Modifiers (if tag_count > 1):**
    * If `{tag_count}` is greater than 1, you may include **one or more descriptive terms** (e.g., "soft female vocals," "energetic beat," "melancholic piano," "driving percussion," "atmospheric soundscape") alongside the music genres.
    * **Crucially, if `{tag_count}` is greater than 1, at least one of the tags MUST still be a core music genre.**
    * If `{tag_count}` is 1, the tag MUST be a core music genre.

{excluded_tags_instruction}
The output MUST ONLY be the comma-separated tags. DO NOT include any conversational filler, explanations, or additional text.
Example Output (for tag_count=3): "synthwave, energetic beat, retro-futurism"
Example Output (for tag_count=1): "ambient"
Output:
"""
        # Corrected: PROMPT_LYRICS_GENERATION_BASE no longer contains embedded f-string logic
        self.PROMPT_LYRICS_GENERATION_BASE = """
You are a creative writer and concept developer. Your task is to generate lyrics or a script for a creative project, or determine if it should be instrumental.

**Instructions:**
1.  **Consider the following project details:**
    * **Initial Concept:** "{initial_concept_prompt}" (This is the primary driver of content and narrative.)
    * **Generated Genres:** "{genre_tags}" (These tags define the aesthetic, mood, rhythmic feel, and typical lyrical themes associated with those genres. They should influence the *style* of writing, not its explicit subject matter.)
2.  **Decision Logic:**
    * If `force_lyrics_generation` is set to `True`, you MUST generate lyrics or a script based on the concept and genres.
    * If `force_lyrics_generation` is set to `False`, you have the discretion to decide if an instrumental piece is more appropriate for these genres and concept.
3.  **Genre Influence vs. Content:**
    * **Crucially, use the `Generated Genres` to inform the *style, vocabulary, tone, rhythm, and underlying themes* of the lyrics or script.**
    * **DO NOT explicitly mention any of the `genre_tags` themselves within the generated lyrics or script.** For example, if "house music" is a genre tag, the lyrics should *feel* like house music but should not say "this is a house music track." The content should be driven by the `Initial Concept`, infused with the `Genre Tags'` essence.
4.  **Output Format:**
    * If generating lyrics/script, provide the complete text.
    * If determining it should be instrumental, your output must ONLY be the word: `[instrumental]`
    * Do not include any conversational filler, explanations, or additional text.

**Example Output Format (Lyrics):**
[Verse 1]
Neon gleam on chrome streets,
Echoes of a forgotten beat.
Synths hum a dystopian tune,
Under a digital moon.

**Example Output Format (Instrumental):**
[instrumental]

Only output the lyrics or instrumental tag, nothing else.
Only use [verse], [chorus], [bridge], and [instrumental] as structure tags.

Genre Tags: "{genre_tags}"
Initial Concept: "{initial_concept_prompt}"
{force_lyrics_instruction}

Consider the creative brief and the genre.
Output:
"""
        self.PROMPT_DURATION_GENERATION_BASE = """
You are an AI assistant tasked with generating a duration for a creative project.
Based on the following information, provide a single floating-point number representing the total duration in seconds.

Initial Concept: "{initial_concept_prompt}"
Generated Genres: "{genre_tags}"
Lyrics/Script: "{lyrics_or_script}"

The duration MUST be between {min_total_seconds:.1f} and {max_total_seconds:.1f} seconds, inclusive.
Ensure your output is ONLY the floating-point number, with no extra text, explanations, or formatting.
Example Output: 180.5
Output:
"""
        self.NOISE_DECAY_MIN = 0.0
        self.NOISE_DECAY_MAX = 10.0
        self.PROMPT_NOISE_DECAY_GENERATION_BASE = """
You are an AI assistant tasked with generating a noise decay strength for image/video generation.
Based on the following creative concept, provide a single floating-point number between {min_noise_decay:.1f} and {max_noise_decay:.1f} (inclusive) representing the noise decay strength.
A higher value means more noise decay, leading to a "cleaner" or more "finished" look. A lower value retains more "raw" or "gritty" noise. Testing has shown a range of 1.5-4 to be ideal in most cases.

Initial Concept: "{initial_concept_prompt}"
Generated Genres: "{genre_tags}"
Lyrics/Script: "{lyrics_or_script}"
Total Duration: {total_seconds} seconds

Ensure your output is ONLY the floating-point number, with no extra text, explanations, or formatting.
Example Output: 2.5
Output:
"""

    CATEGORY = "SceneGenius"

    @classmethod
    def _get_ollama_models_list(cls):
        """
        Attempts to fetch a list of available Ollama models from the default API URL.
        Returns a list of model names or a list with a default/placeholder if connection fails.
        """
        default_ollama_url = "http://localhost:11434"
        try:
            response = requests.get(f"{default_ollama_url}/api/tags", timeout=5)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            models_data = response.json()
            models = [model["name"] for model in models_data.get("models", [])]
            if not models:
                logger.warning(f"SceneGeniusAutocreator Warning: No Ollama models found at {default_ollama_url}/api/tags. Defaulting to 'llama3:8b-instruct-q8_0'.")
                return ["llama3:8b-instruct-q8_0"] # Return default if no models are listed
            return models
        except requests.exceptions.ConnectionError:
            logger.error(f"SceneGeniusAutocreator Warning: Could not connect to Ollama at {default_ollama_url}. Please ensure Ollama is running.")
            return ["llama3:8b-instruct-q8_0 (Ollama not found)"] # Placeholder if connection fails
        except requests.exceptions.Timeout:
            logger.error(f"SceneGeniusAutocreator Warning: Ollama connection timed out at {default_ollama_url}. Defaulting to 'llama3:8b-instruct-q8_0'.")
            return ["llama3:8b-instruct-q8_0 (Connection Timeout)"] # Placeholder if timeout
        except Exception as e:
            logger.error(f"SceneGeniusAutocreator Warning: An unexpected error occurred while fetching Ollama models: {e}. Defaulting to 'llama3:8b-instruct-q8_0'.")
            return ["llama3:8b-instruct-q8_0 (Error fetching models)"] # Generic error fallback

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]: # Changed 's' to 'cls' for clarity as it's a classmethod
        """
        Define the input parameters for the SceneGeniusAutocreator node.
        These parameters allow users to control the LLM's behavior and creative output.
        """
        return {
            "required": {
                "ollama_api_base_url": ("STRING", {"default": "http://localhost:11434", "tooltip": "The full URL to the Ollama API endpoint (e.g., http://localhost:11434).", "placeholder": "http://localhost:11434"}),
                # Correctly calling the class method for model list
                "ollama_model_name": (cls._get_ollama_models_list(), {"default": "llama3:8b-instruct-q8_0", "tooltip": "The name of the LLM model to be used (e.g., llama3:8b-instruct-q8_0). Select from available models or type manually."}),
                "initial_concept_prompt": ("STRING", {"multiline": True, "default": "A dystopian cyberpunk future with a retro-futuristic soundscape.", "tooltip": "The core creative concept driving generation."}),
                "tag_count": ("INT", {"default": 4, "min": 1, "max": 10, "tooltip": "The desired number of genre tags the LLM should generate."}),
                "tag_blend_strength": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Controls how closely the generated tags should relate to the initial_concept_prompt. Higher values (e.g., 0.8-1.0) mean more direct relevance, lower values (e.g., 0.2-0.5) allow for more divergent or abstract tag suggestions."}),
                "excluded_tags": ("STRING", {"default": "", "tooltip": "A comma-separated string of genre tags that the LLM should avoid generating (e.g., 'rock, classical, jazz')."}),
                "force_lyrics_generation": ("BOOLEAN", {"default": False, "tooltip": "If True, the LLM will be instructed to always generate lyrics/script, even if it might otherwise decide an instrumental piece is more suitable."}),
                # Changed min/max_total_seconds to FLOAT as per your prompt, but kept INT defaults for tooltip clarity
                "min_total_seconds": ("FLOAT", {"default": 60.0, "min": 0.0, "max": 3600.0, "tooltip": "The minimum acceptable duration for the generated piece in seconds."}),
                "max_total_seconds": ("FLOAT", {"default": 300.0, "min": 0.0, "max": 3600.0, "tooltip": "The maximum acceptable duration for the generated piece in seconds."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "The seed for random number generation to ensure reproducibility."}),
                "sampler_version": (["Original PingPong Sampler", "FBG Integrated PingPong Sampler"], {"default": "FBG Integrated PingPong Sampler", "tooltip": "Select which version of the PingPong Sampler to generate YAML for. The FBG version includes additional dynamic guidance parameters."}),
            },
            "optional": {
                "prompt_genre_generation": ("STRING", {"multiline": True, "default": "", "tooltip": "Override the genre tags output directly. If provided, the LLM will be skipped for genre generation."}),
                "prompt_lyrics_decision_and_generation": ("STRING", {"multiline": True, "default": "", "tooltip": "Override the lyrics/script output directly. If provided, the LLM will be skipped for lyrics generation."}),
                "prompt_duration_generation": ("STRING", {"multiline": True, "default": "", "tooltip": "Override the duration output directly (e.g., '120.5'). If provided, the LLM will be skipped for duration generation."}),
                "prompt_noise_decay_generation": ("STRING", {"multiline": True, "default": "", "tooltip": "Override the noise decay strength output directly (e.g., '7.5'). If provided, the LLM will be skipped for noise decay generation."}),
                "prompt_apg_yaml_generation": ("STRING", {"multiline": True,
                    "default": "",
                    "tooltip": "Override the APG YAML parameters directly. If provided, the LLM will be skipped for APG YAML generation."}),
                "prompt_sampler_yaml_generation": ("STRING", {"multiline": True,
                    "default": "",
                    "tooltip": "Override the Sampler YAML parameters directly. If provided, the LLM will be skipped for Sampler YAML generation."}),
                "test_mode": ("BOOLEAN", { # Added test_mode back to optional inputs
                    "default": False,
                    "tooltip": "Enable test mode to skip LLM calls and return dummy data."
                })
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "FLOAT", "FLOAT", "STRING", "STRING", "INT",) # Added back RETURN_TYPES
    RETURN_NAMES = ("GENRE_TAGS", "LYRICS_OR_SCRIPT", "TOTAL_SECONDS", "NOISE_DECAY_STRENGTH", "APG_YAML_PARAMS", "SAMPLER_YAML_PARAMS", "SEED",) # Added back RETURN_NAMES

    FUNCTION = "execute" # Updated FUNCTION to point to 'execute' method

    def _strip_think_blocks(self, text):
        """Strips content between <think> and </think> tags (inclusive)."""
        # The re.DOTALL flag is crucial here to ensure '.' matches newlines.
        return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

    def _clean_llm_yaml_output(self, raw_output: str) -> str:
        """Removes common LLM markdown and extra text wrappers from YAML output."""
        cleaned = raw_output.strip()
        # List of prefixes to remove, ordered from most specific to least specific
        prefixes = ["```yaml", "```output", "```", "output:"]
        for prefix in prefixes:
            if cleaned.lower().startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
                break # Remove only one matching prefix
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()
        return cleaned

    def _call_ollama_api(self, api_url: str, payload: Dict[str, Any], timeout: int = 300, max_retries: int = 3) -> str: # Added max_retries
        """Abstract LLM API call to support future backend changes with retry logic."""
        for attempt in range(max_retries):
            try:
                # Added more detailed logging for connection attempts
                logger.info(f"Attempt {attempt + 1}/{max_retries}: Connecting to Ollama at {api_url.split('/api/')[0]} with model {payload.get('model')}")
                response = requests.post(f"{api_url}/api/generate", json=payload, timeout=timeout)
                response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)
                raw_data = response.json().get("response", "").strip()

                # Apply the stripping function here to remove <think> blocks
                data = self._strip_think_blocks(raw_data) # Re-added call to _strip_think_blocks

                return data
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                logger.warning(f"Ollama API call failed (Attempt {attempt + 1}/{max_retries}): {e}. Retrying in {2 ** attempt} seconds...")
                time.sleep(2 ** attempt) # Exponential backoff
            except requests.exceptions.HTTPError as http_err:
                logger.error(f"LLM API HTTP error: {http_err} - Response: {response.text}")
                raise RuntimeError(f"Ollama API HTTP Error: {http_err}") # Re-raising for ComfyUI to catch
            except Exception as e:
                logger.error(f"An unexpected error occurred during LLM API call: {e}")
                raise RuntimeError(f"Unexpected Error during Ollama API call: {e}") # Re-raising for ComfyUI to catch
        
        logger.error(f"Failed to call Ollama API after {max_retries} attempts.")
        raise RuntimeError("Ollama API call failed after multiple retries.")


    def execute(self,
                ollama_api_base_url: str,
                ollama_model_name: str,
                initial_concept_prompt: str,
                tag_count: int,
                tag_blend_strength: float,
                excluded_tags: str,
                force_lyrics_generation: bool,
                min_total_seconds: float, # Changed to float
                max_total_seconds: float, # Changed to float
                seed: int,
                sampler_version: str,
                prompt_genre_generation: str = "",
                prompt_lyrics_decision_and_generation: str = "",
                prompt_duration_generation: str = "",
                prompt_noise_decay_generation: str = "",
                prompt_apg_yaml_generation: str = "",
                prompt_sampler_yaml_generation: str = "",
                test_mode: bool = False
                ) -> tuple:
        """
        Main execution method with enhanced validation, logging, and fallbacks.

        Args:
            All inputs as defined in INPUT_TYPES

        Returns:
            Tuple of generated outputs
        """
        logger.info("Starting SceneGeniusAutocreator execution...")

        api_url_base = f"{ollama_api_base_url}" # Pass base URL, _call_ollama_api adds /api/generate

        # Validate and potentially correct input ranges
        if min_total_seconds < 0:
            logger.warning(f"Invalid min_total_seconds ({min_total_seconds}). Setting to 0.0.")
            min_total_seconds = 0.0
        if max_total_seconds < min_total_seconds:
            logger.warning(f"Invalid max_total_seconds ({max_total_seconds}) less than min_total_seconds ({min_total_seconds}). Setting max_total_seconds = min_total_seconds.")
            max_total_seconds = min_total_seconds
        if max_total_seconds == 0 and min_total_seconds == 0:
             logger.warning("Both min and max duration are 0. Setting min to 60.0 and max to 300.0 to avoid division by zero or infinite loop in LLM.")
             min_total_seconds = 60.0
             max_total_seconds = 300.0


        # Determine which default sampler YAML and FBG prompt section to use
        if sampler_version == "FBG Integrated PingPong Sampler":
            default_sampler_yaml_to_use = self.default_sampler_yaml_fbg
            fbg_section_to_include = self.FBG_PROMPT_SECTION
        else: # "Original PingPong Sampler"
            default_sampler_yaml_to_use = self.default_sampler_yaml_original
            fbg_section_to_include = ""


        # Test Mode: Skip LLMs and return dummy data
        if test_mode:
            logger.info("Test mode enabled. Returning dummy output.")
            # Ensure dummy YAMLs are correctly loaded and dumped
            dummy_apg_yaml = yaml.dump(yaml.safe_load(self.default_apg_yaml), Dumper=CustomDumper, indent=2, sort_keys=False)
            dummy_sampler_yaml_fbg = yaml.dump(yaml.safe_load(self.default_sampler_yaml_fbg), Dumper=CustomDumper, indent=2, sort_keys=False)
            return (
                "electro, synthwave",
                """[Verse 1]
Dust falls on broken signs,
A hollow wind through rusted lines.
Empty eyes in shadowed lanes,
Whispering echoes of forgotten names.

[Chorus]
Beneath a sky of ash and grey,
Hope's a flicker fading away.
A silent scream, a broken plea,
Lost in the ruins, just you and me.

[Verse 2]
Concrete canyons, cold and deep,
Where shadows gather, secrets sleep.
A fragile life, a fading spark,
Lost in the darkness, a world apart.

[Bridge]
The weight of sorrow, a heavy chain,
A constant ache, a lingering pain.
No solace found, no guiding light,
Just endless darkness, and a starless night.

[Chorus]
Beneath a sky of ash and grey,
Hope's a flicker fading away.
A silent scream, a broken plea,
Lost in the ruins, just you and me.

[Outro]
Fading echoes, soft and low,
Where do we go, Where do we go,
In the ruins, we fade away
In the ruins, another day""", # Corrected to triple quotes
                180.0,
                3.5,
                dummy_apg_yaml,
                dummy_sampler_yaml_fbg,
                42
            )

        # --- Stage 1: Genre Tags Generation ---
        logger.info("Stage 1: Generating genre tags...")
        genre_tags = prompt_genre_generation.strip()
        if not genre_tags:
            blend_description = ""
            if tag_blend_strength >= 0.8:
                blend_description = "be very closely related to"
            elif tag_blend_strength >= 0.5:
                blend_description = "be generally related to"
            else:
                blend_description = "allow for more divergent or abstract suggestions based on"

            excluded_tags_instruction = f"Avoid generating any of these tags: {excluded_tags}." if excluded_tags else ""

            try:
                payload = {
                    "model": ollama_model_name,
                    "prompt": self.PROMPT_GENRE_GENERATION_BASE.format(
                        tag_count=tag_count,
                        initial_concept_prompt=initial_concept_prompt,
                        excluded_tags_instruction=excluded_tags_instruction, # Pass instruction here
                        blend_description=blend_description
                    ),
                    "stream": False,
                    "keep_alive": 0
                }
                raw = self._call_ollama_api(api_url_base, payload)
                genre_tags_list = [
                    tag.strip().lower() for tag in raw.strip().split(',')
                    if tag.strip()
                ]
                genre_tags = ", ".join(genre_tags_list)
                if not genre_tags:
                    logger.warning("LLM returned empty genre tags. Falling back to 'ambient'.")
                    genre_tags = "ambient"
            except Exception as e:
                logger.error(f"Genre generation failed: {e}. Falling back to 'ambient'.")
                genre_tags = "ambient"
        else:
            logger.info("Using provided genre tags override.")

        # --- Stage 2: Lyrics/Script ---
        logger.info("Stage 2: Generating lyrics/script...")
        lyrics_or_script = prompt_lyrics_decision_and_generation.strip()
        if not lyrics_or_script:
            force_lyrics_instruction = "The user has requested that you FORCE the generation of lyrics, so do not output [instrumental]." if force_lyrics_generation else ""

            if force_lyrics_generation:
                try:
                    payload = {
                        "model": ollama_model_name,
                        "prompt": self.PROMPT_LYRICS_GENERATION_BASE.format(
                            initial_concept_prompt=initial_concept_prompt,
                            genre_tags=genre_tags,
                            force_lyrics_instruction=force_lyrics_instruction # Pass instruction here
                        ),
                        "stream": False,
                        "keep_alive": 0
                    }
                    raw = self._call_ollama_api(api_url_base, payload)
                    cleaned_lyrics = raw.strip()
                    if cleaned_lyrics.lower() == "[instrumental]":
                        lyrics_or_script = "[instrumental]"
                    else:
                        lyrics_or_script = cleaned_lyrics.strip()

                    if not lyrics_or_script or lyrics_or_script.lower() == "[instrumental]": # Re-check after cleaning
                        logger.warning("LLM returned empty or instrumental lyrics despite force_lyrics_generation. Setting to default instrumental.")
                        lyrics_or_script = "[instrumental]" # Ensure it's not empty if forced
                except Exception as e:
                    logger.error(f"Lyrics generation failed: {e}. Setting to instrumental.")
                    lyrics_or_script = "[instrumental]"
            else:
                logger.info("Force lyrics generation is false. Setting lyrics to instrumental.")
                lyrics_or_script = "[instrumental]"
        else:
            logger.info("Using provided lyrics/script override.")


        # --- Stage 3: Duration ---
        logger.info("Stage 3: Generating duration...")
        total_seconds = min_total_seconds # Initialize with min_total_seconds as a safe fallback
        try:
            if prompt_duration_generation.strip():
                total_seconds = float(prompt_duration_generation.strip())
            else:
                payload = {
                    "model": ollama_model_name,
                    "prompt": self.PROMPT_DURATION_GENERATION_BASE.format(
                        min_total_seconds=min_total_seconds,
                        max_total_seconds=max_total_seconds,
                        initial_concept_prompt=initial_concept_prompt,
                        genre_tags=genre_tags,
                        lyrics_or_script=lyrics_or_script
                    ),
                    "stream": False,
                    "keep_alive": 0
                }
                raw = self._call_ollama_api(api_url_base, payload)
                total_seconds = float(raw.strip())
            total_seconds = max(min_total_seconds, min(total_seconds, max_total_seconds))
            if total_seconds == 0: # Ensure it's not zero if limits allow it
                total_seconds = max(min_total_seconds, 1.0) # Ensure a minimum of 1 second if 0 is possible
        except ValueError:
            logger.warning(f"LLM returned invalid duration ('{raw.strip() if 'raw' in locals() else ''}'). Using min duration: {min_total_seconds}.")
            total_seconds = min_total_seconds
        except Exception as e:
            logger.warning(f"Duration generation failed: {e}. Using min duration: {min_total_seconds}.")
            total_seconds = min_total_seconds
        logger.info(f"Final duration: {total_seconds:.2f} seconds.")

        # --- Stage 4: Noise Decay ---
        logger.info("Stage 4: Generating noise decay strength...")
        noise_decay_strength = self.NOISE_DECAY_MIN # Initialize with min as a safe fallback
        try:
            if prompt_noise_decay_generation.strip():
                noise_decay_strength = float(prompt_noise_decay_generation.strip())
            else:
                payload = {
                    "model": ollama_model_name,
                    "prompt": self.PROMPT_NOISE_DECAY_GENERATION_BASE.format(
                        min_noise_decay=self.NOISE_DECAY_MIN,
                        max_noise_decay=self.NOISE_DECAY_MAX,
                        initial_concept_prompt=initial_concept_prompt,
                        genre_tags=genre_tags,
                        total_seconds=total_seconds,
                        lyrics_or_script=lyrics_or_script # ADDED: Pass lyrics_or_script here
                    ),
                    "stream": False,
                    "keep_alive": 0
                }
                raw = self._call_ollama_api(api_url_base, payload)
                noise_decay_strength = float(raw.strip())
            noise_decay_strength = max(self.NOISE_DECAY_MIN, min(noise_decay_strength, self.NOISE_DECAY_MAX))
        except ValueError:
            logger.warning(f"LLM returned invalid noise decay strength ('{raw.strip() if 'raw' in locals() else ''}'). Using default: 5.0.")
            noise_decay_strength = 5.0 # Fallback to a sensible default if parsing fails
        except Exception as e:
            logger.warning(f"Noise decay generation failed: {e}. Using default: 5.0.")
            noise_decay_strength = 5.0 # Fallback to a sensible default if LLM fails
        logger.info(f"Final noise decay strength: {noise_decay_strength:.2f}.")

        # --- Stage 5: APG YAML Parameters Generation ---
        logger.info("Stage 5: Generating APG YAML Parameters.")
        apg_yaml_params = self.default_apg_yaml # Default fallback
        try:
            if prompt_apg_yaml_generation.strip():
                logger.info("APG YAML override detected. Attempting to parse provided value.")
                cleaned = self._clean_llm_yaml_output(prompt_apg_yaml_generation)
                data = yaml.safe_load(cleaned)
                if isinstance(data, dict):
                    apg_yaml_params = yaml.dump(data, Dumper=CustomDumper, indent=2, sort_keys=False)
                else:
                    logger.warning("Provided APG YAML override was not a valid dictionary. Using default.")
            else:
                payload = {
                    "model": ollama_model_name,
                    "prompt": self.DEFAULT_APG_PROMPT.format(
                        initial_concept_prompt=initial_concept_prompt,
                        genre_tags=genre_tags,
                        lyrics_or_script=lyrics_or_script,
                        total_seconds=total_seconds,
                        noise_decay_strength=noise_decay_strength,
                        min_cfg=self.APG_MIN_CFG,
                        max_cfg=self.APG_MAX_CFG
                    ),
                    "stream": False,
                    "keep_alive": 0
                }
                raw = self._call_ollama_api(api_url_base, payload)
                cleaned = self._clean_llm_yaml_output(raw)
                data = yaml.safe_load(cleaned)
                if isinstance(data, dict):
                    apg_yaml_params = yaml.dump(data, Dumper=CustomDumper, indent=2, sort_keys=False)
                else:
                    logger.warning("LLM-generated APG YAML was not a valid dictionary. Using default.")
        except yaml.YAMLError as ye:
            logger.warning(f"APG YAML parsing error: {ye}. Using default.")
            apg_yaml_params = self.default_apg_yaml
        except Exception as e:
            logger.warning(f"APG YAML generation failed: {e}. Using default.")
            apg_yaml_params = self.default_apg_yaml

        # Re-add the regex substitution for dims for consistent output format
        apg_yaml_params = re.sub(
            r'(\s*)dims:\s*\n\s*- -2\s*\n\s*- -1',
            r'\1dims: [-2, -1]',
            apg_yaml_params
        )
        logger.info(f"Final APG_YAML_PARAMS output: \n---\n{apg_yaml_params}\n---")


        # --- Stage 6: Sampler YAML Parameters Generation ---
        logger.info(f"Stage 6: Generating Sampler YAML Parameters for '{sampler_version}'.")
        # Set initial default based on selected version
        sampler_yaml_params = (
            self.default_sampler_yaml_fbg
            if sampler_version == "FBG Integrated PingPong Sampler"
            else self.default_sampler_yaml_original
        )

        try:
            if prompt_sampler_yaml_generation.strip():
                logger.info("Sampler YAML override detected. Attempting to parse provided value.")
                cleaned = self._clean_llm_yaml_output(prompt_sampler_yaml_generation)
                data = yaml.safe_load(cleaned)

                # Sampler Override Validation: Check if FBG config is present if FBG version is selected
                if sampler_version == "FBG Integrated PingPong Sampler" and (not isinstance(data, dict) or "fbg_config" not in data):
                    logger.warning("Provided Sampler YAML override for FBG version lacks 'fbg_config'. Using default FBG Sampler YAML.")
                    sampler_yaml_params = self.default_sampler_yaml_fbg
                elif isinstance(data, dict):
                    sampler_yaml_params = yaml.dump(data, Dumper=CustomDumper, indent=2, sort_keys=False)
                else:
                    logger.warning(f"Provided Sampler YAML override was not a valid dictionary for '{sampler_version}'. Using its default.")
            else:
                # Dynamically build the prompt based on selected sampler version
                sampler_yaml_prompt_content = self.DEFAULT_SAMPLER_PROMPT_BASE.format(
                    initial_concept_prompt=initial_concept_prompt,
                    genre_tags=genre_tags,
                    total_seconds=total_seconds,
                    lyrics_or_script=lyrics_or_script, # Added back for prompt context
                    apg_yaml_params=apg_yaml_params, # Added back for prompt context
                    fbg_section=fbg_section_to_include # Insert the FBG section if applicable
                )

                payload = {
                    "model": ollama_model_name,
                    "prompt": sampler_yaml_prompt_content,
                    "stream": False,
                    "keep_alive": 0
                }
                raw = self._call_ollama_api(api_url_base, payload)
                cleaned = self._clean_llm_yaml_output(raw)
                data = yaml.safe_load(cleaned)
                if isinstance(data, dict):
                    sampler_yaml_params = yaml.dump(data, Dumper=CustomDumper, indent=2, sort_keys=False)
                else:
                    logger.warning(f"LLM-generated Sampler YAML was not a valid dictionary for '{sampler_version}'. Using its default.")
        except yaml.YAMLError as ye:
            logger.warning(f"Sampler YAML parsing error: {ye}. Using default for '{sampler_version}'.")
            # sampler_yaml_params already set to correct default at the start of the try block
        except Exception as e:
            logger.warning(f"Sampler YAML generation failed: {e}. Using default for '{sampler_version}'.")
            # sampler_yaml_params already set to correct default at the start of the try block

        # Apply the regex substitution for dims to Sampler YAML as well for consistency
        sampler_yaml_params = re.sub(
            r'(\s*)dims:\s*\n\s*- -2\s*\n\s*- -1',
            r'\1dims: [-2, -1]',
            sampler_yaml_params
        )
        logger.info(f"Final SAMPLER_YAML_PARAMS output: \n---\n{sampler_yaml_params}\n---")

        return (
            genre_tags,
            lyrics_or_script,
            total_seconds,
            noise_decay_strength,
            apg_yaml_params,
            sampler_yaml_params,
            seed
        )

# Node Mapping
NODE_CLASS_MAPPINGS = {
    "SceneGeniusAutocreator": SceneGeniusAutocreator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SceneGeniusAutocreator": "Scene Genius Autocreator"
}
