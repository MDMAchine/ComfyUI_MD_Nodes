# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# ████ SCENEGENIUS AUTOCREATOR v0.1.4a – Optimized for Ace-Step Audio/Video ████▓▒░
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
# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄


import requests
import json
import os
import yaml
import re # Import the re module for regular expressions

# Define a custom YAML Dumper for specific list formatting
class CustomDumper(yaml.Dumper):
    def represent_list(self, data):
        # Force flow style (inline) for lists with 2 or fewer numeric elements
        if len(data) <= 2 and all(isinstance(x, (int, float)) for x in data):
            return self.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)
        # For all other lists, let the default behavior of yaml.dump handle it.
        # This typically means block style for sequences unless explicitly forced to flow style elsewhere.
        return self.represent_sequence('tag:yaml.org,2002:seq', data)

class SceneGeniusAutocreator:
    """
    A multi-stage ComfyUI node leveraging local LLMs for dynamic creative content generation.
    This includes genres, lyrics/scripts, duration, and configurable diffusion parameters (APG & Sampler).
    It incorporates advanced reasoning subroutines for enhanced coherence and control over creative output.
    """
    def __init__(self):
        # Default APG YAML to use if LLM output is unparseable or invalid
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
        # Default Sampler YAML for original PingPongSampler_Custom to use if LLM output is unparseable or invalid
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
        # Default Sampler YAML for PingPongSampler_Custom_FBG to use if LLM output is unparseable or invalid
        # This now matches the specific FBG YAML provided by the user.
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
  guidance_max_change: 1.5
  fbg_temp: 0.010
  fbg_offset: -0.030
  pi: 0.35
  t_0: 0.90
  t_1: 0.60
fbg_eta: 0.5
fbg_s_noise: 0.7
"""

        # Base prompt for Sampler YAML generation (common to both versions)
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

* **`verbose` (boolean):** Set to `true` or `false` to enable/disable debug messages for the sampler.
* **`step_random_mode` (string):** **How the internal random number generator (RNG) seed is managed across sampling steps.** This directly influences the temporal coherence (smoothness over time) of generated frames or audio segments.
    * **Purpose:** Controls the "flicker" or consistency.
    * **Decision-Making:**
        * `off`: LEAST random variation per step. Ideal for highly consistent, stable, and coherent output across sequential frames (video) or audio segments.
        * `block`: Randomness changes only after a `step_size` number of steps. Useful if you want visually/audibly distinct "blocks" or segments that are internally consistent but change abruptly at intervals. Works very well in many cases.
        * `step`: Randomness changes with *every single step*. This can lead to very chaotic, flickering visuals or rapidly changing audio textures.
        * `reset`: Resets the random seed completely after each block, leading to potentially strong disconnections between blocks. Use with caution.
        * **LLM Decision**: Consider the `initial_concept_prompt` and `GENRE_TAGS`. For "smooth," "cinematic," or "ambient" outputs, favor `off`. For "segmented" or "rhythmic" changes, consider `block`. For "chaos," `step`.
* **`step_size` (integer):** The size of the "block" or multiplier for `step_random_mode`. Default is `5`.
    * **Purpose:** Defines the granularity of `block` or `reset` mode.
    * **Decision-Making:** Only relevant if `step_random_mode` is `block` or `reset`. Integers like `4`, `5` or `10` are common. Smaller values mean more frequent shifts; larger values mean longer, more consistent segments.
* **`first_ancestral_step` (integer):** The 0-based index in the sigma schedule where ancestral noise injection begins. For Original: `14`. For FBG: can be `0`.
* **`last_ancestral_step` (integer):** The 0-based index in the sigma schedule where ancestral noise injection ends. For Original: `34`. For FBG: can be `-1`.
    * **Purpose of Ancestral Noise:** Ancestral steps re-inject a small, calculated amount of noise back into the latent. This is a crucial technique to prevent "dead spots," over-smoothing, or loss of detail that can occur in purely deterministic samplers. It helps maintain a vibrant, detailed output, especially for complex generative tasks like video/audio.
    * **Decision-Making:** These two parameters define the "active ancestral mixing period." For the Original Sampler, values like `first_ancestral_step: 10-15` and `last_ancestral_step: 30-40` are good starting points. For the FBG Sampler, setting `first_ancestral_step: 0` and `last_ancestral_step: -1` (to cover all steps) can be effective depending on `fbg_eta` and `fbg_s_noise`.
* **`start_sigma_index` (integer):** The 0-based index in the sigma schedule to start sampling from.
    * **Purpose:** Determines which step of the sigma schedule the sampling process *begins* at.
    * **Decision-Making:** **ALWAYS set this to `0` (default). Do not change.** Changing this means skipping initial denoising steps, leading to fundamentally incomplete and noisy results.
* **`end_sigma_index` (integer):** The 0-based index in the sigma schedule to end sampling early.
    * **Purpose:** Determines which step of the sigma schedule the sampling process *ends* at.
    * **Decision-Making:** **ALWAYS set this to `-1` (default, meaning use all steps). Do not change.** Changing this means prematurely stopping the denoising process, resulting in unfinished and noisy outputs.
* **`enable_clamp_output` (boolean):** `true` to clamp the final output values to `[-1, 1]`; `false` otherwise. **For your model's optimal sound quality, it is generally desired to NOT limit the output to `[-1, 1]`. Therefore, for best results, set this to `false`.** Setting to `true` may be considered if strict numerical stability within a specific range is absolutely required, but this is less common for audio models that benefit from a wider dynamic range.
* **`blend_mode` (string):** The blending mode for general internal operations.
    * **Purpose:** Controls how certain internal latent values are combined.
    * **Decision-Making:** `lerp` (linear interpolation) is the default for stability. `slerp` (spherical linear interpolation) can offer smoother transitions, and `geometric_mean`, `add`, `subtract`, `multiply`, `divide`, `a_only`, `b_only` are also available for experimental use. Exercise caution when deviating from `lerp`/`slerp` as other modes can produce unexpected results.
* **`step_blend_mode` (string):** The blending mode specifically for non-ancestral steps.
    * **Purpose:** Controls the blending during non-ancestral steps.
    * **Decision-Making:** Similar to `blend_mode`, `lerp` is default. `geometric_mean` can be very effective in some cases for FBG. Other options are available for experimentation.
* **`sigma_range_preset` (string):** Quick FBG/CFG sigma ranges.
    * **Purpose:** Simplifies setting `cfg_start_sigma`, `cfg_end_sigma`, `fbg_start_sigma`, `fbg_end_sigma`.
    * **Decision-Making:** Choose from `Custom`, `High`, `Mid`, `Low`, `All`. If `Custom` is selected, the individual `_start_sigma` and `_end_sigma` parameters under `fbg_config` will be used.
* **`conditional_blend_mode` (boolean):** Enable dynamic blend function switching based on sigma.
    * **Purpose:** Allows the blend function to change when the sigma value crosses a threshold.
    * **Decision-Making:** Set to `true` to enable. Requires `conditional_blend_sigma_threshold` and `conditional_blend_function_name`.
* **`conditional_blend_sigma_threshold` (float):** Sigma threshold for conditional blending.
    * **Purpose:** The sigma value at which the blend function switches if `conditional_blend_mode` is `true`.
    * **Decision-Making:** A float value (e.g., `0.4`).
* **`conditional_blend_function_name` (string):** Blend function when sigma condition met.
    * **Purpose:** Specifies the blend function to use after the `conditional_blend_sigma_threshold` is crossed.
    * **Decision-Making:** Can be any of the `blend_mode` options (e.g., `slerp`, `geometric_mean`).
* **`conditional_blend_on_change` (boolean):** Enable dynamic blend switching based on step change magnitude.
    * **Purpose:** Allows the blend function to change if the magnitude of change between steps exceeds a threshold.
    * **Decision-Making:** Set to `true` to enable. Requires `conditional_blend_change_threshold`.
* **`conditional_blend_change_threshold` (float):** Change threshold for conditional blending.
    * **Purpose:** The threshold for step change magnitude at which the blend function switches if `conditional_blend_on_change` is `true`.
    * **Decision-Making:** A float value (e.g., `0.15`).
* **`clamp_noise_norm` (boolean):** Enable clamping the L2 norm of injected noise.
    * **Purpose:** Controls whether the magnitude of noise injected during ancestral steps is limited.
    * **Decision-Making:** Set to `true` to clamp noise. Requires `max_noise_norm`.
* **`max_noise_norm` (float):** Maximum allowed noise L2 norm.
    * **Purpose:** The upper limit for the L2 norm of injected noise if `clamp_noise_norm` is `true`.
    * **Decision-Making:** A float value (e.g., `0.2`).
* **`debug_mode` (integer):** Console log verbosity.
    * **Purpose:** Controls the level of debug messages printed to the console.
    * **Decision-Making:** `0`=Off, `1`=Basic, `2`=Verbose.

{fbg_section}

**IMPORTANT: If the FBG section above is present (i.e., you are generating for the FBG Integrated PingPong Sampler), you MUST include the `fbg_config` dictionary and its associated top-level FBG parameters (`fbg_eta`, `fbg_s_noise`, `debug_mode`) in your output. Refer to the structure provided in the FBG section.**

**Your output must ONLY be the valid YAML string, with no conversational filler, explanations, or additional text.**

**Example Output Format (for original PingPong Sampler):**
verbose: true
step_random_mode: block
step_size: 5
first_ancestral_step: 14
last_ancestral_step: 34
start_sigma_index: 0
end_sigma_index: -1
enable_clamp_output: false
blend_mode: lerp
step_blend_mode: lerp

**Example Output Format (for FBG Integrated PingPong Sampler):**
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
  guidance_max_change: 1.5
  fbg_temp: 0.010
  fbg_offset: -0.030
  pi: 0.35
  t_0: 0.90
  t_1: 0.60
fbg_eta: 0.5
fbg_s_noise: 0.7

**IMPORTANT: When generating the actual YAML, DO NOT include the ```yaml or ``` delimiters in your final output. These are only for demonstration in this example.**
"""

        # FBG specific prompt section to be conditionally added
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
    * **`fbg_guidance_multiplier` (float):** Multiplier for the FBG guidance component. **Default: `1.0`**.
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

        # Full default prompt for APG YAML generation
        self.DEFAULT_APG_PROMPT = """
You are an advanced generative AI configuration expert, specializing in the Adaptive Projected Gradient (APG) Guider within ComfyUI. Your primary goal is to generate a YAML-formatted set of "rules" that precisely control the diffusion model's guidance throughout the sampling process, shaping the evolution of the latent space from abstract noise into a cohesive creative output. Each rule defines a specific guidance strategy to be applied at different noise levels (sigmas).

**Your understanding of the creative context:**
* **Initial Concept:** "{initial_concept_prompt}" (This is the overarching vision)
* **Generated Genres:** "{genre_tags}" (These define the aesthetic and mood, e.g., 'synthwave' implies crispness, 'ambient' implies smoothness)
* **Lyrics/Script Status:** "{lyrics_or_script}" (Indicates if the piece is narrative/vocal-driven or instrumental, impacting required detail/cohesion)
* **Total Duration (seconds):** "{total_seconds}" (The intended length, which can subtly influence the need for efficiency or complexity in guidance)

**Your task: Generate a VALID YAML string for the APG guider. The YAML structure MUST include:**
* `verbose` (boolean): Set to `true` if detailed console logging for the APG Guider is desired for debugging, `false` for a cleaner log.
* `rules` (array of dictionaries): This array defines the step-by-step guidance strategy. Each dictionary within `rules` represents a distinct phase of guidance.

**For each rule, intelligently define the following parameters, with a clear understanding of their impact and typical usage patterns:**

* **`start_sigma` (float):** **Defines the exact noise level (sigma) at which this specific rule becomes active.**
    * **Purpose:** Establishes the "timeline" for guidance. Rules are applied in the order of their `start_sigma` values, from highest to lowest.
    * **Critical Usage:**
        * **First Rule:** The very first rule in your `rules` array **MUST have `start_sigma: -1.0`**. This ensures it applies from the absolute beginning of sampling.
        * **Progression:** All subsequent `start_sigma` values **MUST be strictly decreasing** (e.g., `0.85`, `0.70`, `0.55`, `0.40`, `0.30`, `0.15`, `0.05`, `0.01`). This creates a proper sequence of guidance.
        * **Final Rule:** A rule with a very low `start_sigma` (e.g., `0.01` or `0.005`) should typically conclude the list for final detail refinement.
* **`apg_scale` (float):** **The strength of the Adaptive Projected Gradient guidance for this rule.**
    * **Purpose:** `apg_scale` actively sculpts the latent space. Higher values mean more aggressive guidance towards the prompt, potentially leading to sharper details but also risking artifacts if too high. `0.0` turns APG off.
    * **Decision-Making:**
        * **Initial & Final Phases (`start_sigma: -1.0` and lowest `start_sigma`):** Always set `apg_scale: 0.0`. This provides a stable CFG-only phase at the very beginning (for initial composition) and end (for clean finalization).
        * **Intermediate Phases (where `apg_scale` is active):** For optimal performance, `apg_scale` can be set around `5.0` in early to mid-phases, and then gradually taper off to values as low as `1.0` or `0.0` (off) as sampling progresses towards lower sigmas.
            * For concepts requiring **crispness, strong adherence to prompt, or detailed visuals** (e.g., 'cyberpunk', 'photorealistic'), use higher `apg_scale` (e.g., `3.0` to `5.0`).
            * For concepts requiring **softness, ambient qualities, or abstractness** (e.g., 'dreamy', 'lo-fi'), use lower `apg_scale` (e.g., `1.0` to `3.0`).
* **`cfg` (float):** **The standard Classifier-Free Guidance (CFG) scale for this specific rule's phase.**
    * **Purpose:** Controls how strongly the model adheres to the text prompt versus exploring freely.
    * **Decision-Making:** The **optimal performance for `cfg` has been observed to be effective in the `2.5` to `4.0` range, especially when paired with FBG.** While `cfg` can be kept consistent across rules, or adjusted **slightly** (e.g., `0.1` to `0.2` increments/decrements as seen in the example) in phases requiring more direct prompt adherence or subtle shifts in generation, **always keep it within the `2.5-4.0` range for best results** to avoid over-guidance or under-guidance artifacts.
* **`predict_image` (boolean):** **Determines what the model is "predicting" at that step (image content vs. noise).**
    * **Purpose:** Influences how APG guidance is calculated and applied. `true` focuses guidance on the evolving image content, `false` on noise.
    * **Decision-Making:**
        * Generally set to `true` for rules where `apg_scale` is active, especially in early and mid-phases, as this focuses guidance on the evolving image content and often leads to smoother, more coherent results. This is particularly effective when `mode` is `pre_alt2` or `pure_apg` with active `momentum`.
        * **For optimal results, especially when combining with FBG, consider setting `predict_image: true` for the majority of active APG phases (even at lower sigmas down to `0.30` or `0.15`), as this often contributes to better detail preservation and coherence.**
* **`mode` (string):** **Controls the method of APG application and internal blending.**
    * **Purpose:** Dictates the underlying algorithm for projecting guidance and how momentum is applied.
    * **Decision-Making:**
        * `pre_alt2`: **HIGHLY RECOMMENDED for early active APG phases (higher sigmas).** Internally, this activates a "pre-CFG" guidance mechanism (`pre_cfg_mode: true`) and uses `ALT2` for momentum blending, leading to very smooth and coherent results. This mode integrates exceptionally well with `predict_image: true` and active `momentum` (e.g., `0.5` to `0.8`).
        * `pure_apg`: Direct APG application. Best applied from mid-to-lower sigmas until APG is turned off. This corresponds to the `DEFAULT` update mode internally. Use with `predict_image: true` for image prediction or `predict_image: false` for noise prediction, depending on the desired effect at that sigma.
        * `cfg`: Standard CFG behavior (no APG). **MUST be used with `apg_scale: 0.0`**. This mode allows you to have CFG-only phases at the beginning and end of sampling.
* **`momentum` (float, optional):** **Introduces a running average for smoother APG updates.**
    * **Purpose:** Prevents sudden, jarring changes or artifacts in the latent space by smoothing the guidance over several steps.
    * **Decision-Making:**
        * `0.0`: No momentum. Use when `apg_scale` is `0.0` or for very sharp, instantaneous guidance.
        * Higher values (e.g., `0.5` to `0.8`): **Critically important and strongly recommended** when `apg_scale` is active and `mode` is `pre_alt2`. This combination leverages the `lerp` blend mode for `UpdateMode.ALT2` within the APG's internal update mechanism, which is essential for maintaining visual/audio coherence and preventing sudden changes in the latent space, particularly for consistent video/audio generation.
* **`norm_threshold` (float, optional):** **Clamps the magnitude of the guidance vector.**
    * **Purpose:** Prevents APG guidance from becoming excessively strong and causing extreme distortions or noise.
    * **Decision-Making:** Use when `apg_scale` is active (e.g., `2.5` to `3.5`). This acts as a safeguard.
* **`dims` (list of integers, optional):** Specifies which dimensions to normalize.
    * **Purpose:** This parameter is typically used to ensure proper dimensional handling for guidance, especially when working with image latents.
    * **Decision-Making:** **Always include `dims: [-2, -1]`** for image latents (Height and Width) in any rule where `apg_scale` is active (i.e., not `0.0`), as this ensures proper dimensional handling for guidance.
* **`eta` (float, optional):** **Controls the projection orthogonality.**
    * **Purpose:** Influences the direction of the guidance vector.
    * **Decision-Making:** **For this forked version of the APG Guider, `eta` scaling has been removed from the `apg` projection for simplified guidance. Therefore, you should always keep this at `0.0` (fully orthogonal guidance).** Deviating from `0.0` will likely have no effect on the projection in this specific implementation and can lead to unpredictable reasoning by the LLM.
* **`update_blend_mode` (string, optional):** Blending mode for momentum updates.
    * **Purpose:** Defines how the running average is calculated.
    * **Decision-Making:** **Generally keep at `lerp` (linear interpolation)** for smooth transitions.

**Overall Strategy for `rules` Array Construction:**
1.  **Initialization:** A rule at `start_sigma: -1.0` with `apg_scale: 0.0` and a base `cfg`. This sets up the initial CFG-only phase.
2.  **Early Sculpting (Higher Sigmas):** Rules at higher sigmas (e.g., `0.85`, `0.70`) should establish core features and coherence. These rules are ideal for activating APG with higher `apg_scale` (e.g., `3.0` to `5.0`), using `mode: pre_alt2` for robust and smooth guidance, and setting `predict_image: true` to focus on image evolution. Ensure active `momentum` (e.g., `0.6` to `0.8`) and `dims: [-2, -1]` for smooth and dimensionally correct application.
3.  **Mid-Process Refinement (Mid-range Sigmas):** Rules at mid-range sigmas (e.g., `0.55`, `0.40`, `0.30`) should refine details. You can transition `mode` to `pure_apg` here. `apg_scale` might be slightly reduced, and **for optimal results with FBG, continue to use `predict_image: true`** at these stages. Continue to consider `momentum` if needed for continued smoothness, and always include `dims: [-2, -1]` when `apg_scale` is active.
4.  **Final Cleanup:** A rule at a very low `start_sigma` (e.g., `0.01`) with `apg_scale: 0.0` to ensure a clean, CFG-only finish, allowing the sampler to finalize details without aggressive APG.

**Your output must ONLY be the valid YAML string, with no conversational filler, explanations, or additional text.**

**IMPORTANT: Too many apg changes can sometimes be bad, 4-6 including the start where there is no apg used is usually enough.**

**Example Output Format - This setting produces good results and should always be the baseline of your decison making:**
verbose: true

rules:
  - start_sigma: -1
    apg_scale: 0
    cfg: 4

  - start_sigma: 0.85
    apg_scale: 5.0
    predict_image: true
    cfg: 3.0
    mode: pre_alt2
    update_blend_mode: lerp
    dims: [-2, -1]
    momentum: 0.7
    norm_threshold: 3.0
    eta: 0.0

  - start_sigma: 0.70
    apg_scale: 4.5
    predict_image: true
    cfg: 2.9
    mode: pre_alt2
    update_blend_mode: lerp
    dims: [-2, -1]
    momentum: 0.6
    norm_threshold: 2.5
    eta: 0.0

  - start_sigma: 0.55
    apg_scale: 4.0
    predict_image: true
    cfg: 2.8
    mode: pure_apg
    momentum: 0.0

  - start_sigma: 0.40
    apg_scale: 3.8
    predict_image: true
    cfg: 2.7
    momentum: 0.0

  - start_sigma: 0.30
    apg_scale: 3.5
    predict_image: true
    cfg: 2.6
    momentum: 0.0

  - start_sigma: 0.15
    apg_scale: 0
    cfg: 2.5
    momentum: 0.0

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
                print(f"SceneGeniusAutocreator Warning: No Ollama models found at {default_ollama_url}/api/tags. Defaulting to 'llama3:8b-instruct-q8_0'.")
                return ["llama3:8b-instruct-q8_0"] # Return default if no models are listed
            return models
        except requests.exceptions.ConnectionError:
            print(f"SceneGeniusAutocreator Warning: Could not connect to Ollama at {default_ollama_url}. Please ensure Ollama is running.")
            return ["llama3:8b-instruct-q8_0 (Ollama not found)"] # Placeholder if connection fails
        except requests.exceptions.Timeout:
            print(f"SceneGeniusAutocreator Warning: Ollama connection timed out at {default_ollama_url}. Defaulting to 'llama3:8b-instruct-q8_0'.")
            return ["llama3:8b-instruct-q8_0 (Connection Timeout)"] # Placeholder if timeout
        except Exception as e:
            print(f"SceneGeniusAutocreator Warning: An unexpected error occurred while fetching Ollama models: {e}. Defaulting to 'llama3:8b-instruct-q8_0'.")
            return ["llama3:8b-instruct-q8_0 (Error fetching models)"] # Generic error fallback

    @classmethod
    def INPUT_TYPES(s):
        """
        Define the input parameters for the SceneGeniusAutocreator node.
        These parameters allow users to control the LLM's behavior and creative output.
        """
        return {
            "required": {
                "ollama_api_base_url": ("STRING", {"default": "http://localhost:11434", "tooltip": "The full URL to the Ollama API endpoint (e.g., http://localhost:11434).", "placeholder": "http://localhost:11434"}),
                "ollama_model_name": (s._get_ollama_models_list(), {"default": "llama3:8b-instruct-q8_0", "tooltip": "The name of the LLM model to be used (e.g., llama3:8b-instruct-q8_0). Select from available models or type manually."}),
                "initial_concept_prompt": ("STRING", {"multiline": True, "default": "A dystopian cyberpunk future with a retro-futuristic soundscape.", "tooltip": "The foundational text input that guides the LLM's initial creative direction."}),
                "tag_count": ("INT", {"default": 4, "min": 1, "max": 10, "tooltip": "The desired number of genre tags the LLM should generate."}),
                "tag_blend_strength": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Controls how closely the generated tags should relate to the initial_concept_prompt. Higher values (e.g., 0.8-1.0) mean more direct relevance, lower values (e.g., 0.2-0.5) allow for more divergent or abstract tag suggestions."}),
                "excluded_tags": ("STRING", {"default": "", "tooltip": "A comma-separated string of genre tags that the LLM should avoid generating (e.g., 'rock, classical, jazz')."}),
                "force_lyrics_generation": ("BOOLEAN", {"default": False, "tooltip": "If True, the LLM will be instructed to always generate lyrics/script, even if it might otherwise decide an instrumental piece is more suitable."}),
                "min_total_seconds": ("INT", {"default": 60, "min": 30, "max": 1800, "tooltip": "The minimum acceptable duration for the generated piece in seconds."}),
                "max_total_seconds": ("INT", {"default": 300, "min": 60, "max": 3600, "tooltip": "The maximum acceptable duration for the generated piece in seconds."}),
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
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "FLOAT", "FLOAT", "STRING", "STRING", "INT",)
    RETURN_NAMES = ("GENRE_TAGS", "LYRICS_OR_SCRIPT", "TOTAL_SECONDS", "NOISE_DECAY_STRENGTH", "APG_YAML_PARAMS", "SAMPLER_YAML_PARAMS", "SEED",)

    FUNCTION = "generate_content"

    def _strip_think_blocks(self, text):
        """Strips content between <think> and </think> tags (inclusive)."""
        # The re.DOTALL flag is crucial here to ensure '.' matches newlines.
        return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)


    def _call_ollama_api(self, api_url, payload):
        """
        Helper function to make Ollama API calls with error handling and
        to strip <think> blocks from the LLM's raw response.
        """
        try:
            print(f"SceneGeniusAutocreator: Attempting to connect to Ollama at {api_url.split('/api/')[0]} with model {payload.get('model')}")
            response = requests.post(api_url, json=payload, timeout=300)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            # FIX: Use response.json() to get the dictionary, then .get() to access keys
            raw_data = response.json().get("response", "").strip()

            # Apply the stripping function here to remove <think> blocks
            data = self._strip_think_blocks(raw_data)

            return data
        except requests.exceptions.ConnectionError as e:
            print(f"SceneGeniusAutocreator Error: Could not connect to Ollama API at {api_url.split('/api/')[0]}.")
            print(f"Please ensure Ollama is running and the API endpoint is correct.")
            print(f"Error details: {e}")
            raise RuntimeError(f"Ollama API Connection Error: {e}")
        except requests.exceptions.Timeout:
            print(f"SceneGeniusAutocreator Error: Ollama API request timed out after 300 seconds.")
            print(f"The LLM might be taking too long to respond or the model is very large.")
            raise RuntimeError("Ollama API Timeout Error.")
        except requests.exceptions.RequestException as e:
            print(f"SceneGeniusAutocreator Error: An error occurred during the Ollama API request: {e}")
            print(f"Response content: {response.text if 'response' in locals() else 'No response content'}")
            raise RuntimeError(f"Ollama API Request Error: {e}")
        except json.JSONDecodeError as e:
            print(f"SceneGeniusAutocreator Error: Failed to decode JSON response from Ollama API: {e}")
            print(f"Raw response text: {response.text if 'response' in locals() else 'No response'}")
            raise RuntimeError(f"Ollama JSON Decode Error: {e}")
        except Exception as e:
            print(f"SceneGeniusAutocreator Error: An unexpected error occurred: {e}")
            raise RuntimeError(f"Unexpected Error: {e}")

    def generate_content(self, ollama_api_base_url, ollama_model_name, initial_concept_prompt,
                         tag_count, tag_blend_strength, excluded_tags, force_lyrics_generation,
                         min_total_seconds, max_total_seconds, seed, sampler_version,
                         prompt_genre_generation="", prompt_lyrics_decision_and_generation="",
                         prompt_duration_generation="", prompt_noise_decay_generation="",
                         prompt_apg_yaml_generation="", prompt_sampler_yaml_generation=""):
        """
        Main function to generate content using the LLM.
        This phase implements genre generation and initial lyrics generation.
        """
        api_url_base = f"{ollama_api_base_url}/api/generate"
        genre_tags = ""
        lyrics_or_script = ""
        total_seconds = 0.0
        noise_decay_strength = 0.0
        apg_yaml_params = ""
        sampler_yaml_params = ""

        # Determine which default sampler YAML and FBG prompt section to use
        if sampler_version == "FBG Integrated PingPong Sampler":
            default_sampler_yaml_to_use = self.default_sampler_yaml_fbg
            fbg_section_to_include = self.FBG_PROMPT_SECTION
        else: # "Original PingPong Sampler"
            default_sampler_yaml_to_use = self.default_sampler_yaml_original
            fbg_section_to_include = ""


        # --- Stage 1: Genre Generation ---
        print(f"SceneGeniusAutocreator: Initiating Genre Generation for concept: '{initial_concept_prompt}'")
        if prompt_genre_generation.strip():
            genre_tags = prompt_genre_generation.strip()
            print(f"SceneGeniusAutocreator: Genre override detected. Using provided value: '{genre_tags}'")
        else:
            blend_description = ""
            if tag_blend_strength >= 0.8:
                blend_description = "be very closely related to"
            elif tag_blend_strength >= 0.5:
                blend_description = "be generally related to"
            else:
                blend_description = "allow for more divergent or abstract suggestions based on"

            genre_prompt = f"""
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

                {"Avoid generating any of these tags: " + excluded_tags + "." if excluded_tags else ""}
                The output MUST ONLY be the comma-separated tags. DO NOT include any conversational filler, explanations, or additional text.
                Example Output (for tag_count=3): "synthwave, energetic beat, retro-futurism"
                Example Output (for tag_count=1): "ambient"
                Output:
            """

            genre_payload = {
                "model": ollama_model_name,
                "prompt": genre_prompt,
                "stream": False,
                "keep_alive": 0
            }

            raw_genre_output = self._call_ollama_api(api_url_base, genre_payload)
            print(f"SceneGeniusAutocreator: Raw LLM output for genre: '{raw_genre_output}'")

            # Enhanced cleaning for raw_genre_output
            cleaned_genre_output = raw_genre_output.strip()
            common_conversational_prefixes = [
                "output:", "here are the tags:", "i can help you learn about",
                "hello! based on your request,", "i've generated the following tags:",
                "the generated tags are:", "here's your genre output:",
                "based on the concept, the genre tags are:",
                "the genre tags for your project are:",
                "i have generated the genre tags as requested:",
                "please find the genre tags below:",
                "here's a comma-separated list of genres:",
                "genre tags:",
                "is a genre of music and dance that was popular in the late 1970s.",
                "the disco era was characterized by:",
                "key elements of the disco sound include:",
                "would you like to know more about the history of disco or hear some famous disco songs?",
                "disco is a genre of music and dance that was popular in the late 1970s."
            ]

            lower_cleaned_genre_output = cleaned_genre_output.lower()
            for phrase in common_conversational_prefixes:
                if lower_cleaned_genre_output.startswith(phrase):
                    cleaned_genre_output = cleaned_genre_output[len(phrase):].strip()
                    break
            cleaned_genre_output = cleaned_genre_output.strip()

            genre_tags_list = [
                tag.strip().lower() for tag in cleaned_genre_output.split(',')
                if tag.strip()
            ]
            genre_tags = ", ".join(genre_tags_list)
            print(f"SceneGeniusAutocreator: Parsed GENRE_TAGS: '{genre_tags}'")

        # --- Stage 2: Lyrics/Script Decision and Generation ---
        print(f"SceneGeniusAutocreator: Initiating Lyrics/Script Generation for genre: '{genre_tags}'")
        if prompt_lyrics_decision_and_generation.strip():
            lyrics_or_script = prompt_lyrics_decision_and_generation.strip()
            print(f"SceneGeniusAutocreator: Lyrics/Script override detected. Using provided value: '{lyrics_or_script}'")
        else:
            lyrics_prompt = f"""
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
                {"The user has requested that you FORCE the generation of lyrics, so do not output [instrumental]." if force_lyrics_generation else ""}

                Consider the creative brief and the genre.
                Output:
            """

            lyrics_payload = {
                "model": ollama_model_name,
                "prompt": lyrics_prompt,
                "stream": False,
                "keep_alive": 0
            }

            raw_lyrics_output = self._call_ollama_api(api_url_base, lyrics_payload)
            print(f"SceneGeniusAutocreator: Raw LLM output for lyrics/script: '{raw_lyrics_output}'")

            cleaned_lyrics = raw_lyrics_output.strip()
            intro_phrases = (
                "i'll generate lyrics that align with these genres",
                "here are the lyrics:",
                "lyrics:",
                "output:",
                "here is the lyrical script:",
                "lyrical script:",
                "i have decided to generate lyrics.",
                "based on your request, here are the lyrics:",
                "the lyrical script is as follows:",
            )

            lower_cleaned_lyrics = cleaned_lyrics.lower()
            for phrase in intro_phrases:
                if lower_cleaned_lyrics.startswith(phrase):
                    cleaned_lyrics = cleaned_lyrics[len(phrase):].strip()
                    if cleaned_lyrics.startswith('\n'):
                        cleaned_lyrics = cleaned_lyrics.lstrip('\n')
                    break

            if cleaned_lyrics.lower() == "[instrumental]":
                lyrics_or_script = "[instrumental]"
            else:
                lyrics_or_script = cleaned_lyrics.strip()

            print(f"SceneGeniusAutocreator: Parsed LYRICS_OR_SCRIPT: '{lyrics_or_script}'")

        # --- Stage 3: Duration Generation ---
        print(f"SceneGeniusAutocreator: Initiating Duration Generation.")
        if prompt_duration_generation.strip():
            print(f"SceneGeniusAutocreator: Duration override detected. Attempting to parse provided value: '{prompt_duration_generation.strip()}'")
            try:
                parsed_duration = float(prompt_duration_generation.strip())
                if not (float(min_total_seconds) <= parsed_duration <= float(max_total_seconds)):
                    print(f"SceneGeniusAutocreator Warning: Provided duration {parsed_duration} is outside the allowed range [{min_total_seconds}.0, {max_total_seconds}.0]. Clamping to nearest boundary.")
                    total_seconds = max(float(min_total_seconds), min(parsed_duration, float(max_total_seconds)))
                else:
                    total_seconds = parsed_duration
            except ValueError:
                print(f"SceneGeniusAutocreator Error: Could not parse duration from provided override: '{prompt_duration_generation.strip()}'. Defaulting to {float(min_total_seconds)} seconds.")
                total_seconds = float(min_total_seconds)
        else:
            duration_prompt = f"""
                You are an AI assistant tasked with generating a duration for a creative project.
                Based on the following information, provide a single floating-point number representing the total duration in seconds.

                Initial Concept: "{initial_concept_prompt}"
                Generated Genres: "{genre_tags}"
                Lyrics/Script: "{lyrics_or_script}"

                The duration MUST be between {min_total_seconds}.0 and {max_total_seconds}.0 seconds, inclusive.
                Ensure your output is ONLY the floating-point number, with no extra text, explanations, or formatting.
                Example Output: 180.5
                Output:
            """

            duration_payload = {
                "model": ollama_model_name,
                "prompt": duration_prompt,
                "stream": False,
                "keep_alive": 0
            }

            raw_duration_output = self._call_ollama_api(api_url_base, duration_payload)
            print(f"SceneGeniusAutocreator: Raw LLM output for duration: '{raw_duration_output}'")

            try:
                parsed_duration = float(raw_duration_output.strip())
                if not (float(min_total_seconds) <= parsed_duration <= float(max_total_seconds)):
                    print(f"SceneGeniusAutocreator Warning: Generated duration {parsed_duration} is outside the allowed range [0.0, 10.0]. Clamping to nearest boundary.")
                    total_seconds = max(float(min_total_seconds), min(parsed_duration, float(max_total_seconds)))
                else:
                    total_seconds = parsed_duration
            except ValueError:
                print(f"SceneGeniusAutocreator Error: Could not parse duration from LLM output: '{raw_duration_output}'. Defaulting to {float(min_total_seconds)} seconds.")
                total_seconds = float(min_total_seconds)

            print(f"SceneGeniusAutocreator: Parsed TOTAL_SECONDS: {total_seconds}")

        # --- Stage 4: Noise Decay Strength Generation ---
        print(f"SceneGeniusAutocreator: Initiating Noise Decay Strength Generation.")
        if prompt_noise_decay_generation.strip():
            print(f"SceneGeniusAutocreator: Noise Decay Strength override detected. Attempting to parse provided value: '{prompt_noise_decay_generation.strip()}'")
            try:
                parsed_noise_decay = float(prompt_noise_decay_generation.strip())
                if not (0.0 <= parsed_noise_decay <= 10.0):
                    print(f"SceneGeniusAutocreator Warning: Provided noise decay strength {parsed_noise_decay} is outside the allowed range [0.0, 10.0]. Clamping to nearest boundary.")
                    noise_decay_strength = max(0.0, min(parsed_noise_decay, 10.0))
                else:
                    noise_decay_strength = parsed_noise_decay
            except ValueError:
                print(f"SceneGeniusAutocreator Error: Could not parse noise decay strength from provided override: '{prompt_noise_decay_generation.strip()}'. Defaulting to 5.0.")
                noise_decay_strength = 5.0
        else:
            noise_decay_prompt = f"""
                You are an AI assistant tasked with generating a noise decay strength for image/video generation.
                Based on the following creative concept, provide a single floating-point number between 0.0 and 10.0 (inclusive) representing the noise decay strength.
                A higher value means more noise decay, leading to a "cleaner" or more "finished" look. A lower value retains more "raw" or "gritty" noise. Testing has shown a range of 1.5-4 to be ideal in most cases.

                Initial Concept: "{initial_concept_prompt}"
                Generated Genres: "{genre_tags}"
                Lyrics/Script: "{lyrics_or_script}"
                Total Duration: {total_seconds} seconds

                Ensure your output is ONLY the floating-point number, with no extra text, explanations, or formatting.
                Example Output: 2.5
                Output:
            """

            noise_decay_payload = {
                "model": ollama_model_name,
                "prompt": noise_decay_prompt,
                "stream": False,
                "keep_alive": 0
            }

            raw_noise_decay_output = self._call_ollama_api(api_url_base, noise_decay_payload)
            print(f"SceneGeniusAutocreator: Raw LLM output for noise decay strength: '{raw_noise_decay_output}'")

            try:
                parsed_noise_decay = float(raw_noise_decay_output.strip())
                if not (0.0 <= parsed_noise_decay <= 10.0):
                    print(f"SceneGeniusAutocreator Warning: Generated noise decay strength {parsed_noise_decay} is outside the allowed range [0.0, 10.0]. Clamping to nearest boundary.")
                    noise_decay_strength = max(0.0, min(parsed_noise_decay, 10.0))
                else:
                    noise_decay_strength = parsed_noise_decay
            except ValueError:
                print(f"SceneGeniusAutocreator Error: Could not parse noise decay strength from LLM output: '{raw_noise_decay_output}'. Defaulting to 5.0.")
                noise_decay_strength = 5.0

            print(f"SceneGeniusAutocreator: Parsed NOISE_DECAY_STRENGTH: {noise_decay_strength}")

        # --- Stage 5: APG YAML Parameters Generation ---
        print(f"SceneGeniusAutocreator: Initiating APG YAML Parameters Generation.")
        if prompt_apg_yaml_generation.strip():
            print(f"SceneGeniusAutocreator: APG YAML override detected. Attempting to parse provided value.")
            apg_data = None
            try:
                cleaned_apg_yaml = prompt_apg_yaml_generation.strip()
                if cleaned_apg_yaml.startswith("```yaml"):
                    cleaned_apg_yaml = cleaned_apg_yaml[len("```yaml"):].strip()
                if cleaned_apg_yaml.endswith("```"):
                    cleaned_apg_yaml = cleaned_apg_yaml[:-len("```")].strip()
                if cleaned_apg_yaml.lower().startswith("output:"):
                    cleaned_apg_yaml = cleaned_apg_yaml[len("output:"):].strip()
                if cleaned_apg_yaml.lower().startswith("```"):
                    cleaned_apg_yaml = cleaned_apg_yaml[len("```"):].strip()

                apg_data = yaml.safe_load(cleaned_apg_yaml)

                if apg_data is None or not isinstance(apg_data, dict):
                    print("SceneGeniusAutocreator Warning: Provided APG YAML is empty or not a dictionary. Using default APG YAML.")
                    apg_yaml_params = self.default_apg_yaml
                else:
                    apg_yaml_params = yaml.dump(apg_data, Dumper=CustomDumper, indent=2, sort_keys=False)
            except yaml.YAMLError as e:
                print(f"SceneGeniusAutocreator Error: Could not parse APG YAML from provided override (YAMLError): {e}. Using default APG YAML.")
                apg_yaml_params = self.default_apg_yaml
            except Exception as e:
                print(f"SceneGeniusAutocreator Error: An unexpected error occurred during APG YAML override processing: {e}. Using default APG YAML.")
                apg_yaml_params = self.default_apg_yaml
        else:
            actual_apg_prompt = self.DEFAULT_APG_PROMPT

            apg_yaml_prompt_content = actual_apg_prompt.format(
                initial_concept_prompt=initial_concept_prompt,
                genre_tags=genre_tags,
                lyrics_or_script=lyrics_or_script,
                total_seconds=total_seconds,
                noise_decay_strength=noise_decay_strength
            )

            apg_yaml_payload = {
                "model": ollama_model_name,
                "prompt": apg_yaml_prompt_content,
                "stream": False,
                "keep_alive": 0
            }

            raw_apg_yaml_output = self._call_ollama_api(api_url_base, apg_yaml_payload)
            print(f"SceneGeniusAutocreator: Raw LLM output for APG YAML: '{raw_apg_yaml_output}'")

            apg_data = None
            apg_yaml_params = ""
            try:
                cleaned_apg_yaml = raw_apg_yaml_output.strip()

                if cleaned_apg_yaml.startswith("```yaml"):
                    cleaned_apg_yaml = cleaned_apg_yaml[len("```yaml"):].strip()
                if cleaned_apg_yaml.endswith("```"):
                    cleaned_apg_yaml = cleaned_apg_yaml[:-len("```")].strip()
                if cleaned_apg_yaml.lower().startswith("output:"):
                    cleaned_apg_yaml = cleaned_apg_yaml[len("output:"):].strip()
                if cleaned_apg_yaml.lower().startswith("```"):
                    cleaned_apg_yaml = cleaned_apg_yaml[len("```"):].strip()

                print(f"SceneGeniusAutocreator: Cleaned APG YAML string before parsing attempt: \n---\n{cleaned_apg_yaml}\n---")

                apg_data = yaml.safe_load(cleaned_apg_yaml)

                if apg_data is None or not isinstance(apg_data, dict):
                    print("SceneGeniusAutocreator Warning: LLM generated empty or non-dictionary YAML for APG. Using default APG YAML.")
                    apg_yaml_params = self.default_apg_yaml
                else:
                    apg_yaml_params = yaml.dump(apg_data, Dumper=CustomDumper, indent=2, sort_keys=False)

            except yaml.YAMLError as e:
                print(f"SceneGeniusAutocreator Error: Could not parse APG YAML from LLM output (YAMLError): {e}. Raw output: '{raw_apg_yaml_output}'. Using default APG YAML.")
                apg_yaml_params = self.default_apg_yaml
            except Exception as e:
                print(f"SceneGeniusAutocreator Error: An unexpected error occurred during APG YAML processing: {e}. Raw output: '{raw_apg_yaml_output}'. Using default APG YAML.")
                apg_yaml_params = self.default_apg_yaml

        apg_yaml_params = re.sub(
            r'(\s*)dims:\s*\n\s*- -2\s*\n\s*- -1',
            r'\1dims: [-2, -1]',
            apg_yaml_params
        )

        print(f"SceneGeniusAutocreator: Final APG_YAML_PARAMS output: \n---\n{apg_yaml_params}\n---")


        # --- Stage 6: Sampler YAML Parameters Generation ---
        print(f"SceneGeniusAutocreator: Initiating Sampler YAML Parameters Generation for '{sampler_version}'.")
        if prompt_sampler_yaml_generation.strip():
            print(f"SceneGeniusAutocreator: Sampler YAML override detected. Attempting to parse provided value.")
            sampler_data = None
            try:
                cleaned_sampler_yaml = prompt_sampler_yaml_generation.strip()
                if cleaned_sampler_yaml.startswith("```yaml"):
                    cleaned_sampler_yaml = cleaned_sampler_yaml[len("```yaml"):].strip()
                if cleaned_sampler_yaml.endswith("```"):
                    cleaned_sampler_yaml = cleaned_sampler_yaml[:-len("```")].strip()
                if cleaned_sampler_yaml.lower().startswith("output:"):
                    cleaned_sampler_yaml = cleaned_sampler_yaml[len("output:"):].strip()
                if cleaned_sampler_yaml.lower().startswith("```"):
                    cleaned_sampler_yaml = cleaned_sampler_yaml[len("```"):].strip()

                sampler_data = yaml.safe_load(cleaned_sampler_yaml)

                if sampler_data is None or not isinstance(sampler_data, dict):
                    print("SceneGeniusAutocreator Warning: Provided Sampler YAML is empty or not a dictionary. Using default Sampler YAML.")
                    sampler_yaml_params = default_sampler_yaml_to_use
                else:
                    sampler_yaml_params = yaml.dump(sampler_data, Dumper=CustomDumper, indent=2, sort_keys=False)
            except yaml.YAMLError as e:
                print(f"SceneGeniusAutocreator Error: Could not parse Sampler YAML from provided override (YAMLError): {e}. Using default Sampler YAML.")
                sampler_yaml_params = default_sampler_yaml_to_use
            except Exception as e:
                print(f"SceneGeniusAutocreator Error: An unexpected error occurred during Sampler YAML override processing: {e}. Using default Sampler YAML.")
                sampler_yaml_params = default_sampler_yaml_to_use
        else:
            # Dynamically build the prompt based on selected sampler version
            sampler_yaml_prompt_content = self.DEFAULT_SAMPLER_PROMPT_BASE.format(
                initial_concept_prompt=initial_concept_prompt,
                genre_tags=genre_tags,
                lyrics_or_script=lyrics_or_script,
                total_seconds=total_seconds,
                apg_yaml_params=apg_yaml_params, # Pass the generated APG YAML as context
                fbg_section=fbg_section_to_include # Insert the FBG section if applicable
            )

            sampler_yaml_payload = {
                "model": ollama_model_name,
                "prompt": sampler_yaml_prompt_content,
                "stream": False,
                "keep_alive": 0
            }

            raw_sampler_yaml_output = self._call_ollama_api(api_url_base, sampler_yaml_payload)
            print(f"SceneGeniusAutocreator: Raw LLM output for Sampler YAML: '{raw_sampler_yaml_output}'")

            sampler_data = None
            sampler_yaml_params = ""
            try:
                cleaned_sampler_yaml = raw_sampler_yaml_output.strip()

                if cleaned_sampler_yaml.startswith("```yaml"):
                    cleaned_sampler_yaml = cleaned_sampler_yaml[len("```yaml"):].strip()
                if cleaned_sampler_yaml.endswith("```"):
                    cleaned_sampler_yaml = cleaned_sampler_yaml[:-len("```")].strip()
                if cleaned_sampler_yaml.lower().startswith("output:"):
                    cleaned_sampler_yaml = cleaned_sampler_yaml[len("output:"):].strip()
                if cleaned_sampler_yaml.lower().startswith("```"):
                    cleaned_sampler_yaml = cleaned_sampler_yaml[len("```"):].strip()

                print(f"SceneGeniusAutocreator: Cleaned Sampler YAML string before parsing attempt: \n---\n{cleaned_sampler_yaml}\n---")

                sampler_data = yaml.safe_load(cleaned_sampler_yaml)

                if sampler_data is None or not isinstance(sampler_data, dict):
                    print("SceneGeniusAutocreator Warning: LLM generated empty or non-dictionary YAML for Sampler. Using default Sampler YAML.")
                    sampler_yaml_params = default_sampler_yaml_to_use
                else:
                    sampler_yaml_params = yaml.dump(sampler_data, Dumper=CustomDumper, indent=2, sort_keys=False)

            except yaml.YAMLError as e:
                print(f"SceneGeniusAutocreator Error: Could not parse Sampler YAML from LLM output (YAMLError): {e}. Raw output: '{raw_sampler_yaml_output}'. Using default Sampler YAML.")
                sampler_yaml_params = default_sampler_yaml_to_use
            except Exception as e:
                print(f"SceneGeniusAutocreator Error: An unexpected error occurred during Sampler YAML processing: {e}. Raw output: '{raw_sampler_yaml_output}'. Using default Sampler YAML.")
                sampler_yaml_params = default_sampler_yaml_to_use

            print(f"SceneGeniusAutocreator: Final SAMPLER_YAML_PARAMS output: \n---\n{sampler_yaml_params}\n---")


        return (genre_tags, lyrics_or_script, total_seconds, noise_decay_strength, apg_yaml_params, sampler_yaml_params, seed,)

# A dictionary that contains all nodes you want to export with your module
# This is how ComfyUI discovers your nodes.
NODE_CLASS_MAPPINGS = {
    "SceneGeniusAutocreator": SceneGeniusAutocreator
}

# A dictionary that contains the friendly names for the nodes,
# which will be displayed in the ComfyUI menu.
# This makes it easier for users to find your nodes.
NODE_DISPLAY_NAME_MAPPINGS = {
    "SceneGeniusAutocreator": "Scene Genius Autocreator"
}
