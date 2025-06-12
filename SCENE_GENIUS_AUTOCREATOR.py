# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# ████ SCENEGENIUS AUTOCREATOR v0.1.1 – Optimized for Ace-Step Audio/Video ████▓▒░
# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀

# ░▒▓ ORIGIN & DEV:
#   • Cast into the void by: MDMAchine (OG SysOp)
#   • Original mind behind SceneGenius Autocreator
#   • Initial ComfyUI adaptation by: Gemini (Google)
#   • Enhanced & refined by: MDMAchine & Gemini
#   • Critical optimizations & bugfixes: Gemini
#   • Final polish: MDMAchine
#   • License: Apache 2.0 — No cursed floppy disks allowed

# ░▒▓ DESCRIPTION:
#   A multi-stage AI creative weapon node for ComfyUI.
#   Designed to automate Ace-Step diffusion content generation,
#   channeling the chaotic spirit of the demoscene and BBS era.
#   Produces authentic genres, adaptive lyrics, precise durations,
#   and finely tuned APG + Sampler configs with ease.
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
    apg_scale: 0
    cfg: 4

  - start_sigma: 0.85
    apg_scale: 5.0
    predict_image: true
    cfg: 4
    mode: pre_alt2
    update_blend_mode: lerp
    dims: [-2, -1]
    momentum: 0.7
    norm_threshold: 3.0
    eta: 0.0

  - start_sigma: 0.70
    apg_scale: 4.5
    predict_image: true
    cfg: 3.9
    mode: pre_alt2
    update_blend_mode: lerp
    dims: [-2, -1]
    momentum: 0.6
    norm_threshold: 2.5
    eta: 0.0

  - start_sigma: 0.55
    apg_scale: 4
    predict_image: true
    cfg: 3.8
    mode: pure_apg
    momentum: 0.0

  - start_sigma: 0.40
    apg_scale: 3.8
    predict_image: false
    cfg: 3.7
    momentum: 0.0

  - start_sigma: 0.30
    apg_scale: 3.5
    predict_image: false
    cfg: 3.6
    momentum: 0.0

  - start_sigma: 0.15
    apg_scale: 0
    cfg: 3.5
    momentum: 0.0

"""
        # Default Sampler YAML to use if LLM output is unparseable or invalid
        self.default_sampler_yaml = """
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
"""
        # Full default prompt for Sampler YAML generation
        self.DEFAULT_SAMPLER_PROMPT = """
You are an expert diffusion model sampling and parameter configuration specialist, specifically for the "Ping-Pong Sampler" (a custom sampler highly optimized for Ace-Step audio and video diffusion models). Your task is to generate YAML-formatted parameters for this sampler.

**Your understanding of the creative context:**
* **Initial Concept:** "{initial_concept_prompt}" (The core idea guiding the visual/audio style)
* **Generated Genres:** "{genre_tags}" (E.g., 'lo-fi' suggests softer transitions, 'punk' might tolerate more abruptness)
* **Lyrics/Script Status:** "{lyrics_or_script}" (A narrative might need higher coherence; instrumental might allow more abstract patterns)
* **Total Duration (seconds):** "{total_seconds}" (Longer durations emphasize the need for temporal coherence)
* **Generated APG Parameters (Crucial context for overall guidance strategy):** "{apg_yaml_params}"

**Your task: Generate a VALID YAML string containing parameters for the Ping-Pong Sampler. Be acutely aware of the impact of each parameter.**

**CRITICAL GUIDANCE FOR STABILITY AND QUALITY (Read Carefully!):**
Based on extensive testing and user feedback, certain Ping-Pong Sampler parameters (`start_sigma_index`, `end_sigma_index`, `blend_mode`, `step_blend_mode`) are **extremely sensitive and have a high likelihood of producing noisy, corrupted, or unfinished outputs if changed from their default values.** These parameters control fundamental aspects of the sampler's internal blending and sigma schedule traversal. **Therefore, you MUST prioritize using the default values for these specific parameters unless there is an absolute, well-understood technical necessity for alteration (which is exceedingly rare and likely beyond typical creative adjustments).** Focus your creative adjustments on `step_random_mode`, `step_size`, `first_ancestral_step`, and `last_ancestral_step`.

**Key Parameters & How to Think About Them:**

* **`verbose` (boolean):** Set to `true` or `false` to enable/disable debug messages for the sampler.
* **`step_random_mode` (string):** **How the internal random number generator (RNG) seed is managed across sampling steps.** This directly influences the temporal coherence (smoothness over time) of generated frames or audio segments.
    * **Purpose:** Controls the "flicker" or consistency.
    * **Decision-Making:**
        * `off`: LEAST random variation per step.** Ideal for highly consistent, stable, and coherent output across sequential frames (video) or audio segments. for most coherent outputs.
        * `block`: Randomness changes only after a `step_size` number of steps. Useful if you want visually/audibly distinct "blocks" or segments that are internally consistent but change abruptly at intervals. Works very well in many cases.
        * `step`: Randomness changes with *every single step*. This can lead to very chaotic, flickering visuals or rapidly changing audio textures.
        * **LLM Decision**: Consider the `initial_concept_prompt` and `GENRE_TAGS`. For "smooth," "cinematic," or "ambient" outputs, favor `off`. For "segmented" or "rhythmic" changes, consider `block`. For "chaos," `step`.
* **`step_size` (integer):** The size of the "block" or multiplier for `step_random_mode`. Default is `5`.
    * **Purpose:** Defines the granularity of `block` mode.
    * **Decision-Making:** Only relevant if `step_random_mode` is `block` or `step`. Integers like `4`,`5` or `10` are common. Smaller values mean more frequent shifts; larger values mean longer, more consistent segments.
* **`first_ancestral_step` (integer):** The 0-based index in the sigma schedule where ancestral noise injection begins.
* **`last_ancestral_step` (integer):** The 0-based index in the sigma schedule where ancestral noise injection ends.
    * **Purpose of Ancestral Noise:** Ancestral steps re-inject a small, calculated amount of noise back into the latent. This is a crucial technique to prevent "dead spots," over-smoothing, or loss of detail that can occur in purely deterministic samplers. It helps maintain a vibrant, detailed output, especially for complex generative tasks like video/audio.
    * **Decision-Making:** These two parameters define the "active ancestral mixing period."
        * **LLM Decision**: These values should define a window within the total sampling steps where this noise mixing is beneficial. Avoid starting too early (when latents are very noisy) or ending too late (when details are critical). For typical 50-step processes, values like `first_ancestral_step: 10-15` and `last_ancestral_step: 30-40` might be a good starting point to encompass the critical detail-forming stages.
* **`start_sigma_index` (integer):** The 0-based index in the sigma schedule to start sampling from.
    * **Purpose:** Determines which step of the sigma schedule the sampling process *begins* at.
    * **Decision-Making:** **ALWAYS set this to `0` (default). Do not change.** Changing this means skipping initial denoising steps, leading to fundamentally incomplete and noisy results.
* **`end_sigma_index` (integer):** The 0-based index in the sigma schedule to end sampling early.
    * **Purpose:** Determines which step of the sigma schedule the sampling process *ends* at.
    * **Decision-Making:** **ALWAYS set this to `-1` (default, meaning use all steps). Do not change.** Changing this means prematurely stopping the denoising process, resulting in unfinished and noisy outputs.
* **`enable_clamp_output` (boolean):** `true` to clamp the final output values to `[-1, 1]`; `false` otherwise. **For your model's optimal sound quality, it is generally desired to NOT limit the output to `[-1, 1]`. Therefore, for best results, set this to `false`.** Setting to `true` may be considered if strict numerical stability within a specific range is absolutely required, but this is less common for audio models that benefit from a wider dynamic range.
* **`blend_mode` (string):** The blending mode for general internal operations.
    * **Purpose:** Controls how certain internal latent values are combined.
    * **Decision-Making:** **ALWAYS set this to `lerp` (linear interpolation, default). Do not change.** Other modes (`a_only`, `b_only`) are for highly experimental, internal use and will almost certainly produce noise or incorrect results.
* **`step_blend_mode` (string):** The blending mode specifically for ancestral steps.
    * **Purpose:** Controls the blending during ancestral noise re-injection.
    * **Decision-Making:** **ALWAYS set this to `lerp` (linear interpolation, default). Do not change.** Similar to `blend_mode`, other options are highly experimental and detrimental to stable output.

**Your output must ONLY be the valid YAML string, with no conversational filler, explanations, or additional text.**

**Example Output Format:**
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

**IMPORTANT: When generating the actual YAML, DO NOT include the ```yaml or ``` delimiters in your final output. These are only for demonstration in this example.**
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
    * **Decision-Making:** The **optimal performance for `cfg` is typically in the `3.0` to `5.0` range.** While `cfg` can be kept consistent across rules, or adjusted **slightly** (e.g., `0.1` to `0.2` increments/decrements as seen in the example) in phases requiring more direct prompt adherence or subtle shifts in generation, **always keep it within the `3.0-5.0` range for best results** to avoid over-guidance or under-guidance artifacts.
* **`predict_image` (boolean):** **Determines what the model is "predicting" at that step (image content vs. noise).**
    * **Purpose:** Influences how APG guidance is calculated and applied. `true` focuses guidance on the evolving image content, `false` on noise.
    * **Decision-Making:**
        * Generally set to `true` for rules where `apg_scale` is active, especially in early and mid-phases, as this focuses guidance on the evolving image content and often leads to smoother, more coherent results. This is particularly effective when `mode` is `pre_alt2` or `pure_apg` with active `momentum`.
        * Consider `false` (noise prediction) for later stages (lower sigmas, e.g., `start_sigma` below `0.5` or `0.4`), especially when `apg_scale` is lower and the goal is subtle refinement of noise, or if experimenting with alternative guidance strategies. The example's choice of `false` for `0.40` and `0.30` is a good illustration of this.
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
3.  **Mid-Process Refinement (Mid-range Sigmas):** Rules at mid-range sigmas (e.g., `0.55`, `0.40`, `0.30`) should refine details. You can transition `mode` to `pure_apg` here. `apg_scale` might be slightly reduced, and `predict_image` can be `true` or `false` based on whether you want to refine the image or noise prediction at these stages. Continue to consider `momentum` if needed for continued smoothness, and always include `dims: [-2, -1]` when `apg_scale` is active.
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
    cfg: 4
    mode: pre_alt2
    update_blend_mode: lerp
    dims: [-2, -1]
    momentum: 0.7
    norm_threshold: 3.0
    eta: 0.0

  - start_sigma: 0.70
    apg_scale: 4.5
    predict_image: true
    cfg: 3.9
    mode: pre_alt2
    update_blend_mode: lerp
    dims: [-2, -1]
    momentum: 0.6
    norm_threshold: 2.5
    eta: 0.0

  - start_sigma: 0.55
    apg_scale: 4
    predict_image: true
    cfg: 3.8
    mode: pure_apg
    momentum: 0.0

  - start_sigma: 0.40
    apg_scale: 3.8
    predict_image: false
    cfg: 3.7
    momentum: 0.0

  - start_sigma: 0.30
    apg_scale: 3.5
    predict_image: false
    cfg: 3.6
    momentum: 0.0

  - start_sigma: 0.15
    apg_scale: 0
    cfg: 3.5
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
            },
            "optional": {
                "prompt_genre_generation": ("STRING", {"multiline": True, "default": "", "tooltip": "Override the genre tags output directly. If provided, the LLM will be skipped for genre generation."}),
                "prompt_lyrics_decision_and_generation": ("STRING", {"multiline": True, "default": "", "tooltip": "Override the lyrics/script output directly. If provided, the LLM will be skipped for lyrics generation."}),
                "prompt_duration_generation": ("STRING", {"multiline": True, "default": "", "tooltip": "Override the duration output directly (e.g., '120.5'). If provided, the LLM will be skipped for duration generation."}),
                "prompt_noise_decay_generation": ("STRING", {"multiline": True, "default": "", "tooltip": "Override the noise decay strength output directly (e.g., '7.5'). If provided, the LLM will be skipped for noise decay generation."}), # New input for direct override
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
                         min_total_seconds, max_total_seconds, seed,
                         prompt_genre_generation="", prompt_lyrics_decision_and_generation="",
                         prompt_duration_generation="", prompt_noise_decay_generation="", # Added new optional input
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

        # --- Stage 1: Genre Generation ---
        print(f"SceneGeniusAutocreator: Initiating Genre Generation for concept: '{initial_concept_prompt}'")
        if prompt_genre_generation.strip(): # Check if override prompt is provided
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

            if not prompt_genre_generation: # This condition is now redundant due to the outer if/else, but kept for clarity
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
            else:
                genre_prompt = prompt_genre_generation.format(
                    initial_concept_prompt=initial_concept_prompt,
                    tag_count=tag_count,
                    blend_description=blend_description,
                    excluded_tags_clause="Avoid generating any of these tags: " + excluded_tags + "." if excluded_tags else ""
                )

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
                # Added more general conversational prefixes based on user feedback
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
                    break # Stop after finding the first match

            # Remove any leading/trailing newlines or whitespace that might remain
            cleaned_genre_output = cleaned_genre_output.strip()

            # Split and clean individual tags
            genre_tags_list = [
                tag.strip().lower() for tag in cleaned_genre_output.split(',')
                if tag.strip() # Ensure empty strings are not included
            ]
            genre_tags = ", ".join(genre_tags_list)
            print(f"SceneGeniusAutocreator: Parsed GENRE_TAGS: '{genre_tags}'")

        # --- Stage 2: Lyrics/Script Decision and Generation ---
        print(f"SceneGeniusAutocreator: Initiating Lyrics/Script Generation for genre: '{genre_tags}'")
        if prompt_lyrics_decision_and_generation.strip(): # Check if override prompt is provided
            lyrics_or_script = prompt_lyrics_decision_and_generation.strip()
            print(f"SceneGeniusAutocreator: Lyrics/Script override detected. Using provided value: '{lyrics_or_script}'")
        else:
            if not prompt_lyrics_decision_and_generation:
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
            else:
                lyrics_prompt = prompt_lyrics_decision_and_generation.format(
                    genre_tags=genre_tags,
                    initial_concept_prompt=initial_concept_prompt,
                    force_lyrics_clause="The user has requested that you FORCE the generation of lyrics, so do not output [instrumental]." if force_lyrics_generation else ""
                )

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
        if prompt_duration_generation.strip(): # Check if override prompt is provided
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
            if not prompt_duration_generation:
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
            else:
                duration_prompt = prompt_duration_generation.format(
                    initial_concept_prompt=initial_concept_prompt,
                    genre_tags=genre_tags,
                    lyrics_or_script=lyrics_or_script,
                    min_total_seconds=min_total_seconds,
                    max_total_seconds=max_total_seconds
                )

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
                    print(f"SceneGeniusAutocreator Warning: Generated duration {parsed_duration} is outside the allowed range [{min_total_seconds}.0, {max_total_seconds}.0]. Clamping to nearest boundary.")
                    total_seconds = max(float(min_total_seconds), min(parsed_duration, float(max_total_seconds)))
                else:
                    total_seconds = parsed_duration
            except ValueError:
                print(f"SceneGeniusAutocreator Error: Could not parse duration from LLM output: '{raw_duration_output}'. Defaulting to {float(min_total_seconds)} seconds.")
                total_seconds = float(min_total_seconds)

            print(f"SceneGeniusAutocreator: Parsed TOTAL_SECONDS: {total_seconds}")

        # --- Stage 4: Noise Decay Strength Generation ---
        print(f"SceneGeniusAutocreator: Initiating Noise Decay Strength Generation.")
        if prompt_noise_decay_generation.strip(): # Check if override prompt is provided
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
        if prompt_apg_yaml_generation.strip(): # Check if override prompt is provided
            print(f"SceneGeniusAutocreator: APG YAML override detected. Attempting to parse provided value.")
            apg_data = None
            try:
                cleaned_apg_yaml = prompt_apg_yaml_generation.strip()
                # Stripping markdown code block fences and common LLM prefixes if user provided them
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
            # If prompt_apg_yaml_generation is empty, use the class-level default prompt
            actual_apg_prompt = prompt_apg_yaml_generation if prompt_apg_yaml_generation else self.DEFAULT_APG_PROMPT

            apg_yaml_prompt_content = actual_apg_prompt.format(
                initial_concept_prompt=initial_concept_prompt,
                genre_tags=genre_tags,
                lyrics_or_script=lyrics_or_script,
                total_seconds=total_seconds,
                noise_decay_strength=noise_decay_strength
            )

            apg_yaml_payload = {
                "model": ollama_model_name,
                "prompt": apg_yaml_prompt_content, # Use the formatted prompt
                "stream": False,
                "keep_alive": 0
            }

            raw_apg_yaml_output = self._call_ollama_api(api_url_base, apg_yaml_payload)
            print(f"SceneGeniusAutocreator: Raw LLM output for APG YAML: '{raw_apg_yaml_output}'")

            # Process LLM output for APG YAML: clean, attempt to load/dump for formatting, or use default
            apg_data = None
            apg_yaml_params = ""
            try:
                cleaned_apg_yaml = raw_apg_yaml_output.strip()

                # Robust stripping of markdown code block fences and common LLM prefixes
                # This handles cases where ```yaml, ```, or "Output:" might wrap the actual YAML
                if cleaned_apg_yaml.startswith("```yaml"):
                    cleaned_apg_yaml = cleaned_apg_yaml[len("```yaml"):].strip()
                if cleaned_apg_yaml.endswith("```"):
                    cleaned_apg_yaml = cleaned_apg_yaml[:-len("```")].strip()
                if cleaned_apg_yaml.lower().startswith("output:"):
                    cleaned_apg_yaml = cleaned_apg_yaml[len("output:"):].strip()
                if cleaned_apg_yaml.lower().startswith("```"): # Catch cases where ``` is not followed by yaml
                    cleaned_apg_yaml = cleaned_apg_yaml[len("```"):].strip()

                print(f"SceneGeniusAutocreator: Cleaned APG YAML string before parsing attempt: \n---\n{cleaned_apg_yaml}\n---")

                # Attempt to load and then dump the YAML to ensure it's valid and consistently formatted
                apg_data = yaml.safe_load(cleaned_apg_yaml)

                if apg_data is None or not isinstance(apg_data, dict):
                    # If LLM returned empty/invalid YAML, or not a dictionary (expected top-level for rules)
                    print("SceneGeniusAutocreator Warning: LLM generated empty or non-dictionary YAML for APG. Using default APG YAML.")
                    apg_yaml_params = self.default_apg_yaml
                else:
                    # Use CustomDumper for specific list formatting
                    apg_yaml_params = yaml.dump(apg_data, Dumper=CustomDumper, indent=2, sort_keys=False)

            except yaml.YAMLError as e:
                print(f"SceneGeniusAutocreator Error: Could not parse APG YAML from LLM output (YAMLError): {e}. Raw output: '{raw_apg_yaml_output}'. Using default APG YAML.")
                apg_yaml_params = self.default_apg_yaml # Fallback to default on parse error
            except Exception as e:
                print(f"SceneGeniusAutocreator Error: An unexpected error occurred during APG YAML processing: {e}. Raw output: '{raw_apg_yaml_output}'. Using default APG YAML.")
                apg_yaml_params = self.default_apg_yaml # Fallback to default on any other error

        # --- Post-processing for 'dims' to force inline formatting ---
        # This regex looks for 'dims:' followed by any whitespace, then a newline,
        # then any whitespace, then '- -2', then any whitespace, then a newline,
        # then any whitespace, then '- -1'.
        # The 're.M' flag makes '^' and '$' match at the start/end of each line.
        # The 're.DOTALL' flag allows '.' to match newlines.
        # It handles variable indentation before 'dims' and before '- -2' and '- -1'.
        apg_yaml_params = re.sub(
            r'(\s*)dims:\s*\n\s*- -2\s*\n\s*- -1',
            r'\1dims: [-2, -1]',
            apg_yaml_params
        )

        print(f"SceneGeniusAutocreator: Final APG_YAML_PARAMS output: \n---\n{apg_yaml_params}\n---")


        # --- Stage 6: Sampler YAML Parameters Generation ---
        print(f"SceneGeniusAutocreator: Initiating Sampler YAML Parameters Generation.")
        if prompt_sampler_yaml_generation.strip(): # Check if override prompt is provided
            print(f"SceneGeniusAutocreator: Sampler YAML override detected. Attempting to parse provided value.")
            sampler_data = None
            try:
                cleaned_sampler_yaml = prompt_sampler_yaml_generation.strip()
                # Stripping markdown code block fences and common LLM prefixes if user provided them
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
                    sampler_yaml_params = self.default_sampler_yaml
                else:
                    sampler_yaml_params = yaml.dump(sampler_data, Dumper=CustomDumper, indent=2, sort_keys=False)
            except yaml.YAMLError as e:
                print(f"SceneGeniusAutocreator Error: Could not parse Sampler YAML from provided override (YAMLError): {e}. Using default Sampler YAML.")
                sampler_yaml_params = self.default_sampler_yaml
            except Exception as e:
                print(f"SceneGeniusAutocreator Error: An unexpected error occurred during Sampler YAML override processing: {e}. Using default Sampler YAML.")
                sampler_yaml_params = self.default_sampler_yaml
        else:
            # If prompt_sampler_yaml_generation is empty, use the class-level default prompt
            actual_sampler_prompt = prompt_sampler_yaml_generation if prompt_sampler_yaml_generation else self.DEFAULT_SAMPLER_PROMPT

            sampler_yaml_prompt_content = actual_sampler_prompt.format(
                initial_concept_prompt=initial_concept_prompt,
                genre_tags=genre_tags,
                lyrics_or_script=lyrics_or_script,
                total_seconds=total_seconds,
                apg_yaml_params=apg_yaml_params # Pass the generated APG YAML as context
            )

            sampler_yaml_payload = {
                "model": ollama_model_name,
                "prompt": sampler_yaml_prompt_content,
                "stream": False,
                "keep_alive": 0
            }

            raw_sampler_yaml_output = self._call_ollama_api(api_url_base, sampler_yaml_payload)
            print(f"SceneGeniusAutocreator: Raw LLM output for Sampler YAML: '{raw_sampler_yaml_output}'")

            # Process LLM output for Sampler YAML: clean, attempt to load/dump for formatting, or use default
            sampler_data = None
            sampler_yaml_params = ""
            try:
                cleaned_sampler_yaml = raw_sampler_yaml_output.strip()

                # Robust stripping of markdown code block fences and common LLM prefixes
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
                    sampler_yaml_params = self.default_sampler_yaml
                else:
                    # Use CustomDumper for specific list formatting
                    sampler_yaml_params = yaml.dump(sampler_data, Dumper=CustomDumper, indent=2, sort_keys=False)

            except yaml.YAMLError as e:
                print(f"SceneGeniusAutocreator Error: Could not parse Sampler YAML from LLM output (YAMLError): {e}. Raw output: '{raw_sampler_yaml_output}'. Using default Sampler YAML.")
                sampler_yaml_params = self.default_sampler_yaml # Fallback to default on parse error
            except Exception as e:
                print(f"SceneGeniusAutocreator Error: An unexpected error occurred during Sampler YAML processing: {e}. Raw output: '{raw_sampler_yaml_output}'. Using default Sampler YAML.")
                sampler_yaml_params = self.default_sampler_yaml # Fallback to default on any other error

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
