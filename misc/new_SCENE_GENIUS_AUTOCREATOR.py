# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# ████ SCENEGENIUS AUTOCREATOR v1.3.0 – Optimized for Ace-Step Audio/Video ████▓▒░
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
#   • Major Update & Refactor: Gemini (Google) & MDMAchine
#       • Integrated Dual YAML Generation: Node now correctly outputs YAML compatible
#         with *both* the Original PingPong Sampler and the enhanced PingPongSampler_Custom_FBG.
#       • Added Dynamic Parameter Adjustment for PingPongSampler_Custom_FBG (toggleable).
#       • Ensured clean, comment-free YAML output for robust PingPong sampler parsing.
#       • Preserved all original LLM generation logic and fallbacks.
#   • Final polish: MDMAchine
#   • License: Apache 2.0 — No cursed floppy disks allowed

# ░▒▓ DESCRIPTION:
#   A multi-stage AI creative weapon node for ComfyUI.
#   Designed to automate Ace-Step diffusion content generation,
#   channeling the chaotic spirit of the demoscene and BBS era.
#   Produces authentic genres, adaptive lyrics, precise durations,
#   and finely tuned APG + Sampler configs with ease.
#   Now supports *dual* PingPong Sampler output:
#     - Original PingPong Sampler (simpler parameter set)
#     - Enhanced PingPongSampler_Custom_FBG (full FBG dynamics, EMA, Conditional Blending, etc.)
#   Features dynamic parameter adjustment for the FBG sampler based on LLM context (toggleable).
#   Outputs clean YAML for maximum compatibility with the selected PingPong sampler node.
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
#   ✓ Dual PingPong Sampler compatibility (Original vs. FBG Integrated)
#   ✓ **FIXED: FBG Sampler default YAML fallback.**
#   ✓ Dynamic Parameter Adjustment for PingPongSampler_Custom_FBG based on LLM context (toggleable)
#   ✓ Clean, comment-free YAML output for PingPong Sampler compatibility

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
#   - v1.3.0 (Dual Sampler Compatibility & Dynamic Adjustment):
#       • **Major Update:** Integrated *dual* internal YAML generation logic.
#         - When 'Original PingPong Sampler' is selected, generates a simplified,
#           compatible YAML string.
#         - When 'FBG Integrated PingPong Sampler' is selected, generates the full,
#           enhanced YAML string with all new parameters.
#       • **New Feature:** Added 'Dynamic Parameter Adjustment' capability for the
#         FBG sampler.
#         - New input: `enable_dynamic_adjustment` (Boolean).
#         - Adjusts PingPong sampler defaults (pi, sigma_range_preset, blend modes,
#           EMA, etc.) based on LLM-generated genre tags and duration when enabled.
#       • **Optimization:** Refactored internal PingPong YAML generation into dedicated
#         functions for each sampler type.
#       • **Compatibility:** Ensured the generated PingPong YAML strings are clean
#         (no '#' comments) for reliable parsing by the respective sampler nodes.
#       • **Preservation:** Maintained all original LLM interaction logic, prompt templates,
#         APG generation, error handling, and fallback mechanisms.
#       • **UI Integrity:** Added only the single new `enable_dynamic_adjustment` toggle
#         to the node UI, as requested. No other UI elements were added or removed.
#       • **Version Bump:** Significant feature addition warrants a major version increase.

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


# scene_genius_autocreator.py

import json
import yaml
import re
import random
from datetime import datetime

# --- Version ---
SCENE_GENIUS_VERSION = "1.3.0"

# --- Helper function to sanitize string keys for YAML ---
def sanitize_key(key):
    """Replaces spaces and other problematic characters with underscores."""
    if not isinstance(key, str):
        key = str(key)
    # Replace spaces, hyphens, and other non-word characters (except underscore) with underscores
    sanitized = re.sub(r"[^\w]", "_", key)
    # Ensure it doesn't start with a number
    if sanitized and sanitized[0].isdigit():
        sanitized = "key_" + sanitized
    return sanitized

# --- Helper function for dynamic parameter adjustment (for FBG version) ---
def adjust_defaults_based_on_context(genre_tags, total_seconds, enable_dynamic_adjustment):
    """
    Adjusts default PingPong sampler parameters based on LLM output context.
    Only applies adjustments if enable_dynamic_adjustment is True.
    This is specifically for the FBG Integrated PingPong Sampler.
    """
    adjustments = {}
    if not enable_dynamic_adjustment:
        return adjustments

    genre_lower = genre_tags.lower() if genre_tags else ""

    # --- Dynamic Adjustments Based on Context ---
    # Adjust pi based on genre tags
    if any(word in genre_lower for word in ["ambient", "drone", "atmospheric", "calm"]):
        adjustments["pi"] = 0.25
    elif any(word in genre_lower for word in ["aggressive", "metal", "hardcore", "intense"]):
        adjustments["pi"] = 0.55

    # Adjust sigma_range_preset based on duration
    if isinstance(total_seconds, (int, float)):
        if total_seconds > 60:
            adjustments["sigma_range_preset"] = "All"
        elif total_seconds < 15:
            adjustments["sigma_range_preset"] = "High"

    # Adjust conditional blend based on genre tags
    if any(word in genre_lower for word in ["smooth", "ambient", "flowing", "ethereal"]):
        adjustments["conditional_blend_function_name"] = "slerp" # or "geometric_mean"

    # Adjust guidance_max_change for dynamic genres
    if any(word in genre_lower for word in ["fast", "energetic", "rhythmic", "chaotic"]):
        adjustments["guidance_max_change"] = 1.5 # Allow for more dynamic guidance changes

    # Adjust EMA factor for stability vs. responsiveness
    if any(word in genre_lower for word in ["stable", "coherent", "structured"]):
        adjustments["log_posterior_ema_factor"] = 0.4 # More smoothing
    elif any(word in genre_lower for word in ["experimental", "noisy", "chaotic"]):
        adjustments["log_posterior_ema_factor"] = 0.1 # Less smoothing

    return adjustments

# --- Function to generate YAML content for the ORIGINAL PingPong Sampler ---
def generate_original_pingpong_yaml_content(parsed_kwargs):
    """Generates the CLEAN YAML configuration string for the Original PingPongSampler_Custom."""
    # Define base defaults for the Original PingPong Sampler
    # These match the parameters expected by the original node (from Pasted_Text_1753330742230.txt)
    base_defaults = {
        "verbose": True,
        "step_random_mode": parsed_kwargs.get("step_random_mode", "block"),
        "step_size": parsed_kwargs.get("step_size", 5),
        "seed": parsed_kwargs.get("seed", 12345),
        "first_ancestral_step": parsed_kwargs.get("first_ancestral_step", 0),
        "last_ancestral_step": parsed_kwargs.get("last_ancestral_step", -1),
        "start_sigma_index": parsed_kwargs.get("start_sigma_index", 0),
        "end_sigma_index": parsed_kwargs.get("end_sigma_index", -1),
        "enable_clamp_output": parsed_kwargs.get("enable_clamp_output", False),
        "blend_mode": parsed_kwargs.get("blend_mode", "lerp"),
        "step_blend_mode": parsed_kwargs.get("step_blend_mode", "lerp"),
        "debug_mode": parsed_kwargs.get("debug_mode", False), # Original uses BOOLEAN
        # FBG specific parameters are intentionally omitted
    }

    # Construct the CLEAN YAML String for Original PingPong Sampler
    # NO '#' comments are included in the main body to ensure compatibility.
    yaml_content = f"""verbose: {str(base_defaults['verbose']).lower()}
step_random_mode: {base_defaults['step_random_mode']}
step_size: {base_defaults['step_size']}
seed: {base_defaults['seed']}
first_ancestral_step: {base_defaults['first_ancestral_step']}
last_ancestral_step: {base_defaults['last_ancestral_step']}
start_sigma_index: {base_defaults['start_sigma_index']}
end_sigma_index: {base_defaults['end_sigma_index']}
enable_clamp_output: {str(base_defaults['enable_clamp_output']).lower()}
blend_mode: {base_defaults['blend_mode']}
step_blend_mode: {base_defaults['step_blend_mode']}
debug_mode: {str(base_defaults['debug_mode']).lower()}
"""
    return yaml_content.strip()

# --- Function to generate YAML content for the ENHANCED FBG PingPong Sampler ---
def generate_enhanced_pingpong_yaml_content(parsed_kwargs):
    """Generates the CLEAN YAML configuration string for PingPongSampler_Custom_FBG."""

    # --- 1. Collect User Inputs from parsed_kwargs ---
    # Core PingPongSampler Inputs (matching original + new ones)
    step_random_mode = parsed_kwargs.get("step_random_mode", "block")
    step_size = parsed_kwargs.get("step_size", 5)
    seed = parsed_kwargs.get("seed", 12345)
    first_ancestral_step = parsed_kwargs.get("first_ancestral_step", 0)
    last_ancestral_step = parsed_kwargs.get("last_ancestral_step", -1)
    start_sigma_index = parsed_kwargs.get("start_sigma_index", 0)
    end_sigma_index = parsed_kwargs.get("end_sigma_index", -1)
    enable_clamp_output = parsed_kwargs.get("enable_clamp_output", False)
    blend_mode = parsed_kwargs.get("blend_mode", "lerp")
    step_blend_mode = parsed_kwargs.get("step_blend_mode", "lerp")
    debug_mode = parsed_kwargs.get("debug_mode", 0) # Default to off (int 0 for FBG)

    # New PingPongSampler Inputs (from latest enhancements)
    sigma_range_preset = parsed_kwargs.get("sigma_range_preset", "Custom")
    conditional_blend_mode = parsed_kwargs.get("conditional_blend_mode", False)
    conditional_blend_sigma_threshold = parsed_kwargs.get("conditional_blend_sigma_threshold", 1.0)
    conditional_blend_function_name = parsed_kwargs.get("conditional_blend_function_name", "geometric_mean")
    conditional_blend_on_change = parsed_kwargs.get("conditional_blend_on_change", False)
    conditional_blend_change_threshold = parsed_kwargs.get("conditional_blend_change_threshold", 0.1)
    clamp_noise_norm = parsed_kwargs.get("clamp_noise_norm", False)
    max_noise_norm = parsed_kwargs.get("max_noise_norm", 1.0)
    log_posterior_ema_factor = parsed_kwargs.get("log_posterior_ema_factor", 0.0)

    # --- Dynamic Adjustment Context (from LLM or overrides) ---
    genre_tags_context = parsed_kwargs.get("genre_tags_context", "") # Added context input
    total_seconds_context = parsed_kwargs.get("total_seconds_context", 30.0) # Added context input
    enable_dynamic_adjustment = parsed_kwargs.get("enable_dynamic_adjustment", True) # New toggle

    # FBG-Specific Inputs (Remapped names used in YAML fbg_config section)
    fbg_sampler_mode = parsed_kwargs.get("fbg_sampler_mode", "PINGPONG") # Remapped name
    cfg_scale = parsed_kwargs.get("cfg_scale", 150.0) # Adjusted default for high CFG use case
    cfg_start_sigma = parsed_kwargs.get("cfg_start_sigma", 1.0)
    cfg_end_sigma = parsed_kwargs.get("cfg_end_sigma", 0.0)
    fbg_start_sigma = parsed_kwargs.get("fbg_start_sigma", 1.0)
    fbg_end_sigma = parsed_kwargs.get("fbg_end_sigma", 0.0)
    fbg_guidance_multiplier = parsed_kwargs.get("fbg_guidance_multiplier", 1.0)
    ancestral_start_sigma = parsed_kwargs.get("ancestral_start_sigma", 1.0) # Adjusted default
    ancestral_end_sigma = parsed_kwargs.get("ancestral_end_sigma", 0.2) # Adjusted default
    max_guidance_scale = parsed_kwargs.get("max_guidance_scale", 250.0) # Adjusted default
    max_posterior_scale = parsed_kwargs.get("max_posterior_scale", 1.0)
    log_posterior_initial_value = parsed_kwargs.get("log_posterior_initial_value", 0.0) # Remapped name
    initial_guidance_scale = parsed_kwargs.get("initial_guidance_scale", 150.0) # Adjusted default
    guidance_max_change = parsed_kwargs.get("guidance_max_change", 1.0) # Adjusted default
    fbg_temp = parsed_kwargs.get("fbg_temp", 0.005) # Remapped name
    fbg_offset = parsed_kwargs.get("fbg_offset", -0.010) # Remapped name
    pi = parsed_kwargs.get("pi", 0.35) # Adjusted default for high CFG use case
    t_0 = parsed_kwargs.get("t_0", 0.90) # Adjusted default
    t_1 = parsed_kwargs.get("t_1", 0.60) # Adjusted default

    # FBG Internal Step Parameters (Top-Level)
    fbg_eta = parsed_kwargs.get("fbg_eta", 0.1) # Adjusted default
    fbg_s_noise = parsed_kwargs.get("fbg_s_noise", 1.2) # Adjusted default

    # --- 2. Apply Dynamic Parameter Adjustments ---
    context_adjustments = adjust_defaults_based_on_context(
        genre_tags_context, total_seconds_context, enable_dynamic_adjustment
    )

    # --- 3. Construct Base Defaults and Apply Adjustments ---
    base_defaults = {
        "step_random_mode": step_random_mode,
        "step_size": step_size,
        "seed": seed,
        "first_ancestral_step": first_ancestral_step,
        "last_ancestral_step": last_ancestral_step,
        "start_sigma_index": start_sigma_index,
        "end_sigma_index": end_sigma_index,
        "enable_clamp_output": enable_clamp_output,
        "blend_mode": blend_mode,
        "step_blend_mode": step_blend_mode,
        "debug_mode": debug_mode,
        "sigma_range_preset": sigma_range_preset,
        "conditional_blend_mode": conditional_blend_mode,
        "conditional_blend_sigma_threshold": conditional_blend_sigma_threshold,
        "conditional_blend_function_name": conditional_blend_function_name,
        "conditional_blend_on_change": conditional_blend_on_change,
        "conditional_blend_change_threshold": conditional_blend_change_threshold,
        "clamp_noise_norm": clamp_noise_norm,
        "max_noise_norm": max_noise_norm,
        "log_posterior_ema_factor": log_posterior_ema_factor,
        "fbg_sampler_mode": fbg_sampler_mode,
        "cfg_scale": cfg_scale,
        "cfg_start_sigma": cfg_start_sigma,
        "cfg_end_sigma": cfg_end_sigma,
        "fbg_start_sigma": fbg_start_sigma,
        "fbg_end_sigma": fbg_end_sigma,
        "fbg_guidance_multiplier": fbg_guidance_multiplier,
        "ancestral_start_sigma": ancestral_start_sigma,
        "ancestral_end_sigma": ancestral_end_sigma,
        "max_guidance_scale": max_guidance_scale,
        "max_posterior_scale": max_posterior_scale,
        "log_posterior_initial_value": log_posterior_initial_value,
        "initial_guidance_scale": initial_guidance_scale,
        "guidance_max_change": guidance_max_change,
        "fbg_temp": fbg_temp,
        "fbg_offset": fbg_offset,
        "pi": pi,
        "t_0": t_0,
        "t_1": t_1,
        "fbg_eta": fbg_eta,
        "fbg_s_noise": fbg_s_noise,
    }

    # Apply context-based adjustments, then user-provided kwargs (user input overrides all)
    final_inputs = {**base_defaults, **context_adjustments, **parsed_kwargs}

    # --- 4. Construct the CLEAN YAML String for FBG PingPong Sampler ---
    # IMPORTANT: Generate a CLEAN YAML string for parsing by other nodes.
    # NO '#' comments are included in the main body to ensure compatibility.
    # Boolean values are explicitly converted to lowercase strings for YAML compatibility.

    yaml_content = f"""verbose: true
step_random_mode: {final_inputs['step_random_mode']}
step_size: {final_inputs['step_size']}
seed: {final_inputs['seed']}
first_ancestral_step: {final_inputs['first_ancestral_step']}
last_ancestral_step: {final_inputs['last_ancestral_step']}
start_sigma_index: {final_inputs['start_sigma_index']}
end_sigma_index: {final_inputs['end_sigma_index']}
enable_clamp_output: {str(final_inputs['enable_clamp_output']).lower()}
blend_mode: {final_inputs['blend_mode']}
step_blend_mode: {final_inputs['step_blend_mode']}
debug_mode: {final_inputs['debug_mode']}
sigma_range_preset: {final_inputs['sigma_range_preset']}
conditional_blend_mode: {str(final_inputs['conditional_blend_mode']).lower()}
conditional_blend_sigma_threshold: {final_inputs['conditional_blend_sigma_threshold']}
conditional_blend_function_name: {final_inputs['conditional_blend_function_name']}
conditional_blend_on_change: {str(final_inputs['conditional_blend_on_change']).lower()}
conditional_blend_change_threshold: {final_inputs['conditional_blend_change_threshold']}
clamp_noise_norm: {str(final_inputs['clamp_noise_norm']).lower()}
max_noise_norm: {final_inputs['max_noise_norm']}
log_posterior_ema_factor: {final_inputs['log_posterior_ema_factor']}
pingpong_options:
  first_ancestral_step: {final_inputs['first_ancestral_step']}
  last_ancestral_step: {final_inputs['last_ancestral_step']}
fbg_config:
  fbg_sampler_mode: {final_inputs['fbg_sampler_mode']}
  cfg_scale: {final_inputs['cfg_scale']}
  cfg_start_sigma: {final_inputs['cfg_start_sigma']}
  cfg_end_sigma: {final_inputs['cfg_end_sigma']}
  fbg_start_sigma: {final_inputs['fbg_start_sigma']}
  fbg_end_sigma: {final_inputs['fbg_end_sigma']}
  fbg_guidance_multiplier: {final_inputs['fbg_guidance_multiplier']}
  ancestral_start_sigma: {final_inputs['ancestral_start_sigma']}
  ancestral_end_sigma: {final_inputs['ancestral_end_sigma']}
  max_guidance_scale: {final_inputs['max_guidance_scale']}
  max_posterior_scale: {final_inputs['max_posterior_scale']}
  initial_value: {final_inputs['log_posterior_initial_value']}
  initial_guidance_scale: {final_inputs['initial_guidance_scale']}
  guidance_max_change: {final_inputs['guidance_max_change']}
  fbg_temp: {final_inputs['fbg_temp']}
  fbg_offset: {final_inputs['fbg_offset']}
  pi: {final_inputs['pi']}
  t_0: {final_inputs['t_0']}
  t_1: {final_inputs['t_1']}
fbg_eta: {final_inputs['fbg_eta']}
fbg_s_noise: {final_inputs['fbg_s_noise']}
"""
    return yaml_content.strip() # Remove leading/trailing whitespace for clean output

# --- Custom YAML Dumper to prevent aliases and use block sequences ---
class CustomDumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True

    def represent_list(self, data):
        # This typically means block style for sequences unless explicitly forced to flow style elsewhere.
        return self.represent_sequence('tag:yaml.org,2002:seq', data)

# --- Node Class ---
class SceneGeniusAutocreator:
    """
    A multi-stage ComfyUI node leveraging local LLMs for dynamic creative content generation.
    This includes genres, lyrics/scripts, duration, and configurable diffusion parameters (APG & Sampler).
    It incorporates advanced reasoning subroutines for enhanced coherence and control over creative output.
    Version: 1.3.0
    """
    # Default APG YAML to use if LLM output is unparseable or invalid
    DEFAULT_APPLY_PROMPT = """verbose: true
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
  cfg: 2.6"""

    # Default Sampler YAML for Original PingPongSampler_Custom to use if LLM output is unparseable or invalid
    DEFAULT_SAMPLER_YAML_ORIGINAL = """verbose: true
step_random_mode: block
step_size: 5
first_ancestral_step: 25
last_ancestral_step: 40
start_sigma_index: 0
end_sigma_index: -1
enable_clamp_output: false
blend_mode: lerp
step_blend_mode: lerp
debug_mode: false
"""

    # Default Sampler YAML for PingPongSampler_Custom_FBG to use if LLM output is unparseable or invalid
    # This now matches the specific FBG YAML provided by the user.
    DEFAULT_SAMPLER_YAML_FBG = """verbose: true
step_random_mode: block
step_size: 5
first_ancestral_step: 25
last_ancestral_step: 40
enable_clamp_output: false
start_sigma_index: 0
end_sigma_index: -1
blend_mode: lerp
step_blend_mode: lerp
debug_mode: 2
sigma_range_preset: Custom
conditional_blend_mode: false
conditional_blend_sigma_threshold: 1.0
conditional_blend_function_name: geometric_mean
conditional_blend_on_change: false
conditional_blend_change_threshold: 0.1
clamp_noise_norm: false
max_noise_norm: 1.0
log_posterior_ema_factor: 0.3
pingpong_options:
  first_ancestral_step: 25
  last_ancestral_step: 40
fbg_config:
  fbg_sampler_mode: PINGPONG
  cfg_scale: 150.0
  cfg_start_sigma: 1.0
  cfg_end_sigma: 0.0
  fbg_start_sigma: 1.0
  fbg_end_sigma: 0.0
  fbg_guidance_multiplier: 1.0
  ancestral_start_sigma: 0.8
  ancestral_end_sigma: 0.2
  max_guidance_scale: 250.0
  max_posterior_scale: 1.0
  initial_value: 0.0
  initial_guidance_scale: 150.0
  guidance_max_change: 1.5
  fbg_temp: 0.005
  fbg_offset: -0.010
  pi: 0.35
  t_0: 0.90
  t_1: 0.60
fbg_eta: 0.1
fbg_s_noise: 0.7
"""

    # --- Prompts ---
    DEFAULT_GENRE_PROMPT = """You are an expert creative director specializing in tagging conceptual ideas for diffusion models. Your task is to analyze the provided user concept and generate a precise list of descriptive genre and style tags.

Instructions:
1.  Analyze the core theme, mood, and aesthetic of the "User Concept".
2.  Generate exactly {tag_count} distinct, single-word or hyphenated tags.
3.  Prioritize tags that are specific to diffusion model vocabularies (e.g., "synthwave", "cyberpunk", "watercolor", "isometric", "voxel-art").
4.  Blend strength ({tag_blend_strength}) determines variety: 0.0 = all tags highly related, 1.0 = tags highly diverse.
5.  Avoid any tags listed in "Excluded Tags".
6.  Output only the comma-separated list of tags. No other text, formatting, or markdown.

User Concept: {initial_concept_prompt}
Number of Tags: {tag_count}
Tag Blend Strength: {tag_blend_strength}
Excluded Tags: {excluded_tags}
Output Format: tag1, tag2, tag3, ..."""

    DEFAULT_LYRICS_DECISION_PROMPT = """You are an expert in conceptual world-building and narrative design for generative media. Your task is to decide if the provided concept would benefit from a structured lyrical or scriptural component to enhance its diffusion output.

Analyze the "User Concept" and "Generated Tags". If the concept is abstract, purely visual, or atmospheric (e.g., "Ethereal Aurora Borealis", "Abstract Techno Pulse"), answer "No". If it implies a narrative, story, character focus, or song-like structure (e.g., "A Lonesome Cowboy's Tale", "Cyberpunk Anthem for a Digital Heart"), answer "Yes".

User Concept: {initial_concept_prompt}
Generated Tags: {genre_tags}
Force Generation: {force_generation}
Output Format: Answer with a single word: 'Yes' or 'No'."""

    DEFAULT_LYRICS_PROMPT = """You are a talented lyricist and scriptwriter specializing in generative media prompts. Your task is to create a short, evocative piece of lyrics or a narrative script based on the provided concept and tags. This text will guide the semantic understanding of a diffusion model.

Instructions:
1.  Use the "User Concept" and "Generated Tags" as your primary inspiration.
2.  Write 3-5 short lines. For lyrics, use stanzas. For scripts, use concise, descriptive sentences or dialogue.
3.  Focus on imagery, emotion, and atmosphere. Avoid complex plot details.
4.  Do not reference the diffusion process itself.
5.  Output only the lyrics or script. No titles, prefixes, or markdown.

User Concept: {initial_concept_prompt}
Generated Tags: {genre_tags}
Output Format: The lyrics or script itself."""

    DEFAULT_DURATION_PROMPT = """You are a specialist in estimating the temporal duration of conceptual media. Your task is to analyze the provided creative inputs and estimate a suitable duration in seconds.

Consider the "User Concept", "Generated Tags", and "Lyrics/Script". Imagine the conceptual "weight" and complexity.
- Short (10-30s): Simple, static, or fast-paced ideas.
- Medium (30-90s): Moderate narrative, evolving scenes, or musical verses.
- Long (90s+): Complex narratives, detailed world-building, or extended musical pieces.

User Concept: {initial_concept_prompt}
Generated Tags: {genre_tags}
Lyrics/Script: {lyrics_or_script}
Min Duration: {min_duration} seconds
Max Duration: {max_duration} seconds
Output Format: Provide only the estimated duration as a single number in seconds (e.g., 95.5)."""

    DEFAULT_APPLY_PROMPT_GEN = """You are an expert in crafting configuration files for the Accelerated Proximal Guidance (APG) sampler in ComfyUI, specifically for audio/video diffusion models. Your goal is to create a YAML configuration that dynamically modulates the denoising process based on the creative inputs to achieve high-quality, coherent results.

Instructions:
1.  Analyze the "User Concept", "Generated Tags", "Lyrics/Script", and "Estimated Duration".
2.  Construct a `rules` list in the YAML. Each rule defines sampler behavior for a sigma range (from the previous rule's `start_sigma` down to the current rule's `start_sigma`).
3.  **Rule Structure & Parameters:**
    - `start_sigma` (float): The upper sigma boundary for this rule. Start with `-1` (represents infinity, for the initial step) and decrease towards `0`.
    - `apg_scale` (float): The strength of APG guidance. `0.0` disables APG. Typical range 0.0-10.0. Use higher values for complex or noisy concepts where structural coherence is key, lower for simpler or more abstract ideas.
    - `predict_image` (bool): If `true`, APG uses an internal prediction step. Usually `true` when `apg_scale` > 0.
    - `cfg` (float): Classifier-Free Guidance scale for this segment. Adjust based on the need for adherence to the prompt (higher) vs. creative freedom (lower).
    - `mode` (string): APG mode. Common options: `apg_basic`, `pre_alt2`, `pure_apg`. `pre_alt2` is often a good starting point for balancing guidance and quality.
    - `update_blend_mode` (string): How the APG update is blended. `lerp` is standard.
    - `dims` (list): Dimensions to apply APG. `[-2, -1]` is common for last two dims (HxW in latent space).
    - `momentum` (float): APG momentum. Controls update smoothness. 0.0-1.0, often around 0.5-0.8.
    - `norm_threshold` (float): APG norm threshold. Caps update magnitude. Often 1.0-5.0.
    - `eta` (float): Stochasticity. Usually 0.0 for deterministic results.
4.  Tailor the parameters in the rules list to the specific dynamics suggested by the inputs (e.g., faster changes for "fast" tags, higher `apg_scale` for complex narratives).
5.  Ensure the YAML structure is correct and parsable. **Do not use any markdown code block delimiters (```yaml) in your final output.**

User Concept: {initial_concept_prompt}
Generated Tags: {genre_tags}
Lyrics/Script: {lyrics_or_script}
Estimated Duration: {total_seconds} seconds
Output Format: A correctly formatted YAML configuration block for the APG Guider. No other text or explanations."""

    DEFAULT_SAMPLER_PROMPT_BASE = """You are an expert in configuring the PingPong Sampler for ComfyUI, tailored for Ace-Step audio/video diffusion models. Your task is to generate a precise YAML configuration based on the creative context.

Analyze the "User Concept", "Generated Tags", "Lyrics/Script", "Estimated Duration", and the provided "APG YAML Parameters".

{fbg_section}

Instructions:
1.  Interpret the creative intent and technical requirements from the inputs.
2.  Generate a YAML structure that configures the PingPong Sampler appropriately.
3.  **For the Original PingPong Sampler:**
    *   Focus on core parameters: `step_random_mode`, `step_size`, `first_ancestral_step`, `last_ancestral_step`, `blend_mode`, `step_blend_mode`, `debug_mode`.
    *   Do not include any parameters specific to the FBG version (like `sigma_range_preset`, `fbg_config`, `conditional_blend_*`, etc.).
4.  **For the FBG Integrated PingPong Sampler:**
    *   Utilize the full parameter set, including FBG dynamics (`fbg_config`), conditional blending, EMA, etc.
    *   Consider how the `APG YAML Parameters` might influence the overall denoising strategy and adjust PingPong settings for synergy (e.g., if APG is strong in mid-sigma, PingPong noise might be reduced there).
5.  Ensure the YAML structure is correct and parsable. **Do not use any markdown code block delimiters (```yaml) in your final output.**

User Concept: {initial_concept_prompt}
Generated Tags: {genre_tags}
Lyrics/Script: {lyrics_or_script}
Estimated Duration: {total_seconds} seconds
APG YAML Parameters:
{apg_yaml_params}
Output Format: A correctly formatted YAML configuration block for the selected PingPong Sampler. No other text or explanations."""

    # FBG specific prompt section to be conditionally added
    FBG_PROMPT_SECTION = """- **Feedback Guidance (FBG) Parameters (for FBG Integrated PingPong Sampler only):**
These parameters are for the `fbg_config` dictionary and top-level FBG inputs.

* **`fbg_config` (dictionary):** This nested dictionary holds the core FBG configuration.
    * **`sampler_mode` (string):** FBG's base sampler mode for internal calculations. `EULER` is standard and preferable, but `PINGPONG` can also be used and has shown good results in unique instances.
        **Default: `EULER`**
    * **`cfg_scale` (float):** **The base Classifier-Free Guidance (CFG) scale that FBG dynamically modifies.** This is the starting point for FBG's adjustments. **Optimal performance is often seen with higher settings, from `2.0` up to `16.0`, with values "in the middle" like `8.0` through `12.0` being effective.
        **Default: `7.0`**
    * **`cfg_start_sigma` / `cfg_end_sigma` (float):** The noise level (sigma) range where the base `cfg_scale` influence is calculated by FBG.
        **Defaults: `10.0` / `1.0`**
    * **`fbg_start_sigma` / `fbg_end_sigma` (float):** The noise level (sigma) range where the FBG dynamics (adaptive scaling) are active.
        **Defaults: `10.0` / `1.0`**
    * **`fbg_guidance_multiplier` (float):** A scaling factor applied to the guidance component calculated by the FBG algorithm.
        **Default: `1.5`**
    * **`ancestral_start_sigma` / `ancestral_end_sigma` (float):** The sigma range where FBG's *own internal* ancestral sampling step is active (requires `fbg_eta` > 0 and `sampler_mode=EULER`).
        **Defaults: `10.0` / `0.5`**
    * **`max_guidance_scale` (float):** The upper limit for the total guidance scale (base CFG + FBG adjustment).
        **Default: `15.0`**
    * **`max_posterior_scale` (float):** The upper limit for the internal log posterior estimate used by FBG.
        **Default: `3.0`**
    * **`initial_value` (float):** The starting value for the internal log posterior estimate (often remapped from `log_posterior_initial_value`).
        **Default: `0.0`**
    * **`initial_guidance_scale` (float):** The starting value for the internal guidance scale tracker, often matching `cfg_scale`.
        **Default: `7.0`**
    * **`guidance_max_change` (float):** Limits how much the calculated guidance scale can change from one step to the next (e.g., `0.5` = maximum 50% change).
        **Default: `0.5`**
    * **`pi` (float):** The FBG mixing parameter. Influences how the algorithm interprets the model's confidence. Lower values (like 0.2-0.8) are often effective for general Text-to-Image models. Setting `pi` to 1.0 implies the conditional model is perfect.
        **Default: `0.6`**
    * **`t_0` / `t_1` (float):** Internal FBG times (0-1) used to automatically calculate adjustment parameters (`temp`, `offset`).
        **Defaults: `0.8` / `0.3`**
    * **`temp` / `offset` (float):** Manual internal parameters for FBG calculations (often remapped from `fbg_temp` / `fbg_offset`). Used if `t_0`/`t_1` are 0 or overridden.
        **Defaults: `0.0`**

* **Top-Level FBG Parameters:**
    * **`fbg_eta` (float):** Parameter for FBG's *internal* ancestral sampling step (requires `sampler_mode=EULER` and `fbg_eta > 0`).
        **Default: `0.0`**
    * **`fbg_s_noise` (float):** Scale factor for noise in FBG's *internal* ancestral sampling step.
        **Default: `1.0`**

**IMPORTANT: When generating the actual YAML, DO NOT include the ```yaml or ``` delimiters in your final output. These are only for demonstration in this example.**"""

    CATEGORY = "sampling/custom_sampling/samplers"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "FLOAT")
    RETURN_NAMES = ("GENRE_TAGS", "LYRICS_OR_SCRIPT", "TOTAL_SECONDS", "NOISE_DECAY_STRENGTH", "APG_YAML_PARAMS", "SAMPLER_YAML_PARAMS", "SEED")
    FUNCTION = "generate_content"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ollama_api_base_url": ("STRING", {"default": "http://127.0.0.1:11434", "tooltip": "The base URL for the Ollama API endpoint."}),
                "ollama_model_name": ("STRING", {"default": "llama3:8b", "tooltip": "The name of the Ollama model to use for generation (e.g., 'llama3:8b', 'mistral:7b-instruct')."}),
                "initial_concept_prompt": ("STRING", {"default": "A mystical forest at twilight, with glowing mushrooms and a hidden waterfall.", "multiline": True, "tooltip": "Your initial creative concept or prompt to guide the generation process."}),
                "tag_count": ("INT", {"default": 15, "min": 5, "max": 50, "tooltip": "The number of descriptive tags to generate for the concept."}),
                "tag_blend_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Controls the diversity of the generated tags. 0.0 = highly related tags, 1.0 = highly diverse tags."}),
                "excluded_tags": ("STRING", {"default": "3d, render, photo, photograph, realistic, nsfw", "multiline": True, "tooltip": "A comma-separated list of tags to explicitly avoid during generation."}),
                "force_lyrics_generation": ("BOOLEAN", {"default": False, "tooltip": "If True, the node will always attempt to generate lyrics or a script, regardless of the LLM's initial decision."}),
                "min_total_seconds": ("FLOAT", {"default": 10.0, "min": 0.0, "tooltip": "The minimum estimated duration for the generated content (in seconds)."}),
                "max_total_seconds": ("FLOAT", {"default": 120.0, "min": 0.0, "tooltip": "The maximum estimated duration for the generated content (in seconds)."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "The seed for random number generation to ensure reproducibility."}),
                # --- New Toggle Input ---
                "enable_dynamic_adjustment": ("BOOLEAN", {"default": True, "tooltip": "Enable/disable dynamic parameter adjustment for the PingPong Sampler based on LLM context (genre, duration). Applies only to FBG version."}),
                # --- Retained Sampler Version Input ---
                "sampler_version": (["Original PingPong Sampler", "FBG Integrated PingPong Sampler"], {"default": "FBG Integrated PingPong Sampler", "tooltip": "Select which version of the PingPong Sampler to generate YAML for."}),
            },
            "optional": {
                "prompt_genre_generation": ("STRING", {"default": "", "multiline": True, "tooltip": "(Optional) Override the default prompt used for genre/tag generation."}),
                "prompt_lyrics_decision_and_generation": ("STRING", {"default": "", "multiline": True, "tooltip": "(Optional) Override the default prompt used for deciding and generating lyrics/script."}),
                "prompt_duration_generation": ("STRING", {"default": "", "multiline": True, "tooltip": "(Optional) Override the default prompt used for duration estimation."}),
                "prompt_noise_decay_generation": ("STRING", {"default": "", "multiline": True, "tooltip": "(Optional) Override the default prompt used for noise decay adjustment (if implemented)."}),
                "prompt_apg_yaml_generation": ("STRING", {"default": "", "multiline": True, "tooltip": "(Optional) Override the default prompt used for generating APG YAML parameters."}),
                "prompt_sampler_yaml_generation": ("STRING", {"default": "", "multiline": True, "tooltip": "(Optional) Override the default prompt used for generating PingPong Sampler YAML parameters."}),
            }
        }

    def _call_ollama_api(self, api_url, payload):
        """Handles the HTTP POST request to the Ollama API."""
        import requests
        try:
            response = requests.post(api_url, json=payload)
            response.raise_for_status()
            response_data = response.json()
            return response_data.get("response", "").strip()
        except requests.exceptions.RequestException as e:
            print(f"SceneGeniusAutocreator Error: Failed to call Ollama API at {api_url}: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"SceneGeniusAutocreator Error Details: {e.response.text}")
            return f"[API_CALL_FAILED] {str(e)}"
        except Exception as e:
            print(f"SceneGeniusAutocreator Error: An unexpected error occurred during Ollama API call: {e}")
            return f"[UNEXPECTED_ERROR] {str(e)}"

    def _strip_think_blocks(self, text):
        """Removes <think>...</think> blocks and common conversational prefixes from LLM output."""
        # Remove <think>...</think> blocks
        pattern = r"<think>.*?</think>"
        cleaned_text = re.sub(pattern, "", text, flags=re.DOTALL)

        # Remove common conversational prefixes like "Sure, ..." or "Okay, ..."
        cleaned_text = re.sub(r"^(Sure|Okay|Alright|I can|Here is|The output is|Output:)\s*", "", cleaned_text, flags=re.IGNORECASE)

        # Remove leading/trailing whitespace and quotes
        return cleaned_text.strip('",. ')

    def generate_content(self, ollama_api_base_url, ollama_model_name, initial_concept_prompt,
                         tag_count, tag_blend_strength, excluded_tags, force_lyrics_generation,
                         min_total_seconds, max_total_seconds, seed,
                         enable_dynamic_adjustment, # New toggle input
                         sampler_version, # Input for version selection
                         prompt_genre_generation="", prompt_lyrics_decision_and_generation="",
                         prompt_duration_generation="", prompt_noise_decay_generation="",
                         prompt_apg_yaml_generation="", prompt_sampler_yaml_generation="",
                         # Capture all potential PingPongSampler inputs
                         **kwargs # This captures any extra parameters passed
                         ):
        """Main function to generate content using the LLM."""
        try:
            if seed == -1:
                seed = random.randint(0, 0xffffffff)
            random.seed(seed)

            api_url_base = f"{ollama_api_base_url.rstrip('/')}/api/generate"

            # --- 1. Genre/Tag Generation ---
            genre_prompt = prompt_genre_generation if prompt_genre_generation else self.DEFAULT_GENRE_PROMPT
            genre_payload = {
                "model": ollama_model_name,
                "prompt": genre_prompt.format(
                    initial_concept_prompt=initial_concept_prompt,
                    tag_count=tag_count,
                    tag_blend_strength=tag_blend_strength,
                    excluded_tags=excluded_tags
                ),
                "stream": False,
                "keep_alive": 0
            }
            raw_genre_tags = self._call_ollama_api(api_url_base, genre_payload)
            print(f"SceneGeniusAutocreator: Raw LLM output for genre tags: '{raw_genre_tags}'")
            genre_tags = self._strip_think_blocks(raw_genre_tags).strip('",. ')

            # --- 2. Lyrics Decision & Generation ---
            lyrics_decision_prompt = prompt_lyrics_decision_and_generation if prompt_lyrics_decision_and_generation else self.DEFAULT_LYRICS_DECISION_PROMPT
            lyrics_prompt = prompt_lyrics_decision_and_generation if prompt_lyrics_decision_and_generation else self.DEFAULT_LYRICS_PROMPT

            if force_lyrics_generation:
                decision = "Yes"
            else:
                decision_payload = {
                    "model": ollama_model_name,
                    "prompt": lyrics_decision_prompt.format(
                        initial_concept_prompt=initial_concept_prompt,
                        genre_tags=genre_tags,
                        force_generation="Yes" if force_lyrics_generation else "No"
                    ),
                    "stream": False,
                    "keep_alive": 0
                }
                raw_decision = self._call_ollama_api(api_url_base, decision_payload)
                print(f"SceneGeniusAutocreator: Raw LLM output for lyrics decision: '{raw_decision}'")
                decision = "Yes" if "yes" in self._strip_think_blocks(raw_decision).lower() else "No"

            lyrics_or_script = ""
            if decision == "Yes":
                lyrics_payload = {
                    "model": ollama_model_name,
                    "prompt": lyrics_prompt.format(
                        initial_concept_prompt=initial_concept_prompt,
                        genre_tags=genre_tags
                    ),
                    "stream": False,
                    "keep_alive": 0
                }
                raw_lyrics = self._call_ollama_api(api_url_base, lyrics_payload)
                print(f"SceneGeniusAutocreator: Raw LLM output for lyrics: '{raw_lyrics}'")
                lyrics_or_script = self._strip_think_blocks(raw_lyrics).strip()

            # --- 3. Duration Estimation ---
            duration_prompt = prompt_duration_generation if prompt_duration_generation else self.DEFAULT_DURATION_PROMPT
            duration_payload = {
                "model": ollama_model_name,
                "prompt": duration_prompt.format(
                    initial_concept_prompt=initial_concept_prompt,
                    genre_tags=genre_tags,
                    lyrics_or_script=lyrics_or_script,
                    min_duration=min_total_seconds,
                    max_duration=max_total_seconds
                ),
                "stream": False,
                "keep_alive": 0
            }
            raw_duration = self._call_ollama_api(api_url_base, duration_payload)
            print(f"SceneGeniusAutocreator: Raw LLM output for duration: '{raw_duration}'")
            try:
                total_seconds_str = self._strip_think_blocks(raw_duration).strip('",. seconds')
                total_seconds = float(total_seconds_str)
                total_seconds = max(min_total_seconds, min(total_seconds, max_total_seconds))
            except ValueError:
                print(f"SceneGeniusAutocreator Warning: Could not parse duration '{raw_duration}'. Using default 60s.")
                total_seconds = 60.0

            # --- 4. APG YAML Generation ---
            apg_prompt = prompt_apg_yaml_generation if prompt_apg_yaml_generation else self.DEFAULT_APPLY_PROMPT_GEN
            apg_yaml_payload = {
                "model": ollama_model_name,
                "prompt": apg_prompt.format(
                    initial_concept_prompt=initial_concept_prompt,
                    genre_tags=genre_tags,
                    lyrics_or_script=lyrics_or_script,
                    total_seconds=total_seconds
                ),
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

                apg_data = yaml.safe_load(cleaned_apg_yaml)
                if isinstance(apg_data, dict) and 'rules' in apg_data:
                    apg_yaml_params = cleaned_apg_yaml
                else:
                    raise ValueError("Parsed APG YAML does not contain expected 'rules' structure.")
            except (yaml.YAMLError, ValueError) as e:
                print(f"SceneGeniusAutocreator Warning: Could not parse LLM APG YAML output: {e}. Using default.")
                apg_yaml_params = self.DEFAULT_APPLY_PROMPT # Use the default string

            # --- 5. PingPong Sampler YAML Generation ---
            # Prepare context for dynamic adjustment.
            kwargs_for_pingpong_yaml = kwargs.copy()
            kwargs_for_pingpong_yaml["genre_tags_context"] = genre_tags
            kwargs_for_pingpong_yaml["total_seconds_context"] = total_seconds
            kwargs_for_pingpong_yaml["enable_dynamic_adjustment"] = enable_dynamic_adjustment

            # --- Branching Logic Based on sampler_version ---
            if sampler_version == "Original PingPong Sampler":
                print(f"SceneGeniusAutocreator: Generating YAML for '{sampler_version}'.")
                # Use the function designed for the Original PingPong Sampler
                sampler_yaml_params = generate_original_pingpong_yaml_content(kwargs_for_pingpong_yaml)
                # Use the Original PingPong default template for fallback within this branch
                default_sampler_yaml_to_use = self.DEFAULT_SAMPLER_YAML_ORIGINAL
            else: # Default to FBG Integrated if anything else is selected
                print(f"SceneGeniusAutocreator: Generating YAML for '{sampler_version}'.")
                # Use the function designed for the Enhanced FBG PingPong Sampler
                sampler_yaml_params = generate_enhanced_pingpong_yaml_content(kwargs_for_pingpong_yaml)
                # Use the FBG PingPong default template for fallback within this branch
                default_sampler_yaml_to_use = self.DEFAULT_SAMPLER_YAML_FBG

            # --- LLM Override Logic for PingPong Sampler YAML (Simplified) ---
            if prompt_sampler_yaml_generation.strip():
                print(f"SceneGeniusAutocreator: PingPong Sampler YAML override detected. Attempting to parse provided value.")
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

                    print(f"SceneGeniusAutocreator: Cleaned PingPong Sampler YAML string before parsing attempt:\n-{cleaned_sampler_yaml}-")
                    sampler_data = yaml.safe_load(cleaned_sampler_yaml)
                    if sampler_data is None or not isinstance(sampler_data, dict):
                        print("SceneGeniusAutocreator Warning: Provided PingPong Sampler YAML is empty or not a dictionary. Using internally generated YAML.")
                        # Do not override sampler_yaml_params, keep the one generated above
                    else:
                        # Successfully parsed override, use it
                        sampler_yaml_params = yaml.dump(sampler_data, Dumper=CustomDumper, indent=2, sort_keys=False)
                        print("SceneGeniusAutocreator: Successfully used provided PingPong Sampler YAML override.")
                except yaml.YAMLError as e:
                    print(f"SceneGeniusAutocreator Error: Could not parse PingPong Sampler YAML from provided override (YAMLError): {e}. Using internally generated YAML.")
                    # Do not override sampler_yaml_params, keep the one generated above
                except Exception as e:
                    print(f"SceneGeniusAutocreator Error: An unexpected error occurred during PingPong Sampler YAML override processing: {e}. Using internally generated YAML.")
                    # Do not override sampler_yaml_params, keep the one generated above
            else:
                # No override, use the YAML generated by the internal logic above
                pass

            print(f"SceneGeniusAutocreator: Final SAMPLER_YAML_PARAMS output:\n{sampler_yaml_params}")

            # --- 6. Placeholder Outputs ---
            noise_decay_strength = 1.0 # Not currently generated
            mood_keywords = f"mood_{sanitize_key(genre_tags.split(',')[0] if genre_tags else 'default')}"
            sound_design_keywords = f"sound_{sanitize_key(initial_concept_prompt.split()[0] if initial_concept_prompt else 'concept')}"

            print(f"SceneGeniusAutocreator: Generation complete.")
            return (
                genre_tags,
                lyrics_or_script,
                total_seconds,
                noise_decay_strength, # Placeholder
                apg_yaml_params,
                sampler_yaml_params, # Return the correctly chosen YAML string
                float(seed),
            )

        except Exception as e:
            print(f"SceneGeniusAutocreator Error: An unexpected error occurred during generation: {e}")
            import traceback
            traceback.print_exc()
            # Return default/fallback values
            default_apg = self.DEFAULT_APPLY_PROMPT
            # Fallback Sampler YAML based on selected version
            if sampler_version == "Original PingPong Sampler":
                default_sampler = self.DEFAULT_SAMPLER_YAML_ORIGINAL
            else:
                default_sampler = self.DEFAULT_SAMPLER_YAML_FBG

            return (
                "error,genre,tag",
                "Failed to generate lyrics.",
                60.0,
                1.0, # Placeholder noise decay
                default_apg,
                default_sampler,
                float(seed) if 'seed' in locals() else 0.0
            )

    def execute(self, **kwargs):
        """Wrapper for the generate_content function to match ComfyUI's expected interface."""
        return self.generate_content(**kwargs)

# --- Register the node with ComfyUI ---
NODE_CLASS_MAPPINGS = {
    "SceneGeniusAutocreator": SceneGeniusAutocreator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SceneGeniusAutocreator": f"Scene Genius Autocreator v{SCENE_GENIUS_VERSION}",
}
