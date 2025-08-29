# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# ████ SCENEGENIUS AUTOCREATOR v0.2.6 – Complete YAML Generation Fix ████▓▒░
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

# ░▒▓ CHANGELOG:
#   - v0.2.6 (Complete YAML Generation Fix):
#       • FIXED: A critical design flaw where the LLM would only output partial YAMLs.
#       • REINFORCED: The sampler prompt now strictly commands the LLM to output a COMPLETE set of parameters for the chosen sampler.
#       • UPDATED: The internal default YAML configurations are now complete, known-good audio presets, ensuring a robust fallback.
#   - v0.2.5 (Final Sampler Key Fix):
#       • Fixed incorrect `eta`/`s_noise` key names in FBG sampler prompt.
#   - v0.2.4 (Restored Strict Prompts):
#       • Re-implemented highly detailed LLM prompts to prevent parameter hallucination.

import logging
import yaml
import requests
import re
import time
from typing import Dict, Any

# Set up logging for ComfyUI
logger = logging.getLogger(__name__)

try:
    import folder_paths
except ImportError:
    logger.warning("Not running in ComfyUI environment. Mocking folder_paths.")
    class MockFolderPaths:
        def get_full_path(self, filename, subdir):
            return filename
    folder_paths = MockFolderPaths()

# Custom Dumper to preserve list formatting for specific cases
class CustomDumper(yaml.Dumper):
    def represent_sequence(self, tag, data, flow_style=None):
        if len(data) <= 2 and all(isinstance(x, (int, float)) for x in data):
            return super().represent_sequence(tag, data, flow_style=True)
        return super().represent_sequence(tag, data, flow_style)

class SceneGeniusAutocreator:
    """
    A multi-stage ComfyUI node leveraging local LLMs for dynamic creative content generation.
    """

    def __init__(self):
        """Initialize default configurations once at class instantiation."""
        self.APG_MIN_CFG = 2.0
        self.APG_MAX_CFG = 5.0
        self.NOISE_DECAY_MIN = 0.0
        self.NOISE_DECAY_MAX = 10.0

        self.default_apg_yaml = """
verbose: true
rules:
  - start_sigma: -1.0
    apg_scale: 0.0
    cfg: 5.0
  - start_sigma: 8.0
    apg_scale: 4.5
    predict_image: true
    cfg: 4.0
    mode: pure_apg
    dims: [-1]
    momentum: 0.6
    norm_threshold: 2.5
  - start_sigma: 2.0
    apg_scale: 2.0
    predict_image: true
    cfg: 3.0
    mode: pure_apg
    dims: [-1]
    momentum: 0.6
  - start_sigma: 0.8
    apg_scale: 0.0
    cfg: 2.0
"""
        # UPDATED: Complete, known-good audio preset
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
        # UPDATED: Complete, known-good audio preset
        self.default_sampler_yaml_fbg = """
step_random_mode: "block"
step_size: 5
first_ancestral_step: 0
last_ancestral_step: -1
ancestral_noise_type: "gaussian"
start_sigma_index: 0
end_sigma_index: -1
enable_clamp_output: false
blend_mode: "slerp"
step_blend_mode: "lerp"
debug_mode: 0
sigma_range_preset: "All"
log_posterior_ema_factor: 0.3
eta: 0.5
s_noise: 0.8
fbg_config:
  t_0: 0.7
  t_1: 0.4
  sampler_mode: "PINGPONG"
  cfg_scale: 8.0
  pi: 0.35
  max_guidance_scale: 30.0
  max_posterior_scale: 3.0
"""

        # --- LLM PROMPTS (Restored to be highly detailed) ---
        self.DEFAULT_APG_PROMPT = """
You are an expert diffusion model parameter specialist. Your task is to generate APG YAML for an AUDIO generation task.

**Creative Context:**
- Initial Concept: "{initial_concept_prompt}"
- Generated Genres: "{genre_tags}"

**CRITICAL RULES FOR AUDIO APG YAML:**
1.  **Output Format:** MUST be a valid YAML string.
2.  **`dims` for Audio:** The `dims` parameter MUST be `[-1]`. This is critical for normalizing over the frequency dimension correctly. Do NOT use `[-2, -1]`.
3.  **Allowed Rule Parameters:** Each rule in the `rules` list MUST ONLY contain: `start_sigma`, `apg_scale`, `cfg`, `predict_image`, `mode`, `update_blend_mode`, `dims`, `momentum`, `norm_threshold`, `eta`.
4.  **STRICT EXCLUSION:** DO NOT include any other parameters like `sampler`, `seed`, `strength`, `phase`, `duration`, `end_sigma`, etc.
5.  **No Extra Text:** Output ONLY the YAML.

**Example of CORRECT Audio APG YAML Structure:**
```yaml
verbose: true
rules:
  - start_sigma: -1.0
    apg_scale: 0.0
    cfg: 5.0
  - start_sigma: 8.0
    apg_scale: 4.5
    predict_image: true
    cfg: 4.0
    mode: pure_apg
    dims: [-1]
    momentum: 0.6
    norm_threshold: 2.5
  - start_sigma: 0.8
    apg_scale: 0.0
    cfg: 2.0
```

Output:
"""

        self.PROMPT_SECTION_ORIGINAL_SAMPLER = """
**Guidance for Original PingPong Sampler (Lite+):**
This sampler uses presets for noise control, ideal for audio textures.
- `noise_behavior`: Must be one of "Default (Raw)", "Dynamic", "Smooth", "Textured Grain", "Soft (DDIM-Like)", or "Custom".
- `ancestral_strength`: (Only if `noise_behavior` is "Custom"). Float from 0.0 to 2.0. For audio, `0.8` to `1.0` adds rich texture.
- `noise_coherence`: (Only if `noise_behavior` is "Custom"). Float from 0.0 to 1.0. For audio, `0.7` or higher creates smooth, evolving noise.
- `last_ancestral_step`: For audio, setting this to a negative number like `-3` (3 steps from the end) often produces cleaner results.
"""

        self.PROMPT_SECTION_FBG_SAMPLER = """
**Guidance for FBG Integrated PingPong Sampler:**
This sampler offers deep control with Feedback Guidance, excellent for complex audio.
- `ancestral_noise_type`: Must be one of: "gaussian", "uniform", "brownian".
- `eta` and `s_noise` are top-level parameters that control FBG's internal noise.
- `fbg_config`: A nested dictionary for FBG parameters.
  - `t_0` and `t_1` are crucial for automatic FBG sensitivity. Good audio values are often `t_0: 0.7` and `t_1: 0.4`.
  - `cfg_scale`: The base guidance. For audio, this can range from `4.0` to `12.0`.
  - `pi`: For audio, a lower value like `0.35` can work well.
"""

        self.DEFAULT_SAMPLER_PROMPT_BASE = """
You are an expert diffusion model sampling parameter specialist. Your task is to generate YAML-formatted parameters for the selected Ping-Pong Sampler, specifically for an AUDIO generation task.

**Creative Context:**
- Initial Concept: "{initial_concept_prompt}"
- Generated Genres: "{genre_tags}"

**CRITICAL TASK: Generate a VALID and COMPLETE YAML string for the "{sampler_version_name}" sampler.**

{sampler_specific_guidance}

**CRITICAL INSTRUCTION ON YAML STRUCTURE (Follow this to avoid errors):**
- The parameter `fbg_config` is a **nested dictionary**.
- ALL other parameters, including `log_posterior_ema_factor`, `eta`, `s_noise`, `ancestral_noise_type`, etc., MUST be at the **top level**. They DO NOT go inside `fbg_config`.
- **Generate a complete YAML:** Your output MUST include ALL relevant parameters for the selected sampler version, not just the ones you want to change. Use the default YAML for that version as a structural and completeness guide.

**STRICT EXCLUSION - DO NOT INCLUDE THESE PARAMETERS:**
- `sampler`, `seed`, `steps`, `log_posterior_weighting`, `enable_logposterior_guidance`, `fbg_eta`, `fbg_s_noise`. The correct names are `eta` and `s_noise`.

**Output Rules:**
1.  The output MUST be a valid and complete YAML string.
2.  Do NOT include any conversational filler, explanations, or markdown.

Output:
"""

        self.PROMPT_LYRICS_GENERATION_BASE = """
You are a creative writer. Your task is to generate lyrics or a script for a creative project, or determine if it should be instrumental.

**Project Details:**
- **Initial Concept:** "{initial_concept_prompt}"
- **Generated Genres:** "{genre_tags}"

**Decision Logic:**
- If `force_lyrics_generation` is `True`, you MUST generate lyrics.
- Otherwise, you may decide if an instrumental piece is more appropriate.

**CRITICAL OUTPUT FORMATTING RULES:**
1.  If the song should be instrumental, your ENTIRE output MUST be the single word: `[instrumental]`
2.  If generating lyrics, you MUST use structural tags. The ONLY allowed tags are `[Verse 1]`, `[Verse 2]`, `[Chorus]`, `[Bridge]`, `[Outro]`.
3.  Each structural tag MUST be on its own line.
4.  Do NOT include any conversational text. Output ONLY the lyrics or the instrumental tag.

{force_lyrics_instruction}
Now, based on the project details, generate your output.
Output:
"""
        self.PROMPT_GENRE_GENERATION_BASE = """
You are a highly creative AI assistant. Your task is to generate a single, comma-separated string of exactly {tag_count} descriptive genre tags based on the concept: "{initial_concept_prompt}".
{excluded_tags_instruction}
The output MUST ONLY be the comma-separated tags.
Output:
"""
        self.PROMPT_DURATION_GENERATION_BASE = """
You are an AI assistant. Based on the concept "{initial_concept_prompt}", genres "{genre_tags}", and lyrics "{lyrics_or_script}", provide a single floating-point number for the total duration in seconds.
The duration MUST be between {min_total_seconds:.1f} and {max_total_seconds:.1f} seconds.
Output ONLY the number.
Output:
"""
        self.PROMPT_NOISE_DECAY_GENERATION_BASE = """
You are an AI assistant. Based on the creative context, provide a single floating-point number between {min_noise_decay:.1f} and {max_noise_decay:.1f} for noise decay strength.
Concept: "{initial_concept_prompt}"
Genres: "{genre_tags}"
Lyrics: "{lyrics_or_script}"
Duration: {total_seconds} seconds
Output ONLY the number.
Output:
"""

    CATEGORY = "MD_Nodes/SceneGenius"

    @classmethod
    def _get_ollama_models_list(cls):
        default_ollama_url = "http://localhost:11434"
        try:
            response = requests.get(f"{default_ollama_url}/api/tags", timeout=5)
            response.raise_for_status()
            models_data = response.json()
            models = [model["name"] for model in models_data.get("models", [])]
            return models if models else ["llama3:8b-instruct-q8_0"]
        except Exception as e:
            logger.error(f"SceneGenius: Could not fetch Ollama models: {e}")
            return ["llama3:8b-instruct-q8_0 (Ollama not found)"]

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "ollama_api_base_url": ("STRING", {"default": "http://localhost:11434"}),
                "ollama_model_name": (cls._get_ollama_models_list(), {"default": "llama3:8b-instruct-q8_0"}),
                "initial_concept_prompt": ("STRING", {"multiline": True, "default": "A dystopian cyberpunk future with a retro-futuristic soundscape."}),
                "tag_count": ("INT", {"default": 4, "min": 1, "max": 10}),
                "excluded_tags": ("STRING", {"default": ""}),
                "force_lyrics_generation": ("BOOLEAN", {"default": False}),
                "min_total_seconds": ("FLOAT", {"default": 60.0, "min": 0.0, "max": 3600.0}),
                "max_total_seconds": ("FLOAT", {"default": 300.0, "min": 0.0, "max": 3600.0}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "sampler_version": (["Original PingPong Sampler", "FBG Integrated PingPong Sampler"], {"default": "FBG Integrated PingPong Sampler"}),
            },
            "optional": {
                "prompt_genre_generation": ("STRING", {"multiline": True, "default": ""}),
                "prompt_lyrics_decision_and_generation": ("STRING", {"multiline": True, "default": ""}),
                "prompt_duration_generation": ("STRING", {"multiline": True, "default": ""}),
                "prompt_noise_decay_generation": ("STRING", {"multiline": True, "default": ""}),
                "prompt_apg_yaml_generation": ("STRING", {"multiline": True, "default": ""}),
                "prompt_sampler_yaml_generation": ("STRING", {"multiline": True, "default": ""}),
                "test_mode": ("BOOLEAN", {"default": False})
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "FLOAT", "FLOAT", "STRING", "STRING", "INT",)
    RETURN_NAMES = ("GENRE_TAGS", "LYRICS_OR_SCRIPT", "TOTAL_SECONDS", "NOISE_DECAY_STRENGTH", "APG_YAML_PARAMS", "SAMPLER_YAML_PARAMS", "SEED",)
    FUNCTION = "execute"

    def _clean_llm_yaml_output(self, raw_output: str) -> str:
        match = re.search(r'```(?:yaml|output)?\s*(.*?)\s*```', raw_output, re.DOTALL)
        if match:
            return match.group(1).strip()
        cleaned = raw_output.strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()
        return cleaned

    def _call_ollama_api(self, api_url: str, payload: Dict[str, Any], timeout: int = 300, max_retries: int = 3) -> str:
        for attempt in range(max_retries):
            try:
                logger.info(f"Attempt {attempt + 1}/{max_retries}: Connecting to Ollama at {api_url.split('/api/')[0]} with model {payload.get('model')}")
                response = requests.post(f"{api_url}/api/generate", json=payload, timeout=timeout)
                response.raise_for_status()
                return response.json().get("response", "").strip()
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                logger.warning(f"Ollama API call failed (Attempt {attempt + 1}/{max_retries}): {e}. Retrying in {2 ** attempt} seconds...")
                time.sleep(2 ** attempt)
        raise RuntimeError("Ollama API call failed after multiple retries.")

    def execute(self, ollama_api_base_url, ollama_model_name, initial_concept_prompt, tag_count, excluded_tags, force_lyrics_generation, min_total_seconds, max_total_seconds, seed, sampler_version, **kwargs):
        logger.info("Starting SceneGeniusAutocreator execution...")
        
        prompt_overrides = {k: v for k, v in kwargs.items() if k.startswith('prompt_')}
        test_mode = kwargs.get('test_mode', False)

        if test_mode:
            logger.info("Test mode enabled. Returning dummy output.")
            return ("electro, synthwave", "[Verse 1]\n...", 180.0, 3.5, self.default_apg_yaml, self.default_sampler_yaml_fbg, 42)

        # --- Stage 1: Genre Tags ---
        logger.info("Stage 1: Generating genre tags...")
        genre_tags = prompt_overrides.get("prompt_genre_generation", "").strip()
        if not genre_tags:
            try:
                excluded_tags_instruction = f"Avoid generating any of these tags: {excluded_tags}." if excluded_tags else ""
                payload = {
                    "model": ollama_model_name,
                    "prompt": self.PROMPT_GENRE_GENERATION_BASE.format(
                        tag_count=tag_count,
                        initial_concept_prompt=initial_concept_prompt,
                        excluded_tags_instruction=excluded_tags_instruction
                    ), "stream": False, "keep_alive": 0
                }
                genre_tags = self._call_ollama_api(ollama_api_base_url, payload)
            except Exception as e:
                logger.error(f"Genre generation failed: {e}. Falling back to 'ambient'.")
                genre_tags = "ambient"

        # --- Stage 2: Lyrics/Script ---
        logger.info("Stage 2: Generating lyrics/script...")
        lyrics_or_script = prompt_overrides.get("prompt_lyrics_decision_and_generation", "").strip()
        if not lyrics_or_script:
            try:
                force_lyrics_instruction = "The user has set `force_lyrics_generation` to True." if force_lyrics_generation else ""
                payload = {
                    "model": ollama_model_name,
                    "prompt": self.PROMPT_LYRICS_GENERATION_BASE.format(
                        initial_concept_prompt=initial_concept_prompt,
                        genre_tags=genre_tags,
                        force_lyrics_instruction=force_lyrics_instruction
                    ), "stream": False, "keep_alive": 0
                }
                lyrics_or_script = self._call_ollama_api(ollama_api_base_url, payload)
            except Exception as e:
                logger.error(f"Lyrics generation failed: {e}. Setting to instrumental.")
                lyrics_or_script = "[instrumental]"

        # --- Stage 3: Duration ---
        logger.info("Stage 3: Generating duration...")
        total_seconds = 0.0
        duration_override = prompt_overrides.get("prompt_duration_generation", "").strip()
        if duration_override:
            total_seconds = float(duration_override)
        else:
            try:
                payload = {
                    "model": ollama_model_name,
                    "prompt": self.PROMPT_DURATION_GENERATION_BASE.format(
                        min_total_seconds=min_total_seconds, max_total_seconds=max_total_seconds,
                        initial_concept_prompt=initial_concept_prompt, genre_tags=genre_tags, lyrics_or_script=lyrics_or_script
                    ), "stream": False, "keep_alive": 0
                }
                total_seconds = float(self._call_ollama_api(ollama_api_base_url, payload))
            except Exception as e:
                logger.warning(f"Duration generation failed: {e}. Using min duration.")
                total_seconds = min_total_seconds
        total_seconds = max(min_total_seconds, min(total_seconds, max_total_seconds))
        logger.info(f"Final duration: {total_seconds:.2f} seconds.")

        # --- Stage 4: Noise Decay ---
        logger.info("Stage 4: Generating noise decay strength...")
        noise_decay_strength = 0.0
        noise_decay_override = prompt_overrides.get("prompt_noise_decay_generation", "").strip()
        if noise_decay_override:
            noise_decay_strength = float(noise_decay_override)
        else:
            try:
                payload = {
                    "model": ollama_model_name,
                    "prompt": self.PROMPT_NOISE_DECAY_GENERATION_BASE.format(
                        min_noise_decay=self.NOISE_DECAY_MIN, max_noise_decay=self.NOISE_DECAY_MAX,
                        initial_concept_prompt=initial_concept_prompt, genre_tags=genre_tags,
                        lyrics_or_script=lyrics_or_script, total_seconds=total_seconds
                    ), "stream": False, "keep_alive": 0
                }
                noise_decay_strength = float(self._call_ollama_api(ollama_api_base_url, payload))
            except Exception as e:
                logger.warning(f"Noise decay generation failed: {e}. Using default: 5.0.")
                noise_decay_strength = 5.0
        noise_decay_strength = max(self.NOISE_DECAY_MIN, min(noise_decay_strength, self.NOISE_DECAY_MAX))
        logger.info(f"Final noise decay strength: {noise_decay_strength:.2f}.")

        # --- Stage 5: APG YAML ---
        logger.info("Stage 5: Generating APG YAML Parameters.")
        apg_yaml_params = self.default_apg_yaml
        apg_override = prompt_overrides.get("prompt_apg_yaml_generation", "").strip()
        if apg_override:
            apg_yaml_params = apg_override
        else:
            try:
                payload = {
                    "model": ollama_model_name,
                    "prompt": self.DEFAULT_APG_PROMPT.format(
                        initial_concept_prompt=initial_concept_prompt, genre_tags=genre_tags,
                        lyrics_or_script=lyrics_or_script, total_seconds=total_seconds,
                        noise_decay_strength=noise_decay_strength, min_cfg=self.APG_MIN_CFG, max_cfg=self.APG_MAX_CFG
                    ), "stream": False, "keep_alive": 0
                }
                raw = self._call_ollama_api(ollama_api_base_url, payload)
                cleaned = self._clean_llm_yaml_output(raw)
                data = yaml.safe_load(cleaned)
                apg_yaml_params = yaml.dump(data, Dumper=CustomDumper, indent=2, sort_keys=False)
            except Exception as e:
                logger.warning(f"APG YAML generation failed: {e}. Using default.")
        logger.info(f"Final APG YAML:\n{apg_yaml_params}")

        # --- Stage 6: Sampler YAML ---
        logger.info(f"Stage 6: Generating Sampler YAML for '{sampler_version}'.")
        sampler_yaml_params = ""
        sampler_override = prompt_overrides.get("prompt_sampler_yaml_generation", "").strip()
        if sampler_override:
            sampler_yaml_params = sampler_override
        else:
            if sampler_version == "FBG Integrated PingPong Sampler":
                default_sampler_yaml = self.default_sampler_yaml_fbg
                sampler_guidance = self.PROMPT_SECTION_FBG_SAMPLER
            else:
                default_sampler_yaml = self.default_sampler_yaml_original
                sampler_guidance = self.PROMPT_SECTION_ORIGINAL_SAMPLER
            
            sampler_yaml_params = default_sampler_yaml
            try:
                prompt = self.DEFAULT_SAMPLER_PROMPT_BASE.format(
                    initial_concept_prompt=initial_concept_prompt, genre_tags=genre_tags,
                    lyrics_or_script=lyrics_or_script, total_seconds=total_seconds, apg_yaml_params=apg_yaml_params,
                    sampler_version_name=sampler_version, sampler_specific_guidance=sampler_guidance
                )
                payload = {"model": ollama_model_name, "prompt": prompt, "stream": False, "keep_alive": 0}
                raw = self._call_ollama_api(ollama_api_base_url, payload)
                cleaned = self._clean_llm_yaml_output(raw)
                data = yaml.safe_load(cleaned)
                sampler_yaml_params = yaml.dump(data, Dumper=CustomDumper, indent=2, sort_keys=False)
            except Exception as e:
                logger.warning(f"Sampler YAML generation failed: {e}. Using default for {sampler_version}.")
        logger.info(f"Final Sampler YAML:\n{sampler_yaml_params}")

        return (genre_tags, lyrics_or_script, total_seconds, noise_decay_strength, apg_yaml_params, sampler_yaml_params, seed)

# Node Mapping
NODE_CLASS_MAPPINGS = {
    "SceneGeniusAutocreator": SceneGeniusAutocreator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SceneGeniusAutocreator": "Scene Genius Autocreator"
}
