# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# ████ SCENEGENIUS AUTOCREATOR v0.3.8 – Genre Prompt Enhancement ████▓▒░
# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀

# ░▒▓ ORIGIN & DEV:
#    • Cast into the void by: MDMAchine (OG SysOp)
#    • Original mind behind SceneGenius Autocreator
#    • Initial ComfyUI adaptation by: Gemini (Google)
#    • Enhanced & refined by: MDMAchine & Gemini
#    • Critical optimizations & bugfixes: Gemini
#    • Final polish: MDMAchine
#    • License: Apache 2.0 — No cursed floppy disks allowed

# ░▒▓ DESCRIPTION:
#    A multi-stage AI creative weapon node for ComfyUI.
#    Designed to automate Ace-Step diffusion content generation,
#    channeling the chaotic spirit of the demoscene and BBS era.
#    Produces authentic genres, adaptive lyrics, precise durations,
#    and finely tuned APG + Sampler configs with ease.

# ░▒▓ CHANGELOG:
#    - v0.3.8 (Genre Prompt Enhancement):
#       • FIXED: A regression in genre quality caused by an over-simplified prompt.
#       • ENHANCED: Restored a highly detailed prompt for genre generation, including a list of example genres and high-quality tag combinations to ensure the first tag is a valid genre and descriptors are contextually relevant.
#    - v0.3.7 (Lyric Quality Enhancement):
#       • FIXED: A regression in lyric quality.
#       • ENHANCED: Restored a highly detailed, structured lyric prompt with a high-quality example.
#    - v0.3.6 (Final Polish & Refactor):
#       • ADDED: Full, intelligent configuration support for the "Original PingPong Sampler".
#       • ADDED: A `randomize_seed` option.
#       • REFACTORED: The main `execute` method was broken down into clean helper functions.
#    - v0.3.5 (Full Expert Logic Review):
#       • ENHANCED: Noise Decay Scheduler logic is now a two-stage process.
#    - v0.3.4 (Advanced APG Audio Update):
#       • ENHANCED: APG logic now uses a context-aware style selector (`Audio` vs `Image`).

import logging
import yaml
import requests
import re
import time
import random
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
        self.APG_MAX_CFG = 8.0
        self.NOISE_DECAY_MIN = 0.0
        self.NOISE_DECAY_MAX = 10.0

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
blend_mode: "a_only"
step_blend_mode: "b_only"
debug_mode: 2
enable_profiling: true
fbg_eta: 0.5
fbg_s_noise: 0.8
checkpoint_steps: ""
log_posterior_ema_factor: 0.3
tensor_memory_optimization: false
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
fbg_config:
  t_0: 0.7
  t_1: 0.6
  temp: 0.0011
  offset: -0.05
  cfg_scale: 500
  initial_guidance_scale: 1
  pi: 0.36
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
ancestral_strength: 1.0
noise_coherence: 0.0
step_random_mode: block
step_size: 2
seed: 42
first_ancestral_step: 0
last_ancestral_step: -1
start_sigma_index: 0
end_sigma_index: -1
enable_clamp_output: false
"""
        self.default_noise_decay_yaml = """
algorithm_type: polynomial
decay_exponent: 3.0
start_value: 1.0
end_value: 0.0
invert_curve: false
enable_temporal_smoothing: true
smoothing_window: 3
custom_piecewise_points: ""
fourier_frequency: 1.0
use_caching: true
"""

        # --- LLM PROMPTS (v0.3.8 - Genre Prompt Enhancement) ---
        self.PROMPT_GENRE_GENERATION_BASE = """
You are a highly creative AI assistant specializing in music production and audio design. Your task is to generate exactly {tag_count} descriptive elements based on the concept: "{initial_concept_prompt}".

**CRITICAL REQUIREMENTS:**
- The FIRST tag MUST be an actual music genre (e.g., "trap", "ambient", "house", "dnb", "techno", "hip-hop", "downtempo", "breakbeat").
- The remaining tags should be descriptive production techniques and musical elements.
- Include relatable terms like "stutter beat", "chopped vocals", "piano section", "analog warmth", "filtered bass".
- Focus on elements that paint a clear sonic picture.
- Avoid overly technical jargon - use terms producers and musicians would recognize.

**Examples of GOOD tag combinations:**
- "trap, stutter beats, chopped vocal textures, analog synth pads"
- "ambient, warm vinyl crackle, dreamy arpeggios, reverb-drenched guitars"
- "house, sidechained pumping, filtered bass, glitchy percussion"

{excluded_tags_instruction}
The output MUST ONLY be the comma-separated tags with NO extra text or explanations.
Output:
"""
        self.PROMPT_LYRICS_GENERATION_BASE = """
You are a creative writer and lyricist with a deep understanding of evocative language and song structure. Your task is to write lyrics for a creative project.

**Project Details:**
- **Initial Concept:** "{initial_concept_prompt}"
- **Generated Genres:** "{genre_tags}"

**Decision Logic:**
- If `force_instrumental` is `True`, you MUST output exactly: `[instrumental]`
- If `force_lyrics_generation` is `True`, you MUST generate lyrics.
- Otherwise, you may decide if an instrumental piece is more appropriate.

**CRITICAL LYRIC GENERATION RULES:**
1.  **Structure:** If generating lyrics, you MUST use the following structure precisely:
    [Verse 1]
    (4 lines)
    [Chorus]
    (4 lines)
    [Verse 2]
    (4 lines)
    [Bridge]
    (4 lines)
    [Chorus]
    (4 lines, can be same as first)
    [Outro]
    (2-4 lines)
2.  **Quality:** The lyrics should tell a cohesive story, use evocative imagery, and have emotional depth. The chorus should be memorable.
3.  **Formatting:** Each section tag (e.g., `[Verse 1]`) MUST be on its own line. Your ENTIRE output must be ONLY the lyrics or the single word `[instrumental]`. Do NOT include conversational text.

**Example of a High-Quality Output:**
[Verse 1]
Dust falls on broken signs,
A hollow wind through rusted lines.
Empty eyes in shadowed lanes,
Whispering echoes of forgotten names.
[Chorus]
Beneath a sky of ash and grey,
Hope's a flicker fading away.
A silent scream, a broken plea,
Lost in the ruins, just you and me.
[Outro]
Fading echoes, soft and low,
Where do we go, Where do we go.

{force_lyrics_instruction}{force_instrumental_instruction}
Now, based on the project details and the rules above, generate your output.
"""
        self.PROMPT_APG_STYLE_CHOICE = 'Based on the concept "{initial_concept_prompt}", is the target primarily `Audio` or an `Image`? Respond with ONLY the word.'
        self.PROMPT_APG_AUDIO = 'The target is stable audio. Provide a single number for the main APG guidance scale (2.0 to 6.0). A good start is 4.5. Output ONLY the number:'
        self.PROMPT_APG_IMAGE = 'The target is an image. Provide APG Scale (2.0-8.0) and Momentum (-0.95 to 0.95 for sharp/smooth). Output ONLY two numbers separated by a comma:'
        
        self.PROMPT_FBG_TUNING_STYLE_CHOICE = """
Based on "{initial_concept_prompt}", choose the best FBG Sampler tuning style:
- `Aggressive Audio`: For experimental, chaotic, or high-energy audio.
- `Stable Image`: For photorealistic or controlled images.
- `Fine-Tuned Audio`: For detailed or hi-fi audio.
Respond with ONLY the style name (e.g., "Fine-Tuned Audio").
"""
        self.PROMPT_FBG_AUTOPILOT = 'FBG is in AUTOPILOT mode. Provide t_0 (0.3-0.9) and t_1 (0.2-0.8). Output ONLY two numbers like: 0.7,0.4'
        self.PROMPT_FBG_MANUAL_STABLE = 'FBG is in STABLE MANUAL mode. Provide fbg_temp (0.01-0.2) and pi (0.4-0.8). Output ONLY two numbers like: 0.05,0.65'
        
        self.PROMPT_ORIGINAL_SAMPLER_STYLE = """
Based on "{initial_concept_prompt}", choose the best `noise_behavior` for the Original PingPong Sampler:
- `Dynamic`: High energy with some flow. Good for energetic video/audio.
- `Smooth`: Reduced noise. Recommended for clean, cinematic video.
- `Textured Grain`: Consistent noise pattern, like film grain.
- `Soft (DDIM-Like)`: Very little noise, clean output.
- `Default (Raw)`: Chaotic and energetic. Best for glitchy effects.
Respond with ONLY the preset name (e.g., "Smooth").
"""
        
        self.PROMPT_NOISE_DECAY_SCHEDULER_BASE = """
Based on "{initial_concept_prompt}", choose the best algorithm for the Advanced Noise Decay Scheduler: `polynomial`, `sigmoidal`, `gaussian`, `fourier`, or `exponential`. Respond with ONLY the algorithm name.
"""
        self.PROMPT_NOISE_DECAY_EXPONENT = 'Algorithm is `{algo_choice}`. Provide a `decay_exponent` (1.0-8.0). Higher is steeper. Concept: "{initial_concept_prompt}". Output ONLY the number:'
        self.PROMPT_NOISE_DECAY_FOURIER = 'Algorithm is `fourier`. Provide a `fourier_frequency` (1.0-10.0 for number of pulses). Concept: "{initial_concept_prompt}". Output ONLY the number:'
        
        self.PROMPT_DURATION_GENERATION_BASE = 'Based on context, provide a duration in seconds between {min_total_seconds:.1f} and {max_total_seconds:.1f}. Output ONLY the number.'
        self.PROMPT_NOISE_DECAY_GENERATION_BASE = 'Based on context, provide a noise decay strength between {min_noise_decay:.1f} and {max_noise_decay:.1f}. Output ONLY the number.'

    CATEGORY = "MD_Nodes/SceneGenius"

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "llm_backend": (["ollama", "lm_studio"], {"default": "ollama"}),
                "ollama_api_base_url": ("STRING", {"default": "http://localhost:11434"}),
                "ollama_model_name": (cls._get_ollama_models_list(), {"default": "llama3:8b-instruct-q8_0"}),
                "lm_studio_api_base_url": ("STRING", {"default": "http://localhost:1234"}),
                "lm_studio_model_name": (cls._get_lm_studio_models_list(), {"default": "local-model"}),
                "initial_concept_prompt": ("STRING", {"multiline": True, "default": "A dystopian cyberpunk future with a retro-futuristic soundscape."}),
                "tag_count": ("INT", {"default": 4, "min": 1, "max": 10}),
                "excluded_tags": ("STRING", {"default": ""}),
                "force_lyrics_generation": ("BOOLEAN", {"default": False}),
                "force_instrumental": ("BOOLEAN", {"default": False}),
                "min_total_seconds": ("FLOAT", {"default": 60.0, "min": 0.0, "max": 3600.0}),
                "max_total_seconds": ("FLOAT", {"default": 300.0, "min": 0.0, "max": 3600.0}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "randomize_seed": ("BOOLEAN", {"default": True}),
                "sampler_version": (["FBG Integrated PingPong Sampler", "Original PingPong Sampler"], {"default": "FBG Integrated PingPong Sampler"}),
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

    RETURN_TYPES = ("STRING", "STRING", "FLOAT", "FLOAT", "STRING", "STRING", "STRING", "INT",)
    RETURN_NAMES = ("GENRE_TAGS", "LYRICS_OR_SCRIPT", "TOTAL_SECONDS", "NOISE_DECAY_STRENGTH", "APG_YAML_PARAMS", "SAMPLER_YAML_PARAMS", "NOISE_DECAY_YAML_PARAMS", "SEED",)
    FUNCTION = "execute"

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
    def _get_lm_studio_models_list(cls):
        default_lm_studio_url = "http://localhost:1234"
        try:
            response = requests.get(f"{default_lm_studio_url}/v1/models", timeout=5)
            response.raise_for_status()
            models_data = response.json()
            models = [model["id"] for model in models_data.get("data", [])]
            return models if models else ["local-model"]
        except Exception as e:
            logger.error(f"SceneGenius: Could not fetch LM Studio models: {e}")
            return ["local-model (LM Studio not found)"]

    def _clean_llm_output(self, raw_output: str) -> str:
        cleaned = re.sub(r'<think>.*?</think>', '', raw_output, flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(r'</?(?:reasoning|analysis|internal|debug).*?>', '', cleaned, flags=re.IGNORECASE)
        ticks = "```"
        pattern = rf'{ticks}(?:yaml|output)?\s*(.*?)\s*{ticks}'
        match = re.search(pattern, cleaned, re.DOTALL)
        if match:
            cleaned = match.group(1).strip()
        cleaned = cleaned.strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()
        cleaned = '\n'.join(line.strip() for line in cleaned.split('\n') if line.strip())
        return cleaned

    def _call_llm_api(self, backend: str, api_url: str, model_name: str, prompt: str, timeout: int = 300, max_retries: int = 3) -> str:
        if backend == "ollama":
            payload = {"model": model_name, "prompt": prompt, "stream": False, "keep_alive": 0}
            return self._call_ollama_api(api_url, payload, timeout, max_retries)
        elif backend == "lm_studio":
            return self._call_lm_studio_api(api_url, model_name, prompt, timeout, max_retries)
        else:
            raise ValueError(f"Unsupported LLM backend: {backend}")

    def _call_lm_studio_api(self, api_url: str, model_name: str, prompt: str, timeout: int = 300, max_retries: int = 3) -> str:
        for attempt in range(max_retries):
            try:
                # logger.info(f"Attempt {attempt + 1}/{max_retries}: Connecting to LM Studio at {api_url} with model {model_name}")
                payload = {"model": model_name, "messages": [{"role": "user", "content": prompt}], "temperature": 0.7, "max_tokens": 2048, "stream": False}
                headers = {"Content-Type": "application/json"}
                response = requests.post(f"{api_url}/v1/chat/completions", json=payload, headers=headers, timeout=timeout)
                response.raise_for_status()
                response_data = response.json()
                content = response_data["choices"][0]["message"]["content"].strip()
                return self._clean_llm_output(content)
            except Exception as e:
                logger.warning(f"LM Studio API call failed on attempt {attempt+1}: {e}. Retrying...")
                time.sleep(2 ** attempt)
        raise RuntimeError("LM Studio API call failed after multiple retries.")

    def _call_ollama_api(self, api_url: str, payload: Dict[str, Any], timeout: int = 300, max_retries: int = 3) -> str:
        for attempt in range(max_retries):
            try:
                # logger.info(f"Attempt {attempt + 1}/{max_retries}: Connecting to Ollama at {api_url.split('/api/')[0]} with model {payload.get('model')}")
                response = requests.post(f"{api_url}/api/generate", json=payload, timeout=timeout)
                response.raise_for_status()
                raw_response = response.json().get("response", "").strip()
                return self._clean_llm_output(raw_response)
            except Exception as e:
                logger.warning(f"Ollama API call failed on attempt {attempt+1}: {e}. Retrying...")
                time.sleep(2 ** attempt)
        raise RuntimeError("Ollama API call failed after multiple retries.")

    # --- Refactored Generation Stages (v0.3.6) ---

    def _generate_genres(self, llm_backend, api_url, model_name, initial_concept_prompt, tag_count, excluded_tags, override):
        logger.info("Stage 1: Generating genre tags...")
        if override:
            return override
        try:
            excluded_tags_instruction = f"Avoid generating any of these tags: {excluded_tags}." if excluded_tags else ""
            prompt = self.PROMPT_GENRE_GENERATION_BASE.format(tag_count=tag_count, initial_concept_prompt=initial_concept_prompt, excluded_tags_instruction=excluded_tags_instruction)
            return self._call_llm_api(llm_backend, api_url, model_name, prompt)
        except Exception as e:
            logger.error(f"Genre generation failed: {e}. Falling back to 'ambient'.")
            return "ambient"

    def _generate_lyrics(self, llm_backend, api_url, model_name, initial_concept_prompt, genre_tags, force_lyrics, force_instrumental, override):
        logger.info("Stage 2: Generating lyrics/script...")
        if override:
            return override
        try:
            force_lyrics_instruction = "The user has set `force_lyrics_generation` to True." if force_lyrics else ""
            force_instrumental_instruction = "CRITICAL: The user has set `force_instrumental` to True. You MUST output exactly: [instrumental]" if force_instrumental else ""
            prompt = self.PROMPT_LYRICS_GENERATION_BASE.format(initial_concept_prompt=initial_concept_prompt, genre_tags=genre_tags, force_lyrics_instruction=force_lyrics_instruction, force_instrumental_instruction=force_instrumental_instruction)
            lyrics = self._call_llm_api(llm_backend, api_url, model_name, prompt)
            return "[instrumental]" if force_instrumental else lyrics
        except Exception as e:
            logger.error(f"Lyrics generation failed: {e}. Setting to instrumental.")
            return "[instrumental]"

    def _generate_duration(self, llm_backend, api_url, model_name, initial_concept_prompt, genre_tags, lyrics, min_sec, max_sec, override):
        logger.info("Stage 3: Generating duration...")
        total_seconds = -1
        if override:
            try:
                total_seconds = float(override)
                logger.info(f"Using user-provided duration: {total_seconds}s")
            except ValueError:
                logger.warning(f"Could not parse duration override '{override}'. Using LLM.")
        
        if total_seconds == -1:
            try:
                prompt = self.PROMPT_DURATION_GENERATION_BASE.format(min_total_seconds=min_sec, max_total_seconds=max_sec, initial_concept_prompt=initial_concept_prompt, genre_tags=genre_tags, lyrics_or_script=lyrics)
                response = self._call_llm_api(llm_backend, api_url, model_name, prompt)
                match = re.search(r'(\d+\.?\d*)', response)
                total_seconds = float(match.group(1)) if match else min_sec
            except Exception as e:
                logger.warning(f"Duration generation failed: {e}. Using min duration.")
                total_seconds = min_sec
        
        return max(min_sec, min(total_seconds, max_sec))

    def _generate_noise_decay_strength(self, llm_backend, api_url, model_name, initial_concept_prompt, genre_tags, lyrics, total_seconds, override):
        logger.info("Stage 4: Generating noise decay strength...")
        strength = -1
        if override:
            try:
                strength = float(override)
                logger.info(f"Using user-provided noise decay strength: {strength}")
            except ValueError:
                logger.warning(f"Could not parse noise decay override '{override}'. Using LLM.")

        if strength == -1:
            try:
                prompt = self.PROMPT_NOISE_DECAY_GENERATION_BASE.format(min_noise_decay=self.NOISE_DECAY_MIN, max_noise_decay=self.NOISE_DECAY_MAX, initial_concept_prompt=initial_concept_prompt, genre_tags=genre_tags, lyrics_or_script=lyrics, total_seconds=total_seconds)
                response = self._call_llm_api(llm_backend, api_url, model_name, prompt)
                match = re.search(r'(\d+\.?\d*)', response)
                strength = float(match.group(1)) if match else 5.0
            except Exception as e:
                logger.warning(f"Noise decay generation failed: {e}. Using default: 5.0.")
                strength = 5.0
        
        return max(self.NOISE_DECAY_MIN, min(strength, self.NOISE_DECAY_MAX))

    def _configure_apg_yaml(self, llm_backend, api_url, model_name, initial_concept_prompt, genre_tags, override):
        logger.info("Stage 5: Generating Context-Aware APG YAML.")
        if override:
            return override
        try:
            style_prompt = self.PROMPT_APG_STYLE_CHOICE.format(initial_concept_prompt=initial_concept_prompt, genre_tags=genre_tags)
            style_choice = self._call_llm_api(llm_backend, api_url, model_name, style_prompt)
            logger.info(f"LLM chose APG Style: {style_choice}")

            if "Audio" in style_choice:
                data = yaml.safe_load(self.default_apg_yaml_audio)
                prompt = self.PROMPT_APG_AUDIO.format(initial_concept_prompt=initial_concept_prompt, genre_tags=genre_tags)
                raw_response = self._call_llm_api(llm_backend, api_url, model_name, prompt)
                apg_scale = max(2.0, min(6.0, float(raw_response)))
                logger.info(f"LLM suggested Audio APG Scale: {apg_scale:.2f}")
                if 'rules' in data and len(data['rules']) >= 4:
                    data['rules'][1]['apg_scale'] = apg_scale
                    data['rules'][1]['cfg'] = apg_scale + 0.5
                    data['rules'][2]['apg_scale'] = apg_scale / 2.0
            else:
                data = yaml.safe_load(self.default_apg_yaml_image)
                prompt = self.PROMPT_APG_IMAGE.format(initial_concept_prompt=initial_concept_prompt, genre_tags=genre_tags)
                raw_response = self._call_llm_api(llm_backend, api_url, model_name, prompt)
                values = [v.strip() for v in raw_response.split(',')]
                if len(values) >= 2:
                    apg_scale = max(self.APG_MIN_CFG, min(self.APG_MAX_CFG, float(values[0])))
                    momentum = max(-0.95, min(0.95, float(values[1])))
                    logger.info(f"LLM suggested Image APG Scale: {apg_scale:.2f}, Momentum: {momentum:.2f}")
                    if 'rules' in data and len(data['rules']) > 1:
                        data['rules'][1]['apg_scale'] = apg_scale
                        data['rules'][1]['cfg'] = apg_scale
                        data['rules'][1]['momentum'] = momentum
            
            return yaml.dump(data, Dumper=CustomDumper, indent=2, sort_keys=False)
        except Exception as e:
            logger.warning(f"Enhanced APG YAML generation failed: {e}. Using audio default.")
            return self.default_apg_yaml_audio

    def _configure_sampler_yaml(self, llm_backend, api_url, model_name, initial_concept_prompt, genre_tags, sampler_version, override):
        logger.info(f"Stage 6: Generating INTELLIGENT Sampler YAML for '{sampler_version}'.")
        if override:
            return override
        
        if sampler_version == "FBG Integrated PingPong Sampler":
            try:
                style_prompt = self.PROMPT_FBG_TUNING_STYLE_CHOICE.format(initial_concept_prompt=initial_concept_prompt, genre_tags=genre_tags)
                style_choice = self._call_llm_api(llm_backend, api_url, model_name, style_prompt)
                logger.info(f"LLM chose FBG Tuning Style: {style_choice}")

                data = yaml.safe_load(self.default_sampler_yaml_fbg)
                fbg_config = data.get('fbg_config', {})

                if "Aggressive" in style_choice:
                    param_prompt = self.PROMPT_FBG_AUTOPILOT
                    raw_params = self._call_llm_api(llm_backend, api_url, model_name, param_prompt)
                    values = [v.strip() for v in raw_params.split(',')]
                    if len(values) >= 2:
                        fbg_config['t_0'], fbg_config['t_1'], fbg_config['pi'] = max(0.3, min(0.9, float(values[0]))), max(0.2, min(0.8, float(values[1]))), 0.35
                elif "Stable" in style_choice:
                    param_prompt = self.PROMPT_FBG_MANUAL_STABLE
                    raw_params = self._call_llm_api(llm_backend, api_url, model_name, param_prompt)
                    values = [v.strip() for v in raw_params.split(',')]
                    if len(values) >= 2:
                        fbg_config['t_0'], fbg_config['t_1'], fbg_config['temp'], fbg_config['pi'] = 0.0, 0.0, max(0.01, min(0.2, float(values[0]))), max(0.4, min(0.8, float(values[1])))
                else: 
                    logger.info("Defaulting to Fine-Tuned Audio style.")
                    fbg_config.update({'t_0': 0.0, 't_1': 0.0, 'temp': 0.0011, 'pi': 0.35})

                data['fbg_config'] = fbg_config
                return yaml.dump(data, Dumper=CustomDumper, indent=2, sort_keys=False)
            except Exception as e:
                logger.warning(f"Intelligent Sampler YAML generation failed: {e}. Using default.")
                return self.default_sampler_yaml_fbg
        
        elif sampler_version == "Original PingPong Sampler":
            try:
                prompt = self.PROMPT_ORIGINAL_SAMPLER_STYLE.format(initial_concept_prompt=initial_concept_prompt)
                behavior_choice = self._call_llm_api(llm_backend, api_url, model_name, prompt).strip()
                logger.info(f"LLM chose Original Sampler behavior: {behavior_choice}")
                data = yaml.safe_load(self.default_sampler_yaml_original)
                data['noise_behavior'] = behavior_choice
                return yaml.dump(data, Dumper=CustomDumper, indent=2, sort_keys=False)
            except Exception as e:
                logger.warning(f"Original Sampler YAML generation failed: {e}. Using default.")
                return self.default_sampler_yaml_original
        
        return "" # Should not be reached

    def _configure_noise_decay_yaml(self, llm_backend, api_url, model_name, initial_concept_prompt):
        logger.info("Stage 7: Generating INTELLIGENT Noise Decay Scheduler recommendation.")
        try:
            prompt = self.PROMPT_NOISE_DECAY_SCHEDULER_BASE.format(initial_concept_prompt=initial_concept_prompt)
            algo_choice = self._call_llm_api(llm_backend, api_url, model_name, prompt).strip().lower()
            logger.info(f"LLM chose Noise Decay Algorithm: {algo_choice}")
            
            data = yaml.safe_load(self.default_noise_decay_yaml)
            
            if "polynomial" in algo_choice or "sigmoidal" in algo_choice or "exponential" in algo_choice or "gaussian" in algo_choice:
                data['algorithm_type'] = algo_choice
                param_prompt = self.PROMPT_NOISE_DECAY_EXPONENT.format(algo_choice=algo_choice, initial_concept_prompt=initial_concept_prompt)
                raw_response = self._call_llm_api(llm_backend, api_url, model_name, param_prompt)
                exponent = max(1.0, min(8.0, float(raw_response)))
                data['decay_exponent'] = exponent
                logger.info(f"LLM suggested decay_exponent: {exponent}")
            elif "fourier" in algo_choice:
                data['algorithm_type'] = algo_choice
                param_prompt = self.PROMPT_NOISE_DECAY_FOURIER.format(initial_concept_prompt=initial_concept_prompt)
                raw_response = self._call_llm_api(llm_backend, api_url, model_name, param_prompt)
                frequency = max(1.0, min(10.0, float(raw_response)))
                data['fourier_frequency'] = frequency
                logger.info(f"LLM suggested fourier_frequency: {frequency}")
            else:
                logger.warning(f"LLM returned invalid algorithm '{algo_choice}'. Using default.")
            
            return yaml.dump(data, Dumper=CustomDumper, indent=2, sort_keys=False)
        except Exception as e:
            logger.warning(f"Intelligent Noise Decay YAML generation failed: {e}. Using default.")
            return self.default_noise_decay_yaml


    def execute(self, llm_backend, ollama_api_base_url, ollama_model_name, lm_studio_api_base_url, lm_studio_model_name, initial_concept_prompt, tag_count, excluded_tags, force_lyrics_generation, force_instrumental, min_total_seconds, max_total_seconds, seed, randomize_seed, sampler_version, **kwargs):
        logger.info("--- Starting SceneGeniusAutocreator v0.3.7 ---")
        if randomize_seed:
            seed = random.randint(0, 0xffffffffffffffff)
            logger.info(f"Generated random seed: {seed}")

        if force_lyrics_generation and force_instrumental:
            logger.warning("`force_lyrics` and `force_instrumental` are both True. `force_instrumental` will take precedence.")
            force_lyrics_generation = False
        
        prompt_overrides = {k: v for k, v in kwargs.items() if k.startswith('prompt_')}
        test_mode = kwargs.get('test_mode', False)
        
        if llm_backend == "ollama":
            api_url, model_name = ollama_api_base_url, ollama_model_name
        elif llm_backend == "lm_studio":
            api_url, model_name = lm_studio_api_base_url, lm_studio_model_name
        else:
            raise ValueError(f"Unsupported backend: {llm_backend}")
            
        if test_mode:
            logger.info("Test mode enabled. Returning dummy output.")
            return ("electro, synthwave", "[Verse 1]\n...", 180.0, 3.5, self.default_apg_yaml_audio, self.default_sampler_yaml_fbg, self.default_noise_decay_yaml, seed)

        # --- Execute Generation Stages ---
        genre_tags = self._generate_genres(llm_backend, api_url, model_name, initial_concept_prompt, tag_count, excluded_tags, prompt_overrides.get("prompt_genre_generation", "").strip())
        
        lyrics_or_script = self._generate_lyrics(llm_backend, api_url, model_name, initial_concept_prompt, genre_tags, force_lyrics_generation, force_instrumental, prompt_overrides.get("prompt_lyrics_decision_and_generation", "").strip())
        
        total_seconds = self._generate_duration(llm_backend, api_url, model_name, initial_concept_prompt, genre_tags, lyrics_or_script, min_total_seconds, max_total_seconds, prompt_overrides.get("prompt_duration_generation", "").strip())
        
        noise_decay_strength = self._generate_noise_decay_strength(llm_backend, api_url, model_name, initial_concept_prompt, genre_tags, lyrics_or_script, total_seconds, prompt_overrides.get("prompt_noise_decay_generation", "").strip())
        
        apg_yaml_params = self._configure_apg_yaml(llm_backend, api_url, model_name, initial_concept_prompt, genre_tags, prompt_overrides.get("prompt_apg_yaml_generation", "").strip())
        
        sampler_yaml_params = self._configure_sampler_yaml(llm_backend, api_url, model_name, initial_concept_prompt, genre_tags, sampler_version, prompt_overrides.get("prompt_sampler_yaml_generation", "").strip())
        
        noise_decay_yaml_params = self._configure_noise_decay_yaml(llm_backend, api_url, model_name, initial_concept_prompt)
        
        logger.info("--- SceneGeniusAutocreator execution finished ---")
        return (genre_tags, lyrics_or_script, total_seconds, noise_decay_strength, apg_yaml_params, sampler_yaml_params, noise_decay_yaml_params, seed)


# Node Mapping
NODE_CLASS_MAPPINGS = {
    "SceneGeniusAutocreator": SceneGeniusAutocreator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SceneGeniusAutocreator": "Scene Genius Autocreator"
}

