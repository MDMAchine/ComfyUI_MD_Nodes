# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# ████ SCENEGENIUS AUTOCREATOR v1.0.1 – Final Polish (Complete) ████▓▒░
# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀

# ░▒▓ ORIGIN & DEV:
#     • Cast into the void by: MDMAchine (OG SysOp)
#     • Original mind behind SceneGenius Autocreator
#     • Initial ComfyUI adaptation by: Gemini (Google)
#     • Enhanced & refined by: MDMAchine & Gemini & Claude (Anthropic)
#     • Final Polish & Review: Gemini (Google)
#     • License: Apache 2.0 — No cursed floppy disks allowed

# ░▒▓ DESCRIPTION:
#     A multi-stage AI creative weapon node for ComfyUI.
#     Designed to automate Ace-Step diffusion content generation,
#     channeling the chaotic spirit of the demoscene and BBS era.
#     Produces authentic genres, adaptive lyrics, precise durations,
#     and finely tuned APG + Sampler configs with ease.

# ░▒▓ CHANGELOG:
#     - v1.0.1 (Final Polish):
#         • FIXED: Included all methods that were mistakenly omitted in the previous version.
#         • REFACTORED: Implemented final code review suggestions for production readiness.
#         • FIXED: Separated model list cache timestamps to prevent conflicts.
#         • ROBUSTNESS: Added a centralized `try/except` block with a fallback output method for critical failures.
#         • ROBUSTNESS: Added default values for all `kwargs` in the execute method.
#         • QUALITY: Added more specific logging for instrumental mode detection.
#         • QUALITY: Improved the return type hint on the `execute` method for clarity.
#     - v1.0.0 (Professional Refactor):
#         • REFACTORED: Implemented a comprehensive professional code review.
#         • SECURITY: Added URL validation for all user-provided API endpoints.
#         • PERFORMANCE: Implemented lazy-loading for model lists to prevent UI blocking on startup.
#         • MAINTAINABILITY: Moved all large prompt strings to class-level constants.
#         • MAINTAINABILITY: Added comprehensive type hints and docstrings to all methods.

import logging
import yaml
import requests
import re
import time
import random
from typing import Dict, Any, Tuple, List, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

try:
    import folder_paths
except ImportError:
    logger.warning("Not running in ComfyUI environment. Mocking folder_paths.")
    class MockFolderPaths:
        def get_full_path(self, filename: str, subdir: str) -> str:
            return filename
    folder_paths = MockFolderPaths()

class CustomDumper(yaml.Dumper):
    def represent_sequence(self, tag: Any, data: Any, flow_style: Optional[bool] = None) -> Any:
        if len(data) <= 2 and all(isinstance(x, (int, float)) for x in data):
            return super().represent_sequence(tag, data, flow_style=True)
        return super().represent_sequence(tag, data, flow_style=flow_style)

class SceneGeniusAutocreator:
    """
    A multi-stage ComfyUI node leveraging local LLMs for dynamic creative content generation.
    """
    
    # --- Constants ---
    QUALITY_PRESETS: Dict[str, int] = { "Draft": 60, "Standard": 220, "High": 420, "Ultra": 720, "Max": 1000 }
    DEFAULT_OLLAMA_MODEL: str = "llama3:8b-instruct-q8_0"
    DEFAULT_LM_STUDIO_MODEL: str = "local-model"
    DEFAULT_OLLAMA_URL: str = "http://localhost:11434"
    DEFAULT_LM_STUDIO_URL: str = "http://localhost:1234"
    
    MAX_RETRY_ATTEMPTS: int = 3
    API_TIMEOUT_SECONDS: int = 300
    MODEL_LIST_CACHE_SECONDS: int = 300 # 5 minutes

    APG_MIN_CFG: float = 2.0
    APG_MAX_CFG: float = 8.0
    NOISE_DECAY_MIN: float = 0.0
    NOISE_DECAY_MAX: float = 10.0

    # --- Caching ---
    _ollama_models_cache: Optional[List[str]] = None
    _lm_studio_models_cache: Optional[List[str]] = None
    _ollama_cache_timestamp: float = 0.0
    _lm_studio_cache_timestamp: float = 0.0

    # --- LLM PROMPTS ---
    PROMPT_GENRE_GENERATION_BASE: str = """You are a highly creative AI assistant specializing in music production and audio design. Your task is to generate exactly {tag_count} descriptive elements based on the concept: "{initial_concept_prompt}".

**CRITICAL REQUIREMENTS:**
- The FIRST tag MUST be an actual music genre (e.g., "trap", "ambient", "house", "dnb", "techno", "hip-hop", "downtempo", "breakbeat")
- The remaining tags should be descriptive production techniques and musical elements
- Include relatable terms like "stutter beat", "chopped vocals", "piano section", "analog warmth", "filtered bass"
- Focus on elements that paint a clear sonic picture

{excluded_tags_instruction}
The output MUST ONLY be the comma-separated tags with NO extra text or explanations.
Output:
"""
    PROMPT_LYRICS_GENERATION_BASE: str = """You are a creative writer and lyricist. Your task is to write lyrics for a creative project.

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
    PROMPT_DURATION_GENERATION_BASE: str = """Based on the concept "{initial_concept_prompt}", genres "{genre_tags}", and content "{lyrics_or_script}", provide a duration in seconds between {min_total_seconds:.1f} and {max_total_seconds:.1f}. 

Output ONLY the number.
"""
    PROMPT_NOISE_DECAY_GENERATION_BASE: str = """Based on the concept "{initial_concept_prompt}", provide a noise decay strength between {min_noise_decay:.1f} and {max_noise_decay:.1f}.

Output ONLY the number.
"""
    PROMPT_APG_STYLE_CHOICE: str = """Based on the concept "{initial_concept_prompt}", is the target primarily Audio or Image? Respond with ONLY the word "Audio" or "Image"."""
    PROMPT_APG_AUDIO: str = """Based on concept "{initial_concept_prompt}" and genres "{genre_tags}", provide a single number for APG guidance scale (2.0 to 6.0). A balanced value is 4.5.

Output ONLY the number:
"""
    PROMPT_APG_IMAGE: str = """Based on concept "{initial_concept_prompt}", provide two values:
1. APG Scale (2.0-8.0)
2. Momentum (-0.95 to 0.95, negative for sharp, positive for smooth)

Output ONLY two numbers separated by a comma (e.g., 5.2,-0.4):
"""
    PROMPT_SAMPLER_FBG: str = """Based on concept "{initial_concept_prompt}" and genres "{genre_tags}", provide three values for FBG sampler:
1. t_0 (0.7-0.9, higher = more sensitive)
2. cfg_scale (500-800, higher = more aggressive)  
3. max_guidance_scale (800-1200, guidance ceiling)

Output ONLY three numbers separated by commas (e.g., 0.85,650,1000):
"""
    PROMPT_SAMPLER_ORIGINAL: str = """Based on concept "{initial_concept_prompt}", choose the best noise_behavior preset: "Default (Raw)", "Dynamic", "Smooth", "Textured Grain", or "Soft (DDIM-Like)".

Output ONLY the preset name:
"""
    PROMPT_NOISE_DECAY_ALGORITHM: str = """Based on the concept "{initial_concept_prompt}", choose the best algorithm: polynomial, sigmoidal, gaussian, fourier, or exponential.

Output ONLY the algorithm name:
"""
    PROMPT_NOISE_DECAY_EXPONENT: str = """For {algo_choice} algorithm with concept "{initial_concept_prompt}", provide decay_exponent (1.0-8.0, higher = steeper).

Output ONLY the number:
"""
    
    def __init__(self) -> None:
        """Initialize default YAML configurations."""
        self.default_apg_yaml_audio: str = """
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
        self.default_apg_yaml_image: str = """
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
        self.default_sampler_yaml_fbg: str = """
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
debug_mode: 2
enable_profiling: true
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
        self.default_sampler_yaml_original: str = """
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
        self.default_noise_decay_yaml: str = """
algorithm_type: polynomial
decay_exponent: 5.0
start_value: 1.0
end_value: 0.0
invert_curve: false
enable_temporal_smoothing: true
smoothing_window: 3
"""

    CATEGORY: str = "MD_Nodes/SceneGenius"

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "llm_backend": (["ollama", "lm_studio"], {"default": "ollama", "tooltip": "Select the local LLM server backend to use for generation."}),
                "ollama_api_base_url": ("STRING", {"default": cls.DEFAULT_OLLAMA_URL, "tooltip": "The base URL for your Ollama server API."}),
                "ollama_model_name": (cls._get_ollama_models_lazy(), {"default": cls.DEFAULT_OLLAMA_MODEL, "tooltip": "The specific model to use from your Ollama server."}),
                "lm_studio_api_base_url": ("STRING", {"default": cls.DEFAULT_LM_STUDIO_URL, "tooltip": "The base URL for your LM Studio server, which uses an OpenAI-compatible API."}),
                "lm_studio_model_name": (cls._get_lm_studio_models_lazy(), {"default": cls.DEFAULT_LM_STUDIO_MODEL, "tooltip": "The specific model loaded in your LM Studio server."}),
                "initial_concept_prompt": ("STRING", {"multiline": True, "default": "A dystopian cyberpunk future with a retro-futuristic soundscape.", "tooltip": "The core creative idea. This prompt drives all subsequent generation stages, from genre and lyrics to technical parameters."}),
                "tag_count": ("INT", {"default": 4, "min": 1, "max": 10, "tooltip": "The number of descriptive genre/style tags the LLM should generate."}),
                "excluded_tags": ("STRING", {"default": "", "tooltip": "A comma-separated list of tags to prevent the LLM from generating."}),
                "force_lyrics_generation": ("BOOLEAN", {"default": False, "tooltip": "If True, forces the LLM to generate lyrics, overriding its own creative decision."}),
                "force_instrumental": ("BOOLEAN", {"default": False, "tooltip": "If True, forces the output to be '[instrumental]', taking precedence over 'force_lyrics_generation'."}),
                "min_total_seconds": ("FLOAT", {"default": 60.0, "min": 0.0, "max": 3600.0, "tooltip": "The minimum possible duration for the generated content, in seconds."}),
                "max_total_seconds": ("FLOAT", {"default": 300.0, "min": 0.0, "max": 3600.0, "tooltip": "The maximum possible duration for the generated content, in seconds."}),
                "quality_preset": (list(cls.QUALITY_PRESETS.keys()) + ["Manual"], {"default": "Standard", "tooltip": "Select a quality level to determine sampling steps, or choose 'Manual' to specify your own."}),
                "manual_steps": ("INT", {"default": 500, "min": 1, "max": 8192, "step": 1, "tooltip": "Set the total number of sampling steps manually. Only used when quality preset is 'Manual'."}),
                "base_tag_boost": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 10.0, "step": 0.1, "tooltip": "Sets the boost strength for the base genre tags."}),
                "vocal_tag_boost": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1, "tooltip": "Sets the boost strength for the extracted vocal tags."}),
                "vocal_weight": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 0.95, "step": 0.01, "tooltip": "Sets the weight for blending vocal characteristics."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "The seed for all random operations. Use the same seed for reproducible results."}),
                "randomize_seed": ("BOOLEAN", {"default": True, "tooltip": "If True, a new random seed will be generated for each execution, ignoring the value set above."}),
                "sampler_version": (["FBG Integrated PingPong Sampler", "Original PingPong Sampler"], {"default": "FBG Integrated PingPong Sampler", "tooltip": "Selects which PingPong sampler's YAML configuration to generate."}),
                "generate_noise_decay_yaml": ("BOOLEAN", {"default": False, "tooltip": "If True, the LLM will generate a configuration for the Advanced Noise Decay Scheduler. Otherwise, a default is used."}),
            },
            "optional": {
                "genre_input": ("STRING", {"default": "", "multiline": True, "tooltip": "Manually provide genre tags. Can be used to override the LLM or combine with its output via 'Mixed Mode'."}),
                "genre_mixed_mode": ("BOOLEAN", {"default": False, "tooltip": "If True, your 'genre_input' will be prepended to the tags generated by the LLM."}),
                "prompt_vocal_tags_generation": ("STRING", {"multiline": True, "default": "", "tooltip": "ADVANCED: Override the vocal tag extraction with your own text (e.g., 'ethereal vocal chops')."}),
                "prompt_lyrics_decision_and_generation": ("STRING", {"multiline": True, "default": "", "tooltip": "ADVANCED: Override the entire lyrics generation stage with your own text."}),
                "prompt_duration_generation": ("STRING", {"multiline": True, "default": "", "tooltip": "ADVANCED: Override the duration generation with your own value (e.g., '120.0')."}),
                "prompt_noise_decay_generation": ("STRING", {"multiline": True, "default": "", "tooltip": "ADVANCED: Override the noise decay strength generation with your own value (e.g., '4.5')."}),
                "prompt_apg_yaml_generation": ("STRING", {"multiline": True, "default": "", "tooltip": "ADVANCED: Override the APG YAML generation by providing your own complete YAML configuration."}),
                "prompt_sampler_yaml_generation": ("STRING", {"multiline": True, "default": "", "tooltip": "ADVANCED: Override the Sampler YAML generation by providing your own complete YAML configuration."}),
                "test_mode": ("BOOLEAN", {"default": False, "tooltip": "DEV: If True, bypasses all LLM calls and returns dummy data for quick testing of workflow connections."})
            }
        }

    RETURN_TYPES: Tuple[str, ...] = ("STRING", "STRING", "FLOAT", "INT", "INT", "INT", "FLOAT", "FLOAT", "STRING", "STRING", "STRING", "STRING", "INT",)
    RETURN_NAMES: Tuple[str, ...] = ("GENRE_TAGS", "LYRICS_OR_SCRIPT", "TOTAL_SECONDS", "TOTAL_STEPS", "BASE_TAG_BOOST", "VOCAL_TAG_BOOST", "VOCAL_WEIGHT", "NOISE_DECAY_STRENGTH", "VOCAL_TAGS", "APG_YAML_PARAMS", "SAMPLER_YAML_PARAMS", "NOISE_DECAY_YAML_PARAMS", "SEED",)
    FUNCTION: str = "execute"

    @classmethod
    def _is_ollama_cache_valid(cls) -> bool:
        """Checks if the Ollama model list cache is still valid."""
        return (time.time() - cls._ollama_cache_timestamp) < cls.MODEL_LIST_CACHE_SECONDS

    @classmethod
    def _is_lm_studio_cache_valid(cls) -> bool:
        """Checks if the LM Studio model list cache is still valid."""
        return (time.time() - cls._lm_studio_cache_timestamp) < cls.MODEL_LIST_CACHE_SECONDS

    @classmethod
    def _get_ollama_models_lazy(cls) -> List[str]:
        """Lazy-loads Ollama models, using a cache to avoid blocking UI."""
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
            return [f"{cls.DEFAULT_OLLAMA_MODEL} (Ollama not found)"]

    @classmethod
    def _get_lm_studio_models_lazy(cls) -> List[str]:
        """Lazy-loads LM Studio models, using a cache to avoid blocking UI."""
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
            return [f"{cls.DEFAULT_LM_STUDIO_MODEL} (LM Studio not found)"]
    
    def _validate_url(self, url: str) -> bool:
        """Validates that a URL has a valid scheme and network location."""
        try:
            result = urlparse(url)
            return all([result.scheme in ['http', 'https'], result.netloc])
        except (ValueError, AttributeError):
            return False

    def _clean_llm_output(self, raw_output: str) -> str:
        """Cleans raw LLM output by removing thought tags and extracting content from code blocks."""
        cleaned = re.sub(r'<think>.*?</think>', '', raw_output, flags=re.DOTALL | re.IGNORECASE)
        
        backticks = "```"
        code_block_pattern = rf'{backticks}(?:yaml|output|python)?\s*(.*?)\s*{backticks}'
        
        if match := re.search(code_block_pattern, cleaned, re.DOTALL):
            cleaned = match.group(1)
        
        return '\n'.join(line.rstrip() for line in cleaned.strip().splitlines() if line.strip())

    def _call_llm_api(self, backend: str, api_url: str, model_name: str, prompt: str) -> str:
        """Unified and validated API call method for different LLM backends."""
        if not self._validate_url(api_url):
            error_message = f"Invalid or disallowed API URL provided: {api_url}"
            logger.error(error_message)
            raise ValueError(error_message)
            
        if backend == "ollama":
            return self._call_ollama_api(api_url, model_name, prompt)
        elif backend == "lm_studio":
            return self._call_lm_studio_api(api_url, model_name, prompt)
        else:
            raise ValueError(f"Unsupported LLM backend: {backend}")

    def _call_lm_studio_api(self, api_url: str, model_name: str, prompt: str) -> str:
        """Calls the LM Studio API with retry logic."""
        for attempt in range(self.MAX_RETRY_ATTEMPTS):
            try:
                payload = { "model": model_name, "messages": [{"role": "user", "content": prompt}], "temperature": 0.7, "max_tokens": 2048, "stream": False }
                headers = {"Content-Type": "application/json"}
                response = requests.post(f"{api_url}/v1/chat/completions", json=payload, headers=headers, timeout=self.API_TIMEOUT_SECONDS)
                response.raise_for_status()
                response_data = response.json()
                content = response_data["choices"][0]["message"]["content"].strip()
                return self._clean_llm_output(content)
            except (requests.RequestException, KeyError, IndexError) as e:
                logger.warning(f"LM Studio API call failed (Attempt {attempt + 1}/{self.MAX_RETRY_ATTEMPTS}): {e}. Retrying...")
                time.sleep(2 ** attempt)
        raise RuntimeError("LM Studio API call failed after multiple retries.")

    def _call_ollama_api(self, api_url: str, model_name: str, prompt: str) -> str:
        """Calls the Ollama API with retry logic."""
        payload = {"model": model_name, "prompt": prompt, "stream": False, "keep_alive": 0}
        for attempt in range(self.MAX_RETRY_ATTEMPTS):
            try:
                response = requests.post(f"{api_url}/api/generate", json=payload, timeout=self.API_TIMEOUT_SECONDS)
                response.raise_for_status()
                raw_response = response.json().get("response", "").strip()
                return self._clean_llm_output(raw_response)
            except (requests.RequestException, KeyError) as e:
                logger.warning(f"Ollama API call failed (Attempt {attempt + 1}/{self.MAX_RETRY_ATTEMPTS}): {e}. Retrying...")
                time.sleep(2 ** attempt)
        raise RuntimeError("Ollama API call failed after multiple retries.")
    
    def _extract_vocal_tags(self, genre_tags: str) -> Tuple[str, str]:
        """
        Separates vocal-related tags from genre tags using whole-word matching.
        Keywords are sorted by length to prevent substring issues (e.g., 'male' in 'female').
        """
        vocal_keywords = [
            'vocal', 'vocals', 'voice', 'singing', 'lyrics', 'choir', 'chant', 
            'opera', 'ad-lib', 'male', 'female', 'chorus', 'yell', 'spoken', 
            'rap', 'mc', 'acapella', 'harmony', 'beatbox', 'scat', 'whisper', 
            'scream', 'growl'
        ]
        
        vocal_keywords.sort(key=len, reverse=True)
        vocal_pattern = re.compile(r'\b(' + '|'.join(vocal_keywords) + r')\b', re.IGNORECASE)
        
        tags_list = [tag.strip() for tag in genre_tags.split(',')]
        vocal_tags, non_vocal_tags = [], []
        
        for tag in tags_list:
            if vocal_pattern.search(tag):
                vocal_tags.append(tag)
            else:
                non_vocal_tags.append(tag)
                
        return ", ".join(vocal_tags), ", ".join(non_vocal_tags)

    def _generate_genres(self, llm_backend: str, api_url: str, model_name: str, initial_concept_prompt: str, tag_count: int, excluded_tags: str, genre_input: str, genre_mixed_mode: bool) -> str:
        """Generates genre tags, handling user overrides and mixed mode."""
        logger.info("Stage 1: Generating genre tags...")
        user_input = genre_input.strip()

        if user_input and not genre_mixed_mode:
            logger.info(f"Using user-provided genre input exclusively: '{user_input}'")
            return user_input
            
        try:
            prompt = self.PROMPT_GENRE_GENERATION_BASE.format(
                tag_count=tag_count,
                initial_concept_prompt=initial_concept_prompt,
                excluded_tags_instruction=f"Avoid generating any of these tags: {excluded_tags}." if excluded_tags else ""
            )
            llm_tags = self._call_llm_api(llm_backend, api_url, model_name, prompt)

            if user_input and genre_mixed_mode:
                logger.info(f"Mixed mode: Prepending user input '{user_input}' to LLM tags.")
                return f"{user_input}, {llm_tags}"
            else:
                return llm_tags
        except (ValueError, RuntimeError) as e:
            logger.error(f"Genre generation failed: {e}. Falling back...")
            return user_input if user_input else "ambient"

    def _generate_lyrics(self, llm_backend: str, api_url: str, model_name: str, initial_concept_prompt: str, genre_tags: str, force_lyrics: bool, force_instrumental: bool, override: str) -> str:
        """Generates lyrics or an [instrumental] tag."""
        logger.info("Stage 2: Generating lyrics/script...")
        if override and override.strip(): return override
        if force_instrumental: return "[instrumental]"
        try:
            prompt = self.PROMPT_LYRICS_GENERATION_BASE.format(
                initial_concept_prompt=initial_concept_prompt,
                genre_tags=genre_tags,
                force_lyrics_instruction="The user has set `force_lyrics_generation` to True." if force_lyrics else "",
                force_instrumental_instruction=""
            )
            return self._call_llm_api(llm_backend, api_url, model_name, prompt)
        except (ValueError, RuntimeError) as e:
            logger.error(f"Lyrics generation failed: {e}. Setting to instrumental.")
            return "[instrumental]"
    
    def _generate_duration(self, llm_backend: str, api_url: str, model_name: str, initial_concept_prompt: str, genre_tags: str, lyrics: str, min_sec: float, max_sec: float, override: str) -> float:
        """Generates a duration in seconds for the content."""
        logger.info("Stage 3: Generating duration...")
        if override and override.strip():
            try:
                total_seconds = float(override)
                logger.info(f"Using user-provided duration: {total_seconds}s")
                return max(min_sec, min(total_seconds, max_sec))
            except ValueError:
                logger.warning(f"Could not parse duration override '{override}'. Using LLM.")
        try:
            prompt = self.PROMPT_DURATION_GENERATION_BASE.format(min_total_seconds=min_sec, max_total_seconds=max_sec, initial_concept_prompt=initial_concept_prompt, genre_tags=genre_tags, lyrics_or_script=lyrics)
            response = self._call_llm_api(llm_backend, api_url, model_name, prompt)
            match = re.search(r'(\d+\.?\d*)', response)
            total_seconds = float(match.group(1)) if match else min_sec
            return max(min_sec, min(total_seconds, max_sec))
        except (ValueError, RuntimeError, AttributeError) as e:
            logger.warning(f"Duration generation failed: {e}. Using min duration.")
            return min_sec

    def _generate_noise_decay_strength(self, llm_backend: str, api_url: str, model_name: str, initial_concept_prompt: str, genre_tags: str, lyrics: str, total_seconds: float, override: str) -> float:
        """Generates a noise decay strength value."""
        logger.info("Stage 4: Generating noise decay strength...")
        if override and override.strip():
            try:
                strength = float(override)
                logger.info(f"Using user-provided noise decay strength: {strength}")
                return max(self.NOISE_DECAY_MIN, min(strength, self.NOISE_DECAY_MAX))
            except ValueError:
                logger.warning(f"Could not parse noise decay override '{override}'. Using LLM.")
        try:
            prompt = self.PROMPT_NOISE_DECAY_GENERATION_BASE.format(min_noise_decay=self.NOISE_DECAY_MIN, max_noise_decay=self.NOISE_DECAY_MAX, initial_concept_prompt=initial_concept_prompt, genre_tags=genre_tags, lyrics_or_script=lyrics, total_seconds=total_seconds)
            response = self._call_llm_api(llm_backend, api_url, model_name, prompt)
            match = re.search(r'(\d+\.?\d*)', response)
            strength = float(match.group(1)) if match else 5.0
            return max(self.NOISE_DECAY_MIN, min(strength, self.NOISE_DECAY_MAX))
        except (ValueError, RuntimeError, AttributeError) as e:
            logger.warning(f"Noise decay generation failed: {e}. Using default: 5.0.")
            return 5.0

    def _configure_apg_yaml(self, llm_backend: str, api_url: str, model_name: str, initial_concept_prompt: str, genre_tags: str, override: str) -> str:
        """Generates APG YAML parameters."""
        logger.info("Stage 5: Generating APG YAML Parameters.")
        if override and override.strip(): return override
        try:
            style_prompt = self.PROMPT_APG_STYLE_CHOICE.format(initial_concept_prompt=initial_concept_prompt, genre_tags=genre_tags)
            style_choice = self._call_llm_api(llm_backend, api_url, model_name, style_prompt)
            logger.info(f"APG target type: {style_choice}")

            if "audio" in style_choice.lower():
                yaml_config = yaml.safe_load(self.default_apg_yaml_audio)
                prompt = self.PROMPT_APG_AUDIO.format(initial_concept_prompt=initial_concept_prompt, genre_tags=genre_tags)
                raw_response = self._call_llm_api(llm_backend, api_url, model_name, prompt)
                if match := re.search(r'(\d+\.?\d*)', raw_response):
                    apg_scale = max(2.0, min(6.0, float(match.group(1))))
                    logger.info(f"APG Audio scale: {apg_scale:.2f}")
                    if 'rules' in yaml_config and len(yaml_config['rules']) >= 3:
                        yaml_config['rules'][1]['apg_scale'] = apg_scale
                        yaml_config['rules'][2]['apg_scale'] = max(1.0, apg_scale - 1.5)
            else:
                yaml_config = yaml.safe_load(self.default_apg_yaml_image)
                prompt = self.PROMPT_APG_IMAGE.format(initial_concept_prompt=initial_concept_prompt)
                raw_response = self._call_llm_api(llm_backend, api_url, model_name, prompt)
                values = [v.strip() for v in raw_response.split(',')]
                if len(values) >= 2:
                    apg_scale = max(self.APG_MIN_CFG, min(self.APG_MAX_CFG, float(values[0])))
                    momentum = max(-0.95, min(0.95, float(values[1])))
                    logger.info(f"APG Image scale: {apg_scale:.2f}, momentum: {momentum:.2f}")
                    if 'rules' in yaml_config and len(yaml_config['rules']) > 1:
                        yaml_config['rules'][1]['apg_scale'] = apg_scale
                        yaml_config['rules'][1]['momentum'] = momentum
            return yaml.dump(yaml_config, Dumper=CustomDumper, indent=2, sort_keys=False)
        except (ValueError, RuntimeError, yaml.YAMLError) as e:
            logger.warning(f"APG YAML generation failed: {e}. Using audio default.")
            return self.default_apg_yaml_audio

    def _configure_sampler_yaml(self, llm_backend: str, api_url: str, model_name: str, initial_concept_prompt: str, genre_tags: str, sampler_version: str, override: str) -> str:
        """Generates Sampler YAML parameters."""
        logger.info(f"Stage 6: Generating Sampler YAML for '{sampler_version}'.")
        if override and override.strip(): return override
        try:
            if sampler_version == "FBG Integrated PingPong Sampler":
                prompt = self.PROMPT_SAMPLER_FBG.format(initial_concept_prompt=initial_concept_prompt, genre_tags=genre_tags)
                raw_response = self._call_llm_api(llm_backend, api_url, model_name, prompt)
                values = [v.strip() for v in raw_response.split(',')]
                if len(values) >= 3:
                    t_0 = max(0.7, min(0.9, float(values[0])))
                    cfg_scale = max(500, min(800, int(float(values[1]))))
                    max_guidance_scale = max(800, min(1200, int(float(values[2]))))
                    logger.info(f"FBG params: t_0={t_0:.2f}, cfg_scale={cfg_scale}, max_guidance={max_guidance_scale}")
                    
                    yaml_config = yaml.safe_load(self.default_sampler_yaml_fbg)
                    yaml_config['fbg_config'].update({
                        't_0': t_0, 't_1': max(0.2, t_0 - 0.4), 'cfg_scale': cfg_scale, 
                        'initial_guidance_scale': int(cfg_scale * 0.4), 
                        'max_guidance_scale': max_guidance_scale
                    })
                    return yaml.dump(yaml_config, Dumper=CustomDumper, indent=2, sort_keys=False)
                else:
                    logger.warning("Insufficient FBG values from LLM. Using default.")
                    return self.default_sampler_yaml_fbg
            elif sampler_version == "Original PingPong Sampler":
                prompt = self.PROMPT_SAMPLER_ORIGINAL.format(initial_concept_prompt=initial_concept_prompt)
                behavior_choice = self._call_llm_api(llm_backend, api_url, model_name, prompt).strip()
                logger.info(f"Original Sampler behavior: {behavior_choice}")
                valid_behaviors = ["Default (Raw)", "Dynamic", "Smooth", "Textured Grain", "Soft (DDIM-Like)"]
                if behavior_choice not in valid_behaviors:
                    logger.warning(f"Invalid behavior '{behavior_choice}'. Using 'Custom'.")
                    behavior_choice = "Custom"
                yaml_config = yaml.safe_load(self.default_sampler_yaml_original)
                yaml_config['noise_behavior'] = behavior_choice
                return yaml.dump(yaml_config, Dumper=CustomDumper, indent=2, sort_keys=False)
        except (ValueError, RuntimeError, yaml.YAMLError) as e:
            logger.warning(f"Sampler YAML generation failed: {e}. Using default.")
        
        return self.default_sampler_yaml_fbg if sampler_version == "FBG Integrated PingPong Sampler" else self.default_sampler_yaml_original

    def _configure_noise_decay_yaml(self, llm_backend: str, api_url: str, model_name: str, initial_concept_prompt: str) -> str:
        """Generates Noise Decay Scheduler YAML parameters."""
        logger.info("Stage 7: Generating Noise Decay Scheduler YAML.")
        try:
            prompt = self.PROMPT_NOISE_DECAY_ALGORITHM.format(initial_concept_prompt=initial_concept_prompt)
            algo_choice = self._call_llm_api(llm_backend, api_url, model_name, prompt).strip().lower()
            logger.info(f"Noise Decay algorithm: {algo_choice}")
            valid_algorithms = ["polynomial", "sigmoidal", "gaussian", "fourier", "exponential"]
            if algo_choice not in valid_algorithms:
                logger.warning(f"Invalid algorithm '{algo_choice}'. Using 'polynomial'.")
                algo_choice = "polynomial"
            
            yaml_config = yaml.safe_load(self.default_noise_decay_yaml)
            yaml_config['algorithm_type'] = algo_choice
            
            if algo_choice in ["polynomial", "sigmoidal", "exponential", "gaussian"]:
                param_prompt = self.PROMPT_NOISE_DECAY_EXPONENT.format(algo_choice=algo_choice, initial_concept_prompt=initial_concept_prompt)
                raw_response = self._call_llm_api(llm_backend, api_url, model_name, param_prompt)
                if match := re.search(r'(\d+\.?\d*)', raw_response):
                    exponent = max(1.0, min(8.0, float(match.group(1))))
                    yaml_config['decay_exponent'] = exponent
                    logger.info(f"Decay exponent: {exponent:.1f}")
            return yaml.dump(yaml_config, Dumper=CustomDumper, indent=2, sort_keys=False)
        except (ValueError, RuntimeError, yaml.YAMLError) as e:
            logger.warning(f"Noise Decay YAML generation failed: {e}. Using default.")
            return self.default_noise_decay_yaml

    def _get_fallback_outputs(self, seed: int, total_steps: int) -> Tuple[str, str, float, int, int, int, float, float, str, str, str, str, int]:
        """Returns safe default outputs when a critical generation stage fails."""
        logger.warning("Returning fallback outputs due to a critical failure.")
        return ("ambient", "[instrumental]", 120.0, total_steps, 2, 1, 0.5, 5.0, "", self.default_apg_yaml_audio, self.default_sampler_yaml_fbg, self.default_noise_decay_yaml, seed)

    def execute(self, **kwargs: Any) -> Tuple[str, str, float, int, int, int, float, float, str, str, str, str, int]:
        """Main execution method for the node."""
        logger.info("--- Starting SceneGeniusAutocreator v1.0.1 ---")
        
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
        randomize_seed = kwargs.get("randomize_seed", True)
        sampler_version = kwargs.get("sampler_version", "FBG Integrated PingPong Sampler")
        generate_noise_decay_yaml = kwargs.get("generate_noise_decay_yaml", False)

        # --- Seed and Step Logic ---
        if randomize_seed:
            seed = random.randint(0, 0xffffffffffffffff)
            logger.info(f"Generated random seed: {seed}")
        
        total_steps = manual_steps if quality_preset == "Manual" else self.QUALITY_PRESETS.get(quality_preset, 220)
        logger.info(f"Quality preset '{quality_preset}' set to {total_steps} steps")

        # --- Backend and Override Setup ---
        prompt_overrides = {k: v for k, v in kwargs.items() if k.startswith('prompt_')}
        test_mode = kwargs.get('test_mode', False)
        genre_input = kwargs.get('genre_input', "")
        genre_mixed_mode = kwargs.get('genre_mixed_mode', False)
        
        api_url, model_name = (ollama_api_base_url, ollama_model_name) if llm_backend == "ollama" else (lm_studio_api_base_url, lm_studio_model_name)
        
        if test_mode:
            logger.info("Test mode enabled. Returning dummy output.")
            return ("electro, synthwave", "[Verse 1]\n...", 180.0, total_steps, int(base_tag_boost), int(vocal_tag_boost), vocal_weight, 3.5, "ethereal vocals", self.default_apg_yaml_audio, self.default_sampler_yaml_fbg, self.default_noise_decay_yaml, seed)
        
        try:
            # --- Generation Pipeline ---
            raw_genre_tags = self._generate_genres(llm_backend, api_url, model_name, initial_concept_prompt, tag_count, excluded_tags, genre_input, genre_mixed_mode)
            lyrics_or_script = self._generate_lyrics(llm_backend, api_url, model_name, initial_concept_prompt, raw_genre_tags, force_lyrics_generation, force_instrumental, prompt_overrides.get("prompt_lyrics_decision_and_generation", ""))
            
            is_instrumental = False
            if force_instrumental:
                logger.info("Instrumental mode: Forced by user flag.")
                is_instrumental = True
            elif lyrics_or_script.strip().lower() == '[instrumental]':
                logger.info("Instrumental mode: Detected from LLM output.")
                is_instrumental = True

            vocal_override = prompt_overrides.get("prompt_vocal_tags_generation", "")
            if is_instrumental:
                final_vocal_tags, remaining_genre_tags = "", raw_genre_tags
            elif vocal_override and vocal_override.strip():
                final_vocal_tags = vocal_override
                _, remaining_genre_tags = self._extract_vocal_tags(raw_genre_tags)
                logger.info(f"Using user-provided vocal override: '{final_vocal_tags}'")
            else:
                final_vocal_tags, remaining_genre_tags = self._extract_vocal_tags(raw_genre_tags)
            
            total_seconds = self._generate_duration(llm_backend, api_url, model_name, initial_concept_prompt, raw_genre_tags, lyrics_or_script, min_total_seconds, max_total_seconds, prompt_overrides.get("prompt_duration_generation", ""))
            noise_decay_strength = self._generate_noise_decay_strength(llm_backend, api_url, model_name, initial_concept_prompt, raw_genre_tags, lyrics_or_script, total_seconds, prompt_overrides.get("prompt_noise_decay_generation", ""))
            apg_yaml_params = self._configure_apg_yaml(llm_backend, api_url, model_name, initial_concept_prompt, raw_genre_tags, prompt_overrides.get("prompt_apg_yaml_generation", ""))
            sampler_yaml_params = self._configure_sampler_yaml(llm_backend, api_url, model_name, initial_concept_prompt, raw_genre_tags, sampler_version, prompt_overrides.get("prompt_sampler_yaml_generation", ""))
            noise_decay_yaml_params = self._configure_noise_decay_yaml(llm_backend, api_url, model_name, initial_concept_prompt) if generate_noise_decay_yaml else self.default_noise_decay_yaml
            
            logger.info("--- SceneGeniusAutocreator execution complete ---")
            
            return (remaining_genre_tags, lyrics_or_script, total_seconds, total_steps, int(base_tag_boost), int(vocal_tag_boost), vocal_weight, noise_decay_strength, final_vocal_tags, apg_yaml_params, sampler_yaml_params, noise_decay_yaml_params, seed)

        except (ValueError, RuntimeError) as e:
            logger.error(f"A critical failure occurred during generation: {e}")
            return self._get_fallback_outputs(seed, total_steps)

# Node Mapping
NODE_CLASS_MAPPINGS = {
    "SceneGeniusAutocreator": SceneGeniusAutocreator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SceneGeniusAutocreator": "Scene Genius Autocreator"
}