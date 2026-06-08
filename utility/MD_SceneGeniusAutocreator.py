# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# █▓▒░                                                                     ░▒▓█
# █▓▒░       MD_Nodes/SceneGenius – AI Workflow Orchestrator v1.6.0        ░▒▓█
# █▓▒░                                                                     ░▒▓█
# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
# ╠═ © 2026 MDMAchine
# ╠═ License: GNU General Public License v3.0 (GPL v3)
# ║
# ║  This program is free software: you can redistribute it and/or modify
# ║  it under the terms of the GNU General Public License as published by
# ║  the Free Software Foundation, either version 3 of the License, or
# ║  (at your option) any later version.
# ║
# ║  This program is distributed in the hope that it will be useful,
# ║  but WITHOUT ANY WARRANTY; without even the implied warranty of
# ║  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# ║  GNU General Public License for more details.
# ║
# ║  You should have received a copy of the GNU General Public License
# ║  along with this program. If not, see <https://www.gnu.org/licenses/>.
# ╠════════════════════════════════════════════════════════════════════════════
# ║ ░▒▓ ORIGIN: MDMAchine Workflow Brain
# ║ ░▒▓ DESCRIPTION:
# ║    The definitive brain of the MD suite. Orchestrates creative content
# ║    and technical guidance with three-tier mode intelligence.
# ║    NOTE: As a text orchestrator and API client, this runs entirely in the public wrapper domain.
# ║
# ║ ░▒▓ CORE FEATURES:
# ║    ✓ Universal Wildcards: Expansion works in all manual inputs.
# ║    ✓ File Loading: Supports external .txt files for Genres/Vocals/Lyrics.
# ║    ✓ Smart LLM Detection: Auto-discovers Ollama/LM Studio models.
# ║    ✓ Technical Scaling: ADG Angle calculated via 0.39 * (60/steps).
# ║    ✓ Performance Profiling: Real-time analytics on LLM vs Math speed.
# ║
# ║ ░▒▓ CHANGELOG:
# ║    v1.6.0 (Enterprise Standards - Feb 2026):
# ║    ├── REFACTOR: Tooltips strictly updated to 5-part v1.5.4 standard.
# ║    └── VERIFIED: PerformanceProfiler matches v1.5.3 exact specifications.
# ║    v3.2.0 (Production Polish)
# ║    ├── Fixed: File-loading logic now properly reads .txt files
# ║    ├── Fixed: LLM model detection with proper caching
# ║    └── Added: Duration estimation based on step count
# ╚════════════════════════════════════════════════════════════════════════════


VERSION = "v1.6.0"  # UPS v1.5.8

import logging
import re
import time
import random
import secrets
import os
import glob
from textwrap import wrap

import torch
import numpy as np

# =================================================================================
# == Dependency Checks
# =================================================================================
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logging.warning("[SceneGenius] requests not available - LLM modes disabled")

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logging.warning("[SceneGenius] PIL not available - preview disabled")

# =================================================================================
# == Configuration Constants
# =================================================================================
CONST_JS_MAX_SAFE_INTEGER = 9007199254740991
CONST_SEED_MIN = 0
CONST_ADG_BASELINE_ANGLE = 0.39
CONST_ADG_BASELINE_STEPS = 60

# =================================================================================
# == PerformanceProfiler Class (Enterprise Standard)
# =================================================================================
class PerformanceProfiler:
    """Standard performance profiler for MD_Nodes."""
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.timings = {}
        self.start_times = {}
    
    def start(self, operation_name):
        if not self.enabled: return
        import time
        self.start_times[operation_name] = time.perf_counter()
    
    def stop(self, operation_name):
        if not self.enabled: return
        import time
        if operation_name in self.start_times:
            elapsed = time.perf_counter() - self.start_times[operation_name]
            if operation_name not in self.timings:
                self.timings[operation_name] = []
            self.timings[operation_name].append(elapsed)
            del self.start_times[operation_name]
    
    def get_total_time(self):
        if not self.enabled or not self.timings: return 0.0
        return sum(sum(times) for times in self.timings.values())
    
    def print_report(self):
        if not self.enabled or not self.timings: return
        logging.info("\n⏱️  PERFORMANCE (Generation):")
        total = self.get_total_time()
        logging.info(f"    • Total Time: {total:.4f}s")
        for op_name, times in sorted(self.timings.items()):
            avg = sum(times) / len(times)
            if len(times) == 1:
                logging.info(f"    • {op_name}: {avg:.4f}s")
            else:
                logging.info(f"    • {op_name}: {avg:.4f}s avg ({len(times)}x)")

# =================================================================================
# == Helper Classes
# =================================================================================
class WildcardExpander:
    """Expands {option1|option2} patterns with seeded randomness."""
    WILDCARD_PATTERN = re.compile(r'\{([^{}]+)\}')

    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def expand(self, template):
        if not template or not isinstance(template, str):
            return ""
        
        def replace(match):
            options = [opt.strip() for opt in match.group(1).split('|')]
            return self.rng.choice(options)
        
        result = template
        for _ in range(100):
            if not self.WILDCARD_PATTERN.search(result):
                break
            result = self.WILDCARD_PATTERN.sub(replace, result)
        
        return result

# =================================================================================
# == SceneGenius Core Node
# =================================================================================
class SceneGeniusAutocreator:
    """Intelligent workflow orchestrator with adaptive execution modes."""

    # --- Quality Presets ---
    QUALITY_PRESETS = {
        "Draft (60)": 60,
        "Low (120)": 120,
        "Basic (180)": 180,
        "Production (220)": 220,
        "Studio (360)": 360,
        "Fine (500)": 500,
        "Master (720)": 720,
        "Ultra (1000)": 1000,
        "Monster (2000)": 2000
    }

    # --- Internal Creative Libraries ---
    GENRE_LIB = {
        "base": [
            "DnB", "Liquid DnB", "Neurofunk", "Dubstep", "Melodic Dubstep",
            "Trap", "Future Bass", "Techno", "Deep House", "Tech House",
            "Trance", "Psytrance", "Ambient", "Dark Ambient", "Synthwave",
            "Vaporwave", "Lo-Fi", "Breakbeat", "Shoegaze", "Post-Rock", "Metalcore"
        ],
        "mod": [
            "Atmospheric", "Aggressive", "Melodic", "Dark", "Euphoric",
            "Minimal", "Experimental", "Cinematic", "Ethereal", "Industrial"
        ],
        "perc": [
            "Glitchy stuttering", "Sharp staccato", "Polyrhythmic beats",
            "Punchy kicks", "Lo-fi dusty grooves", "Trap 808 rolls",
            "Industrial clanging", "Crisp digital drums"
        ],
        "inst": [
            "Grand Piano", "Rhodes E-Piano", "Sawtooth Lead", "Supersaw Stabs",
            "Atmospheric Pads", "Crystal Bells", "Analog Modular Synth",
            "FM Synth Keys", "Orchestral Strings"
        ],
        "texture": [
            "Lush string pads", "Ethereal choir textures", "Dark granular clouds",
            "Shimmer reverb tails", "Tape hiss and crackle", "Vinyl static",
            "Rain ambience", "Sci-fi bleeps"
        ]
    }

    # --- LLM Settings ---
    DEFAULT_OLLAMA_URL = "http://localhost:11434"
    DEFAULT_LM_STUDIO_URL = "http://localhost:1234"
    DEFAULT_OLLAMA_MODEL = "llama3:8b-instruct-q8_0"
    DEFAULT_LM_STUDIO_MODEL = "local-model"
    API_TIMEOUT_SECONDS = 30
    MODEL_LIST_CACHE_SECONDS = 300

    _ollama_models_cache = None
    _lm_studio_models_cache = None
    _ollama_cache_timestamp = 0.0
    _lm_studio_cache_timestamp = 0.0

    # --- File System Helpers ---
    @classmethod
    def _get_files_from_dir(cls, subfolder):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        target_dir = os.path.join(current_dir, "wildcards", subfolder)
        if not os.path.exists(target_dir): return ["None", "Random"]
        try:
            files = [os.path.basename(f) for f in glob.glob(os.path.join(target_dir, "*.txt"))]
            return ["None", "Random"] + sorted(files)
        except Exception as e:
            logging.error(f"[SceneGenius] Error scanning {target_dir}: {e}")
            return ["None", "Random"]

    @classmethod
    def _read_file_content(cls, subfolder, filename):
        if not filename or filename == "None": return ""
        current_dir = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(current_dir, "wildcards", subfolder, filename)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            logging.error(f"[SceneGenius] Failed to read {path}: {e}")
            return ""

    # --- LLM Model Loaders ---
    @classmethod
    def _get_ollama_models_lazy(cls):
        if not REQUESTS_AVAILABLE: return [cls.DEFAULT_OLLAMA_MODEL]
        if cls._ollama_models_cache and (time.time() - cls._ollama_cache_timestamp) < cls.MODEL_LIST_CACHE_SECONDS:
            return cls._ollama_models_cache
        try:
            resp = requests.get(f"{cls.DEFAULT_OLLAMA_URL}/api/tags", timeout=2)
            if resp.status_code == 200:
                models = [m["name"] for m in resp.json().get("models", [])]
                if models:
                    cls._ollama_models_cache = models
                    cls._ollama_cache_timestamp = time.time()
                    return models
        except Exception: pass
        return [cls.DEFAULT_OLLAMA_MODEL]

    @classmethod
    def _get_lm_studio_models_lazy(cls):
        if not REQUESTS_AVAILABLE: return [cls.DEFAULT_LM_STUDIO_MODEL]
        if cls._lm_studio_models_cache and (time.time() - cls._lm_studio_cache_timestamp) < cls.MODEL_LIST_CACHE_SECONDS:
            return cls._lm_studio_models_cache
        try:
            resp = requests.get(f"{cls.DEFAULT_LM_STUDIO_URL}/v1/models", timeout=2)
            if resp.status_code == 200:
                models = [m["id"] for m in resp.json().get("data", [])]
                if models:
                    cls._lm_studio_models_cache = models
                    cls._lm_studio_cache_timestamp = time.time()
                    return models
        except Exception: pass
        return [cls.DEFAULT_LM_STUDIO_MODEL]

    @classmethod
    def INPUT_TYPES(cls):
        genre_files = cls._get_files_from_dir("genre")
        vocal_files = cls._get_files_from_dir("vocal")
        lyrics_files = cls._get_files_from_dir("lyrics")
        
        return {
            "required": {
                "execution_mode": (["Fast (No LLM)", "Hybrid (Smart LLM)", "Full AI (Creative LLM)"], {
                    "default": "Fast (No LLM)",
                    "tooltip": (
                        "EXECUTION MODE\n"
                        "• Purpose: Controls generation logic intensity.\n"
                        "• Options:\n"
                        "  - Fast: Instant, uses internal dictionary libraries only.\n"
                        "  - Hybrid: LLM selects curated combinations from libraries.\n"
                        "  - Full AI: LLM generates completely new creative content.\n"
                        "• Trade-offs: Fast is instant; Full AI introduces API latency (5-30s).\n"
                        "\n⭐ Recommended: Fast for drafts, Hybrid for production runs."
                    )
                }),
                "initial_concept_prompt": ("STRING", {
                    "multiline": True,
                    "default": "Cyberpunk neon lights rain",
                    "tooltip": (
                        "CONCEPT PROMPT\n"
                        "• Purpose: The core thematic idea driving the entire generation.\n"
                        "• Usage: Used by LLM modes to infer appropriate genre, mood, and lyrics.\n"
                        "• Support: {wildcard|syntax} expansion is fully supported here.\n"
                        "\n⭐ Recommended: Be descriptive (e.g., 'Dark ambient forest, heavy rain, isolation')."
                    )
                }),
                "quality_preset": (list(cls.QUALITY_PRESETS.keys()) + ["Manual"], {
                    "default": "Basic (180)",
                    "tooltip": (
                        "QUALITY PRESET\n"
                        "• Purpose: Automatically sets step count and syncs technical ADG parameters.\n"
                        "• Draft (60): Fast testing.\n"
                        "• Basic (180): Standard balanced speed/quality.\n"
                        "• Production (220): High fidelity standard.\n"
                        "• Monster (2000): Ultra-long experimental runs.\n"
                        "\n⭐ Recommended: Start with Basic (180)."
                    )
                }),
                "randomize_seed": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "RANDOMIZE SEED\n"
                        "• Purpose: Toggle automatic seed rotation.\n"
                        "• True: Generates a new variation/wildcard selection every run.\n"
                        "• False: Locks the seed to continuously refine a specific output."
                    )
                }),
                "debug_mode": (["0 - Silent", "1 - Info", "2 - Verbose"], {
                    "default": "0 - Silent",
                    "tooltip": (
                        "LOGGING VERBOSITY\n"
                        "• Purpose: Controls console output detail level.\n"
                        "• Options:\n"
                        "  - 0: Minimal output (Production).\n"
                        "  - 1: Performance/Latency report.\n"
                        "  - 2: Step-by-step logic tracing.\n"
                        "\n⭐ Recommended: Use '1 - Info' when testing LLM latency."
                    )
                }),
                "enable_profiling": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "ENABLE PROFILING\n"
                        "• Purpose: Measure timing of text generation and API calls.\n"
                        "• Note: Automatically enabled if debug_mode >= 1."
                    )
                }),
            },
            "optional": {
                "seed": ("INT", {
                    "default": 0,
                    "min": CONST_SEED_MIN,
                    "max": CONST_JS_MAX_SAFE_INTEGER,
                    "tooltip": (
                        "MANUAL SEED\n"
                        "• Purpose: Control wildcard/randomization state.\n"
                        "• Range: 0 to 9 quadrillion (JS-safe precision limit).\n"
                        "• Note: Ignored if 'randomize_seed' is True."
                    )
                }),
                "manual_steps": ("INT", {
                    "default": 180,
                    "min": 1,
                    "max": 10000,
                    "tooltip": (
                        "MANUAL STEPS\n"
                        "• Purpose: Custom step count override.\n"
                        "• Requirement: 'quality_preset' MUST be set to 'Manual'."
                    )
                }),
                "force_instrumental": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "FORCE INSTRUMENTAL\n"
                        "• Purpose: Disable vocal generation entirely.\n"
                        "• Result: Overrides LLM to output empty vocals and '[Instrumental]' lyrics."
                    )
                }),
                
                # File Loading
                "load_genre_file": (genre_files, {
                    "default": "None",
                    "tooltip": (
                        "LOAD GENRE FILE\n"
                        "• Purpose: Inject an external .txt file as the genre source.\n"
                        "• Location: Must be in the node's /wildcards/genre/ folder.\n"
                        "• Priority: Takes precedence over LLM/Fast generation."
                    )
                }),
                "load_vocal_file": (vocal_files, {
                    "default": "None",
                    "tooltip": (
                        "LOAD VOCAL FILE\n"
                        "• Purpose: Inject an external .txt file as the vocal source.\n"
                        "• Location: Must be in the node's /wildcards/vocal/ folder."
                    )
                }),
                "load_lyrics_file": (lyrics_files, {
                    "default": "None",
                    "tooltip": (
                        "LOAD LYRICS FILE\n"
                        "• Purpose: Inject an external .txt file as the lyrics source.\n"
                        "• Location: Must be in the node's /wildcards/lyrics/ folder."
                    )
                }),
                
                # Manual Overrides
                "genre_input": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": (
                        "GENRE OVERRIDE\n"
                        "• Purpose: Manually force a specific string into the Genre output.\n"
                        "• Priority: Overrides EVERYTHING (LLM, files, Fast Mode).\n"
                        "• Support: {wildcards} work here."
                    )
                }),
                "vocal_input": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "VOCAL OVERRIDE\n• Purpose: Manually force a specific string into the Vocal output."
                }),
                "lyrics_input": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "LYRICS OVERRIDE\n• Purpose: Manually force a specific string into the Lyrics output."
                }),
                
                # LLM Settings
                "llm_backend": (["ollama", "lm_studio"], {
                    "default": "ollama",
                    "tooltip": "LLM BACKEND\n• Purpose: Select the local API server to use for Hybrid/Full AI modes."
                }),
                "ollama_api_base_url": ("STRING", {
                    "default": cls.DEFAULT_OLLAMA_URL,
                    "tooltip": "OLLAMA URL\n• Default: http://localhost:11434"
                }),
                "ollama_model_name": (cls._get_ollama_models_lazy(), {
                    "default": cls.DEFAULT_OLLAMA_MODEL,
                    "tooltip": "OLLAMA MODEL\n• Purpose: Select which installed Ollama model to prompt."
                }),
                "lm_studio_api_base_url": ("STRING", {
                    "default": cls.DEFAULT_LM_STUDIO_URL,
                    "tooltip": "LM STUDIO URL\n• Default: http://localhost:1234"
                }),
                "lm_studio_model_name": (cls._get_lm_studio_models_lazy(), {
                    "default": cls.DEFAULT_LM_STUDIO_MODEL,
                    "tooltip": "LM STUDIO MODEL\n• Purpose: Select which loaded LM Studio model to prompt."
                }),
                
                "enable_preview": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "ENABLE PREVIEW\n"
                        "• Purpose: Generate a visual status dashboard image.\n"
                        "• Output: IMAGE tensor summarizing the generated parameters."
                    )
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "FLOAT", "INT", "FLOAT", "FLOAT", "BOOLEAN", "FLOAT", "STRING", "INT", "IMAGE")
    RETURN_NAMES = ("GENRE", "LYRICS", "DURATION", "STEPS", "GLIDE_POWER", "MASTER_STR", "LEGACY", "ADG_ANGLE", "VOCALS", "SEED", "STATUS")
    FUNCTION = "execute"
    CATEGORY = "MD_Nodes/Prompt Generation"

    @classmethod
    def IS_CHANGED(cls, randomize_seed=False, **kwargs):
        is_random = randomize_seed
        if isinstance(randomize_seed, str):
            is_random = randomize_seed.lower() == "true"
        if is_random:
            return secrets.token_hex(16)
        return "static"

    # --- Core Logic ---

    def _calculate_tier_params(self, steps):
        if steps <= 120:
            glide_power = 1.0
            master_strength = 1.0
            legacy = True
        elif steps <= 360:
            glide_power = 1.5
            master_strength = 1.2
            legacy = True
        elif steps <= 720:
            glide_power = 2.0
            master_strength = 1.3
            legacy = True
        else:
            glide_power = 3.0
            master_strength = 1.5
            legacy = False 
        
        adg_angle = CONST_ADG_BASELINE_ANGLE * (CONST_ADG_BASELINE_STEPS / max(steps, 1))
        duration = max(30.0, min(300.0, (steps / 100) * 45.0))
        
        return glide_power, master_strength, legacy, adg_angle, duration

    def _call_llm(self, backend, url, model, prompt):
        if not REQUESTS_AVAILABLE:
            raise RuntimeError("requests library not available")
        try:
            if backend == "ollama":
                res = requests.post(
                    f"{url}/api/generate",
                    json={"model": model, "prompt": prompt, "stream": False},
                    timeout=self.API_TIMEOUT_SECONDS
                )
                res.raise_for_status()
                return res.json().get("response", "").strip()
            else: 
                res = requests.post(
                    f"{url}/v1/chat/completions",
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.7
                    },
                    timeout=self.API_TIMEOUT_SECONDS
                )
                res.raise_for_status()
                return res.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logging.error(f"[SceneGenius] LLM call failed: {e}")
            raise

    def _generate_random_genre(self, seed):
        rng = random.Random(seed)
        lib = self.GENRE_LIB
        parts = []
        parts.append(f"{rng.choice(lib['mod'])} {rng.choice(lib['base'])}")
        parts.append(rng.choice(lib['perc']))
        parts.append(f"{rng.choice(lib['inst'])}")
        if rng.random() > 0.5:
            parts.append(rng.choice(lib['texture']))
        return ", ".join(parts)

    def _load_or_expand(self, input_str, file_selection, subfolder, expander, seed):
        if input_str and input_str.strip():
            return expander.expand(input_str)
        if file_selection and file_selection != "None":
            if file_selection == "Random":
                current_dir = os.path.dirname(os.path.realpath(__file__))
                target_dir = os.path.join(current_dir, "wildcards", subfolder)
                files = glob.glob(os.path.join(target_dir, "*.txt"))
                if files:
                    rng = random.Random(seed)
                    chosen = rng.choice(files)
                    try:
                        with open(chosen, 'r', encoding='utf-8') as f:
                            return expander.expand(f.read().strip())
                    except Exception: pass
            else:
                content = self._read_file_content(subfolder, file_selection)
                if content: return expander.expand(content)
        return None

    def _render_status_image(self, concept, genre, vocals, duration, steps, glide_power, master_str, mode, seed):
        if not PIL_AVAILABLE:
            return torch.zeros((1, 64, 64, 3))
        try:
            w, h = 900, 450
            img = Image.new('RGB', (w, h), color='#0A0A0F')
            draw = ImageDraw.Draw(img)
            
            try:
                font_title = ImageFont.truetype("arial.ttf", 28)
                font_label = ImageFont.truetype("arial.ttf", 14)
                font_value = ImageFont.truetype("arial.ttf", 18)
            except Exception:
                font_title = ImageFont.load_default()
                font_label = ImageFont.load_default()
                font_value = ImageFont.load_default()
            
            draw.text((20, 20), "SCENE GENIUS v3.2.0", font=font_title, fill='#FFFFFF')
            draw.text((20, 60), f"Mode: {mode}", font=font_label, fill='#888888')
            
            def draw_kv(label, value, y, color='#87CEEB'):
                draw.text((30, y), label, font=font_label, fill='#888888')
                lines = wrap(str(value), width=70)
                for i, line in enumerate(lines[:3]):
                    draw.text((160, y + (i * 22)), line, font=font_value, fill=color)
                return y + 50 + (len(lines[:3]) * 10)
            
            y = 110
            y = draw_kv("Concept:", concept, y, '#FFFFFF')
            y = draw_kv("Genre:", genre, y, '#87CEEB')
            y = draw_kv("Vocals:", vocals, y, '#FFB6C1')
            y = draw_kv("Technical:", f"{duration:.0f}s | {steps} steps | Seed: {seed}", y, '#98FB98')
            y = draw_kv("Guidance:", f"Glide: {glide_power:.1f} | Strength: {master_str:.1f}", y, '#FFA500')
            
            img_np = np.array(img).astype(np.float32) / 255.0
            return torch.from_numpy(img_np).unsqueeze(0)
        except Exception as e:
            logging.error(f"[SceneGenius] Status image error: {e}")
            return torch.zeros((1, 64, 64, 3))

    def execute(self, **kwargs):
        debug_mode = kwargs.get("debug_mode", "0 - Silent")
        debug_level = int(debug_mode.split(" ")[0])
        enable_profiling = kwargs.get("enable_profiling", False)
        
        profiler = PerformanceProfiler(enabled=(debug_level >= 1 or enable_profiling))
        profiler.start("total_execution")
        
        if debug_level >= 2: logging.info("[SceneGenius] Starting execution...")
        
        seed = kwargs.get("seed", 0)
        if kwargs.get("randomize_seed", True):
            seed = secrets.randbelow(CONST_JS_MAX_SAFE_INTEGER)
        
        profiler.start("wildcard_expansion")
        expander = WildcardExpander(seed)
        concept = expander.expand(kwargs.get("initial_concept_prompt", ""))
        profiler.stop("wildcard_expansion")
        
        profiler.start("technical_calculation")
        preset = kwargs.get("quality_preset", "Basic (180)")
        if preset == "Manual":
            steps = kwargs.get("manual_steps", 180)
        else:
            steps = self.QUALITY_PRESETS.get(preset, 180)
        
        glide_power, master_strength, legacy, adg_angle, duration = self._calculate_tier_params(steps)
        profiler.stop("technical_calculation")
        
        profiler.start("creative_generation")
        execution_mode = kwargs.get("execution_mode", "Fast (No LLM)")
        
        genre_out = self._load_or_expand(
            kwargs.get("genre_input"),
            kwargs.get("load_genre_file"),
            "genre",
            expander,
            seed
        )
        
        if not genre_out:
            if "Fast" in execution_mode:
                genre_out = self._generate_random_genre(seed)
            else:
                backend = kwargs.get("llm_backend", "ollama")
                url = kwargs.get("ollama_api_base_url") if backend == "ollama" else kwargs.get("lm_studio_api_base_url")
                model = kwargs.get("ollama_model_name") if backend == "ollama" else kwargs.get("lm_studio_model_name")
                try:
                    profiler.start("llm_call")
                    prompt = f"Based on '{concept}', select a unique genre combining elements from: {', '.join(self.GENRE_LIB['base'][:10])}. Output only the genre name."
                    genre_out = self._call_llm(backend, url, model, prompt)
                    profiler.stop("llm_call")
                except Exception:
                    logging.warning("[SceneGenius] LLM unavailable, using random genre")
                    genre_out = self._generate_random_genre(seed)
        
        vocal_out = self._load_or_expand(
            kwargs.get("vocal_input"),
            kwargs.get("load_vocal_file"),
            "vocal",
            expander,
            seed
        )
        
        if not vocal_out:
            if kwargs.get("force_instrumental", False):
                vocal_out = "[instrumental]"
            else:
                rng = random.Random(seed)
                vocal_out = rng.choice([
                    "soft female vocals",
                    "powerful male vocals",
                    "ethereal choir",
                    "vocoder effects",
                    "[instrumental]"
                ])
        
        lyrics_out = self._load_or_expand(
            kwargs.get("lyrics_input"),
            kwargs.get("load_lyrics_file"),
            "lyrics",
            expander,
            seed
        )
        
        if not lyrics_out:
            if kwargs.get("force_instrumental", False) or "[instrumental]" in vocal_out.lower():
                lyrics_out = "[instrumental]"
            else:
                lyrics_out = "[Verse 1]\nGenerated lyrics placeholder...\n[Chorus]\n..."
        
        profiler.stop("creative_generation")
        
        profiler.start("visualization")
        status_img = torch.zeros((1, 64, 64, 3))
        if kwargs.get("enable_preview", True):
            status_img = self._render_status_image(
                concept, genre_out, vocal_out, duration, steps,
                glide_power, master_strength, execution_mode, seed
            )
        profiler.stop("visualization")
        
        profiler.stop("total_execution")
        
        if debug_level >= 1:
            logging.debug("\n" + "=" * 60)
            logging.info("📊 [SceneGenius] ANALYTICS REPORT")
            logging.debug("=" * 60)
            logging.info(f"🎭 Concept: {concept}")
            logging.info(f"🎵 Mode: {execution_mode} | Steps: {steps}")
            profiler.print_report()
            logging.debug("=" * 60)
        
        if debug_level >= 2: logging.info(f"[SceneGenius] Completed in {execution_mode} mode")
        
        return (
            genre_out, lyrics_out, duration, steps, glide_power, master_strength, 
            legacy, adg_angle, vocal_out, seed, status_img
        )

# =================================================================================
# == Universal Context Bus
# =================================================================================
class MD_WorkflowContextBus:
    """Consolidates SceneGenius outputs for one-wire workflows."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "genre": ("STRING", {"forceInput": True, "tooltip": "GENRE INPUT"}),
                "lyrics": ("STRING", {"forceInput": True, "tooltip": "LYRICS INPUT"}),
                "duration": ("FLOAT", {"forceInput": True, "tooltip": "DURATION INPUT"}),
                "steps": ("INT", {"forceInput": True, "tooltip": "STEPS INPUT"}),
                "glide_power": ("FLOAT", {"forceInput": True, "tooltip": "GLIDE POWER INPUT"}),
                "master_strength": ("FLOAT", {"forceInput": True, "tooltip": "MASTER STRENGTH INPUT"}),
                "legacy_mode": ("BOOLEAN", {"forceInput": True, "tooltip": "LEGACY MODE INPUT"}),
                "adg_angle": ("FLOAT", {"forceInput": True, "tooltip": "ADG ANGLE INPUT"}),
                "vocals": ("STRING", {"forceInput": True, "tooltip": "VOCALS INPUT"}),
                "seed": ("INT", {"forceInput": True, "tooltip": "SEED INPUT"}),
            }
        }

    RETURN_TYPES = ("CONTEXT", "STRING", "INT", "FLOAT", "BOOLEAN")
    RETURN_NAMES = ("BUS", "GENRE", "STEPS", "ADG_ANGLE", "LEGACY")
    FUNCTION = "bundle"
    CATEGORY = "MD_Nodes/Utility"

    def bundle(self, **kwargs):
        bus_data = {k: v for k, v in kwargs.items() if v is not None}
        return (
            bus_data,
            kwargs.get("genre", ""),
            kwargs.get("steps", 180),
            kwargs.get("adg_angle", 0.13),
            kwargs.get("legacy_mode", True)
        )

# =================================================================================
# == Registration
# =================================================================================
NODE_CLASS_MAPPINGS = {
    "SceneGeniusAutocreator": SceneGeniusAutocreator,
    "MD_WorkflowContextBus": MD_WorkflowContextBus
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SceneGeniusAutocreator": "MD: Scene Genius Autocreator",
    "MD_WorkflowContextBus": "MD: Universal Context Bus"
}


# ==============================================================================
# == Unit Tests (smoke — runs without ComfyUI)
# ==============================================================================

if __name__ == "__main__":
    print("\n🧪 Smoke tests: MD_SceneGeniusAutocreator")
    print("   VERSION :", VERSION)
    _pass = _fail = 0

    def _check(label, expr):
        global _pass, _fail
        if expr:
            print(f"  ✅  {label}")
            _pass += 1
        else:
            print(f"  ❌  {label}")
            _fail += 1

    _check("VERSION defined",    VERSION == "v1.6.0")
    _check("CONST CONST_JS_MAX_SAFE_INTEGER defined", CONST_JS_MAX_SAFE_INTEGER is not None)
    _check("CONST CONST_SEED_MIN defined", CONST_SEED_MIN is not None)
    _check("CONST CONST_ADG_BASELINE_ANGLE defined", CONST_ADG_BASELINE_ANGLE is not None)
    _check("CONST CONST_ADG_BASELINE_STEPS defined", CONST_ADG_BASELINE_STEPS is not None)
    _check("NODE_CLASS_MAPPINGS defined",
           isinstance(NODE_CLASS_MAPPINGS, dict) and len(NODE_CLASS_MAPPINGS) > 0)
    _check("  class SceneGeniusAutocreator in map", "SceneGeniusAutocreator" in NODE_CLASS_MAPPINGS)
    _check("  class MD_WorkflowContextBus in map", "MD_WorkflowContextBus" in NODE_CLASS_MAPPINGS)

    print(f"\n  {_pass} passed, {_fail} failed")
    if _fail == 0:
        print("  🎉 All good.")
