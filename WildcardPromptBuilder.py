# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
# ████ MD_Nodes/WildcardPromptBuilder – Ultimate Prompt Engine ████▓▒░
# © 2025 MDMAchine
# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
# ░▒▓ ORIGIN & DEV:
#   • Cast into the void by: MDMAchine
#   • Enhanced by: Claude & Gemini
#   • License: Apache 2.0 — "The factory must grow."

# ░▒▓ DESCRIPTION:
#   The ultimate prompt generation engine for Ace-Step and generic workflows.
#   Combines massive internal datasets, external file loading, and LLM 
#   hybrid logic to generate Genres, Vocals, Lyrics, and Duration strings.

# ░▒▓ FEATURES:
#   ✓ Massive Internal Library: 500+ options for genres, textures, and vocals.
#   ✓ File Loading: Automatically loads .txt wildcards from a 'wildcards' folder.
#   ✓ Duration Engine: Generates duration strings for automation nodes.
#   ✓ SceneGenius Ready: Outputs align perfectly with SceneGenius overrides.
#   ✓ Rich Text Viz: Renders the generated prompt as an image for easy reading.
#   ✓ Hybrid Mode: Uses LLM to intelligently select from the massive internal lists.

# ░▒▓ CHANGELOG:
#   - v1.2.1 (Compliance Polish):
#       • FIXED: Added tooltips to all inputs.
#       • FIXED: Robust IS_CHANGED widget validation.
#       • FIXED: Top-level error handling with graceful fallback.
#       • FIXED: Cross-platform font safety.
#       • FIXED: Added logging to hybrid vocal fallback.
#   - v1.2.0 (The "Omni" Update):
#       • EXPANDED: Massive expansion of Hybrid Genre/Vocal dictionaries.
#       • ADDED: Duration generation (template + output).
#       • ADDED: Local file loader for external wildcards (.txt files).

# ░▒▓ CONFIGURATION:
#   → Primary Use: Feeding SceneGeniusAutocreator with rich, randomized inputs.
#   → File Loading: Create a 'wildcards' folder next to this .py file.
#     Subfolders: 'genre', 'vocal', 'lyrics'. Place .txt files inside.

# ░▒▓ WARNING:
#   This node may trigger:
#   ▓▒░ Paralysis by analysis due to infinite combinations.
#   ▓▒░ An urge to categorize music genres that don't exist yet.
#   ▓▒░ "Text-based" hallucinations.
# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄

# =================================================================================
# == Standard Library Imports                                                    ==
# =================================================================================
import logging
import random
import re
import secrets
import time
import traceback
import io
import os
import glob
from urllib.parse import urlparse
from textwrap import wrap

# =================================================================================
# == Third-Party Imports                                                         ==
# =================================================================================
import requests
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
# Ensure backend is set to Agg to prevent GUI issues in threads
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# =================================================================================
# == Helper Classes & Dependencies                                               ==
# =================================================================================

class WildcardExpander:
    """
    Expands wildcard patterns like {option1|option2} with seeded randomness.
    """
    WILDCARD_PATTERN = re.compile(r'\{([^{}]+)\}')

    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.selections = []

    def expand(self, template):
        self.selections = []
        if not template:
            return ""

        def replace_wildcard(match):
            options = [opt.strip() for opt in match.group(1).split('|')]
            choice = self.rng.choice(options)
            self.selections.append(choice)
            return choice

        result = template
        max_iterations = 100
        iteration = 0

        while self.WILDCARD_PATTERN.search(result) and iteration < max_iterations:
            result = self.WILDCARD_PATTERN.sub(replace_wildcard, result)
            iteration += 1

        return result

# =================================================================================
# == Core Node Class                                                             ==
# =================================================================================

class WildcardPromptBuilder:
    """
    The ultimate multi-mode prompt generation node.
    """

    # --- Constants ---
    DEFAULT_OLLAMA_URL = "http://localhost:11434"
    DEFAULT_LM_STUDIO_URL = "http://localhost:1234"
    DEFAULT_OLLAMA_MODEL = "llama3:8b-instruct-q8_0"
    DEFAULT_LM_STUDIO_MODEL = "local-model"
    MAX_RETRY_ATTEMPTS = 3
    API_TIMEOUT_SECONDS = 120
    MODEL_LIST_CACHE_SECONDS = 300
    PREVIEW_WIDTH = 800
    PREVIEW_HEIGHT = 400

    # --- Caching ---
    _ollama_models_cache = None
    _lm_studio_models_cache = None
    _ollama_cache_timestamp = 0.0
    _lm_studio_cache_timestamp = 0.0

    # --- MASSIVE INTERNAL LIBRARY ---
    HYBRID_GENRE_OPTIONS = {
        "base_genre": [
            "Drum and Bass", "Liquid DnB", "Neurofunk", "Techstep", "Jump Up", "Breakcore", "Jungle", "Intelligent DnB",
            "Dubstep", "Deathstep", "Melodic Dubstep", "Riddim", "Trap", "Future Bass", "Wave", "Hardwave", "Phonk", "Drift Phonk",
            "Techno", "Hard Techno", "Acid Techno", "Dub Techno", "Minimal Techno", "Industrial Techno", "Peak Time Techno",
            "House", "Deep House", "Tech House", "Progressive House", "Bass House", "Lo-Fi House", "Acid House",
            "Trance", "Psytrance", "Goa Trance", "Progressive Trance", "Uplifting Trance",
            "Ambient", "Drone", "Dark Ambient", "Space Ambient", "Cinematic", "Downtempo", "Psybient", "Solar Fields Style",
            "Synthwave", "Darkwave", "Vaporwave", "Chillwave", "Outrun", "Cyberpunk", "EBM", "Industrial", "Aggrotech",
            "IDM", "Glitch", "Glitch Hop", "Breakbeat", "Big Beat", "Trip Hop",
            "Post-Rock", "Shoegaze", "Math Rock", "Midwest Emo", "Metalcore", "Djent", "Thall"
        ],
        "subgenre_modifier": [
            "Atmospheric", "Aggressive", "Melodic", "Dark", "Euphoric", "Minimal", "Experimental", "Lo-Fi", "Hi-Fi",
            "Symphonic", "Cinematic", "Industrial", "Organic", "Synthetic", "Tribal", "Acid", "Deep", "Hard", "Soft",
            "Ethereal", "Chaotic", "Structured", "Polyrhythmic", "Dissonant", "Harmonic", "Retro", "Futuristic", "Alien"
        ],
        "percussion_style": [
            "Glitchy stuttering percussion", "Sharp staccato drums", "Organic acoustic kit", "Heavily processed breaks",
            "Minimal sparse clicks", "Complex polyrhythmic beats", "Punchy hard-hitting kicks", "Industrial clanging metals",
            "Lo-fi dusty grooves", "Crisp digital drums", "Distorted gabber kicks", "Trap 808 rolls", "Amen break chops",
            "Tribal hand percussion", "Orchestral war drums", "Soft brushed snares", "Gated reverb 80s toms",
            "Granular texture beats", "Click-and-cut micro percussion", "Off-grid swing rhythms"
        ],
        "mood": [
            "Bitter", "Angry", "Vengeful", "Euphoric", "Lonely", "Hopeful", "Anxious", "Contemplative", "Aggressive",
            "Dreamy", "Nostalgic", "Haunting", "Triumphant", "Melancholic", "Chaotic", "Serene", "Ominous", "Romantic",
            "Cold", "Warm", "Detached", "Intimate", "Epic", "Gritty", "Sorrowful", "Blissful", "Hypnotic"
        ],
        "harmonic_instrument": [
            "Grand Piano", "Upright Felt Piano", "Rhodes E-Piano", "Wurlitzer", "Sawtooth Lead", "Square Wave Pluck",
            "Supersaw Stabs", "Atmospheric Pads", "Crystal Bells", "Kalimba", "Marimba", "Vibraphone", "Music Box",
            "Orchestral Strings", "Cello Solo", "Violin Section", "Brass Section", "French Horns", "Church Organ",
            "Hammond B3 Organ", "Electric Guitar Clean", "Electric Guitar Distorted", "Acoustic Guitar", "Harp",
            "Analog Modular Synth", "FM Synth Keys", "Wavetable Arps"
        ],
        "chord_quality": [
            "Minor", "Major", "Diminished", "Augmented", "Suspended 2", "Suspended 4", "Minor 7th", "Major 7th",
            "Dominant 7th", "Half-Diminished", "Modal", "Dorian", "Lydian", "Phrygian", "Cluster", "Atonal", "Open Voicing"
        ],
        "bass_character": [
            "Analog Moog Bass", "Distorted 808", "Clean Sub Bass", "Reese Bass", "Neuro Bass", "Wobble Bass",
            "Acid 303 Squell", "FM Slap Bass", "Upright Double Bass", "Fretless Bass", "Gritty Wavetable Bass",
            "Donk Bass", "Log Drum Bass", "Growl Bass", "Granular Bass textures"
        ],
        "texture": [
            "Lush string pads", "Atmospheric drone beds", "Ethereal choir textures", "Dark granular clouds",
            "Shimmer reverb tails", "Tape hiss and crackle", "Vinyl static", "Rain and thunder field recordings",
            "Cityscape ambience", "Forest nature sounds", "Sci-fi bleeps and bloops", "Radio static", "White noise sweeps",
            "Risers and downlifters", "Reverse piano swells", "Stretched vocal textures"
        ]
    }

    HYBRID_VOCAL_OPTIONS = {
        "vocal_type": [
            "Soft female vocals", "Deep male vocals", "Androgynous vocals", "Gritty male vocals", "Soprano female vocals",
            "Baritone male vocals", "Alto female vocals", "Tenor male vocals", "Child's voice", "Robotic vocoder vocals",
            "Processed vocal chops", "Ethereal choir", "Spoken word poetry", "Rap verse", "Whispered ASMR vocals",
            "Operatic aria", "Throat singing", "Death metal growls", "Black metal screeches", "Clean pop harmonies",
            "Gospel choir", "Gregorian monks chanting", "Soulful diva vocals", "Punk rock shouting"
        ],
        "vocal_quality": [
            "Breathy", "Ethereal", "Aggressive", "Soulful", "Melancholic", "Haunting", "Intimate", "Distant",
            "Processed", "Emotional", "Cold and detached", "Warm", "Powerful", "Raspy", "Smooth", "Clear", "Nasally",
            "Guttural", "Airy", "Strained", "Relaxed", "Urgent", "Languid", "Angelic", "Demonic"
        ],
        "vocal_effect": [
            "Reverb-drenched (Valhalla)", "Heavy ping-pong delay", "Hard Auto-Tune", "Pitch-shifted down", "Pitch-shifted up",
            "Stuttering glitch FX", "Telephone EQ filter", "Megaphone distortion", "Lo-fi bitcrushed", "Harmonizer stack",
            "Double-tracked", "Dry and upfront", "Gated reverb", "Vocoder harmonies", "Tape saturation", "Flanger sweep",
            "Chorus widening", "Granular clouds", "Time-stretched", "Reverse reverb swell", "Sidechain pumping", "Radio static overlay"
        ]
    }

    # --- Internal Default Templates (Massive) ---
    @classmethod
    def _generate_default_genre_template(cls):
        opts = cls.HYBRID_GENRE_OPTIONS
        def to_wc(key): return "{" + "|".join(opts[key]) + "}"
        return f"{to_wc('base_genre')}, {to_wc('subgenre_modifier')}, {to_wc('percussion_style')}, {to_wc('mood')} {to_wc('harmonic_instrument')} {to_wc('chord_quality')} chords, {to_wc('bass_character')}, {to_wc('texture')}"

    @classmethod
    def _generate_default_vocal_template(cls):
        opts = cls.HYBRID_VOCAL_OPTIONS
        def to_wc(key): return "{" + "|".join(opts[key]) + "}"
        return f"{to_wc('vocal_type')}, {to_wc('vocal_quality')}, {to_wc('vocal_effect')}"

    DEFAULT_LYRICS_TEMPLATE = "[Instrumental]"
    DEFAULT_DURATION_TEMPLATE = "{120|180|240}"

    # --- PROMPTS ---
    PROMPT_HYBRID_GENRE = """Based on the concept "{concept}", select the BEST option from each category.
CATEGORIES:
- base_genre: {base_genre_options}
- subgenre_modifier: {subgenre_options}
- percussion_style: {percussion_options}
- mood: {mood_options}
- harmonic_instrument: {instrument_options}
- chord_quality: {chord_options}
- bass_character: {bass_options}
- texture: {texture_options}

Respond in EXACT format:
base_genre=YOUR_CHOICE
subgenre_modifier=YOUR_CHOICE
percussion_style=YOUR_CHOICE
mood=YOUR_CHOICE
harmonic_instrument=YOUR_CHOICE
chord_quality=YOUR_CHOICE
bass_character=YOUR_CHOICE
texture=YOUR_CHOICE"""

    PROMPT_HYBRID_VOCAL = """Based on the concept "{concept}", select the BEST options.
CATEGORIES:
- vocal_type: {vocal_type_options}
- vocal_quality: {vocal_quality_options}
- vocal_effect: {vocal_effect_options}

Respond in EXACT format:
vocal_type=YOUR_CHOICE
vocal_quality=YOUR_CHOICE
vocal_effect=YOUR_CHOICE"""

    PROMPT_LLM_LYRICS = """Write lyrics based on: "{concept}".
Structure:
[Verse 1]
[Chorus]
[Verse 2]
[Bridge]
[Outro]
If concept suggests instrumental, output ONLY: [instrumental]"""

    def __init__(self):
        pass

    @classmethod
    def _get_files_from_dir(cls, subfolder):
        """Scans for .txt files in a subfolder relative to the node."""
        current_dir = os.path.dirname(os.path.realpath(__file__))
        target_dir = os.path.join(current_dir, "wildcards", subfolder)
        
        if not os.path.exists(target_dir):
            return ["None"]
            
        files = [os.path.basename(f) for f in glob.glob(os.path.join(target_dir, "*.txt"))]
        return ["None"] + sorted(files)

    @classmethod
    def INPUT_TYPES(cls):
        """Define inputs including dynamic file loaders."""
        # Note: This is evaluated at load time. If you add files, refresh ComfyUI.
        genre_files = cls._get_files_from_dir("genre")
        vocal_files = cls._get_files_from_dir("vocal")
        lyrics_files = cls._get_files_from_dir("lyrics")

        return {
            "required": {
                "generation_mode": (["wildcard", "llm", "hybrid"], {
                    "default": "wildcard",
                    "tooltip": "GENERATION MODE\n- Wildcard: Fast random expansion.\n- LLM: Full creative generation.\n- Hybrid: LLM selects from huge internal lists."
                }),
                "concept": ("STRING", {
                    "multiline": True, "default": "Deep space ambient, zero gravity, isolation",
                    "tooltip": "CONCEPT\n- The creative driver for LLM and Hybrid modes."
                }),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xffffffffffffffff,
                    "tooltip": "SEED\n- Controls random selection for reproducibility."
                }),
                "randomize_seed": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "RANDOMIZE SEED\n- If True, generates a fresh seed every run (Dynamic)."
                }),
                "duration_template": ("STRING", {
                    "default": "{120|180|240}", 
                    "tooltip": "DURATION TEMPLATE\n- Wildcard pattern for seconds (e.g., {60|120})."
                }),
            },
            "optional": {
                "llm_backend": (["ollama", "lm_studio"], {
                    "default": "ollama",
                    "tooltip": "LLM BACKEND\n- Choose between Ollama or LM Studio server."
                }),
                "ollama_api_url": ("STRING", {
                    "default": cls.DEFAULT_OLLAMA_URL,
                    "tooltip": "OLLAMA URL\n- API endpoint for Ollama."
                }),
                "ollama_model": (cls._get_ollama_models_lazy(), {
                    "default": cls.DEFAULT_OLLAMA_MODEL,
                    "tooltip": "OLLAMA MODEL\n- Select the model to use."
                }),
                "lm_studio_api_url": ("STRING", {
                    "default": cls.DEFAULT_LM_STUDIO_URL,
                    "tooltip": "LM STUDIO URL\n- API endpoint for LM Studio."
                }),
                "lm_studio_model": (cls._get_lm_studio_models_lazy(), {
                    "default": cls.DEFAULT_LM_STUDIO_MODEL,
                    "tooltip": "LM STUDIO MODEL\n- Select the model to use."
                }),
                
                # File Loaders
                "load_genre_file": (genre_files, {
                    "default": "None", 
                    "tooltip": "LOAD GENRE FILE\n- Select .txt file from wildcards/genre folder."
                }),
                "load_vocal_file": (vocal_files, {
                    "default": "None", 
                    "tooltip": "LOAD VOCAL FILE\n- Select .txt file from wildcards/vocal folder."
                }),
                "load_lyrics_file": (lyrics_files, {
                    "default": "None", 
                    "tooltip": "LOAD LYRICS FILE\n- Select .txt file from wildcards/lyrics folder."
                }),

                # Custom Overrides
                "custom_genre_template": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "CUSTOM GENRE\n- Manual wildcard template override."
                }),
                "custom_vocal_template": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "CUSTOM VOCAL\n- Manual wildcard template override."
                }),
                "custom_lyrics_template": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "CUSTOM LYRICS\n- Manual wildcard template override."
                }),
                
                # Toggles
                "generate_genre": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "GENERATE GENRE\n- Toggle genre generation on/off."
                }),
                "generate_vocals": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "GENERATE VOCALS\n- Toggle vocal generation on/off."
                }),
                "generate_lyrics": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "GENERATE LYRICS\n- Toggle lyrics generation on/off."
                }),
                "force_instrumental": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "FORCE INSTRUMENTAL\n- If True, clears vocals and sets lyrics to [Instrumental]."
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "INT", "IMAGE")
    RETURN_NAMES = ("GENRE_TAGS", "VOCAL_TAGS", "LYRICS", "DURATION_STRING", "SEED", "TEXT_PREVIEW")
    FUNCTION = "execute"
    CATEGORY = "MD_Nodes/Prompt Generation"

    @classmethod
    def IS_CHANGED(cls, randomize_seed=False, **kwargs):
        """
        Controls cache behavior with robust widget validation.
        """
        # Normalize widget value (handle cache corruption)
        is_randomize_true = False
        if isinstance(randomize_seed, bool):
            is_randomize_true = randomize_seed
        elif isinstance(randomize_seed, str):
            is_randomize_true = randomize_seed.lower() == "true"
        
        if is_randomize_true:
            return secrets.token_hex(16)
        return "static"

    # --- Lazy Loaders & Helpers ---
    @classmethod
    def _get_ollama_models_lazy(cls):
        """Lazy load Ollama models with logging."""
        if cls._ollama_models_cache and (time.time() - cls._ollama_cache_timestamp) < cls.MODEL_LIST_CACHE_SECONDS:
            return cls._ollama_models_cache
        try:
            resp = requests.get(f"{cls.DEFAULT_OLLAMA_URL}/api/tags", timeout=2)
            if resp.status_code == 200:
                cls._ollama_models_cache = [m["name"] for m in resp.json().get("models", [])]
                cls._ollama_cache_timestamp = time.time()
                return cls._ollama_models_cache
        except Exception as e:
            logging.debug(f"[WildcardPromptBuilder] Ollama fetch failed: {e}")
        return [cls.DEFAULT_OLLAMA_MODEL]

    @classmethod
    def _get_lm_studio_models_lazy(cls):
        """Lazy load LM Studio models with logging."""
        if cls._lm_studio_models_cache and (time.time() - cls._lm_studio_cache_timestamp) < cls.MODEL_LIST_CACHE_SECONDS:
            return cls._lm_studio_models_cache
        try:
            resp = requests.get(f"{cls.DEFAULT_LM_STUDIO_URL}/v1/models", timeout=2)
            if resp.status_code == 200:
                cls._lm_studio_models_cache = [m["id"] for m in resp.json().get("data", [])]
                cls._lm_studio_cache_timestamp = time.time()
                return cls._lm_studio_models_cache
        except Exception as e:
            logging.debug(f"[WildcardPromptBuilder] LM Studio fetch failed: {e}")
        return [cls.DEFAULT_LM_STUDIO_MODEL]

    def _read_file_content(self, subfolder, filename):
        """Safe file reading helper."""
        if not filename or filename == "None": return ""
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "wildcards", subfolder, filename)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            logging.error(f"[Wildcard] Failed to read {path}: {e}")
            return ""

    def _call_llm(self, backend, api_url, model, prompt):
        """Unified LLM Caller."""
        try:
            if backend == "ollama":
                res = requests.post(f"{api_url}/api/generate", json={"model": model, "prompt": prompt, "stream": False}, timeout=self.API_TIMEOUT_SECONDS)
                return res.json().get("response", "").strip()
            else:
                res = requests.post(f"{api_url}/v1/chat/completions", json={
                    "model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.7, "stream": False
                }, timeout=self.API_TIMEOUT_SECONDS)
                return res.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logging.error(f"[Wildcard] LLM Error: {e}")
            raise

    def _render_text_preview(self, genre, vocal, lyrics, duration, seed):
        """
        Renders the generated text to an image for easy viewing in ComfyUI.
        Includes font fallback safety for cross-platform compatibility.
        """
        img = Image.new('RGB', (self.PREVIEW_WIDTH, self.PREVIEW_HEIGHT), color='#1e1e1e')
        draw = ImageDraw.Draw(img)
        
        # Robust Font Loading
        try:
            font_title = ImageFont.truetype("arial.ttf", 20)
            font_text = ImageFont.truetype("arial.ttf", 14)
        except OSError:
            # Fallback for Linux/Mac if Arial is missing
            try:
                font_title = ImageFont.truetype("DejaVuSans.ttf", 20)
                font_text = ImageFont.truetype("DejaVuSans.ttf", 14)
            except OSError:
                # Final fallback to default bitmap font
                font_title = ImageFont.load_default()
                font_text = ImageFont.load_default()

        padding = 20
        y_cursor = padding

        def draw_line(title, text, color='#87CEEB'):
            nonlocal y_cursor
            draw.text((padding, y_cursor), title, font=font_title, fill='#ffffff')
            y_cursor += 25
            lines = wrap(text, width=90) 
            for line in lines[:3]:
                draw.text((padding + 10, y_cursor), line, font=font_text, fill=color)
                y_cursor += 18
            if len(lines) > 3:
                draw.text((padding + 10, y_cursor), "...", font=font_text, fill='gray')
                y_cursor += 18
            y_cursor += 10

        draw_line("GENRE:", genre, '#87CEEB') # Sky Blue
        draw_line("VOCALS:", vocal, '#98FB98') # Pale Green
        draw_line("DURATION:", f"{duration}s (Seed: {seed})", '#FFD700') # Gold
        draw_line("LYRICS SAMPLE:", lyrics.replace('\n', ' / '), '#FFA07A') # Light Salmon

        img_np = np.array(img).astype(np.float32) / 255.0
        return torch.from_numpy(img_np).unsqueeze(0)

    # --- Generation Methods ---

    def _generate_hybrid_genre(self, concept, seed, backend, url, model):
        """
        Generates genre tags using LLM selection from internal lists.
        Args:
            concept (str): User concept.
            seed (int): Seed for fallback.
            backend, url, model: LLM settings.
        Returns:
            str: Comma-separated genre tags.
        """
        try:
            # Format the massive list into the prompt
            opts = self.HYBRID_GENRE_OPTIONS
            prompt = self.PROMPT_HYBRID_GENRE.format(
                concept=concept,
                base_genre_options=", ".join(opts["base_genre"]),
                subgenre_options=", ".join(opts["subgenre_modifier"]),
                percussion_options=", ".join(opts["percussion_style"]),
                mood_options=", ".join(opts["mood"]),
                instrument_options=", ".join(opts["harmonic_instrument"]),
                chord_options=", ".join(opts["chord_quality"]),
                bass_options=", ".join(opts["bass_character"]),
                texture_options=", ".join(opts["texture"])
            )
            response = self._call_llm(backend, url, model, prompt)
            
            selections = {}
            for line in response.split('\n'):
                if '=' in line:
                    k, v = line.split('=', 1)
                    selections[k.strip().lower()] = v.strip()
            
            parts = []
            if 'subgenre_modifier' in selections and 'base_genre' in selections:
                parts.append(f"{selections['subgenre_modifier']} {selections['base_genre']}")
            elif 'base_genre' in selections: parts.append(selections['base_genre'])
            
            if 'percussion_style' in selections: parts.append(selections['percussion_style'])
            
            if 'mood' in selections and 'harmonic_instrument' in selections:
                parts.append(f"{selections['mood']} {selections['harmonic_instrument']}")
            
            if 'bass_character' in selections: parts.append(selections['bass_character'])
            if 'texture' in selections: parts.append(selections['texture'])
            
            return ", ".join(parts)
        except Exception as e:
            logging.warning(f"[Hybrid] Genre failed: {e}. Using wildcard fallback.")
            return WildcardExpander(seed).expand(self._generate_default_genre_template())

    def _generate_hybrid_vocal(self, concept, seed, backend, url, model):
        """
        Generates vocal tags using LLM selection from internal lists.
        Args:
            concept (str): User concept.
            seed (int): Seed for fallback.
            backend, url, model: LLM settings.
        Returns:
            str: Comma-separated vocal tags.
        """
        try:
            opts = self.HYBRID_VOCAL_OPTIONS
            prompt = self.PROMPT_HYBRID_VOCAL.format(
                concept=concept,
                vocal_type_options=", ".join(opts["vocal_type"]),
                vocal_quality_options=", ".join(opts["vocal_quality"]),
                vocal_effect_options=", ".join(opts["vocal_effect"])
            )
            response = self._call_llm(backend, url, model, prompt)
            
            selections = {}
            for line in response.split('\n'):
                if '=' in line:
                    k, v = line.split('=', 1)
                    selections[k.strip().lower()] = v.strip()
            
            parts = [selections.get(k) for k in ['vocal_type', 'vocal_quality', 'vocal_effect'] if k in selections]
            return ", ".join(parts)
        except Exception as e:
            logging.warning(f"[Hybrid] Vocal failed: {e}. Using wildcard fallback.")
            return WildcardExpander(seed).expand(self._generate_default_vocal_template())

    def execute(self, **kwargs):
        """
        Main execution method with safe fallback.
        Routes to appropriate generation mode (Wildcard/LLM/Hybrid).
        
        Args:
            **kwargs: All input parameters from INPUT_TYPES.
            
        Returns:
            Tuple of (genre_tags, vocal_tags, lyrics, duration, seed, preview_image).
        """
        try:
            # --- Setup ---
            mode = kwargs.get("generation_mode")
            seed = kwargs.get("seed", 0)
            if kwargs.get("randomize_seed", True):
                seed = random.randint(0, 0xffffffffffffffff)
            
            concept = kwargs.get("concept", "")
            expander = WildcardExpander(seed)

            # --- Template Resolution ---
            # Genre
            genre_tmpl = self._read_file_content("genre", kwargs.get("load_genre_file"))
            if not genre_tmpl: genre_tmpl = kwargs.get("custom_genre_template")
            if not genre_tmpl: genre_tmpl = self._generate_default_genre_template()

            # Vocal
            vocal_tmpl = self._read_file_content("vocal", kwargs.get("load_vocal_file"))
            if not vocal_tmpl: vocal_tmpl = kwargs.get("custom_vocal_template")
            if not vocal_tmpl: vocal_tmpl = self._generate_default_vocal_template()

            # Lyrics
            lyrics_tmpl = self._read_file_content("lyrics", kwargs.get("load_lyrics_file"))
            if not lyrics_tmpl: lyrics_tmpl = kwargs.get("custom_lyrics_template")
            if not lyrics_tmpl: lyrics_tmpl = self.DEFAULT_LYRICS_TEMPLATE

            # Duration
            dur_tmpl = kwargs.get("duration_template", "{120|180}")

            # --- Generation Logic ---
            genre_out, vocal_out, lyrics_out = "", "", ""
            
            # LLM Settings
            backend = kwargs.get("llm_backend")
            url = kwargs.get("ollama_api_url") if backend == "ollama" else kwargs.get("lm_studio_api_url")
            model = kwargs.get("ollama_model") if backend == "ollama" else kwargs.get("lm_studio_model")

            # 1. DURATION
            duration_out = expander.expand(dur_tmpl)

            # 2. GENRE
            if kwargs.get("generate_genre", True):
                if mode == "hybrid":
                    genre_out = self._generate_hybrid_genre(concept, seed, backend, url, model)
                elif mode == "llm":
                    prompt = f"Generate 5 descriptive music genre tags for: {concept}. Comma separated."
                    try: genre_out = self._call_llm(backend, url, model, prompt)
                    except: genre_out = expander.expand(genre_tmpl)
                else: # Wildcard
                    genre_out = expander.expand(genre_tmpl)

            # 3. VOCALS & LYRICS
            force_inst = kwargs.get("force_instrumental", False)
            
            if force_inst:
                vocal_out = ""
                lyrics_out = "[Instrumental]"
            else:
                if kwargs.get("generate_vocals", True):
                    if mode == "hybrid":
                        vocal_out = self._generate_hybrid_vocal(concept, seed, backend, url, model)
                    elif mode == "llm":
                        prompt = f"Generate 3 vocal style tags for: {concept}. Comma separated."
                        try: vocal_out = self._call_llm(backend, url, model, prompt)
                        except: vocal_out = expander.expand(vocal_tmpl)
                    else:
                        vocal_out = expander.expand(vocal_tmpl)

                if kwargs.get("generate_lyrics", True):
                    if mode in ["llm", "hybrid"]:
                        try:
                            lyrics_out = self._call_llm(backend, url, model, self.PROMPT_LLM_LYRICS.format(concept=concept))
                        except:
                            lyrics_out = expander.expand(lyrics_tmpl)
                    else:
                        lyrics_out = expander.expand(lyrics_tmpl)

            # --- Visualization ---
            preview_image = self._render_text_preview(genre_out, vocal_out, lyrics_out, duration_out, seed)

            return (genre_out, vocal_out, lyrics_out, duration_out, seed, preview_image)

        except Exception as e:
            logging.error(f"[WildcardPromptBuilder] Critical Execution Error: {e}")
            logging.debug(traceback.format_exc())
            
            # Fail-safe return
            error_img = torch.zeros((1, 64, 64, 3), dtype=torch.float32) # Black box
            return ("Ambient, Error", "", "[System Failure]", "120", 0, error_img)

# =================================================================================
# == Node Registration                                                           ==
# =================================================================================
NODE_CLASS_MAPPINGS = {
    "WildcardPromptBuilder": WildcardPromptBuilder
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "WildcardPromptBuilder": "MD: Wildcard Prompt Builder"
}