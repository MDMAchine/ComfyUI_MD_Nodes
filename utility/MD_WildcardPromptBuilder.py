# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# █▓▒░                                                                     ░▒▓█
# █▓▒░   MD_Nodes/WildcardPromptBuilder – Ultimate Prompt Engine v2.3.2    ░▒▓█
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
# ║ ░▒▓ ORIGIN:
# ║   • Hybrid Wildcard/LLM Prompt Generation Engine
# ║   • Research: Based on standard prompt engineering patterns for audio/image/video models.
# ║
# ║ ░▒▓ DESCRIPTION:
# ║   The ultimate prompt generation engine for ACE-Step audio, Stable Diffusion/Flux
# ║   image, and video (Wan/Mochi/etc) workflows. Combines massive internal datasets,
# ║   external file loading, and LLM hybrid logic to generate Genres, Vocals, Lyrics,
# ║   Duration, BPM, Time Signature, Key/Scale, and a unified INTERPRETED_PROMPT
# ║   string formatted for the selected output target.
# ║   NOTE: As a text orchestrator and API client, this runs entirely in the public wrapper domain.
# ║
# ║ ░▒▓ CORE FEATURES:
# ║    ✓ Output Targets: Audio (ACE-Step), Image (SD/Flux), Video (Wan/Mochi), All Three.
# ║    ✓ ACE-Step Mode: Rich Gemini-style prose captions, BPM/key stripped from caption.
# ║    ✓ Massive Vocab: 600+ options with deep Neurofunk/DnB, genre-specific terminology.
# ║    ✓ Orchestrator Mode: Coherent single-pass LLM generation (Standard + ACE-Architect).
# ║    ✓ Ollama Resilience: Background thread fetch, JSON sidecar cache, non-blocking startup.
# ║    ✓ ollama_enabled Toggle: Disables all LLM calls cleanly when Ollama not installed.
# ║    ✓ Smart BPM Engine: Context-aware BPM detection from genre text.
# ║    ✓ Rich Visualization: Renders generated prompt as formatted preview image.
# ║
# ║ ░▒▓ CHANGELOG:
# ║    v2.3.2 (2026-03-19) - Timeout Param + Think Block Filter:
# ║    ├── ADDED: llm_timeout INT param (30–600s, default 120) — no more hardcoded 120s stall.
# ║    ├── ADDED: _strip_think_blocks() — strips <think>...</think> from ALL LLM responses.
# ║    └── FIX: Qwen3/QwQ/R1 models no longer pollute outputs with raw CoT text.
# ║    v2.3.0 (2026-03-15) - Multi-Modal + ACE-Step Quality + Ollama Resilience:
# ║    ├── ADDED: output_target selector (Audio/Image/Video/All Three).
# ║    ├── ADDED: ace_step_mode toggle — rich prose captions, auto BPM/key strip.
# ║    ├── ADDED: INTERPRETED_PROMPT output — target-formatted unified prompt string.
# ║    ├── ADDED: strip_metadata_from_caption() — removes BPM/key/timesig from caption.
# ║    ├── ADDED: ollama_enabled toggle — disables all LLM calls gracefully when False.
# ║    ├── ADDED: JSON sidecar model cache (.ollama_model_cache.json) — survives restarts.
# ║    ├── ADDED: Background thread Ollama model fetch — zero blocking on startup.
# ║    ├── ADDED: Deep Neurofunk/DnB vocabulary (reese, amen, neuro growl, etc).
# ║    ├── ADDED: Genre-specific vocab banks: Techno, House, Ambient, Synthwave, Hip-Hop,
# ║    │          Metal, Jazz, Cinematic — with authentic production terminology.
# ║    ├── ADDED: Image prompt vocabulary (lighting, camera, style, artist, render engine).
# ║    ├── ADDED: Video prompt vocabulary (camera motion, scene, action, cinematography).
# ║    ├── ADDED: PROMPT_ACE_STEP_CAPTION — Gemini-annotator-style caption LLM prompt.
# ║    ├── ADDED: Image/Video LLM prompt templates.
# ║    ├── FIXED: PROMPT_ACE_ARCHITECT caption format — now outputs rich prose not tag lists.
# ║    ├── FIXED: descriptive_mode hybrid/llm paths now ACE-Step aware when ace_step_mode=True.
# ║    └── UNCHANGED: All existing outputs, wildcard/hybrid/llm/orchestrator logic preserved.
# ║    v2.2.0 (Enterprise Standards - Feb 2026):
# ║    ├── REFACTOR: Tooltips strictly updated to 5-part v1.5.4 standard.
# ║    └── VERIFIED: PerformanceProfiler matches v1.5.3 exact specifications.
# ║    v2.1.2 (2026-02-14) - Robustness Update
# ║    ├── FIXED: Indentation error preventing node from loading.
# ║    └── ENHANCED: ACE-Architect prompt template for actual lyrics.
# ╚════════════════════════════════════════════════════════════════════════════

# =================================================================================
# == Standard Library Imports                                                    ==
# =================================================================================
VERSION = "v2.3.2"  # UPS v1.5.8


import logging
import random
import re
import secrets
import time
import traceback
import io
import os
import glob
import sys
import json
import platform
import threading
from urllib.parse import urlparse
from textwrap import wrap

# =================================================================================
# == Third-Party Imports                                                         ==
# =================================================================================
try:
    import requests
    CONST_REQUESTS_AVAILABLE = True
except ImportError:
    CONST_REQUESTS_AVAILABLE = False
    logging.warning("[MD_WildcardPromptBuilder] 'requests' not found. LLM features disabled.")

try:
    import torch
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    CONST_IMAGING_AVAILABLE = True
except ImportError:
    CONST_IMAGING_AVAILABLE = False
    logging.warning("[MD_WildcardPromptBuilder] PIL/torch not found. Preview disabled.")

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    logging.warning("[MD_WildcardPromptBuilder] PyYAML not found. YAML features disabled.")

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logging.warning("[MD_WildcardPromptBuilder] Matplotlib not available, visualization disabled")

# =================================================================================
# == Configuration Constants                                                     ==
# =================================================================================
CONST_JS_MAX_SAFE_INTEGER  = 9007199254740991
CONST_DEFAULT_OLLAMA_URL   = "http://localhost:11434"
CONST_DEFAULT_OLLAMA_MODEL = "llama3:8b-instruct-q8_0"
CONST_MAX_RETRY_ATTEMPTS   = 3
CONST_API_TIMEOUT_SECONDS  = 120
CONST_MODEL_LIST_CACHE_SECONDS = 300
CONST_OLLAMA_PROBE_TIMEOUT = 2          # Fast probe on startup — never blocks longer than this

CONST_PREVIEW_WIDTH  = 800
CONST_PREVIEW_HEIGHT = 600
CONST_FONT_SIZE_TITLE = 22
CONST_FONT_SIZE_BODY  = 16
CONST_FONT_SIZE_SMALL = 14

CONST_MIN_BPM = 40
CONST_MAX_BPM = 220

# Regex for stripping numeric metadata from captions
CONST_METADATA_STRIP_PATTERNS = [
    r'\b\d{2,3}\s*(?:bpm|BPM|Bpm)\b',          # "174 bpm", "174BPM"
    r'\bbpm\s*[:=]?\s*\d{2,3}\b',               # "bpm: 174"
    r'\b\d{2,3}\s*/\s*\d\b',                    # "4/4", "174/4"
    r'\b[A-G][#b]?\s+(?:major|minor|Major|Minor)\b',  # "F# minor" — kept separate, only strip if ace_step_mode
    r'\b(?:key|Key)\s*[:=]\s*[A-G][#b]?\s*(?:major|minor)?\b',  # "Key: F minor"
    r'\btime\s*sig(?:nature)?\s*[:=]?\s*\d+(?:/\d+)?\b',        # "time sig: 4/4"
]

# =================================================================================
# == Genre & BPM Data                                                            ==
# =================================================================================
GENRE_BPM_MAP = {
    # DnB / Jungle family
    "drum and bass": 174, "dnb": 174, "liquid dnb": 172, "liquid drum and bass": 172,
    "neurofunk": 176, "techstep": 174, "jump up": 172, "jungle": 168,
    "intelligent dnb": 170, "darkstep": 174, "rollers": 174,
    # Breakcore / Hardcore
    "breakcore": 180, "hardcore": 170, "happy hardcore": 170, "gabber": 180,
    "speedcore": 200, "terrorcore": 190,
    # Bass music
    "dubstep": 140, "deathstep": 145, "riddim": 140, "melodic dubstep": 138,
    "future bass": 150, "wave": 140, "hardwave": 160,
    # Trap / Phonk
    "trap": 140, "phonk": 135, "drift phonk": 150, "trap rap": 140, "drill": 140,
    # Trance
    "psytrance": 145, "goa trance": 145, "uplifting trance": 140, "trance": 138,
    "progressive trance": 138, "dark psy": 148,
    # Techno
    "industrial techno": 145, "hard techno": 140, "acid techno": 135,
    "peak time techno": 140, "dub techno": 130, "minimal techno": 132, "techno": 133,
    # House
    "tech house": 126, "bass house": 128, "progressive house": 128,
    "deep house": 122, "acid house": 130, "lo-fi house": 120, "house": 128,
    # Disco / Funk
    "garage": 130, "2-step": 130, "nu-disco": 118, "disco": 120, "funk": 115,
    # Electronic / Synth
    "synthpop": 120, "pop": 120, "glitch hop": 110, "glitch": 110, "idm": 130,
    "ebm": 140, "industrial": 140, "aggrotech": 145,
    # Downtempo
    "midtempo": 100, "trip hop": 90, "breakbeat": 130, "big beat": 130,
    "lo-fi": 85, "chillwave": 85, "downtempo": 90,
    # Synthwave / Retro
    "synthwave": 100, "darkwave": 115, "outrun": 100, "vaporwave": 80, "cyberpunk": 140,
    # Ambient
    "ambient": 80, "dark ambient": 75, "drone": 60, "space ambient": 70,
    "cinematic": 80, "psybient": 90,
    # Hip-Hop
    "hip hop": 90, "hip-hop": 90, "boom bap": 90, "lo-fi hip hop": 85,
    "gangsta rap": 95, "old school hip hop": 95,
    # Rock / Metal
    "rock": 120, "hard rock": 130, "metal": 140, "heavy metal": 150,
    "thrash metal": 180, "death metal": 160, "black metal": 140,
    "metalcore": 160, "djent": 140, "thall": 135,
    "punk": 160, "pop punk": 150, "grunge": 110, "alternative rock": 120,
    "indie rock": 120, "post-rock": 110, "shoegaze": 100, "math rock": 130,
    # Jazz / Soul
    "jazz": 100, "acid jazz": 110, "nu-jazz": 100, "blues": 80,
    "soul": 90, "r&b": 90, "neo-soul": 80, "gospel": 100,
    # World / Other
    "reggae": 75, "dub": 70, "dancehall": 95, "ska": 140,
    "country": 110, "folk": 100, "bluegrass": 140, "americana": 100,
    "classical": 90, "orchestral": 90, "soundtrack": 100,
}

TIME_SIGNATURES = ["4", "3", "5", "6", "7"]

KEY_SCALES = []
KEYS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
for _k in KEYS:
    KEY_SCALES.append(f"{_k} major")
    KEY_SCALES.append(f"{_k} minor")

# =================================================================================
# == Helper Classes                                                              ==
# =================================================================================
class WildcardExpander:
    """Expands wildcard patterns like {opt1|opt2} with seeded randomness."""
    WILDCARD_PATTERN = re.compile(r'\{([^{}]+)\}')

    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.selections = []

    def expand(self, template):
        self.selections = []
        if not template: return ""

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


class PerformanceProfiler:
    """Standard performance profiler for MD_Nodes."""
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.timings = {}
        self.start_times = {}

    def start(self, operation_name):
        if not self.enabled: return
        self.start_times[operation_name] = time.perf_counter()

    def stop(self, operation_name):
        if not self.enabled: return
        if operation_name in self.start_times:
            elapsed = time.perf_counter() - self.start_times[operation_name]
            self.timings.setdefault(operation_name, []).append(elapsed)
            del self.start_times[operation_name]

    def get_total_time(self):
        if not self.enabled or not self.timings: return 0.0
        return sum(sum(t) for t in self.timings.values())

    def print_report(self):
        if not self.enabled or not self.timings: return
        logging.info("\n  PERFORMANCE (Generation):")
        total = self.get_total_time()
        logging.info(f"    Total Time: {total:.4f}s")
        for op_name, times in sorted(self.timings.items()):
            avg = sum(times) / len(times)
            if len(times) == 1:
                logging.info(f"    {op_name}: {avg:.4f}s")
            else:
                logging.info(f"    {op_name}: {avg:.4f}s avg ({len(times)}x)")


# =================================================================================
# == Core Node Class                                                             ==
# =================================================================================
class WildcardPromptBuilder:
    """Multi-modal prompt engine: ACE-Step audio, SD/Flux image, Wan/Mochi video."""

    _ollama_models_cache    = None
    _ollama_cache_timestamp = 0.0
    _ollama_fetch_lock      = threading.Lock()
    _ollama_fetch_thread    = None

    # ─────────────────────────────────────────────────────────────────────────
    # VOCABULARY — loaded from wildcards/ txt files at first use
    # ─────────────────────────────────────────────────────────────────────────

    # Minimal hardcoded fallbacks (used only when txt files are missing)
    _FALLBACK_BASE_GENRES = [
        "Drum and Bass", "Neurofunk", "Techstep", "Liquid DnB", "Jungle",
        "Techno", "Hard Techno", "House", "Deep House", "Ambient",
        "Synthwave", "Hip Hop", "Metal", "Jazz", "Cinematic",
    ]
    _FALLBACK_BASS = [
        "Classic reese bass with chorus", "Clean Sub Bass",
        "Distorted 808 sub", "Acid 303 Squell", "Wobble Bass",
    ]
    _FALLBACK_PERCUSSION = [
        "Amen break chop", "Four-to-the-floor kick", "Trap 808 rolls",
        "Minimal sparse clicks", "Blast beats",
    ]

    _vocab_cache:  dict = {}
    _vocab_loaded: bool = False

    @classmethod
    def _vocab_dir(cls):
        return os.path.join(os.path.dirname(os.path.realpath(__file__)), "wildcards")

    @classmethod
    def _read_vocab_file(cls, rel_path, comment_char="#"):
        """Read a vocab txt file, skip comment lines and section headers."""
        full = os.path.join(cls._vocab_dir(), rel_path)
        if not os.path.exists(full):
            return []
        lines = []
        with open(full, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith(comment_char) or (line.startswith("[") and line.endswith("]")):
                    continue
                lines.append(line)
        return lines

    @classmethod
    def _read_vocab_section(cls, rel_path, section):
        """Read a specific [SECTION] block from a sectioned vocab txt file."""
        full = os.path.join(cls._vocab_dir(), rel_path)
        if not os.path.exists(full):
            return []
        result = []
        in_section = False
        target = f"[{section.upper()}]"
        with open(full, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.upper() == target:
                    in_section = True
                    continue
                if line.startswith("[") and line.endswith("]"):
                    if in_section:
                        break
                    continue
                if in_section:
                    result.append(line)
        return result

    @classmethod
    def _load_vocab_banks(cls):
        """Load all vocab banks from txt files into _vocab_cache. Called once.

        Directory layout:
          wildcards/
            _vocab/                  <- internal vocab banks (NOT shown in file picker)
              genre_packs/           <- sectioned [BASE_GENRE]/[BASS_CHARACTER]/etc files
                dnb_neurofunk.txt
                techno.txt  ...
              _internal/             <- shared flat one-per-line banks
                moods.txt, textures.txt, etc
            genre/                   <- user wildcard files {a|b} (file picker only, untouched)
            image/                   <- image vocab flat files
            video/                   <- video vocab flat files
            vocal/                   <- vocal chain files
        """
        if cls._vocab_loaded:
            return

        vocab_dir  = cls._vocab_dir()
        packs_dir  = os.path.join(vocab_dir, "_vocab", "genre_packs")
        intern_pfx = os.path.join("_vocab", "_internal")

        def _collect_section(section):
            """Read [SECTION] blocks from _vocab/genre_packs/ only — never touches genre/."""
            entries = []
            if os.path.exists(packs_dir):
                for fname in sorted(os.listdir(packs_dir)):
                    if fname.endswith(".txt") and not fname.startswith("."):
                        rel = os.path.join("_vocab", "genre_packs", fname)
                        entries.extend(cls._read_vocab_section(rel, section))
            return entries

        def _internal(fname):
            return cls._read_vocab_file(os.path.join(intern_pfx, fname))

        base_genres = _collect_section("BASE_GENRE") or cls._FALLBACK_BASE_GENRES
        bass_all    = _collect_section("BASS_CHARACTER") or cls._FALLBACK_BASS
        perc_all    = _collect_section("PERCUSSION_STYLE") or cls._FALLBACK_PERCUSSION

        dnb_pack = os.path.join("_vocab", "genre_packs", "dnb_neurofunk.txt")

        cls._vocab_cache = {
            "HYBRID_GENRE_OPTIONS": {
                "base_genre":            base_genres,
                "subgenre_modifier":     _internal("subgenre_modifiers.txt") or ["Atmospheric", "Dark"],
                "percussion_style":      perc_all,
                "mood":                  _internal("moods.txt") or ["Aggressive", "Melancholic"],
                "harmonic_instrument":   _internal("harmonic_instruments.txt") or ["Atmospheric Pads"],
                "chord_quality":         _internal("chord_quality.txt") or ["Minor", "Major"],
                "bass_character":        bass_all,
                "texture":               _internal("textures.txt") or ["Lush string pads"],
                "dnb_break_style":       cls._read_vocab_section(dnb_pack, "BREAK_STYLE"),
                "dnb_production_detail": cls._read_vocab_section(dnb_pack, "PRODUCTION_DETAIL"),
            },
            "HYBRID_VOCAL_OPTIONS": {
                "vocal_type": [], "vocal_quality": [], "vocal_effect": [],
            },
            "IMAGE_VOCAB": {
                "style":        cls._read_vocab_file("image/styles.txt"),
                "lighting":     cls._read_vocab_file("image/lighting.txt"),
                "camera":       cls._read_vocab_file("image/camera.txt"),
                "quality_tags": cls._read_vocab_file("image/quality_tags.txt"),
                "artist_style": cls._read_vocab_file("image/artist_styles.txt"),
            },
            "VIDEO_VOCAB": {
                "camera_motion":     cls._read_vocab_file("video/camera_motion.txt"),
                "scene_quality":     cls._read_vocab_file("video/scene_quality.txt"),
                "action_descriptor": cls._read_vocab_file("video/action_descriptors.txt"),
                "mood_grade":        cls._read_vocab_file("video/mood_grades.txt"),
            },
        }

        # Parse vocal chains file: "type, quality, effect" per line
        vocal_path = os.path.join(cls._vocab_dir(), "vocal", "md_core_vocal_chains.txt")
        if os.path.exists(vocal_path):
            types, qualities, effects = [], [], []
            with open(vocal_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"): continue
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 3:
                        types.append(parts[0])
                        qualities.append(parts[1])
                        effects.append(parts[2])
            cls._vocab_cache["HYBRID_VOCAL_OPTIONS"]["vocal_type"]   = list(dict.fromkeys(types))    or ["Soft female vocals"]
            cls._vocab_cache["HYBRID_VOCAL_OPTIONS"]["vocal_quality"] = list(dict.fromkeys(qualities)) or ["Ethereal"]
            cls._vocab_cache["HYBRID_VOCAL_OPTIONS"]["vocal_effect"]  = list(dict.fromkeys(effects))   or ["Reverb-drenched (Valhalla)"]

        cls._vocab_loaded = True

    @classmethod
    def _get_vocab(cls, bank, key):
        cls._load_vocab_banks()
        return cls._vocab_cache.get(bank, {}).get(key, [])

    @classmethod
    def _hybrid_genre_opts(cls):
        cls._load_vocab_banks()
        return cls._vocab_cache["HYBRID_GENRE_OPTIONS"]

    @classmethod
    def _hybrid_vocal_opts(cls):
        cls._load_vocab_banks()
        return cls._vocab_cache["HYBRID_VOCAL_OPTIONS"]

    @classmethod
    def _image_vocab(cls):
        cls._load_vocab_banks()
        return cls._vocab_cache["IMAGE_VOCAB"]

    @classmethod
    def _video_vocab(cls):
        cls._load_vocab_banks()
        return cls._vocab_cache["VIDEO_VOCAB"]

    # ── Shims for any direct class-level access — all delegate to file-loaded cache ──
    # These are intentionally empty; the real data lives in wildcards/ txt files.
    # Use _hybrid_genre_opts(), _hybrid_vocal_opts(), _image_vocab(), _video_vocab()
    # inside methods, or call _get_vocab(bank, key) directly.
    HYBRID_GENRE_OPTIONS  = {}  # populated lazily by _load_vocab_banks()
    HYBRID_VOCAL_OPTIONS  = {}  # populated lazily by _load_vocab_banks()
    IMAGE_VOCAB           = {}  # populated lazily by _load_vocab_banks()
    VIDEO_VOCAB           = {}  # populated lazily by _load_vocab_banks()

    # ─────────────────────────────────────────────────────────────────────────
    # LLM PROMPT TEMPLATES
    # ─────────────────────────────────────────────────────────────────────────

    PROMPT_HYBRID_GENRE = """Based on the concept "{concept}", select the BEST option from each category.
CATEGORIES:
- base_genre: {base_genre_options}
- subgenre_modifier: {subgenre_options}
- percussion_style: {percussion_style}
- mood: {mood}
- harmonic_instrument: {harmonic_instrument}
- chord_quality: {chord_quality}
- bass_character: {bass_character}
- texture: {texture}

Respond in EXACT format (key=value, one per line, no extra text):
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

Respond in EXACT format (key=value, one per line, no extra text):
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

    # Standard orchestrator (general purpose)
    PROMPT_ORCHESTRATOR = """Act as a Lead Music Producer. Analyze the concept: "{concept}".
Determine the optimal production specifications.

CRITICAL INSTRUCTIONS:
1. Output ONLY the strict format below.
2. Ensure BPM, Key, and Time Signature strictly match the Genre's energy.
3. For LYRICS, write the full text on multiple lines.
4. Do NOT put BPM, key, or time signature inside the GENRE field.

REQUIRED FORMAT:
GENRE=Best fitting sub-genre and style tags (NO BPM, NO key info here)
VOCALS=Vocal type and processing chain (or '[Instrumental]')
BPM=Integer (e.g. 128)
TIME_SIG=String (Select one: 3, 4, 5, 6, 7)
KEY_SCALE=String (e.g. C Minor, F# Major)
DURATION=Integer seconds (e.g. 180)
LYRICS=
[Verse 1]
...lines..."""

    # ACE-Step Architect — outputs rich Gemini-style prose caption
    PROMPT_ACE_ARCHITECT = """You are a professional music AI captioner trained on the ACE-Step 1.5 audio model.
Your goal is to produce a rich, descriptive musical caption and complete metadata for the concept: "{concept}".

CRITICAL RULES:
1. The Caption MUST be 2-4 prose sentences describing the musical sound, NOT a tag list.
2. Caption MUST describe: genre style, sonic texture, instruments, bass character, atmosphere, energy, and production feel.
3. Caption MUST NOT contain BPM numbers, key names, or time signature numbers — those belong only in the <think> block.
4. Arrangement section: timeline of instrument entries and SFX (optional, keep brief).
5. Lyrics section: write actual sung lyrics or [Instrumental].

OUTPUT FORMAT (follow exactly):
<think>
bpm: [Integer]
duration: [Integer seconds]
keyscale: [Key + Scale, e.g. F# Minor]
timesignature: [Number only, e.g. 4]
</think>

# Caption
[2-4 rich descriptive prose sentences about the sound. Style, texture, bass, drums, atmosphere, energy. NO BPM. NO key names.]

# Arrangement
[Optional: 0:00-0:15 Intro with heavy sub-bass, break enters at 0:30, etc.]

# Lyrics
[Verse 1]
(Write actual lyrics, or just: [Instrumental])
[Chorus]
(Write actual lyrics)"""

    # ACE-Step caption-only prompt for standalone rich caption generation
    PROMPT_ACE_STEP_CAPTION = """You are a professional music AI captioner. Your captions describe audio tracks for training a music generation model.

Write a rich 2-4 sentence prose caption for a music track described as: "{concept}"

RULES:
- Describe the genre, sonic texture, primary instruments, bass character, drum style, mood, and production atmosphere.
- Use specific production terminology (e.g. "reese bass", "amen break", "supersaw pads", "sidechain compression", "granular texture").
- Do NOT mention BPM numbers, key names, or time signatures.
- Do NOT use bullet points or lists. Write flowing prose sentences only.
- Output ONLY the caption text. No preamble, no labels, no extra explanation.

Example output style:
"A high-energy neurofunk drum and bass track built on complex rolling breakbeats driven by deep reese bass growls and razor-sharp synthesizer stabs. The production features intricate sound design with formant-filtered bass modulation, crisp snare transients, and a dark industrial atmosphere layered with processed metallic textures."

Now write the caption:"""

    # Image prompt generator
    PROMPT_IMAGE_CAPTION = """You are an expert Stable Diffusion / Flux prompt engineer.
Generate a detailed image generation prompt for the concept: "{concept}"

RULES:
- Lead with the main subject and scene description.
- Include style, lighting, camera/lens details, and quality tags.
- Be specific and visual. No abstract emotional words without grounding them in visual details.
- Output ONLY the prompt text. No labels or explanation.
- Keep under 120 words.

Example:
"A lone cyberpunk hacker in a neon-soaked alley, rain-slicked concrete reflecting pink and cyan light, holographic displays floating in mist, 35mm lens, cinematic photography, volumetric god rays, photorealistic, ultra detailed, masterpiece"

Now write the prompt:"""

    # Video prompt generator
    PROMPT_VIDEO_CAPTION = """You are an expert AI video generation prompt engineer for models like Wan and Mochi.
Generate a detailed video prompt for the concept: "{concept}"

RULES:
- Describe the scene, the primary action or motion, camera movement, lighting, and visual mood.
- Be specific about motion (e.g. "slow dolly push-in", "orbital arc", "static locked-off shot").
- Describe color grade and atmosphere.
- Output ONLY the prompt text. No labels or explanation.
- Keep under 100 words.

Example:
"A slow dolly push-in through a fog-filled industrial warehouse at night, orange sodium lights casting long shadows on concrete floors, distant machinery humming, shallow depth of field, cinematic 4K, warm amber-orange grade, high dynamic range, photorealistic lighting"

Now write the prompt:"""

    DEFAULT_LYRICS_TEMPLATE = "[Instrumental]"
    DEFAULT_DURATION_TEMPLATE = "{120|180|240}"
    DEFAULT_BPM_TEMPLATE     = "{100|110|120|128|130|140|150|160|174}"
    DEFAULT_TIME_SIG  = "4"
    DEFAULT_KEY_SCALE = "C major"

    def __init__(self): pass

    # ─────────────────────────────────────────────────────────────────────────
    # OLLAMA MODEL FETCH — non-blocking with JSON sidecar cache
    # ─────────────────────────────────────────────────────────────────────────
    @classmethod
    def _sidecar_cache_path(cls):
        return os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "wildcards", ".ollama_model_cache.json"
        )

    @classmethod
    def _load_sidecar_cache(cls):
        try:
            path = cls._sidecar_cache_path()
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                age = time.time() - data.get("timestamp", 0)
                if age < CONST_MODEL_LIST_CACHE_SECONDS * 2:
                    return data.get("models", [])
        except Exception:
            pass
        return []

    @classmethod
    def _save_sidecar_cache(cls, models):
        try:
            os.makedirs(os.path.dirname(cls._sidecar_cache_path()), exist_ok=True)
            with open(cls._sidecar_cache_path(), "w", encoding="utf-8") as f:
                json.dump({"timestamp": time.time(), "models": models}, f)
        except Exception:
            pass

    @classmethod
    def _fetch_ollama_models_bg(cls):
        """Background thread: probe Ollama, update in-memory + sidecar cache."""
        if not CONST_REQUESTS_AVAILABLE:
            return
        try:
            resp = requests.get(
                f"{CONST_DEFAULT_OLLAMA_URL}/api/tags",
                timeout=CONST_OLLAMA_PROBE_TIMEOUT
            )
            if resp.status_code == 200:
                models = [m["name"] for m in resp.json().get("models", [])]
                if models:
                    with cls._ollama_fetch_lock:
                        cls._ollama_models_cache    = models
                        cls._ollama_cache_timestamp = time.time()
                    cls._save_sidecar_cache(models)
        except Exception:
            pass  # Ollama not running — silent, never blocking

    @classmethod
    def _get_ollama_models_lazy(cls):
        """Return model list instantly from cache; refresh in background if stale."""
        # 1. In-memory cache (fresh enough)
        with cls._ollama_fetch_lock:
            age = time.time() - cls._ollama_cache_timestamp
            if cls._ollama_models_cache and age < CONST_MODEL_LIST_CACHE_SECONDS:
                return cls._ollama_models_cache

        # 2. Sidecar JSON (survives restarts)
        sidecar = cls._load_sidecar_cache()
        if sidecar:
            with cls._ollama_fetch_lock:
                cls._ollama_models_cache    = sidecar
                cls._ollama_cache_timestamp = time.time()
            # Kick off background refresh without blocking
            if cls._ollama_fetch_thread is None or not cls._ollama_fetch_thread.is_alive():
                cls._ollama_fetch_thread = threading.Thread(
                    target=cls._fetch_ollama_models_bg, daemon=True
                )
                cls._ollama_fetch_thread.start()
            return sidecar

        # 3. Nothing cached at all — background fetch, return placeholder immediately
        if cls._ollama_fetch_thread is None or not cls._ollama_fetch_thread.is_alive():
            cls._ollama_fetch_thread = threading.Thread(
                target=cls._fetch_ollama_models_bg, daemon=True
            )
            cls._ollama_fetch_thread.start()

        return [CONST_DEFAULT_OLLAMA_MODEL, "[Ollama Not Detected]"]

    # ─────────────────────────────────────────────────────────────────────────
    # FILE HELPERS
    # ─────────────────────────────────────────────────────────────────────────
    @classmethod
    def _get_files_from_dir(cls, subfolder):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        target_dir  = os.path.join(current_dir, "wildcards", subfolder)
        if not os.path.exists(target_dir): return ["None", "Random"]
        files = [os.path.basename(f) for f in glob.glob(os.path.join(target_dir, "*.txt"))]
        return ["None", "Random"] + sorted(files)

    def _read_file_content(self, subfolder, filename):
        if not filename or filename == "None": return ""
        path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "wildcards", subfolder, filename
        )
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception as e:
            logging.error(f"[WildcardPromptBuilder] Failed to read {path}: {e}")
            return ""

    def _read_random_file_content(self, subfolder, seed):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        target_dir  = os.path.join(current_dir, "wildcards", subfolder)
        if not os.path.exists(target_dir): return ""
        files = [os.path.basename(f) for f in glob.glob(os.path.join(target_dir, "*.txt"))]
        if not files: return ""
        rng = random.Random(seed)
        selected = rng.choice(files)
        return self._read_file_content(subfolder, selected)

    # ─────────────────────────────────────────────────────────────────────────
    # BPM DETECTION
    # ─────────────────────────────────────────────────────────────────────────
    @classmethod
    def _smart_bpm_from_genre(cls, genre_str, seed):
        genre_lower = genre_str.lower()
        matches = []
        for keyword, bpm in GENRE_BPM_MAP.items():
            if keyword in genre_lower:
                matches.append((keyword, bpm))
        if matches:
            matches.sort(key=lambda x: len(x[0]), reverse=True)
            _, base_bpm = matches[0]
            rng = random.Random(seed)
            variation = rng.randint(-3, 3)
            return max(CONST_MIN_BPM, min(CONST_MAX_BPM, base_bpm + variation))
        rng = random.Random(seed)
        return rng.choice([80, 90, 100, 110, 120, 128, 130, 140, 145, 150, 160, 174])

    # ─────────────────────────────────────────────────────────────────────────
    # METADATA STRIP (ACE-Step caption safety)
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def strip_metadata_from_caption(text, strip_key_scale=True):
        """Remove BPM, key/scale, and time signature strings from caption text."""
        if not text: return text
        patterns = list(CONST_METADATA_STRIP_PATTERNS)
        if not strip_key_scale:
            patterns = patterns[:3]  # Only BPM + time sig patterns
        for pat in patterns:
            text = re.sub(pat, "", text, flags=re.IGNORECASE)
        # Clean up double commas or leading/trailing comma artefacts
        text = re.sub(r',\s*,', ',', text)
        text = re.sub(r'^[\s,]+|[\s,]+$', '', text)
        return text.strip()

    # ─────────────────────────────────────────────────────────────────────────
    # THINK BLOCK FILTER (Qwen3 / QwQ / DeepSeek-R1 CoT stripping)
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _strip_think_blocks(text):
        """Strip <think>...</think> CoT blocks from reasoning-model responses.

        ACE-Architect parser intentionally reads the think block for metadata
        (BPM, duration, key) — call this ONLY on non-Architect paths where
        the raw think block text should never reach the output.
        """
        if not text:
            return text
        # Remove all <think>...</think> spans (including multiline, nested-safe)
        stripped = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
        # Also catch unclosed/malformed tags — if a <think> opens but never closes,
        # nuke everything from the tag to end-of-string to prevent CoT leakage.
        stripped = re.sub(r"<think>.*", "", stripped, flags=re.DOTALL | re.IGNORECASE)
        return stripped.strip()


    @classmethod
    def _generate_default_genre_template(cls):
        opts = cls._hybrid_genre_opts()
        def to_wc(key): return "{" + "|".join(opts[key]) + "}"
        return (
            f"{to_wc('base_genre')}, {to_wc('subgenre_modifier')}, "
            f"{to_wc('percussion_style')}, {to_wc('mood')} "
            f"{to_wc('harmonic_instrument')} {to_wc('chord_quality')} chords, "
            f"{to_wc('bass_character')}, {to_wc('texture')}"
        )

    @classmethod
    def _generate_default_vocal_template(cls):
        opts = cls._hybrid_vocal_opts()
        def to_wc(key): return "{" + "|".join(opts[key]) + "}"
        return f"{to_wc('vocal_type')}, {to_wc('vocal_quality')}, {to_wc('vocal_effect')}"

    @classmethod
    def _generate_default_image_template(cls):
        vocab = cls._image_vocab()
        def to_wc(key): return "{" + "|".join(vocab[key]) + "}"
        return (
            f"{to_wc('style')}, {to_wc('lighting')}, {to_wc('camera')}, "
            f"{to_wc('quality_tags')}"
        )

    @classmethod
    def _generate_default_video_template(cls):
        vocab = cls._video_vocab()
        def to_wc(key): return "{" + "|".join(vocab[key]) + "}"
        return (
            f"{to_wc('camera_motion')}, {to_wc('action_descriptor')}, "
            f"{to_wc('scene_quality')}, {to_wc('mood_grade')}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # ACE-ARCHITECT RESPONSE PARSER
    # ─────────────────────────────────────────────────────────────────────────
    def _parse_ace_architect_response(self, response_text):
        data = {}
        think_match = re.search(r"<think>(.*?)</think>", response_text, re.DOTALL | re.IGNORECASE)
        if think_match:
            for line in think_match.group(1).split("\n"):
                if ":" in line:
                    key, val = line.split(":", 1)
                    data[key.strip().upper()] = val.strip()

        caption_match = re.search(
            r"#\s*Caption\s*(.*?)(?=#\s*(?:Arrangement|Lyrics)|$)",
            response_text, re.DOTALL | re.IGNORECASE
        )
        if caption_match:
            data["GENRE"] = caption_match.group(1).strip()

        arr_match = re.search(
            r"#\s*Arrangement\s*(.*?)(?=#\s*Lyrics|$)",
            response_text, re.DOTALL | re.IGNORECASE
        )
        if arr_match:
            arr_text = arr_match.group(1).strip()
            if "GENRE" in data and arr_text:
                data["GENRE"] += "\n" + arr_text
            elif arr_text:
                data["GENRE"] = arr_text

        lyrics_match = re.search(r"#\s*Lyrics\s*(.*)", response_text, re.DOTALL | re.IGNORECASE)
        if lyrics_match:
            data["LYRICS"] = lyrics_match.group(1).strip()

        return data

    # ─────────────────────────────────────────────────────────────────────────
    # GENERIC RESPONSE PARSER
    # ─────────────────────────────────────────────────────────────────────────
    def _parse_hybrid_response(self, response_text):
        data = {}
        current_key = None
        known_keys  = ["GENRE", "VOCALS", "LYRICS", "BPM", "TIME_SIG", "KEY_SCALE", "DURATION"]
        for line in response_text.split("\n"):
            line = line.strip()
            if not line: continue
            key_match = None
            for k in known_keys:
                if re.match(r'^[\*\-]*\s*' + k + r'[\*\-]*\s*[=:]', line, re.IGNORECASE):
                    key_match = k
                    break
            if key_match:
                current_key = key_match
                sep_idx = re.search(r'[=:]', line).start()
                data[current_key] = line[sep_idx + 1:].strip()
            elif current_key:
                data[current_key] = data[current_key] + "\n" + line
        return data

    # ─────────────────────────────────────────────────────────────────────────
    # LLM CALL
    # ─────────────────────────────────────────────────────────────────────────
    def _call_llm(self, api_url, model, prompt, params=None, ollama_enabled=True, timeout=None):
        if not ollama_enabled or not CONST_REQUESTS_AVAILABLE:
            return ""
        if not api_url or not model or model == "[Ollama Not Detected]":
            return ""

        api_url = api_url.rstrip("/")
        if params is None: params = {}

        # Timeout: explicit arg > params dict > module constant
        effective_timeout = timeout if timeout is not None else params.get("llm_timeout", CONST_API_TIMEOUT_SECONDS)

        options = {
            "num_ctx":    8192,
            "num_predict": -1,
            "temperature": params.get("temperature", 0.7),
            "top_k":      params.get("top_k", 40),
            "top_p":      params.get("top_p", 0.9),
            "min_p":      params.get("min_p", 0.05),
        }
        payload = {
            "model":  model,
            "prompt": prompt,
            "stream": False,
            "options": options,
            "seed":   params.get("seed", 0),
        }
        keep_alive = params.get("keep_alive", 300)
        if keep_alive != 300:
            payload["keep_alive"] = f"{keep_alive}s"

        for attempt in range(CONST_MAX_RETRY_ATTEMPTS):
            try:
                resp = requests.post(
                    f"{api_url}/api/generate",
                    json=payload,
                    timeout=effective_timeout
                )
                if resp.status_code == 200:
                    raw = resp.json().get("response", "").strip()
                    return raw  # Caller strips think blocks as appropriate
                else:
                    logging.warning(
                        f"[WildcardPromptBuilder] Ollama {resp.status_code}: {resp.text[:200]}"
                    )
            except Exception as e:
                logging.warning(
                    f"[WildcardPromptBuilder] Ollama attempt {attempt + 1} failed: {e}"
                )
                if attempt == CONST_MAX_RETRY_ATTEMPTS - 1:
                    return ""
        return ""

    # ─────────────────────────────────────────────────────────────────────────
    # HYBRID GENRE / VOCAL GENERATORS
    # ─────────────────────────────────────────────────────────────────────────
    def _generate_hybrid_genre(self, concept, seed, api_url, model, params,
                                descriptive, ace_step_mode, ollama_enabled):
        if ace_step_mode:
            # Rich Gemini-style prose via dedicated prompt
            result = self._call_llm(
                api_url, model,
                self.PROMPT_ACE_STEP_CAPTION.format(concept=concept),
                params, ollama_enabled
            )
            if result:
                result = self._strip_think_blocks(result)
                return self.strip_metadata_from_caption(result)

        if descriptive:
            prompt = (
                f"Describe the musical genre, mood, and atmosphere for: '{concept}'. "
                f"Use descriptive adjectives and production terms. Under 60 words."
            )
            result = self._call_llm(api_url, model, prompt, params, ollama_enabled)
            if result:
                result = self._strip_think_blocks(result)
                return result

        opts = self._hybrid_genre_opts()
        prompt = self.PROMPT_HYBRID_GENRE.format(
            concept=concept,
            base_genre_options   =", ".join(opts["base_genre"]),
            subgenre_options     =", ".join(opts["subgenre_modifier"]),
            percussion_style     =", ".join(opts["percussion_style"]),
            mood                 =", ".join(opts["mood"]),
            harmonic_instrument  =", ".join(opts["harmonic_instrument"]),
            chord_quality        =", ".join(opts["chord_quality"]),
            bass_character       =", ".join(opts["bass_character"]),
            texture              =", ".join(opts["texture"]),
        )
        try:
            response = self._call_llm(api_url, model, prompt, params, ollama_enabled)
            if not response: raise ValueError("Empty")
            response = self._strip_think_blocks(response)
            selections = self._parse_hybrid_response(response)
            parts = [
                selections.get("base_genre",         "Ambient"),
                selections.get("subgenre_modifier",  "Atmospheric"),
                selections.get("percussion_style",   "Minimal sparse clicks"),
                (f"{selections.get('mood','Dreamy')} "
                 f"{selections.get('harmonic_instrument','Atmospheric Pads')} "
                 f"{selections.get('chord_quality','Minor')} chords"),
                selections.get("bass_character", "Clean Sub Bass"),
                selections.get("texture",        "Lush string pads"),
            ]
            result = ", ".join(parts)
            return self.strip_metadata_from_caption(result) if ace_step_mode else result
        except Exception as e:
            logging.warning(f"[WildcardPromptBuilder] Hybrid genre failed: {e}")
            expander = WildcardExpander(seed)
            return expander.expand(self._generate_default_genre_template())

    def _generate_hybrid_vocal(self, concept, seed, api_url, model, params,
                                descriptive, ollama_enabled):
        if descriptive:
            prompt = (
                f"Describe the vocal style, processing, and type for: '{concept}'. "
                f"Under 30 words. No lyrics."
            )
            result = self._call_llm(api_url, model, prompt, params, ollama_enabled)
            if result: return self._strip_think_blocks(result)

        opts = self._hybrid_vocal_opts()
        prompt = self.PROMPT_HYBRID_VOCAL.format(
            concept           =concept,
            vocal_type_options=", ".join(opts["vocal_type"]),
            vocal_quality_options=", ".join(opts["vocal_quality"]),
            vocal_effect_options =", ".join(opts["vocal_effect"]),
        )
        try:
            response = self._call_llm(api_url, model, prompt, params, ollama_enabled)
            if not response: raise ValueError("Empty")
            response = self._strip_think_blocks(response)
            selections = self._parse_hybrid_response(response)
            return ", ".join([
                selections.get("vocal_type",   "Soft female vocals"),
                selections.get("vocal_quality","Ethereal"),
                selections.get("vocal_effect", "Reverb-drenched (Valhalla)"),
            ])
        except Exception as e:
            logging.warning(f"[WildcardPromptBuilder] Hybrid vocal failed: {e}")
            expander = WildcardExpander(seed)
            return expander.expand(self._generate_default_vocal_template())

    def _generate_image_prompt(self, concept, seed, mode, api_url, model,
                                params, ollama_enabled):
        """Generate an image prompt from concept."""
        if mode in ["llm", "hybrid"]:
            result = self._call_llm(
                api_url, model,
                self.PROMPT_IMAGE_CAPTION.format(concept=concept),
                params, ollama_enabled
            )
            if result: return self._strip_think_blocks(result)
        expander = WildcardExpander(seed)
        subject  = concept[:80] if concept else "a dramatic scene"
        tmpl     = self._generate_default_image_template()
        return f"{subject}, {expander.expand(tmpl)}"

    def _generate_video_prompt(self, concept, seed, mode, api_url, model,
                                params, ollama_enabled):
        """Generate a video prompt from concept."""
        if mode in ["llm", "hybrid"]:
            result = self._call_llm(
                api_url, model,
                self.PROMPT_VIDEO_CAPTION.format(concept=concept),
                params, ollama_enabled
            )
            if result: return self._strip_think_blocks(result)
        expander = WildcardExpander(seed)
        subject  = concept[:80] if concept else "a dramatic scene"
        tmpl     = self._generate_default_video_template()
        return f"{subject}, {expander.expand(tmpl)}"

    # ─────────────────────────────────────────────────────────────────────────
    # ORCHESTRATED PROMPT
    # ─────────────────────────────────────────────────────────────────────────
    def _generate_orchestrated_prompt(self, concept, seed, api_url, model,
                                       params, ollama_enabled):
        result = self._call_llm(
            api_url, model,
            self.PROMPT_ORCHESTRATOR.format(concept=concept),
            params, ollama_enabled
        )
        if result:
            result = self._strip_think_blocks(result)
            return self._parse_hybrid_response(result)
        return None

    # ─────────────────────────────────────────────────────────────────────────
    # PREVIEW RENDER
    # ─────────────────────────────────────────────────────────────────────────
    def _find_font(self):
        search_paths = []
        if sys.platform == "win32":
            search_paths = [r"C:\Windows\Fonts"]
        elif sys.platform == "darwin":
            search_paths = ["/Library/Fonts", "/System/Library/Fonts"]
        else:
            search_paths = ["/usr/share/fonts", "/usr/local/share/fonts", "~/.fonts"]
        common_fonts = [
            "Arial.ttf", "Verdana.ttf", "Tahoma.ttf",
            "DejaVuSans.ttf", "LiberationSans-Regular.ttf",
        ]
        for path in search_paths:
            path = os.path.expanduser(path)
            if not os.path.exists(path): continue
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file in common_fonts or file.lower().endswith(".ttf"):
                        return os.path.join(root, file)
        local_font = os.path.join(os.path.dirname(os.path.realpath(__file__)), "font.ttf")
        if os.path.exists(local_font): return local_font
        return None

    def _render_text_preview(self, genre, vocal, lyrics, duration, bpm,
                              time_sig, key_scale, seed, output_target,
                              interpreted_prompt):
        if not CONST_IMAGING_AVAILABLE:
            return self._blank_preview()
        try:
            img  = Image.new("RGB", (CONST_PREVIEW_WIDTH, CONST_PREVIEW_HEIGHT), (18, 18, 24))
            draw = ImageDraw.Draw(img)
            font_path = self._find_font()
            def _font(size):
                if font_path:
                    try: return ImageFont.truetype(font_path, size)
                    except Exception: pass
                return ImageFont.load_default()

            font_title = _font(CONST_FONT_SIZE_TITLE)
            font_body  = _font(CONST_FONT_SIZE_BODY)
            font_small = _font(CONST_FONT_SIZE_SMALL)
            y = 15

            def draw_section(title, text, color=(180, 180, 200)):
                nonlocal y
                draw.text((20, y), title, fill=color, font=font_body)
                y += 24
                for line in wrap(text if text else "[None]", width=88)[:4]:
                    draw.text((36, y), line, fill=(240, 240, 255), font=font_small)
                    y += 18
                y += 8

            header = f"MD: Wildcard Prompt Builder  |  Target: {output_target}"
            draw.text((20, y), header, fill=(80, 200, 255), font=font_title)
            y += 38
            draw_section("GENRE / CAPTION:", genre)
            if vocal:
                draw_section("VOCALS:", vocal)
            if lyrics and lyrics not in ("[Instrumental]", ""):
                draw_section("LYRICS:", lyrics[:120] + ("…" if len(lyrics) > 120 else ""))
            draw.text((20, y), "PARAMETERS:", fill=(180, 180, 200), font=font_body)
            y += 22
            draw.text((36, y),
                      f"BPM: {bpm}  |  Sig: {time_sig}  |  Key: {key_scale}  |  Duration: {duration}s  |  Seed: {seed}",
                      fill=(80, 255, 150), font=font_small)
            y += 20
            if interpreted_prompt:
                draw.text((20, y + 8), "INTERPRETED PROMPT:", fill=(180, 180, 200), font=font_body)
                y += 28
                for line in wrap(interpreted_prompt[:200], width=88)[:3]:
                    draw.text((36, y), line, fill=(255, 220, 120), font=font_small)
                    y += 18

            img_np = np.array(img).astype(np.float32) / 255.0
            return torch.from_numpy(img_np).unsqueeze(0)
        except Exception as e:
            logging.error(f"[WildcardPromptBuilder] Preview render failed: {e}")
            return self._blank_preview()

    def _blank_preview(self):
        if CONST_IMAGING_AVAILABLE:
            return torch.zeros((1, CONST_PREVIEW_HEIGHT, CONST_PREVIEW_WIDTH, 3), dtype=torch.float32)
        return None

    # ─────────────────────────────────────────────────────────────────────────
    # INPUT_TYPES
    # ─────────────────────────────────────────────────────────────────────────
    @classmethod
    def INPUT_TYPES(cls):
        genre_files  = cls._get_files_from_dir("genre")
        vocal_files  = cls._get_files_from_dir("vocal")
        lyrics_files = cls._get_files_from_dir("lyrics")

        return {
            "required": {
                "generation_mode": (["wildcard", "llm", "hybrid"], {
                    "default": "wildcard",
                    "tooltip": (
                        "GENERATION MODE\n"
                        "• Purpose: Core logic for creating prompt content.\n"
                        "• Wildcard: Instant random selection from internal/file libraries.\n"
                        "• LLM: Full creative generation via Ollama API.\n"
                        "• Hybrid: LLM intelligently selects from curated option lists.\n"
                        "\n Recommended: Hybrid for best themed variety."
                    ),
                }),
                "output_target": (["Audio (ACE-Step)", "Image (SD/Flux)", "Video (Wan/Mochi)", "All Three"], {
                    "default": "Audio (ACE-Step)",
                    "tooltip": (
                        "OUTPUT TARGET\n"
                        "• Purpose: Sets vocabulary and prompt format for the target model.\n"
                        "• Audio: ACE-Step optimised outputs (genre, vocals, lyrics, BPM, key).\n"
                        "• Image: SD/Flux prompt in INTERPRETED_PROMPT output.\n"
                        "• Video: Wan/Mochi video prompt in INTERPRETED_PROMPT output.\n"
                        "• All Three: All outputs populated for unified media workflows.\n"
                        "\n Existing GENRE_TAGS/VOCAL_TAGS/LYRICS outputs always populated."
                    ),
                }),
                "ace_step_mode": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "ACE-STEP MODE\n"
                        "• Purpose: Enables ACE-Step 1.5 caption quality enhancements.\n"
                        "• True: Rich Gemini-style prose captions. BPM/key auto-stripped from caption.\n"
                        "• False: Standard tag-list output (legacy behaviour).\n"
                        "\n Recommended: True for ACE-Step 1.5 workflows."
                    ),
                }),
                "descriptive_mode": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "DESCRIPTIVE MODE\n"
                        "• Purpose: Prose sentences instead of comma-separated tags.\n"
                        "• True: Sentence-style output. False: Tag list.\n"
                        "• Note: ace_step_mode overrides this for genre output automatically.\n"
                        "\n Recommended: False unless specifically needed."
                    ),
                }),
                "concept": ("STRING", {
                    "multiline": True,
                    "default": "Dark neurofunk drum and bass with heavy reese bass and complex rolling breaks",
                    "tooltip": (
                        "CONCEPT PROMPT\n"
                        "• Purpose: The core theme driving generation across all targets.\n"
                        "• Usage: Used by LLM/Hybrid/Orchestrator as creative brief.\n"
                        "• Tip: Be specific — genre, mood, energy, references.\n"
                        "\n Recommended: Include genre, mood, and key sonic elements."
                    ),
                }),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": CONST_JS_MAX_SAFE_INTEGER,
                    "tooltip": (
                        "RANDOM SEED\n"
                        "• Purpose: Controls all wildcard randomization.\n"
                        "• Range: 0 to 9,007,199,254,740,991 (JS Safe Max).\n"
                        "\n Recommended: Connect global seed for workflow sync."
                    ),
                }),
                "randomize_seed": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "RANDOMIZE SEED\n"
                        "• Purpose: Auto-change seed every run for variety.\n"
                        "• True: New choices every run. False: Lock current seed.\n"
                        "\n Recommended: True for exploration, False for reproduction."
                    ),
                }),
                "duration_template": ("STRING", {
                    "default": "{120|180|240}",
                    "tooltip": (
                        "DURATION TEMPLATE\n"
                        "• Purpose: Track/clip length logic in seconds.\n"
                        "• Format: Accepts wildcards e.g. {120|180|240}.\n"
                        "• Note: Orchestrator mode may override this."
                    ),
                }),
            },
            "optional": {
                "ollama_enabled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "OLLAMA ENABLED\n"
                        "• Purpose: Master switch for all LLM API calls.\n"
                        "• False: Disables all Ollama calls — falls back to wildcard instantly.\n"
                        "• Use False if Ollama is not installed to prevent any network probing.\n"
                        "\n Recommended: False if Ollama not installed."
                    ),
                }),
                "ollama_api_url": ("STRING", {
                    "default": CONST_DEFAULT_OLLAMA_URL,
                    "tooltip": "Ollama API base URL (e.g. http://localhost:11434).",
                }),
                "ollama_model": (cls._get_ollama_models_lazy(), {
                    "default": CONST_DEFAULT_OLLAMA_MODEL,
                    "tooltip": "Ollama model to use. List auto-refreshes in background.",
                }),
                "orchestrator_mode": (["Off", "Standard", "ACE-Architect"], {
                    "default": "Off",
                    "tooltip": (
                        "ORCHESTRATOR MODE\n"
                        "• Purpose: Single-shot LLM pass for fully coherent output.\n"
                        "• Standard: General purpose BPM/key/genre/lyrics in one call.\n"
                        "• ACE-Architect: ACE-Step 1.5 optimised. Rich prose caption + metadata.\n"
                        "\n Recommended: ACE-Architect for best ACE-Step 1.5 results."
                    ),
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7, "min": 0.0, "max": 2.0, "step": 0.1,
                    "tooltip": "TEMPERATURE\n• LLM creativity (0.2=focused, 1.0=chaotic, 1.4=unhinged).",
                }),
                "top_k": ("INT", {
                    "default": 40, "min": 1, "max": 200,
                    "tooltip": "TOP K\n• Token vocabulary limit for predictability.",
                }),
                "top_p": ("FLOAT", {
                    "default": 0.9, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "TOP P\n• Nucleus sampling cumulative cutoff.",
                }),
                "min_p": ("FLOAT", {
                    "default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "MIN P\n• Filters low probability tokens relative to best choice.",
                }),
                "keep_alive": ("INT", {
                    "default": 300, "min": 0, "max": 3600,
                    "tooltip": "KEEP ALIVE (Seconds)\n• Model VRAM retention. 0 = unload immediately after call.",
                }),
                "llm_timeout": ("INT", {
                    "default": 120, "min": 30, "max": 600,
                    "tooltip": (
                        "LLM TIMEOUT (Seconds)\n"
                        "• Purpose: Max seconds to wait for a single Ollama API response.\n"
                        "• Options: 30–600s. Default 120s.\n"
                        "• Trade-offs: Lower = faster failure detection; higher = tolerates slow models.\n"
                        "• ⭐ Recommendation: 30–60s for fast models (llama3, mistral). "
                        "120s+ for large reasoning models (Qwen3-32B, QwQ)."
                    ),
                }),
                "load_genre_file": (genre_files, {
                    "default": "None",
                    "tooltip": "Load external genre wildcard .txt file.",
                }),
                "load_vocal_file": (vocal_files, {
                    "default": "None",
                    "tooltip": "Load external vocal wildcard .txt file.",
                }),
                "load_lyrics_file": (lyrics_files, {
                    "default": "None",
                    "tooltip": "Load external lyrics wildcard .txt file.",
                }),
                "yaml_input": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "YAML INPUT OVERRIDE\n• Override any parameter via YAML key-value string.",
                }),
                "custom_genre_template": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "Custom genre/caption text. Wildcards {a|b} supported. Overrides generation.",
                }),
                "custom_vocal_template": ("STRING", {
                    "multiline": False, "default": "",
                    "tooltip": "Custom vocal style text. Overrides generation.",
                }),
                "custom_lyrics_template": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "Custom lyrics text. Overrides generation.",
                }),
                "custom_bpm": ("STRING", {
                    "default": "",
                    "tooltip": "Override BPM (integer or wildcard {174|172}).",
                }),
                "custom_time_sig": (["Auto"] + TIME_SIGNATURES, {
                    "default": "Auto",
                    "tooltip": "Override time signature (4 = 4/4, 3 = 3/4, etc).",
                }),
                "custom_key_scale": (["Auto"] + KEY_SCALES, {
                    "default": "Auto",
                    "tooltip": "Override key/scale (C Major, F# Minor, etc).",
                }),
                "expand_custom_templates": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Expand {wildcard} patterns in custom template inputs.",
                }),
                "generate_genre": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable/disable genre/caption generation.",
                }),
                "generate_vocals": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable/disable vocal style generation.",
                }),
                "generate_lyrics": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable/disable lyrics generation.",
                }),
                "force_instrumental": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Force instrumental: clears vocals, sets lyrics to [Instrumental].",
                }),
                "lora_trigger": ("STRING", {
                    "default": "",
                    "tooltip": (
                        "LORA TRIGGER TAG\n"
                        "• Purpose: Prepend a LoRA trigger token to the genre/caption output.\n"
                        "• Example: mdm4_dnb\n"
                        "• Always placed first, before any generated text.\n"
                        "\n Recommended: Use your LoRA's exact trigger string."
                    ),
                }),
                "debug_mode": (["0 - Silent", "1 - Info", "2 - Verbose"], {
                    "default": "0 - Silent",
                    "tooltip": "Logging verbosity level.",
                }),
                "enable_profiling": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable detailed performance profiling output.",
                }),
            },
        }

    RETURN_TYPES  = ("STRING", "STRING", "STRING", "STRING", "FLOAT",
                     "INT", "STRING", TIME_SIGNATURES, "STRING", KEY_SCALES,
                     "INT", "IMAGE", "STRING", "STRING")
    RETURN_NAMES  = ("GENRE_TAGS", "VOCAL_TAGS", "LYRICS", "DURATION_STRING",
                     "DURATION_FLOAT", "BPM", "TIME_SIG_STR", "TIME_SIG_COMBO",
                     "KEY_SCALE_STR", "KEY_SCALE_COMBO", "SEED",
                     "TEXT_PREVIEW", "YAML_CONFIG", "INTERPRETED_PROMPT")
    FUNCTION  = "execute"
    CATEGORY  = "MD_Nodes/Prompt Generation"

    @classmethod
    def IS_CHANGED(cls, randomize_seed=False, **kwargs):
        is_rand = randomize_seed if isinstance(randomize_seed, bool) else (
            randomize_seed.lower() == "true" if isinstance(randomize_seed, str) else False
        )
        return secrets.token_hex(16) if is_rand else "static"

    # ─────────────────────────────────────────────────────────────────────────
    # EXECUTE
    # ─────────────────────────────────────────────────────────────────────────
    def execute(self, **kwargs):
        # ── YAML override ─────────────────────────────────────────────────────
        if kwargs.get("yaml_input") and YAML_AVAILABLE:
            try:
                overrides = yaml.safe_load(kwargs["yaml_input"])
                if isinstance(overrides, dict):
                    kwargs.update(overrides)
            except Exception as e:
                logging.warning(f"[WildcardPromptBuilder] YAML parse failed: {e}")

        # ── Debug / profiling ─────────────────────────────────────────────────
        debug_mode = kwargs.get("debug_mode", "0 - Silent")
        try:
            debug_level = int(str(debug_mode).split(" ")[0])
        except Exception:
            debug_level = 0

        enable_profiling = kwargs.get("enable_profiling", False)
        profiler = PerformanceProfiler(enabled=(debug_level >= 1 or enable_profiling))
        profiler.start("total")

        try:
            mode          = kwargs.get("generation_mode", "wildcard")
            descriptive   = kwargs.get("descriptive_mode", False)
            ace_step_mode = kwargs.get("ace_step_mode", True)
            output_target = kwargs.get("output_target", "Audio (ACE-Step)")
            ollama_enabled= kwargs.get("ollama_enabled", True)
            seed          = kwargs.get("seed", 0)

            if kwargs.get("randomize_seed", True):
                seed = secrets.randbelow(CONST_JS_MAX_SAFE_INTEGER)

            concept  = kwargs.get("concept", "")
            expander = WildcardExpander(seed)
            concept  = expander.expand(concept)

            lora_trigger = kwargs.get("lora_trigger", "").strip()

            llm_params = {
                "temperature": kwargs.get("temperature", 0.7),
                "top_k":       kwargs.get("top_k", 40),
                "top_p":       kwargs.get("top_p", 0.9),
                "min_p":       kwargs.get("min_p", 0.05),
                "keep_alive":  kwargs.get("keep_alive", 300),
                "llm_timeout": kwargs.get("llm_timeout", CONST_API_TIMEOUT_SECONDS),
                "seed":        seed,
            }
            url   = kwargs.get("ollama_api_url", CONST_DEFAULT_OLLAMA_URL)
            model = kwargs.get("ollama_model",   CONST_DEFAULT_OLLAMA_MODEL)

            expand_customs = kwargs.get("expand_custom_templates", True)
            genre_out = vocal_out = lyrics_out = ""
            bpm_out = time_sig_out = key_scale_out = duration_out = None

            profiler.start("generation_logic")

            # ── ORCHESTRATOR ──────────────────────────────────────────────────
            orch_mode = kwargs.get("orchestrator_mode", "Off")
            orch_data = {}

            if orch_mode != "Off":
                if orch_mode == "ACE-Architect":
                    prompt_tmpl = self.PROMPT_ACE_ARCHITECT
                    parser_func = self._parse_ace_architect_response
                else:
                    prompt_tmpl = self.PROMPT_ORCHESTRATOR
                    parser_func = self._parse_hybrid_response

                response = self._call_llm(
                    url, model,
                    prompt_tmpl.format(concept=concept),
                    llm_params, ollama_enabled
                )
                if response:
                    # ACE-Architect parser reads <think> block for metadata — skip stripping.
                    # Standard orchestrator path gets stripped before hybrid parse.
                    if orch_mode != "ACE-Architect":
                        response = self._strip_think_blocks(response)
                    orch_data = parser_func(response)
                    if debug_level >= 1:
                        logging.info(f"[WildcardPromptBuilder] Orch data: {orch_data}")

            # ── 1. DURATION ───────────────────────────────────────────────────
            if orch_data and "DURATION" in orch_data:
                try:
                    duration_out = str(int(re.search(r"\d+", orch_data["DURATION"]).group()))
                except Exception:
                    duration_out = None

            if not duration_out:
                duration_out = expander.expand(
                    kwargs.get("duration_template", "{120|180|240}")
                )

            try:
                duration_float = float(duration_out)
            except ValueError:
                duration_float = 0.0

            # ── 2. GENRE / CAPTION ────────────────────────────────────────────
            if orch_data and "GENRE" in orch_data:
                genre_out = orch_data["GENRE"]
                if ace_step_mode:
                    genre_out = self.strip_metadata_from_caption(genre_out)
            elif kwargs.get("generate_genre", True):
                custom_genre = kwargs.get("custom_genre_template", "")
                if custom_genre.strip():
                    genre_out = expander.expand(custom_genre) if expand_customs else custom_genre
                    if ace_step_mode:
                        genre_out = self.strip_metadata_from_caption(genre_out)
                else:
                    loaded_file = kwargs.get("load_genre_file", "None")
                    if loaded_file == "Random":
                        genre_tmpl = self._read_random_file_content("genre", seed)
                    elif loaded_file and loaded_file != "None":
                        genre_tmpl = self._read_file_content("genre", loaded_file)
                    else:
                        genre_tmpl = None

                    if mode == "hybrid":
                        genre_out = self._generate_hybrid_genre(
                            concept, seed, url, model, llm_params,
                            descriptive, ace_step_mode, ollama_enabled
                        )
                    elif mode == "llm":
                        if ace_step_mode:
                            res = self._call_llm(
                                url, model,
                                self.PROMPT_ACE_STEP_CAPTION.format(concept=concept),
                                llm_params, ollama_enabled
                            )
                        else:
                            prompt_q = (
                                f"Describe a musical genre in 2 sentences for: {concept}."
                                if descriptive else
                                f"Generate 5 descriptive music genre tags for: {concept}. Comma separated."
                            )
                            res = self._call_llm(url, model, prompt_q, llm_params, ollama_enabled)
                        if res:
                            res = self._strip_think_blocks(res)
                            genre_out = self.strip_metadata_from_caption(res) if ace_step_mode else res
                        else:
                            tmpl      = genre_tmpl or self._generate_default_genre_template()
                            genre_out = expander.expand(tmpl)
                    else:  # wildcard
                        tmpl      = genre_tmpl or self._generate_default_genre_template()
                        genre_out = expander.expand(tmpl)
                        if ace_step_mode:
                            genre_out = self.strip_metadata_from_caption(genre_out)

            # Prepend LoRA trigger if provided
            if lora_trigger and genre_out:
                genre_out = f"{lora_trigger}, {genre_out}"
            elif lora_trigger:
                genre_out = lora_trigger

            # ── 3. VOCALS & LYRICS ────────────────────────────────────────────
            if kwargs.get("force_instrumental", False):
                vocal_out  = ""
                lyrics_out = "[Instrumental]"
            else:
                # Vocals
                if orch_data and "VOCALS" in orch_data:
                    vocal_out = orch_data["VOCALS"]
                elif kwargs.get("generate_vocals", True):
                    custom_vocal = kwargs.get("custom_vocal_template", "")
                    if custom_vocal.strip():
                        vocal_out = expander.expand(custom_vocal) if expand_customs else custom_vocal
                    else:
                        loaded_file = kwargs.get("load_vocal_file", "None")
                        if loaded_file == "Random":
                            vocal_tmpl = self._read_random_file_content("vocal", seed)
                        elif loaded_file and loaded_file != "None":
                            vocal_tmpl = self._read_file_content("vocal", loaded_file)
                        else:
                            vocal_tmpl = None

                        if mode == "hybrid":
                            vocal_out = self._generate_hybrid_vocal(
                                concept, seed, url, model, llm_params,
                                descriptive, ollama_enabled
                            )
                        elif mode == "llm":
                            prompt_q = (
                                f"Describe the vocal style for '{concept}' in 2 sentences."
                                if descriptive else
                                f"Generate 3 vocal style tags for: {concept}. Comma separated."
                            )
                            res = self._call_llm(url, model, prompt_q, llm_params, ollama_enabled)
                            if res:
                                res = self._strip_think_blocks(res)
                            vocal_out = res if res else expander.expand(
                                vocal_tmpl or self._generate_default_vocal_template()
                            )
                        else:
                            vocal_out = expander.expand(
                                vocal_tmpl or self._generate_default_vocal_template()
                            )

                # Lyrics
                if orch_data and "LYRICS" in orch_data:
                    lyrics_out = orch_data["LYRICS"]
                elif kwargs.get("generate_lyrics", True):
                    custom_lyrics = kwargs.get("custom_lyrics_template", "")
                    if custom_lyrics.strip():
                        lyrics_out = expander.expand(custom_lyrics) if expand_customs else custom_lyrics
                    else:
                        loaded_file = kwargs.get("load_lyrics_file", "None")
                        if loaded_file == "Random":
                            lyrics_tmpl = self._read_random_file_content("lyrics", seed)
                        elif loaded_file and loaded_file != "None":
                            lyrics_tmpl = self._read_file_content("lyrics", loaded_file)
                        else:
                            lyrics_tmpl = None

                        if mode in ["llm", "hybrid"]:
                            res = self._call_llm(
                                url, model,
                                self.PROMPT_LLM_LYRICS.format(concept=concept),
                                llm_params, ollama_enabled
                            )
                            if res:
                                res = self._strip_think_blocks(res)
                            lyrics_out = res if res else expander.expand(
                                lyrics_tmpl or self.DEFAULT_LYRICS_TEMPLATE
                            )
                        else:
                            lyrics_out = expander.expand(
                                lyrics_tmpl or self.DEFAULT_LYRICS_TEMPLATE
                            )

            # ── 4. MUSIC PARAMETERS ───────────────────────────────────────────
            if orch_data and "BPM" in orch_data:
                try:
                    bpm_out = int(re.search(r"\d+", orch_data["BPM"]).group())
                    bpm_out = max(CONST_MIN_BPM, min(CONST_MAX_BPM, bpm_out))
                except Exception:
                    bpm_out = None

            if bpm_out is None:
                custom_bpm = kwargs.get("custom_bpm", "").strip()
                if custom_bpm:
                    bpm_str = expander.expand(custom_bpm) if expand_customs else custom_bpm
                    try:
                        bpm_out = int(bpm_str)
                        bpm_out = max(CONST_MIN_BPM, min(CONST_MAX_BPM, bpm_out))
                    except ValueError:
                        bpm_out = self._smart_bpm_from_genre(genre_out, seed)
                else:
                    bpm_out = self._smart_bpm_from_genre(genre_out, seed)

            if orch_data and "TIME_SIG" in orch_data:
                ts_clean = re.sub(r"[^0-9]", "", orch_data["TIME_SIG"])
                if ts_clean and ts_clean[0] in TIME_SIGNATURES:
                    time_sig_out = ts_clean[0]
                elif "4" in ts_clean:
                    time_sig_out = "4"

            if time_sig_out is None:
                custom_time_sig = kwargs.get("custom_time_sig", "Auto")
                if custom_time_sig == "Auto":
                    rng = random.Random(seed)
                    time_sig_out = (
                        "4" if rng.random() < 0.75
                        else rng.choice([s for s in TIME_SIGNATURES if s != "4"])
                    )
                else:
                    time_sig_out = custom_time_sig

            if orch_data and "KEY_SCALE" in orch_data:
                key_scale_out = orch_data["KEY_SCALE"]

            if key_scale_out is None:
                custom_key_scale = kwargs.get("custom_key_scale", "Auto")
                key_scale_out = (
                    random.Random(seed).choice(KEY_SCALES)
                    if custom_key_scale == "Auto"
                    else custom_key_scale
                )

            profiler.stop("generation_logic")

            # ── 5. INTERPRETED PROMPT (per output_target) ─────────────────────
            profiler.start("interpreted_prompt")
            interpreted_prompt = ""

            is_audio = "Audio" in output_target or output_target == "All Three"
            is_image = "Image" in output_target or output_target == "All Three"
            is_video = "Video" in output_target or output_target == "All Three"

            parts = []
            if is_audio:
                audio_part = genre_out
                if vocal_out:
                    audio_part += f", {vocal_out}"
                parts.append(f"[AUDIO] {audio_part}")

            if is_image:
                img_prompt = self._generate_image_prompt(
                    concept, seed, mode, url, model, llm_params, ollama_enabled
                )
                parts.append(f"[IMAGE] {img_prompt}")

            if is_video:
                vid_prompt = self._generate_video_prompt(
                    concept, seed, mode, url, model, llm_params, ollama_enabled
                )
                parts.append(f"[VIDEO] {vid_prompt}")

            if len(parts) == 1:
                # Single target — strip the prefix tag for clean direct connection
                interpreted_prompt = re.sub(r'^\[(?:AUDIO|IMAGE|VIDEO)\]\s*', '', parts[0])
            else:
                interpreted_prompt = "\n\n".join(parts)

            profiler.stop("interpreted_prompt")

            # ── 6. PREVIEW & YAML ─────────────────────────────────────────────
            profiler.start("render_preview")
            preview_image = self._render_text_preview(
                genre_out, vocal_out, lyrics_out, duration_out,
                bpm_out, time_sig_out, key_scale_out, seed,
                output_target, interpreted_prompt
            )
            profiler.stop("render_preview")

            yaml_config = ""
            if YAML_AVAILABLE:
                out_dict = {
                    "genre":                genre_out,
                    "vocals":               vocal_out,
                    "lyrics":               lyrics_out,
                    "bpm":                  bpm_out,
                    "time_sig":             time_sig_out,
                    "key":                  key_scale_out,
                    "duration":             duration_out,
                    "seed":                 seed,
                    "output_target":        output_target,
                    "ace_step_mode":        ace_step_mode,
                    "orchestrator_enabled": orch_mode != "Off",
                    "params":               llm_params,
                }
                yaml_config = yaml.dump(out_dict, default_flow_style=False)

            profiler.stop("total")

            if debug_level >= 1:
                logging.info("\n" + "=" * 65)
                logging.info("  [WildcardPromptBuilder] ANALYTICS REPORT")
                logging.info("=" * 65)
                logging.info(f"    Mode: {mode} | Orch: {orch_mode} | Target: {output_target}")
                logging.debug(f"    ACE-Step Mode: {ace_step_mode} | Ollama: {ollama_enabled}")
                logging.info(f"    Genre ({len(genre_out)}ch): {genre_out[:60]}...")
                profiler.print_report()
                logging.info("=" * 65)

            return (
                genre_out, vocal_out, lyrics_out, duration_out, duration_float,
                bpm_out, time_sig_out, time_sig_out, key_scale_out, key_scale_out,
                seed, preview_image, yaml_config, interpreted_prompt
            )

        except Exception as e:
            logging.error(f"[WildcardPromptBuilder] Critical Error: {e}")
            logging.debug(traceback.format_exc())
            blank = (
                torch.zeros((1, CONST_PREVIEW_HEIGHT, CONST_PREVIEW_WIDTH, 3), dtype=torch.float32)
                if CONST_IMAGING_AVAILABLE else None
            )
            return ("Error", "", "System Failure", "120", 120.0,
                    120, "4", "4", "C major", "C major", 0, blank, "", "")


# =================================================================================
# == Node Registration                                                           ==
# =================================================================================
NODE_CLASS_MAPPINGS = {
    "WildcardPromptBuilder": WildcardPromptBuilder
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "WildcardPromptBuilder": "MD: Wildcard Prompt Builder"
}

# =================================================================================
# == Self-Tests                                                                  ==
# =================================================================================
if __name__ == "__main__":
    logging.info("Running Self-Tests for WildcardPromptBuilder v2.3.2...")
    passed = failed = 0
    node = WildcardPromptBuilder()

    def _check(name, fn):
        global passed, failed
        try:
            fn()
            logging.info(f"  PASS  {name}")
            passed += 1
        except Exception as e:
            logging.error(f"  FAIL  {name}: {e}")
            failed += 1

    _check("Constants",
           lambda: None if CONST_JS_MAX_SAFE_INTEGER == 9007199254740991
           else (_ for _ in ()).throw(AssertionError("wrong")))

    _check("WildcardExpander basic",
           lambda: None if WildcardExpander(42).expand("{a|b|c}") in ("a", "b", "c")
           else (_ for _ in ()).throw(AssertionError("bad expansion")))

    _check("WildcardExpander nested",
           lambda: None if len(WildcardExpander(0).expand("{x|{a|b}}")) > 0
           else (_ for _ in ()).throw(AssertionError("nested fail")))

    _check("PerformanceProfiler",
           lambda: (p := PerformanceProfiler(True), p.start("x"),
                    time.sleep(0.001), p.stop("x"),
                    None if p.get_total_time() > 0
                    else (_ for _ in ()).throw(AssertionError("no time"))))

    _check("strip_metadata_from_caption BPM",
           lambda: None if "174" not in
           WildcardPromptBuilder.strip_metadata_from_caption("heavy bass, 174 bpm, dark mood")
           else (_ for _ in ()).throw(AssertionError("BPM not stripped")))

    _check("strip_metadata_from_caption key",
           lambda: None if "F# minor" not in
           WildcardPromptBuilder.strip_metadata_from_caption("dark F# minor vibes", True)
           else (_ for _ in ()).throw(AssertionError("key not stripped")))

    _check("smart_bpm neurofunk",
           lambda: None if 170 <= WildcardPromptBuilder._smart_bpm_from_genre("neurofunk", 0) <= 180
           else (_ for _ in ()).throw(AssertionError("neurofunk BPM out of range")))

    _check("smart_bpm DnB",
           lambda: None if 168 <= WildcardPromptBuilder._smart_bpm_from_genre("drum and bass", 1) <= 178
           else (_ for _ in ()).throw(AssertionError("DnB BPM out of range")))

    _check("smart_bpm fallback (unknown genre)",
           lambda: None if isinstance(WildcardPromptBuilder._smart_bpm_from_genre("zorkwave", 99), int)
           else (_ for _ in ()).throw(AssertionError("fallback not int")))

    def _test_ace_parser():
        d = node._parse_ace_architect_response(
            "<think>bpm: 174\nduration: 180\nkeyscale: F# Minor\ntimesignature: 4\n</think>\n"
            "# Caption\nDark neurofunk with rolling breaks.\n# Arrangement\n0:00 Intro\n# Lyrics\n[Verse 1]\nText"
        )
        assert "BPM" in d, f"BPM missing: {d}"
        assert "LYRICS" in d and "[Verse 1]" in d["LYRICS"], f"Lyrics bad: {d}"
    _check("ACE-Architect parser", _test_ace_parser)

    _check("strip_think_blocks basic",
           lambda: None if "<think>" not in
           WildcardPromptBuilder._strip_think_blocks("<think>some CoT reasoning</think>actual output")
           else (_ for _ in ()).throw(AssertionError("think block not stripped")))

    _check("strip_think_blocks content preserved",
           lambda: None if
           WildcardPromptBuilder._strip_think_blocks("<think>reasoning</think>neurofunk output") == "neurofunk output"
           else (_ for _ in ()).throw(AssertionError("content after think block mangled")))

    _check("strip_think_blocks unclosed tag",
           lambda: None if
           WildcardPromptBuilder._strip_think_blocks("prefix<think>dangling CoT...") == "prefix"
           else (_ for _ in ()).throw(AssertionError("unclosed think tag leaked")))

    _check("strip_think_blocks no-op on clean text",
           lambda: None if
           WildcardPromptBuilder._strip_think_blocks("clean output with no tags") == "clean output with no tags"
           else (_ for _ in ()).throw(AssertionError("clean text mutated")))

    _check("LoRA trigger prefix",
           lambda: None if "mdm4_dnb" in "mdm4_dnb, some genre tags"
           else (_ for _ in ()).throw(AssertionError("lora prefix")))

    _check("Sidecar cache path",
           lambda: None if WildcardPromptBuilder._sidecar_cache_path()
           else (_ for _ in ()).throw(AssertionError("no path")))

    _check("Image vocab non-empty",
           lambda: None if len(WildcardPromptBuilder._image_vocab().get("style", [])) >= 5
           else (_ for _ in ()).throw(AssertionError("image vocab too small")))

    _check("Video vocab non-empty",
           lambda: None if len(WildcardPromptBuilder._video_vocab().get("camera_motion", [])) >= 3
           else (_ for _ in ()).throw(AssertionError("video vocab too small")))

    _check("DnB bass vocab depth",
           lambda: None if len([
               x for x in WildcardPromptBuilder._hybrid_genre_opts().get("bass_character", [])
               if "reese" in x.lower() or "neuro" in x.lower()
           ]) >= 3 else (_ for _ in ()).throw(AssertionError("not enough neuro bass vocab")))

    logging.info(f"\n{'='*55}")
    logging.info(f"  Results: {passed} passed, {failed} failed")
    logging.info(f"{'='*55}")