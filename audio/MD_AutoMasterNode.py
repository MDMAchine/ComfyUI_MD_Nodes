# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# █▓▒░                                                                     ░▒▓█
# █▓▒░         MD_Nodes/AudioAutoMasterPro – v6.32.0 (Enterprise)          ░▒▓█
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
# ║ ░▒▓ DESCRIPTION:
# ║   The ultimate AI-assisted mastering chain wrapper. Manages YAML loading,
# ║   local Ollama vision/text analysis, and parameter resolution before handing
# ║   execution off to the compiled DSP core.
# ║   NOTE: This is a public wrapper. Missing binaries will gracefully pass 
# ║   audio through unchanged.
# ╚════════════════════════════════════════════════════════════════════════════


VERSION = "v6.32.0"  # UPS v1.5.8

import io, os, sys, json, time, logging, requests, yaml, base64
import torch, numpy as np
from PIL import Image

# =================================================================================
# == Dependency Fallback Pattern
# =================================================================================

import logging
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    import pyloudnorm as pln
    PYLOUDNORM_AVAILABLE = True
except ImportError:
    PYLOUDNORM_AVAILABLE = False

# =================================================================================
# == MD_Nodes Universal Binary Loader (v1.6.1)
# =================================================================================

def find_core_paths():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = []
    candidates.append(os.path.abspath(os.path.join(current_dir, "core")))
    candidates.append(os.path.abspath(os.path.join(current_dir, "..", "core")))
    candidates.append(os.path.abspath(os.path.join(current_dir, "..", "..", "core")))
    
    pointer = current_dir
    root_found = None
    for _ in range(4):
        if os.path.basename(pointer) == "ComfyUI_MD_Nodes":
            root_found = pointer
            break
        parent = os.path.dirname(pointer)
        if parent == pointer: break
        pointer = parent
    
    if root_found: candidates.append(os.path.join(root_found, "core"))
    return list(dict.fromkeys(candidates))

CORE_LOCATIONS = find_core_paths()
AM_CORE_LOADED = False
AM_CORE_MODE = None
AM_CORE_ERROR = None

for loc in CORE_LOCATIONS:
    if loc not in sys.path: sys.path.insert(0, loc)

try:
    import automaster_core_bin as am_core
    AM_CORE_LOADED = True
    AM_CORE_MODE = "Binary (Production)"
except ImportError as e1:
    try:
        import automaster_core as am_core
        AM_CORE_LOADED = True
        AM_CORE_MODE = "Source (Development)"
    except ImportError as e2:
        AM_CORE_ERROR = f"Binary: {e1} | Source: {e2}"

# =================================================================================
# == Configuration Constants
# =================================================================================

logger = logging.getLogger("MD_Nodes.Audio.AutoMaster")
CONST_MAX_SAMPLES_PLOT = 150000
CONST_WAVEFORM_COLOR = '#87CEEB'
CONST_PEAK_COLOR = 'orangered'
CONST_BACKGROUND_COLOR = '#1e1e1e'
CONST_PLOT_DPI = 100

MASTERING_PROFILES = {
    "Custom": {"desc": "Manual parameter control", "hp": 0, "lp": 0, "eq": True, "bass": 9.5, "high": 5.5, "adapt": True, "deess": True, "deess_db": -10.0, "mbc": True, "x_low": 300, "x_high": 3000, "x_order": 8, "mbc_L_t": -24.0, "mbc_L_r": 2.5, "mbc_M_t": -22.0, "mbc_M_r": 2.5, "mbc_H_t": -20.0, "mbc_H_r": 2.0, "lim": True, "lim_db": -0.1, "width": 1.0, "tilt": 0.0, "tamer": 0.0, "mud": 0.0, "thump": 0.0, "exciter": 0.0},
    "Standard": {"desc": "Balanced all-purpose mastering", "hp": 30, "lp": 0, "eq": True, "bass": 9.5, "high": 5.5, "adapt": True, "deess": True, "deess_db": -10.0, "mbc": True, "x_low": 250, "x_high": 3000, "x_order": 8, "mbc_L_t": -24.0, "mbc_L_r": 2.5, "mbc_M_t": -22.0, "mbc_M_r": 2.5, "mbc_H_t": -20.0, "mbc_H_r": 2.0, "lim": True, "lim_db": -0.1, "width": 1.0, "tilt": 0.0, "tamer": 0.0, "mud": 0.0, "thump": 0.0, "exciter": 0.0},
    "Diffusion Repair (Clean)": {"desc": "Surgical AI cleanup", "hp": 35, "lp": 18500, "eq": True, "bass": 8.5, "high": 5.0, "adapt": True, "deess": True, "deess_db": -15.0, "mbc": True, "x_low": 200, "x_high": 3500, "x_order": 8, "mbc_L_t": -28.0, "mbc_L_r": 3.0, "mbc_M_t": -26.0, "mbc_M_r": 3.5, "mbc_H_t": -22.0, "mbc_H_r": 2.0, "lim": True, "lim_db": -0.2, "width": 0.85, "tilt": -0.5, "tamer": 1.0, "mud": -7.5, "thump": 5.5, "exciter": 0.1},
    "Aggressive": {"desc": "Heavy compression", "hp": 40, "lp": 0, "eq": True, "bass": 8.5, "high": 4.5, "adapt": True, "deess": True, "deess_db": -12.0, "mbc": True, "x_low": 250, "x_high": 2800, "x_order": 10, "mbc_L_t": -22.0, "mbc_L_r": 3.5, "mbc_M_t": -20.0, "mbc_M_r": 3.5, "mbc_H_t": -18.0, "mbc_H_r": 3.0, "lim": True, "lim_db": -0.1, "width": 1.1, "tilt": 0.0, "tamer": 0.0, "mud": 0.0, "thump": 0.0, "exciter": 0.2},
    "Podcast (Clarity)": {"desc": "Voice-optimized", "hp": 80, "lp": 16000, "eq": True, "bass": 7.5, "high": 6.0, "adapt": True, "deess": True, "deess_db": -15.0, "mbc": True, "x_low": 400, "x_high": 3500, "x_order": 6, "mbc_L_t": -28.0, "mbc_L_r": 2.0, "mbc_M_t": -20.0, "mbc_M_r": 3.5, "mbc_H_t": -18.0, "mbc_H_r": 2.5, "lim": True, "lim_db": -1.0, "width": 0.8, "tilt": 0.0, "tamer": 0.0, "mud": 0.0, "thump": 0.0, "exciter": 0.0},
    "Gentle (Tame)": {"desc": "Minimal processing", "hp": 20, "lp": 0, "eq": True, "bass": 10.5, "high": 6.5, "adapt": True, "deess": False, "deess_db": 0.0, "mbc": True, "x_low": 300, "x_high": 3000, "x_order": 8, "mbc_L_t": -28.0, "mbc_L_r": 1.8, "mbc_M_t": -26.0, "mbc_M_r": 1.8, "mbc_H_t": -24.0, "mbc_H_r": 1.5, "lim": True, "lim_db": -0.5, "width": 1.0, "tilt": 0.0, "tamer": 0.0, "mud": 0.0, "thump": 0.0, "exciter": 0.0},
    "Mastering (Transparent)": {"desc": "Subtle enhancement", "hp": 20, "lp": 0, "eq": True, "bass": 9.0, "high": 5.0, "adapt": True, "deess": True, "deess_db": -12.0, "mbc": True, "x_low": 200, "x_high": 3800, "x_order": 8, "mbc_L_t": -26.0, "mbc_L_r": 2.0, "mbc_M_t": -24.0, "mbc_M_r": 2.0, "mbc_H_t": -22.0, "mbc_H_r": 1.8, "lim": True, "lim_db": -0.3, "width": 1.0, "tilt": 0.0, "tamer": 0.0, "mud": 0.0, "thump": 0.0, "exciter": 0.0},
    "Full Bass (Electronic)": {"desc": "Maximum low-end", "hp": 25, "lp": 0, "eq": True, "bass": 11.5, "high": 7.0, "adapt": True, "deess": True, "deess_db": -8.0, "mbc": True, "x_low": 200, "x_high": 2800, "x_order": 8, "mbc_L_t": -26.0, "mbc_L_r": 2.8, "mbc_M_t": -22.0, "mbc_M_r": 2.5, "mbc_H_t": -20.0, "mbc_H_r": 2.2, "lim": True, "lim_db": -0.1, "width": 1.15, "tilt": 0.0, "tamer": 0.0, "mud": 0.0, "thump": 0.0, "exciter": 0.15}
}

# =================================================================================
# == Performance Profiler
# =================================================================================

class PerformanceProfiler:
    """Standard performance profiler for MD_Nodes."""
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.timings = {}
        self.start_times = {}
    
    def start(self, op):
        if not self.enabled: return
        self.start_times[op] = time.perf_counter()
    
    def stop(self, op):
        if not self.enabled: return
        if op in self.start_times:
            elapsed = time.perf_counter() - self.start_times[op]
            self.timings.setdefault(op, []).append(elapsed)
            del self.start_times[op]
    
    def print_report(self):
        if not self.enabled or not self.timings: return
        logging.info("\n⏱️  PERFORMANCE (AI/DSP):")
        total = sum(sum(times) for times in self.timings.values())
        logging.info(f"    • Total Time: {total:.4f}s")
        for op, times in sorted(self.timings.items()):
            logging.info(f"    • {op}: {sum(times)/len(times):.4f}s avg")

# =================================================================================
# == Main Wrapper Class
# =================================================================================

class MD_AutoMasterNode:
    """
    MD Audio Auto Master Pro v6.32.0 (Enterprise)
    Wrapper with Unified Parameter Resolution and AI Co-Pilot.
    """
    
    def __init__(self):
        self.analysis_log = []
        self.log_verbosity = "0 - Silent"

    @classmethod
    def INPUT_TYPES(cls):
        profile_options = ["Custom", "Auto-Detect Genre", "AI Co-Pilot (Ollama)"] + \
                          [f"{n} - {MASTERING_PROFILES[n]['desc']}" for n in MASTERING_PROFILES.keys() if n != "Custom"]
        
        return {
            "required": {
                "audio": ("AUDIO", {
                    "tooltip": (
                        "AUDIO INPUT\n"
                        "• Purpose: Unprocessed audio waveform to master.\n"
                        "• Requirement: Standard ComfyUI AUDIO dict."
                    )
                }),
                "target_lufs": ("FLOAT", {
                    "default": -14.0, "min": -30.0, "max": -6.0, "step": 0.1,
                    "tooltip": (
                        "TARGET LOUDNESS\n"
                        "• Purpose: The final perceived loudness target (LUFS).\n"
                        "• Options: -14.0 (Streaming), -23.0 (Broadcast).\n"
                        "\n⭐ Recommended: -14.0"
                    )
                }),
                "profile": (profile_options, {
                    "default": "Standard - Balanced all-purpose mastering",
                    "tooltip": (
                        "MASTERING PROFILE\n"
                        "• Purpose: Automatically sets dozens of DSP parameters.\n"
                        "• Options: 'Standard', 'Diffusion Repair' (fixes AI noise), 'Podcast'.\n"
                        "\n⭐ Recommended: 'Diffusion Repair' for raw audio generation outputs."
                    )
                }),
            },
            "optional": {
                # --- Intelligence & Output ---
                "output_mode": (["Mastered Audio", "Delta (Difference)"], {
                    "default": "Mastered Audio",
                    "tooltip": (
                        "OUTPUT MODE\n"
                        "• Purpose: Defines what audio is sent to the output node.\n"
                        "• Options: 'Mastered' (Final result) or 'Delta' (Only what was changed).\n"
                        "\n⭐ Recommended: Mastered Audio."
                    )
                }),
                "enable_ai_helper": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "AI CO-PILOT\n"
                        "• Purpose: Queries a local Ollama LLM to fine-tune EQ based on analysis.\n"
                        "• Requirement: Ollama must be running locally.\n"
                        "\n⭐ Recommended: True for experimental/creative runs."
                    )
                }),
                "genre_hint": ("STRING", {
                    "default": "",
                    "tooltip": "GENRE HINT\n• Purpose: Text clue to help the AI Co-Pilot make better EQ decisions."
                }),
                "ollama_url": ("STRING", {
                    "default": "http://localhost:11434",
                    "tooltip": "OLLAMA URL\n• Purpose: Endpoint for the local LLM API."
                }),
                "ollama_model": ("STRING", {
                    "default": "qwen2.5:14b",
                    "tooltip": "AI MODEL\n• Purpose: Model used for Co-Pilot reasoning."
                }),
                "debug_mode": (["0 - Silent", "1 - Info", "2 - Verbose"], {
                    "default": "1 - Info",
                    "tooltip": "LOGGING VERBOSITY\n• Controls console logging and AI explanation detail."
                }),
                "enable_profiling": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "ENABLE PROFILING\n• Track execution time of LLM vs DSP stages."
                }),
                "yaml_config": ("STRING", {
                    "default": "", "multiline": True,
                    "tooltip": "YAML CONFIG\n• Purpose: Paste exported settings here to override all GUI controls."
                }),
                "export_yaml": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "EXPORT YAML\n• Purpose: Outputs the final computed settings as YAML text for saving."
                }),
                
                # --- DSP Parameters (Calibrated Steps for Sensitivity) ---
                "input_gain_db": ("FLOAT", {
                    "default": 0.0, "min": -36.0, "max": 36.0, "step": 0.1,
                    "tooltip": "INPUT GAIN (dB)\n• Pre-processing volume adjustment."
                }),
                "spectral_tilt": ("FLOAT", {
                    "default": 0.0, "min": -12.0, "max": 12.0, "step": 0.01,
                    "tooltip": "SPECTRAL TILT\n• Extremely sensitive macro EQ.\n• +0.05 = Brighter, -0.05 = Warmer."
                }),
                "vocal_tamer_strength": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "VOCAL TAMER\n• Purpose: Dynamically cuts harsh 1-3kHz resonances common in AI voices."
                }),
                "harmonic_exciter_drive": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 2.0, "step": 0.01,
                    "tooltip": "HARMONIC EXCITER\n• Purpose: Tube-style saturation for warmth. Use sparingly (0.05 - 0.20)."
                }),
                "fix_sub_mud_db": ("FLOAT", {
                    "default": 0.0, "min": -36.0, "max": 0.0, "step": 0.5,
                    "tooltip": "FIX SUB MUD\n• Purpose: Low shelf cut (75Hz) to remove boominess."
                }),
                "fix_kick_thump_db": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 12.0, "step": 0.5,
                    "tooltip": "FIX KICK THUMP\n• Purpose: Targeted narrow boost (90Hz) to restore punch."
                }),
                
                # --- Filters & EQ ---
                "highpass_freq": ("FLOAT", {
                    "default": 0, "min": 0, "max": 1000, "step": 5,
                    "tooltip": "HIGHPASS FILTER\n• Cut frequencies below this point (Hz)."
                }),
                "lowpass_freq": ("FLOAT", {
                    "default": 0, "min": 0, "max": 22000, "step": 100,
                    "tooltip": "LOWPASS FILTER\n• Cut frequencies above this point (Hz)."
                }),
                "do_eq": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "ENABLE ADAPTIVE EQ\n• Auto-balance the spectrum to targets using Librosa FFT analysis."
                }),
                "eq_bass_target": ("FLOAT", {
                    "default": 9.5, "min": 0.0, "max": 20.0, "step": 0.1,
                    "tooltip": "EQ BASS TARGET\n• Desired low-end energy distribution."
                }),
                "eq_high_target": ("FLOAT", {
                    "default": 5.5, "min": 0.0, "max": 20.0, "step": 0.1,
                    "tooltip": "EQ HIGH TARGET\n• Desired high-end energy distribution."
                }),
                "eq_adaptive": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "ADAPTIVE MODE\n• Dynamically scale EQ adjustments based on input deviation."
                }),
                "max_iterations_eq": ("INT", {
                    "default": 5, "min": 1, "max": 20,
                    "tooltip": "EQ ITERATIONS\n• How many analysis/adjustment passes to reach perfect balance."
                }),
                
                # --- Dynamics ---
                "do_deess": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "ENABLE DE-ESSER\n• Dynamically reduces harsh 'S' sounds in the 7kHz range."
                }),
                "deess_amount_db": ("FLOAT", {
                    "default": -10.0, "min": -60.0, "max": 0.0, "step": 0.5,
                    "tooltip": "DE-ESS AMOUNT (dB)\n• Maximum intensity of sibilance reduction."
                }),
                "do_mbc": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "ENABLE MULTIBAND COMPRESSOR\n• Enables independent 3-Band dynamics processing."
                }),
                "mbc_crossover_low": ("FLOAT", {
                    "default": 300, "min": 40, "max": 1000, "step": 10,
                    "tooltip": "MBC CROSSOVER LOW\n• Frequency split point between Bass and Mids."
                }),
                "mbc_crossover_high": ("FLOAT", {
                    "default": 3000, "min": 1000, "max": 16000, "step": 100,
                    "tooltip": "MBC CROSSOVER HIGH\n• Frequency split point between Mids and Highs."
                }),
                "mbc_crossover_order": ("INT", {
                    "default": 8, "min": 2, "max": 8, "step": 2,
                    "tooltip": "CROSSOVER SLOPE\n• Higher numbers create sharper frequency separation."
                }),
                
                # MBC Thresholds & Ratios
                "mbc_low_thresh_db": ("FLOAT", {
                    "default": -24.0, "min": -60.0, "max": 0.0, "step": 0.5,
                    "tooltip": "LOW BAND THRESHOLD\n• Level at which bass compression engages."
                }),
                "mbc_low_ratio": ("FLOAT", {
                    "default": 2.5, "min": 1.0, "max": 20.0, "step": 0.1,
                    "tooltip": "LOW BAND RATIO\n• Severity of bass compression."
                }),
                "mbc_mid_thresh_db": ("FLOAT", {
                    "default": -22.0, "min": -60.0, "max": 0.0, "step": 0.5,
                    "tooltip": "MID BAND THRESHOLD\n• Level at which mid compression engages."
                }),
                "mbc_mid_ratio": ("FLOAT", {
                    "default": 2.5, "min": 1.0, "max": 20.0, "step": 0.1,
                    "tooltip": "MID BAND RATIO\n• Severity of mid compression."
                }),
                "mbc_high_thresh_db": ("FLOAT", {
                    "default": -20.0, "min": -60.0, "max": 0.0, "step": 0.5,
                    "tooltip": "HIGH BAND THRESHOLD\n• Level at which treble compression engages."
                }),
                "mbc_high_ratio": ("FLOAT", {
                    "default": 2.0, "min": 1.0, "max": 20.0, "step": 0.1,
                    "tooltip": "HIGH BAND RATIO\n• Severity of treble compression."
                }),
                
                # --- Finalize ---
                "do_limiter": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "ENABLE LIMITER\n• Engages the final brickwall lookahead limiter to prevent clipping."
                }),
                "limiter_threshold_db": ("FLOAT", {
                    "default": -1.0, "min": -24.0, "max": 0.0, "step": 0.1,
                    "tooltip": "LIMITER CEILING\n• Maximum allowed True Peak level (-1.0 is standard safety margin)."
                }),
                "soft_clip_drive": ("FLOAT", {
                    "default": 1.0, "min": 0.8, "max": 1.5, "step": 0.05,
                    "tooltip": "SOFT CLIP DRIVE\n• Pre-limiter saturation gain. Higher = Louder/Dirtier, Lower = Clean."
                }),
                "stereo_width": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.5, "step": 0.05,
                    "tooltip": "STEREO WIDTH\n• 1.0 = Original, >1.0 = Wider (Haas effect), <1.0 = Narrower."
                }),
                "fast_mode": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "FAST MODE\n• Skips intermediate LUFS normalization passes for a speed boost."
                }),
                "skip_initial_analysis": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "SKIP PRE-ANALYSIS\n• Skips initial chart generation to save time."
                }),
                "mix": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "GLOBAL MIX\n• Final Dry/Wet blend parameter (1.0 = 100% Processed)."
                }),
            }
        }

    RETURN_TYPES = ("AUDIO", "STRING", "STRING", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("audio", "analysis_details", "yaml_config", "waveform_before", "waveform_after", "spectrum_plot", "dynamics_plot", "lufs_history_plot")
    FUNCTION = "master_audio"
    CATEGORY = "MD_Nodes/Audio Processing"
    OUTPUT_NODE = True

    def _log(self, message):
        self.analysis_log.append(message)
        if int(self.log_verbosity.split(" ")[0]) >= 1: logger.info(message)

    def _resolve_all_parameters(self, kwargs, profile_dict):
        def resolve(kw_n, p_k_list, d):
            u_v = kwargs.get(kw_n, d)
            if u_v != d: return u_v
            
            if not isinstance(p_k_list, list): p_k_list = [p_k_list]
            for key in p_k_list:
                if key in profile_dict: return profile_dict[key]
            return d
        
        p = {}
        p['profile_name'] = kwargs.get('profile', 'Custom')
        p['target_lufs'] = resolve('target_lufs', 'target_lufs', -14.0)
        
        p['hp'] = resolve('highpass_freq', 'hp', 0)
        p['lp'] = resolve('lowpass_freq', 'lp', 0)
        p['tilt'] = resolve('spectral_tilt', 'tilt', 0.0)
        p['tamer'] = resolve('vocal_tamer_strength', 'tamer', 0.0)
        p['mud'] = resolve('fix_sub_mud_db', 'mud', 0.0)
        p['thump'] = resolve('fix_kick_thump_db', 'thump', 0.0)
        p['exciter'] = resolve('harmonic_exciter_drive', 'exciter', 0.0)
        
        p['do_eq'] = resolve('do_eq', 'do_eq', True)
        p['eq_bass'] = resolve('eq_bass_target', ['eq_bass', 'bass'], 9.5)
        p['eq_high'] = resolve('eq_high_target', ['eq_high', 'high'], 5.5)
        p['eq_adaptive'] = profile_dict.get('adapt', True) 
        p['max_iterations_eq'] = resolve('max_iterations_eq', 'max_iterations_eq', 5)
        
        p['do_mbc'] = resolve('do_mbc', 'do_mbc', True)
        p['x_low'] = resolve('mbc_crossover_low', 'x_low', 300)
        p['x_high'] = resolve('mbc_crossover_high', 'x_high', 3000)
        p['x_order'] = resolve('mbc_crossover_order', 'x_order', 8)
        
        p['mbc_low_thresh'] = resolve('mbc_low_thresh_db', ['mbc_low_thresh', 'mbc_L_t'], -24.0)
        p['mbc_low_ratio'] = resolve('mbc_low_ratio', ['mbc_low_ratio', 'mbc_L_r'], 2.5)
        p['mbc_mid_thresh'] = resolve('mbc_mid_thresh_db', ['mbc_mid_thresh', 'mbc_M_t'], -22.0)
        p['mbc_mid_ratio'] = resolve('mbc_mid_ratio', ['mbc_mid_ratio', 'mbc_M_r'], 2.5)
        p['mbc_high_thresh'] = resolve('mbc_high_thresh_db', ['mbc_high_thresh', 'mbc_H_t'], -20.0)
        p['mbc_high_ratio'] = resolve('mbc_high_ratio', ['mbc_high_ratio', 'mbc_H_r'], 2.0)
        
        p['do_deess'] = resolve('do_deess', 'do_deess', True)
        p['deess_amount'] = resolve('deess_amount_db', ['deess_amount', 'deess_db'], -10.0)
        
        p['width'] = resolve('stereo_width', 'width', 1.0)
        p['do_limiter'] = resolve('do_limiter', 'do_limiter', True)
        p['lim_db'] = resolve('limiter_threshold_db', 'lim_db', -1.0)
        p['soft_clip_drive'] = resolve('soft_clip_drive', 'soft_clip_drive', 1.0)
        p['fast_mode'] = resolve('fast_mode', 'fast_mode', False)
        return p

    def _sanitize_ai_advice(self, advice):
        clamped = {}
        advice = {k.lower(): v for k, v in advice.items()}
        
        if 'tilt' in advice: clamped['tilt'] = max(-0.5, min(0.5, float(advice['tilt']))) 
        if 'tamer' in advice: clamped['tamer'] = max(0.0, min(1.0, float(advice['tamer']))) 
        if 'mud' in advice: clamped['mud'] = max(-20.0, min(0.0, float(advice['mud'])))
        if 'thump' in advice: clamped['thump'] = max(0.0, min(8.0, float(advice['thump'])))
        if 'exciter' in advice: clamped['exciter'] = max(0.0, min(0.4, float(advice['exciter']))) 
        if 'width' in advice: clamped['width'] = max(0.0, min(2.5, float(advice['width'])))
        if 'lim_db' in advice: clamped['lim_db'] = max(-20.0, min(0.0, float(advice['lim_db'])))
        if 'eq_bass' in advice: clamped['eq_bass'] = max(0.0, min(20.0, float(advice['eq_bass'])))
        if 'eq_high' in advice: clamped['eq_high'] = max(0.0, min(20.0, float(advice['eq_high'])))
        if 'soft_clip_drive' in advice: clamped['soft_clip_drive'] = max(0.8, min(1.5, float(advice['soft_clip_drive'])))
        return clamped

    def _get_ollama_advice(self, metrics, model, hint, url, images=None):
        prompt = f"""
        Role: Senior Mastering Engineer using AutoMaster Pro.
        Analyze the audio metrics and waveform context.
        
        Metrics:
        - Centroid: {metrics['centroid']:.1f}Hz
        - Crest Factor: {metrics['crest']:.1f}dB
        - RMS: {metrics['rms']:.3f}
        
        TOOL SENSITIVITY & RANGES:
        - 'tilt': EXTREMELY SENSITIVE. Range +/- 0.5. Step 0.05. (0.1 is large, 0.3 is huge). Use negatives for warmth.
        - 'exciter': TUBE SATURATION. Range 0.0 - 0.4. (0.1 adds warmth, 0.3 adds crunch).
        - 'tamer': SURGICAL CUT. Range 0.0 - 1.0. (0.3 is standard).
        
        Instruction:
        Return a JSON object containing ONLY the keys you want to change.
        Add a 'reason' key explaining your decision.
        
        CRITICAL RULES:
        1. BE BOLD IN DECISION, PRECISE IN VALUE: If track is dull, use Tilt +0.05, not +1.0.
        2. BODY FIRST: If you cut 'mud', increase 'thump'.
        3. SAFETY: Keep 'lim_db' at -1.0 for Bluetooth safety.
        4. FORMAT: Strict JSON. Lowercase keys.
        """
        payload = {"model": model, "prompt": prompt, "stream": False, "format": "json"}
        if images and ("vl" in model.lower() or "vision" in model.lower()): payload["images"] = images
        try:
            res = requests.post(f"{url}/api/generate", json=payload, timeout=10)
            return json.loads(res.json()['response'])
        except Exception: return None

    def _export_to_yaml(self, params):
        try:
            clean_params = {}
            for k, v in params.items():
                if isinstance(v, (int, float, str, bool)): clean_params[k] = v
                elif isinstance(v, np.ndarray): clean_params[k] = v.tolist()
                elif isinstance(v, torch.Tensor): clean_params[k] = v.cpu().numpy().tolist()
            config = {'md_automaster_v6_27_0': clean_params}
            return yaml.dump(config, default_flow_style=False, sort_keys=False)
        except Exception as e:
            return f"# YAML Export Error: {str(e)}"

    def _generate_4stage_log(self, user_params, ai_advice, final_params, ai_source_tracker):
        lines = ["\n" + "="*60, "📋 AUTOMASTER PROCESSING MANIFEST", "="*60]
        lines.append(f"  Profile: {user_params['profile_name']}")
        lines.append(f"  Target: {user_params['target_lufs']} LUFS")
        
        lines.append("\n" + "-"*60)
        lines.append("📥 STAGE 1: USER SETTINGS")
        lines.append("-"*60)
        lines.append(f"  Tilt: {user_params['tilt']:.2f}")
        lines.append(f"  Tamer: {user_params['tamer']:.2f}")
        lines.append(f"  Mud: {user_params['mud']:.1f} dB")
        lines.append(f"  Thump: {user_params['thump']:.1f} dB")
        lines.append(f"  Exciter: {user_params['exciter']:.2f}")
        lines.append(f"  Stereo Width: {user_params['width']:.2f}")
        
        if ai_advice and any(k != 'reason' for k in ai_advice.keys()):
            lines.append("\n" + "-"*60)
            lines.append("🤖 STAGE 2: AI RECOMMENDATIONS")
            lines.append("-"*60)
            for k, v in ai_advice.items():
                if k == 'reason': lines.append(f"  Reasoning: {v}")
                elif k in user_params: lines.append(f"  {k}: {user_params[k]} → {v} (AI suggests)")
        
        lines.append("\n" + "-"*60)
        lines.append("✅ STAGE 3: FINAL APPLIED SETTINGS")
        lines.append("-"*60)
        lines.append(f"  Tilt: {final_params['tilt']:.2f} {ai_source_tracker.get('tilt', '(USER)')}")
        lines.append(f"  Tamer: {final_params['tamer']:.2f} {ai_source_tracker.get('tamer', '(USER)')}")
        lines.append(f"  Mud: {final_params['mud']:.1f} dB {ai_source_tracker.get('mud', '(USER)')}")
        lines.append(f"  Thump: {final_params['thump']:.1f} dB {ai_source_tracker.get('thump', '(USER)')}")
        lines.append(f"  Exciter: {final_params['exciter']:.2f} {ai_source_tracker.get('exciter', '(USER)')}")
        lines.append(f"  Stereo Width: {final_params['width']:.2f} {ai_source_tracker.get('width', '(USER)')}")
        lines.append(f"  Soft Clip: {final_params['soft_clip_drive']:.2f}x (Saturation)")
        
        lines.append("\n" + "-"*60)
        lines.append("🔧 STAGE 4: PROCESSING LOG")
        lines.append("-"*60)
        return "\n".join(lines)

    def _fig_to_tensor(self, fig):
        b = io.BytesIO()
        fig.savefig(b, format='png', bbox_inches='tight', dpi=CONST_PLOT_DPI, facecolor=CONST_BACKGROUND_COLOR)
        b.seek(0); i = Image.open(b).convert("RGB"); plt.close(fig)
        return torch.from_numpy(np.array(i).astype(np.float32)/255.0).unsqueeze(0)

    def _fig_to_base64(self, fig):
        b = io.BytesIO(); fig.savefig(b, format='png', bbox_inches='tight', dpi=72, facecolor='white')
        b.seek(0); return base64.b64encode(b.read()).decode('utf-8')

    def _plot_spectrum(self, o, p, sr, ret_fig=False):
        if not MATPLOTLIB_AVAILABLE: return torch.zeros((1,64,64,3))
        plt.style.use('dark_background'); fig, ax = plt.subplots(figsize=(10,6))
        def db(x): return librosa.amplitude_to_db(np.abs(librosa.stft(x[:,0] if x.ndim==2 else x)), ref=np.max).mean(axis=1)
        f = librosa.fft_frequencies(sr=sr)
        ax.semilogx(f, db(o), color='gray', alpha=0.5, label='In'); ax.semilogx(f, db(p), color=CONST_WAVEFORM_COLOR, label='Out')
        ax.legend(); ax.set_xlim(20, 20000)
        if ret_fig: return fig
        return self._fig_to_tensor(fig)

    def _plot_dynamics(self, history):
        if not MATPLOTLIB_AVAILABLE: return torch.zeros((1,64,64,3))
        plt.style.use('dark_background'); fig, (ax1,ax2) = plt.subplots(2,1, figsize=(10,8), sharex=True)
        s = list(history['lufs'].keys())
        ax1.plot(s, list(history['lufs'].values()), 'o-', color=CONST_WAVEFORM_COLOR); ax1.set_title("LUFS")
        ax2.plot(s, list(history['peak'].values()), 'o-', color=CONST_PEAK_COLOR); ax2.set_title("Peak")
        return self._fig_to_tensor(fig)

    def _plot_meter(self, c, t):
        if not MATPLOTLIB_AVAILABLE: return torch.zeros((1,64,64,3))
        plt.style.use('dark_background'); fig, ax = plt.subplots(figsize=(6,2))
        ax.barh(0, 1, color='#333'); ax.axvline((t+30)/30, color='cyan', lw=3)
        ax.plot(np.clip((c+30)/30,0,1), 0, 'o', color='green', markersize=15); ax.set_yticks([])
        return self._fig_to_tensor(fig)

    def _plot_waveform(self, a, sr, t):
        if not MATPLOTLIB_AVAILABLE: return torch.zeros((1,64,64,3))
        plt.style.use('dark_background'); fig, ax = plt.subplots(figsize=(10,3))
        d = a[:,0] if a.ndim==2 else a
        if d.size > CONST_MAX_SAMPLES_PLOT: d = d[::d.size//CONST_MAX_SAMPLES_PLOT]
        ax.plot(np.linspace(0, a.shape[0]/sr, d.size), d, color=CONST_WAVEFORM_COLOR, lw=0.5)
        ax.set_title(t); return self._fig_to_tensor(fig)


    def master_audio(self, audio, target_lufs, profile, **kwargs):
        
        # Graceful Degradation: If core is missing, pass audio through unharmed.
        if not AM_CORE_LOADED: 
            error_msg = f"❌ Core Missing: {AM_CORE_ERROR}. Audio passed through unprocessed."
            logging.warning(f"[MD_AutoMaster] {error_msg}")
            return (audio, error_msg, "", *([torch.zeros((1,64,64,3))]*5))
        
        self.log_verbosity = kwargs.get("debug_mode", "1 - Info")
        prof = PerformanceProfiler(enabled=kwargs.get("enable_profiling", False))
        prof.start("total")
        self.analysis_log = []

        sr = audio['sample_rate']
        audio_data = audio['waveform'][0].T.cpu().numpy().astype(np.float32)
        
        if not np.all(np.isfinite(audio_data)):
            audio_data = np.nan_to_num(audio_data, nan=0.0, posinf=1.0, neginf=-1.0)
            self._log("⚠️ WARN: Input audio contained NaNs. Sanitized.")
            
        orig_audio = audio_data.copy()
        
        # 1. Resolve Base Params (USER SETTINGS)
        p_dict = MASTERING_PROFILES.get(profile.split(" - ")[0], MASTERING_PROFILES["Standard"])
        
        yaml_str = kwargs.get("yaml_config", "").strip()
        if yaml_str:
            try: 
                y = yaml.safe_load(yaml_str)
                if isinstance(y, dict):
                    if 'md_automaster_v6_31_0' in y: 
                        p_dict.update(y['md_automaster_v6_31_0'])
                        self._log("📝 YAML: Loaded 'md_automaster_v6_31_0'")
                    elif 'md_master' in y: 
                        p_dict.update(y['md_master'])
                        self._log("📝 YAML: Loaded 'md_master' (Generic)")
                    elif any(k in y for k in ['tilt', 'tamer', 'exciter', 'lim_db']):
                        p_dict.update(y)
                        self._log("📝 YAML: Loaded root dictionary")
                    else:
                        first_val = next(iter(y.values()))
                        if isinstance(first_val, dict): 
                            p_dict.update(first_val)
                            self._log("📝 YAML: Loaded greedy match")
            except Exception as e:
                self._log(f"⚠️ YAML Error: {str(e)}")
        
        user_params = self._resolve_all_parameters(kwargs, p_dict)
        ai_source_tracker = {}
        ai_advice_raw = None

        # 2. AI Intelligence (if enabled)
        if kwargs.get("enable_ai_helper") and kwargs.get("ollama_model"):
            prof.start("ai")
            mono = audio_data[:,0] if audio_data.ndim>1 else audio_data
            cent = librosa.feature.spectral_centroid(y=mono, sr=sr).mean() if LIBROSA_AVAILABLE else 0
            rms = np.sqrt(np.mean(mono**2))
            crest = 20 * np.log10(np.max(np.abs(mono)) / (rms + 1e-6))
            
            imgs = []
            if "vl" in kwargs["ollama_model"].lower():
                fig = self._plot_spectrum(orig_audio, orig_audio, sr, True)
                imgs.append(self._fig_to_base64(fig)); plt.close(fig)

            ai_advice_raw = self._get_ollama_advice(
                {"centroid": cent, "crest": crest, "rms": rms}, 
                kwargs["ollama_model"], 
                kwargs.get("genre_hint"), 
                kwargs.get("ollama_url"), 
                imgs
            )
            prof.stop("ai")

        # 3. Create FINAL params (merge User + AI)
        final_params = user_params.copy()
        if ai_advice_raw:
            safe_advice = self._sanitize_ai_advice(ai_advice_raw)
            for k, v in safe_advice.items():
                if k in final_params:
                    ai_source_tracker[k] = "(AI OVERRIDE)"
                    final_params[k] = v

        # 4. Generate 4-Stage Log
        manifest = self._generate_4stage_log(user_params, ai_advice_raw, final_params, ai_source_tracker)
        self._log(manifest)

        # 5. Execute Core DSP 
        prof.start("dsp")
        pipeline_out = am_core.execute_pipeline(
            audio_data, sr, final_params, 
            lambda m: self._log(m)
        )
        processed = pipeline_out[0]
        history = pipeline_out[1]
        
        prof.stop("dsp")

        # 6. Finalize Output
        prof.start("vis")
        
        if kwargs.get("output_mode") == "Delta (Difference)":
            L = min(len(orig_audio), len(processed))
            processed = orig_audio[:L] - processed[:L]
            
        final_lufs = history['lufs'].get('Final', -14.0)
        
        out = {
            "waveform": torch.from_numpy(processed.T).unsqueeze(0).to(audio['waveform'].device), 
            "sample_rate": sr
        }
        
        # 7. YAML Export
        yaml_out = ""
        if kwargs.get("export_yaml"): 
            yaml_out = self._export_to_yaml(final_params)
        
        # 8. Generate Plots
        wb = self._plot_waveform(orig_audio, sr, "Input")
        wa = self._plot_waveform(processed, sr, "Output")
        sp = self._plot_spectrum(orig_audio, processed, sr)
        dp = self._plot_dynamics(history)
        mp = self._plot_meter(final_lufs, target_lufs)
        
        prof.stop("vis")
        prof.stop("total")
        
        # 9. Performance Report
        if int(self.log_verbosity.split()[0]) >= 1:
            self._log("\n" + "="*60)
            prof.print_report()
            self._log("="*60)

        return (out, "\n".join(self.analysis_log), yaml_out, wb, wa, sp, dp, mp)

# =================================================================================
# == ComfyUI Node Registration
# =================================================================================

NODE_CLASS_MAPPINGS = {"MD_AutoMasterNode": MD_AutoMasterNode}
NODE_DISPLAY_NAME_MAPPINGS = {"MD_AutoMasterNode": "MD: Audio Auto Master Pro"}


# ==============================================================================
# == Unit Tests (smoke — runs without ComfyUI)
# ==============================================================================

if __name__ == "__main__":
    print("\n🧪 Smoke tests: MD_AutoMasterNode")
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

    _check("VERSION defined",    VERSION == "v6.32.0")
    _check("CONST CONST_MAX_SAMPLES_PLOT defined", CONST_MAX_SAMPLES_PLOT is not None)
    _check("CONST CONST_WAVEFORM_COLOR defined", CONST_WAVEFORM_COLOR is not None)
    _check("CONST CONST_PEAK_COLOR defined", CONST_PEAK_COLOR is not None)
    _check("CONST CONST_BACKGROUND_COLOR defined", CONST_BACKGROUND_COLOR is not None)
    _check("CONST CONST_PLOT_DPI defined", CONST_PLOT_DPI is not None)
    _check("NODE_CLASS_MAPPINGS defined",
           isinstance(NODE_CLASS_MAPPINGS, dict) and len(NODE_CLASS_MAPPINGS) > 0)
    _check("  class MD_AutoMasterNode in map", "MD_AutoMasterNode" in NODE_CLASS_MAPPINGS)

    print(f"\n  {_pass} passed, {_fail} failed")
    if _fail == 0:
        print("  🎉 All good.")
