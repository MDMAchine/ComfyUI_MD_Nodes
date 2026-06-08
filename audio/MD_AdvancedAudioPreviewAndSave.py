# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# █▓▒░                                                                     ░▒▓█
# █▓▒░            MD_Nodes/AdvancedAudioPreviewAndSave – v2.3.0            ░▒▓█
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
# ║ ░▒▓ ORIGIN: Custom Implementation (Enhanced Metadata & Normalization)
# ║
# ║ ░▒▓ DESCRIPTION:
# ║    A comprehensive audio processing and export node providing professional-grade
# ║    normalization, effects, and intelligent metadata handling.
# ║    NOTE: As an I/O and file-saving utility, this runs entirely in the public wrapper domain.
# ║
# ║ ░▒▓ CORE FEATURES:
# ║    ✓ Multi-Format Export: MP3 (CBR/VBR), FLAC (lossless), OPUS (efficient)
# ║    ✓ Normalization Presets: Spotify (-14 LUFS), YouTube (-13), Broadcast (-23), ACX
# ║    ✓ Enterprise Logging: Three-tier system (Silent/Info/Verbose) with analytics
# ║    ✓ Advanced Normalization: Peak (0.99), RMS, LUFS (pyloudnorm + FFmpeg fallback)
# ║    ✓ Smart Metadata: Workflow embedding with automatic sidecar for large files (>256KB)
# ║
# ║ ░▒▓ CHANGELOG:
# ║    v2.2.0 (Enterprise Standards - Feb 2026):
# ║    ├── REFACTOR: Tooltips strictly updated to 5-part v1.5.4 standard.
# ║    └── VERIFIED: PerformanceProfiler matches v1.5.3 exact specifications.
# ║    v2.3.0 (2026-06-08) - MIGRATION: torchaudio fully removed
# ║    ├── MIGRATED: torchaudio.save → soundfile.write (lufs_normalize_with_ffmpeg + _save_audio_with_av)
# ║    ├── MIGRATED: torchaudio.load → soundfile.read (lufs_normalize_with_ffmpeg)
# ║    ├── MIGRATED: torchaudio.functional.resample → scipy.signal.resample_poly w/ pure-torch fallback
# ║    └── ADDED: VERSION constant, unit test block
# ║    v2.2.0 (Enterprise Standards - Feb 2026):
# ║    ├── REFACTOR: Tooltips strictly updated to 5-part v1.5.4 standard.
# ║    └── VERIFIED: PerformanceProfiler matches v1.5.3 exact specifications.
# ║    v2.1.9 (2026-02-11) - CRITICAL FIX: BytesIO/TorchCodec Crash
# ║    ├── FIXED: Replaced BytesIO buffer with physical temp file for intermediate WAV generation
# ║    └── FIXED: Resolved 'Couldn't allocate AVFormatContext' in newer torchaudio versions
# ╚════════════════════════════════════════════════════════════════════════════

VERSION = "v2.3.0"

# =================================================================================
# == Standard Library Imports
# =================================================================================
import os
import io
import time
import json
import random
import secrets
import subprocess
import tempfile
import traceback
import logging

# =================================================================================
# == Third-Party Imports
# =================================================================================
import torch
import numpy as np
import av
from PIL import Image

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False
    logging.warning("[AAPS] soundfile not found — audio I/O will fail. Install: uv pip install soundfile")

try:
    from scipy.signal import resample_poly as _scipy_resample_poly
    import math as _math
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# =================================================================================
# == ComfyUI Core Modules
# =================================================================================
import folder_paths
from comfy.cli_args import args

# =================================================================================
# == Dependency Checks
# =================================================================================

try:
    import pedalboard
    _pedalboard_available = True
except ImportError:
    _pedalboard_available = False

try:
    import pyloudnorm as pyln
    _pyloudnorm_available = True
except ImportError:
    _pyloudnorm_available = False

# =================================================================================
# == Configuration Constants
# =================================================================================

logger = logging.getLogger("MD_Nodes.Audio.AAPS")

AUDIO_OUTPUT_DIR = os.path.join(folder_paths.get_output_directory(), "ComfyUI_AdvancedAudioOutputs")
os.makedirs(AUDIO_OUTPUT_DIR, exist_ok=True)

METADATA_SIZE_LIMIT_KB = 256

CONST_WAVEFORM_COLOR = '#87CEEB'
CONST_PEAK_COLOR = 'orangered'
CONST_RMS_COLOR = 'mediumseagreen'
CONST_PLOT_DPI = 96
CONST_MAX_PLOT_SAMPLES = 150000

# =================================================================================
# == Preset Configurations
# =================================================================================

NORMALIZATION_PRESETS = {
    "Custom": None,
    "Spotify Standard (-14 LUFS)": {"method": "LUFS", "target_lufs": -14.0, "use_limiter": True},
    "YouTube Loud (-13 LUFS)": {"method": "LUFS", "target_lufs": -13.0, "use_limiter": True},
    "Apple Music Natural (-16 LUFS)": {"method": "LUFS", "target_lufs": -16.0, "use_limiter": True},
    "Broadcast EBU R128 (-23 LUFS)": {"method": "LUFS", "target_lufs": -23.0, "use_limiter": True},
    "Mastering Headroom (-8 LUFS)": {"method": "LUFS", "target_lufs": -8.0, "use_limiter": False},
    "Podcast ACX (-18 RMS)": {"method": "RMS", "target_rms": -18, "use_limiter": True},
    "Audiobook ACX (-23 RMS)": {"method": "RMS", "target_rms": -23, "use_limiter": True},
    "Peak Normalize Only": {"method": "Peak", "use_limiter": False},
}

FORMAT_PRESETS = {
    "Custom": None,
    "High Quality (MP3 V0)": {"format": "mp3", "quality": "V0"},
    "Web Streaming (MP3 128k)": {"format": "mp3", "quality": "128k"},
    "Archive/Master (FLAC)": {"format": "flac", "quality": None},
    "Voice/Podcast (OPUS 64k)": {"format": "opus", "quality": "64k"},
    "Music Streaming (OPUS 128k)": {"format": "opus", "quality": "128k"},
}

# =================================================================================
# == Performance Profiler (Enterprise Standard)
# =================================================================================

class PerformanceProfiler:
    """Standard performance profiler for MD_Nodes."""
    
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.timings = {}
        self.start_times = {}
    
    def start(self, operation_name):
        if not self.enabled:
            return
        self.start_times[operation_name] = time.perf_counter()
    
    def stop(self, operation_name):
        if not self.enabled:
            return
        if operation_name in self.start_times:
            elapsed = time.perf_counter() - self.start_times[operation_name]
            if operation_name not in self.timings:
                self.timings[operation_name] = []
            self.timings[operation_name].append(elapsed)
            del self.start_times[operation_name]
    
    def get_total_time(self):
        if not self.enabled or not self.timings:
            return 0.0
        return sum(sum(times) for times in self.timings.values())
    
    def print_report(self):
        if not self.enabled or not self.timings:
            return
        logging.info("\n⏱️  PERFORMANCE (I/O):")
        total = self.get_total_time()
        logging.info(f"    • Total Time: {total:.2f}s")
        for op_name, times in sorted(self.timings.items()):
            avg = sum(times) / len(times)
            if len(times) == 1:
                logging.info(f"    • {op_name}: {avg:.3f}s")
            else:
                logging.info(f"    • {op_name}: {avg:.3f}s avg ({len(times)}x)")

# =================================================================================
# == Helper Functions
# =================================================================================

def generate_unique_counter():
    return int(time.time() * 1000) + random.randint(0, 9999)

def apply_fades(audio, sr, fin, fout):
    if fin == 0 and fout == 0:
        return audio
    
    num_samples = audio.shape[1]
    fin_samples = int(fin * sr / 1000)
    fout_samples = int(fout * sr / 1000)
    
    total_fade = fin_samples + fout_samples
    if total_fade > num_samples:
        scale = num_samples / total_fade
        fin_samples = int(fin_samples * scale)
        fout_samples = int(fout_samples * scale)
    
    if fin_samples > 0:
        fade_in_curve = torch.linspace(0, 1, fin_samples, device=audio.device)
        audio[:, :fin_samples] *= fade_in_curve
    
    if fout_samples > 0:
        fade_out_curve = torch.linspace(1, 0, fout_samples, device=audio.device)
        audio[:, -fout_samples:] *= fade_out_curve
    
    return audio

def _resample_audio(audio_tensor, orig_sr, target_sr):
    """Resample audio tensor [C, L] from orig_sr to target_sr. No torchaudio dependency."""
    if orig_sr == target_sr:
        return audio_tensor
    if SCIPY_AVAILABLE:
        gcd = _math.gcd(orig_sr, target_sr)
        up = target_sr // gcd
        down = orig_sr // gcd
        np_audio = audio_tensor.cpu().numpy()   # [C, L]
        resampled = _scipy_resample_poly(np_audio, up, down, axis=1)
        return torch.from_numpy(resampled.astype(np.float32)).to(audio_tensor.device)
    else:
        # Pure-torch linear interpolation fallback (lower quality but zero deps)
        orig_len = audio_tensor.shape[1]
        target_len = int(orig_len * target_sr / orig_sr)
        return torch.nn.functional.interpolate(
            audio_tensor.unsqueeze(0).float(), size=target_len,
            mode='linear', align_corners=False
        ).squeeze(0).to(audio_tensor.device)


def lufs_normalize_with_ffmpeg(audio_tensor, sample_rate, target_lufs):
    if audio_tensor.is_cuda:
        audio_tensor = audio_tensor.cpu()
    
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_in, \
             tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_out:
            
            tmp_in_path = tmp_in.name
            tmp_out_path = tmp_out.name

        # Write temp WAV via soundfile: expects [L, C], tensor is [C, L]
        np_audio = audio_tensor.cpu().numpy().T  # [L, C]
        sf.write(tmp_in_path, np_audio, sample_rate, subtype='FLOAT')

        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", tmp_in_path,
            "-af", f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11:print_format=summary",
            "-ar", str(sample_rate), 
            tmp_out_path
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            logging.warning(f"⚠️ FFmpeg error during normalization: {result.stderr}")
            return audio_tensor

        # Read back normalized WAV via soundfile: returns [L, C], convert to [C, L]
        np_out, _ = sf.read(tmp_out_path, dtype='float32', always_2d=True)
        normalized_audio = torch.from_numpy(np_out.T)  # [C, L]

        for p in [tmp_in_path, tmp_out_path]:
            if os.path.exists(p):
                os.remove(p)

        return normalized_audio

    except Exception as e:
        logging.error(f"❌ FFmpeg LUFS exception: {e}")
        if 'tmp_in_path' in locals() and os.path.exists(tmp_in_path): os.remove(tmp_in_path)
        if 'tmp_out_path' in locals() and os.path.exists(tmp_out_path): os.remove(tmp_out_path)
        return audio_tensor

def save_metadata_sidecar(audio_path, workflow_graph, custom_notes):
    try:
        sidecar_path = os.path.splitext(audio_path)[0] + ".json"
        
        if workflow_graph:
            sidecar_data = workflow_graph.copy()
            if 'version' not in sidecar_data:
                sidecar_data['version'] = 0.4
            
            if 'extra' not in sidecar_data:
                sidecar_data['extra'] = {}
            sidecar_data['extra']['source_audio'] = os.path.basename(audio_path)
            
            if custom_notes:
                sidecar_data['extra']['notes'] = custom_notes
        else:
            sidecar_data = {
                "error": "No workflow data available",
                "source_audio": os.path.basename(audio_path),
                "version": 0.4
            }
        
        with open(sidecar_path, 'w', encoding='utf-8') as f:
            json.dump(sidecar_data, f, indent=2)
        
        return True
        
    except Exception as e:
        logger.error(f"Sidecar save failed: {e}")
        return False

def _save_audio_with_av(audio_tensor, sample_rate, output_path, save_format, metadata, quality, debug_level=0):
    temp_wav_path = None
    try:
        audio_tensor = audio_tensor.to(torch.float32).cpu()
        if audio_tensor.ndim == 3:
            audio_tensor = audio_tensor[0]
        if audio_tensor.ndim == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        
        target_sample_rate = sample_rate
        OPUS_RATES = [8000, 12000, 16000, 24000, 48000]
        
        if save_format == "opus":
            if target_sample_rate not in OPUS_RATES:
                target_sample_rate = min(OPUS_RATES, key=lambda r: abs(r - target_sample_rate))
                audio_tensor = _resample_audio(audio_tensor, sample_rate, target_sample_rate)
                sample_rate = target_sample_rate
                if debug_level >= 1:
                    logging.info(f"    • Resampled to {target_sample_rate}Hz for Opus")
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            temp_wav_path = tf.name
        
        # Write temp WAV via soundfile: expects [L, C], tensor is [C, L]
        np_audio = audio_tensor.cpu().numpy().T  # [L, C]
        sf.write(temp_wav_path, np_audio, sample_rate, subtype='FLOAT')
        
        with av.open(temp_wav_path) as in_container, \
             av.open(output_path, mode='w', format=save_format) as out_container:
                
                if metadata:
                    for key, value in metadata.items():
                        out_container.metadata[str(key)] = str(value)
                
                stream_kwargs = {"rate": sample_rate}
                codec_name = {"mp3": "libmp3lame", "opus": "libopus", "flac": "flac"}.get(
                    save_format, 'aac'
                )
                
                if codec_name == "libmp3lame":
                    quality_str = str(quality) if quality is not None else "128k"
                    
                    if debug_level >= 1:
                        logging.debug(f"\n🔍 [PyAV] MP3 Encoding:")
                        logging.info(f"    • Quality: '{quality_str}'")
                    
                    if quality_str == "V0":
                        stream_kwargs['qscale'] = 1  
                        if debug_level >= 1: print(f"    • Mode: VBR V0 (qscale=1)")
                    elif quality_str == "128k":
                        stream_kwargs['bit_rate'] = 128000
                        if debug_level >= 1: print(f"    • Mode: CBR 128kbps")
                    elif quality_str == "320k":
                        stream_kwargs['bit_rate'] = 320000
                        if debug_level >= 1: print(f"    • Mode: CBR 320kbps")
                    else:
                        stream_kwargs['bit_rate'] = 128000
                        if debug_level >= 1: print(f"    ⚠️  Unknown quality '{quality_str}', defaulting to 128k")
                
                elif codec_name == "libopus" and quality:
                    stream_kwargs['bit_rate'] = int(quality.replace('k', '')) * 1000
                    if debug_level >= 1: print(f"    • OPUS bitrate: {stream_kwargs['bit_rate']}")
                
                out_stream = out_container.add_stream(codec_name, **stream_kwargs)
                
                for frame in in_container.decode(audio=0):
                    frame.pts = None
                    for packet in out_stream.encode(frame):
                        out_container.mux(packet)
                
                for packet in out_stream.encode(None):
                    out_container.mux(packet)
        
        if temp_wav_path and os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)

        if debug_level >= 1:
            logging.info(f"    ✓ Saved: {os.path.basename(output_path)}")
        return True
        
    except Exception as e:
        logger.error(f"PyAV Save Failed: {e}")
        traceback.print_exc()
        if temp_wav_path and os.path.exists(temp_wav_path):
            try: os.remove(temp_wav_path)
            except Exception: pass
        return False

# =================================================================================
# == Main Node Class
# =================================================================================

class AdvancedAudioPreviewAndSave:
    """Comprehensive audio processing and export with enterprise logging."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {
                    "tooltip": (
                        "AUDIO INPUT\n"
                        "• Purpose: The raw audio waveform to be processed and saved.\n"
                        "• Requirement: Standard ComfyUI AUDIO dict."
                    )
                }),
                
                "norm_preset": (list(NORMALIZATION_PRESETS.keys()), {
                    "default": "Spotify Standard (-14 LUFS)",
                    "tooltip": (
                        "NORMALIZATION PRESET\n"
                        "• Purpose: Automatically sets LUFS/RMS targets to match industry standards.\n"
                        "• Options: Spotify (-14), Broadcast (-23), Custom (Manual overrides).\n"
                        "\n⭐ Recommended: 'Spotify Standard' for general distribution."
                    )
                }),
                
                "format_preset": (list(FORMAT_PRESETS.keys()), {
                    "default": "High Quality (MP3 V0)",
                    "tooltip": (
                        "FORMAT PRESET\n"
                        "• Purpose: Sets container format and codec quality.\n"
                        "• Options: High Quality (MP3 V0), Archive (FLAC), Voice (OPUS).\n"
                        "\n⭐ Recommended: 'High Quality (MP3 V0)'"
                    )
                }),
                
                "save_to_disk": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "SAVE TO DISK\n"
                        "• Purpose: Writes the processed audio to the ComfyUI output directory.\n"
                        "• Effect: False will still normalize and preview the audio, but skip file I/O.\n"
                        "\n⭐ Recommended: True."
                    )
                }),
                "filename_prefix": ("STRING", {
                    "default": "ComfyUI_audio_%Y%m%d",
                    "tooltip": (
                        "FILENAME PREFIX\n"
                        "• Purpose: Sets the output file naming scheme.\n"
                        "• Support: Standard strftime patterns (%Y-%m-%d, %H-%M-%S).\n"
                        "• Example: 'audio_%Y%m%d' -> 'audio_20251228_00001.mp3'"
                    )
                }),
                "save_format": (["mp3", "flac", "opus"], {
                    "default": "mp3",
                    "tooltip": (
                        "SAVE FORMAT (Custom)\n"
                        "• Purpose: Manual selection of output container.\n"
                        "• Requirement: Active only if Format Preset is set to 'Custom'."
                    )
                }),
                "save_metadata": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "SAVE METADATA\n"
                        "• Purpose: Embeds workflow JSON directly into the audio file.\n"
                        "• Logic: Automatically generates a .json sidecar if payload > 256KB.\n"
                        "\n⭐ Recommended: True."
                    )
                }),
                "custom_notes": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": (
                        "CUSTOM NOTES\n"
                        "• Purpose: Embeds user text in the 'ComfyUI_Notes' metadata field."
                    )
                }),
                
                "channel_mode": (["Keep Original", "Convert to Mono"], {
                    "default": "Keep Original",
                    "tooltip": (
                        "CHANNEL MODE\n"
                        "• Purpose: Controls stereo imaging downmixing.\n"
                        "• Options: Keep Original (Stereo), Convert to Mono.\n"
                        "\n⭐ Recommended: Keep Original."
                    )
                }),
                "fade_in_ms": ("INT", {
                    "default": 0, "min": 0, "max": 10000, "step": 10,
                    "tooltip": "FADE IN\n• Purpose: Linearly ramps volume up at the start (in ms)."
                }),
                "fade_out_ms": ("INT", {
                    "default": 0, "min": 0, "max": 10000, "step": 10,
                    "tooltip": "FADE OUT\n• Purpose: Linearly ramps volume down at the end (in ms)."
                }),
                
                "normalize_method": (["Off", "Peak", "RMS", "LUFS"], {
                    "default": "LUFS",
                    "tooltip": (
                        "NORMALIZE METHOD (Custom)\n"
                        "• Purpose: Mathematical approach to adjusting volume.\n"
                        "• Options: LUFS (Perceived), RMS (Average), Peak (Max Amplitude).\n"
                        "• Requirement: Active only if Norm Preset is 'Custom'.\n"
                        "\n⭐ Recommended: LUFS."
                    )
                }),
                "target_rms_db": ("INT", {
                    "default": -16, "min": -60, "max": 0,
                    "tooltip": "TARGET RMS\n• Purpose: DB target level for RMS normalization."
                }),
                "target_lufs_db": ("FLOAT", {
                    "default": -14.0, "min": -50.0, "max": -5.0, "step": 0.5,
                    "tooltip": "TARGET LUFS\n• Purpose: LUFS target level for LUFS normalization."
                }),
                "use_limiter": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "USE LIMITER\n"
                        "• Purpose: Applies a fast soft limiter (-1.0dB ceiling) after normalization.\n"
                        "• Effect: Prevents digital clipping from math overshoot.\n"
                        "\n⭐ Recommended: True."
                    )
                }),
                "mp3_quality": (["V0", "128k", "320k"], {
                    "default": "V0",
                    "tooltip": (
                        "MP3 QUALITY\n"
                        "• V0: Best Variable Bitrate (~245kbps).\n"
                        "• 320k: Best Constant Bitrate.\n"
                        "• 128k: Web standard."
                    )
                }),
                "opus_quality": (["64k", "96k", "128k"], {
                    "default": "128k",
                    "tooltip": (
                        "OPUS QUALITY\n"
                        "• 128k: Music / High Quality.\n"
                        "• 64k: Voice / Podcast."
                    )
                }),
                
                "debug_mode": (["0 - Silent", "1 - Info", "2 - Verbose"], {
                    "default": "0 - Silent",
                    "tooltip": (
                        "LOGGING VERBOSITY\n"
                        "• Purpose: Controls console output and structural profiling.\n"
                        "• Options: 0 (Silent), 1 (Analytics Report), 2 (Full trace).\n"
                        "\n⭐ Recommended: 0 - Silent."
                    )
                }),
                "enable_profiling": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "ENABLE PROFILING\n"
                        "• Purpose: Measure file I/O and normalization times.\n"
                        "• Note: Automatically enabled if debug_mode >= 1."
                    )
                }),
                
                "force_save": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "FORCE SAVE\n"
                        "• Purpose: Bypasses ComfyUI caching to always generate a new file on disk.\n"
                        "• Note: Triggers node execution every run."
                    )
                }),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }
    
    @classmethod
    def IS_CHANGED(cls, force_save=False, filename_prefix="", **kwargs):
        if isinstance(force_save, str):
            force_save = (force_save.lower() == "true")

        if force_save:
            return secrets.token_hex(16)

        dynamic_patterns = ['%Y', '%m', '%d', '%H', '%M', '%S']
        if any(pattern in filename_prefix for pattern in dynamic_patterns):
            return secrets.token_hex(16)
        
        return "static"

    RETURN_TYPES = ("AUDIO", "IMAGE", "IMAGE",)
    RETURN_NAMES = ("AUDIO", "waveform_before", "waveform_after",)
    FUNCTION = "process_audio"
    CATEGORY = "MD_Nodes/Save"
    OUTPUT_NODE = True

    def _apply_presets(self, kwargs):
        norm_preset = kwargs.get("norm_preset", "Custom")
        if norm_preset != "Custom" and norm_preset in NORMALIZATION_PRESETS:
            preset = NORMALIZATION_PRESETS[norm_preset]
            if preset:
                kwargs.update({
                    "normalize_method": preset.get("method", kwargs.get("normalize_method")),
                    "target_lufs_db": preset.get("target_lufs", kwargs.get("target_lufs_db")),
                    "target_rms_db": preset.get("target_rms", kwargs.get("target_rms_db")),
                    "use_limiter": preset.get("use_limiter", kwargs.get("use_limiter"))
                })
        
        format_preset = kwargs.get("format_preset", "Custom")
        if format_preset != "Custom" and format_preset in FORMAT_PRESETS:
            preset = FORMAT_PRESETS[format_preset]
            if preset:
                kwargs["save_format"] = preset.get("format", kwargs.get("save_format", "mp3"))
                if preset.get("quality"):
                    if preset["format"] == "mp3":
                        kwargs["mp3_quality"] = preset["quality"]
                    elif preset["format"] == "opus":
                        kwargs["opus_quality"] = preset["quality"]
        
        return kwargs

    def _prepare_metadata(self, **kwargs):
        metadata = {}
        if not kwargs.get("save_metadata") or args.disable_metadata:
            return metadata

        prompt = kwargs.get("prompt")
        extra_pnginfo = kwargs.get("extra_pnginfo")
        
        if prompt is not None:
            metadata["prompt"] = json.dumps(prompt)
        
        if extra_pnginfo is not None:
            for key, value in extra_pnginfo.items():
                metadata[key] = json.dumps(value) if not isinstance(value, str) else value

        if kwargs.get("custom_notes"):
            metadata['ComfyUI_Notes'] = kwargs.get("custom_notes")
            
        return metadata

    def _normalize_audio(self, audio_tensor, sample_rate, profiler=None, **kwargs):
        method = kwargs.get("normalize_method", "Peak")
        debug_level = int(kwargs.get("debug_mode", "0").split(" ")[0])
        
        if method == "Peak":
            if profiler: profiler.start("norm_peak")
            peak_val = torch.max(torch.abs(audio_tensor))
            if peak_val > 1e-6:
                audio_tensor = audio_tensor / peak_val * 0.99
            if profiler: profiler.stop("norm_peak")
                
        elif method == "RMS":
            if profiler: profiler.start("norm_rms")
            current_rms = torch.sqrt(torch.mean(audio_tensor**2))
            target_rms = 10**(kwargs.get("target_rms_db", -16) / 20.0)
            if current_rms > 1e-9:
                audio_tensor *= (target_rms / current_rms)
            if profiler: profiler.stop("norm_rms")
                
        elif method == "LUFS":
            if profiler: profiler.start("norm_lufs")
            target_lufs = kwargs.get("target_lufs_db", -14.0)
            result = None
            
            if _pyloudnorm_available:
                try:
                    import warnings
                    meter = pyln.Meter(sample_rate)
                    loudness = meter.integrated_loudness(audio_tensor.cpu().numpy().T)
                    if loudness > -70.0:
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore", message="Possible clipped samples")
                            normalized_np = pyln.normalize.loudness(
                                audio_tensor.cpu().numpy().T, loudness, target_lufs
                            )
                        result = torch.from_numpy(normalized_np.T).to(audio_tensor.device)
                        if debug_level >= 2: logger.info("LUFS normalized via pyloudnorm")
                except Exception as e:
                    if debug_level >= 1: logger.warning(f"pyloudnorm failed: {e}")
            
            if result is None:
                if debug_level >= 1: logger.info("Using FFmpeg fallback for LUFS")
                result = lufs_normalize_with_ffmpeg(audio_tensor, sample_rate, target_lufs)

            if result is not None:
                audio_tensor = result
            else:
                if debug_level >= 1: logger.warning("LUFS failed, falling back to Peak")
                peak_val = torch.max(torch.abs(audio_tensor))
                if peak_val > 1e-6:
                    audio_tensor = audio_tensor / peak_val * 0.99
            if profiler: profiler.stop("norm_lufs")
        
        if kwargs.get("use_limiter") and method != "Off":
            if profiler: profiler.start("limiter")
            if _pedalboard_available:
                board = pedalboard.Pedalboard([
                    pedalboard.Limiter(threshold_db=-1.0, release_ms=50)
                ])
                audio_tensor = torch.from_numpy(
                    board(audio_tensor.cpu().numpy(), sample_rate)
                ).to(audio_tensor.device)
                if debug_level >= 2: logger.info("Applied Soft Limiter")
            else:
                audio_tensor = torch.clamp(audio_tensor, -1.0, 1.0)
            if profiler: profiler.stop("limiter")
        
        return audio_tensor

    def _plot_waveform_to_tensor(self, audio_data, sample_rate, title="Waveform"):
        if audio_data is None or audio_data.size == 0 or not MATPLOTLIB_AVAILABLE:
            return torch.zeros((1, 64, 64, 3), dtype=torch.float32)
        
        try:
            plt.style.use('dark_background')
            fig, ax = plt.subplots(figsize=(10, 3), dpi=CONST_PLOT_DPI)

            plot_data = audio_data[:, 0] if audio_data.ndim == 2 else audio_data
            
            if len(plot_data) > CONST_MAX_PLOT_SAMPLES:
                ds_factor = len(plot_data) // CONST_MAX_PLOT_SAMPLES
                plot_data = plot_data[::ds_factor]
            
            time_axis = np.linspace(0, len(plot_data) / sample_rate * (len(audio_data)/len(plot_data)), len(plot_data))
            
            ax.plot(time_axis, plot_data, color=CONST_WAVEFORM_COLOR, linewidth=0.5)

            peak_val = np.max(np.abs(plot_data)) if plot_data.size > 0 else 0.0
            rms = np.sqrt(np.mean(plot_data**2)) if plot_data.size > 0 else 0.0

            if peak_val > 0.8:
                ax.axhline(y=peak_val, color=CONST_PEAK_COLOR, ls='--', lw=0.7, alpha=0.6, label=f'Peak: {peak_val:.3f}')
                ax.axhline(y=-peak_val, color=CONST_PEAK_COLOR, ls='--', lw=0.7, alpha=0.6)

            ax.axhline(y=rms, color=CONST_RMS_COLOR, ls=':', lw=0.7, alpha=0.6, label=f'RMS: {rms:.3f}')
            ax.axhline(y=-rms, color=CONST_RMS_COLOR, ls=':', lw=0.7, alpha=0.6)

            ax.set_title(f"{title} | Peak: {peak_val:.3f} | RMS: {rms:.3f}", fontsize=10)
            ax.set_ylim(-1.05, 1.05)
            ax.grid(True, ls=':', lw=0.5, alpha=0.3)
            ax.legend(loc='upper right', fontsize=7, framealpha=0.5)
            
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', facecolor=fig.get_facecolor())
            buf.seek(0)
            plt.close(fig)
            
            img = Image.open(buf).convert("RGB")
            img_np = np.array(img).astype(np.float32) / 255.0
            return torch.from_numpy(img_np).unsqueeze(0)
        
        except Exception as e:
            if 'fig' in locals(): plt.close(fig)
            return torch.zeros((1, 64, 64, 3), dtype=torch.float32)

    def process_audio(self, audio, **kwargs):
        debug_mode = kwargs.get("debug_mode", "0 - Silent")
        debug_level = int(debug_mode.split(" ")[0])
        profiling_enabled = kwargs.get("enable_profiling", False) or (debug_level >= 1)
        profiler = PerformanceProfiler(enabled=profiling_enabled)
        profiler.start("total")

        if debug_level >= 2: logger.setLevel(logging.DEBUG)
        elif debug_level >= 1: logger.setLevel(logging.INFO)
        else: logger.setLevel(logging.WARNING)

        ui_text = []
        placeholder_img = torch.zeros((1, 64, 64, 3), dtype=torch.float32)

        kwargs = self._apply_presets(kwargs)

        try:
            profiler.start("load_audio")
            
            if not isinstance(audio, dict) or 'waveform' not in audio or 'sample_rate' not in audio:
                 raise ValueError("Expected dictionary with 'waveform' and 'sample_rate'")
            waveform_original = audio['waveform']
            samplerate = audio['sample_rate']
            if waveform_original.ndim == 2:
                 waveform_original = waveform_original.unsqueeze(0)
            profiler.stop("load_audio")
        except Exception as e:
            logger.error(f"Error loading audio: {e}")
            return {"ui": {"text": [f"Error: {e}"]}, "result": (audio, placeholder_img, placeholder_img)}

        if waveform_original.numel() == 0:
            return {"ui": {"text": ["Empty audio"]}, "result": (audio, placeholder_img, placeholder_img)}

        profiler.start("plot_before")
        plot_audio_before = waveform_original[0] if waveform_original.ndim == 3 else waveform_original
        waveform_before = self._plot_waveform_to_tensor(plot_audio_before.cpu().numpy().T, samplerate, "Original")
        profiler.stop("plot_before")

        peak_before = float(torch.max(torch.abs(plot_audio_before)))
        rms_before = float(torch.sqrt(torch.mean(plot_audio_before**2)))
        rms_before_db = 20 * np.log10(max(rms_before, 1e-9))

        profiler.start("processing_total")
        processed_audio = waveform_original[0] if waveform_original.ndim == 3 else waveform_original.clone()

        if kwargs.get("channel_mode") == "Convert to Mono" and processed_audio.shape[0] > 1:
            processed_audio = torch.mean(processed_audio, dim=0, keepdim=True)
        
        if kwargs.get("fade_in_ms", 0) > 0 or kwargs.get("fade_out_ms", 0) > 0:
            processed_audio = apply_fades(
                processed_audio, samplerate, kwargs.get("fade_in_ms", 0), kwargs.get("fade_out_ms", 0)
            )
        
        processed_audio = self._normalize_audio(processed_audio, samplerate, profiler=profiler, **kwargs)
        profiler.stop("processing_total")

        peak_after = float(torch.max(torch.abs(processed_audio)))
        rms_after = float(torch.sqrt(torch.mean(processed_audio**2)))
        rms_after_db = 20 * np.log10(max(rms_after, 1e-9))

        profiler.start("plot_after")
        waveform_after = self._plot_waveform_to_tensor(processed_audio.cpu().numpy().T, samplerate, "Processed")
        profiler.stop("plot_after")

        output_path = ""
        sidecar_saved = False
        
        if kwargs.get("save_to_disk"):
            profiler.start("save_to_disk")
            try:
                base_prefix = time.strftime(os.path.basename(kwargs.get("filename_prefix")), time.localtime())
                subfolder = os.path.dirname(kwargs.get("filename_prefix"))
                output_dir = os.path.join(AUDIO_OUTPUT_DIR, subfolder)
                if not os.path.exists(output_dir): os.makedirs(output_dir, exist_ok=True)

                save_format = kwargs.get("save_format", "mp3").lower()
                file_name = f"{base_prefix}_{generate_unique_counter():09}.{save_format}"
                output_path = os.path.join(output_dir, file_name)
                
                metadata = self._prepare_metadata(**kwargs)
                metadata_size = sum(len(v.encode('utf-8')) for v in metadata.values()) / 1024
                
                if save_format in ["mp3", "opus"] and metadata_size > METADATA_SIZE_LIMIT_KB:
                    if debug_level >= 1: logger.info(f"Metadata {metadata_size:.1f}KB exceeds limit. Creating sidecar.")
                    extra_pnginfo = kwargs.get('extra_pnginfo', {})
                    workflow_graph = extra_pnginfo.get('workflow', extra_pnginfo)
                    sidecar_saved = save_metadata_sidecar(output_path, workflow_graph, kwargs.get("custom_notes"))
                    metadata = {"ComfyUI_Notes": kwargs.get("custom_notes", "")} if kwargs.get("custom_notes") else {}

                quality = kwargs.get("mp3_quality") if save_format == "mp3" else kwargs.get("opus_quality")
                
                success = _save_audio_with_av(
                    processed_audio, samplerate, output_path, save_format, metadata, quality, debug_level
                )

                if success:
                    rel_path = os.path.relpath(output_path, folder_paths.get_output_directory())
                    ui_text.append(f"✓ Saved: {rel_path}" + (" + .json" if sidecar_saved else ""))
                else:
                    ui_text.append("✗ Save Failed")
            except Exception as e:
                logger.error(f"Save exception: {e}")
                traceback.print_exc()
                ui_text.append(f"✗ Error: {e}")
            profiler.stop("save_to_disk")

        processed_for_output = processed_audio.unsqueeze(0) if processed_audio.ndim == 2 else processed_audio
        final_audio_output = {"waveform": processed_for_output, "sample_rate": samplerate}
        
        profiler.stop("total")

        if debug_level >= 1:
            logging.info("\n" + "=" * 60)
            logging.info("📊 [AAPS] ANALYTICS REPORT")
            logging.info("=" * 60)
            logging.info("🎵  AUDIO:")
            logging.info(f"    • Peak:         {peak_before:.3f} → {peak_after:.3f}")
            logging.info(f"    • RMS:          {rms_before_db:.1f} dB → {rms_after_db:.1f} dB")
            
            if output_path:
                logging.info("📁  OUTPUT:")
                logging.info(f"    • File:         {os.path.basename(output_path)}")
                logging.info(f"    • Format:       {kwargs.get('save_format', 'mp3').upper()}")
                if sidecar_saved:
                    logging.info("    • Sidecar:      Generated (Metadata > 256KB)")
            
            logging.info("🎛️  PROCESSING:")
            logging.info(f"    • Preset:       {kwargs.get('norm_preset', 'Custom')}")
            logging.info(f"    • Method:       {kwargs.get('normalize_method', 'Off')}")
            logging.info(f"    • Limiter:      {'On' if kwargs.get('use_limiter') else 'Off'}")
            
            profiler.print_report()
            logging.info("=" * 60)

        return {
            "ui": {"text": ui_text},
            "result": (final_audio_output, waveform_before, waveform_after)
        }

# =================================================================================
# == Node Registration
# =================================================================================

NODE_CLASS_MAPPINGS = {
    "AdvancedAudioPreviewAndSave": AdvancedAudioPreviewAndSave
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AdvancedAudioPreviewAndSave": "MD: Advanced Audio Preview & Save"
}


# =================================================================================
# == Unit Tests (smoke test — runs without ComfyUI)
# =================================================================================

if __name__ == "__main__":
    print("✓ MD_AdvancedAudioPreviewAndSave module imports clean")
    print(f"✓ VERSION: {VERSION}")
    print(f"✓ soundfile available: {SOUNDFILE_AVAILABLE}")
    print(f"✓ scipy available: {SCIPY_AVAILABLE}")
    print(f"✓ pyloudnorm available: {_pyloudnorm_available}")
    print(f"✓ pedalboard available: {_pedalboard_available}")

    import torch, numpy as np

    # Test _resample_audio
    dummy = torch.randn(2, 44100)
    if SCIPY_AVAILABLE:
        resampled = _resample_audio(dummy, 44100, 48000)
        assert resampled.shape[0] == 2, "channel count preserved"
        assert resampled.shape[1] == 48000, f"expected 48000 samples, got {resampled.shape[1]}"
        print("✓ _resample_audio (scipy): 44100 -> 48000 OK")
    else:
        resampled = _resample_audio(dummy, 44100, 48000)
        assert resampled.shape[0] == 2
        print("✓ _resample_audio (torch fallback): OK")

    # Test apply_fades
    audio = torch.ones(2, 44100)
    faded = apply_fades(audio, 44100, fin=50, fout=50)
    assert faded.shape == audio.shape
    assert float(faded[:, 0].max()) < 0.1, "fade in should start near zero"
    print("✓ apply_fades: OK")

    # Test PerformanceProfiler
    p = PerformanceProfiler(enabled=True)
    import time
    p.start("test_op")
    time.sleep(0.01)
    p.stop("test_op")
    assert p.get_total_time() > 0
    print("✓ PerformanceProfiler: OK")

    print("\n✓ All AAPS smoke tests passed")