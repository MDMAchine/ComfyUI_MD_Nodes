# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# ████ ADVANCED AUDIO PREVIEW & SAVE (AAPS) v3.0.1 – Production Ready ████▓▒░
# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀

# ░▒▓ PROJECT INFORMATION
#    • Author: MDMAchine (waveform wizard)
#    • Contributors: Claude (Anthropic AI), Gemini (Google AI)
#    • License: Public Domain / MIT
#    • Repository: [Your GitHub URL here]
#    • Support: [Your support link here]

# ░▒▓ DESCRIPTION
#    A comprehensive audio processing and export node for ComfyUI workflows.
#    Provides professional-grade audio normalization, effects, visualization,
#    and intelligent metadata embedding with workflow preservation.

# ░▒▓ KEY FEATURES
#    ✓ Multi-format export: MP3 (universal), FLAC (lossless), OPUS (efficient)
#    ✓ Advanced normalization: Peak, RMS, and LUFS (with pyloudnorm/FFmpeg fallback)
#    ✓ Professional effects: Fade in/out, soft limiting (pedalboard), mono conversion
#    ✓ Smart metadata handling: Embeds workflow JSON with automatic sidecar fallback for large files
#    ✓ Visual feedback: Real-time waveform generation with customizable colors and grid
#    ✓ Optional spectrogram: Frequency analysis visualization (9 colormap options)
#    ✓ Drag-and-drop support: Saved files can be dropped back into ComfyUI to restore workflows
#    ✓ Dynamic filename templating: Supports strftime patterns (e.g., "audio_%Y-%m-%d")

# ░▒▓ DEPENDENCIES
#    Required:
#      - torch, torchaudio (audio tensor processing)
#      - av (PyAV - FFmpeg wrapper for robust encoding)
#      - matplotlib, PIL (waveform/spectrogram visualization)
#      - numpy (numerical operations)
#    
#    Optional (recommended):
#      - pedalboard: Professional audio effects (soft limiter) - pip install pedalboard
#      - pyloudnorm: Fast LUFS normalization - pip install pyloudnorm
#      - ffmpeg: Fallback for LUFS normalization (must be in system PATH)

# ░▒▓ TECHNICAL NOTES
#    • Metadata Embedding Strategy:
#      - FLAC: Always embeds full metadata (robust large tag support)
#      - MP3/Opus: Embeds if <256KB, otherwise creates .json sidecar file
#      - Uses PyAV container metadata (compatible with ComfyUI's loader)
#    
#    • Audio Processing Pipeline:
#      1. Channel conversion (optional mono downmix)
#      2. Fade in/out (with overlap protection)
#      3. Normalization (Peak/RMS/LUFS)
#      4. Soft limiting (prevents clipping after normalization)
#      5. Export with metadata embedding
#    
#    • Opus Sample Rate Handling:
#      - Automatically resamples to nearest supported rate (8/12/16/24/48 kHz)
#      - Ensures compatibility with Opus encoder requirements

# ░▒▓ CHANGELOG
#    - v3.0.1 (Current - Critical Fix):
#        • FIXED: Properly JSON serializes workflow data before size check
#        • FIXED: Enhanced error logging in save function
#        • FIXED: Returns success status from save operation
#        • All features from v3.0.0 retained and stabilized
#    
#    - v3.0.0:
#        • Implemented smart metadata handling with size-based sidecar fallback
#        • Added user-facing warnings when sidecar files are created
#        • FLAC always embeds metadata regardless of size
#    
#    - v2.x.x:
#        • Various attempts at metadata embedding strategies
#        • Restored features from working v0.4.1 baseline
#    
#    - v1.0.x - v0.9.x:
#        • LUFS normalization with triple fallback system
#        • Spectrogram generation with 5 colormap options
#        • Grid overlay for waveform visualization
#    
#    - v0.4.1 (Baseline):
#        • Proven metadata embedding pattern
#        • Core audio processing features established

# ░▒▓ USAGE TIPS
#    • For best quality MP3s: Use V0 setting (highest VBR quality)
#    • For archival: Use FLAC (lossless, unlimited metadata size)
#    • For efficiency: Use OPUS at 128k (excellent quality/size ratio)
#    • For large workflows: FLAC embeds everything; MP3/Opus use sidecar if needed
#    • For mastering: Enable LUFS normalization + limiter for broadcast-ready audio

# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄

import os
import torch
import numpy as np
import io as built_in_io
import time
import json
import random
import subprocess
import torchaudio
import av
import tempfile
import traceback
from typing import Dict, List, Tuple, Optional, Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image

import folder_paths
from comfy.cli_args import args
from ..core import io as md_io

# ═══════════════════════════════════════════════════════════════════════════════
# DEPENDENCY CHECKS
# ═══════════════════════════════════════════════════════════════════════════════

try:
    import pedalboard
    _pedalboard_available = True
    print(f"[{__name__}] ✓ pedalboard available - soft limiter enabled")
except ImportError:
    _pedalboard_available = False
    print(f"[{__name__}] ⚠ pedalboard not found - limiter will use hard clipping fallback")

try:
    import pyloudnorm as pyln
    _pyloudnorm_available = True
    print(f"[{__name__}] ✓ pyloudnorm available - fast LUFS normalization enabled")
except ImportError:
    _pyloudnorm_available = False
    print(f"[{__name__}] ⚠ pyloudnorm not found - LUFS will use FFmpeg fallback")

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

AUDIO_OUTPUT_DIR: str = os.path.join(folder_paths.get_output_directory(), "ComfyUI_AdvancedAudioOutputs")
os.makedirs(AUDIO_OUTPUT_DIR, exist_ok=True)
print(f"[{__name__}] Audio output directory: {AUDIO_OUTPUT_DIR}")

# Metadata size limit in KB - MP3/Opus files larger than this will use sidecar JSON
METADATA_SIZE_LIMIT_KB: int = 256

# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def validate_color(color_str: str, default: str) -> str:
    """Validates matplotlib color string (supports names, hex codes, RGB tuples)."""
    try:
        mcolors.to_rgb(color_str)
        return color_str
    except (ValueError, AttributeError):
        print(f"[{__name__}] Warning: Invalid color '{color_str}', using '{default}'")
        return default

def generate_unique_counter() -> int:
    """Generates truly unique counter for file naming (timestamp + random)."""
    return int(time.time() * 1000) + random.randint(0, 9999)

def apply_fades(audio: torch.Tensor, sr: int, fin: int, fout: int) -> torch.Tensor:
    """Applies fade in/out with automatic overlap protection."""
    _, n_samples = audio.shape
    fin_s = int(sr * fin / 1000.0)
    fout_s = int(sr * fout / 1000.0)
    
    # Prevent overlapping fades on short audio
    if fin_s + fout_s > n_samples and (fin_s + fout_s) > 0:
        scale = n_samples * 0.9 / (fin_s + fout_s)
        fin_s = int(fin_s * scale)
        fout_s = int(fout_s * scale)
        print(f"[{__name__}] Scaled fades to prevent overlap: {fin_s} + {fout_s} samples")
    
    if fin_s > 0: 
        audio[:, :fin_s] *= torch.linspace(0., 1., fin_s, device=audio.device)
    if fout_s > 0: 
        audio[:, -fout_s:] *= torch.linspace(1., 0., fout_s, device=audio.device)
    return audio

def lufs_normalize_with_ffmpeg(audio_tensor: torch.Tensor, sample_rate: int, target_lufs: float) -> Optional[torch.Tensor]:
    """
    LUFS normalization using FFmpeg's loudnorm filter (fallback when pyloudnorm unavailable).
    Returns normalized audio tensor or None if FFmpeg fails.
    """
    try:
        if audio_tensor.ndim == 1:
            audio_tensor = audio_tensor.unsqueeze(0)

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_in, \
             tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_out:
            input_path, output_path = tmp_in.name, tmp_out.name

        try:
            torchaudio.save(input_path, audio_tensor.cpu(), sample_rate, format="WAV")
            creationflags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            cmd = [
                'ffmpeg', '-i', input_path,
                '-filter_complex', f'loudnorm=I={target_lufs}:LRA=7:tp=-1',
                '-ar', str(sample_rate), '-y', output_path
            ]
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                         check=True, creationflags=creationflags)
            normalized_audio, _ = torchaudio.load(output_path)
            return normalized_audio.to(audio_tensor.device)
        finally:
            for path in [input_path, output_path]:
                if os.path.exists(path):
                    os.unlink(path)

    except (subprocess.CalledProcessError, FileNotFoundError, Exception) as e:
        print(f"[{__name__}] FFmpeg loudnorm failed: {str(e)[:300]}")
        return None

def create_spectrogram_image(audio_tensor: torch.Tensor, sample_rate: int, 
                            width: int, height: int, colormap: str = "viridis") -> torch.Tensor:
    """Generates frequency-domain spectrogram visualization."""
    if audio_tensor is None or audio_tensor.numel() == 0:
        return torch.zeros((1, height, width, 3))
    try:
        audio_np = audio_tensor.cpu().numpy()
        if audio_np.ndim > 1:
            audio_np = audio_np[0]

        dpi = 100
        fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
        ax.specgram(audio_np, Fs=sample_rate, NFFT=2048, noverlap=1536, cmap=colormap)
        ax.axis('off')
        fig.tight_layout(pad=0)
        
        with built_in_io.BytesIO() as buffer:
            plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0)
            buffer.seek(0)
            pil_image = Image.open(buffer).convert("RGB").resize((width, height), Image.LANCZOS)
            spec_tensor = (torch.from_numpy(np.array(pil_image)).float() / 255.0).unsqueeze(0)
        
        plt.close(fig)
        return spec_tensor
    except Exception as e:
        print(f"[{__name__}] ERROR: Failed to generate spectrogram: {e}")
        return torch.zeros((1, height, width, 3), dtype=torch.float32)

def save_metadata_sidecar(audio_filepath: str, metadata: Dict[str, Any]) -> None:
    """Saves metadata as a .json sidecar file (used when metadata too large for audio format)."""
    json_path = os.path.splitext(audio_filepath)[0] + ".json"
    try:
        # Parse JSON strings back to dicts for cleaner sidecar file
        parsed_metadata = {}
        for key, value in metadata.items():
            try:
                parsed_metadata[key] = json.loads(value) if isinstance(value, str) else value
            except:
                parsed_metadata[key] = value
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(parsed_metadata, f, indent=2)
        print(f"[{__name__}] Metadata sidecar saved: {json_path}")
    except Exception as e:
        print(f"[{__name__}] FAILED to save metadata sidecar: {e}")

def _save_audio_with_av(
    waveform_tensor: torch.Tensor, sample_rate: int, output_path: str,
    file_format: str, metadata: Dict[str, str], quality_setting: Optional[str] = None
) -> bool:
    """
    Encodes and saves audio file using PyAV with embedded metadata.
    Returns True on success, False on failure.
    """
    try:
        waveform_tensor = waveform_tensor.to(torch.float32).cpu()
        if waveform_tensor.ndim == 3: 
            waveform_tensor = waveform_tensor[0]
        if waveform_tensor.ndim == 1: 
            waveform_tensor = waveform_tensor.unsqueeze(0)
        
        target_sample_rate = sample_rate
        OPUS_RATES = [8000, 12000, 16000, 24000, 48000]
        
        # Handle Opus's strict sample rate requirements
        if file_format == "opus":
            if target_sample_rate not in OPUS_RATES:
                target_sample_rate = min(OPUS_RATES, key=lambda r: abs(r - target_sample_rate))
                waveform_tensor = torchaudio.functional.resample(
                    waveform_tensor, sample_rate, target_sample_rate
                )
                sample_rate = target_sample_rate
                print(f"[{__name__}] Resampled to {target_sample_rate}Hz for Opus compatibility")

        with built_in_io.BytesIO() as wav_buffer, built_in_io.BytesIO() as out_buffer:
            torchaudio.save(wav_buffer, waveform_tensor, sample_rate, format="WAV")
            wav_buffer.seek(0)
            
            with av.open(wav_buffer) as in_container, \
                 av.open(out_buffer, mode='w', format=file_format) as out_container:
                
                # Embed metadata in container (ComfyUI-compatible format)
                if metadata:
                    for key, value in metadata.items():
                        out_container.metadata[key] = str(value)
                
                stream_kwargs = {"rate": sample_rate}
                codec_name = {"mp3": "libmp3lame", "opus": "libopus", "flac": "flac"}.get(
                    file_format, 'aac'
                )
                
                # Configure codec-specific quality settings
                if codec_name == "libmp3lame":
                    if quality_setting == "V0": 
                        stream_kwargs['qscale'] = 1  # VBR highest quality
                    elif quality_setting == "128k": 
                        stream_kwargs['bit_rate'] = 128000
                    elif quality_setting == "320k": 
                        stream_kwargs['bit_rate'] = 320000
                elif codec_name == "libopus" and quality_setting:
                    stream_kwargs['bit_rate'] = int(quality_setting.replace('k', '')) * 1000

                out_stream = out_container.add_stream(codec_name, **stream_kwargs)
                
                # Encode audio frames
                for frame in in_container.decode(audio=0):
                    frame.pts = None
                    for packet in out_stream.encode(frame): 
                        out_container.mux(packet)
                
                # Flush encoder
                for packet in out_stream.encode(None): 
                    out_container.mux(packet)
            
            # Write final file to disk
            with open(output_path, 'wb') as f:
                f.write(out_buffer.getvalue())
        
        print(f"[{__name__}] ✓ Audio saved successfully: {output_path}")
        return True
        
    except Exception as e:
        print(f"[{__name__}] ✗ ERROR in _save_audio_with_av: {e}")
        traceback.print_exc()
        return False

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN NODE CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class AdvancedAudioPreviewAndSave:
    """
    Professional audio processing and export node for ComfyUI.
    Handles normalization, effects, visualization, and intelligent metadata embedding.
    """
    
    CATEGORY = "MD_Nodes/Save"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio_input": ("AUDIO",),
                "filename_prefix": ("STRING", {"default": "ComfyUI_audio_%Y-%m-%d"}),
                "save_format": (["MP3", "FLAC", "OPUS"],),
            },
            "optional": {
                "save_to_disk": ("BOOLEAN", {"default": True}),
                "save_metadata": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Embeds workflow JSON. Large workflows use sidecar .json for MP3/Opus"
                }),
                "custom_notes": ("STRING", {"default": "", "multiline": True}),
                "channel_mode": (["Keep Original", "Convert to Mono"],),
                "fade_in_ms": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 10}),
                "fade_out_ms": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 10}),
                "normalize_method": (["Off", "Peak", "RMS", "LUFS"],),
                "target_rms_db": ("INT", {"default": -16, "min": -60, "max": 0, "step": 1}),
                "target_lufs_db": ("FLOAT", {"default": -14.0, "min": -50.0, "max": -5.0, "step": 0.5}),
                "use_limiter": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Soft limiter prevents clipping. Requires 'pedalboard' package"
                }),
                "mp3_quality": (["V0", "128k", "320k"],),
                "opus_quality": (["64k", "96k", "128k"],),
                "waveform_width": ("INT", {"default": 800, "min": 100, "max": 2048, "step": 16}),
                "waveform_height": ("INT", {"default": 150, "min": 50, "max": 1024, "step": 16}),
                "waveform_color": ("STRING", {"default": "hotpink"}),
                "waveform_background_color": ("STRING", {"default": "black"}),
                "show_grid": ("BOOLEAN", {"default": False}),
                "generate_spectrogram": ("BOOLEAN", {"default": False}),
                "spectrogram_colormap": (["viridis", "plasma", "inferno", "magma", "cividis"],),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("AUDIO", "IMAGE", "IMAGE",)
    RETURN_NAMES = ("AUDIO", "WAVEFORM_IMAGE", "SPECTROGRAM_IMAGE",)
    FUNCTION = "process_audio"
    OUTPUT_NODE = True

    def _prepare_metadata(self, **kwargs: Any) -> Dict[str, str]:
        """
        Prepares metadata dictionary with proper JSON serialization.
        Uses v0.4.1's proven pattern for ComfyUI compatibility.
        """
        metadata = {}
        if not kwargs.get("save_metadata") or args.disable_metadata:
            return metadata

        prompt = kwargs.get("prompt")
        extra_pnginfo = kwargs.get("extra_pnginfo")
        
        # Serialize prompt to JSON string
        if prompt is not None:
            metadata["prompt"] = json.dumps(prompt)
        
        # Serialize all extra_pnginfo items (includes workflow)
        # This is the v0.4.1 pattern that works correctly
        if extra_pnginfo is not None:
            for key, value in extra_pnginfo.items():
                # JSON serialize if not already a string
                metadata[key] = json.dumps(value) if not isinstance(value, str) else value

        # Add custom notes as plain text
        if kwargs.get("custom_notes"):
            metadata['ComfyUI_Notes'] = kwargs.get("custom_notes")
            
        return metadata

    def _normalize_audio(self, audio_tensor: torch.Tensor, sample_rate: int, **kwargs: Any) -> torch.Tensor:
        """
        Applies selected normalization method with appropriate fallbacks.
        Supports Peak, RMS, and LUFS normalization.
        """
        method = kwargs.get("normalize_method")
        
        if method == "Peak":
            peak_val = torch.max(torch.abs(audio_tensor))
            if peak_val > 1e-6:
                audio_tensor = audio_tensor / peak_val * 0.99
                print(f"[{self.__class__.__name__}] Peak normalized to 0.99")
            else:
                print(f"[{self.__class__.__name__}] Audio too quiet for peak normalization")
                
        elif method == "RMS":
            current_rms = torch.sqrt(torch.mean(audio_tensor**2))
            target_rms = 10**(kwargs.get("target_rms_db", -16) / 20.0)
            if current_rms > 1e-9:
                audio_tensor *= (target_rms / current_rms)
                print(f"[{self.__class__.__name__}] RMS normalized to {kwargs.get('target_rms_db')}dB")
            else:
                print(f"[{self.__class__.__name__}] Audio too quiet for RMS normalization")
                
        elif method == "LUFS":
            target_lufs = kwargs.get("target_lufs_db", -14.0)
            result = None
            
            # Try pyloudnorm first (fastest)
            if _pyloudnorm_available:
                try:
                    meter = pyln.Meter(sample_rate)
                    loudness = meter.integrated_loudness(audio_tensor.cpu().numpy().T)
                    if loudness > -70.0:
                        normalized_np = pyln.normalize.loudness(
                            audio_tensor.cpu().numpy().T, loudness, target_lufs
                        )
                        result = torch.from_numpy(normalized_np.T).to(audio_tensor.device)
                        print(f"[{self.__class__.__name__}] LUFS normalized with pyloudnorm")
                except Exception as e:
                    print(f"[{self.__class__.__name__}] pyloudnorm failed: {e}")
            
            # Fall back to FFmpeg
            if result is None:
                result = lufs_normalize_with_ffmpeg(audio_tensor, sample_rate, target_lufs)
                if result is not None:
                    print(f"[{self.__class__.__name__}] LUFS normalized with FFmpeg")

            # Final fallback to Peak if LUFS completely fails
            if result is not None:
                audio_tensor = result
            else:
                print(f"[{self.__class__.__name__}] LUFS failed, falling back to Peak normalization")
                peak_val = torch.max(torch.abs(audio_tensor))
                if peak_val > 1e-6:
                    audio_tensor = audio_tensor / peak_val * 0.99
        
        # Apply soft limiter to prevent clipping (if enabled and normalized)
        if kwargs.get("use_limiter") and method != "Off":
            if _pedalboard_available:
                board = pedalboard.Pedalboard([
                    pedalboard.Limiter(threshold_db=-1.0, release_ms=50)
                ])
                limited_audio = torch.from_numpy(
                    board(audio_tensor.cpu().numpy(), sample_rate)
                ).to(audio_tensor.device)
                print(f"[{self.__class__.__name__}] Applied soft limiter")
                return limited_audio
            else:
                # Hard clipping fallback
                return torch.clamp(audio_tensor, -1.0, 1.0)
        
        return audio_tensor

    def process_audio(self, **kwargs: Any) -> Dict[str, Any]:
        """Main processing function - handles entire audio pipeline."""
        
        waveform_width = kwargs.get("waveform_width", 800)
        waveform_height = kwargs.get("waveform_height", 150)
        audio_input = kwargs.get("audio_input")

        # Load and validate audio input
        try:
            waveform_original, samplerate = md_io.audio_from_comfy_3d(audio_input, try_gpu=True)
            print(f"[{self.__class__.__name__}] Loaded audio: {waveform_original.shape}, {samplerate}Hz")
        except Exception as e:
            empty = torch.zeros((1, waveform_height, waveform_width, 3))
            return {
                "ui": {"text": [f"Error loading audio: {e}"]}, 
                "result": ({"waveform": torch.empty(0), "sample_rate": 0}, empty, empty)
            }

        if waveform_original.numel() == 0:
            empty = torch.zeros((1, waveform_height, waveform_width, 3))
            return {
                "ui": {"text": ["Warning: Empty audio input"]}, 
                "result": (audio_input, empty, empty)
            }
        
        # Extract working audio from batch
        processed_audio = waveform_original[0] if waveform_original.ndim == 3 else waveform_original.clone()
        
        # === AUDIO PROCESSING PIPELINE ===
        
        # 1. Channel conversion
        if kwargs.get("channel_mode") == "Convert to Mono" and processed_audio.shape[0] > 1:
            processed_audio = torch.mean(processed_audio, dim=0, keepdim=True)
            print(f"[{self.__class__.__name__}] Converted to mono")
        
        # 2. Fades (with overlap protection)
        processed_audio = apply_fades(
            processed_audio, samplerate, 
            kwargs.get("fade_in_ms", 0), 
            kwargs.get("fade_out_ms", 0)
        )
        
        # 3. Normalization + Limiting
        processed_audio = self._normalize_audio(processed_audio, samplerate, **kwargs)
        
        # === FILE SAVING ===
        
        ui_text = []
        if kwargs.get("save_to_disk"):
            try:
                base_prefix = time.strftime(
                    os.path.basename(kwargs.get("filename_prefix")), 
                    time.localtime()
                )
                subfolder = os.path.dirname(kwargs.get("filename_prefix"))
            except Exception as e:
                print(f"[{self.__class__.__name__}] Invalid filename prefix: {e}")
                base_prefix = "ComfyUI_audio"
                subfolder = ""
            
            output_dir = os.path.join(AUDIO_OUTPUT_DIR, subfolder)
            os.makedirs(output_dir, exist_ok=True)
            
            save_format = kwargs.get("save_format", "MP3").lower()
            file = f"{base_prefix}_{generate_unique_counter():09}.{save_format}"
            output_path = os.path.join(output_dir, file)
            quality = kwargs.get("mp3_quality") if save_format == "mp3" else kwargs.get("opus_quality")
            
            try:
                # Prepare metadata with proper JSON serialization
                metadata = self._prepare_metadata(**kwargs)
                metadata_to_embed = metadata.copy()
                
                # Smart metadata handling: check size and use sidecar if needed
                if metadata:
                    metadata_size = sum(len(v.encode('utf-8')) for v in metadata.values()) / 1024
                    
                    if save_format in ["mp3", "opus"] and metadata_size > METADATA_SIZE_LIMIT_KB:
                        warning = (
                            f"⚠ Workflow metadata ({metadata_size:.1f}KB) exceeds {save_format.upper()} "
                            f"limit. Saving to sidecar .json file."
                        )
                        print(f"[{self.__class__.__name__}] {warning}")
                        ui_text.append(warning)
                        
                        # Save full metadata to sidecar
                        save_metadata_sidecar(output_path, metadata)
                        
                        # Only embed small metadata in audio file
                        metadata_to_embed = {"ComfyUI_Notes": metadata.get("ComfyUI_Notes", "")} if "ComfyUI_Notes" in metadata else {}
                    else:
                        print(f"[{self.__class__.__name__}] Embedding {metadata_size:.1f}KB metadata")
                
                # Save audio file
                success = _save_audio_with_av(
                    processed_audio, samplerate, output_path, 
                    save_format, metadata_to_embed, quality
                )
                
                if success:
                    rel_path = os.path.relpath(output_path, folder_paths.get_output_directory())
                    ui_text.insert(0, f"✓ Saved: {rel_path}")
                else:
                    ui_text.append("✗ ERROR: File save failed - check console for details")
                    
            except Exception as e:
                error_msg = f"✗ Save error: {e}"
                print(f"[{self.__class__.__name__}] {error_msg}")
                traceback.print_exc()
                ui_text.append(error_msg)
        else:
            ui_text.append("ℹ Not saved to disk")

        # === WAVEFORM VISUALIZATION ===
        
        try:
            audio_plot = processed_audio.cpu().numpy()[0]
            time_axis = np.linspace(0, len(audio_plot) / samplerate, len(audio_plot))
            dpi = 100
            fig, ax = plt.subplots(figsize=(waveform_width / dpi, waveform_height / dpi), dpi=dpi)
            
            bg_color = validate_color(kwargs.get("waveform_background_color"), "black")
            wf_color = validate_color(kwargs.get("waveform_color"), "hotpink")
            
            ax.set_facecolor(bg_color)
            fig.patch.set_facecolor(bg_color)
            ax.plot(time_axis, audio_plot, color=wf_color, linewidth=0.5)
            ax.set_ylim([-1.05, 1.05])

            if kwargs.get("show_grid", False):
                ax.grid(True, alpha=0.3, color='gray', linestyle='--')
            else:
                ax.axis("off")
            
            fig.tight_layout(pad=0)

            with built_in_io.BytesIO() as buffer:
                plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0)
                buffer.seek(0)
                pil_img = Image.open(buffer).convert("RGB").resize(
                    (waveform_width, waveform_height), Image.LANCZOS
                )
                img_array = np.array(pil_img)
                waveform_image = (torch.from_numpy(img_array).float() / 255.0).unsqueeze(0)
            
            plt.close(fig)
            
        except Exception as e:
            print(f"[{self.__class__.__name__}] Waveform generation failed: {e}")
            waveform_image = torch.zeros((1, waveform_height, waveform_width, 3))
        
        # === SPECTROGRAM VISUALIZATION ===
        
        if kwargs.get("generate_spectrogram", False):
            spectrogram_image = create_spectrogram_image(
                processed_audio, samplerate, waveform_width, waveform_height,
                kwargs.get("spectrogram_colormap", "viridis")
            )
        else:
            spectrogram_image = torch.zeros((1, waveform_height, waveform_width, 3))

        # === PREPARE OUTPUT ===
        
        # Ensure processed audio has correct dimensions for output
        processed_for_output = processed_audio.unsqueeze(0) if processed_audio.ndim == 2 else processed_audio
        final_audio_output = {"waveform": processed_for_output, "sample_rate": samplerate}
        
        return {
            "ui": {"text": ui_text}, 
            "result": (final_audio_output, waveform_image, spectrogram_image)
        }

# ═══════════════════════════════════════════════════════════════════════════════
# NODE REGISTRATION
# ═══════════════════════════════════════════════════════════════════════════════

NODE_CLASS_MAPPINGS = {
    "AdvancedAudioPreviewAndSave": AdvancedAudioPreviewAndSave
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AdvancedAudioPreviewAndSave": "Advanced Audio Preview & Save"
}