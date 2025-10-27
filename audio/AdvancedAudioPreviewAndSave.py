# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
# ████ MD_Nodes/AdvancedAudioPreviewAndSave – Audio preview, normalization & export ████▓▒░
# © 2025 MDMAchine
# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
# ░▒▓ ORIGIN & DEV:
#   • Cast into the void by: MDMAchine (waveform wizard)
#   • Enhanced by: Claude (Anthropic AI), Gemini (Google AI)
#   • License: Apache 2.0 — Sharing is caring

# ░▒▓ DESCRIPTION:
#   A comprehensive audio processing and export node for ComfyUI workflows.
#   Provides professional-grade audio normalization, effects, visualization,
#   and intelligent metadata embedding with workflow preservation.

# ░▒▓ FEATURES:
#   ✓ Multi-format export: MP3 (universal), FLAC (lossless), OPUS (efficient)
#   ✓ Advanced normalization: Peak, RMS, and LUFS (with pyloudnorm/FFmpeg fallback)
#   ✓ Professional effects: Fade in/out, soft limiting (pedalboard), mono conversion
#   ✓ Smart metadata handling: Embeds workflow JSON with automatic sidecar fallback
#   ✓ Visual feedback: Real-time "before" and "after" waveform generation
#   ✓ Drag-and-drop support: Saved files/sidecars restore workflows
#   ✓ Dynamic filename templating: Supports strftime patterns (e.g., "audio_%Y-%m-%d")
#   ✓ Cacheable: User-controlled caching via "force_save" (Guide 7.1)

# ░▒▓ CHANGELOG:
#   - v1.4.3 (Plotting Fix - Oct 2025):
#       • FIXED: Reverted `_plot_waveform_to_tensor` logic back to the
#         `fig.savefig(buf)` method. The `canvas.draw()` method was
#         failing and returning a blank 64x64 tensor.
#   - v1.4.2 (Guide Update - Oct 2025):
#       • UPDATED: Converted to ComfyUI_MD_Nodes Guide v1.4.2 standards.
#       • FIXED: Removed all Python type hints (Guide 6.2).
#       • ADDED: Standardized docstrings for all methods (Guide 5.2).
#       • ADDED: Standardized tooltips for all inputs (Guide 8.1).
#       • ADDED: `force_save` input and `IS_CHANGED` logic (Guide 7.1, Approach #2+4).
#       • FIXED: Imports re-ordered to guide standard (Guide 6.4).
#       • FIXED: Added `MD: ` prefix to display name (Guide 5.4).
#       • FIXED: Set `OUTPUT_NODE = True` (Guide 5.4).
#   - v3.0.8 (Original Base - Sidecar Format Fix):
#       • (Features from original base retained)

# ░▒▓ CONFIGURATION:
#   → Primary Use: Archival (FLAC) or high-quality (V0 MP3) audio export with full workflow metadata.
#   → Secondary Use: Mastering audio with LUFS normalization + limiter for broadcast-ready levels.
#   → Edge Use: Embedding massive workflows (>256KB) via FLAC tags or automatic MP3/Opus sidecar files.

# ░▒▓ WARNING:
#   This node may trigger:
#   ▓▒░ Obsessive-compulsive waveform zooming, looking for a single clipped sample.
#   ▓▒░ Realizing your workflow .json is bigger than a 90s-era 64k intro.
#   ▓▒░ A violent, uncontrollable urge to switch to Impulse Tracker and code a .MOD file.
#   Consult your nearest demoscene vet if hallucinations persist.
# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄


# =================================================================================
# == Standard Library Imports                                                    ==
# =================================================================================
import os
import io as built_in_io  # Original io import
import io                 # Added for new plot function
import time
import json
import random
import secrets            # For IS_CHANGED
import subprocess
import tempfile
import traceback

# =================================================================================
# == Third-Party Imports                                                         ==
# =================================================================================
import torch
import numpy as np
import torchaudio
import av
import matplotlib
matplotlib.use('Agg')       # Set backend before pyplot import
import matplotlib.pyplot as plt
from PIL import Image

# =================================================================================
# == ComfyUI Core Modules                                                        ==
# =================================================================================
import folder_paths
from comfy.cli_args import args

# =================================================================================
# == Local Project Imports                                                       ==
# =================================================================================
# --- Fallback md_io (Assuming '../core/io' exists or defining a fallback) ---
try:
    from ..core import io as md_io
except (ImportError, ValueError):
    print(f"[{__name__}] Warning: Could not import md_io from relative path '../core/io'. Using fallback.")
    class MockMdIo:
        def audio_from_comfy_3d(self, audio_data, try_gpu=True):
            if not isinstance(audio_data, dict) or 'waveform' not in audio_data or 'sample_rate' not in audio_data:
                raise ValueError("Fallback md_io expects dict with 'waveform' and 'sample_rate'")
            waveform = audio_data['waveform']
            sample_rate = audio_data['sample_rate']
            if waveform.ndim == 1: waveform = waveform.unsqueeze(0)
            elif waveform.ndim == 3: waveform = waveform[0]
            device = 'cuda' if try_gpu and torch.cuda.is_available() else 'cpu'
            return waveform.to(device), sample_rate
    md_io = MockMdIo()

# =================================================================================
# == Helper Classes & Dependencies                                               ==
# =================================================================================

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

def generate_unique_counter():
    """
    Generates truly unique counter for file naming (timestamp + random).
    
    Returns:
        int: Unique counter value.
    """
    return int(time.time() * 1000) + random.randint(0, 9999)

def apply_fades(audio, sr, fin, fout):
    """
    Applies fade in/out with automatic overlap protection.
    
    Args:
        audio (torch.Tensor): Audio tensor [channels, samples].
        sr (int): Sample rate.
        fin (int): Fade in time in ms.
        fout (int): Fade out time in ms.
    
    Returns:
        torch.Tensor: Faded audio tensor.
    """
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

def lufs_normalize_with_ffmpeg(audio_tensor, sample_rate, target_lufs):
    """
    LUFS normalization using FFmpeg's loudnorm filter (fallback when pyloudnorm unavailable).
    
    Args:
        audio_tensor (torch.Tensor): Input audio tensor.
        sample_rate (int): Sample rate.
        target_lufs (float): Target LUFS level.
        
    Returns:
        torch.Tensor or None: Normalized audio tensor or None if FFmpeg fails.
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

def save_metadata_sidecar(audio_filepath, full_prompt_api_dict, custom_notes=None):
    """
    Saves metadata as a .json sidecar file in ComfyUI's expected export format.
    Requires the full prompt API dictionary.
    
    Args:
        audio_filepath (str): Path to the saved audio file.
        full_prompt_api_dict (dict): The full "prompt" dict containing workflow data.
        custom_notes (str, optional): Additional notes to embed.
        
    Returns:
        bool: True on success, False on failure.
    """
    json_path = os.path.splitext(audio_filepath)[0] + ".json"
    if not full_prompt_api_dict:
        print(f"[{__name__}] ERROR: Cannot save sidecar JSON, full workflow data is missing.")
        return False
    
    try:
        # The full_prompt_api_dict should contain both 'prompt' and 'workflow' or 'extra_pnginfo'
        # We need to extract the workflow data which is usually in extra_pnginfo['workflow']
        
        # Try to get workflow from extra_pnginfo first (this is the standard location)
        workflow_data = None
        if 'extra_pnginfo' in full_prompt_api_dict:
            extra_info = full_prompt_api_dict['extra_pnginfo']
            if isinstance(extra_info, dict) and 'workflow' in extra_info:
                # Parse if it's a JSON string
                workflow_str = extra_info['workflow']
                workflow_data = json.loads(workflow_str) if isinstance(workflow_str, str) else workflow_str
        
        # Fallback: check if 'workflow' is directly in the dict
        if not workflow_data and 'workflow' in full_prompt_api_dict:
            wf = full_prompt_api_dict['workflow']
            workflow_data = json.loads(wf) if isinstance(wf, str) else wf
        
        if not workflow_data:
            print(f"[{__name__}] ERROR: Could not find workflow data in full_prompt_api_dict")
            return False
        
        # workflow_data should now be a dict with the ComfyUI structure
        # Make sure it has the required fields
        data_to_save = {
            "last_node_id": workflow_data.get("last_node_id", 0),
            "last_link_id": workflow_data.get("last_link_id", 0),
            "nodes": workflow_data.get("nodes", []),
            "links": workflow_data.get("links", []),
            "groups": workflow_data.get("groups", []),
            "config": workflow_data.get("config", {}),
            "extra": workflow_data.get("extra", {}),
            "version": workflow_data.get("version", 0.4)
        }
        
        # Add optional fields if present
        if "id" in workflow_data:
            data_to_save["id"] = workflow_data["id"]
        if "revision" in workflow_data:
            data_to_save["revision"] = workflow_data["revision"]
        
        # Add custom notes to extra if provided
        if custom_notes:
            if "extra" not in data_to_save:
                data_to_save["extra"] = {}
            data_to_save["extra"]["ComfyUI_Notes"] = custom_notes
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, indent=2)
        
        print(f"[{__name__}] ✓ Sidecar saved (ComfyUI Export Format): {json_path}")
        return True
        
    except Exception as e:
        print(f"[{__name__}] ✗ FAILED to save metadata sidecar: {e}")
        traceback.print_exc()
        return False

def _save_audio_with_av(
    waveform_tensor, sample_rate, output_path,
    file_format, metadata, quality_setting=None
):
    """
    Encodes and saves audio file using PyAV with embedded metadata.
    
    Args:
        waveform_tensor (torch.Tensor): Audio data.
        sample_rate (int): Sample rate.
        output_path (str): File path to save to.
        file_format (str): "mp3", "flac", or "opus".
        metadata (dict): Metadata dict (all values MUST be strings, per Guide 7.5).
        quality_setting (str, optional): Quality string (e.g., "V0", "320k").
        
    Returns:
        bool: True on success, False on failure.
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
                        # CRITICAL: Guide 7.5 - Ensure value is a string for PyAV
                        out_container.metadata[key] = str(value)
                
                stream_kwargs = {"rate": sample_rate}
                codec_name = {"mp3": "libmp3lame", "opus": "libopus", "flac": "flac"}.get(
                    file_format, 'aac'
                )
                
                # Configure codec-specific quality settings
                if codec_name == "libmp3lame":
                    if quality_setting == "V0": 
                        stream_kwargs['qscale'] = 1 # VBR highest quality
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

# =================================================================================
# == Core Node Class                                                             ==
# =================================================================================

class AdvancedAudioPreviewAndSave:
    """
    Professional audio processing and export node for ComfyUI.
    Handles normalization, effects, visualization, and intelligent metadata embedding.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        """
        Define all input parameters with tooltips.
        """
        return {
            "required": {
                "audio_input": ("AUDIO",),
                "filename_prefix": ("STRING", {
                    "default": "ComfyUI_audio_%Y-%m-%d",
                    "tooltip": (
                        "FILENAME PREFIX\n"
                        "- Base name for the saved audio file.\n"
                        "- Supports strftime patterns (e.g., %Y-%m-%d, %H-%M-%S).\n"
                        "- A unique counter is appended to avoid overwrites."
                    )
                }),
                "save_format": (["MP3", "FLAC", "OPUS"], {
                    "tooltip": (
                        "SAVE FORMAT\n"
                        "- MP3: Best compatibility, good compression.\n"
                        "- FLAC: Lossless, archival quality, supports large metadata.\n"
                        "- OPUS: Modern, high-efficiency, good for web/streaming."
                    )
                }),
            },
            "optional": {
                "save_to_disk": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "SAVE TO DISK\n"
                        "- True: Saves the processed audio file to disk.\n"
                        "- False: Bypasses saving, node acts as processor/previewer."
                    )
                }),
                "force_save": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "FORCE SAVE (CACHE CONTROL)\n"
                        "- False (default): Only save if inputs change (efficient caching).\n"
                        "- True: Always re-run and save, even if inputs are identical."
                    )
                }),
                "save_metadata": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "SAVE METADATA\n"
                        "- Embeds the workflow JSON into the audio file.\n"
                        "- Large workflows (>256KB) use a sidecar .json for MP3/Opus."
                    )
                }),
                "custom_notes": ("STRING", {
                    "default": "", 
                    "multiline": True,
                    "tooltip": "Custom text notes to be embedded in the metadata."
                }),
                "channel_mode": (["Keep Original", "Convert to Mono"], {
                    "tooltip": (
                        "CHANNEL MODE\n"
                        "- Keep Original: Retains the original channel count (e.g., stereo).\n"
                        "- Convert to Mono: Averages all channels into a single mono track."
                    )
                }),
                "fade_in_ms": ("INT", {
                    "default": 0, "min": 0, "max": 10000, "step": 10,
                    "tooltip": (
                        "FADE IN (MS)\n"
                        "- Duration of the fade-in effect in milliseconds.\n"
                        "- 0 = no fade-in."
                    )
                }),
                "fade_out_ms": ("INT", {
                    "default": 0, "min": 0, "max": 10000, "step": 10,
                    "tooltip": (
                        "FADE OUT (MS)\n"
                        "- Duration of the fade-out effect in milliseconds.\n"
                        "- 0 = no fade-out."
                    )
                }),
                "normalize_method": (["Off", "Peak", "RMS", "LUFS"], {
                    "tooltip": (
                        "NORMALIZATION METHOD\n"
                        "- Off: No normalization.\n"
                        "- Peak: Normalizes to 0.99 peak amplitude (prevents clipping).\n"
                        "- RMS: Normalizes to a target average power level.\n"
                        "- LUFS: Normalizes to a target perceived loudness (broadcast standard)."
                    )
                }),
                "target_rms_db": ("INT", {
                    "default": -16, "min": -60, "max": 0, "step": 1,
                    "tooltip": (
                        "TARGET RMS (DB)\n"
                        "- Target average power level (in dB) for RMS normalization.\n"
                        "- Lower values (e.g., -18) = quieter, more dynamic.\n"
                        "- Higher values (e.g., -12) = louder, less dynamic."
                    )
                }),
                "target_lufs_db": ("FLOAT", {
                    "default": -14.0, "min": -50.0, "max": -5.0, "step": 0.5,
                    "tooltip": (
                        "TARGET LUFS (DB)\n"
                        "- Target perceived loudness (in dB) for LUFS normalization.\n"
                        "- -14 LUFS is a common standard for streaming."
                    )
                }),
                "use_limiter": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "USE LIMITER\n"
                        "- Applies a soft limiter after normalization to prevent clipping.\n"
                        "- Highly recommended when normalizing (Peak, RMS, LUFS).\n"
                        "- Requires the 'pedalboard' package."
                    )
                }),
                "mp3_quality": (["V0", "128k", "320k"], {
                    "tooltip": (
                        "MP3 QUALITY\n"
                        "- V0: Highest quality VBR (Variable Bitrate). Recommended.\n"
                        "- 320k: Highest quality CBR (Constant Bitrate).\n"
                        "- 128k: Smaller file size, good for previews."
                    )
                }),
                "opus_quality": (["64k", "96k", "128k"], {
                    "tooltip": (
                        "OPUS QUALITY (BITRATE)\n"
                        "- 128k: High quality, comparable to high-bitrate MP3.\n"
                        "- 96k: Good quality, very efficient.\n"
                        "- 64k: Good for speech or mono audio."
                    )
                }),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }
    
    @classmethod
    def IS_CHANGED(cls, force_save=False, filename_prefix="", **kwargs):
        """
        Handle caching per Guide 7.1.
        Re-run if force_save=True or if filename_prefix is dynamic.
        """
        # Handle potential widget corruption (Guide 8.4)
        if isinstance(force_save, str):
            force_save = (force_save.lower() == "true")

        if force_save:
            return secrets.token_hex(16)  # User requested fresh save

        # Auto-detect dynamic filename patterns (Guide 7.1, Approach #4)
        # Use strftime patterns, as that's what the node uses
        dynamic_patterns = ['%Y', '%m', '%d', '%H', '%M', '%S', '%j', '%w', '%U', '%W', '%c', '%x', '%X']
        if any(pattern in filename_prefix for pattern in dynamic_patterns):
            return secrets.token_hex(16)  # Re-run for dynamic names
        
        # If filename is static and force_save is false, cache
        return "static"

    RETURN_TYPES = ("AUDIO", "IMAGE", "IMAGE",)
    RETURN_NAMES = ("AUDIO", "waveform_before", "waveform_after",)
    FUNCTION = "process_audio"
    CATEGORY = "MD_Nodes/Save" # Changed from /Save to /Audio as per Guide 2.
    OUTPUT_NODE = True

    def _prepare_metadata(self, **kwargs):
        """
        Prepares metadata dictionary with proper JSON serialization.
        
        Args:
            **kwargs: All node inputs.
            
        Returns:
            dict: A dictionary ready for metadata embedding, with complex
                  values (like workflows) serialized to JSON strings.
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
        if extra_pnginfo is not None:
            for key, value in extra_pnginfo.items():
                # JSON serialize if not already a string (per Guide 7.5)
                metadata[key] = json.dumps(value) if not isinstance(value, str) else value

        # Add custom notes as plain text
        if kwargs.get("custom_notes"):
            metadata['ComfyUI_Notes'] = kwargs.get("custom_notes")
            
        return metadata

    def _normalize_audio(self, audio_tensor, sample_rate, **kwargs):
        """
        Applies selected normalization method with appropriate fallbacks.
        Supports Peak, RMS, and LUFS normalization.
        
        Args:
            audio_tensor (torch.Tensor): Audio data [channels, samples].
            sample_rate (int): Sample rate.
            **kwargs: All node inputs, used to find normalization settings.
            
        Returns:
            torch.Tensor: Normalized (and optionally limited) audio tensor.
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

    def _plot_waveform_to_tensor(self, audio_data, sample_rate, title="Waveform", max_samples=150000):
        """
        Plots waveform (downsampled if needed) with peak/RMS and returns as tensor image.
        Takes audio_data as numpy array [samples, channels] or [samples].
        
        Args:
            audio_data (np.array): Numpy array of audio data.
            sample_rate (int): Sample rate.
            title (str): Title for the plot.
            max_samples (int): Max samples to plot before downsampling.
        
        Returns:
            torch.Tensor: A torch tensor [1, height, width, 3] in range [0.0, 1.0]
        """
        # Add basic check for valid audio data
        if audio_data is None or audio_data.size == 0:
            print(f"[Plotting] Skipping plot '{title}': Empty audio data")
            return torch.zeros((1, 64, 64, 3), dtype=torch.float32) # Return blank placeholder
        try:
            plt.style.use('dark_background') # Use dark theme
            fig, ax = plt.subplots(figsize=(10, 3)) # Use fig, ax

            # Use mono or first channel for plotting
            if audio_data.ndim == 2:
                plot_data = audio_data[:, 0]
            else:
                plot_data = audio_data

            # Check plot_data again after potential slicing
            if plot_data.size == 0:
                print(f"[Plotting] Skipping plot '{title}': Empty plot data after channel selection")
                plt.close(fig)
                return torch.zeros((1, 64, 64, 3), dtype=torch.float32)

            num_samples_original = len(plot_data)
            # Downsample if too long
            if num_samples_original > max_samples:
                ds_factor = num_samples_original // max_samples
                plot_data = plot_data[::ds_factor]
                time_axis = np.linspace(0, num_samples_original / sample_rate, len(plot_data))
            else:
                time_axis = np.linspace(0, num_samples_original / sample_rate, len(plot_data))

            # Final check on plot_data before plotting
            if plot_data.size == 0:
                print(f"[Plotting] Skipping plot '{title}': Empty plot data after downsampling")
                plt.close(fig)
                return torch.zeros((1, 64, 64, 3), dtype=torch.float32)

            # Plot waveform line
            ax.plot(time_axis, plot_data, color='#87CEEB', linewidth=0.5) # Sky blue 

            # --- Styling & Info ---
            # Calculate Peak and RMS
            peak_val = np.max(np.abs(plot_data)) if plot_data.size > 0 else 0.0
            rms = np.sqrt(np.mean(plot_data**2)) if plot_data.size > 0 else 0.0

            # Add peak markers (only if significant)
            if peak_val > 0.8:
                ax.axhline(y=peak_val, color='orangered', ls='--', lw=0.7, alpha=0.6, label=f'Peak: {peak_val:.3f}')
                ax.axhline(y=-peak_val, color='orangered', ls='--', lw=0.7, alpha=0.6)

            # Add RMS indicator
            ax.axhline(y=rms, color='mediumseagreen', ls=':', lw=0.7, alpha=0.6, label=f'RMS: {rms:.3f}')
            ax.axhline(y=-rms, color='mediumseagreen', ls=':', lw=0.7, alpha=0.6)

            # Title includes peak/RMS
            ax.set_title(f"{title} | Peak: {peak_val:.3f} | RMS: {rms:.3f}", fontsize=10)
            ax.set_xlabel("Time (s)", fontsize=8)
            ax.set_ylabel("Amplitude", fontsize=8)
            ax.set_xlim(0, num_samples_original / sample_rate)
            ax.set_ylim(-1.05, 1.05)
            ax.set_yticks([-1, -0.5, 0, 0.5, 1])
            ax.grid(True, ls=':', lw=0.5, alpha=0.3, color='gray')
            ax.legend(loc='upper right', fontsize=7, framealpha=0.5)

            # Style tweaks
            ax.tick_params(axis='both', which='major', labelsize=7)
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            plt.tight_layout()
            # --- End Styling & Info ---

            # --- Convert plot to tensor ---
            # *** FIXED: Reverted to original, robust fig.savefig method ***
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=96, facecolor=fig.get_facecolor())
            buf.seek(0)
            plt.close(fig) # CRITICAL: Close figure to prevent memory leak

            img = Image.open(buf).convert("RGB")
            img_np = np.array(img).astype(np.float32) / 255.0
            # Return in ComfyUI format: [batch, height, width, channels]
            img_tensor = torch.from_numpy(img_np).unsqueeze(0)
            return img_tensor
            # --- End Convert ---
            
        except Exception as e:
            # Log error
            print(f"[Plotting] Error during plot generation for '{title}': {e}")
            traceback.print_exc()
            # Return blank placeholder on error
            return torch.zeros((1, 64, 64, 3), dtype=torch.float32)
    # --- END PLOTTING METHOD ---

    def process_audio(self, **kwargs):
        """
        Main execution function. Handles the entire audio processing pipeline.
        
        Args:
            **kwargs: All node inputs, including hidden ones like 'prompt'.
        
        Returns:
            dict: A dictionary for ComfyUI containing the 'ui' (text)
                  and 'result' (a tuple matching RETURN_TYPES).
        """
        audio_input = kwargs.get("audio_input")
        placeholder_img = torch.zeros((1, 64, 64, 3), dtype=torch.float32)

        # Load and validate audio input
        try:
            waveform_original, samplerate = md_io.audio_from_comfy_3d(audio_input, try_gpu=True)
            print(f"[{self.__class__.__name__}] Loaded audio: {waveform_original.shape}, {samplerate}Hz")
        except Exception as e:
            # Graceful Failure (Guide 7.3)
            print(f"[{self.__class__.__name__}] ⚠️ Error loading audio: {e}")
            return {
                "ui": {"text": [f"Error loading audio: {e}"]}, 
                "result": ({"waveform": torch.empty(0), "sample_rate": 0}, placeholder_img, placeholder_img)
            }

        if waveform_original.numel() == 0:
            print(f"[{self.__class__.__name__}] ⚠️ Warning: Empty audio input")
            return {
                "ui": {"text": ["Warning: Empty audio input"]}, 
                "result": (audio_input, placeholder_img, placeholder_img)
            }
        
        # --- Generate Before Plot ---
        # Select first item in batch if needed
        plot_audio_before = waveform_original[0] if waveform_original.ndim == 3 else waveform_original
        # Convert from [channels, samples] tensor to [samples, channels] numpy
        audio_data_before_np = plot_audio_before.cpu().numpy().T
        waveform_before = self._plot_waveform_to_tensor(
            audio_data_before_np, samplerate, "Original Waveform"
        )

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
        
        # --- Generate After Plot ---
        # Convert from [channels, samples] tensor to [samples, channels] numpy
        audio_data_after_np = processed_audio.cpu().numpy().T
        waveform_after = self._plot_waveform_to_tensor(
            audio_data_after_np, samplerate, "Processed Waveform"
        )
        
        # === FILE SAVING ===
        
        ui_text = []
        if kwargs.get("save_to_disk"):
            try:
                # Process filename with strftime
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
            # Use the unique counter, as IS_CHANGED now controls *if* this runs
            file = f"{base_prefix}_{generate_unique_counter():09}.{save_format}"
            output_path = os.path.join(output_dir, file)
            quality = kwargs.get("mp3_quality") if save_format == "mp3" else kwargs.get("opus_quality")
            
            try:
                # Prepare metadata with proper JSON serialization
                metadata = self._prepare_metadata(**kwargs)
                metadata_to_embed = metadata.copy()
                sidecar_saved = False
                
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
                        
                        # Create full_prompt_api_dict for sidecar
                        full_prompt_api_dict = {
                            'prompt': kwargs.get('prompt'),
                            'extra_pnginfo': kwargs.get('extra_pnginfo')
                        }
                        
                        # Save full metadata to sidecar
                        sidecar_saved = save_metadata_sidecar(
                            output_path, 
                            full_prompt_api_dict, 
                            kwargs.get("custom_notes")
                        )
                        
                        if not sidecar_saved:
                            ui_text.append("⚠ Warning: Failed to save workflow sidecar")
                        
                        # Only embed notes in audio file (or nothing if no notes)
                        metadata_to_embed = {"ComfyUI_Notes": kwargs.get("custom_notes", "")} if kwargs.get("custom_notes") else {}
                    else:
                        print(f"[{self.__class__.__name__}] Embedding {metadata_size:.1f}KB metadata")
                
                # Save audio file
                success = _save_audio_with_av(
                    processed_audio, samplerate, output_path, 
                    save_format, metadata_to_embed, quality
                )
                
                if success:
                    rel_path = os.path.relpath(output_path, folder_paths.get_output_directory())
                    sidecar_msg = " + .json" if sidecar_saved else ""
                    ui_text.insert(0, f"✓ Saved: {rel_path}{sidecar_msg}")
                else:
                    ui_text.append("✗ ERROR: File save failed - check console for details")
                    # Clean up sidecar if audio save failed
                    if sidecar_saved:
                        try:
                            os.unlink(os.path.splitext(output_path)[0] + ".json")
                        except:
                            pass
                        
            except Exception as e:
                error_msg = f"✗ Save error: {e}"
                print(f"[{self.__class__.__name__}] {error_msg}")
                traceback.print_exc()
                ui_text.append(error_msg)
        else:
            ui_text.append("ℹ Not saved to disk (save_to_disk=False)")

        # === PREPARE OUTPUT ===
        
        # Ensure processed audio has correct dimensions for output
        processed_for_output = processed_audio.unsqueeze(0) if processed_audio.ndim == 2 else processed_audio
        final_audio_output = {"waveform": processed_for_output, "sample_rate": samplerate}
        
        # Return structure for ComfyUI: UI update + result tuple
        return {
            "ui": {"text": ui_text}, 
            "result": (final_audio_output, waveform_before, waveform_after)
        }

# =================================================================================
# == Node Registration                                                           ==
# =================================================================================

NODE_CLASS_MAPPINGS = {
    "AdvancedAudioPreviewAndSave": AdvancedAudioPreviewAndSave
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AdvancedAudioPreviewAndSave": "MD: Advanced Audio Preview & Save"
}