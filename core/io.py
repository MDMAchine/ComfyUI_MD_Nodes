#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: C0ffymachyne
License: GPLv3
Version: 1.1.1 (Fixed SyntaxError by removing placeholders)
https://github.com/c0ffymachyne/ComfyUI_SignalProcessing

Description:
    Refactored I/O utilities for handling audio files and ComfyUI AUDIO format.
"""

import torch
import torchaudio
from pathlib import Path
from typing import Dict, Tuple, Any, Union, Optional, List
import logging
import numpy as np # Keep numpy import for potential future use or edge cases

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Global debug flag (can be controlled externally if needed)
debug_print = False

## Refactored Audio Conversion Functions

def audio_to_comfy(
    waveform: torch.Tensor,
    sample_rate: int,
    *extra_values: Any,
    cpu: bool = True,
    clip: bool = False,
    target_dtype: torch.dtype = torch.float32 # Changed default to float32
) -> Tuple[Dict[str, Union[torch.Tensor, int]], ...]:
    """
    Converts a waveform tensor to the ComfyUI AUDIO format dictionary,
    ensuring correct shape [batch, channels, samples] and data type.
    Optionally includes extra values in the returned tuple.

    Args:
        waveform: Input audio tensor (can be 2D [channels, samples] or 3D [batch, channels, samples]).
        sample_rate: Audio sample rate.
        *extra_values: Any additional values to include in the output tuple after the audio dict.
        cpu: If True, moves the waveform to the CPU.
        clip: If True, clamps the waveform values to [-1.0, 1.0].
        target_dtype: The target torch dtype for the waveform (default: float32).

    Returns:
        A tuple containing the ComfyUI AUDIO dictionary and any extra values provided.
        Example: ({"waveform": tensor, "sample_rate": sr}, val1, val2)

    Raises:
        RuntimeError: If the input waveform cannot be reshaped to 3D.
        TypeError: If input waveform is not a tensor.
    """
    if not isinstance(waveform, torch.Tensor):
         raise TypeError(f"Input waveform must be a torch.Tensor, got {type(waveform)}")

    # Ensure 3D: [batch, channels, samples]
    if waveform.ndim == 2:
        waveform = waveform.unsqueeze(0) # Add batch dimension
    elif waveform.ndim != 3:
        raise RuntimeError(
            f"Waveform has incorrect dimensions ({waveform.ndim}D with shape {waveform.shape}). "
            "Expected 2D [channels, samples] or 3D [batch, channels, samples]."
        )

    # Move to CPU if requested
    if cpu and waveform.device.type != "cpu":
        waveform = waveform.cpu()

    # Convert dtype if necessary
    if waveform.dtype != target_dtype:
        waveform = waveform.to(dtype=target_dtype)

    # Clamp values if requested
    if clip:
        waveform = torch.clamp(waveform, min=-1.0, max=1.0) # Use torch.clamp

    audio_dict = {"waveform": waveform, "sample_rate": sample_rate}
    return_value = (audio_dict,) + extra_values # Combine audio dict and extra values

    if debug_print:
        logger.debug(
            f"audio_to_comfy -> shape: {waveform.shape}@{waveform.device}, "
            f"dtype: {waveform.dtype}, sr: {sample_rate} Hz. "
            f"Output tuple length: {len(return_value)}"
        )

    return return_value

# --- Keeping the specific named versions for potential backward compatibility ---
# --- or if users prefer the explicit names, but they now wrap the generic one ---

def audio_to_comfy_3d(waveform: torch.Tensor, sample_rate: int, cpu: bool = True, clip: bool = False) -> Tuple[Dict[str, Union[torch.Tensor, int]]]:
    """Wrapper for audio_to_comfy with no extra values."""
    # Type annotation for return value corrected
    result: Tuple[Dict[str, Union[torch.Tensor, int]]] = audio_to_comfy(waveform, sample_rate, cpu=cpu, clip=clip)
    return result


def audio_to_comfy_3dp1(waveform: torch.Tensor, sample_rate: int, value0: Any, cpu: bool = True, clip: bool = False) -> Tuple[Dict[str, Union[torch.Tensor, int]], Any]:
    """Wrapper for audio_to_comfy with one extra value."""
    # Type annotation for return value corrected
    result: Tuple[Dict[str, Union[torch.Tensor, int]], Any] = audio_to_comfy(waveform, sample_rate, value0, cpu=cpu, clip=clip)
    return result


def audio_to_comfy_3dp2(waveform: torch.Tensor, sample_rate: int, value0: Any, value1: Any, cpu: bool = True, clip: bool = False) -> Tuple[Dict[str, Union[torch.Tensor, int]], Any, Any]:
    """Wrapper for audio_to_comfy with two extra values."""
    # Type annotation for return value corrected
    result: Tuple[Dict[str, Union[torch.Tensor, int]], Any, Any] = audio_to_comfy(waveform, sample_rate, value0, value1, cpu=cpu, clip=clip)
    return result


# ---

def audio_from_comfy(
    audio: Dict[str, Union[torch.Tensor, int]],
    keep_batch_dim: bool = False, # Flag to control output dimensions
    convert_to_stereo: bool = True, # Renamed 'repeat' for clarity
    try_gpu: bool = True,
    target_dtype: torch.dtype = torch.float32 # Changed default to float32
) -> Tuple[torch.Tensor, int]:
    """
    Extracts waveform tensor and sample rate from ComfyUI AUDIO format.
    Optionally converts to stereo, moves to GPU, and adjusts dimensions.

    Args:
        audio: ComfyUI AUDIO dictionary ({"waveform": tensor, "sample_rate": sr}).
        keep_batch_dim: If False (default), squeezes the batch dim -> [channels, samples].
                        If True, keeps the batch dim -> [batch, channels, samples].
        convert_to_stereo: If True and input is mono, duplicates the channel to create stereo.
        try_gpu: If True, attempts to move the waveform to the GPU if available.
        target_dtype: The target torch dtype for the waveform (default: float32).

    Returns:
        Tuple: (waveform_tensor, sample_rate)

    Raises:
        ValueError: If input dict is invalid or waveform shape is incorrect.
        TypeError: If waveform is not a tensor or sample_rate is not int.
    """
    if not isinstance(audio, dict):
        raise TypeError("Input 'audio' must be a dictionary.")
    if "waveform" not in audio or "sample_rate" not in audio:
        raise ValueError("Input audio dict must contain 'waveform' and 'sample_rate' keys.")

    waveform: Optional[torch.Tensor] = audio.get("waveform")
    sample_rate: Optional[int] = audio.get("sample_rate")

    if not isinstance(waveform, torch.Tensor):
        raise TypeError(f"Audio 'waveform' must be a torch.Tensor, got {type(waveform)}")
    if not isinstance(sample_rate, int):
         try: # Attempt conversion if sample_rate is tensor or other type
             sample_rate = int(sample_rate)
         except (ValueError, TypeError):
             raise TypeError(f"Audio 'sample_rate' must be an integer, got {type(sample_rate)}")

    if debug_print:
        logger.debug(
            f"audio_from_comfy <- shape: {waveform.shape}@{waveform.device}, "
            f"dtype: {waveform.dtype}, sr: {sample_rate} Hz."
        )

    # Validate input shape (must be 3D initially)
    if waveform.ndim != 3:
        raise ValueError(
            f"Input waveform must be a 3D tensor [batch, channels, samples], "
            f"got {waveform.ndim}D with shape {waveform.shape}@{waveform.device}"
        )

    # Convert dtype if necessary (before potential GPU move)
    if waveform.dtype != target_dtype:
        waveform = waveform.to(dtype=target_dtype)

    # Squeeze batch dimension if requested (do this *before* stereo conversion)
    processed_waveform = waveform if keep_batch_dim else waveform.squeeze(0)
    # Shape is now either [channels, samples] or [batch, channels, samples]

    # Handle stereo conversion more reliably
    if convert_to_stereo:
        channels_dim = 1 if keep_batch_dim else 0 # Dimension index for channels
        if processed_waveform.shape[channels_dim] == 1:
            logger.debug("audio_from_comfy: Converting mono to stereo by repeating channel.")
            # Repeat along the channel dimension
            processed_waveform = processed_waveform.repeat_interleave(2, dim=channels_dim)

    # Move to GPU if requested
    if try_gpu:
        target_device_str = "cuda" if torch.cuda.is_available() else "cpu"
        target_device = torch.device(target_device_str)
        if processed_waveform.device != target_device:
            processed_waveform = processed_waveform.to(target_device)

    # Ensure contiguous memory layout (can sometimes prevent errors downstream)
    processed_waveform = processed_waveform.contiguous()

    if debug_print:
        logger.debug(
            f"audio_from_comfy -> shape: {processed_waveform.shape}@{processed_waveform.device}, "
            f"dtype: {processed_waveform.dtype}, sr: {sample_rate} Hz."
        )

    return processed_waveform, sample_rate

# --- Keeping the old names wrapping the new function for compatibility ---

def audio_from_comfy_2d(audio: Dict[str, Union[torch.Tensor, int]], repeat: bool = True, try_gpu: bool = True) -> Tuple[torch.Tensor, int]:
    """Compatibility wrapper for audio_from_comfy, outputting 2D tensor [channels, samples]."""
    return audio_from_comfy(audio, keep_batch_dim=False, convert_to_stereo=repeat, try_gpu=try_gpu)

def audio_from_comfy_3d(audio: Dict[str, Union[torch.Tensor, int]], repeat: bool = True, try_gpu: bool = True) -> Tuple[torch.Tensor, int]:
    """Compatibility wrapper for audio_from_comfy, outputting 3D tensor [batch, channels, samples]."""
    return audio_from_comfy(audio, keep_batch_dim=True, convert_to_stereo=repeat, try_gpu=try_gpu)

# --- Disk I/O ---

def audio_from_comfy_3d_to_disk(
    audio: Dict[str, Union[torch.Tensor, int]],
    filepath: str,
    # target_dtype: torch.dtype = torch.float32 # Use float32 for saving standard formats
) -> None:
    """
    Saves the waveform from a ComfyUI AUDIO dict to disk using torchaudio.
    Saves as 16-bit WAV by default.

    Args:
        audio: ComfyUI AUDIO dictionary.
        filepath: Path to save the audio file.
    """
    try:
        # Use the standard extraction function, ensuring 2D tensor on CPU, float32
        waveform_2d_cpu, sample_rate = audio_from_comfy(
            audio,
            keep_batch_dim=False,    # Need [channels, samples] for torchaudio.save
            convert_to_stereo=False, # Save original channels
            try_gpu=False,           # Ensure CPU tensor
            target_dtype=torch.float32 # Use float32 for saving standard formats
        )

        logger.info(f"Saving audio to '{filepath}' (Shape: {waveform_2d_cpu.shape}, SR: {sample_rate} Hz)")
        # Ensure parent directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        # Save using torchaudio (e.g., as 16-bit WAV)
        torchaudio.save(filepath, waveform_2d_cpu, sample_rate, encoding="PCM_S", bits_per_sample=16)
        logger.info(f"Successfully saved audio to {filepath}")

    except Exception as e:
        logger.error(f"Failed to save audio to '{filepath}': {e}", exc_info=True)
        # Optionally re-raise the exception if needed by the caller
        raise

def from_disk_as_dict_3d(
    audio_file: str,
    gain: float = 1.0,
    convert_to_stereo: bool = True, # Renamed 'repeat'
    start_time: float = 0.0,
    end_time: float = 0.0,
    target_dtype: torch.dtype = torch.float32 # Added dtype option
) -> Tuple[Dict[str, Union[torch.Tensor, int]]]:
    """
    Loads audio from disk and returns it in ComfyUI AUDIO dict format [batch, channels, samples].

    Args:
        audio_file: Path to the audio file.
        gain: Gain factor to apply.
        convert_to_stereo: If True and input is mono, duplicate channel.
        start_time: Start time in seconds for loading a segment.
        end_time: End time in seconds for loading a segment (requires start_time >= 0 and end_time > start_time).
        target_dtype: Target torch dtype (default: float32).

    Returns:
        Tuple containing the ComfyUI AUDIO dictionary.
    """
    if not Path(audio_file).is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_file}")

    try:
        info = torchaudio.info(audio_file)
        sample_rate = info.sample_rate
        num_frames_total = info.num_frames

        frame_offset = 0
        num_frames = -1 # Load all by default

        # Validate and calculate segment frames
        start_time = max(0.0, start_time) # Ensure start_time is not negative
        end_time = max(0.0, end_time)     # Ensure end_time is not negative

        if end_time > 0 and end_time > start_time:
             frame_offset = int(start_time * sample_rate)
             num_frames_to_read = int((end_time - start_time) * sample_rate)
             # Ensure we don't read past the end of the file or request zero frames
             if frame_offset < num_frames_total and num_frames_to_read > 0:
                 num_frames = min(num_frames_to_read, num_frames_total - frame_offset)
                 logger.debug(f"Loading segment: offset={frame_offset}, frames={num_frames}")
             else:
                  logger.warning(f"Invalid segment: start_time={start_time}, end_time={end_time}. Loading full file.")
                  frame_offset = 0
                  num_frames = -1
        elif start_time > 0: # Only start time given, load from start to end
             frame_offset = int(start_time * sample_rate)
             if frame_offset < num_frames_total:
                  num_frames = -1 # Read to end
                  logger.debug(f"Loading from start time: offset={frame_offset}")
             else:
                  logger.warning(f"Start time ({start_time}s) is beyond or at audio duration. Loading full file.")
                  frame_offset = 0
                  num_frames = -1
        # If only end_time > 0 or both 0, load full file (frame_offset=0, num_frames=-1)


        waveform, loaded_sr = torchaudio.load(
            audio_file, frame_offset=frame_offset, num_frames=num_frames
        )

        if loaded_sr != sample_rate:
             logger.warning(f"Sample rate mismatch during load? Info: {sample_rate}, Load: {loaded_sr}. Using loaded rate: {loaded_sr}.")
             sample_rate = loaded_sr # Trust the rate returned by load

        # Apply gain
        if gain != 1.0:
            waveform = waveform * gain

        # Convert dtype (torchaudio usually loads as float32, but good to ensure)
        if waveform.dtype != target_dtype:
            waveform = waveform.to(dtype=target_dtype)

        # Ensure stereo if requested (waveform is [channels, samples] at this point)
        if convert_to_stereo and waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)

        if debug_print:
            logger.debug(
                f"from_disk_as_dict_3d loaded '{Path(audio_file).name}' -> "
                f"shape: {waveform.shape}@{waveform.device}, dtype: {waveform.dtype}, sr: {sample_rate} Hz."
            )

        # Use the standardized conversion function to add batch dim and return dict
        # Load directly to CPU
        return audio_to_comfy(waveform, sample_rate, cpu=True, clip=False, target_dtype=target_dtype)

    except Exception as e:
        logger.error(f"Failed to load audio file '{audio_file}': {e}", exc_info=True)
        raise # Re-raise the exception after logging

# --- Raw loading functions removed as they are now redundant ---
# def from_disk_as_raw_3d(...)
# def from_disk_as_raw_2d(...)