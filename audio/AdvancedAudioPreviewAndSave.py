# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# ████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████
# █▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀█
# █▓▒░ ADVANCED AUDIO PREVIEW & SAVE (AAPS) v0.3.12 - THE METADATA PRIVACY FILTER! ░▒▓█
# █░▒▓ Embracing harmony between Matplotlib, PIL, and ComfyUI for pixel-perfect waveforms! ▓▒░█
# █▓▒░ Crafted with 386-DX power and a prayer to the digital gods. ░▒▓█
# █▓▒░ Original Code by: MDMAchine (polished and perfected by Gemini) ░▒▓█
# █▓▒░ Special thanks to:                                                ░▒▓█
# █▓▒░  - https://github.com/c0ffymachyne/ComfyUI_SignalProcessing (for waveform display inspiration) ░▒▓█
# █▓▒░  - https://github.com/comfyanonymous/ComfyUI (for foundational audio saving logic) ░▒▓█
# █▓▒░ License: Public Domain, because sharing is caring in the BBS era. ░▒▓█
# █▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄█
# ████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████
#
# ░▒▓ Description:
#    This ComfyUI node is your ultimate audio companion. Think of it as a digital sound
#    engineer, ready to process, preview, and save your generated audio.
#
#    It takes raw audio data (like that sweet sonic output from your VAEAUDIO_Decode
#    or any other audio-producing node) through a standardized "AUDIO" input.
#
#    Then, it does some cool stuff:
#    1. It lets you **preview the audio** right on your ComfyUI canvas! Just connect
#       its 'AUDIO' output to ComfyUI's built-in 'Preview Audio' node. It's like
#       plugging your digital headphones in!
#    2. It can **save your masterpiece to disk** in glorious FLAC (lossless, for
#       audiophiles and workflow archivists), MP3 (universally compatible), or OPUS (efficient and high-quality for web/streaming).
#    3. When saving, it's super smart: it can automatically grab your
#       *entire ComfyUI workflow* (the blueprint of how you made the audio) and
#       embed it directly into the audio file's hidden metadata, if enabled. This means you can
#       drag that file back into ComfyUI later, and BAM! Your whole workflow
#       reappears like magic. It's like a time capsule for your creativity!
#    4. It also whips up a **snazzy waveform image** of your audio. This visual
#       representation of your sound waves is outputted as a standard ComfyUI
#       'IMAGE' tensor, which you can connect to any 'Preview Image' or 'Save Image'
#       node to marvel at your sonic art. We're now armed with *extra* debug to fix
#       any rogue pixels!
#
# ░▒▓ Features at a Glance:
#    - **Audio Output for Preview:** The 'AUDIO' output is ready to connect to ComfyUI's 'Preview Audio' node.
#    - **Waveform Image Output:** The 'WAVEFORM_IMAGE' output is ready for 'Preview Image' or 'Save Image' nodes.
#    - **Save to Disk:** Choose between high-fidelity FLAC, universally compatible MP3, or efficient OPUS. (Default is now OFF, for quick previews!)
#    - **Metadata Privacy Filter:** A new toggle ('Save Metadata') to control whether ANY metadata (including workflow, prompt, notes) is saved with the audio. Keep your sonic secrets safe!
#    - **Custom Notes:** Add your own personal digital post-it note to audio file metadata.
#    - **Normalize Audio:** Automatically scales your audio to prevent annoying clipping and ensure consistent loudness. Because nobody likes distorted sound!
#    - **Customizable Waveform Colors:** Pick your own line and background colors for that perfect visual vibe!
#    - **Customizable Waveform Dimensions:** Now you control the exact width and height of the waveform image!
#
# ░▒▓ Usage:
#    1. Connect the 'AUDIO' output from your audio source (like 'VAEAUDIO_Decode' or your 'EQ' node)
#       to this node's 'audio_input'.
#    2. Wire this node's 'AUDIO' output to a 'Preview Audio' node for listening.
#    3. Wire this node's 'WAVEFORM_IMAGE' output to a 'Preview Image' node to gaze upon your sound waves!
#    4. Set your desired filename, choose your save format, and tweak the visual settings.
#    5. Hit 'Queue Prompt' and witness the digital marvel!
#
# ░▒▓ Warning: May cause excessive enjoyment and workflow recall. Also, high-fives from your computer.
# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄█
# ████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████
#

import os
import torch
import numpy as np
import io as built_in_io # We rename 'io' to 'built_in_io' to avoid confusing it with our own 'md_io'.
import time
import json
import torchaudio # This library is super handy for working with audio tensors and resampling.
import av # The real hero for robust audio saving and embedding metadata. Say hello to FFmpeg's Python wrapper!

# For our snazzy waveform visualization - turning boring numbers into pretty pictures!
import matplotlib.pyplot as plt
# We set the 'Agg' backend. This means Matplotlib won't try to pop up a window,
# which is perfect for server-side operations like ComfyUI. No unexpected pop-ups here!
plt.switch_backend('Agg')
from PIL import Image # Pillow, our trusty image manipulation library, for turning matplotlib plots into usable images.

# ComfyUI's internal utilities for finding folders and communicating with the prompt server.
import folder_paths # Where do we put the files? This guy knows all the good digital hiding spots!
from server import PromptServer # Needed to get the current workflow JSON from ComfyUI's brain.
from comfy.cli_args import args # Checks if the user has forbidden us from embedding metadata (the 'disable_metadata' flag).

# Our custom audio input/output module, designed to standardize how audio flows through ComfyUI.
from ..core import io as md_io # We're using 'md_io' to clearly distinguish it from Python's built_in_io' module.

# --- Configuration: Setting up our Digital Workspace ---
# Define the special directory where all our awesome audio creations will be saved.
AUDIO_OUTPUT_DIR = os.path.join(folder_paths.get_output_directory(), "ComfyUI_AdvancedAudioOutputs")
# Make sure this directory exists. If it doesn't, we create it. No file left behind!
os.makedirs(AUDIO_OUTPUT_DIR, exist_ok=True)
print(f"[{__name__}] Audio output directory: {AUDIO_OUTPUT_DIR}. Go forth, little audio files, and multiply!")

# --- Helper Function: The Audio Saving Ninja! ---
# This function is our secret weapon for robust audio saving, directly inspired by ComfyUI's own internal magic!
def _save_audio_with_av(
    waveform_tensor, # The raw sonic data, fresh from the digital oven.
    sample_rate,     # The frequency of our digital beats (samples per second).
    full_output_folder, # The exact digital address where we'll lay this masterpiece to rest.
    file_basename,   # The cool codename for our audio file (e.g., "my_epic_song").
    counter,         # A unique number to make sure each file is uniquely awesome (and doesn't overwrite others).
    file_format,     # MP3, FLAC, or Opus? Choose your fighter!
    metadata,        # The hidden secrets we'll embed within the audio (like workflow JSON, notes). Shhh!
    quality_setting=None # For MP3/Opus, how good should it sound? Like a pristine CD or a nostalgic tape deck?
):
    results = [] # A list to hold the glorious paths of our saved files for ComfyUI to know about.

    # First, we need to make sure our audio data (the 'waveform_tensor') is in the perfect shape
    # for 'torchaudio.save'. It expects something like (channels, samples).
    if waveform_tensor.ndim == 1:
        waveform_tensor = waveform_tensor.unsqueeze(0) # If it's just a single line of sound (mono), we give it a 'channel' dimension.
    elif waveform_tensor.ndim == 3: # If it's a batch of sounds (batch, channels, samples)
         waveform_tensor = waveform_tensor[0] # We'll just take the first item in the batch. Sorry, other batches, first come, first serve!

    # Convert the audio data to float32. This is the standard data type for audio processing.
    # Because precision matters in the digital realm!
    if waveform_tensor.dtype != torch.float32:
        waveform_tensor = waveform_tensor.to(torch.float32)

    # We start by respecting the original sample rate. It's like listening to the source's rhythm.
    target_sample_rate = sample_rate

    # Handle Opus's picky sample rate requirements. Opus is a great codec, but it has its quirks!
    OPUS_RATES = [8000, 12000, 16000, 24000, 48000] # These are the only sample rates Opus likes.
    if file_format == "opus":
        if target_sample_rate > 48000:
            target_sample_rate = 48000 # If the original is too high, we cap it at Opus's max.
        elif target_sample_rate not in OPUS_RATES:
            # If the rate isn't one of Opus's favorites, we find the next closest supported one.
            for rate in sorted(OPUS_RATES):
                if rate > target_sample_rate:
                    target_sample_rate = rate
                    break
            if target_sample_rate not in OPUS_RATES: # Fallback if we still can't find a good match.
                target_sample_rate = 48000 # When all else fails, 48k is a safe bet for Opus.

        # Resample the audio if needed to match Opus's preferred rate.
        if target_sample_rate != sample_rate:
            waveform_tensor = torchaudio.functional.resample(waveform_tensor, sample_rate, target_sample_rate)
            print(f"[{__name__}] Resampled for Opus from {sample_rate} to {target_sample_rate} Hz. Keeping Opus happy!")

        sample_rate = target_sample_rate # Update the sample_rate variable for PyAV. PyAV likes to know the *real* rate.

    # Construct the final filename. It's gotta be unique, like a digital fingerprint!
    file = f"{file_basename}_{counter:05}.{file_format}"
    output_path = os.path.join(full_output_folder, file)

    try:
        # Create a temporary in-memory WAV buffer. Think of it as a temporary digital holding tank for the audio.
        wav_buffer = built_in_io.BytesIO()
        # Save our audio tensor into this temporary WAV buffer.
        torchaudio.save(wav_buffer, waveform_tensor, sample_rate, format="WAV")
        wav_buffer.seek(0) # Rewind the buffer to the beginning, ready for reading. Don't miss the start of the song!

        # Now, we use PyAV (our FFmpeg wrapper) to convert the WAV data to our desired format and add metadata.
        # This is where the magic happens, powered by FFmpeg's audio encoding prowess!
        input_container = av.open(wav_buffer) # Open the temporary WAV data as an input.

        # Create an output container in memory for our final audio file.
        output_buffer = built_in_io.BytesIO()
        output_container = av.open(output_buffer, mode='w', format=file_format)

        # Set the metadata on the container. This is how we embed secrets like the workflow JSON!
        # This check is now the final gate for *any* metadata being written to the file by PyAV.
        if metadata: # Only write if the metadata dictionary is not empty (controlled by save_metadata in process_audio)
            for key, value in metadata.items():
                output_container.metadata[key] = str(value) # Ensure all metadata values are strings for PyAV.

        # Set up the output audio stream. Each format has its own best practices and encoding settings.
        in_stream = input_container.streams.audio[0] # Grab the first audio stream from our input.

        out_stream = None # Our output stream, waiting for its destiny.
        if file_format == "opus":
            out_stream = output_container.add_stream("libopus", rate=sample_rate)
            # Opus quality settings. Choosing your fidelity!
            if quality_setting == "64k": out_stream.bit_rate = 64000
            elif quality_setting == "96k": out_stream.bit_rate = 96000
            elif quality_setting == "128k": out_stream.bit_rate = 128000
            elif quality_setting == "192k": out_stream.bit_rate = 192000
            elif quality_setting == "320k": out_stream.bit_rate = 320000
        elif file_format == "mp3":
            out_stream = output_container.add_stream("libmp3lame", rate=sample_rate)
            # MP3 quality settings. V0 is generally considered the best VBR quality for MP3!
            # It's like asking the chef for the "best tasting" dish, not just the "biggest" one.
            if quality_setting == "V0": out_stream.codec_context.qscale = 1 # V0 is highest quality variable bitrate. Max fidelity with smart file size!
            elif quality_setting == "128k": out_stream.bit_rate = 128000
            elif quality_setting == "320k": out_stream.bit_rate = 320000
        elif file_format == "flac":
            out_stream = output_container.add_stream("flac", rate=sample_rate) # FLAC, the lossless champion! Pure, uncompressed glory.
        else: # This should ideally not be reached if file_format is one of 'flac', 'mp3', 'opus'.
            # If we end up here, something funky happened! We'll just default to AAC for robustness.
            print(f"[{__name__}] Warning: Unhandled format '{file_format}' for PyAV stream setup. Using generic AAC default.")
            out_stream = output_container.add_stream('aac', rate=sample_rate)

        # Copy frames from the input stream to the output stream. It's like a digital assembly line!
        for frame in input_container.decode(audio=0):
            frame.pts = None # Let PyAV handle timestamps automatically. It's often smarter than us!
            for packet in out_stream.encode(frame):
                output_container.mux(packet) # Add the encoded audio packet to the output file.

        # Flush the encoder. Make sure all the digital bits are pushed through and nothing is left behind!
        for packet in out_stream.encode(None):
            output_container.mux(packet)

        # Close the containers. Clean up your workspace, digital citizen!
        output_container.close()
        input_container.close()

        # Write the entire in-memory output buffer to a real file on your hard drive. Finally, permanent storage!
        output_buffer.seek(0) # Rewind the output buffer to the beginning.
        with open(output_path, 'wb') as f: # Open the file in binary write mode.
            f.write(output_buffer.getbuffer()) # Write the audio data to the file.

        results.append({ # Tell ComfyUI about the newly saved file.
            "filename": file,
            "subfolder": os.path.basename(full_output_folder), # Just the last part of the subfolder for UI.
            "type": "output" # Mark it as an 'output' file type.
        })
        print(f"[{__name__}] Audio saved to: {output_path}. Mission accomplished! Your ears will thank you!")

    except Exception as e:
        print(f"[{__name__}] ERROR: Failed to save audio to disk using PyAV: {e}. Oh no! The digital quill broke!")
        # If something went wrong, we'll still report an error so you know.
        results.append({
            "filename": f"ERROR_SAVE_{int(time.time())}.{file_format}",
            "subfolder": os.path.basename(full_output_folder),
            "type": "output",
            "error": str(e)
        })
    return results

# --- Node Class Definition: Where the Magic Happens! ---
class AdvancedAudioPreviewAndSave:
    # CATEGORY: This tells ComfyUI where to find our node in the "Add Node" menu.
    # Look for us under 'audio/output'! We're making sounds and saving them!
    CATEGORY = "audio/output"

    # INPUT_TYPES: Defines all the inputs our node needs to do its awesome work.
    # Think of these as the ingredients for our digital recipe.
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { # These inputs are absolutely essential; the node won't run without them!
                # audio_input: The standardized audio dictionary coming from upstream nodes.
                # It's like the main course, bringing us the sound waves and their speed.
                "audio_input": ("AUDIO", {"description": "Audio data (waveform and sample rate) from previous nodes."}),
                # filename_prefix: A cool codename for your saved file (e.g., "my_epic_song").
                "filename_prefix": ("STRING", {"default": "ComfyUI_generated_audio"}),
                # save_format: Choose your audio destiny! MP3 for universal vibes, FLAC for pure lossless glory, or OPUS for efficient quality.
                "save_format": (["MP3", "FLAC", "OPUS"],),
            },
            "optional": { # These inputs are optional; they have default values if you don't connect anything.
                # save_to_disk: Should we immortalize this sound on your hard drive, or just preview it?
                # DEFAULT IS NOW FALSE! Because sometimes you just want to hear the vibes without cluttering your drive!
                "save_to_disk": ("BOOLEAN", {"default": False}),
                # NEW: save_metadata - Primary toggle to control whether ANY metadata is saved.
                "save_metadata": ("BOOLEAN", {"default": True, "tooltip": "If checked, all available metadata (workflow, prompt, notes) will be embedded. If unchecked, NO metadata will be saved."}),
                # custom_notes: A personal whisper to your audio file's metadata. Your secret message to the future!
                "custom_notes": ("STRING", {"default": ""}),
                # normalize_audio: Automatically scales the audio. Prevents harsh clipping (too loud!) or being too quiet.
                # Because nobody likes audio that screams or whispers too much!
                "normalize_audio": ("BOOLEAN", {"default": True}),
                # waveform_color: The color of the main line in your waveform visualization.
                # Use HTML color names (like "red", "blue") or #HEX codes (like "#00FFFF" for neon cyan).
                "waveform_color": ("STRING", {"default": "hotpink"}),
                # waveform_background_color: The canvas behind your sound waves. Make it dramatic!
                "waveform_background_color": ("STRING", {"default": "black"}),
                # mp3_quality: For those discerning ears who demand high-fidelity MP3s!
                # V0 is the top-tier Variable Bitrate (VBR) setting, offering amazing quality with smart file sizes.
                # 128k is good for general use, 320k is max Constant Bitrate (CBR). V0 is often the sweet spot!
                "mp3_quality": (["V0", "128k", "320k"], {"default": "V0"}),
                # opus_quality: New! Quality settings for Opus. Higher values mean better quality.
                "opus_quality": (["64k", "96k", "128k", "192k", "320k"], {"default": "128k"}),
                # waveform_width: The desired width of the waveform image in pixels.
                "waveform_width": ("INT", {"default": 800, "min": 100, "max": 2048, "step": 16}),
                # waveform_height: The desired height of the waveform image in pixels.
                "waveform_height": ("INT", {"default": 150, "min": 50, "max": 1024, "step": 16}),
            },
            # Hidden inputs for metadata embedding. ComfyUI whispers secrets into our node's ear!
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    # RETURN_TYPES: Defines what goodies our node spits out after it's done processing.
    # We output two two things: the 'AUDIO' itself (for playback) and the 'IMAGE' (for the waveform).
    RETURN_TYPES = ("AUDIO", "IMAGE",)
    # RETURN_NAMES: Gives friendly names to our outputs, so you know what's what on the node.
    RETURN_NAMES = ("AUDIO", "WAVEFORM_IMAGE",)

    # FUNCTION: This tells ComfyUI which Python method (function) to call when the node runs.
    # Our main event!
    FUNCTION = "process_audio"

    # OUTPUT_NODE: True means this node is typically an end-point in a workflow.
    # It's where your audio journey often concludes (or at least pauses for a listen!).
    OUTPUT_NODE = True

    # --- The Main Processing Function: Where the Magic ACTUALLY Happens! ---
    def process_audio(self, audio_input, filename_prefix, save_format, save_to_disk, save_metadata, custom_notes, normalize_audio, waveform_color, waveform_background_color, mp3_quality, opus_quality, waveform_width, waveform_height, prompt=None, extra_pnginfo=None):
        print(f"[{__name__}] Processing audio data stream... Stand by for sonic glory!")

        # Step 1: Get the audio data from our input. It's like uncorking a bottle of digital sound!
        try:
            # Our 'md_io' module helps us translate ComfyUI's audio input into a usable waveform tensor.
            # waveform_tensor_original will be in a format like (batch, channels, samples).
            waveform_tensor_original, samplerate = md_io.audio_from_comfy_3d(audio_input, try_gpu=True)
            print(f"[{__name__}] Received audio: Waveform shape: {waveform_tensor_original.shape}, Sample Rate: {samplerate} Hz, Device: {waveform_tensor_original.device}")
        except Exception as e:
            # Uh oh! If the audio input is funky, we catch the error and gracefully exit.
            print(f"[{__name__}] ERROR: Failed to decode audio input: {e}. Did the digital stream get clogged? Returning empty audio.")
            # Return empty audio and image if input is invalid to prevent crashes.
            return ({"waveform": torch.empty(0, 0, 0), "sample_rate": 0}, torch.empty(0, 0, 0, 3).permute(0,2,3,1)) # Ensure 4D (B,H,W,C) for ComfyUI.image

        # If the audio input is empty (e.g., if a previous node failed), we skip processing.
        if waveform_tensor_original.numel() == 0:
            print("WARNING: Input waveform is empty. Skipping audio saving and preview. Nothing to hear here but the sound of silence (literally!)")
            return ({"waveform": torch.empty(0, 0, 0), "sample_rate": 0}, torch.empty(0, 0, 0, 3).permute(0,2,3,1)) # Ensure 4D (B,H,W,C) for ComfyUI.image


        # Prepare audio data for processing (ensure it's on the CPU and in the correct 2D shape).
        # We need to make sure the tensor is shaped just right, like a puzzle piece.
        audio_data_for_processing = waveform_tensor_original.cpu() # Move data to CPU, most audio processing likes it there.
        if audio_data_for_processing.ndim == 3: # If it's (batch, channels, samples)
            audio_data_for_processing = audio_data_for_processing[0] # Take the first item in the batch.

        # Step 2: Normalization - Making sure our audio isn't too loud or too quiet!
        if normalize_audio:
            peak_val = torch.max(torch.abs(audio_data_for_processing)) # Find the loudest point in the audio.
            if peak_val > 1e-6: # Avoid dividing by zero or near-zero, which would make the universe explode!
                # Normalize to 0.99 to give a little headroom. We like our audio to breathe and not clip!
                audio_data_for_processing = audio_data_for_processing / peak_val * 0.99
                print(f"[{__name__}] Audio normalized to peak value of ~0.99. No clipping allowed on my watch!")
            else:
                print(f"[{__name__}] Audio has very low amplitude (or is silent); skipping normalization. Nothing to normalize here!")

        # Step 3: Dynamic Workflow JSON and Metadata Assembly - Building our secret messages!
        metadata = {} # This dictionary will hold all the extra info we want to embed.

        # Now, `save_metadata` is the master switch.
        if save_metadata and not args.disable_metadata:
            # The 'prompt' and 'extra_pnginfo' are generally for image metadata in ComfyUI,
            # but we can still pass them to PyAV if a format supports generic text tags.
            if prompt is not None:
                metadata["prompt"] = json.dumps(prompt)
            if extra_pnginfo is not None:
                # ComfyUI often puts the workflow into `extra_pnginfo` under the key 'workflow'
                # or other relevant keys. We will just pass along whatever it provides here.
                for key, value in extra_pnginfo.items():
                    # Check if the value is already a string to avoid double-dumping JSON strings
                    if isinstance(value, str):
                        try:
                            # Attempt to load as JSON to check if it's a JSON string.
                            # If it is, we'll try to re-dump it only if it's a simple type
                            # or just pass it through if it's already complex JSON.
                            # For simplicity, if it's already a string, we'll just assign it.
                            # PyAV will handle string values for metadata.
                            json.loads(value) # Just check if it's valid JSON
                            metadata[key] = value # Keep it as is if it's already a JSON string
                        except json.JSONDecodeError:
                            metadata[key] = value # Not JSON, just a regular string
                    else:
                        metadata[key] = json.dumps(value) # Dump non-string values to JSON string

                # Debugging workflow presence:
                if 'workflow' in extra_pnginfo:
                    print(f"[{__name__}] Found workflow in `extra_pnginfo`. It will be embedded if save_metadata is ON.")
                else:
                    print(f"[{__name__}] No workflow found directly in `extra_pnginfo` for embedding.")

            if custom_notes: # For those personal touches, like a digital post-it note on your audio file!
                if 'ComfyUI_Notes' in metadata: # If workflow JSON is already there, we append to existing notes.
                    metadata['ComfyUI_Notes'] += f"\n{custom_notes}"
                else: # Otherwise, we start a fresh note.
                    metadata['ComfyUI_Notes'] = custom_notes
                print(f"[{__name__}] Adding custom notes to metadata. Your digital thoughts are now immortalized!")
        else:
            # Provide specific feedback based on why metadata is skipped
            if not save_metadata:
                print(f"[{__name__}] User chose to save audio with NO metadata via node input. All metadata embedding skipped.")
            elif args.disable_metadata:
                print(f"[{__name__}] Metadata embedding globally disabled by ComfyUI --disable-metadata flag. Node 'save_metadata' setting is overridden (no metadata will be saved).")


        # Step 4: Save to Disk - Time to write the digital masterpiece to permanent storage!
        if save_to_disk:
            timestamp = int(time.time()) # A unique ID for this moment in time. Like a digital fingerprint!

            # Create a specific output directory, inside our main audio output folder.
            output_dir_local = os.path.join(AUDIO_OUTPUT_DIR, os.path.dirname(filename_prefix))
            os.makedirs(output_dir_local, exist_ok=True) # Make sure the folder exists. No digital homelessness here!

            # Use our powerful _save_audio_with_av helper. It handles all the dirty work of encoding!
            _save_audio_with_av(
                audio_data_for_processing,
                samplerate,
                output_dir_local,
                os.path.basename(filename_prefix), # Pass only the base name for the counter logic.
                timestamp % 100000, # Use part of the timestamp as a starting counter for uniqueness.
                save_format.lower(), # 'mp3', 'flac', or 'opus' (PyAV prefers lowercase).
                metadata, # Pass the potentially empty metadata dictionary
                quality_setting=mp3_quality if save_format == "MP3" else (opus_quality if save_format == "OPUS" else None) # Send quality for MP3s/Opus.
            )
            print(f"[{__name__}] Audio saved to disk (via PyAV). Your masterpiece is now safely stored!")

        # Step 5: Generate Waveform Image (now as an IMAGE output) - Visualizing the sound waves!
        waveform_image_tensor = torch.empty(1, waveform_height, waveform_width, 3) # Default empty tensor in case of errors. Ensure 4D (B,H,W,C) for ComfyUI.image
        fig = None # Initialize fig to None to handle potential errors before its creation

        try:
            # Convert our pristine audio tensor to NumPy for plotting. Matplotlib loves NumPy arrays!
            audio_data_np = audio_data_for_processing.cpu().numpy()

            # For plotting, if we have stereo audio, we typically plot one channel (e.g., the left one)
            # or an average. This makes the graph clear and easy to read.
            if audio_data_np.ndim == 2: # If it's (channels, samples)
                if audio_data_np.shape[0] > 1: # Stereo audio (e.g., (2, samples))
                    audio_data_np_mono = audio_data_np[0, :] # Plot the first channel. Left channel's time to shine!
                else: # Mono but 2D (1, samples) - just flatten it for plotting.
                    audio_data_np_mono = audio_data_np.flatten()
            elif audio_data_np.ndim == 1: # Already mono (samples,)
                audio_data_np_mono = audio_data_np
            else:
                raise ValueError(f"Unsupported audio data shape for plotting: {audio_data_np.shape}. My pixels are confused and refusing to draw!")

            # --- ADDED: Calculate time_axis for plotting ---
            if samplerate <= 0: # Handle cases where samplerate might be invalid (e.g., empty audio input)
                print(f"[{__name__}] WARNING: Samplerate is zero or negative ({samplerate}). Cannot generate waveform image. Returning blank image.")
                raise ValueError("Invalid samplerate for waveform plotting.") # Raise error to trigger except block and return empty image

            num_samples = audio_data_np_mono.shape[-1] # Get the number of samples
            time_axis = np.linspace(0, num_samples / samplerate, num_samples)
            # --- END ADDED ---

            # --- COLOR CHECK DEBUG ---
            print(f"[{__name__}] DEBUG: Waveform color received: '{waveform_color}'")
            print(f"[{__name__}] DEBUG: Waveform background color received: '{waveform_background_color}'")

            # Use provided width/height for figsize, divided by DPI to get inches.
            # This directly controls the pixel resolution of the saved image.
            dpi_val = 100 # Standard DPI for matplotlib
            fig, ax = plt.subplots(figsize=(waveform_width / dpi_val, waveform_height / dpi_val), dpi=dpi_val)

            # --- EXPLICIT AXIS CONTROL FROM PROVIDED CODE ---
            ax.axis("off") # Turn off all axes
            ax.set_position([0, 0, 1, 1]) # Set the position to fill the entire figure without margins

            # Plot the waveform itself. This is the sound's unique visual signature!
            # Ensure color is passed directly.
            ax.plot(time_axis, audio_data_np_mono, color=waveform_color, linewidth=0.5)


            # Customize plot appearance for that cool retro BBS aesthetic or whatever wild color scheme you choose!
            ax.set_facecolor(waveform_background_color) # Background color for the plot area.
            fig.patch.set_facecolor(waveform_background_color) # Match the figure background for seamless art.
            ax.set_ylim([-1, 1]) # Set the Y-axis limits. Assuming audio is normalized between -1 and 1. Keep it in bounds!

            # Adjust layout to prevent overlapping titles/labels and save figure
            plt.tight_layout(pad=0) # Pad=0 is crucial for tight bounding box without extra whitespace.

            # Convert our beautiful matplotlib figure into a PIL Image, then into a PyTorch tensor.
            # It's like taking a high-res digital photo and preparing it for display.
            buffer = built_in_io.BytesIO() # Create a temporary in-memory buffer to hold the image data.
            # Save the plot to the buffer as a PNG. Use bbox_inches='tight' from the reference code.
            plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0)
            buffer.seek(0) # Rewind the buffer to the beginning.
            pil_image = Image.open(buffer).convert("RGB") # Open the PNG data as a PIL Image and ensure it's in RGB format.

            # Resize to the requested output dimensions. PIL's LANCZOS filter is like
            # a master upscaler/downscaler for pixel perfection!
            pil_image = pil_image.resize((waveform_width, waveform_height), Image.LANCZOS)

            # Convert our PIL Image into a PyTorch tensor.
            # ComfyUI expects images in (Batch, Height, Width, Channels) format,
            # with pixel values normalized to [0,1].
            waveform_image_tensor = torch.from_numpy(np.array(pil_image)).float() / 255.0
            waveform_image_tensor = waveform_image_tensor.unsqueeze(0) # Add the batch dimension (we only have one image)

        except Exception as e:
            print(f"[{__name__}] ERROR: Failed to generate waveform image: {e}. The digital artist had a meltdown! Returning blank image.")
            # If image generation fails, return a blank image to avoid crashing the workflow.
            waveform_image_tensor = torch.zeros((1, waveform_height, waveform_width, 3), dtype=torch.float32)
        finally:
            # ALWAYS close the figure to prevent memory leaks! This is super important!
            if fig is not None:
                plt.close(fig)

        # Step 6: Return the goods! The processed audio for preview and the waveform image.
        # ComfyUI expects node outputs as a tuple of tensors.
        return ({"waveform": waveform_tensor_original, "sample_rate": samplerate}, waveform_image_tensor,)

# NODE_CLASS_MAPPINGS: This is how ComfyUI knows about our amazing node!
# It maps the internal class name to a friendly string.
NODE_CLASS_MAPPINGS = {
    "AdvancedAudioPreviewAndSave": AdvancedAudioPreviewAndSave,
}

# NODE_DISPLAY_NAME_MAPPINGS: This is the name you'll see in the "Add Node" menu.
NODE_DISPLAY_NAME_MAPPINGS = {
    "AdvancedAudioPreviewAndSave": "Advanced Audio Preview & Save"
}