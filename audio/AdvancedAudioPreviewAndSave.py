# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# ████ ADVANCED AUDIO PREVIEW & SAVE (AAPS) v0.4.1 – Optimized for Ace-Step Audio/Video ████▓▒░
# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀

# ░▒▓ ORIGIN & DEV:
#   • Cast into the void by: MDMAchine (waveform conjurer)
#   • Original AAPS code & concept by: MDMAchine & Gemini (pixel perfectionists)
#   • Inspired by: c0ffymachyne’s ComfyUI_SignalProcessing & ComfyAnonymous audio saving logic
#   • License: Public Domain — Sharing the sonic love of the BBS era

# ░▒▓ DESCRIPTION:
#   The ultimate audio companion node for ComfyUI with Ace-Step precision.
#   Processes, previews, and saves audio from your generation workflows.
#   Outputs audio streams ready for immediate playback and
#   generates crisp waveform images for visualization or archival.
#   Includes smart metadata embedding that keeps your workflow blueprints
#   locked inside your audio files — or filtered out for privacy.

# ░▒▓ FEATURES:
#   ✓ Audio output compatible with ComfyUI Preview Audio node
#   ✓ Waveform image output for real-time visual feedback
#   ✓ Save audio in FLAC (lossless), MP3 (universal), or OPUS (efficient)
#   ✓ Metadata Privacy Filter toggle — keep your secrets safe or share your process
#   ✓ Add custom notes embedded into audio metadata
#   ✓ Automatic audio normalization (Peak, RMS) to prevent clipping
#   ✓ Optional soft limiter to prevent harsh clipping after normalization (requires pedalboard)
#   ✓ Fade In / Fade Out controls for smooth starts and stops
#   ✓ Stereo-to-Mono conversion option
#   ✓ Dynamic filename templating (e.g., 'my_audio_%Y-%m-%d')
#   ✓ Fully customizable waveform colors, size, and grid display

# ░▒▓ CHANGELOG:
#   - v0.1-v0.3.19 (See previous versions for details on initial features, bug fixes, and LUFS removal)
#   - v0.4.0 (Feature Pack & Enhancements):
#       • Added Fade In / Fade Out, Mono Conversion, Limiter, Dynamic Filenames, and UI path display.
#   - v0.4.1 (Bugfix):
#       • Fixed `NameError: name 'sample_rate' is not defined` by correcting a typo (`sample_rate` -> `samplerate`) in the limiter processing call.

# ░▒▓ CONFIGURATION:
#   → Primary Use: Audio preview and file export within ComfyUI workflows
#   → Secondary Use: Visual waveform generation for audio debugging or art
#   → Edge Use: Metadata steganography and workflow archival in sound files

# ░▒▓ WARNING:
#   This node may trigger:
#   ▓▒░ Uncontrollable groove sessions
#   ▓▒░ Over-sharing or over-hiding your creative workflow
#   ▓▒░ Obsessive waveform customization urges

# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄


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

# --- NEW IMPORTS FOR ADVANCED NORMALIZATION AND EFFECTS ---
try:
    import pedalboard
    _pedalboard_available = True
    print(f"[{__name__}] pedalboard imported successfully. Advanced gain and effects available!")
except ImportError:
    _pedalboard_available = False
    print(f"[{__name__}] Warning: pedalboard not found. Final gain application will fall back to direct tensor manipulation.")
# --- END NEW IMPORTS ---

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

    # Move to CPU for saving libraries
    waveform_tensor = waveform_tensor.cpu()

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
            "type": "output",
            "filepath": output_path # Store the full path for UI display
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
    CATEGORY = "MD_Nodes/Save"

    # INPUT_TYPES: Defines all the inputs our node needs to do its awesome work.
    # Think of these as the ingredients for our digital recipe.
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { # These inputs are absolutely essential; the node won't run without them!
                "audio_input": ("AUDIO", {"description": "Audio data (waveform and sample rate) from previous nodes."}),
                "filename_prefix": ("STRING", {"default": "ComfyUI_audio_%Y-%m-%d"}),
                "save_format": (["MP3", "FLAC", "OPUS"],),
            },
            "optional": { # These inputs are optional; they have default values if you don't connect anything.
                "save_to_disk": ("BOOLEAN", {"default": False}),
                "save_metadata": ("BOOLEAN", {"default": True, "tooltip": "If checked, all available metadata (workflow, prompt, notes) will be embedded in the audio file. If unchecked, NO metadata will be saved."}),
                "custom_notes": ("STRING", {"default": "", "multiline": True, "tooltip": "Add custom text notes to be embedded in the audio file's metadata. Visible in players that support metadata."}),
                "channel_mode": (["Keep Original", "Convert to Mono"], {"default": "Keep Original", "tooltip": "Choose whether to keep the original audio channels or downmix to mono."}),
                "fade_in_ms": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 10, "tooltip": "Duration of the fade-in in milliseconds."}),
                "fade_out_ms": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 10, "tooltip": "Duration of the fade-out in milliseconds."}),
                "normalize_method": (["Off", "Peak", "RMS"], {"default": "Peak", "tooltip": "Select the audio normalization method.\n- 'Off': No normalization applied.\n- 'Peak': Normalizes the audio so its loudest sample reaches a target peak amplitude (usually just below 0 dBFS to prevent clipping).\n- 'RMS': Normalizes to an average loudness level based on the Root Mean Square of the waveform."}),
                "target_rms_db": ("INT", {"default": -16, "min": -60, "max": 0, "step": 1, "tooltip": "Target RMS level in dBFS for RMS normalization."}),
                "use_limiter": ("BOOLEAN", {"default": True, "tooltip": "If checked, applies a soft limiter after normalization to prevent harsh clipping. Recommended for RMS normalization. Requires 'pedalboard'."}),
                "waveform_color": ("STRING", {"default": "hotpink", "tooltip": "Color of the waveform line."}),
                "waveform_background_color": ("STRING", {"default": "black", "tooltip": "Background color of the waveform image."}),
                "mp3_quality": (["V0", "128k", "320k"], {"default": "V0", "tooltip": "Select the quality for MP3 output."}),
                "opus_quality": (["64k", "96k", "128k", "192k", "320k"], {"default": "128k", "tooltip": "Select the quality for OPUS output."}),
                "waveform_width": ("INT", {"default": 800, "min": 100, "max": 2048, "step": 16, "tooltip": "Width of the generated waveform image in pixels."}),
                "waveform_height": ("INT", {"default": 150, "min": 50, "max": 1024, "step": 16, "tooltip": "Height of the generated waveform image in pixels."}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    # RETURN_TYPES: Defines what goodies our node spits out after it's done processing.
    RETURN_TYPES = ("AUDIO", "IMAGE",)
    # RETURN_NAMES: Gives friendly names to our outputs, so you know what's what on the node.
    RETURN_NAMES = ("AUDIO", "WAVEFORM_IMAGE",)
    FUNCTION = "process_audio"
    OUTPUT_NODE = True

    def process_audio(self, audio_input, filename_prefix, save_format, save_to_disk, save_metadata, custom_notes, channel_mode, fade_in_ms, fade_out_ms, normalize_method, target_rms_db, use_limiter, waveform_color, waveform_background_color, mp3_quality, opus_quality, waveform_width, waveform_height, prompt=None, extra_pnginfo=None):
        print(f"[{__name__}] Processing audio data stream... Stand by for sonic glory!")

        # Initialize UI output text
        ui_text = []

        try:
            waveform_tensor_original, samplerate = md_io.audio_from_comfy_3d(audio_input, try_gpu=True)
            print(f"[{__name__}] Received audio: Shape: {waveform_tensor_original.shape}, SR: {samplerate} Hz, Device: {waveform_tensor_original.device}")
        except Exception as e:
            print(f"[{__name__}] ERROR: Failed to decode audio input: {e}. Returning empty audio.")
            return {"ui": {"text": ["Error: Invalid audio input."]}, "result": ({"waveform": torch.empty(0,0,0), "sample_rate": 0}, torch.empty(1, 1, 1, 3))}

        if waveform_tensor_original.numel() == 0:
            print("WARNING: Input waveform is empty. Skipping processing.")
            return {"ui": {"text": ["Warning: Empty audio input."]}, "result": ({"waveform": torch.empty(0,0,0), "sample_rate": 0}, torch.empty(1, 1, 1, 3))}

        # Clone the tensor for processing to keep the original for preview
        processed_audio = waveform_tensor_original.clone()
        if processed_audio.ndim == 3:
             processed_audio = processed_audio[0]

        # --- Step 1: Channel Conversion ---
        if channel_mode == "Convert to Mono" and processed_audio.shape[0] > 1:
            processed_audio = torch.mean(processed_audio, dim=0, keepdim=True)
            print(f"[{__name__}] Converted audio to mono.")

        # --- Step 2: Fades ---
        num_channels, num_samples = processed_audio.shape
        if fade_in_ms > 0:
            fade_in_samples = int(samplerate * (fade_in_ms / 1000.0))
            if fade_in_samples > 0 and fade_in_samples <= num_samples:
                fade_in_ramp = torch.linspace(0.0, 1.0, fade_in_samples, device=processed_audio.device)
                processed_audio[:, :fade_in_samples] *= fade_in_ramp
                print(f"[{__name__}] Applied {fade_in_ms}ms fade-in.")

        if fade_out_ms > 0:
            fade_out_samples = int(samplerate * (fade_out_ms / 1000.0))
            if fade_out_samples > 0 and fade_out_samples <= num_samples:
                fade_out_ramp = torch.linspace(1.0, 0.0, fade_out_samples, device=processed_audio.device)
                processed_audio[:, -fade_out_samples:] *= fade_out_ramp
                print(f"[{__name__}] Applied {fade_out_ms}ms fade-out.")

        # --- Step 3: Normalization ---
        if normalize_method == "Peak":
            peak_val = torch.max(torch.abs(processed_audio))
            if peak_val > 1e-6:
                processed_audio = processed_audio / peak_val * 0.99
                print(f"[{__name__}] Audio normalized to peak value of ~0.99.")
            else:
                print(f"[{__name__}] Audio has very low amplitude; skipping peak normalization.")
        elif normalize_method == "RMS":
            current_rms_linear = torch.sqrt(torch.mean(processed_audio**2))
            target_rms_linear = 10**(target_rms_db / 20.0)
            if current_rms_linear > 1e-9:
                scale_factor = target_rms_linear / current_rms_linear
                processed_audio = processed_audio * scale_factor
                print(f"[{__name__}] Audio normalized to target RMS: {target_rms_db} dB.")
                # We don't clip here, we let the limiter handle it
            else:
                print(f"[{__name__}] Audio has very low amplitude; skipping RMS normalization.")

        # --- Step 4: Limiter ---
        if use_limiter and normalize_method != "Off":
            if _pedalboard_available:
                # pedalboard works on numpy arrays with shape (channels, samples)
                audio_np = processed_audio.cpu().numpy()
                board = pedalboard.Pedalboard([pedalboard.Limiter(threshold_db=-1.0, release_ms=50)])
                processed_np = board(audio_np, samplerate) # <-- BUG FIX HERE
                processed_audio = torch.from_numpy(processed_np).to(processed_audio.device)
                print(f"[{__name__}] Applied soft limiter to prevent clipping.")
            else:
                print(f"[{__name__}] Warning: 'use_limiter' is enabled but 'pedalboard' is not installed. Clipping may occur. Falling back to hard clipping.")
                processed_audio = torch.clamp(processed_audio, -1.0, 1.0)
        elif normalize_method == "RMS": # If limiter is off but RMS is on, we still need to prevent clipping
            processed_audio = torch.clamp(processed_audio, -1.0, 1.0)


        # --- Step 5: Metadata Assembly ---
        metadata = {}
        if save_metadata and not args.disable_metadata:
            if prompt is not None: metadata["prompt"] = json.dumps(prompt)
            if extra_pnginfo is not None:
                for key, value in extra_pnginfo.items():
                    metadata[key] = json.dumps(value) if not isinstance(value, str) else value
            if custom_notes:
                metadata['ComfyUI_Notes'] = custom_notes
            print(f"[{__name__}] Metadata will be embedded.")
        else:
            print(f"[{__name__}] Metadata embedding is disabled.")

        # --- Step 6: Save to Disk ---
        if save_to_disk:
            # Dynamic filename templating
            try:
                # Sanitize prefix for path operations
                base_prefix = time.strftime(os.path.basename(filename_prefix), time.localtime())
                subfolder_prefix = os.path.dirname(filename_prefix)
            except Exception as e:
                print(f"[{__name__}] Warning: Invalid filename prefix format. Using default. Error: {e}")
                base_prefix = "ComfyUI_generated_audio"
                subfolder_prefix = ""

            output_dir_local = os.path.join(AUDIO_OUTPUT_DIR, subfolder_prefix)
            os.makedirs(output_dir_local, exist_ok=True)

            timestamp = int(time.time())
            save_results = _save_audio_with_av(
                processed_audio,
                samplerate,
                output_dir_local,
                base_prefix,
                timestamp % 100000,
                save_format.lower(),
                metadata,
                quality_setting=mp3_quality if save_format == "MP3" else (opus_quality if save_format == "OPUS" else None)
            )

            # Update UI text with the saved path
            if save_results and "filepath" in save_results[0]:
                ui_text.append(f"Saved: {os.path.relpath(save_results[0]['filepath'], folder_paths.get_output_directory())}")
            else:
                ui_text.append("Error during save.")

        else:
            ui_text.append("Not saved to disk.")

        # --- Step 7: Generate Waveform Image ---
        waveform_image_tensor = torch.zeros((1, waveform_height, waveform_width, 3), dtype=torch.float32)
        fig = None
        try:
            audio_for_plot = processed_audio.cpu().numpy()
            if audio_for_plot.ndim > 1:
                audio_for_plot = audio_for_plot[0] # Plot first channel

            num_samples = audio_for_plot.shape[-1]
            time_axis = np.linspace(0, num_samples / samplerate, num_samples)

            dpi_val = 100
            fig, ax = plt.subplots(figsize=(waveform_width / dpi_val, waveform_height / dpi_val), dpi=dpi_val)
            ax.axis("off")
            ax.set_position([0, 0, 1, 1])
            ax.plot(time_axis, audio_for_plot, color=waveform_color, linewidth=0.5)
            ax.set_facecolor(waveform_background_color)
            fig.patch.set_facecolor(waveform_background_color)
            ax.set_ylim([-1.05, 1.05])

            buffer = built_in_io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0)
            buffer.seek(0)
            pil_image = Image.open(buffer).convert("RGB")
            pil_image = pil_image.resize((waveform_width, waveform_height), Image.LANCZOS)

            waveform_image_tensor = torch.from_numpy(np.array(pil_image)).float() / 255.0
            waveform_image_tensor = waveform_image_tensor.unsqueeze(0)
        except Exception as e:
            print(f"[{__name__}] ERROR: Failed to generate waveform image: {e}")
        finally:
            if fig is not None:
                plt.close(fig)

        # --- Step 8: Return the goods! ---
        # The first output is the ORIGINAL audio for previewing without processing
        # The second output is the waveform image of the PROCESSED audio
        return {"ui": {"text": ui_text}, "result": ({"waveform": waveform_tensor_original, "sample_rate": samplerate}, waveform_image_tensor,)}

# NODE_CLASS_MAPPINGS: This is how ComfyUI knows about our amazing node!
NODE_CLASS_MAPPINGS = {
    "AdvancedAudioPreviewAndSave": AdvancedAudioPreviewAndSave,
}

# NODE_DISPLAY_NAME_MAPPINGS: This is the name you'll see in the "Add Node" menu.
NODE_DISPLAY_NAME_MAPPINGS = {
    "AdvancedAudioPreviewAndSave": "Advanced Audio Preview & Save"
}