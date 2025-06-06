# Advanced Audio Preview & Save (AAPS) — ComfyUI Node Manual

---

## 1. What is the Advanced Audio Preview & Save Node?

The Advanced Audio Preview & Save (AAPS) node is your all-in-one solution for managing audio within ComfyUI workflows. It acts as a digital sound engineer, allowing you to process, preview, and save generated audio with advanced controls.

---

### How it Works

The AAPS node takes raw audio data from other ComfyUI nodes (like `VAEAUDIO_Decode` or any other audio-producing node) through a standardized `AUDIO` input. Once it receives the audio, it performs several key functions:

- **Real-time Audio Preview**  
  Outputs the processed audio in a format compatible with ComfyUI's built-in "Preview Audio" node.

- **Intelligent Audio Saving**  
  Save your audio to disk in FLAC (lossless), MP3 (universally compatible), or OPUS (efficient for web/streaming).

- **Workflow Metadata Embedding**  
  Automatically embeds your entire ComfyUI workflow into the audio file's hidden metadata. Drag the saved file back into ComfyUI to reload the workflow.

- **Visual Waveform Generation**  
  Generates a waveform image of your audio, output as an IMAGE tensor—compatible with Preview Image and Save Image nodes.

---

### What it Does (Key Features)

- **Audio Output for Preview**  
  Connects directly to ComfyUI's "Preview Audio" node.

- **Waveform Image Output**  
  Generates a visual waveform for "Preview Image" or "Save Image" nodes.

- **Save to Disk Control**  
  Toggle saving on/off; choose between FLAC, MP3, OPUS formats.

- **Metadata Privacy Filter**  
  Toggle to control embedding of metadata, including workflow, prompts, and notes.

- **Custom Notes**  
  Add your own text notes into the audio file’s metadata.

- **Audio Normalization**  
  Adjusts audio loudness to prevent clipping and ensure consistent volume.

- **Customizable Waveform Appearance**  
  Set line/background colors and image dimensions.

---

### How to Use It

1. **Connect Audio Input**  
   Connect the `AUDIO` output from your source node (e.g., `VAEAUDIO_Decode`) to the `audio_input` of the AAPS node.

2. **Connect for Preview**  
   Connect AAPS’s `AUDIO` output to a `Preview Audio` node.

3. **Connect Waveform Image**  
   Connect AAPS’s `WAVEFORM_IMAGE` output to a `Preview Image` or `Save Image` node.

4. **Configure Settings**  
   Adjust parameters such as `filename_prefix`, `save_format`, `save_to_disk`, and waveform appearance.

5. **Queue Prompt**  
   Click "Queue Prompt" in ComfyUI. The node will process, preview, and optionally save the audio and waveform.

This node is especially useful for audio diffusion models like **Ace-Step**, and while it's optimized for audio, it could benefit other media workflows with further testing.

---

## 2. Detailed Parameter Information

---

### audio_input

- **Type**: AUDIO (Required)  
- **Description**: The primary audio input. Expects a dictionary with waveform tensor and sample rate.  
- **Use & Why**: Main entry point for sound. Required for processing.

---

### filename_prefix

- **Type**: STRING (Required)  
- **Default**: `"ComfyUI_generated_audio"`  
- **Description**: Prefix for saved file names. Supports subdirectory notation (e.g., `project1/song`).  
- **Use & Why**: Organize saved files. Subdirectories will be auto-created under `ComfyUI_AdvancedAudioOutputs`.

---

### save_format

- **Type**: ENUM (Required)  
- **Options**: `"MP3"`, `"FLAC"`, `"OPUS"`  
- **Description**: Output format for saving audio.  
- **Use & Why**:
  - **MP3**: Widely compatible, lossy compression.
  - **FLAC**: Lossless, large file size, best for archiving.
  - **OPUS**: Efficient, ideal for streaming, high quality at small size.

---

### save_to_disk

- **Type**: BOOLEAN (Optional)  
- **Default**: `False`  
- **Description**: If `True`, saves audio to disk; if `False`, skips saving.  
- **Use & Why**: Prevents clutter during previews. Enable for final exports.

---

### save_metadata

- **Type**: BOOLEAN (Optional)  
- **Default**: `True`  
- **Description**: If `True`, embeds ComfyUI workflow and notes into metadata.  
- **Use & Why**: Great for reproducibility. Turn off to reduce file size or maintain privacy.

---

### custom_notes

- **Type**: STRING (Optional)  
- **Default**: `""`  
- **Description**: Personal notes saved into audio metadata.  
- **Use & Why**: Add reminders, notes, or other context. Saved only if `save_metadata = True`.

---

### normalize_audio

- **Type**: BOOLEAN (Optional)  
- **Default**: `True`  
- **Description**: Scales audio to peak amplitude ~0.99.  
- **Use & Why**: Prevents clipping. Uncheck only for specific use cases (e.g., dynamic range preservation).

---

### waveform_color

- **Type**: STRING (Optional)  
- **Default**: `"hotpink"`  
- **Description**: Color of the waveform line (HTML name or hex code).  
- **Use & Why**: Customize the waveform’s look.

---

### waveform_background_color

- **Type**: STRING (Optional)  
- **Default**: `"black"`  
- **Description**: Background color of waveform image.  
- **Use & Why**: For better contrast or theme-matching.

---

### mp3_quality

- **Type**: ENUM (Optional)  
- **Options**: `"V0"`, `"128k"`, `"320k"`  
- **Default**: `"V0"`  
- **Description**: Quality setting for MP3 output.  
- **Use & Why**:
  - `V0`: Best balance (high-quality VBR).
  - `128k`: Standard streaming quality.
  - `320k`: Maximum CBR quality.

---

### opus_quality

- **Type**: ENUM (Optional)  
- **Options**: `"64k"`, `"96k"`, `"128k"`, `"192k"`, `"320k"`  
- **Default**: `"128k"`  
- **Description**: Bitrate setting for OPUS output.  
- **Use & Why**: Higher bitrates = better quality, larger files.

---

### waveform_width

- **Type**: INT (Optional)  
- **Default**: `800`  
- **Range**: 100–2048 (step 16)  
- **Description**: Width of waveform image (pixels).  
- **Use & Why**: Wider images show more detail over time.

---

### waveform_height

- **Type**: INT (Optional)  
- **Default**: `150`  
- **Range**: 50–1024 (step 16)  
- **Description**: Height of waveform image (pixels).  
- **Use & Why**: Taller images show amplitude more clearly.

---

## 3. In-Depth Nerd Technical Information

---

### Core Audio Handling (torchaudio & md_io)

- Uses `torch.Tensor` for waveform data.
- `md_io.audio_from_comfy_3d`: Extracts waveform/sample rate from ComfyUI `AUDIO` input.
- Handles GPU tensors by moving them to CPU.
- Normalization: Scales to `0.99 / peak_val`, unless signal is silent.

---

### Robust Audio Saving (PyAV / FFmpeg)

- Uses **PyAV** (FFmpeg bindings) to encode MP3, FLAC, OPUS.
- **In-memory WAV buffer** is created using `torchaudio.save` before PyAV encodes it.
- **Opus Resampling**: Uses `torchaudio.functional.resample` if needed (Opus requires 48k or lower).
- **MP3** uses `libmp3lame`; `V0` = `qscale=1`, others = constant bitrate.
- **FLAC** requires no extra options (lossless).
- Metadata dictionary is injected using `output_container.metadata`.
- If `save_metadata = False`, nothing is embedded (honors global `--disable-metadata` flag).
- Failures handled with a try-except block and console errors.

---

### Waveform Visualization (Matplotlib & PIL)

- Uses `matplotlib` with `'Agg'` backend (headless rendering).
- Pixel-accurate dimensions calculated from `waveform_width` and `waveform_height`.
- Axis, padding, and labels removed using:
  ```python
  ax.axis("off")
  ax.set_position([0, 0, 1, 1])
  plt.tight_layout(pad=0)
  ```
- Saved to in-memory PNG buffer → `PIL.Image.open(...)`.
- Resized with `Image.LANCZOS`.
- Converted to float32 tensor with shape `[1, H, W, 3]` for ComfyUI.

---

### File Management and Naming

- Files are saved to:  
  `ComfyUI_AdvancedAudioOutputs/`

- Filenames include:  
  `filename_prefix + timestamp_modulo + 5-digit counter`

- Prevents overwriting and ensures uniqueness.

---

This node exemplifies best practices in Python-based media processing, integrating **torchaudio**, **PyAV**, **Matplotlib**, and **Pillow** to offer a powerful, user-friendly tool for ComfyUI audio workflows.
