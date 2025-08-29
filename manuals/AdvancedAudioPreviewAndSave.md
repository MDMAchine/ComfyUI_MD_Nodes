# Comprehensive Manual: Advanced Audio Preview & Save (AAPS v0.4.1)

Welcome to the complete guide for the **Advanced Audio Preview & Save (AAPS v0.4.1)** node, an all-in-one audio toolkit for ComfyUI. This manual provides everything you need to know, from basic setup to advanced processing techniques and technical details.

---

### **Table of Contents**

1.  **Introduction**
    * What is the AAPS Node?
    * Who is this Node For?
    * Key Features in Version 0.4.1
2.  **Installation**
3.  **Core Concepts: The Audio Processing Chain**
    * Audio Normalization: Peak vs. RMS
    * The Soft Limiter: Preventing Distortion
    * Metadata Embedding: Archiving Your Workflow
    * Finishing Touches: Fades & Mono Conversion
4.  **Node Setup and Workflow**
    * How to Use It: A Step-by-Step Guide
5.  **Parameter Deep Dive**
    * Primary Controls: Input & Output
    * Processing Controls: Audio Shaping
    * Normalization Controls: Volume & Dynamics
    * Metadata Controls: Notes & Workflow
    * Waveform Controls: Visualization
    * Format-Specific Controls: Quality
6.  **Practical Recipes & Use Cases**
    * Recipe 1: Finalizing Music for Consistent Volume
    * Recipe 2: Quick Preview & Visualization (No Save)
    * Recipe 3: Archiving a Generative Audio Workflow
7.  **Technical Deep Dive**
    * The Order of Operations
8.  **Troubleshooting & FAQ**

---

## 1. Introduction

### What is the AAPS Node?

The **Advanced Audio Preview & Save (AAPS)** node is a powerful and versatile "swiss-army knife" for handling audio at the end of a ComfyUI workflow. It's designed to be the final step for any audio generation process, providing essential tools for processing, previewing, visualizing, and saving your sound. It takes raw audio and transforms it into a polished, shareable file, complete with embedded workflow data for perfect reproducibility.

### Who is this Node For?

* **AI Musicians & Sound Designers:** Anyone generating music, sound effects, or dialogue who needs to normalize, finalize, and save their creations in a professional format.
* **Video and Animation Creators:** Users who need to process generated audio to match their visuals, applying fades and ensuring consistent volume levels.
* **Workflow Archivists:** Anyone who wants a foolproof way to save their exact ComfyUI workflow *inside* the audio file it generated.
* **All ComfyUI Audio Users:** Its intuitive interface makes it an essential replacement for basic audio-saving nodes.

### Key Features in Version 0.4.1

* **Multi-Format Saving:** Save audio in high-quality lossless **FLAC**, universal **MP3**, or efficient **OPUS**.
* **Advanced Normalization:** Go beyond simple volume changes with **Peak** and **RMS** normalization to achieve professional loudness standards.
* **Built-in Soft Limiter:** A crucial tool (using the `pedalboard` library) that prevents the harsh digital distortion (clipping) that can occur after boosting volume.
* **Workflow Metadata Embedding:** Automatically save the entire workflow and your custom notes directly into the audio file's metadata.
* **Audio Finishing Tools:** Add professional **fade-ins/fade-outs** and convert stereo audio to **mono** with ease.
* **Dynamic Filenaming:** Use time and date codes (e.g., `%Y-%m-%d`) in your filenames for automatic organization.
* **Customizable Waveform Visualizer:** Get immediate visual feedback with a waveform image output, with full control over colors and dimensions.

---

## 2. 🧰 INSTALLATION: JACK INTO THE MATRIX

This node is part of the **MD Nodes** package. All required Python libraries, including `pedalboard` for the limiter, are listed in the `requirements.txt` and should be installed automatically.

### Method 1: ComfyUI Manager (Recommended)

1.  Open the **ComfyUI Manager**.
2.  Click "Install Custom Nodes".
3.  Search for `MD Nodes` and click "Install".
4.  The manager will download the package and automatically install its dependencies.
5.  **Restart ComfyUI.**

### Method 2: Manual Installation (Git)

1.  Open a terminal or command prompt.
2.  Navigate to your `ComfyUI/custom_nodes/` directory.
3.  Run the following command to clone the repository:
    ```bash
    git clone [https://github.com/MDMAchine/ComfyUI_MD_Nodes.git](https://github.com/MDMAchine/ComfyUI_MD_Nodes.git)
    ```
4.  Install the required dependencies by running:
    ```bash
    pip install -r ComfyUI_MD_Nodes/requirements.txt
    ```
5.  **Restart ComfyUI.**

After restarting, the node and all its features should be fully available. Don’t forget, even gods need to reboot.

---

## 3. Core Concepts: The Audio Processing Chain

### Audio Normalization: Peak vs. RMS

Normalization is the process of adjusting audio volume to a standard level. AAPS offers two professional methods:

* **Peak Normalization:** This is the simplest method. It finds the single loudest point (the "peak") in your audio and turns the entire volume up or down so that this peak hits a target level (usually just below the maximum to avoid distortion). It's great for making sure nothing clips, but it doesn't guarantee the audio will *sound* consistently loud.
* **RMS Normalization:** This method measures the **average** volume (Root Mean Square) of the audio and adjusts it to a target. This is much closer to how humans perceive loudness. It's the professional standard for making different songs, dialogue clips, or sound effects sound like they belong together.

### The Soft Limiter: Preventing Distortion

When you boost the volume (especially with RMS normalization), some individual peaks might be pushed beyond the maximum digital level (0 dBFS). A simple "clamp" would just chop these peaks off, creating a harsh, ugly clicking distortion.

A **limiter** is a smarter tool. It acts like an ultra-fast, transparent volume knob that automatically turns down any sound about to distort, preserving the sound quality. **It is highly recommended to keep `use_limiter` enabled when using RMS normalization.**

### Metadata Embedding: Archiving Your Workflow

One of the most powerful features of ComfyUI is reproducibility. The AAPS node can embed the entire workflow (the arrangement of nodes and their settings) directly into the metadata of the saved FLAC, MP3, or OPUS file.

This means you can simply drag and drop the saved audio file back into ComfyUI in the future, and it will load the exact workflow that created it. This is invaluable for archiving your work or sharing your process. The `save_metadata` toggle gives you full privacy control.

### Finishing Touches: Fades & Mono Conversion

* **Fades:** Abrupt starts and stops can sound jarring. The `fade_in_ms` and `fade_out_ms` controls allow you to apply smooth volume ramps to the beginning and end of your audio for a polished, professional finish.
* **Mono Conversion:** Some applications require a single-channel (mono) audio file. The `channel_mode` option allows you to instantly downmix your stereo audio to mono.

---

## 4. Node Setup and Workflow

### How to Use It: A Step-by-Step Guide

1.  **Connect Audio Source:** Connect the `AUDIO` output from your audio generation node (e.g., a VAE Decode for audio) to the `audio_input` of the AAPS node.
2.  **Set Filename and Format:**
    * Enter a name in `filename_prefix`. You can use date/time codes like `MySong_%Y%m%d`.
    * Select `MP3`, `FLAC`, or `OPUS` from the `save_format` dropdown.
3.  **Choose Processing Options:**
    * Set your desired `normalize_method`. If using `RMS`, set the `target_rms_db` (e.g., -14 for music, -18 for dialogue).
    * Keep `use_limiter` enabled for best results with RMS.
    * Set `fade_in_ms` or `fade_out_ms` if needed.
4.  **Enable Saving:** Check the `save_to_disk` box to save the file. If this is unchecked, the node will still process the audio and generate the waveform, but it won't write a file.
5.  **Connect Outputs (Optional):**
    * The `AUDIO` output passes through the *original, unprocessed* audio. You can connect this to an "Audio Preview" node for an A/B comparison.
    * The `WAVEFORM_IMAGE` output can be connected to a "Save Image" node to save the visualization.
6.  **Queue Prompt:** After running, the node's UI will display the path to the saved file for easy access.

---

## 5. Parameter Deep Dive

### Primary Controls: Input & Output

* **`audio_input`** (Required): The audio data stream from an upstream node.
* **`filename_prefix`** (`STRING`, default: `ComfyUI_audio_%Y-%m-%d`): The base name for your saved file. Supports `strftime` codes for dynamic naming (e.g., `%H` for hour, `%M` for minute).
* **`save_format`** (`ENUM`): The file format for saving: `MP3`, `FLAC`, or `OPUS`.
* **`save_to_disk`** (`BOOLEAN`, default: `False`): Master switch to enable or disable writing the audio file to your hard drive.

### Processing Controls: Audio Shaping

* **`channel_mode`** (`ENUM`, default: `Keep Original`): Choose to keep the original channels (e.g., stereo) or downmix to `Convert to Mono`.
* **`fade_in_ms`** (`INT`, default: `0`): Duration in milliseconds for a smooth volume ramp at the start of the audio.
* **`fade_out_ms`** (`INT`, default: `0`): Duration in milliseconds for a smooth volume ramp at the end of the audio.

### Normalization Controls: Volume & Dynamics

* **`normalize_method`** (`ENUM`, default: `Peak`): The algorithm used for volume adjustment. `Off`, `Peak`, or `RMS`.
* **`target_rms_db`** (`INT`, default: `-16`): The target average loudness in dBFS for RMS normalization. Lower values (e.g., -20) are quieter; higher values (e.g., -12) are louder.
* **`use_limiter`** (`BOOLEAN`, default: `True`): If enabled, applies a soft limiter after normalization to prevent digital clipping. **Requires the `pedalboard` library.**

### Metadata Controls: Notes & Workflow

* **`save_metadata`** (`BOOLEAN`, default: `True`): If enabled, the ComfyUI workflow and prompt data will be embedded in the saved file.
* **`custom_notes`** (`STRING`): A multiline text field where you can write personal notes that will be saved into the file's metadata.

### Waveform Controls: Visualization

* **`waveform_color`** (`STRING`, default: `hotpink`): The color of the line in the waveform image. Accepts names (`blue`) or hex codes (`#FF00FF`).
* **`waveform_background_color`** (`STRING`, default: `black`): The background color of the waveform image.
* **`waveform_width`** / **`waveform_height`** (`INT`): The dimensions of the output waveform image in pixels.

### Format-Specific Controls: Quality

* **`mp3_quality`** (`ENUM`, default: `V0`): Quality setting for MP3 files. `V0` is a high-quality variable bitrate, while `128k` and `320k` are constant bitrates.
* **`opus_quality`** (`ENUM`, default: `128k`): The bitrate for OPUS files. Higher values mean better quality and larger files.

---

## 6. Practical Recipes & Use Cases

### Recipe 1: Finalizing Music for Consistent Volume

Goal: Save a generated music track so it has a standard, competitive loudness without distortion.

* **`save_format`**: `FLAC` (for mastering) or `MP3` (for sharing)
* **`mp3_quality`**: `V0`
* **`fade_in_ms`**: `100` (for a subtle fade-in)
* **`fade_out_ms`**: `2000` (for a 2-second fade-out)
* **`normalize_method`**: `RMS`
* **`target_rms_db`**: `-14` (a common target for music streaming)
* **`use_limiter`**: `True` (Essential to catch peaks)
* **`save_to_disk`**: `True`

### Recipe 2: Quick Preview & Visualization (No Save)

Goal: Quickly check the audio and see its waveform without cluttering your output folder.

* **`save_to_disk`**: `False`
* All other settings can be left at their defaults. The node will still process the audio and output the waveform image, which you can preview.

### Recipe 3: Archiving a Generative Audio Workflow

Goal: Save a sound effect losslessly and ensure you can recreate it perfectly later.

* **`save_format`**: `FLAC` (lossless is key for archives)
* **`filename_prefix`**: `SoundEffect_AlienShip_%Y%m%d`
* **`save_metadata`**: `True`
* **`custom_notes`**: `Generated with custom XYZ model at 0.8 CFG. This version has a nice metallic texture.`
* **`normalize_method`**: `Peak` (to prevent clipping without altering dynamics)
* **`save_to_disk`**: `True`

---

## 7. Technical Deep Dive

### The Order of Operations

Understanding the sequence of events is key to predicting the final output. The node processes the audio in this specific order:

1.  **Channel Conversion:** The audio is first converted to mono, if selected.
2.  **Fades:** Fade-in and fade-out ramps are applied next.
3.  **Normalization:** The volume is adjusted using the selected Peak or RMS method. This happens *after* fades.
4.  **Limiter:** The limiter is the very last step in the audio chain, catching any peaks that were pushed too high by the normalization stage.
5.  **Saving & Encoding:** This processed audio is then sent to the encoder (`PyAV`) to be written to a file.

The audio sent to the `AUDIO` output port is the **original, unprocessed audio**, allowing for direct comparison with the processed version you hear in the saved file.

---

## 8. Troubleshooting & FAQ

* **"My audio sounds distorted or has clicking noises."**
    * This is almost always digital clipping. Ensure `use_limiter` is set to `True`. If you are using RMS normalization with a high `target_rms_db` (e.g., -8), you may be pushing the audio too hard. Try a lower value like -14.
* **"The limiter option is on, but it's not working."**
    * This feature requires the `pedalboard` library. It should have been installed automatically from the package's `requirements.txt` file. Check your ComfyUI console for any errors during startup related to `pedalboard` or missing packages. If it failed, you may need to run `pip install -r path/to/ComfyUI/custom_nodes/ComfyUI_MD_Nodes/requirements.txt` manually and restart.
* **"I can't find my saved audio file."**
    * By default, files are saved in `ComfyUI/output/ComfyUI_AdvancedAudioOutputs/`. The node will also display the relative path of the saved file in its UI after it finishes running.
* **"My filename has `_2025-08-28` in it, why?"**
    * The default `filename_prefix` uses `strftime` codes (`%Y-%m-%d`) to automatically insert the current year, month, and day. You can change this to a static name or use other codes like `%H` (hour) and `%M` (minute).
* **"Why does the audio from the output port sound different from my saved file?"**
    * This is by design! The `AUDIO` output port passes through the **original, unprocessed audio** so you can use a Preview node to hear the "before" version. The saved file contains the "after" version with all your processing (fades, normalization, etc.) applied.