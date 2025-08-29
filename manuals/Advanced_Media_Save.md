# Comprehensive Manual: Advanced Media Save (AMS v1.0.0)

Welcome to the complete guide for the **Advanced Media Save (AMS v1.0.0)** node, the ultimate media output toolkit for ComfyUI. This manual provides everything you need to know, from basic setup to advanced format controls and workflow archiving.

---

### **Table of Contents**

1.  **Introduction**
    * What is the AMS Node?
    * Who is this Node For?
    * Key Features in Version 1.0.0
2.  **Installation**
3.  **Core Concepts: Media Saving & Archiving**
    * Image vs. Animation Formats
    * Compression: Lossy vs. Lossless
    * Metadata Embedding: Archiving Your Workflow
4.  **Node Setup and Workflow**
    * How to Use It: A Step-by-Step Guide
5.  **Parameter Deep Dive**
    * Primary Controls: Input & Output
    * Metadata Controls: Notes & Workflow
    * Format-Specific Controls: Quality & Animation
6.  **Practical Recipes & Use Cases**
    * Recipe 1: Saving a High-Quality Portfolio Image
    * Recipe 2: Creating an Animated GIF from a Batch
    * Recipe 3: Archiving a Workflow with Full Metadata
7.  **Technical Deep Dive**
    * The Logic Flow
8.  **Troubleshooting & FAQ**

---

## 1. Introduction

### What is the AMS Node?

The **Advanced Media Save (AMS)** node is a powerful and versatile "swiss-army knife" for handling images and video at the end of a ComfyUI workflow. It's designed to be the final step for any visual generation process, providing all the tools you need to save single images, image batches, or animations. It takes raw image tensors and transforms them into polished, shareable files in a variety of formats, complete with embedded workflow data for perfect reproducibility.

### Who is this Node For?

* **AI Artists & Image Creators:** Anyone generating still images who needs control over format (PNG, JPEG, WEBP), quality, and metadata.
* **Animators:** Users creating image sequences who want to easily convert them into animated GIFs, MP4s, or WEBMs.
* **Workflow Archivists:** Anyone who wants a foolproof way to save their exact ComfyUI workflow *inside* the image file it generated.
* **All ComfyUI Users:** Its intuitive interface and powerful features make it an essential replacement for the basic "Save Image" node.

### Key Features in Version 1.0.0

* **Multi-Format Saving:** Save images in **PNG**, **JPEG**, or **WEBP** (lossy/lossless).
* **Animation Conversion:** Automatically convert image batches into **GIF**, **MP4 (H.264)**, or **WEBM (VP9)**.
* **Workflow Metadata Embedding:** Automatically save the entire workflow and your custom notes directly into the media file's metadata.
* **Quality & Framerate Control:** Fine-tune the compression quality for JPEGs and WEBP, and set the precise framerate for animations.
* **Dynamic Filenaming:** Use time and date codes (e.g., `%Y-%m-%d`) in your filenames for automatic organization.
* **UI Feedback:** The node displays the final save path after execution for easy access to your files.

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

## 3. Core Concepts: Media Saving & Archiving

### Image vs. Animation Formats

The AMS node intelligently handles both static images and animations based on your `save_format` selection.

* **Image Formats (`PNG`, `JPEG`, `WEBP`):** If you select one of these, the node will save every image in the input batch as a separate file. This is ideal for generating variations or a series of still images.
* **Animation Formats (`GIF`, `MP4`, `WEBM`):** If you select a format with `(from batch)` in its name, the node will treat the input image batch as a sequence of frames and compile them into a single animated file.

### Compression: Lossy vs. Lossless

* **Lossless (PNG, WEBP Lossless):** These formats store the image data perfectly, with no loss in quality. This is best for archival or when you need the absolute highest quality, but results in larger file sizes.
* **Lossy (JPEG, WEBP, MP4, WEBM):** These formats use clever algorithms to reduce file size by discarding some visual information that the human eye is less likely to notice. This is great for sharing on the web, but can introduce artifacts at lower quality settings.

### Metadata Embedding: Archiving Your Workflow

One of the most powerful features of ComfyUI is reproducibility. The AMS node can embed the entire workflow (the arrangement of nodes and their settings) directly into the metadata of the saved file.

For **PNG** files, this means you can simply drag and drop the saved image back into ComfyUI in the future, and it will load the exact workflow that created it. For other formats like JPEG and MP4, the data is still embedded but may require external tools to extract. This is invaluable for archiving your work or sharing your process. The `save_metadata` toggle gives you full privacy control.

---

## 4. Node Setup and Workflow

### How to Use It: A Step-by-Step Guide

1.  **Connect Image Source:** Connect the `IMAGE` output from your generation or processing node (e.g., VAE Decode, KSampler) to the `images` input of the AMS node. This can be a single image or a batch.
2.  **Set Filename and Format:**
    * Enter a name in `filename_prefix`. You can use date/time codes like `MyRender_%Y%m%d`.
    * Select your desired format from the `save_format` dropdown.
3.  **Choose Quality Options:**
    * If saving a JPEG or lossy WEBP, adjust the `jpeg_quality` or `webp_quality` slider.
    * If saving an animation, set the `framerate` and `video_quality`.
4.  **Set Metadata Options:**
    * Ensure `save_metadata` is checked if you want to embed the workflow.
    * Add any personal descriptions in `custom_notes`.
5.  **Queue Prompt:** After running, the node's UI will display the path to the saved file(s) for easy access.

---

## 5. Parameter Deep Dive

### Primary Controls: Input & Output

* **`images`** (Required): The image data stream from an upstream node. Accepts single images or batches.
* **`filename_prefix`** (`STRING`, default: `ComfyUI_media_%Y-%m-%d`): The base name for your saved file(s). Supports `strftime` codes for dynamic naming (e.g., `%H` for hour, `%M` for minute). You can also specify subdirectories like `characters/my_char_`.
* **`save_format`** (`ENUM`): The file format for saving. Formats with `(from batch)` will create a single animated file from the input batch.

### Metadata Controls: Notes & Workflow

* **`save_metadata`** (`BOOLEAN`, default: `True`): If enabled, the ComfyUI workflow and prompt data will be embedded in the saved file.
* **`custom_notes`** (`STRING`): A multiline text field where you can write personal notes that will be saved into the file's metadata.

### Format-Specific Controls: Quality & Animation

* **`jpeg_quality`** (`INT`, default: `95`): The compression quality for JPEG files (1-100). Higher is better quality and larger file size.
* **`webp_quality`** (`INT`, default: `90`): The compression quality for lossy WEBP files (1-100).
* **`webp_lossless`** (`BOOLEAN`, default: `True`): If enabled, saves WEBP files in a lossless format, ignoring the `webp_quality` setting.
* **`framerate`** (`FLOAT`, default: `8.0`): The number of frames per second for animated formats (GIF, MP4, WEBM).
* **`video_quality`** (`INT`, default: `8`): A general quality setting for video encoders (1-10). 10 is the highest quality.

---

## 6. Practical Recipes & Use Cases

### Recipe 1: Saving a High-Quality Portfolio Image

Goal: Save a final render with maximum quality for a portfolio or for further editing.

* **`save_format`**: `PNG` (for perfect lossless quality)
* **`filename_prefix`**: `Portfolio_FinalRender_`
* **`save_metadata`**: `True` (to remember how you made it)
* **`custom_notes`**: `Final version using the 'Cinematic' LoRA at 0.6 strength.`

### Recipe 2: Creating an Animated GIF from a Batch

Goal: Turn a sequence of 16 generated frames into a looping animation for sharing online.

* **`save_format`**: `GIF (from batch)`
* **`filename_prefix`**: `MyAnimation_Loop_`
* **`framerate`**: `12.0` (for a smoother animation)
* **`save_metadata`**: `False` (to keep the file size smaller for sharing)

### Recipe 3: Archiving a Workflow with Full Metadata

Goal: Save a test image and ensure you can recreate it perfectly later by reloading the workflow.

* **`save_format`**: `PNG` (required for ComfyUI's drag-and-drop workflow loading)
* **`filename_prefix`**: `tests/workflow_archive_`
* **`save_metadata`**: `True`
* **`custom_notes`**: `Initial test of the new 'DreamShaper V8' model. Sampler: DPM++ 2M Karras, Steps: 25.`

---

## 7. Technical Deep Dive

### The Logic Flow

The node follows a clear sequence to process and save your media:

1.  **Prepare File Paths:** It first interprets the `filename_prefix`, creates any necessary subdirectories, and applies date/time formatting.
2.  **Prepare Metadata:** It gathers the prompt, workflow, and custom notes into a single data structure.
3.  **Convert Tensors to Images:** The raw PyTorch `IMAGE` tensors are converted into a list of standard PIL (Pillow) images, ready for saving.
4.  **Check Format Type:** The node checks if the selected `save_format` is an animation or static image format.
5.  **Route to Saver:**
    * If it's an **animation format**, it sends the entire list of PIL images to the `_save_animation` function, which uses the `imageio` library to create a single GIF, MP4, or WEBM file.
    * If it's a **static image format**, it sends the list to the `_save_static_images` function, which loops through each image and saves it individually, applying the correct metadata format (PNG chunks or EXIF) for each one.

---

## 8. Troubleshooting & FAQ

* **"The options for GIF, MP4, etc., are missing!"**
    * This means the `imageio` library is not installed correctly in your ComfyUI's Python environment. The node requires it for all animation saving. Try to run the `pip install` command from the manual installation instructions again and restart ComfyUI.
* **"I tried to save a GIF but it saved a PNG instead."**
    * The animation formats require an input batch with more than one image. If you only provide a single image, the node will print a warning and automatically fall back to saving a single PNG to prevent an error.
* **"I can't find my saved file."**
    * By default, files are saved in `ComfyUI/output/ComfyUI_AdvancedMediaOutputs/`. If you used slashes in your `filename_prefix` (e.g., `renders/today/`), it will create those subdirectories inside the main output folder. The node's UI also displays the relative path after it finishes.
* **"I dragged my saved JPEG/MP4 file into ComfyUI and it didn't load my workflow."**
    * This is expected behavior. ComfyUI's drag-and-drop workflow-loading feature currently only supports metadata stored in **PNG** files. The workflow data *is* embedded in your JPEG or MP4, but you would need an external tool (like `exiftool`) to view it. For guaranteed reloading, save as PNG.
