# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# ████ ADVANCED MEDIA SAVE (AMS) v1.0.0 – Optimized for Ace-Step Visuals ████▓▒░
# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀

# ░▒▓ ORIGIN & DEV:
#   • Forged in the digital ether by: MDMAchine & Gemini (pixel perfectionists)
#   • Inspired by: The robust logic of AAPS and the ComfyUI media saving ecosystem
#   • License: Public Domain — Spreading the visual love

# ░▒▓ DESCRIPTION:
#   The ultimate visual media companion node for ComfyUI.
#   Processes, previews, and saves images, image batches, and videos from your workflows.
#   Handles everything from single PNGs to animated GIFs and MP4s with embedded metadata.
#   Your one-stop-shop for saving any visual output from ComfyUI.

# ░▒▓ FEATURES:
#   ✓ Save single images or batches with ease.
#   ✓ Supported Image Formats: PNG, JPEG, WEBP (lossy/lossless).
#   ✓ Supported Animation/Video Formats: GIF, MP4 (H.264), WEBM (VP9).
#   ✓ Metadata Privacy Filter: Toggle to embed or strip workflow data.
#   ✓ Add custom text notes to your media's metadata.
#   ✓ Quality controls for JPEG, WEBP, and video formats.
#   ✓ Set the framerate for animated GIFs and videos.
#   ✓ Dynamic filename templating (e.g., 'my_render_%Y-%m-%d').
#   ✓ Displays the final saved file path in the node UI for easy access.

# ░▒▓ CHANGELOG:
#   - v1.0.0 (Initial Release):
#       • Core functionality for saving images, batches, and videos.
#       • Full metadata embedding support for all relevant formats.
#       • UI controls for format, quality, and framerate.
#       • Dynamic filename parsing.

# ░▒▓ CONFIGURATION:
#   → Primary Use: Saving final images, animations, or video clips from workflows.
#   → Secondary Use: Converting image batches into different animated formats.
#   → Edge Use: Archiving workflows directly inside the generated media files.

# ░▒▓ WARNING:
#   This node may cause:
#   ▓▒░ An overwhelming desire to save everything in multiple formats.
#   ▓▒░ Extreme organization of your output folders.
#   ▓▒░ A sudden appreciation for embedded metadata.

# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄

import os
import torch
import numpy as np
import time
import json
from PIL import Image
from PIL.PngImagePlugin import PngInfo

# Metadata and saving utilities
import piexif
from comfy.cli_args import args
import folder_paths

# Video/Animation library
try:
    import imageio.v2 as imageio
    _imageio_available = True
except ImportError:
    print("[AdvancedMediaSave] Warning: imageio not found. Saving animations (GIF, MP4, WEBM) will be disabled.")
    _imageio_available = False

# --- Configuration: Setting up our Digital Workspace ---
MEDIA_OUTPUT_DIR = os.path.join(folder_paths.get_output_directory(), "ComfyUI_AdvancedMediaOutputs")
os.makedirs(MEDIA_OUTPUT_DIR, exist_ok=True)
print(f"[{__name__}] Media output directory: {MEDIA_OUTPUT_DIR}")


class AdvancedMediaSave:
    # We are an output node, so we don't need to return anything to other nodes.
    OUTPUT_NODE = True
    CATEGORY = "MD_Nodes/Save"

    @classmethod
    def INPUT_TYPES(cls):
        # Define available formats, separating image and animation types
        IMAGE_FORMATS = ["PNG", "JPEG", "WEBP"]
        ANIMATION_FORMATS = []
        if _imageio_available:
            ANIMATION_FORMATS = ["GIF (from batch)", "MP4 (from batch)", "WEBM (from batch)"]
        
        ALL_FORMATS = IMAGE_FORMATS + ANIMATION_FORMATS

        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "ComfyUI_media_%Y-%m-%d"}),
                "save_format": (ALL_FORMATS, {"default": "PNG"}),
            },
            "optional": {
                "save_metadata": ("BOOLEAN", {"default": True, "tooltip": "Embed workflow, prompt, and notes into the media file."}),
                "custom_notes": ("STRING", {"default": "", "multiline": True, "tooltip": "Add custom text notes to the metadata."}),
                "jpeg_quality": ("INT", {"default": 95, "min": 1, "max": 100, "step": 1, "tooltip": "Quality for JPEG format."}),
                "webp_quality": ("INT", {"default": 90, "min": 1, "max": 100, "step": 1, "tooltip": "Quality for WEBP format (lossy)."}),
                "webp_lossless": ("BOOLEAN", {"default": True, "tooltip": "Use lossless compression for WEBP."}),
                "framerate": ("FLOAT", {"default": 8.0, "min": 0.1, "max": 60.0, "step": 0.1, "tooltip": "Framerate for animated formats (GIF, MP4, WEBM)."}),
                "video_quality": ("INT", {"default": 8, "min": 1, "max": 10, "step": 1, "tooltip": "Quality for video formats (10 is best, 1 is worst)."}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    FUNCTION = "save_media"

    def save_media(self, images, filename_prefix, save_format, save_metadata, custom_notes, 
                   jpeg_quality, webp_quality, webp_lossless, framerate, video_quality,
                   prompt=None, extra_pnginfo=None):

        # --- 1. Prepare filenames and directories ---
        try:
            # Sanitize prefix and apply time formatting
            base_prefix = time.strftime(os.path.basename(filename_prefix), time.localtime())
            subfolder = os.path.dirname(filename_prefix)
        except Exception as e:
            print(f"[{self.__class__.__name__}] Warning: Invalid filename prefix format. Using default. Error: {e}")
            base_prefix = "ComfyUI_media"
            subfolder = ""

        output_dir = os.path.join(MEDIA_OUTPUT_DIR, subfolder)
        os.makedirs(output_dir, exist_ok=True)

        # --- 2. Prepare Metadata ---
        metadata = {}
        if save_metadata and not args.disable_metadata:
            if prompt is not None:
                metadata["prompt"] = json.dumps(prompt)
            if extra_pnginfo is not None and "workflow" in extra_pnginfo:
                metadata["workflow"] = json.dumps(extra_pnginfo["workflow"])
            if custom_notes:
                metadata['notes'] = custom_notes
            print(f"[{self.__class__.__name__}] Metadata will be embedded.")
        else:
            print(f"[{self.__class__.__name__}] Metadata embedding is disabled.")

        # --- 3. Process and Save Media ---
        # Convert tensor to a list of PIL Images
        pil_images = []
        for image_tensor in images:
            img_np = image_tensor.cpu().numpy() * 255.0
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)
            pil_images.append(Image.fromarray(img_np))

        num_images = len(pil_images)
        is_animation_format = "batch" in save_format

        results = []
        
        if is_animation_format:
            # --- Save as Animation/Video ---
            if num_images > 1:
                file_ext = save_format.split(" ")[0].lower()
                filepath = self._save_animation(pil_images, output_dir, base_prefix, file_ext, framerate, video_quality, metadata)
                if filepath:
                    results.append({"filename": os.path.basename(filepath), "subfolder": subfolder, "type": "output", "filepath": filepath})
            else:
                print(f"[{self.__class__.__name__}] Warning: Only one frame provided for an animation format. Saving as a PNG instead.")
                save_format = "PNG" # Fallback to PNG
        
        if not is_animation_format: # This will also catch the fallback from animation
            # --- Save as Static Images ---
            file_ext = save_format.lower()
            filepaths = self._save_static_images(pil_images, output_dir, base_prefix, file_ext, jpeg_quality, webp_quality, webp_lossless, metadata)
            for fp in filepaths:
                results.append({"filename": os.path.basename(fp), "subfolder": subfolder, "type": "output", "filepath": fp})

        # --- 4. Prepare UI Output ---
        ui_text = [f"Saved {len(results)} file(s) to {os.path.relpath(output_dir, folder_paths.get_output_directory())}"]
        if results:
            # Display the path of the first saved file for convenience
            ui_text.append(f"First file: {os.path.basename(results[0]['filepath'])}")

        return {"ui": {"text": ui_text, "images": []}} # No image preview in the node itself, ComfyUI handles previews for output nodes

    def _save_static_images(self, pil_images, output_dir, base_prefix, ext, jpeg_quality, webp_quality, webp_lossless, metadata):
        saved_paths = []
        counter = folder_paths.get_output_counter(output_dir, base_prefix)
        
        for img in pil_images:
            filename = f"{base_prefix}_{counter:05}.{ext}"
            filepath = os.path.join(output_dir, filename)
            
            save_params = {}
            if ext == 'png':
                png_info = PngInfo()
                for k, v in metadata.items():
                    png_info.add_text(k, str(v))
                save_params['pnginfo'] = png_info
            elif ext in ['jpeg', 'webp']:
                exif_bytes = b''
                if metadata:
                    exif_dict = {"Exif": {piexif.ExifIFD.UserComment: json.dumps(metadata).encode('utf-8')}}
                    exif_bytes = piexif.dump(exif_dict)
                save_params['exif'] = exif_bytes
                save_params['quality'] = jpeg_quality if ext == 'jpeg' else webp_quality
                if ext == 'webp':
                    save_params['lossless'] = webp_lossless

            img.save(filepath, **save_params)
            saved_paths.append(filepath)
            counter += 1
            
        return saved_paths

    def _save_animation(self, pil_images, output_dir, base_prefix, ext, framerate, video_quality, metadata):
        if not _imageio_available:
            print(f"[{self.__class__.__name__}] ERROR: imageio library is required to save animations.")
            return None
            
        counter = folder_paths.get_output_counter(output_dir, base_prefix)
        filename = f"{base_prefix}_{counter:05}.{ext}"
        filepath = os.path.join(output_dir, filename)
        
        # Convert metadata to a string for embedding
        metadata_str = json.dumps(metadata) if metadata else ""

        try:
            if ext == 'gif':
                imageio.mimsave(filepath, pil_images, duration=(1000 / framerate), loop=0)
                # Note: Standard GIF format does not support metadata like EXIF or PNG chunks.
            elif ext == 'mp4':
                # H.264 is widely supported
                writer = imageio.get_writer(filepath, fps=framerate, codec='libx264', quality=video_quality,
                                            ffmpeg_params=['-metadata', f'comment={metadata_str}'])
                for img in pil_images:
                    writer.append_data(np.array(img))
                writer.close()
            elif ext == 'webm':
                # VP9 is a modern, efficient codec
                writer = imageio.get_writer(filepath, fps=framerate, codec='libvpx-vp9', quality=video_quality,
                                            ffmpeg_params=['-metadata', f'comment={metadata_str}'])
                for img in pil_images:
                    writer.append_data(np.array(img))
                writer.close()
            
            return filepath
        except Exception as e:
            print(f"[{self.__class__.__name__}] ERROR saving animation to {filepath}: {e}")
            return None


# --- Node Registration ---
NODE_CLASS_MAPPINGS = {
    "AdvancedMediaSave": AdvancedMediaSave
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AdvancedMediaSave": "Advanced Media Save 🖼️"
}
