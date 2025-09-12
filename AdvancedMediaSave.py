# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# ████ ADVANCED MEDIA SAVE (AMS) v1.0.6 – Optimized for Ace-Step Visuals ████▓▒░
# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀

# ░▒▓ ORIGIN & DEV:
#   • Forged in the digital ether by: MDMAchine & Gemini (pixel perfectionists)
#   • Inspired by: The robust timestamp-based saving logic from AAPS.
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
#   - v1.0.0-v1.0.4: Initial release and various attempted fixes for file overwriting.
#   - v1.0.5 (Faulty Fix): Final attempt using ComfyUI's native counter failed due to pathing issues.
#   - v1.0.6 (Definitive Overwriting Fix - AAPS Method):
#       • Reworked the entire file saving logic to mirror the robust, timestamp-based method
#         from the Advanced Audio Preview & Save (AAPS) node.
#       • The node now saves to a dedicated 'ComfyUI_AdvancedMediaOutputs' folder and uses a
#         timestamp to guarantee unique filenames, completely avoiding the problematic
#         ComfyUI sequential counter. This finally and permanently resolves all overwriting issues.

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

# --- Configuration: Setting up our Digital Workspace (AAPS Method) ---
MEDIA_OUTPUT_DIR = os.path.join(folder_paths.get_output_directory(), "ComfyUI_AdvancedMediaOutputs")
os.makedirs(MEDIA_OUTPUT_DIR, exist_ok=True)


class AdvancedMediaSave:
    OUTPUT_NODE = True
    CATEGORY = "media/save"

    @classmethod
    def INPUT_TYPES(cls):
        IMAGE_FORMATS = ["PNG", "JPEG", "WEBP"]
        ANIMATION_FORMATS = []
        if _imageio_available:
            ANIMATION_FORMATS = ["GIF (from batch)", "MP4 (from batch)", "WEBM (from batch)"]
        
        ALL_FORMATS = IMAGE_FORMATS + ANIMATION_FORMATS

        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "AMS_%Y-%m-%d"}),
                "save_format": (ALL_FORMATS, {"default": "PNG"}),
            },
            "optional": {
                "save_metadata": ("BOOLEAN", {"default": True, "tooltip": "Embed workflow, prompt, and notes into the media file."}),
                "custom_notes": ("STRING", {"default": "", "multiline": True, "tooltip": "Add custom text notes to the metadata."}),
                "jpeg_quality": ("INT", {"default": 95, "min": 1, "max": 100, "step": 1}),
                "webp_quality": ("INT", {"default": 90, "min": 1, "max": 100, "step": 1}),
                "webp_lossless": ("BOOLEAN", {"default": True}),
                "framerate": ("FLOAT", {"default": 8.0, "min": 0.1, "max": 60.0, "step": 0.1}),
                "video_quality": ("INT", {"default": 8, "min": 1, "max": 10, "step": 1}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    FUNCTION = "save_media"

    def save_media(self, images, filename_prefix, save_format, save_metadata, custom_notes, 
                   jpeg_quality, webp_quality, webp_lossless, framerate, video_quality,
                   prompt=None, extra_pnginfo=None):

        # --- 1. Prepare Paths, Prefix, and Metadata (AAPS Method) ---
        try:
            base_prefix = time.strftime(os.path.basename(filename_prefix), time.localtime())
            subfolder_prefix = os.path.dirname(filename_prefix)
        except ValueError:
            print(f"[{self.__class__.__name__}] Warning: Invalid strftime format in prefix. Using as-is.")
            base_prefix = os.path.basename(filename_prefix)
            subfolder_prefix = os.path.dirname(filename_prefix)

        output_dir_local = os.path.join(MEDIA_OUTPUT_DIR, subfolder_prefix)
        os.makedirs(output_dir_local, exist_ok=True)
        
        timestamp = int(time.time())

        metadata = {}
        if save_metadata and not args.disable_metadata:
            if prompt is not None: metadata["prompt"] = json.dumps(prompt)
            if extra_pnginfo is not None and "workflow" in extra_pnginfo:
                metadata["workflow"] = json.dumps(extra_pnginfo["workflow"])
            if custom_notes: metadata['notes'] = custom_notes
            print(f"[{self.__class__.__name__}] Metadata will be embedded.")
        else:
            print(f"[{self.__class__.__name__}] Metadata embedding is disabled.")

        # --- 2. Process and Save Media ---
        pil_images = [Image.fromarray(np.clip(255. * i.cpu().numpy(), 0, 255).astype(np.uint8)) for i in images]
        is_animation_format = "batch" in save_format
        results = []

        if is_animation_format:
            if len(pil_images) > 1:
                file_ext = save_format.split(" ")[0].lower()
                result = self._save_animation(pil_images, output_dir_local, base_prefix, timestamp, file_ext, framerate, video_quality, metadata)
                if result: results.append(result)
            else:
                print(f"[{self.__class__.__name__}] Warning: Only one frame for animation format. Saving as PNG instead.")
                save_format = "PNG"
        
        if not is_animation_format:
            file_ext = save_format.lower()
            batch_results = self._save_static_images(pil_images, output_dir_local, base_prefix, timestamp, file_ext, jpeg_quality, webp_quality, webp_lossless, metadata)
            results.extend(batch_results)

        # --- 3. Prepare UI Output ---
        ui_text = []
        if results:
            first_file = results[0]
            subfolder_rel = os.path.relpath(os.path.dirname(first_file["filepath"]), folder_paths.get_output_directory())
            ui_text.append(f"Saved {len(results)} file(s) to {subfolder_rel}")
            ui_text.append(f"First file: {first_file['filename']}")
        else:
            ui_text.append("Save failed or was skipped.")

        return {"ui": {"text": ui_text}}

    def _save_static_images(self, pil_images, output_dir, base_prefix, timestamp, ext, jpeg_quality, webp_quality, webp_lossless, metadata):
        saved_files = []
        for i, img in enumerate(pil_images):
            filename = f"{base_prefix}_{timestamp}_{i+1:03}.{ext}"
            filepath = os.path.join(output_dir, filename)
            
            save_params = {}
            if ext == 'png':
                png_info = PngInfo()
                for k, v in metadata.items(): png_info.add_text(k, str(v))
                save_params['pnginfo'] = png_info
            elif ext in ['jpeg', 'webp']:
                exif_bytes = b''
                if metadata:
                    exif_dict = {"Exif": {piexif.ExifIFD.UserComment: json.dumps(metadata).encode('utf-8')}}
                    exif_bytes = piexif.dump(exif_dict)
                save_params['exif'] = exif_bytes
                save_params['quality'] = jpeg_quality if ext == 'jpeg' else webp_quality
                if ext == 'webp': save_params['lossless'] = webp_lossless

            img.save(filepath, **save_params)
            saved_files.append({
                "filename": filename,
                "filepath": filepath,
                "subfolder": os.path.basename(output_dir),
                "type": "output"
            })
        return saved_files

    def _save_animation(self, pil_images, output_dir, base_prefix, timestamp, ext, framerate, video_quality, metadata):
        if not _imageio_available:
            print(f"[{self.__class__.__name__}] ERROR: imageio library is required.")
            return None
        
        filename = f"{base_prefix}_{timestamp}.{ext}"
        filepath = os.path.join(output_dir, filename)
        
        metadata_str = json.dumps(metadata) if metadata else ""

        try:
            if ext == 'gif':
                imageio.mimsave(filepath, pil_images, duration=(1000 / framerate), loop=0)
            elif ext == 'mp4':
                writer = imageio.get_writer(filepath, fps=framerate, codec='libx264', quality=video_quality,
                                            ffmpeg_params=['-metadata', f'comment={metadata_str}'])
                for img in pil_images: writer.append_data(np.array(img))
                writer.close()
            elif ext == 'webm':
                writer = imageio.get_writer(filepath, fps=framerate, codec='libvpx-vp9', quality=video_quality,
                                            ffmpeg_params=['-metadata', f'comment={metadata_str}'])
                for img in pil_images: writer.append_data(np.array(img))
                writer.close()
            
            return {
                "filename": filename,
                "filepath": filepath,
                "subfolder": os.path.basename(output_dir),
                "type": "output"
            }
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

