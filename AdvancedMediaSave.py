# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
# ████ MD_Nodes/AdvancedMediaSave – Multi-format media saving node ████▓▒░
# © 2025 MDMAchine
# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
# ░▒▓ ORIGIN & DEV:
#   • Cast into the void by: MDMAchine & Gemini
#   • Enhanced by: AAPS Save Logic (Inspiration)
#   • License: License: Apache 2.0 — Spreading the visual love

# ░▒▓ DESCRIPTION:
#   Processes, previews, and saves images (PNG/JPEG/WEBP), image batches, and
#   videos (GIF/MP4/WEBM) with robust, timestamp-based saving to prevent
#   overwrites and optional metadata embedding control.

# ░▒▓ FEATURES:
#   ✓ Save single images, batches, or animations.
#   ✓ Formats: PNG, JPEG, WEBP (lossy/lossless), GIF, MP4 (H.264), WEBM (VP9).
#   ✓ Metadata Privacy Filter: Toggle embedding or stripping workflow data.
#   ✓ Quality & Framerate controls for video/animation formats.
#   ✓ Dynamic filename templating (e.g., 'render_%Y-%m-%d').
#   ✓ Robust timestamp-based saving to prevent file overwrites (AAPS method).
#   ✓ Smart Sidecar: Automatically saves workflow to a .json sidecar file
#     if it's too large for EXIF (e.g., > 256KB).

# ░▒▓ CHANGELOG:
#   - v1.4.4 (Sidecar Update - Oct 2025):
#       • FIXED: Implemented sidecar logic from AdvancedAudioSave.
#       • ROBUST: Node now checks metadata size *before* saving.
#       • ROBUST: If metadata > 256KB, workflow is saved to a .json
#         sidecar file, preventing "EXIF data is too long" crash.
#   - v1.4.2 (Guideline Update - Oct 2025):
#       • REFACTOR: Full compliance update to v1.4.2 guidelines.
#       • CRITICAL: Removed all type hints from function signatures.
#       • STYLE: Standardized imports, docstrings, error handling, logging.
#       • STYLE: Rewrote all tooltips to new standard format.
#       • CACHE: Added IS_CHANGED method to ensure node always runs (saver node).
#       • ROBUST: Added try/except blocks to helper save functions.
#       • STYLE: Updated category and display name.
#   - v1.0.6 (Overwriting Fix):
#       • FIXED: Reworked save logic to mirror AAPS timestamp method, resolving overwrites.
#   - v1.0.0-v1.0.5 (Base):
#       • ADDED: Initial release with multi-format support, metadata controls.

# ░▒▓ CONFIGURATION:
#   → Primary Use: Saving final PNGs or GIFs with workflow metadata embedded.
#   → Secondary Use: Saving MP4/WEBM videos with specific quality/framerate.
#   → Edge Use: Using "Save Metadata" toggle for privacy control when sharing images.

# ░▒▓ WARNING:
#   This node may trigger:
#   ▓▒░ The digital ghost of that .IFF file you overwrote on 'WORK:3' back in '92.
#   ▓▒░ A paranoid, compulsive need to toggle the "Save Metadata" switch... just in case.
#   ▓▒░ Realizing your 120-frame MP4 is just a 10-second loop of a 256-byte intro.
#   ▓▒░ A sudden, deep appreciation for a simple, un-incremented, timestamped filename.
#   Consult your nearest demoscene vet if hallucinations persist.
# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄


# =================================================================================
# == Standard Library Imports
# =================================================================================
import json
import logging
import os
import secrets # For IS_CHANGED
import time
import traceback

# =================================================================================
# == Third-Party Imports
# =================================================================================
import numpy as np
import piexif
import torch
from PIL import Image
from PIL.PngImagePlugin import PngInfo

# Video/Animation library (Optional)
try:
    import imageio.v2 as imageio
    _imageio_available = True
except ImportError:
    logging.warning("[AdvancedMediaSave] imageio not found. Saving animations (GIF, MP4, WEBM) will be disabled.")
    _imageio_available = False

# =================================================================================
# == ComfyUI Core Modules
# =================================================================================
from comfy.cli_args import args
import folder_paths

# =================================================================================
# == Configuration & Setup
# =================================================================================
MEDIA_OUTPUT_DIR = os.path.join(folder_paths.get_output_directory(), "ComfyUI_AdvancedMediaOutputs")
os.makedirs(MEDIA_OUTPUT_DIR, exist_ok=True)

# Metadata size limit in KB - JPEG/WEBP files larger than this will use sidecar JSON
METADATA_SIZE_LIMIT_KB: int = 256

# =================================================================================
# == Helper Function (from AdvancedAudioSave)
# =================================================================================

def save_metadata_sidecar(media_filepath, full_prompt_api_dict, custom_notes=None):
    """
    Saves metadata as a .json sidecar file in ComfyUI's expected export format.
    Requires the full prompt API dictionary.
    
    Args:
        media_filepath (str): Path to the saved media file (image, audio, etc.).
        full_prompt_api_dict (dict): The full "prompt" dict containing workflow data.
        custom_notes (str, optional): Additional notes to embed.
        
    Returns:
        bool: True on success, False on failure.
    """
    json_path = os.path.splitext(media_filepath)[0] + ".json"
    if not full_prompt_api_dict:
        logging.error("[AdvancedMediaSave] ERROR: Cannot save sidecar JSON, full workflow data is missing.")
        return False
    
    try:
        # Try to get workflow from extra_pnginfo first (standard location)
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
            logging.error("[AdvancedMediaSave] ERROR: Could not find workflow data in full_prompt_api_dict")
            return False
        
        # workflow_data should now be a dict with the ComfyUI structure
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
        
        logging.info(f"[AdvancedMediaSave] ✓ Sidecar saved (ComfyUI Export Format): {json_path}")
        return True
        
    except Exception as e:
        logging.error(f"[AdvancedMediaSave] ✗ FAILED to save metadata sidecar: {e}")
        logging.debug(traceback.format_exc())
        return False

# =================================================================================
# == Core Node Class
# =================================================================================

class AdvancedMediaSave:
    """
    Saves images or animations in various formats with metadata control.
    Uses timestamped filenames inspired by AAPS to prevent overwrites.
    """
    OUTPUT_NODE = True # This is a terminal node

    @classmethod
    def INPUT_TYPES(cls):
        """Define input parameters with standardized tooltips."""
        IMAGE_FORMATS = ["PNG", "JPEG", "WEBP"]
        ANIMATION_FORMATS = []
        if _imageio_available:
            ANIMATION_FORMATS = ["GIF (from batch)", "MP4 (from batch)", "WEBM (from batch)"]
        else:
            logging.warning("[AdvancedMediaSave] imageio not available, animation formats disabled in input types.")

        ALL_FORMATS = IMAGE_FORMATS + ANIMATION_FORMATS

        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": (
                        "IMAGE INPUT\n"
                        "- The image or batch of images to save.\n"
                        "- Expects standard ComfyUI IMAGE tensor format: [batch, height, width, 3]."
                    )
                }),
                "filename_prefix": ("STRING", {
                    "default": "AMS_%Y-%m-%d",
                    "tooltip": (
                        "FILENAME PREFIX\n"
                        "- Prefix for the saved file(s).\n"
                        "- Supports strftime codes (e.g., %Y, %m, %d, %H, %M, %S) for dynamic naming.\n"
                        "- Example: 'renders/projectX_%Y%m%d' will save to 'ComfyUI_AdvancedMediaOutputs/renders/projectX_...'."
                    )
                }),
                "save_format": (ALL_FORMATS, {
                    "default": "PNG",
                    "tooltip": (
                        "SAVE FORMAT\n"
                        "- Choose the output file format.\n"
                        "- Image formats (PNG, JPEG, WEBP) save each image in the batch individually.\n"
                        "- Animation formats (GIF, MP4, WEBM) save the entire batch as one animation file."
                    )
                }),
            },
            "optional": {
                "save_metadata": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "SAVE METADATA\n"
                        "- If True, embeds workflow, prompt, and custom notes into the media file (format permitting).\n"
                        "- If False (Privacy Filter), metadata is stripped.\n"
                        "- respects ComfyUI's global '--disable-metadata' flag."
                    )
                }),
                "custom_notes": ("STRING", {
                    "default": "", "multiline": True,
                    "tooltip": (
                        "CUSTOM NOTES\n"
                        "- Add custom text notes to the file's metadata.\n"
                        "- Only embedded if 'Save Metadata' is True."
                    )
                }),
                "jpeg_quality": ("INT", {
                    "default": 95, "min": 1, "max": 100, "step": 1,
                    "tooltip": (
                        "JPEG QUALITY\n"
                        "- Quality setting for JPEG format (1-100).\n"
                        "- Higher = better quality, larger file size."
                    )
                }),
                "webp_quality": ("INT", {
                    "default": 90, "min": 1, "max": 100, "step": 1,
                    "tooltip": (
                        "WEBP QUALITY\n"
                        "- Quality setting for lossy WEBP format (1-100).\n"
                        "- Higher = better quality, larger file size.\n"
                        "- Ignored if 'WEBP Lossless' is True."
                    )
                }),
                "webp_lossless": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "WEBP LOSSLESS\n"
                        "- If True, saves WEBP images losslessly (perfect quality, larger size).\n"
                        "- If False, uses the 'WEBP Quality' setting for lossy compression."
                    )
                }),
                "framerate": ("FLOAT", {
                    "default": 8.0, "min": 0.1, "max": 60.0, "step": 0.1,
                    "tooltip": (
                        "FRAMERATE (FPS)\n"
                        "- Frames per second for animation formats (GIF, MP4, WEBM).\n"
                        "- Higher = smoother animation."
                    )
                }),
                "video_quality": ("INT", {
                    "default": 8, "min": 1, "max": 10, "step": 1,
                    "tooltip": (
                        "VIDEO QUALITY (MP4/WEBM)\n"
                        "- Quality setting for MP4 (H.264) and WEBM (VP9) formats (1-10).\n"
                        "- Higher = better quality, larger file size.\n"
                        "- Note: imageio uses a 1-10 scale internally."
                    )
                }),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"}, # Standard hidden inputs for metadata
        }

    RETURN_TYPES = () # Output nodes have empty return types
    FUNCTION = "save_media"
    CATEGORY = "MD_Nodes/Save"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """
        Force node to re-run every time. Saver nodes should not be cached.
        """
        # Always return a unique value to ensure execution.
        return secrets.token_hex(16)

    def save_media(self, images, filename_prefix, save_format, save_metadata, custom_notes,
                   jpeg_quality, webp_quality, webp_lossless, framerate, video_quality,
                   prompt=None, extra_pnginfo=None):
        """
        Main execution function to process images and save them.
        """
        results = []
        try:
            # --- 1. Prepare Paths, Prefix, and Metadata (AAPS Method) ---
            try:
                # Process strftime codes in the prefix
                base_prefix = time.strftime(os.path.basename(filename_prefix), time.localtime())
                subfolder_prefix = os.path.dirname(filename_prefix)
            except ValueError:
                logging.warning(f"[AdvancedMediaSave] Invalid strftime format in prefix '{filename_prefix}'. Using as-is.")
                base_prefix = os.path.basename(filename_prefix)
                subfolder_prefix = os.path.dirname(filename_prefix)

            output_dir_local = os.path.join(MEDIA_OUTPUT_DIR, subfolder_prefix)
            os.makedirs(output_dir_local, exist_ok=True)
            logging.info(f"[AdvancedMediaSave] Saving to directory: {output_dir_local}")

            timestamp = int(time.time()) # Used for unique filenames

            metadata = {}
            # Respect both node setting and global ComfyUI flag
            should_save_metadata = save_metadata and not args.disable_metadata
            if should_save_metadata:
                if prompt is not None: metadata["prompt"] = json.dumps(prompt)
                if extra_pnginfo is not None and "workflow" in extra_pnginfo:
                    metadata["workflow"] = json.dumps(extra_pnginfo["workflow"])
                if custom_notes: metadata['notes'] = custom_notes
                logging.info("[AdvancedMediaSave] Metadata embedding enabled.")
            else:
                logging.info("[AdvancedMediaSave] Metadata embedding disabled (by node setting or global flag).")

            # --- 2. Process and Save Media ---
            # Convert tensors to PIL Images
            pil_images = []
            for i in images:
                img_np = np.clip(255. * i.cpu().numpy(), 0, 255).astype(np.uint8)
                pil_images.append(Image.fromarray(img_np))

            is_animation_format = "batch" in save_format
            logging.info(f"[AdvancedMediaSave] Processing {len(pil_images)} image(s) for format: {save_format}")

            if is_animation_format:
                if len(pil_images) > 1:
                    # Extract format like 'gif', 'mp4', 'webm'
                    file_ext = save_format.split(" ")[0].lower()
                    result = self._save_animation(pil_images, output_dir_local, base_prefix, timestamp, file_ext, framerate, video_quality, metadata)
                    if result: results.append(result)
                else:
                    logging.warning("[AdvancedMediaSave] Only one frame provided for animation format. Saving as PNG instead.")
                    save_format = "PNG" # Force fallback to static image
                    is_animation_format = False # Update flag

            # Handle static image saving (either originally selected or fallback from animation)
            if not is_animation_format:
                file_ext = save_format.lower()
                # *** MODIFIED: Pass prompt, extra_pnginfo, and custom_notes for sidecar logic ***
                batch_results = self._save_static_images(
                    pil_images, output_dir_local, base_prefix, timestamp, file_ext, 
                    jpeg_quality, webp_quality, webp_lossless, 
                    metadata, prompt, extra_pnginfo, custom_notes
                )
                results.extend(batch_results)

        except Exception as e:
            logging.error(f"[AdvancedMediaSave] Failed during save preparation: {e}")
            logging.debug(traceback.format_exc())
            # Fall through to UI output preparation, which will show failure

        # --- 3. Prepare UI Output ---
        ui_text = []
        if results:
            first_file = results[0]
            # Get relative path for cleaner UI display
            try:
                subfolder_rel = os.path.relpath(os.path.dirname(first_file["filepath"]), folder_paths.get_output_directory())
            except ValueError: # Handle cases where paths are on different drives (Windows)
                subfolder_rel = os.path.dirname(first_file["filepath"]) # Show absolute path as fallback

            ui_text.append(f"✅ Saved {len(results)} file(s) to '{subfolder_rel}'")
            ui_text.append(f"First file: {first_file['filename']}")
            
            # *** ADDED: Check for and report sidecar saves ***
            sidecar_count = sum(1 for f in results if f.get('sidecar_saved'))
            if sidecar_count > 0:
                ui_text.append(f"✓ Saved {sidecar_count} sidecar .json(s) for large workflows.")

            logging.info(f"[AdvancedMediaSave] Successfully saved {len(results)} file(s). First: {first_file['filename']}")
        else:
            ui_text.append("❌ Save failed or was skipped. Check console/logs.")
            logging.error("[AdvancedMediaSave] Save operation resulted in no files saved.")

        # Return format expected by ComfyUI frontend
        return {"ui": {"text": ui_text}}

    def _save_static_images(self, pil_images, output_dir, base_prefix, timestamp, ext, 
                            jpeg_quality, webp_quality, webp_lossless, 
                            metadata, prompt, extra_pnginfo, custom_notes):
        """
        Helper to save a batch of images individually.
        Includes sidecar logic for large metadata.
        """
        saved_files = []
        num_images = len(pil_images)
        
        for i, img in enumerate(pil_images):
            # Generate unique filename using timestamp and index
            zfill_count = max(3, len(str(num_images)))
            filename = f"{base_prefix}_{timestamp}_{i+1:0{zfill_count}d}.{ext}"
            filepath = os.path.join(output_dir, filename)

            metadata_to_embed = metadata.copy()
            sidecar_saved = False

            # --- Sidecar Logic ---
            if metadata and ext in ['jpeg', 'jpg', 'webp']:
                try:
                    metadata_json_str = json.dumps(metadata)
                    metadata_size_kb = len(metadata_json_str.encode('utf-8')) / 1024

                    if metadata_size_kb > METADATA_SIZE_LIMIT_KB:
                        logging.warning(f"[AdvancedMediaSave] Metadata ({metadata_size_kb:.1f}KB) exceeds limit for {filename}. Saving sidecar .json.")
                        
                        full_prompt_api_dict = {'prompt': prompt, 'extra_pnginfo': extra_pnginfo}
                        sidecar_saved = save_metadata_sidecar(filepath, full_prompt_api_dict, custom_notes)
                        
                        if not sidecar_saved:
                            logging.warning(f"[AdvancedMediaSave] Failed to save sidecar for {filename}.")
                        
                        # Only embed notes in the image, or nothing
                        metadata_to_embed = {}
                        if custom_notes:
                            metadata_to_embed['notes'] = custom_notes
                            
                except Exception as e:
                    logging.warning(f"[AdvancedMediaSave] Failed to check metadata size or save sidecar: {e}")
                    # Proceed with full metadata and risk failure
                    metadata_to_embed = metadata

            # --- Prepare Save Parameters ---
            save_params = {}
            if ext == 'png':
                png_info = PngInfo()
                if metadata_to_embed:
                    for k, v in metadata_to_embed.items():
                        png_info.add_text(k, str(v)) # PNG-Info handles large data well
                save_params['pnginfo'] = png_info
                logging.debug(f"[AdvancedMediaSave] Saving PNG: {filepath} with metadata: {bool(metadata_to_embed)}")
            
            elif ext in ['jpeg', 'jpg', 'webp']:
                exif_bytes = b''
                if metadata_to_embed: # This is now the *small* metadata if sidecar was used
                    try:
                        user_comment_data = json.dumps(metadata_to_embed)
                        exif_dict = {"Exif": {
                            piexif.ExifIFD.UserComment: b"UNICODE\x00" + user_comment_data.encode("utf-16-be")
                        }}
                        exif_bytes = piexif.dump(exif_dict)
                    except Exception as exif_e:
                        logging.warning(f"[AdvancedMediaSave] Failed to encode metadata for EXIF: {exif_e}")
                
                save_params['exif'] = exif_bytes
                if ext in ['jpeg', 'jpg']:
                    save_params['quality'] = jpeg_quality
                else: # webp
                    save_params['quality'] = webp_quality
                    save_params['lossless'] = webp_lossless
                logging.debug(f"[AdvancedMediaSave] Saving {ext.upper()}: {filepath} with quality={save_params.get('quality', 'N/A')}, lossless={save_params.get('lossless', 'N/A')}, metadata: {bool(metadata_to_embed)}")

            # --- Execute Save ---
            try:
                img.save(filepath, **save_params)
                
                saved_files.append({
                    "filename": filename,
                    "filepath": filepath,
                    "subfolder": os.path.basename(output_dir),
                    "type": "output",
                    "sidecar_saved": sidecar_saved # Report status
                })

            except Exception as e:
                # Handle "data too long" error *even* with sidecar, in case notes are too long
                if "EXIF data is too long" in str(e) and ext in ['jpeg', 'jpg', 'webp']:
                    logging.warning(f"[AdvancedMediaSave] Metadata was still too long for EXIF (even after sidecar). Saving {filename} without any metadata.")
                    try:
                        save_params.pop('exif', None) # Remove problematic key
                        img.save(filepath, **save_params) # Try again
                        
                        saved_files.append({
                            "filename": filename,
                            "filepath": filepath,
                            "subfolder": os.path.basename(output_dir),
                            "type": "output",
                            "sidecar_saved": sidecar_saved
                        })
                    except Exception as e2:
                        logging.error(f"[AdvancedMediaSave] Failed to save {filename} even after removing metadata: {e2}")
                        logging.debug(traceback.format_exc())
                else:
                    logging.error(f"[AdvancedMediaSave] Failed to save static image {filename}: {e}")
                    logging.debug(traceback.format_exc())
            
            # Continue to the next image in the batch

        return saved_files

    def _save_animation(self, pil_images, output_dir, base_prefix, timestamp, ext, framerate, video_quality, metadata):
        """
        Helper to save a batch of images as an animation (GIF, MP4, WEBM).
        Note: Sidecar logic is NOT implemented for animations yet.
        """
        if not _imageio_available:
            logging.error("[AdvancedMediaSave] Cannot save animation: imageio library is not installed.")
            return None

        filename = f"{base_prefix}_{timestamp}.{ext}"
        filepath = os.path.join(output_dir, filename)
        logging.debug(f"[AdvancedMediaSave] Saving animation: {filepath} (FPS: {framerate}, Quality: {video_quality}, Format: {ext})")

        # TODO: Implement sidecar logic for animations if metadata_str is too large
        metadata_str = json.dumps(metadata) if metadata else ""

        try:
            if ext == 'gif':
                imageio.mimsave(filepath, pil_images, duration=(1000 / framerate), loop=0)
                logging.debug(f"[AdvancedMediaSave] Saved GIF: {filepath}")
            elif ext == 'mp4':
                writer = imageio.get_writer(filepath, fps=framerate, codec='libx264', quality=video_quality,
                                            ffmpeg_params=['-metadata', f'comment={metadata_str}'])
                for img in pil_images: writer.append_data(np.array(img))
                writer.close()
                logging.debug(f"[AdvancedMediaSave] Saved MP4: {filepath} with metadata: {bool(metadata)}")
            elif ext == 'webm':
                writer = imageio.get_writer(filepath, fps=framerate, codec='libvpx-vp9', quality=video_quality,
                                            ffmpeg_params=['-metadata', f'comment={metadata_str}'])
                for img in pil_images: writer.append_data(np.array(img))
                writer.close()
                logging.debug(f"[AdvancedMediaSave] Saved WEBM: {filepath} with metadata: {bool(metadata)}")
            else:
                logging.error(f"[AdvancedMediaSave] Unsupported animation format requested: {ext}")
                return None

            return {
                "filename": filename,
                "filepath": filepath,
                "subfolder": os.path.basename(output_dir),
                "type": "output",
                "sidecar_saved": False # Sidecar not implemented for video yet
            }
        except Exception as e:
            logging.error(f"[AdvancedMediaSave] ERROR saving animation to {filepath}: {e}")
            logging.debug(traceback.format_exc())
            return None # Indicate failure

# =================================================================================
# == Node Registration
# =================================================================================

NODE_CLASS_MAPPINGS = {
    "AdvancedMediaSave": AdvancedMediaSave
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AdvancedMediaSave": "MD: Advanced Media Save"
}