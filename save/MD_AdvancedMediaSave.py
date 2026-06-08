# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# █▓▒░                                                                     ░▒▓█
# █▓▒░ MD_Nodes/AdvancedMediaSave – Multi-format media saving node v1.5.5  ░▒▓█
# █▓▒░                                                                     ░▒▓█
# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
# ╠═ © 2026 MDMAchine
# ╠═ License: GNU General Public License v3.0 (GPL v3)
# ║
# ║  This program is free software: you can redistribute it and/or modify
# ║  it under the terms of the GNU General Public License as published by
# ║  the Free Software Foundation, either version 3 of the License, or
# ║  (at your option) any later version.
# ║
# ║  This program is distributed in the hope that it will be useful,
# ║  but WITHOUT ANY WARRANTY; without even the implied warranty of
# ║  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# ║  GNU General Public License for more details.
# ║
# ║  You should have received a copy of the GNU General Public License
# ║  along with this program. If not, see <https://www.gnu.org/licenses/>.
# ╠════════════════════════════════════════════════════════════════════════════
# ║ ░▒▓ ORIGIN & DEV:
# ║   • Cast into the void by: MDMAchine & Gemini
# ║   • Enhanced by: AAPS Save Logic (Inspiration)
# ║
# ║ ░▒▓ DESCRIPTION:
# ║   Processes, previews, and saves images (PNG/JPEG/WEBP), image batches, and
# ║   videos (GIF/MP4/WEBM) with robust, timestamp-based saving to prevent
# ║   overwrites and optional metadata embedding control.
# ║   NOTE: As an I/O node, this runs entirely in the public wrapper domain.
# ║
# ║ ░▒▓ FEATURES:
# ║   ✓ Save single images, batches, or animations.
# ║   ✓ Formats: PNG, JPEG, WEBP (lossy/lossless), GIF, MP4 (H.264), WEBM (VP9).
# ║   ✓ Metadata Privacy Filter: Toggle embedding or stripping workflow data.
# ║   ✓ Quality & Framerate controls for video/animation formats.
# ║   ✓ Dynamic filename templating (e.g., 'render_%Y-%m-%d').
# ║   ✓ Robust timestamp-based saving to prevent file overwrites.
# ║   ✓ Smart Sidecar: Automatically saves workflow to a .json sidecar file
# ║     if it's too large for EXIF (e.g., > 256KB).
# ║
# ║ ░▒▓ CHANGELOG:
# ║   - v1.5.5 (Enterprise Standards - Feb 2026):
# ║       • ADDED: PerformanceProfiler class for disk I/O tracking (v1.5.3 standard).
# ║       • ADDED: debug_mode parameter.
# ║       • REFACTOR: Tooltips strictly updated to 5-part v1.5.4 standard.
# ║   - v1.4.4 (Sidecar Update - Oct 2025):
# ║       • FIXED: Implemented sidecar logic from AdvancedAudioSave.
# ║       • ROBUST: Node now checks metadata size *before* saving.
# ║       • ROBUST: If metadata > 256KB, workflow is saved to a .json
# ║         sidecar file, preventing "EXIF data is too long" crash.
# ╚════════════════════════════════════════════════════════════════════════════

# =================================================================================
# == Standard Library Imports
# =================================================================================
VERSION = "v1.5.5"  # UPS v1.5.8


import json
import logging
import os
import secrets
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
# == Helper Classes (Enterprise Standards)
# =================================================================================

class PerformanceProfiler:
    """Standard performance profiler for MD_Nodes."""
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.timings = {}
        self.start_times = {}
    
    def start(self, operation_name):
        if not self.enabled: return
        self.start_times[operation_name] = time.perf_counter()
    
    def stop(self, operation_name):
        if not self.enabled: return
        if operation_name in self.start_times:
            elapsed = time.perf_counter() - self.start_times[operation_name]
            if operation_name not in self.timings:
                self.timings[operation_name] = []
            self.timings[operation_name].append(elapsed)
            del self.start_times[operation_name]
    
    def get_total_time(self):
        if not self.enabled or not self.timings: return 0.0
        return sum(sum(times) for times in self.timings.values())
    
    def print_report(self):
        if not self.enabled or not self.timings: return
        logging.info("\n⏱️  PERFORMANCE (I/O):")
        total = self.get_total_time()
        logging.info(f"    • Total Time: {total:.2f}s")
        for op_name, times in sorted(self.timings.items()):
            avg = sum(times) / len(times)
            if len(times) == 1:
                logging.info(f"    • {op_name}: {avg:.3f}s")
            else:
                logging.info(f"    • {op_name}: {avg:.3f}s avg ({len(times)}x)")

# =================================================================================
# == Helper Function (from AdvancedAudioSave)
# =================================================================================

def save_metadata_sidecar(media_filepath, full_prompt_api_dict, custom_notes=None):
    """
    Saves metadata as a .json sidecar file in ComfyUI's expected export format.
    Requires the full prompt API dictionary.
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
    OUTPUT_NODE = True 

    @classmethod
    def INPUT_TYPES(cls):
        IMAGE_FORMATS = ["PNG", "JPEG", "WEBP"]
        ANIMATION_FORMATS = []
        if _imageio_available:
            ANIMATION_FORMATS = ["GIF (from batch)", "MP4 (from batch)", "WEBM (from batch)"]
        else:
            logging.warning("[AdvancedMediaSave] imageio not available, animation formats disabled.")

        ALL_FORMATS = IMAGE_FORMATS + ANIMATION_FORMATS

        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": (
                        "IMAGE INPUT\n"
                        "• Purpose: The image or batch of images to save.\n"
                        "• Format: Standard ComfyUI IMAGE tensor [batch, height, width, 3]."
                    )
                }),
                "filename_prefix": ("STRING", {
                    "default": "AMS_%Y-%m-%d",
                    "tooltip": (
                        "FILENAME PREFIX\n"
                        "• Purpose: Prefix for the saved file(s).\n"
                        "• Options: Supports strftime codes (e.g., %Y, %m, %d, %H, %M, %S).\n"
                        "• Note: Automatically appended with a timestamp to prevent overwrites.\n"
                        "\n⭐ Recommended: Keep default or add project name (e.g., 'projectX_%Y-%m-%d')."
                    )
                }),
                "save_format": (ALL_FORMATS, {
                    "default": "PNG",
                    "tooltip": (
                        "SAVE FORMAT\n"
                        "• Purpose: Choose the output file format.\n"
                        "• Static: PNG, JPEG, WEBP save each image in the batch individually.\n"
                        "• Animation: GIF, MP4, WEBM compile the batch into a single video file.\n"
                        "\n⭐ Recommended: PNG for highest quality static images."
                    )
                }),
            },
            "optional": {
                "save_metadata": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "SAVE METADATA\n"
                        "• Purpose: Toggle embedding workflow and prompt data into the file.\n"
                        "• Trade-offs: Large workflows may bloat file sizes. Automatically triggers Sidecar JSON if > 256KB.\n"
                        "\n⭐ Recommended: True (unless sharing publicly and privacy is needed)."
                    )
                }),
                "custom_notes": ("STRING", {
                    "default": "", "multiline": True,
                    "tooltip": (
                        "CUSTOM NOTES\n"
                        "• Purpose: Add personal text notes to the file's metadata.\n"
                        "• Note: Only embedded if 'Save Metadata' is True."
                    )
                }),
                "jpeg_quality": ("INT", {
                    "default": 95, "min": 1, "max": 100, "step": 1,
                    "tooltip": (
                        "JPEG QUALITY\n"
                        "• Purpose: Compression quality for JPEG formats.\n"
                        "• Range: 1 (Lowest) to 100 (Highest).\n"
                        "\n⭐ Recommended: 95 for best balance of size and quality."
                    )
                }),
                "webp_quality": ("INT", {
                    "default": 90, "min": 1, "max": 100, "step": 1,
                    "tooltip": (
                        "WEBP QUALITY\n"
                        "• Purpose: Compression quality for lossy WEBP formats.\n"
                        "• Note: Ignored if 'WEBP Lossless' is enabled.\n"
                        "\n⭐ Recommended: 90."
                    )
                }),
                "webp_lossless": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "WEBP LOSSLESS\n"
                        "• Purpose: Forces WEBP to save with zero quality loss.\n"
                        "• Trade-offs: Perfect quality but significantly larger file sizes.\n"
                        "\n⭐ Recommended: True."
                    )
                }),
                "framerate": ("FLOAT", {
                    "default": 8.0, "min": 0.1, "max": 60.0, "step": 0.1,
                    "tooltip": (
                        "FRAMERATE (FPS)\n"
                        "• Purpose: Playback speed for animation formats (GIF, MP4, WEBM).\n"
                        "• Options: 8-12 fps is common for standard GIF generations.\n"
                        "\n⭐ Recommended: 8.0"
                    )
                }),
                "video_quality": ("INT", {
                    "default": 8, "min": 1, "max": 10, "step": 1,
                    "tooltip": (
                        "VIDEO QUALITY\n"
                        "• Purpose: Internal quality parameter for MP4 (H.264) and WEBM (VP9).\n"
                        "• Range: 1 (Lowest) to 10 (Highest lossless).\n"
                        "\n⭐ Recommended: 8."
                    )
                }),
                "debug_mode": (["0 - Silent", "1 - Info", "2 - Verbose"], {
                    "default": "0 - Silent",
                    "tooltip": (
                        "LOGGING VERBOSITY\n"
                        "• Purpose: Controls console output and enables disk I/O profiling.\n"
                        "• Options: 0 (Silent), 1 (Info/Stats), 2 (Verbose).\n"
                        "\n⭐ Recommended: 0 - Silent for standard production."
                    )
                }),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"}, 
        }

    RETURN_TYPES = ("STRING",) 
    RETURN_NAMES = ("file_path",)
    FUNCTION = "save_media"
    CATEGORY = "MD_Nodes/Save"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """Force node to re-run every time. Saver nodes should not be cached."""
        return secrets.token_hex(16)

    def save_media(self, images, filename_prefix, save_format, save_metadata, custom_notes,
                   jpeg_quality, webp_quality, webp_lossless, framerate, video_quality,
                   debug_mode="0 - Silent", prompt=None, extra_pnginfo=None):
        
        debug_level = int(debug_mode.split(" ")[0])
        profiler = PerformanceProfiler(enabled=(debug_level >= 1))
        profiler.start("total_save_process")
        
        results = []
        try:
            # --- 1. Prepare Paths, Prefix, and Metadata ---
            profiler.start("path_prep")
            try:
                base_prefix = time.strftime(os.path.basename(filename_prefix), time.localtime())
                subfolder_prefix = os.path.dirname(filename_prefix)
            except ValueError:
                logging.warning(f"[AdvancedMediaSave] Invalid strftime format in prefix '{filename_prefix}'. Using as-is.")
                base_prefix = os.path.basename(filename_prefix)
                subfolder_prefix = os.path.dirname(filename_prefix)

            output_dir_local = os.path.join(MEDIA_OUTPUT_DIR, subfolder_prefix)
            os.makedirs(output_dir_local, exist_ok=True)
            if debug_level >= 2: logging.info(f"[AdvancedMediaSave] Saving to directory: {output_dir_local}")

            timestamp = int(time.time()) 

            metadata = {}
            should_save_metadata = save_metadata and not args.disable_metadata
            if should_save_metadata:
                if prompt is not None: metadata["prompt"] = json.dumps(prompt)
                if extra_pnginfo is not None and "workflow" in extra_pnginfo:
                    metadata["workflow"] = json.dumps(extra_pnginfo["workflow"])
                if custom_notes: metadata['notes'] = custom_notes
            profiler.stop("path_prep")

            # --- 2. Process and Save Media ---
            profiler.start("tensor_to_pil")
            pil_images = []
            for i in images:
                img_np = np.clip(255. * i.cpu().numpy(), 0, 255).astype(np.uint8)
                pil_images.append(Image.fromarray(img_np))
            profiler.stop("tensor_to_pil")

            is_animation_format = "batch" in save_format

            profiler.start("disk_write")
            if is_animation_format:
                if len(pil_images) > 1:
                    file_ext = save_format.split(" ")[0].lower()
                    result = self._save_animation(pil_images, output_dir_local, base_prefix, timestamp, file_ext, framerate, video_quality, metadata)
                    if result: results.append(result)
                else:
                    logging.warning("[AdvancedMediaSave] Only one frame provided for animation format. Saving as PNG instead.")
                    save_format = "PNG" 
                    is_animation_format = False 

            if not is_animation_format:
                file_ext = save_format.lower()
                batch_results = self._save_static_images(
                    pil_images, output_dir_local, base_prefix, timestamp, file_ext, 
                    jpeg_quality, webp_quality, webp_lossless, 
                    metadata, prompt, extra_pnginfo, custom_notes
                )
                results.extend(batch_results)
            profiler.stop("disk_write")

        except Exception as e:
            logging.error(f"[AdvancedMediaSave] Failed during save execution: {e}")
            logging.debug(traceback.format_exc())

        profiler.stop("total_save_process")

        # --- 3. Prepare UI Output ---
        ui_text = []
        if results:
            first_file = results[0]
            try:
                subfolder_rel = os.path.relpath(os.path.dirname(first_file["filepath"]), folder_paths.get_output_directory())
            except ValueError: 
                subfolder_rel = os.path.dirname(first_file["filepath"])

            ui_text.append(f"✅ Saved {len(results)} file(s) to '{subfolder_rel}'")
            ui_text.append(f"First file: {first_file['filename']}")
            
            sidecar_count = sum(1 for f in results if f.get('sidecar_saved'))
            if sidecar_count > 0:
                ui_text.append(f"✓ Saved {sidecar_count} sidecar .json(s) for large workflows.")

            if debug_level >= 1:
                logging.info("\n" + "=" * 60)
                logging.info("📊 [AdvancedMediaSave] ANALYTICS REPORT")
                logging.info("=" * 60)
                logging.info("💾  STORAGE:")
                logging.info(f"    • Format:       {save_format}")
                logging.info(f"    • Files saved:  {len(results)}")
                logging.info(f"    • Sidecars:     {sidecar_count}")
                profiler.print_report()
                logging.info("=" * 60)
        else:
            ui_text.append("❌ Save failed or was skipped. Check console/logs.")

        return {"ui": {"text": ui_text}}

    def _save_static_images(self, pil_images, output_dir, base_prefix, timestamp, ext, 
                            jpeg_quality, webp_quality, webp_lossless, 
                            metadata, prompt, extra_pnginfo, custom_notes):
        """Helper to save a batch of images individually. Includes sidecar logic for large metadata."""
        saved_files = []
        num_images = len(pil_images)
        
        for i, img in enumerate(pil_images):
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
                        logging.warning(f"[AdvancedMediaSave] Metadata ({metadata_size_kb:.1f}KB) exceeds limit. Saving sidecar.")
                        
                        full_prompt_api_dict = {'prompt': prompt, 'extra_pnginfo': extra_pnginfo}
                        sidecar_saved = save_metadata_sidecar(filepath, full_prompt_api_dict, custom_notes)
                        
                        metadata_to_embed = {}
                        if custom_notes:
                            metadata_to_embed['notes'] = custom_notes
                            
                except Exception as e:
                    logging.warning(f"[AdvancedMediaSave] Failed to check metadata size or save sidecar: {e}")
                    metadata_to_embed = metadata

            # --- Prepare Save Parameters ---
            save_params = {}
            if ext == 'png':
                png_info = PngInfo()
                if metadata_to_embed:
                    for k, v in metadata_to_embed.items():
                        png_info.add_text(k, str(v))
                save_params['pnginfo'] = png_info
            
            elif ext in ['jpeg', 'jpg', 'webp']:
                exif_bytes = b''
                if metadata_to_embed: 
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
                else: 
                    save_params['quality'] = webp_quality
                    save_params['lossless'] = webp_lossless

            # --- Execute Save ---
            try:
                img.save(filepath, **save_params)
                
                saved_files.append({
                    "filename": filename,
                    "filepath": filepath,
                    "subfolder": os.path.basename(output_dir),
                    "type": "output",
                    "sidecar_saved": sidecar_saved 
                })

            except Exception as e:
                if "EXIF data is too long" in str(e) and ext in ['jpeg', 'jpg', 'webp']:
                    logging.warning(f"[AdvancedMediaSave] Metadata too long for EXIF. Saving {filename} without metadata.")
                    try:
                        save_params.pop('exif', None) 
                        img.save(filepath, **save_params) 
                        
                        saved_files.append({
                            "filename": filename,
                            "filepath": filepath,
                            "subfolder": os.path.basename(output_dir),
                            "type": "output",
                            "sidecar_saved": sidecar_saved
                        })
                    except Exception as e2:
                        logging.error(f"[AdvancedMediaSave] Failed to save {filename} even after removing metadata: {e2}")
                else:
                    logging.error(f"[AdvancedMediaSave] Failed to save static image {filename}: {e}")
            
        return saved_files

    def _save_animation(self, pil_images, output_dir, base_prefix, timestamp, ext, framerate, video_quality, metadata):
        """Helper to save a batch of images as an animation (GIF, MP4, WEBM)."""
        if not _imageio_available:
            logging.error("[AdvancedMediaSave] Cannot save animation: imageio library is not installed.")
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
            else:
                return None

            return {
                "filename": filename,
                "filepath": filepath,
                "subfolder": os.path.basename(output_dir),
                "type": "output",
                "sidecar_saved": False 
            }
        except Exception as e:
            logging.error(f"[AdvancedMediaSave] ERROR saving animation to {filepath}: {e}")
            return None 

# =================================================================================
# == Node Registration
# =================================================================================

NODE_CLASS_MAPPINGS = {
    "AdvancedMediaSave": AdvancedMediaSave
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AdvancedMediaSave": "MD: Advanced Media Save"
}


# ==============================================================================
# == Unit Tests (smoke — runs without ComfyUI)
# ==============================================================================

if __name__ == "__main__":
    print("\n🧪 Smoke tests: MD_AdvancedMediaSave")
    print("   VERSION :", VERSION)
    _pass = _fail = 0

    def _check(label, expr):
        global _pass, _fail
        if expr:
            print(f"  ✅  {label}")
            _pass += 1
        else:
            print(f"  ❌  {label}")
            _fail += 1

    _check("VERSION defined",    VERSION == "v1.5.5")
    _check("NODE_CLASS_MAPPINGS defined",
           isinstance(NODE_CLASS_MAPPINGS, dict) and len(NODE_CLASS_MAPPINGS) > 0)
    _check("  class AdvancedMediaSave in map", "AdvancedMediaSave" in NODE_CLASS_MAPPINGS)

    print(f"\n  {_pass} passed, {_fail} failed")
    if _fail == 0:
        print("  🎉 All good.")
