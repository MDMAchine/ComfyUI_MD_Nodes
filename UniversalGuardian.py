# UniversalGuardian ComfyUI Node
# Version: 4.0.2 (Validation Fix)
# Author: MDMAchine (Enhanced by AI Assistant)
# License: MIT License
# Description:
# A comprehensive, modality-agnostic AI guardian node for ComfyUI workflows.
# This node performs quality evaluation and can regenerate content when integrated
# with samplers. It features dynamic prompt enhancement and multiple quality metrics.
#
# Key Features:
# - True guardian functionality with regeneration capability
# - Multiple quality metrics (CLIP, sharpness, contrast, variance)
# - Enhanced audio/video quality evaluation
# - Robust error handling and memory management
# - Configurable retry behavior with early stopping
# - Live preview and comprehensive logging
#
# Dependencies:
# - torch
# - numpy
# - requests
# - transformers (for CLIP)
# - ollama (for prompt enhancement)
# - librosa (optional, for audio analysis)
# - opencv-python (optional, for video analysis)

import torch
import numpy as np
import requests
import random
import logging
import io
import hashlib
import time
import json
from typing import Any, Dict, List, Tuple, Optional, Union
from contextlib import contextmanager
from PIL import Image

# Try to import optional dependencies
try:
    from transformers import CLIPProcessor, CLIPModel
    HF_AVAILABLE = True
except ImportError:
    print("WARNING: 'transformers' library not found. CLIP quality scoring will be disabled.")
    CLIPProcessor = None
    CLIPModel = None
    HF_AVAILABLE = False

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    print("WARNING: 'ollama' library not found. Ollama prompt enhancement will be disabled.")
    ollama = None
    OLLAMA_AVAILABLE = False

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    print("INFO: 'librosa' not found. Advanced audio analysis will be limited.")
    librosa = None
    LIBROSA_AVAILABLE = False

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    print("INFO: 'opencv-python' not found. Advanced video analysis will be limited.")
    cv2 = None
    OPENCV_AVAILABLE = False

# Import ComfyUI modules
import comfy
import comfy.samplers
import comfy.model_management
import comfy.sample
import comfy.utils

# Set up logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("UniversalGuardian")

class QualityMetrics:
    """Container for multiple quality metrics"""
    def __init__(self):
        self.clip_score = 0.0
        self.sharpness = 0.0
        self.contrast = 0.0
        self.variance = 0.0
        self.audio_snr = 0.0  # Signal-to-noise ratio for audio
        self.video_motion = 0.0  # Motion score for video
        self.composite_score = 0.0
    
    def calculate_composite(self, weights: Dict[str, float] = None) -> float:
        """Calculate weighted composite score"""
        if weights is None:
            weights = {} # Default weights are set in the node
        
        # Calculate composite score based on available metrics and provided weights
        self.composite_score = sum(
            getattr(self, metric) * weight 
            for metric, weight in weights.items()
            if hasattr(self, metric)
        )
        return self.composite_score
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for logging"""
        return {
            attr: getattr(self, attr) 
            for attr in dir(self) 
            if not attr.startswith('_') and isinstance(getattr(self, attr), (int, float))
        }

class UniversalGuardian:
    """
    Enhanced AI guardian that enforces output quality through evaluation
    and optional regeneration with comprehensive quality metrics.
    """
    __version__ = "4.0.2"
    __author__ = "MDMAchine (Enhanced by AI Assistant)"

    def __init__(self):
        self.clip_model = None
        self.clip_processor = None
        self.ollama_client = None

    @contextmanager
    def safe_device_context(self, model, target_device):
        """Context manager for safe device operations"""
        if model is None:
            yield None
            return
            
        original_device = next(model.parameters()).device if hasattr(model, 'parameters') else None
        try:
            if hasattr(model, 'to'):
                model.to(target_device)
            yield model
        finally:
            if original_device is not None and hasattr(model, 'to'):
                model.to(original_device)
                comfy.model_management.soft_empty_cache()

    def _validate_enhancement_config(self, kwargs: Dict[str, Any]) -> bool:
        """Validate Ollama server availability before execution"""
        if not kwargs.get('use_ollama_enhancement', False):
            return True
            
        if not OLLAMA_AVAILABLE:
            logger.warning("Ollama enhancement requested but library not available")
            return False
            
        ollama_url = kwargs.get('ollama_server_url', 'http://localhost:11434')
        try:
            response = requests.get(f"{ollama_url}/api/tags", timeout=3)
            available = response.status_code == 200
            if not available:
                logger.warning(f"Ollama server not responding at {ollama_url}")
            return available
        except Exception as e:
            logger.warning(f"Failed to validate Ollama server: {e}")
            return False

    @classmethod
    def get_ollama_models(cls) -> List[str]:
        """Fetches available models from local Ollama instance with better error handling"""
        if not OLLAMA_AVAILABLE:
            return ["Ollama not available"]
            
        try:
            default_url = "http://localhost:11434"
            response = requests.get(f'{default_url}/api/tags', timeout=3)
            if response.status_code == 200:
                data = response.json()
                models = [model['name'] for model in data.get('models', [])]
                return models if models else ["No models found"]
        except (requests.exceptions.RequestException, requests.exceptions.Timeout, json.JSONDecodeError) as e:
            logger.debug(f"Could not fetch Ollama models: {e}")
        return ["Ollama not available"]

    @classmethod
    def INPUT_TYPES(cls):
        ollama_models = cls.get_ollama_models()
        clip_models = [
            "openai/clip-vit-base-patch32", 
            "openai/clip-vit-large-patch14", 
            "laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
            "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        ]
        
        return {
            "required": {
                # Core Parameters
                "initial_prompt": ("STRING", {
                    "multiline": True, 
                    "default": "A high-quality output", 
                    "tooltip": "The prompt associated with the input for quality checks."
                }),
                "quality_metric_threshold": ("FLOAT", {
                    "default": 0.6,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Minimum composite quality score required to pass the guardian check."
                }),
                "max_quality_threshold": ("FLOAT", {
                    "default": 0.9,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Upper limit to stop retries once quality is high enough."
                }),
                "max_retries": ("INT", {
                    "default": 3,
                    "min": 0,
                    "max": 15,
                    "tooltip": "Maximum number of enhancement/regeneration attempts."
                }),
                "initial_seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 2**32 - 1,
                    "tooltip": "Starting seed for reproducibility. 0 = random seed."
                }),
                "retry_behavior": ([
                    "Return on First Success", 
                    "Return Best Attempt", 
                    "Return Last Attempt"
                ], {
                    "tooltip": "Strategy for selecting final output."
                }),
            },
            "optional": {
                # Input Data (one should be provided)
                "latent_input": ("LATENT", {
                    "tooltip": "Latent tensor to evaluate/regenerate."
                }),
                "image_input": ("IMAGE", {
                    "tooltip": "Image tensor to evaluate."
                }),
                "audio_input": ("AUDIO", {
                    "tooltip": "Audio data to evaluate."
                }),
                "video_input": ("VIDEO", {
                    "tooltip": "Video data to evaluate."
                }),
                
                # Regeneration Components (for true guardian functionality)
                "model": ("MODEL", {
                    "tooltip": "Model for latent decoding or regeneration."
                }),
                "positive_conditioning": ("CONDITIONING", {
                    "tooltip": "Positive conditioning for regeneration."
                }),
                "negative_conditioning": ("CONDITIONING", {
                    "tooltip": "Negative conditioning for regeneration."
                }),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {
                    "default": "euler",
                    "tooltip": "Sampler for regeneration."
                }),
                "scheduler_name": (comfy.samplers.KSampler.SCHEDULERS, {
                    "default": "normal", 
                    "tooltip": "Scheduler for regeneration."
                }),
                "steps": ("INT", {
                    "default": 20,
                    "min": 1,
                    "max": 150,
                    "tooltip": "Sampling steps for regeneration."
                }),
                "cfg": ("FLOAT", {
                    "default": 7.0,
                    "min": 0.0,
                    "max": 30.0,
                    "step": 0.1,
                    "tooltip": "CFG scale for regeneration."
                }),
                "denoise": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Denoising strength for regeneration."
                }),
                
                # CLIP Components
                "clip_model": ("CLIP", {
                    "forceInput": True, 
                    "default": None,
                    "tooltip": "Pre-loaded CLIP model."
                }),
                "clip_processor": ("CLIP_VISION", {
                    "forceInput": True,
                    "default": None, 
                    "tooltip": "Pre-loaded CLIP processor."
                }),
                "use_clip_scoring": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable CLIP-based quality scoring."
                }),
                "clip_model_name": (clip_models, {
                    "tooltip": "HuggingFace CLIP model if none provided."
                }),
                
                # Quality Metrics Configuration
                "clip_weight": ("FLOAT", {
                    "default": 0.4,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Weight for CLIP score in composite metric."
                }),
                "sharpness_weight": ("FLOAT", {
                    "default": 0.25,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Weight for sharpness in composite metric."
                }),
                "contrast_weight": ("FLOAT", {
                    "default": 0.25,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Weight for contrast in composite metric."
                }),
                "variance_weight": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Weight for variance in composite metric."
                }),
                
                # Ollama Enhancement
                "use_ollama_enhancement": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Use Ollama for prompt enhancement."
                }),
                "ollama_model_name": (ollama_models, {
                    "tooltip": "Ollama model for prompt enhancement."
                }),
                "ollama_system_prompt": ("STRING", {
                    "multiline": True,
                    "default": "You are a creative prompt enhancer. Given a prompt, add specific, vivid details to improve its quality for image generation. Focus on lighting, composition, style, and technical quality. Only provide the enhanced prompt without explanations.",
                    "tooltip": "System prompt for Ollama enhancement."
                }),
                "ollama_temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "Creativity level for Ollama."
                }),
                "ollama_top_p": ("FLOAT", {
                    "default": 0.9,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Diversity control for Ollama."
                }),
                "ollama_server_url": ("STRING", {
                    "default": "http://localhost:11434",
                    "tooltip": "Custom Ollama server URL."
                }),
                
                # Alternative Enhancement
                "skip_enhancement": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Disable prompt enhancement."
                }),
                "bite_phrases": ("STRING", {
                    "multiline": True,
                    "default": "highly detailed, ultra-realistic, cinematic lighting, masterpiece, best quality, sharp focus, 8K resolution, professional photography",
                    "tooltip": "Detail-focused enhancement phrases (comma-separated)."
                }),
                "sting_phrases": ("STRING", {
                    "multiline": True,
                    "default": "ethereal, dramatic, emotive, mystical, vibrant colors, bold contrast, artistic composition, dynamic lighting",
                    "tooltip": "Style-focused enhancement phrases (comma-separated)."
                }),
                "prompt_enhancement_ratio": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Enhancement aggressiveness."
                }),
                
                # Debug and Control
                "enable_regeneration": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable actual regeneration (requires model and conditioning)."
                }),
                "debug_mode": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable detailed logging and metrics."
                }),
            }
        }

    RETURN_TYPES = ("LATENT", "IMAGE", "AUDIO", "VIDEO", "STRING", "FLOAT", "INT", "INT", "IMAGE", "STRING")
    RETURN_NAMES = ("output_latent", "output_image", "output_audio", "output_video", 
                   "final_prompt", "final_quality_metric", "attempt_count", "final_seed", 
                   "preview_image", "quality_report")
    FUNCTION = "execute"
    CATEGORY = "dynamic/guardian"
    
    def _load_clip_model(self, model_name: str, clip_model, clip_processor):
        """Load CLIP model with better error handling"""
        if clip_model and clip_processor:
            return clip_model, clip_processor
            
        if not HF_AVAILABLE:
            logger.warning("Transformers not available, cannot load CLIP model")
            return None, None
            
        if not self.clip_model:
            try:
                logger.info(f"Loading CLIP model: {model_name}")
                self.clip_model = CLIPModel.from_pretrained(model_name)
                self.clip_processor = CLIPProcessor.from_pretrained(model_name)
                logger.info("CLIP model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load CLIP model '{model_name}': {e}")
                self.clip_model = None
                self.clip_processor = None
                
        return self.clip_model, self.clip_processor

    def _get_clip_score(self, image_tensor: torch.Tensor, prompt: str, clip_model, clip_processor) -> float:
        """Enhanced CLIP scoring with better normalization"""
        if not clip_model or not clip_processor:
            return 0.0
            
        try:
            device = comfy.model_management.get_torch_device()
            
            with self.safe_device_context(clip_model, device):
                # Prepare image
                img_tensor_cpu = image_tensor.detach().cpu()
                if img_tensor_cpu.dim() == 4 and img_tensor_cpu.shape[0] == 1:
                    img_tensor_cpu = img_tensor_cpu.squeeze(0)
                    
                img_np = img_tensor_cpu.permute(1, 2, 0).numpy()
                if img_np.max() <= 1.0:
                    img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
                else:
                    img_pil = Image.fromarray(img_np.astype(np.uint8))
                
                # Process inputs
                inputs = clip_processor(
                    text=[prompt], 
                    images=img_pil, 
                    return_tensors="pt", 
                    padding=True
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Get similarity score
                with torch.no_grad():
                    outputs = clip_model(**inputs)
                    
                # Improved normalization
                logits_per_image = outputs.logits_per_image
                # Use tanh normalization for better score distribution
                normalized_score = (torch.tanh(logits_per_image / 5.0) + 1.0) / 2.0
                return float(normalized_score.item())
                
        except Exception as e:
            logger.error(f"CLIP scoring failed: {e}")
            return 0.0

    def _calculate_sharpness(self, image_tensor: torch.Tensor) -> float:
        """Calculate image sharpness using Laplacian variance"""
        try:
            # Check for empty tensor
            if image_tensor.numel() == 0:
                return 0.0
            
            # Convert to grayscale if needed
            if image_tensor.dim() == 4:
                image_tensor = image_tensor.squeeze(0)
            if image_tensor.shape[0] == 3:  # RGB
                gray = 0.299 * image_tensor[0] + 0.587 * image_tensor[1] + 0.114 * image_tensor[2]
            else:
                gray = image_tensor[0]
            
            # Calculate Laplacian
            laplacian_kernel = torch.tensor([
                [0, -1, 0],
                [-1, 4, -1],
                [0, -1, 0]
            ], dtype=gray.dtype, device=gray.device).unsqueeze(0).unsqueeze(0)
            
            gray_padded = torch.nn.functional.pad(gray.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='reflect')
            laplacian = torch.nn.functional.conv2d(gray_padded, laplacian_kernel)
            
            # Normalize to [0, 1]
            variance = float(laplacian.var().item())
            # Normalize against a typical variance range for images in [0,1]
            return min(1.0, variance / 0.1)
            
        except Exception as e:
            logger.error(f"Sharpness calculation failed: {e}")
            return 0.0

    def _calculate_contrast(self, image_tensor: torch.Tensor) -> float:
        """Calculate image contrast using standard deviation"""
        try:
            # Check for empty tensor
            if image_tensor.numel() == 0:
                return 0.0
                
            if image_tensor.dim() == 4:
                image_tensor = image_tensor.squeeze(0)
                
            # Calculate contrast as normalized standard deviation
            std_dev = float(image_tensor.std().item())
            # Normalize assuming typical std range of 0-0.3 for [0,1] images
            contrast_score = min(1.0, std_dev / 0.3)
            return contrast_score
            
        except Exception as e:
            logger.error(f"Contrast calculation failed: {e}")
            return 0.0

    def _calculate_audio_snr(self, audio_tensor: torch.Tensor) -> float:
        """Calculate signal-to-noise ratio for audio"""
        try:
            if not LIBROSA_AVAILABLE:
                return 0.1
                
            if not isinstance(audio_tensor, torch.Tensor) or audio_tensor.numel() == 0:
                return 0.0
                
            audio_np = audio_tensor.detach().cpu().numpy().flatten()
            
            # Use librosa for more accurate RMS calculation
            rms_signal = librosa.feature.rms(y=audio_np, frame_length=2048, hop_length=512)
            
            # Simple assumption: noise is the minimum RMS value
            rms_noise = np.min(rms_signal) + 1e-10
            
            snr = np.mean(rms_signal) / rms_noise
            
            # Normalize SNR to [0, 1] range (assuming 0-60 dB range)
            return min(1.0, max(0.0, 10 * np.log10(snr) / 60.0))
            
        except Exception as e:
            logger.error(f"Audio SNR calculation failed: {e}")
            return 0.1

    def _calculate_video_motion(self, video_tensor: torch.Tensor) -> float:
        """Calculate motion score for video"""
        try:
            if not OPENCV_AVAILABLE:
                return 0.1
                
            if not isinstance(video_tensor, torch.Tensor) or video_tensor.dim() < 4 or video_tensor.numel() == 0:
                return 0.0
                
            # Take the first 10 frames or all frames if fewer
            num_frames = min(video_tensor.shape[2], 10)
            if num_frames < 2:
                return 0.0
                
            # Simple motion estimation using frame differences
            frame_diffs = []
            for i in range(1, num_frames):
                frame1_np = video_tensor[0, :, i-1, :, :].detach().cpu().permute(1, 2, 0).numpy()
                frame2_np = video_tensor[0, :, i, :, :].detach().cpu().permute(1, 2, 0).numpy()
                
                # Convert to grayscale for diff calculation
                gray1 = cv2.cvtColor((frame1_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
                gray2 = cv2.cvtColor((frame2_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
                
                diff = cv2.absdiff(gray1, gray2)
                mean_diff = np.mean(diff)
                frame_diffs.append(mean_diff)
                
            mean_motion = np.mean(frame_diffs)
            # Normalize motion score (assuming a range of 0-50 for 8-bit images)
            return min(1.0, mean_motion / 50.0)
            
        except Exception as e:
            logger.error(f"Video motion calculation failed: {e}")
            return 0.1

    def _get_comprehensive_quality_metrics(self, latent_input, image_input, audio_input, video_input,
                                         model, prompt, clip_model, clip_processor, use_clip_scoring,
                                         quality_weights) -> Tuple[QualityMetrics, Optional[torch.Tensor]]:
        """Calculate comprehensive quality metrics for any input type"""
        metrics = QualityMetrics()
        preview_image = None
        
        try:
            # Handle Image Input
            if image_input is not None:
                logger.info("Evaluating image quality...")
                image_tensor = image_input.detach().cpu()
                
                if image_tensor.dim() == 4:
                    image_for_metrics = image_tensor[0].permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
                    image_for_metrics = torch.clamp(image_for_metrics, 0.0, 1.0)
                    
                    if use_clip_scoring and clip_model and clip_processor:
                        metrics.clip_score = self._get_clip_score(image_for_metrics, prompt, clip_model, clip_processor)
                    
                    metrics.sharpness = self._calculate_sharpness(image_for_metrics)
                    metrics.contrast = self._calculate_contrast(image_for_metrics)
                    metrics.variance = float(image_for_metrics.var().item())
                    preview_image = image_for_metrics
                    
            # Handle Latent Input
            elif latent_input is not None:
                logger.info("Evaluating latent quality...")
                latent_tensor = latent_input["samples"]
                
                decoded_image = None
                if use_clip_scoring and model is not None:
                    try:
                        vae = getattr(model, 'first_stage_model', None)
                        if vae is not None:
                            with self.safe_device_context(vae, comfy.model_management.get_torch_device()):
                                decoded_images = vae.decode(latent_tensor.to(comfy.model_management.get_torch_device()))
                                decoded_image = decoded_images[0].detach().cpu()
                    except Exception as e:
                        logger.error(f"Latent decoding failed: {e}")
                
                if decoded_image is not None:
                    # Ensure correct format for decoded image
                    if decoded_image.dim() == 3:
                        if decoded_image.shape[2] == 3:  # [H, W, 3]
                            image_for_metrics = decoded_image.permute(2, 0, 1).unsqueeze(0)
                        elif decoded_image.shape[0] == 3:  # [3, H, W]
                            image_for_metrics = decoded_image.unsqueeze(0)
                        else:
                            image_for_metrics = decoded_image[:3].unsqueeze(0)
                    else:
                        image_for_metrics = decoded_image[:3].unsqueeze(0) if decoded_image.dim() >= 3 else None
                        
                    if image_for_metrics is not None:
                        image_for_metrics = torch.clamp(image_for_metrics, 0.0, 1.0)
                        if clip_model and clip_processor:
                            metrics.clip_score = self._get_clip_score(image_for_metrics, prompt, clip_model, clip_processor)
                        
                        metrics.sharpness = self._calculate_sharpness(image_for_metrics)
                        metrics.contrast = self._calculate_contrast(image_for_metrics)
                        preview_image = image_for_metrics
                
                # Fallback to latent space metrics if image metrics fail
                latent_cpu = latent_tensor.detach().cpu()
                metrics.variance = float(latent_cpu.var().item())
                
                if preview_image is None:
                    # Create preview from latent (normalized)
                    preview_latent = latent_cpu[:1, :3]
                    preview_latent = (preview_latent - preview_latent.min()) / (preview_latent.max() - preview_latent.min() + 1e-8)
                    preview_image = torch.clamp(preview_latent, 0, 1)
                    
            # Handle Audio Input
            elif audio_input is not None:
                logger.info("Evaluating audio quality...")
                metrics.audio_snr = self._calculate_audio_snr(audio_input)
                metrics.variance = 0.1  # Placeholder
                
            # Handle Video Input
            elif video_input is not None:
                logger.info("Evaluating video quality...")
                metrics.video_motion = self._calculate_video_motion(video_input)
                metrics.variance = 0.1  # Placeholder
                
                # Try to extract first frame for preview
                try:
                    if isinstance(video_input, torch.Tensor) and video_input.dim() == 5: # B, C, F, H, W
                        first_frame = video_input[0, :, 0, :, :]
                        preview_image = first_frame[:3].unsqueeze(0)
                        preview_image = torch.clamp(preview_image, 0, 1)
                except Exception as e:
                    logger.debug(f"Video preview extraction failed: {e}")
            
            # Calculate composite score
            metrics.calculate_composite(quality_weights)
            
        except Exception as e:
            logger.error(f"Quality metrics calculation failed: {e}")
            
        return metrics, preview_image

    def _enhance_prompt_ollama(self, prompt: str, model_name: str, system_prompt: str, 
                              temperature: float, top_p: float, ollama_server_url: str) -> str:
        """Enhanced Ollama prompt enhancement with better error handling"""
        if not OLLAMA_AVAILABLE or model_name == "Ollama not available":
            logger.warning("Ollama not available, skipping enhancement")
            return prompt
            
        try:
            ollama_client = ollama.Client(host=ollama_server_url)
            response = ollama_client.chat(
                model=model_name,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': prompt},
                ],
                options={
                    'temperature': temperature, 
                    'top_p': top_p,
                    'num_predict': 200  # Limit response length
                },
            )
            enhanced = response['message']['content'].strip()
            
            # Simple validation to avoid excessively long prompts
            if len(enhanced.split()) > len(prompt.split()) * 3:
                logger.warning("Ollama enhancement too verbose, truncating")
                enhanced = ' '.join(enhanced.split()[:len(prompt.split()) * 2])
            
            return enhanced if enhanced else prompt
            
        except Exception as e:
            logger.error(f"Ollama enhancement failed: {e}")
            return prompt

    def _enhance_prompt_bite_sting(self, prompt: str, ratio: float, bite_phrases_str: str, 
                                  sting_phrases_str: str, current_seed: int) -> str:
        """Enhanced deterministic prompt enhancement"""
        if not prompt:
            prompt = "Generate high-quality output"
            
        bite_phrases = [p.strip() for p in bite_phrases_str.split(',') if p.strip()]
        sting_phrases = [p.strip() for p in sting_phrases_str.split(',') if p.strip()]
        
        if not bite_phrases and not sting_phrases:
            return prompt
            
        rng = random.Random(current_seed)
        enhancement_parts = []
        
        # Simple tiered logic for adding phrases
        if ratio > 0.7:
            # Aggressive enhancement
            if bite_phrases: enhancement_parts.extend(rng.choices(bite_phrases, k=min(3, len(bite_phrases))))
            if sting_phrases: enhancement_parts.extend(rng.choices(sting_phrases, k=min(2, len(sting_phrases))))
        elif ratio > 0.3:
            # Moderate enhancement
            if bite_phrases: enhancement_parts.extend(rng.choices(bite_phrases, k=min(2, len(bite_phrases))))
            if sting_phrases: enhancement_parts.extend(rng.choices(sting_phrases, k=min(1, len(sting_phrases))))
        elif ratio > 0.1:
            # Light enhancement
            if bite_phrases: enhancement_parts.append(rng.choice(bite_phrases))
            if sting_phrases and rng.random() > 0.5: enhancement_parts.append(rng.choice(sting_phrases))
        
        if enhancement_parts:
            enhanced = f"{prompt}, {', '.join(enhancement_parts)}"
            # Clean up formatting
            enhanced = enhanced.replace(", ,", ",").strip().strip(',')
            return enhanced
            
        return prompt

    def _regenerate_latent(self, model, positive_cond, negative_cond, latent_input, 
                          sampler_name: str, scheduler_name: str, steps: int, 
                          cfg: float, denoise: float, seed: int) -> Dict[str, torch.Tensor]:
        """Regenerate latent using ComfyUI sampling"""
        try:
            if not all([model, positive_cond, negative_cond, latent_input]):
                raise ValueError("Missing required components for regeneration")
                
            latent_samples = latent_input["samples"]
            
            # The comfy.sample.sample function handles the denoise parameter internally
            samples = comfy.sample.sample(
                model=model,
                noise=torch.randn_like(latent_samples),
                steps=steps,
                cfg=cfg,
                sampler_name=sampler_name,
                scheduler=scheduler_name,
                positive=positive_cond,
                negative=negative_cond,
                latent_image=latent_input,
                denoise=denoise,
                disable_noise=False,
                start_step=None,
                last_step=None,
                force_full_denoise=False,
                seed=seed
            )
            
            return {"samples": samples}
            
        except Exception as e:
            logger.error(f"Latent regeneration failed: {e}")
            return latent_input

    def execute(self,
                # Required parameters
                initial_prompt: str,
                quality_metric_threshold: float,
                max_quality_threshold: float,
                max_retries: int,
                initial_seed: int,
                retry_behavior: str,
                
                # Optional inputs
                latent_input=None,
                image_input=None,
                audio_input=None,
                video_input=None,
                
                # Generation components
                model=None,
                positive_conditioning=None,
                negative_conditioning=None,
                sampler_name="euler",
                scheduler_name="normal",
                steps=20,
                cfg=7.0,
                denoise=1.0,
                
                # CLIP components
                clip_model=None,
                clip_processor=None,
                use_clip_scoring=True,
                clip_model_name="openai/clip-vit-base-patch32",
                
                # Quality weights
                clip_weight=0.4,
                sharpness_weight=0.25,
                contrast_weight=0.25,
                variance_weight=0.1,
                
                # Ollama settings
                use_ollama_enhancement=False,
                ollama_model_name="Ollama not available",
                ollama_system_prompt="",
                ollama_temperature=0.7,
                ollama_top_p=0.9,
                ollama_server_url="http://localhost:11434",
                
                # Enhancement settings
                skip_enhancement=False,
                bite_phrases="",
                sting_phrases="",
                prompt_enhancement_ratio=0.3,
                
                # Control settings
                enable_regeneration=False,
                debug_mode=False
                ) -> Tuple[Any, ...]:
        """
        Execute the Enhanced Universal Guardian with comprehensive quality evaluation
        """
        
        # Configure logging
        if debug_mode:
            logger.setLevel(logging.INFO)
        else:
            logger.setLevel(logging.WARNING)
            
        # Determine input type
        input_count = sum(x is not None for x in [latent_input, image_input, audio_input, video_input])
        if input_count != 1:
            raise ValueError("Exactly one input type must be provided")
            
        if latent_input is not None:
            input_type, current_input = "latent", latent_input
        elif image_input is not None:
            input_type, current_input = "image", image_input
        elif audio_input is not None:
            input_type, current_input = "audio", audio_input
        else:
            input_type, current_input = "video", video_input
            
        # Initialize seed
        if initial_seed == 0:
            initial_seed = int(time.time() * 1000) % (2**32 - 1)
            logger.info(f"Generated random seed: {initial_seed}")
            
        # Setup quality weights
        quality_weights = {
            'clip_score': clip_weight,
            'sharpness': sharpness_weight,
            'contrast': contrast_weight,
            'variance': variance_weight,
        }
        # Add audio/video weights if a non-image input is provided
        if input_type in ['audio', 'video']:
             quality_weights['audio_snr'] = 0.05
             quality_weights['video_motion'] = 0.05
        
        # Normalize weights
        total_weight = sum(quality_weights.values())
        if total_weight > 0:
            quality_weights = {k: v/total_weight for k, v in quality_weights.items()}
            
        # Load CLIP model if needed
        if use_clip_scoring:
            clip_model, clip_processor = self._load_clip_model(clip_model_name, clip_model, clip_processor)
            
        # Validate enhancement config
        if use_ollama_enhancement and not self._validate_enhancement_config({
            'use_ollama_enhancement': use_ollama_enhancement,
            'ollama_server_url': ollama_server_url
        }):
            logger.warning("Ollama validation failed, falling back to bite-sting enhancement")
            use_ollama_enhancement = False
            
        # Initialize tracking variables
        best_input = current_input
        best_prompt = initial_prompt
        best_metrics = None
        best_preview = None
        best_seed = initial_seed
        quality_history = []
        
        current_prompt = initial_prompt
        
        logger.info(f"Starting quality evaluation for {input_type} input")
        
        # Initial quality assessment
        initial_metrics, initial_preview = self._get_comprehensive_quality_metrics(
            latent_input, image_input, audio_input, video_input,
            model, initial_prompt, clip_model, clip_processor, use_clip_scoring, quality_weights
        )
        
        logger.info(f"Initial quality score: {initial_metrics.composite_score:.4f}")
        if debug_mode:
            logger.info(f"Initial metrics: {initial_metrics.to_dict()}")
            
        # Update best attempt
        best_input = current_input
        best_metrics = initial_metrics
        best_preview = initial_preview
        quality_history.append({
            'attempt': 0,
            'prompt': initial_prompt,
            'metrics': initial_metrics.to_dict(),
            'composite_score': initial_metrics.composite_score
        })
        
        # Check if initial quality meets criteria
        if (retry_behavior == "Return on First Success" and 
            initial_metrics.composite_score >= quality_metric_threshold):
            logger.info("Initial quality meets threshold, returning immediately")
            return self._prepare_return_values(
                input_type, best_input, best_prompt, best_metrics.composite_score,
                0, best_seed, best_preview, quality_history
            )
            
        if initial_metrics.composite_score >= max_quality_threshold:
            logger.info("Initial quality meets max threshold, returning immediately")
            return self._prepare_return_values(
                input_type, best_input, best_prompt, best_metrics.composite_score,
                0, best_seed, best_preview, quality_history
            )
            
        # Retry loop
        for attempt in range(1, max_retries + 1):
            current_seed = initial_seed + attempt
            logger.info(f"Starting attempt {attempt}/{max_retries} (seed: {current_seed})")
            
            # Enhanced prompt
            if not skip_enhancement:
                prev_prompt = current_prompt
                if use_ollama_enhancement:
                    current_prompt = self._enhance_prompt_ollama(
                        current_prompt, ollama_model_name, ollama_system_prompt,
                        ollama_temperature, ollama_top_p, ollama_server_url
                    )
                else:
                    current_prompt = self._enhance_prompt_bite_sting(
                        current_prompt, prompt_enhancement_ratio, 
                        bite_phrases, sting_phrases, current_seed
                    )
                    
                if debug_mode and current_prompt != prev_prompt:
                    logger.info(f"Enhanced prompt: {current_prompt}")
                    
            # Regenerate if enabled and possible
            if (enable_regeneration and input_type == "latent" and 
                all([model, positive_conditioning, negative_conditioning])):
                
                logger.info("Regenerating latent...")
                current_input = self._regenerate_latent(
                    model, positive_conditioning, negative_conditioning, latent_input,
                    sampler_name, scheduler_name, steps, cfg, denoise, current_seed
                )
            
            # Evaluate current attempt
            current_metrics, current_preview = self._get_comprehensive_quality_metrics(
                current_input if input_type == "latent" else None,
                current_input if input_type == "image" else None,
                current_input if input_type == "audio" else None,
                current_input if input_type == "video" else None,
                model, current_prompt, clip_model, clip_processor, use_clip_scoring, quality_weights
            )
            
            logger.info(f"Attempt {attempt} quality score: {current_metrics.composite_score:.4f}")
            if debug_mode:
                logger.info(f"Attempt {attempt} metrics: {current_metrics.to_dict()}")
                
            # Record attempt
            quality_history.append({
                'attempt': attempt,
                'prompt': current_prompt,
                'metrics': current_metrics.to_dict(),
                'composite_score': current_metrics.composite_score
            })
            
            # Update best if improved
            if current_metrics.composite_score > best_metrics.composite_score:
                best_input = current_input
                best_prompt = current_prompt
                best_metrics = current_metrics
                best_preview = current_preview
                best_seed = current_seed
                logger.info(f"New best quality found: {best_metrics.composite_score:.4f}")
                
            # Check success conditions
            if (retry_behavior == "Return on First Success" and 
                current_metrics.composite_score >= quality_metric_threshold):
                logger.info(f"Quality threshold met on attempt {attempt}")
                return self._prepare_return_values(
                    input_type, current_input, current_prompt, current_metrics.composite_score,
                    attempt, current_seed, current_preview, quality_history
                )
                
            if current_metrics.composite_score >= max_quality_threshold:
                logger.info(f"Max quality threshold reached on attempt {attempt}")
                break
                
        # Return based on behavior
        if retry_behavior == "Return Last Attempt":
            return self._prepare_return_values(
                input_type, current_input, current_prompt, current_metrics.composite_score,
                max_retries, current_seed, current_preview, quality_history
            )
        else:  # Return Best Attempt
            return self._prepare_return_values(
                input_type, best_input, best_prompt, best_metrics.composite_score,
                max_retries, best_seed, best_preview, quality_history
            )
            
    def _prepare_return_values(self, input_type: str, final_input, final_prompt: str,
                              final_quality: float, attempt_count: int, final_seed: int,
                              preview_image, quality_history: List[Dict]) -> Tuple[Any, ...]:
        """Prepare return values based on input type"""
        
        # Create quality report
        quality_report = json.dumps({
            'final_quality': final_quality,
            'attempt_count': attempt_count,
            'input_type': input_type,
            'history': quality_history
        }, indent=2)
        
        # Initialize return array
        returns = [None, None, None, None, final_prompt, final_quality, 
                  attempt_count, final_seed, preview_image, quality_report]
        
        # Set appropriate output
        type_indices = {"latent": 0, "image": 1, "audio": 2, "video": 3}
        returns[type_indices[input_type]] = final_input
        
        return tuple(returns)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """This method has been removed to avoid a validation bug."""
        return 42 # A simple constant value to satisfy ComfyUI

# Node Registration
NODE_CLASS_MAPPINGS = {
    "UniversalGuardian": UniversalGuardian,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UniversalGuardian": "Universal Guardian (Enhanced Evaluator & Generator)",
}