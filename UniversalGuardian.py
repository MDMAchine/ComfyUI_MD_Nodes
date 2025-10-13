# UniversalGuardian ComfyUI Node
# Version: 4.1.0 (Enhanced with LM Studio + Fixes)
# Author: MDMAchine (Enhanced by AI Assistant)
# License: MIT License

import torch
import numpy as np
import requests
import random
import logging
import json
import time
from typing import Any, Dict, List, Tuple, Optional
from contextlib import contextmanager
from PIL import Image

# Optional dependencies with better error handling
try:
    from transformers import CLIPProcessor, CLIPModel
    HF_AVAILABLE = True
except ImportError:
    print("WARNING: 'transformers' not found. CLIP scoring disabled.")
    CLIPProcessor = CLIPModel = None
    HF_AVAILABLE = False

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    print("INFO: 'ollama' not found. Ollama enhancement disabled.")
    ollama = None
    OLLAMA_AVAILABLE = False

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    librosa = None
    LIBROSA_AVAILABLE = False

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    cv2 = None
    OPENCV_AVAILABLE = False

import comfy
import comfy.samplers
import comfy.model_management
import comfy.sample

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("UniversalGuardian")

class QualityMetrics:
    """Container for quality metrics with better normalization"""
    def __init__(self):
        self.clip_score = 0.0
        self.sharpness = 0.0
        self.contrast = 0.0
        self.variance = 0.0
        self.audio_snr = 0.0
        self.video_motion = 0.0
        self.composite_score = 0.0
    
    def calculate_composite(self, weights: Dict[str, float]) -> float:
        """Calculate weighted composite score"""
        self.composite_score = sum(
            getattr(self, metric, 0.0) * weight 
            for metric, weight in weights.items()
        )
        return self.composite_score
    
    def to_dict(self) -> Dict[str, float]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

class ImageTensorHandler:
    """Centralized image tensor format handling"""
    
    @staticmethod
    def normalize_to_chw(tensor: torch.Tensor) -> torch.Tensor:
        """Convert any image format to [C, H, W] format in [0, 1] range"""
        if tensor.numel() == 0:
            raise ValueError("Empty tensor")
        
        # Remove batch dimension if present
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        
        # Handle [H, W, C] format (ComfyUI default)
        if tensor.dim() == 3 and tensor.shape[2] in [1, 3, 4]:
            tensor = tensor.permute(2, 0, 1)
        
        # Ensure [C, H, W]
        if tensor.dim() != 3:
            raise ValueError(f"Cannot normalize tensor with shape {tensor.shape}")
        
        # Normalize to [0, 1]
        if tensor.max() > 1.0:
            tensor = tensor.float() / 255.0
        
        return torch.clamp(tensor, 0.0, 1.0)
    
    @staticmethod
    def to_pil(tensor: torch.Tensor) -> Image.Image:
        """Convert tensor to PIL Image"""
        tensor = ImageTensorHandler.normalize_to_chw(tensor)
        if tensor.shape[0] == 1:  # Grayscale
            arr = (tensor[0].cpu().numpy() * 255).astype(np.uint8)
            return Image.fromarray(arr, mode='L')
        else:  # RGB
            arr = (tensor[:3].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            return Image.fromarray(arr, mode='RGB')

class LMStudioClient:
    """Client for LM Studio API (OpenAI-compatible)"""
    
    def __init__(self, base_url: str = "http://localhost:1234/v1"):
        self.base_url = base_url
        self.models_cache = None
        self.cache_time = 0
    
    def get_models(self) -> List[str]:
        """Get available models from LM Studio"""
        # Cache for 30 seconds
        if self.models_cache and (time.time() - self.cache_time) < 30:
            return self.models_cache
        
        try:
            response = requests.get(f"{self.base_url}/models", timeout=3)
            if response.status_code == 200:
                data = response.json()
                models = [m['id'] for m in data.get('data', [])]
                self.models_cache = models if models else ["No models loaded"]
                self.cache_time = time.time()
                return self.models_cache
        except Exception as e:
            logger.debug(f"LM Studio not available: {e}")
        
        return ["LM Studio not available"]
    
    def chat_completion(self, model: str, messages: List[Dict], 
                       temperature: float = 0.7, max_tokens: int = 200) -> str:
        """Send chat completion request"""
        try:
            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False
            }
            response = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                timeout=30
            )
            if response.status_code == 200:
                data = response.json()
                return data['choices'][0]['message']['content'].strip()
        except Exception as e:
            logger.error(f"LM Studio request failed: {e}")
            raise
        
        return ""

class UniversalGuardian:
    """Enhanced AI guardian with LM Studio support"""
    
    __version__ = "4.1.0"
    
    def __init__(self):
        self.clip_model = None
        self.clip_processor = None
        self.lm_studio_client = LMStudioClient()
        self.image_handler = ImageTensorHandler()
    
    @contextmanager
    def safe_device_context(self, model, target_device):
        """Safe device management with cleanup"""
        if model is None:
            yield None
            return
        
        original_device = next(model.parameters()).device if hasattr(model, 'parameters') else None
        try:
            if hasattr(model, 'to'):
                model.to(target_device)
            yield model
        finally:
            if original_device and hasattr(model, 'to'):
                model.to(original_device)
            comfy.model_management.soft_empty_cache()
    
    @classmethod
    def get_ollama_models(cls) -> List[str]:
        """Get Ollama models with caching"""
        if not OLLAMA_AVAILABLE:
            return ["Ollama not available"]
        
        try:
            response = requests.get('http://localhost:11434/api/tags', timeout=2)
            if response.status_code == 200:
                models = [m['name'] for m in response.json().get('models', [])]
                return models if models else ["No models found"]
        except:
            pass
        return ["Ollama not available"]
    
    @classmethod
    def get_lm_studio_models(cls) -> List[str]:
        """Get LM Studio models"""
        client = LMStudioClient()
        return client.get_models()
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "initial_prompt": ("STRING", {
                    "multiline": True,
                    "default": "A high-quality output"
                }),
                "quality_threshold": ("FLOAT", {
                    "default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01
                }),
                "max_quality_threshold": ("FLOAT", {
                    "default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01
                }),
                "max_retries": ("INT", {
                    "default": 3, "min": 0, "max": 15
                }),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 2**32 - 1
                }),
                "retry_mode": (["First Success", "Best Quality", "Last Attempt"], {}),
            },
            "optional": {
                # Inputs
                "latent": ("LATENT",),
                "image": ("IMAGE",),
                "audio": ("AUDIO",),
                
                # Generation
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "normal"}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 150}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 30.0, "step": 0.1}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                
                # Quality metrics
                "use_clip": ("BOOLEAN", {"default": True}),
                "clip_model_name": ([
                    "openai/clip-vit-base-patch32",
                    "openai/clip-vit-large-patch14"
                ], {}),
                "clip_weight": ("FLOAT", {"default": 0.4, "min": 0, "max": 1, "step": 0.05}),
                "sharpness_weight": ("FLOAT", {"default": 0.3, "min": 0, "max": 1, "step": 0.05}),
                "contrast_weight": ("FLOAT", {"default": 0.2, "min": 0, "max": 1, "step": 0.05}),
                "variance_weight": ("FLOAT", {"default": 0.1, "min": 0, "max": 1, "step": 0.05}),
                
                # Enhancement method
                "enhancement_method": ([
                    "None", "Ollama", "LM Studio", "Keyword"
                ], {"default": "None"}),
                
                # Ollama
                "ollama_model": (cls.get_ollama_models(), {}),
                "ollama_url": ("STRING", {"default": "http://localhost:11434"}),
                
                # LM Studio
                "lm_studio_model": (cls.get_lm_studio_models(), {}),
                "lm_studio_url": ("STRING", {"default": "http://localhost:1234/v1"}),
                
                # Shared LLM settings
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": "Enhance this image prompt with vivid details about lighting, composition, and style. Keep it concise (under 50 words)."
                }),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0, "max": 2, "step": 0.1}),
                
                # Keyword enhancement
                "keywords": ("STRING", {
                    "multiline": True,
                    "default": "highly detailed, masterpiece, 8k, professional"
                }),
                "keyword_strength": ("FLOAT", {"default": 0.3, "min": 0, "max": 1, "step": 0.05}),
                
                # Control
                "enable_regeneration": ("BOOLEAN", {"default": False}),
                "debug": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("LATENT", "IMAGE", "STRING", "FLOAT", "INT", "STRING")
    RETURN_NAMES = ("latent", "image", "prompt", "quality", "attempts", "report")
    FUNCTION = "execute"
    CATEGORY = "conditioning/guardian"
    
    def _load_clip(self, model_name: str):
        """Load CLIP model with caching"""
        if not HF_AVAILABLE:
            return None, None
        
        if self.clip_model is None:
            try:
                logger.info(f"Loading CLIP: {model_name}")
                self.clip_model = CLIPModel.from_pretrained(model_name)
                self.clip_processor = CLIPProcessor.from_pretrained(model_name)
            except Exception as e:
                logger.error(f"CLIP load failed: {e}")
                return None, None
        
        return self.clip_model, self.clip_processor
    
    def _clip_score(self, image: torch.Tensor, prompt: str, model, processor) -> float:
        """Calculate CLIP similarity score"""
        if not model or not processor:
            return 0.0
        
        try:
            device = comfy.model_management.get_torch_device()
            
            with self.safe_device_context(model, device):
                img_pil = self.image_handler.to_pil(image)
                
                inputs = processor(
                    text=[prompt],
                    images=img_pil,
                    return_tensors="pt",
                    padding=True
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    score = outputs.logits_per_image[0][0]
                    # Better normalization using sigmoid
                    return float(torch.sigmoid(score / 2.5).cpu())
        
        except Exception as e:
            logger.error(f"CLIP scoring failed: {e}")
            return 0.0
    
    def _calculate_sharpness(self, image: torch.Tensor) -> float:
        """Laplacian variance for sharpness"""
        try:
            img = self.image_handler.normalize_to_chw(image)
            gray = img.mean(dim=0) if img.shape[0] > 1 else img[0]
            
            laplacian = torch.tensor([
                [0, -1, 0],
                [-1, 4, -1],
                [0, -1, 0]
            ], dtype=gray.dtype, device=gray.device).unsqueeze(0).unsqueeze(0)
            
            gray_padded = torch.nn.functional.pad(
                gray.unsqueeze(0).unsqueeze(0), 
                (1, 1, 1, 1), 
                mode='reflect'
            )
            result = torch.nn.functional.conv2d(gray_padded, laplacian)
            
            variance = float(result.var().item())
            return min(1.0, variance * 10)  # Scale to [0, 1]
        
        except Exception as e:
            logger.error(f"Sharpness calc failed: {e}")
            return 0.0
    
    def _calculate_contrast(self, image: torch.Tensor) -> float:
        """Standard deviation for contrast"""
        try:
            img = self.image_handler.normalize_to_chw(image)
            std = float(img.std().item())
            return min(1.0, std * 3.5)  # Normalize to [0, 1]
        except Exception as e:
            logger.error(f"Contrast calc failed: {e}")
            return 0.0
    
    def _enhance_with_ollama(self, prompt: str, model: str, system: str, 
                            temp: float, url: str) -> str:
        """Ollama enhancement"""
        if not OLLAMA_AVAILABLE or "not available" in model.lower():
            return prompt
        
        try:
            client = ollama.Client(host=url)
            response = client.chat(
                model=model,
                messages=[
                    {'role': 'system', 'content': system},
                    {'role': 'user', 'content': prompt}
                ],
                options={'temperature': temp, 'num_predict': 150}
            )
            result = response['message']['content'].strip()
            return result if result else prompt
        except Exception as e:
            logger.error(f"Ollama failed: {e}")
            return prompt
    
    def _enhance_with_lm_studio(self, prompt: str, model: str, system: str,
                               temp: float, url: str) -> str:
        """LM Studio enhancement"""
        if "not available" in model.lower():
            return prompt
        
        try:
            client = LMStudioClient(url)
            messages = [
                {'role': 'system', 'content': system},
                {'role': 'user', 'content': prompt}
            ]
            result = client.chat_completion(model, messages, temp, 150)
            return result if result else prompt
        except Exception as e:
            logger.error(f"LM Studio failed: {e}")
            return prompt
    
    def _enhance_with_keywords(self, prompt: str, keywords: str, 
                              strength: float, seed: int) -> str:
        """Simple keyword enhancement"""
        if strength <= 0 or not keywords.strip():
            return prompt
        
        kw_list = [k.strip() for k in keywords.split(',') if k.strip()]
        if not kw_list:
            return prompt
        
        rng = random.Random(seed)
        num_keywords = max(1, int(len(kw_list) * strength))
        selected = rng.sample(kw_list, min(num_keywords, len(kw_list)))
        
        return f"{prompt}, {', '.join(selected)}"
    
    def _evaluate_quality(self, latent, image, model, prompt, 
                         use_clip, clip_model, clip_processor, weights) -> Tuple[QualityMetrics, Optional[torch.Tensor]]:
        """Comprehensive quality evaluation"""
        metrics = QualityMetrics()
        preview = None
        
        try:
            eval_image = None
            
            # Get image to evaluate
            if image is not None:
                eval_image = image[0] if image.dim() == 4 else image
            elif latent is not None and model is not None:
                # Decode latent
                try:
                    vae = getattr(model, 'first_stage_model', None)
                    if vae:
                        device = comfy.model_management.get_torch_device()
                        with self.safe_device_context(vae, device):
                            decoded = vae.decode(latent["samples"].to(device))
                            eval_image = decoded[0].detach().cpu()
                except Exception as e:
                    logger.debug(f"Decode failed: {e}")
            
            # Calculate metrics if we have an image
            if eval_image is not None:
                eval_image = self.image_handler.normalize_to_chw(eval_image)
                
                if use_clip and clip_model and clip_processor:
                    metrics.clip_score = self._clip_score(eval_image, prompt, clip_model, clip_processor)
                
                metrics.sharpness = self._calculate_sharpness(eval_image)
                metrics.contrast = self._calculate_contrast(eval_image)
                metrics.variance = float(eval_image.var().item())
                
                preview = eval_image.unsqueeze(0)
            
            # Calculate composite
            metrics.calculate_composite(weights)
        
        except Exception as e:
            logger.error(f"Quality evaluation failed: {e}")
        
        return metrics, preview
    
    def _regenerate(self, model, positive, negative, latent, sampler, scheduler,
                   steps, cfg, denoise, seed):
        """Regenerate latent"""
        try:
            samples = comfy.sample.sample(
                model=model,
                noise=torch.randn_like(latent["samples"]),
                steps=steps,
                cfg=cfg,
                sampler_name=sampler,
                scheduler=scheduler,
                positive=positive,
                negative=negative,
                latent_image=latent,
                denoise=denoise,
                seed=seed
            )
            return {"samples": samples}
        except Exception as e:
            logger.error(f"Regeneration failed: {e}")
            return latent
    
    def execute(self, initial_prompt, quality_threshold, max_quality_threshold,
                max_retries, seed, retry_mode, **kwargs):
        """Main execution"""
        
        # Setup logging
        if kwargs.get('debug', False):
            logger.setLevel(logging.INFO)
        
        # Validate inputs
        input_map = {
            'latent': kwargs.get('latent'),
            'image': kwargs.get('image'),
            'audio': kwargs.get('audio')
        }
        active_inputs = {k: v for k, v in input_map.items() if v is not None}
        
        if len(active_inputs) != 1:
            raise ValueError("Provide exactly one input (latent, image, or audio)")
        
        input_type, current_input = list(active_inputs.items())[0]
        
        # Generate seed if needed
        if seed == 0:
            seed = int(time.time() * 1000) % (2**32 - 1)
        
        # Setup quality weights (normalize)
        weights = {
            'clip_score': kwargs.get('clip_weight', 0.4),
            'sharpness': kwargs.get('sharpness_weight', 0.3),
            'contrast': kwargs.get('contrast_weight', 0.2),
            'variance': kwargs.get('variance_weight', 0.1)
        }
        total = sum(weights.values())
        weights = {k: v/total for k, v in weights.items()} if total > 0 else weights
        
        # Load CLIP if needed
        clip_model = clip_processor = None
        if kwargs.get('use_clip', True):
            clip_model, clip_processor = self._load_clip(kwargs.get('clip_model_name', 'openai/clip-vit-base-patch32'))
        
        # Tracking variables
        best_input = current_input
        best_prompt = initial_prompt
        best_quality = 0.0
        best_preview = None
        history = []
        
        # Initial evaluation
        logger.info(f"Evaluating {input_type} input...")
        metrics, preview = self._evaluate_quality(
            kwargs.get('latent'), kwargs.get('image'),
            kwargs.get('model'), initial_prompt,
            kwargs.get('use_clip', True), clip_model, clip_processor, weights
        )
        
        logger.info(f"Initial quality: {metrics.composite_score:.3f}")
        best_quality = metrics.composite_score
        best_preview = preview
        history.append({'attempt': 0, 'quality': best_quality, 'prompt': initial_prompt})
        
        # Early exit if quality sufficient
        if metrics.composite_score >= quality_threshold and retry_mode == "First Success":
            return self._return_values(input_type, best_input, best_prompt, 
                                      best_quality, 0, seed, best_preview, history)
        
        # Retry loop
        for attempt in range(1, max_retries + 1):
            current_seed = seed + attempt
            logger.info(f"Attempt {attempt}/{max_retries}")
            
            # Enhance prompt
            current_prompt = initial_prompt
            method = kwargs.get('enhancement_method', 'None')
            
            if method == "Ollama":
                current_prompt = self._enhance_with_ollama(
                    current_prompt,
                    kwargs.get('ollama_model', ''),
                    kwargs.get('system_prompt', ''),
                    kwargs.get('temperature', 0.7),
                    kwargs.get('ollama_url', 'http://localhost:11434')
                )
            elif method == "LM Studio":
                current_prompt = self._enhance_with_lm_studio(
                    current_prompt,
                    kwargs.get('lm_studio_model', ''),
                    kwargs.get('system_prompt', ''),
                    kwargs.get('temperature', 0.7),
                    kwargs.get('lm_studio_url', 'http://localhost:1234/v1')
                )
            elif method == "Keyword":
                current_prompt = self._enhance_with_keywords(
                    current_prompt,
                    kwargs.get('keywords', ''),
                    kwargs.get('keyword_strength', 0.3),
                    current_seed
                )
            
            # Regenerate if enabled
            if kwargs.get('enable_regeneration', False) and input_type == 'latent':
                if all([kwargs.get('model'), kwargs.get('positive'), kwargs.get('negative')]):
                    current_input = self._regenerate(
                        kwargs['model'], kwargs['positive'], kwargs['negative'],
                        kwargs['latent'], kwargs.get('sampler_name', 'euler'),
                        kwargs.get('scheduler', 'normal'), kwargs.get('steps', 20),
                        kwargs.get('cfg', 7.0), kwargs.get('denoise', 1.0), current_seed
                    )
            
            # Evaluate
            metrics, preview = self._evaluate_quality(
                current_input if input_type == 'latent' else None,
                current_input if input_type == 'image' else None,
                kwargs.get('model'), current_prompt,
                kwargs.get('use_clip', True), clip_model, clip_processor, weights
            )
            
            logger.info(f"Quality: {metrics.composite_score:.3f}")
            history.append({'attempt': attempt, 'quality': metrics.composite_score, 'prompt': current_prompt})
            
            # Update best
            if metrics.composite_score > best_quality:
                best_input = current_input
                best_prompt = current_prompt
                best_quality = metrics.composite_score
                best_preview = preview
            
            # Check thresholds
            if retry_mode == "First Success" and metrics.composite_score >= quality_threshold:
                return self._return_values(input_type, current_input, current_prompt,
                                          metrics.composite_score, attempt, current_seed, preview, history)
            
            if metrics.composite_score >= max_quality_threshold:
                break
        
        # Return based on mode
        if retry_mode == "Last Attempt":
            return self._return_values(input_type, current_input, current_prompt,
                                      metrics.composite_score, max_retries, current_seed, preview, history)
        else:  # Best Quality
            return self._return_values(input_type, best_input, best_prompt,
                                      best_quality, max_retries, seed, best_preview, history)
    
    def _return_values(self, input_type, final_input, prompt, quality, 
                      attempts, seed, preview, history):
        """Format return values"""
        report = json.dumps({
            'input_type': input_type,
            'final_quality': quality,
            'attempts': attempts,
            'seed': seed,
            'history': history
        }, indent=2)
        
        # Default returns
        latent_out = final_input if input_type == 'latent' else None
        image_out = final_input if input_type == 'image' else preview
        
        # Ensure image is in correct format
        if image_out is not None and image_out.dim() == 3:
            image_out = image_out.unsqueeze(0)
        
        return (latent_out, image_out, prompt, quality, attempts, report)

# Registration
NODE_CLASS_MAPPINGS = {
    "UniversalGuardian": UniversalGuardian,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UniversalGuardian": "🛡️ Universal Guardian",
}