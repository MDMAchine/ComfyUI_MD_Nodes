# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
# ████ MD_Nodes/utils/self_cross_utils – Self-Cross Components ████▓▒░
# © 2025 MDMAchine
# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
# ░▒▓ DESCRIPTION:
#   Self-Cross specific components: config, attention storage, loss computation.
#   Built on the universal guidance engine (guidance_core.py).
#
# ░▒▓ CHANGELOG:
#   - v1.1.0 (Current - CRITICAL FIX):
#       FIXED: __post_init__ no longer calls super().__post_init__() (doesn't exist)
#       WORKING: Proper validation without inheritance issues
# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄

import torch
import torch.nn.functional as F
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

# =================================================================================
# == Imports from guidance_core
# =================================================================================
try:
    from .guidance_core import GuidanceConfig, GuidanceContext, BaseGuidanceLoss
except ImportError:
    from guidance_core import GuidanceConfig, GuidanceContext, BaseGuidanceLoss

# =================================================================================
# == Self-Cross Configuration
# =================================================================================
@dataclass
class SelfCrossConfig(GuidanceConfig):
    """Configuration for Self-Cross Guidance."""
    
    # Subject-specific settings
    subject_indices: List[int] = field(default_factory=list)
    guidance_scale: float = 10.0
    param_lambda: float = 0.3  # Attend-and-Excite weight
    
    # Attention filtering
    target_layers: List[str] = field(default_factory=lambda: ["middle", "output_4", "output_5"])
    min_resolution: int = 8
    max_resolution: int = 256
    
    # Otsu settings
    use_2d_otsu: bool = False  # Use 1D by default (audio-safe)
    
    def __post_init__(self):
        """Validate configuration (FIXED - no super() call)."""
        # CRITICAL FIX: Don't call super().__post_init__() - parent class doesn't have it
        
        # Validation
        if not self.subject_indices:
            raise ValueError(
                "subject_indices cannot be empty. "
                "Use MD_CLIPTokenFinder to get token indices."
            )
        
        if len(self.subject_indices) < 2:
            raise ValueError(
                f"Need at least 2 subject indices for separation, got {len(self.subject_indices)}"
            )
        
        if self.guidance_scale <= 0:
            raise ValueError(f"guidance_scale must be positive, got {self.guidance_scale}")
        
        if self.max_iters < 0:
            raise ValueError(f"max_iters must be non-negative, got {self.max_iters}")
        
        if self.param_lambda < 0 or self.param_lambda > 1:
            raise ValueError(f"param_lambda must be in [0, 1], got {self.param_lambda}")
        
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        
        logging.info(
            f"[SelfCross] Config validated: {len(self.subject_indices)} subjects, "
            f"scale={self.guidance_scale}, iters={self.max_iters}"
        )

# =================================================================================
# == Otsu Thresholding (Audio-Safe 1D Version)
# =================================================================================
def torch_otsu_threshold(attention_map: torch.Tensor, use_2d: bool = False) -> torch.Tensor:
    """
    Audio-safe Otsu thresholding.
    
    Args:
        attention_map: Attention values, shape [H, W] or [tokens]
        use_2d: If True, treat as 2D image. If False, use 1D (audio-safe)
    
    Returns:
        Binary mask
    """
    if use_2d and attention_map.dim() == 2:
        # 2D Otsu (for square images)
        flat = attention_map.flatten()
    else:
        # 1D Otsu (audio-safe, works for any shape)
        flat = attention_map.flatten()
    
    # Normalize to [0, 255]
    flat_norm = ((flat - flat.min()) / (flat.max() - flat.min() + 1e-8) * 255).long()
    flat_norm = torch.clamp(flat_norm, 0, 255)
    
    # Compute histogram
    hist = torch.zeros(256, device=flat.device)
    hist = hist.scatter_add(0, flat_norm, torch.ones_like(flat_norm, dtype=torch.float32))
    
    # Otsu algorithm
    total = flat_norm.numel()
    sum_total = torch.sum(torch.arange(256, device=flat.device).float() * hist)
    
    sum_background = 0.0
    weight_background = 0.0
    max_variance = 0.0
    threshold = 0
    
    for t in range(256):
        weight_background += hist[t]
        if weight_background == 0:
            continue
        
        weight_foreground = total - weight_background
        if weight_foreground == 0:
            break
        
        sum_background += t * hist[t]
        
        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground
        
        variance = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
        
        if variance > max_variance:
            max_variance = variance
            threshold = t
    
    # Apply threshold
    mask = (flat_norm > threshold).float()
    return mask.reshape(attention_map.shape)

# =================================================================================
# == Attention Storage
# =================================================================================
class AttentionStore:
    """Stores attention maps from model forward pass."""
    
    def __init__(self, config: SelfCrossConfig):
        self.config = config
        self.attention_maps: Dict[str, List[torch.Tensor]] = {
            'self': [],
            'cross': []
        }
        self.step_index = 0
    
    def append(self, attn_type: str, attn_map: torch.Tensor, layer_name: str):
        """Store attention map with filtering."""
        # Extract resolution from layer name if possible
        resolution = self._extract_resolution(attn_map)
        
        # Filter by resolution
        if resolution < self.config.min_resolution or resolution > self.config.max_resolution:
            return
        
        # Filter by layer name
        if not any(target in layer_name for target in self.config.target_layers):
            return
        
        self.attention_maps[attn_type].append(attn_map.detach())
    
    def _extract_resolution(self, attn_map: torch.Tensor) -> int:
        """Extract approximate resolution from attention map shape."""
        if attn_map.dim() >= 3:
            # Shape like [batch, tokens, tokens] or [batch, heads, tokens, tokens]
            token_dim = attn_map.shape[-1]
            # Approximate spatial resolution (sqrt for square, or use directly for audio)
            return int(token_dim ** 0.5) if token_dim > 16 else token_dim
        return 0
    
    def get_attention(self, attn_type: str) -> Optional[torch.Tensor]:
        """Get last (deepest) attention map of specified type."""
        maps = self.attention_maps.get(attn_type, [])
        if not maps:
            return None
        return maps[-1]  # Return most recent/deepest layer
    
    def reset(self):
        """Clear stored attention maps."""
        self.attention_maps = {'self': [], 'cross': []}
        self.step_index += 1

# =================================================================================
# == Attention Hook Manager
# =================================================================================
class AttentionHookManager:
    """Manages attention hooks on model."""
    
    def __init__(self, model, attn_store: AttentionStore):
        self.model = model
        self.attn_store = attn_store
        self.hooks = []
    
    def inject_hooks(self):
        """Inject hooks to capture attention."""
        def make_hook(attn_type: str, layer_name: str):
            def hook(module, args, output):
                # Extract attention from output
                if isinstance(output, tuple):
                    attn = output[0] if len(output) > 0 else None
                else:
                    attn = output
                
                if attn is not None and torch.is_tensor(attn):
                    self.attn_store.append(attn_type, attn, layer_name)
                
                return output
            return hook
        
        # Hook self-attention (attn1)
        if hasattr(self.model, 'set_model_attn1_patch'):
            self.model.set_model_attn1_patch(make_hook('self', 'attn1'))
            logging.info("[Hooks] Injected self-attention (attn1) hook")
        
        # Hook cross-attention (attn2)
        if hasattr(self.model, 'set_model_attn2_patch'):
            self.model.set_model_attn2_patch(make_hook('cross', 'attn2'))
            logging.info("[Hooks] Injected cross-attention (attn2) hook")
    
    def remove_hooks(self):
        """Remove all hooks."""
        if hasattr(self.model, 'set_model_attn1_patch'):
            self.model.set_model_attn1_patch(None)
        if hasattr(self.model, 'set_model_attn2_patch'):
            self.model.set_model_attn2_patch(None)
        logging.info("[Hooks] Removed attention hooks")

# =================================================================================
# == Self-Cross Loss Computation
# =================================================================================
class SelfCrossLoss(BaseGuidanceLoss):
    """Computes Self-Cross loss (subject overlap penalty)."""
    
    def __init__(self, config: SelfCrossConfig, attn_store: AttentionStore, 
                 latent_shape: Optional[Tuple[int, ...]] = None):
        super().__init__(config)
        self.config: SelfCrossConfig = config  # Type hint for IDE
        self.attn_store = attn_store
        self.latent_shape = latent_shape
    
    def compute(self, context: GuidanceContext) -> torch.Tensor:
        """
        Compute Self-Cross overlap loss.
        
        Loss = Σ_i,j min(Agg_Si, Cross_Sj) for all subject pairs
        where Agg_Si = Σ_p self_attn(p,p') * cross_attn(Si,p') * mask(Si,p')
        """
        # Get attention maps
        cross_attn = self.attn_store.get_attention('cross')
        self_attn = self.attn_store.get_attention('self')
        
        if cross_attn is None or self_attn is None:
            logging.warning("[SelfCross] No attention maps captured, returning zero loss")
            return torch.tensor(0.0, device=context.z_noisy.device, requires_grad=True)
        
        # Handle batch dimension
        if cross_attn.dim() == 4:  # [batch, heads, tokens, text_tokens]
            cross_attn = cross_attn[0]  # First batch
        if cross_attn.dim() == 3:  # [heads, tokens, text_tokens]
            cross_attn = cross_attn.mean(dim=0)  # Average over heads
        
        if self_attn.dim() == 4:
            self_attn = self_attn[0]
        if self_attn.dim() == 3:
            self_attn = self_attn.mean(dim=0)
        
        # Now: cross_attn [tokens, text_tokens], self_attn [tokens, tokens]
        
        # Aggregate self-attention for each subject
        aggregated_maps = []
        
        for subj_idx in self.config.subject_indices:
            if subj_idx >= cross_attn.shape[1]:
                logging.warning(f"[SelfCross] Subject index {subj_idx} out of range")
                continue
            
            # Extract cross-attention for this subject
            ca_k = cross_attn[:, subj_idx]  # [tokens]
            
            # Apply Otsu threshold
            mask = torch_otsu_threshold(ca_k, use_2d=self.config.use_2d_otsu)
            
            # Aggregate self-attention: weighted by cross-attention and mask
            # Agg_Si = Σ_p self_attn(·,p) * ca_k(p) * mask(p)
            weighted_sa = self_attn * (ca_k.unsqueeze(0) * mask.unsqueeze(0))
            aggregated = weighted_sa.sum(dim=1)  # [tokens]
            
            aggregated_maps.append(aggregated)
        
        if len(aggregated_maps) < 2:
            logging.warning("[SelfCross] Insufficient subjects for loss computation")
            return torch.tensor(0.0, device=context.z_noisy.device, requires_grad=True)
        
        # Compute pairwise overlap
        loss = torch.tensor(0.0, device=context.z_noisy.device)
        
        for i in range(len(aggregated_maps)):
            for j in range(i + 1, len(aggregated_maps)):
                agg_i = aggregated_maps[i]
                cross_j = cross_attn[:, self.config.subject_indices[j]]
                
                # Overlap = min(Agg_Si, Cross_Sj)
                overlap = torch.min(agg_i, cross_j).sum()
                loss = loss + overlap
        
        # Scale by guidance_scale
        loss = loss * self.config.guidance_scale
        
        # Normalize by number of pairs
        num_pairs = len(aggregated_maps) * (len(aggregated_maps) - 1) / 2
        loss = loss / num_pairs
        
        return loss

# =================================================================================
# == Exports
# =================================================================================
__all__ = [
    'SelfCrossConfig',
    'torch_otsu_threshold',
    'AttentionStore',
    'AttentionHookManager',
    'SelfCrossLoss',
]