# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
# ████ MD_Nodes/utils/self_cross_utils.py – Self-Cross Specific Logic ████▓▒░
# © 2025 MDMAchine
# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
# ░▒▓ DESCRIPTION:
#   Utilities specific to Self-Cross Diffusion Guidance (Qiu et al., CVPR 2025).
#   Includes attention storage, Otsu thresholding, and overlap loss.
#
# ░▒▓ KEY FEATURE:
#   Audio-Safe 1D Otsu thresholding (no sqrt assumption).
#   Works for spectrograms (Time x Freq), images (H x W), and video.
# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄

import torch
import torch.nn.functional as F
import math
import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional

# Import base guidance infrastructure
from .guidance_core import GuidanceConfig, BaseGuidanceLoss, GuidanceContext

# =================================================================================
# == Self-Cross Configuration
# =================================================================================
@dataclass
class SelfCrossConfig(GuidanceConfig):
    """
    Configuration for Self-Cross Diffusion Guidance.
    Extends base GuidanceConfig with Self-Cross specific parameters.
    """
    # Subject identification
    subject_indices: List[int] = None  # Token indices for subjects (e.g., [2, 5])
    
    # Guidance strength
    guidance_scale: float = 10.0
    param_lambda: float = 0.3  # Weight for Attend-and-Excite term
    
    # Layer selection (performance critical)
    target_layers: Optional[List[str]] = None
    min_resolution: int = 8     # Audio-friendly (fewer freq bins)
    max_resolution: int = 256   # Audio-friendly (long time sequences)
    
    # Thresholding mode
    use_2d_otsu: bool = False   # False = 1D (audio-safe), True = 2D (images)
    threshold: float = 0.2      # Fallback if Otsu fails
    
    def __post_init__(self):
        super().__post_init__()
        if self.target_layers is None:
            self.target_layers = ["middle", "output_4", "output_5"]
        if self.subject_indices is None:
            self.subject_indices = []

# =================================================================================
# == Otsu Thresholding (Audio-Safe)
# =================================================================================
def torch_otsu_threshold(tensor: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """
    Fast Otsu thresholding on GPU (1D-safe for audio).
    Returns binary mask.
    """
    if tensor.numel() == 0:
        return torch.zeros_like(tensor, dtype=torch.bool)
    
    flat = tensor.flatten()
    
    if normalize:
        min_val, max_val = flat.min(), flat.max()
        if min_val == max_val:
            return torch.zeros_like(tensor, dtype=torch.bool)
        flat = (flat - min_val) / (max_val - min_val + 1e-8)
    
    scaled = (flat * 255).long().clamp(0, 255)
    hist = torch.bincount(scaled, minlength=256).float()
    bin_centers = torch.arange(256, device=tensor.device, dtype=torch.float32)
    
    weight1 = hist.cumsum(0)
    weight2 = hist.sum() - weight1
    weight1 = torch.clamp(weight1, min=1e-8)
    weight2 = torch.clamp(weight2, min=1e-8)
    
    mean1 = (hist * bin_centers).cumsum(0) / weight1
    mean2 = ((hist * bin_centers).sum() - (hist * bin_centers).cumsum(0)) / weight2
    
    variance = weight1 * weight2 * (mean1 - mean2) ** 2
    threshold_idx = variance.argmax()
    threshold_val = threshold_idx.float() / 255.0
    
    return tensor > threshold_val

# =================================================================================
# == Attention Storage
# =================================================================================
class AttentionStore:
    """Stores attention maps with filtering."""
    
    def __init__(self, config: SelfCrossConfig):
        self.config = config
        self.self_attn: List[torch.Tensor] = []
        self.cross_attn: List[torch.Tensor] = []
        self.layer_names: List[str] = []
    
    def add_self_attn(self, attn: torch.Tensor, layer_name: str):
        """Add self-attention map if it passes filters."""
        tokens = attn.shape[-2]
        if tokens < (self.config.min_resolution ** 2):
            return
        if tokens > (self.config.max_resolution ** 2):
            return
        if self.config.target_layers:
            if not any(t in layer_name for t in self.config.target_layers):
                return
        self.self_attn.append(attn.detach())
        self.layer_names.append(layer_name)
    
    def add_cross_attn(self, attn: torch.Tensor, layer_name: str):
        """Add cross-attention map with same filtering."""
        tokens = attn.shape[-2]
        if tokens < (self.config.min_resolution ** 2):
            return
        if tokens > (self.config.max_resolution ** 2):
            return
        if self.config.target_layers:
            if not any(t in layer_name for t in self.config.target_layers):
                return
        self.cross_attn.append(attn.detach())
    
    def reset(self):
        """Clear storage (call between timesteps)."""
        self.self_attn.clear()
        self.cross_attn.clear()
        self.layer_names.clear()
    
    def get_best_layer(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Returns the most semantic layer pair (last = deepest)."""
        if not self.self_attn or not self.cross_attn:
            return None, None
        return self.self_attn[-1], self.cross_attn[-1]

# =================================================================================
# == Hook Manager
# =================================================================================
class AttentionHookManager:
    """Manages hook injection into ComfyUI's model patcher."""
    
    def __init__(self, model_patcher, attn_store: AttentionStore):
        self.model_patcher = model_patcher
        self.attn_store = attn_store
        self.hooks_active = False
    
    def inject_hooks(self):
        """Inject attention capture hooks into model."""
        if self.hooks_active:
            return
        
        def attn1_hook(q, k, v, extra_options):
            """Self-attention hook (Attn1)."""
            try:
                layer_name = extra_options.get("block", "unknown")
                scale = q.shape[-1] ** -0.5
                sim = torch.bmm(q, k.transpose(-2, -1)) * scale
                attn_map = F.softmax(sim, dim=-1)
                self.attn_store.add_self_attn(attn_map, layer_name)
            except Exception:
                pass
            return q, k, v
        
        def attn2_hook(q, k, v, extra_options):
            """Cross-attention hook (Attn2)."""
            try:
                layer_name = extra_options.get("block", "unknown")
                scale = q.shape[-1] ** -0.5
                sim = torch.bmm(q, k.transpose(-2, -1)) * scale
                attn_map = F.softmax(sim, dim=-1)
                self.attn_store.add_cross_attn(attn_map, layer_name)
            except Exception:
                pass
            return q, k, v
        
        self.model_patcher.set_model_attn1_patch(attn1_hook)
        self.model_patcher.set_model_attn2_patch(attn2_hook)
        self.hooks_active = True
    
    def remove_hooks(self):
        """Clean up hooks (prevents memory leaks)."""
        if not self.hooks_active:
            return
        
        def noop(q, k, v, extra_options):
            return q, k, v
        
        self.model_patcher.set_model_attn1_patch(noop)
        self.model_patcher.set_model_attn2_patch(noop)
        self.hooks_active = False

# =================================================================================
# == Self-Cross Loss (Audio-Safe)
# =================================================================================
class SelfCrossLoss(BaseGuidanceLoss):
    """
    Implements Self-Cross overlap loss from Qiu et al. (CVPR 2025).
    Audio-compatible via 1D Otsu thresholding.
    """
    
    def __init__(self, config: SelfCrossConfig, attn_store: AttentionStore, 
                 latent_shape: Optional[Tuple[int, ...]] = None):
        self.config = config
        self.attn_store = attn_store
        self.latent_shape = latent_shape
        self._aspect_ratio = None
    
    def _infer_aspect_ratio(self, num_tokens: int) -> Optional[Tuple[int, int]]:
        """Infer (H, W) from token count."""
        if self._aspect_ratio is not None:
            return self._aspect_ratio
        
        if self.latent_shape is not None and len(self.latent_shape) >= 4:
            h, w = self.latent_shape[2], self.latent_shape[3]
            if h * w == num_tokens:
                self._aspect_ratio = (h, w)
                return self._aspect_ratio
        
        sqrt = math.sqrt(num_tokens)
        if sqrt == int(sqrt):
            side = int(sqrt)
            self._aspect_ratio = (side, side)
            return self._aspect_ratio
        
        return None
    
    def compute(self, context: GuidanceContext) -> torch.Tensor:
        """Compute Self-Cross overlap loss."""
        # Get attention maps from store
        s_map, c_map = self.attn_store.get_best_layer()
        
        if s_map is None or c_map is None:
            return torch.tensor(0.0, device=context.z_noisy.device)
        
        # Handle batch/head dimensions
        if s_map.dim() > 2:
            s_map = s_map[-1] if s_map.dim() == 3 else s_map.mean(dim=1)[-1]
        if c_map.dim() > 2:
            c_map = c_map[-1] if c_map.dim() == 3 else c_map.mean(dim=1)[-1]
        
        device = s_map.device
        num_tokens = c_map.shape[0]
        aspect_ratio = self._infer_aspect_ratio(num_tokens)
        
        # Aggregate self-attention for each subject
        agg_maps = {}
        
        for idx in self.config.subject_indices:
            if idx >= c_map.shape[-1]:
                continue
            
            ca_k = c_map[:, idx]
            
            # ═══════════════════════════════════════════════════════
            # AUDIO FIX: 1D Otsu by default (Universal Mode)
            # ═══════════════════════════════════════════════════════
            if aspect_ratio is not None and self.config.use_2d_otsu:
                try:
                    h, w = aspect_ratio
                    ca_2d = ca_k.view(h, w)
                    mask = torch_otsu_threshold(ca_2d, normalize=True).flatten().float()
                except RuntimeError:
                    mask = torch_otsu_threshold(ca_k, normalize=True).float()
            else:
                mask = torch_otsu_threshold(ca_k, normalize=True).float()
            # ═══════════════════════════════════════════════════════
            
            if mask.sum() < 1:
                agg_maps[idx] = torch.zeros_like(ca_k)
                continue
            
            weights = ca_k * mask
            weighted_sa = s_map * weights.unsqueeze(-1)
            agg_map = weighted_sa.sum(dim=0) / (weights.sum() + 1e-8)
            agg_maps[idx] = agg_map
        
        # Compute pairwise overlap loss
        total_loss = torch.tensor(0.0, device=device)
        pairs = 0
        
        indices = list(agg_maps.keys())
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                idx_i, idx_j = indices[i], indices[j]
                overlap_ij = torch.min(agg_maps[idx_i], c_map[:, idx_j]).sum()
                overlap_ji = torch.min(agg_maps[idx_j], c_map[:, idx_i]).sum()
                total_loss += (overlap_ij + overlap_ji)
                pairs += 1
        
        if pairs > 0:
            total_loss = total_loss / pairs
        
        return total_loss

# =================================================================================
# == Export
# =================================================================================
__all__ = [
    'SelfCrossConfig',
    'torch_otsu_threshold',
    'AttentionStore',
    'AttentionHookManager',
    'SelfCrossLoss',
]