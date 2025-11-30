# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
# ████ MD_Nodes/utils/guidance_core – Universal Guidance Engine ████▓▒░
# © 2025 MDMAchine
# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
# ░▒▓ DESCRIPTION:
#   Universal optimization engine for diffusion guidance.
#   Separates optimization mechanics (HOW) from loss computation (WHAT).
#
# ░▒▓ CHANGELOG:
#   - v1.1.0 (Current - CRITICAL FIX):
#       FIXED: BaseGuidanceLoss now has proper __init__ accepting config
# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄

import torch
import logging
from dataclasses import dataclass
from typing import Optional, Callable, Dict, Any
from abc import ABC, abstractmethod

# =================================================================================
# == Configuration Base Class
# =================================================================================
@dataclass
class GuidanceConfig:
    """Base configuration for guidance methods."""
    max_iters: int = 3
    learning_rate: float = 0.01
    loss_threshold: float = 1e-4
    gradient_clip: float = 1.0
    adaptive_lr: bool = True  # Scale LR by sigma

# =================================================================================
# == Guidance Context
# =================================================================================
@dataclass
class GuidanceContext:
    """Data container passed to loss functions."""
    z_noisy: torch.Tensor
    sigma: torch.Tensor
    noise_pred: Optional[torch.Tensor] = None
    x_0_pred: Optional[torch.Tensor] = None
    attention_maps: Optional[Dict[str, torch.Tensor]] = None
    extras: Optional[Dict[str, Any]] = None

# =================================================================================
# == Base Loss Class
# =================================================================================
class BaseGuidanceLoss(ABC):
    """Abstract base class for guidance loss functions."""
    
    def __init__(self, config: GuidanceConfig):
        """
        Initialize loss function with configuration.
        
        Args:
            config: Guidance configuration (subclass of GuidanceConfig)
        """
        self.config = config
    
    @abstractmethod
    def compute(self, context: GuidanceContext) -> torch.Tensor:
        """
        Compute guidance loss.
        
        Args:
            context: Current optimization context
        
        Returns:
            Scalar loss tensor with gradient
        """
        pass

# =================================================================================
# == Universal Optimizer
# =================================================================================
class GuidanceOptimizer:
    """Universal optimization loop for diffusion guidance."""
    
    def __init__(self, config: GuidanceConfig):
        self.config = config
    
    def optimize_latent(
        self,
        z: torch.Tensor,
        sigma: torch.Tensor,
        loss_fn: BaseGuidanceLoss,
        model_forward: Optional[Callable] = None,
        populate_context: Optional[Callable[[GuidanceContext], None]] = None
    ) -> torch.Tensor:
        """
        Universal optimization loop.
        
        Args:
            z: Noisy latent to optimize
            sigma: Current noise level
            loss_fn: Loss function to minimize
            model_forward: Optional model forward pass function
            populate_context: Optional callback to populate context (e.g., capture attention)
        
        Returns:
            Optimized latent
        """
        if self.config.max_iters == 0:
            return z
        
        # Enable gradients
        z_opt = z.detach().clone().requires_grad_(True)
        
        # Adaptive learning rate
        if self.config.adaptive_lr and sigma is not None:
            lr = self.config.learning_rate * sigma.item() if sigma.numel() == 1 else self.config.learning_rate
        else:
            lr = self.config.learning_rate
        
        for iteration in range(self.config.max_iters):
            # Build context
            context = GuidanceContext(
                z_noisy=z_opt,
                sigma=sigma
            )
            
            # Optional: Run model forward pass
            if model_forward is not None:
                with torch.enable_grad():
                    noise_pred = model_forward(z_opt, sigma)
                    context.noise_pred = noise_pred
            
            # Optional: Populate context (e.g., capture attention maps)
            if populate_context is not None:
                populate_context(context)
            
            # Compute loss
            loss = loss_fn.compute(context)
            
            if not loss.requires_grad:
                logging.warning(f"[Guidance] Loss has no gradient at iter {iteration}")
                break
            
            # Compute gradients
            grad = torch.autograd.grad(loss, z_opt, create_graph=False)[0]
            
            # Clip gradients
            if self.config.gradient_clip > 0:
                grad = torch.clamp(grad, -self.config.gradient_clip, self.config.gradient_clip)
            
            # Update latent
            with torch.no_grad():
                z_opt = z_opt - lr * grad
                z_opt.requires_grad_(True)
            
            # Logging
            logging.info(f"[Guidance] Iter {iteration+1}/{self.config.max_iters}: loss={loss.item():.6f}")
            
            # Early stopping
            if loss.item() < self.config.loss_threshold:
                logging.info(f"[Guidance] Early stop at iter {iteration+1}")
                break
        
        return z_opt.detach()

# =================================================================================
# == Exports
# =================================================================================
__all__ = [
    'GuidanceConfig',
    'GuidanceContext',
    'BaseGuidanceLoss',
    'GuidanceOptimizer',
]