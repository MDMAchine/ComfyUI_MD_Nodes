# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
# ████ MD_Nodes/utils/guidance_core.py – Universal Guidance Engine ████▓▒░
# © 2025 MDMAchine
# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
# ░▒▓ DESCRIPTION:
#   Core optimization infrastructure for guided diffusion sampling.
#   Implements iterative latent refinement for:
#     • Self-Cross Guidance (Qiu et al., CVPR 2025)
#     • Universal Guidance (Bansal et al., CVPR 2023) [FUTURE]
#     • Custom guidance methods
#
# ░▒▓ ARCHITECTURE:
#   Separates "How to optimize" (this file) from "What to optimize" (loss functions).
#   Loss functions subclass BaseGuidanceLoss and implement compute().
# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄

import torch
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Callable

# =================================================================================
# == Base Configuration
# =================================================================================
@dataclass
class GuidanceConfig:
    """
    Universal configuration for iterative guidance optimization.
    All guidance methods inherit from this.
    """
    # Optimization parameters
    max_iters: int = 3              # Number of refinement steps per timestep
    learning_rate: float = 0.01     # Step size for gradient descent
    loss_threshold: float = 1e-4    # Stop if loss < this (early stopping)
    gradient_clip: float = 1.0      # Clip gradients to prevent explosion
    
    # Adaptive learning rate (optional)
    use_adaptive_lr: bool = False   # Scale LR by sigma (noise level)
    lr_scale_factor: float = 1.0    # Multiplier for adaptive LR
    
    # Logging
    verbose: bool = False           # Log per-iteration loss values
    log_prefix: str = "[Guidance]"  # Prefix for log messages

# =================================================================================
# == Guidance Context (Data Container)
# =================================================================================
@dataclass
class GuidanceContext:
    """
    Container for all data passed to loss functions.
    Allows loss functions to access whatever they need.
    """
    # Required fields
    z_noisy: torch.Tensor           # Current noisy latent [B, C, H, W] or [B, C, T, F]
    sigma: torch.Tensor             # Current noise level (scalar or [B])
    timestep: Optional[int] = None  # Current diffusion timestep (optional)
    
    # Model outputs (optional, populated by optimizer)
    noise_pred: Optional[torch.Tensor] = None       # ε_θ(z_t, t, c)
    x_0_pred: Optional[torch.Tensor] = None         # Predicted clean sample
    v_pred: Optional[torch.Tensor] = None           # Velocity prediction (SD3, etc.)
    
    # Attention maps (optional, for attention-based guidance)
    self_attn_maps: Optional[Dict[str, torch.Tensor]] = None
    cross_attn_maps: Optional[Dict[str, torch.Tensor]] = None
    
    # Custom fields (extensible)
    extras: Dict[str, Any] = field(default_factory=dict)

# =================================================================================
# == Abstract Base Loss
# =================================================================================
class BaseGuidanceLoss(ABC):
    """
    Abstract base class for all guidance loss functions.
    
    Subclass this to implement custom guidance methods.
    The optimizer calls compute() in each iteration.
    """
    
    @abstractmethod
    def compute(self, context: GuidanceContext) -> torch.Tensor:
        """
        Compute the guidance loss for the current latent state.
        
        Args:
            context: GuidanceContext with z_noisy, sigma, model outputs, etc.
        
        Returns:
            Scalar loss tensor (must support .backward())
        """
        pass
    
    def pre_optimization_hook(self, context: GuidanceContext) -> None:
        """
        Optional hook called BEFORE optimization starts.
        Useful for setup (e.g., initializing buffers).
        """
        pass
    
    def post_optimization_hook(self, context: GuidanceContext) -> None:
        """
        Optional hook called AFTER optimization completes.
        Useful for cleanup or logging final state.
        """
        pass

# =================================================================================
# == Universal Guidance Optimizer
# =================================================================================
class GuidanceOptimizer:
    """
    Universal optimization loop for guided diffusion.
    
    Powers Self-Cross, Universal Classifier Guidance, and future methods.
    Handles gradient computation, clipping, adaptive LR, and early stopping.
    """
    
    def __init__(self, config: GuidanceConfig):
        self.config = config
    
    def optimize_latent(
        self,
        z: torch.Tensor,
        sigma: torch.Tensor,
        loss_fn: BaseGuidanceLoss,
        model_forward: Optional[Callable] = None,
        timestep: Optional[int] = None,
        populate_context: Optional[Callable] = None,
    ) -> torch.Tensor:
        """
        Optimize a noisy latent via iterative gradient descent.
        
        Args:
            z: Initial noisy latent [B, C, H, W]
            sigma: Noise level (scalar or [B])
            loss_fn: Guidance loss function (subclass of BaseGuidanceLoss)
            model_forward: Optional function to run model (for x_0 prediction)
            timestep: Optional timestep index
            populate_context: Optional function to add custom data to context
        
        Returns:
            Optimized latent (same shape as z)
        
        Algorithm:
            for iter in range(max_iters):
                1. Create context (with model outputs if model_forward provided)
                2. Compute loss = loss_fn.compute(context)
                3. Backprop: grad = ∂loss/∂z
                4. Update: z = z - lr * grad
                5. Check early stopping
        """
        # Enable gradients on latent
        z_opt = z.detach().clone().requires_grad_(True)
        
        # Pre-optimization hook
        initial_context = GuidanceContext(z_noisy=z_opt, sigma=sigma, timestep=timestep)
        loss_fn.pre_optimization_hook(initial_context)
        
        # Optimization loop
        for iter_idx in range(self.config.max_iters):
            
            # ═══════════════════════════════════════════════════════
            # STEP 1: Build Context
            # ═══════════════════════════════════════════════════════
            context = GuidanceContext(
                z_noisy=z_opt,
                sigma=sigma,
                timestep=timestep
            )
            
            # Optional: Run model to populate predictions
            if model_forward is not None:
                with torch.enable_grad():
                    model_output = model_forward(z_opt, sigma)
                    
                    # Parse model output (format varies by sampler)
                    if isinstance(model_output, dict):
                        context.noise_pred = model_output.get('noise_pred')
                        context.x_0_pred = model_output.get('x_0_pred')
                        context.v_pred = model_output.get('v_pred')
                    else:
                        # Assume it's noise prediction
                        context.noise_pred = model_output
            
            # Optional: Custom context population (e.g., attention maps)
            if populate_context is not None:
                populate_context(context)
            
            # ═══════════════════════════════════════════════════════
            # STEP 2: Compute Loss
            # ═══════════════════════════════════════════════════════
            with torch.enable_grad():
                loss = loss_fn.compute(context)
                
                if self.config.verbose:
                    logging.info(f"{self.config.log_prefix} Iter {iter_idx+1}/{self.config.max_iters}: loss={loss.item():.6f}")
                
                # Early stopping
                if loss.item() < self.config.loss_threshold:
                    if self.config.verbose:
                        logging.info(f"{self.config.log_prefix} Early stop (loss < {self.config.loss_threshold})")
                    break
                
                # ═══════════════════════════════════════════════════════
                # STEP 3: Backprop
                # ═══════════════════════════════════════════════════════
                grad = torch.autograd.grad(
                    outputs=loss,
                    inputs=z_opt,
                    retain_graph=False,
                    create_graph=False
                )[0]
                
                # Gradient clipping (prevent instability)
                if self.config.gradient_clip > 0:
                    grad = torch.clamp(
                        grad,
                        -self.config.gradient_clip,
                        self.config.gradient_clip
                    )
                
                # ═══════════════════════════════════════════════════════
                # STEP 4: Update Latent
                # ═══════════════════════════════════════════════════════
                with torch.no_grad():
                    # Compute learning rate (adaptive or fixed)
                    if self.config.use_adaptive_lr:
                        # Scale LR by noise level (higher noise = larger steps)
                        sigma_val = sigma[0].item() if sigma.numel() > 1 else sigma.item()
                        lr = self.config.learning_rate * sigma_val * self.config.lr_scale_factor
                    else:
                        lr = self.config.learning_rate
                    
                    # Gradient descent step
                    z_opt = z_opt - lr * grad
                    
                    # Re-enable grad for next iteration
                    z_opt = z_opt.detach().requires_grad_(True)
        
        # Post-optimization hook
        final_context = GuidanceContext(z_noisy=z_opt.detach(), sigma=sigma, timestep=timestep)
        loss_fn.post_optimization_hook(final_context)
        
        return z_opt.detach()

# =================================================================================
# == Export Public API
# =================================================================================
__all__ = [
    'GuidanceConfig',
    'GuidanceContext',
    'BaseGuidanceLoss',
    'GuidanceOptimizer',
]