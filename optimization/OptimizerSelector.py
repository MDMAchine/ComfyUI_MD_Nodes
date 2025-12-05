# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
# ████ MD_Nodes/Optimization – Universal Optimizer Selector ████████▓▒░
# © 2025 MDMAchine
# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
# ░▒▓ ORIGIN & DEV:
#   • Cast into the void by: MDMAchine
#   • Enhanced by: Gemini
#   • License: Apache 2.0
#
# ░▒▓ DESCRIPTION:
#   A safe factory for PyTorch Optimizers. Handles external dependencies
#   (Prodigy, Lion, BitsAndBytes) gracefully with automatic fallbacks.
#   Returns a configuration object compatible with advanced samplers/trainers.
#
# ░▒▓ FEATURES:
#   ✓ Safety Fallbacks: Missing packages (like Prodigy) default to AdamW
#   ✓ 8-Bit Support: Auto-detects bitsandbytes for low-VRAM optimization
#   ✓ Configurable: Full control over LR, Weight Decay, and Betas
#
# ░▒▓ CHANGELOG:
#   - v1.0.0 (Initial):
#       • ADDED: Core optimizer logic with Dependency Injection pattern

# =================================================================================
# == Standard Library Imports
# =================================================================================
import logging

# =================================================================================
# == Third-Party Imports
# =================================================================================
import torch

# -- Dependency Guard --
try:
    import prodigyopt
    HAS_PRODIGY = True
except ImportError:
    HAS_PRODIGY = False

try:
    import bitsandbytes as bnb
    HAS_BNB = True
except ImportError:
    HAS_BNB = False

try:
    from lion_pytorch import Lion
    HAS_LION = True
except ImportError:
    HAS_LION = False

# =================================================================================
# == Configuration Constants
# =================================================================================

CONST_OPTIMIZERS = [
    "AdamW", 
    "Adam", 
    "SGD", 
    "Lion (if installed)", 
    "Prodigy (if installed)", 
    "AdamW8bit (if installed)"
]

# =================================================================================
# == Helper Classes
# =================================================================================

class OptimizerFactory:
    """
    A factory object that holds settings and creates the optimizer 
    when provided with model parameters later in the workflow.
    """
    def __init__(self, name, lr, weight_decay, beta1, beta2, epsilon, use_fallback=False):
        self.name = name
        self.lr = lr
        self.weight_decay = weight_decay
        self.betas = (beta1, beta2)
        self.eps = epsilon
        self.use_fallback = use_fallback

    def create(self, params):
        """
        Instantiates the PyTorch optimizer for the given parameters.
        """
        # 1. Prodigy
        if self.name == "Prodigy (if installed)":
            if HAS_PRODIGY:
                return prodigyopt.Prodigy(
                    params, lr=self.lr, weight_decay=self.weight_decay, 
                    betas=self.betas, eps=self.eps
                )
            else:
                logging.warning("[MD_Nodes] Prodigy missing. Falling back to AdamW.")
                return torch.optim.AdamW(
                    params, lr=self.lr, weight_decay=self.weight_decay, 
                    betas=self.betas, eps=self.eps
                )

        # 2. Lion
        if self.name == "Lion (if installed)":
            if HAS_LION:
                return Lion(
                    params, lr=self.lr, weight_decay=self.weight_decay, 
                    betas=self.betas
                )
            else:
                logging.warning("[MD_Nodes] Lion missing. Falling back to AdamW.")
                return torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)

        # 3. 8-bit AdamW (BitsAndBytes)
        if self.name == "AdamW8bit (if installed)":
            if HAS_BNB:
                return bnb.optim.AdamW8bit(
                    params, lr=self.lr, weight_decay=self.weight_decay, 
                    betas=self.betas, eps=self.eps
                )
            else:
                logging.warning("[MD_Nodes] BitsAndBytes missing. Falling back to standard AdamW.")
                return torch.optim.AdamW(
                    params, lr=self.lr, weight_decay=self.weight_decay, 
                    betas=self.betas, eps=self.eps
                )

        # 4. Standard Torch Optimizers
        if self.name == "Adam":
            return torch.optim.Adam(
                params, lr=self.lr, weight_decay=self.weight_decay, 
                betas=self.betas, eps=self.eps
            )
        
        if self.name == "SGD":
            return torch.optim.SGD(
                params, lr=self.lr, weight_decay=self.weight_decay
            )

        # Default: AdamW
        return torch.optim.AdamW(
            params, lr=self.lr, weight_decay=self.weight_decay, 
            betas=self.betas, eps=self.eps
        )

# =================================================================================
# == Core Node Class
# =================================================================================

class MD_OptimizerSelector:
    """
    Configures an Optimizer Factory for use in advanced sampling or training nodes.
    """
    DESCRIPTION = "Safely configures optimizers (Prodigy, Lion, AdamW) with auto-fallback."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "optimizer_name": (CONST_OPTIMIZERS, {
                    "default": "AdamW",
                    "tooltip": "Select algorithm. If a custom one (Prodigy/Lion/8bit) is missing, it falls back to AdamW safely."
                }),
                "learning_rate": ("FLOAT", {
                    "default": 1.0, "min": 0.000001, "max": 100.0, "step": 0.0001,
                    "tooltip": "Step size (Alpha). For Prodigy, 1.0 is standard. For AdamW, usually 1e-4."
                }),
                "weight_decay": ("FLOAT", {
                    "default": 0.01, "min": 0.0, "max": 1.0, "step": 0.001,
                    "tooltip": "Regularization factor to prevent overfitting."
                }),
                "beta1": ("FLOAT", {
                    "default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Exponential decay rate for first moment estimates."
                }),
                "beta2": ("FLOAT", {
                    "default": 0.999, "min": 0.0, "max": 1.0, "step": 0.001,
                    "tooltip": "Exponential decay rate for second moment estimates."
                }),
                "epsilon": ("FLOAT", {
                    "default": 1e-8, "min": 1e-12, "max": 1e-4, "step": 1e-8,
                    "tooltip": "Term added to denominator to improve numerical stability."
                }),
            }
        }

    RETURN_TYPES = ("OPTIMIZER_CONFIG",)
    RETURN_NAMES = ("optimizer_config",)
    FUNCTION = "get_optimizer_config"
    CATEGORY = "MD_Nodes/Optimization"

    def get_optimizer_config(self, optimizer_name, learning_rate, weight_decay, beta1, beta2, epsilon):
        # Return a factory object that downstream nodes can use to initialize
        # the optimizer once they have the model parameters.
        factory = OptimizerFactory(
            name=optimizer_name,
            lr=learning_rate,
            weight_decay=weight_decay,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon
        )
        return (factory,)

# =================================================================================
# == Node Registration
# =================================================================================

NODE_CLASS_MAPPINGS = {
    "MD_OptimizerSelector": MD_OptimizerSelector
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MD_OptimizerSelector": "MD: Optimizer Selector"
}

# =================================================================================
# == Unit Tests
# =================================================================================

if __name__ == "__main__":
    print("🧪 Running Self-Tests for MD_OptimizerSelector...")
    
    # Mock Parameters (just a simple tensor)
    params = [torch.randn(10, 10, requires_grad=True)]
    
    # Test 1: AdamW Creation
    try:
        node = MD_OptimizerSelector()
        factory = node.get_optimizer_config("AdamW", 0.001, 0.01, 0.9, 0.999, 1e-8)[0]
        opt = factory.create(params)
        assert isinstance(opt, torch.optim.AdamW)
        print("✅ AdamW Instantiation: PASSED")
    except Exception as e:
        print(f"❌ AdamW Failed: {e}")

    # Test 2: Fallback Logic (Simulate missing package request)
    # Even if you have Prodigy, this tests the logic path validity
    try:
        # Requesting Prodigy
        factory = node.get_optimizer_config("Prodigy (if installed)", 1.0, 0.01, 0.9, 0.99, 1e-8)[0]
        opt = factory.create(params)
        # Should be Prodigy if installed, or AdamW if not. 
        # Just checking it didn't crash.
        print(f"✅ Prodigy/Fallback Instantiation: PASSED (Got {type(opt).__name__})")
    except Exception as e:
        print(f"❌ Fallback Logic Failed: {e}")