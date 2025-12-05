# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
# ████ MD_Nodes/Workflow – Universal Context Bus ███████████████████▓▒░
# © 2025 MDMAchine
# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
# ░▒▓ ORIGIN & DEV:
#   • Cast into the void by: MDMAchine
#   • Enhanced by: Gemini
#   • License: Apache 2.0
#
# ░▒▓ DESCRIPTION:
#   The ultimate "Cable Bundle" for ComfyUI.
#   Carries 18 different data types in a single CONTEXT connection.
#   Allows "Bus" style workflow organization (plug in once, use everywhere).
#
# ░▒▓ FEATURES:
#   ✓ Core: Model, CLIP, VAE
#   ✓ Conditioning: Positive, Negative
#   ✓ Advanced Sampling: Sampler, Sigmas, Guider, Noise
#   ✓ Data: Latent, Image, Mask
#   ✓ Params: Seed, Steps, CFG
#   ✓ Utilities: 2x Generic Strings (for paths/notes)
#   ✓ JS-Safe Seed Capping
#
# ░▒▓ CHANGELOG:
#   - v1.5.0 (Enterprise Universal):
#       • ADDED: Noise, Guider, Sampler, Sigmas inputs/outputs
#       • ADDED: String_A and String_B for generic text passing
#       • REFACTOR: Dynamic mapping loop for reliable I/O handling

# =================================================================================
# == Standard Library Imports
# =================================================================================
import logging

# =================================================================================
# == Configuration Constants
# =================================================================================

CONST_JS_MAX_SAFE_INTEGER = 9007199254740991
CONST_SEED_MIN = 0

# -- Context Keys (Central Definition) --
# This ensures keys are identical across all context operations
CONST_CTX_MODEL = "model"
CONST_CTX_CLIP = "clip"
CONST_CTX_VAE = "vae"

CONST_CTX_POS = "positive"
CONST_CTX_NEG = "negative"

CONST_CTX_SAMPLER = "sampler"
CONST_CTX_SIGMAS = "sigmas"
CONST_CTX_GUIDER = "guider"
CONST_CTX_NOISE = "noise"

CONST_CTX_LATENT = "latent"
CONST_CTX_IMAGE = "image"
CONST_CTX_MASK = "mask"

CONST_CTX_SEED = "seed"
CONST_CTX_STEPS = "steps"
CONST_CTX_CFG = "cfg"

CONST_CTX_STR_A = "string_A"
CONST_CTX_STR_B = "string_B"

# =================================================================================
# == Utility Functions
# =================================================================================

def validate_seed(seed_value):
    """Ensure seed is within safe range if it exists."""
    if seed_value is None:
        return None
    try:
        val = int(seed_value)
    except (ValueError, TypeError):
        return CONST_SEED_MIN
    return max(CONST_SEED_MIN, min(val, CONST_JS_MAX_SAFE_INTEGER))

# =================================================================================
# == Core Node Class
# =================================================================================

class MD_WorkflowContextBus:
    """
    Universal Context Bus.
    Aggregates inputs into a dictionary and outputs them individually.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "base_context": ("MD_CONTEXT", {"tooltip": "Incoming Context bundle (optional)"}),
                
                # -- Core --
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                
                # -- Conditioning --
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                
                # -- Advanced Sampling --
                "sampler": ("SAMPLER",),
                "sigmas": ("SIGMAS",),
                "guider": ("GUIDER",),
                "noise": ("NOISE",),
                
                # -- Data --
                "latent": ("LATENT",),
                "image": ("IMAGE",),
                "mask": ("MASK",),
                
                # -- Parameters --
                "seed": ("INT", {
                    "default": 0, "min": CONST_SEED_MIN, "max": CONST_JS_MAX_SAFE_INTEGER,
                    "forceInput": True 
                }),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000, "forceInput": True}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1, "forceInput": True}),
                
                # -- Utilities --
                "string_A": ("STRING", {"multiline": False, "forceInput": True}),
                "string_B": ("STRING", {"multiline": False, "forceInput": True}),
            }
        }

    # Definition of output order - CRITICAL to match process_context return
    RETURN_TYPES = (
        "MD_CONTEXT", 
        "MODEL", "CLIP", "VAE", 
        "CONDITIONING", "CONDITIONING",
        "SAMPLER", "SIGMAS", "GUIDER", "NOISE",
        "LATENT", "IMAGE", "MASK", 
        "INT", "INT", "FLOAT", 
        "STRING", "STRING"
    )
    
    RETURN_NAMES = (
        "CONTEXT", 
        "model", "clip", "vae", 
        "positive", "negative", 
        "sampler", "sigmas", "guider", "noise",
        "latent", "image", "mask", 
        "seed", "steps", "cfg", 
        "string_A", "string_B"
    )
    
    FUNCTION = "process_context"
    CATEGORY = "MD_Nodes/Workflow"

    def process_context(self, base_context=None, **kwargs):
        """
        Dynamically updates context and returns strictly ordered values.
        """
        
        # 1. Initialize Context
        current_ctx = base_context.copy() if base_context else {}

        # 2. Define Mapping: Kwarg Name -> Context Key
        # This list defines the exact order of processing and RETURN TYPES (after the Context object itself)
        # Structure: (Input Name, Context Key, Default Value)
        # Note: Default Value is used ONLY for fallback if not in context. 
        # Most objects default to None.
        
        param_map = [
            # Core
            ("model", CONST_CTX_MODEL, None),
            ("clip", CONST_CTX_CLIP, None),
            ("vae", CONST_CTX_VAE, None),
            
            # Cond
            ("positive", CONST_CTX_POS, None),
            ("negative", CONST_CTX_NEG, None),
            
            # Adv Sampling
            ("sampler", CONST_CTX_SAMPLER, None),
            ("sigmas", CONST_CTX_SIGMAS, None),
            ("guider", CONST_CTX_GUIDER, None),
            ("noise", CONST_CTX_NOISE, None),
            
            # Data
            ("latent", CONST_CTX_LATENT, None),
            ("image", CONST_CTX_IMAGE, None),
            ("mask", CONST_CTX_MASK, None),
            
            # Params
            ("seed", CONST_CTX_SEED, 0),
            ("steps", CONST_CTX_STEPS, 20),
            ("cfg", CONST_CTX_CFG, 8.0),
            
            # Utils
            ("string_A", CONST_CTX_STR_A, ""),
            ("string_B", CONST_CTX_STR_B, ""),
        ]

        # 3. Update Logic
        for input_name, ctx_key, _ in param_map:
            input_val = kwargs.get(input_name)
            
            # If User connected an input, update the context
            if input_val is not None:
                # Seed Safety Check
                if ctx_key == CONST_CTX_SEED:
                    input_val = validate_seed(input_val)
                    
                current_ctx[ctx_key] = input_val

        # 4. Prepare Ordered Return Tuple
        # Start with the Context object
        return_values = [current_ctx]
        
        # Iterate mapping to pull values in correct order
        for _, ctx_key, default_val in param_map:
            # Get from context, or use default if missing
            val = current_ctx.get(ctx_key, default_val)
            return_values.append(val)

        return tuple(return_values)

# =================================================================================
# == Node Registration
# =================================================================================

NODE_CLASS_MAPPINGS = {
    "MD_WorkflowContextBus": MD_WorkflowContextBus
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MD_WorkflowContextBus": "MD: Universal Context Bus"
}

# =================================================================================
# == Development & Testing
# =================================================================================

if __name__ == "__main__":
    print("🧪 Running Self-Tests for MD_WorkflowContextBus...")
    
    try:
        node = MD_WorkflowContextBus()
        
        # Test 1: Advanced Sampling Data
        mock_noise = "NOISE_OBJ"
        mock_guider = "GUIDER_OBJ"
        
        # Simulate inputting noise/guider
        res = node.process_context(
            base_context=None, 
            noise=mock_noise, 
            guider=mock_guider,
            seed=123
        )
        
        ctx_out = res[0]
        # Indexes: 
        # 0:Ctx, 1:Model, 2:Clip, 3:Vae, 4:Pos, 5:Neg, 
        # 6:Sampler, 7:Sigmas, 8:Guider, 9:Noise
        
        guider_out = res[8]
        noise_out = res[9]
        
        assert ctx_out[CONST_CTX_NOISE] == mock_noise
        assert noise_out == mock_noise
        assert guider_out == mock_guider
        print("✅ Advanced Types (Noise/Guider): PASSED")
        
        # Test 2: Passthrough
        # If we pass the previous context but don't add inputs, they should persist
        res_2 = node.process_context(base_context=ctx_out)
        assert res_2[9] == mock_noise
        print("✅ Context Passthrough: PASSED")
        
    except Exception as e:
        print(f"❌ Test Failed: {e}")