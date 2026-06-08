# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# █▓▒░                                                                     ░▒▓█
# █▓▒░  MD_Nodes/MD_ApplyTPG – Token Perturbation Guidance Patcher v1.6.1  ░▒▓█
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
# ║ ░▒▓ ORIGIN: Token Perturbation Guidance for Diffusion Models (Rajabi et al., 2025)
# ║ ░▒▓ DESCRIPTION:
# ║    Patches the Diffusion Model to implement Token Perturbation Guidance (TPG).
# ║    Intercepts Self-Attention (Attn1) layers during the Unconditional pass
# ║    and shuffles tokens to create a robust "Negative Score".
# ╚════════════════════════════════════════════════════════════════════════════


VERSION = "v1.6.1"  # UPS v1.5.8

import os
import sys
import logging
import torch
import comfy.model_management

# =================================================================================
# == MD_Nodes Universal Binary Loader (v1.6.1)
# =================================================================================

def find_core_paths():
    """Locate core directory in dev and production environments."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(current_dir, "core"),
        os.path.join(current_dir, "..", "core"),
        os.path.join(current_dir, "..", "..", "core")
    ]
    return [os.path.abspath(c) for c in candidates if os.path.exists(c)]

CORE_LOCATIONS = find_core_paths()
CORE_LOADED = False
CORE_MODE = None
CORE_ERROR = None

for loc in CORE_LOCATIONS:
    if loc not in sys.path:
        sys.path.insert(0, loc)

try:
    import apply_tpg_core_bin as core
    CORE_LOADED = True
    CORE_MODE = "Binary (Production)"
except ImportError as e1:
    try:
        import apply_tpg_core as core
        CORE_LOADED = True
        CORE_MODE = "Source (Development)"
    except ImportError as e2:
        CORE_ERROR = f"Binary: {e1}\nSource: {e2}"

# =================================================================================
# == Configuration Constants
# =================================================================================
CONST_LOG_PREFIX = "[MD_TPG]"
CONST_JS_MAX_SAFE_INTEGER = 9007199254740991
CONST_SEED_MIN = 0

CONST_DEFAULT_PROTECT = 1
CONST_DEFAULT_STRENGTH = 1.0
CONST_DEFAULT_START_SIGMA = 1000.0
CONST_DEFAULT_END_SIGMA = 0.1

# =================================================================================
# == Core Node Class
# =================================================================================

class MD_ApplyTPG:
    """ComfyUI wrapper node for Token Perturbation Guidance (TPG)."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {
                    "tooltip": (
                        "MODEL INPUT\n"
                        "• Purpose: The diffusion model to patch with TPG.\n"
                        "• Compatibility: Works with most Unet/DiT architectures (SD1.5, SDXL, FLUX).\n"
                        "• Output: Returns a patched model ready for sampling."
                    )
                }),
                "enable_tpg": ("BOOLEAN", {
                    "default": True,
                    "label_on": "Enabled",
                    "label_off": "Disabled",
                    "tooltip": (
                        "ENABLE TPG\n"
                        "• Purpose: Master switch for the effect.\n"
                        "• Use Case: Quickly toggle for A/B testing without rewiring nodes.\n"
                        "\n⭐ Recommended: True"
                    )
                }),
                "target_layers": (["Down (Encoder)", "Mid (Bottleneck)", "Up (Decoder)", "All"], {
                    "default": "Down (Encoder)",
                    "tooltip": (
                        "TARGET LAYERS\n"
                        "• Purpose: Select which U-Net blocks to patch.\n"
                        "• Options:\n"
                        "  - Down (Encoder): Best FID scores (Paper recommendation)\n"
                        "  - Mid: Affects global structure/composition\n"
                        "  - Up (Decoder): Affects fine details and textures\n"
                        "  - All: Strongest effect, may be destructive\n"
                        "\n⭐ Recommended: Down (Encoder)"
                    )
                }),
                "perturbation_strength": ("FLOAT", {
                    "default": CONST_DEFAULT_STRENGTH, 
                    "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": (
                        "PERTURBATION STRENGTH\n"
                        "• Purpose: Controls the blend between original and shuffled tokens.\n"
                        "• Range: 0.0 (No effect) to 1.0 (Full Shuffle).\n"
                        "• Trade-offs: Lower values are safer but less effective at guiding.\n"
                        "\n⭐ Recommended: 1.0 (Control intensity via Sigma instead)"
                    )
                }),
                "start_sigma": ("FLOAT", {
                    "default": CONST_DEFAULT_START_SIGMA, 
                    "min": 0.0, "max": 10000.0, "step": 0.1,
                    "tooltip": (
                        "START SIGMA\n"
                        "• Purpose: Noise level to START applying TPG.\n"
                        "• Note: High value (1000+) = Start immediately at beginning of generation.\n"
                        "• Use Case: TPG is most effective at high noise levels (structural formation).\n"
                        "\n⭐ Recommended: 1000.0 (Always on at start)"
                    )
                }),
                "end_sigma": ("FLOAT", {
                    "default": CONST_DEFAULT_END_SIGMA, 
                    "min": 0.0, "max": 1000.0, "step": 0.01,
                    "tooltip": (
                        "END SIGMA\n"
                        "• Purpose: Noise level to STOP applying TPG.\n"
                        "• Use Case: Stop before 0.1 to prevent damaging fine textures.\n"
                        "• Range: 0.0 (End of gen) to 1000.0 (Start of gen).\n"
                        "\n⭐ Recommended: 0.1 - 0.5"
                    )
                }),
                "protect_first_tokens": ("INT", {
                    "default": CONST_DEFAULT_PROTECT, 
                    "min": 0, "max": 16,
                    "tooltip": (
                        "PROTECT TOKENS\n"
                        "• Purpose: Prevent shuffling of initial special tokens (like [CLS]).\n"
                        "• Range: 0-16 tokens.\n"
                        "• Importance: Preserves global conditioning context.\n"
                        "\n⭐ Recommended: 1 (Protects [CLS])"
                    )
                }),
                "split_mode": (["Uncond First (Standard)", "Cond First (Inverted)"], {
                    "default": "Uncond First (Standard)",
                    "tooltip": (
                        "SPLIT MODE HEURISTIC\n"
                        "• Purpose: How the node detects the Unconditional batch.\n"
                        "• Options:\n"
                        "  - Standard: Assumes Uncond batch comes first (ComfyUI default)\n"
                        "  - Inverted: Use if generation looks corrupted or inverted\n"
                        "\n⭐ Recommended: Uncond First (Standard)"
                    )
                }),
                "seed": ("INT", {
                    "default": 0, 
                    "min": CONST_SEED_MIN, 
                    "max": CONST_JS_MAX_SAFE_INTEGER,
                    "tooltip": (
                        "PERMUTATION SEED\n"
                        "• Purpose: Base seed for the token shuffle generator.\n"
                        "• Note: Actual permutation changes every step/layer deterministically.\n"
                        "• Range: 0 to 9,007,199,254,740,991 (JS-safe limit).\n"
                        "\n⭐ Most users: Leave random or fixed for reproducibility."
                    )
                }),
                "debug_mode": (["Off", "Basic (Stats)", "Visual (ASCII)"], {
                    "default": "Off",
                    "tooltip": (
                        "DEBUG MODE\n"
                        "• Purpose: Console feedback level.\n"
                        "• Options:\n"
                        "  - Off: Silent (Production)\n"
                        "  - Basic: Prints active layers/sigma info\n"
                        "  - Visual: Prints ASCII map of token shuffles (WARNING: Spammy!)\n"
                        "\n⭐ Recommended: Off"
                    )
                }),
            }
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("patched_model", "debug_info")
    FUNCTION = "apply_tpg"
    CATEGORY = "MD_Nodes/Optimization"

    def apply_tpg(self, model, enable_tpg, target_layers, perturbation_strength, start_sigma, end_sigma, protect_first_tokens, split_mode, seed, debug_mode):
        
        # Early Exit
        if not enable_tpg or perturbation_strength <= 0.0:
            return (model, "TPG Disabled")

        # Graceful Degradation: Missing Core
        if not CORE_LOADED:
            error_msg = f"❌ ERROR: TPG Core missing.\nMode: {CORE_MODE or 'Not Loaded'}\nError: {CORE_ERROR}"
            logging.error(f"{CONST_LOG_PREFIX} {error_msg}")
            return (model, error_msg) # Return model unpatched

        # Validate seed using Core function
        valid_seed = core.validate_seed(seed)
        m = model.clone()

        # Parse Targets
        target_blocks = []
        if "Down" in target_layers or "All" in target_layers: target_blocks.append("input")
        if "Mid" in target_layers or "All" in target_layers:  target_blocks.append("middle")
        if "Up" in target_layers or "All" in target_layers:   target_blocks.append("output")

        # Define the patch function to inject
        def tpg_attn1_patch(q, k, v, extra_options):
            try:
                # A. Safety Check
                if q.ndim != 3: return q, k, v

                # B. Layer Filter
                block_info = extra_options.get("block", None)
                if block_info:
                    block_type = block_info[0]
                    layer_id_salt = block_info[1] if len(block_info) > 1 else 0
                else:
                    block_type = "unknown"
                    layer_id_salt = 0

                if block_type not in target_blocks:
                    return q, k, v 

                # C. Sigma Extraction & Temporal Scheduling
                transformer_options = extra_options.get("transformer_options", {})
                current_sigma = 0.0
                sigmas = transformer_options.get("sigmas", None)
                
                if sigmas is not None:
                    if isinstance(sigmas, torch.Tensor) and sigmas.numel() > 0:
                        current_sigma = float(sigmas.flatten()[0])
                    elif isinstance(sigmas, (list, tuple)) and len(sigmas) > 0:
                        current_sigma = float(sigmas[0])
                elif "step" in transformer_options:
                    current_sigma = float(transformer_options["step"])

                # Check Schedule
                if current_sigma > start_sigma or current_sigma < end_sigma:
                    return q, k, v

                # D. Debug Logging Check
                should_log = False
                if debug_mode != "Off":
                    if not hasattr(tpg_attn1_patch, "_call_count"): tpg_attn1_patch._call_count = 0
                    tpg_attn1_patch._call_count += 1
                    if tpg_attn1_patch._call_count % 20 == 0: should_log = True

                # E. Processing via Core
                cond_map = transformer_options.get("cond_or_uncond", None)
                
                # Generate deterministic seed
                step_seed = core.generate_step_seed(valid_seed, layer_id_salt, current_sigma)
                
                # Apply permutation math
                q_out = core.process_uncond_batch(
                    q, 
                    cond_map, 
                    step_seed, 
                    protect_first_tokens, 
                    perturbation_strength, 
                    split_mode
                )

                # Visual Debugging (Only runs on designated log cycles)
                if should_log and getattr(q_out, "data_ptr", lambda: 0)() != getattr(q, "data_ptr", lambda: 1)():
                     msg = f"{CONST_LOG_PREFIX} Layer: {block_type} | Sigma: {current_sigma:.2f}"
                     if debug_mode == "Visual (ASCII)":
                         orig = list(range(10))
                         g_cpu = torch.Generator().manual_seed(step_seed)
                         p = torch.randperm(10-protect_first_tokens, generator=g_cpu) + protect_first_tokens
                         prot = list(range(protect_first_tokens))
                         shuf = p.tolist()
                         final = prot + shuf
                         msg += f"\n  Map: {orig} -> {final[:10]}..."
                     print(msg)

                return q_out, k, v

            except Exception as e:
                print(f"{CONST_LOG_PREFIX} Patch Error: {e}")
                return q, k, v

        # Inject the patch into the model
        m.set_model_attn1_patch(tpg_attn1_patch)
        
        # Status String
        status = (f"TPG Active | Layers: {target_layers}\n"
                  f"Strength: {perturbation_strength} | Sigma Range: {start_sigma} -> {end_sigma}\n"
                  f"Protect: First {protect_first_tokens} tokens")
        
        logging.info(f"{CONST_LOG_PREFIX} Configured. {status}")
        
        return (m, status)

# =================================================================================
# == Node Registration
# =================================================================================

NODE_CLASS_MAPPINGS = {
    "MD_ApplyTPG": MD_ApplyTPG,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MD_ApplyTPG": "MD: Apply TPG (Token Perturbation)",
}


# ==============================================================================
# == Unit Tests (smoke — runs without ComfyUI)
# ==============================================================================

if __name__ == "__main__":
    print("\n🧪 Smoke tests: MD_ApplyTPG")
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

    _check("VERSION defined",    VERSION == "v1.6.1")
    _check("CONST CONST_LOG_PREFIX defined", CONST_LOG_PREFIX is not None)
    _check("CONST CONST_JS_MAX_SAFE_INTEGER defined", CONST_JS_MAX_SAFE_INTEGER is not None)
    _check("CONST CONST_SEED_MIN defined", CONST_SEED_MIN is not None)
    _check("CONST CONST_DEFAULT_PROTECT defined", CONST_DEFAULT_PROTECT is not None)
    _check("CONST CONST_DEFAULT_STRENGTH defined", CONST_DEFAULT_STRENGTH is not None)
    _check("NODE_CLASS_MAPPINGS defined",
           isinstance(NODE_CLASS_MAPPINGS, dict) and len(NODE_CLASS_MAPPINGS) > 0)
    _check("  class MD_ApplyTPG in map", "MD_ApplyTPG" in NODE_CLASS_MAPPINGS)

    print(f"\n  {_pass} passed, {_fail} failed")
    if _fail == 0:
        print("  🎉 All good.")
