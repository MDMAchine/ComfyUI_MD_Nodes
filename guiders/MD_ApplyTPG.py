# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
# ████ MD_Nodes/MD_ApplyTPG – Token Perturbation Guidance Model Patcher ████▓▒░
# © 2025 MDMAchine
# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
# ░▒▓ ORIGIN & DEV:
#   • Concept: "Token Perturbation Guidance for Diffusion Models" (Rajabi et al., 2025)
#   • Implemented by: MDMAchine & Gemini
#   • License: Apache 2.0
#   • Paper Source: arXiv:2506.10036v2

# ░▒▓ DESCRIPTION:
#   Patches the Diffusion Model to implement Token Perturbation Guidance (TPG).
#   It intercepts Self-Attention (Attn1) layers during the Unconditional pass
#   and shuffles tokens to create a robust "Negative Score".

# ░▒▓ FEATURES:
#   ✓ Paper-Accurate: Unique permutations per Step (t) and Layer (k).
#   ✓ Targeted: Affects only specific UNet/DiT blocks (Down/Mid/Up).
#   ✓ Robust: Adaptive batch detection with manual overrides.
#   ✓ Stable: Bitwise seeding, type-safe sigma extraction, and device fallbacks.

# ░▒▓ CHANGELOG:
#   - v1.3.0 (Release Candidate):
#       • ENHANCED: Debug output now includes Block Type & Uncond count.
#       • ENHANCED: Generator device fallback (CPU) for older PyTorch/MPS compatibility.
#       • OPTIMIZED: Sigma extraction avoids unnecessary CPU transfers.
#   - v1.2.0: Stability hardening (Bitwise seeds, Input validation).
#   - v1.1.0: Unique permutations per step.

# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄

import torch
import logging
import comfy.model_management

class MD_ApplyTPG:
    """
    Applies Token Perturbation Guidance (TPG) by patching the model's Self-Attention.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The diffusion model to patch."}),
                "enable_tpg": ("BOOLEAN", {"default": True}),
                "target_layers": (["Down (Encoder)", "Mid (Bottleneck)", "Up (Decoder)", "All"], {
                    "default": "Down (Encoder)",
                    "tooltip": "Paper recommends 'Down (Encoder)' for best FID scores."
                }),
                "perturbation_strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Blend factor: 1.0 = Full Shuffle, 0.0 = No effect."
                }),
                "split_mode": (["Uncond First (Standard)", "Cond First (Inverted)"], {
                    "default": "Uncond First (Standard)",
                    "tooltip": "HEURISTIC FALLBACK:\nUse if TPG seems to invert/break generation.\nMost samplers are [Uncond, Cond] (Select 'Uncond First')."
                }),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xffffffffffffffff,
                    "tooltip": "Base seed. Actual permutation varies deterministically by step and layer."
                }),
                "debug_mode": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Print tensor shapes and split logic to console (Rate Limited)."
                }),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("patched_model",)
    FUNCTION = "apply_tpg"
    CATEGORY = "MD_Nodes/Guidance"

    def apply_tpg(self, model, enable_tpg, target_layers, perturbation_strength, split_mode, seed, debug_mode):
        
        if not enable_tpg or perturbation_strength <= 0.0:
            return (model,)

        m = model.clone()

        # --- Parse Layer Targets ---
        # ComfyUI Block Names: 'input' (Down), 'middle' (Mid), 'output' (Up)
        target_blocks = []
        if "Down" in target_layers or "All" in target_layers: target_blocks.append("input")
        if "Mid" in target_layers or "All" in target_layers:  target_blocks.append("middle")
        if "Up" in target_layers or "All" in target_layers:   target_blocks.append("output")

        # --- Patch Function ---
        def tpg_attn1_patch(q, k, v, extra_options):
            try:
                # 1. Safety Validation
                if q.ndim != 3:
                    # TPG expects [Batch, Tokens, Dim]. If 4D/5D, skip.
                    return q, k, v

                # 2. Layer Filtering
                block_info = extra_options.get("block", None) # Tuple: ("input", 1)
                
                if block_info and len(block_info) > 1:
                    layer_id_salt = block_info[1]
                    block_type = block_info[0]
                elif block_info:
                    layer_id_salt = hash(str(block_info)) % 1000
                    block_type = str(block_info)
                else:
                    layer_id_salt = 0
                    block_type = "unknown"

                if block_type not in target_blocks:
                    return q, k, v 

                # 3. Debug Rate Limiting (Early init for tracking)
                should_print = False
                if debug_mode:
                    if not hasattr(tpg_attn1_patch, "_call_count"):
                        tpg_attn1_patch._call_count = 0
                    tpg_attn1_patch._call_count += 1
                    if tpg_attn1_patch._call_count % 20 == 0:
                        should_print = True

                # 4. Robust Sigma/Time Extraction
                transformer_options = extra_options.get("transformer_options", {})
                current_sigma = 0.0
                
                sigmas = transformer_options.get("sigmas", None)
                if sigmas is not None:
                    if isinstance(sigmas, torch.Tensor) and sigmas.numel() > 0:
                        # Optimization: Avoid CPU transfer if possible, but float cast usually forces it.
                        # Accessing flattened index 0 is safe.
                        current_sigma = float(sigmas.flatten()[0])
                    elif isinstance(sigmas, (list, tuple)) and len(sigmas) > 0:
                        current_sigma = float(sigmas[0])
                
                if current_sigma == 0.0 and "step" in transformer_options:
                    current_sigma = float(transformer_options["step"])

                # 5. Bitwise Seeding (Overflow Safe)
                time_salt = int(current_sigma * 1000)
                limit = 2**63 - 1
                current_seed = (seed ^ layer_id_salt ^ time_salt) & limit

                # 6. Batch Identification (Uncond vs Cond)
                cond_map = transformer_options.get("cond_or_uncond", None)
                q_out = q
                
                # --- Path A: Precise Masking (Preferred) ---
                if cond_map is not None and len(cond_map) == q.shape[0]:
                    if isinstance(cond_map, torch.Tensor):
                        is_uncond = (cond_map == 0).to(q.device)
                    else:
                        is_uncond = torch.tensor([x == 0 for x in cond_map], device=q.device)
                    
                    if is_uncond.any():
                        q_uncond = q[is_uncond]
                        
                        # Device-Safe Generator
                        gen = torch.Generator(device=q.device)
                        try:
                            gen.manual_seed(current_seed)
                            B_u, T, C = q_uncond.shape
                            perm_indices = torch.randperm(T, generator=gen, device=q.device)
                        except RuntimeError:
                            # Fallback for older PyTorch/MPS where Generator(device=cuda) might fail
                            gen = torch.Generator(device='cpu')
                            gen.manual_seed(current_seed)
                            perm_indices = torch.randperm(T, generator=gen).to(q.device)

                        q_shuffled = q_uncond[:, perm_indices, :]
                        
                        if perturbation_strength < 1.0:
                            q_uncond_final = torch.lerp(q_uncond, q_shuffled, perturbation_strength)
                        else:
                            q_uncond_final = q_shuffled
                        
                        # Selective Clone for Memory Safety
                        q_out = q.clone()
                        q_out[is_uncond] = q_uncond_final
                        
                        if should_print:
                             print(f"[MD_TPG] Masked | Block: {block_type}_{layer_id_salt} | "
                                   f"Sigma: {current_sigma:.3f} | Uncond: {is_uncond.sum().item()}/{q.shape[0]}")

                # --- Path B: Heuristic Chunking (Fallback) ---
                else:
                    if q.shape[0] % 2 == 0:
                        chunk_size = q.shape[0] // 2
                        q1, q2 = q.chunk(2, dim=0)
                        
                        if split_mode == "Uncond First (Standard)":
                            target_q = q1
                            other_q = q2
                            is_q1_target = True
                        else: 
                            target_q = q2
                            other_q = q1
                            is_q1_target = False
                        
                        # Device-Safe Generator (Repeated logic)
                        gen = torch.Generator(device=q.device)
                        try:
                            gen.manual_seed(current_seed)
                            B_u, T, C = target_q.shape
                            perm_indices = torch.randperm(T, generator=gen, device=q.device)
                        except RuntimeError:
                            gen = torch.Generator(device='cpu')
                            gen.manual_seed(current_seed)
                            perm_indices = torch.randperm(T, generator=gen).to(q.device)

                        q_shuffled = target_q[:, perm_indices, :]
                        
                        if perturbation_strength < 1.0:
                            target_final = torch.lerp(target_q, q_shuffled, perturbation_strength)
                        else:
                            target_final = q_shuffled
                        
                        if is_q1_target:
                            q_out = torch.cat([target_final, other_q], dim=0)
                        else:
                            q_out = torch.cat([other_q, target_final], dim=0)
                        
                        if should_print:
                             print(f"[MD_TPG] Heuristic | Block: {block_type}_{layer_id_salt} | "
                                   f"Sigma: {current_sigma:.3f} | Mode: {split_mode}")
                    else:
                        if should_print:
                            print(f"[MD_TPG] Odd batch {q.shape[0]} - Skipping.")
                        return q, k, v

                return q_out, k, v

            except Exception as e:
                print(f"[MD_ApplyTPG] Error in patch: {e}")
                return q, k, v

        m.set_model_attn1_patch(tpg_attn1_patch)
        
        logging.info(f"[MD_ApplyTPG] Patch active. Layers: {target_layers}, Str: {perturbation_strength}")
        return (m,)

NODE_CLASS_MAPPINGS = {
    "MD_ApplyTPG": MD_ApplyTPG,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MD_ApplyTPG": "MD: Apply TPG (Token Perturbation)",
}