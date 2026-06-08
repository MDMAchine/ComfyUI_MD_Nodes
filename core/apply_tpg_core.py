# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# █▓▒░                                                                     ░▒▓█
# █▓▒░              apply_tpg_core.py - Core Algorithm v1.6.1              ░▒▓█
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
# ║ CORE RESPONSIBILITIES:
# ║   • Token Perturbation Logic (Shuffling)
# ║   • Deterministic Seed Generation
# ║   • Tensor slicing and reconstruction
# ║   • Stateless processing (pure tensor math)
# ╚════════════════════════════════════════════════════════════════════════════


VERSION = "v1.6.1"  # UPS v1.5.8

import torch
import math

# Constants
CONST_JS_MAX_SAFE_INTEGER = 9007199254740991
CONST_SEED_MIN = 0

def validate_seed(seed_value):
    """Ensure seed is within safe range."""
    try:
        val = int(seed_value)
    except (ValueError, TypeError):
        return CONST_SEED_MIN
    return max(CONST_SEED_MIN, min(val, CONST_JS_MAX_SAFE_INTEGER))

def generate_step_seed(base_seed, layer_id_salt, current_sigma):
    """
    Generate a deterministic seed for a specific layer and timestep.
    """
    time_salt = int(current_sigma * 1000)
    return (base_seed ^ layer_id_salt ^ time_salt) & CONST_JS_MAX_SAFE_INTEGER

def shuffle_tokens(target_q, current_seed, protect_first_tokens, perturbation_strength):
    """
    Shuffles tokens in the sequence dimension (T) while protecting initial tokens.
    
    Args:
        target_q: Tensor of shape [Batch, Sequence, Channels]
        current_seed: Deterministic seed for this step/layer
        protect_first_tokens: Number of initial tokens to skip (e.g., [CLS])
        perturbation_strength: Blend factor (0.0 to 1.0)
        
    Returns:
        Shuffled/blended tensor of same shape
    """
    B_u, T, C = target_q.shape
    num_shuffled = T - protect_first_tokens
    
    if num_shuffled <= 1: 
        return target_q  # Nothing to shuffle

    device = target_q.device
    
    try:
        # Attempt device-native generator first
        gen = torch.Generator(device=device)
        gen.manual_seed(current_seed)
        perm_indices = torch.randperm(num_shuffled, generator=gen, device=device)
    except RuntimeError:
        # Fallback to CPU if MPS/CUDA generator fails
        gen = torch.Generator(device='cpu')
        gen.manual_seed(current_seed)
        perm_indices = torch.randperm(num_shuffled, generator=gen).to(device)

    # Shift indices to account for protected tokens
    perm_indices += protect_first_tokens
    
    # Prepend protected indices [0, 1, ... N]
    protected_indices = torch.arange(protect_first_tokens, device=device)
    final_indices = torch.cat([protected_indices, perm_indices])
    
    # Apply Shuffle across sequence dimension
    q_shuffled = target_q[:, final_indices, :]
    
    # Apply Strength interpolation
    if perturbation_strength < 1.0:
        return torch.lerp(target_q, q_shuffled, perturbation_strength)
    
    return q_shuffled

def process_uncond_batch(q, cond_map, current_seed, protect_first, strength, split_mode):
    """
    Identifies and processes ONLY the unconditional portion of the batch.
    """
    # Path A: Explicit Masking (if cond_map provided)
    if cond_map is not None and len(cond_map) == q.shape[0]:
        if not isinstance(cond_map, torch.Tensor):
            cond_map = torch.tensor(cond_map, device=q.device)
            
        is_uncond = (cond_map == 0)
        
        if is_uncond.any():
            q_uncond = q[is_uncond]
            q_processed = shuffle_tokens(q_uncond, current_seed, protect_first, strength)
            
            # Safe cloning and injection
            q_out = q.clone()
            q_out[is_uncond] = q_processed
            return q_out
            
        return q # No uncond found

    # Path B: Heuristic Split (Fallback for symmetric batches)
    if q.shape[0] % 2 == 0:
        q1, q2 = q.chunk(2, dim=0)
        if split_mode == "Uncond First (Standard)":
            target, other = q1, q2
            recombine = lambda t, o: torch.cat([t, o], dim=0)
        else:
            target, other = q2, q1
            recombine = lambda t, o: torch.cat([o, t], dim=0)
            
        target_processed = shuffle_tokens(target, current_seed, protect_first, strength)
        return recombine(target_processed, other)

    # Fallback if neither condition met
    return q


# ==============================================================================
# == Unit Tests (smoke — runs without ComfyUI)
# ==============================================================================

if __name__ == "__main__":
    print("\n🧪 Smoke tests: apply_tpg_core")
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
    _check("CONST CONST_JS_MAX_SAFE_INTEGER defined", CONST_JS_MAX_SAFE_INTEGER is not None)
    _check("CONST CONST_SEED_MIN defined", CONST_SEED_MIN is not None)
    _check("fn validate_seed is callable", callable(validate_seed))
    _check("fn generate_step_seed is callable", callable(generate_step_seed))
    _check("fn shuffle_tokens is callable", callable(shuffle_tokens))

    print(f"\n  {_pass} passed, {_fail} failed")
    if _fail == 0:
        print("  🎉 All good.")
