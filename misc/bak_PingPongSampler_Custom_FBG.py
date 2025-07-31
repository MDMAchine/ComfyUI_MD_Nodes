# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# ████ PINGPONGSAMPLER v0.9.9 – Optimized for Ace-Step Audio/Video with FBG and Node Debug! ████▓▒░
# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀

# ░▒▓ ORIGIN & DEV:
#   • Foundational principles for iterative sampling, including concepts that underpin 'ping-pong sampling'
#   • Consistency Models by Song et al. (2023)
#   •      https://arxiv.org/abs/2303.01469
#   • The term 'ping-pong sampling' is explicitly introduced and applied in the context of fast text-to-audio
#   • generationin the paper "Fast Text-to-Audio Generation with Adversarial Post-Training" by Novack et al.
#   • (2025) from Stability AI
#   •      https://arxiv.org/abs/2505.08175
#   • original concept for the PingPong Sampler for ace-step diffusion by: Junmin Gong (Ace-Step team)
#   • ComfyUI adaptation by: blepping (original ComfyUI port with quirks)
#   • Disassembled & warped by: MD (Machine Damage)
#   • Critical fixes & re-engineering by: Gemini (Google) based on user feedback
#   • Feedback Guidance integration by: Gemini (Google)
#   • Completionist fixups via: devstral / qwen3 (local heroes)
#   • License: Apache 2.0 — Sharing is caring, no Voodoo hoarding here
#   • Original source gist: https://gist.github.com/blepping/b372ef6c5412080af136aad942d9d76c
#   • FBG addition based off of Feedback-Guidance-of-Diffusion-Models:
#   •      https://github.com/FelixKoulischer/Feedback-Guidance-of-Diffusion-Models/
#   •      https://arxiv.org/abs/2506.06085
#   • ComfyUI FBG adaptation by: blepping:
#   •      https://gist.github.com/blepping/d424e8fd27d76845ad27997820a57f6b

# ░▒▓ DESCRIPTION:
#   A sampler node for ComfyUI, now enhanced with Feedback Guidance (FBG) and
#   a convenient node-level debug toggle. It combines PingPong's unique
#   ancestral noise mixing strategy, tuned for Ace-Step audio and video
#   diffusion models, with FBG's dynamic, content-aware guidance scale adjustment.
#   This means your generations can now benefit from both precise noise control
#   and adaptive guidance based on the model's "understanding" of the content's needs.
#   Maintain ancestral noise sync with the original blepping sampler.
#   May work with image models but results can vary wildly — consider it your
#   party-ready audio/video workhorse, now with adaptive intelligence.
#   Warning: May induce flashbacks to 256-byte intros or
#   compulsive byte-optimization urges. Use responsibly.

# ░▒▓ CHANGELOG HIGHLIGHTS:
#   - v0.9.0 "Feedback Fusion":
#       • Integrated Feedback Guidance (FBG) for dynamic guidance scale adjustment.
#       • FBG parameters (pi, t_0, t_1, fbg_guidance_multiplier, etc.) added as inputs.
#       • Enhanced debugging capabilities with DEBUG_MODE (global constant).
#       • Comprehensive docstrings and comments for better understanding.
#   - v0.9.1 "Interactive Debug":
#       • Moved DEBUG_MODE from a global constant to a node input boolean.
#       • Users can now enable/disable debug messages directly from the ComfyUI node.
#   - v0.9.2 "Syntax Fixes":
#       • Corrected 'non-default argument follows default argument' error in get_sampler
#         by providing default values for all FBG parameters. (Initial attempt, required refinement)
#   - v0.9.3 "Signature Refinement":
#       • Corrected Python function signature to remove redundant default assignments
#         for parameters already declared as 'required' in INPUT_TYPES, resolving
#         '_collections._tuplegetter' object has no attribute 'default' error.
#   - v0.9.4 "Coexistence Enablement":
#       • Changed NODE_CLASS_MAPPINGS key to 'PingPongSampler_Custom_FBG' to allow
#         coexistence with the original 'PingPongSampler_Custom' node.
#   - v0.9.5 "AttributeError Fix":
#       • Implemented `go` as a `@staticmethod` within `PingPongSampler` to correctly
#         interface with ComfyUI's KSAMPLER wrapper.
#   - v0.9.6 "YAML Key Remap (Attempt 1)":
#       • Attempted remapping logic in `PingPongSampler.go` to accept `fbg_sampler_mode`
#         as `sampler_mode` within the `fbg_config` section of YAML input.
#   - v0.9.7 "Robust YAML Key Fix":
#       • Implemented a more robust remapping mechanism for FBGConfig parameters
#         parsed from YAML to ensure correct argument names are passed to FBGConfig.__new__().
#       • Improved error handling for minimal_log_posterior calculation in PingPongSampler.__init__.
#       • Added checks for zero divisors in _update_log_posterior.
#   - v0.9.8 "Sigma Precision":
#       • Updated `step` values for all sigma-related float inputs in `INPUT_TYPES`
#         to `0.001` to allow finer precision (e.g., 0.004).
#   - v0.9.9 "Default Sigma Range Alignment":
#       • Adjusted default `cfg_start_sigma`, `cfg_end_sigma`, `fbg_start_sigma`, `fbg_end_sigma`,
#         `ancestral_start_sigma`, and `ancestral_end_sigma` to match common model sigma ranges
#         (e.g., 1.0 down to 0.004).


# ░▒▓ CONFIGURATION:
#   → Primary Use: High-quality audio/video generation with Ace-Step diffusion
#   → Secondary Use: Experimental visual generation (expect wild results)
#   → Edge Use: For demoscene veterans and bytecode wizards only

# ░▒▓ WARNING:
#   This node may trigger:
#   ▓▒░ Flashbacks to ANSI art and 256-byte intros
#   ▓▒░ Spontaneous breakdancing or synesthesia
#   ▓▒░ Urges to reinstall Impulse Tracker
#   ▓▒░ Extreme focus on byte-level optimization
#   Consult your nearest demoscene vet if hallucinations persist.

# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄


#!/usr/bin/env python3

import math
import torch
from tqdm.auto import trange
from enum import Enum, auto
from typing import NamedTuple, Optional, Callable, Dict, Any

from comfy import model_sampling, model_patcher # model_patcher is crucial for CFG handling
from comfy.samplers import KSAMPLER, cfg_function
from comfy.k_diffusion.sampling import get_ancestral_step # For ancestral step calculation
import numpy as np
import yaml # YAML for handling optional string inputs


# --- Internal Blend Modes (from original PingPong) ---
_INTERNAL_BLEND_MODES = {
    "lerp": torch.lerp,
    "a_only": lambda a, _b, _t: a, # Returns the first input (a)
    "b_only": lambda _a, b, _t: b  # Returns the second input (b)
}


# --- FBG Specific Classes and Functions (Moved from fbg_sampler_node.py) ---
class SamplerMode(Enum):
    """
    Defines the sampling modes supported for the underlying k-diffusion step in FBG.
    - EULER: Standard Euler ancestral sampling (default for FBG's internal step).
    - PINGPONG: A mode where noise completely replaces the current state under certain conditions (from FBG).
    Note: This is separate from PingPongSampler's overall noise strategy.
    """
    EULER = auto()
    PINGPONG = auto()


class FBGConfig(NamedTuple):
    """
    Configuration parameters for the Feedback Guidance (FBG) algorithm.
    These parameters control the dynamic adjustment of the guidance scale.
    """
    sampler_mode: SamplerMode = SamplerMode.PINGPONG
    cfg_start_sigma: float = 1.0 # Adjusted default for common model ranges
    cfg_end_sigma: float = 0.004 # Adjusted default for common model ranges
    fbg_start_sigma: float = 1.0 # Adjusted default for common model ranges
    fbg_end_sigma: float = 0.004 # Adjusted default for common model ranges
    fbg_guidance_multiplier: float = 1.0
    ancestral_start_sigma: float = 1.0 # Adjusted default for common model ranges
    ancestral_end_sigma: float = 0.004 # Adjusted default for common model ranges
    cfg_scale: float = 120.0 # This is the base CFG scale that FBG modifies
    max_guidance_scale: float = 550.0
    max_posterior_scale: float = 1.0
    initial_value: float = 0.5
    initial_guidance_scale: float = 120.0
    guidance_max_change: float = 1.0
    temp: float = 0.01
    offset: float = -0.03
    pi: float = 0.35
    t_0: float = 0.95
    t_1: float = 0.75


def batch_mse_loss(a: torch.Tensor, b: torch.Tensor, *, start_dim: int = 1) -> torch.Tensor:
    """
    Calculates the Mean Squared Error (MSE) loss across batch dimensions.
    Used for comparing the distance between denoised predictions in FBG.
    """
    return ((a - b) ** 2).sum(dim=tuple(range(start_dim, a.ndim)))


# --- End FBG Specific Classes and Functions ---


class PingPongSampler:
    """
    A custom sampler implementing a "ping-pong" ancestral noise mixing strategy,
    now integrated with Feedback Guidance (FBG) for dynamic guidance scale adjustment.
    """

    # Removed FUNCTION = "go" here, it's now a static method below.
    # CATEGORY and RETURN_TYPES are for the Node wrapper, not the internal sampler.

    def __init__(
        self,
        model,
        x,
        sigmas,
        extra_args=None,
        callback=None,
        disable=None,
        noise_sampler=None,
        # PingPong-specific parameters
        start_sigma_index: int = 0,
        end_sigma_index: int = -1,
        enable_clamp_output: bool = False,
        step_random_mode: str = "off",
        step_size: int = 5,
        seed: int = 42,
        blend_function: Callable = torch.lerp,
        step_blend_function: Callable = torch.lerp,
        scheduler=None,
        pingpong_options: Optional[Dict[str, Any]] = None,
        # FBG-specific configuration
        fbg_config: FBGConfig = FBGConfig(),
        # New: Debug mode flag
        debug_mode: bool = False,
        # FBG specific
        eta: float = 0.0,
        s_noise: float = 0.7,
        **kwargs # Catch any remaining kwargs, though they should be handled by now
    ):
        # --- Core ComfyUI Sampler Inputs ---
        self.model_ = model # Underlying diffusion model function
        self.x = x # Initial noisy tensor to sample from
        self.sigmas = sigmas # Array of noise levels to step through
        self.extra_args = extra_args.copy() if extra_args is not None else {} # Additional arguments for model (e.g., conditioning)
        self.callback_ = callback # External callback to report progress (for ComfyUI's progress bar)
        self.disable_pbar = disable # Disable progress bar if truthy

        # --- PingPong Sampler Specific Controls ---
        self.start_sigma_index = start_sigma_index
        self.end_sigma_index = end_sigma_index
        self.enable_clamp_output = enable_clamp_output

        # Noise injection and random seed controls
        self.step_random_mode = step_random_mode
        self.step_size = step_size

        # Use the provided seed, otherwise fallback to a default if not set (or if None)
        self.seed = seed if seed is not None else 42


        # Blend functions
        self.blend_function = blend_function
        self.step_blend_function = step_blend_function

        # New: Store debug mode
        self.debug_mode = debug_mode

        # Determine the number of total steps available from the sigmas schedule
        num_steps_available = len(sigmas) - 1 if len(sigmas) > 0 else 0

        # Ancestral (ping-pong) operation boundaries (from original PingPong)
        if pingpong_options is None:
            pingpong_options = {}
        raw_first_ancestral_step = pingpong_options.get("first_ancestral_step", 0)
        raw_last_ancestral_step = pingpong_options.get("last_ancestral_step", num_steps_available - 1)
        
        self.first_ancestral_step = max(0, min(raw_first_ancestral_step, raw_last_ancestral_step))
        if num_steps_available > 0:
            self.last_ancestral_step = min(num_steps_available - 1, max(raw_first_ancestral_step, raw_last_ancestral_step))
        else:
            self.last_ancestral_step = -1

        # Detect if model uses ComfyUI CONST sampling (e.g., for Reflow-like noise injection behavior)
        self.is_rf = False
        current_model_check = model
        try:
            while hasattr(current_model_check, 'inner_model') and current_model_check.inner_model is not None:
                current_model_check = current_model_check.inner_model
            if hasattr(current_model_check, 'model_sampling') and current_model_check.model_sampling is not None:
                self.is_rf = isinstance(current_model_check.model_sampling, model_sampling.CONST)
        except AttributeError:
            if self.debug_mode:
                print("PingPongSampler Warning: Could not definitively determine model_sampling type, assuming not CONST.")
            self.is_rf = False

        # Default noise sampler: NOW RETURNS RAW, UNCONDITIONED UNIT VARIANCE NOISE
        self.noise_sampler = noise_sampler
        if self.noise_sampler is None:
            def default_noise_sampler(sigma, sigma_next):
                # Ancestral sampling needs noise scaled by a specific factor based on sigmas.
                # Here, we generate unit variance noise and PingPongSampler's main loop
                # will scale it correctly. This differs from k_diffusion's default ancestral_step
                # which adds s_noise * noise * (sigma_current**2 - sigma_next**2)**0.5
                return torch.randn_like(x)
            self.noise_sampler = default_noise_sampler

        # Build noise decay array
        if num_steps_available > 0:
            if scheduler is not None and hasattr(scheduler, 'get_decay'):
                try:
                    arr = scheduler.get_decay(num_steps_available)
                    decay = np.asarray(arr, dtype=np.float32)
                    assert decay.shape == (num_steps_available,), (
                        f"Expected {num_steps_available} values from scheduler.get_decay, got {decay.shape}"
                    )
                    self.noise_decay = torch.tensor(decay, device=x.device)
                except Exception as e:
                    if self.debug_mode:
                        print(f"PingPongSampler Warning: Could not get decay from scheduler: {e}. Using zeros for noise decay.")
                    self.noise_decay = torch.zeros((num_steps_available,), dtype=torch.float32, device=x.device)
            else:
                self.noise_decay = torch.zeros((num_steps_available,), dtype=torch.float32, device=x.device)
        else:
            self.noise_decay = torch.empty((0,), dtype=torch.float32, device=x.device)


        # --- FBG Sampler Specific Initialization ---
        self.config = fbg_config # Store the FBG configuration
        # Store eta and s_noise which are used for FBG's internal k-diffusion step if it's not EULER
        self.eta = eta
        self.s_noise = s_noise
        
        # Calculate or update temperature and offset based on t_0 and t_1 in FBG config.
        self.update_fbg_config_params()

        cfg = self.config # Use the (potentially updated) FBG config
        
        # Calculate the minimal log posterior for FBG, used for clamping.
        # Added error handling for division by zero / log of non-positive numbers
        if cfg.cfg_scale > 1 and cfg.cfg_start_sigma > 0:
            numerator = (1.0 - cfg.pi) * (cfg.max_guidance_scale - cfg.cfg_scale + 1)
            denominator = (cfg.max_guidance_scale - cfg.cfg_scale)
            if denominator <= 0 or numerator <= 0:
                self.minimal_log_posterior = float('-inf')
                if self.debug_mode:
                    print(f"Warning: FBG minimal_log_posterior calculation for CFG > 1 resulted in non-positive log argument. Setting to -inf. Numerator={numerator}, Denominator={denominator}")
            else:
                self.minimal_log_posterior = math.log(numerator / denominator)
        else:
            numerator = (1.0 - cfg.pi) * cfg.max_guidance_scale
            denominator = (cfg.max_guidance_scale - 1.0)
            if denominator <= 0 or numerator <= 0:
                self.minimal_log_posterior = float('-inf')
                if self.debug_mode:
                    print(f"Warning: FBG minimal_log_posterior calculation for CFG <= 1 resulted in non-positive log argument. Setting to -inf. Numerator={numerator}, Denominator={denominator}")
            else:
                self.minimal_log_posterior = math.log(numerator / denominator)

        
        # Initial FBG internal states
        self.log_posterior = x.new_full((x.shape[0],), cfg.initial_value)
        self.guidance_scale = x.new_full((x.shape[0], * (1,) * (x.ndim - 1)), cfg.initial_guidance_scale)

        if self.debug_mode:
            print(f"PingPongSampler initialized with FBG config: {self.config}")
            print(f"Minimal log posterior (clamping lower bound): {self.minimal_log_posterior:.4f}")
            print(f"Critical posterior threshold for guidance divergence (1 - pi): {(1.0 - self.config.pi):.4f}")
            if kwargs:
                print(f"PingPongSampler initialized with unused extra_options: {kwargs}")


    def _stepped_seed(self, step: int) -> Optional[int]:
        """
        Determines the RNG seed for the current step based on the selected random mode.
        """
        if self.step_random_mode == "off":
            return None
        
        current_step_size = self.step_size if self.step_size > 0 else 1

        if self.step_random_mode == "block":
            return self.seed + (step // current_step_size)
        elif self.step_random_mode == "reset":
            return self.seed + (step * current_step_size)
        elif self.step_random_mode == "step":
            return self.seed + step
        else:
            if self.debug_mode:
                print(f"PingPongSampler Warning: Unknown step_random_mode '{self.step_random_mode}'. Using base seed.")
            return self.seed

    def _get_sigma_square_tilde(self, sigmas: torch.Tensor) -> torch.Tensor:
        """
        Calculates sigma_square_tilde for each step in the sigma schedule, used by FBG.
        """
        # Ensure sigmas has at least two elements for slicing
        if len(sigmas) < 2:
            return torch.tensor([], device=sigmas.device) # Return empty tensor if not enough sigmas

        s_sq, sn_sq = sigmas[:-1] ** 2, sigmas[1:] ** 2
        
        # Avoid division by zero if s_sq contains zeros
        safe_s_sq = torch.where(s_sq == 0, torch.full_like(s_sq, 1e-6), s_sq) # Add epsilon if zero

        return ((s_sq - sn_sq) * sn_sq / safe_s_sq).flip(dims=(0,))

    def _get_offset(
        self,
        steps: int,
        sigma_square_tilde: torch.Tensor,
        *,
        lambda_ref: float = 3.0,
        decimals: int = 4,
    ) -> float:
        """
        Calculates the offset hyperparameter for FBG log posterior update.
        """
        cfg = self.config
        t_0_clamped = max(0.0, min(1.0, cfg.t_0))
        
        if t_0_clamped == 1.0:
            if self.debug_mode:
                print("Warning: t_0 is 1.0. Offset calculation may be undefined. Returning 0.0.")
            return 0.0

        # Ensure (lambda_ref - 1.0) is not zero or negative for the log term
        if (lambda_ref - 1.0) <= 0:
            if self.debug_mode:
                print("Warning: lambda_ref - 1.0 is <= 0. Offset calculation may be undefined. Returning 0.0 for offset.")
            return 0.0
            
        # Ensure (1.0 - cfg.pi) is not zero or negative for the log term
        if (1.0 - cfg.pi) <= 0:
            if self.debug_mode:
                print("Warning: (1.0 - pi) is <= 0. Offset calculation may be undefined. Returning 0.0 for offset.")
            return 0.0

        log_term = math.log((1.0 - cfg.pi) * lambda_ref / (lambda_ref - 1.0))
        
        # Ensure ((1.0 - t_0_clamped) * steps) is not zero
        denominator = (1.0 - t_0_clamped) * steps
        if denominator == 0:
            if self.debug_mode:
                print("Warning: Denominator for offset calculation is zero. Returning 0.0 for offset.")
            return 0.0

        result = (
            1.0
            / denominator
            * log_term
        )
        return round(result, decimals)

    def _get_temp(
        self,
        steps: int,
        offset: float,
        sigma_square_tilde: torch.Tensor,
        *,
        alpha: float = 10.0,
        decimals: int = 4,
    ) -> float:
        """
        Calculates the temperature hyperparameter for FBG log posterior update.
        """
        cfg = self.config
        t_1_clamped = max(0.0, min(1.0, cfg.t_1))

        t1_lower_idx = int(math.floor(t_1_clamped * (steps - 1)))
        
        if len(sigma_square_tilde) == 0:
            if self.debug_mode:
                print("Warning: sigma_square_tilde is empty. Returning 0.0 for temp.")
            return 0.0
        
        # Clamp t1_lower_idx to valid range
        t1_lower_idx = max(0, min(t1_lower_idx, len(sigma_square_tilde) - 1))

        sst = 0.0 # Default value in case of edge cases
        if len(sigma_square_tilde) == 1:
            sst = sigma_square_tilde[0].item() if isinstance(sigma_square_tilde[0], torch.Tensor) else sigma_square_tilde[0]
            if self.debug_mode:
                print(f"  Only one step, using sst: {sst:.4f}")
        elif t1_lower_idx == len(sigma_square_tilde) - 1:
            sst = sigma_square_tilde[t1_lower_idx].item() if isinstance(sigma_square_tilde[t1_lower_idx], torch.Tensor) else sigma_square_tilde[t1_lower_idx]
            if self.debug_mode:
                print(f"  t_1 clamped to last index. Using sst_t1: {sst:.4f}")
        else: # Interpolate
            sst_t1 = sigma_square_tilde[t1_lower_idx].item() if isinstance(sigma_square_tilde[t1_lower_idx], torch.Tensor) else sigma_square_tilde[t1_lower_idx]
            sst_t1_next = sigma_square_tilde[t1_lower_idx + 1].item() if isinstance(sigma_square_tilde[t1_lower_idx + 1], torch.Tensor) else sigma_square_tilde[t1_lower_idx + 1]
            a = (t_1_clamped * (steps - 1)) - t1_lower_idx
            sst = (sst_t1 * (1 - a) + sst_t1_next * a) # Manual lerp for scalar items
            if self.debug_mode:
                print(f"  Interpolating sst between {sst_t1:.4f} and {sst_t1_next:.4f} with alpha {a:.4f}")

        # Ensure alpha is not zero for division
        if alpha == 0:
            if self.debug_mode:
                print("Warning: Alpha for temp calculation is zero. Returning 0.0 for temp.")
            return 0.0

        temp = (2 * sst / alpha * offset) # No .abs().item() here, let it be float and handle abs in return
        return round(temp, decimals)

    def update_fbg_config_params(self):
        """
        Updates the FBGConfig's 'temp' and 'offset' parameters based on 't_0' and 't_1'
        if these are specified in the config (i.e., not zero).
        """
        if self.config.t_0 == 0 and self.config.t_1 == 0:
            if self.debug_mode:
                print("t_0 and t_1 are 0. Using manual temp and offset from FBG config.")
            return
            
        sigmas = self.sigmas
        steps = len(sigmas) - 1
        
        if steps <= 0:
            if self.debug_mode:
                print("Not enough steps to calculate FBG temp/offset automatically. Keeping defaults.")
            return

        sst = self._get_sigma_square_tilde(sigmas)
        
        calculated_offset = self._get_offset(steps, sst)
        calculated_temp = self._get_temp(steps, calculated_offset, sst)
        
        kwargs = self.config._asdict() # Start with current config dict
        kwargs["offset"] = calculated_offset
        kwargs["temp"] = calculated_temp
        
        self.config = self.config.__class__(**kwargs) # Recreate NamedTuple with updated values

        if self.debug_mode:
            print(f"Updated FBGConfig: offset={self.config.offset:.4f}, temp={self.config.temp:.4f}")

    def get_dynamic_guidance_scale(
        self,
        log_posterior_val: torch.Tensor,
        guidance_scale_prev: torch.Tensor,
        sigma_item: float,
    ) -> torch.Tensor:
        """
        Calculates the dynamic guidance scale for the current step,
        combining Feedback Guidance (FBG) with optional Classifier-Free Guidance (CFG).
        """
        config = self.config
        
        # Determine if FBG and CFG are active based on sigma ranges.
        using_fbg = config.fbg_end_sigma <= sigma_item <= config.fbg_start_sigma
        using_cfg = config.cfg_scale != 1 and (
            config.cfg_end_sigma <= sigma_item <= config.cfg_start_sigma
        )

        # Initialize guidance scale to 1.0 (no guidance)
        guidance_scale = log_posterior_val.new_ones(guidance_scale_prev.shape[0])

        if using_fbg:
            # Calculate FBG component based on log_posterior (Eq. 8 in paper).
            denom = log_posterior_val.exp() - (1.0 - config.pi)
            # Add a small epsilon to prevent division by zero if denom is exactly zero
            safe_denom = torch.where(denom.abs() < 1e-6, torch.full_like(denom, 1e-6), denom)
            
            fbg_component = log_posterior_val.exp() / safe_denom
            fbg_component *= config.fbg_guidance_multiplier
            guidance_scale = fbg_component
                
            # Clamp the FBG component itself before adding CFG, to prevent intermediate explosion
            guidance_scale = guidance_scale.clamp(1.0, config.max_guidance_scale)

            if self.debug_mode:
                print(f"  FBG active (sigma {sigma_item:.3f}). Raw FBG component: {guidance_scale.flatten().detach().tolist()}")
        else:
            if self.debug_mode:
                print(f"  FBG inactive (sigma {sigma_item:.3f} not in [{config.fbg_end_sigma:.3f}, {config.fbg_start_sigma:.3f}]).")

        if using_cfg:
            # Add the Classifier-Free Guidance component.
            guidance_scale += config.cfg_scale - 1.0
            if self.debug_mode:
                print(f"  CFG active (sigma {sigma_item:.3f}). CFG component added: {config.cfg_scale - 1.0:.2f}.")
        else:
            if self.debug_mode:
                print(f"  CFG inactive (sigma {sigma_item:.3f} not in [{config.cfg_end_sigma:.3f}, {config.cfg_start_sigma:.3f}]).")

        # Clamp the combined guidance scale to the overall maximum allowed value.
        guidance_scale = guidance_scale.clamp(1.0, config.max_guidance_scale).view(
            guidance_scale_prev.shape
        )
        
        # Apply the guidance_max_change constraint.
        # Ensure guidance_scale_prev is not zero for division
        safe_guidance_scale_prev = torch.where(guidance_scale_prev.abs() < 1e-6, torch.full_like(guidance_scale_prev, 1e-6), guidance_scale_prev)
        
        change_diff = guidance_scale - guidance_scale_prev
        change_pct = (change_diff / safe_guidance_scale_prev).clamp_(
            -config.guidance_max_change, config.guidance_max_change
        )
        
        guidance_scale_new = guidance_scale_prev + guidance_scale_prev * change_pct
        
        final_guidance_scale = guidance_scale_new.clamp_(1.0, config.max_guidance_scale)

        if self.debug_mode:
            print(f"  Sigma: {sigma_item:.4f}, Prev GS: {guidance_scale_prev.mean().item():.4f}, "
                  f"Calculated GS: {guidance_scale.mean().item():.4f}, "
                  f"Final GS: {final_guidance_scale.mean().item():.4f} (max change: {config.guidance_max_change})")

        return final_guidance_scale

    def _model_denoise_with_guidance(self, x_tensor: torch.Tensor, sigma_scalar: torch.Tensor, override_cfg: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Wrapper around the underlying diffusion model's denoising function,
        modified to handle Classifier-Free Guidance (CFG) and capture conditional/unconditional outputs
        as required by FBG.
        """
        sigma_tensor = sigma_scalar * x_tensor.new_ones((x_tensor.shape[0],))
        cond = uncond = None

        def post_cfg_function(args: Dict[str, torch.Tensor]) -> torch.Tensor:
            """
            Internal function to capture conditional and unconditional denoised
            outputs after the model's internal CFG optimization.
            """
            nonlocal cond, uncond
            cond, uncond = args["cond_denoised"], args["uncond_denoised"]
            return args["denoised"]

        extra_args = self.extra_args.copy()
        orig_model_options = extra_args.get("model_options", {})
        model_options = orig_model_options.copy()
        
        model_options["disable_cfg1_optimization"] = True # Essential for capturing cond/uncond via post_cfg_function
        
        extra_args["model_options"] = model_patcher.set_model_options_post_cfg_function(
            model_options, post_cfg_function
        )
        inner_model = self.model_.inner_model

        if (override_cfg is None or (isinstance(override_cfg, torch.Tensor) and override_cfg.numel() < 2)) and hasattr(inner_model, "cfg"):
            # If no override or a simple scalar override, and model has internal CFG,
            # temporarily set model's CFG and predict noise.
            orig_cfg = inner_model.cfg
            try:
                if override_cfg is not None:
                    # If override_cfg is a scalar tensor, use its item() value
                    if isinstance(override_cfg, torch.Tensor) and override_cfg.numel() == 1:
                        inner_model.cfg = override_cfg.detach().item()
                    elif isinstance(override_cfg, torch.Tensor): # Multi-element tensor, use mean as a fallback
                        if self.debug_mode:
                            print(f"Warning: override_cfg is not scalar ({override_cfg.shape}). Using mean for inner_model.cfg.")
                        inner_model.cfg = override_cfg.mean().detach().item()
                    else: # If it's a simple float/int
                        inner_model.cfg = override_cfg

                denoised = inner_model.predict_noise(
                    x_tensor,
                    sigma_tensor,
                    model_options=extra_args["model_options"],
                    seed=extra_args.get("seed"),
                )
            finally:
                inner_model.cfg = orig_cfg
        else:
            # Otherwise, use the general model call and apply cfg_function manually.
            _ = self.model_(x_tensor, sigma_tensor, **extra_args) # No kwargs here, already in extra_args
            denoised = cfg_function(
                inner_model.inner_model,
                cond,
                uncond,
                override_cfg, # Dynamic guidance scale passed here
                x_tensor,
                sigma_tensor,
                model_options=orig_model_options,
            )
        return denoised, cond, uncond

    def _update_log_posterior(
        self,
        prev_log_posterior: torch.Tensor,
        x_curr: torch.Tensor,
        x_next: torch.Tensor,
        t_curr: torch.Tensor,
        t_next: torch.Tensor,
        uncond: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        Updates the FBG log posterior estimate.
        """
        # Ensure t_curr is not zero to prevent division by zero
        if torch.isclose(t_curr, torch.tensor(0.0)).all():
            if self.debug_mode:
                print("Warning: t_curr is zero in _update_log_posterior. Skipping update.")
            return prev_log_posterior # Return previous value, or handle as error

        t_csq = t_curr**2
        
        # Ensure t_csq is not zero before division
        if torch.isclose(t_csq, torch.tensor(0.0)).all():
            if self.debug_mode:
                print("Warning: t_csq is zero in _update_log_posterior. Skipping update.")
            return prev_log_posterior


        t_ndc = t_next**2 / t_csq
        t_cmn = t_csq - t_next**2
        sigma_square_tilde_t = t_cmn * t_ndc

        # Ensure sigma_square_tilde_t is not zero for the division below.
        # This can happen if t_cmn is zero (t_curr == t_next), or t_ndc is zero (t_next == 0).
        if torch.isclose(sigma_square_tilde_t, torch.tensor(0.0)).all():
             if self.debug_mode:
                print("Warning: sigma_square_tilde_t is zero in _update_log_posterior. Skipping update.")
             return prev_log_posterior

        pred_base = t_ndc * x_curr
        uncond_pred_mean = pred_base + t_cmn / t_csq * uncond
        cond_pred_mean = pred_base + t_cmn / t_csq * cond

        diff = batch_mse_loss(x_next, cond_pred_mean) - batch_mse_loss(
            x_next, uncond_pred_mean
        )

        # Check for potential division by zero before applying config.temp / (2 * sigma_square_tilde_t)
        if sigma_square_tilde_t == 0: # Already checked above, but double-safety for this specific line
             result = prev_log_posterior + self.config.offset
             if self.debug_mode:
                print("Warning: sigma_square_tilde_t is zero during log posterior update. Temp term ignored.")
        else:
            result = (
                prev_log_posterior
                - self.config.temp / (2 * sigma_square_tilde_t) * diff
                + self.config.offset
            )
        
        return result.clamp_(
            self.minimal_log_posterior, self.config.max_posterior_scale
        )


    def _do_callback(self, step_idx, current_x, current_sigma, denoised_sample):
        """
        Forwards progress information and intermediate results to ComfyUI's callback system.
        """
        if self.callback_:
            self.callback_({
                "i": step_idx,
                "x": current_x,
                "sigma": current_sigma,
                "sigma_hat": current_sigma,
                "denoised": denoised_sample
            })

    def __call__(self): # This is the main sampling loop execution
        """
        Executes the main "ping-pong" sampling loop, integrating FBG's dynamic guidance.
        """
        if self.debug_mode:
            print("PingPongSampler.__call__ (sampling loop) started.")

        x_current = self.x.clone()
        num_steps = len(self.sigmas) - 1

        if num_steps <= 0:
            if self.enable_clamp_output:
                x_current = torch.clamp(x_current, -1.0, 1.0)
            return x_current
            
        astart = self.first_ancestral_step
        aend = self.last_ancestral_step
        
        actual_iteration_end_idx = self.end_sigma_index if self.end_sigma_index >= 0 else num_steps - 1
        actual_iteration_end_idx = min(actual_iteration_end_idx, num_steps - 1)


        for idx in trange(num_steps, disable=self.disable_pbar):
            if idx < self.start_sigma_index or idx > actual_iteration_end_idx:
                continue

            sigma_current, sigma_next = self.sigmas[idx], self.sigmas[idx + 1]
            sigma_item = sigma_current.max().detach().item() # Scalar sigma for FBG checks
            sigma_next_item = sigma_next.min().detach().item() # Scalar next sigma

            # --- FBG: Calculate Dynamic Guidance Scale ---
            # Update the internal guidance_scale for the next step.
            self.guidance_scale = self.get_dynamic_guidance_scale(
                self.log_posterior, self.guidance_scale, sigma_item
            )

            # --- Call the model with the dynamic guidance scale ---
            denoised_sample, cond, uncond = self._model_denoise_with_guidance(
                x_current, sigma_current, override_cfg=self.guidance_scale
            )
            
            self._do_callback(idx, x_current, sigma_current, denoised_sample)

            if self.debug_mode:
                pretty_scales = ", ".join(f"{scale:.02f}" for scale in self.guidance_scale.flatten().detach().tolist())
                print(f"Step {idx}: Sigma {sigma_item:.3f}, Next {sigma_next_item:.3f}, GS: {pretty_scales}")
                print(f"  Log Posterior mean: {self.log_posterior.mean().item():.4f}")
            
            # --- FBG: Update Log Posterior for Next Iteration ---
            # This uses x_current (before sampler step), x_next (after sampler step),
            # and the cond/uncond predictions.
            # We need x_next for this, which means we must perform the sampler step first.
            # So, we save x_current (x_orig) for the log_posterior update BEFORE it gets modified.
            x_orig_for_fbg_update = x_current.clone() # Clone before it's modified by the sampler step


            # --- PingPong Sampler Step Logic (Ancestral / DDIM-like) ---
            # Determine whether ancestral noise injection should be used for this specific step
            use_anc = (astart <= idx <= aend) if astart <= aend else False

            if not use_anc:
                # Non-Ancestral Step (DDIM-like interpolation)
                blend = sigma_next / sigma_current if sigma_current > 0 else 0.0
                x_current = self.step_blend_function(denoised_sample, x_current, blend)
            else:
                # Ancestral Step Logic: Injecting controlled noise
                local_seed = self._stepped_seed(idx)
                if local_seed is not None:
                    torch.manual_seed(local_seed)

                noise_sample = self.noise_sampler(sigma_current, sigma_next) 
                final_noise_to_add = noise_sample # No s_noise/dynamic_noise as per PingPong's optimization

                if self.is_rf:
                    x_current = self.step_blend_function(denoised_sample, final_noise_to_add, sigma_next)
                else:
                    x_current = denoised_sample + final_noise_to_add * sigma_next

            # --- FBG: Perform Log Posterior Update with the new x_current (as x_next) ---
            # The x_current here is the x_next for the FBG update.
            # We use x_orig_for_fbg_update as x_curr, and the current x_current as x_next.
            self.log_posterior = self._update_log_posterior(
                self.log_posterior, x_orig_for_fbg_update, x_current, sigma_current, sigma_next, uncond, cond
            )

            # Check for early exit conditions (clamping or sigma_next near zero)
            if self.enable_clamp_output and sigma_next_item < 1e-3:
                if self.debug_mode:
                    print(f"Final clamp condition met (sigma_next {sigma_next_item:.6f} < 1e-3). Clamping and breaking.")
                x_current = torch.clamp(x_current, -1.0, 1.0)
                break
            
            if sigma_next_item <= 1e-6:
                if self.debug_mode:
                    print(f"Sigma_next {sigma_next_item:.6f} is near zero. Finishing sampling.")
                x_current = denoised_sample
                break

        # Final clamping for the very last step if not already handled
        if self.enable_clamp_output and sigma_next_item > 1e-6: # Only clamp if not already broken by small sigma_next
             if self.debug_mode:
                 print("Final clamping after loop.")
             x_current = torch.clamp(x_current, -1.0, 1.0)

        if self.debug_mode:
            print("PingPongSampler.__call__ (sampling loop) finished.")
        return x_current

    @staticmethod
    def go(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, **kwargs):
        """
        Static method to serve as the entry point for ComfyUI's KSAMPLER.
        It instantiates PingPongSampler and runs its __call__ method.
        All parameters from the ComfyUI node are passed via kwargs.
        """
        # Extract FBGConfig from kwargs, then build the FBGConfig object
        fbg_config_kwargs_raw = kwargs.pop("fbg_config", {})
        
        # --- Robust YAML Key Remapping for FBGConfig Parameters ---
        # This ensures that even if YAML uses the node input names (e.g., 'fbg_sampler_mode')
        # instead of the FBGConfig's internal field names (e.g., 'sampler_mode'),
        # the FBGConfig object is created correctly.
        fbg_config_kwargs = {}
        remap_rules = {
            "fbg_sampler_mode": "sampler_mode",
            "fbg_temp": "temp",
            "fbg_offset": "offset",
            "log_posterior_initial_value": "initial_value",
            # Add other remapping if needed, but these are the primary ones based on INPUT_TYPES
        }

        for key, value in fbg_config_kwargs_raw.items():
            if key in remap_rules:
                fbg_config_kwargs[remap_rules[key]] = value
                if kwargs.get("debug_mode", False):
                    print(f"PingPongSampler.go Info: Remapped YAML FBG config key '{key}' to '{remap_rules[key]}'.")
            else:
                fbg_config_kwargs[key] = value

        # Special handling for sampler_mode: ensure it's converted to Enum member if present
        if "sampler_mode" in fbg_config_kwargs and isinstance(fbg_config_kwargs["sampler_mode"], str):
            try:
                fbg_config_kwargs["sampler_mode"] = getattr(SamplerMode, fbg_config_kwargs["sampler_mode"].upper())
            except AttributeError:
                if kwargs.get("debug_mode", False):
                    print(f"PingPongSampler.go Warning: Invalid FBG sampler_mode '{fbg_config_kwargs['sampler_mode']}'. Falling back to default.")
                # If conversion fails, remove it to let FBGConfig use its default
                fbg_config_kwargs.pop("sampler_mode", None)
        elif "sampler_mode" not in fbg_config_kwargs: # Ensure it has a default if not provided
             fbg_config_kwargs["sampler_mode"] = FBGConfig().sampler_mode # Use FBGConfig's default

        fbg_config_instance = FBGConfig(**fbg_config_kwargs)

        # Extract pingpong_options from kwargs
        pingpong_options_kwargs = kwargs.pop("pingpong_options", {})

        # Extract blend functions which are passed by name from node but need to be actual functions
        # Use .get with a default to prevent KeyError if these aren't present (e.g., in YAML without these keys)
        blend_function = _INTERNAL_BLEND_MODES.get(kwargs.pop("blend_function_name", "lerp"), torch.lerp)
        step_blend_function = _INTERNAL_BLEND_MODES.get(kwargs.pop("step_blend_function_name", "lerp"), torch.lerp)


        # Instantiate PingPongSampler with all required arguments
        # The remaining kwargs should directly map to PingPongSampler.__init__ parameters
        sampler_instance = PingPongSampler(
            model=model,
            x=x,
            sigmas=sigmas,
            extra_args=extra_args,
            callback=callback,
            disable=disable,
            noise_sampler=noise_sampler,
            blend_function=blend_function,
            step_blend_function=step_blend_function,
            fbg_config=fbg_config_instance,
            pingpong_options=pingpong_options_kwargs,
            **kwargs # Pass remaining kwargs which should be direct __init__ parameters
        )
        return sampler_instance() # Execute the sampling loop

class PingPongSamplerNode:
    """
    ComfyUI node wrapper to register PingPongSampler as a custom sampler.
    This class defines the input parameters that will be exposed in the ComfyUI user interface.
    """
    CATEGORY = "sampling/custom_sampling/samplers"
    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "get_sampler"

    @classmethod
    def INPUT_TYPES(cls):
        """
        Defines the input parameters that will be exposed in the ComfyUI node editor.
        """
        # We need this defaults object to populate the tooltip values correctly.
        defaults_fbg_config = FBGConfig() 
        return {
            "required": {
                # --- PingPong Sampler Specific Inputs ---
                "step_random_mode":    (["off", "block", "reset", "step"], {"default": "block", "tooltip": "Controls how the RNG seed varies per sampling step.\n- 'off': Seed is constant. Predictable, but where's the fun in that?\n- 'block': Seed changes every 'step_size' frames. Like a glitch in the matrix, but intentional.\n- 'reset': Seed is reset based on 'step_size' multiplied by the frame index, offering more varied randomness.\n- 'step': Seed changes incrementally by the frame index at each step, providing subtle variations."}),
                "step_size":           ("INT",    {"default": 5,    "min": 1,        "max": 100, "tooltip": "Used by 'block' and 'reset' step random modes to define the block/reset interval for the seed."}),
                "seed":                ("INT",    {"default": 80085,  "min": 0,        "max": 2**32 - 1, "tooltip": "Base random seed. The cosmic initializer. Change it for new universes, keep it for deja vu."}),
                "first_ancestral_step": ("INT",    {"default": 0, "min": -1, "max": 10000, "tooltip": "The sampler step index (0-based) to begin ancestral\nnoise injection (ping-pong behavior). Use -1 to effectively disable ancestral noise if last_ancestral_step is also -1."}),
                "last_ancestral_step":  ("INT",    {"default": -1, "min": -1, "max": 10000, "tooltip": "The sampler step index (0-based) to end ancestral\nnoise injection (ping-pong behavior). Use -1 to extend ancestral noise to the end of the sampling process."}),
                "start_sigma_index":    ("INT",    {"default": 0,    "min": 0,        "max": 10000, "tooltip": "The index in the sigma array (denoising schedule) to begin sampling from. Allows skipping initial high-noise steps, potentially speeding up generation or changing visual character."}),
                "end_sigma_index":      ("INT",    {"default": -1,   "min": -10000,  "max": 10000, "tooltip": "The index in the sigma array to end sampling at. -1 means sample all steps. To the bitter end, or a graceful exit? You decide."}),
                "enable_clamp_output":  ("BOOLEAN", {"default": False, "tooltip": "If true, clamps the final output latent to the range [-1.0, 1.0]. Useful for preventing extreme values that might lead to artifacts during decoding."}),
                "scheduler":           ("SCHEDULER", {"tooltip": "Connect a ComfyUI scheduler node (e.g., KSamplerScheduler) to define the noise decay curve for the sampler. Essential for proper sampling progression. It's the tempo track for your pixels!"}),
                "blend_mode":          (tuple(_INTERNAL_BLEND_MODES.keys()), {"default": "lerp", "tooltip": "Blend mode to use for blending noise in ancestral steps. Defaults to 'lerp' (linear interpolation). Choose your flavor: 'lerp' (smooth blend), 'a_only' (take noise as is), 'b_only' (take other input as is). Fancy, right?"}),
                "step_blend_mode":     (tuple(_INTERNAL_BLEND_MODES.keys()), {"default": "lerp", "tooltip": "Blend mode to use for non-ancestral steps (regular denoising progression). Changing this from 'lerp' is generally not recommended unless you're feeling particularly chaotic. Like trying to render Doom on a 386SX with 2MB RAM."}),
                
                # --- FBG Specific Inputs ---
                "fbg_sampler_mode": (
                    tuple(SamplerMode.__members__), # Exposes "EULER", "PINGPONG" as selectable strings for FBG's base sampler
                    {
                        "default": defaults_fbg_config.sampler_mode.name, # Use .name for Enum string
                        "tooltip": (
                            "FBG's base sampler mode for internal calculations. 'EULER' is standard. 'PINGPONG' "
                            "influences how FBG calculates its internal step, but the main PingPongSampler's "
                            "noise injection logic remains dominant."
                        ),
                    },
                ),
                "cfg_scale": (
                    "FLOAT",
                    {
                        "default": defaults_fbg_config.cfg_scale,
                        "min": -1000.0,
                        "max": 1000.0,
                        "step": 0.01,
                        "tooltip": "Base Classifier-Free Guidance (CFG) scale. FBG dynamically modifies this value during sampling.",
                    },
                ),
                "cfg_start_sigma": (
                    "FLOAT",
                    {
                        "default": defaults_fbg_config.cfg_start_sigma, # Updated default
                        "min": 0.0,
                        "max": 9999.0,
                        "step": 0.001, 
                        "tooltip": "The noise level (sigma) at which CFG (and thus FBG's influence over it) begins to be active.",
                    },
                ),
                "cfg_end_sigma": (
                    "FLOAT",
                    {
                        "default": defaults_fbg_config.cfg_end_sigma, # Updated default
                        "min": 0.0,
                        "max": 9999.0,
                        "step": 0.001, 
                        "tooltip": "The noise level (sigma) at which CFG (and thus FBG's influence over it) ceases to be active.",
                    },
                ),
                "fbg_start_sigma": (
                    "FLOAT",
                    {
                        "default": defaults_fbg_config.fbg_start_sigma, # Updated default
                        "min": 0.0,
                        "max": 9999.0,
                        "step": 0.001, 
                        "tooltip": "The noise level (sigma) at which Feedback Guidance (FBG) actively calculates and applies its dynamic scale.",
                    },
                ),
                "fbg_end_sigma": (
                    "FLOAT",
                    {
                        "default": defaults_fbg_config.fbg_end_sigma, # Updated default
                        "min": 0.0,
                        "max": 9999.0,
                        "step": 0.001, 
                        "tooltip": "The noise level (sigma) at which FBG ceases to actively calculate and apply its dynamic scale.",
                    },
                ),
                "ancestral_start_sigma": (
                    "FLOAT",
                    {
                        "default": defaults_fbg_config.ancestral_start_sigma, # Updated default
                        "min": 0.0,
                        "max": 9999.0,
                        "step": 0.001, 
                        "tooltip": "FBG internal parameter: First sigma ancestral/pingpong sampling (for FBG's base sampler) will be active. Note: ETA (for FBG) must also be non-zero.",
                    },
                ),
                "ancestral_end_sigma": (
                    "FLOAT",
                    {
                        "default": defaults_fbg_config.ancestral_end_sigma, # Updated default
                        "min": 0.0,
                        "max": 9999.0,
                        "step": 0.001, 
                        "tooltip": "FBG internal parameter: Last sigma ancestral/pingpong sampling (for FBG's base sampler) will be active.",
                    },
                ),
                "max_guidance_scale": (
                    "FLOAT",
                    {
                        "default": defaults_fbg_config.max_guidance_scale,
                        "min": 1.0,
                        "max": 1000.0,
                        "step": 0.01,
                        "tooltip": "Upper limit for the total guidance scale after FBG and CFG adjustments.",
                    },
                ),
                "initial_guidance_scale": (
                    "FLOAT",
                    {
                        "default": defaults_fbg_config.initial_guidance_scale,
                        "min": 1.0,
                        "max": 1000.0,
                        "step": 0.01,
                        "tooltip": "Initial value for FBG's internal guidance scale. Primarily affects how 'guidance_max_change' starts.",
                    },
                ),
                "guidance_max_change": (
                    "FLOAT",
                    {
                        "default": defaults_fbg_config.guidance_max_change,
                        "min": 0.0,
                        "max": 1000.0,
                        "step": 0.01,
                        "tooltip": "Limits the percentage change of the FBG guidance scale per step. A value like 0.5 means max 50% change. High value (e.g., 1000) disables limiting.",
                    },
                ),
                "pi": (
                    "FLOAT",
                    {
                        "default": defaults_fbg_config.pi,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": (
                            "The mixing parameter (pi) from the FBG paper. Higher values (e.g., 0.95-0.999) "
                            "are typically for very well-learned models (like those benchmarked in the FBG paper). "
                            "Lower values (e.g., 0.2-0.8) may be more effective for less optimized or general "
                            "Text-to-Image models (e.g., Stable Diffusion), as it causes guidance to activate at "
                            "higher posterior probabilities. Adjust based on model type and desired effect. "
                            "Setting pi to 1.0 implies the conditional model is perfect and needs no guidance."
                        ),
                    },
                ),
                "t_0": (
                    "FLOAT",
                    {
                        "default": defaults_fbg_config.t_0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Normalized diffusion time (0-1) where FBG guidance scale reaches a reference value. If 0, 'fbg_temp' and 'fbg_offset' are used directly.",
                    },
                ),
                "t_1": (
                    "FLOAT",
                    {
                        "default": defaults_fbg_config.t_1,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Normalized diffusion time (0-1) where FBG guidance is estimated to reach its maximum. If 0, 'fbg_temp' and 'fbg_offset' are used directly.",
                    },
                ),
                "fbg_temp": (
                    "FLOAT",
                    {
                        "default": defaults_fbg_config.temp,
                        "min": -1000.0,
                        "max": 1000.0,
                        "step": 0.001,
                        "tooltip": "Temperature parameter for FBG log posterior update. Only applies if both t_0 and t_1 are 0, otherwise it is calculated automatically.",
                    },
                ),
                "fbg_offset": (
                    "FLOAT",
                    {
                        "default": defaults_fbg_config.offset,
                        "min": -1000.0,
                        "max": 1000.0,
                        "step": 0.001,
                        "tooltip": "Offset parameter for FBG log posterior update. Only applies if both t_0 and t_1 are 0, otherwise it is calculated automatically.",
                    },
                ),
                "log_posterior_initial_value": (
                    "FLOAT",
                    {
                        "default": defaults_fbg_config.initial_value,
                        "min": -1000.0,
                        "max": 1000.0,
                        "step": 0.01,
                        "tooltip": "Initial value for FBG's internal log posterior estimate. Typically, this does not need to be changed.",
                    },
                ),
                "fbg_guidance_multiplier": (
                    "FLOAT",
                    {
                        "default": defaults_fbg_config.fbg_guidance_multiplier,
                        "min": 0.001,
                        "max": 1000.0,
                        "step": 0.01,
                        "tooltip": "A scalar multiplier applied to the FBG guidance scale before combining with base CFG.",
                    },
                ),
                 "fbg_eta": (
                    "FLOAT",
                    {
                        "default": 0.0, # Default as per FBG paper's suggested eta for optimal results
                        "min": -1000.0,
                        "max": 1000.0,
                        "step": 0.01,
                        "tooltip": "FBG internal parameter: Controls the amount of noise added during ancestral sampling *within FBG's model prediction step*. Must be >0 for ancestral to activate for FBG.",
                    },
                ),
                "fbg_s_noise": (
                    "FLOAT",
                    {
                        "default": 0.7,
                        "min": -1000.0,
                        "max": 1000.0,
                        "step": 0.01,
                        "tooltip": "FBG internal parameter: Scale for noise added during ancestral sampling *within FBG's model prediction step*.",
                    },
                ),
                # New Debug Mode Input
                "debug_mode": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Enable verbose debug messages in the ComfyUI console. Useful for troubleshooting FBG and sampler behavior.",
                    },
                ),
            },
            "optional": {
                "yaml_settings_str": ("STRING", {"multiline": True, "default": "", "dynamic_prompt": False, "tooltip": "YAML string to configure sampler parameters. Parameters provided directly via the ComfyUI node's inputs will now be **OVERRIDDEN** by any corresponding values set in the YAML string. If the YAML is empty, node inputs are used. YAML is the boss now, respect its authority!"}),
            }
        }

    def get_sampler(
        self,
        # All of these are 'required' in INPUT_TYPES and will be provided by ComfyUI.
        # They do NOT need default values in the function signature here.
        step_random_mode: str,
        step_size: int,
        seed: int,
        first_ancestral_step: int,
        last_ancestral_step: int,
        start_sigma_index: int,
        end_sigma_index: int,
        enable_clamp_output: bool,
        scheduler, # ComfyUI handles its default/None if not connected
        blend_mode: str, # Passed as string, convert to function
        step_blend_mode: str, # Passed as string, convert to function
        fbg_sampler_mode: str,
        cfg_scale: float,
        cfg_start_sigma: float,
        cfg_end_sigma: float,
        fbg_start_sigma: float,
        fbg_end_sigma: float,
        ancestral_start_sigma: float,
        ancestral_end_sigma: float,
        max_guidance_scale: float,
        log_posterior_initial_value: float,
        initial_guidance_scale: float,
        guidance_max_change: float,
        pi: float,
        t_0: float,
        t_1: float,
        fbg_temp: float,
        fbg_offset: float,
        fbg_guidance_multiplier: float,
        fbg_eta: float,
        fbg_s_noise: float,
        debug_mode: bool,
        # Only 'optional' parameters should have defaults here.
        yaml_settings_str: str = ""
    ):
        """
        This method gathers all input parameters from the ComfyUI node,
        merges them with any provided YAML settings (prioritizing YAML),
        and returns a KSAMPLER object configured with the PingPongSampler.
        """
        # Create FBGConfig object from FBG-specific node inputs
        # The direct node inputs already map correctly due to get_sampler's arguments
        fbg_config_instance_from_direct_inputs = FBGConfig(
            sampler_mode=getattr(SamplerMode, fbg_sampler_mode.upper()),
            cfg_start_sigma=cfg_start_sigma,
            cfg_end_sigma=cfg_end_sigma,
            fbg_start_sigma=fbg_start_sigma,
            fbg_end_sigma=fbg_end_sigma,
            ancestral_start_sigma=ancestral_start_sigma,
            ancestral_end_sigma=ancestral_end_sigma,
            cfg_scale=cfg_scale,
            max_guidance_scale=max_guidance_scale,
            initial_guidance_scale=initial_guidance_scale,
            guidance_max_change=guidance_max_change,
            temp=fbg_temp,
            offset=fbg_offset,
            pi=pi,
            t_0=t_0,
            t_1=t_1,
            initial_value=log_posterior_initial_value,
            fbg_guidance_multiplier=fbg_guidance_multiplier,
        )


        # Create a dictionary with direct node inputs for PingPongSampler's __init__.
        # Pass blend_mode strings and convert later inside PingPongSampler.go
        direct_inputs = {
            "step_random_mode": step_random_mode,
            "step_size": step_size,
            "seed": seed,
            "pingpong_options": {
                "first_ancestral_step": first_ancestral_step,
                "last_ancestral_step": last_ancestral_step,
            },
            "start_sigma_index": start_sigma_index,
            "end_sigma_index": end_sigma_index,
            "enable_clamp_output": enable_clamp_output,
            "scheduler": scheduler,
            "blend_function_name": blend_mode, # Pass the string name
            "step_blend_function_name": step_blend_mode, # Pass the string name
            "fbg_config": fbg_config_instance_from_direct_inputs._asdict(), # Pass FBGConfig as dict to be recreated in go()
            "eta": fbg_eta, 
            "s_noise": fbg_s_noise, 
            "debug_mode": debug_mode,
        }

        # Initialize final_options with direct node inputs first.
        final_options = direct_inputs.copy()

        # Attempt to load YAML data.
        yaml_data = {}
        if yaml_settings_str:
            try:
                temp_yaml_data = yaml.safe_load(yaml_settings_str)
                if isinstance(temp_yaml_data, dict):
                    yaml_data = temp_yaml_data
                    # Convert boolean strings in YAML to actual booleans
                    for k, v in yaml_data.items():
                        if isinstance(v, str):
                            if v.lower() == 'true':
                                yaml_data[k] = True
                            elif v.lower() == 'false':
                                yaml_data[k] = False

            except yaml.YAMLError as e:
                print(f"WARNING: PingPongSamplerNode YAML parsing error: {e}. Using direct node inputs only.")
            except Exception as e:
                print(f"WARNING: PingPongSamplerNode unexpected error during YAML parsing: {e}. Using direct node inputs only.")

        # --- MERGE YAML DATA ON TOP OF DIRECT INPUTS ---
        # YAML data now overrides direct inputs.
        for key, value in yaml_data.items():
            if key == "pingpong_options" and isinstance(value, dict) and key in final_options and isinstance(final_options[key], dict):
                final_options[key].update(value)
            elif key == "fbg_config" and isinstance(value, dict):
                # Ensure fbg_config in final_options is a dict for updating
                if "fbg_config" not in final_options or not isinstance(final_options["fbg_config"], dict):
                    final_options["fbg_config"] = {}
                final_options["fbg_config"].update(value)
            else:
                final_options[key] = value

        return (KSAMPLER(PingPongSampler.go, extra_options=final_options),)

# Dictionary mapping class names to identifiers used by ComfyUI
NODE_CLASS_MAPPINGS = {
    # Changed the key here to make it unique from the original PingPongSampler_Custom
    "PingPongSampler_Custom_FBG": PingPongSamplerNode,
}

# Dictionary mapping class names to display names in the ComfyUI UI
NODE_DISPLAY_NAME_MAPPINGS = {
    "PingPongSampler_Custom_FBG": "PingPong Sampler (Custom V0.9.9 FBG)", # Updated version number!
}
