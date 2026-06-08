# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# █▓▒░                                                                     ░▒▓█
# █▓▒░           MD_Nodes Wrapper: PingPong Sampler FBG (v1.6.0)           ░▒▓█
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
# ║ ░▒▓ DESCRIPTION:
# ║   Advanced Sampler integrating Forward-Backward Guidance (FBG)
# ║   and dynamic restart scheduling.
# ║   NOTE: This is a public wrapper. Missing binaries will safely degrade
# ║   to standard ComfyUI Euler sampling.
# ╚════════════════════════════════════════════════════════════════════════════


VERSION = "v1.6.0"  # UPS v1.5.8

import logging
import torch
import sys
import os
import yaml
from comfy.samplers import KSAMPLER
import comfy.model_management
import comfy.model_patcher
import comfy.model_sampling

# =================================================================================
# == MD_Nodes Universal Binary Loader (v1.6.1)
# =================================================================================

def find_core_paths():
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
    import pingpong_sampler_core_bin as core
    CORE_LOADED = True
    CORE_MODE = "Binary (Production)"
except ImportError as e1:
    try:
        import pingpong_sampler_core as core
        CORE_LOADED = True
        CORE_MODE = "Source (Development)"
    except ImportError as e2:
        CORE_ERROR = f"Binary: {e1}\nSource: {e2}"

logger = logging.getLogger("MD_Nodes.Samplers.PingPong")

# =================================================================================
# == Wrapper Logic (The Bridge)
# =================================================================================

def WrapperDenoiseFn(model, x, sigma, guidance_scale, extra_args):
    """Handles ComfyUI specific model patching and execution."""
    sigma_t = sigma * x.new_ones((x.shape[0],))
    cond = uncond = None
    
    def post_cfg(args):
        nonlocal cond, uncond
        cond, uncond = args.get("cond_denoised"), args.get("uncond_denoised")
        return args["denoised"]
    
    extra = extra_args.copy()
    mo = extra.get("model_options", {}).copy()
    mo["disable_cfg1_optimization"] = True
    extra["model_options"] = comfy.model_patcher.set_model_options_post_cfg_function(mo, post_cfg)
    
    try:
        inner = model.inner_model
        if (guidance_scale is None or guidance_scale.numel() < 2) and hasattr(inner, "cfg"):
            orig = inner.cfg
            try:
                if guidance_scale is not None:
                    inner.cfg = guidance_scale.item() if guidance_scale.numel() == 1 else guidance_scale.mean().item()
                denoised = inner.predict_noise(x, sigma_t, model_options=extra["model_options"], seed=extra.get("seed"))
            finally:
                inner.cfg = orig
        else:
            denoised = model(x, sigma_t, **extra)
            
    except Exception as e:
        logger.error(f"WrapperDenoiseFn Error: {e}")
        denoised = model(x, sigma_t, **extra)
        
    if cond is None: cond = uncond = denoised
    return denoised, cond, uncond

def WrapperDetectRF(model):
    """Detects if model is Rectified Flow."""
    try:
        curr = model
        while hasattr(curr, 'inner_model') and curr.inner_model is not None:
            curr = curr.inner_model
        if hasattr(curr, 'model_sampling') and curr.model_sampling is not None:
            return isinstance(curr.model_sampling, comfy.model_sampling.CONST)
    except Exception: pass
    return False

def SamplerEntry(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, **kwargs):
    """Static entry point for KSAMPLER to initialize Core or Graceful Fallback."""
    
    # --- Graceful Degradation Fallback ---
    if not CORE_LOADED:
        logger.warning(f"⚠️ PingPong Core Missing: {CORE_ERROR}. Falling back to standard Euler step.")
        total_steps = len(sigmas) - 1
        for i in range(total_steps):
            sigma_curr = sigmas[i]
            sigma_next = sigmas[i + 1]
            sigma_t = sigma_curr * x.new_ones((x.shape[0],))
            denoised = model(x, sigma_t, **extra_args)
            d = (x - denoised) / sigma_curr
            x = x + d * (sigma_next - sigma_curr)
            if callback:
                callback({'x': x, 'i': i, 'sigma': sigma_curr, 'sigma_hat': sigma_curr, 'denoised': x})
        return x
    
    # --- Standard Core Path ---
    fbg_config_raw = kwargs.pop("fbg_config", {})
    fbg_config_kwargs = {}
    if isinstance(fbg_config_raw, core.FBGConfig):
        fbg_config_kwargs = fbg_config_raw._asdict()
    elif isinstance(fbg_config_raw, dict):
        remap = {
            "fbg_sampler_mode": "sampler_mode",
            "fbg_temp": "temp",
            "fbg_offset": "offset",
            "log_posterior_initial_value": "initial_value"
        }
        for k, v in fbg_config_raw.items():
            fbg_config_kwargs[remap.get(k, k)] = v
            
    if "sampler_mode" in fbg_config_kwargs and isinstance(fbg_config_kwargs["sampler_mode"], str):
        try: fbg_config_kwargs["sampler_mode"] = getattr(core.SamplerMode, fbg_config_kwargs["sampler_mode"].upper())
        except Exception: fbg_config_kwargs.pop("sampler_mode", None)
    elif "sampler_mode" not in fbg_config_kwargs:
        fbg_config_kwargs["sampler_mode"] = core.FBGConfig().sampler_mode
        
    fbg_config_instance = core.FBGConfig(**fbg_config_kwargs)
    
    pingpong_options = kwargs.pop("pingpong_options", {})
    blend_fn = core._INTERNAL_BLEND_MODES.get(kwargs.pop("blend_function_name", "lerp"), torch.lerp)
    step_blend_fn = core._INTERNAL_BLEND_MODES.get(kwargs.pop("step_blend_function_name", "lerp"), torch.lerp)
    cond_blend_fn = core._INTERNAL_BLEND_MODES.get(kwargs.pop("conditional_blend_function_name", "slerp"), core.slerp)
    
    is_rf = WrapperDetectRF(model)
    
    sampler = core.PingPongSamplerCore(
        model=model,
        x=x,
        sigmas=sigmas,
        denoise_impl=lambda x, s, g, e: WrapperDenoiseFn(model, x, s, g, e),
        is_rf=is_rf,
        extra_args=extra_args,
        callback=callback,
        disable=disable,
        noise_sampler=noise_sampler,
        blend_function=blend_fn,
        step_blend_function=step_blend_fn,
        conditional_blend_function=cond_blend_fn,
        fbg_config=fbg_config_instance,
        pingpong_options=pingpong_options,
        **kwargs
    )
    
    return sampler.execute_sampling()

# =================================================================================
# == Node 1: Full Control (FBG)
# =================================================================================

class PingPongSamplerNodeFBG:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "step_random_mode": (["off", "block", "reset", "step"], {
                    "default": "block", 
                    "tooltip": (
                        "STEP RANDOMIZATION MODE\n"
                        "• Purpose: Controls how the seed evolves during sampling steps.\n"
                        "• Options: off (static), block (periodic), reset (per-step), step (incremental).\n"
                        "• Trade-offs: 'reset' is chaotic, 'off' is rigid.\n"
                        "\n⭐ Recommended: 'block' for best texture variation."
                    )
                }),
                "step_size": ("INT", {
                    "default": 4, "min": 1, "max": 100, 
                    "tooltip": (
                        "STEP SIZE INTERVAL\n"
                        "• Purpose: Defines how many sampling steps occur before the seed rotates.\n"
                        "\n⭐ Recommended: 4-10 steps."
                    )
                }),
                "seed": ("INT", {
                    "default": 80085, "min": 0, "max": 9007199254740991, 
                    "tooltip": "RANDOM SEED\n• Purpose: Base JS-Safe seed for noise generation."
                }),
                "first_ancestral_step": ("INT", {
                    "default": 0, 
                    "tooltip": (
                        "FIRST ANCESTRAL STEP\n"
                        "• Purpose: The step index to begin injecting ancestral noise.\n"
                        "\n⭐ Recommended: Increase to 3-5 if initial structural composition is messy."
                    )
                }),
                "last_ancestral_step": ("INT", {
                    "default": -1, 
                    "tooltip": (
                        "LAST ANCESTRAL STEP\n"
                        "• Purpose: The step index to stop injecting ancestral noise.\n"
                        "\n⭐ Recommended: -1 (Disable entirely) for standard deterministic diffusion."
                    )
                }),
                "ancestral_noise_type": (["gaussian", "uniform", "brownian"], {
                    "default": "gaussian", 
                    "tooltip": (
                        "ANCESTRAL NOISE TYPE\n"
                        "• Purpose: The statistical distribution of the injected noise.\n"
                        "• Options: Gaussian (standard bell), Uniform (flat limits), Brownian (correlated).\n"
                        "\n⭐ Recommended: Brownian for highly organic/analog textures."
                    )
                }),
                "start_sigma_index": ("INT", {"default": 0, "tooltip": "START SIGMA INDEX\n• Purpose: Skip early sampling steps (used for img2img routing)."}),
                "end_sigma_index": ("INT", {"default": -1, "tooltip": "END SIGMA INDEX\n• Purpose: Halt sampling early."}),
                "enable_clamp_output": ("BOOLEAN", {
                    "default": False, 
                    "tooltip": (
                        "CLAMP OUTPUT\n"
                        "• Purpose: Forces all output latent values strictly into the [-1, 1] range.\n"
                        "\n⭐ Recommended: False for audio generation (audio latents often exceed 1.0)."
                    )
                }),
                "scheduler": ("SCHEDULER", {"tooltip": "SCHEDULER INPUT\n• Purpose: Connects the sigma schedule (from Basic/Hybrid/GITS node)."}),
                "blend_mode": (["lerp", "slerp", "cosine", "cubic", "add", "a_only", "b_only"], {
                    "default": "lerp", 
                    "tooltip": (
                        "BLEND MODE (PRIMARY)\n"
                        "• Purpose: Mathematical approach for mixing new noise with the latent.\n"
                        "\n⭐ Recommended: 'lerp' for speed, 'slerp' for spherical structural integrity."
                    )
                }),
                "step_blend_mode": (["lerp", "slerp", "cosine", "cubic", "add", "a_only", "b_only"], {
                    "default": "lerp", 
                    "tooltip": "STEP BLEND MODE (SECONDARY)\n• Purpose: Interpolation used for internal dual-step calculations."
                }),
                
                # FBG Params
                "fbg_sampler_mode": (["EULER", "PINGPONG"], {
                    "default": "EULER", 
                    "tooltip": (
                        "FBG INTERNAL MODE\n"
                        "• Purpose: Defines the fundamental ODE solver step logic.\n"
                        "\n⭐ Recommended: EULER for standard Ace-Step audio models."
                    )
                }),
                "cfg_scale": ("FLOAT", {"default": 1.0, "min": -1000, "max": 1000, "step": 0.01, "tooltip": "CFG SCALE (BASE)\n• Purpose: Classifier-Free Guidance magnitude."}),
                "cfg_start_sigma": ("FLOAT", {"default": 1.0, "tooltip": "CFG START SIGMA"}),
                "cfg_end_sigma": ("FLOAT", {"default": 0.004, "tooltip": "CFG END SIGMA"}),
                "fbg_start_sigma": ("FLOAT", {"default": 1.0, "tooltip": "FBG START SIGMA"}),
                "fbg_end_sigma": ("FLOAT", {"default": 0.004, "tooltip": "FBG END SIGMA"}),
                "max_guidance_scale": ("FLOAT", {"default": 10.0, "tooltip": "MAX GUIDANCE CAP\n• Purpose: Hard ceiling limit for dynamic CFG scaling."}),
                "initial_guidance_scale": ("FLOAT", {"default": 1.0, "tooltip": "INITIAL GUIDANCE\n• Purpose: Starting CFG magnitude."}),
                "fbg_guidance_multiplier": ("FLOAT", {"default": 1.0, "tooltip": "FBG MULTIPLIER\n• Purpose: Strength of the feedback loop correction."}),
                "guidance_max_change": ("FLOAT", {"default": 1000.0, "tooltip": "MAX CHANGE PER STEP\n• Purpose: Velocity limit for guidance scaling."}),
                "pi": ("FLOAT", {"default": 0.5, "step": 0.01, "tooltip": "PI (POSTERIOR FACTOR)"}),
                "t_0": ("FLOAT", {"default": 0.5, "step": 0.01, "tooltip": "T0 OFFSET"}),
                "t_1": ("FLOAT", {"default": 0.4, "step": 0.01, "tooltip": "T1 TEMP FACTOR"}),
                "fbg_temp": ("FLOAT", {"default": 0.0, "tooltip": "FBG TEMPERATURE"}),
                "fbg_offset": ("FLOAT", {"default": 0.0, "tooltip": "FBG OFFSET"}),
                "fbg_eta": ("FLOAT", {"default": 0.0, "tooltip": "DEPRECATED: Legacy ETA parameter."}),
                "fbg_s_noise": ("FLOAT", {"default": 1.0, "tooltip": "DEPRECATED: Legacy noise scale."}),
                "max_posterior_scale": ("FLOAT", {"default": 3.0, "tooltip": "MAX POSTERIOR SCALE"}),
                "log_posterior_initial_value": ("FLOAT", {"default": 0.0, "tooltip": "INITIAL LOG POSTERIOR"}),
                "log_posterior_ema_factor": ("FLOAT", {"default": 0.0, "tooltip": "POSTERIOR EMA"}),
                
                # Enhanced
                "adaptive_noise_scaling": ("BOOLEAN", {"default": False, "tooltip": "ADAPTIVE NOISE SCALING"}),
                "noise_scale_factor": ("FLOAT", {"default": 1.0, "tooltip": "NOISE SCALE FACTOR"}),
                "progressive_blend_mode": ("BOOLEAN", {"default": False, "tooltip": "PROGRESSIVE BLEND"}),
                "conditional_blend_mode": ("BOOLEAN", {"default": False, "tooltip": "CONDITIONAL BLEND"}),
                "conditional_blend_sigma_threshold": ("FLOAT", {"default": 0.5, "tooltip": "BLEND THRESHOLD"}),
                "conditional_blend_function_name": (["lerp", "slerp", "cosine", "cubic", "add", "a_only", "b_only"], {"default": "slerp", "tooltip": "CONDITIONAL BLEND FUNC"}),
                "conditional_blend_on_change": ("BOOLEAN", {"default": False, "tooltip": "BLEND ON CHANGE"}),
                "conditional_blend_change_threshold": ("FLOAT", {"default": 0.1, "tooltip": "CHANGE THRESHOLD"}),
                "clamp_noise_norm": ("BOOLEAN", {"default": False, "tooltip": "CLAMP NOISE NORM"}),
                "max_noise_norm": ("FLOAT", {"default": 1.0, "tooltip": "MAX NOISE NORM"}),
                "gradient_norm_tracking": ("BOOLEAN", {"default": False, "tooltip": "GRADIENT TRACKING: Unused."}),
                "enable_profiling": ("BOOLEAN", {"default": False, "tooltip": "ENABLE PROFILING"}),
                "debug_mode": ("INT", {"default": 0, "min": 0, "max": 2, "tooltip": "DEBUG MODE"}),
                "tensor_memory_optimization": ("BOOLEAN", {"default": False, "tooltip": "MEMORY OPTIMIZATION"}),
                "early_exit_threshold": ("FLOAT", {"default": 1e-6, "tooltip": "EARLY EXIT THRESHOLD"}),
                "ancestral_start_sigma": ("FLOAT", {"default": 1.0, "tooltip": "ANCESTRAL START"}),
                "ancestral_end_sigma": ("FLOAT", {"default": 0.004, "tooltip": "ANCESTRAL END"}),
                "sigma_range_preset": (["Custom", "High", "Mid", "Low", "All"], {"default": "Custom", "tooltip": "SIGMA RANGE PRESET"}),
                
                # Res 2
                "enable_restarts": ("BOOLEAN", {"default": False, "tooltip": "ENABLE RES 2 RESTARTS"}),
            },
            "optional": {
                "yaml_settings_str": ("STRING", {"multiline": True, "default": "", "tooltip": "YAML OVERRIDE"}),
                "checkpoint_steps_str": ("STRING", {"default": "", "tooltip": "CHECKPOINT STEPS"}),
                "restart_mode": (["balanced", "aggressive", "conservative", "detail_focus", "composition_focus"], {"default": "balanced", "tooltip": "RESTART MODE"}),
                "restart_noise_scale": ("FLOAT", {"default": 0.5, "tooltip": "RESTART NOISE SCALE"}),
                "restart_s_noise": ("FLOAT", {"default": 1.0, "tooltip": "RESTART S_NOISE"}),
                "restart_steps": ("STRING", {"default": "", "tooltip": "CUSTOM RESTART STEPS"}),
            }
        }

    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "get_sampler"
    CATEGORY = "MD_Nodes/Samplers"
    
    def get_sampler(self, **kwargs):
        yaml_str = kwargs.pop("yaml_settings_str", "")
        if yaml_str:
            try:
                overrides = yaml.safe_load(yaml_str)
                if isinstance(overrides, dict):
                    for k, v in overrides.items():
                        if k == "fbg_config" and isinstance(v, dict):
                            for fk, fv in v.items():
                                kwargs[fk] = fv
                        else:
                            kwargs[k] = v
            except Exception: pass
        
        cp_str = kwargs.pop("checkpoint_steps_str", "")
        if cp_str:
            try: kwargs["checkpoint_steps"] = [int(x.strip()) for x in cp_str.split(",") if x.strip()]
            except Exception: pass
            
        kwargs['blend_function_name'] = kwargs.pop("blend_mode", "lerp")
        kwargs['step_blend_function_name'] = kwargs.pop("step_blend_mode", "lerp")
        
        return (KSAMPLER(SamplerEntry, extra_options=kwargs),)

# =================================================================================
# == Node 2: Basic
# =================================================================================

class PingPongSamplerNodeBasic:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "noise_behavior": (["Default (Raw)", "Dynamic", "Smooth"], {
                    "default": "Default (Raw)", 
                    "tooltip": (
                        "NOISE BEHAVIOR PRESETS\n"
                        "• Purpose: High-level mapping of ancestral noise generation.\n"
                        "\n⭐ Recommended: 'Default (Raw)' for standard generation."
                    )
                }),
                "seed": ("INT", {"default": 0, "min": 0, "max": 9007199254740991, "tooltip": "RANDOM SEED"}),
                "enable_restarts": ("BOOLEAN", {
                    "default": False, 
                    "tooltip": (
                        "ENABLE RESTARTS (Boomerang)\n"
                        "• Purpose: Briefly raises noise levels mid-generation to allow the model to fix detail mistakes.\n"
                        "• Trade-offs: Increases render time but significantly improves high-detail micro-textures.\n"
                        "\n⭐ Recommended: True for high-res final outputs."
                    )
                }),
                "scheduler": ("SCHEDULER", {"tooltip": "Connect Sigma Schedule node here."}),
            }
        }
    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "get_sampler"
    CATEGORY = "MD_Nodes/Samplers"
    
    def get_sampler(self, noise_behavior, seed, enable_restarts, scheduler):
        preset_map = {
            "Default (Raw)": {"ancestral_strength": 1.0, "noise_coherence": 0.0},
            "Dynamic": {"ancestral_strength": 1.0, "noise_coherence": 0.25},
            "Smooth": {"ancestral_strength": 0.8, "noise_coherence": 0.5},
        }
        preset = preset_map.get(noise_behavior, preset_map["Default (Raw)"])
        
        # Safely instantiate config if core is loaded, otherwise use dict for fallback
        if CORE_LOADED:
            fbg_cfg = core.FBGConfig(
                sampler_mode=core.SamplerMode.PINGPONG,
                cfg_scale=6.0,
                fbg_guidance_multiplier=400.0,
                max_guidance_scale=5000.0,
                initial_guidance_scale=250.0,
                guidance_max_change=500.0,
                pi=0.35, t_0=0.7, t_1=0.4
            )
        else:
            fbg_cfg = {}
        
        kwargs = {
            'fbg_config': fbg_cfg,
            'seed': seed,
            'enable_restarts': enable_restarts,
            'ancestral_strength': preset['ancestral_strength'],
            'noise_coherence': preset['noise_coherence'],
            'step_random_mode': "block",
            'step_size': 25,
            'blend_function_name': "lerp",
            'step_blend_function_name': "lerp",
            'restart_mode': "conservative" if enable_restarts else "balanced",
            'restart_noise_scale': 0.1,
            'start_sigma_index': 0, 'end_sigma_index': -1,
            'checkpoint_steps': []
        }
        return (KSAMPLER(SamplerEntry, extra_options=kwargs),)

# =================================================================================
# == Node 3: Lite
# =================================================================================

class PingPongSamplerNodeLite:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "noise_behavior": (["Default (Raw)", "Dynamic", "Smooth", "Textured Grain", "Soft", "Custom"], {"default": "Default (Raw)", "tooltip": "NOISE BEHAVIOR"}),
                "step_random_mode": (["off", "block", "reset", "step"], {"default": "block", "tooltip": "STEP RANDOMIZATION"}),
                "step_size": ("INT", {"default": 4, "tooltip": "STEP SIZE"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 9007199254740991, "tooltip": "SEED"}),
                "first_ancestral_step": ("INT", {"default": 0, "tooltip": "FIRST STEP"}),
                "last_ancestral_step": ("INT", {"default": -1, "tooltip": "LAST STEP"}),
                "start_sigma_index": ("INT", {"default": 0, "tooltip": "START IDX"}),
                "end_sigma_index": ("INT", {"default": -1, "tooltip": "END IDX"}),
                "enable_clamp_output": ("BOOLEAN", {"default": False, "tooltip": "CLAMP"}),
                "blend_mode": (["lerp", "slerp", "cosine", "cubic", "add", "a_only", "b_only"], {"default": "lerp", "tooltip": "BLEND"}),
                "enable_restarts": ("BOOLEAN", {"default": False, "tooltip": "RESTARTS"}),
                "scheduler": ("SCHEDULER", {"tooltip": "SCHEDULER"}),
            },
            "optional": {
                "ancestral_strength": ("FLOAT", {"default": 1.0, "tooltip": "STRENGTH"}),
                "noise_coherence": ("FLOAT", {"default": 0.0, "tooltip": "COHERENCE"}),
                "debug_mode": ("INT", {"default": 0, "tooltip": "DEBUG"}),
                "enable_profiling": ("BOOLEAN", {"default": False, "tooltip": "PROFILE"}),
                "restart_mode": (["balanced", "aggressive", "conservative", "detail_focus"], {"default": "balanced", "tooltip": "RESTART MODE"}),
                "yaml_settings_str": ("STRING", {"multiline": True, "default": "", "tooltip": "YAML"}),
            }
        }
    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "get_sampler"
    CATEGORY = "MD_Nodes/Samplers"
    
    def get_sampler(self, noise_behavior, step_random_mode, step_size, seed,
                    first_ancestral_step, last_ancestral_step, start_sigma_index,
                    end_sigma_index, enable_clamp_output, blend_mode, enable_restarts,
                    scheduler, **optional_kwargs):
        
        if noise_behavior != "Custom":
            preset_map = {
                "Default (Raw)": {"ancestral_strength": 1.0, "noise_coherence": 0.0},
                "Dynamic": {"ancestral_strength": 1.0, "noise_coherence": 0.25},
                "Smooth": {"ancestral_strength": 0.8, "noise_coherence": 0.5},
                "Textured Grain": {"ancestral_strength": 0.9, "noise_coherence": 0.9},
                "Soft": {"ancestral_strength": 0.2, "noise_coherence": 0.0},
            }
            preset = preset_map.get(noise_behavior, preset_map["Default (Raw)"])
            ancestral_strength = preset["ancestral_strength"]
            noise_coherence = preset["noise_coherence"]
        else:
            ancestral_strength = optional_kwargs.get("ancestral_strength", 1.0)
            noise_coherence = optional_kwargs.get("noise_coherence", 0.0)
            
        if CORE_LOADED:
            fbg_cfg = core.FBGConfig(
                sampler_mode=core.SamplerMode.PINGPONG,
                cfg_scale=6.0,
                fbg_guidance_multiplier=400.0,
                max_guidance_scale=5000.0,
                initial_guidance_scale=250.0,
                guidance_max_change=500.0,
                pi=0.35, t_0=0.7, t_1=0.4
            )
        else:
            fbg_cfg = {}
        
        kwargs = {
            'fbg_config': fbg_cfg,
            'seed': seed,
            'step_random_mode': step_random_mode,
            'step_size': step_size,
            'first_ancestral_step': first_ancestral_step,
            'last_ancestral_step': last_ancestral_step,
            'ancestral_noise_type': "gaussian",
            'start_sigma_index': start_sigma_index,
            'end_sigma_index': end_sigma_index,
            'enable_clamp_output': enable_clamp_output,
            'blend_function_name': blend_mode,
            'step_blend_function_name': blend_mode,
            'ancestral_strength': ancestral_strength,
            'noise_coherence': noise_coherence,
            'adaptive_noise_scaling': False,
            'noise_scale_factor': 1.0,
            'clamp_noise_norm': False,
            'progressive_blend_mode': False,
            'conditional_blend_mode': False,
            'log_posterior_ema_factor': 0.8,
            'enable_restarts': enable_restarts,
            'restart_mode': optional_kwargs.get("restart_mode", "balanced"),
            'restart_noise_scale': 0.5,
            'restart_s_noise': 1.0,
            'debug_mode': optional_kwargs.get("debug_mode", 0),
            'enable_profiling': optional_kwargs.get("enable_profiling", False),
            'tensor_memory_optimization': False,
            'early_exit_threshold': CONST_EARLY_EXIT_THRESHOLD,
            'sigma_range_preset': "Custom",
            'checkpoint_steps': [],
        }
        return (KSAMPLER(SamplerEntry, extra_options=kwargs),)

# =================================================================================
# == Node Registration
# =================================================================================

NODE_CLASS_MAPPINGS = {
    "PingPongSamplerNodeFBG": PingPongSamplerNodeFBG,
    "PingPongSamplerNodeBasic": PingPongSamplerNodeBasic,
    "PingPongSamplerNodeLite": PingPongSamplerNodeLite,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PingPongSamplerNodeFBG": "MD: PingPong FBG (Full Control)",
    "PingPongSamplerNodeBasic": "MD: PingPong Basic (Presets)",
    "PingPongSamplerNodeLite": "MD: PingPong Lite (Classic)",
}


# ==============================================================================
# == Unit Tests (smoke — runs without ComfyUI)
# ==============================================================================

if __name__ == "__main__":
    print("\n🧪 Smoke tests: MD_PingPongSamplerFBG_Legacy")
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

    _check("VERSION defined",    VERSION == "v1.6.0")
    _check("NODE_CLASS_MAPPINGS defined",
           isinstance(NODE_CLASS_MAPPINGS, dict) and len(NODE_CLASS_MAPPINGS) > 0)
    _check("  class PingPongSamplerNodeFBG in map", "PingPongSamplerNodeFBG" in NODE_CLASS_MAPPINGS)
    _check("  class PingPongSamplerNodeBasic in map", "PingPongSamplerNodeBasic" in NODE_CLASS_MAPPINGS)
    _check("  class PingPongSamplerNodeLite in map", "PingPongSamplerNodeLite" in NODE_CLASS_MAPPINGS)

    print(f"\n  {_pass} passed, {_fail} failed")
    if _fail == 0:
        print("  🎉 All good.")
