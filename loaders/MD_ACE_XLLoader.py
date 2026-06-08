# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# █▓▒░                                                                     ░▒▓█
# █▓▒░     MD_ACE_XLLoader — ACE-Step 1.5 XL Model Loader Suite           ░▒▓█
# █▓▒░                                                                     ░▒▓█
# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
# ╠═ © 2026 Alexander Allan (MDMAchine) | A&E Concepts
# ╠═ License: GNU General Public License v3.0 (GPL v3)
# ╠════════════════════════════════════════════════════════════════════════════
# ║ ░▒▓ NODES IN THIS FILE:
# ║
# ║   1. MD_ACE_XLLoader          — Load XL model variants with tunable
# ║                                  AuraFlow shift baked in. Outputs MODEL
# ║                                  + sigma plot IMAGE + analytics STRING.
# ║
# ║   2. MD_ACE_XLLatentProcessor — Post-sampling latent correction node.
# ║                                  DC offset correction (latent shift) +
# ║                                  optional mean normalization + scale.
# ║                                  Plugs between sampler and VAE Decode.
# ║
# ║ ░▒▓ WHY THIS EXISTS:
# ║    The XL models (acestep-v15-xl-turbo, xl-sft, xl-base) use a wider
# ║    hidden dimension (2560) vs the original 2.6B model (2048). The stock
# ║    ComfyUI ACE-Step node hardcodes the old architecture and throws a
# ║    RuntimeError on size mismatch when loading XL weights.
# ║
# ║    This loader wraps the XL model in a full ComfyUI ModelPatcher and
# ║    bakes in the correct flow-matching sampling config internally.
# ║    No external ModelSamplingAuraFlow node needed — this IS the patch.
# ║
# ║ ░▒▓ AURAFLOW SHIFT vs SD3/FLUX SHIFT:
# ║    AuraFlow uses multiplier=1.0. SD3/Flux use multiplier=1000.
# ║    Default AuraFlow shift = 1.73 (sqrt(3)). ACE-Step XL uses 3.0.
# ║    shift > 1.0 biases toward higher-noise early steps.
# ║    σ(t) = t / (t + shift*(1−t))   — nonlinear front-loading.
# ║    For undistilled SFT/base, lower shift (1.5–2.5) improves coherence.
# ║    For Turbo, 3.0 is the ByteDance-validated value.
# ║
# ║ ░▒▓ LATENT SHIFT (DC OFFSET CORRECTION):
# ║    ACE-Step silence_latent has non-zero mean. After denoising, the
# ║    output latent sits slightly off-center. Small additive correction
# ║    before VAE decode re-centers it to the VAE's optimal range.
# ║    Typical: −0.10 to −0.20 depending on variant.
# ║    0.0 = passthrough (disabled).
# ║
# ║ ░▒▓ ARCHITECTURE (from XL config.json):
# ║    decoder hidden_size:   2560  (was 2048 in 2.6B)
# ║    encoder hidden_size:   2048  (lyric/timbre — unchanged)
# ║    num_hidden_layers:      32   (DiT blocks)
# ║    patch_size:              2
# ║    in_channels:           192   (64 audio + 128 context)
# ║    layer_types: alternating sliding/full attention
# ║
# ║ ░▒▓ CONDITIONING ARCHITECTURE:
# ║    CLIP encoder → c_crossattn + conditioning_lyrics (per step)
# ║    extra_conds() packages as CONDRegular objects.
# ║    apply_model() passes to diffusion_model.forward() which calls
# ║    prepare_condition() + decoder() INTERNALLY every step.
# ║    NEVER call prepare_condition() from outside the model.
# ║
# ║ ░▒▓ CHANGELOG:
# ║    v1.3.0 (2026-04-08) — AuraFlow shift + Latent Processor + Plot + Analytics
# ║      NEW:  aura_shift parameter on loader (default 3.0, range 0.1–10.0).
# ║            No external ModelSamplingAuraFlow node needed.
# ║      NEW:  MD_ACE_XLLatentProcessor — post-sampling DC correction.
# ║            latent_shift + mean_normalize + scale. Stats STRING output.
# ║      NEW:  Sigma plot IMAGE output — matplotlib σ curve with velocity
# ║            panel, shift annotation, semantic midpoint marker.
# ║      NEW:  analytics STRING output — full load report for text display node.
# ║      NEW:  Manual AuraFlow fallback implements σ(t) math directly.
# ║    v1.2.0 (2026-04-08) — Bug fixes (see detailed notes below)
# ║      FIX:  model_output tensor check inverted — pure noise cause.
# ║      FIX:  lyric_embed guard checked cross_attn not conditioning_lyrics.
# ║      FIX:  context double-feed (c_crossattn in both context= and **kwargs).
# ║      FIX:  _ace15 import inside per-step hot path.
# ║      FIX:  torch.no_grad() absent from _apply_model.
# ║      QoL:  Session model cache (skip 90s reload on re-queue).
# ║      QoL:  Step-1 conditioning diagnostic.
# ║      QoL:  missing_keys threshold (warn only if >10).
# ║    v1.1.0 (2026-04-08) — Correct architecture (mirrors ACEStep15 source diff).
# ║    v1.0.0 (2026-04-07) — Initial implementation.
# ╚════════════════════════════════════════════════════════════════════════════
# ==============================================================================
# Part of ComfyUI_MD_Nodes by MDMAchine (A&E Concepts)
# Repository: https://github.com/MDMAchine/ComfyUI_MD_Nodes
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
# ==============================================================================


VERSION = "v3.0.0"  # UPS v1.5.8

import io
import os
import sys
import types
import logging
import importlib
import importlib.util
from pathlib import Path

import torch
import numpy as np
import folder_paths

logger = logging.getLogger("MD_ACE_XLLoader")

# ── Module-level ace_step15 import ───────────────────────────────────────────
_ace15 = None
try:
    import comfy.ldm.ace.ace_step15 as _ace15
except Exception as _e:
    logger.warning(
        f"[MD_ACE_XLLoader] comfy.ldm.ace.ace_step15 not importable: {_e}\n"
        f"    Silence latent falls back to stored patcher._silence_latent."
    )

# ── ComfyUI imports ───────────────────────────────────────────────────────────
try:
    import comfy.model_patcher
    import comfy.model_management
    import comfy.model_sampling
    import comfy.conds
    COMFY_AVAILABLE = True
except ImportError:
    COMFY_AVAILABLE = False
    logger.warning("[MD_ACE_XLLoader] ComfyUI comfy module not found.")

# ── Matplotlib ────────────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("[MD_ACE_XLLoader] matplotlib unavailable — plot output disabled.")


# =============================================================================
# == XL Variant Registry
# =============================================================================

XL_VARIANTS = {
    "xl-turbo": {
        "display":              "ACE-Step 1.5 XL Turbo",
        "model_file":           "modeling_acestep_v15_xl_turbo.py",
        "config_file":          "configuration_acestep_v15.py",
        "model_class":          "AceStepConditionGenerationModel",
        "config_class":         "AceStepConfig",
        "dtype":                torch.bfloat16,
        "notes":                "DMD2 distilled. 4–8 steps. hidden_size=2560.",
        "hint_steps":           "4–8  (distilled — do NOT exceed 12)",
        "hint_cfg":             "1.5–4.0  (distilled — high CFG collapses structure)",
        "hint_aura_shift":      "3.0  (ByteDance ACEStep15 validated default)",
        "hint_latent_shift":    "-0.10 to -0.15",
        "hint_scheduler":       "HT: ke=0.15, damping=2.0, ct=0.50, target_entropy=5.0",
        "hint_tpg":             "Reduce vs Turbo 2.6B baseline — wider XL manifold",
        "default_aura_shift":   3.0,
        "default_latent_shift": -0.12,
    },
    "xl-sft": {
        "display":              "ACE-Step 1.5 XL SFT",
        "model_file":           "modeling_acestep_v15_xl_base.py",
        "config_file":          "configuration_acestep_v15.py",
        "model_class":          "AceStepConditionGenerationModel",
        "config_class":         "AceStepConfig",
        "dtype":                torch.bfloat16,
        "notes":                "50-step undistilled. Broader manifold. Uses base modeling file.",
        "hint_steps":           "40–60  (undistilled — sweet spot 50)",
        "hint_cfg":             "1.5–3.5  (no CFG dropout training — keep low)",
        "hint_aura_shift":      "2.5  (start here — tune down for more coherence)",
        "hint_latent_shift":    "-0.15 to -0.20",
        "hint_scheduler":       "HT: ke=0.18, damping=2.0, ct=0.88, target_entropy=8.5",
        "hint_tpg":             "0.3–0.5 cascade (lower than Turbo — wider SFT manifold)",
        "default_aura_shift":   2.5,
        "default_latent_shift": -0.17,
    },
    "xl-base": {
        "display":              "ACE-Step 1.5 XL Base",
        "model_file":           "modeling_acestep_v15_xl_base.py",
        "config_file":          "configuration_acestep_v15.py",
        "model_class":          "AceStepConditionGenerationModel",
        "config_class":         "AceStepConfig",
        "dtype":                torch.bfloat16,
        "notes":                "Unaligned base weights. Research only.",
        "hint_steps":           "50–100  (unaligned — highly variable)",
        "hint_cfg":             "1.0–2.0  (unaligned — CFG behavior unpredictable)",
        "hint_aura_shift":      "1.73  (AuraFlow default sqrt(3) as baseline)",
        "hint_latent_shift":    "-0.10 to -0.20  (empirical — test both ends)",
        "hint_scheduler":       "HT: ke=0.15, damping=1.8, ct=0.90, target_entropy=9.0",
        "hint_tpg":             "0.2–0.3 or off — manifold too wide for strong guidance",
        "default_aura_shift":   1.73,
        "default_latent_shift": -0.15,
    },
}

# =============================================================================
# == Session model cache
# =============================================================================
_MODEL_CACHE: dict = {}


# =============================================================================
# == Sigma plot
# =============================================================================

def _render_sigma_plot(aura_shift: float, variant_key: str, steps: int = 60):
    """
    Render a styled matplotlib σ(t) curve for the loaded model.
    Returns a ComfyUI IMAGE tensor [1, H, W, 3] or None if matplotlib missing.

    Top panel: sigma trajectories (linear reference, sqrt(3) AuraFlow default,
               current shift) with semantic midpoint annotation.
    Bottom panel: dσ/dt velocity — shows where denoising effort concentrates.
    """
    if not MATPLOTLIB_AVAILABLE:
        return None

    t = np.linspace(1e-6, 1.0 - 1e-6, steps + 1)

    def sigma_curve(s):
        return t / (t + s * (1.0 - t) + 1e-9)

    s_linear  = t.copy()
    s_sqrt3   = sigma_curve(1.73)
    s_current = sigma_curve(aura_shift)
    dsdt      = np.gradient(s_current, t)

    # Semantic midpoint (σ=0.5)
    mid_idx = int(np.argmin(np.abs(s_current - 0.5)))

    # Per-variant accent colors
    accent = {"xl-turbo": "#00d4ff", "xl-sft": "#a78bfa", "xl-base": "#34d399"}.get(
        variant_key, "#00d4ff"
    )

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7.2, 5.0),
        gridspec_kw={"height_ratios": [2.4, 1.0], "hspace": 0.06}
    )
    bg = "#0b0d12"
    fig.patch.set_facecolor(bg)

    for ax in (ax1, ax2):
        ax.set_facecolor(bg)
        for spine in ax.spines.values():
            spine.set_edgecolor("#1e2130")
            spine.set_linewidth(0.7)
        ax.tick_params(colors="#505570", labelsize=7.5)
        ax.grid(True, color="#161825", linewidth=0.55, linestyle="--", alpha=0.9)

    # ── Top: sigma curves ─────────────────────────────────────────────────────
    ax1.plot(t, s_linear,  color="#242636", lw=1.0, ls=":",
             label="linear  shift=1.0",    zorder=1)
    ax1.plot(t, s_sqrt3,   color="#3a3d54", lw=1.1, ls="--",
             label=f"AuraFlow default  shift=√3≈1.73", zorder=2)
    ax1.plot(t, s_current, color=accent,    lw=2.3,
             label=f"current  shift={aura_shift:.2f}  "
                   f"[{XL_VARIANTS[variant_key]['display']}]",
             zorder=3)

    ax1.fill_between(t, s_linear, s_current, color=accent, alpha=0.07, zorder=0)

    # Semantic midpoint marker
    ax1.axvline(t[mid_idx], color=accent, lw=0.65, ls=":", alpha=0.55)
    ax1.annotate(
        f"  σ=0.5\n  t={t[mid_idx]:.2f}",
        xy=(t[mid_idx], 0.5),
        xytext=(min(t[mid_idx] + 0.08, 0.88), 0.38),
        color=accent, fontsize=6.5, fontfamily="monospace",
        arrowprops=dict(arrowstyle="-|>", color=accent, lw=0.65),
        bbox=dict(boxstyle="round,pad=0.25", fc=bg, ec=accent, lw=0.55, alpha=0.9),
    )

    ax1.set_xlim(0.0, 1.0)
    ax1.set_ylim(-0.03, 1.06)
    ax1.set_ylabel("σ  (noise level)", color="#7a7f9a", fontsize=8, labelpad=5)
    ax1.xaxis.set_ticklabels([])
    ax1.set_title(
        f"MD ACE-Step XL  ·  {XL_VARIANTS[variant_key]['display']}"
        f"  ·  AuraFlow Shift = {aura_shift:.2f}",
        color="#e8eaf0", fontsize=8.8, fontweight="bold",
        pad=9, fontfamily="monospace",
    )
    leg = ax1.legend(
        fontsize=6.8, loc="upper left",
        facecolor="#0f1118", edgecolor="#1e2130",
        labelcolor="#b0b4cc", framealpha=0.93,
        handlelength=1.8,
    )

    # ── Bottom: velocity dσ/dt ────────────────────────────────────────────────
    ax2.fill_between(t, 0, dsdt, color=accent, alpha=0.22, zorder=0)
    ax2.plot(t, dsdt, color=accent, lw=1.6, zorder=2)
    ax2.axhline(0, color="#1e2130", lw=0.6)

    peak_idx = int(np.argmax(dsdt))
    ax2.axvline(t[peak_idx], color="#f59e0b", lw=0.65, ls=":", alpha=0.75)
    ax2.annotate(
        f" peak\n t={t[peak_idx]:.2f}",
        xy=(t[peak_idx], dsdt[peak_idx]),
        xytext=(min(t[peak_idx] + 0.06, 0.88), dsdt[peak_idx] * 0.68),
        color="#f59e0b", fontsize=6, fontfamily="monospace",
        arrowprops=dict(arrowstyle="-|>", color="#f59e0b", lw=0.6),
    )

    ax2.set_xlim(0.0, 1.0)
    ax2.set_xlabel("t  (noise  →  clean)", color="#7a7f9a", fontsize=7.8, labelpad=5)
    ax2.set_ylabel("dσ/dt", color="#7a7f9a", fontsize=7.8, labelpad=5)

    fig.text(0.988, 0.012, "MD_Nodes  ©  MDMAchine",
             ha="right", va="bottom", fontsize=5.2,
             color="#1e2130", fontfamily="monospace")

    plt.tight_layout(pad=0.45)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)

    try:
        from PIL import Image as _PIL
        pil = _PIL.open(buf).convert("RGB")
        arr = np.array(pil).astype(np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0)   # [1, H, W, 3]
    except ImportError:
        # PIL not available — return blank
        return torch.zeros(1, 4, 4, 3)


# =============================================================================
# == Analytics string builder
# =============================================================================

def _build_analytics_string(
    variant_key: str, model_dir, config,
    aura_shift: float, size_bytes: int,
    missing_keys: int, unexpected_keys: int,
    cache_hit: bool,
) -> str:
    vinfo  = XL_VARIANTS[variant_key]
    size_gb = size_bytes / (1024 ** 3)
    # Computed sigma stats for the loaded shift
    t_half  = 0.5 / (0.5 + aura_shift * 0.5 + 1e-9)
    t_sem   = aura_shift / (1.0 + aura_shift)

    lines = [
        "╔══════════════════════════════════════════════════════════╗",
        "║          MD: ACE-Step XL Loader  ·  Load Report          ║",
        "╚══════════════════════════════════════════════════════════╝",
        "",
        f"  Variant      : {vinfo['display']}",
        f"  Path         : {model_dir}",
        f"  Cache        : {'♻  HIT — reused session weights' if cache_hit else '🔄  MISS — loaded from disk'}",
        "",
        "── ARCHITECTURE ─────────────────────────────────────────────",
        f"  hidden_size         : {config.hidden_size}",
        f"  encoder_hidden_size : {config.encoder_hidden_size}",
        f"  num_hidden_layers   : {config.num_hidden_layers}",
        f"  dtype               : {vinfo['dtype']}",
        f"  model_size          : {size_gb:.2f} GB",
        f"  missing_keys        : {missing_keys}{'  (expected — EMA/buffers)' if missing_keys <= 10 else '  ⚠️  HIGH'}",
        f"  unexpected_keys     : {unexpected_keys}",
        "",
        "── FLOW-MATCHING CONFIG ──────────────────────────────────────",
        f"  class               : CONST + ModelSamplingDiscreteFlow",
        f"  multiplier          : 1.0  (AuraFlow — NOT SD3/Flux 1000)",
        f"  aura_shift          : {aura_shift:.3f}",
        f"  σ at t=0.5          : {t_half:.4f}",
        f"  semantic midpoint t : {t_sem:.3f}",
        f"  schedule bias       : {'front-loaded (aggressive early)' if aura_shift > 1.5 else 'near-linear'}",
        "",
        "── INFERENCE HINTS ──────────────────────────────────────────",
        f"  Steps        : {vinfo['hint_steps']}",
        f"  CFG scale    : {vinfo['hint_cfg']}",
        f"  Aura shift   : {vinfo['hint_aura_shift']}",
        f"  Latent shift : {vinfo['hint_latent_shift']}  [MD_ACE_XLLatentProcessor]",
        f"  HT Scheduler : {vinfo['hint_scheduler']}",
        f"  TPG          : {vinfo['hint_tpg']}",
        "",
        "── PIPELINE ─────────────────────────────────────────────────",
        "  MD_ACE_XLLoader  →  MODEL  →  [TPG]  →  [NAG]  →  Guider",
        "  KSampler  →  MD_ACE_XLLatentProcessor  →  VAE Decode",
        "  ✅ No external ModelSamplingAuraFlow needed (baked in)",
        "  ✅ No MD_ACE_XLConditioner needed (architecture removed)",
        "",
        f"  Notes : {vinfo['notes']}",
        "",
        "─────────────────────────────────────────────────────────────",
        "  © 2026 Alexander Allan (MDMAchine) | A&E Concepts",
    ]
    return "\n".join(lines)


# =============================================================================
# == Model loading utilities
# =============================================================================

def _load_module_from_path(module_name: str, file_path: str):
    parent_dir = str(Path(file_path).parent)
    _injected  = parent_dir not in sys.path
    if _injected:
        sys.path.insert(0, parent_dir)
    try:
        spec   = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        if _injected and parent_dir in sys.path:
            sys.path.remove(parent_dir)


def load_xl_model(model_dir, variant_key: str, device: str = "cuda") -> tuple:
    """
    Load ACE-Step XL model from local directory.
    Returns (model, config, silence_latent, load_stats).
    load_stats = {"missing": int, "unexpected": int, "cache_hit": bool}
    Caches on (resolved_path, variant_key) — same call skips disk load.
    """
    model_dir = Path(model_dir).resolve()
    variant   = XL_VARIANTS[variant_key]
    cache_key = (str(model_dir), variant_key)

    if cache_key in _MODEL_CACHE:
        logger.info("[MD_ACE_XLLoader] ♻  Cache hit — skipping reload.")
        model, config, silence_latent, stats = _MODEL_CACHE[cache_key]
        return model, config, silence_latent, {**stats, "cache_hit": True}

    # ── File validation ───────────────────────────────────────────────────────
    for f in ["config.json", variant["config_file"], variant["model_file"], "silence_latent.pt"]:
        if not (model_dir / f).exists():
            raise FileNotFoundError(
                f"[MD_ACE_XLLoader] Required file missing: {model_dir / f}\n"
                f"Run ace_fetcher.py to download and merge."
            )

    merged = list(model_dir.glob("*_merged.safetensors"))
    shards  = list(model_dir.glob("model-*-of-*.safetensors"))
    single  = model_dir / "model.safetensors"
    index   = model_dir / "model.safetensors.index.json"

    if not merged and not shards and not single.exists():
        raise FileNotFoundError(
            f"[MD_ACE_XLLoader] No weights found in {model_dir}.\n"
            f"Expected: *_merged.safetensors OR model.safetensors OR shards+index."
        )
    if shards and not merged and not single.exists() and not index.exists():
        raise RuntimeError(
            f"[MD_ACE_XLLoader] Shards without index.json. Run ace_fetcher.py merge."
        )

    logger.info(f"[MD_ACE_XLLoader] Loading {variant['display']} from {model_dir}")

    # ── Dynamic class import ──────────────────────────────────────────────────
    cfg_mod  = _load_module_from_path(
        f"acestep_xl_config_{variant_key.replace('-','_')}",
        str(model_dir / variant["config_file"])
    )
    mdl_mod  = _load_module_from_path(
        f"acestep_xl_model_{variant_key.replace('-','_')}",
        str(model_dir / variant["model_file"])
    )
    ConfigClass = getattr(cfg_mod, variant["config_class"])
    ModelClass  = getattr(mdl_mod, variant["model_class"])

    config = ConfigClass.from_pretrained(str(model_dir))
    logger.info(
        f"[MD_ACE_XLLoader] Config loaded — "
        f"hidden={config.hidden_size} enc={config.encoder_hidden_size} "
        f"layers={config.num_hidden_layers}"
    )
    if config.hidden_size == 2048:
        logger.warning(
            "[MD_ACE_XLLoader] ⚠  hidden_size=2048 — looks like base 2.6B, not XL."
        )

    # ── Instantiate on CPU (bypass meta device) ───────────────────────────────
    # from_pretrained + accelerate triggers meta device init; ResidualFSQ.__init__
    # calls .item() which is illegal on meta tensors. Manual instantiate is safe.
    class _PatchedModelClass(ModelClass):
        @property
        def device(self):
            if hasattr(self, '_comfy_device'):
                return self._comfy_device
            try:
                return next(self.parameters()).device
            except StopIteration:
                return torch.device('cpu')

        @device.setter
        def device(self, value):
            self._comfy_device = torch.device(value) if isinstance(value, str) else value

    with torch.device("cpu"):
        model = _PatchedModelClass(config)
    model = model.to(dtype=variant["dtype"])

    # ── Load weights ──────────────────────────────────────────────────────────
    from safetensors.torch import load_file as _load_file

    _missing, _unexpected = 0, 0

    def _apply(sd, label):
        nonlocal _missing, _unexpected
        m, u = model.load_state_dict(sd, strict=False)
        _missing, _unexpected = len(m), len(u)
        if _missing > 10:
            logger.warning(f"[MD_ACE_XLLoader] {label}: {_missing} missing keys")
        elif _missing:
            logger.info(f"[MD_ACE_XLLoader] {label}: {_missing} missing (expected)")
        if _unexpected:
            logger.info(f"[MD_ACE_XLLoader] {label}: {_unexpected} unexpected (harmless)")

    if merged:
        w = merged[0]
        if len(merged) > 1:
            logger.warning(f"[MD_ACE_XLLoader] Multiple merged files — using {w.name}")
        logger.info(f"[MD_ACE_XLLoader] Loading {w.name}  (60–120s)...")
        sd = _load_file(str(w), device="cpu")
        _apply(sd, "merged"); del sd

    elif single.exists():
        logger.info("[MD_ACE_XLLoader] Loading model.safetensors...")
        sd = _load_file(str(single), device="cpu")
        _apply(sd, "single"); del sd

    elif index.exists():
        import json
        with open(str(index)) as f:
            idx = json.load(f)
        sd = {}
        for shard in sorted(set(idx["weight_map"].values())):
            logger.info(f"[MD_ACE_XLLoader]   Shard: {shard}")
            sd.update(_load_file(str(model_dir / shard), device="cpu"))
        _apply(sd, "shards"); del sd

    else:
        raise RuntimeError(f"[MD_ACE_XLLoader] No loadable weights in {model_dir}")

    logger.info(f"[MD_ACE_XLLoader] Moving to {device}...")
    model.eval()
    model = model.to(device)

    silence_latent = torch.load(
        str(model_dir / "silence_latent.pt"),
        map_location=device, weights_only=True,
    )
    if silence_latent.dtype != variant["dtype"]:
        silence_latent = silence_latent.to(variant["dtype"])

    logger.info("[MD_ACE_XLLoader] ✅ Model ready")
    stats  = {"missing": _missing, "unexpected": _unexpected, "cache_hit": False}
    result = (model, config, silence_latent, stats)
    _MODEL_CACHE[cache_key] = result
    return result


# =============================================================================
# == ModelPatcher wrapper
# =============================================================================

def _make_xl_patcher(inner_model, config, silence_latent,
                     variant_key: str, model_dir, aura_shift: float):
    """
    Wrap XL model in ComfyUI ModelPatcher.
    aura_shift baked into model_sampling — no external AuraFlow node required.
    Returns (patcher, size_bytes).
    """
    import comfy.model_patcher
    import comfy.model_management
    import comfy.model_sampling
    import comfy.conds

    load_device    = comfy.model_management.get_torch_device()
    offload_device = comfy.model_management.unet_offload_device()
    size_bytes     = sum(p.nelement() * p.element_size()
                         for p in inner_model.parameters())

    # ── Latent format (attached to inner_model — survives clone()) ────────────
    class ACEStepLatentFormat:
        latent_channels         = 64
        latent_dimensions       = 2
        scale_factor            = 1.0
        taesd_decoder_name      = None
        latent_rgb_factors      = None
        latent_rgb_factors_bias = None
        def process_in(self, x):  return x
        def process_out(self, x): return x

    inner_model.latent_format  = ACEStepLatentFormat()
    inner_model.silence_latent = silence_latent

    def _proc_out(self, x): return x
    def _proc_in(self, x):  return x
    inner_model.process_latent_out = types.MethodType(_proc_out, inner_model)
    inner_model.process_latent_in  = types.MethodType(_proc_in,  inner_model)

    # ── model_sampling — AuraFlow shifted timestep ───────────────────────────
    # Confirmed from native loader inspection: ACEStep15 XL uses _ManualAuraFlow
    # with shift=3.0. timestep([1,0.5,0.1]) must produce [1.0, 0.75, 0.25].
    # CONST+ModelSamplingDiscreteFlow gives linear [1.0, 0.5, 0.1] — WRONG.
    # _ManualAuraFlow is the correct implementation. Use it directly.
    class _ManualAuraFlow:
        """
        AuraFlow σ math with configurable shift.
        timestep(σ) = σ*shift / (1 + σ*(shift-1))
        Confirmed: shift=3.0 → timestep([1,0.5,0.1]) = [1.0, 0.75, 0.25]
        calculate_input: CONST — returns noise unchanged (flow matching)
        calculate_denoised: x0 = x - output * σ
        """
        def __init__(self, s):
            self._s = float(s)
            self.sigma_min = torch.tensor(0.0)
            self.sigma_max = torch.tensor(1.0)

        def timestep(self, sigma):
            s = sigma.clamp(1e-7, 1.0 - 1e-7)
            return s * self._s / (1.0 - s + s * self._s)

        def sigma(self, t):
            t = t.clamp(0.0, 1.0)
            return t / (t + self._s * (1.0 - t) + 1e-9)

        def percent_to_sigma(self, pct):
            t = torch.tensor(float(pct))
            return t / (t + self._s * (1.0 - t) + 1e-9)

        def noise_scaling(self, sigma, noise, latent, max_denoise=True):
            s = float(sigma.max()) if hasattr(sigma, 'max') else float(sigma)
            return noise * s + latent * (1.0 - s)

        def inverse_noise_scaling(self, sigma, latent):
            return latent

        def calculate_input(self, sigma, noise):
            # CONST behavior — flow matching passes noise unchanged
            return noise

        def calculate_denoised(self, sigma, model_output, model_input):
            s = sigma.view(sigma.shape[:1] + (1,) * (model_input.ndim - 1))
            return model_input - model_output * s

    _ms = _ManualAuraFlow(aura_shift)
    logger.info(f"[MD_ACE_XLLoader] model_sampling: ManualAuraFlow shift={aura_shift} ✅")
    inner_model.model_sampling = _ms

    # ── model_config stub (LoRA loader compatibility) ─────────────────────────
    # comfy.lora.model_lora_keys_unet() has a dedicated ACEStep15 branch that
    # builds key maps by iterating state dict keys directly — it does NOT need
    # model_config.unet_config. But it only takes that path if:
    #   isinstance(model, comfy.model_base.ACEStep15)
    # Our _PatchedModelClass is a HuggingFace PreTrainedModel subclass, so it
    # fails that check and falls through to a generic branch that DOES call
    # model.model_config.unet_config — crashing with AttributeError.
    #
    # Fix A: register inner_model as ACEStep15 via __class__ reassignment so
    #         isinstance() returns True and the correct key-map branch fires.
    # Fix B: attach a model_config stub as a belt-and-suspenders fallback in
    #         case any other ComfyUI path (future versions, plugins) calls it.
    #
    # Why __class__ reassignment instead of subclassing:
    #   We can't subclass ACEStep15 because it's a ComfyUI BaseModel subclass
    #   that expects a completely different __init__ signature. Reassigning
    #   __class__ makes isinstance() and type checks pass without touching
    #   __init__ or the actual object structure.
    try:
        import comfy.model_base as _cmb
        if hasattr(_cmb, 'ACEStep15'):
            inner_model.__class__ = type(
                '_XLPatchedACEStep15',
                (_cmb.ACEStep15, inner_model.__class__),
                {}
            )
            logger.info(
                "[MD_ACE_XLLoader] LoRA compat: registered as ACEStep15 subclass ✅"
            )
        else:
            logger.warning(
                "[MD_ACE_XLLoader] comfy.model_base.ACEStep15 not found — "
                "LoRA key mapping will use model_config stub fallback."
            )
    except Exception as _lora_e:
        logger.warning(
            f"[MD_ACE_XLLoader] ACEStep15 registration failed: {_lora_e} — "
            f"falling back to model_config stub."
        )

    # model_config stub: minimal interface to satisfy any path that calls
    # model.model_config.unet_config without crashing.
    # unet_config = {} → comfy.utils.unet_to_diffusers({}) returns empty dict
    # → model_lora_keys_unet produces no diffusers keys → falls through to
    # the ACEStep15 branch or key-by-key iteration. Safe no-op.
    class _ModelConfigStub:
        unet_config              = {}
        supported_inference_dtypes = [torch.bfloat16, torch.float16, torch.float32]
        def process_unet_state_dict(self, sd): return sd
        def process_unet_state_dict_for_saving(self, sd): return sd

    if not hasattr(inner_model, 'model_config'):
        inner_model.model_config = _ModelConfigStub()

    # ── diffusion_model alias ─────────────────────────────────────────────────
    # ComfyUI's model_base.py get_dtype() calls self.diffusion_model.dtype.
    # ACE-Step XL exposes the DiT as self.decoder, not self.diffusion_model.
    # Add alias so ComfyUI infrastructure (Super Guider, samplers, etc.) works.
    if not hasattr(inner_model, 'diffusion_model'):
        if hasattr(inner_model, 'decoder'):
            inner_model.diffusion_model = inner_model.decoder
        else:
            # Fallback: point at self so dtype calls don't crash
            inner_model.diffusion_model = inner_model

    # ── Stubs ─────────────────────────────────────────────────────────────────
    def _ecs(self, **kw):          return {}
    def _mr(self, shape, cond_shapes=None):
        e = 1
        for s in shape: e *= s
        return int(e * 16)

    # ── extra_conds ───────────────────────────────────────────────────────────
    _diag_done = [False]

    def _extra_conds(self, **kwargs):
        out    = {}
        device = kwargs["device"]
        noise  = kwargs["noise"]

        # Step-1 diagnostic — fires once per session
        if not _diag_done[0]:
            _diag_done[0] = True
            ca = kwargs.get("cross_attn", None)
            ly = kwargs.get("conditioning_lyrics", None)
            ra = kwargs.get("reference_audio_timbre_latents", None)
            logger.info(
                f"\n[MD_ACE_XLLoader] ── STEP-1 CONDITIONING DIAGNOSTIC ──\n"
                f"  noise.shape            : {noise.shape}\n"
                f"  cross_attn             : {ca.shape if ca is not None else 'None'}\n"
                f"  conditioning_lyrics    : {ly.shape if ly is not None else 'None'}\n"
                f"  reference_audio_timbre : len={len(ra) if ra is not None else 0}\n"
                f"  all kwargs keys        : {sorted(kwargs.keys())}\n"
                f"────────────────────────────────────────────────────"
            )

        # cross_attn / lyric embed
        cross_attn = kwargs.get("cross_attn", None)
        if cross_attn is not None:
            if torch.count_nonzero(cross_attn) == 0:
                out['replace_with_null_embeds'] = comfy.conds.CONDConstant(True)
            out['c_crossattn'] = comfy.conds.CONDRegular(cross_attn)

        # FIX v1.2: guard on conditioning_lyrics presence, not cross_attn
        conditioning_lyrics = kwargs.get("conditioning_lyrics", None)
        if conditioning_lyrics is not None:
            out['lyric_embed'] = comfy.conds.CONDRegular(conditioning_lyrics)

        # refer_audio
        refer_audio      = kwargs.get("reference_audio_timbre_latents", None)
        pass_audio_codes = False
        T                = noise.shape[2]

        if refer_audio is None or len(refer_audio) == 0:
            if _ace15 is not None:
                refer_audio = _ace15.get_silence_latent(T, device)
            else:
                sl  = self.silence_latent
                rep = (T // sl.shape[2]) + 1
                refer_audio = sl.repeat(1, 1, rep)[:, :, :T].to(device)
            pass_audio_codes = True
        else:
            refer_audio = refer_audio[-1][:, :, :T]
            out['is_covers'] = comfy.conds.CONDConstant(True)

        if pass_audio_codes:
            ac = kwargs.get("audio_codes", None)
            if ac is not None:
                out['audio_codes'] = comfy.conds.CONDRegular(
                    torch.tensor(ac, device=device))
                refer_audio = refer_audio[:, :, :750]
            else:
                out['is_covers'] = comfy.conds.CONDConstant(False)

        if refer_audio.shape[2] < T:
            if _ace15 is not None:
                pad = _ace15.get_silence_latent(T, device)
            else:
                sl  = self.silence_latent
                rep = (T // sl.shape[2]) + 1
                pad = sl.repeat(1, 1, rep)[:, :, :T].to(device)
            refer_audio = torch.cat(
                [refer_audio.to(pad.dtype).to(pad.device),
                 pad[:, :, refer_audio.shape[2]:]], dim=2
            )

        out['refer_audio'] = comfy.conds.CONDRegular(refer_audio)
        return out

    # ── apply_model ───────────────────────────────────────────────────────────
    def _apply_model(self, x, t, c_concat=None, c_crossattn=None,
                     control=None, transformer_options=None, **kwargs):
        if transformer_options is None: transformer_options = {}
        with torch.no_grad():
            sigma   = t
            xc      = self.model_sampling.calculate_input(sigma, x)
            context = c_crossattn

            xc     = xc.to(torch.bfloat16)
            device = xc.device
            if context is not None:
                context = context.to(device, torch.bfloat16)

            extra_conds = {}
            for k, v in kwargs.items():
                if k == 'c_crossattn':
                    continue   # strip — already passed as context=
                if hasattr(v, 'dtype'):
                    if v.dtype not in (torch.int32, torch.int64, torch.int, torch.long):
                        v = v.to(device, torch.bfloat16)
                    else:
                        v = v.to(device)
                elif isinstance(v, list):
                    v = [
                        e.to(device, torch.bfloat16)
                        if (hasattr(e, 'dtype') and
                            e.dtype not in (torch.int32, torch.int64, torch.int, torch.long))
                        else (e.to(device) if hasattr(e, 'dtype') else e)
                        for e in v
                    ]
                extra_conds[k] = v

            t_ts = self.model_sampling.timestep(t).float()

            model_output = self.diffusion_model(
                xc, t_ts,
                context=context,
                control=control,
                transformer_options=transformer_options,
                **extra_conds
            )

            # FIX v1.2: is_tensor FIRST — original had len() first, fired
            # True on batch=2 tensors, called pack_latents on valid tensor.
            if not torch.is_tensor(model_output) and len(model_output) > 1:
                from comfy import utils as _cu
                model_output, _ = _cu.pack_latents(model_output)

            if not torch.is_tensor(model_output):
                raise RuntimeError(
                    f"[MD_ACE_XLLoader] diffusion_model returned non-tensor: "
                    f"{type(model_output)}. Check XL modeling file forward()."
                )

            return self.model_sampling.calculate_denoised(sigma, model_output.float(), x)

    # apply_model wrapper — routes through WrapperExecutor so all ComfyUI
    # transformer_options patches (TPG, SequentialAttention, ModelStabilizer etc.)
    # actually fire. Without this they are silently skipped.
    def _apply_model_wrapped(self, x, t, c_concat=None, c_crossattn=None,
                             control=None, transformer_options=None, **kwargs):
        if transformer_options is None: transformer_options = {}
        try:
            import comfy.patcher_extension as _pe
            return _pe.WrapperExecutor.new_class_executor(
                _apply_model,
                self,
                _pe.get_all_wrappers(_pe.WrappersMP.APPLY_MODEL, transformer_options)
            ).execute(x, t, c_concat, c_crossattn, control, transformer_options, **kwargs)
        except Exception:
            # Fallback: call directly if WrapperExecutor unavailable
            return _apply_model(self, x, t, c_concat=c_concat, c_crossattn=c_crossattn,
                                control=control, transformer_options=transformer_options, **kwargs)

    inner_model.extra_conds_shapes = types.MethodType(_ecs,                   inner_model)
    inner_model.memory_required    = types.MethodType(_mr,                    inner_model)
    inner_model.extra_conds        = types.MethodType(_extra_conds,           inner_model)
    inner_model._apply_model       = types.MethodType(_apply_model,           inner_model)
    inner_model.apply_model        = types.MethodType(_apply_model_wrapped,   inner_model)

    # ── Build patcher ─────────────────────────────────────────────────────────
    patcher = comfy.model_patcher.ModelPatcher(
        inner_model,
        load_device=load_device,
        offload_device=offload_device,
        size=size_bytes,
    )
    inner_model.current_patcher = patcher
    patcher._silence_latent = silence_latent
    patcher._xl_config      = config
    patcher._xl_variant     = variant_key
    patcher._xl_model_dir   = model_dir
    patcher._aura_shift     = aura_shift

    return patcher, size_bytes


# =============================================================================
# == NODE 1: MD_ACE_XLLoader
# =============================================================================

class MD_ACE_XLLoader:
    """
    MD: ACE-Step 1.5 XL Loader  v1.3.0

    Loads XL model variants (hidden_size=2560) with tunable AuraFlow shift.
    No external ModelSamplingAuraFlow node required — shift is baked in.

    Outputs:
      MODEL       — wire into TPG → NAG → Guider chain as normal
      sigma_plot  — σ(t) curve IMAGE, wire into Preview Image node
      analytics   — load report STRING, wire into text display node
    """

    @classmethod
    def INPUT_TYPES(cls):
        xl_dirs = []
        try:
            for base in folder_paths.get_folder_paths("ace_step"):
                xl_root = os.path.join(base, "xl")
                if os.path.isdir(xl_root):
                    for d in sorted(os.listdir(xl_root)):
                        full = os.path.join(xl_root, d)
                        if os.path.isdir(full):
                            xl_dirs.append(full)
        except Exception:
            pass
        if not xl_dirs:
            xl_dirs = ["[place XL model folder path here]"]

        return {
            "required": {
                "model_path": ("STRING", {
                    "default":   xl_dirs[0] if xl_dirs else "",
                    "multiline": False,
                    "tooltip": (
                        "Full path to the XL model directory.\n"
                        "Must contain: config.json, modeling_*.py, "
                        "silence_latent.pt, and weights (.safetensors).\n"
                        "Session cache: same path+variant skips the 90s reload on re-queue."
                    ),
                }),
                "variant": (list(XL_VARIANTS.keys()), {
                    "default": "xl-turbo",
                    "tooltip": (
                        "xl-turbo : DMD2 distilled, 4–8 steps  (fastest)\n"
                        "xl-sft   : 50-step undistilled, broader manifold  (best quality)\n"
                        "xl-base  : Unaligned research weights"
                    ),
                }),
                "aura_shift": ("FLOAT", {
                    "default": 3.0,
                    "min":     0.1,
                    "max":     10.0,
                    "step":    0.05,
                    "tooltip": (
                        "AuraFlow sigma shift — controls denoising schedule curvature.\n"
                        "Formula: σ(t) = t / (t + shift*(1−t)), multiplier=1.0\n\n"
                        "shift = 1.0  → linear schedule\n"
                        "shift = 1.73 → AuraFlow default (√3)\n"
                        "shift = 3.0  → ACE-Step XL validated (ByteDance default)\n"
                        "shift > 1.0  → front-loads denoising (more aggressive early)\n\n"
                        "Per-variant recommendations:\n"
                        "  xl-turbo : 3.0   (validated)\n"
                        "  xl-sft   : 2.5   (start here, tune down for coherence)\n"
                        "  xl-base  : 1.73  (AuraFlow default as baseline)\n\n"
                        "No external ModelSamplingAuraFlow node needed —\n"
                        "this loader bakes the shift in directly."
                    ),
                }),
                "device": (["cuda", "cpu"], {
                    "default": "cuda",
                }),
            }
        }

    RETURN_TYPES  = ("MODEL", "IMAGE", "STRING")
    RETURN_NAMES  = ("model", "sigma_plot", "analytics")
    FUNCTION      = "load_xl"
    CATEGORY      = "MD_Nodes/Loaders"
    DESCRIPTION   = (
        "🎵 MD: ACE-Step 1.5 XL Loader — Loads XL variants (hidden_size=2560). "
        "Tunable AuraFlow shift baked in — no external node needed. "
        "Outputs MODEL + sigma plot IMAGE + analytics STRING. "
        "Session cache skips 90s reload on re-queue."
    )

    def load_xl(self, model_path: str, variant: str,
                aura_shift: float, device: str):

        if not model_path or model_path == "[place XL model folder path here]":
            raise ValueError("[MD_ACE_XLLoader] No model path provided.")
        if variant not in XL_VARIANTS:
            raise ValueError(f"[MD_ACE_XLLoader] Unknown variant '{variant}'.")

        vinfo = XL_VARIANTS[variant]
        logger.info(
            f"\n[MD_ACE_XLLoader] ══════════════════════════════════════════\n"
            f"  Variant     : {vinfo['display']}\n"
            f"  Path        : {model_path}\n"
            f"  Aura shift  : {aura_shift}  (AuraFlow, multiplier=1.0)\n"
            f"═══════════════════════════════════════════════════════"
        )

        model, config, silence_latent, load_stats = load_xl_model(
            model_dir=model_path, variant_key=variant, device=device,
        )
        patcher, size_bytes = _make_xl_patcher(
            inner_model=model, config=config, silence_latent=silence_latent,
            variant_key=variant, model_dir=model_path, aura_shift=aura_shift,
        )

        plot_tensor = _render_sigma_plot(aura_shift, variant)
        if plot_tensor is None:
            plot_tensor = torch.zeros(1, 4, 4, 3, dtype=torch.float32)

        analytics = _build_analytics_string(
            variant_key=variant, model_dir=model_path, config=config,
            aura_shift=aura_shift, size_bytes=size_bytes,
            missing_keys=load_stats.get("missing", 0),
            unexpected_keys=load_stats.get("unexpected", 0),
            cache_hit=load_stats.get("cache_hit", False),
        )

        logger.info(
            f"\n[MD_ACE_XLLoader] ── INFERENCE HINTS ──\n"
            f"  Steps        : {vinfo['hint_steps']}\n"
            f"  CFG          : {vinfo['hint_cfg']}\n"
            f"  Aura shift   : {vinfo['hint_aura_shift']}\n"
            f"  Latent shift : {vinfo['hint_latent_shift']}  (MD_ACE_XLLatentProcessor)\n"
            f"  HT Scheduler : {vinfo['hint_scheduler']}\n"
            f"  TPG          : {vinfo['hint_tpg']}\n"
            f"─────────────────────────────────────────────"
        )
        logger.info(
            f"[MD_ACE_XLLoader] ✅ Ready — "
            f"hidden={config.hidden_size} enc={config.encoder_hidden_size} "
            f"layers={config.num_hidden_layers} shift={aura_shift}"
        )

        return (patcher, plot_tensor, analytics)


# =============================================================================
# == NODE 2: MD_ACE_XLLatentProcessor
# =============================================================================

class MD_ACE_XLLatentProcessor:
    """
    MD: ACE-Step XL Latent Processor  v1.0.0

    Post-sampling latent correction. Place between KSampler and VAE Decode.

    Operations (in order):
      1. mean_normalize  — subtract per-channel mean (stronger DC correction)
      2. latent_shift    — additive DC offset. Re-centers ACE-Step's
                           non-zero silence_latent distribution.
      3. scale           — multiplicative output level control

    Per-variant defaults:
      xl-turbo : -0.12
      xl-sft   : -0.17
      xl-base  : -0.15
    """

    @classmethod
    def INPUT_TYPES(cls):
        hint = "\n".join(
            f"  {k:10s}: {v['default_latent_shift']}"
            for k, v in XL_VARIANTS.items()
        )
        return {
            "required": {
                "samples": ("LATENT", {
                    "tooltip": "Connect to KSampler LATENT output.",
                }),
                "latent_shift": ("FLOAT", {
                    "default": -0.15,
                    "min":     -2.0,
                    "max":      2.0,
                    "step":     0.01,
                    "tooltip": (
                        "Additive DC offset applied to all latent channels.\n"
                        "Re-centers ACE-Step's non-zero silence_latent.\n\n"
                        "0.0    = passthrough (disabled)\n"
                        "< 0.0  = shift toward center (typical — reduces DC bias)\n"
                        "> 0.0  = shift away from center\n\n"
                        "Per-variant recommendations:\n"
                        f"{hint}\n\n"
                        "Tune ±0.05 by ear from the variant default.\n"
                        "Too negative = thin/quiet. Too positive = distortion."
                    ),
                }),
                "scale": ("FLOAT", {
                    "default": 1.0,
                    "min":     0.1,
                    "max":     3.0,
                    "step":    0.01,
                    "tooltip": (
                        "Multiplicative output level. Applied after shift.\n"
                        "1.0 = no change (start here).\n"
                        ">1.0 = amplify (clipping risk at high values).\n"
                        "<1.0 = attenuate (safe headroom).\n"
                        "Use sparingly — adjust latent_shift first."
                    ),
                }),
                "mean_normalize": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Subtract per-channel mean before applying latent_shift.\n"
                        "More aggressive DC correction than shift alone.\n"
                        "Enable if latent_shift doesn't fully resolve DC artifacts.\n"
                        "Runs before latent_shift."
                    ),
                }),
            },
            "optional": {
                "passthrough": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "True = bypass all ops. Use for A/B comparison.",
                }),
            },
        }

    RETURN_TYPES  = ("LATENT", "STRING")
    RETURN_NAMES  = ("samples", "stats")
    FUNCTION      = "process"
    CATEGORY      = "MD_Nodes/Loaders"
    DESCRIPTION   = (
        "🎛️ MD: ACE-Step XL Latent Processor — Post-sampling DC correction. "
        "Place between KSampler and VAE Decode. "
        "latent_shift re-centers the distribution before decode. "
        "Outputs corrected LATENT + stats STRING."
    )

    def process(self, samples: dict, latent_shift: float, scale: float,
                mean_normalize: bool, passthrough: bool = False) -> tuple:

        result = samples.copy()
        x      = samples["samples"].clone()

        if passthrough:
            s = "MD_ACE_XLLatentProcessor: PASSTHROUGH — no operations applied."
            result["samples"] = x
            return (result, s)

        pre_mean = float(x.mean())
        pre_std  = float(x.std())
        pre_min  = float(x.min())
        pre_max  = float(x.max())

        # 1. Mean normalization (runs first)
        if mean_normalize:
            channel_means = x.mean(dim=-1, keepdim=True)
            x = x - channel_means

        # 2. DC offset
        if latent_shift != 0.0:
            x = x + latent_shift

        # 3. Scale
        if scale != 1.0:
            x = x * scale

        post_mean = float(x.mean())
        post_std  = float(x.std())
        post_min  = float(x.min())
        post_max  = float(x.max())

        clip_frac = float((x.abs() > 4.0).float().mean())

        lines = [
            "╔══════════════════════════════════════════════════╗",
            "║    MD: ACE-Step XL Latent Processor  ·  Stats    ║",
            "╚══════════════════════════════════════════════════╝",
            "",
            f"  shape          : {list(x.shape)}",
            f"  latent_shift   : {latent_shift:+.3f}",
            f"  scale          : {scale:.3f}",
            f"  mean_normalize : {mean_normalize}",
            "",
            "  ── BEFORE ─────────────────────────────────────",
            f"  mean : {pre_mean:+.5f}   std : {pre_std:.5f}",
            f"  min  : {pre_min:+.5f}   max : {pre_max:+.5f}",
            "",
            "  ── AFTER ──────────────────────────────────────",
            f"  mean : {post_mean:+.5f}   std : {post_std:.5f}",
            f"  min  : {post_min:+.5f}   max : {post_max:+.5f}",
            "",
            f"  Δ mean : {post_mean - pre_mean:+.5f}",
            f"  Δ std  : {post_std - pre_std:+.5f}",
        ]
        if clip_frac > 0.001:
            lines += [
                "",
                f"  ⚠️  CLIPPING: {clip_frac*100:.2f}% values > |4.0|",
                "     Reduce scale or reduce latent_shift magnitude.",
            ]

        stats_str = "\n".join(lines)
        logger.info(
            f"[MD_ACE_XLLatentProcessor] "
            f"shift={latent_shift:+.3f} scale={scale:.3f} norm={mean_normalize} | "
            f"mean {pre_mean:+.4f}→{post_mean:+.4f}  std {pre_std:.4f}→{post_std:.4f}"
        )

        result["samples"] = x
        return (result, stats_str)


# =============================================================================
# == Node Registration
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "MD_ACE_XLLoader":          MD_ACE_XLLoader,
    "MD_ACE_XLLatentProcessor": MD_ACE_XLLatentProcessor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MD_ACE_XLLoader":          "MD: ACE-Step XL Loader 🎵",
    "MD_ACE_XLLatentProcessor": "MD: ACE-Step XL Latent Processor 🎛️",
}


# ==============================================================================
# == Unit Tests (smoke — runs without ComfyUI)
# ==============================================================================

if __name__ == "__main__":
    print("\n🧪 Smoke tests: MD_ACE_XLLoader")
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

    _check("VERSION defined",    VERSION == "v3.0.0")
    _check("NODE_CLASS_MAPPINGS defined",
           isinstance(NODE_CLASS_MAPPINGS, dict) and len(NODE_CLASS_MAPPINGS) > 0)
    _check("  class MD_ACE_XLLoader in map", "MD_ACE_XLLoader" in NODE_CLASS_MAPPINGS)
    _check("  class MD_ACE_XLLatentProcessor in map", "MD_ACE_XLLatentProcessor" in NODE_CLASS_MAPPINGS)

    print(f"\n  {_pass} passed, {_fail} failed")
    if _fail == 0:
        print("  🎉 All good.")
