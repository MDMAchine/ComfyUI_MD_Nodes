# MD_NODES

![Synth Approved](https://img.shields.io/badge/VIBES-CHAOTIC_NEUTRAL-ff00ff?style=flat-square&logo=md&logoColor=white)
![Node Count](https://img.shields.io/badge/NODES-45+_MODULES-cyan?style=flat-square)
![L33T MODE](https://img.shields.io/badge/L33T-MODE_ENABLED-red?style=flat-square&logo=hackaday)
![ComfyUI](https://img.shields.io/badge/ComfyUI-Custom_Nodes-orange?style=flat-square)

```text
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                                                                        ┃
┃  ███╗   ███╗██████╗        ███╗   ██╗ ██████╗ ██████╗ ███████╗███████╗  ┃
┃  ████╗ ████║██╔══██╗       ████╗  ██║██╔═══██╗██╔══██╗██╔════╝██╔════╝  ┃
┃  ██╔████╔██║██║  ██║       ██╔██╗ ██║██║   ██║██║  ██║█████╗  ███████╗  ┃
┃  ██║╚██╔╝██║██║  ██║       ██║╚██╗██║██║   ██║██║  ██║██╔══╝  ╚════██║  ┃
┃  ██║ ╚═╝ ██║██████╔╝       ██║ ╚████║╚██████╔╝██████╔╝███████╗███████║  ┃
┃  ╚═╝     ╚═╝╚═════╝        ╚═╝  ╚═══╝ ╚═════╝ ╚═════╝ ╚══════╝╚══════╝  ┃
┃                                                                        ┃
┃         🎛️ ComfyUI Custom Node Central - v1.5.1-RELEASE 🎛️              ┃
┃      "Latent Space Debauchery & Digital Sorcery Since 56k Days"        ┃
┃                                                                        ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
```

## 📟 Welcome to MD_NODES

**Strap in, traveler.** This isn't just another GitHub repo—it's a wormhole into raw, modular creativity with ComfyUI at its freaky finest.

Brought to you by a confederation of node-smiths, mod wizards, and digital dropouts, this is where unhinged meets unfiltered. Whether you're forging dreamscapes from the void or just want to vibe with your VAE, this is your personal BBS of brilliance.

Packaged with ASCII-flavored love, open-source chaos, and a healthy disregard for conventional sanity.

> *"Why do it the easy way when you can do it the aesthetic way?"*
> — probably MDMAchine

**🎯 Universal Toolkit:** While MD_Nodes began with a focus on audio generation (Ace-Step), most nodes are now **universal tools** that work with any diffusion model—Stable Diffusion, SDXL, Flux, video models, and beyond. Advanced sampling, scheduling, and guidance nodes are model-agnostic.

---

## 🔥 Node Collection

### 🎨 Wildcards & Prompt Engineering

#### `MD_WILDCARD_PROMPT_BUILDER` ⭐
**The ultimate multi-mode prompt generation node.**
Far more than a simple text box, this is a seed-controlled prompt engine. It supports massive wildcard expansion, dynamic list choices, and robust string manipulation to keep your generations fresh yet controllable.

---

### 🕹️ Advanced Sampling & Schedulers

#### `PINGPONG_SAMPLER` (Lite+ & FBG Variants)
**Advanced ancestral sampler with sophisticated noise control.** While originally optimized for Ace-Step audio, this is a powerful universal sampler suitable for any diffusion model (Stable Diffusion, SDXL, Flux, etc.).
* **v1.5.0 Update:** Adds **Res 2 (Restart)** logic for iterative error correction with 5 restart modes.
* **Features:** FBG dynamic Feedback Guidance, Noise Behavior presets, strength/coherence control, advanced blend modes, and NaN recovery.

📖 **[Read the Manual](https://github.com/MDMAchine/ComfyUI_MD_Nodes/blob/main/manuals/Master%20Technical%20Manual_%20PingPong%20Sampler%20(Custom%20v0.9.9-p2%20FBG).md)**

#### `HYBRID_SIGMA_SCHEDULER`
**Universal precision noise scheduling.** The ultimate sigma/noise scheduler.
* **Features:** 14+ modes (Karras, Poly, AYS, Beta, Exponential, Bong Tangent), auto-detects model sigmas, supports slicing, visual plots, and comparison.
* **Utilities Included:**
    * `SigmaSmooth`: Smoothes step transitions.
    * `SigmaConcatenate`: Joins different schedules together.

📖 **[Read the Manual](https://github.com/MDMAchine/ComfyUI_MD_Nodes/blob/main/manuals/Master%20Technical%20Manual_%20Hybrid%20Sigma%20Scheduler.md)**

#### `NOISEDECAY_SCHEDULER` (Advanced)
**Universal custom decay curves.** Generate sophisticated decay curves (polynomial, sigmoidal, gaussian, exponential) with start/end values, inversion, and smoothing.

---

### 🧭 Universal Guidance & Control

#### `APG_GUIDER_FORKED`
**Universal adaptive gradient guidance.** Enhanced Adaptive Projected Gradient guidance that works with any diffusion model. Schedule APG scale, CFG, and momentum via YAML based on sigma. Surgical latent steering with minimal artifacts.

#### `MD_APPLY_TPG` (Token Perturbation Guidance) 🆕
**Controlled chaos.** Applies Token Perturbation Guidance by patching the model's Self-Attention mechanism. Breaks repetitive patterns and improves prompt adherence without high CFG burn.

---

### 🎚️ Noise & Latent Generation

#### `MD_CUSTOM_NOISE_GENERATOR` 🆕
**Beyond Gaussian.** Generates advanced noise patterns for use with custom samplers.
* **Algorithms:** Perlin, Voronoi (Worley), Collatz Orbit (Fractal), Wavelet, Pyramid, Pink Noise, and more.

#### `MD_MULTI_NOISE_BLENDER` (5-Layer) 🆕
**Sonic alchemy for images.** Mixes up to 5 different noise sources using advanced blend modes (Add, Multiply, Screen, Overlay, Difference) to create complex initial noise states. Inspired by the Sonar suite.

#### `MD_EMPTY_LATENT_RATIO_SELECTOR` 🆕
**Perfect aspect ratios.** Generates empty latents based on strict ratios (16:9, 21:9, etc.) and target megapixels. Auto-calculates exact dimensions divisible by 8 for SDXL/Flux compatibility. Includes orientation toggle.

---

### ⚡ Optimization

#### `MD_OPTIMIZER_SELECTOR` 🆕
**Crash-proof training config.** A safety factory for PyTorch Optimizers.
* **Features:** Selects **Prodigy**, **Lion**, or **AdamW**.
* **Safety:** Automatically detects if `prodigyopt` or `bitsandbytes` are missing and falls back to standard AdamW instead of crashing your workflow.
* **VRAM:** Auto-enables 8-bit optimizers if available.

---

### 🎵 Audio Processing

#### `MASTERING_CHAIN_NODE` (Full & Modular)
**Make your audio thicc.** Pro-grade mastering suite.
* **Modular Nodes:** `MD_Mastering_Gain`, `MD_Mastering_EQ`, `MD_Mastering_Compressor` (Multiband), `MD_Mastering_Limiter`.
* **Features:** True stereo multiband compression, surgical EQ, transparent limiting.

#### `AUDIO_AUTO_MASTER_PRO`
**Iterative mastering magic.** Automatically analyzes and processes audio to hit target LUFS and spectral profile using 3-band compression and stereo widening.

#### `AUDIO_AUTO_EQ`
**One-click audio shaping.** Analyzes audio spectrum and applies multi-band EQ based on 18+ target profiles (Vocal Clarity, Podcast, ASMR, etc.).

#### `ADVANCED_AUDIO_PREVIEW_AND_SAVE` (AAPS)
**Hear it before you overthink it.** Professional export: MP3/FLAC/OPUS, normalization (Peak/RMS/LUFS), fades, waveform/spectrogram preview, and metadata.

---

### ⚡ ACE T5 Audio Suite

#### `ACE_T5_CONDITIONING`
**The main event.** Intelligently blends T5 & CLIP embeddings for Ace-Step. Includes `ACE_T5_CONDITIONING_SCHEDULED` and `ACE_T5_CONDITIONING_ANALYZER`.

#### `ACE_T5_MODEL_LOADER`
**The key to the kingdom.** Custom loader for ACE T5 models enabling specialized tokenizers.

---

### 🛠️ Workflow & Utilities

#### `SCENE_GENIUS_AUTOCREATOR`
**Prompter's divine sidekick.** Uses local LLMs (Ollama/LM Studio) to auto-generate genres, lyrics, and tuned YAML configs for samplers based on a core concept.

#### `MD_WORKFLOW_CONTEXT_BUS` 🆕
**The cable management king.** A universal "Bus" node that bundles 18+ data types (Model, Clip, VAE, Cond, Latent, Image, Audio, Sampler, Noise, etc.) into a single line. Plug it in once, route it everywhere.

#### `UNIVERSAL_ROUTING_HUB`
**The traffic controller.** Advanced signal routing with multiple inputs/outputs, intelligent type detection, and visual status mapping.

#### `SMART_FILENAME_BUILDER` Suite
**Dynamic filename toolkit.** Includes `SmartFilenameBuilder`, `FilenameTokenReplacer`, and `FilenameCounterNode` for persistent numbering.

#### `MD_ENHANCED_SEED_SAVER`
**Professional seed management.** Save, load, search, favorite, and backup seeds.

#### `LLM_VRAM_MANAGER`
**VRAM peace talks.** Unloads models from Ollama/LM Studio to prevent VRAM conflicts.

#### `GPU_TEMPERATURE_PROTECTION` (Enhanced)
**Keep your cool.** Monitors GPU temp/VRAM via `nvidia-smi` and pauses the queue if thresholds are exceeded.

---

### 🏗️ Maintenance & Repo Management

#### `MD_REPO_FORTRESS` (v2.2.1) 🆕
**The Safety Net.** A node-based maintenance tool.
* **Updates:** Checks git status and pulls updates.
* **Backups:** Creates zip snapshots with manifest metadata.
* **Restore:** "Time Travel" to previous versions.
* **Safety:** Disk space checks, race condition protection, and "Dry Run" simulation mode.

#### `MD_GLOBAL_UPDATE_FIXER` (v1.7.1) 🆕
**The Overseer.** Manages your *entire* `custom_nodes` folder.
* **Scan:** Reports status of all nodes.
* **Repair:** "Inject Git" mode converts manual Zip installs into updateable Git repos.
* **Conflict:** Analyzes `requirements.txt` across all nodes to find version conflicts.

---

### 🗂️ Organization

#### `AUTO_LAYOUT_OPTIMIZER`
**Untangle your spaghetti.** Automatically reorganizes workflows using force-directed graphs.

#### `WORKFLOW_SECTION_ORGANIZER`
**Divide and conquer.** Universal passthrough node that acts as a visual chapter marker with auto-coloring.

#### `SMART_COLOR_PALETTE_MANAGER`
**Color-code your chaos.** Generate and manage professional color palettes for your workflow.

#### `ENHANCED_ANNOTATION_NODE`
**Notes with style.** Add markdown-supported sticky notes and banners.

---

### 🎨 Visual Tools

#### `ACE_LATENT_VISUALIZER`
**Decode the noise gospel.** 9 visualization modes (Waveform, Spectrum, Heatmap) for latent tensors.

#### `ADVANCED_MEDIA_SAVE` (AMS)
**Your one-stop media vault.** Saves images and videos (GIF/MP4/WEBM) with metadata privacy filters.

---

## 🧰 Installation

```bash
cd path/to/ComfyUI/custom_nodes
git clone [https://github.com/MDMAchine/ComfyUI_MD_Nodes.git](https://github.com/MDMAchine/ComfyUI_MD_Nodes.git)
cd ComfyUI_MD_Nodes
pip install -r requirements.txt
```

Or search for **MD_Nodes** in the ComfyUI Manager!

**Don't forget to restart ComfyUI.** Even gods need to reboot.

---

## 💾 Credits

These legends walked so you could sample:

| Handle            | Role                            |
|-------------------|---------------------------------|
| **MDMAchine** | Core concept, main chaos wizard |
| **blepping** | Original PingPong/APG mind      |
| **c0ffymachyne** | Signal alchemist / audio IO     |
| **devstral** | Local l33t, fix-ologist         |
| **Gemini (Google)** | Kernel rewriter, patch priest   |
| **Claude (Anthropic)** | Refactor architect, stability   |
| **qwen3** | Completionist with RGB soul     |
| **w-e-w** | OG GPU Temp Protect Concept     |
| **meap158** | Comfy GPU Temp Protect Adapt.   |

---

## 📜 License

This project is licensed under the **Apache License 2.0**.
See [`LICENSE.md`](LICENSE.md) for details.

You are free to use, modify, and distribute this software in accordance with the license terms.

---

```text
                        .-----.
                       /       \
                      |  RAVE   |
                      |_________|
                     /  _     _  \
                    |  | |   | |  |
                    |  |_|___|_|  |
                    |  /       \  |
                    |_|_________|_|
                         \_______/
                          \_____/
                           \___/
                            `-'

            LOGGING OFF FROM MD_NODES 🛰️
      STAY SYNTHETIC, STAY STRANGE, STAY COMFYUI 💽
```

---

**⭐ Star this repo if it tickles your neurons**
**🐛 Report issues if reality breaks**
**🔀 PRs welcome from fellow digital shamans**