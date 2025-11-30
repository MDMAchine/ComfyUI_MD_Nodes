# MD_NODES

![Synth Approved](https://img.shields.io/badge/VIBES-CHAOTIC_NEUTRAL-ff00ff?style=flat-square&logo=md&logoColor=white)
![Node Count](https://img.shields.io/badge/NODES-35+_MODULES-cyan?style=flat-square)
![L33T MODE](https://img.shields.io/badge/L33T-MODE_ENABLED-red?style=flat-square&logo=hackaday)
![ComfyUI](https://img.shields.io/badge/ComfyUI-Custom_Nodes-orange?style=flat-square)

```text
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                                                                        ┃
┃  ███╗   ███╗██████╗       ███╗   ██╗ ██████╗ ██████╗ ███████╗███████╗  ┃
┃  ████╗ ████║██╔══██╗      ████╗  ██║██╔═══██╗██╔══██╗██╔════╝██╔════╝  ┃
┃  ██╔████╔██║██║  ██║      ██╔██╗ ██║██║   ██║██║  ██║█████╗  ███████╗  ┃
┃  ██║╚██╔╝██║██║  ██║      ██║╚██╗██║██║   ██║██║  ██║██╔══╝  ╚════██║  ┃
┃  ██║ ╚═╝ ██║██████╔╝      ██║ ╚████║╚██████╔╝██████╔╝███████╗███████║  ┃
┃  ╚═╝     ╚═╝╚═════╝       ╚═╝  ╚═══╝ ╚═════╝ ╚═════╝ ╚══════╝╚══════╝  ┃
┃                                                                        ┃
┃         🎛️ ComfyUI Custom Node Central - v1.5.0-RELEASE 🎛️             ┃
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

**🎯 Universal Toolkit:** While MD_Nodes began with a focus on audio generation (Ace-Step), most nodes are now **universal tools** that work with any diffusion model—Stable Diffusion, SDXL, Flux, video models, and beyond. Advanced sampling, scheduling, and guidance nodes are model-agnostic. Audio-specific nodes are clearly marked.

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
* **v1.5.0 Update:** Adds **Res 2 (Restart)** logic for iterative error correction with 5 restart modes (Balanced, Detail Focus, Composition Focus, Aggressive Correction, Conservative).
* **Features:** FBG dynamic Feedback Guidance, Noise Behavior presets, strength/coherence control, advanced blend modes, and NaN recovery. Works with image, audio, and video diffusion models.

📖 **[Read the Manual (v0.9.9-p4 FBG)](https://github.com/MDMAchine/ComfyUI_MD_Nodes/blob/main/manuals/Master%20Technical%20Manual_%20PingPong%20Sampler%20(Custom%20v0.9.9-p2%20FBG).md)**
📖 **[Read the Manual (Lite+ v0.8.21)](https://github.com/MDMAchine/ComfyUI_MD_Nodes/blob/main/manuals/Master%20Technical%20Manual_%20PingPong%20Sampler%20(Lite%2B%20V0.8.20).md)**

---

### 🧭 Universal Guidance & Control

#### `HYBRID_SIGMA_SCHEDULER`
**Universal precision noise scheduling for any diffusion model.** The ultimate sigma/noise scheduler with comprehensive visualization and analysis. Works with SD1.5, SDXL, Flux, Ace-Step, and any other diffusion architecture.
* **New in v1.4.1+:** **Bong Tangent** scheduler (Composition vs. Detail focus), Percentage-based splitting, Relative Linear steps, and step density analysis.
* **Features:** 14+ modes (Karras, Poly, AYS, Beta, Exponential, etc.), auto-detects model sigmas, supports slicing, visual plots, and comparison. Includes `SigmaSmooth` and `SigmaConcatenate` utilities.

📖 **[Read the Manual](https://github.com/MDMAchine/ComfyUI_MD_Nodes/blob/main/manuals/Master%20Technical%20Manual_%20Hybrid%20Sigma%20Scheduler.md)**

#### `NOISEDECAY_SCHEDULER` (Advanced)
**Universal custom decay curves for any diffusion workflow.** Generate sophisticated decay curves (polynomial, sigmoidal, gaussian, exponential, etc.) with start/end values, inversion, and smoothing. Perfect for image generation, video interpolation, audio synthesis, or any process requiring controlled noise modulation.

📖 **[Read the Manual](https://github.com/MDMAchine/ComfyUI_MD_Nodes/blob/main/manuals/Master%20Technical%20Manual_%20Advanced%20Noise%20Decay%20Scheduler.md)**

#### `APG_GUIDER_FORKED`
**Universal adaptive gradient guidance for precise control.** Enhanced Adaptive Projected Gradient guidance that works with any diffusion model. Schedule APG scale, CFG, and momentum via YAML based on sigma. Surgical latent steering with minimal artifacts. Excellent for both image and audio generation.

📖 **[Read the Manual](https://github.com/MDMAchine/ComfyUI_MD_Nodes/blob/main/manuals/Master%20Technical%20Manual_%20APG%20Guider%20(Forked).md)**

#### `MD_APPLY_TPG` (Token Perturbation Guidance) 🆕
**Universal guidance enhancement through controlled chaos.** Applies Token Perturbation Guidance (TPG) by patching the model's Self-Attention mechanism. This technique introduces controlled noise during the guidance phase to break repetitive patterns and improve prompt adherence without the burn of high CFG. Works with any text-conditioned diffusion model (SD, SDXL, Flux, Ace-Step).

---

### 🎵 Audio Processing

#### `MASTERING_CHAIN_NODE` (Full & Modular)
**Make your audio thicc.** Pro-grade mastering suite with both all-in-one and modular components.

**Modular Nodes Included:**
* `MD_Mastering_Gain`: Simple gain staging with visual feedback
* `MD_Mastering_EQ`: Parametric, Shelf, and Cutoff filters
* `MD_Mastering_Compressor`: Single or Multiband (True stereo-linked)
* `MD_Mastering_Limiter`: Lookahead brickwall limiting

**Features:** True stereo multiband compression, surgical EQ, transparent limiting. Optimized for LUFS-normalized input. Slaps waveforms with attitude.

📖 **[Read the Manual](https://github.com/MDMAchine/ComfyUI_MD_Nodes/blob/main/manuals/Master%20Technical%20Manual_%20Custom%20Audio%20Mastering%20Chain.md)**

#### `AUDIO_AUTO_MASTER_PRO`
**Iterative mastering magic.** Automatically analyzes and processes audio using filters, iterative EQ, de-essing, 3-band compression, stereo widening, and limiting to hit target LUFS and spectral profile. Set it and forget it.

#### `AUDIO_AUTO_EQ`
**One-click audio shaping.** Analyzes audio spectrum (librosa) and applies multi-band EQ (pedalboard) based on 18+ target profiles (Vocal Clarity, De-esser, Podcast, Music Master, ASMR, Radio Voice, etc.). Intelligent spectral correction.

#### `ADVANCED_AUDIO_PREVIEW_AND_SAVE` (AAPS)
**Hear it before you overthink it.** Comprehensive audio export with professional features: MP3/FLAC/OPUS formats, advanced normalization (Peak/RMS/LUFS), effects (fades, limiting), waveform/spectrogram preview, and robust metadata embedding.

📖 **[Read the Manual](https://github.com/MDMAchine/ComfyUI_MD_Nodes/blob/main/manuals/Master%20Technical%20Manual_%20Advanced%20Audio%20Preview%20&%20Save%20(AAPS).md)**

---

### ⚡ ACE T5 Audio Suite

#### `ACE_T5_CONDITIONING`
**The main event.** Intelligently blends T5 & CLIP embeddings for Ace-Step. Creates high-quality audio conditioning with separate controls for genre, vocal style, and lyrical content. Supports dual CLIP & LoRAs (when used with native loader). Use this for your positive prompt!

#### `ACE_T5_MODEL_LOADER`
**The key to the kingdom.** Custom loader for ACE T5 models (`umt5_with_tokenizer.safetensors`) enabling specialized tokenizers for the highest quality conditioning. *Note: Does NOT support LoRAs.*

🔗 **[Get the Model](https://huggingface.co/ACE-Step/ACE-Step-v1-3.5B/blob/main/umt5-base/model.safetensors)**

#### `ACE_T5_ANALYZER`
**The debugger.** Inspect conditioning tensors (shape, stats, lyric tokens) to troubleshoot distortion or verify encoding. Essential for sanity checks when audio output sounds weird.

---

### 🛠️ Workflow & Utilities

#### `SCENE_GENIUS_AUTOCREATOR`
**Prompter's divine sidekick.** Uses local LLMs (Ollama/LM Studio) to auto-generate genres, lyrics, duration, and tuned YAML configs for APG/Sampler/NoiseDecay based on a core concept. Automate your creative process and let the AI handle the boring parts.

📖 **[Read the Manual](https://github.com/MDMAchine/ComfyUI_MD_Nodes/blob/main/manuals/Master%20Technical%20Manual_%20Scene%20Genius%20Autocreator.md)**

#### `ADVANCED_TEXT_INPUT` & `TEXT_FILE_LOADER`
**External text powerhouse.** Load large prompts or configs directly from `.txt` files. Supports wildcards and dynamic token replacement. No more tiny text boxes.

#### `SMART_FILENAME_BUILDER` Suite
**Dynamic filename toolkit.** Organize your outputs like a pro.
* `SmartFilenameBuilder`: Complex names with presets (Vocal/Instrumental), metadata toggles, and date codes
* `FilenameTokenReplacer`: Simple `{token}` substitution for custom patterns
* `FilenameCounterNode`: Persistent `#0001` counters stored on disk to prevent overwrites

#### `MD_ENHANCED_SEED_SAVER`
**Professional seed management.** Save, load, search, and manage seeds with subdirectories, favorites, statistics, backup, and export/import. Features dynamic action & static pass-through modes for workflow stability. Never lose a good seed again.

📖 **[Read the Manual](https://github.com/MDMAchine/ComfyUI_MD_Nodes/blob/main/manuals/Master%20Technical%20Manual_%20MD%20Seed%20Saver.md)**

#### `LLM_VRAM_MANAGER`
**VRAM peace talks.** Unloads models from Ollama (API) or force-stops LM Studio (process kill) to prevent VRAM conflicts between ComfyUI and local LLMs directly from your workflow. No more manual switching.

#### `GPU_TEMPERATURE_PROTECTION` (Enhanced)
**Keep your cool.** Monitors GPU temp/VRAM via `nvidia-smi` and pauses the queue if thresholds are exceeded. Multi-GPU support, CSV logging, adaptive cooling profiles. Prevents meltdowns and extends hardware life.

---

### 🗂️ Organization & Workflow Management

#### `UNIVERSAL_ROUTING_HUB` (Advanced)
**The traffic controller.** Advanced signal routing with multiple inputs/outputs and intelligent type detection.
* **v2.0.0 Update:** Visual status map output showing active connections and data types. Performance metrics and better debugging.
* **Features:** Daisy-chaining hubs, custom labeling, conditional routing, and signal transformation. Route any data type anywhere. Your workflow's nervous system.

#### `AUTO_LAYOUT_OPTIMIZER`
**Untangle your spaghetti.** Automatically analyzes and reorganizes ComfyUI workflows using force-directed graphs and hierarchical layouts. Detects node clusters, optimizes spacing, and prevents overlaps. Makes your workflow readable again.

#### `WORKFLOW_SECTION_ORGANIZER`
**Divide and conquer.** Universal passthrough node that acts as a visual chapter marker. Group related nodes into labeled sections with visual boundaries. Auto-colors based on section type (Audio, Image, Utility, Control). Perfect for complex multi-stage workflows.

#### `SMART_COLOR_PALETTE_MANAGER`
**Color-code your chaos.** Generate and manage professional color palettes for your workflow. Create, save, and load color schemes. Features auto-detection based on node keywords and visual preview output. Quickly identify node types by color. Visual organization at its finest.

#### `ENHANCED_ANNOTATION_NODE`
**Notes with style.** Add styled annotations to your workflow with customizable styles (Sticky Note, Banner, Callout), colors, fonts, borders, and opacity. Markdown support for documentation. Purely visual—zero execution cost. Make your workflow self-documenting.

---

### 🎨 Visual Tools

#### `ACE_LATENT_VISUALIZER`
**Decode the noise gospel.** 9 visualization modes to peer into latent tensors:
* Waveform, Spectrum (FFT), Heatmap, Histogram
* Phase Analysis, Difference Map, Channel Comparison
* Statistical Overlays, Peak Detection

Includes batch/channel selection, stats overlay, and export options. See what the AI sees.

📖 **[Read the Manual](https://github.com/MDMAchine/ComfyUI_MD_Nodes/blob/main/manuals/Master%20Technical%20Manual_%20ACE%20Latent%20Visualizer.md)**

#### `ADVANCED_MEDIA_SAVE` (AMS)
**Your one-stop media vault.** Saves images (PNG/JPEG/WEBP), batches, and videos (GIF/MP4/WEBM). Features metadata privacy filter, quality/framerate control, and robust timestamp-based naming to prevent overwrites. Professional output management.

📖 **[Read the Manual](https://github.com/MDMAchine/ComfyUI_MD_Nodes/blob/main/manuals/Master%20Technical%20Manual_%20Advanced%20Media%20Save%20(AMS).md)**

---

## 💾 Included Tools (`/tools`)

We now include a suite of professional automation tools located in the `tools/` directory of this repository:

1.  **Project Updater** (`ComfyUI_Project_Updater.py`):
    * Self-maintaining script to auto-generate `__init__.py`, update requirements, and manage node metadata.
    * Includes "Re-launch Guard" to ensure it runs in your correct Python venv.
2.  **AIO Manager**:
    * Launchers and management utilities for ComfyUI.
3.  **Ace-Step Training Suite**:
    * Full LoRA training automation tools (`lora_trainer_automation_v2.1_FINAL.py`).
    * Audio normalization and tagging utilities.
    * Setup checklists and best practice guides for audio model training.

---

## 🧰 Installation

```bash
cd path/to/ComfyUI/custom_nodes
git clone https://github.com/MDMAchine/ComfyUI_MD_Nodes.git
cd ComfyUI_MD_Nodes
pip install -r requirements.txt
```

Or search for **MD_Nodes** in the ComfyUI Manager!

**Don't forget to restart ComfyUI.** Even gods need to reboot.

---

## 📛 Warnings from the Digital Frontier

- **Adult humor & digital irony** throughout. If offended, please Ctrl+Alt+Del your expectations.
- **Side effects may include:**
  - Late-night node tweaking
  - Caffeine-fueled revelations
  - Obsessive tensor visualizations
  - Latent hallucinations (the good kind)
- We're not liable for friendships lost over signal chains.

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
