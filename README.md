# MD_NODES

![Synth Approved](https://img.shields.io/badge/VIBES-CHAOTIC_NEUTRAL-ff00ff?style=flat-square&logo=md&logoColor=white)
![Node Count](https://img.shields.io/badge/NODES-28_MODULES-cyan?style=flat-square)
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
┃         🎛️ ComfyUI Custom Node Central - v0.69.420-BETA 🎛️            ┃
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

---

## 🔥 Node Collection

### Scheduling & Guidance

#### `HYBRID_SIGMA_SCHEDULER`
**Your vibe, your noise.** 10+ modes (Karras, Poly, AYS, etc.) for precision noise scheduling. Auto-detects model sigmas, supports slicing, visualization, and comparison. Your sigma sommelier.

📖 **[Read the Manual](https://github.com/MDMAchine/ComfyUI_MD_Nodes/blob/main/manuals/Master%20Technical%20Manual_%20Hybrid%20Sigma%20Scheduler.md)**
#### `NOISEDECAY_SCHEDULER` (Advanced)
**Controlled fade into darkness.** Generate custom decay curves (polynomial, sigmoidal, gaussian, etc.) with start/end values, inversion, and smoothing. Modulate noise like a sad synth player.

📖 **[Read the Manual](https://github.com/MDMAchine/ComfyUI_MD_Nodes/blob/main/manuals/Master%20Technical%20Manual_%20Advanced%20Noise%20Decay%20Scheduler.md)**

#### `APG_GUIDER_FORKED`
**Low-key guiding, high-key results.** Enhanced fork for adaptive projected gradient guidance. Schedule APG scale, CFG, and momentum via YAML based on sigma. Surgical latent steering.

📖 **[Read the Manual](https://github.com/MDMAchine/ComfyUI_MD_Nodes/blob/main/manuals/Master%20Technical%20Manual_%20APG%20Guider%20(Forked).md)**

---

### 🎵 Audio Processing

#### `ADVANCED_AUDIO_PREVIEW_AND_SAVE` (AAPS)
**Hear it before you overthink it.** Comprehensive audio export: MP3/FLAC/OPUS, advanced normalization (Peak/RMS/LUFS), effects (fades, limiting), waveform/spectrogram preview, and robust metadata embedding.

📖 **[Read the Manual](https://github.com/MDMAchine/ComfyUI_MD_Nodes/blob/main/manuals/Master%20Technical%20Manual_%20Advanced%20Audio%20Preview%20&%20Save%20(AAPS).md)**

#### `AUDIO_AUTO_EQ`
**One-click audio shaping.** Analyzes audio spectrum (librosa) and applies multi-band EQ (pedalboard) based on 18+ target profiles (Vocal Clarity, De-esser, Podcast, Music Master, ASMR, etc.).

📖 `[Manual Coming Soon]`

#### `AUDIO_AUTO_MASTER_PRO`
**Iterative mastering magic.** Automatically analyzes and processes audio using filters, iterative EQ, de-essing, 3-band compression, stereo widening, and limiting to hit target LUFS and spectral profile.

📖 `[Manual Coming Soon]`

#### `MASTERING_CHAIN_NODE`
**Make your audio thicc.** Pro-grade mastering: true stereo multiband compression, surgical EQ, transparent limiting. Optimized for LUFS-normalized input. Slaps waveforms with attitude.

📖 **[Read the Manual](https://github.com/MDMAchine/ComfyUI_MD_Nodes/blob/main/manuals/Master%20Technical%20Manual_%20Custom%20Audio%20Mastering%20Chain.md)**

---

### ⚡ ACE T5 Audio Suite

#### `ACE_T5_CONDITIONING`
**The main event.** Intelligently blends T5 & CLIP embeddings. Creates high-quality audio conditioning with separate controls for genre, vocal style, and lyrical content. Supports dual CLIP & LoRAs (when used with native loader). Use this for your positive prompt!

#### `ACE_T5_MODEL_LOADER`
**The key to the kingdom.** Custom loader for ACE T5 models (`umt5_with_tokenizer.safetensors`) enabling specialized tokenizers for the highest quality conditioning. *Note: Does NOT support LoRAs.*

🔗 **[Get the Model](https://huggingface.co/ACE-Step/ACE-Step-v1-3.5B/blob/main/umt5-base/model.safetensors)**

#### `ACE_T5_ANALYZER`
**The debugger.** Inspect conditioning tensors (shape, stats, lyric tokens) to troubleshoot distortion or verify encoding. Essential for sanity checks.

---

### 🕹️ Samplers

#### `PINGPONG_SAMPLER` (Lite+ & FBG Variants)
**Symphonic frequencies & lyrical chaos.** Ancestral sampler optimized for Ace-Step. Features noise behavior presets, strength/coherence control, advanced blend modes, NaN recovery. FBG variant adds dynamic Feedback Guidance.

📖 **[Read the Manual (v0.9.9-p4 FBG)](https://github.com/MDMAchine/ComfyUI_MD_Nodes/blob/main/manuals/Master%20Technical%20Manual_%20PingPong%20Sampler%20(Custom%20v0.9.9-p2%20FBG).md).md)**
📖 **[Read the Manual (Lite+ v0.8.21)](https://github.com/MDMAchine/ComfyUI_MD_Nodes/blob/main/manuals/Master%20Technical%20Manual_%20PingPong%20Sampler%20(Lite%2B%20V0.8.20).md)**

---

### 🛠️ Workflow & Utilities

#### `SCENE_GENIUS_AUTOCREATOR`
**Prompter's divine sidekick.** Uses local LLMs (Ollama/LM Studio) to auto-generate genres, lyrics, duration, and tuned YAML configs for APG/Sampler/NoiseDecay based on a core concept. Automate your creative process.

📖 **[Read the Manual](https://github.com/MDMAchine/ComfyUI_MD_Nodes/blob/main/manuals/Master%20Technical%20Manual_%20Scene%20Genius%20Autocreator.md)**

#### `ADVANCED_TEXT_INPUT` & `TEXT_FILE_LOADER`
**Wildcard magic & external text.** Use `{option1|option2}` syntax with seed control for reproducible dynamic text. Load large prompts or configs directly from `.txt` files.

#### `SMART_FILENAME_BUILDER` Suite
**Dynamic filename toolkit.** Includes `SmartFilenameBuilder` (presets, toggles), `FilenameTokenReplacer` (simple `{token}` substitution), and `FilenameCounterNode` (persistent `#0001` counters). Organize your chaos.

📖 `[Manual Coming Soon]`

#### `MD_ENHANCED_SEED_SAVER`
**Professional seed management.** Save, load, search, manage seeds with subdirs, favorites, stats, backup, export/import. Features dynamic action & static pass-through modes for workflow stability.

📖 **[Read the Manual](https://github.com/MDMAchine/ComfyUI_MD_Nodes/blob/main/manuals/Master%20Technical%20Manual_%20MD%20Seed%20Saver.md)**

#### `LLM_VRAM_MANAGER`
**VRAM peace talks.** Unloads models from Ollama (API) or force-stops LM Studio (process kill) to prevent VRAM conflicts between ComfyUI and local LLMs directly from your workflow.

📖 `[Manual Coming Soon]`

#### `GPU_TEMPERATURE_PROTECTION` (Enhanced)
**Keep your cool.** Monitors GPU temp/VRAM via `nvidia-smi` and pauses the queue if thresholds are exceeded. Multi-GPU support, logging, adaptive cooling profiles. Prevents meltdowns.

📖 `[Manual Coming Soon]`

---

### 🗂️ Organization & Workflow Management

#### `AUTO_LAYOUT_OPTIMIZER`
**Untangle your spaghetti.** Automatically analyzes and reorganizes ComfyUI workflows using force-directed graphs. Detects node clusters, optimizes spacing, and prevents overlaps. Makes your workflow readable again.

📖 `[Manual Coming Soon]`

#### `ENHANCED_ANNOTATION_NODE`
**Notes with style.** Add styled annotations to your workflow with customizable colors, fonts, borders, and opacity. Markdown support for documentation. Make your workflow self-documenting.

📖 `[Manual Coming Soon]`

#### `SMART_COLOR_PALETTE_MANAGER`
**Color-code your chaos.** Manage and apply color schemes across workflow nodes. Create, save, and load palettes. Quickly identify node types by color. Visual organization at its finest.

📖 `[Manual Coming Soon]`

#### `UNIVERSAL_ROUTING_HUB`
**The traffic controller.** Advanced signal routing with multiple inputs/outputs, conditional routing, and signal transformation. Route any data type anywhere. Your workflow's nervous system.

📖 `[Manual Coming Soon]`

#### `WORKFLOW_SECTION_ORGANIZER`
**Divide and conquer.** Group related nodes into labeled sections with visual boundaries. Collapse/expand sections, apply section-wide settings. Perfect for complex multi-stage workflows.

📖 `[Manual Coming Soon]`

---

### 🎨 Visual Tools

#### `ACE_LATENT_VISUALIZER`
**Decode the noise gospel.** 9 modes (waveform, spectrum, heatmap, histogram, phase, difference, etc.) to visualize latent tensors. Includes batch/channel selection, peak detection, stats overlay. See what the AI sees.

📖 **[Read the Manual](https://github.com/MDMAchine/ComfyUI_MD_Nodes/blob/main/manuals/Master%20Technical%20Manual_%20ACE%20Latent%20Visualizer.md)**

#### `ADVANCED_MEDIA_SAVE` (AMS)
**Your one-stop media vault.** Saves images (PNG/JPEG/WEBP), batches, and videos (GIF/MP4/WEBM). Features metadata privacy filter, quality/framerate control, and robust timestamp-based naming to prevent overwrites.

📖 **[Read the Manual](https://github.com/MDMAchine/ComfyUI_MD_Nodes/blob/main/manuals/Master%20Technical%20Manual_%20Advanced%20Media%20Save%20(AMS).md)**

---

## 🧰 Installation

```bash
cd path/to/ComfyUI/custom_nodes
git clone https://github.com/MDMAchine/ComfyUI_MD_Nodes.git
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
| **meap158** | Comfy GPU Temp Protect Adapt. |

---

## 📜 License

Primarily **Apache 2.0**, but check individual node headers. Some are **Public Domain** or **MIT**. Open source chaos reigns.

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