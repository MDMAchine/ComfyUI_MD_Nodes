# MD_NODES

![Build Status](https://img.shields.io/badge/STATUS-ACTIVE_DEVELOPMENT-00ff00?style=flat-square)
![License](https://img.shields.io/badge/LICENSE-GPL_v3-blue?style=flat-square)
![Free To Use](https://img.shields.io/badge/PUBLIC_NODES-FREE_FOREVER-brightgreen?style=flat-square)
![ComfyUI](https://img.shields.io/badge/ComfyUI-Custom_Nodes-orange?style=flat-square)
![HOT-Step](https://img.shields.io/badge/HOT--Step--CPP-Lua_Plugins-purple?style=flat-square)

```text
▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
█▓▒░                                                                   ░▒▓█
█▓▒░                        ▄▄▄████████████▄▄▄                         ░▒▓█
█▓▒░                    ▄▄██████████████████████▄▄                     ░▒▓█
█▓▒░                  ▄███████▓▓▒▒░░██░░▒▒▓▓███████▄                   ░▒▓█
█▓▒░                ▄██████▓▓  ▄▄▄▄ ▒▒ ▄▄▄▄  ▓▓██████▄                 ░▒▓█
█▓▒░               ██████▓▒  ▄██████  ██████▄  ▒▓██████                ░▒▓█
█▓▒░        ▄▄    ██████▒░  ████████  ████████  ░▒██████    ▄▄         ░▒▓█
█▓▒░      ▄████  ▐█████▒   █████▓▒░█  █░▒▓█████   ▒█████▌  ████▄       ░▒▓█
█▓▒░    ▄██████  ██████░   █████████  █████████   ░██████  ██████▄     ░▒▓█
█▓▒░   ██████▀   ██████░   █████████  █████████   ░██████   ▀██████    ░▒▓█
█▓▒░    ▀██████  ▐█████▒   ▀█████▓▒░  ░▒▓█████▀   ▒█████▌  ██████▀     ░▒▓█
█▓▒░      ▀████   ██████▒░  ▀███████  ███████▀  ░▒██████   ████▀       ░▒▓█
█▓▒░        ▀▀     ██████▓▒   ▀▀████  ████▀▀   ▒▓██████     ▀▀         ░▒▓█
█▓▒░                ▀██████▓▓    ▀▀▀  ▀▀▀    ▓▓██████▀                 ░▒▓█
█▓▒░                 ▀███████▓▓▒▒░      ░▒▒▓▓███████▀                  ░▒▓█
█▓▒░           ▄▄▄     ▀▀██████████▄▄▄▄██████████▀▀     ▄▄▄            ░▒▓█
█▓▒░         ▄██████▄▄     ▀▀▀██████████████▀▀▀     ▄▄██████▄          ░▒▓█
█▓▒░       ▄████████████▄▄▄       ▀▀▀▀▀▀       ▄▄▄████████████▄        ░▒▓█
█▓▒░      ██████▓▓▒▒░▀███████▄▄▄          ▄▄▄███████▀░▒▒▓▓██████       ░▒▓█
█▓▒░     ██████▒░     ░▒▀█████████▄▄  ▄▄█████████▀▒░     ░▒██████      ░▒▓█
█▓▒░     ██████░          ░▒▀▀██████████████▀▀▒░          ░██████      ░▒▓█
█▓▒░     ▀█████▄                ░▒▀▀██▀▀▒░                ▄█████▀      ░▒▓█
█▓▒░       ▀█████▄                  ░░                  ▄█████▀        ░▒▓█
█▓▒░         ▀█████▄  ┌────────────────────────────┐  ▄█████▀          ░▒▓█
█▓▒░           ▀█████ │     M D M A   N O D E S    │ █████▀            ░▒▓█
█▓▒░             ▀▀██ │ P U R E  D I F F U S I O N │ ██▀▀              ░▒▓█
█▓▒░                ▀ └────────────────────────────┘ ▀                 ░▒▓█
█▓▒░  ┌─────────────────────────────────────────────────────────────┐  ░▒▓█
█▓▒░  │  CRACKER..: MDMAchine           PACKER....: PYTHON 3.10+    │  ░▒▓█
█▓▒░  │  STATUS...: EUPHORIC            NODES.....: 80+ UNLOCKED    │  ░▒▓█
█▓▒░  │  SUPPLY...: 115200 BAUD         PROTECT...: GPL v3          │  ░▒▓█
█▓▒░  └─────────────────────────────────────────────────────────────┘  ░▒▓█
█▓▒░                                                                   ░▒▓█
▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
```

## 📟 What is MD_Nodes?

**MD_Nodes** is a collection of **80+ advanced ComfyUI custom nodes** built by MDMAchine (A&E Concepts). Originally focused on audio generation workflows with ACE-Step, the suite has expanded to cover advanced sampling and scheduling, guidance systems, professional audio mastering, workflow automation, GGUF model loading, and system safety tools.

**Everything in this repository is free to use. All code is open source GPL v3.**

---

## 🏗️ About This Repository

**License: GPL v3 — pure Python source, no compiled binaries in this release.**

All `.py` files in this repo are complete, readable source code under the GNU General Public License v3. There are no hidden `.pyd` or `.so` binary cores in this initial public release — what you see is what you get.

### Why GPL v3?
- ✅ **Transparency** — Audit exactly how every node works.
- ✅ **Community protection** — Improvements to the wrappers must be shared back.
- ✅ **Commercial freedom** — Use these nodes in personal or commercial workflows, no restrictions.

### Future Compiled Cores
The architecture is designed to support optional `.pyd`/`.so` compiled cores for select high-value algorithms in future releases. When that happens, the compiled cores will be free to use in workflows but protected against reverse engineering. The GPL v3 Python source layer will always remain fully open. This release ships source-only.

---

## 🔥 Node Categories

### 🎨 Samplers & Schedulers
*The engines that drive the diffusion process.*

#### **PingPong Sampler FBG**
**Bidirectional ancestral sampler with feedback guidance.**
- **Classes**: `PingPongSamplerNodeBasic`, `PingPongSamplerNodeFBG`, `PingPongSamplerNodeLite`
- **Features**: Feedback guidance logic, restart modes, look-back SNR smoothing, NaN recovery
- **Use Case**: Resolves hyper-detailed textures in images or complex harmonics in audio that standard samplers turn to mush. The FBG variant is the primary production sampler for ACE-Step audio workflows.

#### **AMED Sampler**
**Adaptive multi-engine diffusion sampler.**
- **Class**: `MD_AMED_Sampler`
- **Features**: Multi-mode ancestral/predictor-corrector hybrid
- **Use Case**: Finding the middle ground between fast render speed and high-quality output.

#### **F-Sampler**
**Fast sampler using Richardson extrapolation to skip steps.**
- **Class**: `FSampler`
- **Features**: Conservative/aggressive skip strategies, quality-preserving step reduction
- **Use Case**: Cuts rendering time significantly while maintaining comparable output quality.

#### **Hybrid Sigma Scheduler**
**Universal precision noise scheduling with 14 curve algorithms.**
- **Classes**: `HybridAdaptiveSigmas_Advanced`, `_Basic`, `_Lite`
- **Modes**: Karras, Poly, AYS, Fibonacci, Exponential, Tangent, LINA, and more
- **Use Case**: Replaces the default ComfyUI scheduler with fine-grained control over when the model focuses on broad structure vs. fine detail.

#### **GITS Scheduler**
**Gradient-informed timestep scaling.**
- **Class**: `MD_GITS_Scheduler`
- **Use Case**: Allocates steps where the latent is changing most rapidly, skipping over stable regions. Efficient for both audio and image generation.

#### **Noise Decay Scheduler**
**Custom decay curves for sigma scheduling.**
- **Class**: `NoiseDecayScheduler_Custom`
- **Use Case**: Manual control over the exact noise decay profile.

---

### 🧭 Guidance & Optimization
*Tools that steer generation without destroying output quality.*

#### **APG Guider (Forked)**
**Adaptive Projected Gradient guidance.**
- **Class**: `APGGuiderForked`
- **Features**: Scheduled APG scale, CFG, and momentum by sigma
- **Use Case**: Projects guidance math safely, allowing high prompt adherence without burning or oversaturating output.

#### **Apply TPG**
**Token Perturbation Guidance — breaks repetitive patterns.**
- **Class**: `MD_ApplyTPG`
- **Use Case**: Shuffles prompt tokens at the attention level to prevent repetitive structural artifacts.

---

### 🎵 Audio Processing
*Professional audio tools built directly into the ComfyUI node graph.*

#### **Auto Master Pro**
**Iterative mastering to hit a target LUFS and spectral profile.**
- **Class**: `MD_AutoMasterNode`
- **Features**: 3-band compression, stereo widening, automatic gain staging
- **Use Case**: Makes AI-generated audio sound broadcast-ready in a single node.

#### **Audio Auto EQ**
**One-click adaptive EQ with 18+ target profiles.**
- **Class**: `MD_AudioAutoEQ_Adaptive`
- **Profiles**: Vocal Clarity, Podcast, Cinematic, Warm Analog, and more
- **Use Case**: Instantly corrects muddy, muffled, or harsh AI audio.

#### **Mastering Chain (Modular)**
**Individual mastering components for custom pipelines.**
- **Classes**: `MasteringChainNode`, `MD_Mastering_Gain`, `MD_Mastering_EQ`, `MD_Mastering_Compressor`, `MD_Mastering_Limiter`
- **Use Case**: Manual control over every stage of the mastering signal chain.

#### **Broadcast Tools**
**Loudness normalization for platform and broadcast standards.**
- **Classes**: `MD_LUFS_Normalizer`, `MD_Stereo_Width_Controller`
- **Use Case**: Hit exact LUFS targets for Spotify, YouTube, EBU R128.

#### **Advanced Audio Preview & Save (AAPS)**
**Professional audio export with metadata embedding.**
- **Class**: `AdvancedAudioPreviewAndSave`
- **Features**: MP3/FLAC/OPUS export, LUFS normalization presets, waveform visualization, workflow JSON embedding
- **Use Case**: The final output node for audio workflows. Normalizes, exports, and embeds generation metadata in a single step.

#### **Audio Simple Editor**
**Sample-accurate trimming and fading.**
- **Class**: `MD_AudioSimpleEditor`
- **Use Case**: Slice, trim, and apply linear/exponential fades to audio tensors.

---

### 📦 ACE Engine
*Nodes specifically for ACE-Step audio generation models.*

#### **ACE-Step XL Loader**
**Model loader for ACE-Step 1.5 XL variants.**
- **Class**: `MD_ACE_XLLoader`
- **Features**: Adapter support, tunable AuraFlow shift
- **Use Case**: Loads ACE-Step base, SFT, and turbo UNet variants with correct architecture detection.

#### **ACE Sigma Denoise Patcher**
**Slices an existing sigma schedule for audio-to-audio workflows.**
- **Class**: `MD_ACE_SigmaDenoisePatcher`
- **Use Case**: Feed it an existing audio file and re-generate with altered style or instrumentation while preserving the original structure.

#### **AceStep Inpaint (Generative Fill)**
**Time-based generative fill mask for audio latents.**
- **Class**: `MD_ACE_LatentInpaintMask`
- **Use Case**: Mask a specific time region in an audio latent and regenerate just that section.

---


### 🛡️ Guardian Suite
*Crash prevention and output quality protection.*

#### **NaN / Image / Audio Guardians**
**Multi-modal protection against math errors and artifacts.**
- **Classes**: `MD_NaN_Guardian`, `MD_Image_Guardian`, `MD_Audio_Guardian`
- **Use Case**: Detects and repairs NaN/Inf values in tensors before they crash ComfyUI or produce garbage output.

#### **Universal Latent Sanitizer**
**Deep-level latent repair.**
- **Class**: `MD_LatentSanitizer`
- **Use Case**: Clamps wild outlier values before VAE decode, eliminating static pops and decoding artifacts.

---

### 🛠️ Workflow Automation & Utilities

#### **Prompting & Wildcards**
**Wildcard expansion, LLM integration, and scene automation.**
- **Classes**: `WildcardPromptBuilder`, `UniversalWildcardOrchestrator`, `SceneGeniusAutocreator`
- **Use Case**: Automate prompt generation with nested wildcard logic, local LLM routing (Ollama/LM Studio), and genre/style preset libraries.

#### **YAML Configuration**
**YAML-driven parameter systems for complex nodes.**
- **Classes**: `MD_YAML_Generator`, `MD_YAML_Utils`
- **Use Case**: Load and manage complex node configurations from human-readable YAML files. Used extensively by the PingPong sampler nodes.

#### **Smart Filenames & Saving**
**Intelligent filename generation with metadata embedding.**
- **Classes**: `SmartFilenameBuilder`, `AdvancedMediaSave`
- **Use Case**: Auto-generate organized filenames and embed generation metadata.

#### **Hardware & System Management**
**Real-time VRAM/GPU monitoring and protection.**
- **Classes**: `MD_VRAMCanary`, `LLMVRAMManager`, `GPUTemperatureProtectionEnhanced`
- **Use Case**: Monitor GPU temperature and VRAM headroom, pause the queue if limits are hit.

#### **Seeds & Conditioning**
**Seed management and conditioning cache.**
- **Classes**: `EnhancedSeedSaver`, `MD_AdvancedSeedGenerator`, `MD_LoadConditioning`, `MD_SaveConditioning`
- **Use Case**: Save favorite seeds to disk, cache expensive text encoding to speed up subsequent runs.

#### **Maintenance Tools**
**Node-based repo management.**
- **Classes**: `MD_RepoMaintenance`, `MD_GlobalUpdateManager`, `MD_ModelStateReset`
- **Use Case**: Update custom nodes, clear VRAM caches, roll back broken updates.

#### **Math, Logic & Modulators**
**Conditional routing and signal modulation.**
- **Classes**: `MD_Math_Add`, `MD_Math_Subtract`, `MD_MultiSwitch`, `MD_LFO_Generator`, `MD_CustomNoiseGenerator`
- **Use Case**: Build decision-branching workflows, modulate parameters over time with LFOs, generate custom noise patterns.

---

## 🔗 Coming Soon

These algorithms are being released as standalone repositories and ComfyUI node packages. Each pairs directly with MD_Nodes workflows.

| Repo | Description |
|------|-------------|
| [MDMAchine/STORM-Sampler](https://github.com/MDMAchine/STORM-Sampler) | STORM adaptive hybrid ODE solver (STORK4 + DPM++3M) with LookBack SNR smoother |
| [MDMAchine/MD-Causal-Scheduler](https://github.com/MDMAchine/MD-Causal-Scheduler) | 14-mode causal sigma scheduler with LINA time-axis warp |
| [MDMAchine/MD-HAP-Scheduler](https://github.com/MDMAchine/MD-HAP-Scheduler) | Hamiltonian potential well sigma scheduler |
| [MDMAchine/MD-Audio-VAE-Tiled](https://github.com/MDMAchine/MD-Audio-VAE-Tiled) | Tiled VAE decoder for long-form audio with LSS, HPC, SCE |

---

## 🌐 HOT-Step-CPP

MDMAchine has contributed several plugins to [HOT-Step-CPP](https://github.com/scragnog/HOT-Step-CPP), a C++ inference runtime with a Lua plugin system maintained by [scragnog](https://github.com/scragnog).

Contributions merged upstream include the PingPong solver, Causal and HAP schedulers, STORM Guidance V2, the STORM sampler core, Seed Manager UI, and DSP improvements to the tiled decoder. More in progress including negative prompt support.

If you're running HOT-Step-CPP, these plugins ship with it — no separate install needed. Check the HOT-Step-CPP repo for the full list and release notes.

---

## 🧰 Installation

```bash
cd path/to/ComfyUI/custom_nodes
git clone https://github.com/MDMAchine/ComfyUI_MD_Nodes.git
cd ComfyUI_MD_Nodes
pip install -r requirements.txt
```

**Or via ComfyUI Manager:** search for **MD_Nodes** and click Install.

Restart ComfyUI after installation.

---

## 📋 Requirements

**Python**: 3.10+
**ComfyUI**: Latest (Nodes 2.0 compatible)
**PyTorch**: 2.0+ with CUDA (2.11+cu130 recommended for full feature set)

**Audio nodes**: `soundfile`, `librosa`, `pyloudnorm`, `pedalboard`
**Visualization**: `matplotlib`

See `requirements.txt` for the full list.

---

## 🔐 License

**GPL v3 — full source, no compiled binaries in this release.**

You are free to:
- ✅ Use in personal or commercial projects
- ✅ Read, modify, and learn from the source code
- ✅ Redistribute with attribution
- ✅ Fork and create derivative works (GPL v3 terms apply)

Your generated content (audio, images, video) is always 100% yours.

See [`LICENSE.md`](LICENSE.md) for full details.

---

## 🚀 Roadmap

### Current: June 2026 Public Release
- [x] Full audit — GPL headers, VERSION constants, unit tests
- [x] torchaudio fully migrated to soundfile
- [x] IP firewall verified — no proprietary code in public repo
- [x] HOT-Step-CPP contributions documented
- [ ] Tooltip coverage pass (currently 66%, targeting 100%)
- [ ] Full node documentation

### Next Phase
- [ ] Compiled core releases for select algorithms (.pyd/.so)
- [ ] API service endpoints (Captain Quantum, SCT Analyzer, FidelityX)
- [ ] Additional ACE-Step nodes
- [ ] Video generation utilities

---

## 💾 Credits

| Handle | Contribution |
|--------|-------------|
| **MDMAchine (Alex)** | Core architecture, all nodes, HOT-Step-CPP plugins |
| **scragnog** | HOT-Step-CPP maintainer, upstream collaboration |
| **blepping** | Original PingPong/APG concepts |
| **c0ffymachyne** | Audio I/O and signal processing research |
| **Community** | Bug reports, testing, feedback |

---

## 🐛 Support

**Issues**: [GitHub Issues](https://github.com/MDMAchine/ComfyUI_MD_Nodes/issues)
**Discussions**: [GitHub Discussions](https://github.com/MDMAchine/ComfyUI_MD_Nodes/discussions)
**Consulting / custom development**: mdmachine@gmail.com

---

```text
▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
█▓▒░                                                                   ░▒▓█
█▓▒░                 ▄▄███████▄             ▄███████▄▄                 ░▒▓█
█▓▒░              ▄█████████████▄         ▄█████████████▄              ░▒▓█
█▓▒░            ▄██████▀▀   ▀▀████▄     ▄████▀▀   ▀▀██████▄            ░▒▓█
█▓▒░          ▄██████▓▒░     ░▒▓████▄ ▄████▓▒░     ░▒▓██████▄          ░▒▓█
█▓▒░         ██████▓▒░         ░▒▓███████▓▒░         ░▒▓██████         ░▒▓█
█▓▒░        ██████▒░    ▄▄▄▄▄    ░▒█████▒░    ▄▄▄▄▄    ░▒██████        ░▒▓█
█▓▒░       ██████░   ▄█████████▄   ░███░   ▄█████████▄   ░██████       ░▒▓█
█▓▒░      ▐█████▌  ▄█████████████▄  ▐█▌  ▄█████████████▄  ▐█████▌      ░▒▓█
█▓▒░      ██████  ▐██████▓▒░░▒▓████  █  ████▓▒░░▒▓██████▌  ██████      ░▒▓█
█▓▒░      ██████  ███████▒░  ░▒█████   █████▒░  ░▒███████  ██████      ░▒▓█
█▓▒░      ██████  ███████▓▒░░▒▓█████   █████▓▒░░▒▓███████  ██████      ░▒▓█
█▓▒░      ▐█████▌  ▀██████████████▀  ▄  ▀██████████████▀  ▐█████▌      ░▒▓█
█▓▒░       ██████░   ▀██████████▀   ▄█▄   ▀██████████▀   ░██████       ░▒▓█
█▓▒░        ██████▒░    ▀▀▀▀▀▀    ▄█████▄    ▀▀▀▀▀▀    ░▒██████        ░▒▓█
█▓▒░         ██████▓▒░          ▄████▀████▄          ░▒▓██████         ░▒▓█
█▓▒░          ▀███████▓▓▒▒▒▒▒▓▓█████▀ ▀█████▓▓▒▒▒▒▒▓▓███████▀          ░▒▓█
█▓▒░            ▀██████████████████▀   ▀██████████████████▀            ░▒▓█
█▓▒░               ▀▀███████████▀▀       ▀▀███████████▀▀               ░▒▓█
█▓▒░                                                                   ░▒▓█
█▓▒░  ┌──[ SYSTEM OFFLINE ]─────────────────────────[ STAY PURE ]──┐   ░▒▓█
█▓▒░  │                                                            │   ░▒▓█
█▓▒░  │  >_ TERMINAL DISCONNECTED.                                 │   ░▒▓█
█▓▒░  │  >_ DATA DUMP SUCCESSFUL. NO CARRIER.                      │   ░▒▓█
█▓▒░  └────────────────────────────────────────────────────────────┘   ░▒▓█
█▓▒░                                                                   ░▒▓█
▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
```
