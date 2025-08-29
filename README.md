![Synth Approved](https://img.shields.io/badge/VIBES-CHAOTIC_NEUTRAL-ff00ff?style=flat-square&logo=md&logoColor=white)
![Node Count](https://img.shields.io/badge/NODES-12_MODULES_ACTIVE-00ffff?style=flat-square)
![L33T MODE](https://img.shields.io/badge/L33T-MODE_ENABLED-red?style=flat-square&logo=hackaday)

```text
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                                                                                  ┃
┃  ███╗   ███╗██████╗      ████╗   █████╗ ██████╗     ███╗   ██╗ ██████╗ ███████╗  ┃
┃  ████╗ ████║██╔══██╗    ██╔══██╗██╔══██╗██╔══██╗    ████╗  ██║██╔═══██╗██╔════╝  ┃
┃  ██╔████╔██║██████╔╝    ███████║███████║██║  ██║    ██╔██╗ ██║██║   ██║███████╗  ┃
┃  ██║╚██╔╝██║██╔═══╝     ██╔══██║██╔══██║██║  ██║    ██║╚██╗██║██║   ██║╚════██║  ┃
┃  ██║ ╚═╝ ██║██║         ██║  ██║██║  ██║██████╔╝    ██║ ╚████║╚██████╔╝███████║  ┃
┃  ╚═╝     ╚═╝╚═╝         ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝     ╚═╝  ╚═══╝ ╚═════╝ ╚══════╝  ┃
┃                                                                                  ┃
┃   ✪ COMFYUI CUSTOM NODE CENTRAL ✪           v1.4.0 - L33T APPROVED              ┃
┃   "Latent Space Debauchery & Digital Sorcery Unleashed Since 56k Modem Days!"    ┃
┃                                                                                  ┃
┃   Logged in as: [L33T_USER]              Last Sync: 2025-08-28 22:42 EST         ┃
┃                                                                                  ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
```

### 📟 **WELCOME TO MD\_NODES: WHERE PIXELS PARTY & NOISE IS HOLY** 📟

**Strap in, traveler.** This isn’t just another GitHub repo—it’s a wormhole into raw, modular creativity with *ComfyUI* at its freaky finest.
Brought to you by a confederation of node-smiths, mod wizards, and digital dropouts, this is where unhinged meets unfiltered.

Whether you're forging dreamscapes from the void or just want to vibe with your VAE, this is your personal BBS of brilliance.
Packaged with ASCII-flavored love, open-source chaos, and a healthy disregard for conventional sanity.

---

### 🧰 INSTALLATION: JACK INTO THE MATRIX

This node is part of the **MD Nodes** package. All required Python libraries are listed in the `requirements.txt` and should be installed automatically.

#### Method 1: ComfyUI Manager (Recommended)

1.  Open the **ComfyUI Manager**.
2.  Click "Install Custom Nodes".
3.  Search for `MD Nodes` and click "Install".
4.  The manager will download the package and automatically install its dependencies.
5.  **Restart ComfyUI.**

#### Method 2: Manual Installation (Git)

1.  Open a terminal or command prompt.
2.  Navigate to your `ComfyUI/custom_nodes/` directory.
3.  Run the following command to clone the repository:
    ```bash
    git clone https://github.com/MDMAchine/ComfyUI_MD_Nodes.git
    ```
4.  Install the required dependencies by running:
    ```bash
    pip install -r ComfyUI_MD_Nodes/requirements.txt
    ```
5.  **Restart ComfyUI.**

> **Note for Audio Nodes:** This node pack uses `torchaudio` for audio processing. Most ComfyUI installations include it, but if you encounter errors with audio nodes, you may need to install it manually. Please follow the official [PyTorch installation instructions](https://pytorch.org/get-started/locally/) to get the correct version that matches your existing PyTorch and system configuration (CPU/CUDA).

---

### 📡 **A WORD FROM THE SYSOP: PROJECT PHILOSOPHY**

This node pack is a living, breathing experiment. It's built on a foundation of chaotic creativity, rapid prototyping, and a healthy dose of digital anarchy. Think of it less as a polished commercial product and more as a tricked-out pirate radio station broadcasting from the fringes of latent space.

* **Embrace the Chaos:** These nodes are designed to be powerful and, at times, unpredictable. They are tools for exploration. Sometimes the best results come from pushing a parameter until something beautiful and unexpected breaks.
* **Built for Audio, Hacked for Image/Video:** The core samplers and schedulers were born from a deep dive into audio synthesis. They have been adapted and tuned for image and video, but their soul is sonic. This is why they produce unique temporal effects.
* **The Bleeding Edge is Sharp:** We prioritize adding new, strange features over maintaining perfect backward compatibility. A workflow that worked last week might need a tweak to work today. This is the price of admission for riding the bleeding edge.

---

### 📛 **WARNING: THIS IS A HIGHLY EXPERIMENTAL ZONE** 📛

Consider this entire repository a volatile, experimental playground. Nodes may appear, disappear, change names, or have their inputs completely rewired between updates without warning. A feature you love today might be gone tomorrow, only to reappear in a new, weirder form next week.

* **Things Will Break:** Expect it. Embrace it. Report it.
* **Functionality is Fluid:** A node that works perfectly today might develop a sudden case of existential dread tomorrow. We'll fix it... eventually.
* **Documentation May Lag Behind:** We try to keep the manuals updated, but sometimes the code evolves faster than the words can keep up. The truest manual is the source code itself.

**Your feedback, bug reports, and pull requests are the lifeblood of this project.** You are not just a user; you are a fellow explorer in this chaotic neutral territory.

---

### 🔥 **THE HOLY NODES OF CHAOTIC NEUTRALITY** 🔥
Click the names for the full user manuals.

**🔮 [`SCENE GENIUS AUTOCREATOR`](manuals/SCENE_GENIUS_AUTOCREATOR.md)**
*‣ v0.2.6 – Your AI creative weapon for automated content generation!*

A multi-stage AI "weapon" node that automates the entire creative front-end of a workflow. It uses a local LLM via Ollama to intelligently generate everything from genre and lyrics down to the intricate YAML configurations for advanced samplers like the `APG Guider` and `PingPong Sampler`.

**🛡️ [`UNIVERSAL GUARDIAN`](manuals/UniversalGuardian.md)**
*‣ v4.0.2 – The ultimate quality control gatekeeper for any workflow.*

A comprehensive, modality-agnostic AI guardian that performs quality evaluation (CLIP, sharpness, etc.) and can regenerate content to meet your standards. Features dynamic prompt enhancement, configurable retry behavior, and live previews for images, audio, and video.

**🔁 [`PINGPONG SAMPLER (LITE+)`](manuals/PingPong_Sampler_Custom.md)**
*‣ v0.8.20 – The classic iterative denoise/re-noise dance with enhanced behavior control!*

A powerful and intuitive custom sampler optimized for audio and video. This version introduces "Noise Behavior" presets to control the **magnitude (strength)** and **evolution (coherence)** of ancestral noise, unlocking a vast range of artistic effects from smooth and cinematic to chaotic and glitchy.

**💫 [`PINGPONG SAMPLER (CUSTOM FBG)`](manuals/PingPong_Sampler_Custom_FBG.md)**
*‣ v0.9.9-p2 – Denoise with dynamic Feedback Guidance for ultimate consistency!*

The evolution of the PingPong Sampler, integrating Feedback Guidance (FBG) for intelligent, dynamic adjustment of the guidance scale. It combines ancestral noise injection with adaptive guidance to achieve high fidelity and temporal consistency, perfect for complex audio and video.

**📡 [`APG GUIDER (FORKED)`](manuals/APG_Guider_Forked.md)**
*‣ v0.2.0 – A plug-and-play guider module for surgical precision in latent space.*

A powerful fork of the original APG Guider that injects Adaptive Projected Gradient (APG) guidance into your sampling pipeline. While standard CFG pushes, APG steers. Expect surgical precision... or beautiful chaos, depending on your config.

**🧠 [`HYBRID SIGMA SCHEDULER`](manuals/Hybrid_Sigma_Scheduler.md)**
*‣ v0.71 – Multi-mode sigma scheduling for pure generative groove.*

A precision noise scheduler that automatically detects the optimal sigma range from your model. Features multiple modes (Karras, Polynomial, etc.), schedule slicing for img2img, and an `actual_steps` output to keep your sampler perfectly in sync.

**📉 [`NOISE DECAY SCHEDULER (ADVANCED)`](manuals/NoiseDecayScheduler_Custom.md)**
*‣ v0.6.0 – An elite toolkit for designing custom noise schedules.*

A "curve designer" node that generates a highly customized mathematical curve to tell a compatible sampler *how* to remove noise at every step. The ultimate tool for overriding default sampler behavior and dictating the precise "vibe" and texture of your output.

**💾 [`MD SEED SAVER`](manuals/MD_Seed_Saver_Manual.md)**
*‣ v1.0 – Consistency is key.*

Save, load, delete, and manage your generation seeds using simple named files. Features subdirectory organization, random seed loading, and automatic naming to enhance reproducibility and organize your creative projects.

**🎨 [`ACE LATENT VISUALIZER`](manuals/ACE_LATENT_VISUALIZER.md)**
*‣ v0.3.1 – A multi-mode viewport into the soul of your latent tensor.*

A powerful debugging and inspection tool that transforms the abstract, numerical data inside a latent tensor into a clear, human-readable plot. Offers waveform, spectrum, and RGB split views to help you understand *what* is happening inside your generative workflows.

**🔊 [`CUSTOM AUDIO MASTERING CHAIN`](manuals/mastering_chain_node.md)**
*‣ v1.2a – Your personal audio mastering suite inside ComfyUI.*

An all-in-one audio finalization node that brings the essential tools of a professional mastering studio into your workflow. It provides sequential processing for gain, equalization (shelf and parametric), dual-mode compression (single-band or multiband), and a lookahead peak limiter.

**🎛️ [`ADVANCED AUDIO PREVIEW & SAVE`](manuals/AdvancedAudioPreviewAndSave.md)**
*‣ v0.4.1 – The ultimate audio companion with waveform previews and metadata control.*

Processes, previews, and saves audio in FLAC, MP3, or OPUS. Generates crisp waveform images and includes a metadata privacy filter to embed your workflow or keep your secrets safe. Features normalization, fades, and a soft limiter.

**🖼️ [`ADVANCED MEDIA SAVE`](manuals/Advanced_Media_Save.md)**
*‣ v1.0.0 – Your one-stop-shop for saving any visual output from ComfyUI.*

Processes, previews, and saves images, image batches, and videos. Supports PNG, JPEG, WEBP, GIF, MP4, and WEBM formats with quality controls, framerate settings, and full metadata embedding.

---

### 💾 THE CREDITS - THESE LEGENDS WALKED SO YOU COULD SAMPLE

The foundational principles for iterative sampling, including concepts that underpin 'ping-pong sampling', are explored in works such as **Consistency Models** by Song et al. (2023) \[[https://arxiv.org/abs/2303.01469](https://arxiv.org/abs/2303.01469)].

The term 'ping-pong sampling' is explicitly introduced and applied in the context of fast text-to-audio generation in the paper **"Fast Text-to-Audio Generation with Adversarial Post-Training"** by Novack et al. (2025) from Stability AI [\[arXiv:2505.08175\]](https://arxiv.org/abs/2505.08175), where it is described as a method alternating between denoising and re-noising for iterative refinement.

The original concept for the **PingPong Sampler in the context of ace-step diffusion** was implemented by **Junmin Gong** (Ace-Step team member).

The first ComfyUI implementation of the PingPong Sampler per ace-step was created by **blepping**.

FBG addition based off of [Feedback-Guidance-of-Diffusion-Models](https://github.com/FelixKoulischer/Feedback-Guidance-of-Diffusion-Models/) - [Paper](https://arxiv.org/abs/2506.06085)

[ComfyUI FBG adaptation by: blepping](https://gist.github.com/blepping/d424e8fd27d76845ad27997820a57f6b)

| 🕶️ Handle                                           | 🧠 Role                                                                                                     |
| --------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| **MDMAchine** | Main chaos wizard                                                                                           |
| **[Junmin Gong](https://github.com/ChuxiJ)** | Ace-Step implementation of PingPongSampler - Ace-Step Team                                                  |
| **[blepping](https://github.com/blepping)** | PingPongSampler ComfyUI node implementation with some tweaks<br>Mind behind OG APG guider node                |
| **[c0ffymachyne](https://github.com/c0ffymachyne)** | Signal alchemist / audio IO / image output                                                                  |

### 💾 BORG SUPPORT TEAM 🤖

| **Handle** | **Contribution** |
| ------------------- | ----------------------------------------------------------------------------- |
| **devstral** | Local l33t fix-ologist — patching holes and breaking stuff since dial-up days |
| **Gemini (Google)** | Kernel rewritin’ wizard and patch priest — holy hacks served fresh            |
| **qwen3** | RGB soul, code completions on point — basically your digital wingman          |

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
