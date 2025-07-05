![Synth Approved](https://img.shields.io/badge/VIBES-CHAOTIC_NEUTRAL-ff00ff?style=flat-square&logo=md&logoColor=white)
![Node Count](https://img.shields.io/badge/NODES-9_MODULES_ACTIVE-00ffff?style=flat-square)
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
┃   ✪ COMFYUI CUSTOM NODE CENTRAL ✪           v0.69.420-BETA - L33T APPROVED      ┃
┃   "Latent Space Debauchery & Digital Sorcery Unleashed Since 56k Modem Days!"    ┃
┃                                                                                  ┃
┃   Logged in as: [L33T_USER]              Last Sync: 2025-06-13 23:37 EST         ┃
┃                                                                                  ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
````

### 📟 **WELCOME TO MD\_NODES: WHERE PIXELS PARTY & NOISE IS HOLY** 📟

**Strap in, traveler.** This isn’t just another GitHub repo—it’s a wormhole into raw, modular creativity with *ComfyUI* at its freaky finest. 
Brought to you by a confederation of node-smiths, mod wizards, and digital dropouts, this is where unhinged meets unfiltered.

Whether you're forging dreamscapes from the void or just want to vibe with your VAE, this is your personal BBS of brilliance. 
Packaged with ASCII-flavored love, open-source chaos, and a healthy disregard for conventional sanity.

---

> “*Why do it the easy way when you can do it the aesthetic way?*”
> — probably MDMAchine

---

### 🔥 **THE HOLY NODES OF CHAOTIC NEUTRALITY** 🔥
Click the names for manuals

**🧠 [`HYBRID_SIGMA_SCHEDULER`](https://github.com/MDMAchine/ComfyUI_MD_Nodes/blob/main/manuals/Hybrid_Sigma_Scheduler.md)**
*‣ v0.69.420.1 🍆💦 – Karras & Linear dual-mode sigma scheduler with curve blending, featuring KL-optimal and linear-quadratic adaptations*

Outputs a tensor of sigmas to control diffusion noise levels with flexible start and end controls. Switch freely between Karras and Linear sampling styles, or blend them both using a configurable Bézier spline for full control over your denoising journey. This scheduler is designed for precision noise scheduling in ComfyUI workflows, with built-in pro tips for dialing in your noise. Perfect for artists, scientists, and late-night digital shamans.

**🔊 [`MASTERING_CHAIN_NODE`](https://github.com/MDMAchine/ComfyUI_MD_Nodes/blob/main/manuals/mastering_chain_node.md)**
*‣ v1.2 – Audio mastering for generative sound!*

This ComfyUI custom node is an audio transformation station that applies audio-style mastering techniques, making it like "Ableton Live for your tensors." It features Global Gain control to crank it to 11, a Multi-band Equalizer for sculpting frequencies, advanced Compression for dynamic shaping, and a Lookahead Limiter to prevent pesky digital overs. Now with more cowbell and less clipping, putting your sweet audio through the wringer in a good way.

**🔁 [`PINGPONG_SAMPLER_CUSTOM`](https://github.com/MDMAchine/ComfyUI_MD_Nodes/blob/main/manuals/PingPong_Sampler_Custom.md)**
*‣ v0.8.15 – Iterative denoise/re-noise dance!*

A sampler that alternates between denoising and renoising to refine media over time, acting like a finely tuned echo chamber for your latent space. You set how "pingy" (denoise) or "pongy" (re-noise) it gets, allowing for precise control over the iterative refinement process, whether aiming for crisp details or a more ethereal quality. It works beautifully for both image and text-to-audio latents, and allows for advanced configuration via YAML parameters that can override direct node inputs.

**💫 [`PINGPONG_SAMPLER_CUSTOM_FBG`](https://github.com/MDMAchine/ComfyUI_MD_Nodes/blob/main/manuals/PingPong_Sampler_Custom_FBG.md)**
*‣ v0.9.9 FBG – Denoise with Feedback Guidance for dynamic control & consistency!*

A powerful evolution of the PingPong Sampler, this version integrates Feedback Guidance (FBG) for intelligent, dynamic adjustment of the guidance scale during denoising. It combines controlled ancestral noise injection with adaptive guidance to achieve both high fidelity and temporal consistency, particularly effective for challenging time-series data like audio and video. FBG adapts the guidance on-the-fly, leading to potentially more efficient sampling and improved results.


**🔮 [`SCENE_GENIUS_AUTOCREATOR`](https://github.com/MDMAchine/ComfyUI_MD_Nodes/blob/main/manuals/SCENE_GENIUS_AUTOCREATOR.md)**
*‣ v0.1.1 – Automatic scene prompt & input generation for batch jobs, powered by AI creative weapon node!*

This multi-stage AI (ollama) creative weapon node for ComfyUI allows you to plug in basic concepts or seeds. Designed to automate Ace-Step diffusion content generation, it produces authentic genres, adaptive lyrics, precise durations, finely tuned Noise Decay, APG and PingPong Sampler YAML configs with ease, making batch experimentation a breeze.

**🎨 [`ACE_LATENT_VISUALIZER`](https://github.com/MDMAchine/ComfyUI_MD_Nodes/blob/main/manuals/ACE_LATENT_VISUALIZER.md)**
*‣ v0.3.1 – Latent-space decoder with zoom, color maps, channels, optimized for Ace-Step Audio/Video!*

Peek behind the curtain to see what your model sees. This visualization node decodes 4D latent madness into clean, readable 2D tensor maps, offering multi-mode insight including waveform, spectrum, and RGB channel split visualizations. You can choose your slice, style, and level of cognitive dissonance, making it ideal for debugging, pattern spotting, or simply admiring your AI’s hidden guts.

**📉 [`NOISEDECAY_SCHEDULER`](https://github.com/MDMAchine/ComfyUI_MD_Nodes/blob/main/manuals/NoiseDecayScheduler_Custom.md)**
*‣ v0.4.4 – Variable-step decay scheduling with cosine-based curve control*

A custom noise decay scheduler inspired by adversarial re-noising research, this node outputs a cosine-based decay curve raised to your decay_power to control steepness. It's great for stylized outputs, consistent animations, and model guidance training. Designed for use with pingpongsampler_custom or anyone seeking to escape aesthetic purgatory, use with PingPong Sampler Custom if you're feeling brave and want to precisely modulate noise like a sad synth player modulates a filter envelope.

**📡 [`APG_GUIDER_FORKED`](https://github.com/MDMAchine/ComfyUI_MD_Nodes/blob/main/manuals/APG_Guider_Forked.md)**
*‣ v0.2.2 – Plug-and-play guider module for surgical precision in latent space*

A powerful fork of the original APG Guider, this module drops into any suitable sampler to inject Adaptive Projected Gradient (APG) guidance, offering easy plug-in guidance behavior. It features better logic and adjustable strength, providing advanced control over latent space evolution for surgical precision in your ComfyUI sampling pipeline. Expect precise results, or chaos, depending on your configuration. Allows for advanced configuration via YAML parameters that can override direct node inputs

**🎛️ [`ADVANCED_AUDIO_PREVIEW_AND_SAVE`](https://github.com/MDMAchine/ComfyUI_MD_Nodes/blob/main/manuals/AdvancedAudioPreviewAndSave.md)**
*‣ v1.0 – Realtime audio previews with advanced WAV save logic and metadata privacy!*

The ultimate audio companion node for ComfyUI with Ace-Step precision. Preview generated audio directly in the UI, process it with normalization. This node saves your audio with optional suffix formatting and generates crisp waveform images for visualization. It also includes smart metadata embedding that can keep your workflow blueprints locked inside your audio files, or filter them out for privacy, offering flexible control over your sonic creations.

**💾 `MD_SEED_SAVER`**
*‣ v1.0 – Consistency is key.*

Save, load, and manage your seeds with persistent storage. Recall specific artistic directions, ensure reproducibility, and access your seed history. Never lose that perfect vibe again.


---

### 🧰 INSTALLATION: JACK INTO THE MATRIX

Should now be available to install in ComfyUI Manager under "MD Nodes"

```bash
cd path/to/ComfyUI/custom_nodes
git clone https://github.com/MDMAchine/ComfyUI_MD_Nodes.git
```
> Don’t forget to restart ComfyUI. Even gods need to reboot.


---

### 📛 WARNINGS FROM THE DIGITAL FRONTIER 📛

* **Adult humor & digital irony** throughout. If offended, please Ctrl+Alt+Del your expectations.
* **Side effects** may include:

  * Late-night node tweaking.
  * Caffeine-fueled revelations.
  * Obsessive tensor visualizations.
  * Latent hallucinations (the good kind).
* We’re not liable for friendships lost over signal chains.

---

## 💾 THE CREDITS - THESE LEGENDS WALKED SO YOU COULD SAMPLE

The foundational principles for iterative sampling, including concepts that underpin 'ping-pong sampling', are explored in works such as **Consistency Models** by Song et al. (2023) \[[https://arxiv.org/abs/2303.01469](https://arxiv.org/abs/2303.01469)].

The term 'ping-pong sampling' is explicitly introduced and applied in the context of fast text-to-audio generation in the paper **"Fast Text-to-Audio Generation with Adversarial Post-Training"** by Novack et al. (2025) from Stability AI [\[arXiv:2505.08175\]](https://arxiv.org/abs/2505.08175), where it is described as a method alternating between denoising and re-noising for iterative refinement.

The original concept for the **PingPong Sampler in the context of ace-step diffusion** was implamented by **Junmin Gong** (Ace-Step team member).

The first ComfyUI implementation of the PingPong Sampler per ace-step was created by **blepping**.

FBG addition based off of [Feedback-Guidance-of-Diffusion-Models](https://github.com/FelixKoulischer/Feedback-Guidance-of-Diffusion-Models/) - [Paper](https://arxiv.org/abs/2506.06085)

[ComfyUI FBG adaptation by: blepping](https://gist.github.com/blepping/d424e8fd27d76845ad27997820a57f6b)

| 🕶️ Handle         | 🧠 Role                                                  |
|--------------------|--------------------------------------------------------- |
| **MDMAchine**      | Main chaos wizard                                        |
| **[Junmin Gong](https://github.com/ChuxiJ)**    | Ace-Step implementation of PingPongSampler - Ace-Step Team      |
| **[blepping](https://github.com/blepping)**       | PingPongSampler ComfyUI node implementation with some tweaks<br>Mind behind OG APG guider node                           |
| **[c0ffymachyne](https://github.com/c0ffymachyne)**   | Signal alchemist / audio IO / image output               |

### 💾 BORG SUPPORT TEAM 🤖

| **Handle**          | **Contribution**                                                               |
|--------------------- |------------------------------------------------------------------------------ |
| **devstral**         | Local l33t fix-ologist — patching holes and breaking stuff since dial-up days |
| **Gemini (Google)**  | Kernel rewritin’ wizard and patch priest — holy hacks served fresh            |
| **qwen3**            | RGB soul, code completions on point — basically your digital wingman          |

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