# MD_NODES

![Synth Approved](https://img.shields.io/badge/VIBES-CHAOTIC_NEUTRAL-ff00ff?style=flat-square&logo=md&logoColor=white)
![Node Count](https://img.shields.io/badge/NODES-15+_MODULES-cyan?style=flat-square)
![L33T MODE](https://img.shields.io/badge/L33T-MODE_ENABLED-red?style=flat-square&logo=hackaday)
![ComfyUI](https://img.shields.io/badge/ComfyUI-Custom_Nodes-orange?style=flat-square)

```text
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                                                                       ┃
┃  ███╗   ███╗██████╗      ███╗   ██╗ ██████╗ ██████╗ ███████╗███████╗  ┃
┃  ████╗ ████║██╔══██╗     ████╗  ██║██╔═══██╗██╔══██╗██╔════╝██╔════╝  ┃
┃  ██╔████╔██║██║  ██║     ██╔██╗ ██║██║   ██║██║  ██║█████╗  ███████╗  ┃
┃  ██║╚██╔╝██║██║  ██║     ██║╚██╗██║██║   ██║██║  ██║██╔══╝  ╚════██║  ┃
┃  ██║ ╚═╝ ██║██████╔╝     ██║ ╚████║╚██████╔╝██████╔╝███████╗███████║  ┃
┃  ╚═╝     ╚═╝╚═════╝      ╚═╝  ╚═══╝ ╚═════╝ ╚═════╝ ╚══════╝╚══════╝  ┃
┃                                                                       ┃
┃         🎛️ ComfyUI Custom Node Central - v0.69.420-BETA 🎛️           ┃
┃      "Latent Space Debauchery & Digital Sorcery Since 56k Days"       ┃
┃                                                                       ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
```

## 📟 Welcome to MD_NODES

**Strap in, traveler.** This isn't just another GitHub repo—it's a wormhole into raw, modular creativity with ComfyUI at its freaky finest.

Brought to you by a confederation of node-smiths, mod wizards, and digital dropouts, this is where unhinged meets unfiltered. Whether you're forging dreamscapes from the void or just want to vibe with your VAE, this is your personal BBS of brilliance.

Packaged with ASCII-flavored love, open-source chaos, and a healthy disregard for conventional sanity.

> *"Why do it the easy way when you can do it the aesthetic way?"*  
> — probably MDMAchine

---

## 🔥 Node Collection

### 🎵 Audio & Scheduling

#### `HYBRID_SIGMA_SCHEDULER`
**Your vibe, your noise.** Get Karras Fury (for when subtlety is dead) or Linear Chill (for flat, vibe-checked diffusion). Instantly generates noise levels like a bootleg synth wave generator trapped in a tensor. Built on rage, love, and nostalgia.

📖 **[Read the Manual](https://github.com/MDMAchine/ComfyUI_MD_Nodes/blob/main/manuals/Master%20Technical%20Manual_%20Hybrid%20Sigma%20Scheduler.md)**

#### `MASTERING_CHAIN_NODE`
**Make your audio thicc.** Think mastering, but with attitude. Slaps your waveform until it begs for release. Now with less clipping and more cowbell.

📖 **[Read the Manual](https://github.com/MDMAchine/ComfyUI_MD_Nodes/blob/main/manuals/Master%20Technical%20Manual_%20Custom%20Audio%20Mastering%20Chain.md)**

#### `PINGPONG_SAMPLER_CUSTOM`
**Symphonic frequencies & lyrical chaos.** Imagine your noise bouncing like a rave ball in a VHS tape. Originally coded in a fever dream, fixed with duct tape and dark energy.

📖 **[Read the Manual (v0.9.9-p2 FBG)](https://github.com/MDMAchine/ComfyUI_MD_Nodes/blob/main/manuals/Master%20Technical%20Manual_%20PingPong%20Sampler%20(Custom%20v0.9.9-p2%20FBG).md)**  
📖 **[Read the Manual (Lite+ v0.8.20)](https://github.com/MDMAchine/ComfyUI_MD_Nodes/blob/main/manuals/Master%20Technical%20Manual_%20PingPong%20Sampler%20(Lite+%20V0.8.20).md)**

#### `NOISEDECAY_SCHEDULER`
**Controlled fade into darkness.** Apply custom decay curves to your sigma schedule. Modulate noise like a sad synth player modulates a filter envelope. Comes with built-in cinematic moodiness.

📖 **[Read the Manual](https://github.com/MDMAchine/ComfyUI_MD_Nodes/blob/main/manuals/Master%20Technical%20Manual_%20Advanced%20Noise%20Decay%20Scheduler.md)**

#### `APG_GUIDER_FORKED`
**Low-key guiding, high-key results.** Forked and retooled for subtle prompt reinforcement that nudges rather than steamrolls. Now with a Chaos/Order slider.

📖 **[Read the Manual](https://github.com/MDMAchine/ComfyUI_MD_Nodes/blob/main/manuals/Master%20Technical%20Manual_%20APG%20Guider%20(Forked).md)**

#### `ADVANCED_AUDIO_PREVIEW_AND_SAVE`
**Hear it before you overthink it.** Preview audio waveforms inside the workflow before exporting. Includes safe saving, better waveform drawing, and normalized output. Finally, listen without guessing.

📖 **[Read the Manual](https://github.com/MDMAchine/ComfyUI_MD_Nodes/blob/main/manuals/Master%20Technical%20Manual_%20Advanced%20Audio%20Preview%20&%20Save%20(AAPS).md)**

---

### ⚡ ACE T5 Audio Suite

#### `ACE_T5_CONDITIONING`
**The main event.** Creates high-quality audio conditioning with separate controls for genre, vocal style, and lyrical content. Use this for your positive prompt!

#### `ACE_T5_MODEL_LOADER`
**The key to the kingdom.** Custom loader for ACE T5 models that enables the highest quality conditioning. Customized for use with "comfy" oriented UMT5 models.

🔗 **[Get the Model](https://huggingface.co/ACE-Step/ACE-Step-v1-3.5B/blob/main/umt5-base/model.safetensors)**

#### `ACE_T5_ANALYZER`
**The debugger.** Inspect conditioning tensors to troubleshoot distortion or verify lyrics are properly encoded.

---

### 🔮 Prompting & Utilities

#### `SCENE_GENIUS_AUTOCREATOR`
**Prompter's divine sidekick.** Feed it vibes, and it returns raw latent prophecy. Prompting was never supposed to be this dangerous. You're welcome.

📖 **[Read the Manual](https://github.com/MDMAchine/ComfyUI_MD_Nodes/blob/main/manuals/Master%20Technical%20Manual_%20Scene%20Genius%20Autocreator.md)**

#### `ADVANCED_TEXT_INPUT`
**Wildcard magic for your prompts.** Use `{option1|option2}` syntax with seed control for reproducible, dynamic text generation.

#### `TEXT_FILE_LOADER`
**External text integration.** Pulls text from external files directly into your workflow. Perfect for managing large prompt libraries or configs.

#### `UNIVERSAL_GUARDIAN`
**Your workflow's digital bodyguard.** Monitors, sanitizes, and validates inputs to prevent chaos and ensure stability. Never generate without protection.

📖 **[Read the Manual](https://github.com/MDMAchine/ComfyUI_MD_Nodes/blob/main/manuals/Master%20Technical%20Manual_%20Universal%20Guardian.md)**

---

### 🎨 Visual & Workflow Tools

#### `ACE_LATENT_VISUALIZER`
**Decode the noise gospel.** Waveform. Spectrum. RGB channel hell. Perfect for those who need to know what the AI sees behind the curtain. Because latent space is beautiful and terrifying.

📖 **[Read the Manual](https://github.com/MDMAchine/ComfyUI_MD_Nodes/blob/main/manuals/Master%20Technical%20Manual_%20ACE%20Latent%20Visualizer.md)**

#### `ADVANCED_MEDIA_SAVE`
**Your one-stop media vault.** The ultimate save node. Handles images, batches, GIFs, and videos with metadata control and guaranteed unique filenames.

📖 **[Read the Manual](https://github.com/MDMAchine/ComfyUI_MD_Nodes/blob/main/manuals/Master%20Technical%20Manual_%20Advanced%20Media%20Save%20(AMS).md)**

#### `MD_ENHANCED_SEED_SAVER`
**Professional seed management.** Save, load, search, and manage your seeds. Features dynamic action and static pass-through modes for ultimate workflow stability.

📖 **[Read the Manual](https://github.com/MDMAchine/ComfyUI_MD_Nodes/blob/main/manuals/Master%20Technical%20Manual_%20MD%20Seed%20Saver.md)**

---

## 🧰 Installation

```bash
cd path/to/ComfyUI/custom_nodes
git clone https://github.com/MDMAchine/ComfyUI_MD_Nodes.git
```

Or search for MD_Nodes in Manager!

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

| Handle | Role |
|--------|------|
| **MDMAchine** | Core concept, main chaos wizard |
| **blepping** | Original mind behind PingPongSampler |
| **c0ffymachyne** | Signal alchemist / audio IO |
| **devstral** | Local l33t, fix-ologist |
| **Gemini (Google)** | Kernel rewriter, patch priest |
| **qwen3** | Completionist with RGB soul |

---

## 📜 License

Open source chaos.

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
