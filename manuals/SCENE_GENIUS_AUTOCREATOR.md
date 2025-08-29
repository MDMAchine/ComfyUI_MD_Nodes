# Comprehensive Manual: Scene Genius Autocreator (v0.2.6)

Welcome to the complete guide for the **Scene Genius Autocreator (v0.2.6)**, your AI co-pilot for creative content generation in ComfyUI. This manual provides everything you need to know, from basic setup to advanced YAML hacking and technical details.

---

### **Table of Contents**

1.  **Introduction**
    * What is the Scene Genius Autocreator?
    * Who is this Node For?
    * Key Features in Version 0.2.6
2.  **Installation**
3.  **Core Concepts: The Generative Pipeline**
    * Stage 1: Genre Synthesis
    * Stage 2: Lyrical Conception
    * Stage 3 & 4: Temporal & Aesthetic Tuning (Duration & Noise)
    * Stage 5 & 6: Generating the DNA (APG & Sampler YAML)
4.  **Node Setup and Workflow**
    * How to Use It: A Step-by-Step Guide
5.  **Parameter Deep Dive**
    * Primary Controls: LLM & Concept
    * Creative Controls: Genre, Lyrics, & Duration
    * Advanced Controls: Sampler & Overrides
6.  **Practical Recipes & Use Cases**
    * Recipe 1: Fully Automated Audio Concept Generation
    * Recipe 2: Batch Processing with a Fixed Vibe
    * Recipe 3: Manual Override for Surgical Control
7.  **Technical Deep Dive**
    * The Order of Operations
    * Working with YAML Overrides
8.  **Troubleshooting & FAQ**

---

## 1. Introduction

### What is the Scene Genius Autocreator?

The **Scene Genius Autocreator** is a multi-stage AI "weapon" node that automates the entire creative front-end of a generation workflow. It uses a local Large Language Model (LLM) via Ollama to intelligently generate all the necessary parameters—from high-level concepts like genre and lyrics down to the intricate YAML configurations for advanced samplers like the `APG Guider` and `PingPong Sampler`. It's designed to be the "brain" of your workflow, turning a simple idea into a fully-defined, ready-to-render set of instructions.

### Who is this Node For?

* **AI Musicians & Artists:** Anyone who wants to rapidly prototype ideas without manually tweaking dozens of parameters for every new generation.
* **Batch Job Enthusiasts:** Users who run large batches and want each generation to have unique, context-aware settings based on a changing seed or concept.
* **Creative Explorers:** Anyone who wants to be surprised by the AI's interpretation of a concept and discover unexpected creative directions.
* **Power Users:** Those who want a robust, automated base configuration that they can then manually override for fine-tuning.

### Key Features in Version 0.2.6

* **Full Automation Pipeline:** Generates genre, lyrics, duration, noise decay, APG YAML, and Sampler YAML from a single prompt.
* **Local LLM Integration:** Leverages the power of your own machine via Ollama for privacy and control.
* **Intelligent Sampler Support:** Dynamically generates correct YAML for both the "Lite+" and "FBG" versions of the PingPong Sampler.
* **Robust Error Handling:** Features intelligent fallbacks to known-good default configurations if the LLM fails or produces invalid output.
* **Strict Prompt Engineering:** Heavily reinforced prompts prevent the LLM from hallucinating invalid parameters, ensuring stable and reliable YAML generation.
* **Full Override Capability:** Every stage of the generation can be manually overridden with your own text inputs for surgical control.

---

## 2. 🧰 INSTALLATION: JACK INTO THE MATRIX

This node is part of the **MD Nodes** package. All required Python libraries, including `ollama` and `PyYAML`, are listed in the `requirements.txt`.

### Method 1: ComfyUI Manager (Recommended)

1.  Open the **ComfyUI Manager**.
2.  Click "Install Custom Nodes".
3.  Search for `MD Nodes` and click "Install".
4.  The manager will download the package and automatically install its dependencies.
5.  **Restart ComfyUI.**

### Method 2: Manual Installation (Git)

1.  Open a terminal or command prompt.
2.  Navigate to your `ComfyUI/custom_nodes/` directory.
3.  Run the following command to clone the repository:
    ```bash
    git clone [https://github.com/MDMAchine/ComfyUI_MD_Nodes.git](https://github.com/MDMAchine/ComfyUI_MD_Nodes.git)
    ```
4.  Install the required dependencies by running:
    ```bash
    pip install -r ComfyUI_MD_Nodes/requirements.txt
    ```
5.  **Restart ComfyUI.**

After restarting, the node and all its features should be fully available. Don’t forget, even gods need to reboot.

---

## 3. Core Concepts: The Generative Pipeline

The Scene Genius operates in a sequential, six-stage pipeline. The output of each stage becomes part of the creative context for the next, creating a coherent and context-aware set of parameters.

### Stage 1: Genre Synthesis
The node takes your `initial_concept_prompt` and asks the LLM to brainstorm a set of descriptive genre tags. This sets the overall mood and aesthetic.

### Stage 2: Lyrical Conception
Using the concept and the newly generated genres, the LLM either writes lyrics/script or decides the piece should be instrumental. The genres inform the *style* of the writing, not the literal content.

### Stage 3 & 4: Temporal & Aesthetic Tuning (Duration & Noise)
The LLM now has a full creative brief (concept, genres, lyrics). It uses this to determine an appropriate duration in seconds and a `noise_decay_strength` for the Noise Decay Scheduler, which influences the final texture.

### Stage 5 & 6: Generating the DNA (APG & Sampler YAML)
This is the final and most complex step. The LLM receives the entire creative package and generates the highly-structured YAML configurations required by the `APG Guider` and the selected `PingPong Sampler`. The prompts are heavily reinforced to ensure the output is technically correct and optimized for audio generation.

---

## 4. Node Setup and Workflow

### How to Use It: A Step-by-Step Guide

1.  **Place the Node:** Add a `Scene Genius Autocreator` node to your workflow. It should be placed early, as its outputs will feed many other nodes.
2.  **Configure the LLM:**
    * Ensure your Ollama server is running.
    * Select your desired `ollama_model_name` from the dropdown. Smaller, faster models are often sufficient.
3.  **Define Your Concept:** Write your core creative idea in the `initial_concept_prompt` field. This is the most important input!
4.  **Set Creative Constraints:** Adjust `tag_count`, `min/max_total_seconds`, and other parameters to guide the generation.
5.  **Select Sampler Version:** Choose which PingPong Sampler you are using in your workflow (`Original` or `FBG Integrated`). This is crucial for generating the correct YAML.
6.  **Connect the Outputs:**
    * Connect `APG_YAML_PARAMS` to the `yaml_parameters_opt` input of your `APGGuiderForked` node.
    * Connect `SAMPLER_YAML_PARAMS` to the `yaml_settings_str` input of the corresponding `PingPongSampler` node.
    * Connect `NOISE_DECAY_STRENGTH` to the `decay_exponent` of a `NoiseDecayScheduler_Custom` node.
    * Connect `TOTAL_SECONDS` to any nodes that require a duration.
    * The other outputs (`GENRE_TAGS`, `LYRICS_OR_SCRIPT`, `SEED`) can be connected to text display nodes for review.
7.  **Queue Prompt:** Run the workflow. The Scene Genius will call the LLM for each stage and output the complete set of parameters.

---

## 5. Parameter Deep Dive

### Primary Controls: LLM & Concept

* **`ollama_api_base_url`**: The URL of your running Ollama instance. The default is usually correct.
* **`ollama_model_name`**: The specific model Ollama should use for generation.
* **`initial_concept_prompt`**: The seed of your creative idea. Be descriptive!
* **`seed`**: The master seed for reproducibility.

### Creative Controls: Genre, Lyrics, & Duration

* **`tag_count`**: How many genre tags to generate.
* **`excluded_tags`**: A comma-separated list of tags to forbid the LLM from using.
* **`force_lyrics_generation`**: If `True`, forces the LLM to write lyrics even if it thinks instrumental is better.
* **`min_total_seconds` / `max_total_seconds`**: The allowed range for the generated duration.

### Advanced Controls: Sampler & Overrides

* **`sampler_version`**: **Crucial.** Must match the sampler node you are using in your workflow to ensure the correct YAML is generated.
* **`prompt_*_generation`**: These optional text inputs allow you to **override** any stage of the pipeline. If you fill in `prompt_apg_yaml_generation`, for example, the node will use your text and skip calling the LLM for that stage.
* **`test_mode`**: If `True`, skips all LLM calls and returns pre-filled dummy data. Useful for quickly testing workflow connections without waiting for the AI.

---

## 6. Practical Recipes & Use Cases

### Recipe 1: Fully Automated Audio Concept Generation

Goal: Turn a single sentence into a complete, ready-to-render set of audio parameters.

* **`initial_concept_prompt`**: "A melancholic piano melody in a rainy, futuristic city."
* **`sampler_version`**: `FBG Integrated PingPong Sampler`
* Leave all `prompt_*` overrides empty.
* Connect all outputs to their respective nodes (`APGGuiderForked`, `PingPongSampler_Custom_FBG`, `NoiseDecayScheduler_Custom`, etc.).
* **Result:** A fully-formed set of parameters, from genre to intricate sampler settings, all derived from your initial concept.

### Recipe 2: Batch Processing with a Fixed Vibe

Goal: Generate 10 variations of a track, keeping the sampler settings consistent but letting the AI generate new lyrics and genres for each.

* **`initial_concept_prompt`**: "An energetic synthwave track with a driving beat."
* **`prompt_sampler_yaml_generation`**: Paste in your known-good, hand-tuned sampler YAML here.
* **`prompt_apg_yaml_generation`**: Paste in your known-good APG YAML here.
* Set your KSampler to increment the seed on each run.
* **Result:** The APG and Sampler settings will remain identical for every generation, but the LLM will still generate new genres and lyrics for each seed, providing creative variation within a stable technical framework.

### Recipe 3: Manual Override for Surgical Control

Goal: You like the generated genres and lyrics, but want to manually set the duration and noise decay.

* Run the workflow once to generate the genres and lyrics.
* Copy the `GENRE_TAGS` and `LYRICS_OR_SCRIPT` outputs into the `prompt_genre_generation` and `prompt_lyrics_decision_and_generation` override fields.
* Fill in `prompt_duration_generation` with a specific number (e.g., `90.5`).
* Fill in `prompt_noise_decay_generation` with a specific number (e.g., `2.5`).
* **Result:** The node will now use your exact manual values for the overridden stages, while still generating the YAML configs based on that context.

---

## 7. Technical Deep Dive

### The Order of Operations

The node executes its internal pipeline in a strict sequence. This is important because the output of one stage is used as context for the next.

1.  **Genre Generation** (uses `initial_concept_prompt`)
2.  **Lyrics Generation** (uses `initial_concept_prompt`, `genre_tags`)
3.  **Duration Generation** (uses `initial_concept_prompt`, `genre_tags`, `lyrics_or_script`)
4.  **Noise Decay Generation** (uses all of the above)
5.  **APG YAML Generation** (uses all of the above)
6.  **Sampler YAML Generation** (uses all of the above)

Any manual override intercepts this chain, providing the data for that stage and all subsequent stages.

### Working with YAML Overrides

When you provide a YAML override, the node will perform a basic cleaning step to remove common LLM artifacts like markdown code fences (```yaml ... ```). However, it still expects the content to be valid YAML. If parsing fails, the node will log a warning and fall back to its internal, known-good default configuration for that stage.

---

## 8. Troubleshooting & FAQ

* **"The node is throwing an error about connecting to Ollama."**
    * Make sure your Ollama server is running locally and is accessible at the `ollama_api_base_url`. Test the URL in your browser; you should see a message like "Ollama is running".

* **"The generated YAML is causing a crash in my sampler/guider node."**
    * This can happen if the LLM ignores the prompt's strict rules. The latest version (v0.2.6+) has heavily reinforced prompts to prevent this. Ensure you are on the latest version. If it still happens, try a different, more instruction-focused Ollama model.

* **"The generation is slow."**
    * The node makes multiple calls to the LLM, so its speed is entirely dependent on your hardware and the size of the Ollama model you've chosen. For faster iteration, use a smaller model (e.g., a 4B or 7B model).

* **"Can I use this for image generation?"**
    * While it's *possible*, the prompts and default configurations are heavily optimized for **audio**. The APG prompt specifically enforces `dims: [-1]`, which is correct for audio but incorrect for images. Using this for image generation will likely produce suboptimal or unpredictable results without significant changes to the prompts inside the Python code.
