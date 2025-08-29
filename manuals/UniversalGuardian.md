# Comprehensive Manual: Universal Guardian (v4.0.2)

Welcome to the complete guide for the **Universal Guardian (v4.0.2)**, a comprehensive, modality-agnostic AI guardian node for ComfyUI. This manual covers everything from installation and core concepts to advanced regeneration workflows and technical details.

---

### **Table of Contents**

1.  **Introduction**
    * What is the Universal Guardian Node?
    * Who is this Node For?
    * Key Features in Version 4.0.2
2.  **Installation**
3.  **Core Concepts: The Guardian Loop**
    * The Composite Quality Score: Defining "Good"
    * Dynamic Prompt Enhancement: Getting Creative
    * Regeneration: The Self-Correcting Workflow
    * Retry Behavior: Choosing Your Endgame
4.  **Node Setup and Workflow**
    * How to Use It: A Step-by-Step Guide
5.  **Parameter Deep Dive**
    * Core Parameters: The Guardian's Rules
    * Input Data: What to Guard
    * Regeneration Controls: The Generation Engine
    * Quality Metric Weights: Fine-Tuning a "Good Eye"
    * Ollama Enhancement Controls: The AI Co-Writer
    * Bite/Sting Enhancement Controls: The Keyword Booster
    * Debug and Control Switches
6.  **Practical Recipes & Use Cases**
    * Recipe 1: Simple Quality Gate (Evaluation Only)
    * Recipe 2: The Full Guardian Loop (Latent Regeneration)
    * Recipe 3: Creative Exploration with Ollama
7.  **Technical Deep Dive**
    * The Order of Operations
8.  **Troubleshooting & FAQ**

---

## 1. Introduction

### What is the Universal Guardian Node?

The **Universal Guardian** is an intelligent quality control system for your ComfyUI workflows. It acts as an automated art director, inspecting the output of a generation process, and deciding if it's good enough. If the quality is below your standard, it can intelligently enhance the prompt and re-run the generation, trying again and again to create the perfect result. It's modality-agnostic, meaning it can evaluate images, latents, and even has experimental support for audio and video.

### Who is this Node For?

* **Automators & Power Users:** Anyone who wants to build "fire-and-forget" workflows that automatically filter out bad results and retry until a high-quality output is achieved.
* **Creative Explorers:** Users who want to leverage LLMs like Ollama to dynamically evolve a simple prompt into something more complex and detailed.
* **Batch Processors:** Anyone running large batches who needs to ensure a minimum level of quality across all generated outputs without manual inspection.
* **Anyone Tired of Bad Generations:** If you've ever been frustrated by a blurry, low-contrast, or off-prompt result, this node is your solution.

### Key Features in Version 4.0.2

* **True Guardian Functionality:** A complete loop to evaluate, enhance the prompt, and regenerate content.
* **Multi-Modal Evaluation:** Works with **LATENT**, **IMAGE**, **AUDIO**, and **VIDEO** inputs.
* **Comprehensive Quality Metrics:** Calculates a composite score from **CLIP** similarity, **sharpness**, **contrast**, and **variance**.
* **Dual Prompt Enhancement:**
    * **Ollama Integration:** Use a local LLM for sophisticated, context-aware prompt rewriting.
    * **Bite/Sting Phrases:** A simple, effective keyword-appending system as a fallback.
* **Configurable Retry Logic:** Choose to stop on the first success, return the best of all attempts, or the very last attempt.
* **Detailed Reporting:** Outputs the final score, attempt count, final prompt, and a full JSON report of the entire process.

---

## 2. 🧰 INSTALLATION: JACK INTO THE MATRIX

This node is part of the **MD Nodes** package. All required Python libraries, including `pedalboard` for the limiter, are listed in the `requirements.txt` and should be installed automatically.

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

## 3. Core Concepts: The Guardian Loop

The node operates on a powerful three-stage cycle: **Evaluate → Enhance → Regenerate**.



### The Composite Quality Score: Defining "Good"

The Guardian doesn't just guess if an image is good. It calculates a score based on several metrics. You control how important each metric is by adjusting its weight.
* **CLIP Score:** How well does the image match the text prompt? (Requires `transformers`)
* **Sharpness:** How clear and in-focus are the details?
* **Contrast:** Is there a good dynamic range between light and dark areas?
* **Variance:** How much information and complexity is in the image?

The node combines these into a single `composite_score` from 0.0 to 1.0. If this score is below your `quality_metric_threshold`, the loop continues.

### Dynamic Prompt Enhancement: Getting Creative

When a generation fails the quality check, the Guardian tries to improve the prompt for the next attempt.
* **Ollama (The AI Co-Writer):** If enabled, the node sends the prompt to a local LLM. You give the LLM a persona (e.g., "You are a creative prompt enhancer...") and it rewrites your prompt with more vivid detail, lighting, and style cues. This is the most powerful and creative method.
* **Bite/Sting Phrases (The Keyword Booster):** If Ollama is off or unavailable, the node falls back to a simpler method. It appends descriptive keywords to your prompt, like "masterpiece, best quality" (Bite phrases) or "ethereal, dramatic" (Sting phrases).

### Regeneration: The Self-Correcting Workflow

This is the Guardian's ultimate power. If `enable_regeneration` is on and you've connected a model, conditioning, and latent, the node will use the newly enhanced prompt and a new seed to run the sampler again, creating a brand new image to evaluate. This allows it to actively fix its own mistakes.

If regeneration is disabled, the node acts as a simple "Quality Gate," either passing the input through or stopping the workflow.

### Retry Behavior: Choosing Your Endgame

What happens after the retries are done? You decide:
* **Return on First Success:** The most efficient mode. The node stops and outputs the very first result that meets the quality threshold.
* **Return Best Attempt:** The "perfectionist" mode. The node runs through *all* `max_retries`, keeping track of the highest score it saw, and returns that one at the end.
* **Return Last Attempt:** Useful for seeing the full evolution of the prompt. It simply returns whatever was generated on the final attempt, regardless of its score.

---

## 4. Node Setup and Workflow

### How to Use It: A Step-by-Step Guide

1.  **Connect an Input Source:** Connect a `LATENT`, `IMAGE`, `AUDIO`, or `VIDEO` to the corresponding input. You can only connect **one** of these.
2.  **Set Core Rules:**
    * Write your `initial_prompt`.
    * Set the `quality_metric_threshold` (e.g., 0.6). A higher value means stricter quality control.
    * Set the `max_retries` (e.g., 3).
3.  **Configure for Regeneration (Optional but powerful):**
    * Check the `enable_regeneration` box.
    * Connect `model`, `positive_conditioning`, `negative_conditioning`, and the `latent_input`. The Guardian now has everything it needs to function like a KSampler.
4.  **Choose a Prompt Enhancer:**
    * To use Ollama, check `use_ollama_enhancement` and select your model. Make sure your Ollama server is running!
    * Otherwise, customize the `bite_phrases` and `sting_phrases`.
5.  **Adjust Weights (Optional):** If you care more about prompt adherence than sharpness, increase `clip_weight` and decrease `sharpness_weight`.
6.  **Connect Outputs:**
    * Connect the `output_latent` or `output_image` to the next part of your workflow (e.g., VAE Decode or Save Image).
    * Use the `final_prompt`, `final_quality_metric`, and `quality_report` outputs for debugging or logging.
7.  **Queue Prompt:** The node will now run its loop. Watch the console or connect a preview node to see the attempts.

---

## 5. Parameter Deep Dive

### Core Parameters: The Guardian's Rules

* **`initial_prompt`**: The starting prompt for the first evaluation.
* **`quality_metric_threshold`**: The minimum composite score an output needs to be considered a "success."
* **`max_quality_threshold`**: An "early exit" threshold. If a result is this good, the loop stops immediately, even if it could retry more.
* **`max_retries`**: The maximum number of times to enhance and regenerate after the initial check.
* **`initial_seed`**: The seed for the first generation. Each retry increments this seed by 1.
* **`retry_behavior`**: `Return on First Success`, `Return Best Attempt`, or `Return Last Attempt`.

### Input Data: What to Guard

* **`latent_input` / `image_input` / `audio_input` / `video_input`**: Connect your data source here. **Only connect one.**

### Regeneration Controls: The Generation Engine

*These are only used if `enable_regeneration` is True.*
* **`model`, `positive_conditioning`, `negative_conditioning`**: The standard inputs needed for sampling.
* **`sampler_name`, `scheduler_name`, `steps`, `cfg`, `denoise`**: The KSampler settings to use for each regeneration attempt.

### Quality Metric Weights: Fine-Tuning a "Good Eye"

* **`clip_weight`, `sharpness_weight`, `contrast_weight`, `variance_weight`**: Adjust the importance of each metric in the final score. They are automatically normalized, so they don't have to add up to 1.0.

### Ollama Enhancement Controls: The AI Co-Writer

* **`use_ollama_enhancement`**: Master switch to enable Ollama.
* **`ollama_model_name`**: The model to use for enhancement (e.g., `llama3`).
* **`ollama_system_prompt`**: The instructions you give the LLM. The default is excellent for creative image prompts.
* **`ollama_temperature` / `ollama_top_p`**: Controls the creativity and randomness of the LLM's response.

### Bite/Sting Enhancement Controls: The Keyword Booster

* **`skip_enhancement`**: A master switch to disable all prompt enhancement.
* **`bite_phrases`**: Comma-separated list of quality/detail keywords.
* **`sting_phrases`**: Comma-separated list of style/artistic keywords.
* **`prompt_enhancement_ratio`**: Controls how many keywords are added per attempt.

### Debug and Control Switches

* **`enable_regeneration`**: The main switch to turn the node from a passive evaluator into an active generator.
* **`debug_mode`**: Prints detailed logs and metric scores to the console for each attempt.

---

## 6. Practical Recipes & Use Cases

### Recipe 1: Simple Quality Gate (Evaluation Only)

Goal: Check an image from a previous step. If it's blurry or low-contrast, stop the workflow.

* **Connect**: `image_input` from a VAE Decode.
* **`quality_metric_threshold`**: `0.55`
* **`max_retries`**: `0` (we only want to check once)
* **`enable_regeneration`**: `False`
* **Result**: The `output_image` will only contain data if the input image's score was >= 0.55. Otherwise, it will be empty, stopping downstream nodes.

### Recipe 2: The Full Guardian Loop (Latent Regeneration)

Goal: Generate an image and automatically re-roll if it's not good enough.

* **Connect**: `model`, `positive`, `negative` from your CLIP Text Encode nodes. Connect a `latent_input` from an Empty Latent node.
* **`initial_prompt`**: "A majestic space whale"
* **`quality_metric_threshold`**: `0.65`
* **`max_retries`**: `5`
* **`retry_behavior`**: `Return Best Attempt`
* **`enable_regeneration`**: `True`
* **`use_ollama_enhancement`**: `False` (using Bite/Sting for this example)
* **Result**: The node will generate the space whale. If the score is low, it will start adding phrases like "cinematic lighting, masterpiece" and try again up to 5 times, finally outputting the latent for the highest-scoring image it created.

### Recipe 3: Creative Exploration with Ollama

Goal: Start with a simple idea and let an LLM creatively expand on it over several iterations.

* **Connect**: Same as Recipe 2.
* **`initial_prompt`**: "A cat in a library"
* **`quality_metric_threshold`**: `0.75` (set it high to encourage retries)
* **`max_retries`**: `4`
* **`retry_behavior`**: `Return Last Attempt`
* **`enable_regeneration`**: `True`
* **`use_ollama_enhancement`**: `True`
* **`ollama_model_name`**: `llama3` or your preferred model.
* **Result**: The Guardian might fail the first attempt. For the next, Ollama might change the prompt to "A fluffy ginger cat sleeping on a pile of ancient, leather-bound books in a sun-drenched, dusty library, cinematic lighting." The next attempt might be even more detailed. This recipe uses the Guardian as a tool for emergent creativity.

---

## 7. Technical Deep Dive

### The Order of Operations

Understanding the node's internal sequence helps you predict its behavior:
1.  **Initial Evaluation:** The node first evaluates the incoming data with the `initial_prompt`.
2.  **Threshold Check:** It compares the score against `quality_metric_threshold` and `max_quality_threshold`. If it passes, it exits immediately.
3.  **Enter Retry Loop:** If the check fails, the loop begins.
4.  **Seed Increment:** The current seed is increased by 1.
5.  **Prompt Enhancement:** The prompt is modified using either Ollama or the Bite/Sting method.
6.  **Regeneration:** If enabled, the `comfy.sample.sample` function is called with the new prompt and seed to create a new latent. If disabled, this step is skipped, and the original input is re-evaluated against the new prompt.
7.  **New Evaluation:** The new (or original) data is evaluated against the new prompt.
8.  **Update Best:** The score is compared to the best score found so far.
9.  **Loop or Exit:** The node checks the thresholds again. If it meets the criteria or has run out of retries, it exits the loop.
10. **Final Output:** The node returns the appropriate data based on the chosen `retry_behavior`.

---

## 8. Troubleshooting & FAQ

* **"My CLIP score is always 0.0."**
    * The `transformers` library is likely not installed correctly. Check the ComfyUI console on startup for import errors. You may need to manually install dependencies (see Installation).

* **"Ollama enhancement isn't working."**
    * First, ensure your local Ollama server is running. Second, verify the `ollama_server_url` is correct. Third, make sure the `ollama_model_name` you selected is actually downloaded and available in Ollama. The node will print a warning to the console if it can't connect.

* **"Regeneration isn't happening, even though `enable_regeneration` is on."**
    * For regeneration, you MUST provide all the necessary inputs: `model`, `positive_conditioning`, `negative_conditioning`, and `latent_input`. If any of these are missing, the node cannot perform sampling and will skip the regeneration step.

* **"The node seems to re-run every time, even if I don't change anything."**
    * This is by design. The node's `IS_CHANGED` function is intentionally set to always trigger execution. This is necessary for the node to function correctly, especially when using random seeds or when it's part of a larger, iterating workflow.

* **"What's a good `quality_metric_threshold`?"**
    * This is highly subjective and depends on your model and what you're generating. A good starting point is between **0.55 and 0.65**. Set it lower to be more lenient and higher to be stricter. Use `debug_mode` to see the scores your typical generations are getting, then set your threshold accordingly.