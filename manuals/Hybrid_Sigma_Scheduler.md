# Comprehensive Manual: Hybrid Sigma Scheduler (v0.71)

Welcome to the complete guide for the **Hybrid Sigma Scheduler (v0.71)** node, an advanced noise scheduler for precision control over the diffusion process in ComfyUI. This manual covers everything from core concepts to practical workflow recipes and technical details.

---

### **Table of Contents**

1.  **Introduction**
    * What is the Hybrid Sigma Scheduler?
    * Who is this Node For?
    * Key Features in Version 0.71
2.  **Installation**
3.  **Core Concepts: The Art of Noise Scheduling**
    * What Are Sigmas? The Heartbeat of Diffusion
    * Why Scheduling Matters: The Path from Chaos to Clarity
    * The Scheduler Modes: Your Noise Profile Toolkit
    * Sigma Range: The Canvas Boundaries (`sigma_min` & `sigma_max`)
4.  **Node Setup and Workflow**
    * How to Use It: A Step-by-Step Guide
5.  **Parameter Deep Dive**
    * Primary Controls: Model, Steps & Mode
    * Slicing Controls: Precision Timing
    * Scheduler-Specific Controls: Fine-Tuning Your Curve
    * Sigma Override Controls: Expert Mode
    * Optional & Passthrough Controls
    * Node Outputs: The Final Schedule
6.  **Practical Recipes & Use Cases**
    * Recipe 1: Standard High-Quality Text-to-Image
    * Recipe 2: Controlled img2img with Schedule Slicing
    * Recipe 3: Experimental High-Detail Generation
7.  **Technical Deep Dive**
    * The Order of Operations
8.  **Troubleshooting & FAQ**

---

## 1. Introduction

### What is the Hybrid Sigma Scheduler?

The **Hybrid Sigma Scheduler** is a powerful utility node that gives you fine-grained control over the noise schedule used by samplers in a diffusion workflow. Instead of relying on the sampler's default, often simplistic schedule, this node allows you to generate a custom sequence of sigma (noise) values. This sequence dictates the exact noise level at every step, directly influencing the final image's coherence, detail, and style. It's the difference between letting the car drive itself and being in the driver's seat with a map and a mission.

### Who is this Node For?

* **Generative Artists & Detail Fanatics:** Anyone who wants to push their image quality to the limit by perfectly tailoring the denoising process to their model and subject.
* **Workflow Technicians:** Users creating advanced workflows for tasks like img2img, inpainting, or latent upscaling, where precise control over the start and end of the denoising process is critical.
* **Experimenters & Sampler Theorists:** Anyone curious about how different noise distributions affect the final output and wants a toolkit to explore those effects.
* **All ComfyUI Users:** Its ability to automatically pull the correct sigma range from the model makes it a smarter, more reliable choice than a sampler's default settings.

### Key Features in Version 0.71

* **Automatic Sigma Range Detection:** Intelligently reads the optimal `sigma_min` and `sigma_max` directly from the connected model, ensuring you always start with the right values.
* **Multiple Scheduler Modes:** Go beyond the basics with a suite of schedulers: `karras_rho`, `adaptive_linear`, `polynomial`, `blended_curves`, `kl_optimal`, and `linear_quadratic`.
* **Precision Schedule Slicing:** Trim the start and end of the noise schedule with `start_at_step` and `end_at_step` for perfect control in img2img or multi-stage workflows.
* **`actual_steps` Output:** A crucial output that provides the exact number of steps in the final *sliced* schedule, preventing mismatches with your sampler settings.
* **Manual Overrides:** Full power for experts to override the model's default sigma range for specific effects.
* **Schedule Reversal:** An experimental feature to reverse the schedule from low noise to high noise for unique effects.

---

## 2. 🧰 INSTALLATION: JACK INTO THE MATRIX

This node is part of the **MD Nodes** package. All required Python libraries are listed in the `requirements.txt` and should be installed automatically.

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

## 3. Core Concepts: The Art of Noise Scheduling

### What Are Sigmas? The Heartbeat of Diffusion

In a diffusion model, "sigma" ($\sigma$) is a number that represents the amount of noise present in the latent image at a given time. A high sigma value means the image is mostly random noise (like at the very beginning of generation). A low sigma value means the image is very clean and detailed (like at the end). The entire generation process is a journey from a high sigma to a low sigma.

### Why Scheduling Matters: The Path from Chaos to Clarity

A **noise schedule** is the series of sigma values used at each step of the sampling process. It's the "path" the sampler takes from chaos to clarity. The shape of this path is incredibly important:
* A **steep curve** at the beginning helps the model quickly establish the main composition.
* A **flatter curve** at the end allows the model to spend more time refining fine details.
The scheduler determines the shape of this curve. A good schedule helps the sampler converge on a high-quality image more efficiently.

### The Scheduler Modes: Your Noise Profile Toolkit

* **`karras_rho`**: The gold standard for many. Based on the Karras paper, it's known for producing high-quality, coherent images. The `rho` parameter controls the curve's steepness.
* **`adaptive_linear`**: A simple, straight-line descent from high noise to low noise. Predictable and stable.
* **`polynomial`**: A curve defined by a power exponent. Offers a smooth, controllable curve that can be more or less aggressive than linear.
* **`blended_curves`**: Your personal DJ mixer for blending the `karras_rho` and `adaptive_linear` schedules.
* **`kl_optimal`**: A schedule based on the "Align Your Steps" paper, designed to be mathematically optimal for sample quality.
* **`linear_quadratic`**: A two-part schedule that starts linear and finishes with a more aggressive quadratic curve.

### Sigma Range: The Canvas Boundaries (`sigma_min` & `sigma_max`)

Every diffusion model is trained to operate within a specific range of noise levels.
* **`sigma_max`**: The highest noise level the model understands (the starting point).
* **`sigma_min`**: The lowest noise level the model can effectively work with (the ending point).
Using the wrong range is like giving an artist a canvas that's too big or too small. This node automatically detects the correct range from the model, but also allows you to override it for advanced techniques.

---

## 4. Node Setup and Workflow

### How to Use It: A Step-by-Step Guide

1.  **Connect the Model:** Connect the `MODEL` output from your "Load Checkpoint" node to the `model` input of the Hybrid Sigma Scheduler. This is crucial for automatic sigma range detection.
2.  **Set Total Steps & Mode:**
    * Choose the total number of steps you want to generate in the `steps` input. This is the "resolution" of your schedule *before* slicing.
    * Select your desired `mode` from the dropdown (e.g., `karras_rho`).
3.  **Configure Slicing (for img2img or advanced workflows):**
    * Leave `start_at_step` at `0` for standard text-to-image. For img2img, you might start at a later step to match your denoise value.
    * Leave `end_at_step` at its default high value (`9999`) to run to the end.
4.  **Tune Mode-Specific Parameters:** Adjust `rho`, `power`, `blend_factor`, etc., based on the mode you selected.
5.  **Connect Outputs to Sampler:**
    * Connect the `sigmas` output directly to the `sigmas` input on your KSampler node.
    * Connect the `actual_steps` output to the `steps` input on your KSampler. This is **critical** to ensure the sampler runs for the correct number of steps after slicing.
6.  **Set Sampler to `external`:** In your KSampler node, set the `scheduler` parameter to **`external`**. This tells the sampler to use the custom sigmas you are providing.
7.  **Queue Prompt:** Run your workflow. The scheduler will generate the custom sigma schedule, and the sampler will use it to generate the image.



---

## 5. Parameter Deep Dive

### Primary Controls: Model, Steps & Mode

* **`model`** (Required): The diffusion model. It's used to automatically determine the correct `sigma_min` and `sigma_max`.
* **`steps`** (`INT`, default: `60`): The total number of steps to generate for the schedule *before* any slicing is applied.
* **`mode`** (`ENUM`): The core algorithm for generating the noise curve. See "Core Concepts" for a breakdown.

### Slicing Controls: Precision Timing

* **`start_at_step`** (`INT`, default: `0`): The step index to *begin* the final schedule from. A value of `10` will discard the first 10 steps.
* **`end_at_step`** (`INT`, default: `9999`): The step index to *end* the final schedule at (exclusive). The default high value effectively means "run to the very end."

### Scheduler-Specific Controls: Fine-Tuning Your Curve

* **`rho`** (`FLOAT`, default: `1.5`): For `karras_rho` mode. Controls the curve's steepness. Values between 1.1 and 4.5 are common. Higher values are more aggressive.
* **`blend_factor`** (`FLOAT`, default: `0.5`): For `blended_curves` mode. A value of `0.0` is pure Karras, `1.0` is pure Linear.
* **`power`** (`FLOAT`, default: `2.0`): For `polynomial` mode. A value of `1.0` is linear. `>1.0` is a faster start, `<1.0` is a slower start.
* **`threshold_noise` / `linear_steps`**: Advanced controls for the `linear_quadratic` mode.

### Sigma Override Controls: Expert Mode

* **`start_sigma_override`** (`FLOAT`, optional): Manually set the `sigma_max` value. This will override the value read from the model.
* **`end_sigma_override`** (`FLOAT`, optional): Manually set the `sigma_min` value. This will override the value read from the model.

### Optional & Passthrough Controls

* **`denoise`** (`FLOAT`, default: `1.0`): This is a **passthrough** value. The node does not use it. You can connect it to your KSampler's `denoise` input for convenience.
* **`reverse_sigmas`** (`BOOLEAN`, default: `False`): If enabled, flips the final schedule to go from low noise to high noise. For experimental use.

### Node Outputs: The Final Schedule

* **`sigmas`** (SIGMAS): The final, sliced tensor of sigma values. This plugs directly into the `sigmas` input of a KSampler.
* **`actual_steps`** (INT): The final number of steps in the `sigmas` tensor. This plugs into the `steps` input of a KSampler.

---

## 6. Practical Recipes & Use Cases

### Recipe 1: Standard High-Quality Text-to-Image

Goal: Generate a standard image with a reliable, high-quality schedule.

* **`model`**: Connect your checkpoint (e.g., SDXL Base).
* **`steps`**: `40`
* **`mode`**: `karras_rho`
* **`rho`**: `3.0`
* **`start_at_step`**: `0` (start from the beginning)
* **`end_at_step`**: `9999` (run to the end)
* **Sampler Setup**: Connect `sigmas` and `actual_steps` to your KSampler. Set sampler `scheduler` to `external`.

### Recipe 2: Controlled img2img with Schedule Slicing

Goal: Apply a specific amount of denoising to an existing image. Let's say your KSampler `denoise` is set to `0.6`.

* **`model`**: Connect your checkpoint.
* **`steps`**: `50` (a higher base resolution for more slicing precision)
* **`mode`**: `karras_rho`
* **`start_at_step`**: `20` (Calculated as `total_steps * (1 - denoise)`. So, `50 * (1 - 0.6) = 20`. We are skipping the first 40% of the steps).
* **`end_at_step`**: `9999`
* **Sampler Setup**: Connect `sigmas` and `actual_steps` to your KSampler. `actual_steps` will now output `30`. Set sampler `scheduler` to `external` and `denoise` to `0.6`.

### Recipe 3: Experimental High-Detail Generation

Goal: Spend more time on fine details at the end of the generation.

* **`model`**: Connect your checkpoint.
* **`steps`**: `60`
* **`mode`**: `polynomial`
* **`power`**: `0.7` (A power less than 1.0 creates a curve that is flatter at the end, meaning more steps are spent at lower noise levels).
* **`start_at_step`**: `0`
* **`end_at_step`**: `9999`
* **Sampler Setup**: Connect `sigmas` and `actual_steps` to your KSampler. Set sampler `scheduler` to `external`.

---

## 7. Technical Deep Dive

### The Order of Operations

The node processes the data in this specific order to generate the final schedule:

1.  **Determine Sigma Range:** The node first tries to get `sigma_min` and `sigma_max` from the connected `model`.
2.  **Apply Overrides:** If `start_sigma_override` or `end_sigma_override` are provided, these values replace the ones from the model.
3.  **Apply Fallback:** If no values could be determined from the model or overrides, it uses hardcoded fallback values (defaults for SD1.5).
4.  **Generate Full Schedule:** Using the final sigma range, total `steps`, and selected `mode`, it generates the complete, unsliced noise schedule.
5.  **Slice Schedule:** The full schedule is then sliced according to the `start_at_step` and `end_at_step` values.
6.  **Reverse (Optional):** If `reverse_sigmas` is enabled, the final sliced schedule is flipped.
7.  **Output:** The final `sigmas` tensor and the calculated `actual_steps` are outputted.

---

## 8. Troubleshooting & FAQ

* **"My output looks noisy, blurry, or generally low-quality."**
    * This can happen if the sigma range is wrong. Ensure you have the `model` connected correctly. If you are using manual overrides, double-check that they are appropriate for your model (e.g., SD1.5 `sigma_max` is ~14.6, while SDXL is ~1.0). Also, try adjusting the `rho` or `power` values.
* **"Why is my KSampler only running for 25 steps when I set the scheduler to 40?"**
    * This is the exact problem the `actual_steps` output solves! You likely used `start_at_step` or `end_at_step` to slice the schedule. The original schedule had 40 steps, but the sliced version has 25. You must connect the `actual_steps` output to the sampler's `steps` input to keep them in sync.
* **"I'm getting an error that says `sigma_min` is greater than `sigma_max`."**
    * This happens if your `start_sigma_override` is a smaller number than your `end_sigma_override`. Remember that `start_sigma` corresponds to the maximum noise (`sigma_max`), and `end_sigma` corresponds to the minimum noise (`sigma_min`). The start value must be larger.
* **"What's the difference between the KSampler's `steps` and this node's `steps`?"**
    * This node's `steps` input defines the "resolution" of the schedule *before* slicing. The KSampler's `steps` input is how many steps it will actually execute. They should be the same for txt2img, but for img2img, this node's `steps` might be higher to allow for more granular slicing. Always connect this node's `actual_steps` output to the KSampler's `steps` input for best results.
