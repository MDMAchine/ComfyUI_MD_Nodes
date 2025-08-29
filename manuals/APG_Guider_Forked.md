# Comprehensive Manual: APG Guider (Forked) v0.2.0

Welcome to the complete guide for the **APG Guider (Forked) v0.2.0**, an advanced guidance controller for ComfyUI. This manual covers everything from the core concepts of projected guidance to practical recipes that will give you surgical control over your generations.

---

### **Table of Contents**

1.  **Introduction**
    * What is the APG Guider?
    * Who is this Node For?
    * Key Features
2.  **Installation**
3.  **Core Concepts: Beyond Standard CFG**
    * CFG: The Scaled Difference
    * APG: The Orthogonal Steer
    * Momentum: The Smoothing & Sharpening Engine
    * Scheduling with Sigma: Guiding Through the Fog
4.  **Node Setup and Workflow**
    * How to Use It: A Step-by-Step Guide
5.  **Parameter Deep Dive**
    * Primary Connections
    * Master Controls
    * Main Guidance Controls
    * Scheduling Controls
    * Advanced Dynamics
    * Technical & YAML Controls
6.  **Practical Recipes & Use Cases**
    * Recipe 1: The "Crisp & Detailed" Look
    * Recipe 2: The "Smooth & Painterly" Look
    * Recipe 3: Advanced Audio Guidance
7.  **Technical Deep Dive**
    * The Guidance Process
8.  **Troubleshooting & FAQ**

---

## 1. 🧠 Introduction

### What is the APG Guider?

The **APG Guider (Forked)** is a custom sampler guider that offers a powerful alternative to ComfyUI's standard Classifier-Free Guidance (CFG). While standard CFG simply pushes the generation towards a prompt, Adaptive Projected Gradient (APG) guidance acts like a rudder, steering it in more nuanced and controlled ways. It analyzes the guidance vector and uses its **orthogonal** (perpendicular) component to add detail and influence the generation's style without simply "shouting" the prompt louder. It's the difference between a hammer and a scalpel.

### Who is this Node For?

* **Advanced AI Artists:** Users who are comfortable with CFG but want a deeper level of control over texture, detail, and coherence.
* **Technical Experimenters:** Anyone who enjoys diving into the mechanics of diffusion and manipulating the latent space with precision.
* **Audio Alchemists:** Those working with audio diffusion models who need stable, responsive guidance to shape timbre and character.
* **Workflow Tinkerers:** Users who want to build complex, multi-stage guidance schedules using the powerful YAML input.

### Key Features

* **Advanced Orthogonal Guidance:** The core APG mechanism provides a unique method of steering the diffusion process.
* **Full Guidance Scheduling:** Control every parameter at different stages of the generation based on the current noise level (`sigma`).
* **Dynamic Momentum Engine:** Use **positive** momentum to smooth guidance for a painterly feel, or **negative** momentum to sharpen details and create crisp results.
* **Hybrid CFG/APG System:** Seamlessly switch between standard CFG and APG guidance at different points in the sampling process.
* **Full YAML Control:** Bypass the node's UI and define complex, multi-rule guidance schedules for ultimate power and portability.
* **Built-in Debugging:** A verbose toggle prints detailed step-by-step information to the console to help you understand exactly how your rules are being applied.

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

## 3. ⚙️ Core Concepts: Beyond Standard CFG

### CFG: The Scaled Difference

Standard Classifier-Free Guidance is simple and effective. It calculates the difference between what the model predicts for your prompt (`cond`) and what it predicts for a blank prompt (`uncond`). It then applies this difference vector, scaled by the CFG value, to push the generation toward your prompt. It's a direct, brute-force push.

### APG: The Orthogonal Steer



Adaptive Projected Gradient guidance is more elegant. It also calculates the difference vector, but then it **projects** it to find the component that is **orthogonal** (perpendicular) to the `cond` prediction.

Think of it like a rocket (`cond` prediction) with a main engine pushing it forward. Standard CFG just cranks up the main engine. APG adds side-thrusters (the orthogonal vector) that can steer the rocket and add complex maneuvers without just making it go faster in a straight line. This allows it to add details and stylistic flair that aren't just a stronger version of the prompt.

### Momentum: The Smoothing & Sharpening Engine

Momentum applies a running average to the guidance vector, influencing how it behaves over multiple steps.

* **Positive Momentum (e.g., `0.75`):** This smooths the guidance, like running a sponge over a rough surface. It averages the current step's guidance with previous steps, leading to more coherent, blended, and often painterly results. It's great for stability.
* **Negative Momentum (e.g., `-0.75`):** This sharpens the guidance, like using a chisel. It actively pushes away from the previous step's average, making the guidance more responsive and decisive. This is excellent for enhancing fine details, creating crisp textures, and achieving a sharper final look.

### Scheduling with Sigma: Guiding Through the Fog

`Sigma` represents the amount of noise in the image at any given step. High `sigma` means the image is mostly noise (the beginning), and low `sigma` means it's almost finished (the end). The APG Guider allows you to create rules that activate at different `sigma` thresholds, letting you apply different guidance strategies as the image emerges from the noise. For example, you can use strong, broad guidance at the start and switch to fine-tuned, detail-oriented guidance at the end.

---

## 4. Node Setup and Workflow

### How to Use It: A Step-by-Step Guide

1.  **Connect Core Components:** Connect your `MODEL`, `positive` conditioning, and `negative` conditioning to the corresponding inputs on the APG Guider node.
2.  **Connect to Sampler:** Connect the `GUIDER` output to the `guider` input on a **Custom Sampler** node (e.g., SamplerCustom, SamplerDPMAdaptative). This node does **not** work with standard samplers.
3.  **Choose a Mode:** For the simplest setup, leave the node's settings as they are. This will apply a basic APG guidance.
4.  **Set Basic Parameters:**
    * Adjust `apg_scale` to control the strength of the effect.
    * Adjust `momentum` to control the character (smooth vs. sharp).
    * Use `start_sigma` and `end_sigma` to define when the APG effect is active.
5.  **For Advanced Control (YAML):**
    * Clear the `start_sigma`/`end_sigma` fields (set to `-1`).
    * Paste a YAML configuration (see recipes below) into the `yaml_parameters_opt` box. This will override all other settings.
6.  **Queue Prompt:** Run the generation. If `verbose_debug` is on, check your console to see the rules being applied at each step.

---

## 5. Parameter Deep Dive

### Primary Connections

* **`model`**: The diffusion model to be guided.
* **`positive`**: The positive conditioning from your text prompt.
* **`negative`**: The negative conditioning.

### Master Controls

* **`disable_apg`** (`BOOLEAN`): A master switch. If `True`, the node is completely disabled and acts like a standard CFG guider using only the `cfg_after` value.
* **`verbose_debug`** (`BOOLEAN`): If `True`, prints detailed information about which rule is active at each sigma step to the console. Invaluable for debugging your YAML schedules.

### Main Guidance Controls

* **`apg_scale`** (`FLOAT`): The strength of the APG effect. `1.0` is neutral. Higher values apply stronger orthogonal guidance. Can be a good idea to keep this and `cfg` values in a similar range.
* **`mode`** (`ENUM`): The core guidance algorithm.
    * `pure_apg`: Applies APG guidance directly during the guidance step. The most common and stable mode.
    * `pre_cfg`: Modifies the conditioning *before* it's sent to the sampler's CFG step. More experimental.
    * `alt1`/`alt2`: Alternative momentum calculation modes that can produce unique results.

### Scheduling Controls

* **`start_sigma`** (`FLOAT`): The noise level at which APG guidance begins. A value of `-1` means it's active from the very start (infinity sigma).
* **`end_sigma`** (`FLOAT`): The noise level at which APG guidance ends. After this point, the guider will revert to using `cfg_after`. A value of `-1` means it never ends.
* **`cfg_before`** (`FLOAT`): The standard CFG value to use *before* `start_sigma` is reached.
* **`cfg_after`** (`FLOAT`): The standard CFG value to use *after* `end_sigma` is reached (or if APG is disabled).

### Advanced Dynamics

* **`momentum`** (`FLOAT`): Controls the guidance character. Positive values smooth; negative values sharpen. `0` disables momentum.
* **`norm_threshold`** (`FLOAT`): A safety valve. If the magnitude of the guidance vector exceeds this value, it will be scaled down. Prevents overly strong guidance that can lead to artifacts.
* **`eta`** (`FLOAT`): An advanced parameter for re-introducing some of the parallel guidance component. `0.0` is the standard for pure orthogonal guidance.

### Technical & YAML Controls

* **`dims`** (`STRING`): Comma-separated list of tensor dimensions to normalize over. For images, `"-1, -2"` (height, width) is standard. For audio, `"-1"` (frequency) is often used.
* **`predict_image`** (`BOOLEAN`): If `True`, guides the denoised image prediction (`x0`). If `False`, guides the noise prediction (`epsilon`). `True` is generally recommended for better results.
* **`yaml_parameters_opt`** (`STRING`): A multiline text field to input a YAML configuration that overrides all other settings. This is the most powerful feature for advanced users.

---

## 6. 🧪 Practical Recipes & Use Cases

### Recipe 1: The "Crisp & Detailed" Look

Goal: Generate a sharp, highly detailed image, excellent for intricate subjects.

```yaml
# Crisp & Detailed Look
verbose: false
rules:
  - start_sigma: -1.0
    apg_scale: 0.0
    cfg: 7.0
  - start_sigma: 14.0
    apg_scale: 6.5
    predict_image: true
    cfg: 6.5
    mode: pure_apg
    momentum: -0.85      # Strong negative momentum for sharpness
    norm_threshold: 2.0
  - start_sigma: 1.5
    apg_scale: 0.0
    cfg: 5.0             # Lower CFG for final clean-up
```

### Recipe 2: The "Smooth & Painterly" Look

Goal: Generate an image with soft, blended details and a more organic, painterly feel.

```yaml
# Smooth & Painterly Look
verbose: false
rules:
  - start_sigma: -1.0
    apg_scale: 0.0
    cfg: 5.0
  - start_sigma: 10.0
    apg_scale: 4.5
    predict_image: true
    cfg: 4.0
    mode: pure_apg
    momentum: 0.9      # Strong positive momentum for smoothing
    norm_threshold: 3.5
  - start_sigma: 2.0
    apg_scale: 0.0
    cfg: 3.0
```

### Recipe 3: Advanced Audio Guidance

Goal: Guide an audio diffusion model with stability, shaping the timbre without creating artifacts. Best used with a `PingPongSampler`.

```yaml
# Stable Audio Guidance
verbose: false
rules:
  - start_sigma: -1.0
    apg_scale: 0.0
    cfg: 5.0
  - start_sigma: 8.0
    apg_scale: 4.5
    predict_image: true
    cfg: 4.0
    mode: pure_apg
    dims: [-1]             # CRITICAL: Normalize over frequency dimension
    momentum: 0.6          # Gentle positive momentum for stability
    norm_threshold: 2.5
  - start_sigma: 2.0
    apg_scale: 0.0
    cfg: 3.0
```

---

## 7. Technical Deep Dive

### The Guidance Process

When a custom sampler calls the APG Guider at each step, the following sequence occurs:

1.  **Get Sigma:** The guider receives the current `timestep` and calculates the corresponding `sigma` (noise level).
2.  **Find Rule:** It iterates through its list of rules (sorted by `start_sigma`) and finds the first rule that matches the current `sigma`.
3.  **Reset State:** It calls the `reset()` method on all *other* rules to clear their momentum, preventing state from one rule from leaking into the next.
4.  **Inject Function:** If the matched rule has an active `apg_scale`, the guider injects its custom guidance function (`rule.cfg_function` or `rule.pre_cfg_function`) into the model options for the current step.
5.  **Set CFG:** It sets the sampler's CFG value to the `cfg` value specified in the matched rule.
6.  **Execute:** It then calls the original `predict_noise` function, which now executes with the injected APG logic and the correct CFG for that specific step.

This process repeats for every step of the generation, allowing for a dynamic and highly controllable guidance schedule.

---

## 8. Troubleshooting & FAQ

* **"My images look overly noisy, strange, or chaotic."**
    * Your guidance is likely too strong. Try lowering `apg_scale` and/or `momentum` (closer to 0). Also, check your `norm_threshold`; a lower value like `2.0` can help tame extreme guidance.
* **"The node doesn't seem to be doing anything."**
    * Ensure `disable_apg` is `False`. Check that your `apg_scale` is not `0.0` during the steps you want it to be active. Make sure your `start_sigma` is high enough to be triggered early in the sampling process.
* **"I'm getting a 'Could not get APG rule' error in the console."**
    * This means no rule matched the current sigma. The node automatically adds a fallback rule to prevent this, but if you are using a very old version or have a strange YAML, ensure you have a rule that covers all sigma values, typically one with `start_sigma: -1.0`.
* **"What's the real difference between `pure_apg` and `pre_cfg` modes?"**
    * `pure_apg` integrates into the sampler's guidance calculation, which is generally more stable. `pre_cfg` modifies the conditioning tensors *before* they even get to the sampler, which can be more powerful but also more unpredictable. Stick with `pure_apg` unless you are specifically experimenting.
* **"My YAML config isn't working or gives an error."**
    * YAML is very sensitive to indentation. Make sure your `rules:` list and all parameters within each rule are indented correctly with spaces, not tabs. Use the recipes above as a template. Turn on `verbose_debug` to see if your rules are being loaded correctly.
