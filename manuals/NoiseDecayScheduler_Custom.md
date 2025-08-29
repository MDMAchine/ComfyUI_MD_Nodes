# Comprehensive Manual: Advanced Noise Decay Scheduler (v0.6.0)

Welcome to the complete guide for the **Advanced Noise Decay Scheduler (v0.6.0)**, an elite toolkit for designing custom noise schedules in ComfyUI. This manual provides everything you need to know, from foundational concepts to advanced curve-shaping techniques and technical details.

---

### **Table of Contents**

1.  **Introduction**
    * What is the Advanced Noise Decay Scheduler?
    * Who is this Node For?
    * Key Features in Version 0.6.0
2.  **Installation**
3.  **Core Concepts: The Art of Decay**
    * The Noise Schedule: Your Generation's Blueprint
    * Shaping the Curve: From Vibe to Visuals
4.  **Node Setup and Workflow**
    * How to Use It: A Step-by-Step Guide
5.  **Parameter Deep Dive**
    * Primary Controls: Algorithm Selection
    * Curve Manipulation Controls
    * Algorithm-Specific Controls
    * Performance & Quality Controls
6.  **Practical Recipes & Use Cases**
    * Recipe 1: The Cliff Drop
    * Recipe 2: The Gaussian Detail Push
    * Recipe 3: The Rhythmic Pulse
7.  **Technical Deep Dive**
    * The Order of Operations
8.  **Troubleshooting & FAQ**

---

## 1. Introduction

### What is the Advanced Noise Decay Scheduler?

The **Advanced Noise Decay Scheduler** is a specialized "curve designer" node for ComfyUI. It doesn't perform sampling itself; instead, it generates a highly customized mathematical curve that tells a compatible sampler *how* to remove noise at every step of the generation process. It's the ultimate tool for overriding default sampler behavior and dictating the precise "vibe" and texture of your output.

### Who is this Node For?

* **Experimental Artists & Creative Coders:** Anyone who wants to break free from standard sampler looks and create truly unique, signature visual styles.
* **Workflow Architects:** Tinkerers who build complex, non-standard workflows and require granular control over the diffusion process.
* **Animation and Video Specialists:** Users who need to create specific temporal effects, like pulsing, breathing, or strobing textures that evolve over time.

### Key Features in Version 0.6.0

* **Six Unique Algorithms:** Go beyond simple curves with `Polynomial`, `Sigmoidal`, `Piecewise`, `Fourier`, `Exponential`, and `Gaussian` functions.
* **Total Curve Control:** Manipulate your curve with precision using `start/end` value overrides and a powerful `invert_curve` toggle for radical effects.
* **Configurable Smoothing:** A built-in moving average filter with a configurable window to soften sharp transitions for cleaner, more organic results.
* **Intelligent Caching:** Automatically saves computed curves, dramatically speeding up iterative work and reruns with the same settings.
* **Designed for Custom Samplers:** The perfect companion for nodes like `pingpongsampler_custom` that accept an external scheduler input.

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

## 3. Core Concepts: The Art of Decay

### The Noise Schedule: Your Generation's Blueprint

Think of the image generation process as a sculptor carving a statue from a block of marble. The noise is the marble, and the final image is the statue. A standard scheduler is like giving the sculptor one type of chisel. This node is like giving them a full set of custom-forged tools.

The **decay curve** is the blueprint for this process. It dictates the energy and focus at each step:
* A **steep initial drop** (like a cliff) carves out the main shapes immediately.
* A **long, flat tail** spends more time refining fine details and textures.
* A **wavy, oscillating curve** tells the process to add and remove energy rhythmically, creating pulsing effects.



### Shaping the Curve: From Vibe to Visuals

By designing your own curve, you are no longer just a passenger in the generation process; you are the pilot.

* **Control Texture:** A slow, gentle decay can produce soft, painterly textures. A noisy, chaotic curve can create gritty, raw, or glitchy effects.
* **Direct the Focus:** Use a Gaussian curve to force the model to hold onto noise for longer, then resolve details rapidly in the middle steps.
* **Create Motion:** In animations, an oscillating Fourier curve can make textures feel like they are breathing or pulsing in time with your visuals.

---

## 4. Node Setup and Workflow

### How to Use It: A Step-by-Step Guide

This node acts as a "plugin" for a sampler. It doesn't replace your KSampler.

1.  **Add the Scheduler Node:** Add the `Noise Decay Scheduler (Advanced)` node to your workflow.
2.  **Design Your Curve:** This is the creative step.
    * Choose your base `algorithm_type`. `Polynomial` is a great starting point.
    * Adjust the algorithm-specific controls (like `decay_exponent`) to shape the curve.
    * Fine-tune with the manipulation controls (`start_value`, `invert_curve`, etc.).
3.  **Connect to a Custom Sampler:** Drag the `scheduler` output noodle from this node and plug it into the `scheduler` input of a compatible sampler (e.g., `pingpongsampler_custom` or other samplers that expose this input).
4.  **Connect Sampler Outputs:** Connect your sampler's `LATENT` output to your VAE Decode as usual.
5.  **Queue Prompt:** The sampler will now ignore its internal scheduler and follow the blueprint you designed.

---

## 5. Parameter Deep Dive

### Primary Controls: Algorithm Selection

* **`algorithm_type`** (`ENUM`): Selects the core mathematical function for the curve's shape.
    * `polynomial`: A powerful, aggressive curve. The workhorse.
    * `sigmoidal`: An S-curve. Gentle start and end, fast middle. Very "cinematic."
    * `piecewise`: Full manual control. Define the curve point-by-point.
    * `fourier`: A cosine wave for rhythmic, oscillating effects.
    * `exponential`: A smooth, natural decay. Less harsh than polynomial.
    * `gaussian`: Inverted bell curve. Holds noise high, drops fast, holds low. Great for a mid-point detail push.

### Curve Manipulation Controls

* **`start_value`** / **`end_value`** (`FLOAT`, default: `1.0`/`0.0`): Overrides the start and end points of the curve. You can make it decay from 0.8 to 0.1, for example, to avoid pure black or white noise levels.
* **`invert_curve`** (`BOOLEAN`): Flips the curve vertically. A decay becomes a growth curve. An experimental tool for effects like scheduled noise *injection*.

### Algorithm-Specific Controls

* **`decay_exponent`** (`FLOAT`): Controls the steepness/shape for `polynomial`, `sigmoidal`, and `exponential` curves. Higher values = more dramatic curves.
* **`custom_piecewise_points`** (`STRING`): A comma-separated list of values for the `piecewise` algorithm (e.g., `"1.0,0.9,0.2,0.0"`).
* **`fourier_frequency`** (`FLOAT`): Sets the number of waves for the `fourier` algorithm. `2.0` creates twice as many waves as `1.0`.

### Performance & Quality Controls

* **`use_caching`** (`BOOLEAN`): Enables/disables caching. Keep this on for faster iterations.
* **`enable_temporal_smoothing`** (`BOOLEAN`): Applies a smoothing filter to the curve, ideal for preventing jitter in animations.
* **`smoothing_window`** (`INT`): The strength of the smoothing effect. A larger window creates a smoother, less detailed curve.

---

## 6. Practical Recipes & Use Cases

### Recipe 1: The Cliff Drop

Goal: Define the composition very early and spend the rest of the time refining.

* **`algorithm_type`**: `polynomial`
* **`decay_exponent`**: `5.0` (or higher)
* **`enable_temporal_smoothing`**: `True`
* **`smoothing_window`**: `3` (to slightly soften the sharp "knee" of the curve)

### Recipe 2: The Gaussian Detail Push

Goal: Let the image remain noisy and undefined for longer, then force a rapid clarification of details in the middle steps.

* **`algorithm_type`**: `gaussian`
* **`decay_exponent`**: `2.5` (a lower value makes the central drop wider)

### Recipe 3: The Rhythmic Pulse

Goal: Create a "breathing" texture in an animation or a unique layered look in a still image.

* **`algorithm_type`**: `fourier`
* **`fourier_frequency`**: `3.0` (for three full pulses)
* **`start_value`**: `0.9`
* **`end_value`**: `0.1` (to prevent the noise from ever reaching 100% or 0%)

---

## 7. Technical Deep Dive

### The Order of Operations

To master this node, understand its internal processing chain. When a sampler requests the curve from `get_decay`, the node performs these steps in a strict sequence:

1.  **Generate Base Curve:** The selected algorithm (`polynomial`, `gaussian`, etc.) is used to create a "normalized" curve. This base curve is always calculated to run from a high value (like 1.0) down to a low one (like 0.0).
2.  **Apply Smoothing:** If `enable_temporal_smoothing` is on, this base curve is passed through the moving average filter, softening its shape.
3.  **Apply Inversion:** If `invert_curve` is checked, the curve is flipped vertically (`new_value = 1.0 - old_value`).
4.  **Rescale to Final Values:** This is the last step. The resulting curve is mathematically stretched and shifted to fit the exact `start_value` and `end_value` defined by the user.

This predictable order ensures that controls like `start_value` are the final word on the curve's range, no matter how complex the underlying shape is.

---

## 8. Troubleshooting & FAQ

* **"My output looks like a mess."**
    * Welcome to the bleeding edge! This is an experimental tool, and extreme settings will produce extreme, often non-photorealistic results. This is a feature, not a bug. Start with mild settings and work your way up.
* **"I'm moving the `decay_exponent` slider, but nothing is changing."**
    * That parameter is only active for the `polynomial`, `sigmoidal`, and `exponential` algorithms. It has no effect on the other modes. Check the "Algorithm-Specific Controls" section for which slider controls which mode.
* **"How is this different from the `KarrasScheduler` or `SimpleScheduler` nodes?"**
    * Those are complete, self-contained schedulers with their own internal, fixed curves. This node **is not a sampler or a full scheduler.** It's a "curve factory" that *generates* a schedule object. It does nothing on its own and must be plugged into a custom sampler that is designed to accept an external schedule.