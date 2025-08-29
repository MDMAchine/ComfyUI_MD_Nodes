# Comprehensive Manual: PingPong Sampler (Lite+ V0.8.20)

Welcome to the complete guide for the **PingPong Sampler (Lite+ V0.8.20)**, a powerful and intuitive custom sampler for ComfyUI. This manual provides everything you need to know, from basic setup to advanced techniques and technical details.

---

### **Table of Contents**

1.  **Introduction**
    * What is the PingPong Sampler?
    * Who is this Sampler For?
    * Key Features in Version "Lite+"
2.  **Installation**
3.  **Core Concepts: The "Ping-Pong" Method**
    * Ancestral Noise Injection
    * Controlling the Noise: Strength and Coherence
4.  **Node Setup and Workflow**
    * How to Use It: A Step-by-Step Guide
5.  **Parameter Deep Dive**
    * Primary Controls: Noise Behavior
    * Timing Controls: Ancestral Steps
    * Randomness Controls: Seed and Mode
    * Advanced / Custom Controls
    * Standard Sampler Inputs
6.  **Practical Recipes & Use Cases**
    * Recipe 1: Smooth, Cinematic Video
    * Recipe 2: Glitchy, Energetic Audio Visualizer
    * Recipe 3: Classic Film Grain for Still Images
7.  **Advanced Usage: YAML Overrides**
8.  **Technical Deep Dive**
    * The `__call__` Function: Core Logic
    * How Strength and Coherence are Implemented
9.  **Troubleshooting & FAQ**

---

## 1. Introduction

### What is the PingPong Sampler?

The PingPong Sampler is a specialized sampling node for ComfyUI, optimized for generating high-quality audio and video with Ace-Step diffusion models. Unlike standard samplers that follow a linear denoising path, it employs a "ping-pong" strategy—alternating between denoising the latent and injecting carefully controlled ancestral noise. This "Lite+" version introduces powerful but simple controls over the **magnitude (strength)** and **evolution (coherence)** of that noise, unlocking a vast range of artistic effects.

### Who is this Sampler For?

* **Video and Audio Creators:** Its primary audience. The controls for noise coherence are designed to manage temporal effects, from smooth transitions to chaotic energy.
* **Creative Coders & Experimental Artists:** Anyone looking for unique textures and behaviors not found in standard samplers.
* **Still Image Generators:** The noise controls can be used to create beautiful, consistent textures like film grain or atmospheric effects.

### Key Features in Version "Lite+"

* **Intuitive Noise Behavior Presets:** A single dropdown menu to instantly select different noise styles, from smooth to chaotic.
* **Ancestral Strength Control:** A powerful dial to control the *magnitude* of the injected noise, blending between a smooth, DDIM-like output and a highly textured one.
* **Noise Coherence Control:** A unique parameter to control the *evolution* of noise over time, enabling effects from flickering static to flowing, organic patterns.
* **Full Backwards Compatibility:** The "Default (Raw)" preset is 100% identical to the original v0.8.15 sampler.
* **Simple UI with Advanced Options:** The interface remains clean and simple, with advanced sliders hidden until you need them.

---

## 2. Installation

As a custom node, the PingPong Sampler needs to be placed in the correct ComfyUI directory.

1.  Navigate to your ComfyUI installation folder.
2.  Open the `ComfyUI/custom_nodes/` directory.
3.  Save the `PingPongSampler_Lite.py` file inside this directory.
4.  Restart ComfyUI. The "PingPong Sampler (Lite+ V0.8.20)" will now be available when you search for nodes.

---

## 3. Core Concepts: The "Ping-Pong" Method

### Ancestral Noise Injection

At its heart, the sampler performs two actions in a loop:

1.  **Denoise:** It asks the model to predict a cleaner version of your image/video frame.
2.  **Inject Noise (The "Ping-Pong"):** It then strategically adds a small amount of new, random noise back in.

This prevents the output from becoming overly smooth or "baked" and is the key to creating rich textures and details. The "Lite+" version gives you unprecedented control over step #2.

### Controlling the Noise: Strength and Coherence

Imagine the ancestral noise is like TV static being overlaid on your image at each step.

* **Ancestral Strength** is the **volume** of that static. A high value makes the static very visible and influential, creating a raw, energetic texture. A low value makes it a faint whisper, resulting in a much smoother image.
* **Noise Coherence** is how **quickly the static pattern changes**.
    * **Low Coherence (0.0):** The static pattern is completely random and different on every single frame. This creates a chaotic, flickering, high-energy effect.
    * **High Coherence (1.0):** The *exact same* static pattern is used on every frame. This creates a fixed-pattern noise, like a stable film grain or a texture overlay that doesn't change over time.
    * **Mid-range Coherence (0.5):** The static pattern smoothly evolves, with 50% of the pattern carrying over from the previous frame. This is perfect for organic, flowing effects.

---

## 4. Node Setup and Workflow

### How to Use It: A Step-by-Step Guide

1.  **Replace your KSampler:** The PingPong Sampler is a direct replacement for your existing `KSampler` or `SamplerCustom` nodes.
2.  **Connect Core Inputs:** Connect your `model`, `positive`, `negative`, `latent`, and `scheduler` as you normally would.
3.  **Select a Noise Behavior (New!):** This is your primary creative control. Start with a preset from the `noise_behavior` dropdown. "Default (Raw)" is the original, and "Smooth" is great for video.
4.  **Adjust Timing:** Use `first_ancestral_step` and `last_ancestral_step` to define *when* the ping-pong effect is active during the sampling process. It's often best to stop it before the final steps (e.g., set `last_ancestral_step` to 18 on a 20-step sample).
5.  **Fine-Tune (Optional):** If you need more specific control, set `noise_behavior` to "Custom". This will activate the `ancestral_strength` and `noise_coherence` sliders for manual adjustment.
6.  **Connect Latent Output:** Connect the `LATENT` output to your VAE Decode node.

---

## 5. Parameter Deep Dive

### Primary Controls: Noise Behavior

* **`noise_behavior`** (`ENUM`, default: `Default (Raw)`)
    * **What it is**: A simple dropdown menu to select a preset for the ancestral noise characteristics.
    * **Presets & Use Cases**:
        * `Default (Raw)`: The original v0.8.15 behavior. Energetic and chaotic. Best for highly textured still images or glitchy effects. (Strength: 1.0, Coherence: 0.0)
        * `Dynamic`: High energy, but with a subtle flow between steps. A good starting point for dynamic video where you want texture but not full chaos. (Strength: 1.0, Coherence: 0.25)
        * `Smooth`: Reduced noise strength and smooth evolution. **The recommended preset for clean, cinematic video.** (Strength: 0.8, Coherence: 0.5)
        * `Textured Grain`: A highly consistent noise pattern. **Excellent for adding a stable film grain effect to videos or still images.** (Strength: 0.9, Coherence: 0.9)
        * `Soft (DDIM-Like)`: Very little ancestral noise. Use this if you want a very clean, smooth output that is close to a non-ancestral sampler. (Strength: 0.2, Coherence: 0.0)

### Timing Controls: Ancestral Steps

* **`first_ancestral_step`** (`INT`, default: `0`)
    * The step index (0-based) where noise injection *begins*. Leave at 0 to apply the effect from the start.
* **`last_ancestral_step`** (`INT`, default: `-1`)
    * The step index where noise injection *ends*. A value of `-1` means it continues to the final step. **It is highly recommended to set this to a few steps below your total step count** (e.g., 18 for 20 steps) to ensure a clean final output.

### Randomness Controls: Seed and Mode

* **`step_random_mode`** (`ENUM`, default: `block`)
    * Controls how the RNG seed for the noise varies at each step. `block` is great for creating structured variations, while `step` or `reset` can create more dynamic randomness.
* **`step_size`** (`INT`, default: `4`)
    * The interval used by the `block` and `reset` random modes.
* **`seed`** (`INT`, Required): The master random seed for reproducibility.

### Advanced / Custom Controls

These sliders are **only active when `noise_behavior` is set to "Custom"**.

* **`ancestral_strength`** (`FLOAT`, default: `1.0`, Optional)
    * The magnitude of the injected noise, from 0.0 (none) to 1.0 (full) and beyond. Allows you to dial in the exact amount of texture.
* **`noise_coherence`** (`FLOAT`, default: `0.0`, Optional)
    * Controls how much of the previous step's noise is blended into the current step's noise. 0.0 is fully random, 1.0 re-uses the same noise pattern.

### Standard Sampler Inputs

* **`start_sigma_index`** / **`end_sigma_index`**: Control the start/end point within the scheduler's sigma steps.
* **`enable_clamp_output`**: Clamps the final output latent to the range `[-1.0, 1.0]`.
* **`scheduler`**: The noise schedule used by the sampler (e.g., from a KarrasScheduler node).

---

## 6. Practical Recipes & Use Cases

### Recipe 1: Smooth, Cinematic Video

Goal: Create clean, smoothly evolving video with minimal flickering.

* **`noise_behavior`**: `Smooth`
* **`last_ancestral_step`**: `[Total Steps - 2]` (e.g., `18` if you have 20 steps)
* **`step_random_mode`**: `block`
* **`step_size`**: `1` or `2`

### Recipe 2: Glitchy, Energetic Audio Visualizer

Goal: Create chaotic, flickering noise that reacts wildly.

* **`noise_behavior`**: `Default (Raw)` or `Dynamic`
* **`last_ancestral_step`**: `[Total Steps]` (Let it run to the end for maximum effect)
* **`step_random_mode`**: `reset` or `step`

### Recipe 3: Classic Film Grain for Still Images

Goal: Add a stable, consistent grain texture to a still image or video.

* **`noise_behavior`**: `Textured Grain`
* **`last_ancestral_step`**: `[Total Steps - 2]`
* **`step_random_mode`**: `off` (This is key to ensure the noise pattern is identical on every step)

---

## 7. Advanced Usage: YAML Overrides

The `yaml_settings_str` input is a powerful feature for saving and loading presets. Any parameter defined in the YAML string will **override** the UI controls.

**Example YAML for a custom "Smooth" preset:**
```yaml
# My Custom Smooth Video Settings
noise_behavior: Custom
ancestral_strength: 0.75
noise_coherence: 0.6
step_random_mode: block
step_size: 2
last_ancestral_step: 15
```

---

## 8. Technical Deep Dive

### The `__call__` Function: Core Logic

The main sampling loop has been updated to incorporate strength and coherence.

1.  **State Management**: The sampler now maintains a `self.previous_noise` state variable to store the noise from the last step.
2.  **Coherence Blending**: If `noise_coherence > 0`, the noise for the current step is a `torch.lerp` (linear interpolation) between brand new random noise and `self.previous_noise`.
3.  **Strength Application**: Instead of simply adding the final noise, the sampler calculates the result *with* full noise and the result *without* any noise (the clean `denoised_sample`). It then uses `torch.lerp` to blend between these two states using `ancestral_strength` as the factor. This provides a smooth and stable way to control the noise magnitude.

---

## 9. Troubleshooting & FAQ

* **"My output is too noisy or chaotic."**
    * Lower the `ancestral_strength` (via a preset like "Soft" or "Custom").
    * Lower the `last_ancestral_step` to stop noise injection earlier.
* **"My video feels static or has a fixed pattern."**
    * Lower the `noise_coherence`. A high coherence value re-uses the same noise pattern.
* **"The `ancestral_strength` and `noise_coherence` sliders aren't doing anything."**
    * You must set the `noise_behavior` dropdown to **"Custom"** to activate the manual sliders.
* **"What's the difference between this and the FBG version?"**
    * This "Lite+" version focuses on controlling the *character of the ancestral noise*. The FBG (Feedback Guidance) version is a much more complex sampler that focuses on *dynamically adjusting the CFG guidance scale* based on the model's confidence. They solve different problems.
