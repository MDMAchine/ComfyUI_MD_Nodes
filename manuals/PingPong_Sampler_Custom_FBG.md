# Comprehensive Manual: PingPong Sampler (Custom v0.9.9-p2 FBG)

Welcome to the complete guide for the **PingPong Sampler (Custom v0.9.9-p2 FBG)**, a specialized ComfyUI node designed for advanced, dynamic control over the diffusion sampling process.

Originally optimized for Ace-Step audio and video models, its powerful feature set makes it an exceptional tool for experimental image generation as well. This manual provides everything you need to know, from basic setup to the masterclass in tuning its powerful Feedback Guidance engine.

---

### **Table of Contents**

1.  **Introduction**
    * What is the PingPong Sampler?
    * Who is this Node For?
    * Key Features in Version 0.9.9-p2
2.  **Installation**
3.  **Core Concepts: The Art of Dynamic Sampling**
    * "Ping-Pong" Ancestral Noise: Creative Chaos
    * Feedback Guidance (FBG): The Intelligent Autopilot
    * Conditional Blending: "If/Then" Rules for Creativity
4.  **Node Setup and Workflow**
    * How to Use It: A Step-by-Step Guide
5.  **Parameter Deep Dive**
    * Ping-Pong Specific Parameters
    * Feedback Guidance (FBG) & Enhancement Parameters
    * Advanced (YAML) Override
6.  **A Masterclass in Tuning Feedback Guidance**
    * The Golden Rule: Disabling Autopilot
    * The Troubleshooting Playbook
    * Symptom: Guidance Explodes to Max Value Instantly
    * Symptom: Guidance is Stable but "Stuck"
    * Reading the Signs: Anatomy of a Perfect Log
7.  **Practical Recipes & Use Cases**
    * Recipe 1: Stable & Controlled Image Generation
    * Recipe 2: Aggressive & Experimental Audio Generation
    * Recipe 3: The "Slow Burn" Finishing Kick
8.  **Technical Deep Dive**
    * The Order of Operations

---

## 1. Introduction

### What is the PingPong Sampler?

The **PingPong Sampler (FBG)** is not a standard sampler; it is a *sampler provider*. It acts as an advanced "brain" that plugs into a standard KSampler node, hijacking its internal logic to provide a far more dynamic and controllable generation process. It enhances the denoising process in two key ways:

* **"Ping-Pong" Ancestral Noise Injection:** It employs a strategy of intelligently alternating between denoising your latent data and strategically re-injecting controlled ancestral noise. This is particularly beneficial for maintaining coherence in time-series data (like audio or video frames) and can introduce rich textures in static images.

* **Feedback Guidance (FBG):** This is the core innovation. Instead of a fixed guidance scale (CFG), FBG acts like an adaptive cruise control, dynamically adjusting the guidance strength based on the model's "certainty" at each step.

### Who is this Node For?

* **AI Musicians & Sound Designers:** Anyone generating audio who needs to push their models to the creative limit and achieve unique sonic textures.
* **Experimental Visual Artists:** Users who want to move beyond the "feel" of standard samplers and explore chaotic, highly-textured, or uniquely detailed image styles.
* **Workflow Tinkerers & Power Users:** Anyone who loves having deep control over the sampling process and wants to fine-tune every aspect of the generation.

### Key Features in Version 0.9.9-p2

* **Dynamic Feedback Guidance (FBG):** The intelligent, self-regulating guidance system that adapts to your model on the fly.
* **Advanced Ancestral Noise Control:** Go beyond simple noise injection with selectable noise types (`gaussian`, `uniform`, `brownian`) and precise scheduling.
* **Full Tuning Control:** A comprehensive suite of parameters allows you to switch between a powerful "Autopilot" mode and a precise "Manual" mode for FBG.
* **Conditional Blending:** Create complex "if/then" rules to change how the sampler behaves at different stages of the generation.
* **Multi-Level Debug Mode:** Get unparalleled insight into the sampler's internal state with a verbose console log to perfect your settings.
* **Full YAML Preset Support:** Define and save every single parameter in a portable YAML string for perfect reproducibility.

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

## 3. Core Concepts: The Art of Dynamic Sampling

### "Ping-Pong" Ancestral Noise: Creative Chaos

Standard samplers only remove noise. An ancestral sampler, like this one, re-injects a small, controlled amount of new noise at certain steps. This "ping-pong" between denoising and re-noising prevents the image from becoming overly smooth and can introduce beautiful, complex textures. You have full control over when this happens (`first/last_ancestral_step`) and what kind of noise is used (`ancestral_noise_type`).

### Feedback Guidance (FBG): The Intelligent Autopilot

Think of standard CFG as a simple cruise control—you set it to 7 and it stays there. FBG is an **adaptive cruise control**. It constantly measures the difference between what the model wants to do (unconditional) and what you told it to do (conditional).

* If the model is on the right track, FBG might lower the guidance to allow for more nuance.
* If the model starts to stray, FBG increases the guidance to steer it back.

This dynamic adjustment can lead to more detailed and coherent results than a fixed CFG value.

### Conditional Blending: "If/Then" Rules for Creativity

This is an advanced feature that lets you change the sampler's blending algorithm mid-generation. You can set up triggers, such as "if the noise level drops below 0.3" or "if the image changes by more than 20% in one step," and tell the sampler to switch to a different blend function. This allows for complex behaviors, like using a stable blend for the initial composition and a sharper one for the final details.

---

## 4. Node Setup and Workflow

### How to Use It: A Step-by-Step Guide

1.  **Add the Node:** Add the `PingPong Sampler (Custom FBG)` node to your workflow.
2.  **Connect the Scheduler:** Connect a `KSamplerScheduler` node to the `scheduler` input. This is mandatory.
3.  **Configure the Sampler:** Adjust the parameters on the PingPong Sampler node to your liking. Use the "Masterclass in Tuning" section below to guide you.
4.  **Connect to KSampler:** Connect the `SAMPLER` output of the PingPong node to the `sampler` input of a standard `KSampler` node.
5.  **Configure KSampler:** Your KSampler still controls the main process: `model`, `positive`, `negative`, `latent_image`, `steps`, `seed`, and the base `cfg` value. The PingPong node will take this base `cfg` and modify it dynamically.
6.  **Queue Prompt:** Run the generation. If `debug_mode` is on, check your console to see the FBG system in action.

---

## 5. Parameter Deep Dive

### 5.1. Ping-Pong Specific Parameters

* **`step_random_mode`**: Controls how the seed changes for ancestral noise injection (`off`, `block`, `reset`, `step`).
* **`step_size`**: The interval used by `block` and `reset` modes.
* **`seed`**: A base seed for the sampler's internal randomness (often overridden by the KSampler's seed).
* **`first_ancestral_step` / `last_ancestral_step`**: The step range where "ping-pong" noise injection is active.
* **`ancestral_noise_type`**: The type of noise to inject (`gaussian`, `uniform`, `brownian`).
* **`start_sigma_index` / `end_sigma_index`**: The range of the sigma schedule to process.
* **`enable_clamp_output`**: If `True`, clamps the final latent to `[-1.0, 1.0]`.
* **`scheduler`**: **Required.** Connects a `KSamplerScheduler`.
* **`blend_mode`**: The function for blending the *denoised sample* with *new noise* during ancestral steps.
* **`step_blend_mode`**: The function for blending the *denoised sample* with the *previous latent* during non-ancestral steps.

### 5.2. Feedback Guidance (FBG) & Enhancement Parameters

* **`debug_mode`**: Console log verbosity (`0`: Off, `1`: Basic, `2`: Verbose).
* **`sigma_range_preset`**: Quick presets for the FBG/CFG active range (`Custom`, `High`, `Mid`, `Low`, `All`).
* **`conditional_blend_mode`**: Enables the conditional blending triggers.
* **`conditional_blend_sigma_threshold`**: The noise level that triggers the conditional blend.
* **`conditional_blend_function_name`**: The blend function to switch to.
* **`conditional_blend_on_change`**: Enables the "big change" trigger.
* **`conditional_blend_change_threshold`**: The percentage of change that triggers the conditional blend.
* **`clamp_noise_norm`**: Enables limiting the magnitude of injected noise.
* **`max_noise_norm`**: The maximum magnitude for clamped noise.
* **`log_posterior_ema_factor`**: The smoothing factor for FBG's internal state (`0.0` is off).
* **`fbg_sampler_mode`**: The internal sampler for FBG's own calculations (`EULER`, `PINGPONG`).
* **`cfg_scale`**: The base CFG value (usually overridden by the KSampler).
* **`cfg_start_sigma` / `fbg_start_sigma`**: The noise level where guidance becomes active.
* **`cfg_end_sigma` / `fbg_end_sigma`**: The noise level where guidance becomes inactive.
* **`max_guidance_scale`**: The absolute maximum "ceiling" for the guidance scale.
* **`pi (π)`**: The core sensitivity parameter for FBG. Lower values are more aggressive.
* **`t_0` / `t_1`**: The **Autopilot controls**. If either is not `0`, they automatically calculate the guidance curve.
* **`fbg_temp` / `fbg_offset`**: The **Manual controls**. These are the primary tuning knobs for sensitivity and drift, but are **only active when `t_0` and `t_1` are both `0`**.

### 5.3. Advanced (YAML) Override

* **`yaml_settings_str`**: A text box where you can paste a YAML configuration that will **override all settings** on the node.

---

## 6. A Masterclass in Tuning Feedback Guidance

This guide will walk you through the entire process of taming and tuning the FBG system. Our goal is to achieve a **stable and dynamic guidance curve**.



### 6.1. The Golden Rule: Disabling Autopilot 🛑

The sampler has two modes for its sensitivity: **Automatic** and **Manual**.

* **Automatic Mode (Autopilot):** If **either `t_0` or `t_1` is NOT `0`**, the system is in Autopilot. It will **IGNORE** any values you set for `fbg_temp` and `fbg_offset`.
* **Manual Mode:** If **BOTH `t_0` AND `t_1` are set to `0`**, Autopilot is disengaged. You now have full, direct control.

**The first step in any serious tuning process is to engage Manual Mode.**

### 6.2. The Troubleshooting Playbook

#### Symptom: Guidance Explodes to Max Value Instantly 🚀

If your `Guidance Scale (GS)` shoots to its maximum value in the first few steps, the system is too sensitive.

**Solution: The "Safe Mode" Reset**

We need to force the sampler into a state of maximum stability.

1.  **Engage Manual Mode** by setting `t_0` and `t_1` to `0`.
2.  **Apply Safe Start Settings** to desensitize the system.

Copy this "Safe Mode" `fbg_config` block into your YAML:

```yaml
fbg_config:
  # --- Step 1: Engage Manual Mode ---
  t_0: 0.0
  t_1: 0.0
  
  # --- Step 2: Apply Safe Start Settings ---
  initial_value: 2.0            # Start the internal state high for a safety buffer
  fbg_temp: 0.01                # Start with extremely low sensitivity
  fbg_offset: 0.0                 # Start with no baseline drift
  guidance_max_change: 1000.0   # Ensure the rate limiter is disabled
  
  # --- Your other personal settings go here ---
  pi: 0.65
  max_guidance_scale: 250.0
  # etc...
```

#### Symptom: Guidance is Stable but "Stuck" 🧊

If the `Guidance Scale (GS)` barely moves, our "Safe Mode" reset was successful, but the sensitivity is too low.

**Solution: Tuning the `fbg_temp` Dial**

With the sampler in Manual Mode, **`fbg_temp` is now your primary sensitivity dial.**

1.  **Make a big jump first.** Change `fbg_temp` from `0.01` to **`0.1`**.
2.  **Observe the log.** The guidance will likely start moving.
3.  **Fine-tune.** The perfect value is usually between your last two attempts. Try values like **`0.03`**, **`0.05`**, etc., until you see a smooth, gradual change.

### 6.3. Reading the Signs - Anatomy of a Perfect Log

After tuning, you should get a log that tells a story of a controlled, dynamic generation.

```
# LOG EXCERPT
Step 0: ... GS: 2.55 | Log Posterior mean: 2.0000  # Starts at initial values
Step 1: ... GS: 2.62 | Log Posterior mean: 1.2110  # GS rises, Posterior falls
...
Step 6: ... GS: 250.00| Log Posterior mean: -1.0458 # Final push to max guidance!
...
--- PingPongSampler FBG Summary ---
  Guidance Scale Min: 2.550
  Guidance Scale Max: 249.998
  Guidance Scale Avg: 38.988
---------------------------------
```

**What this log shows:**
* **Gradual Ramp-Up:** The GS doesn't explode. It climbs steadily.
* **The "Crescendo":** In the final step, the guidance pushes to its maximum. This is desired behavior, not an error.
* **Healthy Average:** The average guidance is moderate, proving the process was controlled.

By following this process—**1. Engage Manual Mode, 2. Start with Safe Settings, 3. Tune the `fbg_temp` Dial**—you can achieve a perfect FBG curve for any model.

---

## 7. Practical Recipes & Use Cases

### Recipe 1: Stable & Controlled Image Generation

This recipe uses Manual Mode for a gentle, predictable result ideal for high-fidelity images.

```yaml
debug_mode: 1
sigma_range_preset: "Custom"
fbg_config:
  t_0: 0.0
  t_1: 0.0
  fbg_temp: 0.05
  fbg_offset: 0.0
  initial_value: 1.0
  pi: 0.5
  max_guidance_scale: 22.0
```

### Recipe 2: Aggressive & Experimental Audio Generation

This recipe uses Autopilot for a powerful, front-loaded guidance curve that is great for energetic audio.

```yaml
debug_mode: 2
sigma_range_preset: "All"
blend_mode: "a_only"
step_blend_mode: "b_only"
fbg_config:
  t_0: 0.7
  t_1: 0.4
  pi: 0.35
  max_guidance_scale: 350.0
```

### Recipe 3: The "Slow Burn" Finishing Kick

This recipe uses Autopilot but flips the controls to create a gentle start and a powerful finish, ideal for complex compositions.

```yaml
debug_mode: 1
sigma_range_preset: "All"
fbg_config:
  t_0: 0.3 # Target is early
  t_1: 0.7 # Peak is late
  pi: 0.65
  max_guidance_scale: 30.0
```

---

## 8. Technical Deep Dive

### The Order of Operations

1.  **Parse & Override:** The node reads its UI parameters, then overrides them with any valid settings from the `yaml_settings_str`.
2.  **Instantiate Core:** The `PingPongSamplerCore` class is created.
3.  **Check FBG Mode:** It checks if `t_0` and `t_1` are both zero.
    * If **NO**, it enters **Autopilot** and calculates `temp` and `offset` automatically.
    * If **YES**, it enters **Manual Mode** and uses the `fbg_temp` and `fbg_offset` you provided.
4.  **Loop per Step:** For each sampling step, it...
    * Calculates the dynamic guidance scale using the FBG logic.
    * Calls the model to get a denoised prediction.
    * Updates its internal `log_posterior` state based on the model's feedback.
    * Performs the "Ping-Pong" or standard DDIM-like step using the selected blend modes.
5.  **Return:** After all steps, it returns the final latent.
