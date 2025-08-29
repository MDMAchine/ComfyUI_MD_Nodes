# PingPong Sampler (Custom v0.9.9-p2 FBG) ComfyUI Node Manual

Welcome to the comprehensive manual for the PingPong Sampler (Custom v0.9.9-p2 FBG), a specialized ComfyUI node designed for advanced control over diffusion model sampling. This version integrates Feedback Guidance (FBG), enabling dynamic and intelligent adjustment of the guidance scale during the denoising process.

Originally optimized for Ace-Step audio and video diffusion models due to its focus on temporal coherence, it's also a powerful tool for visual experimentation. This manual will guide you through its features, parameters, and inner workings, catering to both new and advanced users.

---

## 1. Understanding the PingPong Sampler Node with FBG

### 1.1. What is it?

The PingPong Sampler (Custom v0.9.9-p2 FBG) is a unique sampling node for ComfyUI that enhances the traditional denoising process in several key ways:

* **"Ping-Pong" Ancestral Noise Injection:** It employs a strategy of intelligently alternating between denoising your latent data and strategically re-injecting controlled ancestral noise. This is particularly beneficial for maintaining coherence in time-series data (like audio or video frames) and can also influence the diversity and texture in static image generation.

* **Feedback Guidance (FBG):** This is the major new addition. FBG uses a state-dependent coefficient to self-regulate the guidance amount (similar to Classifier-Free Guidance, but dynamic) based on whether the model perceives a particular sample as needing more or less "correction." Instead of a fixed guidance scale, FBG adapts it on-the-fly.

* **Advanced Feature Suite:** This version includes numerous enhancements for fine-grained control, including selectable noise types, sigma range presets for scheduling, conditional blending rules, and noise norm clamping for stability.

### 1.2. How it Works (The Synergy)

At its core, a diffusion model refines noisy data step by step. Samplers dictate this transformation. The PingPong Sampler, now with FBG, enhances this process by:

* **Denoising:** At each step, it asks the underlying diffusion model to predict a cleaner version of the current noisy latent.

* **Ancestral Noise (Ping-Pong Action):** For a configurable range of steps, instead of just using the denoised prediction directly, it strategically adds a fresh burst of random noise back into the latent. This "ping-pong" between denoising and controlled noise injection helps maintain diversity and prevent over-smoothing.

* **Dynamic Guidance (FBG Action):** Based on an internal "log posterior" estimate (which measures how "certain" the model is about its conditional prediction relative to its unconditional one), FBG continuously adjusts the effective guidance scale during inference.
    * If the model is highly certain, FBG can reduce guidance, promoting diversity.
    * If the model is uncertain or the sample needs more correction, FBG can increase guidance, promoting fidelity.

### 1.3. What it Does

* **Generates High-Quality Audio/Video:** By optimizing for temporal coherence and controlled noise, it excels at creating smooth transitions and detailed outputs in time-series data.
* **Adaptive Guidance:** FBG provides intelligent, dynamic guidance, potentially leading to more optimal sampling paths and improved results without constant manual `cfg_scale` adjustments.
* **Enhanced Control over Ancestral Noise:** It provides direct control over the start and end points of ancestral noise injection, including the type of noise used (`gaussian`, `uniform`, etc.).
* **Flexible Scheduling & Blending:** Allows for nuanced control over when guidance is active and how the sampler progresses from step to step.
* **Debug Visibility:** A built-in multi-level `debug_mode` allows verbose console logging to understand the sampler's internal state and FBG's dynamic adjustments.

### 1.4. How to Use It (Quick Start)

* **Use as a Sampler Provider:** This node does **not** replace your KSampler. Instead, it generates a custom `SAMPLER` object.
* **Connect Inputs:**
    * `scheduler`: Connect a ComfyUI KSamplerScheduler node. This is crucial for the sampler's internal calculations.
* **Connect Output to a KSampler:**
    * Connect the `SAMPLER` output of this node to the `sampler` input of a standard KSampler (or similar) node.
    * The KSampler node will still control the overall `steps`, `cfg` (which FBG will use as a base), `denoise`, `seed`, model, positive/negative conditioning, and latent image.
* **Adjust Parameters:** Use the extensive parameters on this node to control the FBG and Ping-Pong behavior.
* **Enable Debug (Optional):** Set `debug_mode` to `1` or `2` to see detailed information in your ComfyUI console about FBG's calculations and guidance scale adjustments. This is invaluable for understanding its behavior.
* **Advanced (YAML) Control:** For expert users, the `yaml_settings_str` input provides a powerful way to override node parameters with a YAML configuration.



---

## 2. Detailed Parameter Information

This node's parameters configure the `SAMPLER` object that you pass to a KSampler. The KSampler itself still controls the main generation process.

### 2.1. Ping-Pong Specific Parameters

These parameters control the unique "ping-pong" noise injection behavior.

* **`step_random_mode`** (ENUM, default: `block`)
    * **Description:** Controls how the RNG seed varies per sampling step. Options: `off`, `block`, `reset`, `step`.

* **`step_size`** (INT, default: `4`)
    * **Description:** Used by `block` and `reset` random modes to define the interval for seed changes.

* **`seed`** (INT, default: `80085`)
    * **Description:** The base random seed for reproducibility. Note: This is often overridden by the seed in your KSampler node.

* **`first_ancestral_step`** (INT, default: `0`)
    * **Description:** The sampler step index (0-based) to begin ancestral noise injection.

* **`last_ancestral_step`** (INT, default: `-1`)
    * **Description:** The sampler step index to end ancestral noise injection. `-1` means it continues until the last step.

* **`ancestral_noise_type`** (ENUM, default: `gaussian`)
    * **Description:** The type of random noise to inject during ancestral steps. Options: `gaussian`, `uniform`, `brownian`.

* **`start_sigma_index`** (INT, default: `0`)
    * **Description:** The index in the sigma array to begin sampling from.

* **`end_sigma_index`** (INT, default: `-1`)
    * **Description:** The index in the sigma array to end sampling at. `-1` means sample all steps.

* **`enable_clamp_output`** (BOOLEAN, default: `False`)
    * **Description:** If `True`, clamps the final output latent to the range `[-1.0, 1.0]`.

* **`scheduler`** (SCHEDULER, Required)
    * **Description:** Connect a ComfyUI KSamplerScheduler node. Essential for defining the noise decay curve.

* **`blend_mode`** (ENUM, default: `lerp`)
    * **Description:** The function used to blend the *denoised sample* and the *injected noise* during ancestral steps. Options: `lerp`, `slerp`, `add`, `a_only`, `b_only`.

* **`step_blend_mode`** (ENUM, default: `lerp`)
    * **Description:** The function used to blend the *denoised sample* with the *previous step's latent* during non-ancestral (DDIM-like) steps.

### 2.2. Feedback Guidance (FBG) & Enhancement Parameters

These parameters control the dynamic guidance behavior and other advanced features.

* **`debug_mode`** (INT, default: `0`)
    * **Description:** Controls the level of detail printed to the console. `0`: Off, `1`: Basic Info & Summary, `2`: Verbose Step-by-Step Details.

* **`sigma_range_preset`** (ENUM, default: `Custom`)
    * **Description:** Quickly sets the active sigma range for FBG and CFG, overriding the manual sigma inputs. Options: `Custom`, `High`, `Mid`, `Low`, `All`.

* **`conditional_blend_mode`** (BOOLEAN, default: `False`)
    * **Description:** Enables switching the blend function for non-ancestral steps based on sigma level.

* **`conditional_blend_sigma_threshold`** (FLOAT, default: `0.5`)
    * **Description:** If the current sigma is below this value, the `conditional_blend_function_name` is used.

* **`conditional_blend_function_name`** (ENUM, default: `slerp`)
    * **Description:** The blend function to use when the conditional sigma threshold is met.

* **`conditional_blend_on_change`** (BOOLEAN, default: `False`)
    * **Description:** Enables switching the blend function based on the magnitude of change in a step.

* **`conditional_blend_change_threshold`** (FLOAT, default: `0.1`)
    * **Description:** Threshold for relative change that triggers the conditional blend.

* **`clamp_noise_norm`** (BOOLEAN, default: `False`)
    * **Description:** If `True`, limits the L2 norm (magnitude) of the injected ancestral noise vector.

* **`max_noise_norm`** (FLOAT, default: `1.0`)
    * **Description:** The maximum allowed L2 norm for the noise vector when clamping is enabled.

* **`log_posterior_ema_factor`** (FLOAT, default: `0.0`)
    * **Description:** Applies Exponential Moving Average smoothing to FBG's internal state. `0.0` is off. Small values like `0.1-0.3` can reduce jitter.

* **`fbg_sampler_mode`** (ENUM, default: `EULER`)
    * **Description:** FBG's internal sampler mode for its calculations.

* **`cfg_scale`** (FLOAT, default: `1.0`)
    * **Description:** The base CFG scale that FBG dynamically modifies. This is usually overridden by the CFG value in your KSampler node.

* **`cfg_start_sigma` / `cfg_end_sigma`** (FLOAT)
    * **Description:** Defines the sigma range where the base CFG is active.

* **`fbg_start_sigma` / `fbg_end_sigma`** (FLOAT)
    * **Description:** Defines the sigma range where FBG calculations are active.

* **`ancestral_start_sigma` / `ancestral_end_sigma`** (FLOAT)
    * **Description:** FBG internal parameters defining its own ancestral sampling range.

* **`max_guidance_scale`** (FLOAT, default: `10.0`)
    * **Description:** An absolute upper limit (ceiling) for the final, combined guidance scale.

* **`initial_guidance_scale`** (FLOAT, default: `1.0`)
    * **Description:** Initial value for FBG's internal guidance scale tracking.

* **`guidance_max_change`** (FLOAT, default: `1000.0`)
    * **Description:** Limits the percentage change of the guidance scale per step. A high value (`1000.0`) effectively disables this limiter.

* **`pi (π)`** (FLOAT, default: `0.5`)
    * **Description:** The mixing parameter ($\pi$) from the FBG paper. Higher values (`0.9+`) are for highly coherent models. Lower values (`0.2-0.8`) are often better for general T2I models.

* **`t_0` / `t_1`** (FLOAT, default: `0.5` / `0.4`)
    * **Description:** Normalized time values that control the FBG's automatic calculation of `temp` and `offset`. **To enable Manual Mode for tuning, both must be set to `0.0`**.

* **`fbg_temp` / `fbg_offset`** (FLOAT, default: `0.0`)
    * **Description:** The core sensitivity and drift parameters for FBG. **Only used if both `t_0` and `t_1` are `0.0`**.

* **`log_posterior_initial_value`** (FLOAT, default: `0.0`)
    * **Description:** The starting value for FBG's internal state.

* **`fbg_guidance_multiplier`** (FLOAT, default: `1.0`)
    * **Description:** A final multiplier for the FBG component of the guidance.

* **`fbg_eta` / `fbg_s_noise`** (FLOAT)
    * **Description:** FBG internal parameters for its own noise calculations.

### 2.3. Advanced (YAML) Override

* **`yaml_settings_str`** (STRING, Optional)
    * **Description:** A multiline text input to provide a YAML string that overrides all other node parameters. This is excellent for saving and sharing complex presets.

---

## 3. A Masterclass in Tuning Feedback Guidance

This guide will walk you through the entire process of taming and tuning the FBG system. By the end, you will not only have a perfectly tuned sampler but will also understand the philosophy behind adjusting its parameters to suit any model.

Our goal is to achieve a **stable and dynamic guidance curve**: one that starts low, responds to the model's needs, and avoids getting "stuck" or "exploding."



### 3.1. The Golden Rule: Disabling Autopilot 🛑

The sampler has two modes for its sensitivity parameters (`temp` and `offset`): **Automatic** and **Manual**.

* **Automatic Mode (Autopilot):** If **either `t_0` or `t_1` is NOT `0`**, the system is in Autopilot. It will **IGNORE** any values you set for `fbg_temp` and `fbg_offset`.
* **Manual Mode:** If **BOTH `t_0` AND `t_1` are set to `0`**, Autopilot is disengaged. You now have full, direct control.

**The first step in any serious tuning process is to engage Manual Mode.**

### 3.2. The Troubleshooting Playbook

#### Symptom: Guidance Explodes to Max Value Instantly 🚀

If your `Guidance Scale (GS)` shoots to its maximum value in the first few steps, the system is too sensitive to your model's initial feedback.

**Solution: The "Safe Mode" Reset**

We need to force the sampler into a state of maximum stability. This gives us a safe baseline from which to begin tuning.

1.  **Engage Manual Mode** by setting `t_0` and `t_1` to `0`.
2.  **Apply Safe Start Settings** to desensitize the system and create a safety buffer.

Here is the "Safe Mode" `fbg_config` block. Copy this into your YAML to establish a stable starting point.

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
  cfg_scale: 2.5
  max_guidance_scale: 250.0
  # etc...
```

#### Symptom: Guidance is Stable but "Stuck" 🧊

If the `Guidance Scale (GS)` barely moves, our "Safe Mode" settings were successful, but the sensitivity is too low.

**Solution: Tuning the `fbg_temp` Dial**

With the sampler in Manual Mode, **`fbg_temp` is now your primary sensitivity dial.** We started it at `0.01`. Now, we will carefully turn it up.

1.  **Make a big jump first.** Change `fbg_temp` from `0.01` to **`0.1`**. This will show a noticeable change.
2.  **Observe the log.** The guidance will likely start moving, but it might be too aggressive.
3.  **Fine-tune.** The perfect value is usually between your last two attempts. Try values like **`0.03`**, **`0.05`**, etc., until you see a smooth, gradual change in the Guidance Scale.

### 3.3. Reading the Signs - Anatomy of a Perfect Log

After tuning, you should get a log that tells a story of a controlled, dynamic generation.

```
# LOG EXCERPT
Updated FBGConfig: offset=-0.0930, temp=-0.0001 # Autopilot is OFF, using gentle auto-settings

...
Step 0: ... GS: 2.55 | Log Posterior mean: 2.0000  # Starts at initial values
Step 1: ... GS: 2.62 | Log Posterior mean: 1.2110  # GS rises, Posterior falls
Step 2: ... GS: 2.84 | Log Posterior mean: 0.3180  # Continues the gradual trend
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
* **Gradual Ramp-Up:** The Guidance Scale (GS) doesn't explode. It climbs steadily, showing the system is responding dynamically.
* **The "Crescendo":** In the final step, as the sampler locks in the details, the guidance pushes to its maximum. This is desired behavior, not an error.
* **Healthy Average:** The average guidance is moderate, proving the process was controlled.

By following this process—**1. Engage Manual Mode, 2. Start with Safe Settings, 3. Tune the `fbg_temp` Dial**—you can achieve a perfect FBG curve for any model.

---

## 4. In-Depth Technical Information

This section explores the underlying mechanics and algorithms.

### 4.1. Sampling Flow Overview

The node parses its parameters, merges them with any YAML overrides, creates an `FBGConfig` object, and finally constructs a `KSAMPLER` object configured to use its internal `go` function. This `KSAMPLER` is what gets passed to the rest of the workflow.

### 4.2. The Core `go` Logic

* **Initialization:** Instantiates the `PingPongSamplerCore` class. If `t_0` and `t_1` are not both zero, it automatically calculates `temp` and `offset`, overriding any manual values.
* **Dynamic Guidance Calculation:** At each step, it calculates the effective guidance scale based on the active sigma ranges, the current `log_posterior` value, and the base `cfg_scale`.
* **Model Denoising:** It calls the model, ensuring both conditional and unconditional predictions are available for the FBG update.
* **Log Posterior Update:** It updates its internal `log_posterior` estimate based on the difference between the model's predictions, forming the "feedback" loop.
* **The "Ping-Pong" Action:** During ancestral steps, it generates and blends new noise with the denoised prediction, using the selected `blend_mode`.

### 4.3. Internal Blend Modes

The `_INTERNAL_BLEND_MODES` dictionary maps string names to functions. In this sampler, they are used for two distinct purposes:
* **`blend_mode`**: Blends the **denoised sample** with **newly injected noise** during an ancestral step.
* **`step_blend_mode`**: Blends the **denoised sample** with the **previous latent state** during a non-ancestral (DDIM-like) step.
