# PingPong Sampler (Custom V0.9.9 FBG) ComfyUI Node Manual

Welcome to the comprehensive manual for the PingPong Sampler (Custom V0.9.9 FBG), a specialized ComfyUI node designed for advanced control over diffusion model sampling. This version integrates Feedback Guidance (FBG), enabling dynamic and intelligent adjustment of the guidance scale during the denoising process.

Originally optimized for Ace-Step audio and video diffusion models due to its focus on temporal coherence, it's also a powerful tool for visual experimentation. This manual will guide you through its features, parameters, and inner workings, catering to both new and advanced users.

---

## 1. Understanding the PingPong Sampler Node with FBG

### 1.1. What is it?

The PingPong Sampler (Custom V0.9.9 FBG) is a unique sampling node for ComfyUI that enhances the traditional denoising process in two key ways:

* **"Ping-Pong" Ancestral Noise Injection:** It employs a strategy of intelligently alternating between denoising your latent data and strategically re-injecting controlled ancestral noise. This is particularly beneficial for maintaining coherence in time-series data (like audio or video frames) and can also influence the diversity and texture in static image generation.

* **Feedback Guidance (FBG):** This is the major new addition. FBG uses a state-dependent coefficient to self-regulate the guidance amount (similar to Classifier-Free Guidance, but dynamic) based on whether the model perceives a particular sample as needing more or less "correction." Instead of a fixed guidance scale, FBG adapts it on-the-fly, leading to potentially more efficient sampling and improved fidelity.

### 1.2. How it Works (The Synergy)

At its core, a diffusion model refines noisy data step by step. Samplers dictate this transformation. The PingPong Sampler, now with FBG, enhances this process by:

* **Denoising:** At each step, it asks the underlying diffusion model to predict a cleaner version of the current noisy latent.

* **Ancestral Noise (Ping-Pong Action):** For a configurable range of steps, instead of just using the denoised prediction directly, it strategically adds a fresh burst of random noise back into the latent. This "ping-pong" between denoising and controlled noise injection helps maintain diversity and prevent over-smoothing. The strength and decay of this noise can be influenced by an attached scheduler.

* **Dynamic Guidance (FBG Action):** Based on an internal "log posterior" estimate (which measures how "certain" the model is about its conditional prediction relative to its unconditional one), FBG continuously adjusts the effective guidance scale (cfg\_scale) during inference.
    * If the model is highly certain, FBG can reduce guidance, promoting diversity.
    * If the model is uncertain or the sample needs more correction, FBG can increase guidance, promoting fidelity.

* **Conditional Blending:** It offers advanced options (`blend_mode`, `step_blend_mode`) to blend the positive and negative conditioning, providing flexible control over how prompts guide the generation.

### 1.3. What it Does

* **Generates High-Quality Audio/Video:** By optimizing for temporal coherence and controlled noise, it excels at creating smooth transitions and detailed outputs in time-series data.
* **Adaptive Guidance:** FBG provides intelligent, dynamic guidance, potentially leading to more optimal sampling paths and improved results without constant manual `cfg_scale` adjustments.
* **Enhanced Control over Ancestral Noise:** It provides direct control over the start and end points of ancestral noise injection (the "ping-pong" effect).
* **Flexible Prompt Blending:** Allows for nuanced interpretation of positive and negative prompts.
* **Diverse & Consistent Outputs:** The strategic noise injection combined with adaptive guidance can help prevent mode collapse, encourage more varied generations, and improve consistency.
* **Debug Visibility:** A built-in `debug_mode` allows verbose console logging to understand the sampler's internal state and FBG's dynamic adjustments.

### 1.4. How to Use It (Quick Start)

* **Replace your KSampler:** This node is a direct replacement for your existing KSampler or SamplerCustom nodes.
* **Connect Inputs:**
    * `model`: Connect your loaded diffusion model.
    * `noise`: Connect your initial latent noise.
    * `positive`: Connect your positive conditioning.
    * `negative`: Connect your negative conditioning.
    * `scheduler`: Connect a ComfyUI KSamplerScheduler node. This is crucial for the PingPong sampler to generate its internal noise decay array.
* **Adjust Parameters:** The node provides a comprehensive set of parameters. Many FBG parameters have intelligent defaults (e.g., `cfg_start_sigma: 1.0`, `fbg_end_sigma: 0.006`) that align with common Stable Diffusion models.
* **Connect Latent Output:** Connect the `LATENT` output to your VAE Decode or other latent processing nodes.
* **Enable Debug (Optional):** Toggle `debug_mode` to `True` to see detailed information in your ComfyUI console about FBG's calculations and guidance scale adjustments. This is invaluable for understanding its behavior.
* **Advanced (YAML) Control:** For expert users, the `yaml_settings_str` input provides a powerful way to override node parameters with a YAML configuration, allowing for complex presets and fine-grained control.

---

## 2. Detailed Parameter Information

The PingPong Sampler node comes with a comprehensive set of parameters. Note that standard KSampler inputs like `sampler_name`, `steps`, `cfg` (as a direct input), `denoise`, `start_at_step`, `end_at_step` are not direct inputs on this node. This node outputs a `SAMPLER` object that is then connected to a standard KSampler node (or similar) in your workflow, which handles those overall sampling parameters.

### 2.1. Ping-Pong Specific Parameters

These parameters control the unique "ping-pong" noise injection behavior.

* **`step_random_mode`** (ENUM, `off`, `block`, `reset`, `step`, default: `block`)
    * **Description:** Controls how the RNG seed varies per sampling step, influencing the randomness of ancestral noise.
        * `off`: Seed is constant.
        * `block`: Seed changes every `step_size` frames.
        * `reset`: Seed is reset based on `step_size` multiplied by the frame index, offering more varied randomness.
        * `step`: Seed changes incrementally by the frame index at each step, providing subtle variations.

* **`step_size`** (INT, default: `4`, min: `1`, max: `100`)
    * **Description:** Used by `block` and `reset` random modes to define the interval for seed changes.

* **`seed`** (INT, default: `80085`, min: `0`, max: `2^32 - 1`)
    * **Description:** The base random seed for reproducibility.

* **`first_ancestral_step`** (INT, default: `0`, min: `-1`, max: `10000`)
    * **Description:** The sampler step index (0-based) to begin ancestral noise injection (the "ping-pong" behavior). Use `-1` to effectively disable ancestral noise if `last_ancestral_step` is also `-1`.

* **`last_ancestral_step`** (INT, default: `-1`, min: `-1`, max: `10000`)
    * **Description:** The sampler step index (0-based) to end ancestral noise injection. `-1` means ancestral noise continues until the end of the sampling process.

* **`start_sigma_index`** (INT, default: `0`, min: `0`, max: `10000`)
    * **Description:** The index in the sigma array (denoising schedule) to begin sampling from. Allows skipping initial high-noise steps.

* **`end_sigma_index`** (INT, default: `-1`, min: `-10000`, max: `10000`)
    * **Description:** The index in the sigma array to end sampling at. `-1` means sample all steps.

* **`enable_clamp_output`** (BOOLEAN, default: `False`)
    * **Description:** If `True`, clamps the final output latent to the range `[-1.0, 1.0]`. Useful for preventing extreme values that might lead to artifacts during decoding.

* **`scheduler`** (SCHEDULER, Required)
    * **Description:** Connect a ComfyUI KSamplerScheduler node. This is essential for defining the noise decay curve and progression for the sampler, used internally for calculations.

* **`blend_mode`** (ENUM, `lerp`, `a_only`, `b_only`, default: `lerp`)
    * **Description:** Determines how the noise prediction from the conditional (positive) and unconditional (negative) prompts are combined during ancestral steps.
        * `lerp` (Linear Interpolation): Standard blend.
        * `a_only`: Uses only the unconditional (negative) prediction. Highly experimental.
        * `b_only`: Uses only the conditional (positive) prediction. Bypasses negative prompt.

* **`step_blend_mode`** (ENUM, `lerp`, `a_only`, `b_only`, default: `lerp`)
    * **Description:** Controls how predictions from previous steps or internal model states are blended with the current step's prediction in non-ancestral steps. Generally left at `lerp`.

### 2.2. Feedback Guidance (FBG) Specific Parameters

These parameters control the dynamic guidance behavior, and are encapsulated within the FBG algorithm.

* **`fbg_sampler_mode`** (ENUM, `EULER`, `PINGPONG`, default: `EULER`)
    * **Description:** FBG's internal sampler mode for its calculations. `EULER` is standard. `PINGPONG` influences how FBG calculates its internal step, but the main PingPongSampler's noise injection logic remains dominant.

* **`cfg_scale`** (FLOAT, default: `1.0`, min: `-1000.0`, max: `1000.0`, step: `0.01`)
    * **Description:** This is the base Classifier-Free Guidance (CFG) scale that FBG dynamically modifies during sampling.

* **`cfg_start_sigma`** (FLOAT, default: `1.0`, min: `0.0`, max: `9999.0`, step: `0.001`)
    * **Description:** The noise level (sigma) at which regular CFG (and thus FBG's influence over it) begins to be active.

* **`cfg_end_sigma`** (FLOAT, default: `0.004`, min: `0.0`, max: `9999.0`, step: `0.001`)
    * **Description:** The noise level (sigma) at which regular CFG (and thus FBG's influence over it) ceases to be active.

* **`fbg_start_sigma`** (FLOAT, default: `1.0`, min: `0.0`, max: `9999.0`, step: `0.001`)
    * **Description:** The noise level (sigma) at which Feedback Guidance (FBG) actively calculates and applies its dynamic scale.

* **`fbg_end_sigma`** (FLOAT, default: `0.004`, min: `0.0`, max: `9999.0`, step: `0.001`)
    * **Description:** The noise level (sigma) at which FBG ceases to actively calculate and apply its dynamic scale.

* **`ancestral_start_sigma`** (FLOAT, default: `1.0`, min: `0.0`, max: `9999.0`, step: `0.001`)
    * **Description:** FBG internal parameter: First sigma ancestral/pingpong sampling (for FBG's base sampler) will be active. Note: `fbg_eta` must also be non-zero for this to have an effect within FBG's internal calculations.

* **`ancestral_end_sigma`** (FLOAT, default: `0.004`, min: `0.0`, max: `9999.0`, step: `0.001`)
    * **Description:** FBG internal parameter: Last sigma ancestral/pingpong sampling (for FBG's base sampler) will be active.

* **`max_guidance_scale`** (FLOAT, default: `10.0`, min: `1.0`, max: `1000.0`, step: `0.01`)
    * **Description:** Upper limit for the total guidance scale (`cfg_scale` + FBG_component) after FBG and CFG adjustments.

* **`initial_guidance_scale`** (FLOAT, default: `1.0`, min: `1.0`, max: `1000.0`, step: `0.01`)
    * **Description:** Initial value for FBG's internal guidance scale tracking. Primarily affects how `guidance_max_change` behaves at the very beginning of sampling.

* **`guidance_max_change`** (FLOAT, default: `1000.0`, min: `0.0`, max: `1000.0`, step: `0.01`)
    * **Description:** Limits the percentage change of the FBG guidance scale per step. A value like `0.5` means a maximum 50% change from the previous step's guidance scale. A high value (e.g., `1000.0`) effectively disables this limiting.

* **`pi (π)`** (FLOAT, default: `0.95`, min: `0.0`, max: `1.0`, step: `0.01`)
    * **Description:** The mixing parameter ($\pi$) from the FBG paper. Higher values (e.g., `0.95-0.999`) are typically for very well-learned models (like those benchmarked in the FBG paper). Lower values (e.g., `0.2-0.8`) may be more effective for less optimized or general Text-to-Image models (e.g., Stable Diffusion), as it causes guidance to activate at higher posterior probabilities. Adjust based on model type and desired effect. Setting $\pi$ to `1.0` implies the conditional model is perfect and needs no guidance.

* **`t_0`** (FLOAT, default: `0.5`, min: `0.0`, max: `1.0`, step: `0.01`)
    * **Description:** Normalized diffusion time (0-1) where FBG guidance scale is expected to reach a reference value. If `0.0` and `t_1` is also `0.0`, then `fbg_temp` and `fbg_offset` are used directly. Otherwise, these are auto-calculated.

* **`t_1`** (FLOAT, default: `0.4`, min: `0.0`, max: `1.0`, step: `0.01`)
    * **Description:** Normalized diffusion time (0-1) where FBG guidance is estimated to reach its maximum. If `0.0` and `t_0` is also `0.0`, then `fbg_temp` and `fbg_offset` are used directly. Otherwise, these are auto-calculated.

* **`fbg_temp`** (FLOAT, default: `0.0`, min: `-1000.0`, max: `1000.0`, step: `0.001`)
    * **Description:** Temperature parameter for FBG log posterior update. Only applies if both `t_0` and `t_1` are `0.0`, otherwise it is calculated automatically based on `t_0`, `t_1`, and the sigma schedule.

* **`fbg_offset`** (FLOAT, default: `0.0`, min: `-1000.0`, max: `1000.0`, step: `0.001`)
    * **Description:** Offset parameter for FBG log posterior update. Only applies if both `t_0` and `t_1` are `0.0`, otherwise it is calculated automatically based on `t_0`, `t_1`, and the sigma schedule.

* **`log_posterior_initial_value`** (FLOAT, default: `0.0`, min: `-1000.0`, max: `1000.0`, step: `0.01`)
    * **Description:** Initial value for FBG's internal log posterior estimate. Typically, this does not need to be changed.

* **`fbg_guidance_multiplier`** (FLOAT, default: `1.0`, min: `0.001`, max: `1000.0`, step: `0.01`)
    * **Description:** A scalar multiplier applied to the FBG guidance component before it's added to the base CFG.

* **`fbg_eta`** (FLOAT, default: `0.0`, min: `-1000.0`, max: `1000.0`, step: `0.01`)
    * **Description:** FBG internal parameter: Controls the amount of noise added during ancestral sampling within FBG's internal model prediction step. Must be `>0` for ancestral noise to activate for FBG's internal calculations.

* **`fbg_s_noise`** (FLOAT, default: `1.0`, min: `-1000.0`, max: `1000.0`, step: `0.01`)
    * **Description:** FBG internal parameter: Scale for noise added during ancestral sampling within FBG's internal model prediction step.

* **`debug_mode`** (BOOLEAN, default: `False`)
    * **Description:** Enable verbose debug messages in the ComfyUI console. Invaluable for troubleshooting FBG and sampler behavior, showing dynamic guidance scales, log posterior values, and more.

### 2.3. Advanced (YAML) Override

* **`yaml_settings_str`** (STRING, Multiline, Optional)
    * **Description:** A multiline text input allowing you to provide a YAML (YAML Ain't Markup Language) string to override any of the node's parameters.
    * **Use & Why:** This is a powerful feature for saving and loading complex sampler presets. You can define all node parameters within the YAML string, and they will take precedence over the individual node inputs. This is excellent for creating consistent workflows or sharing specific settings. Boolean values from YAML strings (e.g., `"true"`, `"false"`) are automatically converted to Python booleans.
    * **Important Note on FBG Parameters in YAML:** For FBG parameters nested under `fbg_config` in the YAML, it's best practice to use the internal `FBGConfig` field names rather than the node's input names. For example, use `sampler_mode` instead of `fbg_sampler_mode`, and `initial_value` instead of `log_posterior_initial_value`. The node includes robust remapping, but using the correct internal names in your YAML is clearer.

* **Example YAML Structure** (matching V0.9.9 defaults and common use):
    ```yaml
    # Example YAML settings for PingPong Sampler (Custom V0.9.9 FBG)

    # --- PingPong Sampler Specific Parameters (Top-level) ---
    step_random_mode: "block"
    step_size: 4
    seed: 12345
    first_ancestral_step: 15
    last_ancestral_step: 35
    start_sigma_index: 0
    end_sigma_index: -1
    enable_clamp_output: false
    blend_mode: "lerp"
    step_blend_mode: "lerp"

    # --- Feedback Guidance (FBG) Specific Parameters (Nested under fbg_config) ---
    fbg_config:
      # Use internal FBGConfig names where applicable for clarity, though remapping handles node names.
      sampler_mode: "EULER" # Matches 'fbg_sampler_mode' node input
      cfg_scale: 7.0
      cfg_start_sigma: 1.0 # Aligned with typical model sigma start
      cfg_end_sigma: 0.004 # Aligned with typical model sigma end
      fbg_start_sigma: 1.0 # Aligned with typical model sigma start
      fbg_end_sigma: 0.006 # Your preferred FBG end sigma
      ancestral_start_sigma: 1.0 # Aligned with typical model sigma start
      ancestral_end_sigma: 0.004 # Aligned with typical model sigma end
      max_guidance_scale: 15.0
      initial_guidance_scale: 1.0
      guidance_max_change: 0.5 # Example: Limit guidance change to 50% per step
      pi: 0.5 # Common adjustment for T2I
      t_0: 0.5
      t_1: 0.4
      # temp and offset are usually auto-calculated if t_0/t_1 are non-zero.
      # Only set explicitly if t_0 and t_1 are both 0.0.
      temp: 0.0
      offset: 0.0
      initial_value: 0.0 # Matches 'log_posterior_initial_value' node input
      fbg_guidance_multiplier: 1.0

    # --- Top-level FBG specific parameters ---
    fbg_eta: 0.0
    fbg_s_noise: 1.0

    # --- Debug Mode ---
    debug_mode: true
    ```

---

## 3. In-Depth Technical Information

This section explores the underlying mechanics and algorithms that power the PingPong Sampler (Custom FBG).

### 3.1. Sampling Flow Overview

The PingPong Sampler integrates into ComfyUI by providing a custom `SAMPLER` output. This output is designed to connect directly to the sampler input of a standard KSampler node. This means our custom node doesn't handle the overall `sampler_name`, `steps`, or `denoise` parameters directly; instead, it configures how the underlying KSampler will perform its denoising steps.

* **Input Parsing:** The `PingPongSamplerNode.get_sampler` method first reads all direct input parameters from the ComfyUI UI.

* **YAML Override Logic:** It then attempts to parse the `yaml_settings_str`. If valid YAML is found and contains parameters that also exist as direct node inputs, the YAML values take precedence and override the direct node inputs. This makes YAML a powerful way to manage complex presets. Crucially, specific FBG parameter names from YAML (e.g., `log_posterior_initial_value`) are remapped internally to the correct `FBGConfig` field names (e.g., `initial_value`) to ensure compatibility.

* **Blend Mode Resolution:** It resolves the `blend_mode` and `step_blend_mode` strings into actual `torch` functions (e.g., `torch.lerp`, or custom lambda functions for `a_only`/`b_only`) using the `_INTERNAL_BLEND_MODES` dictionary.

* **FBGConfig Instantiation:** An `FBGConfig` NamedTuple is created, encapsulating all the Feedback Guidance parameters (many of which have new, more relevant default sigma ranges).

* **KSAMPLER Construction:** Finally, it constructs and returns a `KSAMPLER` object. This `KSAMPLER` is configured to use `PingPongSampler.go` as its core sampling function, passing all the resolved parameters (after YAML merging and `FBGConfig` creation) as `extra_options` (which are then unpacked as `**kwargs` into `PingPongSampler.go`).

### 3.2. The `PingPongSampler.go` Static Method (The Core Logic)

This static method is the heart of the custom sampler, executed by the underlying ComfyUI KSampler at each step.

* **Initialization:** When `PingPongSampler.go` is first called, it instantiates the `PingPongSampler` class, passing all the `extra_options` (which includes all the node's parameters and the `FBGConfig`). During this `__init__`, FBG's `temp` and `offset` values are automatically calculated if `t_0` and `t_1` are non-zero, simplifying user configuration.

* **Dynamic Guidance Scale Calculation (`get_dynamic_guidance_scale`):** At each sampling step, this method is called to determine the effective guidance scale.
    * It checks the current sigma value against `cfg_start_sigma`/`cfg_end_sigma` and `fbg_start_sigma`/`fbg_end_sigma` to determine if CFG and FBG are active for that step.
    * If FBG is active, it calculates an FBG-specific component based on the current `log_posterior` estimate.
    * It then combines this FBG component with the `cfg_scale`.
    * The total guidance scale is clamped by `max_guidance_scale` and further limited by `guidance_max_change` to ensure smooth transitions.

* **Model Denoising (`_model_denoise_with_guidance`):** This wrapper function calls the underlying diffusion model. Crucially, it sets `disable_cfg1_optimization = True` in `model_options` to ensure that both the conditional and unconditional predictions are available. This is essential because FBG needs both to update its `log_posterior` estimate. The dynamically calculated `guidance_scale` is passed here to influence the model's behavior.

* **Log Posterior Update (`_update_log_posterior`):** After the model provides its predictions, FBG updates its internal `log_posterior` estimate. This estimate tracks how well the conditional and unconditional predictions align with the actual latent state, forming the "feedback" mechanism. This value is used in the next step to calculate the new dynamic guidance scale. Error handling is included to prevent issues with very small sigma values or division by zero.

* **Ancestral Step Determination:** It determines whether the current step is within the `first_ancestral_step` and `last_ancestral_step` range (from the PingPong specific parameters). If so, ancestral noise injection is active.

* **The "Ping-Pong" Action:** When ancestral steps are active:
    * A new random noise tensor is generated (scaled based on the current sigma and `sigma_next`).
    * This new noise is then blended with the model's denoised prediction using the selected `blend_function` (`blend_mode`). This is where the core "ping-pong" effect happens, introducing controlled variability and contributing to temporal coherence.

* **Callback:** Progress information is regularly sent to ComfyUI's callback system for UI updates.

* **Final Output:** After iterating through all steps, the final denoised latent is returned.

### 3.3. Internal Blend Modes

The `_INTERNAL_BLEND_MODES` dictionary is a simple lookup table that maps user-friendly string names (`"lerp"`, `"a_only"`, `"b_only"`) to their corresponding Python functions, used by `blend_mode` and `step_blend_mode`.

* **`"lerp"`:** Uses `torch.lerp(a, b, weight)`, which is standard linear interpolation. `a` is typically the unconditional prediction, `b` is the conditional prediction, and `weight` is derived from the guidance scale.

* **`"a_only"`:** Returns `a`. This means the `cfg_function` will effectively ignore the conditional (positive) prediction and rely solely on the unconditional (negative) prediction. This is usually not what you want for standard generation but can be an experimental setting.

* **`"b_only"`:** Returns `b`. This means the `cfg_function` will ignore the unconditional (negative) prediction and rely solely on the conditional (positive) prediction. This essentially turns off negative prompting and can lead to over-saturation or less diverse results.

In summary, the PingPong Sampler (Custom V0.9.9 FBG) is a powerful and flexible tool for diffusion sampling. Its combined approach of controlled ancestral noise injection and dynamic Feedback Guidance offers a sophisticated method for achieving both high fidelity and temporal consistency, particularly in challenging time-series generation tasks.
