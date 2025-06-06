# PingPong Sampler (Custom V0.8.15) ComfyUI Node Manual

Welcome to the manual for the PingPong Sampler, a custom ComfyUI node specifically optimized for Ace-Step audio and video diffusion models. While its primary purpose is high-quality audio/video generation, it's also a powerful tool for visual experimentation. This manual will guide you through its features, parameters, and inner workings.

---

## 1. Understanding the PingPong Sampler Node

### What is it?

The PingPong Sampler is a specialized sampling node for ComfyUI. Unlike standard samplers that often follow a fixed path through the denoising process, the PingPong Sampler employs a "ping-pong" strategy, intelligently alternating between denoising your latent data and injecting carefully controlled ancestral noise. This nuanced approach is designed to produce results with distinct characteristics, especially beneficial for time-series data like audio and video.

### How it Works

At its core, a diffusion model refines noisy data step by step, gradually transforming it into a clean output. Samplers dictate how this transformation occurs. The PingPong Sampler enhances this process by:

* **Denoising**: At each step, it asks the underlying diffusion model to predict a cleaner version of the current noisy latent.
* **Ancestral Noise Injection (Ping-Pong Action)**: For a configurable range of steps (the "ancestral steps"), instead of just using the denoised prediction directly, it strategically adds a fresh burst of random noise back into the latent. This "ping-pong" between denoising and controlled noise injection helps maintain diversity, prevent over-smoothing, and contribute to temporal coherence in sequential data.
* **Noise Scheduling Integration**: It can accept a custom noise decay schedule from the `NoiseDecayScheduler (Custom)` node, allowing precise control over how ancestral noise fades over time.
* **Conditional Blending**: It offers advanced options to blend the positive and negative conditioning, providing flexible control over how prompts guide the generation.

### What it Does

* **Generates High-Quality Audio/Video**: By optimizing for temporal coherence and controlled noise, it excels at creating smooth transitions and detailed outputs in time-series data.
* **Enhanced Control over Ancestral Noise**: Unlike standard samplers, it provides direct control over the amount and decay of ancestral noise.
* **Flexible Prompt Blending**: Allows for nuanced interpretation of positive and negative prompts, potentially leading to more artistic or specific results.
* **Diverse Outputs**: The strategic noise injection can help prevent mode collapse and encourage more varied generations.
* **Predictable Workflow**: Designed to integrate smoothly into existing ComfyUI workflows, acting as a direct replacement for standard `KSampler` nodes.

### How to Use It

1.  **Replace your KSampler**: The PingPong Sampler (Custom) node is intended as a direct replacement for your existing `KSampler` or `SamplerCustom` nodes in your ComfyUI workflow.
2.  **Connect Inputs**:
    * **model**: Connect your loaded diffusion model (e.g., from `Load Checkpoint`).
    * **noise**: Connect your initial noise latent (e.g., from `Latent Image` or `Empty Latent Image`).
    * **positive**: Connect your positive conditioning (e.g., from `CLIPTextEncode`).
    * **negative**: Connect your negative conditioning (e.g., from `CLIPTextEncode`).
    * **seed**: Connect your seed (e.g., from `CR Seed`).
    * **cfg**: Set your Classifier-Free Guidance scale.
    * **sampler_name**: Choose your preferred sampler algorithm (e.g., `euler_ancestral`, `dpmpp_2m`).
    * **scheduler_name**: Choose your scheduler (e.g., `karras`, `normal`).
    * **steps**: Set the total number of sampling steps.
    * **`optional_noise_decay_scheduler`**: (Crucial for PingPong's unique behavior) Connect the `SCHEDULER` output from the `NoiseDecayScheduler (Custom)` node here. This is where you define how ancestral noise decays.
3.  **Adjust Ping-Pong Specific Parameters**: Fine-tune the `ancestral_steps_start_ratio`, `ancestral_steps_end_ratio`, `ancestral_noise_strength`, and `blend_mode` to control the unique "ping-pong" behavior.
4.  **Connect Latent Output**: Connect the `LATENT` output of the PingPong Sampler to your VAE Decode or other latent processing nodes.
5.  **Advanced (YAML) Control**: For expert users, the `yaml_settings_str` input provides a powerful way to override node parameters with a YAML configuration, allowing for complex presets and fine-grained control.

---

## 2. Detailed Parameter Information

The PingPong Sampler node comes with a comprehensive set of parameters, combining standard sampler controls with its unique PingPong-specific options.

### Standard Sampler Inputs (Common to KSampler)

* **model** (`MODEL`, Required)
    * **Description**: The diffusion model to be used for sampling.
* **noise** (`LATENT`, Required)
    * **Description**: The initial latent noise to be denoised.
* **positive** (`CONDITIONING`, Required)
    * **Description**: Positive conditioning (e.g., from text prompts).
* **negative** (`CONDITIONING`, Required)
    * **Description**: Negative conditioning (e.g., from negative text prompts).
* **seed** (`INT`, Required)
    * **Description**: The random seed for reproducibility.
* **cfg** (`FLOAT`, default: `8.0`, min: `0.1`, max: `100.0`, step: `0.1`)
    * **Description**: Classifier-Free Guidance scale. Higher values make the output more aligned with the prompt but can lead to artifacts.
* **sampler_name** (`ENUM`, various sampler algorithms like `euler`, `euler_ancestral`, `dpmpp_2m`, etc.)
    * **Description**: The core algorithm used for denoising. `euler_ancestral` is often used with ancestral samplers.
* **scheduler_name** (`ENUM`, `normal`, `karras`, `exponential`, `sgm_uniform`, `simple`)
    * **Description**: The noise schedule used by the sampler. `karras` is often preferred for quality.
* **steps** (`INT`, default: `20`, min: `1`, max: `1000`, step: `1`)
    * **Description**: The total number of denoising steps. More steps generally mean better quality but longer generation times.
* **denoise** (`FLOAT`, default: `1.0`, min: `0.0`, max: `1.0`, step: `0.01`)
    * **Description**: Denoising strength. `1.0` means full denoising from noise, `0.0` means no denoising (input latent is unchanged). Useful for img2img.
* **start_at_step** (`INT`, default: `0`, min: `0`, max: `1000`, step: `1`)
    * **Description**: Starts sampling from a specific step, skipping initial steps.
* **end_at_step** (`INT`, default: `10000`, min: `0`, max: `10000`, step: `1`)
    * **Description**: Ends sampling at a specific step, potentially leaving some noise.

### Ping-Pong Specific Parameters

* **optional_noise_decay_scheduler** (`SCHEDULER`, Optional)
    * **What it is**: An optional input for a custom noise decay scheduler.
    * **Use & Why**: This is the primary way to inject a custom ancestral noise decay curve from the `NoiseDecayScheduler (Custom)` node. If provided, it overrides the `ancestral_noise_strength` parameter, using the custom schedule for noise injection. This provides much finer control over the "ping-pong" behavior. If not connected, `ancestral_noise_strength` is used.

* **ancestral_steps_start_ratio** (`FLOAT`, default: `0.0`, min: `0.0`, max: `1.0`, step: `0.001`)
    * **What it is**: The ratio of total steps at which ancestral noise injection *begins*.
    * **Use & Why**: A value of `0.0` means ancestral noise can be injected from the very first step. A value of `0.5` means ancestral noise only begins after 50% of the steps are completed. This allows you to delay the "ping-pong" effect.

* **ancestral_steps_end_ratio** (`FLOAT`, default: `1.0`, min: `0.0`, max: `1.0`, step: `0.001`)
    * **What it is**: The ratio of total steps at which ancestral noise injection *ends*.
    * **Use & Why**: A value of `1.0` means ancestral noise can be injected until the very last step. A value of `0.5` means ancestral noise stops after 50% of the steps. This is crucial for preventing excessive noise in the final stages of denoising, which can lead to artifacts.

* **ancestral_noise_strength** (`FLOAT`, default: `1.0`, min: `0.0`, max: `2.0`, step: `0.001`)
    * **What it is**: The baseline strength of ancestral noise to inject at each applicable step.
    * **Use & Why**: This parameter is active only if `optional_noise_decay_scheduler` is *not* connected. Higher values inject more noise, potentially leading to more "ancestral" characteristics or artifacts. Lower values make the process smoother.

* **blend_mode** (`ENUM`, `lerp`, `a_only`, `b_only`, default: `lerp`)
    * **What it is**: Determines how the noise prediction from the conditional (positive) and unconditional (negative) prompts are combined.
    * **Use & Why**:
        * `lerp` (Linear Interpolation): This is the standard blend, similar to what most samplers do for CFG. The final prediction is a linear mix of the conditional and unconditional.
        * `a_only`: Only uses the conditional (positive) prediction. This bypasses the negative prompt's influence almost entirely, which can lead to very strong prompt adherence but potentially less creative outputs or artifacts.
        * `b_only`: Only uses the unconditional (negative) prediction. This is highly experimental and will likely result in very noisy or unguided outputs, as it effectively ignores your positive prompt.

* **step_blend_mode** (`ENUM`, `lerp`, `a_only`, `b_only`, default: `lerp`)
    * **What it is**: Controls how predictions from previous steps or internal model states are blended with the current step's prediction. Primarily used in advanced internal calculations within the sampler.
    * **Use & Why**: This parameter is more subtle and generally affects the temporal coherence or stability of the generation. It's often left at `lerp` unless you are actively debugging or experimenting with specific temporal effects.

### Advanced (YAML) Override

* **yaml_settings_str** (`STRING`, Multiline, Optional)
    * **What it is**: A multiline text input allowing you to provide a YAML (YAML Ain't Markup Language) string to override any of the node's parameters.
    * **Use & Why**: This is a powerful feature for saving and loading complex sampler presets. You can define all node parameters within the YAML string, and they will take precedence over the individual node inputs. This is excellent for creating consistent workflows or sharing specific settings.
    * **Example YAML Structure**:
        ```yaml
        # Example PingPong Sampler YAML Settings
        steps: 30
        cfg: 7.5
        sampler_name: "dpmpp_2m"
        scheduler_name: "karras"
        ancestral_steps_start_ratio: 0.1
        ancestral_steps_end_ratio: 0.95
        ancestral_noise_strength: 0.8
        blend_mode: "lerp"
        step_blend_mode: "lerp"
        ```
    * **Note**: Ensure valid YAML syntax. Boolean values (e.g., `True`/`False`) should be represented as `true`/`false` in YAML.

---

## 3. In-Depth Technical Information

This section explores the underlying mechanics and algorithms that power the PingPong Sampler.

### Sampling Flow Overview

The PingPong Sampler integrates into ComfyUI by providing a custom `KSAMPLER` output. This means it essentially wraps a standard KSampler's functionality while injecting its specialized "ping-pong" and blending logic.

1.  **Input Parsing**: The node first reads all direct input parameters.
2.  **YAML Override Logic**: It then attempts to parse the `yaml_settings_str`. If valid YAML is found and contains parameters that also exist as direct node inputs, the YAML values take precedence and override the direct node inputs. This means you can paste a complex YAML configuration to instantly change multiple settings, acting as a powerful preset manager. Boolean values from YAML strings are automatically converted to Python booleans.
3.  **Blend Mode Resolution**: It resolves the `blend_mode` and `step_blend_mode` strings into actual `torch` functions (e.g., `torch.lerp`, or custom lambda functions for `a_only`/`b_only`) using the `_INTERNAL_BLEND_MODES` dictionary.
4.  **KSampler Construction**: Finally, it constructs and returns a `KSAMPLER` object, passing `PingPongSampler.go` as the sampling function and all the resolved parameters (after YAML merging) as `extra_options` (which are then unpacked as `**kwargs` into `PingPongSampler.go`).

### The `PingPongSampler.go` Function (The Core Logic)

This static method performs the actual sampling loop.

* **Ancestral Step Determination**: It determines whether the current step is within the `ancestral_steps_start_ratio` and `ancestral_steps_end_ratio` range. If so, ancestral noise injection is active.
* **Noise Decay Schedule**:
    * If `optional_noise_decay_scheduler` is provided, it calls `scheduler.get_decay(num_steps)` to get the custom noise decay curve.
    * It then calculates `noise_strength_mult = decay_curve[step]` to determine the specific multiplier for ancestral noise at the current step.
    * If no custom scheduler is provided, `noise_strength_mult` defaults to `ancestral_noise_strength`.
* **The "Ping-Pong" Action (Ancestral Noise Injection)**:
    * When an ancestral sampler is used (e.g., `euler_ancestral`) and ancestral steps are active:
        * After the model predicts the noise, a new `random_noise` tensor is generated.
        * This `random_noise` is then blended with the existing predicted noise based on `noise_strength_mult` and the `blend_mode`. This is where the core "ping-pong" effect happens, introducing controlled variability.
* **Conditional Blending**: The `blend_mode` and `step_blend_mode` influence how the model's predictions (conditional and unconditional) are combined, affecting the overall guidance.
* **Dynamic `model_options`**: The `PingPongSampler.go` function dynamically modifies `model_options` to pass its custom `cfg_function` (which implements the `blend_mode` logic) to the underlying KSampler. This allows the PingPong Sampler to override how the model's output is interpreted.

### Internal Blend Modes

The `_INTERNAL_BLEND_MODES` dictionary is a simple lookup table that maps user-friendly string names (`"lerp"`, `"a_only"`, `"b_only"`) to their corresponding Python functions.

* `"lerp"`: Uses `torch.lerp(a, b, weight)`, which is standard linear interpolation. `a` is typically the unconditional prediction, `b` is the conditional prediction, and `weight` is derived from `cfg`.
* `"a_only"`: Returns `a`. This means the `cfg_function` will effectively ignore the conditional (positive) prediction and rely solely on the unconditional (negative) prediction. This is usually not what you want for standard generation.
* `"b_only"`: Returns `b`. This means the `cfg_function` will ignore the unconditional (negative) prediction and rely solely on the conditional (positive) prediction. This essentially turns off negative prompting and can lead to over-saturation or less diverse results.

In summary, the PingPong Sampler is a finely tuned instrument for diffusion sampling, particularly adept at handling the temporal coherence required by audio and video models. Its internal logic, especially regarding noise injection and the "ping-pong" ancestral steps, is meticulously crafted to produce consistent and high-quality results.
