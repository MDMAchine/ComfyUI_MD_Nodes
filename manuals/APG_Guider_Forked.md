# APG Guider (Forked) ComfyUI Node Manual

---

## 1. What is the APG Guider (Forked) Node?

The APG Guider (Forked) node is a powerful tool for anyone looking to gain more precise control over the creative process in ComfyUI, especially when working with diffusion models for tasks like image, video, or audio generation.

### What it is:
Imagine you're sculpting. Standard tools (like CFG) give you broad control. The APG Guider is like a set of specialized chisels and brushes that allow you to refine your work with much greater detail, especially in how your AI model "understands" and "interprets" your creative directions (prompts). APG stands for Adaptive Projected Gradient, which is a fancy way of saying it intelligently adjusts how the AI tries to match your positive prompt while avoiding your negative prompt.

### How it works:
Diffusion models generate content by gradually removing noise from an initial noisy input, guided by your prompts. During this denoising process, the model takes many small "steps." The APG Guider doesn't just apply a single, constant guidance strength throughout these steps. Instead, it dynamically changes the way and strength of guidance based on the current "noise level" (or sigma) of the latent image, video, or audio.

It does this by:
1.  **Calculating a "difference"**: It figures out how much the positive guidance wants to change the output compared to the negative guidance.
2.  **Projecting this difference**: It then takes this difference and "projects" it. Think of it like shining a light on a 3D object from a specific angle to see its shadow. This projection helps the guidance focus on the most relevant aspects.
3.  **Applying "Adaptive" adjustments**: Based on internal calculations and your chosen parameters (like momentum or norm_threshold), it subtly nudges the generation process in the desired direction, preventing overshooting or introducing artifacts.

### What it does:
* **Enhanced Control**: Moves beyond basic CFG (Classifier-Free Guidance) to offer a more nuanced approach to guiding your diffusion process.
* **Dynamic Guidance**: Allows you to define different guidance behaviors at different stages of the denoising process (e.g., strong guidance early on, subtle refinements later).
* **Problem Solving**: Can help address issues like over-saturation, lack of detail, or "prompt adherence" problems that sometimes occur with standard CFG.
* **Creative Exploration**: Opens up new possibilities for artistic expression by allowing you to experiment with unique guidance profiles.

### How to use it:
The APG Guider node acts as a replacement for your standard `MODEL` input in your sampling workflow. Instead of connecting your raw model directly to your sampler, you connect your model, positive conditioning, and negative conditioning to the APG Guider node. The `apg_guider` output from this node then connects to the `model` input of your sampler (e.g., `SamplerCustom`, `KSampler`).

You can use it in two primary ways:
1.  **Simple Mode (Node Inputs)**: By adjusting the sliders and dropdowns directly on the node, you can set a single, continuous APG guidance profile that applies between a `start_sigma` and `end_sigma`.
2.  **Advanced Mode (YAML Parameters)**: For much finer control, you can define multiple "rules" in YAML format. Each rule specifies different APG parameters (like `apg_scale`, `mode`, `momentum`) to apply at specific `start_sigma` values. This allows for highly customized, stage-by-stage guidance throughout your sampling process. The YAML input overrides the individual node parameters if present.

---

## 2. Detailed Information about the Parameters

The APG Guider (Forked) node offers a range of parameters to fine-tune its behavior. Understanding each one is key to mastering this node.

### Node Inputs (Simple Mode Parameters)
These parameters define a single APG guidance profile. If you provide content in `yaml_parameters_opt`, these individual parameters will be overridden by your YAML rules.

* **model** (`MODEL`, Required)
    * **What it is**: This is the core diffusion model (e.g., a Stable Diffusion model checkpoint) that you want to apply APG guidance to.
    * **Use & How**: Connect your `MODEL` output from a `Load Checkpoint` or similar node here. This is the model whose noise prediction will be guided by APG.
    * **Why**: The APG Guider intercepts and modifies the model's behavior during the denoising steps, so it needs to know which model to operate on.

* **positive** (`CONDITIONING`, Required)
    * **What it is**: Your positive conditioning, typically derived from your text prompt (e.g., "a majestic castle, highly detailed").
    * **Use & How**: Connect the `CONDITIONING` output from your positive text encoder (e.g., `CLIPTextEncode (Prompt)`) here.
    * **Why**: APG guidance uses the positive conditioning to steer the generation process towards your desired output.

* **negative** (`CONDITIONING`, Required)
    * **What it is**: Your negative conditioning, typically derived from your negative text prompt (e.g., "blurry, low quality, deformed").
    * **Use & How**: Connect the `CONDITIONING` output from your negative text encoder here.
    * **Why**: APG guidance uses the negative conditioning to steer the generation process away from undesired features.

* **apg_scale** (`FLOAT`, default: `4.5`, min: `-1000.0`, max: `1000.0`, step: `0.1`)
    * **What it is**: This is the primary strength multiplier for the APG guidance. Similar to CFG scale, but specific to the APG calculation.
    * **Use & How**:
        * A value of `1.0` means no APG effect (it effectively becomes CFG-like in its base behavior).
        * Higher values increase the strength of APG guidance, pushing the generation more aggressively towards the positive conditioning and away from the negative.
        * If set to `0`, APG is effectively off for the steps where this rule applies.
        * Experiment with values between `1.0` and `10.0` as a starting point. Very high values can lead to artifacts or over-saturation, while very low values might not have enough impact.
    * **Why**: Controls how much the APG "chisel" carves into your latent space.

* **cfg_before** (`FLOAT`, default: `4.0`, min: `1.0`, max: `1000.0`, step: `0.1`)
    * **What it is**: The standard Classifier-Free Guidance (CFG) scale that will be used for sampling steps where the current noise level (sigma) is greater than your `start_sigma`.
    * **Use & How**: Set this to the CFG value you want to use before APG kicks in. For example, if you want APG to start at `start_sigma: 0.8` (higher noise levels), `cfg_before` will be active for all steps from the beginning until sigma reaches `0.8`.
    * **Why**: Provides a baseline CFG for the initial stages of denoising before APG becomes active, allowing for a smoother transition or different guidance strategies at different noise levels.

* **cfg_after** (`FLOAT`, default: `3.0`, min: `1.0`, max: `1000.0`, step: `0.1`)
    * **What it is**: The standard Classifier-Free Guidance (CFG) scale to use for sampling steps where the current noise level (sigma) is less than your `end_sigma`, or if APG is completely turned off for a particular rule.
    * **Use & How**: Set this to the CFG value you want to use after APG disengages. This allows you to finish the generation with a different CFG strength for final refinements.
    * **Why**: Useful for preventing artifacts that can sometimes occur at very low noise levels with strong APG, or for applying a different final aesthetic.

* **eta** (`FLOAT`, default: `0.0`, min: `-1000.0`, max: `1000.0`, step: `0.1`)
    * **What it is**: A parameter that influences how the guidance vector is projected. It controls the mix between the orthogonal component and a component parallel to the prediction difference.
    * **Use & How**:
        * Values closer to `0.0` (the default) emphasize guidance that is "orthogonal" (perpendicular) to the difference between conditional and unconditional predictions. This often leads to more stable and focused guidance.
        * Non-zero values introduce a component of guidance that is "parallel" to the prediction difference. Experimentation is key, as this can lead to unique visual effects or instabilities depending on the model and content.
    * **Why**: Provides a fine-grained control over the directionality of the APG guidance in the latent space.

* **norm_threshold** (`FLOAT`, default: `2.5`, min: `-1000.0`, max: `1000.0`, step: `0.1`)
    * **What it is**: If the "strength" (L2 norm) of the calculated APG guidance vector exceeds this threshold, the vector is scaled down (clamped) to prevent it from becoming too strong.
    * **Use & How**:
        * Set to a positive value (e.g., `2.5`) to prevent overly aggressive guidance that can lead to blown-out highlights, oversaturation, or other artifacts. It acts as a safety brake.
        * Set to `0.0` or a negative value to disable thresholding, allowing the guidance vector to be as strong as calculated. This can sometimes be useful for strong stylistic effects but carries a higher risk of artifacts.
    * **Why**: Essential for maintaining image quality and preventing guidance from "running away" with the generation, especially during sensitive stages of denoising.

* **momentum** (`FLOAT`, default: `0.75`, min: `-1000.0`, max: `1000.0`, step: `0.01`)
    * **What it is**: Applies a running average to the APG guidance updates, influencing how much past guidance influences the current step.
    * **Use & How**:
        * A positive value (e.g., `0.5` to `0.9`) introduces a smoothing effect, making the guidance more consistent across steps. This can lead to cleaner results and prevent flickering in video/audio.
        * A negative value (e.g., `-0.5` to `-0.9`) can make the guidance more reactive or "sharper," potentially emphasizing details but also risking instability.
        * `0.0` disables momentum, meaning each step's guidance is calculated independently.
    * **Why**: Useful for achieving smoother transitions, reducing noise, or creating more aggressive, detailed results.

* **start_sigma** (`FLOAT`, default: `-1.0`, min: `-1.0`, max: `10000.0`, step: `0.01`)
    * **What it is**: The noise level (sigma) at which the APG guidance defined by the current node's parameters begins to be applied.
    * **Use & How**:
        * If set to `-1.0` or any negative value, APG guidance will be active from the very beginning of the sampling process (effectively sigma = infinity). This is a common setting if you want APG to influence the entire generation.
        * Set a positive value (e.g., `0.8`, `0.5`) to have APG kick in only when the image has been denoised to a certain degree. This allows `cfg_before` to handle the very noisy stages.
    * **Why**: Enables stage-based guidance, allowing different guidance strategies at different noise levels, which can be crucial for quality.

* **end_sigma** (`FLOAT`, default: `-1.0`, min: `-1.0`, max: `10000.0`, step: `0.01`)
    * **What it is**: The noise level (sigma) at which the APG guidance defined by the current node's parameters ends. Below this sigma, the `cfg_after` value will take over.
    * **Use & How**:
        * If set to `-1.0` or any negative value, APG guidance will remain active until the very end of the sampling process.
        * Set a positive value (e.g., `0.1`, `0.05`) to have APG stop at a specific noise level, allowing standard CFG (`cfg_after`) to handle the final very subtle refinements.
    * **Why**: Can prevent artifacts or over-processing that might occur with strong APG at very low noise levels.

* **dims** (`STRING`, default: `"-1, -2"`, tooltip: "Comma-separated list of dimensions (e.g., -1 for width, -2 for height) to normalize the guidance vector along.")
    * **What it is**: Specifies which dimensions of the latent space the guidance vector's normalization (and projection) should operate on.
    * **Use & How**:
        * `-1` typically refers to the width dimension of the latent space.
        * `-2` typically refers to the height dimension of the latent space.
        * The default `"-1, -2"` means the guidance is normalized across the spatial dimensions (height and width), which is standard for image generation.
        * For other modalities (like audio), you might need to adjust this depending on the latent shape. There is no error checking on the input string, so ensure correct integer format (e.g., `"-1"`, `"-2"`, `"0, 1"`).
    * **Why**: Critical for ensuring the APG guidance is applied meaningfully across the relevant dimensions of your latent data, whether it's an image, video frame, or audio chunk.

* **predict_image** (`BOOLEAN`, default: `True`)
    * **What it is**: Determines whether the APG guidance is based on a prediction of the final denoised latent (image) or a prediction of the noise itself.
    * **Use & How**:
        * `True` (default): APG guides based on the predicted image/denoised latent. This often yields better, more coherent results, especially for image generation.
        * `False`: APG guides based on the predicted noise. This can sometimes lead to different stylistic effects or be useful for specific experimental setups.
    * **Why**: Influences the fundamental target of the APG guidance, which can significantly alter the outcome. Only has an effect in "pure_apg" mode.

* **mode** (Dropdown: `"pure_apg"`, `"pre_cfg"`, `"pure_alt1"`, `"pre_alt1"`, `"pure_alt2"`, `"pre_alt2"`, default: `"pure_apg"`)
    * **What it is**: Defines the specific algorithmic approach APG uses to inject guidance.
    * **Use & How**:
        * `pure_apg`: Applies APG guidance directly to the noise prediction, modifying how the model moves from noise to image. This is a common and robust mode.
        * `pre_cfg`: Modifies the conditional input before the standard CFG calculation. In this mode, the `apg_scale` acts more like a CFG scale. This can lead to a different feel in guidance, sometimes more subtle or integrated.
        * `pure_alt1` / `pre_alt1` / `pure_alt2` / `pre_alt2`: These are experimental alternative "update modes" for how APG's internal momentum is applied. They may produce unique or unexpected results. Experiment with these if `pure_apg` doesn't give you the desired outcome, but be prepared for varied effects.
    * **Why**: Offers different mathematical frameworks for applying guidance, each with its own characteristics and potential for different outputs.

### Optional Input (Advanced Mode Parameter)
* **yaml_parameters_opt** (`STRING`, Multiline, Default: Empty)
    * **What it is**: A text input area where you can define multiple, complex APG guidance "rules" using YAML (YAML Ain't Markup Language) syntax. This provides the most granular control.
    * **Use & How**:
        * **YAML Syntax**: YAML uses indentation to define structure. A list of rules starts with a hyphen (`-`). Each rule is a dictionary of key-value pairs.
        * **Example Structure** (from node header):
            ```yaml
            # verbose: true # Uncomment to see debug messages in ComfyUI console.
            rules:
               - start_sigma: -1.0 # Applies from the start
                 apg_scale: 0.0
                 cfg: 4.0
               - start_sigma: 0.85 # New rule kicks in at sigma 0.85
                 apg_scale: 5.0
                 predict_image: true
                 mode: pre_alt2
                 update_blend_mode: lerp
                 dims: [-2, -1]
                 momentum: 0.7
                 norm_threshold: 3.0
                 eta: 0.0
               - start_sigma: 0.70 # Another rule at sigma 0.70
                 apg_scale: 4.0
                 # ... other parameters for this rule
            ```
        * **Rule Parameters**: Inside each rule, you can specify almost any of the parameters listed above (`start_sigma`, `apg_scale`, `cfg`, `momentum`, `eta`, `norm_threshold`, `dims`, `predict_image`, `mode`, `update_blend_mode`).
        * `start_sigma` is Key: Each rule is triggered when the current sigma value in the sampler drops below or equals that rule's `start_sigma`. Rules are sorted by `start_sigma`, so they will apply in order.
        * `cfg` vs `apg_scale`: In YAML rules, you can explicitly set `cfg` for a rule. If `apg_scale` is `0.0` or `apg_blend` is `0.0` (which is not exposed as a direct input but is `1.0` by default), only `cfg` will apply.
        * `update_blend_mode`: This is a YAML-specific parameter used with `pure_alt2`/`pre_alt2` modes. It determines how the momentum update blends. Options are `lerp` (linear interpolation), `a_only`, `b_only`. `lerp` is generally recommended.
        * **Overrides**: Parameters defined in `yaml_parameters_opt` take precedence over the individual node inputs. If a parameter is not specified in a YAML rule, it will revert to its internal default for that rule.
    * **Why**: Provides ultimate flexibility to create complex, multi-stage guidance profiles that adapt precisely to the denoising process, offering unparalleled control and creative possibilities.

---

## 3. In-Depth Nerd Technical Information

For those who want to peek under the hood and understand the mathematical and architectural nuances of the APG Guider (Forked) node, this section delves into the technical details.

### Core Concepts
* **Adaptive Projected Gradient (APG)**:
    * At its heart, APG is a guidance mechanism that goes beyond simply scaling the difference between conditional and unconditional noise predictions (as in CFG).
    * It calculates a "gradient" (direction of change) that moves the latent state towards the positive prompt and away from the negative.
    * The "Projected Gradient" aspect refers to decomposing this gradient into components that are orthogonal (perpendicular) and parallel to a reference vector (often the conditional prediction). By primarily using the orthogonal component, APG aims to steer the generation without simply amplifying existing features or creating artifacts due to excessive parallel movement.
    * The "Adaptive" part comes from its ability to use momentum, norm thresholding, and different update modes to dynamically adjust the guidance strength and direction throughout the sampling process.

* **sigma (Noise Level)**:
    * In diffusion models, sigma represents the standard deviation of the noise added to the latent image.
    * Sampling starts at a high sigma (very noisy latent) and iteratively moves to a low sigma (denoised latent).
    * The APG Guider strategically applies different rules based on the current sigma value, allowing for precise control at various stages of the denoising process.

* **CFGGuider Integration**:
    * The `APGGuider` class inherits from ComfyUI's `CFGGuider`. This is crucial for its integration into the ComfyUI sampling pipeline.
    * By overriding the `predict_noise` method, `APGGuider` intercepts the noise prediction calls, injects its custom APG logic, and then calls the `super().predict_noise` (the original `CFGGuider`'s noise prediction) with the modified parameters or inputs.

### Key Components and Mechanics
* **APGConfig (NamedTuple)**:
    * This `NamedTuple` serves as a structured container for all the parameters associated with a single APG rule. It ensures type safety and readability.
    * The `fixup_param` static method handles special parsing, such as converting `start_sigma` values less than `0` to `math.inf` (representing the start of sampling) and parsing the `dims` string into a tuple of integers.
    * The `build` class method is a constructor that allows for flexible instantiation of `APGConfig` objects, mapping mode strings (like `"pure_apg"`, `"pre_alt2"`) to their internal enum representations (`UpdateMode`).

* **APG Class**:
    * This class encapsulates the core APG logic for a single rule. An `APGGuider` instance holds a tuple of `APG` instances, one for each defined rule.
    * `running_average`: Manages the momentum. It stores a stateful running average of the (cond - uncond) difference, which is updated at each step. This allows for temporal smoothing or sharpening of the guidance.
    * `update(val: torch.Tensor)`: Applies the momentum.
        * If `momentum` is `0`, `val` is returned directly.
        * If `running_average` is a `float` (initial state) or mismatching in type/shape, it's initialized with `val.clone()`.
        * `UpdateMode.DEFAULT`: `result = val + self.momentum * avg`, and `self.running_average = result`. This is a simple exponential moving average.
        * `UpdateMode.ALT1`: `self.running_average = val + abs(self.momentum) * avg`. This variant changes how the running average itself is updated, potentially leading to different convergence behavior.
        * `UpdateMode.ALT2`: Utilizes external `BLEND_MODES` (from `blepping_integrations` if available, otherwise a fallback `lerp`). `result` and `self.running_average` are both computed using blending functions, offering more complex momentum dynamics.
    * `project(v0_orig: torch.Tensor, v1_orig: torch.Tensor)`:
        * Performs the projection. `v1_orig` is normalized along `self.dims`.
        * `v0_p` (parallel component) is calculated as `(v0 * v1).sum(...) * v1`.
        * `v0_o` (orthogonal component) is `v0 - v0_p`.
        * Crucially, tensors are temporarily cast to double precision on "mps" (Apple Silicon) devices for numerical stability during projection, then converted back to original dtype/device.
    * `apg(cond: torch.Tensor, uncond: torch.Tensor)`:
        * Calculates `pred_diff = self.update(cond - uncond)`. This is the conditional-unconditional difference, potentially smoothed by momentum.
        * Applies `norm_threshold`: If `diff_norm` exceeds `norm_threshold`, `pred_diff` is scaled down. This prevents explosion of gradients and preserves image quality.
        * Performs projection: `diff_p, diff_o = self.project(pred_diff, cond)`.
        * **Key Fork Change**: The original line `update += self.eta * diff_p` was removed. This means the guidance primarily relies on the orthogonal component (`diff_o`), simplifying the projection's effect and providing a cleaner guidance path. The `eta` parameter, while still present, now only influences the calculation of the parallel component within `project` but that parallel component is not added back to the final update returned by `apg` directly. Its effect is more subtle in this forked version, influencing the `v0_p` during projection, which is then subtracted from `v0` to get `v0_o`.
        * Returns `diff_o` (the orthogonal component), which is the core APG guidance vector.
    * `cfg_function(args: dict)`:
        * This function is passed to `model_options["sampler_cfg_function"]` when the current rule's mode is `"pure_apg"`.
        * It determines whether to predict image or noise based on `self.predict_image`.
        * The core calculation is `cond + (self.apg_scale - 1.0) * self.apg(cond, uncond)`. This effectively scales the APG guidance and adds it to the conditional prediction.
        * If `predict_image` is `True`, it returns `args["input"] - result` (denoised latent - APG-modified prediction). If `False`, it returns `result` (the APG-modified noise prediction).
    * `pre_cfg_function(args: dict)`:
        * This function is appended to `model_options["sampler_pre_cfg_function"]` when the current rule's mode is `"pre_cfg"`.
        * It modifies `cond_apg = uncond + update + (cond - uncond) / self.apg_scale`. This adjusts the conditional input before the standard CFG calculation, allowing APG to influence the initial conditional vector that enters the CFG loop.

* **APGGuider Class (Main Guider)**:
    * Initializes with a list of `APG` instances (`self.apg_rules`) derived from the `APGConfig` rules.
    * `apg_reset()`: Resets the `running_average` of all `APG` rules, ensuring a clean state before and after each sampling process, and preventing state leakage between samples. An `exclude` parameter allows a specific rule to maintain its state.
    * `apg_get_match(sigma: float)`: Iterates through the sorted `self.apg_rules` and returns the first rule whose `start_sigma` is greater than or equal to the current sigma. This implements the stage-based rule application.
    * `outer_sample()`: Calls `apg_reset()` before and after `super().outer_sample()` to manage the state of momentum-based rules, ensuring consistent behavior for each new generation.
    * `predict_noise(x, timestep, ...)`: This is the most critical method, overriding the base `CFGGuider`'s logic.
        * It determines the current sigma from the timestep.
        * It finds the active rule using `apg_get_match(sigma)`.
        * It calls `apg_reset(exclude=rule)` to reset momentum for other rules, ensuring only the currently active rule maintains its momentum state.
        * **Dynamic model_options Modification**:
            * If APG is active for the current rule (`rule.apg_blend != 0` and `rule.apg_scale != 0`), it sets `model_options["disable_cfg1_optimization"] = True`. This is important because ComfyUI's internal CFG=1 optimization can interfere with custom guidance functions.
            * If `rule.pre_cfg_mode` is `True`, it appends `rule.pre_cfg_function` to `sampler_pre_cfg_function` in `model_options`.
            * Otherwise (for `pure_apg` modes), it sets `model_options["sampler_cfg_function"] = rule.cfg_function`.
            * It dynamically sets `self.cfg` (the base `CFGGuider`'s CFG value) to the `rule.cfg` or `rule.apg_scale` (for `pre_cfg` mode) for the current step, then restores it in a `finally` block.
            * Finally, it calls `super().predict_noise` with the potentially modified `model_options` and adjusted `self.cfg`, allowing the base sampler to perform the actual noise prediction under the influence of APG.

### Internal Logic and Potential Implementations
The code contains internal logic and parameters not directly exposed as node inputs, such as:
* `apg_blend`: This parameter within `APGConfig` controls the blend strength of APG. It is currently set to `1.0` by default when building rules from node inputs, meaning full APG effect. If `yaml_parameters_opt` is used, it can be set per rule. Future versions might expose this as a direct input.
* `apg_blend_mode`: Similar to `update_blend_mode`, this governs how the `apg_blend` is applied. Defaulted to `lerp`. Not currently exposed.
* `verbose`: A `params` option that can be set in the YAML input to enable debug messages in the ComfyUI console. This is useful for troubleshooting and understanding rule activation.

These internal parameters could, in theory, be exposed as direct node inputs in future updates, offering even more granular control.
