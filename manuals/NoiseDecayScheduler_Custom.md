# ComfyUI Noise Decay Scheduler (Custom) Manual

---

## 1. What is the Noise Decay Scheduler (Custom) Node?

The "Noise Decay Scheduler (Custom)" is a specialized ComfyUI node designed to control how noise is faded out during the sampling process in diffusion models. Think of it like a dimmer switch for the "ancestral" noise in your generation: it doesn't remove the noise entirely, but rather dictates how quickly or slowly its influence diminishes over the steps of your sampling process.

### How it Works:

At its core, this node generates a mathematical curve based on a cosine function, which is then shaped by a single parameter: `decay_power`. This curve acts as a "schedule" for the noise. When connected to a compatible sampler (like `pingpongsampler_custom`), the sampler uses this schedule to adjust the amount of noise applied at each step.

### What it Does:

By manipulating the `decay_power` parameter, you can influence the visual or auditory characteristics of your generated output.

* **Steeper decay curves (higher `decay_power`)** mean the noise reduces more quickly in the early stages of sampling. This can lead to outputs that are:
    * Sharper and more detailed: Less noise influence for longer means the model refines the output more aggressively.
    * Potentially less "noisy" overall: The final output might have a cleaner appearance.
* **Gentler decay curves (lower `decay_power`)** prolong the influence of noise throughout the sampling process. This can result in outputs that are:
    * Softer or more "dreamlike": The continued noise influence can blend elements more.
    * Potentially more prone to artifacts: If too much noise is left in for too long, it can interfere with coherent generation.

### How to Use It:

1.  **Add the Node**: In your ComfyUI workflow, right-click on the graph, go to `MD_Nodes -> Schedulers`, and select `Noise Decay Scheduler (Custom)`.
2.  **Connect to Sampler**: Connect the `SCHEDULER` output of this node to the `noise_decay_scheduler` input of a compatible sampler node (e.g., `pingpongsampler_custom`).
3.  **Adjust `decay_power`**: Experiment with the `decay_power` parameter to observe its effect on your generated output. Start with small adjustments and gradually increase or decrease to find the desired aesthetic.

---

## 2. Detailed Parameter Information

The Noise Decay Scheduler (Custom) node exposes a single, crucial parameter that controls its behavior.

* **decay_power**
    * **Type**: `FLOAT`
    * **Default**: `0.75`
    * **Min**: `0.0`
    * **Max**: `2.0`
    * **Step**: `0.01`
    * **Description**: This parameter controls the "aggressiveness" or "steepness" of the noise decay curve. It directly influences how quickly the ancestral noise component diminishes over the sampling steps.
    * **Use Cases & Why**:
        * **`0.0` (Linear Decay)**: The noise influence decays linearly across the steps.
        * **`1.0` (Cosine Decay)**: A standard cosine-based decay, where noise reduction is initially slower and then accelerates.
        * **Greater than `1.0` (Steeper Decay)**: The noise influence diminishes very rapidly in the early steps, then levels off. This can lead to outputs that are quickly refined and might appear sharper.
        * **Less than `1.0` (Gentler Decay)**: The noise influence persists longer into the sampling process. This can lead to softer, more blended results, or introduce more textural noise depending on the context.
    * **How to Use**: Experimentation is key. Start with the default `0.75` and incrementally adjust. Higher values make the decay steeper, lower values make it gentler.
    * **Analogy**: Imagine a hill. `decay_power` shapes how steep the hill is. If it's a very steep hill (high `decay_power`), you drop fast at the beginning. If it's a gentle slope (low `decay_power`), you descend slowly throughout.

---

## 3. In-Depth Technical Information

This section provides a more detailed look at the mathematical and architectural aspects of the Noise Decay Scheduler (Custom) node.

### Mathematical Basis: Cosine Decay

The core of this scheduler is a cosine-based decay function. This type of function is often chosen in signal processing and machine learning for its smooth, non-linear progression.

The internal calculation for the decay curve generally follows a pattern similar to:

`decay_factor = (1 - cos(x)) / 2`

where `x` ranges from `0` to `π` (or `0` to `2π`) over the course of the sampling steps. The `decay_power` parameter is then applied to this `decay_factor` (e.g., `decay_factor^decay_power`) to further shape the curve.

* When `decay_power` is `1.0`, it's a standard cosine decay.
* When `decay_power` is greater than `1.0`, the curve becomes concave, meaning initial drops are larger.
* When `decay_power` is less than `1.0`, the curve becomes convex, meaning initial drops are smaller.

### Integration with Samplers

The node does not directly apply noise reduction. Instead, it generates a `SCHEDULER` object. This object is designed to be passed to a compatible sampler (like `pingpongsampler_custom`).

The `SCHEDULER` object typically contains a method (e.g., `get_decay(num_steps)`) that, when called by the sampler, generates the actual noise decay curve (a NumPy array or PyTorch tensor) with the specified number of steps. This allows the scheduler to be flexible and adapt to the number of steps chosen in the sampler, rather than being fixed.

### Design Principles

* **Lazy Generation**: The actual noise decay curve is not calculated until the sampler explicitly requests it and provides the `num_steps`. This is efficient as the decay curve is only generated when it's actually needed by the sampler.
* **NumPy-Powered**: The node relies solely on `NumPy` for numerical operations. This makes it more lightweight and potentially more compatible across different environments without requiring a full PyTorch installation for just the scheduler logic.
* **Object-Oriented Output**: Instead of returning a raw tensor (as in older versions), the node returns a `SCHEDULER` object. This is a cleaner, more encapsulated approach. The object contains the logic and the `decay_power` parameter, allowing the sampler to interact with it via a defined interface (`get_decay` method).

### Internal (Non-Exposed) Parameters/Concepts:

While the node's internal logic utilizes concepts like linear spacing for `0` to `π` radians, these are part of the implementation of the cosine decay and are not intended as direct user-adjustable parameters. They are fixed elements of how the decay curve is shaped.

### Future Implementations (May or May Not Be):

Currently, the node focuses on a single `decay_power` parameter for a cosine-based decay. Potential future implementations might explore:

* Alternative decay functions (e.g., exponential, polynomial).
* More complex multi-parameter decay controls.
* Ability to input custom decay arrays.

However, these are speculative and not part of the current node's functionality.
