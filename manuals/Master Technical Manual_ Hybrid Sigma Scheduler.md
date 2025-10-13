---

## **Master Technical Manual: Hybrid Sigma Scheduler**

### **Node Name: HybridAdaptiveSigmas**

Display Name: Hybrid Sigma Scheduler  
Category: MD\_Nodes/Schedulers  
Version: 0.89  
Last Updated: 2025-09-15

---

### **Table of Contents**

1. Introduction  
   1.1. Executive Summary  
   1.2. Conceptual Category  
   1.3. Problem Domain & Intended Application  
   1.4. Key Features (Functional & Technical)  
2. Core Concepts & Theory  
   2.1. Theoretical Background  
   2.2. Mathematical & Algorithmic Formulation  
   2.3. Data I/O Deep Dive  
   2.4. Strategic Role in the ComfyUI Graph  
3. Node Architecture & Workflow  
   3.1. Node Interface & Anatomy  
   3.2. Input Port Specification  
   3.3. Output Port Specification  
   3.4. Workflow Schematics (Minimal & Advanced)  
4. Parameter Specification  
   4.1. Parameter: model  
   4.2. Parameter: steps  
   4.3. Parameter: denoise\_mode  
   4.4. Parameter: mode  
   4.5. Parameter: denoise  
   4.6. Parameter: start\_at\_step  
   4.7. Parameter: end\_at\_step  
   4.8. Parameter: min\_sliced\_steps  
   4.9. Parameter: detail\_preservation  
   4.10. Parameter: low\_denoise\_color\_fix  
   4.11. Parameter: split\_schedule  
   4.12. Parameter: mode\_b  
   4.13. Parameter: split\_at\_step  
   4.14. Parameter: start\_sigma\_override  
   4.15. Parameter: end\_sigma\_override  
   4.16. Parameter: rho  
   4.17. Parameter: blend\_factor  
   4.18. Parameter: power  
   4.19. Parameter: threshold\_noise  
   4.20. Parameter: linear\_steps  
   4.21. Parameter: reverse\_sigmas  
5. Applied Use-Cases & Recipes  
   5.1. Recipe 1: Standard High-Quality Text-to-Image Generation  
   5.2. Recipe 2: Precise Img2Img Denoising with Schedule Slicing  
   5.3. Recipe 3: Late-Stage Detail Enhancement using a Polynomial Curve  
6. Implementation Deep Dive  
   6.1. Source Code Walkthrough  
   6.2. Dependencies & External Calls  
   6.3. Performance & Resource Analysis  
   6.4. Tensor Lifecycle Analysis  
7. Troubleshooting & Diagnostics  
   7.1. Error Code Reference  
   7.2. Unexpected Visual Artifacts & Behavior

---

### **1\. Introduction**

#### **1.1. Executive Summary**

The **Hybrid Sigma Scheduler** is a highly configurable utility node for ComfyUI that provides precise, expert-level control over the noise schedule used in the diffusion sampling process. It functions as a procedural generator for the sequence of sigma (σ) values that a sampler uses to denoise a latent image. By offering multiple mathematical modes for curve generation (e.g., Karras, Polynomial, Blended), automatic detection of a model's native sigma range, and advanced controls for slicing and reshaping the schedule, this node allows users to move beyond default sampler behavior. It enables the fine-tuning of the entire denoising trajectory, directly impacting the final image's detail, coherence, and stylistic character. Its key outputs, the sigmas tensor and the actual\_steps integer, are designed to integrate seamlessly with custom sampler nodes that accept an external sigma schedule.

#### **1.2. Conceptual Category**

**Diffusion Process Controller / Sampler Utility.** This node is a procedural generator that does not process image or latent data itself. Instead, it creates a critical control signal—the sigmas tensor—that dictates the behavior of a downstream sampler node. It is a fundamental building block for constructing advanced, customized sampling workflows.

#### **1.3. Problem Domain & Intended Application**

* **Problem Domain:** The default schedulers built into ComfyUI's standard samplers offer limited to no user control. They operate as black boxes, generating a fixed schedule based on the number of steps. This lack of control is a significant limitation in advanced workflows. For instance, in an img2img task, simply reducing the number of steps to match the denoise value creates a completely new, compressed schedule, rather than correctly using a *segment* of the full, optimal schedule. This can lead to a mismatch in the noise levels and degrade quality. The Hybrid Sigma Scheduler solves this by externalizing schedule generation, providing explicit tools for slicing, reshaping, and overriding the schedule to perfectly match the workflow's intent.  
* **Intended Application (Use-Cases):**  
  * **High-Fidelity Text-to-Image:** Generating a mathematically precise and model-appropriate noise schedule (like karras\_rho or kl\_optimal) that is often superior to a sampler's default.  
  * **Precision Img2Img and Inpainting:** Slicing a full schedule using start\_at\_step to perfectly correspond to a custom sampler's denoise value, ensuring the sampler operates on the correct noise levels for the given task.  
  * **Multi-Stage Workflows:** Creating partial schedules for multi-pass techniques, such as a latent upscale workflow where one schedule handles the initial pass and a second, sliced schedule handles a high-resolution detail pass.  
  * **Aesthetic Exploration:** Experimenting with different curve shapes (polynomial, exponential, etc.) to intentionally alter the character of the generation, for example, by concentrating steps in the low-noise region to enhance detail.  
* **Non-Application (Anti-Use-Cases):**  
  * The node has no function without a downstream sampler (like SamplerCustom) designed to consume its sigmas output.  
  * The node cannot be used with the standard KSampler node, as it lacks the necessary sigmas input to accept an external schedule.

#### **1.4. Key Features (Functional & Technical)**

* **Functional Features:**  
  * **Multiple Generation Modes:** Provides a comprehensive suite of scheduler algorithms, including the industry-standard karras\_rho, polynomial curves, and mathematically-derived schedules like kl\_optimal.  
  * **Automatic Model Calibration:** Reads the diffusion model's intended sigma\_min and sigma\_max values directly from the model object, removing guesswork and ensuring the generated schedule is correctly scaled.  
  * **Precision Schedule Slicing:** Allows users to discard steps from the beginning (start\_at\_step) or end (end\_at\_step) of a full schedule, which is essential for img2img and other partial denoising tasks.  
  * **Synchronized Step Output:** Outputs the exact number of steps (actual\_steps) remaining in the schedule *after* slicing, which must be piped to a custom sampler to ensure perfect synchronization between the schedule length and the sampler's step count.  
  * **Split Scheduling:** An advanced feature (split\_schedule) to combine two different scheduler modes within a single generation, transitioning from one curve shape to another at a specific step.  
* **Technical Features:**  
  * **Model-Aware Sigma Detection:** Accesses the model.get\_model\_object("model\_sampling") object to retrieve the sigma\_min and sigma\_max attributes, providing a robust, model-specific baseline for all calculations.  
  * **Denoise Mode Logic:** Implements three distinct modes (Hybrid, Subtractive, Repaced) for interpreting the denoise parameter, offering flexible strategies for how a partial denoise operation is translated into a sigma schedule.  
  * **Procedural Tensor Generation:** The core of the node is a set of functions that procedurally generate a 1D PyTorch tensor (sigmas) based on mathematical formulas (e.g., torch.linspace, pow, tan).  
  * **Safe Fallbacks:** Includes hard-coded fallback sigma\_min and sigma\_max values to ensure functionality even if a connected model does not expose its sigma range correctly.

### **2\. Core Concepts & Theory**

#### **2.1. Theoretical Background**

* **Diffusion and Sigmas (σ):** The generative process of a diffusion model is a denoising sequence. It begins with a latent tensor of pure Gaussian noise, which has a very high standard deviation, or sigma. At each step, the model predicts a slightly less noisy version of the latent, and the sampler moves the latent towards that prediction. The sequence of sigma values from high to low is the **noise schedule**. This schedule is the roadmap for the denoising process.  
* **The Importance of the Schedule's Shape:** The distribution of sigma values over the given steps—the "shape" of the schedule—is non-trivial. The distance between consecutive sigma values determines how large a "jump" the sampler makes in the latent space.  
  * **Large jumps (steep curve)** at high sigma values allow the model to quickly establish the global composition and major forms from the initial chaos.  
  * Small jumps (flat curve) at low sigma values force the sampler to take smaller, more careful steps, allowing the model to refine intricate details and textures without making drastic changes that could disrupt the already-formed image.  
    The various modes (karras\_rho, polynomial, etc.) are different mathematical formalisms for generating curves with these desirable properties.  
* **Karras Schedule (karras\_rho):** This refers to the schedule proposed in the paper "Elucidating the Design Space of Diffusion-Based Generative Models" (Karras et al., 2022). It is a perceptually-informed schedule that is known to achieve high-quality results in fewer steps than simpler schedules. The rho (ρ) parameter is a power that controls the steepness of the curve, allowing it to be tuned. It has become a de facto standard for high-quality image generation.  
* **Slicing vs. Rescaling:** A critical concept for img2img is the difference between slicing a schedule and rescaling it. When performing img2img with a denoise of 0.5, one should not simply generate a new, shorter schedule that covers the full sigma range. The correct approach, which this node enables, is to generate a **full** schedule and then **slice** it, using only the first 50% of the sigma values. This ensures the sampler operates in the intended high-noise region of the model's training, leading to far more coherent and predictable results.

#### **2.2. Mathematical & Algorithmic Formulation**

The node generates a tensor of sigma values, Σ=\[σ0​,σ1​,…,σN​\], where N is the number of steps.

1. Linear Schedule: The simplest form, a linear interpolation between σmax​ and σmin​.

   σi​=σmax​−i⋅Nσmax​−σmin​​fori=0,…,N  
2. Polynomial Schedule: A curve defined by a power, p.

   σi​=(σmax1/p​−i⋅Nσmax1/p​−σmin1/p​​)p

   A power p=1.0 is equivalent to a linear schedule. p\>1.0 concentrates steps at high sigmas, while p\<1.0 concentrates them at low sigmas.  
3. Karras Schedule: A more complex formula based on the rho (ρ) parameter.

   σi​=(σmax1/ρ​+N−1i​(σmin1/ρ​−σmax1/ρ​))ρ

   The additional sigma value at the end (for a total of N+1 points) is typically 0\.  
4. Schedule Slicing: Given a full schedule Σfull​ of length N+1, and parameters start\_at\_step (Sstart​) and end\_at\_step (Send​), the final sliced schedule Σsliced​ is:

   Σsliced​=Σfull​\[Sstart​:Send​\]

   The actual\_steps is then calculated as length(Σsliced​)−1.

#### **2.3. Data I/O Deep Dive**

* **Input:**  
  * model (MODEL): A reference to the loaded diffusion model object. The node specifically queries this object for its model\_sampling attribute to get .sigma\_min and .sigma\_max.  
* **Outputs:**  
  * sigmas (SIGMAS): This is a dedicated ComfyUI data type for noise schedules.  
    * **Tensor Specification:** It is a 1-dimensional PyTorch tensor.  
    * **Shape:** \[actual\_steps \+ 1\]. For example, if actual\_steps is 25, the tensor will have a shape of \[26\].  
    * **dtype:** torch.float32.  
    * **Value Range:** A monotonically decreasing sequence of values from a sigma\_max approximation down to a sigma\_min approximation.  
  * actual\_steps (INT):  
    * **Data Specification:** A standard Python integer. It is **not** a tensor. This value represents the number of denoising steps the sampler should perform, which is one less than the number of points in the sigmas tensor.

#### **2.4. Strategic Role in the ComfyUI Graph**

* **Placement Context:** This node acts as a **pre-sampler controller**. It must be placed before the sampler in the workflow. Its primary function is to prepare the control signals (sigmas and steps) that the sampler will use.  
* **Integration with Custom Samplers:** This node is designed to work with samplers that expose a sigmas input socket, such as **SamplerCustom** or its advanced variants. This integration pattern allows the scheduler node to completely take over the responsibility of defining the denoising path from the sampler. This is fundamentally different from a GUIDER node (like APG Guider), which connects to a guider input to replace the sampler's *guidance calculation logic*. The Hybrid Sigma Scheduler provides the *path*, while a GUIDER provides the *steering* along that path.  
* **Synergistic Nodes:**  
  * **SamplerCustom / SamplerCustomAdvanced:** The primary consumers of this node's outputs. The sigmas output must connect to the sigmas input on the custom sampler, and the actual\_steps output must connect to the steps input to ensure the sampler executes the correct number of iterations.  
* **Conflicting Nodes:**  
  * **Standard KSampler:** This node is incompatible with the standard KSampler, as KSampler lacks the required sigmas input socket and generates its own schedule internally based on its dropdown menu. Attempting to use this node without a compatible custom sampler will render it non-functional.

### **3\. Node Architecture & Workflow**

#### **3.1. Node Interface & Anatomy**

1. model (Input Port)  
2. steps (Integer Widget)  
3. denoise\_mode (Dropdown Widget)  
4. mode (Dropdown Widget)  
5. denoise (Float Widget)  
6. start\_at\_step (Integer Widget)  
7. end\_at\_step (Integer Widget)  
8. min\_sliced\_steps (Integer Widget)  
9. detail\_preservation (Float Widget)  
10. low\_denoise\_color\_fix (Float Widget)  
11. split\_schedule (Toggle Widget)  
12. mode\_b (Dropdown Widget)  
13. split\_at\_step (Integer Widget)  
14. start\_sigma\_override (Float Widget)  
15. end\_sigma\_override (Float Widget)  
16. rho (Float Widget)  
17. blend\_factor (Float Widget)  
18. power (Float Widget)  
19. threshold\_noise (Float Widget)  
20. linear\_steps (Integer Widget)  
21. reverse\_sigmas (Toggle Widget)  
22. sigmas (Output Port)  
23. actual\_steps (Output Port)

#### **3.2. Input Port Specification**

* **model (MODEL)** \- (Anatomy Ref: \#1)  
  * **Description:** This input receives the main diffusion model. Its primary purpose is to allow the node to programmatically access the model's model\_sampling properties to determine the optimal sigma\_min and sigma\_max for the schedule. This automatic calibration is the node's core feature for ensuring compatibility and quality.  
  * **Required/Optional:** Required.

#### **3.3. Output Port Specification**

* **sigmas (SIGMAS)** \- (Anatomy Ref: \#22)  
  * **Description:** This port outputs the final, processed noise schedule. It is a 1D tensor of monotonically decreasing float values, representing the noise level at the start of each sampling step, plus a final value (often 0). This tensor is the primary control signal for an external sampler.  
  * **Resulting Tensor Spec:** A torch.float32 tensor of shape \[actual\_steps \+ 1\].  
* **actual\_steps (INT)** \- (Anatomy Ref: \#23)  
  * **Description:** This port outputs a single integer representing the exact number of steps in the final, sliced schedule. This value is critical for synchronizing the custom sampler's step count with the length of the provided sigmas tensor, preventing errors or incomplete generations.  
  * **Data Characteristics:** A Python int.

#### **3.4. Workflow Schematics**

* Minimal Functional Graph (Text-to-Image):  
  A LoadCheckpoint node's MODEL output is connected to the model input of the HybridAdaptiveSigmas node. The sigmas and actual\_steps outputs are then connected to the corresponding sigmas and steps inputs on a SamplerCustom.  
* Advanced Img2Img Graph:  
  The setup is similar to the minimal graph, but now the parameters on the scheduler are used for control. For example, the denoise widget on a VAEEncode or similar node could be connected to the denoise input of the scheduler. The scheduler's denoise\_mode (Subtractive or Hybrid) will then automatically calculate the correct start\_at\_step and slice the sigmas tensor accordingly, ensuring the SamplerCustom begins denoising from the correct point in the schedule.

### **4\. Parameter Specification**

#### **4.1. Parameter: model**

* **UI Label:** model (Input Port)  
* **Internal Variable Name:** model  
* **Data Type & Constraints:** MODEL object.  
* **Algorithmic Impact:** This is the primary data source for automatic schedule calibration. The generate method attempts to access model.get\_model\_object("model\_sampling") to retrieve the sigma\_min and sigma\_max attributes. These two float values define the absolute boundaries of the noise space the model was trained on, and all generated schedules are scaled to fit within this range. If this access fails, the node reverts to hard-coded fallback values.  
* **Interaction with Other Parameters:** The values read from the model can be completely overridden by start\_sigma\_override and end\_sigma\_override.  
* **Default Value & Rationale:** No default; this is a required input connection.

#### **4.2. Parameter: steps**

* **UI Label:** steps  
* **Internal Variable Name:** steps  
* **Data Type & Constraints:** INT, min: 1, max: 200\.  
* **Algorithmic Impact:** This integer determines the "resolution" or number of discrete points in the full, unsliced noise schedule. The various \_get\_sigmas helper functions use this value N to generate a tensor of N+1 sigma points. A higher steps value creates a more granular schedule, which can be useful when heavy slicing is anticipated.  
* **Interaction with Other Parameters:** This value is the basis for calculating the denoise\_start\_step when denoise is used. It also defines the maximum possible values for start\_at\_step and split\_at\_step.  
* **Default Value & Rationale:** 60\. A relatively high number of steps that provides a high-quality baseline and sufficient granularity for most slicing operations.

#### **4.3. Parameter: denoise\_mode**

* **UI Label:** denoise\_mode  
* **Internal Variable Name:** denoise\_mode  
* **Data Type & Constraints:** COMBO (Hybrid (Adaptive Steps), Subtractive (Slice), Repaced (Full Steps)).  
* **Algorithmic Impact:** This parameter dictates how the denoise value is interpreted to modify the schedule.  
  * Subtractive (Slice): Performs a direct slice. If steps=40 and denoise=0.25, the first 40 \* (1-0.25) \= 30 steps are discarded, and a final schedule of 10 steps is returned.  
  * Repaced (Full Steps): Slices the *sigma range* but not the steps. In the above example, it finds the sigma values at step 30 and 40, and then generates a *new*, full 40-step schedule that only spans this narrow, low-noise range.  
  * Hybrid (Adaptive Steps): Acts like Subtractive, but if the resulting number of steps is less than min\_sliced\_steps, it triggers a "repace" using min\_sliced\_steps to ensure a smooth output.  
* **Interaction with Other Parameters:** This parameter fundamentally changes the interplay between steps, denoise, start\_at\_step, and min\_sliced\_steps.  
* **Default Value & Rationale:** Hybrid (Adaptive Steps). This is the most intelligent and flexible default, preventing issues with very low denoise values while behaving predictably for mid-to-high values.

#### **4.4. Parameter: mode**

* **UI Label:** mode  
* **Internal Variable Name:** mode  
* **Data Type & Constraints:** COMBO (list of strings, e.g., karras\_rho).  
* **Algorithmic Impact:** This string is the primary switch in the \_get\_sigmas helper function. It selects which mathematical formula will be used to generate the distribution of sigma values between sigma\_max and sigma\_min. Each mode creates a uniquely shaped curve, impacting the convergence path of the sampler.  
* **Interaction with Other Parameters:** The selection of mode determines which of the fine-tuning parameters (rho, power, blend\_factor, etc.) will be active.  
* **Default Value & Rationale:** Not specified in code, but typically defaults to karras\_rho in the UI, as it's a widely-used standard for high-quality results.

#### **4.5. Parameter: denoise**

* **UI Label:** denoise  
* **Internal Variable Name:** denoise  
* **Data Type & Constraints:** FLOAT, min: 0.0, max: 1.0.  
* **Algorithmic Impact:** This value is used to calculate a denoise\_start\_step (int(steps \* (1.0 \- denoise))). This calculated step number is then compared with start\_at\_step, and the larger of the two is used as the final slice point. This provides a convenient, intuitive way to link the schedule's start point to a standard img2img denoise parameter.  
* **Interaction with Other Parameters:** Its effect is determined by denoise\_mode. A denoise of 1.0 results in a denoise\_start\_step of 0, having no effect on slicing.  
* **Default Value & Rationale:** 1.0. This corresponds to a full text-to-image generation, so no slicing is performed by default.

#### **4.6. Parameter: start\_at\_step**

* **UI Label:** start\_at\_step  
* **Internal Variable Name:** start\_at\_step  
* **Data Type & Constraints:** INT, min: 0\.  
* **Algorithmic Impact:** Provides a direct, manual override for the start of the schedule slice. The node calculates final\_start\_step \= max(start\_at\_step, denoise\_start\_step). This means start\_at\_step can be used to enforce a later start than the denoise value would imply, offering more explicit control. The final schedule is then sliced using this value as the starting index.  
* **Interaction with Other Parameters:** Overrides the denoise parameter if it specifies a later step.  
* **Default Value & Rationale:** 0\. By default, the schedule is not manually sliced from the beginning.

#### **4.7. Parameter: end\_at\_step**

* **UI Label:** end\_at\_step  
* **Internal Variable Name:** end\_at\_step  
* **Data Type & Constraints:** INT, min: 0\.  
* **Algorithmic Impact:** Provides a manual override for the end of the schedule slice. The full schedule tensor is sliced as full\_sigmas\[start\_idx:end\_idx\]. This allows a user to isolate a specific middle segment of the noise schedule for highly specialized workflows.  
* **Interaction with Other Parameters:** Defines the end point of the slice, which directly affects the final value of actual\_steps.  
* **Default Value & Rationale:** 9999\. A large integer used as a sentinel value to mean "do not slice the end," ensuring the schedule runs to completion by default.

#### **4.8. Parameter: min\_sliced\_steps**

* **UI Label:** min\_sliced\_steps  
* **Internal Variable Name:** min\_sliced\_steps  
* **Data Type & Constraints:** INT, min: 1, max: 20\.  
* **Algorithmic Impact:** This parameter is active only when denoise\_mode is set to Hybrid. After the schedule is sliced, the node checks if the resulting number of steps is greater than 0 but less than min\_sliced\_steps. If this condition is met, the node discards the sliced schedule and generates a *new* schedule of length min\_sliced\_steps that spans the sigma range of the (discarded) short slice. This prevents workflows with very low denoise from using only 1 or 2 steps, which can cause quality issues like color shifting or tiling seams.  
* **Interaction with Other Parameters:** Only active in Hybrid mode.  
* **Default Value & Rationale:** 3\. Provides a reasonable minimum number of steps to ensure smooth convergence even at very low denoise values.

#### **4.9. Parameter: detail\_preservation**

* **UI Label:** detail\_preservation  
* **Internal Variable Name:** detail\_preservation  
* **Data Type & Constraints:** FLOAT, min: 0.0, max: 1.0.  
* **Algorithmic Impact:** This parameter modifies the very last sigma value in the final schedule. It prevents the schedule from reaching absolute zero noise by interpolating the final sigma towards a small, non-zero upper bound. A value of 0.0 has no effect. A value of 1.0 pushes the final sigma to its maximum allowed corrected value. This is useful in multi-pass workflows to avoid "over-cooking" an image and destroying fine texture by denoising it too far.  
* **Interaction with Other Parameters:** This is one of the final tweaks applied to the schedule after all slicing and generation is complete.  
* **Default Value & Rationale:** 0.0. Disabled by default to produce a standard, full denoising schedule.

#### **4.10. Parameter: low\_denoise\_color\_fix**

* **UI Label:** low\_denoise\_color\_fix  
* **Internal Variable Name:** low\_denoise\_color\_fix  
* **Data Type & Constraints:** FLOAT, min: 0.0, max: 1.0.  
* **Algorithmic Impact:** Similar to detail\_preservation, this parameter adjusts the final sigma value. It is specifically designed to mitigate color shifting artifacts (e.g., a green tint) that can occur in low-denoise or tiling workflows. It interpolates the final sigma value towards a safe target (a fraction of the second-to-last sigma), effectively pulling the last step away from the unstable near-zero noise region where color artifacts can emerge.  
* **Interaction with Other Parameters:** A final tweak applied after slicing and generation.  
* **Default Value & Rationale:** 0.0. Disabled by default as it is a specific fix for advanced workflows and not needed for standard text-to-image.

#### **4.11. Parameter: split\_schedule**

* **UI Label:** split\_schedule  
* **Internal Variable Name:** split\_schedule  
* **Data Type & Constraints:** BOOLEAN.  
* **Algorithmic Impact:** This toggle activates the split schedule functionality. If True, the node generates two partial schedules instead of one. The first schedule uses mode and runs from sigma\_max down to the sigma value at split\_at\_step. The second schedule uses mode\_b and runs from the split sigma value down to sigma\_min. The two are then concatenated to form the final full schedule.  
* **Interaction with Other Parameters:** If True, it brings mode\_b and split\_at\_step into play.  
* **Default Value & Rationale:** False. Disabled by default to keep the node's behavior simple and predictable.

#### **4.12. Parameter: mode\_b**

* **UI Label:** mode\_b  
* **Internal Variable Name:** mode\_b  
* **Data Type & Constraints:** COMBO.  
* **Algorithmic Impact:** This parameter is only active if split\_schedule is True. It selects the scheduler algorithm to be used for the *second part* of the schedule, from split\_at\_step down to the end. This allows for creating complex hybrid curves, for instance, starting with an aggressive polynomial curve and finishing with a high-quality karras\_rho curve.  
* **Interaction with Other Parameters:** Only used when split\_schedule is active.  
* **Default Value & Rationale:** "karras\_rho". A safe and high-quality default for the detail-oriented second half of a schedule.

#### **4.13. Parameter: split\_at\_step**

* **UI Label:** split\_at\_step  
* **Internal Variable Name:** split\_at\_step  
* **Data Type & Constraints:** INT, min: 0\.  
* **Algorithmic Impact:** Active only when split\_schedule is True. This integer determines the step number in the original steps count at which the transition from mode to mode\_b occurs.  
* **Interaction with Other Parameters:** Only used when split\_schedule is active.  
* **Default Value & Rationale:** 30\. A reasonable midpoint for a default steps value of 60\.

#### **4.14. Parameter: start\_sigma\_override**

* **UI Label:** start\_sigma\_override  
* **Internal Variable Name:** start\_sigma\_override  
* **Data Type & Constraints:** FLOAT, min: 0.0.  
* **Algorithmic Impact:** Provides an absolute, manual override for the sigma\_max value of the schedule. If a value is provided to this optional input, the node will completely ignore the sigma\_max detected from the model. This is an expert feature for intentionally creating schedules that are mismatched with the model's training, which can be used for artistic effect or specific technical goals.  
* **Interaction with Other Parameters:** Overrides the sigma\_max obtained from the model input.  
* **Default Value & Rationale:** 1.000 in UI, but None in code. The input is optional; if not provided (None), the model's value is used.

#### **4.15. Parameter: end\_sigma\_override**

* **UI Label:** end\_sigma\_override  
* **Internal Variable Name:** end\_sigma\_override  
* **Data Type & Constraints:** FLOAT, min: 0.0.  
* **Algorithmic Impact:** Provides an absolute, manual override for the sigma\_min value of the schedule. Similar to its start\_ counterpart, this value, if provided, will be used as the lower boundary of the noise range regardless of what the model specifies.  
* **Interaction with Other Parameters:** Overrides the sigma\_min obtained from the model input.  
* **Default Value & Rationale:** 0.006 in UI, but None in code. The input is optional; if not provided (None), the model's value is used.

#### **4.16. Parameter: rho**

* **UI Label:** rho  
* **Internal Variable Name:** rho  
* **Data Type & Constraints:** FLOAT, min: 1.0.  
* **Algorithmic Impact:** This parameter is only active when mode (or mode\_b) is karras\_rho. It controls the exponent (ρ) in the Karras scheduling formula. Higher rho values create a steeper curve at the beginning of the schedule, concentrating more steps in the low-noise (high detail) regions. Values typically range from 1.1 to 7.0 for different effects.  
* **Interaction with Other Parameters:** Only used by karras\_rho mode.  
* **Default Value & Rationale:** 1.5. A low-to-moderate rho that produces a balanced, high-quality schedule suitable for general use.

#### **4.17. Parameter: blend\_factor**

* **UI Label:** blend\_factor  
* **Internal Variable Name:** blend\_factor  
* **Data Type & Constraints:** FLOAT, min: 0.0, max: 1.0.  
* **Algorithmic Impact:** Active only when mode is blended\_curves. This parameter performs a linear interpolation between a karras\_rho schedule and an adaptive\_linear schedule. The final schedule is calculated as (1.0 \- blend\_factor) \* karras \+ blend\_factor \* linear. A value of 0.0 yields a pure Karras schedule, while 1.0 yields a pure linear schedule.  
* **Interaction with Other Parameters:** Only used by blended\_curves mode.  
* **Default Value & Rationale:** 0.5. A perfect 50/50 blend between the two curves by default.

#### **4.18. Parameter: power**

* **UI Label:** power  
* **Internal Variable Name:** power  
* **Data Type & Constraints:** FLOAT, min: 0.1.  
* **Algorithmic Impact:** Active only when mode is polynomial. It serves as the exponent p in the polynomial scheduling formula. A value of 1.0 is equivalent to a linear schedule. Values greater than 1.0 make the curve more convex (steeper start, flatter end). Values less than 1.0 make the curve more concave (flatter start, steeper end), which concentrates steps in the low-sigma region.  
* **Interaction with Other Parameters:** Only used by polynomial mode.  
* **Default Value & Rationale:** 2.0. A quadratic curve that provides a noticeably different shape from linear without being too extreme.

#### **4.19. Parameter: threshold\_noise**

* **UI Label:** threshold\_noise  
* **Internal Variable Name:** threshold\_noise  
* **Data Type & Constraints:** FLOAT, min: 0.0.  
* **Algorithmic Impact:** Active only when mode is linear\_quadratic. This parameter defines the transition point, or "knee," in the curve where the schedule switches from a linear descent to a quadratic one. It represents the noise level at this transition point.  
* **Interaction with Other Parameters:** Works in conjunction with linear\_steps.  
* **Default Value & Rationale:** 0.0025. A small value that places the transition very late in the schedule.

#### **4.20. Parameter: linear\_steps**

* **UI Label:** linear\_steps  
* **Internal Variable Name:** linear\_steps  
* **Data Type & Constraints:** INT, min: 0\.  
* **Algorithmic Impact:** Active only when mode is linear\_quadratic. This specifies how many of the total steps are dedicated to the initial linear portion of the schedule. The code implements the logic as linear\_steps\_actual \= max(0, min(linear\_steps, steps)). Therefore, a user-provided value of 0 results in linear\_steps\_actual being 0, which causes the schedule to be purely quadratic. A value greater than or equal to the total steps will result in a purely linear schedule.  
* **Interaction with Other Parameters:** Works in conjunction with threshold\_noise to define the shape of the linear\_quadratic curve.  
* **Default Value & Rationale:** 30\. Allocates a significant portion of a standard 60-step schedule to the stable linear section.

#### **4.21. Parameter: reverse\_sigmas**

* **UI Label:** reverse\_sigmas  
* **Internal Variable Name:** reverse\_sigmas  
* **Data Type & Constraints:** BOOLEAN.  
* **Algorithmic Impact:** This is an experimental toggle that applies torch.flip to the final, sliced schedule tensor. This completely inverts the schedule, making it go from low noise (sigma\_min) to high noise (sigma\_max). This is not a standard diffusion process and is intended for creative, experimental, or "glitch art" effects.  
* **Interaction with Other Parameters:** This is the absolute final operation performed on the sigmas tensor before it is outputted.  
* **Default Value & Rationale:** False. Disabled by default to ensure standard, correct diffusion behavior.

### **5\. Applied Use-Cases & Recipes**

#### **5.1. Recipe 1: Standard High-Quality Text-to-Image Generation**

* **Objective:** To create a standard text-to-image generation using a reliable, high-quality noise schedule that is automatically calibrated to the loaded model.  
* **Rationale:** The karras\_rho schedule is a proven, perceptually-tuned standard. By connecting the model, the node automatically fetches the correct sigma\_min and sigma\_max. A rho of 3.0 provides a well-balanced curve. Connecting actual\_steps to the sampler ensures perfect synchronization. This is the baseline "set it and forget it" recipe for quality.  
* **Parameter Configuration:**  
  * model: Connect checkpoint.  
  * steps: 40  
  * mode: karras\_rho  
  * rho: 3.0  
  * start\_at\_step: 0  
  * end\_at\_step: 9999  
  * **Sampler Setup**: Connect sigmas and actual\_steps to a SamplerCustom.

#### **5.2. Recipe 2: Precise Img2Img Denoising with Schedule Slicing**

* **Objective:** To perform an img2img operation with a denoise value of exactly 0.6, ensuring the sampler uses the mathematically correct portion of the noise schedule.  
* **Rationale:** The correct way to handle a denoise of 0.6 is to skip the first 1.0 \- 0.6 \= 0.4 (or 40%) of the diffusion steps. By setting denoise\_mode to Subtractive, the node will calculate the starting step as total\_steps \* (1 \- denoise) (50 \* 0.4 \= 20). It then slices the full 50-step schedule, discarding the first 20 steps and returning the remaining 30\. The actual\_steps output of 30 perfectly syncs the sampler.  
* **Parameter Configuration:**  
  * model: Connect checkpoint.  
  * steps: 50  
  * denoise\_mode: Subtractive (Slice)  
  * denoise: 0.6  
  * mode: karras\_rho  
  * **Sampler Setup**: Connect sigmas and actual\_steps to a SamplerCustom (and denoise to the sampler's denoise input).

#### **5.3. Recipe 3: Late-Stage Detail Enhancement using a Polynomial Curve**

* **Objective:** To create a schedule that spends proportionally more time in the low-noise, high-detail phase of generation, potentially enhancing fine textures and details.  
* **Rationale:** A polynomial schedule with a power less than 1.0 creates a concave curve. The sigma values decrease slowly at the beginning and then more rapidly towards the end. This means that for a fixed number of steps, more of those steps are "clustered" in the low-sigma region of the schedule. This forces the sampler to take smaller, more numerous steps when the image is nearly complete, giving the model more opportunities to refine fine details.  
* **Parameter Configuration:**  
  * model: Connect checkpoint.  
  * steps: 60  
  * mode: polynomial  
  * power: 0.7  
  * **Sampler Setup**: Connect sigmas and actual\_steps to a SamplerCustom.

### **6\. Implementation Deep Dive**

#### **6.1. Source Code Walkthrough**

The node's logic is primarily contained within the generate method.

1. **Sigma Range Determination:** The method begins by wrapping a try-except block around the access to model.get\_model\_object("model\_sampling").sigma\_min and sigma\_max. This is the primary method of calibration. If it fails, it prints a warning and falls back to class constants FALLBACK\_SIGMA\_MIN and FALLBACK\_SIGMA\_MAX.  
2. **Override Application:** It then immediately checks if start\_sigma\_override or end\_sigma\_override are None. If they have been provided, their values overwrite the sigma\_max and sigma\_min variables, respectively. A final sanity check ensures sigma\_min \< sigma\_max.  
3. **Full Schedule Generation:** The logic then enters a large if/else block for split\_schedule.  
   * If split\_schedule is True, it first generates a temporary full schedule to find the split\_sigma value at the split\_at\_step. It then calls the internal \_get\_sigmas helper function twice: once for mode (from sigma\_max to split\_sigma) and once for mode\_b (from split\_sigma to sigma\_min). The results are concatenated.  
   * If False, it simply calls \_get\_sigmas once to generate the full\_sigmas tensor.  
4. **Slice Calculation:** A universal slice calculation is performed. denoise\_start\_step is calculated from the denoise value. final\_start\_step is determined by max(start\_at\_step, denoise\_start\_step). This determines the start index start\_idx. The end\_idx is calculated from end\_at\_step.  
5. **Denoise Mode Logic:** A new if/elif block applies the selected denoise\_mode. Subtractive simply slices full\_sigmas\[start\_idx:end\_idx\]. Repaced uses the sigma values at the slice boundaries to generate a completely new schedule. Hybrid first slices, then checks the length and potentially triggers a repace based on min\_sliced\_steps.  
6. **Final Tweaks:** The resulting final\_sigmas tensor is cloned. The logic for detail\_preservation and low\_denoise\_color\_fix then perform in-place modification of the last element of this cloned tensor. If reverse\_sigmas is true, a final torch.flip is applied.  
7. **Output Calculation:** actual\_step\_count is calculated as final\_sigmas.shape\[0\] \- 1\. The final\_sigmas tensor and this integer are returned.

#### **6.2. Dependencies & External Calls**

* **torch:** The fundamental library for all tensor creation and manipulation. Functions like torch.linspace, .pow, and .flip are used extensively to procedurally generate the schedules.  
* **comfy.model\_management:** Used for get\_torch\_device() to ensure tensors are created on the correct compute device.  
* **comfy.k\_diffusion.sampling:** The get\_sigmas\_karras function is imported directly from ComfyUI's internal k-diffusion implementation. This ensures that the generated Karras schedule is identical to the one the built-in samplers would use, guaranteeing compatibility and quality.

#### **6.3. Performance & Resource Analysis**

* **Execution Target:** All operations are performed on the **CPU**. The generated tensors are small 1D arrays, and the mathematical operations are computationally trivial. The generate method is extremely fast.  
* **VRAM Usage:** Negligible. The only VRAM interaction is the initial read of the model object's attributes. All generated sigma tensors are created on the CPU by default, though the code specifies .to(device) which would move them to the GPU if necessary (a very small memory footprint).  
* **Bottlenecks:** There are no performance bottlenecks in this node. Its execution time is effectively instantaneous relative to the time taken by the sampler or model.

#### **6.4. Tensor Lifecycle Analysis**

1. **Stage 1 (Input):** A MODEL object is received. No tensors are input.  
2. **Stage 2 (Generation):** Inside the \_get\_sigmas helper functions, new PyTorch tensors are created from scratch. For example, torch.linspace(sigma\_max, sigma\_min, steps \+ 1\) creates the initial tensor for a linear schedule on the specified compute device.  
3. **Stage 3 (Manipulation):** This initial tensor (full\_sigmas) undergoes slicing, which creates a *view* of the original tensor without copying data. It may then be cloned (final\_sigmas.clone()) to create a new, separate tensor in memory.  
4. **Stage 4 (In-Place Modification):** The final tweaks (detail\_preservation, etc.) modify the last element of this cloned tensor in-place.  
5. **Stage 5 (Output):** The final, manipulated 1D final\_sigmas tensor is returned through the sigmas output port. Its lifecycle continues until it is consumed by the downstream custom sampler node.

### **7\. Troubleshooting & Diagnostics**

#### **7.1. Error Code Reference**

* **Error Message/Traceback Snippet:** (Console Warning) Hybrid Sigma Scheduler WARNING: Using fallback sigmas.  
  * **Root Cause Analysis:** The generate method failed to access the sigma\_min and sigma\_max attributes from the connected model object. This typically happens with older or non-standard model architectures that do not conform to the expected ComfyUI model wrapper structure.  
  * **Primary Solution:** For most common models (SD1.5, SDXL, etc.), ensure the model is loaded correctly. If the warning persists with a custom model, you must use the start\_sigma\_override and end\_sigma\_override inputs to manually provide the correct values for your model.  
* **Error Message/Traceback Snippet:** (Console ERROR) Hybrid Sigma Scheduler ERROR: sigma\_min (X) \>= sigma\_max (Y). Adjusting.  
  * **Root Cause Analysis:** The final sigma range determined after applying overrides is invalid, with the minimum noise level being greater than or equal to the maximum. This is most often caused by user error when setting the start\_sigma\_override and end\_sigma\_override values.  
  * **Primary Solution:** Correct the override values. Remember that start\_sigma\_override corresponds to sigma\_max (the highest noise level, a larger number) and end\_sigma\_override corresponds to sigma\_min (the lowest noise level, a smaller number).

#### **7.2. Unexpected Visual Artifacts & Behavior**

* **Artifact:** The custom sampler produces a noisy, unfinished, or blurry image.  
  * **Likely Cause(s):** There is a mismatch between the number of steps the sampler *thinks* it should run and the number of sigmas provided.  
  * **Correction Strategy:** This is the most common user error. You **must** connect the actual\_steps output of the scheduler to the steps input of the SamplerCustom. This is especially critical when using any form of slicing (denoise, start\_at\_step, end\_at\_step).  
* **Artifact:** The generated image has a green tint or other color shifts, especially when using low denoise values.  
  * **Likely Cause(s):** At very low denoise values, the schedule may contain only a few steps that are all clustered very close to zero noise. Some models can become unstable in this near-zero sigma region, leading to color artifacts.  
  * **Correction Strategy:** This is precisely what the low\_denoise\_color\_fix parameter is designed for. Try setting it to a small value like 0.1 or 0.2. This will subtly lift the final sigma value away from zero, into a more stable region for the model, which can often resolve the color shift. Alternatively, use the Hybrid denoise mode with a min\_sliced\_steps of 3 or 4 to ensure the sampler always takes at least a few steps.  
* **Artifact:** The img2img output does not seem to respect the denoise value correctly.  
  * **Likely Cause(s):** The interaction between denoise and start\_at\_step can be confusing. The node uses max(start\_at\_step, denoise\_start\_step), meaning the later of the two start times will be used.  
  * **Correction Strategy:** For simple img2img, leave start\_at\_step at its default of 0 and control the process using only the denoise parameter. For more complex workflows where you need to guarantee a specific starting step regardless of denoise, use start\_at\_step and set denoise to 1.0.