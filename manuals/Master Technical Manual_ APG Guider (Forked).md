---

## **Master Technical Manual: APG Guider (Forked)**

### **Node Name: APGGuiderNode**

Display Name: APG Guider (Forked)  
Category: MD\_Nodes/Guiders  
Version: 0.2.1  
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
   4.2. Parameter: positive  
   4.3. Parameter: negative  
   4.4. Parameter: disable\_apg  
   4.5. Parameter: verbose\_debug  
   4.6. Parameter: apg\_scale  
   4.7. Parameter: cfg\_before  
   4.8. Parameter: cfg\_after  
   4.9. Parameter: eta  
   4.10. Parameter: norm\_threshold  
   4.11. Parameter: momentum  
   4.12. Parameter: start\_sigma  
   4.13. Parameter: end\_sigma  
   4.14. Parameter: dims  
   4.15. Parameter: predict\_image  
   4.16. Parameter: mode  
   4.17. Parameter: yaml\_parameters\_opt  
5. Applied Use-Cases & Recipes  
   5.1. Recipe 1: High-Frequency Detail Enhancement ("Crisp Look")  
   5.2. Recipe 2: Coherent Painterly Style ("Smooth Look")  
   5.3. Recipe 3: Stable Audio Generation Guidance  
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

The **APG Guider (Forked)** is a sophisticated, highly configurable guidance controller engineered for ComfyUI's custom sampler ecosystem. It fundamentally replaces the standard Classifier-Free Guidance (CFG) mechanism with a more advanced system based on **Adaptive Projected Gradient (APG)**. Where standard CFG applies a first-order linear force to push the generation towards a prompt, APG introduces a second-order, geometric approach. It analyzes the guidance vector in high-dimensional latent space and decomposes it, primarily utilizing the **orthogonal (perpendicular) component** to steer the diffusion process. This allows for surgical control over stylistic elements, texture, and detail without simply increasing prompt adherence. The node also integrates a momentum engine for temporal smoothing or sharpening of the guidance signal and a comprehensive sigma-based scheduling system, enabling dynamic changes to the guidance strategy throughout the generation process.

#### **1.2. Conceptual Category**

**Advanced Sampler Guidance Controller.** This categorization is critical: the node does not directly process or transform data tensors in a sequential workflow. Instead, it constructs a functional Python object—an instance of the APGGuider class—which encapsulates a complete guidance strategy. This object is then injected into a compatible custom sampler, where it hooks into the core prediction loop to exert its influence at each denoising step.

#### **1.3. Problem Domain & Intended Application**

* **Problem Domain:** The standard Classifier-Free Guidance (CFG) mechanism, while effective, suffers from several limitations that become apparent in advanced use cases. Its linear scaling nature means that increasing the cfg value acts as a blunt instrument, amplifying all prompt-related signals indiscriminately. This can lead to common issues such as color oversaturation, "over-baked" or "fried" textures, and a loss of nuanced detail as the model is forced too aggressively along a single guidance vector. Furthermore, standard CFG is static; the same guidance strength is applied uniformly across all steps, failing to account for the different needs of the generation process as it transitions from a high-noise (structural) phase to a low-noise (detail) phase.  
* **Intended Application (Use-Cases):** This node is engineered for precision and control, targeting users who have reached the limits of standard CFG.  
  * **Fine Art & Stylistic Control:** To generate images with specific aesthetic qualities, such as painterly smoothness (via positive momentum) or crisp, photorealistic detail (via negative momentum), that are difficult to achieve with CFG alone.  
  * **Complex Prompt Coherence:** To improve the coherence of generations with complex prompts by using scheduled guidance to focus on structure first (high sigma) and details later (low sigma).  
  * **Advanced Modalities (Audio):** To provide stable, responsive guidance for non-image diffusion models where standard CFG can easily lead to chaotic or artifact-laden output.  
  * **Workflow Automation & Sharing:** To encapsulate complex guidance strategies into a portable YAML format, allowing for perfect replication and sharing of advanced techniques.  
* **Non-Application (Anti-Use-Cases):**  
  * The node is fundamentally **incompatible** with any standard sampler that uses a simple float cfg input (e.g., KSampler, KSampler Advanced). Attempting to use it with these will result in a workflow error, as they lack the necessary guider input socket to accept the functional object this node produces.

#### **1.4. Key Features (Functional & Technical)**

* **Functional Features:**  
  * **Sigma-Based Scheduling:** Provides the ability to define multiple, distinct sets of guidance rules, with each rule activating automatically when the generation's noise level (sigma) drops below a specified threshold. This enables complex, phased guidance strategies.  
  * **Full YAML Configuration:** A multiline text input allows for the definition of the entire guidance schedule using the YAML format. This powerful feature overrides all UI controls, enabling the creation, saving, and sharing of intricate and precise guidance recipes.  
  * **Dynamic Momentum Engine:** Integrates a momentum term that temporally filters the guidance vector. Positive values smooth the guidance across steps for coherence, while negative values sharpen it by amplifying changes, enhancing detail.  
  * **Hybrid Guidance System:** Allows for seamless transitions between pure APG, standard CFG, and various experimental hybrid modes, all within a single scheduled generation.  
* **Technical Features:**  
  * **Orthogonal Vector Projection:** The core of the APG method. The guidance vector is mathematically projected onto the conditional prediction vector, and the resulting orthogonal component is used to steer the generation. This provides a guidance signal that is decorrelated from the simple "more like the prompt" direction.  
  * **Dynamic Guider Injection:** The node outputs an APGGuider object that inherits from ComfyUI's CFGGuider. A compatible custom sampler will call this object's predict\_noise method at each step, allowing the node's logic to completely take over the guidance calculation.  
  * **Selectable Prediction Target:** Guidance can be configured to operate on the model's direct noise prediction (epsilon) or its predicted clean image (x0), offering different stability and quality characteristics.

### **2\. Core Concepts & Theory**

#### **2.1. Theoretical Background**

To fully grasp the APG Guider's operation, it is essential to understand the progression from standard CFG to the more advanced concepts it employs.

* **Classifier-Free Guidance (CFG):** At each step, a diffusion model makes two predictions: a **conditional** prediction based on the user's prompt (x\_cond) and an **unconditional** prediction based on a null prompt (x\_uncond). The vector difference, v\_g=(x\_cond−x\_uncond), represents the direction in latent space that "points" toward the prompt's concepts. CFG guidance creates a final prediction by taking the unconditional prediction and moving it along this guidance vector, scaled by the cfg value. It is a linear extrapolation, a simple and powerful method for increasing prompt adherence.  
* **Adaptive Projected Gradient (APG):** APG posits that the guidance vector v\_g contains more information than just a single direction. It can be geometrically decomposed relative to the conditional prediction vector, v\_c=x\_cond. Using vector projection, v\_g is split into two unique, perpendicular components:  
  1. A **parallel component**, v\_p, which lies along the same direction as v\_c. This component is conceptually similar to what standard CFG reinforces.  
  2. An **orthogonal component**, v\_o, which is perpendicular to v\_c. This component represents a direction that is still aligned with the prompt's influence but explores a different dimension of the solution space than simply reinforcing the existing prediction. It is hypothesized that this orthogonal direction corresponds to higher-frequency details, textures, and stylistic nuances. The APG Guider isolates and scales this component to steer the generation in a more refined manner.  
* **Guidance Momentum:** In signal processing, momentum is often implemented as an Infinite Impulse Response (IIR) filter, or more simply, an Exponential Moving Average (EMA). It maintains a "memory" of previous states.  
  * **Positive Momentum ($0 \\\< m \\\< 1$):** Functions as a **low-pass filter** on the sequence of guidance vectors. By averaging the current vector with the memory of past vectors, it smooths out rapid, high-frequency changes. In generation, this translates to more stable, coherent, and fluid transitions from step to step, often resulting in a painterly aesthetic.  
  * **Negative Momentum ($-1 \\\< m \\\< 0$):** Functions as a **high-pass filter**. By subtracting the memory of past vectors from the current one, it amplifies the *change* or *delta* between steps. This makes the guidance hyper-responsive to new information in the prediction, effectively sharpening the final result and enhancing fine details.  
* **Sigma (σ) Scheduling:** The diffusion process is defined by a noise schedule, where the latent image starts with a high level of Gaussian noise (high sigma) and is progressively denoised (low sigma). The *character* of the required guidance changes with sigma. At high sigma, the model is determining broad structures and composition, requiring strong, stable guidance. At low sigma, the model is refining fine details and textures, where overly strong guidance can be destructive. Sigma scheduling, the core control principle of this node, allows the user to define a piecewise function for all guidance parameters, tailoring the strategy to the specific needs of each phase of the generation.

#### **2.2. Mathematical & Algorithmic Formulation**

1. Classifier-Free Guidance (CFG) Prediction (x′\_denoised):

   xdenoised′​=xuncond​+scfg​⋅(xcond​−xuncond​)

   where s\_cfg is the CFG scale.  
2. APG Vector Decomposition:  
   Let the guidance vector be v\_g=x\_cond−x\_uncond and the conditional prediction be v\_c=x\_cond.  
   * Parallel Component (v\_p): The vector projection of v\_g onto v\_c.

     vp​=∥vc​∥2vg​⋅vc​​vc​=∑(vc​⊙vc​)∑(vg​⊙vc​)​vc​  
   * Orthogonal Component (v\_o): The vector rejection of v\_g from v\_c.

     vo​=vg​−vp​  
3. Momentum (EMA) Update:  
   The running average vector avg is updated at each step t for the active rule.

   avgt​=vg,t​+m⋅avgt−1​

   where v\_g,t is the guidance vector at step t and m is the momentum parameter. The vector passed to the projection step is this updated avg\_t.  
4. **Rule Matching Pseudocode:**  
   function get\_active\_rule(current\_sigma, all\_rules\_sorted):  
       for rule in all\_rules\_sorted:  
           if current\_sigma \<= rule.start\_sigma:  
               return rule  
       return fallback\_rule // Should not be reached

#### **2.3. Data I/O Deep Dive**

* **Inputs:**  
  * model (MODEL): A reference to the loaded diffusion model object. This object contains the U-Net and other components necessary for prediction.  
  * positive / negative (CONDITIONING): These are complex Python objects containing one or more conditioning tensors of shape \[Batch Size, Sequence Length, Embedding Dim\] (e.g., \[1, 77, 768\] for SD1.5), along with pooled embeddings and other metadata.  
* **Output:**  
  * apg\_guider (GUIDER): This is not a data tensor but a Python **object instance** of the APGGuider class. This object encapsulates the model, conditioning, and all the configured guidance rules. It is designed to be passed directly to a compatible sampler's guider input.

#### **2.4. Strategic Role in the ComfyUI Graph**

* **Placement Context:** The APG Guider is a **controller** that sits between the conditioning and the sampler. It takes the core components (model, positive, negative) and produces a new functional unit (GUIDER).  
* **Synergistic Nodes:**  
  * SamplerCustom, SamplerDPMAdaptative: **Required.** Any sampler node that has a guider input socket.  
* **Conflicting Nodes:**  
  * KSampler, KSampler Advanced: **Incompatible.** These nodes have their own built-in CFG implementation and do not accept an external guider object.

### **3\. Node Architecture & Workflow**

#### **3.1. Node Interface & Anatomy**

1. model (Input Port)  
2. positive (Input Port)  
3. negative (Input Port)  
4. apg\_guider (Output Port)  
5. disable\_apg (Toggle Widget)  
6. verbose\_debug (Toggle Widget)  
7. apg\_scale (Float Widget)  
8. cfg\_before (Float Widget)  
9. cfg\_after (Float Widget)  
10. eta (Float Widget)  
11. norm\_threshold (Float Widget)  
12. momentum (Float Widget)  
13. start\_sigma (Float Widget)  
14. end\_sigma (Float Widget)  
15. dims (String Widget)  
16. predict\_image (Toggle Widget)  
17. mode (Dropdown Widget)  
18. yaml\_parameters\_opt (Multiline String Widget)

#### **3.2. Input Port Specification**

* **model (MODEL)** \- (Anatomy Ref: \#1)  
  * **Description:** This input receives the main diffusion model. The APGGuider object will store a reference to this model and will invoke its forward pass during the prediction step. It is the computational engine that the guider directs.  
  * **Required/Optional:** Required.  
* **positive (CONDITIONING)** \- (Anatomy Ref: \#2)  
  * **Description:** This input receives the positive conditioning generated from the user's prompt. This data structure contains the text embeddings that the guider uses as the target for the conditional prediction (x\_cond).  
  * **Required/Optional:** Required.  
* **negative (CONDITIONING)** \- (Anatomy Ref: \#3)  
  * **Description:** This input receives the negative conditioning. This is used as the target for the unconditional prediction (x\_uncond) and forms the baseline from which the guidance vector is calculated.  
  * **Required/Optional:** Required.

#### **3.3. Output Port Specification**

* **apg\_guider (GUIDER)** \- (Anatomy Ref: \#4)  
  * **Description:** This port outputs a single, functional Python object, an instance of the APGGuider class. This object is a self-contained guidance engine, fully configured with all the rules and parameters defined in the node's UI or YAML input. It is not a data tensor but a "strategy package" that is handed off to the sampler.  
  * **Data Characteristics:** The object contains references to the model and conditioning, a tuple of configured APG rule objects, and the critical predict\_noise method which the sampler will call at every step.

#### **3.4. Workflow Schematics**

* Minimal Functional Graph:  
  This schematic demonstrates the essential connections. The conditioning from CLIPTextEncode nodes and the MODEL from a LoadCheckpoint node are fed into the APGGuiderNode. The resulting GUIDER object is then passed to a SamplerCustom. The sampler performs the denoising loop, calling the guider at each step, and outputs the final latent, which is then decoded.  
* Advanced YAML-Based Graph:  
  The workflow topology is identical to the minimal graph. The key difference is that the control logic is defined textually in the yaml\_parameters\_opt widget instead of through the individual UI sliders and toggles. This workflow is functionally more complex but visually the same, highlighting the node's ability to encapsulate complexity.

### **4\. Parameter Specification**

#### **4.1. Parameter: model, positive, negative**

* **UI Label:** model, positive, negative (Input Ports)  
* **Internal Variable Name:** model, positive, negative  
* **Data Type & Constraints:** MODEL, CONDITIONING.  
* **Algorithmic Impact:** These are the foundational inputs required to perform any guidance. They are passed to the APGGuider constructor and stored as instance attributes. The model is used to make predictions, while positive and negative are used to define the conditional and unconditional states for those predictions. They are not parameters in the sense of numeric controls, but the essential data context for the guider's operation.  
* **Interaction with Other Parameters:** All other parameters on the node serve to control how these three core components are used during the sampling loop.  
* **Default Value & Rationale:** No default; these are required connections.

#### **4.2. Parameter: disable\_apg**

* **UI Label:** disable\_apg (Toggle: APG Enabled / APG Disabled)  
* **Internal Variable Name:** disable\_apg  
* **Data Type & Constraints:** BOOLEAN.  
* **Algorithmic Impact:** This parameter acts as a global master switch for all APG functionality. Inside the go method, if disable\_apg is True, the entire rule-building logic (from both UI and YAML) is bypassed. Instead, a single, hard-coded rule is created: APGConfig.build(cfg=cfg\_after, start\_sigma=math.inf, apg\_blend=0.0). This rule has an apg\_blend of zero, which effectively turns off all APG calculations and causes the guider to behave as a standard CFG guider, using only the cfg\_after value for the entire generation.  
* **Interaction with Other Parameters:** If True, this parameter overrides **all** other guidance-related parameters (apg\_scale, momentum, yaml\_parameters\_opt, etc.), with the sole exception of cfg\_after.  
* **Default Value & Rationale:** False. The node is intended to provide APG guidance, so this functionality is enabled by default.

#### **4.3. Parameter: verbose\_debug**

* **UI Label:** verbose\_debug (Toggle: Verbose Off / Verbose On)  
* **Internal Variable Name:** verbose\_debug  
* **Data Type & Constraints:** BOOLEAN.  
* **Algorithmic Impact:** This parameter controls the verbosity of console logging. The boolean value is passed to the APGGuider constructor and stored as self.apg\_verbose. Inside the predict\_noise method, a check for if self.apg\_verbose: determines whether to print the detailed debug message: tqdm.write(f"\* APG rule matched: sigma={sigma:.4f}, rule={rule.config}"). This provides a step-by-step trace of which rule is active at which sigma level.  
* **Interaction with Other Parameters:** This is a purely diagnostic control and does not affect the final generated image. Its state is also respected if set within a YAML configuration (verbose: true).  
* **Default Value & Rationale:** False. Verbose logging can clutter the console during normal operation, so it is disabled by default.

#### **4.4. Parameter: apg\_scale**

* **UI Label:** apg\_scale  
* **Internal Variable Name:** apg\_scale (a field within the APGConfig named tuple for the active rule)  
* **Data Type & Constraints:** FLOAT, with a range from \-1000.0 to 1000.0.  
* **Algorithmic Impact:** This parameter is the primary scalar multiplier for the orthogonal component of the guidance vector. In the cfg\_function, the final prediction is calculated as result \= cond \+ (self.apg\_scale \- 1.0) \* self.apg(cond, uncond). A value of 1.0 results in cond \+ 0 \* ..., effectively disabling the APG influence. Higher values increase the magnitude of the orthogonal "push," leading to a stronger stylistic and detail-enhancing effect. Negative values will push the generation *away* from the orthogonal direction, an experimental effect.  
* **Interaction with Other Parameters:** The effect of apg\_scale is entirely nullified if disable\_apg is True or if the active rule has apg\_blend set to 0.0. Its perceived strength is heavily modulated by momentum and constrained by norm\_threshold.  
* **Default Value & Rationale:** 4.5. This provides a moderate, noticeable APG effect that is a good starting point for experimentation without being immediately overpowering.

#### **4.5. Parameter: cfg\_before / cfg\_after**

* **UI Label:** cfg\_before, cfg\_after  
* **Internal Variable Name:** cfg\_before, cfg\_after  
* **Data Type & Constraints:** FLOAT, min: 1.0, max: 1000.0.  
* **Algorithmic Impact:** These parameters define the standard CFG scale to be used at different stages of generation when using the simple UI-based scheduler. cfg\_before sets the cfg value for the main rule (active between start\_sigma and end\_sigma). cfg\_after sets the cfg value for the final rule (active after end\_sigma). Inside predict\_noise, the value rule.cfg is retrieved from the active rule and passed to self.cfg, controlling the strength of the standard guidance component.  
* **Interaction with Other Parameters:** These are completely ignored if a yaml\_parameters\_opt configuration is provided. cfg\_after is uniquely used by the disable\_apg toggle as the sole CFG value for the entire generation.  
* **Default Value & Rationale:** cfg\_before: 4.0, cfg\_after: 3.0. This reflects a common strategy of using slightly stronger guidance during the main composition phase and then reducing it for the final detail refinement steps to prevent over-baking.

#### **4.6. Parameter: eta**

* **UI Label:** eta  
* **Internal Variable Name:** eta  
* **Data Type & Constraints:** FLOAT.  
* **Algorithmic Impact:** In the original research and some implementations of projected guidance, eta is a parameter that controls the re-introduction of the parallel component of the guidance vector. However, in this specific forked implementation (v0.2.1), the code in the APG.apg method intentionally uses only the orthogonal component (return diff\_o). Therefore, while the eta parameter is present in the UI and configuration structure for compatibility and potential future development, it currently has **no algorithmic impact** on the output.  
* **Interaction with Other Parameters:** None in the current implementation.  
* **Default Value & Rationale:** 0.0. Set to zero as it is inactive, representing a pure orthogonal guidance approach.

#### **4.7. Parameter: norm\_threshold**

* **UI Label:** norm\_threshold  
* **Internal Variable Name:** norm\_threshold  
* **Data Type & Constraints:** FLOAT.  
* **Algorithmic Impact:** This parameter acts as a crucial stabilizer. Inside the APG.apg method, after momentum is applied, the L2 norm (magnitude) of the guidance vector pred\_diff is calculated. If this magnitude exceeds norm\_threshold, the vector is rescaled (normalized) back down to have a magnitude exactly equal to norm\_threshold. This is a form of vector clipping that prevents excessively large, potentially divergent guidance steps that can cause visual artifacts. A value of 0 or less disables this check.  
* **Interaction with Other Parameters:** This parameter is a direct counter-measure to aggressive settings for apg\_scale and negative momentum. Higher guidance strengths often necessitate a lower, more restrictive norm\_threshold to maintain stability.  
* **Default Value & Rationale:** 2.5. This value provides a reasonable ceiling for the guidance vector's magnitude, preventing most common artifacting issues while still allowing for strong guidance.

#### **4.8. Parameter: momentum**

* **UI Label:** momentum  
* **Internal Variable Name:** momentum  
* **Data Type & Constraints:** FLOAT.  
* **Algorithmic Impact:** This parameter, m, is the coefficient for the running average term in the momentum update formula: avg\_t \= v\_g,t \+ m \* avg\_{t-1}.  
  * When m \> 0, it creates a low-pass filter, smoothing the guidance by incorporating previous steps. Values closer to 1.0 (e.g., 0.95) create very strong smoothing.  
  * When m \< 0, it creates a high-pass filter, sharpening the guidance by subtracting the influence of previous steps, thereby amplifying the change between the current step and the historical average.  
  * When m \= 0, the update method returns the value val unmodified, completely disabling the momentum effect.  
* **Interaction with Other Parameters:** Momentum is most impactful when apg\_scale is active. High positive momentum can help stabilize high apg\_scale values, while high negative momentum can exacerbate them, often requiring a lower norm\_threshold to prevent artifacts.  
* **Default Value & Rationale:** 0.75. A substantial positive value chosen to promote stable, coherent generations by default, as this is often desirable and a key strength of the momentum feature.

#### **4.9. Parameter: start\_sigma**

* **UI Label:** start\_sigma  
* **Internal Variable Name:** start\_sigma  
* **Data Type & Constraints:** FLOAT, min: \-1.0.  
* **Algorithmic Impact:** When using the UI scheduler, this defines the upper noise boundary for the main APG rule. The rule becomes active for any step where the sampler's current sigma is less than or equal to this value. In the APGConfig.fixup\_param method, a value of \-1.0 is programmatically converted to math.inf (infinity), ensuring that a rule with this setting is active from the very beginning of the sampling process.  
* **Interaction with Other Parameters:** Defines the point at which cfg\_before is used instead of a preceding rule's cfg. Ignored if YAML is used.  
* **Default Value & Rationale:** \-1.0. This makes the primary APG rule active for the entire generation by default, providing the most straightforward user experience.

#### **4.10. Parameter: end\_sigma**

* **UI Label:** end\_sigma  
* **Internal Variable Name:** end\_sigma  
* **Data Type & Constraints:** FLOAT, min: \-1.0.  
* **Algorithmic Impact:** When using the UI scheduler, this defines the lower noise boundary for the main APG rule. The \_build\_rules\_from\_inputs helper function creates a second rule that starts at end\_sigma (technically math.nextafter(end\_sigma, \-math.inf) for precision) and uses the cfg\_after value with APG disabled. This creates the transition from the main guidance phase to the final refinement phase. A value of \-1.0 means this second rule is never created, and the main rule persists to the end.  
* **Interaction with Other Parameters:** Determines the activation point for the cfg\_after parameter. Ignored if YAML is used.  
* **Default Value & Rationale:** \-1.0. By default, there is no separate "end" phase; the main rule is active for the entire duration.

#### **4.11. Parameter: dims**

* **UI Label:** dims  
* **Internal Variable Name:** dims  
* **Data Type & Constraints:** STRING (comma-separated integers).  
* **Algorithmic Impact:** This string is parsed into a tuple of integers that specifies which dimensions of the latent tensor the vector mathematics (normalization, dot product summation) should operate over. For standard image latents with shape \[B, C, H, W\], the height and width are the last two dimensions, hence "-1, \-2". For audio or other 1D data with shape \[B, C, L\], this should be "-1". Incorrectly setting this will cause the vector math to fail or produce nonsensical results.  
* **Interaction with Other Parameters:** This is a fundamental technical parameter that affects all calculations involving vector norms and projections, including norm\_threshold and the core apg method.  
* **Default Value & Rationale:** "-1, \-2". This is the correct setting for the vast majority of use cases involving standard image diffusion models.

#### **4.12. Parameter: predict\_image**

* **UI Label:** predict\_image  
* **Internal Variable Name:** predict\_image  
* **Data Type & Constraints:** BOOLEAN.  
* **Algorithmic Impact:** This toggle controls the target of the guidance calculation.  
  * If True, the guider uses the model's predicted denoised image (cond\_denoised, or x0) for its calculations. The final result is then converted back to a noise prediction.  
  * If False, the guider uses the model's direct noise prediction (cond, or epsilon).  
    Guiding on the predicted image is often more stable and can produce qualitatively better results, as it operates in a more semantically meaningful space.  
* **Interaction with Other Parameters:** This is a fundamental choice in how guidance is applied and affects the behavior of all other guidance parameters.  
* **Default Value & Rationale:** True. Guiding on the x0 prediction is generally considered a more modern and stable approach, so it is the default.

#### **4.13. Parameter: mode**

* **UI Label:** mode  
* **Internal Variable Name:** mode  
* **Data Type & Constraints:** COMBO (list of strings like "pure\_apg", "pre\_cfg", etc.).  
* **Algorithmic Impact:** This selects the core guidance algorithm. The string is split by \_.  
  * The first part (pure or pre) determines the injection point. pure injects a sampler\_cfg\_function, replacing the CFG calculation. pre injects a sampler\_pre\_cfg\_function, modifying the conditioning *before* it gets to the standard CFG calculation.  
  * The second part (apg, alt1, alt2) determines the momentum update logic inside the APG.update method. apg is the default EMA, while alt1 and alt2 are experimental variations.  
* **Interaction with Other Parameters:** pre\_cfg modes can behave very differently from pure modes and may require different apg\_scale and cfg values to achieve a similar effect.  
* **Default Value & Rationale:** "pure\_apg". This is the most direct, stable, and well-understood implementation of the APG concept, making it the safest and most useful default.

#### **4.14. Parameter: yaml\_parameters\_opt**

* **UI Label:** yaml\_parameters\_opt  
* **Internal Variable Name:** yaml\_parameters\_opt  
* **Data Type & Constraints:** STRING (multiline). Must be valid YAML format.  
* **Algorithmic Impact:** This is the master override. If the string in this field is not empty, the go method of the node will invoke yaml.safe\_load to parse it. The resulting Python dictionary or list is used to construct the rules tuple of APGConfig objects, **completely bypassing and ignoring all other parameter widgets on the node UI** (with the exception of the master disable\_apg and verbose\_debug toggles). This allows for the definition of schedules far more complex than the simple start/end sigma model offered by the UI.  
* **Interaction with Other Parameters:** Overrides apg\_scale, cfg\_before, cfg\_after, momentum, start\_sigma, end\_sigma, etc. It is the single source of truth for the guidance schedule when used.  
* **Default Value & Rationale:** "" (empty string). The node defaults to using its interactive UI widgets, making it accessible to users who are not familiar with YAML. The YAML input is an optional power-user feature.

### **5\. Applied Use-Cases & Recipes**

#### **5.1. Recipe 1: High-Frequency Detail Enhancement ("Crisp Look")**

* **Objective:** To generate an image with extremely sharp, well-defined details and textures, suitable for subjects like intricate machinery, detailed fabrics, or sharp landscapes.  
* **Rationale:** This recipe leverages a strong negative momentum (-0.85). This transforms the momentum accumulator into a sharpening filter. At each step, instead of averaging with the previous guidance direction, it subtracts a scaled version of it, thus amplifying the change (the "delta") from the previous step. This forces the guidance to be highly responsive and aggressive in carving out fine details in the latent space. The effect is active during the main denoising phase (sigma 14.0 down to 1.5) and is then turned off for the final, low-noise cleanup steps to ensure a clean result.  
* **Parameter Configuration (YAML):**  
  YAML  
  \# Crisp & Detailed Look  
  verbose: false  
  rules:  
    \- start\_sigma: \-1.0  
      apg\_scale: 0.0  
      cfg: 7.0  
    \- start\_sigma: 14.0  
      apg\_scale: 6.5  
      predict\_image: true  
      cfg: 6.5  
      mode: pure\_apg  
      momentum: \-0.85  
      norm\_threshold: 2.0  
    \- start\_sigma: 1.5  
      apg\_scale: 0.0  
      cfg: 5.0

#### **5.2. Recipe 2: Coherent Painterly Style ("Smooth Look")**

* **Objective:** To generate an image that eschews sharp digital details in favor of a soft, blended, and organic appearance reminiscent of traditional painting.  
* **Rationale:** This approach uses a very high positive momentum (0.9). This creates a powerful low-pass filter on the guidance vector's trajectory through the sampling steps. By heavily averaging the current guidance direction with the history of previous directions, it smooths out any abrupt, jerky movements. This prevents the model from locking onto high-frequency noise and instead encourages fluid, blended transitions, which manifests visually as a painterly or smoothed style. The apg\_scale and cfg are kept moderate to complement this soft approach.  
* **Parameter Configuration (YAML):**  
  YAML  
  \# Smooth & Painterly Look  
  verbose: false  
  rules:  
    \- start\_sigma: \-1.0  
      apg\_scale: 0.0  
      cfg: 5.0  
    \- start\_sigma: 10.0  
      apg\_scale: 4.5  
      predict\_image: true  
      cfg: 4.0  
      mode: pure\_apg  
      momentum: 0.9  
      norm\_threshold: 3.5  
    \- start\_sigma: 2.0  
      apg\_scale: 0.0  
      cfg: 3.0

#### **5.3. Recipe 3: Stable Audio Generation Guidance**

* **Objective:** To guide an audio diffusion model with enhanced stability, shaping the timbre and character of the sound without introducing chaotic noise or artifacts common with high CFG on audio models.  
* **Rationale:** Audio latents are fundamentally different from image latents. Their dimensions often represent time and frequency. This recipe makes one critical change: dims: \[-1\]. This tells the guider to perform all its vector math along the last dimension only (e.g., frequency), which is appropriate for many audio models. It uses a gentle positive momentum (0.6) to stabilize the guidance, preventing the chaotic oscillations that can occur in audio generation, ensuring a more coherent and listenable result.  
* **Parameter Configuration (YAML):**  
  YAML  
  \# Stable Audio Guidance  
  verbose: false  
  rules:  
    \- start\_sigma: \-1.0  
      apg\_scale: 0.0  
      cfg: 5.0  
    \- start\_sigma: 8.0  
      apg\_scale: 4.5  
      predict\_image: true  
      cfg: 4.0  
      mode: pure\_apg  
      dims: \[\-1\]  
      momentum: 0.6  
      norm\_threshold: 2.5  
    \- start\_sigma: 2.0  
      apg\_scale: 0.0  
      cfg: 3.0

### **6\. Implementation Deep Dive**

#### **6.1. Source Code Walkthrough**

Execution flows from node instantiation to per-step application within the sampler.

1. **Instantiation (APGGuiderNode.go):** When the workflow is queued, the go method is invoked. Its sole purpose is to parse the configuration and instantiate the APGGuider object. It first checks if yaml\_parameters\_opt contains text.  
   * **YAML Path:** If text is present, yaml.safe\_load is called. The resulting data structure is used to build a list of APGConfig named tuples by iterating through the rules list in the YAML data.  
   * **UI Path:** If the YAML field is empty, the \_build\_rules\_from\_inputs helper function is called. This function constructs a simple 2 or 3-rule schedule based on the values of the UI widgets (start\_sigma, end\_sigma, cfg\_before, cfg\_after, etc.).  
   * **Finalization:** The resulting list of APGConfig objects is sorted by start\_sigma in descending order, a fallback rule is added if necessary, and the final tuple of rules is passed to the APGGuider class constructor, which returns the functional guider object.  
2. **Sampler Interaction (APGGuider.predict\_noise):** During generation, the custom sampler calls the guider's predict\_noise method for each timestep. This is the heart of the execution.  
   * The method first gets the numerical sigma value for the current timestep.  
   * It then calls self.apg\_get\_match(sigma), which iterates through its sorted self.apg\_rules tuple and returns the first APG object for which sigma \<= rule.start\_sigma.  
   * Crucially, self.apg\_reset(exclude=rule) is called to clear the running\_average momentum state of all *other*, inactive rules. This prevents state from one sigma range from incorrectly influencing another.  
   * It checks if the matched rule is active (rule.apg\_scale \!= 0). If so, it injects the appropriate guidance function (rule.cfg\_function or rule.pre\_cfg\_function) into the model\_options dictionary that will be passed to the model's forward pass. It also sets the cfg value for the sampler's superclass logic.  
   * Finally, it calls super().predict\_noise(...), which executes the standard CFG forward pass, but now the sampler will use the injected function during its internal guidance calculation.  
3. **APG Calculation (APG.apg):** The injected function from the active rule calls self.apg(...). This method performs the core mathematics:  
   * It calculates the raw guidance vector cond \- uncond.  
   * It passes this vector to self.update(), which applies the momentum calculation.  
   * It then applies the norm\_threshold clamp if configured.  
   * The resulting, processed guidance vector is passed to self.project(), which performs the dot product and vector subtraction to isolate the orthogonal component diff\_o.  
   * This orthogonal vector is returned to the injected function, which scales it by apg\_scale and adds it to the conditional prediction to produce the final guided result.

#### **6.2. Dependencies & External Calls**

* **PyYAML:** This external library is essential for the node's most powerful feature. It is used to safely parse the user-provided YAML string into a nested Python dictionary/list structure that the node can then interpret. yaml.safe\_load is used specifically to prevent arbitrary code execution, a critical security measure.  
* **torch:** The foundational library for all numerical operations. The core APG logic relies heavily on PyTorch's optimized tensor functions, including element-wise arithmetic, torch.norm for calculating vector magnitude, .sum for dot products, and torch.nn.functional.normalize.  
* **comfy.samplers.CFGGuider:** This is the base class from which APGGuider inherits. This inheritance provides the fundamental structure for storing conditioning, setting a cfg value, and the original predict\_noise method which is called via super(). The node extends and overrides this base behavior.

#### **6.3. Performance & Resource Analysis**

* **Execution Target:** The guidance logic itself—vector math and rule matching—is executed on the **CPU**. However, the mathematical operations are performed on PyTorch tensors which reside on the selected compute device (typically GPU).  
* **VRAM Usage:** VRAM consumption is not meaningfully increased. The APG logic operates on the cond and uncond prediction tensors that are already required for standard CFG and are present in VRAM. The temporary tensors created for the guidance vectors (pred\_diff, diff\_o, etc.) are relatively small and have a very short lifecycle, being released from memory after each step.  
* **Computational Overhead:** The APG calculation adds a small but non-zero overhead to each sampling step compared to standard CFG. This overhead consists of a few dot products, normalizations, and element-wise operations per step. For a typical 20-30 step generation, this overhead is negligible compared to the massive computational cost of the U-Net's forward pass at each step, which remains the primary performance bottleneck.

#### **6.4. Tensor Lifecycle Analysis**

As a controller, this node doesn't have a simple input-to-output tensor flow. Instead, it interacts with tensors during the sampler's loop.

1. **Initialization:** The positive and negative conditioning tensors are received and stored as attributes within the APGGuider object instance. They persist for the duration of the sampling process.  
2. **Per-Step Interaction:** At each step t, the sampler provides the model's prediction tensors, cond\_t and uncond\_t, which reside on the GPU.  
3. **Internal State Tensor:** The active rule's running\_average tensor (from step t-1) is retrieved. This tensor also resides on the GPU and has the same shape as the prediction tensors.  
4. **Temporary Calculation Tensors:** Inside the apg method, the following short-lived tensors are created on the GPU:  
   * pred\_diff: The raw or momentum-updated guidance vector.  
   * diff\_norm, scale\_factor: Tensors used for the norm\_threshold calculation.  
   * v0\_p, v0\_o: The parallel and orthogonal component tensors created during projection.  
5. **Modification & Return:** These temporary tensors are used to compute the final guided prediction tensor for step t. This final tensor is returned to the sampler.  
6. **Cleanup:** Once the prediction is returned, all temporary calculation tensors (pred\_diff, diff\_o, etc.) go out of scope and their memory is freed by PyTorch's garbage collector. The only internal tensor that persists to the next step (t+1) is the updated running\_average.

### **7\. Troubleshooting & Diagnostics**

#### **7.1. Error Code Reference**

* **Error Message/Traceback Snippet:** RuntimeError: Could not get APG rule for sigma=X  
  * **Root Cause Analysis:** This error is explicitly raised by the apg\_get\_match method when its loop completes without finding a rule in self.apg\_rules where current\_sigma \<= rule.start\_sigma. The node's go method is designed to prevent this by always appending a final fallback rule with start\_sigma=math.inf. Therefore, seeing this error indicates a critical failure in the rule-building logic, likely from a malformed or empty rules list provided via YAML.  
  * **Primary Solution:** Ensure your YAML configuration is not empty and contains at least one rule. If not using YAML, this may indicate a bug in the node's \_build\_rules\_from\_inputs helper. The simplest fix is to provide a basic YAML rule that covers all sigmas, such as rules: \[{start\_sigma: \-1.0, cfg: 7.0, apg\_scale: 0.0}\].  
* **Error Message/Traceback Snippet:** ValueError: Error parsing YAML parameters: ... followed by a yaml.YAMLError.  
  * **Root Cause Analysis:** The string in the yaml\_parameters\_opt input is not syntactically valid YAML. The most common cause by far is inconsistent or incorrect indentation. YAML is whitespace-sensitive and strictly requires spaces (not tabs) for indentation.  
  * **Primary Solution:** Scrutinize the indentation of your YAML text. Ensure all items in the rules list begin with a hyphen (- ) at the same indentation level, and all parameters for a given rule (start\_sigma, apg\_scale, etc.) are indented further, consistently. Use an online YAML validator to check your syntax.

#### **7.2. Unexpected Visual Artifacts & Behavior**

* **Artifact:** Generated images are chaotic, noisy, have extreme color burning, or appear "torn."  
  * **Likely Cause(s):** The guidance vector's magnitude is becoming pathologically large, causing the sampler to make divergent, unstable steps in the latent space. This is a classic sign of over-guidance, often caused by a combination of high apg\_scale and high negative momentum.  
  * **Correction Strategy:** The primary tool to combat this is the norm\_threshold. Lower this value (e.g., to 2.5, 2.0, or even 1.5) to enforce a hard clamp on the magnitude of the guidance vector at each step. Additionally, reduce apg\_scale or bring momentum closer to zero.  
* **Artifact:** The node appears to have no effect, and the output is identical to a standard generation.  
  * **Likely Cause(s):** The APG logic is being bypassed or nullified. This can happen for several reasons:  
    1. The disable\_apg toggle is set to True.  
    2. The apg\_scale for the active rule(s) is set to 0.0 or 1.0.  
    3. The start\_sigma of your main APG rule is set too low (e.g., 2.0), meaning it only activates for the last few refinement steps where its impact will be minimal.  
  * **Correction Strategy:** First, confirm disable\_apg is False. Second, enable verbose\_debug and check the console output. This will show you exactly which rule is active at each sigma and what its parameters are. If you see your APG rule is not activating until late in the process, increase its start\_sigma value (e.g., to 14.0 or higher, or \-1.0 for the very start).  
* **Artifact:** The YAML configuration is being ignored, and the node is using the UI slider values.  
  * **Likely Cause(s):** The yaml\_parameters\_opt text field likely contains text that is parsed as empty by the YAML loader, such as only comments (\#...) or whitespace. If yaml.safe\_load returns None or an empty structure, the node will fall back to its UI-based \_build\_rules\_from\_inputs logic.  
  * **Correction Strategy:** Ensure your YAML text is valid and not commented out. The simplest test is to add verbose: true at the top level of the YAML. If you do not see verbose output in the console after queuing, the YAML was not parsed correctly.