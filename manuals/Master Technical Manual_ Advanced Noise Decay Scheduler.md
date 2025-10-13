---

## **Master Technical Manual: Advanced Noise Decay Scheduler**

### **Node Name: NoiseDecayScheduler\_Custom**

Display Name: Noise Decay Scheduler (Advanced)  
Category: MD\_Nodes/Schedulers  
Version: 0.6.0  
Last Updated: 2025-09-15

---

### **Table of Contents**

1. Introduction  
   1.1. Executive Summary  
   1.2. Conceptual Category  
   1.3. Problem Domain & Intended Application  
   1.4. Key Features (Functional & Technical)  
2. Core Concepts & Theory  
   2.1. Theoretical Background: Custom Noise Schedules  
   2.2. Mathematical & Algorithmic Formulation  
   2.3. Data I/O Deep Dive  
   2.4. Strategic Role in the ComfyUI Graph  
3. Node Architecture & Workflow  
   3.1. Node Interface & Anatomy  
   3.2. Input Port Specification  
   3.3. Output Port Specification  
   3.4. Workflow Schematics (Minimal & Advanced)  
4. Parameter Specification  
   4.1. Parameter: algorithm\_type  
   4.2. Parameter: decay\_exponent  
   4.3. Parameter: start\_value  
   4.4. Parameter: end\_value  
   4.5. Parameter: invert\_curve  
   4.6. Parameter: use\_caching  
   4.7. Parameter: enable\_temporal\_smoothing  
   4.8. Parameter: smoothing\_window  
   4.9. Parameter: custom\_piecewise\_points  
   4.10. Parameter: fourier\_frequency  
5. Applied Use-Cases & Recipes  
   5.1. Recipe 1: Aggressive Compositional "Cliff Drop"  
   5.2. Recipe 2: Mid-Point Detail Push with a Gaussian Curve  
   5.3. Recipe 3: Rhythmic "Breathing" Effect for Animation  
6. Implementation Deep Dive  
   6.1. Source Code Walkthrough  
   6.2. Dependencies & External Calls  
   6.3. Performance & Resource Analysis  
   6.4. Data Lifecycle Analysis  
7. Troubleshooting & Diagnostics  
   7.1. Error Code Reference  
   7.2. Unexpected Visual Artifacts & Behavior

---

### **1\. Introduction**

#### **1.1. Executive Summary**

The **Advanced Noise Decay Scheduler** is a specialized utility node for ComfyUI that functions as a high-precision, procedural curve generator. Its sole purpose is to create a custom **noise decay schedule**, which is a functional object that dictates the magnitude of noise to be added or removed at each step of a compatible custom sampler's process. By offering a suite of distinct mathematical algorithms—including Polynomial, Sigmoidal, Fourier, and Gaussian—and a powerful set of manipulation controls, the node allows users to design highly specific denoising trajectories. This externalizes the scheduling logic from the sampler, enabling artists and technicians to exert surgical control over the final image's texture, detail clarity, and temporal behavior in animations. It is a tool for fundamentally altering the "character" of the diffusion process.

#### **1.2. Conceptual Category**

**Sampler Control Object Generator.** This node does not process any image, latent, or audio data. It is a factory that takes a set of user-defined parameters and constructs a SCHEDULER object. This functional object is then passed to a compatible custom sampler (like the PingPong Sampler series) to override its default scheduling behavior.

#### **1.3. Problem Domain & Intended Application**

* **Problem Domain:** Standard diffusion samplers and even advanced sigma schedulers typically rely on a predefined set of monotonically decreasing curves (like Karras). While effective for photorealistic or standard outputs, these fixed schedules offer limited artistic control over the *dynamics* of the generation process. It is difficult or impossible to create effects like rhythmic pulsing, mid-point focus shifts, or non-standard noise injection using these tools. The Advanced Noise Decay Scheduler addresses this limitation by providing a toolkit of diverse mathematical functions, allowing for the design of virtually any curve shape, thereby unlocking a new dimension of experimental and stylistic control.  
* **Intended Application (Use-Cases):**  
  * **Stylistic Control:** Designing unique visual textures by creating non-standard decay curves. A slow, gentle decay can produce soft, painterly looks, while a noisy, chaotic curve can generate glitch or raw aesthetics.  
  * **Advanced Animation:** Creating temporal effects by using oscillating algorithms like fourier. When each frame of an animation uses the same schedule, the texture of the image will appear to "breathe" or "pulse" in sync with the curve's oscillations.  
  * **Process Control:** Forcing the sampler to spend more or less time in specific noise regions to influence the final output. A gaussian curve, for example, can force the model to resolve the bulk of the image's detail within a very narrow window of steps.  
  * **Experimental Effects:** Using invert\_curve to create noise *growth* schedules, which can be used for controlled noise injection or other abstract generative techniques.  
* **Non-Application (Anti-Use-Cases):**  
  * The node is **not a sampler**. It does nothing on its own and will have no effect on the workflow unless its scheduler output is connected to a compatible sampler that has a scheduler input socket.  
  * It is **not a sigma scheduler**. It does not output a SIGMAS tensor. The curve it generates is typically used by custom samplers to modulate other aspects of the sampling step, not to define the sigma values themselves.

#### **1.4. Key Features (Functional & Technical)**

* **Functional Features:**  
  * **Six Generation Algorithms:** Provides a versatile toolkit of curve shapes: polynomial for aggressive curves, sigmoidal for cinematic S-curves, piecewise for manual point-by-point design, fourier for oscillations, exponential for natural decay, and gaussian for an inverted bell curve.  
  * **Total Curve Manipulation:** Full control over the final output range with start\_value and end\_value overrides, plus a powerful invert\_curve toggle to flip the entire schedule vertically.  
  * **Configurable Temporal Smoothing:** An optional moving average filter (enable\_temporal\_smoothing) can be applied to any generated curve to smooth out sharp transitions, which is critical for preventing jitter and creating more organic effects in animations.  
  * **Designed for Custom Samplers:** The primary purpose of this node is to provide a SCHEDULER object to advanced samplers, such as the PingPong Sampler family, that are built to accept external schedule controllers.  
* **Technical Features:**  
  * **Object-Oriented Design:** The node's generate function instantiates and returns a NoiseDecayObject. This object encapsulates all the configuration and logic, allowing it to be passed around the workflow as a single unit. The sampler then calls the object's get\_decay method.  
  * **Intelligent Caching:** Implemented via the use\_caching toggle. When enabled, the node generates a unique hash key from all configuration parameters. The computed decay curve (a NumPy array) is stored in a dictionary against this key. On subsequent runs with identical settings, the curve is instantly retrieved from this cache, avoiding redundant computation and dramatically speeding up iterative workflows.  
  * **Procedural Curve Generation:** All curves are generated procedurally using the numpy library. The logic first computes a normalized base curve (typically spanning 1.0 to 0.0) and then applies a fixed sequence of transformations (smoothing, inversion, rescaling) to arrive at the final output.

### **2\. Core Concepts & Theory**

#### **2.1. Theoretical Background: Custom Noise Schedules**

In a typical diffusion process, a **sigma schedule** dictates the absolute noise level at each step, guiding the image from pure noise to a clean output. This node introduces a parallel concept: a **decay schedule** or **modulation curve**. This is a separate schedule, usually normalized between 0.0 and 1.0, that is used by a custom sampler as a *multiplier* or *modulator* for some internal process at each step.

For example, a PingPong Sampler might use this decay curve to control the amount of noise to *re-inject* at each step.

* A decay value of 1.0 might mean "re-inject 100% of the intended noise for this step."  
* A decay value of 0.2 might mean "only re-inject 20% of the intended noise."

By designing the *shape* of this modulation curve, the user is not setting the absolute noise level, but rather controlling the *intensity* of a sampler's action at each point along the pre-existing sigma schedule. This provides a powerful, orthogonal layer of control over the generation's dynamics.

#### **2.2. Mathematical & Algorithmic Formulation**

The node generates a curve C, which is an array of N floating-point values, where N is the number of steps requested by the sampler.

1. **Base Curve Generation (Cbase​):** A normalized curve is generated first. For a given number of steps N, an array of indices x is created, where x=\[0,N−11​,N−12​,…,1\].  
   * **Polynomial:** Cbase​(x)=(1−x)p, where p is decay\_exponent.  
   * **Fourier:** Cbase​(x)=(cos(f⋅π⋅x)+1)/2, where f is fourier\_frequency.  
   * **Gaussian:** Cbase​(x)=1−e−((2x−1)2/(2σ2)), where σ is derived from decay\_exponent.  
2. Temporal Smoothing (if enabled):  
   The base curve is convolved with a moving average window. For a window of size W:

   Csmoothed​\[i\]=W1​j=0∑W−1​Cbase​\[i+j−W//2\]  
3. Inversion (if enabled):

   Cinverted​=1.0−Csmoothed​  
4. Rescaling to Final Values:  
   The final curve is linearly rescaled from its native range (e.g., \[0, 1\]) to the user-defined range \[end\_value, start\_value\].

   Cfinal​=Cinverted​⋅(start\_value−end\_value)+end\_value

#### **2.3. Data I/O Deep Dive**

* **Inputs:** The node has no data inputs (like MODEL or LATENT). All inputs are control parameters (floats, integers, booleans, strings).  
* **Output:**  
  * scheduler (SCHEDULER):  
    * **Data Specification:** This is not a data tensor. It is a custom ComfyUI type, representing a Python **object instance** of the internal NoiseDecayObject class. This object contains all the user-configured parameters and the core get\_decay method that the custom sampler will call.

#### **2.4. Strategic Role in the ComfyUI Graph**

* **Placement Context:** This node is a **controller factory** for a custom sampler. It sits alongside the main workflow and plugs *into* the sampler, but is not part of the main MODEL \-\> CLIP \-\> KSampler \-\> VAE data flow.  
* **Integration with Custom Samplers:** This node's primary and sole purpose is to connect to a compatible custom sampler that has a scheduler input socket, most notably the **PingPong Sampler** family. The sampler, upon receiving this scheduler object, will call its get\_decay(num\_steps) method at the beginning of its process to retrieve the custom curve. It will then use the values from this curve at each step to modulate its behavior.  
* **Synergistic Nodes:**  
  * **PingPongSampler\_Custom (and variants):** The intended target for the scheduler output. The sampler uses the curve from this node to control its unique "ping-pong" noise injection and removal process.  
  * **Reroute:** Useful for sending the same custom scheduler object to multiple samplers, ensuring they all share the exact same dynamic behavior.  
* **Conflicting Nodes:**  
  * **Standard KSampler / SamplerCustom:** These nodes do not have a scheduler input socket and cannot accept the object this node produces. Connecting it to them is not possible. It is not a SIGMAS or GUIDER node and cannot be used in their place.

### **3\. Node Architecture & Workflow**

#### **3.1. Node Interface & Anatomy**

1. algorithm\_type (Dropdown Widget)  
2. decay\_exponent (Float Widget)  
3. start\_value (Float Widget)  
4. end\_value (Float Widget)  
5. invert\_curve (Toggle Widget)  
6. use\_caching (Toggle Widget)  
7. enable\_temporal\_smoothing (Toggle Widget)  
8. smoothing\_window (Integer Widget)  
9. custom\_piecewise\_points (String Widget)  
10. fourier\_frequency (Float Widget)  
11. scheduler (Output Port)

#### **3.2. Input Port Specification**

This node has no input ports. All parameters are configured via widgets on the node's UI.

#### **3.3. Output Port Specification**

* **scheduler (SCHEDULER)** \- (Anatomy Ref: \#11)  
  * **Description:** This port outputs a functional NoiseDecayObject instance, fully configured with the user's settings. This object is a self-contained "curve generator" that a compatible sampler can query to get a custom decay schedule.  
  * **Resulting Data:** A Python object, not a tensor.

#### **3.4. Workflow Schematics**

* Minimal Functional Graph:  
  The Advanced Noise Decay Scheduler node is created. Its scheduler output is connected to the scheduler input of a PingPongSampler\_Custom. The sampler's other inputs (model, cond, latent, etc.) are connected as usual. This demonstrates the core integration pattern.

### **4\. Parameter Specification**

#### **4.1. Parameter: algorithm\_type**

* **UI Label:** algorithm\_type  
* **Internal Variable Name:** self.algorithm\_type (within the NoiseDecayObject)  
* **Data Type & Constraints:** COMBO (Dropdown list of strings).  
* **Algorithmic Impact:** This parameter is the primary control switch for the node's behavior. It selects which of the six base curve generation methods (\_compute\_polynomial\_decay, \_compute\_sigmoidal\_decay, etc.) is called within the get\_decay function. The choice of algorithm determines the fundamental mathematical shape of the normalized decay curve before any further manipulations (smoothing, inversion, rescaling) are applied.  
* **Default Value & Rationale:** "polynomial". This is a versatile and powerful default, offering a wide range of aggressive or gentle curves controlled by the decay\_exponent, making it an intuitive starting point.

#### **4.2. Parameter: decay\_exponent**

* **UI Label:** decay\_exponent  
* **Internal Variable Name:** self.decay\_exponent  
* **Data Type & Constraints:** FLOAT, min: 0.1, max: 10.0.  
* **Algorithmic Impact:** This parameter controls the shape or steepness for three of the algorithms:  
  * **polynomial:** Serves as the exponent p in (1 \- x)^p. Higher values create a much steeper initial drop (a "cliff").  
  * **sigmoidal:** Used as a multiplier to control the steepness of the S-curve's transition phase.  
  * **exponential:** Serves as the exponent in exp(-p \* x). Higher values cause a faster decay.  
  * **gaussian:** Used to derive the standard deviation (sigma), controlling the "width" of the inverted bell curve.  
* **Interaction with Other Parameters:** This parameter is ignored if the algorithm\_type is piecewise or fourier.  
* **Default Value & Rationale:** 2.0. A quadratic polynomial decay, which is a common and well-behaved curve shape, providing a noticeable but not extreme effect.

#### **4.3. Parameter: start\_value**

* **UI Label:** start\_value  
* **Internal Variable Name:** self.start\_value  
* **Data Type & Constraints:** FLOAT, min: \-2.0, max: 2.0.  
* **Algorithmic Impact:** This is the final step in the processing chain. After the base curve has been generated, smoothed, and inverted, it is linearly rescaled to fit a new range. This parameter defines the value of the very first point of the final output curve. It allows the user to define the exact starting intensity of the schedule.  
* **Interaction with Other Parameters:** Works in conjunction with end\_value to define the final output range of the curve.  
* **Default Value & Rationale:** 1.0. A standard normalized starting value, representing 100% intensity.

#### **4.4. Parameter: end\_value**

* **UI Label:** end\_value  
* **Internal Variable Name:** self.end\_value  
* **Data Type & Constraints:** FLOAT, min: \-2.0, max: 2.0.  
* **Algorithmic Impact:** This parameter defines the value of the very last point of the final output curve. It is used in the final rescaling calculation: final\_curve \= base\_curve \* (start\_value \- end\_value) \+ end\_value. This allows the user to specify that a schedule should not decay to complete zero, but instead to a small residual value, for example.  
* **Interaction with Other Parameters:** Works in conjunction with start\_value to define the final output range of the curve.  
* **Default Value & Rationale:** 0.0. A standard normalized ending value, representing 0% intensity.

#### **4.5. Parameter: invert\_curve**

* **UI Label:** invert\_curve  
* **Internal Variable Name:** self.invert\_curve  
* **Data Type & Constraints:** BOOLEAN.  
* **Algorithmic Impact:** If True, this toggle inserts an inversion step in the processing chain immediately after base curve generation and smoothing. The operation is decay\_values \= 1.0 \- decay\_values. This vertically flips the normalized curve. A decay that goes from 1 to 0 becomes a growth curve that goes from 0 to 1\. This is a powerful tool for experimental effects, such as creating a schedule for progressive noise *injection* rather than removal.  
* **Interaction with Other Parameters:** This operation is applied *before* the final rescaling to start\_value and end\_value.  
* **Default Value & Rationale:** False. Standard behavior is a decay, not a growth, so inversion is disabled by default.

#### **4.6. Parameter: use\_caching**

* **UI Label:** use\_caching  
* **Internal Variable Name:** self.use\_caching  
* **Data Type & Constraints:** BOOLEAN.  
* **Algorithmic Impact:** This toggle controls the caching mechanism. If True, the get\_decay method first calls \_generate\_cache\_key to create an MD5 hash of all configuration parameters (including the number of steps). It checks if this hash exists in its internal self.\_cache dictionary. If so, it returns the cached NumPy array instantly, skipping all computation. If not, it computes the curve, stores it in the cache against the key, and then returns it. Disabling this forces re-computation on every run.  
* **Default Value & Rationale:** True. Caching provides a significant performance increase for iterative workflows where settings are not changed between runs, so it is enabled by default.

#### **4.7. Parameter: enable\_temporal\_smoothing**

* **UI Label:** enable\_temporal\_smoothing  
* **Internal Variable Name:** self.enable\_temporal\_smoothing  
* **Data Type & Constraints:** BOOLEAN.  
* **Algorithmic Impact:** If True, this enables the moving average filter. After the base curve is generated, it is passed to the \_apply\_temporal\_smoothing method, which applies a convolution with a uniform kernel of size smoothing\_window. This has the effect of averaging each point with its neighbors, smoothing out sharp transitions or "corners" in the curve.  
* **Interaction with Other Parameters:** The strength of the smoothing is controlled by smoothing\_window.  
* **Default Value & Rationale:** False. Smoothing alters the pure mathematical shape of the curve, so it is an optional effect disabled by default.

#### **4.8. Parameter: smoothing\_window**

* **UI Label:** smoothing\_window  
* **Internal Variable Name:** self.smoothing\_window  
* **Data Type & Constraints:** INT, min: 2, max: 20\.  
* **Algorithmic Impact:** This parameter is active only if enable\_temporal\_smoothing is True. It defines the size of the window for the moving average filter. A larger window size results in a stronger smoothing effect, averaging each point over a wider range of its neighbors and creating a much softer, less detailed curve. A smaller window size produces a more subtle smoothing.  
* **Interaction with Other Parameters:** Only used when enable\_temporal\_smoothing is True.  
* **Default Value & Rationale:** 3\. A small window size that provides gentle smoothing without excessively blurring the original curve shape.

#### **4.9. Parameter: custom\_piecewise\_points**

* **UI Label:** custom\_piecewise\_points  
* **Internal Variable Name:** self.custom\_piecewise\_points  
* **Data Type & Constraints:** STRING (comma-separated floats).  
* **Algorithmic Impact:** Active only when algorithm\_type is piecewise. The string is parsed into a list of float values. The \_compute\_piecewise\_decay method then uses these values as the Y-points for a linear interpolation. For example, "1.0, 0.2, 0.0" will create a curve that drops sharply from 1.0 to 0.2 over the first half of the steps, then decays more slowly from 0.2 to 0.0 over the second half. This offers complete manual control over the curve's shape.  
* **Interaction with Other Parameters:** This parameter is ignored by all other algorithms.  
* **Default Value & Rationale:** "1.0,0.5,0.0". A simple three-point linear decay, providing a basic and predictable starting point.

#### **4.10. Parameter: fourier\_frequency**

* **UI Label:** fourier\_frequency  
* **Internal Variable Name:** self.fourier\_frequency  
* **Data Type & Constraints:** FLOAT, min: 0.1, max: 10.0.  
* **Algorithmic Impact:** Active only when algorithm\_type is fourier. This value acts as the frequency multiplier f in the cosine function: (cos(f \* pi \* x) \+ 1\) / 2\. A value of 1.0 creates a single half-cosine wave (a simple decay). A value of 2.0 creates a full cosine wave. A value of 3.0 creates one and a half waves, and so on. This directly controls the number of oscillations in the schedule.  
* **Interaction with Other Parameters:** Ignored by all other algorithms.  
* **Default Value & Rationale:** 1.0. A single, simple decay curve, making the Fourier mode behave predictably by default.

### **5\. Applied Use-Cases & Recipes**

#### **5.1. Recipe 1: Aggressive Compositional "Cliff Drop"**

* **Objective:** To force the sampler to establish the main shapes and composition of the image very early in the process, spending the vast majority of steps on refinement.  
* **Rationale:** The polynomial algorithm with a high decay\_exponent creates a curve that drops extremely rapidly from its start value and then has a long, low-value "tail." When a PingPong sampler uses this curve to modulate noise re-injection, it means noise is aggressively removed in the first few steps and very little is re-added thereafter, locking in the composition early.  
* **Parameter Configuration:**  
  * algorithm\_type: polynomial  
  * decay\_exponent: 6.0  
  * enable\_temporal\_smoothing: True  
  * smoothing\_window: 3 (to slightly soften the sharp "knee" of the cliff)

#### **5.2. Recipe 2: Mid-Point Detail Push with a Gaussian Curve**

* **Objective:** To create an image where the details resolve rapidly in the middle of the generation process, rather than linearly throughout.  
* **Rationale:** The gaussian algorithm creates an inverted bell curve. The decay value stays high (e.g., near 1.0) for the initial steps, drops very quickly in the middle, and then stays low (e.g., near 0.0) for the final steps. This forces the sampler to maintain a high level of noise/chaos for longer than usual, then undergo a very rapid clarification phase, before entering a final, low-energy refinement stage.  
* **Parameter Configuration:**  
  * algorithm\_type: gaussian  
  * decay\_exponent: 2.5 (a moderate width for the bell curve)

#### **5.3. Recipe 3: Rhythmic "Breathing" Effect for Animation**

* **Objective:** To create a visual pulsing or "breathing" effect in the texture of an animation sequence.  
* **Rationale:** The fourier algorithm generates a cosine wave. By setting fourier\_frequency to a value greater than 1 (e.g., 3.0), the decay curve will oscillate up and down multiple times. When this schedule is used for every frame of an animation, the sampler's modulated action will oscillate in sync, causing the textures and details to appear and recede rhythmically. Setting start\_value and end\_value to be less than 1.0 and greater than 0.0 respectively contains the oscillation, preventing extreme noise states.  
* **Parameter Configuration:**  
  * algorithm\_type: fourier  
  * fourier\_frequency: 3.0  
  * start\_value: 0.9  
  * end\_value: 0.1

### **6\. Implementation Deep Dive**

#### **6.1. Source Code Walkthrough**

The node consists of a main class NoiseDecayScheduler\_Custom and a nested class NoiseDecayObject.

1. **Instantiation (generate method):** When the workflow runs, the main class's generate method is called. It is extremely simple: it takes all the parameters from the UI as keyword arguments (\*\*kwargs) and uses them to instantiate the NoiseDecayObject. It then returns a tuple containing this single object, conforming to the RETURN\_TYPES \= ("SCHEDULER",).  
2. **Sampler Call (get\_decay method):** Later, the custom sampler that received this object will call its get\_decay(num\_steps) method. This is where the core logic executes.  
3. **Caching:** The get\_decay method first checks if self.use\_caching is True. If so, it calls \_generate\_cache\_key which creates an MD5 hash from a string representation of all functional parameters. If this key exists in the self.\_cache dictionary, the stored NumPy array is returned immediately, and the process ends.  
4. **Base Curve Generation:** If no cached value is found, the method proceeds to a dictionary lookup on self.algorithm\_type. This calls the appropriate private method (e.g., \_compute\_polynomial\_decay). Each of these methods uses numpy.linspace to create a normalized step array and then applies its specific mathematical formula to generate a base curve, which is a NumPy array.  
5. Chain of Transformations: The generated decay\_values array is then passed through a fixed sequence of optional transformations:  
   a. It's passed to \_apply\_temporal\_smoothing if enable\_temporal\_smoothing is true.  
   b. The result is then inverted (1.0 \- decay\_values) if invert\_curve is true.  
   c. Finally, the result of the previous steps is linearly rescaled using the start\_value and end\_value parameters.  
6. **Cache Storage and Return:** The final NumPy array is stored in self.\_cache using the generated key (if caching is enabled) and then returned to the sampler.

#### **6.2. Dependencies & External Calls**

* **numpy:** The exclusive backend for all numerical operations. It is used for creating the initial step array (linspace), applying all mathematical functions (cos, exp, \*\*), and performing the convolution for the moving average smoothing (convolve).  
* **hashlib:** A standard Python library used to implement the caching mechanism. hashlib.md5 is used to create a short, unique fingerprint of the node's settings to use as a dictionary key.

#### **6.3. Performance & Resource Analysis**

* **Execution Target:** All operations are performed on the **CPU**. The curve generation is a small NumPy calculation.  
* **VRAM Usage:** Zero. The node does not interact with the GPU or handle any large tensors.  
* **Bottlenecks:**  
  * **Without Caching:** The computation is trivial and effectively instantaneous. It will never be a bottleneck.  
  * **With Caching:** The first run involves the trivial computation plus a fast MD5 hash. All subsequent runs with the same settings are reduced to a single dictionary lookup, which is even faster. Performance is not a consideration for this node.

#### **6.4. Data Lifecycle Analysis**

1. **Stage 1 (Instantiation):** The node's generate method is called. It creates a NoiseDecayObject in memory. This object stores all the user-defined parameters as its attributes. No arrays or tensors are created yet. This object is outputted.  
2. **Stage 2 (Sampler Query):** The custom sampler receives the object and calls its get\_decay(num\_steps) method.  
3. **Stage 3 (Computation):** Inside get\_decay, a NumPy array of shape (num\_steps,) is created. It is processed in-place or by creating new arrays through the transformation chain.  
4. **Stage 4 (Return):** The final NumPy array is returned to the sampler. The sampler will then typically iterate through this array, using one value per step. The array persists in the sampler's memory for the duration of its sampling loop.  
5. **Stage 5 (Caching):** A copy of the final NumPy array is stored in the NoiseDecayObject's internal \_cache dictionary, where it persists until the workflow is changed or ComfyUI is restarted.

### **7\. Troubleshooting & Diagnostics**

#### **7.1. Error Code Reference**

This node is computationally simple and has few hard failure points. Most issues arise from misconfiguration rather than runtime errors.

* **Error Message/Traceback Snippet:** ValueError or TypeError from self.piecewise\_points \= \[float(x.strip()) ...\]  
  * **Root Cause Analysis:** The string provided to custom\_piecewise\_points is malformed. It contains non-numeric characters (other than commas or periods) that cannot be converted to a float.  
  * **Primary Solution:** Ensure the custom\_piecewise\_points string contains only numbers and commas, for example: "1.0, 0.8, 0.5, 0.1, 0.0".

#### **7.2. Unexpected Visual Artifacts & Behavior**

* **Artifact:** The generated image is chaotic, overly noisy, or pure black/gray.  
  * **Likely Cause(s):** The generated curve has an extreme or unintended shape. This is the most common issue and is considered a part of the experimental process. A start\_value much greater than 1.0, an inverted curve, or a very high-frequency Fourier wave can easily produce non-photorealistic results.  
  * **Correction Strategy:** This is a feature, not a bug. To return to a "normal" image, start with simple settings: algorithm\_type: polynomial, decay\_exponent: 2.0, start\_value: 1.0, end\_value: 0.0, and all toggles disabled. From this baseline, make one change at a time to understand its effect.  
* **Artifact:** Making changes to the parameters has no effect on the output.  
  * **Likely Cause(s):**  
    1. The use\_caching feature is enabled, and you are re-running a prompt without changing any settings on this node. The cached curve is being used.  
    2. You are adjusting a parameter that is not active for the selected algorithm\_type (e.g., changing fourier\_frequency when the mode is polynomial).  
  * **Correction Strategy:** If you suspect a stale cache, toggle use\_caching off and on again to clear it, or simply change any parameter slightly (like decay\_exponent from 2.0 to 2.01) to force a re-computation. Ensure you are adjusting the correct parameters for your chosen algorithm as detailed in the Parameter Specification section.  
* **Artifact:** The effect of the curve seems very weak or subtle.  
  * **Likely Cause(s):** The custom sampler you are using may only apply the decay curve as a weak modulator. The visual impact of this scheduler is highly dependent on how the receiving sampler chooses to interpret and use the curve it provides.  
  * **Correction Strategy:** Check the documentation or parameters of your custom sampler. It may have a "strength" or "influence" slider that controls how strongly it applies the external schedule. For example, a PingPong sampler might have a parameter that controls the maximum amount of noise to re-inject, which would act as a master control over this scheduler's effect.