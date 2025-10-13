---

## **Master Technical Manual: PingPong Sampler (Custom v0.9.9-p2 FBG)**

### **Node Name: PingPongSamplerNodeFBG**

Display Name: PingPong Sampler Enhanced (FBG \+ Optimizations)  
Category: MD\_Nodes/Samplers  
Version: 0.9.9-p3 (based on source code)  
Last Updated: 2025-09-16

---

### **Table of Contents**

1. Introduction  
   1.1. Executive Summary  
   1.2. Conceptual Category  
   1.3. Problem Domain & Intended Application  
   1.4. Key Features (Functional & Technical)  
2. Core Concepts & Theory  
   2.1. Theoretical Background: Ancestral Sampling vs. FBG  
   2.2. Feedback Guidance (FBG): The Self-Regulating Engine  
   2.3. Conditional Blending  
   2.4. Data I/O Deep Dive  
   2.5. Strategic Role in the ComfyUI Graph  
3. Node Architecture & Workflow  
   3.1. Node Interface & Anatomy  
   3.2. Input Port Specification  
   3.3. Output Port Specification  
   3.4. Workflow Schematics (Minimal & Advanced)  
4. Parameter Specification  
   4.1. Parameter Group: Ancestral Noise & Timing Controls  
   4.2. Parameter Group: Feedback Guidance (FBG) \- Core Engine  
   4.3. Parameter Group: FBG \- Autopilot vs. Manual Tuning  
   4.4. Parameter Group: FBG \- Sigma Range & Scheduling  
   4.5. Parameter Group: Advanced & Experimental Controls  
   4.6. Parameter Group: Diagnostic & Override Controls  
5. A Masterclass in Tuning Feedback Guidance  
   5.1. The Golden Rule: Disabling Autopilot for Manual Control  
   5.2. The Troubleshooting Playbook & "Safe Mode" Reset  
   5.3. Tuning fbg\_temp: The Primary Sensitivity Dial  
   5.4. Reading the Signs: Anatomy of a Healthy FBG Log  
6. Applied Use-Cases & Recipes  
   6.1. Recipe 1: Stable & Controlled Image Generation (Manual Tune)  
   6.2. Recipe 2: Aggressive & Experimental Audio Generation (Autopilot)  
   6.3. Recipe 3: The "Slow Burn" Finishing Kick (Flipped Autopilot)  
7. Implementation Deep Dive  
   7.1. Source Code Walkthrough  
   7.2. Dependencies & External Calls  
   7.3. Performance & Resource Analysis  
   7.4. Data Lifecycle Analysis  
8. Troubleshooting & Diagnostics  
   8.1. Error Code Reference  
   8.2. Unexpected Visual Artifacts & Behavior

---

### **1\. Introduction**

#### **1.1. Executive Summary**

The **PingPong Sampler (FBG)** is an advanced **sampler configuration node** that creates a highly dynamic and intelligent sampler object for ComfyUI. It combines two powerful concepts: the texture-rich "ping-pong" ancestral noise method and a sophisticated **Feedback Guidance (FBG)** engine. Unlike standard samplers with a fixed Classifier-Free Guidance (CFG) scale, the FBG system functions as an adaptive autopilot, dynamically adjusting the guidance strength at each step based on the model's posterior estimation—a measure of its "confidence" or "surprise." This allows the sampler to apply strong guidance when the model is uncertain and gentle guidance when it is confident, leading to potentially more detailed, coherent, and artifact-free results. The node is a factory for producing this complex sampler logic, which is then passed to an execution node (like SamplerCustomAdvanced) to run the sampling process.

#### **1.2. Conceptual Category**

**Advanced Sampler Configuration.** This node's function is to construct a functional SAMPLER object. It does not execute the sampling loop itself. It is a "brain" that encapsulates an entire sampling strategy, which is then plugged into a "body" (an execution sampler) that provides the core data (model, latents, etc.) and runs the step loop. This modular design separates the complex *logic* of the sampler from the *act* of sampling.

#### **1.3. Problem Domain & Intended Application**

* **Problem Domain:** The primary challenge with standard samplers is the static nature of the CFG scale. Finding the "perfect" CFG value is a manual process of trial and error, and a single value is often a compromise—too low and the image is incoherent, too high and it becomes oversaturated and artifact-heavy. The FBG system is designed to solve this by automating the process, creating a dynamic CFG scale that adapts to the needs of the generation in real-time. It aims to find the "sweet spot" of guidance at every single step, a task that is impossible with a fixed CFG value.  
* **Intended Application (Use-Cases):**  
  * **Experimental Image Generation:** For artists who want to break away from the characteristic "look" of standard samplers and explore outputs with unique textures and detail structures that can only arise from a dynamically guided process.  
  * **Complex Audio Generation:** For AI musicians using models like Ace-Step, where the FBG can help navigate the complex latent space of audio to produce more coherent and interesting sonic textures without exploding into noise.  
  * **Power-User Workflow Tuning:** For technicians who want the deepest possible level of control over the sampling process. The extensive set of FBG tuning parameters allows for the precise sculpting of the guidance curve's behavior.  
  * **Automated Quality Seeking:** Using the FBG "Autopilot" mode (t\_0/t\_1) to allow the sampler to self-regulate, potentially achieving high-quality results across different models without extensive manual CFG tuning.  
* **Non-Application (Anti-Use-Cases):**  
  * This node is not a simple "drop-in" replacement for KSampler for beginners. Its complexity and the number of interacting parameters require a solid understanding of the diffusion process to troubleshoot and tune effectively.  
  * It cannot function on its own and **must** be connected to an execution node like SamplerCustomAdvanced.

#### **1.4. Key Features (Functional & Technical)**

* **Functional Features:**  
  * **Dynamic Feedback Guidance (FBG):** The core feature. A self-regulating guidance system that algorithmically adjusts the CFG scale at each step based on model feedback, moving beyond a single, static cfg value.  
  * **Autopilot & Manual FBG Modes:** Offers both an automatic mode where the guidance curve is calculated based on start/end targets (t\_0, t\_1) and a fully manual mode for direct control over sensitivity (fbg\_temp) and drift (fbg\_offset).  
  * **Advanced Ancestral Noise Control:** Provides a choice of gaussian, uniform, or brownian noise types for injection, allowing for different textural effects.  
  * **Conditional Blending:** An expert feature to create if/then rules that switch the sampler's mathematical blending function mid-generation based on sigma thresholds or the magnitude of change in the latent.  
  * **Multi-Level Debugging:** A debug\_mode setting provides unparalleled console insight into the sampler's internal state, printing the dynamic guidance scale and log posterior at every step.  
* **Technical Features:**  
  * **Posterior Estimation:** The FBG engine is driven by a log\_posterior value, which is updated at each step. This value is derived from the Mean Squared Error (MSE) difference between the next-step predictions made from the conditional and unconditional model outputs. It serves as a proxy for the model's "surprise" or the magnitude of the guidance vector.  
  * **PID-like Control Loop:** The FBG's update rule for the guidance scale is analogous to a PID (Proportional-Integral-Derivative) controller. The pi (proportional), fbg\_temp (integral), and fbg\_offset (derivative/drift) parameters work together to create a responsive but stable control system.  
  * **Modular Sampler Object Output:** The node outputs a standard SAMPLER object, ensuring compatibility with the ecosystem of custom execution samplers like SamplerCustomAdvanced.  
  * **Full YAML Overrides:** A string input for a YAML configuration allows every single one of the node's numerous parameters to be defined in a portable, shareable text format.

### **2\. Core Concepts & Theory**

#### **2.1. Theoretical Background: Ancestral Sampling vs. FBG**

This sampler combines two distinct advanced concepts:

* **Ancestral Sampling:** As detailed in previous manuals, this is the process of injecting new random noise at each step to increase detail and exploration of the solution space. This node provides this feature with configurable noise types (gaussian, etc.) and timing controls.  
* **Feedback Guidance (FBG):** This is an entirely separate concept that operates on the **guidance scale (CFG)**. It can be used with or without ancestral noise. Its goal is to make the cfg value dynamic. While ancestral sampling adds *stochasticity*, FBG adds *adaptivity*.

#### **2.2. Feedback Guidance (FBG): The Self-Regulating Engine**

Standard CFG is an "open-loop" system: you set a value, and it is applied blindly. FBG creates a "closed-loop" system.

1. **Action:** The sampler applies guidance with the current scale, GSi​.  
2. **Measurement:** It runs the model and calculates the difference between the conditional and unconditional predictions. This difference is used to update an internal state variable called the log\_posterior. This measurement tells the system how much the prompt is influencing the model at this step.  
3. **Control:** The FBG algorithm uses the new log\_posterior value and its tuning parameters (pi, fbg\_temp, fbg\_offset) to calculate the *next* step's guidance scale, GSi+1​.  
4. **Repeat:** The loop continues.

This feedback mechanism allows the sampler to "correct its course." If the guidance is too weak, the posterior difference will be small, and the FBG will increase the guidance scale. If the guidance is too strong and causing instability, the posterior difference will be large, and the FBG will lower the scale.

#### **2.3. Conditional Blending**

This is an expert-level feature that allows for dynamic changes to the core mathematical operations of the sampler. The blending functions (lerp, slerp, add, etc.) determine how two tensors are combined. By setting up a trigger (e.g., conditional\_blend\_sigma\_threshold), you can instruct the sampler to use one function (like the stable lerp) for the high-noise compositional phase and automatically switch to another (like the sharper add or slerp) for the low-noise detail phase.

#### **2.4. Data I/O Deep Dive**

* **Input (to this node):**  
  * scheduler (SCHEDULER): A functional object from a node like Advanced Noise Decay Scheduler. Unlike the Lite+ version, the FBG version *can* use this object's get\_decay() method to modulate its ancestral noise.  
* **Output (from this node):**  
  * SAMPLER (SAMPLER): A functional KSAMPLER object that wraps the PingPongSamplerCore logic.  
* **Inputs/Outputs (of the downstream execution sampler, e.g., SamplerCustomAdvanced):**  
  * **Inputs:** model, positive, negative, latent\_image, sigmas, sampler (which receives the output of this node).  
  * **Output:** LATENT (the final denoised latent tensor).

#### **2.5. Strategic Role in the ComfyUI Graph**

* **Placement Context:** This node is a **sampler logic provider**. It must be placed upstream of an execution sampler. It does not touch the main data path of the workflow.  
* **The Configuration vs. Execution Pattern:**  
  * **PingPongSamplerNodeFBG (Configuration):** This node's role is to act as a dashboard for the incredibly complex PingPongSamplerCore object. It gathers dozens of parameters and packages them into a SAMPLER object.  
  * **SamplerCustomAdvanced (Execution):** This node receives the configured SAMPLER object and is responsible for feeding it the necessary data (model, latent, etc.) and running its sampling loop. This separation is essential for managing the complexity of the FBG system.

### **3\. Node Architecture & Workflow**

#### **3.1. Node Interface & Anatomy**

The PingPongSamplerNodeFBG node itself consists of a large number of widgets for its extensive parameters, a single scheduler input, and a single SAMPLER output.

#### **3.2. Input Port Specification**

* **scheduler (SCHEDULER)**:  
  * **Description:** Receives a scheduler object. The PingPongSamplerCore object will call this scheduler's get\_decay() method to get a modulation curve for the ancestral noise.  
  * **Required/Optional:** Required.

#### **3.3. Output Port Specification**

* **SAMPLER (SAMPLER)**:  
  * **Description:** Outputs a configured KSAMPLER object containing the full FBG and PingPong logic, ready to be executed by a downstream node.  
  * **Resulting Data:** A functional Python object.

#### **3.4. Workflow Schematics**

* Minimal Functional Graph:  
  The workflow requires at least three nodes for sampling: 1\) A sigma scheduler like Hybrid Sigma Scheduler. 2\) The PingPongSamplerNodeFBG node itself, configured as desired. 3\) An execution node like SamplerCustomAdvanced. The sigmas from the sigma scheduler connect to the SamplerCustomAdvanced. The SAMPLER from the PingPong node connects to the sampler input on the SamplerCustomAdvanced. All other inputs (model, conditioning, latent) also connect to the SamplerCustomAdvanced.

### **4\. Parameter Specification**

This node has an exceptionally large number of parameters. They are grouped here by function.

---

#### **4.1. Parameter Group: Ancestral Noise & Timing Controls**

These parameters control the "Ping-Pong" aspect of the sampler.

* **step\_random\_mode**: A COMBO (off, block, reset, step) that controls how the seed for the per-step ancestral noise is generated, allowing for fixed, block-wise, or continuously changing noise patterns.  
* **step\_size**: An INT that defines the interval for block and reset modes.  
* **seed**: The base INT seed for the ancestral noise's random number generator.  
* **first\_ancestral\_step / last\_ancestral\_step**: INT values defining the 0-indexed window of steps during which ancestral noise is actively injected. Setting last\_ancestral\_step below the total step count creates final "cleanup" steps.  
* **ancestral\_noise\_type**: A COMBO (gaussian, uniform, brownian) that selects the statistical distribution of the injected noise, affecting the resulting texture. gaussian is standard, uniform is harsher, and brownian is temporally correlated (smoother).  
* **start\_sigma\_index / end\_sigma\_index**: INT values for slicing the main sigma schedule, allowing the sampler to operate on only a sub-section of the denoising process.  
* **enable\_clamp\_output**: A BOOLEAN that, if True, clamps the final latent tensor's values to the \[-1.0, 1.0\] range.  
* **scheduler**: A required SCHEDULER object input. The get\_decay() method of this object is called to get a curve that can modulate the ancestral noise.  
* **blend\_mode**: A COMBO of blend functions (lerp, slerp, add, etc.) used to combine the denoised sample and the new noise during an ancestral step.  
* **step\_blend\_mode**: A COMBO of blend functions used to combine the denoised sample and the previous latent during a non-ancestral step (DDIM-like).

---

#### **4.2. Parameter Group: Feedback Guidance (FBG) \- Core Engine**

These are the core mathematical controls for the FBG algorithm.

* **fbg\_sampler\_mode**: A COMBO (EULER, PINGPONG) that selects the internal algorithm used for the FBG's own prediction calculations.  
* **cfg\_scale**: The base FLOAT CFG value. While the FBG will make this dynamic, this value can be used in a hybrid mode. Often overridden by the execution sampler's cfg input.  
* **max\_guidance\_scale**: A FLOAT that acts as the absolute "ceiling" or upper limit for the dynamic guidance scale. The FBG cannot push the guidance beyond this value.  
* **pi (π)**: A FLOAT between 0.0 and 1.0. This is the core **proportional gain** of the FBG system. It controls how strongly the guidance scale reacts to changes in the log\_posterior. Lower values make the system more sensitive and aggressive; higher values make it more dampened and stable.  
* **max\_posterior\_scale**: A FLOAT that clamps the maximum value of the internal log\_posterior state variable, preventing it from growing uncontrollably.  
* **fbg\_guidance\_multiplier**: A FLOAT that acts as a final multiplier on the FBG component of the guidance scale, allowing for an overall boost or reduction in its influence.  
* **fbg\_eta / fbg\_s\_noise**: Advanced FLOAT parameters corresponding to the eta (for DDIM-like steps) and noise level for the internal FBG sampler.

---

#### **4.3. Parameter Group: FBG \- Autopilot vs. Manual Tuning**

This is the most critical group for controlling the FBG behavior.

* **t\_0 / t\_1**: These two FLOAT parameters (0.0 to 1.0) control the FBG's **Autopilot mode**. They represent points in the generation timeline (0.0=start, 1.0=end). If **either t\_0 or t\_1 is non-zero**, Autopilot is engaged. The system will automatically calculate the fbg\_temp and fbg\_offset values needed to achieve a specific curve shape defined by these targets.  
* **fbg\_temp**: The primary **Manual mode** FLOAT control for FBG **sensitivity (Integral gain)**. It is **only active if t\_0 and t\_1 are both 0**. Higher values make the guidance scale change more slowly and smoothly; lower values make it more reactive and aggressive.  
* **fbg\_offset**: The primary **Manual mode** FLOAT control for FBG **drift (Derivative gain)**. Also **only active if t\_0 and t\_1 are both 0**. It adds a constant drift to the log\_posterior at each step, influencing the overall upward or downward trend of the guidance scale.  
* **log\_posterior\_initial\_value / initial\_guidance\_scale**: FLOAT values that set the starting state for the log\_posterior and the guidance scale at the very first step.

---

#### **4.4. Parameter Group: FBG \- Sigma Range & Scheduling**

These parameters control *when* the FBG and standard CFG components are active.

* **sigma\_range\_preset**: A COMBO that provides quick presets (High, Mid, Low, All) to set the four sigma values below, restricting FBG/CFG to certain phases of the generation.  
* **cfg\_start\_sigma / cfg\_end\_sigma**: FLOAT values defining the sigma range within which the standard cfg\_scale is active.  
* **fbg\_start\_sigma / fbg\_end\_sigma**: FLOAT values defining the sigma range within which the FBG system is active.  
* **ancestral\_start\_sigma / ancestral\_end\_sigma**: Deprecated/legacy FLOAT controls for timing, superseded by the first/last\_ancestral\_step integer-based controls.

---

#### **4.5. Parameter Group: Advanced & Experimental Controls**

* **conditional\_blend\_mode**: BOOLEAN to enable conditional blending.  
* **conditional\_blend\_sigma\_threshold**: A FLOAT sigma value. If the current sigma drops below this, the blend function switches to conditional\_blend\_function\_name.  
* **conditional\_blend\_on\_change / conditional\_blend\_change\_threshold**: BOOLEAN and FLOAT to enable a trigger that switches the blend function if the relative change in the latent between steps exceeds the threshold.  
* **clamp\_noise\_norm / max\_noise\_norm**: BOOLEAN and FLOAT to enable and set a limit on the L2 norm (magnitude) of the injected ancestral noise vector, which can help stabilize very chaotic generations.  
* **log\_posterior\_ema\_factor**: A FLOAT to apply an Exponential Moving Average to the log\_posterior state, smoothing its value over time for potentially more stable guidance.

---

#### **4.6. Parameter Group: Diagnostic & Override Controls**

* **debug\_mode**: An INT (0, 1, 2\) that controls the verbosity of console logging. 0 is off. 1 shows a summary per step. 2 shows verbose internal calculations.  
* **yaml\_settings\_str**: A multiline STRING input. If it contains a valid YAML configuration, it will **override all other parameters** on the node. This is the master control for saving and loading presets.  
* **checkpoint\_steps\_str**: A comma-separated STRING of step numbers at which to save a checkpoint of the latent state (a debug feature).

### **5\. A Masterclass in Tuning Feedback Guidance**

This guide details the process for manually tuning the FBG system for a new model or desired effect.

#### **5.1. The Golden Rule: Disabling Autopilot for Manual Control**

The sampler has two modes for its sensitivity: **Automatic (Autopilot)** and **Manual**.

* **Autopilot:** Engaged if **either t\_0 or t\_1 is NOT 0**. It will **IGNORE** fbg\_temp and fbg\_offset.  
* Manual: Engaged if BOTH t\_0 AND t\_1 are set to 0\. You now have full, direct control via fbg\_temp and fbg\_offset.  
  The first step in any serious tuning is to engage Manual Mode.

#### **5.2. The Troubleshooting Playbook & "Safe Mode" Reset**

If the guidance scale behaves erratically (e.g., explodes to its maximum value instantly), the system is too sensitive. The solution is to force it into a maximally stable state and tune from there.

1. Engage Manual Mode (t\_0=0, t\_1=0).  
2. Apply "Safe Mode" settings: fbg\_temp: 0.01 (very low sensitivity), fbg\_offset: 0.0 (no drift), initial\_value: 2.0 (high starting buffer).

#### **5.3. Tuning fbg\_temp: The Primary Sensitivity Dial**

With the sampler in a stable Manual Mode, **fbg\_temp is the primary sensitivity control.**

1. Incrementally increase fbg\_temp (e.g., from 0.01 to 0.05, then 0.1).  
2. Observe the debug\_mode log in the console.  
3. Fine-tune the value until the Guidance Scale (GS) shows a smooth, dynamic curve over the steps without becoming stuck or exploding.

#### **5.4. Reading the Signs: Anatomy of a Healthy FBG Log**

A well-tuned FBG system will produce a console log (with debug\_mode: 1\) that shows a controlled, dynamic process. The GS (Guidance Scale) value should start near its initial value and evolve smoothly throughout the steps, reacting to the content being generated. A final push to the max\_guidance\_scale in the last few steps is often desirable and indicates the system is working correctly.

### **6\. Implementation Deep Dive**

#### **6.1. Source Code Walkthrough**

1. **Configuration (PingPongSamplerNodeFBG.get\_sampler):** This method acts as the entry point. It gathers all \~40 parameters from the UI widgets. It then checks if yaml\_settings\_str is populated; if so, it uses yaml.safe\_load to parse it and updates a dictionary of options, with the YAML values taking precedence. This final, comprehensive dictionary of options is passed to the KSAMPLER constructor.  
2. **Instantiation (PingPongSamplerCore.\_\_init\_\_):** The KSAMPLER object, when called by the execution node, instantiates PingPongSamplerCore, passing all the options. The constructor is a massive initialization block that sets up all the state variables. Critically, it checks if Autopilot is engaged (t\_0 or t\_1 are non-zero). If so, it calls helper methods (\_get\_offset, \_get\_temp) to calculate the dynamic temp and offset values. Otherwise, it uses the manually provided fbg\_temp and fbg\_offset.  
3. Sampling Loop (PingPongSamplerCore.\_\_call\_\_): This is the main execution loop. For each step:  
   a. Guidance Calculation: It calls get\_dynamic\_guidance\_scale, which uses the current log\_posterior and the core FBG formula to compute the guidance scale for this specific step.  
   b. Model Prediction: It calls \_model\_denoise\_with\_guidance, passing the dynamically calculated guidance scale as an override. This method cleverly uses a post\_cfg\_function to intercept the model's cond and uncond denoised predictions.  
   c. Posterior Update: It calls \_update\_log\_posterior, feeding it the results from the model prediction. This method calculates the MSE difference and updates the log\_posterior state variable according to the FBG formula, preparing it for the next step.  
   d. Sampling Step: It performs the actual ancestral or DDIM-like step to compute the next latent, x\_current, using the selected blend modes and ancestral noise type.  
   e. The loop repeats.

### **7\. Troubleshooting & Diagnostics**

#### **7.2. Unexpected Visual Artifacts & Behavior**

* **Artifact:** The guidance scale immediately shoots to max\_guidance\_scale and stays there.  
  * **Likely Cause(s):** The FBG system is far too sensitive for the model. This is the most common issue when first tuning.  
  * **Correction Strategy:** Follow the "Masterclass in Tuning" guide. Engage Manual Mode (t\_0=0, t\_1=0) and start with "Safe Mode" settings (fbg\_temp=0.01). The system is highly sensitive, and very small fbg\_temp values are often correct.  
* **Artifact:** The guidance scale stays low and barely moves.  
  * **Likely Cause(s):** The FBG system has been successfully stabilized but is now *too insensitive*.  
  * **Correction Strategy:** You are in Manual Mode. Incrementally increase fbg\_temp (e.g., from 0.01 to 0.02, then 0.03). A tiny change can have a large effect. If the scale still doesn't move, you can try adding a small amount of positive drift with fbg\_offset.  
* **Artifact:** The workflow fails, or the sampler behaves like a standard sampler.  
  * **Likely Cause(s):** The workflow integration is incorrect. This node only *configures* the sampler.  
  * **Correction Strategy:** Ensure the SAMPLER output of this node is connected to the sampler input of an execution node like SamplerCustomAdvanced, and that all data inputs (model, latent, etc.) are connected to the execution node.