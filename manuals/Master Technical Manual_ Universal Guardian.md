---

## **Master Technical Manual: Universal Guardian**

### **Node Name: UniversalGuardian**

Display Name: Universal Guardian (Enhanced Evaluator & Generator)  
Category: MD\_Nodes/Utility  
Version: 4.0.2  
Last Updated: 2025-09-16

---

### **Table of Contents**

1. Introduction  
   1.1. Executive Summary  
   1.2. Conceptual Category  
   1.3. Problem Domain & Intended Application  
   1.4. Key Features (Functional & Technical)  
2. Core Concepts & Theory  
   2.1. The Guardian Loop: Evaluate, Enhance, Regenerate  
   2.2. The Composite Quality Score  
   2.3. Prompt Enhancement Strategies  
   2.4. Data I/O Deep Dive  
   2.5. Strategic Role in the ComfyUI Graph  
3. Node Architecture & Workflow  
   3.1. Node Interface & Anatomy  
   3.2. Input Port Specification  
   3.3. Output Port Specification  
   3.4. Workflow Schematics (Minimal & Advanced)  
4. Parameter Specification  
   4.1. Parameter Group: Core Guardian Logic  
   4.2. Parameter Group: Data Inputs  
   4.3. Parameter Group: Regeneration Engine  
   4.4. Parameter Group: Quality Metric Configuration  
   4.5. Parameter Group: Ollama Prompt Enhancement  
   4.6. Parameter Group: Bite/Sting Phrase Enhancement  
   4.7. Parameter Group: Diagnostic & Control Switches  
5. Applied Use-Cases & Recipes  
   5.1. Recipe 1: Simple Quality Gate (Evaluation Only)  
   5.2. Recipe 2: The Full Guardian Loop (Automated Latent Regeneration)  
   5.3. Recipe 3: Creative Prompt Evolution with Ollama  
6. Implementation Deep Dive  
   6.1. Source Code Walkthrough  
   6.2. Dependencies & External Calls  
   6.3. Performance & Resource Analysis  
   6.4. Data Lifecycle Analysis  
7. Troubleshooting & Diagnostics  
   7.1. Error Code Reference  
   7.2. Unexpected Behavior & Output

---

### **1\. Introduction**

#### **1.1. Executive Summary**

The **Universal Guardian** is an advanced, modality-agnostic workflow automation and quality control node for ComfyUI. It functions as an intelligent, closed-loop system that evaluates generated content against a user-defined quality standard, and if the standard is not met, can autonomously attempt to improve the result. The core of the node is its "Guardian Loop," a cycle of **Evaluate, Enhance, and Regenerate**. It calculates a composite quality score from a weighted combination of metrics (CLIP similarity, sharpness, contrast, variance). If the score is below a set threshold, it can dynamically enhance the text prompt using either a local Large Language Model (LLM) via Ollama or a simpler keyword-appending system. With regeneration enabled, it then uses this enhanced prompt to re-run an internal sampler, creating a new candidate for evaluation. This process repeats until the quality standard is met or a maximum number of retries is reached.

#### **1.2. Conceptual Category**

**Workflow Automation / Quality Control System.** This node is unique in that it combines the roles of a data evaluator, a generative controller, and an LLM-powered text processor. It can act as a simple quality "gate" (evaluating and passing/blocking data) or as a complete, self-correcting generative engine. Its primary function is to introduce conditional logic and iterative refinement into a ComfyUI graph.

#### **1.3. Problem Domain & Intended Application**

* **Problem Domain:** The output of generative AI models is inherently stochastic. For any given prompt, a significant percentage of outputs may be of low quality due to undesirable compositions, blurriness, low contrast, or poor prompt adherence. Manually curating these outputs, especially in large batch jobs, is a tedious and inefficient process. Furthermore, improving a prompt to achieve a better result is an iterative, manual task. The Universal Guardian is designed to automate both of these processes: the filtering of low-quality results and the iterative refinement of the prompt to generate better ones.  
* **Intended Application (Use-Cases):**  
  * **Automated Quality Assurance:** Building "fire-and-forget" workflows that guarantee a minimum level of output quality. The node can be configured to run until it produces a result with a composite score of, for example, 0.7 or higher.  
  * **Batch Processing Enhancement:** Ensuring that every item in a large batch meets a quality baseline, automatically re-rolling any that fail.  
  * **Creative Prompt Exploration:** Using the Ollama integration as a creative partner. A user can provide a simple prompt and allow the Guardian to iteratively "brainstorm" more detailed and evocative versions until a high-quality result is achieved.  
  * **Building Resilient Workflows:** Using the node as an intelligent filter that prevents low-quality latents or images from being passed to subsequent, computationally expensive stages like upscaling or video processing.  
* **Non-Application (Anti-Use-Cases):**  
  * The node is not a replacement for a well-tuned model or a well-written initial prompt. It is a tool for refinement and filtering, not for salvaging incoherent or fundamentally flawed concepts.  
  * The regeneration loop can be computationally expensive, as it involves running the sampler multiple times. It is not intended for workflows where speed is the absolute highest priority.  
  * The audio and video evaluation metrics are experimental and less robust than the image/latent metrics.

#### **1.4. Key Features (Functional & Technical)**

* **Functional Features:**  
  * **Closed-Loop Regeneration:** The core feature allowing the node to evaluate, enhance a prompt, and re-sample in an automated loop.  
  * **Multi-Modal Input:** Can accept and evaluate LATENT, IMAGE, AUDIO, or VIDEO data types, making it highly versatile.  
  * **Composite Quality Metric:** Calculates a final quality score based on a weighted average of up to four metrics: CLIP Score (prompt-image similarity), Sharpness (Laplacian variance), Contrast (standard deviation), and Variance (of the latent or image).  
  * **Dual Prompt Enhancement Engines:**  
    1. **Ollama (LLM):** For sophisticated, context-aware prompt rewriting using a local LLM.  
    2. **Bite/Sting Phrases:** A simple, deterministic fallback that appends pre-defined quality and style keywords to the prompt.  
  * **Configurable Retry Logic:** Offers three distinct strategies for the final output: Return on First Success (fastest), Return Best Attempt (highest quality), and Return Last Attempt (for observing evolution).  
* **Technical Features:**  
  * **Integrated Sampler:** When regeneration is enabled, the node internally calls comfy.sample.sample, effectively packaging a complete KSampler within itself.  
  * **On-Demand CLIP Model Loading:** The CLIP model and processor required for scoring are only loaded into memory if use\_clip\_scoring is enabled, conserving VRAM in workflows where it is not needed.  
  * **Safe Device Management:** Uses a context manager (safe\_device\_context) to temporarily move models (like the VAE or CLIP model) to the correct compute device for processing and then move them back, minimizing VRAM usage.  
  * **Robust Error Handling:** Optional dependencies (transformers, ollama, librosa, cv2) are wrapped in try-except blocks, allowing the node to function in a degraded state (e.g., without CLIP scoring) if a library is missing. API calls and regeneration steps are also wrapped to prevent workflow crashes.  
  * **Detailed JSON Reporting:** Outputs a comprehensive JSON string (quality\_report) that contains the full history of the Guardian loop, including the prompt, seed, and detailed metric scores for every attempt.

### **2\. Core Concepts & Theory**

#### **2.1. The Guardian Loop: Evaluate, Enhance, Regenerate**

The node's operation is best understood as a stateful loop that seeks to optimize a quality score.

1. **Evaluate:** The process begins by taking an input (either from an upstream node or its own first generation) and calculating its composite\_score.  
2. **Compare:** This score is compared to the quality\_metric\_threshold. If score \>= threshold, the loop terminates successfully.  
3. **Enhance:** If the score is too low, the prompt enhancement logic is triggered. The current prompt is modified by either the Ollama LLM or the Bite/Sting keyword system.  
4. **Regenerate:** If enable\_regeneration is active, the node increments its internal seed and calls the sampler with the newly enhanced prompt to generate a new candidate latent/image.  
5. Repeat: The process returns to Step 1, evaluating the new candidate.  
   This continues until a success condition is met or max\_retries is exhausted.

#### **2.2. The Composite Quality Score**

The node does not rely on a single, subjective measure of "good." It quantifies quality using a weighted sum of objective metrics, normalized to a \[0.0, 1.0\] range.

* **CLIP Score:** Measures semantic similarity. It uses a CLIP model to compare the generated image to the text prompt, yielding a score that reflects prompt adherence.  
* **Sharpness (Laplacian Variance):** A classic computer vision technique. The Laplacian operator highlights edges and fine details. A high variance in the Laplacian image indicates a large amount of sharp detail, while a low variance indicates blurriness.  
* **Contrast (Standard Deviation):** Measures the standard deviation of pixel/latent values. A high standard deviation implies a wide range of values (high contrast), while a low standard deviation implies a narrow range (low contrast, washed-out).  
* **Variance:** The statistical variance of all values in the tensor. This serves as a proxy for overall information content and complexity.

By adjusting the **weights** of these metrics, the user can define what "quality" means for their specific use case.

#### **2.3. Prompt Enhancement Strategies**

When a generation fails, the Guardian attempts to improve the input for the next attempt.

* **Ollama (LLM-based):** This is the more intelligent strategy. It leverages the reasoning and creative capabilities of a Large Language Model. The node sends the current prompt along with a system\_prompt (e.g., "You are a creative prompt enhancer...") to the LLM. The LLM then rewrites the prompt, often adding more descriptive adjectives, specifying lighting conditions, or suggesting artistic styles. This can guide the diffusion model toward more complex and aesthetically pleasing regions of the latent space.  
* **Bite/Sting (Keyword-based):** This is a simpler, deterministic fallback. It maintains two lists of keywords: bite\_phrases (often quality terms like "masterpiece, 8k, sharp focus") and sting\_phrases (often stylistic terms like "cinematic lighting, ethereal"). At each retry, it randomly selects a number of these keywords (controlled by prompt\_enhancement\_ratio) and appends them to the current prompt.

### **2.4. Data I/O Deep Dive**

* **Inputs:**  
  * latent\_input (LATENT): A dictionary containing a "samples" tensor of shape \[B, C, H, W\].  
  * image\_input (IMAGE): A tensor of shape \[B, H, W, C\].  
  * audio\_input (AUDIO): A dictionary containing a "waveform" tensor.  
  * video\_input (VIDEO): A tensor representing video frames.  
  * **Regeneration Inputs:** MODEL, CONDITIONING, etc., which are standard ComfyUI objects.  
* **Outputs:**  
  * output\_latent, output\_image, output\_audio, output\_video: The node will only output on the port corresponding to its input type. The output will be the successful candidate that passed the quality check.  
  * final\_prompt (STRING): The final, potentially enhanced, prompt that was used to generate the successful output.  
  * final\_quality\_metric (FLOAT): The composite score of the successful output.  
  * attempt\_count (INT): The number of retries it took to achieve a successful result (0 for the initial success).  
  * final\_seed (INT): The seed used for the successful generation.  
  * preview\_image (IMAGE): A preview image of the successful output, automatically generated even if the input was a latent.  
  * quality\_report (STRING): A JSON-formatted string containing a detailed log of all attempts, including the prompt, seed, and full metric breakdown for each.

### **2.5. Strategic Role in the ComfyUI Graph**

* **Placement Context:** The Universal Guardian is a **workflow controller** that typically sits *after* an initial generation stage or acts as a self-contained generative stage itself.  
  * **As a Quality Gate:** It is placed after a sampler and VAE Decode. It receives an image, evaluates it, and if it passes, forwards it. If it fails (and regeneration is off), it outputs an empty tensor, effectively stopping that branch of the workflow.  
  * **As a Regenerative Engine:** It replaces a KSampler. It takes the same inputs (model, conditioning, latent) and performs the sampling loop internally, only outputting a final latent that meets the quality criteria.

### **3\. Node Architecture & Workflow**

#### **3.1. Node Interface & Anatomy**

The node is exceptionally large, with four optional data inputs on the left, numerous parameter widgets, and ten distinct outputs on the right.

#### **3.2. Input Port Specification**

* **Data Inputs (latent\_input, image\_input, etc.)**:  
  * **Description:** These four optional ports are where the content to be evaluated is connected. The node's logic requires that **exactly one** of these be connected for a given run. The connected input determines the node's operating modality.  
  * **Required/Optional:** At least one is functionally required.  
* **Regeneration Inputs (model, positive\_conditioning, etc.)**:  
  * **Description:** These are the standard inputs required by any sampler. They are only used if enable\_regeneration is set to True. If regeneration is enabled, these inputs become functionally required for the node to operate correctly.  
  * **Required/Optional:** Optional, but required for regeneration.

#### **3.3. Output Port Specification**

* **Data Outputs (output\_latent, output\_image, etc.)**:  
  * **Description:** These four ports correspond to the input data types. The node will only provide an output on the port that matches the input modality. For example, if latent\_input is used, only output\_latent will contain data. The output data is the final, successful result of the Guardian loop.  
* **Metadata Outputs (final\_prompt, final\_quality\_metric, etc.)**:  
  * **Description:** These ports provide detailed information about the final state of the successful generation. They are invaluable for logging, debugging, or for passing the enhanced prompt or seed to other parts of the workflow. The quality\_report provides the most comprehensive summary.

### **4\. Parameter Specification**

(This section provides a complete, unabridged specification for every parameter.)

#### **4.1. Parameter Group: Core Guardian Logic**

* **initial\_prompt**: A multiline STRING that serves as the prompt for the first evaluation attempt. This is the creative seed for all subsequent enhancements.  
* **quality\_metric\_threshold**: A FLOAT (0.0 to 1.0) that defines the "pass" score. The composite\_score of an attempt must be greater than or equal to this value to be considered a success.  
* **max\_quality\_threshold**: A FLOAT (0.0 to 1.0) that defines an "excellent" score. If any attempt achieves a score at or above this value, the Guardian loop terminates immediately, even if max\_retries has not been reached. This acts as an early-exit for high-quality results.  
* **max\_retries**: An INT that sets the maximum number of times the node will **re-try** after the initial attempt fails. A value of 3 means a total of 1 (initial) \+ 3 (retries) \= 4 generations can occur.  
* **initial\_seed**: The INT seed for the first attempt. For each subsequent retry, the seed used for regeneration will be initial\_seed \+ attempt\_number. If set to 0, a random seed based on the current time is used for the first attempt.  
* **retry\_behavior**: A COMBO dropdown that dictates which attempt's output is returned upon completion of the loop.  
  * Return on First Success: The loop terminates and returns the data for the very first attempt that meets the quality\_metric\_threshold. Most efficient.  
  * Return Best Attempt: The loop always runs for the full max\_retries (unless the max\_quality\_threshold is met). It keeps track of the attempt with the highest score and returns that one at the end. Most thorough.  
  * Return Last Attempt: The loop runs for the full max\_retries and simply returns the data from the final attempt, regardless of its score. Useful for observing the full evolution of the prompt.

#### **4.2. Parameter Group: Data Inputs**

* **latent\_input / image\_input / audio\_input / video\_input**: Four optional input ports for LATENT, IMAGE, AUDIO, and VIDEO data types respectively. The node's execute method begins by checking that exactly one of these inputs is connected and not None, raising a ValueError if this condition is not met. The connected input determines the modality and the data that will be evaluated.

#### **4.3. Parameter Group: Regeneration Engine**

These parameters are only active if enable\_regeneration is True.

* **model / positive\_conditioning / negative\_conditioning**: The standard MODEL and CONDITIONING objects required for any diffusion sampling process. They are passed directly to the internal comfy.sample.sample function.  
* **sampler\_name / scheduler\_name**: COMBO dropdowns populated with ComfyUI's available samplers and schedulers. These strings are passed to comfy.sample.sample to define the sampling algorithm.  
* **steps / cfg / denoise**: The standard INT and FLOAT parameters controlling the sampling process, which are also passed directly to comfy.sample.sample.

#### **4.4. Parameter Group: Quality Metric Configuration**

* **use\_clip\_scoring**: A BOOLEAN toggle that enables or disables the CLIP score calculation. If False, the clip\_weight is effectively zero.  
* **clip\_model / clip\_processor**: Optional CLIP and CLIP\_VISION inputs. If provided, the node will use these pre-loaded models instead of loading its own from HuggingFace, which is more memory-efficient.  
* **clip\_model\_name**: A COMBO dropdown to select a specific CLIP model from HuggingFace if the dedicated inputs are not used.  
* **clip\_weight / sharpness\_weight / contrast\_weight / variance\_weight**: FLOAT values that determine the relative importance of each metric in the final composite\_score. Internally, the code sums these weights and then normalizes them so that the normalized weights sum to 1.0. This means the user can think in terms of relative importance (e.g., "twice as much weight on CLIP as sharpness") without needing to manually make them sum to 1\.

#### **4.5. Parameter Group: Ollama Prompt Enhancement**

These parameters are only active if use\_ollama\_enhancement is True.

* **use\_ollama\_enhancement**: A BOOLEAN to enable the LLM-based prompt enhancement strategy.  
* **ollama\_model\_name**: A COMBO dropdown of available Ollama models, polled from the local server. Selects which LLM will perform the rewriting.  
* **ollama\_system\_prompt**: A multiline STRING that provides the instructions or "persona" for the LLM. The default prompt instructs it to act as a creative enhancer focusing on details relevant to image generation.  
* **ollama\_temperature / ollama\_top\_p**: FLOAT parameters that control the creativity of the LLM. Higher temperature and lower top\_p lead to more random and diverse responses. These are passed in the options dictionary of the Ollama API call.  
* **ollama\_server\_url**: A STRING for the base URL of the Ollama server, in case it is not running on the default localhost:11434.

#### **4.6. Parameter Group: Bite/Sting Phrase Enhancement**

These parameters are only active if use\_ollama\_enhancement is False.

* **skip\_enhancement**: A BOOLEAN master switch. If True, it disables both Ollama and Bite/Sting enhancement, causing the same prompt to be used for all retry attempts.  
* **bite\_phrases**: A multiline, comma-separated STRING of keywords, typically focused on quality and technical detail (e.g., "masterpiece, best quality").  
* **sting\_phrases**: A multiline, comma-separated STRING of keywords, typically focused on style and artistic mood (e.g., "cinematic lighting, ethereal").  
* **prompt\_enhancement\_ratio**: A FLOAT that controls the aggressiveness of this method. A tiered logic system uses this ratio to determine how many keywords from the bite\_phrases and sting\_phrases lists are randomly selected and appended to the prompt for a given retry.

#### **4.7. Parameter Group: Diagnostic & Control Switches**

* **enable\_regeneration**: A critical BOOLEAN switch. If False, the node acts only as a passive evaluator. If True, it activates the internal sampler and becomes an active, self-correcting generator.  
* **debug\_mode**: A BOOLEAN toggle that, if True, sets the node's logger level to INFO, causing it to print detailed information about each attempt's scores and the final decisions to the console.

### **5\. Applied Use-Cases & Recipes**

#### **5.1. Recipe 1: Simple Quality Gate (Evaluation Only)**

* **Objective:** To check an image generated by a separate sampler. If it's blurry or low-contrast, stop the workflow for that image.  
* **Rationale:** This recipe uses the Guardian in its simplest form. By disabling regeneration and setting max\_retries to 0, it performs exactly one evaluation. If the input image fails the quality\_metric\_threshold, the node outputs None (empty tensors), which will cause downstream nodes like Save Image to fail or be skipped for that item, effectively filtering it out.  
* **Parameter Configuration:**  
  * enable\_regeneration: False  
  * max\_retries: 0  
  * quality\_metric\_threshold: 0.55  
  * **Connections:** Connect a VAE Decode output to the image\_input.

#### **5.2. Recipe 2: The Full Guardian Loop (Automated Latent Regeneration)**

* **Objective:** To create a self-correcting workflow that generates a latent and automatically re-rolls with an improved prompt if the result is not good enough, returning the best result after all attempts.  
* **Rationale:** This is the canonical use case. The node is given all the components needed to sample. It performs the initial sample (as it has no latent input), evaluates it, and enters the retry loop if needed. The Bite/Sting method provides a reliable, deterministic way to enhance the prompt. Return Best Attempt ensures that even if the last attempt is worse, the highest-quality result is preserved.  
* **Parameter Configuration:**  
  * enable\_regeneration: True  
  * initial\_prompt: "A majestic space whale"  
  * quality\_metric\_threshold: 0.65  
  * max\_retries: 5  
  * retry\_behavior: Return Best Attempt  
  * use\_ollama\_enhancement: False  
  * **Connections:** Connect model, positive\_conditioning, negative\_conditioning, and an Empty Latent to their respective inputs. Connect the output\_latent to a VAE Decode.

#### **5.3. Recipe 3: Creative Prompt Evolution with Ollama**

* **Objective:** To start with a very simple concept and use the LLM as a creative co-writer to expand upon it over several iterations, returning the final, most evolved version.  
* **Rationale:** This recipe leans into the creative potential of the LLM. By setting a high quality threshold, we encourage the node to retry. At each retry, the LLM will generate a more descriptive prompt. By setting the retry\_behavior to Return Last Attempt, we ensure we get the output from the most complex and detailed prompt, allowing us to see the full result of the AI's creative evolution.  
* **Parameter Configuration:**  
  * enable\_regeneration: True  
  * initial\_prompt: "A cat in a library"  
  * quality\_metric\_threshold: 0.75 (high to force retries)  
  * max\_retries: 4  
  * retry\_behavior: Return Last Attempt  
  * use\_ollama\_enhancement: True  
  * **Connections:** Same as Recipe 2\.

### **6\. Implementation Deep Dive**

#### **6.1. Source Code Walkthrough**

The logic is contained within the execute method of the UniversalGuardian class.

1. **Initialization and Validation:** The method starts by validating that exactly one data input is provided. It initializes the seed and normalizes the quality metric weights.  
2. **Initial Evaluation:** It calls \_get\_comprehensive\_quality\_metrics on the initial input data and prompt. This helper function is a large switch that determines the data modality. For images/latents, it decodes the latent if necessary, then calls helpers like \_get\_clip\_score and \_calculate\_sharpness. It returns a QualityMetrics object.  
3. **Threshold Check:** The initial score is checked against quality\_metric\_threshold and max\_quality\_threshold. If either condition is met and the behavior is Return on First Success, the node formats its output via \_prepare\_return\_values and exits early.  
4. Retry Loop: If the initial check fails, it enters a for loop from 1 to max\_retries \+ 1\.  
   a. Seed & Prompt: The seed is incremented. The prompt is enhanced by calling either \_enhance\_prompt\_ollama or \_enhance\_prompt\_bite\_sting.  
   b. Regeneration: If enable\_regeneration is true and the input is a latent, it calls the \_regenerate\_latent helper, which is a wrapper around the core comfy.sample.sample function.  
   c. Re-evaluation: It calls \_get\_comprehensive\_quality\_metrics again on the newly generated (or original) data with the newly enhanced prompt.  
   d. State Update: It compares the current\_metrics.composite\_score to the best\_metrics.composite\_score and updates the best\_input, best\_prompt, best\_seed, etc., if the current attempt is better.  
   e. Exit Checks: It re-checks the quality\_metric\_threshold and max\_quality\_threshold to see if an early exit is warranted based on the retry behavior.  
5. **Final Return:** After the loop finishes (either by completing all retries or an early exit), it selects which data to return based on the retry\_behavior (best\_input for Best Attempt, current\_input for Last Attempt) and calls \_prepare\_return\_values to format the final tuple of 10 outputs.

#### **6.2. Dependencies & External Calls**

* **transformers:** An optional but critical dependency for quality scoring. The CLIPProcessor and CLIPModel classes are used to perform the prompt-image similarity calculation. If not found, CLIP scoring is disabled.  
* **ollama:** An optional dependency for LLM-based prompt enhancement. The ollama.Client is used to communicate with the local Ollama server.  
* **requests:** Used to validate the Ollama server connection before attempting to use the ollama library.  
* **librosa / opencv-python:** Optional dependencies for experimental audio and video quality metrics. Their absence does not break the node but limits its functionality for those modalities.  
* **comfy.sample.sample:** The node directly calls this core ComfyUI function to perform latent regeneration, effectively embedding a KSampler's functionality within itself.

#### **6.3. Performance & Resource Analysis**

* **Execution Target:** A mix of CPU and GPU.  
  * **GPU-bound:** The \_regenerate\_latent call (sampling) and the \_get\_clip\_score call (CLIP model inference) are VRAM and compute-intensive GPU operations.  
  * **CPU-bound:** All other quality metrics (sharpness, contrast), prompt enhancement (LLM calls are CPU/RAM intensive), and the main loop logic are executed on the CPU.  
* **VRAM Usage:** VRAM usage can be significant. The node may need to hold the main diffusion model, a VAE, and a CLIP model in VRAM simultaneously. The safe\_device\_context manager helps mitigate this by loading and unloading models as needed, but a machine with sufficient VRAM is recommended for using all features.  
* **Bottlenecks:** The primary bottleneck is the regeneration loop. If the node retries 5 times with a 20-step sampler, it is performing 100 sampling steps in total. The LLM calls via Ollama can also add significant latency depending on the model size and hardware.

### **7\. Troubleshooting & Diagnostics**

#### **7.2. Unexpected Behavior & Output**

* **Issue:** The CLIP Score is always 0.0, and the quality metric seems inaccurate.  
  * **Likely Cause(s):** The transformers library is not installed, or the node failed to download the selected clip\_model\_name from HuggingFace. CLIP scoring is a "soft" dependency and the node will function without it, but the composite\_score will be calculated without the crucial prompt-adherence metric.  
  * **Correction Strategy:** Check the ComfyUI console on startup for warnings about transformers not being found. If present, install it in your Python environment (pip install transformers). Also ensure you have an active internet connection the first time you run the node so it can download the CLIP model. For offline use, pre-load the CLIP model with a CLIPLoader node and connect it to the clip\_model and clip\_processor inputs.  
* **Issue:** Regeneration is enabled, but the output latent is identical to the input latent on every retry.  
  * **Likely Cause(s):** One of the required inputs for regeneration (model, positive\_conditioning, negative\_conditioning) is missing. The node logs an error and skips the \_regenerate\_latent call, causing it to re-evaluate the *original* input with the *new* prompt on every loop iteration.  
  * **Correction Strategy:** Ensure that all four regeneration-related inputs (model, positive\_conditioning, negative\_conditioning, and latent\_input) are correctly connected when enable\_regeneration is True.  
* **Issue:** The workflow seems to execute twice or re-runs unexpectedly.  
  * **Likely Cause(s):** This is the intended behavior due to the node's internal state management and how it interacts with the ComfyUI execution graph. The IS\_CHANGED method is intentionally designed to always trigger a re-execution to ensure the loop can function correctly.  
  * **Correction Strategy:** This is not an error to be corrected. Understand that this node forces re-execution of its branch of the workflow. To prevent re-running upstream nodes, use primitive inputs (like Seed (any)) that can be converted to widgets to "freeze" their values between runs.