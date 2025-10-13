---

## **Master Technical Manual: Scene Genius Autocreator**

### **Node Name: SceneGeniusAutocreator**

Display Name: Scene Genius Autocreator  
Category: MD\_Nodes/SceneGenius  
Version: 0.3.8 (based on source code)  
Last Updated: 2025-09-16

---

### **Table of Contents**

1. Introduction  
   1.1. Executive Summary  
   1.2. Conceptual Category  
   1.3. Problem Domain & Intended Application  
   1.4. Key Features (Functional & Technical)  
2. Core Concepts & Theory  
   2.1. The Generative Pipeline: A Chain of Context  
   2.2. LLM Integration via API  
   2.3. The Override Mechanism  
   2.4. Data I/O Deep Dive  
   2.5. Strategic Role in the ComfyUI Graph  
3. Node Architecture & Workflow  
   3.1. Node Interface & Anatomy  
   3.2. Input Port Specification  
   3.3. Output Port Specification  
   3.4. Workflow Schematics (Minimal & Advanced)  
4. Parameter Specification  
   4.1. Parameter Group: LLM Backend Configuration  
   4.2. Parameter Group: Core Creative Inputs  
   4.3. Parameter Group: Generation Constraints  
   4.4. Parameter Group: Sampler Configuration  
   4.5. Parameter Group: Manual Override Inputs  
   4.6. Parameter: test\_mode  
5. Applied Use-Cases & Recipes  
   5.1. Recipe 1: Fully Automated Concept-to-Parameters Generation  
   5.2. Recipe 2: Batch Processing with a Fixed Technical "Vibe"  
   5.3. Recipe 3: Surgical Override for Fine-Tuning a Single Stage  
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

The **Scene Genius Autocreator** is a high-level, multi-stage generative orchestration node for ComfyUI. It functions as an automated "creative director" by leveraging a local Large Language Model (LLM) via an API (supporting Ollama and LM Studio) to procedurally generate a complete set of contextually coherent parameters for an advanced audio/video generation workflow. From a single high-level initial\_concept\_prompt, the node executes a sequential pipeline to synthesize genre tags, lyrics or scripts, duration, and crucially, the intricate YAML configurations required by complex downstream nodes like the APG Guider and PingPong Sampler. It is designed to be the starting point of a complex workflow, transforming a simple idea into a fully-defined, ready-to-render scene, while retaining full manual override capability at every stage for expert users.

#### **1.2. Conceptual Category**

**Workflow Automation / Generative Controller.** This node does not process or generate any media (image, audio, latent) itself. Instead, it generates **metadata and configuration data** (strings, floats, and integers). It acts as an intelligent front-end, using an LLM as a reasoning engine to automate the tedious task of parameter tuning and creative brainstorming, outputting the results as control signals for other nodes in the graph.

#### **1.3. Problem Domain & Intended Application**

* **Problem Domain:** Modern generative workflows, especially those using advanced samplers and guiders, involve a daunting number of highly abstract and interdependent parameters. Manually configuring these for every new creative idea is time-consuming, repetitive, and can be a barrier to rapid prototyping. Furthermore, ensuring that all parameters are thematically and technically coherent (e.g., that the sampler settings match the "vibe" of the genre) is a complex creative task. The Scene Genius node solves this by automating the entire creative and technical setup process, using an LLM to bridge the gap between a high-level idea and a low-level, machine-readable configuration.  
* **Intended Application (Use-Cases):**  
  * **Rapid Prototyping:** Quickly generating a wide variety of complete, ready-to-render scenes from a list of simple concept prompts without any manual parameter adjustment.  
  * **Automated Batch Processing:** In large batch runs, connecting an incrementing seed to this node allows it to generate a completely unique and thematically consistent set of parameters for *each individual image or audio clip* in the batch.  
  * **Creative Augmentation:** Using the LLM as a creative partner to suggest genres, lyrics, and technical settings that the user might not have considered, breaking creative blocks.  
  * **Foundation for Fine-Tuning:** Generating a robust, AI-created baseline configuration and then using the override inputs to manually tweak only the specific parameters that require adjustment, saving significant setup time.  
* **Non-Application (Anti-Use-Cases):**  
  * The node is not a substitute for a well-configured workflow. It requires the user to have the necessary downstream nodes (APG Guider, PingPong Sampler, etc.) already in their graph.  
  * The quality of the output is heavily dependent on the capability of the selected local LLM. It is not designed for users without a functioning local LLM server (Ollama or LM Studio).  
  * While it has fallbacks, it is inherently an experimental tool. It is not intended for workflows requiring 100% deterministic, manually-set parameters (in which case, the override fields should be used, or the node bypassed entirely).

#### **1.4. Key Features (Functional & Technical)**

* **Functional Features:**  
  * **Sequential Generation Pipeline:** Executes a six-stage pipeline (Genre → Lyrics → Duration → Noise → APG YAML → Sampler YAML), where the context from each stage informs the next.  
  * **Local LLM Integration:** Connects to a local Ollama or LM Studio server, ensuring user privacy, no API costs, and the ability to use custom-tuned models.  
  * **Intelligent Sampler Support:** Dynamically generates distinct, syntactically correct YAML configurations for both the FBG Integrated and Original (Lite+) versions of the PingPong Sampler.  
  * **Context-Aware APG Tuning:** Intelligently determines if the target is "Audio" or "Image" and generates an APG Guider YAML with the correct dims and appropriate parameter ranges.  
  * **Full Manual Override:** Every single generated output can be bypassed and replaced with a user-provided string, allowing for any degree of manual control, from full automation to surgical fine-tuning.  
* **Technical Features:**  
  * **Robust Prompt Engineering:** The internal prompts sent to the LLM are heavily structured and reinforced with examples and strict formatting rules (e.g., Output ONLY the number.) to minimize the chance of the LLM hallucinating incorrect data types or invalid YAML syntax.  
  * **Graceful Fallbacks:** Each generation stage is wrapped in a try-except block. If an API call fails or the LLM returns unusable data, the node logs a warning and falls back to a hard-coded, known-good default configuration for that stage, preventing workflow failure.  
  * **YAML Cleaning and Parsing:** The node automatically cleans common LLM artifacts (like markdown code fences) from the YAML outputs. It uses a custom yaml.Dumper to ensure specific data formats (like the dims list in APG YAML) are preserved correctly.  
  * **Dynamic Model Polling:** The ollama\_model\_name and lm\_studio\_model\_name dropdowns are dynamically populated by polling the respective local API endpoints when the UI is loaded, providing an up-to-date list of available models.

### **2\. Core Concepts & Theory**

#### **2.1. The Generative Pipeline: A Chain of Context**

The node's core principle is the creation of a "chain of thought" or a cascading context. This mimics a human creative process.

1. **Idea:** initial\_concept\_prompt  
2. **Moodboard:** The idea is expanded into genre\_tags.  
3. **Narrative:** The idea and moodboard inform the lyrics\_or\_script.  
4. **Technical Outline:** The full creative brief (idea, mood, narrative) is used to decide on technical parameters like total\_seconds and noise\_decay\_strength.  
5. **Detailed Blueprint:** Finally, the entire context package is given to the LLM to create the highly detailed, low-level APG and Sampler YAML blueprints.

This sequential process ensures that the final, highly technical YAML configurations are not random, but are contextually and thematically linked to the original high-level concept.

#### **2.2. LLM Integration via API**

The node functions as an API client. It does not run an LLM itself. It requires a separate, continuously running server application (Ollama or LM Studio) to be active on the local machine or network. The node formats a specific, carefully engineered prompt for each stage of its pipeline and sends it as an HTTP POST request to the server's API endpoint. It then waits for the response, cleans the returned text, and uses it as the output for that stage. This client-server architecture makes the node lightweight and modular.

#### **2.3. The Override Mechanism**

The override inputs (prompt\_\*\_generation) are the primary mechanism for user intervention. The logic for each stage in the execute method follows a simple but powerful pattern:  
if override\_string\_is\_provided: use\_override\_string else: call\_llm\_api()  
This allows the user to "short-circuit" the generative pipeline at any point. If an override is provided for an early stage (e.g., prompt\_genre\_generation), that manual data is then used as the context for all subsequent, automated stages. This provides a powerful blend of AI generation and manual control.

#### **2.4. Data I/O Deep Dive**

* **Inputs:** The node primarily takes STRING, INT, and FLOAT parameters from its UI widgets. It has no MODEL, LATENT, or other standard data inputs.  
* **Outputs:** The node produces only primitive data types, which are intended to be wired into the control inputs of other nodes.  
  * GENRE\_TAGS, LYRICS\_OR\_SCRIPT: STRING.  
  * TOTAL\_SECONDS, NOISE\_DECAY\_STRENGTH: FLOAT.  
  * APG\_YAML\_PARAMS, SAMPLER\_YAML\_PARAMS, NOISE\_DECAY\_YAML\_PARAMS: STRING (formatted as YAML).  
  * SEED: INT.

#### **2.5. Strategic Role in the ComfyUI Graph**

* **Placement Context:** This node should be placed at the **very beginning** of a generation workflow. It is a "source node" for configuration data. Its outputs are designed to be fanned out to multiple downstream nodes (Guiders, Samplers, Schedulers, etc.) to configure them before the main generation process begins.  
* **Synergistic Nodes:**  
  * **APGGuiderForked:** The APG\_YAML\_PARAMS output connects directly to the yaml\_parameters\_opt input of this node.  
  * **PingPongSampler\_Custom\_FBG or PingPongSampler\_Custom\_Lite:** The SAMPLER\_YAML\_PARAMS output connects directly to the yaml\_settings\_str input of the corresponding sampler configuration node.  
  * **NoiseDecayScheduler\_Custom:** The NOISE\_DECAY\_YAML\_PARAMS output connects to the yaml\_settings\_str input of this scheduler. (Note: The older NOISE\_DECAY\_STRENGTH output is for direct connection to the decay\_exponent slider for simpler workflows).  
  * **Text Display Nodes:** The GENRE\_TAGS and LYRICS\_OR\_SCRIPT outputs can be connected to any node that displays text, allowing the user to review the AI-generated creative content.

### **3\. Node Architecture & Workflow**

#### **3.1. Node Interface & Anatomy**

The node consists entirely of input widgets and output ports. It has no data inputs.

1. LLM Backend Controls (llm\_backend, URLs, model names)  
2. Core Creative Inputs (initial\_concept\_prompt, seed, randomize\_seed)  
3. Creative Constraint Widgets (tag\_count, excluded\_tags, force\_lyrics, duration limits)  
4. Sampler Version Selector  
5. Manual Override Text Boxes (prompt\_...\_generation)  
6. test\_mode Toggle  
7. Data Outputs (GENRE\_TAGS, LYRICS\_OR\_SCRIPT, TOTAL\_SECONDS, etc.)

#### **3.2. Input Port Specification**

This node has no input ports. All control is managed via its UI widgets.

#### **3.3. Output Port Specification**

* **GENRE\_TAGS (STRING):** A comma-separated string of genre and descriptive tags.  
* **LYRICS\_OR\_SCRIPT (STRING):** A multiline string containing generated lyrics with structural tags (e.g., \[Verse 1\]) or the single tag \[instrumental\].  
* **TOTAL\_SECONDS (FLOAT):** The generated duration of the piece in seconds.  
* **NOISE\_DECAY\_STRENGTH (FLOAT):** The generated strength/exponent for the noise decay curve.  
* **APG\_YAML\_PARAMS (STRING):** A fully-formed, multiline YAML string for configuring the APGGuiderForked node.  
* **SAMPLER\_YAML\_PARAMS (STRING):** A fully-formed, multiline YAML string for configuring the selected PingPongSampler node.  
* **NOISE\_DECAY\_YAML\_PARAMS (STRING):** A fully-formed, multiline YAML string for configuring the NoiseDecayScheduler\_Custom node.  
* **SEED (INT):** The final seed value to be used, either the input seed or a new one if randomize\_seed is active.

#### **3.4. Workflow Schematics**

* Fully Automated Workflow:  
  The Scene Genius Autocreator node sits at the top. Its APG\_YAML\_PARAMS, SAMPLER\_YAML\_PARAMS, NOISE\_DECAY\_YAML\_PARAMS, and SEED outputs are connected to the corresponding inputs on an APGGuiderForked node, a PingPongSampler... node, a NoiseDecayScheduler\_Custom node, and the execution sampler, respectively. The user only needs to edit the initial\_concept\_prompt and queue the workflow.

### **4\. Parameter Specification**

(This section provides a complete, unabridged specification for every parameter.)

#### **4.1. Parameter Group: LLM Backend Configuration**

* **llm\_backend**: A COMBO (ollama, lm\_studio) that selects the target API. This switch determines which URL and model name are used, and which API calling convention (\_call\_ollama\_api or \_call\_lm\_studio\_api) is invoked.  
* **ollama\_api\_base\_url / lm\_studio\_api\_base\_url**: STRING inputs for the base URL of the respective local LLM server. The code appends the specific API endpoint (/api/generate or /v1/chat/completions) internally.  
* **ollama\_model\_name / lm\_studio\_model\_name**: COMBO dropdowns that are dynamically populated by polling the respective APIs on UI load. The selected string is passed in the API request payload to specify which model should handle the generation.

#### **4.2. Parameter Group: Core Creative Inputs**

* **initial\_concept\_prompt**: A multiline STRING that serves as the primary seed for the entire generative pipeline. This text is formatted into the system prompts for every stage of the LLM generation process. Its content is the single most influential factor on the final output.  
* **seed**: An INT that is used as the base seed. It is either passed through directly or ignored if randomize\_seed is active.  
* **randomize\_seed**: A BOOLEAN toggle. If True, the execute method will generate a new random 64-bit integer at the start of the run, overwriting the value from the seed widget. If False, the seed widget's value is used.

#### **4.3. Parameter Group: Generation Constraints**

* **tag\_count**: An INT that is formatted into the prompt for Stage 1 (Genre Synthesis) to instruct the LLM on exactly how many tags to generate.  
* **excluded\_tags**: A STRING of comma-separated tags. If not empty, an instruction is added to the genre prompt telling the LLM to avoid using these specific tags.  
* **force\_lyrics\_generation**: A BOOLEAN that adds a specific instruction to the Stage 2 (Lyrics) prompt, forcing the LLM to generate lyrics and preventing it from choosing the \[instrumental\] path.  
* **force\_instrumental**: A BOOLEAN that adds a critical instruction to the Stage 2 prompt. The internal logic also checks this flag; if True, it will return \[instrumental\] regardless of the LLM's output, and an instruction is sent to the LLM to reflect this. Takes precedence over force\_lyrics\_generation.  
* **min\_total\_seconds / max\_total\_seconds**: FLOAT values that define the valid range for the duration. They are formatted into the Stage 3 (Duration) prompt to constrain the LLM. The node's code also performs a final clamp on the LLM's output to ensure the returned value strictly adheres to this range.

#### **4.4. Parameter Group: Sampler Configuration**

* **sampler\_version**: A COMBO dropdown (FBG Integrated PingPong Sampler, Original PingPong Sampler). This parameter acts as a crucial switch in the \_configure\_sampler\_yaml method. It determines which set of prompts to use and which hard-coded default YAML to fall back on, ensuring the generated YAML is syntactically and logically correct for the intended downstream sampler node.

#### **4.5. Parameter Group: Manual Override Inputs**

* **prompt\_genre\_generation / ...lyrics... / ...duration... / ...noise\_decay... / ...apg\_yaml... / ...sampler\_yaml...**: These six optional STRING inputs correspond to the main stages of the pipeline. In the execute method, before each call to a generation helper function (e.g., \_generate\_genres), the node checks if the corresponding override string is non-empty. If it is, the helper function is skipped entirely, and the override string is used as the output for that stage. This provides a mechanism for surgical manual intervention at any point in the automated process.

#### **4.6. Parameter: test\_mode**

* **UI Label:** test\_mode  
* **Internal Variable Name:** test\_mode  
* **Data Type & Constraints:** BOOLEAN.  
* **Algorithmic Impact:** This is a global override switch checked at the beginning of the execute method. If True, the entire multi-stage LLM pipeline is bypassed. The node immediately returns a set of hard-coded dummy data (e.g., "electro, synthwave", 180.0, default YAML strings, etc.). Its purpose is purely for workflow debugging, allowing a user to quickly test node connections without waiting for multiple slow LLM API calls.  
* **Default Value & Rationale:** False. The node's primary purpose is to call the LLM, so this debug mode is disabled by default.

### **5\. Applied Use-Cases & Recipes**

#### **5.1. Recipe 1: Fully Automated Concept-to-Parameters Generation**

* **Objective:** To leverage the full power of the node to turn a single creative sentence into a complete, ready-to-render set of audio parameters without manual tuning.  
* **Rationale:** This is the primary intended use case. By leaving all override fields empty, the user allows the node to execute its full sequential pipeline, creating a chain of context that ensures the final YAML is thematically aligned with the generated genres and lyrics.  
* **Parameter Configuration:**  
  * initial\_concept\_prompt: "A forgotten vinyl record playing a melancholic lofi hip-hop beat in a dusty, sunlit attic."  
  * sampler\_version: FBG Integrated PingPong Sampler  
  * All prompt\_\*\_generation overrides: Leave empty.

#### **5.2. Recipe 2: Batch Processing with a Fixed Technical "Vibe"**

* **Objective:** To generate a batch of 10 variations, keeping the advanced sampler settings identical for technical consistency, but allowing the LLM to generate fresh creative content (genres, lyrics) for each one.  
* **Rationale:** This recipe uses the override mechanism to "lock" the technical, late-stage parts of the pipeline. By providing manual YAML for the APG and Sampler nodes, the user ensures a consistent sonic texture and dynamic behavior across all outputs. However, by leaving the early-stage overrides empty, the LLM is still free to brainstorm new genres and lyrics for each run, which are then passed to the CLIP Text Encode node.  
* **Parameter Configuration:**  
  * initial\_concept\_prompt: "An energetic drum and bass track with sci-fi themes."  
  * randomize\_seed: True  
  * prompt\_apg\_yaml\_generation: Paste in your hand-tuned APG YAML.  
  * prompt\_sampler\_yaml\_generation: Paste in your hand-tuned Sampler YAML.

#### **5.3. Recipe 3: Surgical Override for Fine-Tuning a Single Stage**

* **Objective:** The user likes the genres and lyrics the AI generated on a previous run but finds the duration too long and wants to manually specify it.  
* **Rationale:** The override system allows for re-running the generation with a mix of locked and dynamic stages. By copying the previous run's GENRE\_TAGS and LYRICS\_OR\_SCRIPT outputs into their respective override fields, the user "freezes" those stages. By providing a manual number to prompt\_duration\_generation, they take control of that stage. The remaining stages (noise, YAML) are left empty, so the LLM will re-generate them based on the new, manually-defined context.  
* **Parameter Configuration:**  
  * prompt\_genre\_generation: "downtempo, vinyl crackle, soulful vocal chops, melancholic piano" (copied from previous run)  
  * prompt\_lyrics\_decision\_and\_generation: \[instrumental\] (copied from previous run)  
  * prompt\_duration\_generation: 120 (manual override)

### **6\. Implementation Deep Dive**

#### **6.1. Source Code Walkthrough**

The node's logic is a clear, sequential pipeline within the execute method, with each stage handled by a private helper method.

1. **Initialization:** The execute method begins by checking randomize\_seed. It resolves a conflict between force\_lyrics and force\_instrumental. It determines which LLM backend to use based on the llm\_backend switch. It then checks for test\_mode and returns dummy data if enabled.  
2. **Stage 1 (\_generate\_genres):** It checks the prompt\_genre\_generation override. If empty, it formats PROMPT\_GENRE\_GENERATION\_BASE with the user's concept and constraints, then calls \_call\_llm\_api. The result is returned as genre\_tags.  
3. **Stage 2 (\_generate\_lyrics):** It checks the prompt\_lyrics... override. If empty, it formats PROMPT\_LYRICS\_GENERATION\_BASE, including the genre\_tags from the previous stage as context. The result is returned as lyrics\_or\_script.  
4. **Stages 3-7 (Duration, Noise, APG, Sampler, Noise Decay YAML):** This pattern repeats for all subsequent stages. Each helper function (\_generate\_duration, \_configure\_apg\_yaml, etc.) first checks its corresponding override input. If the override is empty, the function formats its specific prompt using *all* the data generated in the preceding stages as context, calls the LLM, parses the result, and returns it. For example, the prompt to generate the APG YAML contains the initial\_concept\_prompt, the genre\_tags, the lyrics\_or\_script, the total\_seconds, and the noise\_decay\_strength.  
5. **LLM Call (\_call\_llm\_api):** All helper methods funnel into this function, which contains the actual requests.post call to the Ollama or LM Studio API endpoint. It includes retry logic and calls \_clean\_llm\_output on the response.  
6. **Output:** The execute method concludes by returning a tuple containing all the final generated or overridden values in the correct order for the output ports.

#### **6.2. Dependencies & External Calls**

* **requests:** This external library is essential for all communication with the LLM server. It is used to make the HTTP POST requests to the Ollama/LM Studio API endpoints.  
* **PyYAML:** Used to load the default YAML configurations from the node's internal strings (self.default\_apg\_yaml\_audio, etc.) and to dump the AI-modified data back into a cleanly formatted YAML string for the output. The CustomDumper ensures that short lists like dims are formatted inline for readability.  
* **Standard Libraries:** re (for cleaning LLM output), json (for handling API payloads), time (for retry delays), random (for randomize\_seed), logging.

#### **6.3. Performance & Resource Analysis**

* **Execution Target:** The node is **entirely CPU-bound**. Its execution time is dominated by the latency of the network API calls to the LLM server.  
* **VRAM Usage:** Zero. The node does not load any models or handle any large tensors.  
* **Bottlenecks:** The primary and only significant bottleneck is the **speed of the configured LLM**. A large model running on a slow CPU can cause this node to take several minutes to execute its full pipeline of 6-7 sequential API calls. The node's own Python logic is effectively instantaneous. The use of a smaller, faster LLM is highly recommended for interactive use.

#### **6.4. Data Lifecycle Analysis**

The node exclusively handles primitive data types (strings, floats, booleans, integers).

1. **Input:** User-configured parameters are read from the UI.  
2. **Processing:** A series of strings are created through LLM generation or from the override inputs. These strings are passed from one stage to the next as context. Some strings are parsed into floats (e.g., duration, noise decay). Dictionaries are created from default YAML strings, modified with the AI-generated values, and then dumped back into formatted YAML strings.  
3. **Output:** The final strings, floats, and the single integer seed are returned. The lifecycle of all data is contained within the single execute call; the node has no persistent state between runs.

### **7\. Troubleshooting & Diagnostics**

#### **7.1. Error Code Reference**

* **Error Message/Traceback Snippet:** (Console) requests.exceptions.ConnectionError...  
  * **Root Cause Analysis:** The node cannot establish a network connection with the LLM server at the specified URL. This means the Ollama/LM Studio server is not running, is not accessible from the ComfyUI environment (e.g., due to firewall or Docker networking issues), or the ...\_api\_base\_url is incorrect.  
  * **Primary Solution:** Verify that your Ollama or LM Studio server is running. Open the specified URL (e.g., http://localhost:11434) in a web browser to confirm it is accessible. Check for typos in the URL.  
* **Error Message/Traceback Snippet:** (Console) requests.exceptions.HTTPError: 404 Client Error: Not Found for url: ...  
  * **Root Cause Analysis:** The connection to the server was successful, but the server does not recognize the API endpoint, or the specified ollama\_model\_name does not exist on the server.  
  * **Primary Solution:** Ensure your Ollama/LM Studio version is up to date. Verify that the model name selected in the dropdown exactly matches a model you have installed on the server.

#### **7.2. Unexpected Behavior & Output**

* **Issue:** The generated YAML is malformed and causes the downstream sampler/guider to crash.  
  * **Likely Cause(s):** The LLM has failed to follow the strict formatting instructions in the prompt and has "hallucinated" an invalid parameter or incorrect YAML syntax.  
  * **Correction Strategy:** First, ensure you are using the latest version of the node, as the internal prompts are continuously improved to be more robust. Second, try using a different LLM that is known to be better at following instructions (e.g., an instruct-tuned model). As a final resort, use the manual override for the failing YAML stage and paste in the node's known-good default YAML from its \_\_init\_\_ method.  
* **Issue:** The node takes an excessively long time to run.  
  * **Likely Cause(s):** The selected LLM is too large for your hardware, or the LLM server is under heavy load.  
  * **Correction Strategy:** Switch to a smaller, quantized model in the ollama\_model\_name dropdown (e.g., a q4\_K\_M or q5\_K\_M version of a 7B or 8B model). These models are significantly faster than larger models or those with higher quantization levels.  
* **Issue:** The generated creative content (genres, lyrics) is low-quality or nonsensical.  
  * **Likely Cause(s):** The quality of the output is a direct reflection of the capability of the underlying LLM. A smaller or less capable model may struggle with creative tasks.  
  * **Correction Strategy:** Use a more capable (often larger) model for generation. Also, improve the initial\_concept\_prompt to be more descriptive and provide more context for the LLM to work with.