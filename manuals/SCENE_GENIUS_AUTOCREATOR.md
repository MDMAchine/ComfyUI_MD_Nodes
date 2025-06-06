# SceneGenius Autocreator Node Manual V0.1

---

## 1. What is the SceneGenius Autocreator Node?

The SceneGenius Autocreator is a powerful ComfyUI node designed to streamline your creative content generation process by acting as an "AI SysOp." While it was originally developed for use with audio diffusion models like Ace-Step, its capabilities hold significant promise for enhancing video and image generation workflows as well, though further testing and modifications to the code is recommended for these applications.

At its core, this node integrates with local Large Language Models (LLMs), specifically via Ollama, to intelligently produce a wide array of creative outputs based on a single conceptual prompt. Think of it as a multi-stage AI creative assistant that automates complex tasks and parameter tuning for your generative art projects.

### How it Works

The SceneGenius Autocreator operates through a sophisticated multi-stage process, orchestrating various AI subroutines to generate cohesive and contextually relevant outputs:

1.  **Concept Interpretation**: It takes your primary input, the `initial_concept_prompt`, and uses a local LLM to understand your overarching creative vision.
2.  **Creative Generation**: Based on this concept, the LLM dynamically generates several key elements:
    * **Authentic & Varied Genres**: It goes beyond simple keywords to produce realistic music genres and diverse combinations (e.g., "Gabba Cyber-Grindcore with a hint of Industrial Noise").
    * **Lyrics or Script**: It can author detailed lyrics or scripts. Alternatively, if your creative concept suggests an instrumental piece, it will intelligently decide to output an `[instrumental]` tag instead, demonstrating an "art of suggestion."
    * **Precise Durations**: It calculates an optimal total duration in seconds for your generated content.
    * **Diffusion Parameters (YAML)**: This is a critical feature where the LLM dynamically configures parameters for two crucial diffusion components:
        * **Adaptive Projected Gradient (APG) Guider**: It forges finely tuned YAML parameters for APG, aiming for precise guidance for your diffusion model, which helps achieve clean visuals and audio without unwanted artifacts.
        * **Ping-Pong Sampler**: It configures YAML parameters specifically optimized for models like Ace-Step, aiming for tight, high-quality visual or audio output.
    * **Noise Decay Adjustment**: It dynamically adjusts a `noise_decay_strength` parameter, allowing you to fine-tune the "cleanliness" or "grit" of the final output, from pixel-perfect clarity to a desired retro noise effect.
3.  **Output**: The node then provides all these generated elements as structured outputs, ready to be seamlessly fed into downstream ComfyUI nodes for your diffusion workflow.

### What it Does (Key Features)

* **Local LLM Integration**: Connects seamlessly with local Ollama instances for powerful, offline AI generation, ensuring privacy and control over your models.
* **Multi-Stage Generation**: Orchestrates a complex creative pipeline from an initial concept to final diffusion parameter configuration, automating tasks that would typically require extensive manual input and expertise.
* **Context-Aware Creativity**: The LLM maintains context across its internal processes, remembering previous generations to ensure a coherent creative narrative and consistent, thematically aligned outputs.
* **Dynamic Content Creation**: Generates diverse genres, intelligent lyrics/scripts (or instrumental tags), and precise durations, effectively acting as a creative co-pilot.
* **Automated Diffusion Parameter Generation**: Synthesizes Adaptive Projected Gradient (APG) and Ping-Pong Sampler YAML configurations tailored to the creative concept, saving significant time in manual parameter tuning.
* **Robust Error Handling & YAML Override**: Includes built-in fallbacks if the LLM output is unparseable or malformed, and provides options for manual YAML overrides for advanced users seeking ultimate control or fine-tuning.

### How to Use It

The SceneGenius Autocreator node is typically the initial step in your ComfyUI workflow when you want to leverage AI for creative content and parameter generation.

1.  **Add the Node**: In your ComfyUI workflow, right-click on the canvas, navigate to `Add Node`, find and select `SceneGeniusAutocreator`.
2.  **Connect to Ollama**:
    * **Prerequisite**: Ensure you have Ollama installed and running locally on your system, with your desired LLM model downloaded (e.g., `llama3:8b-instruct-q8_0`).
    * **`ollama_api_url`**: Input the full URL where your Ollama server is accessible. For a default local setup, this is typically `http://localhost:11434`.
    * **`ollama_model_name`**: Specify the exact name of the Ollama model you wish to use for generation (e.g., `llama3:8b-instruct-q8_0`). The node will attempt to fetch a list of available models from your Ollama instance and use a default if it cannot connect or find the specified model.
3.  **Provide the Initial Concept**:
    * **`initial_concept_prompt`**: Enter your creative vision or idea here. This is the core textual input that guides the LLM in generating all subsequent outputs (genres, lyrics, duration, and diffusion parameters). Be as descriptive or concise as you like; the LLM is designed to interpret a wide range of inputs.
4.  **Optional Parameters (for fine-tuning)**:
    * **`force_instrumental`**: `BOOLEAN` (Default: `False`). If set to `True`, the LLM will always generate an `[instrumental]` tag for the lyrics/script output, regardless of the `initial_concept_prompt`. Useful when you specifically want instrumental pieces.
    * **`min_duration_seconds`**: `INT` (Default: `15`). Sets the minimum allowed duration for the generated content.
    * **`max_duration_seconds`**: `INT` (Default: `45`). Sets the maximum allowed duration for the generated content.
    * **`llm_temperature`**: `FLOAT` (Default: `0.7`). Controls the randomness of the LLM's output.
        * Higher values (e.g., `1.0`+) lead to more diverse and creative (but potentially less coherent) outputs.
        * Lower values (e.g., `0.1`-`0.5`) make the output more focused and deterministic.
    * **`yaml_parameters_opt`**: `STRING` (Optional). This is a powerful advanced feature. You can paste a YAML string here that contains custom diffusion parameters for APG and Ping-Pong Sampler. If provided, these values will override the LLM's dynamically generated diffusion parameters, giving you ultimate control. This is useful for:
        * Applying specific, pre-tested diffusion settings.
        * Troubleshooting or isolating issues.
        * Using custom parameters that the LLM might not generate.
5.  **Connect Outputs**: The node provides several outputs that you can connect to downstream nodes:
    * **`genres`**: The generated music genres or descriptive tags.
    * **`lyrics_or_script`**: The generated lyrics, script, or `[instrumental]` tag.
    * **`duration_seconds`**: The calculated optimal duration in seconds.
    * **`apg_guider_params`**: YAML string containing parameters for the APG Guider node.
    * **`pingpong_sampler_params`**: YAML string containing parameters for the PingPong Sampler node.
    * **`noise_decay_strength`**: The dynamically adjusted noise decay strength.

---

## 2. Detailed Parameter Information

This section provides a more in-depth look at each parameter of the SceneGenius Autocreator node.

### Core Input

* **`initial_concept_prompt`**
    * **Type**: `STRING`
    * **Description**: The foundational text input that describes your creative idea, theme, or desired outcome. This prompt guides the entire multi-stage generation process of the LLM.
    * **Use & Why**: This is where you convey your artistic vision. It should be descriptive enough for the LLM to understand and expand upon. The quality and specificity of this prompt directly influence the relevance and creativity of all generated outputs (genres, lyrics, duration, and diffusion parameters).

### Ollama Connection Parameters

* **`ollama_api_url`**
    * **Type**: `STRING`
    * **Default**: `http://localhost:11434`
    * **Description**: The network address (URL) where your local Ollama server is running and accessible.
    * **Use & Why**: Essential for the node to communicate with the Ollama instance to perform LLM inference. Ensure this URL points to your running Ollama server.
* **`ollama_model_name`**
    * **Type**: `STRING`
    * **Default**: `llama3:8b-instruct-q8_0`
    * **Description**: The exact name of the Large Language Model hosted on your Ollama server that you want the node to use for generating content.
    * **Use & Why**: Specifies which LLM model the node should load and use for creative generation. It's crucial that this name matches a model you have downloaded and made available via Ollama.

### Output Control Parameters

* **`force_instrumental`**
    * **Type**: `BOOLEAN`
    * **Default**: `False`
    * **Description**: If set to `True`, the LLM will bypass its internal logic for generating lyrics/scripts and will always output the `[instrumental]` tag for the `lyrics_or_script` output.
    * **Use & Why**: Provides a direct override when you are certain you want an instrumental piece, preventing the LLM from attempting to generate lyrical content.
* **`min_duration_seconds`**
    * **Type**: `INT`
    * **Default**: `15`
    * **Range**: `1` to `3600` (1 hour)
    * **Description**: The minimum desired duration, in seconds, for the generated content. The LLM will try to adhere to this lower bound when calculating the `duration_seconds` output.
    * **Use & Why**: Helps to constrain the LLM's duration predictions, ensuring the generated content meets minimum length requirements for your project.
* **`max_duration_seconds`**
    * **Type**: `INT`
    * **Default**: `45`
    * **Range**: `1` to `3600` (1 hour)
    * **Description**: The maximum desired duration, in seconds, for the generated content. The LLM will try to adhere to this upper bound when calculating the `duration_seconds` output.
    * **Use & Why**: Helps to constrain the LLM's duration predictions, ensuring the generated content does not exceed a desired maximum length, which is important for managing generation times and resource usage.
* **`llm_temperature`**
    * **Type**: `FLOAT`
    * **Default**: `0.7`
    * **Range**: `0.0` to `2.0`
    * **Step Size**: `0.01`
    * **Description**: Controls the "creativity" or randomness of the LLM's responses. A higher temperature makes the LLM's output more varied and unpredictable, while a lower temperature makes it more deterministic and focused.
    * **Use & Why**: Adjust this to fine-tune the exploratory nature of the LLM. For more stable, predictable genres and lyrics, use a lower temperature. For more experimental, surprising, or "out-of-the-box" ideas, increase the temperature.

### Advanced Override Parameter

* **`yaml_parameters_opt`**
    * **Type**: `STRING` (Multi-line text input)
    * **Default**: `(empty string)`
    * **Description**: An optional input where you can provide a YAML formatted string containing specific diffusion parameters for the APG Guider and Ping-Pong Sampler. If this input is populated with valid YAML, it will override the parameters dynamically generated by the LLM.
    * **Use & Why**: This provides an escape hatch and advanced control. If you have a known set of optimal diffusion parameters or wish to test specific configurations, you can hardcode them here. The YAML structure should follow the expected format for APG Guider and Ping-Pong Sampler configurations.

---

## 3. Detailed Technical Information

This section delves into the internal workings and architecture of the SceneGenius Autocreator node.

### Core Architecture

The node's `generate` method orchestrates a sequence of internal steps, each designed to leverage the connected Ollama LLM for specific creative and technical tasks.

1.  **Ollama Client Initialization**:
    * It first attempts to establish a connection with the Ollama server using the provided `ollama_api_url`.
    * It then verifies if the specified `ollama_model_name` is available on the server. If not, or if connection fails, it attempts to use a fallback default model (`llama3:8b-instruct-q8_0`). This ensures robustness.

2.  **Schema and Prompt Loading**:
    * The node loads predefined `system_prompt`, `user_prompt`, and `json_schema` from internal static definitions. These are crucial for guiding the LLM to produce structured, relevant JSON outputs. The JSON schema enforces the expected format for genres, lyrics, duration, and the nested diffusion parameters.

3.  **LLM Inference Loop**:
    * The core of the creative generation is an iterative loop that attempts to get a valid JSON response from the LLM.
    * **`try-except` block**: This robust error handling mechanism is critical. If the LLM generates malformed JSON, or if there's any other error during the parsing, it catches the exception.
    * **Retries**: In case of a parsing error, it instructs the LLM to "try again" with a refined prompt, emphasizing the need for valid JSON. This helps to improve the reliability of the LLM's structured output.
    * **Temperature Adjustment**: If the LLM repeatedly fails to produce valid JSON, the node dynamically lowers the `llm_temperature` to encourage more deterministic and predictable (and thus, more likely to be valid) outputs.

4.  **Content Extraction and Conversion**:
    * Once a valid JSON response is received, the node extracts the `genres`, `lyrics_or_script` (handling the `force_instrumental` override), `duration_seconds`, `apg_guider_params`, `pingpong_sampler_params`, and `noise_decay_strength`.
    * **YAML Conversion**: The `apg_guider_params` and `pingpong_sampler_params` are converted into YAML strings.

### Internal Logic and Best Practices

* **Robustness via Fallbacks and Retries**: The primary design philosophy behind this node is resilience. By incorporating default Ollama URLs/models, `try-except` blocks around LLM calls, and a retry mechanism with temperature annealing, the node is designed to handle common LLM eccentricities (like generating invalid JSON) gracefully. This makes it more suitable for production ComfyUI workflows where stability is paramount.
* **Structured Output Enforcement**: The use of a `json_schema` in the LLM prompt is a powerful technique to coerce the LLM into producing a specific, parseable data structure. This is crucial for automating the configuration of downstream nodes.
* **Dynamic Parameter Generation**: The node intelligently generates diffusion parameters (`apg_guider_params`, `pingpong_sampler_params`, `noise_decay_strength`) based on the creative prompt. This automates a complex, often manual, tuning process, making advanced diffusion models more accessible.
* **YAML Override Logic**: The `yaml_parameters_opt` input is a strategic design choice. It allows advanced users to bypass the LLM's parameter generation for diffusion components, providing a critical level of control and enabling integration with external parameter libraries or pre-tuned configurations.
* **Context Management**: While not explicitly shown in the `generate` method's signature, the LLM's ability to maintain context over a conversation is leveraged (implicitly via the system/user prompt structure) to ensure coherence between the creative concept and the generated technical parameters.

---

## 4. Error Handling and Graceful Degradation

The SceneGenius Autocreator node is built with robustness in mind, featuring several layers of error handling and graceful degradation to ensure a stable user experience.

### LLM Communication Errors

* **Connection Issues**: If the node cannot connect to the Ollama API at the specified `ollama_api_url`, it will log an error message to the ComfyUI console, indicating a network problem or an incorrect URL. It will then attempt to proceed with fallback parameters if available.
* **Model Not Found**: If the `ollama_model_name` specified by the user is not found on the connected Ollama server, the node will log a warning and attempt to use a default fallback model (`llama3:8b-instruct-q8_0`). This prevents a complete failure due to a misspelled or unavailable model.

### LLM Output Parsing Errors

* **Invalid JSON**: The most common issue with LLM integration is when the model fails to produce syntactically correct JSON, despite being prompted with a schema. The node has a robust `try-except` block to catch `json.JSONDecodeError`.
    * **Retries**: Upon detecting invalid JSON, the node attempts to retry the LLM call, providing an explicit instruction to the LLM to correct its output and adhere to the JSON format.
    * **Temperature Adjustment**: If repeated retries (up to a predefined limit) still result in invalid JSON, the node will incrementally lower the `llm_temperature`. This makes the LLM's output more deterministic and increases the likelihood of generating valid JSON.
* **Missing Fields**: If the LLM generates valid JSON but omits crucial fields defined in the schema (e.g., `genres`, `duration_seconds`), the node is designed to handle these gracefully. It will log a warning and attempt to use sensible defaults or empty strings for the missing data.
* **Schema Validation Errors**: While not explicitly coded with a full JSON schema validator for runtime, the internal structure of the prompting encourages adherence to the expected types and ranges. Deviations are primarily handled by the parsing and data conversion steps.

### Fallback Parameters

* **Default Values**: If the LLM fails to produce valid diffusion parameters (e.g., `apg_guider_params`, `pingpong_sampler_params`) or if the `yaml_parameters_opt` input is empty or malformed, the node has internal default YAML configurations for these parameters. This means your diffusion workflow can still proceed, albeit with less tailored settings.
* **Error Logging**: All errors and warnings are logged to the ComfyUI console. This provides users with crucial feedback for debugging and understanding why certain outputs might not be as expected.

### Overall Resilience

The extensive implementation of `try-except` blocks around all LLM API calls and YAML parsing operations, coupled with the pre-defined default YAML fallbacks, makes the node highly resilient. It can gracefully handle network issues, Ollama failures, or unpredictable LLM output formats. This "fail-safe" design is paramount for stable integration into dynamic ComfyUI workflows, preventing a single LLM hiccup from crashing the entire generation process.

---

## 5. Modularity and Extensibility

### Separation of Concerns

By relying on YAML for the configuration of diffusion parameters, the node maintains a clear separation between the AI's creative output (genres, lyrics) and the diffusion model's technical parameters. This modularity simplifies future updates, allows for easy swapping out of diffusion models, or enables the implementation of new parameter generation stages in subsequent versions, fostering long-term adaptability.

### "Code-as-Creative-Director" Philosophy

This node embodies a "code-as-creative-director" philosophy, leveraging LLMs to automate complex parameter generation and creative decision-making. This empowers users to focus on high-level concepts and artistic direction rather than intricate manual tuning, acting as a sophisticated bridge between abstract creative intent and the concrete, technical requirements of advanced diffusion models.
