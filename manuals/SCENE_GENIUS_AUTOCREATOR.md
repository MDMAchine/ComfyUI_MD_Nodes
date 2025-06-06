SceneGenius Autocreator Node Manual V0.1
The SceneGenius Autocreator is a powerful ComfyUI node designed to streamline your creative content generation process by acting as an "AI SysOp." While it was originally developed for use with audio diffusion models like Ace-Step, its capabilities hold significant promise for enhancing video and image generation workflows as well, though further testing and modifcations to the code is recommended for these applications.

At its core, this node integrates with local Large Language Models (LLMs), specifically via Ollama, to intelligently produce a wide array of creative outputs based on a single conceptual prompt. Think of it as a multi-stage AI creative assistant that automates complex tasks and parameter tuning for your generative art projects.

1. What is the SceneGenius Autocreator Node?
How it Works:
The SceneGenius Autocreator operates through a sophisticated multi-stage process, orchestrating various AI subroutines to generate cohesive and contextually relevant outputs:

Concept Interpretation: It takes your primary input, the initial_concept_prompt, and uses a local LLM to understand your overarching creative vision.

Creative Generation: Based on this concept, the LLM dynamically generates several key elements:

Authentic & Varied Genres: It goes beyond simple keywords to produce realistic music genres and diverse combinations (e.g., "Gabba Cyber-Grindcore with a hint of Industrial Noise").

Lyrics or Script: It can author detailed lyrics or scripts. Alternatively, if your creative concept suggests an instrumental piece, it will intelligently decide to output an [instrumental] tag instead, demonstrating an "art of suggestion."

Precise Durations: It calculates an optimal total duration in seconds for your generated content.

Diffusion Parameters (YAML): This is a critical feature where the LLM dynamically configures parameters for two crucial diffusion components:

Adaptive Projected Gradient (APG) Guider: It forges finely tuned YAML parameters for APG, aiming for precise guidance for your diffusion model, which helps achieve clean visuals and audio without unwanted artifacts.

Ping-Pong Sampler: It configures YAML parameters specifically optimized for models like Ace-Step, aiming for tight, high-quality visual or audio output.

Noise Decay Adjustment: It dynamically adjusts a noise_decay_strength parameter, allowing you to fine-tune the "cleanliness" or "grit" of the final output, from pixel-perfect clarity to a desired retro noise effect.

Output: The node then provides all these generated elements as structured outputs, ready to be seamlessly fed into downstream ComfyUI nodes for your diffusion workflow.

What it Does (Key Features):
Local LLM Integration: Connects seamlessly with local Ollama instances for powerful, offline AI generation, ensuring privacy and and control over your models.

Multi-Stage Generation: Orchestrates a complex creative pipeline from an initial concept to final diffusion parameter configuration, automating tasks that would typically require extensive manual input and expertise.

Context-Aware Creativity: The LLM maintains context across its internal processes, remembering previous generations to ensure a coherent creative narrative and consistent, thematically aligned outputs.

Dynamic Content Creation: Generates diverse genres, intelligent lyrics/scripts (or instrumental tags), and precise durations, effectively acting as a creative co-pilot.

Automated Diffusion Parameter Generation: Synthesizes Adaptive Projected Gradient (APG) and Ping-Pong Sampler YAML configurations tailored to the creative concept, saving significant time in manual parameter tuning.

Robust Error Handling & YAML Override: Includes built-in fallbacks if the LLM output is unparseable or malformed, and provides options for manual YAML overrides for advanced users seeking ultimate control or fine-tuning.

How to Use It:
The SceneGenius Autocreator node is typically the initial step in your ComfyUI workflow when you want to leverage AI for creative content and parameter generation.

Add the Node: In your ComfyUI workflow, right-click on the canvas, navigate to Add Node, find and select SceneGeniusAutocreator.

Connect to Ollama:

Prerequisite: Ensure you have Ollama installed and running locally on your system, with your desired LLM model downloaded (e.g., llama3:8b-instruct-q8_0).

ollama_api_url: Input the full URL where your Ollama server is accessible. For a default local setup, this is typically http://localhost:11434.

ollama_model_name: Specify the exact name of the Ollama model you wish to use for generation (e.g., llama3:8b-instruct-q8_0). The node will attempt to fetch a list of available models from your Ollama instance and use a default if it cannot connect or find the specified model.

Provide the Initial Concept:

initial_concept_prompt: Enter your creative vision or idea here. This is the core textual input that guides the LLM in generating all subsequent outputs (genres, lyrics, duration, and diffusion parameters). Be as descriptive and evocative as possible to achieve more targeted results (e.g., "a driving dark techno track with glitchy, abstract visuals" instead of just "techno"). This field supports multiline input for longer descriptions.

Control Lyrics/Instrumental:

force_lyrics: This is a boolean input, typically represented as a checkbox.

If checked (true): The LLM will be explicitly instructed to generate narrative lyrics or a script, regardless of the nuanced interpretation of your initial_concept_prompt.

If unchecked (false): The LLM will intelligently decide based on your initial_concept_prompt whether to generate lyrics/script or simply output an [instrumental] tag, providing a dynamic co-writing experience.

Adjust Noise Decay:

noise_decay_strength: Tweak this floating-point value to influence the "cleanliness" or "grit" of the final generated output. Higher values (e.g., closer to 2.0) typically lead to clearer, more refined results, while lower values (e.g., closer to 0.0) might introduce more noise or stylistic texture.

Set Seed (for Reproducibility):

seed: Input an integer value for the random number generator. Providing a specific seed ensures that if all other inputs remain identical, the node will consistently produce the same genre, lyrics, duration, and YAML parameters. This is crucial for iterating on specific results or sharing workflows.

Optional Overrides (For Advanced Users):

apg_yaml_override: If you possess a pre-defined or custom YAML string for the Adaptive Projected Gradient (APG) Guider, you can paste it here. If this field contains valid YAML, the node will use your provided YAML for APG parameters, completely bypassing the LLM's automatic generation for this component. Leave empty ("") to allow the LLM to generate.

sampler_yaml_override: Similarly, for the Ping-Pong Sampler, you can paste a custom YAML string here. If valid, this will override the LLM's sampler parameter generation. Leave empty ("") to allow the LLM to generate. These override fields are invaluable for fine-tuning, debugging, or integrating with external parameter optimization tools.

Connect Outputs: Connect the node's outputs (e.g., genre_tags, lyrics_or_script, total_seconds, apg_yaml_params, sampler_yaml_params, noise_decay_strength, seed) to the corresponding inputs of your downstream diffusion model or other processing nodes in your ComfyUI workflow.

2. Detailed Information About the Parameters
The SceneGenius Autocreator node takes the following direct inputs, allowing you to guide its AI-powered creative process:

initial_concept_prompt (string):

Use: This is the foundational text input that describes your overarching creative vision or idea. It's the primary instruction to the LLM.

Purpose: The LLM interprets this prompt to generate all subsequent outputs: genres, lyrics/scripts, duration, and diffusion parameters.

How to Use: Be descriptive and evocative. For example, "a melancholic synthwave track with visuals of a rain-soaked cyberpunk city" will yield more targeted results than just "music." This field supports multiline input.

force_lyrics (boolean):

Use: Controls whether the LLM is explicitly instructed to produce narrative lyrics/script.

Purpose: If true (checked), the LLM will always attempt to generate lyrics or a script regardless of the initial_concept_prompt. If false (unchecked), the LLM will use its own judgment based on the initial_concept_prompt to decide if lyrics are appropriate, or if an [instrumental] tag should be generated instead.

How to Use: Check this box (true) if your creative vision absolutely requires a narrative or lyrical component. Uncheck it (false) if you want the AI to intelligently determine if the concept lends itself more to an instrumental piece.

noise_decay_strength (float):

Use: A numerical value that influences the dynamic adjustment of noise decay in the generated output.

Purpose: This parameter allows you to fine-tune the perceived "cleanliness" or "grit" of the final generated content. Higher values generally lead to clearer outputs, while lower values might retain more noise or texture.

How to Use: Experiment with values, typically between 0.0 and 2.0 (with a suggested step of 0.01). For a very polished look, try higher values closer to 2.0. For a more raw or lo-fi aesthetic, consider values closer to 0.0.

ollama_api_url (string):

Use: Specifies the network address of your local Ollama instance.

Purpose: This is essential for the node to establish communication with the Large Language Model running on your system. Without a correct URL, the node cannot interact with the LLM to generate creative content.

How to Use: Enter the URL where your Ollama server is accessible (e.g., http://localhost:11434 for a default local setup). Ensure Ollama is running and accessible from where ComfyUI is installed.

ollama_model_name (string):

Use: Defines which specific Ollama model the node should utilize for its generation tasks.

Purpose: Allows you to select the LLM that will perform the creative generation, giving you control over the AI's capabilities and characteristics (e.g., model size, instruction following, creative style).

How to Use: Enter the exact name of the model you have downloaded and made available in Ollama (e.g., llama3:8b-instruct-q8_0). The node attempts to fetch a list of available models and defaults to a specific model if none are found or the connection fails.

seed (integer):

Use: A numerical value used to initialize the random number generator.

Purpose: Provides reproducibility. If you use the same seed along with identical initial_concept_prompt and other inputs, the node will consistently produce the same genres, lyrics/scripts, duration, and YAML parameters.

How to Use: Set to 0 for a default starting point (often leading to varied results on each run). Change to any integer value (from 0 to 0xffffffffffffffff) for different results or to preserve a specific generation.

apg_yaml_override (string, optional):

Use: Allows you to input a custom YAML string to define the Adaptive Projected Gradient (APG) Guider parameters.

Purpose: This provides advanced users direct control. If this field contains a valid YAML string, the node will use this YAML for the APG Guider, completely bypassing the LLM's automatic generation for this component.

How to Use: Leave empty ("") to let the LLM generate the APG parameters based on your concept. Provide a multi-line YAML string conforming to the APG Guider's expected schema if you need precise, manual control over the APG settings. This is useful for experimentation or leveraging pre-defined optimal settings.

sampler_yaml_override (string, optional):

Use: Allows you to input a custom YAML string to define the Ping-Pong Sampler parameters.

Purpose: Similar to the APG override, this gives direct control over the sampler settings, overriding the LLM's generated parameters.

How to Use: Leave empty ("") to let the LLM generate the Sampler parameters based on your concept. Provide a multi-line YAML string conforming to the Ping-Pong Sampler's expected schema for advanced, manual control over its behavior.

Node Outputs:
The node will output the following parameters, which can be connected to other nodes in your ComfyUI workflow:

genre_tags (string): The LLM-generated genre tags for your creative concept.

lyrics_or_script (string): The generated lyrics, script, or the [instrumental] tag.

total_seconds (integer): The calculated duration in seconds.

noise_decay_strength (float): The noise decay strength, either passed through from input or dynamically adjusted.

apg_yaml_params (string): The YAML-formatted string containing the Adaptive Projected Gradient parameters.

sampler_yaml_params (string): The YAML-formatted string containing the Ping-Pong Sampler parameters.

seed (integer): The seed used for the current generation, useful for tracing reproducibility.

Internal LLM Parameters (Not Directly Exposed to User):
The following parameters are integral to the node's internal logic and the LLM's behavior, but they are not exposed as direct user-configurable inputs on the ComfyUI node's interface in this version (V0.1). They are currently hardcoded or dynamically determined by the node's internal logic. Future versions may expose these for even finer control:

llm_temperature: Controls the randomness and creativity of the LLM's text generation.

llm_max_output_tokens_per_stage: Limits the length of the LLM's output for each stage of generation.

llm_num_gpu_layers: Specifies how many GPU layers Ollama should use for the model.

llm_context_window_size: Determines how much prior conversation history the LLM retains for context.

3. In-Depth Nerd Technical Information Regarding the Node
The SceneGenius Autocreator (V0.1) is engineered as a robust, multi-stage AI orchestration system within the ComfyUI framework. Its core design principle is to seamlessly interlace the generative capabilities of a local Large Language Model with precise control over diffusion model parameters.

Architecture and Flow:
Initialization (__init__): Upon instantiation, the node pre-loads default_apg_yaml and default_sampler_yaml strings. These serve as resilient fallbacks in scenarios where the LLM's output is unparseable or fails internal validation. This proactive measure ensures continuous operation even under unexpected LLM responses, significantly mitigating potential workflow interruptions.

ComfyUI Integration (INPUT_TYPES, RETURN_TYPES, CATEGORY, FUNCTION):

INPUT_TYPES: This static method rigorously defines the node's user-facing configurable parameters, including initial_concept_prompt, force_lyrics, noise_decay_strength, ollama_api_url, ollama_model_name, seed, and the optional apg_yaml_override and sampler_yaml_override strings. This method is crucial as it dictates the node's appearance and the manipulable inputs within the ComfyUI graph editor.

RETURN_TYPES: This method explicitly declares the node's output structure, which comprises genre_tags (STRING), lyrics_or_script (STRING), total_seconds (INT), noise_decay_strength (FLOAT), apg_yaml_params (STRING), sampler_yaml_params (STRING), and seed (INT). These structured outputs are designed to be readily consumed by subsequent nodes in a typical diffusion workflow.

FUNCTION: This attribute maps to the run method, which encapsulates the entire core execution logic of the node.

Core Execution (run method):

Ollama API Endpoint Handling: Dynamically constructs the /api/generate endpoint using the provided ollama_api_url. This serves as the primary conduit for sending prompts and receiving generative responses from the local LLM.

Prompt Construction (Multi-Stage, Context-Aware): The run method intelligently and dynamically builds prompts for the LLM based on the current stage of generation, ensuring a cohesive creative narrative.

Stage 1 (Genre, Lyrics/Script, Duration): The initial prompt is meticulously crafted, incorporating the user's initial_concept_prompt and the force_lyrics boolean. The force_lyrics parameter directly influences the LLM's internal instruction to either explicitly generate lyrics or allow for an [instrumental] tag, providing precise creative control.

Stage 2 (APG & Sampler YAML): Crucially, the outputs generated from Stage 1 (e.g., generated genres, lyrics/script, and duration) are fed back into the prompts for subsequent LLM calls. This establishes a context-aware, multi-stage reasoning chain. For instance, the apg_yaml_params generated in one sub-step might inform the prompt for sampler_yaml_params generation, ensuring a holistic parameter synthesis tailored to the initial creative output. This iterative self-correction and refinement is a key architectural strength enabling sophisticated results.

LLM Invocation: Utilizes the requests.post library to facilitate robust communication with the local Ollama API. Error handling is comprehensively implemented, catching common issues such as network connection problems (requests.exceptions.ConnectionError) and general HTTP errors, enhancing the node's reliability.

Output Parsing and Validation:

The LLM's raw text output is meticulously parsed using regular expressions (re.search) to extract specific, tagged sections like GENRE_TAGS:, LYRICS_OR_SCRIPT:, TOTAL_SECONDS:, APG_YAML_PARAMS:, and SAMPLER_YAML_PARAMS:. This relies on the LLM adhering to a strict output format specified in its internal instructions for reliable extraction.

The extracted YAML segments (APG_YAML_PARAMS, SAMPLER_YAML_PARAMS) are then parsed using yaml.safe_load.

Robust Fallbacks: If YAML parsing fails due to malformed LLM output (yaml.YAMLError) or other exceptions during the process, the node gracefully reverts to its pre-defined self.default_apg_yaml and self.default_sampler_yaml. This crucial mechanism prevents crashes and ensures the workflow can continue with sensible default parameters.

Parameter Propagation: The noise_decay_strength input value is directly passed through as an output, allowing it to be used by downstream nodes. Similarly, the seed integer input is directly propagated as an output, ensuring that if all inputs are consistent, the entire generative process, including the LLM's output, remains reproducible.

Key Technical Aspects:
Custom YAML Dumper (CustomDumper): A specialized yaml.Dumper is implemented (CustomDumper overriding yaml.Dumper.represent_list) to enforce a "flow style" (inline) representation specifically for lists containing two or fewer numeric elements (e.g., dims: [-2, -1]). This technical detail addresses a common formatting nuisance in generated YAML for diffusion models, ensuring cleaner, more compact, and easily human-readable output, particularly for parameters like dims which are crucial in APG configurations.

Deep Prompt Engineering: The profound strength of this node lies in its meticulously crafted, multi-stage, iterative prompting of the LLM. Instead of a single, monolithic query, the LLM's previous outputs (genres, lyrics, duration) are intelligently incorporated as contextual information into subsequent prompts. The internal prompts for generating APG and Sampler YAMLs are highly detailed, providing explicit guidance to the LLM on:

Parameter Semantics: Explaining the precise purpose and impact of each parameter (e.g., apg_scale, momentum, predict_image for APG; step_random_mode, ancestral_steps for Sampler).

Optimal Ranges and Interdependencies: Suggesting ideal value ranges and highlighting crucial relationships between parameters (e.g., the mode: pre_alt2 synergy with momentum and predict_image).

Critical Constraints: Enforcing specific values for stability and desired behavior (e.g., eta: 0.0 for APG in this specific fork, or start_sigma_index: 0 and end_sigma_index: -1 for the Sampler to ensure full denoising).
This deep, granular instruction aims to consistently produce technically sound and creatively effective diffusion configurations.

Error Tolerance and Graceful Degradation: The extensive implementation of try-except blocks around all LLM API calls and YAML parsing operations, coupled with the pre-defined default YAML fallbacks, makes the node highly resilient. It can gracefully handle network issues, Ollama failures, or unpredictable LLM output formats. This "fail-safe" design is paramount for stable integration into dynamic ComfyUI workflows, preventing a single LLM hiccup from crashing the entire generation process.

Modularity and Extensibility: By relying on YAML for the configuration of diffusion parameters, the node maintains a clear separation between the AI's creative output (genres, lyrics) and the diffusion model's technical parameters. This modularity simplifies future updates, allows for easy swapping out of diffusion models, or enables the implementation of new parameter generation stages in subsequent versions, fostering long-term adaptability.

This node embodies a "code-as-creative-director" philosophy, leveraging LLMs to automate complex parameter generation and creative decision-making. This empowers users to focus on high-level concepts and artistic direction rather than intricate manual tuning, acting as a sophisticated bridge between abstract creative intent and the concrete, technical requirements of advanced diffusion models.