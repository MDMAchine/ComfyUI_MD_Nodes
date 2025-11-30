# Master Technical Manual Template (v2.0 – Exhaustive)
-----------------------------------------------------
### Node Name:
**WildcardPromptBuilder**

**Display Name:** MD: Wildcard Prompt Builder

**Category:** MD_Nodes/Prompt Generation

**Version:** v1.3.0

**Last Updated:** 2023-10-XX

---

## Table of Contents
1. Introduction  
   1.1 Executive Summary  
   1.2 Conceptual Category  
   1.3 Problem Domain & Intended Application  
   1.4 Key Features (Functional & Technical)  
2. Core Concepts & Theory  
   2.1 Theoretical Background  
   2.2 Mathematical & Algorithmic Formulation  
   2.3 Data I/O Deep Dive  
   2.4 Strategic Role in the ComfyUI Graph  
3. Node Architecture & Workflow  
   3.1 Node Interface & Anatomy  
   3.2 Input Port Specification  
   3.3 Output Port Specification  
   3.4 Workflow Schematics (Minimal & Advanced)  
4. Parameter Specification  
5. Applied Use-Cases & Recipes  
6. Implementation Deep Dive  
7. Troubleshooting & Diagnostics

---

## 1. Introduction

### 1.1 Executive Summary
The **WildcardPromptBuilder** node is an advanced, multi-mode prompt generation engine developed for music production workflows in the ComfyUI environment. It leverages a hybrid approach—combining deterministic wildcard expansion with stochastic large language model (LLM) selection—to generate descriptive tags and lyrical prompts dynamically. Its robust design ensures fallback functionality across various modes: custom inputs, file-based templates, or direct LLM generation.

### 1.2 Conceptual Category
This node is categorized under **Prompt Generation** within the MD_Nodes framework. It specifically caters to music production pipelines by automating the creation of creative descriptions for musical elements such as genre tags, vocal styles, lyrics, and duration strings. Its dual approach—using both internally seeded wildcard expansion and LLM-based refinement—is designed to ensure high variability and contextual relevance.

### 1.3 Problem Domain & Intended Application
- **Problem Domain:**  
  Automating the generation of creative prompts in music production is often error-prone when done manually, particularly when integrating descriptive tags for genre, vocal style, or lyrics. The node addresses this challenge by providing a robust and flexible method to generate contextually appropriate content.

- **Intended Application (Use-Cases):**  
  - Generate dynamic descriptions for music genres using internally seeded wildcard expansion or hybrid LLM calls.
  - Produce creative vocal tags that adapt based on user-defined concepts.
  - Create lyrical samples or enforce an instrumental mode via file loading, custom templates, or direct LLM generation.
  - Supply duration strings to control playback timing in subsequent processing nodes.

- **Non-Application (Anti-Use-Cases):**  
  The node is less suited when manual curation is desired, where the input data is too restrictive for creative prompt generation, or when real-time human intervention is preferred over automated text creation.

### 1.4 Key Features (Functional & Technical)
* **Functional Features:**
  - **Multi-Mode Operation:** Supports modes “wildcard”, “llm”, and “hybrid” to adapt the generation process.
  - **Custom Template Overrides:** Direct custom inputs for genre, vocal, or lyric prompts that take absolute priority over file-based or mode-generated content.
  - **File Loader Integration:** Dynamically loads external wildcard templates from user-defined `.txt` files located in subfolders (e.g., “genre”, “vocal”, “lyrics”).
  - **Preview Rendering:** Renders a visual preview image of generated text components using robust font fallback and PIL-based image drawing.
  
* **Technical Features:**
  - Implements the **WildcardExpander** class to process patterns such as `{option1|option2}` using seeded random selection, with up to 100 iterations for full template substitution.
  - Uses lazy-loading functions (`_get_ollama_models_lazy` and `_get_lm_studio_models_lazy`) that cache remote LLM model lists for improved performance.
  - Employs a unified LLM caller via `_call_llm`, supporting both “ollama” and “lm_studio” backends with built-in timeout and exception handling.
  - Converts rendered preview images from PIL/Numpy arrays to normalized torch.Tensor objects, ensuring GPU compatibility in downstream nodes.

---

## 2. Core Concepts & Theory

### 2.1 Theoretical Background
The **WildcardPromptBuilder** merges deterministic template processing with stochastic language generation techniques. Wildcard expansion leverages regular expressions to substitute placeholders (e.g., `{option1|option2}`) using a seeded random mechanism. Meanwhile, the node integrates large language models (LLM) for contextually aware content generation, drawing on established generative prompt engineering practices.

### 2.2 Mathematical & Algorithmic Formulation
The wildcard expansion algorithm can be abstracted as follows:
  • **Input:** A template string T containing pattern matches like `{option1|option2}`.
  • **Process:**
   1. Identify all substrings matching the regular expression `r'\{([^{}]+)\}'`.
   2. For each match, randomly select one option using a seeded random function:
     Choice = rand({option1, option2, …})
   3. Replace the pattern with the selected option.
   4. Repeat until no patterns remain or a maximum of 100 iterations is reached.
  • **Output:** A fully expanded template string.

In pseudocode:

  function expand(template T):
   while exists {pattern} in T and iteration < MAX_ITER:
    for each pattern in T:
     selected = rand(options)
     replace pattern with selected
   return T

Similar substitution logic is applied within the `_generate_hybrid_genre` and `_generate_hybrid_vocal` methods to prepare prompts for LLM calls.

### 2.3 Data I/O Deep Dive
- **STRING Outputs (Genre, Vocal, Lyrics):**
  - Expected as comma-separated values derived either from wildcard expansion or returned by LLM responses.
  - For example: "Space ambient with deep bass" or "[Instrumental]" if forced.
  
- **IMAGE Tensors (Preview Image):**
  - Constructed in `_render_text_preview` using PIL, then converted to a numpy array and normalized (values between 0 and 1).
  - Expected shape: [1, PREVIEW_HEIGHT, PREVIEW_WIDTH, 3] with RGB channel ordering.

- **INT Output (Seed):**
  - Reflects the effective random seed used during generation, provided for debugging purposes.
  
- **No explicit latent or vocal data types are processed internally; instead, extensive dictionaries store available descriptive options.**

### 2.4 Strategic Role in the ComfyUI Graph
The node is intended to be placed at a junction where creative input (from concept nodes) feeds into further music production pipelines:
 • **Preceding Nodes:** Concept generators or file loader nodes providing template content.
 • **Following Nodes:** Audio samplers, genre selectors, and visualizers that consume the generated prompts.

Its design emphasizes flexibility so it can integrate seamlessly with both manual overrides (via custom templates) and automated processing pipelines.

---

## 3. Node Architecture & Workflow

### 3.1 Node Interface & Anatomy
The WildcardPromptBuilder interface is structured with clearly numbered input and output ports:

**Input Ports (with Numbers):**
1. **generation_mode**  
2. **concept**  
3. **seed**  
4. **randomize_seed**  
5. **duration_template**  
6. **llm_backend**  
7. **ollama_api_url**  
8. **ollama_model**  
9. **lm_studio_api_url**  
10. **lm_studio_model**  
11. **load_genre_file** (populated dynamically)  
12. **load_vocal_file** (populated dynamically)  
13. **load_lyrics_file** (populated dynamically)  
14. **custom_genre_template**  
15. **custom_vocal_template**  
16. **custom_lyrics_template**  
17. **expand_custom_templates**  
18. **generate_genre**  
19. **generate_vocals**  
20. **generate_lyrics**  
21. **force_instrumental**

**Output Ports (with Numbers):**
1. **genre_tags**  
2. **vocal_tags**  
3. **lyrics**  
4. **duration_string**  
5. **seed**  
6. **text_preview**

*A high-resolution screenshot of the node interface (annotated with these port numbers) should be included in the user guide for visual reference.*

### 3.2 Input Port Specification

- **#1: generation_mode (STRING)**  
 • **Description:** Determines whether to use wildcard expansion (“wildcard”), direct LLM calls (“llm”), or a combined hybrid approach.
 • **Default:** "wildcard"
 • **Impact:** Dictates the branch of logic executed in the main `execute` method.

- **#2: concept (STRING, multiline)**  
 • **Description:** A creative descriptor that guides prompt generation; examples include “deep space ambient” or “electro fusion.”
 • **Default:** ""
 • **Impact:** Used to form LLM prompts and influence the style of generated content.

- **#3: seed (INT)**  
 • **Description:** Controls the starting value for randomness. If overridden by randomize_seed, a new integer is generated.
 • **Default:** 0
 • **Impact:** Ensures reproducibility when required and governs wildcard choices.

- **#4: randomize_seed (BOOLEAN)**  
 • **Description:** When set to True, the node generates a fresh random seed each run.
 • **Default:** True
 • **Impact:** Determines deterministic versus non-deterministic behavior in template processing.

- **#5: duration_template (STRING)**  
 • **Description:** Contains numeric alternatives (e.g., "{120|180}") that are processed by the wildcard expander to output a final duration.
 • **Default:** "{120|180}"
 • **Impact:** Directly affects the computed duration string used in further processing.

- **#6: llm_backend (ARRAY of STRING, e.g., ["ollama", "lm_studio"])**  
 • **Description:** Selects which LLM API to use when mode is “llm” or “hybrid.”
 • **Default:** "ollama"
 • **Impact:** Influences the URL endpoints and models invoked in `_call_llm`.

- **#7: ollama_api_url (STRING)**  
 • **Description:** The endpoint for the OLLAMA backend.
 • **Default:** Provided via lazy-loading (typically DEFAULT_OLLAMA_URL)
 • **Impact:** Used when llm_backend is “ollama.”

- **#8: ollama_model (STRING)**  
 • **Description:** Specifies which OLLAMA model to use for generation.
 • **Default:** Fallback from `_get_ollama_models_lazy` if unavailable.
 • **Impact:** Determines the style and content of LLM responses.

- **#9: lm_studio_api_url (STRING)**  
 • **Description:** The endpoint URL for LM Studio when selected as the backend.
 • **Default:** Provided via lazy-loading.
 • **Impact:** Activated when llm_backend is “lm_studio.”

- **#10: lm_studio_model (STRING)**  
 • **Description:** Specifies the model identifier for LM Studio.
 • **Default:** Fallback from `_get_lm_studio_models_lazy`
 • **Impact:** Affects LLM-generated content in this mode.

- **#11: load_genre_file (DYNAMIC ARRAY of STRING)**  
 • **Description:** Dynamically populated list of available genre template files.
 • **Default:** ["None"]
 • **Impact:** Enables selection from file-based templates or a “Random” trigger for file selection.

- **#12: load_vocal_file (DYNAMIC ARRAY of STRING)**  
 • **Description:** Similar to #11 but for vocal template files.
 • **Default:** ["None"]

- **#13: load_lyrics_file (DYNAMIC ARRAY of STRING)**  
 • **Description:** Provides available lyric templates from the “lyrics” subfolder.
 • **Default:** ["None"]

- **#14: custom_genre_template (STRING, multiline)**  
 • **Description:** User-defined genre template that overrides default generation methods when non-empty.
 • **Default:** ""
 • **Impact:** If provided, takes absolute priority over file and LLM modes.

- **#15: custom_vocal_template (STRING, multiline)**  
 • **Description:** Direct vocal style template input.
 • **Default:** ""
 • **Impact:** Overrides standard generation if non-empty; subject to wildcard expansion if expand_custom_templates is True.

- **#16: custom_lyrics_template (STRING, multiline)**  
 • **Description:** User-defined lyric content that bypasses automatic generation.
 • **Default:** ""
 • **Impact:** When provided, its contents are used directly or processed for wildcards based on toggle settings.

- **#17: expand_custom_templates (BOOLEAN)**  
 • **Description:** Indicates whether wildcard patterns within custom templates should be expanded.
 • **Default:** True
 • **Impact:** Determines literal versus dynamic processing of user-supplied text.

- **#18: generate_genre (BOOLEAN)**  
 • **Description:** Toggles generation of genre tags.
 • **Default:** True

- **#19: generate_vocals (BOOLEAN)**  
 • **Description:** Enables or disables vocal tag generation.
 • **Default:** True

- **#20: generate_lyrics (BOOLEAN)**  
 • **Description:** Controls whether lyrics are generated.
 • **Default:** True

- **#21: force_instrumental (BOOLEAN)**  
 • **Description:** When true, suppresses vocal and lyric outputs to enforce an instrumental mode.
 • **Default:** False
 • **Impact:** Sets both vocals to an empty string and lyrics to “[Instrumental]”.

### 3.3 Output Port Specification

- **#1: genre_tags (STRING):**  
 • Generated via wildcard expansion, LLM call, or custom template override.  
 • Format: A comma-separated list of descriptive tags.

- **#2: vocal_tags (STRING):**  
 • Similar in function to genre tags but pertains to vocal style descriptors.  
 • May be empty if `force_instrumental` is activated.

- **#3: lyrics (STRING):**  
 • Contains generated lyrical content or “[Instrumental]” if overridden by force_instrumental.  
 • Typically a multiline string that may be shortened in the preview display.

- **#4: duration_string (STRING):**  
 • Result of processing the provided duration_template via wildcard expansion.  
 • Format example: "180s (Seed: …)".

- **#5: seed (INT):**  
 • Reflects the effective random seed used during generation for reproducibility tracking.

- **#6: text_preview (IMAGE, torch.Tensor):**  
 • A rendered image generated in `_render_text_preview`.  
 • Expected tensor shape is [1, PREVIEW_HEIGHT, PREVIEW_WIDTH, 3] with normalized pixel values between 0 and 1.

### 3.4 Workflow Schematics
- **Minimal Workflow Example:**  
  • The node receives a concept input along with a wildcard duration template and default settings (generation_mode = “wildcard”).  
  • It performs seeded random expansion on the duration template and processes genre/vocal/lyrics via internal logic, returning both text outputs and a preview image.

- **Advanced Workflow Example:**  
  • Integration with file loader nodes that supply external templates overrides custom inputs.  
  • The node’s output is fed into downstream nodes such as audio samplers or visualizers.
  • A sample JSON workflow diagram (refer to ComfyUI documentation) would illustrate connections from concept, file loaders, and this node.

---

## 4. Parameter Specification
Each parameter—documented below—is directly mapped from the INPUT_TYPES defined within the source code.

**Parameter: generation_mode**  
- *Type:* STRING  
- **Default:** "wildcard"  
- **Description:** Chooses between:
 • “wildcard” – pure template expansion;  
 • “llm” – direct language model prompting;  
 • “hybrid” – a combination of internal libraries and LLM calls.  
- **Impact:** Determines the execution path in the main `execute` method.

**Parameter: concept**  
- *Type:* STRING (multiline)  
- **Default:** ""  
- **Description:** Provides creative context for generating prompts.  
- **Impact:** Influences both the style and content of generated genre, vocal, and lyric outputs.

**Parameter: seed**  
- *Type:* INT  
- **Default:** 0  
- **Description:** Controls the random number generator’s state to ensure reproducibility when needed.  
- **Impact:** Affects wildcard expansion results and file selection in randomized modes.

**Parameter: randomize_seed**  
- *Type:* BOOLEAN  
- **Default:** True  
- **Description:** When enabled, overrides the provided seed with a randomly generated integer at runtime.  
- **Impact:** Determines if outputs will vary between runs or remain consistent when manually seeded.

**Parameter: duration_template**  
- *Type:* STRING  
- **Default:** "{120|180}"  
- **Description:** Template string used to generate a final duration output after wildcard expansion.  
- **Impact:** Directly affects the computed value returned as part of the node’s outputs.

**Parameter: llm_backend**  
- *Type:* ARRAY of STRING (e.g., ["ollama", "lm_studio"])  
- **Default:** "ollama"  
- **Description:** Selects which LLM service is used when generation mode is “llm” or “hybrid.”  
- **Impact:** Influences the endpoints and model calls in `_call_llm`.

**Parameter: ollama_api_url**  
- *Type:* STRING  
- **Default:** As defined by lazy-loading constants (e.g., DEFAULT_OLLAMA_URL)  
- **Description:** The URL endpoint for the OLLAMA backend.  
- **Impact:** Used when llm_backend is “ollama.”

**Parameter: ollama_model**  
- *Type:* STRING  
- **Default:** Fallback model name from `_get_ollama_models_lazy`  
- **Description:** Specifies which OLLAMA model to invoke for prompt generation.  
- **Impact:** Affects the creativity and specificity of LLM outputs.

**Parameter: lm_studio_api_url**  
- *Type:* STRING  
- **Default:** As defined by lazy-loading constants (e.g., DEFAULT_LM_STUDIO_URL)  
- **Description:** URL endpoint used when llm_backend is “lm_studio.”  
- **Impact:** Directly determines the LLM service called via `_call_llm`.

**Parameter: lm_studio_model**  
- *Type:* STRING  
- **Default:** Fallback model identifier from `_get_lm_studio_models_lazy`  
- **Description:** Specifies which LM Studio model to use when llm_backend is “lm_studio.”  
- **Impact:** Influences the generated content’s style and context.

**Parameter: load_genre_file**  
- *Type:* DYNAMIC ARRAY of STRING  
- **Default:** ["None"]  
- **Description:** Provides a dynamically populated list of genre template file names available in the "wildcards/genre" folder.  
- **Impact:** Facilitates selection between using default templates or external files (with support for “Random” triggers).

**Parameter: load_vocal_file**  
- *Type:* DYNAMIC ARRAY of STRING  
- **Default:** ["None"]  
- **Description:** Similar to load_genre_file but targets the "wildcards/vocal" folder.  
- **Impact:** Enables file-based template overrides for vocal tag generation.

**Parameter: load_lyrics_file**  
- *Type:* DYNAMIC ARRAY of STRING  
- **Default:** ["None"]  
- **Description:** Lists available lyric templates from the "wildcards/lyrics" directory.  
- **Impact:** Allows external lyric content to be used in place of generated text.

**Parameter: custom_genre_template**  
- *Type:* STRING (multiline)  
- **Default:** ""  
- **Description:** A user-provided genre template that, if non-empty, takes precedence over file or mode-based generation.  
- **Impact:** Overrides automatic generation; processed for wildcards based on expand_custom_templates flag.

**Parameter: custom_vocal_template**  
- *Type:* STRING (multiline)  
- **Default:** ""  
- **Description:** User-defined input for vocal style descriptions with priority over other methods.  
- **Impact:** Bypasses file and LLM generation when provided; subject to wildcard expansion if enabled.

**Parameter: custom_lyrics_template**  
- *Type:* STRING (multiline)  
- **Default:** ""  
- **Description:** Direct lyric content input that overrides automatic generation.  
- **Impact:** Ensures literal or expanded output based on toggle settings, bypassing file and LLM methods.

**Parameter: expand_custom_templates**  
- *Type:* BOOLEAN  
- **Default:** True  
- **Description:** When enabled, any embedded wildcard patterns within custom templates are processed for dynamic expansion.  
- **Impact:** Controls whether custom inputs are treated as static text or subject to further wildcard processing.

**Parameter: generate_genre**  
- *Type:* BOOLEAN  
- **Default:** True  
- **Description:** Enables the generation of genre tags; if set to False, this output remains empty.  
- **Impact:** Directly toggles the execution path for genre tag creation.

**Parameter: generate_vocals**  
- *Type:* BOOLEAN  
- **Default:** True  
- **Description:** Controls whether vocal style descriptions are generated.  
- **Impact:** Influences downstream processing; if False, vocal outputs may be suppressed unless overridden by force_instrumental.

**Parameter: generate_lyrics**  
- *Type:* BOOLEAN  
- **Default:** True  
- **Description:** Dictates if lyrics should be automatically produced.  
- **Impact:** Affects LLM call utilization and fallback to “[Instrumental]” when forced.

**Parameter: force_instrumental**  
- *Type:* BOOLEAN  
- **Default:** False  
- **Description:** When enabled, both vocal tags and lyric outputs are cleared (or replaced with "[Instrumental]") regardless of other settings.  
- **Impact:** Provides a mechanism to bypass creative generation in favor of an instrumental production mode.

---

## 5. Applied Use-Cases & Recipes

### Recipe 1: Automated Genre & Vocal Generation via Hybrid Mode
- **Setup:**
  • Set `generation_mode` to "hybrid".  
  • Provide a rich concept such as “space ambient with deep bass.”  
  • Leave file-based overrides empty (or set to “None”).  
  • Ensure custom template fields remain blank or use expand_custom_templates.
- **Outcome:** The node uses internal dictionaries and the LLM backend to produce creative, context-sensitive genre and vocal tag outputs. A preview image visually displaying these tags is also rendered.

### Recipe 2: Direct Custom Template Override
- **Setup:**
  • Populate `custom_genre_template`, `custom_vocal_template`, and/or `custom_lyrics_template` with user-provided text containing potential wildcards.  
  • Leave generation_mode at any valid setting since custom inputs override default behavior.
- **Outcome:** The node bypasses file-based or LLM-generated content in favor of the directly supplied templates—processing embedded wildcards if enabled—resulting in outputs that fully reflect user intent.

### Recipe 3: Forced Instrumental Mode
- **Setup:**
  • Set `force_instrumental` to True.  
  • Optionally, adjust other settings as needed.
- **Outcome:** Regardless of other inputs, the node clears both vocal and lyric outputs (vocal becomes an empty string and lyrics default to “[Instrumental]”), making it suitable for productions focusing on purely instrumental tracks.

*Additional recipes may vary based on choosing different LLM backends or modifying duration_template values. Each recipe is designed to be integrated within a broader ComfyUI workflow for music production.*

---

## 6. Implementation Deep Dive

### 6.1 Source Code Walkthrough
- **WildcardExpander Class:**  
  - The constructor accepts and stores the seed value, which is later used in the `expand` method.
  - Uses the regular expression `r'\{([^{}]+)\}'` to detect patterns and substitutes each with a randomly selected option from within the curly braces. Iteration continues until no patterns remain or up to a defined limit (e.g., 100 iterations).

- **File Reading & Template Loading:**  
  - Functions like `_read_file_content(subfolder, filename)` perform safe file access using `os` and `glob`, returning an empty string upon failure.
  - `_get_files_from_dir()` (conceptually represented by load genre/vocal/lyrics ports) dynamically gathers available templates from the designated "wildcards" directory.

- **LLM Integration (_generate_hybrid_genre & _generate_hybrid_vocal):**  
  - These methods format internal dictionaries—populated with creative alternatives—and then invoke `_call_llm`, which abstracts API calls to either OLLAMA or LM Studio.
  - Timeout and error logging are implemented within `_call_llm` to handle network issues gracefully.

- **Preview Rendering (_render_text_preview):**  
  - A new RGB image is created using PIL with fixed dimensions defined by `PREVIEW_WIDTH` and `PREVIEW_HEIGHT`.
  - The method attempts to load a robust font (Arial) with fallbacks (DejaVuSans, then PIL’s default), ensuring cross-platform consistency.
  - Text is drawn incrementally: genre tags at the top, vocal tags, duration string (with seed info), followed by a truncated snippet of lyrics.
  - After drawing, the image is converted to a numpy array and normalized into a torch.Tensor for compatibility with subsequent ComfyUI processing nodes.

- **execute Method:**  
  - This central function performs:
    1. Initialization and validation of all input parameters.
    2. Determination of generation path: custom template overrides take priority; if absent, the system checks file-based overrides; otherwise, mode-dependent logic is executed.
    3. Wildcard expansion applied to the duration_template.
    4. Calls to `_generate_hybrid_genre` and/or `_generate_hybrid_vocal`, or LLM methods if in “llm” or “hybrid” modes.
    5. Rendering of a visual preview via `_render_text_preview`.
    6. Exception handling throughout ensures fallback values (including a “black box” image tensor) are returned on critical failures.

### 6.2 Dependencies & Library Utilization
- **Standard Python Libraries:**  
  - `re` for regex operations in wildcard processing.
  - `os`, `glob` for file path management and content retrieval.
  - `random` and `secrets` for generating and managing random seed values.

- **External Libraries:**  
  - `requests`: For making HTTP API calls to remote LLM backends.
  - `PIL` (Python Imaging Library): To create, draw on, and manipulate the preview image.
  - `numpy` and `torch`: For converting images into normalized tensor formats suitable for downstream nodes.

- **Algorithmic Dependencies:**  
  - The node combines deterministic template expansion with stochastic LLM responses to achieve both consistency and creative variability.

### 6.3 Performance & Resource Analysis
- **Preview Rendering Efficiency:**  
  - The rendering process uses a fixed-size canvas (as defined by `PREVIEW_WIDTH` and `PREVIEW_HEIGHT`), ensuring predictable resource usage.
  - Font fallback mechanisms minimize performance degradation on platforms lacking standard fonts.

- **Processing Overhead of Wildcard Expansion:**  
  - Although the algorithm allows up to 100 iterations, most templates require far fewer passes. This ensures minimal computational overhead even under heavy load.

- **LLM API Call Optimizations:**  
  - Timeout settings (API_TIMEOUT_SECONDS) prevent indefinite hangs.
  - Model list caching (MODEL_LIST_CACHE_SECONDS = 300 seconds) reduces redundant network requests, improving overall node responsiveness in recurrent workflows.

### 6.4 Tensor Lifecycle Analysis
- **In _render_text_preview:**  
  - A PIL image is created and populated with text elements.
  - The final image is converted into a numpy array, normalized to float32 (values between 0 and 1), and then reshaped as [1, H, W, 3].
- **Downstream Compatibility:**  
  - The output tensor is immediately compatible with other ComfyUI nodes that expect image data in tensor form.

---

## 7. Troubleshooting & Diagnostics

### 7.1 Common Error Messages & Symptoms
- **"OLLAMA fetch failed:" / "LM Studio fetch failed:"**  
 • Indicates an inability to retrieve model lists from the respective backend API.
 • Check network connectivity, verify URL endpoints (DEFAULT_OLLAMA_URL and DEFAULT_LM_STUDIO_URL), and inspect any firewall restrictions.

- **Tensor Dimension Mismatch:**  
 • May occur if the preview image rendering returns unexpected dimensions.
 • Validate that `PREVIEW_WIDTH` and `PREVIEW_HEIGHT` are set correctly and that font fallbacks resolve all missing-font issues.

- **File Read Errors:**  
 • Log messages such as "Failed to read [filepath]: [error]" indicate issues with file permissions or missing files in the wildcards directory.
 • Confirm that files exist within “wildcards/genre”, “wildcards/vocal”, and “wildcards/lyrics” folders and that the node has read access.

### 7.2 Diagnostics & Remediation Strategies
- **LLM API Issues:**  
  - Verify network connectivity and validate API endpoints.
  - Ensure that any required authentication tokens or configuration settings are correctly supplied.
  - Consult backend server logs for additional error details.

- **File Access Problems:**  
  - Ensure the correct file paths are provided in input parameters.
  - Confirm file permissions on the host system to allow read operations.
  - If using a “Random” selection, verify that at least one valid `.txt` file is present.

- **Preview Rendering Issues:**  
  - Confirm that necessary fonts (Arial or DejaVuSans) are installed. If not, rely on PIL’s default fallback.
  - Check the version of PIL installed; update if compatibility issues arise with image drawing functions.

### 7.3 Logging & Debugging Recommendations
- Enable verbose logging to capture detailed debug output at each step:
  • **Debug Level:** Outputs seed regeneration events and successful wildcard expansion occurrences.
  • **Warning Level:** Captures file access errors, API connectivity warnings, or fallback triggers (e.g., model selection issues).
  • **Error Level:** Provides full exception tracebacks in critical failure scenarios such as unhandled LLM API exceptions.
- Review logs periodically to identify recurring patterns that may indicate configuration issues or network problems.

---

*End of Master Technical Manual Template (v2.0 – Exhaustive)*