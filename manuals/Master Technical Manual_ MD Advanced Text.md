***

## **Master Technical Manual Template (v2.0 - Exhaustive)**

### **Node Names:**
1.  **`AdvancedTextNode`** (Display: `MD: Advanced Text Input`)
2.  **`TextFileLoader`** (Display: `MD: Text File Loader`)

**Category:** `MD_Nodes/Text`
**Version:** `1.8.0`
**Last Updated:** `Nov 2025`

---

### **Table of Contents**

1.  **Introduction**
    1.1. Executive Summary
    1.2. Conceptual Category
    1.3. Problem Domain & Intended Application
    1.4. Key Features (Functional & Technical)
2.  **Core Concepts & Theory**
    2.1. Theoretical Background: Deterministic Text Generation
    2.2. Mathematical & Algorithmic Formulation
    2.3. Data I/O Deep Dive
    2.4. Strategic Role in the ComfyUI Graph
3.  **Node Architecture & Workflow**
    3.1. Node Interface & Anatomy
    3.2. Input Port Specification
    3.3. Output Port Specification
    3.4. Workflow Schematics (Minimal & Advanced)
4.  **Parameter Specification**
    4.1. Text Processing Parameters
    4.2. Seeding & Randomization Parameters
    4.3. Wildcard Configuration
    4.4. File Loader Parameters
5.  **Applied Use-Cases & Recipes**
    5.1. Recipe 1: Dynamic Prompting with Nested Wildcards
    5.2. Recipe 2: Batch Processing with Auto-Incrementing Seeds
    5.3. Recipe 3: External Configuration Injection
6.  **Implementation Deep Dive**
    6.1. Source Code Walkthrough
    6.2. Dependencies & External Calls
    6.3. Performance & Resource Analysis
    6.4. String Lifecycle Analysis
7.  **Troubleshooting & Diagnostics**
    7.1. Error Code Reference
    7.2. Unexpected Visual Artifacts & Mitigation

---

### **1. Introduction**

#### **1.1. Executive Summary**
The **MD: Advanced Text Input** suite consists of two synergistic nodes designed to revolutionize text handling within ComfyUI.
The primary node, **AdvancedTextNode**, is a robust string processor capable of generating dynamic prompts via recursive wildcards, managing deterministic randomness through rigorous seed controls, and sanitizing text input.
The companion node, **TextFileLoader**, facilitates the ingestion of external text data (TXT, JSON, YAML) into the workflow, supporting dynamic reloading based on file modification events.

#### **1.2. Conceptual Category**
**Text Pre-Processor / Data Ingestion Utility**
These nodes function as the entry point for textual data in a generative workflow, transforming static strings or external files into dynamic, formatted prompts suitable for conditioning (CLIP) or configuration.

#### **1.3. Problem Domain & Intended Application**
* **Problem Domain:** Standard string nodes in ComfyUI lack logic for randomization, formatting, or external data loading. Users often struggle with repeating prompts in batch runs or managing massive text blocks within the UI. Furthermore, Python-to-JavaScript integer precision mismatches often cause seed drifting in web interfaces.
* **Intended Application (Use-Cases):**
    * **Dynamic Prompting:** Generating variations of a prompt (e.g., "A {red|blue} car") using a fixed or incrementing seed.
    * **Batch Variation:** Ensuring every image in a batch gets a unique, reproducible prompt variation.
    * **Configuration Management:** Loading complex JSON/YAML schemas from external files to keep the ComfyUI workspace clean.
* **Non-Application (Anti-Use-Cases):**
    * **Binary File Loading:** The loader is strictly for text-encoded files, not images or binaries.

#### **1.4. Key Features (Functional & Technical)**
* **Functional Features:**
    * **Recursive Wildcards:** Supports deeply nested options (e.g., `{style_{A|B}|style_C}`).
    * **Precision Seeding:** Implements "Increment" mode for batch runs and "Seed Lists" for specific curation.
    * **Text Hygiene:** Auto-removes extra spaces, strips whitespace, and handles casing (Upper/Lower).
    * **Live Reloading:** The File Loader detects file changes on disk and triggers workflow execution.
* **Technical Features:**
    * [cite_start]**JS-Safe Integer Clamping:** Limits seeds to $2^{53}-1$ to prevent integer overflow/precision loss in the web frontend[cite: 1].
    * [cite_start]**Compiled Regex:** Uses `re.compile` class attributes for high-performance pattern matching[cite: 1].
    * [cite_start]**Cryptographic Randomness:** Utilizes `secrets` module for high-entropy random seed generation when in 'random' mode[cite: 1].

### **2. Core Concepts & Theory**

#### **2.1. Theoretical Background**
The node operates on the principle of **Deterministic Chaos** in text generation. By binding wildcard expansion to a specific seed state, the node ensures that a prompt like `{cat|dog}` will always resolve to `cat` given Seed X, and `dog` given Seed Y. This allows for "explored randomness"—users can find a variation they like and reproduce it perfectly by locking the seed.

#### **2.2. Mathematical & Algorithmic Formulation**
**Seed Safety Constraint:**
To ensure compatibility between the Python backend (64-bit integers) and the JavaScript frontend (Double-precision float), the seed input $S$ is clamped to the JavaScript Safe Integer range:
$$ S_{safe} = \min(\max(S, 0), 2^{53} - 1) $$
Where $2^{53} - 1 = 9,007,199,254,740,991$. This prevents the "last digit jitter" often seen in web UIs handling large Python integers.

**Recursive Wildcard Resolution:**
For a text $T$ containing pattern $P$ (e.g., `{...}`), the expansion function $E(T, S)$ is defined recursively:
1.  Find innermost match $M$ in $T$.
2.  Split $M$ into options $O = [o_1, o_2, ..., o_n]$.
3.  Select $o_i$ using Pseudo-Random Number Generator (PRNG) initialized with $S$.
4.  Replace $M$ with $o_i$ in $T$.
5.  Repeat until no matches for $P$ remain.

#### **2.3. Data I/O Deep Dive**
* **Inputs:**
    * **`text` (STRING):** Multiline string buffer. Supports standard Python formatting chars (`\n`, `\t`).
    * **`seed` (INT):** 64-bit integer, internally clamped.
* **Outputs:**
    * **`processed_text` (STRING):** The final string after wildcard expansion and formatting.
    * **`selected_seed` (INT):** The actual seed used for logic (vital when using Seed Lists).
    * **`text_length` (INT):** Character count of the output.
    * **`wildcard_count` (INT):** Total number of substitutions performed.

#### **2.4. Strategic Role in the ComfyUI Graph**
* **Placement:** Start of the graph (Encoders) or Pre-sampling.
* **Synergistic Nodes:** Connects to `CLIP Text Encode`, `ShowText`, or custom prompt parsers.
* **File Loader Role:** Acts as a "Source of Truth" node. Changing the file content updates the graph state without requiring manual node editing.

### **3. Node Architecture & Workflow**

#### **3.1. Node Interface & Anatomy**
The **AdvancedTextNode** features a large text area for input, flanked by toggle switches for text formatting (Lowercase, Strip Whitespace) and seed controls.
The **TextFileLoader** is a compact node accepting a file path and encoding selector.

#### **3.2. Input Port Specification**
* **AdvancedTextNode:**
    * **`text` (STRING)** - (Required)
        * **Description:** The primary content buffer. Supports dynamic wildcards.
    * **`seed` (INT)** - (Optional)
        * **Default:** 0.
        * **Constraint:** clamped to $[0, 9,007,199,254,740,991]$.
    * **`seed_mode` (COMBO)** - (Optional)
        * **Options:** `fixed`, `random`, `increment`.
        * **Impact:** Controls how the `seed` parameter is interpreted between runs.
    * **`seed_list` (STRING)** - (Optional)
        * **Description:** A wildcard-formatted list of seeds (e.g., `{100|200|300}`). The node picks one and outputs it as `selected_seed`.

* **TextFileLoader:**
    * **`file_path` (STRING)** - (Required)
        * **Description:** Absolute or relative path to the target file.
    * **`encoding` (COMBO)** - (Optional)
        * **Default:** `utf-8`.
        * **Options:** `utf-8`, `ascii`, `latin-1`.

#### **3.3. Output Port Specification**
* **AdvancedTextNode:**
    * **`processed_text` (STRING):** Final result.
    * **`original_text` (STRING):** Input pass-through.
    * **`seed_used` (INT):** The active PRNG state.
    * **`text_length` (INT):** `len(processed_text)`.
    * **`wildcard_count` (INT):** Number of replacements.

#### **3.4. Workflow Schematics**

**Minimal Functional Graph:**
`[TextFileLoader] -> (text) -> [AdvancedTextNode] -> (processed_text) -> [CLIP Text Encode]`

**Advanced Batch Graph:**
```json
{
  "nodes": [
    {
      "type": "AdvancedTextNode",
      "inputs": {
        "text": "A {beautiful|stunning} landscape",
        "seed_mode": "increment",
        "wildcard_mode": true
      }
    }
  ]
}
```

### **4. Parameter Specification**

#### **4.1. Text Processing Parameters**
* **`strip_whitespace` (BOOLEAN)**
    * **Function:** Performs `.strip()` on the final string.
    * **Impact:** Removes accidental line breaks at the start/end of pasted text.
* **`remove_extra_spaces` (BOOLEAN)**
    * **Function:** Replaces regex `r' +'` with `' '`.
    * **Impact:** Collapses "A   red    car" to "A red car".
* **`lowercase` / `uppercase` (BOOLEAN)**
    * **Priority:** `lowercase` takes precedence if both are True.

#### **4.2. Seeding & Randomization Parameters**
* **`seed_mode`**:
    * **`fixed`:** Uses the `seed` widget value exactly.
    * **`random`:** Generates a new `secrets.randbelow` integer every execution, ignoring the widget.
    * **`increment`:** Adds 1 to the input seed for each run (wraps at `SEED_MAX`). Essential for batch generation where every frame needs a new, predictable prompt.
* **`seed_list` & `seed_offset`**:
    * Allows providing a specific pool of seeds.
    * **Formula:** `index = (input_seed + seed_offset) % list_length`.

#### **4.3. Wildcard Configuration**
* **`wildcard_mode` (BOOLEAN)**: Master toggle. If False, curly braces are treated as literals.
* **`wildcard_syntax`**:
    * **`curly_braces`:** Uses `{option1|option2}`. Standard for dynamic prompts.
    * **`double_underscore`:** Uses `__option1|option2__`. Useful if `{}` conflicts with JSON syntax.

#### **4.4. File Loader Parameters**
* **`file_path`**: Supports forward (`/`) and backward (`\`) slashes.
* **`IS_CHANGED` Logic**: The node calculates a hash of `(file_path, modification_time)`. If the file is saved externally, the hash changes, triggering an auto-queue in ComfyUI (if set to "Queue Prompt").

### **5. Applied Use-Cases & Recipes**

#### **5.1. Recipe: Dynamic Prompting with Nested Wildcards**
* **Objective:** Generate varied character descriptions.
* **Input Text:** `A {fantasy|sci-fi} character wearing {{iron|steel} armor|{leather|cloth} robes}.`
* **Settings:**
    * `wildcard_mode`: True
    * `seed_mode`: `increment`
* **Result (Run 1):** "A fantasy character wearing iron armor."
* **Result (Run 2):** "A sci-fi character wearing cloth robes."

#### **5.2. Recipe: Batch Processing with Auto-Incrementing Seeds**
* **Objective:** Generate 100 variations of a prompt for a grid.
* **Setup:**
    1.  Set Batch Size in KSampler to 100.
    2.  Set `AdvancedTextNode` -> `seed_mode` to `increment`.
    3.  Set `wildcard_mode` to `True`.
* **Outcome:** The node executes 100 times. Each execution receives `seed + n`, ensuring 100 unique wildcard combinations.

#### **5.3. Recipe: External Configuration Injection**
* **Objective:** Load a large negative prompt from a shared file.
* **Setup:**
    1.  Create `negative_prompts.txt` in your ComfyUI input folder.
    2.  Use `TextFileLoader` pointing to `input/negative_prompts.txt`.
    3.  Connect output to `CLIP Text Encode (Negative)`.
* **Benefit:** Update the negative prompt for all workflows instantly by editing one text file.

### **6. Implementation Deep Dive**

#### **6.1. Source Code Walkthrough**
**Wildcard Processing (`_process_wildcards_recursive`):**
[cite_start][cite: 1]
```python
while iteration < max_iterations:
    match = pattern.search(text)
    if not match: break
    options = [opt.strip() for opt in match.group(1).split('|') if opt.strip()]
    chosen_option = rng.choice(options)
    text = text[:match.start()] + chosen_option + text[match.end():]
```
The logic uses a `while` loop to repeatedly scan for the *innermost* regex match. This allows it to resolve `{outer|{inner}}` by first resolving `{inner}` then re-scanning to resolve the resulting `{outer|result}`.

**Seed Validation (`_validate_seed`):**
[cite_start][cite: 1]
```python
if int_value > SEED_MAX:
    logging.warning(f"... Clamping to {SEED_MAX}.")
    return SEED_MAX
```
Ensures strict adherence to the JavaScript integer limit to prevent frontend display bugs.

#### **6.2. Dependencies & External Calls**
* **`re`**: Used for `_PATTERN_CURLY` (`r'\{([^{}]+?)\}'`) and `_PATTERN_UNDERSCORE`. The patterns use `+?` (non-greedy) matching to find the smallest possible enclosed group (innermost nesting).
* **`secrets`**: Used in `IS_CHANGED` (for random mode) and `_generate_random_seed` for cryptographic strength randomness.

#### **6.3. Performance & Resource Analysis**
* **Complexity:** Wildcard processing is $O(N \times M)$ where $N$ is text length and $M$ is nesting depth.
* **Optimization:** Regex patterns are pre-compiled as class attributes (`_PATTERN_CURLY`) to avoid compilation overhead on every execution.
* **Safety:** A hard limit of 100 iterations prevents infinite loops if a user creates a recursive wildcard trap (though logically difficult with the current regex).

#### **6.4. String Lifecycle Analysis**
1.  **Input:** Raw string from Widget or File Loader.
2.  **Wildcard Pass:** Recursive regex expansion using Seeded Randomness.
3.  **Sanitization:** `strip()`, regex `sub(r' +', ' ')`.
4.  **Casing:** `.lower()` or `.upper()`.
5.  **Output:** Final string passed to ComfyUI string bus.

### **7. Troubleshooting & Diagnostics**

#### **7.1. Error Code Reference**
* **Warning: "Seed X exceeds JS-safe range"**
    * **Cause:** You entered a 64-bit integer (e.g., $10^{18}$) manually.
    * **Mitigation:** The node automatically clamps it to $9 \times 10^{15}$. No action needed, but be aware the seed used differs from input.
* **Warning: "Max wildcard processing iterations reached"**
    * **Cause:** Extremely deep nesting (>100 levels) or a malformed pattern.
    * **Mitigation:** Simplify prompt structure.
* **Error: "File not found" (TextFileLoader)**
    * **Cause:** Typo in path or relative path confusion.
    * **Mitigation:** Use absolute paths or check the ComfyUI console for the precise CWD (Current Working Directory).

#### **7.2. Unexpected Visual Artifacts & Mitigation**
* **Artifact:** Wildcards not replacing (`{option1|option2}` appears in output).
    * **Cause:** `wildcard_mode` is False.
    * **Correction:** Enable the toggle switch.
* **Artifact:** Text is all lowercase.
    * **Cause:** `lowercase` toggle is active.
    * **Correction:** Disable the toggle.