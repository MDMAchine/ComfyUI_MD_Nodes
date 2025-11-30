***

## **Master Technical Manual Template (v2.0 - Exhaustive)**

### **Node Suite: MD Dynamic Filename Toolkit**
**Nodes Included:**
1.  `SmartFilenameBuilder` (Display: **MD: Smart Filename Builder**)
2.  `FilenameTokenReplacer` (Display: **MD: Filename Token Replacer**)
3.  `FilenameCounterNode` (Display: **MD: Filename Counter**)

**Category:** `MD_Nodes/Utility`
**Version:** `1.1.0`
**Last Updated:** `Nov 2025`

---

### **Table of Contents**

1.  **Introduction**
    1.1. Executive Summary
    1.2. Conceptual Category
    1.3. Problem Domain & Intended Application
    1.4. Key Features (Functional & Technical)
2.  **Core Concepts & Theory**
    2.1. Theoretical Background: Filename Hygiene
    2.2. Mathematical & Algorithmic Formulation
    2.3. Data I/O Deep Dive
    2.4. Strategic Role in the ComfyUI Graph
3.  **Node Architecture & Workflow**
    3.1. Node Interface & Anatomy
    3.2. Input Port Specification
    3.3. Output Port Specification
    3.4. Workflow Schematics (Minimal & Advanced)
4.  **Parameter Specification**
    4.1. Smart Filename Builder Parameters
    4.2. Filename Token Replacer Parameters
    4.3. Filename Counter Parameters
5.  **Applied Use-Cases & Recipes**
    5.1. Recipe 1: Organized Daily Outputs
    5.2. Recipe 2: Persistent Dataset Numbering
    5.3. Recipe 3: Dynamic Template Substitution
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
The **MD Dynamic Filename Toolkit** is a suite of three specialized nodes designed to solve the chaos of file organization in generative workflows.
* **Smart Filename Builder** constructs robust, sanitized, and informative file paths based on predefined logic (Presets) or custom assembly.
* **Filename Token Replacer** offers a flexible string substitution engine for users who prefer template-based naming (e.g., `{project}_{date}`).
* **Filename Counter** introduces a persistent, file-based auto-incrementing integer system, ensuring that sequential numbering survives server restarts and workflow reloads.

#### **1.2. Conceptual Category**
**Workflow Utility / Data Management**
These nodes operate on the "Meta" layer of the generation pipeline, handling string manipulation and state persistence rather than pixel or latent data.

#### **1.3. Problem Domain & Intended Application**
* **Problem Domain:** Standard ComfyUI save nodes often result in filenames like `ComfyUI_00001_.png`. As project complexity grows, users lose track of *which* parameters created *which* image. Furthermore, basic counters reset when the server restarts, leading to file overwrites or confusion.
* **Intended Application:**
    * **Archival:** Creating folders named by date (`2025-01-01/`) and files named by parameters (`Steps_30-CFG_7`).
    * **Dataset Creation:** Generating thousands of images with strictly sequential, padded indices (`#0001`, `#0002`).
    * **A/B Testing:** Automatically embedding prompt tags or model names into filenames for comparison.

#### **1.4. Key Features (Functional & Technical)**
* **Functional Features:**
    * **Presets:** One-click configurations for specific workflows (e.g., "Vocal" tracks vs. "Instrumental").
    * **Sanitization:** Automatically strips illegal characters (`< > : " / \ | ? *`) to prevent OS file system errors.
    * **Persistence:** The Counter node saves its state to `md_filename_counters.json` in the input directory, ensuring continuity.
* **Technical Features:**
    * **Datetime formatting:** Full support for Python's `strftime` directives (e.g., `%Y-%m-%d`).
    * **Path Structure:** Capable of generating full subdirectory paths (e.g., `Project/Section/File`) which downstream Save nodes interpret as folders.
    * **Token Logic:** Regex-based substitution engine for `{token}` style templates.

### **2. Core Concepts & Theory**

#### **2.1. Theoretical Background: Filename Hygiene**
Operating systems have strict rules regarding filenames. Beyond forbidden characters, "hygiene" involves:
1.  **Uniformity:** Using consistent separators (e.g., underscores or hyphens) to make files parseable by scripts.
2.  **Sortability:** Placing date components (`YYYY-MM-DD`) at the beginning to ensure chronological sorting.
3.  **Uniqueness:** Using seeds or incrementing counters to prevent collision.

This node suite enforces these principles via its `_sanitize_filename` and `_clean_genre` algorithms.

#### **2.2. Mathematical & Algorithmic Formulation**
**Counter Persistence:**
The counter $C$ for a given context $K$ is stored in a persistent hash map (JSON).
On execution, if Trigger ($T$) is true:
$$C_{next} = C_{current} + \max(1, I)$$
Where $I$ is the increment value. The result is formatted as a string $S$:
$$S = \text{Prefix} + \text{Pad}(C_{current}, P) + \text{Suffix}$$
Where $\text{Pad}(V, P)$ adds leading zeros to value $V$ until length $P$ is reached.

**Token Replacement:**
Given a template string $T$ and a dictionary of tokens $D$:
$$T_{final} = \text{Replace}(T, \{k\} \to v \mid \forall (k, v) \in D)$$
Post-replacement, the algorithm performs a "cleanup pass" to remove empty tokens, double separators (`--`), or trailing delimiters.

#### **2.3. Data I/O Deep Dive**
* **SmartFilenameBuilder:**
    * **Input:** Integers (Steps, Seed), Strings (Tags, Project Path).
    * **Output:** `full_path_prefix` (String) - effectively `path/to/filename_prefix`.
* **FilenameCounterNode:**
    * **Input:** `context_key` (String ID for the counter).
    * **Output:** `formatted_counter` (String), `current_value` (Int).
* **FilenameTokenReplacer:**
    * **Input:** Template String (`{date}_{project}`).
    * **Output:** Processed String.

#### **2.4. Strategic Role in the ComfyUI Graph**
* **Placement:** These nodes should be placed **near the end** of the workflow, feeding directly into `Save Image`, `Save Audio`, or `Video Combine` nodes.
* **Counter Placement:** The Counter can feed into the `SmartFilenameBuilder`'s `custom_tag` or `counter_start` input, or be used independently.

### **3. Node Architecture & Workflow**

#### **3.1. Node Interface & Anatomy**
* **SmartFilenameBuilder:** A comprehensive dashboard of toggles and text fields. The `preset` dropdown is the master control, potentially overriding other defaults.
* **Counter:** A specialized utility with inputs for defining the "scope" (`context_key`) and formatting the number.

#### **3.2. Input Port Specification**
*(See Parameter Specification for detailed breakdown)*

#### **3.3. Output Port Specification**

**SmartFilenameBuilder:**
* **`full_path_prefix` (STRING):** The complete relative path and filename prefix. Connect this to the `filename_prefix` widget of a Save node.
* **`filename_preview` (STRING):** A text block showing the directory and filename for debugging/UI display.

**FilenameTokenReplacer:**
* **`result_string` (STRING):** The fully resolved string.
* **`preview` (STRING):** Debug information showing which tokens were replaced.

**FilenameCounterNode:**
* **`formatted_counter` (STRING):** The number as a string with prefix/padding (e.g., `#0042`).
* **`current_value` (INT):** The raw integer value (useful for math nodes).
* **`info` (STRING):** Log of the action taken (Incremented, Reset, or Read-Only).

#### **3.4. Workflow Schematics**

**Minimal Functional Graph (Builder):**
`[KSampler] -> (steps/seed) -> [SmartFilenameBuilder] -> (full_path_prefix) -> [Save Image]`

**Persistent Counter Graph:**
`[FilenameCounterNode (Context: "DailyGen")] -> (current_value) -> [SmartFilenameBuilder (Counter Input)] -> [Save Image]`

### **4. Parameter Specification**

#### **4.1. Smart Filename Builder Parameters**
* **`preset` (COMBO)**
    * **Options:** `Custom`, `Instrumental`, `Vocal`, `Master`, `Raw Output`, `AB Test`.
    * **Impact:** Pre-configures boolean toggles (include steps, include seed, etc.). `Custom` allows full manual control.
* **`base_template` (STRING)**
    * **Default:** `MD_Nodes_Workflow %Y-%m-%d`
    * **Function:** The root of the filename. Supports standard Python datetime formatting (e.g., `%H-%M` for time).
* **`project_path` (STRING)**
    * **Function:** Defines the folder structure. Slashes (`/` or `\`) are interpreted as directory separators.
    * **Example:** `ClientA/ProjectB/` results in the OS creating those folders.
* **`mode_tag` (STRING)**
    * **Function:** A specific tag appended after the base template (e.g., `(Vocal)`).
* **`include_[steps|schedule|seed|genre|counter]` (BOOLEAN)**
    * **Function:** Toggles whether these specific components are appended to the filename.
* **`separator` (STRING)**
    * **Default:** ` - `
    * **Function:** The string inserted between every enabled component.

#### **4.2. Filename Token Replacer Parameters**
* **`template` (STRING)**
    * **Function:** Defines the structure using curly braces.
    * **Supported Tokens:** `{project}`, `{mode}`, `{steps}`, `{seed}`, `{genre}`, `{custom1}`, `{custom2}`, `{date}`, `{time}`, `{year}`, `{month}`, `{day}`, `{hour}`, `{minute}`, `{second}`.
* **`date_format` / `time_format` (STRING)**
    * **Function:** Defines the output format for `{date}` and `{time}` tokens using `strftime` syntax.

#### **4.3. Filename Counter Parameters**
* **`context_key` (STRING)**
    * **Function:** The unique ID for this counter.
    * **Important:** Changing this switches to a completely different number stream. Use descriptive names like `dataset_v1` or `client_x_renders`.
* **`start_value` (INT)**
    * **Function:** The fallback value if the `context_key` does not yet exist in the JSON file.
* **`increment` (INT)**
    * **Function:** How much to add per execution. Can be set to 0 to prevent incrementing (though `trigger` is preferred).
* **`padding` (INT)**
    * **Function:** Zero-padding width. `4` results in `0001`. `0` results in `1`.
* **`trigger` (BOOLEAN)**
    * **Label On:** `INCREMENT` | **Label Off:** `READ ONLY`
    * **Function:** If OFF, the node reads the current value *without* updating the JSON file. Useful for re-running a workflow without advancing the file numbering.

### **5. Applied Use-Cases & Recipes**

#### **5.1. Recipe: Organized Daily Outputs**
* **Objective:** Save images in folders by date, named by time and prompt steps.
* **Node:** `SmartFilenameBuilder`
* **Settings:**
    * `project_path`: `Daily_Renders/%Y-%m-%d/` (Note: % directives work in path too).
    * `base_template`: `%H-%M-%S`
    * `include_steps`: True
    * `separator`: `_`
* **Result:** `Output/Daily_Renders/2025-11-28/23-15-00_20S.png`

#### **5.2. Recipe: Persistent Dataset Numbering**
* **Objective:** Create a dataset where every file is guaranteed to have a unique, sequential ID, even if ComfyUI crashes.
* **Node:** `FilenameCounterNode` connected to `SmartFilenameBuilder`.
* **Counter Settings:**
    * `context_key`: `lora_dataset_v1`
    * `padding`: 5
    * `prefix`: `img_`
* **Builder Settings:**
    * `base_template`: (Empty string or generic tag)
    * `include_counter`: True (Connect Counter output to `counter_start`)
* **Result:** `img_00001`, `img_00002`, ... (persisted in JSON).

#### **5.3. Recipe: Dynamic Template Substitution**
* **Objective:** Create filenames based on external variables like "Client" and "Version".
* **Node:** `FilenameTokenReplacer`
* **Template:** `{custom1}/{year}_{month}_{custom2}_v{steps}`
* **Inputs:**
    * `custom1`: "Client_A"
    * `custom2`: "Concept_Art"
    * `steps`: 3 (Used as version number here)
* **Result:** `Client_A/2025_11_Concept_Art_v3`

### **6. Implementation Deep Dive**

#### **6.1. Source Code Walkthrough**
**SmartFilenameBuilder Logic:**
1.  **Preset Loading:** Checks `self.PRESETS`. If a known preset is selected, it overrides specific boolean flags (e.g., `include_steps`).
2.  **Date Formatting:** `datetime.now().strftime(base_template)` resolves time codes.
3.  **Sanitization:** `_sanitize_filename` removes OS-illegal characters using regex `[<>:"/\\|?*]`.
4.  **Assembly:** Components are added to a list `parts` only if their corresponding inputs are valid/true. They are joined by the `separator`.
5.  **Path Construction:** `project_path` is split and cleaned to ensure valid directory structures.

**FilenameCounterNode Logic:**
1.  **File Access:** Locates `md_filename_counters.json` in `folder_paths.get_input_directory()`.
2.  **Load:** Reads the entire JSON dict.
3.  **Context Check:** Looks for `context_key`. If missing, initializes with `start_value`.
4.  **Action:**
    * If `reset_counter` is True -> Force value to `start_value`.
    * If `trigger` is True -> `next_value = current + increment`.
5.  **Save:** Atomic write back to JSON.

#### **6.2. Dependencies & External Calls**
* **`os`, `json`, `re`**: Standard Python libraries for file/text handling.
* **`folder_paths`**: ComfyUI core module used to locate the safe Input directory for storing the counter JSON.

#### **6.3. Performance & Resource Analysis**
* **Computation:** String manipulation is negligible (microseconds).
* **IO Overhead:** The Counter node performs synchronous file I/O (Read/Write JSON) on every execution.
    * *Optimization:* The file is small (text), so latency is minimal, but massive parallel execution of this node *could* theoretically cause race conditions on the JSON file (though unlikely in standard ComfyUI linear execution).

#### **6.4. String Lifecycle Analysis**
1.  **Input:** "My : Cool : Project"
2.  **Sanitization:** Regex replaces `:` with `_` or removes it $\rightarrow$ "My_Cool_Project".
3.  **Formatting:** Joins with Separator $\rightarrow$ "My_Cool_Project - 20S".
4.  **Pathing:** Prepend Project Path $\rightarrow$ "Render/My_Cool_Project - 20S".

### **7. Troubleshooting & Diagnostics**

#### **7.1. Error Code Reference**
* **`[FilenameCounter] Save Error` / `Load Error`**:
    * **Cause:** Permissions issue in the ComfyUI `input` directory, or corrupted JSON file.
    * **Solution:** Check write permissions. Delete `md_filename_counters.json` to reset all counters.
* **`[SmartFilenameBuilder] Invalid date format`**:
    * **Cause:** Incorrect `%` syntax in `base_template`.
    * **Solution:** Refer to Python `strftime` documentation (e.g., use `%Y` not `%y` for 4-digit year).

#### **7.2. Unexpected Visual Artifacts & Mitigation**
* **Artifact:** Filename looks like `__ - 20S`.
    * **Cause:** `base_template` is empty, and sanitation converted spaces to underscores.
    * **Mitigation:** Ensure `base_template` has content, or `mode_tag` is set.
* **Artifact:** Counter not incrementing.
    * **Cause:** `trigger` toggle is set to `False` (Read Only).
    * **Mitigation:** Enable the Trigger.