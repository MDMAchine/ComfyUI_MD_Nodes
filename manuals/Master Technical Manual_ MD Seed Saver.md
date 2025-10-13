---

## **Master Technical Manual: MD Seed Saver**

### **Node Name: SeedSaverNode**

Display Name: MD Seed Saver  
Category: MD\_Nodes/Utility  
Version: 1.0 (based on code features)  
Last Updated: 2025-09-15

---

### **Table of Contents**

1. Introduction  
   1.1. Executive Summary  
   1.2. Conceptual Category  
   1.3. Problem Domain & Intended Application  
   1.4. Key Features (Functional & Technical)  
2. Core Concepts & Theory  
   2.1. Theoretical Background: Seed Management  
   2.2. File-Based Storage Architecture  
   2.3. Data I/O Deep Dive  
   2.4. Strategic Role in the ComfyUI Graph  
3. Node Architecture & Workflow  
   3.1. Node Interface & Anatomy  
   3.2. Input Port Specification  
   3.3. Output Port Specification  
   3.4. Workflow Schematics (Minimal & Advanced)  
4. Parameter Specification  
   4.1. Parameter: seed\_input  
   4.2. Parameter: action  
   4.3. Parameter: seed\_name\_input  
   4.4. Parameter: seed\_to\_load\_name  
   4.5. Parameter: subdirectory  
5. Applied Use-Cases & Recipes  
   5.1. Recipe 1: Saving and Reloading a Foundational "Golden Seed"  
   5.2. Recipe 2: Organizing a Project with Character-Specific Seeds  
   5.3. Recipe 3: Creative Ideation Using Randomized Seeds  
6. Implementation Deep Dive  
   6.1. Source Code Walkthrough  
   6.2. Dependencies & External Calls  
   6.3. Performance & Resource Analysis  
   6.4. Data Lifecycle Analysis  
7. Troubleshooting & Diagnostics  
   7.1. Error Code Reference  
   7.2. Unexpected Behavior & Limitations

---

### **1\. Introduction**

#### **1.1. Executive Summary**

The **MD Seed Saver** is a workflow utility node for ComfyUI designed to externalize and manage generation seeds through a persistent, file-based system. It addresses the ephemeral nature of seed values by providing a robust framework to save, name, load, and organize seeds as discrete files on disk. The node allows users to capture and recall specific seeds that produce desirable results, thereby enhancing reproducibility and establishing an organized asset library for creative projects. It offers multiple modes of operation, including direct saving, loading from a list, loading the most recent or a random seed, and organizational capabilities via a subdirectory structure. The node acts as an intelligent switch and storage interface for the integer seed value that is fundamental to the diffusion sampling process.

#### **1.2. Conceptual Category**

**Workflow Utility / State Management.** This node does not perform any image or audio processing. Its sole function is to manage a critical piece of workflow state—the seed—by reading from and writing to the local filesystem. It acts as an intermediary for the seed value, either passing it through, replacing it with a stored value, or capturing it for storage.

#### **1.3. Problem Domain & Intended Application**

* **Problem Domain:** In standard ComfyUI workflows, the seed is either a fixed integer manually entered by the user or a randomized value that is lost after the generation is complete. There is no built-in mechanism for saving a particularly "good" random seed with a memorable name for later use. Users must manually copy the seed number from the output and paste it into a primitive node, a process that is cumbersome, error-prone, and lacks organization. This node solves the problem of persistent, named seed management.  
* **Intended Application (Use-Cases):**  
  * **Reproducibility:** Saving a seed that produced a specific desired outcome (e.g., a perfect character face, a unique composition) with a descriptive name for guaranteed reproduction in future sessions.  
  * **Project Organization:** Using the subdirectory feature to create isolated libraries of seeds for different projects, characters, or artistic styles, preventing cross-contamination and simplifying asset management.  
  * **Creative Exploration:** Building a curated library of "golden seeds" and then using the LOAD\_RANDOM\_SEED action to explore novel combinations of a successful seed with new prompts or models.  
  * **Workflow Simplification:** Using the LOAD\_LATEST\_SAVED\_SEED action as a quick "undo" or "revisit" function for the last seed that was explicitly saved.  
* **Non-Application (Anti-Use-Cases):**  
  * The node is not a seed generator. It requires an incoming seed value from another node (like a Seed (any) primitive or another node's seed output).  
  * It is not a comprehensive database. It is a simple file-based system; for managing tens of thousands of seeds with complex metadata, a more advanced external system would be required.

#### **1.4. Key Features (Functional & Technical)**

* **Functional Features:**  
  * **Multi-Action Control:** A single dropdown (action) provides five distinct operations: Save, Load Selected, Delete Selected, Load Latest, and Load Random.  
  * **Named Seed Storage:** Allows users to save seeds with human-readable names (e.g., perfect\_sunrise\_v1), abstracting away the raw 64-bit integer.  
  * **Subdirectory Organization:** A subdirectory input enables the creation of a hierarchical folder structure within the main seeds directory for project-based organization.  
  * **Automatic Naming:** If a name is not provided during a save operation, a unique name is automatically generated using the seed value and a timestamp.  
  * **UI Feedback:** Provides a status\_info text output that summarizes the action performed, the seed name, and the final output value.  
* **Technical Features:**  
  * **Persistent File-Based Storage:** Creates and manages files within the ComfyUI/output/seeds/ directory, making the seed library tangible, portable, and easy to back up.  
  * **JSON Data Format:** Saves seeds in a structured JSON format ({"seed": value, "saved\_at": timestamp}), which is robust and extensible for potential future features.  
  * **Backward Compatibility:** The load\_seed\_from\_file function is designed to first check for a .json file, and if not found, to fall back to reading legacy .txt files, ensuring a seamless user experience with older saved seeds.  
  * **Dynamic Dropdown Population:** The seed\_to\_load\_name dropdown is dynamically populated on UI load by scanning the root seeds directory for all valid .json and .txt files.

### **2\. Core Concepts & Theory**

#### **2.1. Theoretical Background: Seed Management**

In generative AI, the **seed** is an integer that initializes the pseudo-random number generator (PRNG). For a given model, prompt, and set of parameters, the same seed will always produce the exact same initial noise tensor, which in turn leads to the exact same final output. This deterministic property is the foundation of reproducibility.

Effective seed management is the practice of identifying, storing, and retrieving seeds that have proven to be aesthetically or compositionally successful. This transforms the seed from a simple random number into a reusable creative asset. This node provides the tooling to implement a seed management strategy directly within the workflow environment.

#### **2.2. File-Based Storage Architecture**

The node's architecture is intentionally simple and transparent. It leverages the local filesystem as a key-value store, where the **filename is the key** (the seed\_name) and the **file content is the value** (the seed integer and its metadata).

* **Root Directory:** A primary storage location is established at ComfyUI/output/seeds/.  
* **Subdirectories:** The subdirectory parameter allows for the creation of an additional layer of organization within this root folder. This mirrors common file management paradigms and is immediately intuitive.  
* **Data Format:** The use of JSON (.json) provides a structured and human-readable format. The inclusion of a "saved\_at" timestamp, while not currently used for sorting in LOAD\_LATEST\_SAVED\_SEED (which uses file modification time), provides valuable metadata and future-proofs the format. The system's ability to fall back to plain text (.txt) files demonstrates robust design for backward compatibility.

#### **2.3. Data I/O Deep Dive**

* **Inputs:**  
  * seed\_input (INT):  
    * **Data Specification:** A 64-bit integer (0 to 0xffffffffffffffff). This is the raw data that the node operates on.  
* **Outputs:**  
  * seed\_output (INT):  
    * **Data Specification:** A 64-bit integer. This is the final value that will be passed to the sampler. Depending on the action, this may be the original seed\_input or a value loaded from a file.  
  * status\_info (STRING):  
    * **Data Specification:** A standard Python string. This string is dynamically constructed during execution to provide a human-readable summary of the node's operations, including any actions taken, warnings, or errors encountered.

#### **2.4. Strategic Role in the ComfyUI Graph**

* **Placement Context:** This node is a **passthrough utility** that should be placed directly between a seed source and a seed consumer. The canonical workflow is Seed Generator \-\> MD Seed Saver \-\> KSampler. It acts as an interception point on the "seed wire," allowing the user to either observe and save the seed passing through it or to break the connection and inject a new seed from its library.  
* **Synergistic Nodes:**  
  * **Seed (any) or Integer Primitive:** The most common source for the seed\_input.  
  * **KSampler / SamplerCustom:** The primary consumers of the seed\_output.  
  * **Reroute:** Can be used to cleanly wire the seed\_output to multiple samplers to ensure they all use the same managed seed.  
* **Conflicting Nodes:** There are no direct conflicts, but incorrect wiring can lead to logical errors. Placing this node *after* the sampler has no effect. Connecting its status\_info output to a numerical input will cause a type mismatch error.

### **3\. Node Architecture & Workflow**

#### **3.1. Node Interface & Anatomy**

1. seed\_input (Input Port)  
2. action (Dropdown Widget)  
3. seed\_name\_input (String Widget)  
4. seed\_to\_load\_name (Dropdown Widget)  
5. subdirectory (String Widget)  
6. seed\_output (Output Port)  
7. status\_info (Output Port)

#### **3.2. Input Port Specification**

* **seed\_input (INT)** \- (Anatomy Ref: \#1)  
  * **Description:** This port receives the integer seed value from an upstream node. This is the value that will be saved when the SAVE\_CURRENT\_SEED action is used. For all other actions, this value serves as the default or fallback output if the selected action fails (e.g., if a file cannot be found).  
  * **Required/Optional:** Required. The node is designed to have a forced input to ensure a seed value is always present.

#### **3.3. Output Port Specification**

* **seed\_output (INT)** \- (Anatomy Ref: \#6)  
  * **Description:** This port outputs the final integer seed value. If a "LOAD" action is successful, it will be the value read from the file. If any other action is selected or if a "LOAD" action fails, it will be a direct passthrough of the seed\_input.  
  * **Resulting Data:** A 64-bit integer.  
* **status\_info (STRING)** \- (Anatomy Ref: \#7)  
  * **Description:** This port outputs a multiline string that provides a summary of the node's execution for the last run. It details which action was performed, the names of any files affected, the seed values involved, and any warnings or errors that occurred.  
  * **Resulting Data:** A human-readable Python string.

#### **3.4. Workflow Schematics**

* Minimal Functional Graph (Save Operation):  
  A Seed (any) node with control\_after\_generate set to fixed is connected to seed\_input. The action is set to SAVE\_CURRENT\_SEED. The seed\_name\_input is filled. The seed\_output is connected to a KSampler's seed input. When run, the fixed seed is passed through to the sampler and simultaneously saved to a named file.  
* Advanced Organization Graph (Load Operation):  
  A Seed (any) node is connected to seed\_input as a fallback. The subdirectory is set to a project name, e.g., "landscapes". The action is set to LOAD\_LATEST\_SAVED\_SEED. The seed\_output is connected to a KSampler. When run, the node ignores the input seed, finds the most recently modified seed file in the seeds/landscapes/ directory, loads its value, and outputs it to the sampler.

### **4\. Parameter Specification**

#### **4.1. Parameter: seed\_input**

* **UI Label:** seed\_input (Input Port)  
* **Internal Variable Name:** seed\_input  
* **Data Type & Constraints:** INT, min: 0, max: 0xffffffffffffffff. The forceInput flag is set to True, meaning this port must have a connection.  
* **Algorithmic Impact:** This is the primary data source for the SAVE\_CURRENT\_SEED action, where its value is written to a file. For all other actions, it serves as the default passthrough value. The execute method initializes output\_seed \= seed\_input, and this value is only ever overwritten if a LOAD action is successful. This ensures that the node always outputs a valid seed, preventing workflow failures.  
* **Default Value & Rationale:** No default value in the widget itself, as it is a forced input. A value must be provided by an upstream node.

#### **4.2. Parameter: action**

* **UI Label:** action  
* **Internal Variable Name:** action  
* **Data Type & Constraints:** COMBO (Dropdown list of strings).  
* **Algorithmic Impact:** This parameter is the main control switch for the node's execute method. Its value is evaluated in a large if/elif block that routes the program flow to the appropriate logic (saving, loading, deleting, etc.). The entire behavior of the node is dictated by the selection made here. An action of (None) or an unrecognized string will cause the node to perform no file operations and act as a simple passthrough.  
* **Default Value & Rationale:** (None). The default action performs no file operations, making the node inert and safe by default. This prevents accidental saving or loading when the node is first added to a workflow.

#### **4.3. Parameter: seed\_name\_input**

* **UI Label:** seed\_name\_input  
* **Internal Variable Name:** seed\_name\_input  
* **Data Type & Constraints:** STRING.  
* **Algorithmic Impact:** This parameter is active only when action is SAVE\_CURRENT\_SEED. Its string value is used as the filename (without extension) for the saved seed file. The code first calls .strip() to remove leading/trailing whitespace. If the resulting string is empty, the node enters an automatic naming mode, creating a name from the seed value and a precise timestamp (e.g., seed\_12345\_20250915235200). This ensures every saved seed has a unique filename.  
* **Interaction with Other Parameters:** This parameter is only used for the SAVE\_CURRENT\_SEED action. It is ignored for all other actions.  
* **Default Value & Rationale:** "" (empty string). This default encourages the user to provide a meaningful name but gracefully falls back to the robust auto-naming feature if they do not.

#### **4.4. Parameter: seed\_to\_load\_name**

* **UI Label:** seed\_to\_load\_name  
* **Internal Variable Name:** seed\_to\_load\_name  
* **Data Type & Constraints:** COMBO (Dropdown list of strings).  
* **Algorithmic Impact:** This dropdown is used for the LOAD\_SELECTED\_SEED and DELETE\_SELECTED\_SEED actions. Its list of options is dynamically generated when the ComfyUI frontend loads, by calling the get\_all\_saved\_seed\_names function with an empty subdirectory. **Crucially, this means the dropdown only ever shows seeds from the root seeds/ directory.** The selected string from this list is passed to the load\_seed\_from\_file or delete\_seed\_file function. A selection of (None) causes the action to be skipped.  
* **Interaction with Other Parameters:** Although the subdirectory widget is visible, the code explicitly calls the file operations for these two actions with subdirectory="", meaning the subdirectory parameter is **ignored** for LOAD\_SELECTED\_SEED and DELETE\_SELECTED\_SEED.  
* **Default Value & Rationale:** (None). The default selection is a no-op, preventing accidental loading or deletion.

#### **4.5. Parameter: subdirectory**

* **UI Label:** subdirectory  
* **Internal Variable Name:** subdirectory  
* **Data Type & Constraints:** STRING.  
* **Algorithmic Impact:** This parameter provides a relative path that is appended to the base OUTPUT\_SEEDS\_DIR. For example, a value of project\_alpha/characters will cause the node to operate within ComfyUI/output/seeds/project\_alpha/characters/. The ensure\_output\_directory\_exists function is called to create this path if it doesn't exist. This parameter is respected by the SAVE\_CURRENT\_SEED, LOAD\_LATEST\_SAVED\_SEED, and LOAD\_RANDOM\_SEED actions.  
* **Interaction with Other Parameters:** As noted above, this parameter is **ignored** for the LOAD\_SELECTED\_SEED and DELETE\_SELECTED\_SEED actions, which are hard-coded to operate on the root directory to match the dropdown's contents.  
* **Default Value & Rationale:** "" (empty string). The node operates on the root seeds directory by default, providing the simplest experience for new users.

### **5\. Applied Use-Cases & Recipes**

#### **5.1. Recipe 1: Saving and Reloading a Foundational "Golden Seed"**

* **Objective:** You have generated an image with a specific seed that has an excellent composition or character, and you want to save it for future use as a reliable starting point.  
* **Rationale:** This is the primary use case. By saving the seed with a memorable name, you decouple the creative asset (the seed's result) from the transient, numerical value. This allows you to build a library of reliable starting points. Reloading is done via the dropdown, which requires a browser refresh to see newly saved seeds.  
* **Workflow Steps:**  
  1. **Save:** With the desired seed passing through seed\_input, set action to SAVE\_CURRENT\_SEED. Set seed\_name\_input to "golden\_seed\_v1". Queue prompt.  
  2. **Refresh:** Press F5 to refresh the ComfyUI web interface.  
  3. **Load:** Set action to LOAD\_SELECTED\_SEED. The seed\_to\_load\_name dropdown will now contain "golden\_seed\_v1". Select it. Queue prompt. The seed\_output will now be the saved value.

#### **5.2. Recipe 2: Organizing a Project with Character-Specific Seeds**

* **Objective:** You are working on a project with a specific character, "cyborg\_ninja," and want to keep all successful seeds for this character separate from your other projects.  
* **Rationale:** The subdirectory feature creates an isolated namespace for your seeds. This prevents the main dropdown from becoming cluttered and allows project-specific actions like LOAD\_LATEST\_SAVED\_SEED to function correctly without being influenced by seeds from other projects.  
* **Workflow Steps:**  
  1. In the subdirectory field, enter "cyborg\_ninja".  
  2. Use SAVE\_CURRENT\_SEED with names like "cyborg\_ninja\_profile" and "cyborg\_ninja\_action\_pose" to save seeds into the seeds/cyborg\_ninja/ folder.  
  3. To get the most recently saved character seed, set action to LOAD\_LATEST\_SAVED\_SEED while the subdirectory is still set to "cyborg\_ninja".

#### **5.3. Recipe 3: Creative Ideation Using Randomized Seeds**

* **Objective:** You have a well-developed prompt but want to explore how it looks with a variety of previously successful seeds to generate unexpected but high-quality variations.  
* **Rationale:** The LOAD\_RANDOM\_SEED action leverages your existing library of saved seeds as a source of curated randomness. Instead of pulling from the entire space of 64-bit integers, it pulls from a much smaller set of seeds that you have already vetted as being "good." This can be a powerful tool for breaking creative blocks.  
* **Workflow Steps:**  
  1. Set action to LOAD\_RANDOM\_SEED.  
  2. (Optional) Specify a subdirectory to limit the random selection to a specific project.  
  3. Queue the prompt multiple times. Each time, the node will select a different seed from the specified directory and output it to the sampler, producing a new variation.

### **6\. Implementation Deep Dive**

#### **6.1. Source Code Walkthrough**

The node's logic is primarily contained within the execute method, supported by several file operation helpers.

1. **Initialization:** The method starts by setting output\_seed \= seed\_input, establishing the default passthrough behavior. It then calls get\_all\_saved\_seed\_names(subdirectory) to get a list of existing seeds for context and begins building the status\_message string.  
2. **Action Routing:** A large if/elif block checks the action string.  
   * **SAVE\_CURRENT\_SEED:** It strips whitespace from seed\_name\_input. If the name is empty, it generates a timestamped name. It checks if the name already exists and adds a warning to the status. Finally, it calls the save\_seed\_to\_file helper.  
   * **LOAD\_SELECTED\_SEED:** It checks that seed\_to\_load\_name is not the default (None). It then calls load\_seed\_from\_file, crucially passing subdirectory="" to force it to look in the root directory, matching the dropdown's source. If the load is successful, output\_seed is updated.  
   * **DELETE\_SELECTED\_SEED:** Similar to loading, it checks for a valid selection and then calls delete\_seed\_file, also hard-coded to the root directory.  
   * **LOAD\_LATEST\_SAVED\_SEED:** This is more complex. It lists all files in the target subdirectory, gets the modification time for each using os.path.getmtime, sorts the list of files by this time, and then takes the first item. The name of this file is then passed to load\_seed\_from\_file to get the seed value.  
   * **LOAD\_RANDOM\_SEED:** It gets the list of seeds in the subdirectory and uses Python's built-in random.choice to select one name, which is then passed to load\_seed\_from\_file.  
3. **Finalization:** The status\_message is appended with the final output\_seed value. The method then returns a tuple containing the final output\_seed integer and the status\_message string.

#### **6.2. Dependencies & External Calls**

* **Standard Libraries:** The node relies entirely on Python's built-in libraries, making it highly portable and dependency-free.  
  * **os:** Used extensively for all filesystem operations: os.path.join for constructing paths, os.makedirs for creating directories, os.listdir for getting contents, os.path.splitext for parsing filenames, os.remove for deletion, and os.path.getmtime for finding the latest file.  
  * **json:** Used for serializing (json.dump) and deserializing (json.load) the seed data into the .json file format.  
  * **datetime:** Used to generate timestamps for metadata (datetime.now().isoformat()) and for the automatic naming of seeds.  
  * **random:** The random.choice function is used for the LOAD\_RANDOM\_SEED action.  
* **ComfyUI Libraries:**  
  * **folder\_paths:** folder\_paths.get\_output\_directory() is used to get the base path for the seeds directory, ensuring the node correctly integrates with the ComfyUI environment.

#### **6.3. Performance & Resource Analysis**

* **Execution Target:** All operations are **CPU-bound** and **I/O-bound**.  
* **VRAM Usage:** Zero. This node does not interact with the GPU or handle any large tensors.  
* **Bottlenecks:** Performance is not a concern for this node. All operations are filesystem lookups on a small number of files. Even with thousands of saved seeds, the I/O operations (like os.listdir) are effectively instantaneous and will never be the bottleneck in a generation workflow. The LOAD\_LATEST\_SAVED\_SEED action is technically the most "expensive" as it has to stat every file in a directory, but this is still measured in microseconds or milliseconds.

#### **6.4. Data Lifecycle Analysis**

The node manages simple integer and string data.

1. **Input:** An integer seed\_input is received.  
2. **Processing:**  
   * **Save:** The seed\_input integer is packaged into a Python dictionary, serialized into a JSON string by the json library, and written to disk as a file.  
   * **Load:** A file is read from disk, its JSON content is parsed into a Python dictionary, the "seed" value is extracted, and it is cast back to an integer.  
3. **Output:** The final integer output\_seed is returned. Its lifecycle continues until it is consumed by the downstream sampler. The status\_info string is created and destroyed within the scope of the execute method call.

### **7\. Troubleshooting & Diagnostics**

#### **7.1. Error Code Reference**

* **Error Message/Traceback Snippet:** (Console) ERROR: Could not save seed 'X' to file: \[Errno Y\] ...  
  * **Root Cause Analysis:** A filesystem permission error. The user account running ComfyUI does not have write permissions for the ComfyUI/output/seeds/ directory or the specific subdirectory.  
  * **Primary Solution:** Check the folder permissions for your ComfyUI/output/ directory and ensure they are read/write accessible by the current user.  
* **Error Message/Traceback Snippet:** (Console) ERROR: Could not load JSON seed 'X': json.decoder.JSONDecodeError...  
  * **Root Cause Analysis:** A saved .json seed file has become corrupted and is no longer valid JSON. This could happen if it was edited manually and saved incorrectly.  
  * **Primary Solution:** Navigate to the seeds folder and either delete the corrupted file or fix its formatting. A valid file should look like: {"seed": 12345, "saved\_at": "..."}.

#### **7.2. Unexpected Behavior & Limitations**

* **Issue:** After saving a new seed, it does not appear in the seed\_to\_load\_name dropdown menu.  
  * **Cause:** This is a known limitation of how ComfyUI populates dropdowns. The INPUT\_TYPES method, which scans the directory to create the list, is only called when the graph is first loaded or the browser page is refreshed.  
  * **Solution:** You must perform a hard refresh of the ComfyUI web page (typically by pressing F5 or Ctrl+F5) to force the node to re-scan the directory and update the list. The node's status message includes a reminder for this.  
* **Issue:** Seeds saved in a subdirectory are not visible in the seed\_to\_load\_name dropdown.  
  * **Cause:** This is an explicit design choice in the code. The dropdown is hard-coded to only scan the root seeds directory (subdirectory="") to keep the list manageable and predictable. It is not a dynamic list that updates when you type in the subdirectory field.  
  * **Solution:** This is the intended behavior. To access seeds within a subdirectory, you must use the actions that operate on the subdirectory field, which are SAVE\_CURRENT\_SEED, LOAD\_LATEST\_SAVED\_SEED, and LOAD\_RANDOM\_SEED. The dropdown is only for quick access to globally saved seeds.  
* **Issue:** The LOAD\_LATEST\_SAVED\_SEED action loads an older seed than expected.  
  * **Cause:** This action relies on the filesystem's "last modified" timestamp (os.path.getmtime). While saving a seed updates this time, manually editing or copying a file can also change it, potentially leading to an unexpected file being identified as the "latest."  
  * **Solution:** Avoid manually modifying files in the seeds directory if you rely on the LOAD\_LATEST\_SAVED\_SEED functionality. If you need to edit a seed, it's better to load, modify, and re-save it through the node to ensure the timestamp is updated correctly.