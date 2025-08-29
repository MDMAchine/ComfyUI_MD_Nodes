# Comprehensive Manual: MD Seed Saver

Welcome to the complete guide for the **MD Seed Saver**, a powerful utility node for ComfyUI. This manual provides everything you need to know, from basic setup to advanced organizational techniques and technical details.

---

### **Table of Contents**

1.  **Introduction**
    * What is the MD Seed Saver?
    * Who is this Node For?
    * Key Features
2.  **Installation**
3.  **Core Concepts: File-Based Seed Management**
    * How Seeds are Stored
    * The Power of Subdirectories
4.  **Node Setup and Workflow**
    * How to Use It: A Step-by-Step Guide
5.  **Parameter Deep Dive**
    * Primary Controls: Action & Naming
    * Organization Controls: Subdirectory
    * Selection Controls: Load/Delete Dropdown
    * Core I/O
6.  **Practical Recipes & Use Cases**
    * Recipe 1: Save and Reload a "Golden Seed"
    * Recipe 2: Organize a Project with Subdirectories
    * Recipe 3: Creative Exploration with "Load Random"
7.  **Technical Details**
    * Storage Format (JSON)
    * Backward Compatibility
8.  **Troubleshooting & FAQ**

---

## 1. Introduction

### What is the MD Seed Saver?

The MD Seed Saver is a specialized utility node for ComfyUI that allows you to save, load, and manage your generation seeds using simple named files. Instead of manually copying and pasting seed numbers, you can save a seed with a memorable name (e.g., `perfect_face_seed`) and easily recall it for any workflow. It enhances reproducibility and helps organize your creative projects.

### Who is this Node For?

* **Systematic Artists:** Anyone who wants to save and document specific seeds that produce great results.
* **Workflow Organizers:** Users who manage multiple projects and want to keep seeds grouped by theme (e.g., `portraits`, `landscapes`).
* **Creative Explorers:** Artists who want to build a library of their favorite seeds to revisit or randomize for unexpected results.

### Key Features

* **Intuitive Save/Load System:** Simple actions to save the current seed or load a previously saved one.
* **Subdirectory Organization:** Group seeds into folders directly from the node to keep your library clean.
* **Random Seed Loading:** Inject creativity by loading a random seed from your collection.
* **Load Latest Seed:** Instantly get back to your last saved seed.
* **Automatic Naming:** If you don't provide a name, a unique one is generated for you.
* **Future-Proof JSON Storage:** Seeds are saved with metadata, with full backward compatibility for older `.txt` seed files.

---

## 2. Installation

As a custom node, the MD Seed Saver needs to be placed in the correct ComfyUI directory.

1.  Navigate to your ComfyUI installation folder.
2.  Open the `ComfyUI/custom_nodes/` directory.
3.  Save the `seed_saver_node.py` file inside this directory.
4.  Restart ComfyUI. The "MD Seed Saver" will now be available under `Add Node` > `MD_Nodes` > `Utility`.

---

## 3. Core Concepts: File-Based Seed Management

### How Seeds are Stored

The node creates a `seeds` folder inside your main `ComfyUI/output/` directory. Every time you save a seed, it creates a small `.json` file in this folder. For example, saving a seed named `glowing_forest` creates a file called `glowing_forest.json`. This simple file-based approach makes your seeds easy to back up and manage.

### The Power of Subdirectories

To avoid a cluttered dropdown menu with hundreds of seeds, you can use the `subdirectory` input. Typing `portraits` in this field will make the node save and load from `ComfyUI/output/seeds/portraits/`. This allows you to create project-specific seed libraries that are isolated from each other.

---

## 4. Node Setup and Workflow

### How to Use It: A Step-by-Step Guide

1.  **Place the Node:** Add the "MD Seed Saver" node to your workflow.
2.  **Connect the Input Seed:** Connect the output of your seed generator (e.g., a Primitive node) to the `seed_input` of the Seed Saver.
3.  **Connect the Output Seed:** Connect the `seed_output` of the Seed Saver to the `seed` input of your KSampler.
4.  **Choose an Action:** Select what you want to do from the `action` dropdown (e.g., `SAVE_CURRENT_SEED`).
5.  **Provide Names/Folders (If Needed):** Fill in the `seed_name_input` or `subdirectory` fields depending on the action.
6.  **Queue Prompt:** Run your workflow. The node will perform the action and provide a summary in the `status_info` output.

---

## 5. Parameter Deep Dive

### Primary Controls: Action & Naming

* **`action`** (`ENUM`): Your main control.
    * `SAVE_CURRENT_SEED`: Saves the number from `seed_input` using the name from `seed_name_input`.
    * `LOAD_SELECTED_SEED`: Loads the seed chosen in the `seed_to_load_name` dropdown.
    * `DELETE_SELECTED_SEED`: Deletes the seed chosen in the `seed_to_load_name` dropdown.
    * `LOAD_LATEST_SAVED_SEED`: Finds and loads the most recently saved seed in the specified `subdirectory`.
    * `LOAD_RANDOM_SEED`: Loads a random seed from the specified `subdirectory`.
* **`seed_name_input`** (`STRING`): The filename for the seed you are saving.

### Organization Controls: Subdirectory

* **`subdirectory`** (`STRING`, Optional): The folder inside `ComfyUI/output/seeds/` to use. If left blank, it uses the root `seeds` folder.

### Selection Controls: Load/Delete Dropdown

* **`seed_to_load_name`** (`ENUM`): A dropdown list of all seeds found in the **root `seeds` folder only**. This is used for the `LOAD_SELECTED_SEED` and `DELETE_SELECTED_SEED` actions. **Note: This list does not see inside subdirectories.**

### Core I/O

* **`seed_input`** (`INT`, Required): The incoming seed to be saved or passed through.
* **`seed_output`** (`INT`): The final seed passed to the KSampler.
* **`status_info`** (`STRING`): A text box with a summary of the node's actions.

---

## 6. Practical Recipes & Use Cases

### Recipe 1: Save and Reload a "Golden Seed"

Goal: You found a perfect seed and want to use it again later.

1.  With the perfect image generated, set `action` to `SAVE_CURRENT_SEED`.
2.  Type a name like `golden_seed_v1` into `seed_name_input`.
3.  Queue Prompt. The seed is now saved.
4.  Later, set `action` to `LOAD_SELECTED_SEED` and pick `golden_seed_v1` from the dropdown to get the exact same seed.

### Recipe 2: Organize a Project with Subdirectories

Goal: Keep all seeds for a character project separate from your landscape seeds.

1.  In the `subdirectory` input, type `my_character`.
2.  When you find a good seed for your character, save it using `SAVE_CURRENT_SEED`. It will be saved in the `seeds/my_character/` folder.
3.  To get the latest seed from this project, set `action` to `LOAD_LATEST_SAVED_SEED` while `my_character` is still in the subdirectory field.

### Recipe 3: Creative Exploration with "Load Random"

Goal: Break a creative block by using one of your previously successful seeds.

1.  Set `action` to `LOAD_RANDOM_SEED`.
2.  If you want to pull from a specific project, enter its name in the `subdirectory` field. Otherwise, leave it blank to pull from your main seeds.
3.  Queue Prompt. The node will pick one of your saved seeds at random and use it for the generation.

---

## 7. Technical Details

### Storage Format (JSON)

When you save a seed, the node creates a `.json` file containing the seed value and the timestamp it was saved.

**Example `my_seed.json`:**
```json
{
    "seed": 1234567890123456,
    "saved_at": "2023-10-27T10:30:00.123456"
}
```

### Backward Compatibility

The node can still read old seeds you may have saved as simple `.txt` files, ensuring a smooth upgrade. When deleting, it will attempt to remove both `.json` and `.txt` versions of a file if they exist.

---

## 8. Troubleshooting & FAQ

* **"I saved a new seed, but it's not in the `seed_to_load_name` dropdown."**
    * You must **refresh your browser (F5)**. The dropdown list is only created when the page loads.
* **"The dropdown is empty, but I saved seeds in a subdirectory."**
    * This is expected behavior. The `seed_to_load_name` dropdown **only** shows seeds from the main `seeds` folder, not from subdirectories. Use actions like `LOAD_LATEST_SAVED_SEED` to work with seeds in subdirectories.
* **"I deleted a seed, but it's still in the list."**
    * As with saving, you must **refresh your browser (F5)** to update the list after deleting a file.
