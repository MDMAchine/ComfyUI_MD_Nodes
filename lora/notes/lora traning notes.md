# ACE-Step LoRA Training Workflow (Windows)

This guide outlines the steps to train a LoRA model for ACE-Step on a Windows system, including environment setup, data preparation, and training.  
**Repository:** https://github.com/woct0rdho/ACE-Step

---

## 1. Environment Setup

It's highly recommended to use a Python virtual environment to manage your dependencies. Set up and activate your virtual environment before proceeding.

### 🔹 Virtual Environment Setup (Prerequisite)

Navigate to your working directory, e.g., `D:\Ace-Step_Fork\`, and create a virtual environment:

```bash
python -m venv ACE-Step
```

Activate the environment:

```bash
D:\Ace-Step_Fork\ACE-Step\Scripts\activate.bat
```

(Your prompt should now show: `(ACE-Step)`)

---

### 🔹 Installation Steps

Navigate to your ACE-Step directory:

```bash
cd D:\Ace-Step_Fork\ACE-Step
```

Activate your environment again:

```bash
.\Scripts\activate.bat
```

Install PyTorch with CUDA 12.x support:

```bash
pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu128
```

> ⚠️ If you don’t have a compatible NVIDIA GPU, replace `cu128` with `cpu`.

Upgrade essential tools:

```bash
pip install --upgrade setuptools wheel
```

Install `gptqmodel`:

```bash
pip install gptqmodel==2.2.0 --no-build-isolation
```

Update `transformers` from Hugging Face’s main branch:

```bash
pip uninstall transformers
pip install git+https://github.com/huggingface/transformers.git
```

Install other project dependencies:

```bash
pip install -r requirements.txt
```

---

### 🔹 Triton and Flash-Attention Installation

If `triton-windows` is working:

```bash
pip install triton-windows
pip install flash-attn --no-build-isolation
```

If it fails, try from source:

```bash
pip install packaging ninja
pip install flash-attn --no-build-isolation --no-binary :all: flash-attn
```

If it still fails:

1. Open `generate_prompts_lyrics.py`
2. Locate `load_model` function
3. Change:
   ```python
   attn_implementation="flash_attention_2"
   ```
   to:
   ```python
   attn_implementation="eager"
   ```

---

## 2. Data Preparation

### 🔹 Organize Your Music Data

Place your `.wav`, `.mp3`, etc., in:

```
C:\Users\Admin\Desktop\MDMAchine\normalized
```

### 🔹 Generate Prompts and Lyrics

```bash
python generate_prompts_lyrics.py --lyrics --data_dir "C:\Users\Admin\Desktop\MDMAchine\normalized"
```

---

### 🔹 Add Custom Tags to Prompts (Optional)

Save this script as `add_tags.py` inside `D:\Ace-Step_Fork\ACE-Step\`:

<details>
<summary>▶ Click to view script</summary>

```python
import os
import argparse

def add_tags_to_prompts(data_directory: str, tags_to_add: list[str]):
    print(f"\n--- Starting Tag Addition ---")
    print(f"Processing directory: '{data_directory}'")
    print(f"Tags to append: {tags_to_add}")
    total_files_updated = 0
    total_tags_appended = 0

    if not os.path.exists(data_directory):
        print(f"Error: The specified data directory '{data_directory}' does not exist. Please check the path.")
        return

    for root, _, files in os.walk(data_directory):
        for file_name in files:
            if file_name.endswith("_prompt.txt"):
                file_path = os.path.join(root, file_name)
                tags_added_to_current_file = 0

                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        current_content = f.read().strip()

                    existing_tags_set = set(tag.strip().lower() for tag in current_content.split(',') if tag.strip())
                    new_content = current_content
                    for tag in tags_to_add:
                        clean_tag = tag.strip()
                        if clean_tag and clean_tag.lower() not in existing_tags_set:
                            new_content = f"{new_content}, {clean_tag}" if new_content else clean_tag
                            existing_tags_set.add(clean_tag.lower())
                            tags_added_to_current_file += 1
                            total_tags_appended += 1

                    if tags_added_to_current_file > 0:
                        with open(file_path, 'w', encoding='utf-8', newline='\n') as f:
                            f.write(new_content)
                        print(f"  Updated: '{file_path}' (added {tags_added_to_current_file} new tags)")
                        total_files_updated += 1
                    else:
                        print(f"  Skipped: '{file_path}' (all tags already present or no valid tags to add)")

                except Exception as e:
                    print(f"  Error processing '{file_path}': {e}")

    print(f"\n--- Tag Addition Summary ---")
    print(f"Total files where tags were appended: {total_files_updated}")
    print(f"Total individual new tags added across all files: {total_tags_appended}")
    if total_files_updated == 0 and total_tags_appended == 0 and os.path.exists(data_directory) and not any(f.endswith("_prompt.txt") for r, d, f_list in os.walk(data_directory) for f in f_list):
        print("Note: No '_prompt.txt' files were found in the specified directory.")
    print(f"---------------------------\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add tags to all _prompt.txt files in a directory.")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to your _prompt.txt files.")
    parser.add_argument("--tags", type=str, required=True, help="Comma-separated list of tags to add.")
    args = parser.parse_args()
    tags_list = [tag.strip() for tag in args.tags.split(',') if tag.strip()]
    if not tags_list:
        print("Error: No valid tags provided.")
    else:
        add_tags_to_prompts(args.data_dir, tags_list)
```

</details>

Run the script:

```bash
python add_tags.py --data_dir "C:\Users\Admin\Desktop\MDMAchine\normalized" --tags "mdmachine"
```

---

### 🔹 Create Processed Dataset

Create the processed data folder:

```bash
mkdir D:\Ace-Step_Fork\processed_data
```

Convert filenames into a HuggingFace dataset:

```bash
python convert2hf_dataset_new.py --data_dir "C:\Users\Admin\Desktop\MDMAchine\normalized" --output_name "D:\Ace-Step_Fork\processed_data\mdmachine_filenames"
```

Preprocess audio features:

```bash
python preprocess_dataset_new.py --input_name "D:\Ace-Step_Fork\processed_data\mdmachine_filenames" --output_dir "D:\Ace-Step_Fork\processed_data\mdmachine_prep"
```

---

## 3. LoRA Training

### 🔹 Clear Existing Checkpoints (optional)

Delete contents of:

```
D:\Ace-Step_Fork\ACE-Step\checkpoints
```

### 🔹 Start Training

Run with default steps (2000):

```bash
python trainer_new.py --dataset_path "D:\Ace-Step_Fork\processed_data\mdmachine_prep"
```

Or with custom steps:

```bash
python trainer_new.py --dataset_path "D:\Ace-Step_Fork\processed_data\mdmachine_prep" --max_steps 5000
```

> ✅ If prompted by Weights & Biases (wandb), choose `3` to disable or set `WANDB_MODE=offline` as an environment variable.

---

## 4. Finalizing Your LoRA Model

### 🔹 Locate Final Checkpoint

Navigate to:

```
D:\Ace-Step_Fork\ACE-Step\checkpoints
```

Find the folder with the **highest epoch/step** (e.g., `epoch=36-step=5000_lora`).

### 🔹 Add LoRA Alpha Value

```bash
python add_alpha_in_lora.py --input_name checkpoints/epoch=36-step=5000_lora/pytorch_lora_weights.safetensors --output_name out.safetensors --lora_config_path config/lora_config_transformer_only.json
```

This creates your final model file: `out.safetensors` in your working directory.

---

**You're done! Your LoRA model is now trained and ready for use.**
