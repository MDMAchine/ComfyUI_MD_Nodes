# ACE-Step LoRA Training Automation Suite v2.1
## Comprehensive Toolkit for Custom Audio LoRA Training

**Created by:** MDMAchine / ComfyUI_MD_Nodes
**Version:** 2.1 Final
**Last Updated:** 2025

---

## Overview

This package provides a complete ecosystem for training high-quality LoRA models for ACE-Step audio generation. It emphasizes automation, reproducibility, and workflow efficiency by integrating dataset management, audio normalization, and training execution into a unified pipeline.

### Included Components:
- **Automation Script:** `lora_trainer_automation_v2.1_FINAL.py` - Orchestrates the entire pipeline from tagging to final weight adjustment.
- **Dataset Utilities:** Tools for audio normalization, tag injection, and directory cleanup.
- **Documentation:** A full suite of guides ranging from quick-start references to deep technical deep dives.

---

## Installation & Quick Start

### 1. Clone Repository
First, clone the base ACE-Step repository. This provides the core training environment required for the suite to function.

```bash
git clone [https://github.com/woct0rdho/ACE-Step](https://github.com/woct0rdho/ACE-Step) D:\Ace-Step_Fork\ACE-Step
```

### 2. Setup Directory Structure
Ensure the automation files are placed in the root directory (parent of the cloned `ACE-Step` folder), resulting in the following structure:

```text
D:\Ace-Step_Fork\                  <-- Root Directory
├── lora_trainer_automation_v2.1_FINAL.py
├── ACE-Step\                      <-- Cloned Repo
│   └── ... (training scripts)
└── Tools\                         <-- Your Utilities
    ├── add_tag.py
    ├── lora_magician_pro.py
    └── normalize_audio.py
```

### 3. Launch Automation
Open a terminal in your root directory and execute the script:

```bash
cd D:\Ace-Step_Fork
python lora_trainer_automation_v2.1_FINAL.py
```

### 4. Configuration
The script provides an interactive wizard to configure:
- Data source directories
- Training hyperparameters (Steps, Learning Rate, Batch Size)
- Processing options (Tagging, Cleanup, Normalization)
- Resume functionality (Auto-detection of checkpoints)

---

## Package Manifest

### Automation & Utilities

| File | Description | Location |
|------|-------------|----------|
| `lora_trainer_automation_v2.1_FINAL.py` | Primary automation controller. Manages the 6-step training pipeline. | Root |
| `add_tag.py` | Utility to recursively inject style tags into prompt text files. | Tools/ |
| `lora_magician_pro.py` | Dataset validator. Removes empty files and organizes prompts by length. | Tools/ |
| `normalize_audio.py` | Audio pre-processor. Normalizes to -14 LUFS and converts formats. | Tools/ |

### Documentation

| File | Purpose |
|------|---------|
| `README_MASTER.md` | General overview and installation guide. |
| `LoRA_Training_Quick_Reference.md` | CLI command cheat sheet and common parameters. |
| `Troubleshooting_Guide.md` | Solutions for common errors (Flash Attention, OOM, etc.). |
| `ACE_Step_LoRA_Training_Guide.docx` | In-depth technical reference manual. |
| `BEST_PRACTICES.md` | Recommended configurations for different training tiers. |

---

## Workflow Architecture

### Phase 1: Pre-Processing
Before training, it is recommended to standardize audio inputs.
**Tool:** `normalize_audio.py`
- Normalizes audio to -14.0 LUFS (Streaming Standard).
- Standardizes sample rates and formats (WAV/MP3).

### Phase 2: Training Pipeline
The automation script executes the following sequence:

1.  **Prompt Generation:** Generates descriptive text/lyrics for audio files.
    * *Options:* Qwen (GPU-accelerated) or Ollama (CPU/Low-VRAM fallback).
2.  **Tag Injection:** Appends unique trigger words (e.g., `style_name`) to prompts via `add_tag.py`.
3.  **Dataset Validation:** Cleans and organizes dataset structure via `lora_magician_pro.py`.
4.  **Dataset Formatting:** Converts raw audio/text pairs into HuggingFace dataset format.
5.  **Feature Extraction:** Pre-processes audio for the transformer model.
6.  **Training:** Executes the main training loop with specified parameters.
7.  **Weight Adjustment:** Extracts and adjusts final LoRA weights for inference.

---

## Utility Features

### Tag Injection (`add_tag.py`)
* **Function:** Recursively adds specific tags to text files.
* **Usage:** Ensures consistent trigger words across the entire dataset.
* **Features:** Duplicate prevention and case-insensitive matching.

### Dataset Management (`lora_magician_pro.py`)
* **Function:** Audits the dataset for quality control.
* **Organization:** Automatically sorts files into folders based on prompt validity (Empty, Too Short, Too Long).
* **Safety:** Creates backups of original files before modification.

### Audio Normalization (`normalize_audio.py`)
* **Function:** Batch audio processing.
* **Standardization:** ensures consistent volume levels across training data, preventing the model from learning volume artifacts.
* **Features:** Metadata preservation and multi-format support.

---

## Version 2.1 Improvements

### Critical Fixes
* **Resume Logic:** Corrected parameters (`--resume_ckpt_path` and `--resume_id`) to ensure proper training continuation.
* **Checkpoint Detection:** Implemented auto-detection logic to locate the most recent `.ckpt` files across directory structures.

### New Capabilities
* **Hybrid Captioning:** Integrated support for both Qwen (GPU) and Ollama (CPU) for prompt generation.
* **Integrated Tagging:** Tag injection is now a seamless step within the automation pipeline.
* **Checkpoint Management:** Interactive prompts to safely clear old checkpoints before starting new runs.
* **W&B Control:** Added options to disable Weights & Biases logging for offline training.

---

## Best Practices & Recommendations

### Training Configurations
Refer to `BEST_PRACTICES.md` for detailed parameter sets.

* **Testing (Tier 2):** 10,000 Steps. For rapid iteration and validation.
* **Production (Tier 1):** 20,000 Steps. Recommended balance of quality and time.
* **High Fidelity (Tier 3):** 35,000 Steps. For final release candidates.

### Hardware Requirements
* **GPU:** NVIDIA GPU with 12GB+ VRAM recommended.
* **Storage:** NVMe SSD recommended for data loading performance.
* **System:** Windows 10/11 or Linux.

---

## Support & Troubleshooting

For technical issues, consult `Troubleshooting_Guide.md`.

**Common Solutions:**
* **OOM Errors:** Increase `--accumulate_grad_batches` or enable mixed precision (`bf16-mixed`).
* **Slow Training:** Increase `--num_workers` (up to CPU core count - 2).
* **Resume Failures:** Ensure both the checkpoint path and Run ID are specified.

---

**MDMAchine / ComfyUI_MD_Nodes**
*2025 Release*