# ACE-Step LoRA Training Suite v2.1 - DEPLOYMENT GUIDE

## Directory Structure

```text
D:\Ace-Step_Fork\
├── README_MASTER.md                      ← Start here
├── FINAL_PACKAGE_SUMMARY.md              ← Integration details
├── LoRA_Training_Quick_Reference.md      ← Command cheat sheet
├── Troubleshooting_Guide.md              ← Problem solving
├── Changelog_and_Fixes.md                ← What changed
├── ACE_Step_LoRA_Training_Guide.docx     ← Complete reference
├── lora_trainer_automation_v2.1_FINAL.py ← MAIN AUTOMATION SCRIPT
│
├── Tools\                                ← Your utilities
│   ├── add_tag.py                        ← Tag addition
│   ├── lora_magician_pro.py              ← Dataset cleanup
│   └── normalize_audio.py                ← Audio processing
│
├── ACE-Step\                             ← Main training environment
│   ├── Scripts\                          ← Virtual environment
│   │   └── python.exe
│   ├── config\
│   │   └── lora_config_transformer_only.json
│   ├── trainer_new.py                    ← YOUR modified trainer
│   ├── generate_prompts_lyrics.py        ← YOUR modified prompt gen
│   ├── add_alpha_in_lora.py
│   ├── convert2hf_dataset_new.py
│   ├── preprocess_dataset_new.py
│   └── ... (other ACE-Step files)
│
├── processed_data\                       ← Generated during training
│   ├── {project}_filenames/
│   └── {project}_prep/
│
├── runs\                                 ← Training outputs
│   └── lora_checkpoints/
│
└── LoRa\                                 ← Final models (optional)
```

---

## Quick Setup (3 Steps)

### Step 1: Verify Your Structure

Check that you have these files in place:

```bash
# Check automation script
D:\Ace-Step_Fork\lora_trainer_automation_v2.1_FINAL.py

# Check your tools
D:\Ace-Step_Fork\Tools\add_tag.py
D:\Ace-Step_Fork\Tools\lora_magician_pro.py
D:\Ace-Step_Fork\Tools\normalize_audio.py

# Check ACE-Step environment
D:\Ace-Step_Fork\ACE-Step\Scripts\python.exe
D:\Ace-Step_Fork\ACE-Step\trainer_new.py
D:\Ace-Step_Fork\ACE-Step\generate_prompts_lyrics.py
```

### Step 2: Activate Environment

```bash
cd D:\Ace-Step_Fork\ACE-Step
.\Scripts\activate.bat
```

You should see: `(ACE-Step) D:\Ace-Step_Fork\ACE-Step>`

### Step 3: Run Automation

```bash
cd D:\Ace-Step_Fork
python lora_trainer_automation_v2.1_FINAL.py
```

---

## File Locations Explained

### Root Level (`D:\Ace-Step_Fork\`)
- **Documentation** - All MD and DOCX files
- **Main automation script** - `lora_trainer_automation_v2.1_FINAL.py`
- **Working directories** - `processed_data/`, `runs/`, `LoRa/`

### Tools Directory (`D:\Ace-Step_Fork\Tools\`)
- **add_tag.py** - Adds custom identifiers to prompts
- **lora_magician_pro.py** - Cleans up and organizes dataset
- **normalize_audio.py** - Audio preprocessing (LUFS normalization)

### ACE-Step Directory (`D:\Ace-Step_Fork\ACE-Step\`)
- **Virtual environment** - `Scripts/python.exe`
- **Training scripts** - `trainer_new.py`, etc.
- **Config files** - `config/lora_config_transformer_only.json`

---

## Parameter Corrections for YOUR Version

### Important: Your trainer_new.py uses DIFFERENT parameter names!

| Standard Name | YOUR Version | Script Uses |
|---------------|--------------|-------------|
| `--resume_ckpt_path` | `--resume_full_ckpt_path` | Fixed |
| `--resume_id` | `--wandb_run_id` | Fixed |

The automation script has been **corrected** to use YOUR parameter names.

### Your Modified Defaults:
```python
--max_steps: 5000           (was: 2000)
--warmup_steps: 100         (was: 10)
--accumulate_grad_batches: 2 (was: 1)
--save_every_n_train_steps: 500 (was: 100)
--speaker_dropout: 0.05     (was: 0.0)
--lyrics_dropout: 0.05      (was: 0.0)
```

These are better defaults! The automation script respects them.

---

## Complete Training Workflow

### Method 1: Full Automation (Recommended)

```bash
# 1. (Optional) Pre-normalize audio
cd D:\Ace-Step_Fork
python Tools\normalize_audio.py
# Input: Your raw audio folder
# Output: D:\Desktop\MDMAchine\normalized\MP3_Files

# 2. Run complete training pipeline
cd D:\Ace-Step_Fork
python lora_trainer_automation_v2.1_FINAL.py

# Follow prompts:
# → Project name: mdmachine
# → Audio path: D:\Desktop\MDMAchine\normalized\MP3_Files
# → Start step: 1
# → Add tags: Yes → "mdmachine"
# → Run cleanup: Yes
# → Max steps: 20000
# → [Configure other parameters]
# → Start!
```

### Method 2: Manual Commands

```bash
cd D:\Ace-Step_Fork\ACE-Step
.\Scripts\activate.bat

# Step 1: Generate prompts
python generate_prompts_lyrics.py --data_dir "D:\Desktop\MDMAchine\normalized\MP3_Files" --lyrics

# Step 1.5: Add tags
cd D:\Ace-Step_Fork
python Tools\add_tag.py --data_dir "D:\Desktop\MDMAchine\normalized\MP3_Files" --tags "mdmachine"

# Step 2: Dataset cleanup
python Tools\lora_magician_pro.py

# Step 3: Create HF dataset
cd ACE-Step
python convert2hf_dataset_new.py \
    --data_dir "D:\Desktop\MDMAchine\normalized\MP3_Files" \
    --output_name "D:\Ace-Step_Fork\processed_data\mdmachine_filenames"

# Step 4: Preprocess audio
python preprocess_dataset_new.py \
    --input_name "D:\Ace-Step_Fork\processed_data\mdmachine_filenames" \
    --output_dir "D:\Ace-Step_Fork\processed_data\mdmachine_prep"

# Step 5: Train LoRA
python trainer_new.py \
    --dataset_path "D:\Ace-Step_Fork\processed_data\mdmachine_prep" \
    --max_steps 20000 \
    --num_workers 10 \
    --output_dir "D:\Ace-Step_Fork\runs" \
    --exp_name "mdmachine"

# Step 6: Adjust LoRA strength
# Find latest checkpoint in D:\Ace-Step_Fork\ACE-Step\checkpoints\
python add_alpha_in_lora.py \
    --input_name checkpoints/epoch=XX-step=YYYY_lora/pytorch_lora_weights.safetensors \
    --output_name mdmachine_20k.safetensors \
    --lora_config_path config/lora_config_transformer_only.json
```

---

## Resuming Training

Your `trainer_new.py` uses these parameters for resuming:

```bash
python trainer_new.py \
    --dataset_path "D:\Ace-Step_Fork\processed_data\mdmachine_prep" \
    --max_steps 20000 \
    --num_workers 10 \
    --output_dir "D:\Ace-Step_Fork\runs" \
    --exp_name "mdmachine" \
    --resume_full_ckpt_path "D:\Ace-Step_Fork\ACE-Step\ace_step_lora\{run_id}\checkpoints\epoch=XX-step=YYYY.ckpt" \
    --wandb_run_id "{run_id}"
```

**Both parameters are required!**

The automation script will:
1. Auto-detect the latest checkpoint
2. Extract the wandb_run_id from the directory name
3. Prompt you to confirm
4. Resume with correct parameters

---

## Checkpoint Locations

### Full Training Checkpoints (.ckpt files):
```text
D:\Ace-Step_Fork\ACE-Step\ace_step_lora\{wandb_run_id}\checkpoints\
└── epoch=XX-step=YYYY.ckpt
```

### LoRA Weights (for inference):
```text
D:\Ace-Step_Fork\ACE-Step\checkpoints\
└── epoch=XX-step=YYYY_lora\
    └── pytorch_lora_weights.safetensors
```

### Final Adjusted LoRA:
```text
D:\Ace-Step_Fork\ACE-Step\
└── mdmachine_20k.safetensors  ← Use this in ComfyUI
```

### Optional: Move to LoRa folder:
```bash
move D:\Ace-Step_Fork\ACE-Step\mdmachine_20k.safetensors D:\Ace-Step_Fork\LoRa\
```

---

## Tool Integration

The automation script automatically finds and uses your tools:

| Tool | Path | Called at |
|------|------|-----------|
| **add_tag.py** | `D:\Ace-Step_Fork\Tools\` | Step 1.5 |
| **lora_magician_pro.py** | `D:\Ace-Step_Fork\Tools\` | Step 2 |
| **normalize_audio.py** | `D:\Ace-Step_Fork\Tools\` | Manual (Step 0) |

### Manual Tool Usage:

```bash
# Normalize audio
cd D:\Ace-Step_Fork
python Tools\normalize_audio.py

# Add tags
python Tools\add_tag.py --data_dir "path" --tags "mdmachine"

# Clean dataset
python Tools\lora_magician_pro.py
```

---

## Pre-Flight Checklist

Before training, verify:

- [ ] Virtual environment activated
- [ ] GPU is available (`nvidia-smi`)
- [ ] Audio files are in one folder
- [ ] At least 50GB free disk space
- [ ] Your tools are in `Tools/` directory
- [ ] Documentation is accessible

---

## Your Modified Scripts

### trainer_new.py Changes:
- Better default values (5000 steps, 100 warmup, etc.)
- Uses `--resume_full_ckpt_path` and `--wandb_run_id`
- Default output to `./runs`
- Improved dropout defaults (0.05)

### generate_prompts_lyrics.py Changes:
- Ollama integration support
- Low VRAM mode support
- Loguru logging
- Automatic package installation

---

## Training Examples

### Quick Test (1 hour):
```bash
python lora_trainer_automation_v2.1_FINAL.py
# Max steps: 2000
# Workers: 4
```

### Development (4-6 hours):
```bash
python lora_trainer_automation_v2.1_FINAL.py
# Max steps: 20000
# Workers: 10
# Learning rate: 1e-4
```

### Maximum Quality (10-15 hours):
```bash
python lora_trainer_automation_v2.1_FINAL.py
# Max steps: 50000
# Workers: 10
# Learning rate: 5e-5
# Save every: 1000 steps
```

---

## Troubleshooting

### Script not finding tools:
```bash
# Check paths
dir D:\Ace-Step_Fork\Tools\add_tag.py
dir D:\Ace-Step_Fork\Tools\lora_magician_pro.py
```

### Python not found:
```bash
# Verify environment
D:\Ace-Step_Fork\ACE-Step\Scripts\python.exe --version
```

### Import errors:
```bash
cd D:\Ace-Step_Fork\ACE-Step
.\Scripts\activate.bat
pip install -r requirements.txt
```

### Resume fails:
Verify both parameters:
- `--resume_full_ckpt_path` → Full path to .ckpt file
- `--wandb_run_id` → Directory name from ace_step_lora/

---

## Quick Reference

### Most Used Paths:
```bash
# Root
D:\Ace-Step_Fork\

# Tools
D:\Ace-Step_Fork\Tools\

# Training environment
D:\Ace-Step_Fork\ACE-Step\

# Data
D:\Ace-Step_Fork\processed_data\
D:\Ace-Step_Fork\runs\

# Final models
D:\Ace-Step_Fork\ACE-Step\*.safetensors
D:\Ace-Step_Fork\LoRa\  (optional)
```

### Most Used Commands:
```bash
# Activate environment
cd D:\Ace-Step_Fork\ACE-Step
.\Scripts\activate.bat

# Run automation
cd D:\Ace-Step_Fork
python lora_trainer_automation_v2.1_FINAL.py

# Check GPU
nvidia-smi

# Disable W&B
set WANDB_MODE=disabled
```

---

## You're Ready!

Everything is configured for your exact directory structure:
- Correct paths for all tools
- Correct parameter names for YOUR trainer
- Proper checkpoint detection
- Full automation support

**Run this to start:**
```bash
cd D:\Ace-Step_Fork
python lora_trainer_automation_v2.1_FINAL.py
```

Good luck with your training!

---

**Version:** 2.1 Final
**Configured for:** Your exact directory structure
**Tested with:** Your modified trainer_new.py and generate_prompts_lyrics.py