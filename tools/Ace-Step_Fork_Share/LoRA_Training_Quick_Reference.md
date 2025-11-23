# ACE-Step LoRA Training Quick Reference

## Environment Setup (First Time Only)

```bash
cd D:\Ace-Step_Fork\ACE-Step
.\Scripts\activate.bat

# Install dependencies
pip install torch torchaudio torchvision --index-url [https://download.pytorch.org/whl/cu128](https://download.pytorch.org/whl/cu128)
pip install --upgrade setuptools wheel
pip install gptqmodel==2.2.0 --no-build-isolation
pip uninstall transformers
pip install git+[https://github.com/huggingface/transformers.git](https://github.com/huggingface/transformers.git)
pip install -r requirements.txt
```

## Basic Training Pipeline (Manual Mode)

### 1. Generate Prompts
```bash
# GPU Mode (Qwen - Recommended)
python generate_prompts_lyrics.py --data_dir "D:\Desktop\MDMAchine\normalized\MP3_Files" --lyrics --model_type qwen

# CPU Mode (Ollama - Low VRAM)
python generate_prompts_lyrics.py --data_dir "D:\Desktop\MDMAchine\normalized\MP3_Files" --lyrics --model_type ollama
```

### 1.5. Add Custom Tags (Optional but Recommended)
```bash
# Note: Tool is in the parent Tools directory
python ..\Tools\add_tag.py --data_dir "D:\Desktop\MDMAchine\normalized\MP3_Files" --tags "mdmachine"

# Multiple tags
python ..\Tools\add_tag.py --data_dir "D:\Desktop\MDMAchine\normalized\MP3_Files" --tags "mdmachine, synthwave, custom_style"
```

**Why add tags?** This embeds your dataset identifier (e.g., "mdmachine") into all prompts, making it easier to trigger your trained style during inference.

### 2. Create HuggingFace Dataset
```bash
python convert2hf_dataset_new.py \
    --data_dir "D:\Desktop\MDMAchine\normalized\MP3_Files" \
    --output_name "D:\Ace-Step_Fork\processed_data\mdmachine_filenames"
```

### 3. Preprocess Audio
```bash
python preprocess_dataset_new.py \
    --input_name "D:\Ace-Step_Fork\processed_data\mdmachine_filenames" \
    --output_dir "D:\Ace-Step_Fork\processed_data\mdmachine_prep"
```

### 4. Clear Old Checkpoints (Recommended for New Training)
```bash
# Clear these directories before starting a NEW training session:
D:\Ace-Step_Fork\ACE-Step\checkpoints\
D:\Ace-Step_Fork\ACE-Step\ace_step_lora\  # Optional
```

**Important:** Do NOT clear if you plan to resume training!

### 5. Train LoRA
```bash
# Quick test (2000 steps)
python trainer_new.py --dataset_path "D:\Ace-Step_Fork\processed_data\mdmachine_prep"

# Full training (20k steps)
python trainer_new.py \
    --dataset_path "D:\Ace-Step_Fork\processed_data\mdmachine_prep" \
    --max_steps 20000 \
    --num_workers 10
```

### 6. Adjust LoRA Strength
```bash
python add_alpha_in_lora.py \
    --input_name checkpoints\epoch=62-step=20000_lora\pytorch_lora_weights.safetensors \
    --output_name mdmachine_20k.safetensors \
    --lora_config_path config/lora_config_transformer_only.json
```

## Resuming Training

```bash
python trainer_new.py \
    --dataset_path "D:\Ace-Step_Fork\processed_data\mdmachine_prep" \
    --max_steps 20000 \
    --num_workers 10 \
    --resume_ckpt_path "D:\Ace-Step_Fork\ACE-Step\ace_step_lora\{run_id}\checkpoints\epoch=27-step=3500.ckpt" \
    --resume_id "{run_id}"
```

**Important:** Both `--resume_ckpt_path` AND `--resume_id` are required!

## Common Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--max_steps` | Maximum training steps | 2000 |
| `--num_workers` | Data loading workers | 0 |
| `--learning_rate` | Learning rate | 1e-4 |
| `--batch_size` | Batch size | 1 |
| `--precision` | Training precision | bf16-mixed |
| `--save_every_n_train_steps` | Checkpoint frequency | 100 |
| `--warmup_steps` | LR warmup steps | 10 |
| `--gradient_clip_val` | Gradient clipping | 1.0 |
| `--speaker_dropout` | Speaker dropout rate | 0.0 |
| `--lyrics_dropout` | Lyrics dropout rate | 0.0 |

## Before New Training Session

Clear these directories to start fresh:
```bash
# Clear old checkpoints (optional, not if resuming)
D:\Ace-Step_Fork\ACE-Step\checkpoints\

# Clear old run data (optional, not if resuming)
D:\Ace-Step_Fork\ACE-Step\ace_step_lora\
```

## Checkpoint Locations

### Training Checkpoints (.ckpt files)
```
D:\Ace-Step_Fork\ACE-Step\ace_step_lora\{run_id}\checkpoints\epoch=X-step=Y.ckpt
```

### LoRA Weights (for inference)
```
# Old location
D:\Ace-Step_Fork\ACE-Step\checkpoints\epoch=X-step=Y_lora\pytorch_lora_weights.safetensors

# New location (if using --output_dir)
D:\Ace-Step_Fork\ACE-Step\runs\lora_checkpoints\epoch=X-step=Y_lora\pytorch_lora_weights.safetensors
```

## Disabling Weights & Biases (W&B)

Set environment variable before training:
```bash
# Windows (PowerShell)
$env:WANDB_MODE="disabled"

# Windows (CMD)
set WANDB_MODE=disabled

# Linux/Mac
export WANDB_MODE=disabled
```

## Troubleshooting

### Out of GPU Memory
- Reduce `--batch_size`
- Increase `--accumulate_grad_batches`
- Use `--precision bf16-mixed`

### Slow Data Loading
- Increase `--num_workers` (try 8-12)
- Move data to SSD if possible

### Resume Fails
- Verify both `--resume_ckpt_path` and `--resume_id` are correct
- Check that checkpoint file exists
- Ensure the run_id matches the directory name

### Training Loss Not Decreasing
- Check learning rate (try 1e-4 to 1e-5)
- Verify data preprocessing completed successfully
- Ensure sufficient training steps (20k+ recommended)
- Check that dropout rates aren't too high

## Using the Automation Script

```bash
cd D:\Ace-Step_Fork
python lora_trainer_automation_v2.1_FINAL.py
```

The script will:
1. Guide you through all configuration options
2. Auto-detect checkpoints for resuming
3. Handle all paths and parameters
4. Provide detailed logging
5. Allow starting from any pipeline step

## Recommended Training Settings

### For Testing/Iteration
```bash
--max_steps 2000 \
--num_workers 4 \
--save_every_n_train_steps 500
```

### For Production Training
```bash
--max_steps 20000 \
--num_workers 10 \
--learning_rate 1e-4 \
--warmup_steps 100 \
--save_every_n_train_steps 500 \
--save_last 10
```

### For Fine-tuning Existing LoRA
```bash
--max_steps 5000 \
--learning_rate 5e-5 \
--warmup_steps 50
```

---

**MDMAchine / ComfyUI_MD_Nodes**