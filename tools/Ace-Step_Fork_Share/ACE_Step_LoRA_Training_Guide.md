# ACE-Step LoRA Training Guide
## Complete Reference & Best Practices

---

## 1. Environment Setup

### 1.1 Initial Setup
* Navigate to ACE-Step directory
* Activate virtual environment
* Install PyTorch with CUDA support
* Install required dependencies

```bash
cd D:\Ace-Step_Fork\ACE-Step
.\Scripts\activate.bat

pip install torch torchaudio torchvision --index-url [https://download.pytorch.org/whl/cu128](https://download.pytorch.org/whl/cu128)
pip install --upgrade setuptools wheel
pip install gptqmodel==2.2.0 --no-build-isolation
pip uninstall transformers
pip install git+[https://github.com/huggingface/transformers.git](https://github.com/huggingface/transformers.git)
pip install -r requirements.txt
```

### 1.2 Flash Attention (Optional)
For improved training performance:

```bash
pip install flash-attn --no-build-isolation

# Alternative if above fails:
pip install packaging ninja
pip install flash-attn --no-build-isolation --no-binary :all: flash-attn
pip install triton-windows
```

---

## 2. Training Pipeline Steps

### 2.1 Step 1: Generate Prompts
Generate text prompts from your audio files, optionally including lyrics:

```bash
# Option A: GPU (Qwen - Recommended)
python generate_prompts_lyrics.py --data_dir "D:\Desktop\MDMAchine\normalized\MP3_Files" --lyrics --model_type qwen

# Option B: CPU (Ollama - Low VRAM)
python generate_prompts_lyrics.py --data_dir "D:\Desktop\MDMAchine\normalized\MP3_Files" --lyrics --model_type ollama
```

### 2.2 Step 2: Clean Dataset (Optional)
Use LoRA Magician Pro to clean up dataset and text files:

```bash
python ..\Tools\lora_magician_pro.py
```

### 2.3 Step 3: Create HuggingFace Dataset
Convert your audio files to HuggingFace dataset format:

```bash
python convert2hf_dataset_new.py \
    --data_dir "D:\Desktop\MDMAchine\normalized\MP3_Files" \
    --output_name "D:\Ace-Step_Fork\processed_data\mdmachine_filenames"
```

### 2.4 Step 4: Preprocess Audio
Preprocess the audio data for training:

```bash
python preprocess_dataset_new.py \
    --input_name "D:\Ace-Step_Fork\processed_data\mdmachine_filenames" \
    --output_dir "D:\Ace-Step_Fork\processed_data\mdmachine_prep"
```

### 2.5 Step 5: LoRA Training
Start the training process:

```bash
# Basic training (2000 steps default)
python trainer_new.py \
    --dataset_path "D:\Ace-Step_Fork\processed_data\mdmachine_prep"

# Extended training with custom parameters (Recommended)
python trainer_new.py \
    --dataset_path "D:\Ace-Step_Fork\processed_data\mdmachine_prep" \
    --max_steps 20000 \
    --num_workers 10 \
    --learning_rate 5e-5 \
    --warmup_steps 100
```

### 2.6 Step 6: Adjust LoRA Strength
After training, adjust the LoRA alpha value:

```bash
python add_alpha_in_lora.py \
    --input_name checkpoints\epoch=62-step=20000_lora\pytorch_lora_weights.safetensors \
    --output_name mdmachine_20k.safetensors \
    --lora_config_path config/lora_config_transformer_only.json
```

---

## 3. Resuming Training

To resume training from a checkpoint:

```bash
python trainer_new.py \
    --dataset_path "D:\Ace-Step_Fork\processed_data\mdmachine_prep" \
    --max_steps 20000 \
    --num_workers 10 \
    --output_dir "D:\Ace-Step_Fork\runs" \
    --resume_ckpt_path "D:\Ace-Step_Fork\ACE-Step\ace_step_lora\i9501mzk\checkpoints\epoch=27-step=3500.ckpt" \
    --wandb_run_id "i9501mzk"
```

**Important:** Both `--resume_ckpt_path` AND `--wandb_run_id` are required for proper resumption.

---

## 4. Training Parameters Reference

### 4.1 Model Arguments
- `--checkpoint_dir`: Directory containing base ACE-Step model checkpoint
- `--shift`: Parameter for FlowMatchEulerDiscreteScheduler (default: 3.0)
- `--lora_config_path`: Path to LoRA configuration JSON file
- `--last_lora_path`: Path to .safetensors file with LoRA weights to load

### 4.2 Data Arguments
- `--dataset_path`: Path to preprocessed HDF5 dataset directory
- `--batch_size`: Number of samples per batch (default: 1)
- `--num_workers`: Number of data loading workers (default: 0)
- `--tag_dropout`: Dropout rate for tags (default: 0.5)
- `--speaker_dropout`: Dropout rate for speaker embeddings (default: 0.0)
- `--lyrics_dropout`: Dropout rate for lyrics (default: 0.0)

### 4.3 Optimizer Arguments
- `--optimizer`: Optimizer to use (adamw or prodigy, default: adamw)
- `--learning_rate`: Initial learning rate (default: 1e-4)
- `--beta1`: Beta1 for optimizer (default: 0.9)
- `--beta2`: Beta2 for optimizer (default: 0.99)
- `--weight_decay`: Weight decay/L2 penalty (default: 1e-2)
- `--max_steps`: Maximum optimization steps (default: 2000)
- `--warmup_steps`: Linear LR warmup steps (default: 10)
- `--accumulate_grad_batches`: Gradient accumulation (default: 1)
- `--gradient_clip_val`: Gradient clipping value (default: 1.0)
- `--gradient_clip_algorithm`: Clipping algorithm (norm/value, default: norm)

### 4.4 Checkpoint & Logging Arguments
- `--output_dir`: Base directory for logs and checkpoints (default: ./output)
- `--exp_name`: Experiment name for W&B (default: ace_step_lora)
- `--wandb_run_id`: W&B run ID to resume from
- `--resume_ckpt_path`: Path to PyTorch Lightning .ckpt file to resume from
- `--precision`: Training precision (bf16-mixed/16-mixed/32-true)
- `--save_every_n_train_steps`: Save checkpoint every N steps (default: 100)
- `--save_last`: Keep last N LoRA checkpoints (default: 5)

---

## 5. Best Practices & Tips

### 5.1 Before Starting a New Training Session
- Clear old checkpoints from `D:\Ace-Step_Fork\ACE-Step\checkpoints`
- Optionally clear `D:\Ace-Step_Fork\ACE-Step\ace_step_lora` (unless resuming)
- Ensure sufficient disk space for checkpoints
- Verify GPU memory availability
- Consider disabling W&B if not needed (set `WANDB_MODE=disabled`)

### 5.2 Checkpoint Management
- Training creates .ckpt files in `ace_step_lora/{run_id}/checkpoints/`
- LoRA weights are saved separately in `checkpoints/epoch=X-step=Y_lora/`
- Use `--save_last` to control number of kept checkpoints
- Full checkpoints (.ckpt) are needed for resuming
- Only `pytorch_lora_weights.safetensors` is needed for inference

### 5.3 Training Optimization
- Use `bf16-mixed` precision on supported GPUs for best performance
- Increase `num_workers` (8-12) for faster data loading
- Use gradient accumulation if running out of GPU memory
- Start with 2000 steps for testing, scale up to 20000+ for final training
- Monitor training loss - it should steadily decrease
- Test generated audio at different checkpoint intervals

### 5.4 Common Issues & Solutions
- **Out of Memory:** Reduce batch_size, increase accumulate_grad_batches, or use gradient checkpointing
- **Slow Data Loading:** Increase num_workers, ensure data is on fast storage (SSD)
- **Training Loss Not Decreasing:** Check learning rate, verify data preprocessing, ensure sufficient training steps
- **Resume Fails:** Verify both `--resume_ckpt_path` and `--wandb_run_id` are provided
- **Missing Dependencies:** Reinstall requirements.txt, check virtual environment activation

---

## 6. Directory Structure

```text
ACE-Step/
├── Scripts/                     # Virtual environment (Windows)
│   └── python.exe
├── config/
│   └── lora_config_transformer_only.json
├── checkpoints/                 # OLD checkpoint location
│   └── epoch=X-step=Y_lora/
│       └── pytorch_lora_weights.safetensors
├── ace_step_lora/               # Training run directories
│   └── {run_id}/
│       ├── checkpoints/
│       │   └── epoch=X-step=Y.ckpt
│       └── wandb/
├── processed_data/              # Preprocessed datasets
│   ├── {project}_filenames/
│   └── {project}_prep/
├── runs/                        # NEW output directory (recommended)
│   ├── lora_checkpoints/
│   │   └── epoch=X-step=Y_lora/
│   └── {experiment_name}/
└── Tools/                       # Optional utilities
    └── lora_magician_pro.py
```

---

## 7. Using the Automation Script

The enhanced automation script (`lora_trainer_automation_v2.1_FINAL.py`) provides:
- Interactive configuration with sensible defaults
- Automatic checkpoint detection for resuming
- Proper handling of `wandb_run_id` and `resume_ckpt_path`
- Fixed checkpoint path detection logic
- Step-by-step pipeline execution with validation
- Comprehensive logging
- Error recovery options

### 7.1 Running the Automation Script
1. Place the script in your ACE-Step fork root directory
2. Run: `python lora_trainer_automation_v2.1_FINAL.py`
3. Follow the interactive prompts
4. Choose starting step (1-6)
5. Configure training parameters
6. Monitor training progress

---

**MDMAchine / ComfyUI_MD_Nodes** *2025 Release*