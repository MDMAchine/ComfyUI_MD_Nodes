# ACE-Step LoRA Training - Troubleshooting Guide

## Flash-Attention Installation Issues

### Problem: Flash-Attention Won't Install on Windows

Flash-attention requires `triton`, which can be problematic on Windows.

### Solution 1: Install triton-windows (Recommended)

```bash
pip install triton-windows
pip install flash-attn --no-build-isolation
```

### Solution 2: Install from Source

```bash
pip install packaging ninja
pip install flash-attn --no-build-isolation --no-binary :all: flash-attn
```

### Solution 3: Use Eager Attention (Fallback)

If flash-attention cannot be installed, you can fall back to "eager" attention mode:

1. Open `generate_prompts_lyrics.py` in a text editor
2. Find the `load_model` function
3. Locate this line:
   ```python
   attn_implementation="flash_attention_2"
   ```
4. Change it to:
   ```python
   attn_implementation="eager"
   ```
5. Save the file

**Note:** Eager attention is slower but more compatible.

---

## GPU Memory Issues

### Out of Memory (OOM) Errors

**Symptoms:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**

1. **Reduce Batch Size**
   ```bash
   --batch_size 1  # Already the default, but can't go lower
   ```

2. **Enable Gradient Accumulation**
   ```bash
   --accumulate_grad_batches 4  # Effectively batch_size=4 without memory cost
   ```

3. **Use Mixed Precision**
   ```bash
   --precision bf16-mixed  # Recommended for modern GPUs
   ```

4. **Reduce Model Size (Advanced)**
   - Edit LoRA config to use lower rank
   - Reduce number of workers: `--num_workers 4`

5. **Enable Gradient Checkpointing** (if available)
   - Check if trainer script has `--gradient_checkpointing` flag

### Checking GPU Memory Usage

```bash
# Windows
nvidia-smi

# Check continuously
nvidia-smi -l 1
```

---

## Data Loading Issues

### Slow Data Loading

**Symptoms:**
- Training stalls between batches
- Low GPU utilization
- High CPU usage

**Solutions:**

1. **Increase Workers**
   ```bash
   --num_workers 8  # Try 4, 8, 10, or 12
   ```

2. **Move Data to SSD**
   - Preprocessed data should be on fast storage
   - HDD will bottleneck training

3. **Check Disk Usage**
   ```bash
   # Windows
   perfmon  # Look at disk queue length
   ```

### Dataset Not Found

**Error:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'D:\...\processed_data\...'
```

**Solutions:**

1. Verify preprocessing completed successfully
2. Check that paths match exactly
3. Look for the `.hdf5` files in the prep directory

---

## Resume Training Issues

### Resume Fails with Error

**Symptoms:**
```
Error loading checkpoint
KeyError: 'state_dict'
```

**Solutions:**

1. **Verify Both Parameters**
   ```bash
   --resume_ckpt_path "path/to/checkpoint.ckpt" \
   --resume_id "wandb_run_id"
   ```
   BOTH are required!

2. **Check Checkpoint File**
   - File should be `.ckpt` (not `.safetensors`)
   - File size should be > 100MB
   - Located in `ace_step_lora/{run_id}/checkpoints/`

3. **Find the Resume ID**
   The resume_id is the directory name under `ace_step_lora/`:
   ```
   ace_step_lora/
   └── abc123xyz/  ← This is your resume_id
       └── checkpoints/
           └── epoch=10-step=1000.ckpt
   ```

### W&B Creates New Run Instead of Resuming

**Cause:** Missing or incorrect `--resume_id`

**Solution:** Always provide both parameters:
```bash
--resume_ckpt_path "..." --resume_id "abc123xyz"
```

---

## Training Not Learning

### Loss Not Decreasing

**Symptoms:**
- Training loss stays constant or increases
- Validation metrics don't improve

**Diagnostic Steps:**

1. **Check Learning Rate**
   ```bash
   # Try lower LR
   --learning_rate 5e-5  # Instead of 1e-4
   ```

2. **Verify Data Preprocessing**
   - Check that audio files were processed correctly
   - Ensure prompts exist for all audio files
   - Verify tags were added if using custom tags

3. **Check Training Steps**
   ```bash
   --max_steps 20000  # 2000 may be too few
   ```

4. **Review Dropout Rates**
   ```bash
   --speaker_dropout 0.0  # Lower if too high
   --lyrics_dropout 0.0
   ```

5. **Monitor Gradients**
   - Check logs for gradient norm
   - If gradients are NaN, reduce learning rate

### Generated Audio Sounds Nothing Like Training Data

**Solutions:**

1. **Train Longer**
   - Try 20,000-50,000 steps

2. **Use Custom Tags**
   - Add "mdmachine" or similar identifier to prompts
   - Use this tag during inference

3. **Check LoRA Strength**
   - During inference, try strength 0.8-1.2
   - Make sure you ran `add_alpha_in_lora.py`

4. **Verify Training Data Quality**
   - All audio should be similar style
   - Remove outliers or low-quality samples

---

## Checkpoint and File Issues

### Checkpoint Not Found After Training

**Check These Locations:**

1. **LoRA Weights (for inference):**
   ```
   D:\Ace-Step_Fork\ACE-Step\checkpoints\epoch=X-step=Y_lora\pytorch_lora_weights.safetensors
   ```

2. **Full Checkpoints (for resuming):**
   ```
   D:\Ace-Step_Fork\ACE-Step\ace_step_lora\{run_id}\checkpoints\epoch=X-step=Y.ckpt
   ```

3. **New Output Directory (if using --output_dir):**
   ```
   D:\Ace-Step_Fork\ACE-Step\runs\lora_checkpoints\
   ```

### Cannot Find Latest Checkpoint

**Manual Search:**

```bash
# Windows Command Prompt
dir /s /b D:\Ace-Step_Fork\ACE-Step\*.ckpt
dir /s /b D:\Ace-Step_Fork\ACE-Step\*.safetensors
```

```powershell
# PowerShell
Get-ChildItem -Path "D:\Ace-Step_Fork\ACE-Step" -Recurse -Filter "*.ckpt"
Get-ChildItem -Path "D:\Ace-Step_Fork\ACE-Step" -Recurse -Filter "*.safetensors"
```

---

## Weights & Biases (W&B) Issues

### Disable W&B Completely

**Method 1: Environment Variable (Recommended)**

```bash
# Windows Command Prompt
set WANDB_MODE=disabled

# Windows PowerShell
$env:WANDB_MODE="disabled"

# Then run training
python trainer_new.py ...
```

**Method 2: Use Offline Mode**

```bash
set WANDB_MODE=offline
```

### W&B Asking for Login

If you don't want to use W&B:
1. Choose option 3: "Don't visualize my results"
2. Or set environment variable before training (see above)

---

## Preprocessing Errors

### Qwen2.5-Omni Model Download Issues

**Error:**
```
Connection timeout / Download failed
```

**Solutions:**

1. **Check Internet Connection**
   - Model is ~10GB, needs stable connection

2. **Use HuggingFace Token (if needed)**
   ```bash
   huggingface-cli login
   ```

3. **Manual Download**
   ```python
   from transformers import AutoModel
   model = AutoModel.from_pretrained("Qwen/Qwen2.5-Omni-7B")
   ```

### Audio Files Not Recognized

**Supported Formats:**
- `.wav` (preferred)
- `.mp3`
- `.flac`
- `.ogg`

**Solution:**
Convert unsupported formats using ffmpeg:
```bash
ffmpeg -i input.m4a output.wav
```

---

## Python Environment Issues

### Module Not Found Errors

```
ModuleNotFoundError: No module named 'torch'
```

**Solution:**

1. **Verify Virtual Environment is Activated**
   ```bash
   # You should see (Ace-Step) in prompt
   cd D:\Ace-Step_Fork\ACE-Step
   .\Scripts\activate.bat
   ```

2. **Reinstall Requirements**
   ```bash
   pip install -r requirements.txt
   ```

3. **Check Python Version**
   ```bash
   python --version  # Should be 3.10 or 3.11
   ```

### Package Conflicts

**Symptoms:**
```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed.
```

**Solution:**

1. **Create Fresh Environment**
   ```bash
   cd D:\Ace-Step_Fork
   python -m venv ACE-Step-fresh
   cd ACE-Step-fresh
   .\Scripts\activate.bat
   ```

2. **Install in Order**
   ```bash
   pip install torch torchaudio torchvision --index-url [https://download.pytorch.org/whl/cu128](https://download.pytorch.org/whl/cu128)
   pip install --upgrade setuptools wheel
   pip install -r requirements.txt
   ```

---

## Performance Optimization

### Training Too Slow

**Optimizations:**

1. **Use BF16 Precision**
   ```bash
   --precision bf16-mixed
   ```

2. **Increase Workers**
   ```bash
   --num_workers 10
   ```

3. **Enable Flash Attention**
   - See Flash-Attention section above

4. **Check GPU Utilization**
   ```bash
   nvidia-smi -l 1
   ```
   Should be 90-100% during training

5. **Reduce Checkpoint Frequency**
   ```bash
   --save_every_n_train_steps 1000  # Instead of 100
   ```

### Disk Space Issues

**Check Available Space:**
```bash
# Windows
dir D:\
```

**Space Requirements:**
- Raw audio: ~1-5GB (depends on dataset)
- Preprocessed data: ~2-10GB
- Checkpoints: ~500MB-5GB per checkpoint
- **Recommended:** 50GB free space minimum

**Cleanup:**
```bash
# Remove old checkpoints
D:\Ace-Step_Fork\ACE-Step\checkpoints\
D:\Ace-Step_Fork\ACE-Step\ace_step_lora\
```

---

## Getting Help

### Information to Provide When Asking for Help

1. **Error Message** (full traceback)
2. **Command Used**
3. **System Info:**
   ```bash
   python --version
   pip list | grep torch
   nvidia-smi
   ```
4. **Last 50 Lines of Log**
5. **Steps Already Tried**

### Useful Debugging Commands

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Check CUDA version
python -c "import torch; print(torch.version.cuda)"

# List all checkpoints
dir /s D:\Ace-Step_Fork\ACE-Step\*.ckpt

# Check disk space
wmic logicaldisk get size,freespace,caption
```

---

## Common Error Messages and Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| `CUDA out of memory` | GPU memory full | Reduce batch size, enable gradient accumulation |
| `Module 'flash_attn' has no attribute...` | Flash-attention not installed | Use eager attention fallback |
| `No such file or directory` | Wrong path | Verify paths, check for typos |
| `KeyError: 'state_dict'` | Bad checkpoint | Use different checkpoint, start fresh |
| `AttributeError: 'NoneType'` | Missing model/data | Check preprocessing completed |
| `RuntimeError: Sizes of tensors must match` | Shape mismatch | Clear checkpoints, restart fresh |
| `ConnectionError` | Network issue | Check internet, try again |
| `PermissionError` | File locked/no access | Close programs, run as admin |

---

**Last Updated:** 2025  
**Version:** 2.1  
**MDMAchine / ComfyUI_MD_Nodes**