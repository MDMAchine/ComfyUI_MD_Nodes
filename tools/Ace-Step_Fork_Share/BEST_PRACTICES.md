# ACE-Step LoRA Training - Best Practices Configuration

## Proven Parameter Sets
**Based on successful training runs with high-quality results**

---

## Training Tiers

### Tier 1: Production Quality (Recommended)
**Best balance of quality, time, and reliability**

```bash
--max_steps 20000
--learning_rate 5e-05
--warmup_steps 100
--save_every_n_train_steps 5000
--num_workers 10
--accumulate_grad_batches 2
--speaker_dropout 0.05
--lyrics_dropout 0.05
--gradient_clip_val 1.0
--gradient_clip_algorithm norm
--precision bf16-mixed
--save_last 5
--beta1 0.9
--beta2 0.99
--weight_decay 0.01
```

**Expected Results:**
- Training time: 6-8 hours
- Checkpoints: 4 saves (5k, 10k, 15k, 20k)
- Disk usage: ~3GB checkpoints
- Quality: Excellent, production-ready
- Risk: Low (proven safe parameters)

**Use When:**
- Training for the first time with new data
- You want reliable, high-quality results
- Time is not critical (can run overnight)
- This is your "go-to" configuration

---

### Tier 2: Quick Iteration
**For testing changes, experimenting, or limited time**

```bash
--max_steps 10000
--learning_rate 5e-05
--warmup_steps 50
--save_every_n_train_steps 2500
--num_workers 10
--accumulate_grad_batches 2
--speaker_dropout 0.05
--lyrics_dropout 0.05
--gradient_clip_val 1.0
--gradient_clip_algorithm norm
--precision bf16-mixed
--save_last 5
--beta1 0.9
--beta2 0.99
--weight_decay 0.01
```

**Expected Results:**
- Training time: 3-4 hours
- Checkpoints: 4 saves (2.5k, 5k, 7.5k, 10k)
- Disk usage: ~2GB checkpoints
- Quality: Good, suitable for testing
- Risk: Low

**Use When:**
- Testing dataset changes
- Experimenting with prompts/tags
- Need results quickly
- Validating your setup works

---

### Tier 3: Maximum Quality
**For final production models when quality is paramount**

```bash
--max_steps 35000
--learning_rate 5e-05
--warmup_steps 100
--save_every_n_train_steps 7000
--num_workers 10
--accumulate_grad_batches 2
--speaker_dropout 0.05
--lyrics_dropout 0.05
--gradient_clip_val 1.0
--gradient_clip_algorithm norm
--precision bf16-mixed
--save_last 5
--beta1 0.9
--beta2 0.99
--weight_decay 0.01
```

**Expected Results:**
- Training time: 12-16 hours
- Checkpoints: 5 saves (7k, 14k, 21k, 28k, 35k)
- Disk usage: ~3GB checkpoints
- Quality: Maximum, fully converged
- Risk: Very low (proven stable)

**Use When:**
- Creating final production model
- Maximum quality is required
- You have overnight/weekend time
- Previous 20k run was good but you want more

---

## Parameter Explanations

### Core Training Parameters

#### `--max_steps` (10000 / 20000 / 35000)
- **What it does:** Total number of training steps
- **Why these values:**
  - 10k = Basic style learning
  - 20k = Full style convergence (sweet spot)
  - 35k = Maximum quality, diminishing returns after
- **Don't go below:** 5000 (insufficient training)
- **Don't go above:** 50000 (wastes time, minimal gains)

#### `--learning_rate 5e-05` (0.00005)
- **What it does:** How fast the model learns
- **Why this value:**
  - Conservative for long training runs
  - Prevents instability and overshooting
  - Proven stable for 15k-35k steps
- **Alternative:** `1e-4` for shorter runs (<10k steps)
- **Don't use:** >1e-4 (unstable) or <1e-5 (too slow)

#### `--warmup_steps 100` (50 for 10k run)
- **What it does:** Gradual learning rate ramp-up
- **Why this value:**
  - Small warmup = more time at full LR
  - ~0.5% of total steps is optimal
  - Prevents initial instability
- **Rule of thumb:** 0.3-0.5% of max_steps

#### `--save_every_n_train_steps` (2500 / 5000 / 7000)
- **What it does:** Checkpoint frequency
- **Why these values:**
  - More frequent = more safety, more disk space
  - 5000 is sweet spot for 20k run (4 checkpoints)
  - 7000 saves space for 35k run (5 checkpoints)
- **Consideration:** Each checkpoint ~500MB

---

### Optimization Parameters

#### `--num_workers 10`
- **What it does:** Parallel data loading threads
- **Why 10:** Good balance for 6-12 core CPUs
- **Adjust based on CPU:**
  - 4-6 cores: use 4-6
  - 8-12 cores: use 8-10
  - 16+ cores: use 12-16
- **Don't exceed:** CPU cores - 2

#### `--accumulate_grad_batches 2`
- **What it does:** Effective batch size without memory cost
- **Why 2:** Smooths gradients, stable updates
- **Benefits:**
  - Reduces variance in gradient updates
  - Simulates batch_size=2
  - No extra VRAM needed
- **Can increase:** To 4 if training is noisy

#### `--gradient_clip_val 1.0`
- **What it does:** Prevents exploding gradients
- **Why 1.0:** Conservative, prevents instability
- **Keep as-is:** Works for all scenarios

---

### Regularization Parameters

#### `--speaker_dropout 0.05` and `--lyrics_dropout 0.05`
- **What it does:** Random removal of inputs during training
- **Why 0.05:** Light regularization
- **Benefits:**
  - Prevents overfitting
  - Improves generalization
  - Maintains detail (not too aggressive)
- **Don't exceed:** 0.2 (loses too much information)

#### `--weight_decay 0.01`
- **What it does:** L2 regularization on weights
- **Why 0.01:** Standard value, prevents overfitting
- **Keep as-is:** Proven effective

---

### Technical Parameters

#### `--precision bf16-mixed`
- **What it does:** Mixed precision training
- **Why bf16:**
  - Faster training
  - Lower memory usage
  - Maintains numerical stability
- **Requirement:** GPU with BF16 support (RTX 30xx+, RTX 40xx)
- **Alternative:** `16-mixed` (older GPUs) or `32-true` (slow but stable)

#### `--beta1 0.9` and `--beta2 0.99`
- **What it does:** Adam optimizer momentum parameters
- **Why these:** Standard values, proven effective
- **Keep as-is:** No need to change

---

## Quick Copy-Paste Commands

### For Automation Script:
```bash
cd D:\Ace-Step_Fork
python lora_trainer_automation_v2.1_FINAL.py

# When prompted, use these values:
Max steps: 20000
Learning rate: 5e-05
Warmup steps: 100
Save every N steps: 5000
Workers: 10
Accumulate grad batches: 2
Speaker dropout: 0.05
Lyrics dropout: 0.05
```

### For Manual Training:

**Production (20k):**
```bash
python trainer_new.py \
    --dataset_path "D:\Ace-Step_Fork\processed_data\mdmachine_prep" \
    --max_steps 20000 \
    --learning_rate 5e-05 \
    --warmup_steps 100 \
    --save_every_n_train_steps 5000 \
    --num_workers 10 \
    --accumulate_grad_batches 2 \
    --speaker_dropout 0.05 \
    --lyrics_dropout 0.05 \
    --gradient_clip_val 1.0 \
    --gradient_clip_algorithm norm \
    --precision bf16-mixed \
    --save_last 5 \
    --beta1 0.9 \
    --beta2 0.99 \
    --weight_decay 0.01 \
    --output_dir "D:\Ace-Step_Fork\runs" \
    --exp_name "mdmachine"
```

**Quick Test (10k):**
```bash
python trainer_new.py \
    --dataset_path "D:\Ace-Step_Fork\processed_data\mdmachine_prep" \
    --max_steps 10000 \
    --learning_rate 5e-05 \
    --warmup_steps 50 \
    --save_every_n_train_steps 2500 \
    --num_workers 10 \
    --accumulate_grad_batches 2 \
    --speaker_dropout 0.05 \
    --lyrics_dropout 0.05 \
    --gradient_clip_val 1.0 \
    --gradient_clip_algorithm norm \
    --precision bf16-mixed \
    --save_last 5 \
    --output_dir "D:\Ace-Step_Fork\runs" \
    --exp_name "mdmachine_test"
```

**Maximum Quality (35k):**
```bash
python trainer_new.py \
    --dataset_path "D:\Ace-Step_Fork\processed_data\mdmachine_prep" \
    --max_steps 35000 \
    --learning_rate 5e-05 \
    --warmup_steps 100 \
    --save_every_n_train_steps 7000 \
    --num_workers 10 \
    --accumulate_grad_batches 2 \
    --speaker_dropout 0.05 \
    --lyrics_dropout 0.05 \
    --gradient_clip_val 1.0 \
    --gradient_clip_algorithm norm \
    --precision bf16-mixed \
    --save_last 5 \
    --output_dir "D:\Ace-Step_Fork\runs" \
    --exp_name "mdmachine_max"
```

---

## Decision Tree

```text
Do you have trained model before?
├─ No → Use Tier 2 (10k) for first test
│       Then → Tier 1 (20k) for production
│
└─ Yes → Is this the final model?
         ├─ No → Use Tier 1 (20k)
         └─ Yes → Use Tier 3 (35k)
```

---

## Comparison Table

| Aspect | Tier 2 (10k) | Tier 1 (20k) | Tier 3 (35k) |
|--------|-------------|----------------|--------------|
| **Time** | 3-4 hrs | 6-8 hrs | 12-16 hrs |
| **Quality** | Good | Excellent | Maximum |
| **Disk Space** | 2GB | 3GB | 3GB |
| **Checkpoints** | 4 | 4 | 5 |
| **Use Case** | Testing | Production | Final model |
| **Risk** | Low | Very Low | Very Low |
| **Cost** | Low | Medium | High |

---

## Pro Tips

### Monitoring Training

Watch these indicators:
```bash
# Training loss should decrease
Step 0:     Loss ~1.5-2.0
Step 5k:    Loss ~0.8-1.2
Step 10k:   Loss ~0.5-0.8
Step 15k:   Loss ~0.4-0.7
Step 20k:   Loss ~0.3-0.6
```

If loss plateaus or increases:
- Normal: Small fluctuations
- Warning: No decrease after 5k steps
- Problem: Loss increasing consistently

### Checkpoint Selection

After training, test these checkpoints:
1. **Last checkpoint** (step 20k/35k)
2. **Middle checkpoint** (step 10k/17.5k)
3. **Early-late blend** (step 15k/21k)

Sometimes middle checkpoints provide better generalization.

### GPU Utilization

```bash
nvidia-smi -l 1
```

Should show:
- 90-100% GPU utilization
- Memory near capacity (but not OOM)
- <80% GPU = data loading bottleneck (increase num_workers)

---

## Troubleshooting

### Training Too Slow
```bash
# Increase workers
--num_workers 12  # (was 10)

# Check data location
# Should be on SSD, not HDD
```

### Out of Memory
```bash
# Increase accumulation
--accumulate_grad_batches 4  # (was 2)

# This simulates batch_size=4 without extra VRAM
```

### Loss Not Decreasing
```bash
# Try slightly higher LR
--learning_rate 7e-05  # (was 5e-05)

# Or increase warmup
--warmup_steps 200  # (was 100)
```

### Training Unstable
```bash
# Lower learning rate
--learning_rate 3e-05  # (was 5e-05)

# Or increase gradient clipping
--gradient_clip_val 0.5  # (was 1.0)
```

---

## Training Checklist

Before starting a production run (Tier 1 or 3):

- [ ] Dataset has 20+ high-quality audio files
- [ ] All files are normalized (LUFS ~-14.0)
- [ ] Custom tags added (e.g., "style_name")
- [ ] Dataset cleaned with lora_magician_pro.py
- [ ] At least 50GB free disk space
- [ ] GPU has at least 12GB VRAM
- [ ] Tested with Tier 2 (10k) first
- [ ] W&B disabled if not needed
- [ ] Ready to wait 6-16 hours

---

## Summary

**Default Recommendation: Tier 1 (20k steps)**

This configuration is:
- Proven reliable in production
- Excellent quality results
- Reasonable training time (6-8 hours)
- Balanced disk space usage
- Low risk, high reward

**Use Tier 2 (10k)** for testing/iteration  
**Use Tier 3 (35k)** for final polish

---

**Version:** 2.1  
**Last Updated:** 2025  
**Based On:** Successful training runs with proven results  
**Tested With:** MDMAchine LoRA models