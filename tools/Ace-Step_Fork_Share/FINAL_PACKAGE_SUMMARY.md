# ACE-Step LoRA Training Suite v2.1 - Package Summary

## Complete Toolkit Overview

### Core Scripts

1. **lora_trainer_automation_v2.1_FINAL.py** (Root Directory)
   - **Purpose:** The main controller. Orchestrates the entire 6-step pipeline.
   - **Key Features:** Interactive configuration, auto-resume detection, hybrid captioning selection (Qwen/Ollama).

2. **add_tag.py** (Tools Directory)
   - **Purpose:** Recursively injects style tags into training prompts.
   - **Key Features:** Duplicate prevention, case-insensitive matching.

3. **lora_magician_pro.py** (Tools Directory)
   - **Purpose:** Dataset hygiene and validation.
   - **Key Features:** Sorts empty/short/long prompts, creates safety backups.

4. **normalize_audio.py** (Tools Directory)
   - **Purpose:** Audio standardization.
   - **Key Features:** LUFS normalization (-14.0), format conversion, metadata preservation.

### Modified ACE-Step Files

5. **trainer_new.py** (ACE-Step Directory)
   - **Purpose:** The training engine.
   - **Modifications:** Fixed resume logic (`checkpoint.clear()` bug), exposed advanced parameters (dropout, accum_grad).

6. **generate_prompts_lyrics.py** (ACE-Step Directory)
   - **Purpose:** Caption generation.
   - **Modifications:** Added support for Ollama (CPU) fallback and robust JSON parsing.

---

## Workflow Integration

### Pre-Processing
Run `normalize_audio.py` first to standardize your raw audio inputs. This ensures the model learns musical features rather than volume differences.

### Training
Run `lora_trainer_automation_v2.1_FINAL.py`. It will automatically call the tools in the `Tools/` directory and the scripts in the `ACE-Step/` directory in the correct order.

### Directory Structure
For the suite to function correctly, maintain this hierarchy:

```text
D:\Ace-Step_Fork\
├── lora_trainer_automation_v2.1_FINAL.py
├── Tools\
│   ├── add_tag.py
│   ├── lora_magician_pro.py
│   └── normalize_audio.py
└── ACE-Step\
    ├── trainer_new.py
    └── generate_prompts_lyrics.py
```

---

## Version Comparison

| Feature | v1.0 | v2.1 Final |
|---------|------|------------|
| Resume Capability | Broken | Fixed (Auto-detect) |
| Captioning | Qwen Only | Hybrid (Qwen + Ollama) |
| Tagging | Manual | Integrated |
| Cleanup | Manual | Integrated |
| Documentation | Basic | Comprehensive Suite |
| Stability | Low | Production Ready |

---

**MDMAchine / ComfyUI_MD_Nodes**
*2025 Release*