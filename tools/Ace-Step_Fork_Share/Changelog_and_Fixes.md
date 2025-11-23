# ACE-Step LoRA Training Suite - Changelog

## Version 2.1 Final (2025 Release)

### New Features
* **Hybrid Captioning System:** Added an interactive menu in Step 1 to choose between:
    * **Qwen:** GPU-accelerated, high accuracy (Requires ~6GB VRAM).
    * **Ollama:** CPU-based, lower resource usage (Ideal for low-VRAM setups).
* **Integrated Tagging:** The `add_tag.py` utility is now seamlessly called within the automation pipeline (Step 1.5).
* **Smart Resume:** Added auto-detection for the latest checkpoint (`.ckpt`) and Run ID to fix W&B resuming issues.
* **Checkpoint Cleanup:** Added an interactive safety check before deleting old checkpoints to prevent data loss.

### Critical Bug Fixes
* **Variable Scope Fix:** Fixed a critical crash in the automation script where `ace_step_dir` was referenced before assignment.
* **Path Resolution:** Fixed issues where the script couldn't locate tools in the `Tools/` subdirectory.
* **Resume Logic:** Corrected the command flags passed to `trainer_new.py`. It now correctly passes both `--resume_ckpt_path` AND `--wandb_run_id`.

### Configuration Updates
* **Defaults:** Updated default training parameters to "Tier 1" standards (20k steps, 5e-5 LR, 5000 save interval).
* **Documentation:** Completely rewrote all guides for 2025, removing deprecated info and adding low-VRAM troubleshooting.

---

## Version 2.0 (Legacy)

### Improvements
* **W&B Control:** Added support for disabling Weights & Biases logging via environment variables.
* **Error Handling:** Improved error messages when audio files were missing or corrupt.
* **Structure:** Moved helper scripts into a dedicated `Tools/` directory.

### Fixes
* Fixed an issue where the script would crash if the `runs/` directory didn't exist.
* Corrected the `num_workers` calculation logic to prevent CPU bottlenecks.

---

## Migration Guide (v1.0 -> v2.1)

If you are coming from the original v1.0 script:

1. **Delete** the old `lora_trainer_automation.py`.
2. **Place** `lora_trainer_automation_v2.1_FINAL.py` in your root folder.
3. **Move** `add_tag.py`, `lora_magician_pro.py`, and `normalize_audio.py` into a `Tools/` folder.
4. **Update** `trainer_new.py` and `generate_prompts_lyrics.py` in your `ACE-Step/` folder with the versions included in this package.

**Note:** Old checkpoints from v1.0 are compatible, but you must manually point to them using the full path if you wish to resume.