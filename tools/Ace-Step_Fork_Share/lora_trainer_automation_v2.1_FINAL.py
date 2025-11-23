#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════
    LoRA Training Automation Script v2.1 - FINAL
    MDMAchine / ComfyUI_MD_Nodes
═══════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import subprocess
import platform
from pathlib import Path
from datetime import datetime
import json

# --- Tag Addition Function ---
def add_tags_to_prompts(data_directory: Path, tags_to_add: list[str], python_exec: Path, ace_step_fork_dir: Path) -> bool:
    """
    Add custom tags to all _prompt.txt files using the add_tag.py script.
    """
    add_tags_script = ace_step_fork_dir / "Tools" / "add_tag.py"
    
    # Check if add_tag.py exists
    if not add_tags_script.exists():
        print(f"\n⚠️  Warning: add_tag.py not found at {add_tags_script}")
        print("This script should be in your ACE-Step directory.")
        
        skip = input("\nSkip tag addition step? (Y/N): ").strip().lower()
        if skip == 'y':
            return True
        else:
            print("Please place add_tag.py in your ACE-Step directory and try again.")
            return False
    
    tags_str = ','.join(tags_to_add)
    command = [
        str(python_exec),
        str(add_tags_script),
        "--data_dir", str(data_directory),
        "--tags", tags_str
    ]
    
    print(f"\nAdding tags: {tags_str}")
    
    try:
        # FIXED: Ensure cwd is the root fork directory so imports work if needed
        result = subprocess.run(command, cwd=ace_step_fork_dir, check=True, capture_output=True, text=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error adding tags: {e}")
        print(e.stderr)
        return False

# --- Helper Function for Path Validation ---
def get_validated_path(prompt: str, default_path: Path, must_exist: bool = True, path_type: str = "dir") -> Path:
    while True:
        user_input = input(f"{prompt} (Default: {default_path}): ").strip()
        raw_path = user_input if user_input else default_path
        
        if not isinstance(raw_path, Path):
            raw_path = Path(raw_path)

        try:
            resolved_path = raw_path.resolve()
            path_is_valid = True
            if must_exist:
                if not resolved_path.exists():
                    print(f"Error: Path '{resolved_path}' does not exist.")
                    path_is_valid = False
                elif path_type == "dir" and not resolved_path.is_dir():
                    print(f"Error: Path '{resolved_path}' is not a directory.")
                    path_is_valid = False
                elif path_type == "file" and not resolved_path.is_file():
                    print(f"Error: Path '{resolved_path}' is not a file.")
                    path_is_valid = False
            
            if path_is_valid:
                return resolved_path

        except Exception as e:
            print(f"Error: Invalid path format: {e}")

# --- Helper Functions for Input Types ---
def get_int_input(prompt: str, default: int) -> int:
    while True:
        user_input = input(f"{prompt} (Default: {default}): ").strip()
        if not user_input:
            return default
        try:
            return int(user_input)
        except ValueError:
            print("Invalid input. Please enter an integer.")

def get_float_input(prompt: str, default: float) -> float:
    while True:
        user_input = input(f"{prompt} (Default: {default}): ").strip()
        if not user_input:
            return default
        try:
            return float(user_input)
        except ValueError:
            print("Invalid input. Please enter a floating-point number.")

def get_boolean_input(prompt: str, default: bool) -> bool:
    default_str = 'Y' if default else 'N'
    while True:
        user_input = input(f"{prompt} (Y/N) (Default: {default_str}): ").strip().lower()
        if not user_input:
            return default
        if user_input in ['y', 'yes']:
            return True
        if user_input in ['n', 'no']:
            return False
        print("Invalid input. Please enter 'Y' or 'N'.")

def get_string_input(prompt: str, default: str) -> str:
    user_input = input(f"{prompt} (Default: {default}): ").strip()
    return user_input if user_input else default

# --- Centralized Command Execution Function ---
def run_script_command(command_parts: list[str], log_file_path: Path, cwd_dir: Path) -> bool:
    cmd_str = ' '.join(str(p) for p in command_parts)
    print(f"\n{'='*80}")
    print(f"Executing: {cmd_str}")
    print(f"{'='*80}\n")

    try:
        with open(log_file_path, 'a', encoding='utf-8') as f:
            f.write(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Executing: {cmd_str}\n")
            
        # Using shell=True on Windows can sometimes help with path resolution, but usually not needed if full paths provided
        process = subprocess.run(command_parts, cwd=cwd_dir, text=True, check=False)
        
        with open(log_file_path, 'a', encoding='utf-8') as f:
            f.write(f"Command finished with exit code {process.returncode}\n")

        if process.returncode != 0:
            print(f"\n{'!'*80}")
            print(f"ERROR: Command failed with exit code {process.returncode}")
            print(f"Check '{log_file_path}' for details.")
            print(f"{'!'*80}\n")
            return False
        return True
    except Exception as e:
        print(f"Error executing command: {e}", file=sys.stderr)
        with open(log_file_path, 'a', encoding='utf-8') as f:
            f.write(f"Error: {e}\n")
        return False

def find_latest_checkpoint(checkpoints_dir: Path) -> tuple[Path | None, str | None]:
    if not checkpoints_dir.exists():
        return None, None
    
    run_dirs = [d for d in checkpoints_dir.iterdir() if d.is_dir()]
    if not run_dirs:
        return None, None
    
    # Sort by modification time to get latest
    latest_run_dir = sorted(run_dirs, key=lambda p: p.stat().st_mtime, reverse=True)[0]
    resume_id = latest_run_dir.name
    
    checkpoints_subdir = latest_run_dir / "checkpoints"
    if not checkpoints_subdir.exists():
        return None, None
    
    # Find the latest .ckpt file
    ckpt_files = list(checkpoints_subdir.glob("epoch=*-step=*.ckpt"))
    if not ckpt_files:
        return None, None
    
    def extract_step(ckpt_path):
        try:
            name = ckpt_path.stem
            step_part = name.split("step=")[1]
            return int(step_part.split("-")[0] if "-" in step_part else step_part)
        except:
            return 0
    
    latest_ckpt = sorted(ckpt_files, key=extract_step, reverse=True)[0]
    return latest_ckpt, resume_id

def find_latest_lora_weights(ace_step_fork_dir: Path, base_output_dir: Path, exp_name: str) -> Path | None:
    old_checkpoints_dir = ace_step_fork_dir / "ACE-Step" / "checkpoints"
    new_checkpoints_dir = base_output_dir / "lora_checkpoints"
    
    for checkpoints_dir in [new_checkpoints_dir, old_checkpoints_dir]:
        if not checkpoints_dir.exists():
            continue
            
        # Find all LoRA checkpoint directories
        lora_dirs = sorted(
            [d for d in checkpoints_dir.iterdir() 
             if d.is_dir() and "epoch=" in d.name and "step=" in d.name and "_lora" in d.name],
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        
        if lora_dirs:
            latest_lora_dir = lora_dirs[0]
            safetensors_path = latest_lora_dir / "pytorch_lora_weights.safetensors"
            if safetensors_path.is_file():
                return safetensors_path
    return None

def main():
    print("""
═══════════════════════════════════════════════════════════════════════════
    ACE-Step LoRA Training Automation v2.1 FINAL
    MDMAchine / ComfyUI_MD_Nodes
    Based off: https://github.com/woct0rdho/ACE-Step
═══════════════════════════════════════════════════════════════════════════
""")
    
    script_dir = Path(__file__).resolve().parent
    
    # Default paths
    default_ace_step_fork_dir = script_dir
    default_data_source_dir = Path("D:/Desktop/MDMAchine/normalized/MP3_Files")
    # FIXED: Variable initialization order
    default_base_output_dir = default_ace_step_fork_dir / "runs"
    
    log_filename = f"lora_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    error_log_path = script_dir / log_filename
    print(f"Logging to: {error_log_path}\n")
    
    with open(error_log_path, 'w', encoding='utf-8') as f:
        f.write(f"ACE-Step LoRA Training Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
    
    # ========================================================================
    # CONFIGURATION SECTION
    # ========================================================================
    
    print("\n--- Path Configuration ---")
    ace_step_fork_dir = get_validated_path(
        "Enter the path to your Ace-Step_Fork directory",
        default_ace_step_fork_dir
    )
    
    # Detect Python executable
    if platform.system() == "Windows":
        python_exec = ace_step_fork_dir / "ACE-Step" / "Scripts" / "python.exe"
        if not python_exec.is_file():
            python_exec = ace_step_fork_dir / "ACE-Step" / "venv" / "Scripts" / "python.exe"
    else:
        python_exec = ace_step_fork_dir / "ACE-Step" / "bin" / "python"
        if not python_exec.is_file():
            python_exec = ace_step_fork_dir / "ACE-Step" / "venv" / "bin" / "python"
    
    if not python_exec.is_file():
        print(f"Error: Python executable not found at {python_exec}")
        input("Press Enter to exit.")
        sys.exit(1)
    
    print(f"Using Python: {python_exec}")
    
    print("\n--- Project Configuration ---")
    project_name = get_string_input("Enter project name", "mdmachine")
    
    mp3_files_dir = get_validated_path(
        "Enter the path to your MP3 files directory",
        default_data_source_dir
    )
    
    base_output_dir = get_validated_path(
        "Enter the base output directory for runs",
        default_base_output_dir,
        must_exist=False
    )
    base_output_dir.mkdir(parents=True, exist_ok=True)
    
    processed_data_base = ace_step_fork_dir / "processed_data"
    processed_data_base.mkdir(exist_ok=True)
    
    hf_output_name = processed_data_base / f"{project_name}_filenames"
    prep_output_dir = processed_data_base / f"{project_name}_prep"
    
    # ========================================================================
    # PIPELINE STEPS
    # ========================================================================
    
    print("\n--- Pipeline Step Selection ---")
    print("1. Prompt Generation (Qwen/Ollama)")
    print("2. Dataset Cleanup (LoRA Magician)")
    print("3. Create HuggingFace Dataset")
    print("4. Preprocess Audio Data")
    print("5. LoRA Training")
    print("6. Adjust LoRA Strength")
    
    start_step = get_int_input("Enter the step number to start from", 1)
    initial_selected_step = start_step
    
    # --- Resume Config ---
    resume = False
    resume_ckpt_path = None
    resume_id = None
    
    if start_step >= 5:
        resume = get_boolean_input("Do you want to resume from a previous checkpoint?", False)
        if resume:
            ace_step_lora_dir = ace_step_fork_dir / "ACE-Step" / "ace_step_lora"
            auto_ckpt, auto_id = find_latest_checkpoint(ace_step_lora_dir)
            
            if auto_ckpt:
                print(f"\nFound latest checkpoint: {auto_ckpt}")
                print(f"Resume ID: {auto_id}")
                if get_boolean_input("Use this checkpoint?", True):
                    resume_ckpt_path = auto_ckpt
                    resume_id = auto_id
            
            if not resume_ckpt_path:
                resume_ckpt_path = get_validated_path(
                    "Enter the path to the checkpoint (.ckpt) file",
                    ace_step_lora_dir / "latest" / "checkpoints" / "last.ckpt",
                    path_type="file"
                )
                resume_id = get_string_input("Enter the W&B run ID", "")
    
    # --- Training Parameters ---
    print("\n--- Training Parameters ---")
    max_training_steps = get_int_input("Maximum training steps", 20000)
    num_workers = get_int_input("Number of data loading workers", 10)
    
    print("\n--- Optimizer Settings ---")
    learning_rate = get_float_input("Learning rate", 5e-5)
    adam_beta1 = get_float_input("Adam beta1", 0.9)
    adam_beta2 = get_float_input("Adam beta2", 0.99)
    weight_decay = get_float_input("Weight decay", 0.01)
    warmup_steps = get_int_input("Warmup steps", 100)
    accumulate_grad_batches = get_int_input("Gradient accumulation batches", 2)
    gradient_clip_val = get_float_input("Gradient clip value", 1.0)
    gradient_clip_algorithm = get_string_input("Gradient clip algorithm (norm/value)", "norm")
    
    print("\n--- Training Settings ---")
    precision = get_string_input("Training precision (bf16-mixed/16-mixed/32-true)", "bf16-mixed")
    save_every_n_train_steps = get_int_input("Save checkpoint every N steps", 5000)
    save_last = get_int_input("Keep last N LoRA checkpoints", 5)
    
    print("\n--- Dropout Settings ---")
    speaker_dropout = get_float_input("Speaker dropout rate", 0.05)
    lyrics_dropout = get_float_input("Lyrics dropout rate", 0.05)
    
    print("\n--- Logging Settings ---")
    disable_wandb = get_boolean_input("Disable Weights & Biases logging?", False)
    
    run_adjust_lora_strength = False
    if initial_selected_step <= 5:
        run_adjust_lora_strength = get_boolean_input("Adjust LoRA strength after training?", True)
    
    print("\n" + "="*80)
    print("Starting LoRA Training Pipeline...")
    print("="*80 + "\n")
    
    # --- Step 1: Prompt Generation ---
    if start_step <= 1:
        print("\n--- Step 1: Generating Prompts ---")
        
        # NEW: Interactive Model Selection
        print("\nChoose a model for captioning:")
        print("1. Qwen (Recommended) - Uses GPU VRAM. Faster, better quality.")
        print("2. Ollama - Uses System RAM/CPU. Slower, but saves VRAM.")
        model_choice = input("Enter choice (1/2) [Default: 1]: ").strip()
        
        use_ollama = (model_choice == '2')
        model_type = "ollama" if use_ollama else "qwen"
        print(f"Selected: {model_type.upper()}\n")
        
        use_lyrics = get_boolean_input("Include lyrics in prompts?", True)
        
        command = [
            str(python_exec),
            "generate_prompts_lyrics.py",
            "--data_dir", str(mp3_files_dir),
            "--model_type", model_type
        ]
        
        if use_lyrics:
            command.append("--lyrics")
        
        if use_ollama:
            ollama_model_name = input("Enter Ollama model tag (Press Enter for default 'qwen2.5'): ").strip()
            if ollama_model_name:
                command.extend(["--ollama_model", ollama_model_name])

        if not run_script_command(command, error_log_path, ace_step_fork_dir / "ACE-Step"):
            input("Press Enter to exit.")
            sys.exit(1)
        
        print("✓ Prompt generation complete.")
        input("Press Enter to continue...")
    else:
        print("\n✗ Skipping Prompt Generation (starting from step {})".format(start_step))
    
    # --- Step 1.5: Tags ---
    if start_step <= 1:
        print("\n--- Step 1.5: Add Custom Tags (Optional) ---")
        if get_boolean_input("Add custom tags to prompts?", True):
            tags_input = input("Enter comma-separated tags (e.g., 'mdmachine, synthwave'): ").strip()
            if tags_input:
                tags_list = [tag.strip() for tag in tags_input.split(',') if tag.strip()]
                if tags_list:
                    if not add_tags_to_prompts(mp3_files_dir, tags_list, python_exec, ace_step_fork_dir):
                        print("Tag addition failed, but continuing with pipeline...")
                    else:
                        print("✓ Tags added successfully.")
                else:
                    print("No valid tags provided, skipping...")
            input("Press Enter to continue...")
        else:
            print("✗ Skipping tag addition.")
    else:
        print("\n✗ Skipping Tag Addition (starting from step {})".format(start_step))
    
    # --- Step 2: Cleanup ---
    if start_step <= 2:
        print("\n--- Step 2: Dataset Cleanup ---")
        if get_boolean_input("Run LoRA Magician cleanup?", True):
            lora_magician_script_path = ace_step_fork_dir / "Tools" / "lora_magician_pro.py"
            
            if not lora_magician_script_path.is_file():
                print(f"Warning: LoRA Magician script not found at {lora_magician_script_path}")
                alt_path = get_validated_path("Enter path to lora_magician_pro.py", lora_magician_script_path, path_type="file")
                lora_magician_script_path = alt_path
            
            command = [str(python_exec), str(lora_magician_script_path)]
            if not run_script_command(command, error_log_path, ace_step_fork_dir / "ACE-Step"):
                input("Press Enter to exit.")
                sys.exit(1)
            print("✓ Dataset cleanup complete.")
            input("Press Enter to continue...")
        else:
            print("✗ Skipping dataset cleanup.")
    else:
        print("\n✗ Skipping Dataset Cleanup (starting from step {})".format(start_step))
    
    # --- Step 3: HF Dataset ---
    if start_step <= 3:
        print("\n--- Step 3: Creating HuggingFace Dataset ---")
        command = [
            str(python_exec),
            "convert2hf_dataset_new.py",
            "--data_dir", str(mp3_files_dir),
            "--output_name", str(hf_output_name)
        ]
        if not run_script_command(command, error_log_path, ace_step_fork_dir / "ACE-Step"):
            input("Press Enter to exit.")
            sys.exit(1)
        print(f"✓ HuggingFace dataset created: {hf_output_name}")
        input("Press Enter to continue...")
    else:
        print("\n✗ Skipping HuggingFace Dataset Creation (starting from step {})".format(start_step))
    
    # --- Step 4: Preprocess ---
    if start_step <= 4:
        print("\n--- Step 4: Preprocessing Audio Data ---")
        command = [
            str(python_exec),
            "preprocess_dataset_new.py",
            "--input_name", str(hf_output_name),
            "--output_dir", str(prep_output_dir)
        ]
        if not run_script_command(command, error_log_path, ace_step_fork_dir / "ACE-Step"):
            input("Press Enter to exit.")
            sys.exit(1)
        print(f"✓ Audio preprocessing complete: {prep_output_dir}")
        input("Press Enter to continue...")
    else:
        print("\n✗ Skipping Audio Preprocessing (starting from step {})".format(start_step))
    
    # --- Checkpoint Cleanup ---
    if start_step <= 5 and not resume:
        print("\n" + "="*80)
        print("--- Checkpoint Cleanup Reminder ---")
        print("="*80)
        checkpoints_dir = ace_step_fork_dir / "ACE-Step" / "checkpoints"
        ace_step_lora_dir = ace_step_fork_dir / "ACE-Step" / "ace_step_lora"
        
        print(f"\nBefore starting NEW training (not resuming), consider clearing:")
        print(f"  • {checkpoints_dir}")
        print(f"  • {ace_step_lora_dir}")
        
        if get_boolean_input("\nClear old checkpoints now?", False):
            import shutil
            if checkpoints_dir.exists():
                try:
                    shutil.rmtree(checkpoints_dir)
                    checkpoints_dir.mkdir()
                    print(f"  ✓ Cleared {checkpoints_dir}")
                except Exception as e:
                    print(f"  ⚠️  Could not clear {checkpoints_dir}: {e}")
            if ace_step_lora_dir.exists():
                if get_boolean_input("  Also clear ace_step_lora directory?", False):
                    try:
                        shutil.rmtree(ace_step_lora_dir)
                        ace_step_lora_dir.mkdir()
                        print(f"  ✓ Cleared {ace_step_lora_dir}")
                    except Exception as e:
                        print(f"  ⚠️  Could not clear {ace_step_lora_dir}: {e}")
        print("="*80 + "\n")
        input("Press Enter to continue...")
    
    # --- Step 5: Training ---
    if start_step <= 5:
        print("\n" + "="*80)
        print("--- Step 5: LoRA Training ---")
        print(f"Training for {max_training_steps} steps with {num_workers} workers")
        print("="*80 + "\n")
        
        command = [
            str(python_exec),
            "trainer_new.py",
            "--dataset_path", str(prep_output_dir),
            "--max_steps", str(max_training_steps),
            "--num_workers", str(num_workers),
            "--output_dir", str(base_output_dir),
            "--exp_name", project_name,
            "--learning_rate", str(learning_rate),
            "--beta1", str(adam_beta1),
            "--beta2", str(adam_beta2),
            "--weight_decay", str(weight_decay),
            "--warmup_steps", str(warmup_steps),
            "--accumulate_grad_batches", str(accumulate_grad_batches),
            "--gradient_clip_val", str(gradient_clip_val),
            "--gradient_clip_algorithm", gradient_clip_algorithm,
            "--precision", precision,
            "--save_every_n_train_steps", str(save_every_n_train_steps),
            "--save_last", str(save_last),
            "--speaker_dropout", str(speaker_dropout),
            "--lyrics_dropout", str(lyrics_dropout),
        ]
        
        if resume and resume_ckpt_path:
            command.extend(["--resume_full_ckpt_path", str(resume_ckpt_path)])
            if resume_id:
                command.extend(["--wandb_run_id", resume_id])
        
        if disable_wandb:
            env = os.environ.copy()
            env["WANDB_MODE"] = "disabled"
            # Note: To support env vars, we'd need to modify run_script_command. 
            # For now, user sets it globally or script assumes standard exec.
        
        if not run_script_command(command, error_log_path, ace_step_fork_dir / "ACE-Step"):
            input("Press Enter to exit.")
            sys.exit(1)
        
        print("\n✓ LoRA Training complete!")
        if initial_selected_step <= 5:
            if get_boolean_input("\nAdjust LoRA strength now?", True):
                run_adjust_lora_strength = True
        input("Press Enter to continue...")
    else:
        print("\n✗ Skipping LoRA Training (starting from step {})".format(start_step))
    
    # --- Step 6: Strength ---
    if run_adjust_lora_strength or start_step == 6:
        print("\n" + "="*80)
        print("--- Step 6: Adjusting LoRA Strength ---")
        print("="*80 + "\n")
        
        input_safetensors = find_latest_lora_weights(ace_step_fork_dir, base_output_dir, project_name)
        
        if not input_safetensors:
            print("Error: Could not find LoRA weights file.")
            print(f"Searched in:")
            print(f"  - {base_output_dir / 'lora_checkpoints'}")
            # FIXED: Changed ace_step_dir to ace_step_fork_dir / "ACE-Step"
            print(f"  - {ace_step_fork_dir / 'ACE-Step' / 'checkpoints'}")
            
            manual_path = input("\nEnter path to pytorch_lora_weights.safetensors (or press Enter to skip): ").strip()
            if manual_path:
                input_safetensors = Path(manual_path)
                if not input_safetensors.is_file():
                    print("File not found. Skipping strength adjustment.")
                    input_safetensors = None
        
        if input_safetensors:
            print(f"Found LoRA weights: {input_safetensors}")
            step_match = None
            for part in input_safetensors.parts:
                if "step=" in part:
                    try:
                        step_match = part.split("step=")[1].split("_")[0]
                    except:
                        pass
            
            output_name = f"{project_name}_{step_match}k_adjusted.safetensors" if step_match else f"{project_name}_adjusted.safetensors"
            lora_config_path = ace_step_fork_dir / "ACE-Step" / "config" / "lora_config_transformer_only.json"
            
            if not lora_config_path.is_file():
                print(f"Error: LoRA config not found at {lora_config_path}")
                input("Press Enter to exit.")
                sys.exit(1)
            
            command = [
                str(python_exec),
                str(ace_step_fork_dir / "ACE-Step" / "add_alpha_in_lora.py"),
                "--input_name", str(input_safetensors),
                "--output_name", output_name,
                "--lora_config_path", str(lora_config_path)
            ]
            
            if not run_script_command(command, error_log_path, ace_step_fork_dir / "ACE-Step"):
                input("Press Enter to exit.")
                sys.exit(1)
            
            # FIXED: Changed ace_step_dir to ace_step_fork_dir / "ACE-Step"
            output_path = ace_step_fork_dir / "ACE-Step" / output_name
            print(f"\n✓ LoRA strength adjustment complete!")
            print(f"  Output: {output_path}")
            input("Press Enter to continue...")
        else:
            print("✗ Skipping strength adjustment.")
    else:
        print("\n✗ Skipping LoRA Strength Adjustment")
    
    print("\n" + "="*80 + "\nLoRA Training Pipeline Complete!\n" + "="*80)
    print(f"\nLog file: {error_log_path}")
    input("Press Enter to exit.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        input("Press Enter to exit.")
        sys.exit(1)