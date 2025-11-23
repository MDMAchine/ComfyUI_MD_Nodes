# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# ████ ComfyUI Project Updater v4.3.7 - Enhanced Edition  ████▓▒░
# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
# ░▒▓ ORIGIN & DEV:
#  • Crafted with care by MDMAchine / MD_Nodes
#  • Enhanced by Claude, Gemini, etc.
#  • License: Apache 2.0 — do what you want, just give credit
# ░▒▓ DESCRIPTION:
#  A professional-grade CLI tool for ComfyUI project management with
#  enhanced dynamic node detection and interactive menu system.
# ░▒▓ NEW IN v4.3.7:
#  ✓ CRITICAL FIX (Backwards Comp.): The generated __init__.py now uses the 
#   original NODE_CLASS_MAPPINGS key (e.g., "APGGuiderForked") instead of 
#   the Python class name (e.g., "APGGuiderNode") for the dictionary key, 
#     ensuring compatibility with existing user workflows.
#  ✓ FIXED (Static Parser): Upgraded the "Static Fallback Parser"
#   (`extract_node_mappings_static`) to support type-hinted assignments.
#  ✓ FIXED: `NameError` crash in `main()` from v4.3.6 re-launch guard.
# ░▒▓ FROM v4.3.6:
#  ✓ ADDED: "Re-launch Guard" to auto-run script in the correct venv.
# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄

import os
import sys
import json
import ast
import shutil
import subprocess
from pathlib import Path
from datetime import datetime
import pkgutil
import argparse
import hashlib
import importlib.util
from typing import Any, Dict, List, Optional, Set
from types import ModuleType # Needed for mock modules

# --- !! START VENV GUARD CONFIGURATION !! ---
# Path to your ComfyUI virtual environment's Python executable
VENV_PYTHON_EXE = r"C:\Stable Diffusion\venvs\comfy_py313\Scripts\python.exe"
# --- !! END VENV GUARD CONFIGURATION !! ---

# Try to import optional libraries
try:
  import tomli
  import tomli_w
  HAS_TOML = True
except ImportError:
  HAS_TOML = False

try:
  from tqdm import tqdm
  HAS_TQDM = True
except ImportError:
  HAS_TQDM = False

# =================================================================================
# --- CONFIGURATION MANAGEMENT ---
# =================================================================================

PROJECT_CONFIG_FILENAME = "project_updater_config.json"
UPDATER_SETTINGS_FILENAME = "updater_settings.json"

DEFAULT_PROJECT_CONFIG = {
  "project_version": "1.0.0",
  "import_to_package_map": {"PIL": "Pillow", "cv2": "opencv-python", "yaml": "PyYAML", "skimage": "scikit-image", "sklearn": "scikit-learn", "torchaudio": "torchaudio"},
  "manual_imports": ["filelock", "fsspec", "Jinja2", "librosa", "MarkupSafe", "mpmath", "networkx", "ollama", "pedalboard", "piexif", "pyloudnorm", "requests", "scipy", "sympy", "tokenizers", "tqdm", "imageio", "av"],
  "denylist_packages": ["torch", "numpy", "Pillow", "PyYAML", "torchaudio", "transformers", "opencv-python", "comfy", "nodes", "folder_paths", "setuptools", "wheel", "typing_extensions"],
  "exclude_dirs": [".git", ".cache", ".pytest_cache", ".mypy_cache", ".ruff_cache", "__pycache__", "venv", ".venv", "env", ".env", "node_modules", ".ipynb_checkpoints", ".project_backups", "misc", "Internal", "docs", "tests", "tools", "build", "dist", "*.egg-info"],
  "exclude_files": ["update_project.py", "__init__.py", "README.md", "LICENSE"],
  "exclude_extensions": [".rar", ".zip", ".7z", ".bak"],
  "gitignore_entries": ["# Python cache", "__pycache__/", "*.py[cod]", "*$py.class", "# Virtual environments", "venv/", ".venv/", "env/", ".env/", "# IDE files", ".vscode/", ".idea/", "*.swp", "# OS files", ".DS_Store", "Thumbs.db", "# Backups", ".project_backups/", "# ComfyUI specific", ".ipynb_checkpoints/", "# Large files", "*.bin", "*.safetensors", "*.ckpt", "*.pt", "*.pth", "# Local configs", "*.ini", "*.local", "project_updater_config.json", "updater_settings.json"]
}

def get_updater_settings_path() -> Path:
  return Path(__file__).resolve().parent / UPDATER_SETTINGS_FILENAME

def load_updater_settings() -> Dict[str, Any]:
  settings_path = get_updater_settings_path()
  if not settings_path.exists(): return {"last_project_path": "", "file_cache": {}}
  try:
    with open(settings_path, 'r', encoding='utf-8') as f:
      settings = json.load(f)
      if "file_cache" not in settings or not isinstance(settings.get("file_cache"), dict):
        settings["file_cache"] = {}
      return settings
  except (json.JSONDecodeError, IOError):
    return {"last_project_path": "", "file_cache": {}}

def save_updater_settings(settings: Dict[str, Any]):
  settings_path = get_updater_settings_path()
  try:
    with open(settings_path, 'w', encoding='utf-8') as f: json.dump(settings, f, indent=4)
  except IOError as e: print(f"⚠️ Could not save updater settings: {e}")

def load_or_create_project_config(project_root: Path) -> Dict[str, Any]:
  config_path = project_root / PROJECT_CONFIG_FILENAME
  if not config_path.exists():
    print(f"👋 Project config not found. Creating a default `{PROJECT_CONFIG_FILENAME}`.")
    with open(config_path, 'w', encoding='utf-8') as f: json.dump(DEFAULT_PROJECT_CONFIG, f, indent=4)
    return DEFAULT_PROJECT_CONFIG
  try:
    with open(config_path, 'r', encoding='utf-8') as f: config = json.load(f)
  except (json.JSONDecodeError, IOError) as e:
    print(f"⚠️ Error loading {PROJECT_CONFIG_FILENAME}: {e}. Using defaults.")
    return DEFAULT_PROJECT_CONFIG.copy()

  updated = False
  for key, value in DEFAULT_PROJECT_CONFIG.items():
    if key not in config:
      config[key] = value
      updated = True
  return config

def validate_config(config: Dict[str, Any]) -> List[str]:
  issues = []
  for key, expected_type in [("denylist_packages", list), ("exclude_dirs", list), ("exclude_files", list), ("exclude_extensions", list)]:
    if key in config and not isinstance(config.get(key), expected_type):
      issues.append(f"Config Error: '{key}' must be a list.")
  if not issues:
    print(" ✓ Configuration validated successfully")
  return issues

# =================================================================================
# --- VERSIONING, UI, FILE I/O ---
# =================================================================================

def handle_versioning(args: argparse.Namespace, current_version: str) -> str:
  if args.set_version: return args.set_version
  if args.bump:
    try:
      major, minor, patch = map(int, current_version.split('.'))
      if args.bump == 'major': major += 1; minor, patch = 0, 0
      elif args.bump == 'minor': minor += 1; patch = 0
      elif args.bump == 'patch': patch += 1
      else: return current_version
      new_version = f"{major}.{minor}.{patch}"
      print(f"🚀 Bumping version: {current_version} -> {new_version}")
      return new_version
    except ValueError:
      print(f"⚠️ Could not parse version '{current_version}'. Skipping bump.")
      return current_version
  return current_version

def parse_arguments() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Update ComfyUI project files automatically")
  parser.add_argument("--project-root", type=str, help="Path to the custom node project root directory.")
  parser.add_argument("--dry-run", action="store_true", help="Perform a dry run without modifying files.")
  parser.add_argument("--git-commit", action="store_true", help="Automatically commit changes to Git.")
  parser.add_argument("--no-backup", action="store_true", help="Skip creating a backup of modified files.")
  parser.add_argument("--interactive", action="store_true", help="Force interactive mode even if other flags are set.")
  parser.add_argument("--debug", action="store_true", help="Enable detailed debug output during parsing.")
  parser.add_argument("--set-version", type=str, help="Set a specific project version (e.g., '1.5.0').")
  parser.add_argument("--bump", choices=['major', 'minor', 'patch'], help="Increment project version (major, minor, or patch).")
  parser.add_argument("--pin-versions", action="store_true", help="Preserve existing version numbers in requirements.txt.")
  parser.add_argument("--clear-cache", action="store_true", help="Clear the file cache and re-parse all files.")
  return parser.parse_args()

def print_header():
  print("\n" + "█" * 70)
  print("█" + "   ComfyUI Project Updater v4.3.7 - BY MDMACHINE    ".center(68) + "█")
  print("█" + "   Automate Your Node Management & Release Workflow   ".center(68) + "█")
  print("█" * 70 + "\n")

def print_section_header(title: str, icon: str = "▓"):
  print("\n" + "▓" * 70); print(f"▓▓▓ {icon} {title}"); print("▓" * 70)

def get_file_hash(path: Path) -> str:
  hash_md5 = hashlib.md5()
  try:
    with open(path, "rb") as f:
      for chunk in iter(lambda: f.read(4096), b""): hash_md5.update(chunk)
    return hash_md5.hexdigest()
  except Exception: return ""

def prompt_for_project_root(updater_settings: Dict[str, Any]) -> Path:
  print("\n" + "=" * 70)
  print("📁 PROJECT ROOT SELECTION")
  print("=" * 70)

  last_path = updater_settings.get('last_project_path', '')

  while True:
    if last_path and Path(last_path).is_dir():
      print(f"\n💾 Last used project:")
      print(f" {last_path}")
      use_last = input("\n Use this path? (Y/n): ").strip().lower()
      if use_last in ('', 'y', 'yes'):
        print(f" ✓ Using: {last_path}")
        return Path(last_path).resolve()

    print("\n🗂️ Enter the project root path:")
    print(" Examples:")
    print(" - C:\\Stable Diffusion\\ComfyUI_Base\\custom_nodes\\ComfyUI_MD_Nodes")
    print(" - C:\\Projects\\MyComfyNodes")
    print(" - . (for current directory)")
    print()

    user_path = input(" Path: ").strip()

    if not user_path:
      print(" ⚠️ No path entered. Please provide a valid path.")
      last_path = ""
      continue

    if user_path == '.':
      resolved_path = Path.cwd().resolve()
    else:
      user_path = user_path.strip('"').strip("'")
      resolved_path = Path(user_path).resolve()

    if not resolved_path.exists():
      print(f" ❌ Path does not exist: {resolved_path}")
      retry = input(" Try again? (Y/n): ").strip().lower()
      if retry == 'n' or retry == 'no':
        print("\n❌ Operation cancelled.")
        sys.exit(1)
      continue

    if not resolved_path.is_dir():
      print(f" ❌ Path is not a directory: {resolved_path}")
      retry = input(" Try again? (Y/n): ").strip().lower()
      if retry == 'n' or retry == 'no':
        print("\n❌ Operation cancelled.")
        sys.exit(1)
      continue

    print(f"\n ✓ Found directory: {resolved_path}")

    has_init = (resolved_path / "__init__.py").exists()
    py_files = list(resolved_path.glob("*.py"))
    subdirs = [d for d in resolved_path.iterdir() if d.is_dir() and not d.name.startswith('.')]

    print(f" 📊 Contains:")
    print(f"   - __init__.py: {'✓ Found' if has_init else '✗ Not found'}")
    print(f"   - Python files: {len(py_files)} in root")
    print(f"   - Subdirectories: {len(subdirs)}")

    if not has_init and len(py_files) == 0 and len(subdirs) == 0:
      print(f" ⚠️ Warning: This doesn't look like a ComfyUI node directory")
      print(f"   (No __init__.py, .py files, or subdirectories found)")

    confirm = input("\n Use this directory? (Y/n): ").strip().lower()
    if confirm in ('', 'y', 'yes'):
      print(f" ✓ Selected: {resolved_path}")
      return resolved_path
    else:
      print(" → Let's try again...")
      last_path = ""

def show_preset_menu() -> Optional[Dict[str, Any]]:
  """Show quick preset options for common scenarios"""
  print_section_header("QUICK PRESETS", "🎯")
  print("\n Choose a preset or customize manually:")
  print()
  print(" 1. 🔍 Full Re-scan (Clear cache + Debug)")
  print("   → Clear cache, debug mode, no changes made")
  print()
  print(" 2. 🚀 Quick Update")
  print("   → Standard update with backup")
  print()
  print(" 3. 📦 Release Build (Bump + Commit)")
  print("   → Bump patch version, create backup, commit to git")
  print()
  print(" 4. 🐞 Debug Mode")
  print("   → Dry run with debug output, no changes")
  print()
  print(" 5. ⚙️ Custom (Manual configuration)")
  print("   → Choose all options manually")
  print()
  print(" 0. ❌ Exit")
  print()

  choice = input(" Select preset [1-5, 0=exit]: ").strip()

  if choice == "0":
    print("\n❌ Operation cancelled.")
    sys.exit(0)

  elif choice == "1":
    print("\n ✓ Preset: Full Re-scan")
    return {
      'clear_cache': True, 'bump': None, 'set_version': None,
      'git_commit': False, 'no_backup': False, 'dry_run': False,
      'pin_versions': False, 'debug': True
    }
  elif choice == "2":
    print("\n ✓ Preset: Quick Update")
    return {
      'clear_cache': False, 'bump': None, 'set_version': None,
      'git_commit': False, 'no_backup': False, 'dry_run': False,
      'pin_versions': False, 'debug': False
    }
  elif choice == "3":
    print("\n ✓ Preset: Release Build")
    return {
      'clear_cache': False, 'bump': 'patch', 'set_version': None,
      'git_commit': True, 'no_backup': False, 'dry_run': False,
      'pin_versions': False, 'debug': False
    }
  elif choice == "4":
    print("\n ✓ Preset: Debug Mode")
    return {
      'clear_cache': False, 'bump': None, 'set_version': None,
      'git_commit': False, 'no_backup': True, 'dry_run': True,
      'pin_versions': False, 'debug': True
    }
  elif choice == "5":
    print("\n ✓ Manual configuration selected")
    return None # Signal to use detailed menu
  else:
    print("\n ⚠️ Invalid choice. Using manual configuration.")
    return None

def interactive_menu(config: Dict[str, Any]) -> Dict[str, Any]:
  preset = show_preset_menu()
  if preset is not None:
    print("\n" + "=" * 70)
    print("📋 PRESET CONFIGURATION")
    print("=" * 70)
    print(f" Clear Cache:  {'YES ✓' if preset.get('clear_cache') else 'NO'}")
    print(f" Version Bump:  {preset.get('bump', 'None') or preset.get('set_version', 'Keep current')}")
    print(f" Pin Versions:  {'YES ✓' if preset.get('pin_versions') else 'NO'}")
    print(f" Git Commit:   {'YES ✓' if preset.get('git_commit') else 'NO'}")
    print(f" Create Backup: {'NO ⚠️' if preset.get('no_backup') else 'YES ✓'}")
    print(f" Dry Run:    {'YES ✓' if preset.get('dry_run') else 'NO'}")
    print(f" Debug Mode:   {'YES ✓' if preset.get('debug') else 'NO'}")
    print("=" * 70)

    confirm = input("\n✅ Proceed with preset? (Y/n): ").strip().lower()
    if confirm == 'n' or confirm == 'no':
      print("\n❌ Operation cancelled by user.")
      sys.exit(0)
    return preset

  # Detailed menu
  print_section_header("INTERACTIVE OPTIONS", "⚙️")
  options = {}

  print("\n💾 CACHE MANAGEMENT")
  print(" The cache stores previous scan results. Clear it to force re-scanning all files.")
  cache_action = input("🗑️ Clear cache and force re-scan? (y/N): ").strip().lower()
  options['clear_cache'] = cache_action in ('y', 'yes')
  if options['clear_cache']: print(" ✓ Cache will be cleared before scanning")

  print("\n📦 VERSION MANAGEMENT")
  version_action = input("📌 Bump version? (major/minor/patch/skip) [skip]: ").strip().lower()
  if version_action in ['major', 'minor', 'patch']:
    options['bump'] = version_action; options['set_version'] = None
    print(f" ✓ Will bump {version_action} version")
  else:
    options['bump'] = None
    custom_version = input(" Set custom version? (leave empty to skip): ").strip()
    options['set_version'] = custom_version if custom_version else None
    if options['set_version']: print(f" ✓ Will set version to {custom_version}")
    else: print(" → Keeping current version")

  print("\n📍 DEPENDENCY MANAGEMENT")
  print(" Pin versions keeps existing version numbers in requirements.txt")
  pin_action = input("📍 Pin existing versions in requirements.txt? (y/N): ").strip().lower()
  options['pin_versions'] = pin_action in ('y', 'yes')
  if options['pin_versions']: print(" ✓ Will preserve version numbers")

  print("\n🔀 GIT INTEGRATION")
  git_action = input("🚀 Commit changes to git after update? (y/N): ").strip().lower()
  options['git_commit'] = git_action in ('y', 'yes')
  if options['git_commit']: print(" ✓ Will commit changes")

  print("\n💾 BACKUP")
  print(" Backups are stored in .project_backups/ folder")
  backup_action = input("💾 Skip backup? (NOT RECOMMENDED) (y/N): ").strip().lower()
  options['no_backup'] = backup_action in ('y', 'yes')
  if options['no_backup']: print(" ⚠️ Backup will be SKIPPED")
  else: print(" ✓ Backup will be created")

  print("\n🔍 DRY RUN MODE")
  print(" Dry run shows what WOULD change without actually modifying any files")
  dry_action = input("🔍 Enable dry run mode? (y/N): ").strip().lower()
  options['dry_run'] = dry_action in ('y', 'yes')
  if options['dry_run']: print(" ✓ DRY RUN MODE - No files will be modified")

  print("\n🐞 DEBUG MODE")
  print(" Debug mode shows detailed information about file processing")
  debug_action = input("🐞 Enable debug mode? (y/N): ").strip().lower()
  options['debug'] = debug_action in ('y', 'yes')
  if options['debug']: print(" ✓ Debug output enabled")

  print("\n" + "=" * 70)
  print("📋 CONFIGURATION SUMMARY")
  print("=" * 70)
  print(f" Clear Cache:  {'YES ✓' if options.get('clear_cache') else 'NO'}")
  print(f" Version Bump:  {options.get('bump', 'None') or options.get('set_version', 'Keep current')}")
  print(f" Pin Versions:  {'YES ✓' if options.get('pin_versions') else 'NO'}")
  print(f" Git Commit:   {'YES ✓' if options.get('git_commit') else 'NO'}")
  print(f" Create Backup: {'NO ⚠️' if options.get('no_backup') else 'YES ✓'}")
  print(f" Dry Run:    {'YES ✓' if options.get('dry_run') else 'NO'}")
  print(f" Debug Mode:   {'YES ✓' if options.get('debug') else 'NO'}")
  print("=" * 70)

  confirm = input("\n✅ Proceed with these settings? (Y/n): ").strip().lower()
  if confirm == 'n' or confirm == 'no':
    print("\n❌ Operation cancelled by user.")
    sys.exit(0)

  return options

def create_backup(project_root: Path, dry_run: bool):
  backup_dir = project_root / ".project_backups"; timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  backup_name = f"backup_{timestamp}"
  print(f"💾 Creating backup: {backup_name}")
  if dry_run: print(" (Dry run: no backup actually created)"); return
  try:
    backup_dir.mkdir(exist_ok=True); backup_path = backup_dir / backup_name
    files_to_backup = ["__init__.py", "requirements.txt", "comfyui.json", "pyproject.toml", ".gitignore"]
    backup_path.mkdir()
    for file in files_to_backup:
      src = project_root / file
      if src.exists(): shutil.copy2(src, backup_path / file)
    print(f" ✓ Backup created at {backup_path}")
  except Exception as e:
    print(f" ⚠️ Error creating backup: {e}")


def git_commit_changes(project_root: Path, version: str, dry_run: bool):
  print("🔀 Running git commit...")
  if dry_run: print(" (Dry run: no git commands executed)"); return
  try:
    if not (project_root / ".git").is_dir():
      print(" ⚠️ Not a git repository. Skipping commit.")
      return
    subprocess.run(["git", "add", "__init__.py", "requirements.txt", "comfyui.json", "pyproject.toml", ".gitignore"], cwd=project_root, check=True, capture_output=True)
    status_result = subprocess.run(["git", "status", "--porcelain"], cwd=project_root, check=True, capture_output=True, text=True)
    if not status_result.stdout.strip():
      print(" ✓ No changes detected to commit.")
      return
    subprocess.run(["git", "commit", "-m", f"Auto-update project files to v{version}"], cwd=project_root, check=True, capture_output=True)
    print(f" ✓ Committed changes with message: 'Auto-update project files to v{version}'")
  except subprocess.CalledProcessError as e:
    print(f" ⚠️ Git command failed: {e}")
    print(f" Stderr: {e.stderr.decode()}")
  except FileNotFoundError: print(" ⚠️ Git command not found. Skipping commit.")
  except Exception as e: print(f" ⚠️ An unexpected error occurred during git commit: {e}")

# =================================================================================
# --- ENVIRONMENT CHECK (Used by Re-launch Guard) ---
# =================================================================================

def is_in_correct_venv(debug: bool = False) -> bool:
  """Checks if running in the specified venv."""
  try:
    current_exe = Path(sys.executable).resolve()
    target_exe = Path(VENV_PYTHON_EXE).resolve()
   
    is_correct_exe = False
    try:
      is_correct_exe = current_exe.samefile(target_exe)
    except (FileNotFoundError, OSError):
      if debug: print(f"[Venv Guard] Target EXE not found at: {target_exe}")
      pass
     
    if is_correct_exe:
      if debug: print("[Venv Guard] Check 1: PASSED. Running from configured VENV_PYTHON_EXE.")
      return True
   
    if debug:
      print("[Venv Guard] Check 1: FAILED.")
      print(f" Current sys.executable: {current_exe}")
      print(f" Expected VENV_PYTHON_EXE: {target_exe}")
     
    # If not the correct exe, it's the wrong environment.
    return False
   
  except Exception as e:
    print(f"[Venv Guard] Error during check: {e}. Assuming incorrect environment.")
    return False

# =================================================================================
# --- UPDATE FILE FUNCTIONS ---
# =================================================================================

def update_init_py(project_root: Path, parsed_nodes: List[Dict[str, Any]], dry_run: bool):
  print("✏️ Updating __init__.py...")
  file_path = project_root / "__init__.py"

  # Container for public (standard) nodes
  public_imports = set()
  public_mappings = []
  public_display = {}

  # Container for private (testing) nodes
  private_imports = set()
  private_mappings = []
  private_display = {}
  
  # Track counts
  public_count = 0
  private_count = 0

  for node in parsed_nodes:
    class_name_str = node['type']
    node_title = node['title']
    mapping_key = node['key']
    
    # Calculate relative import path
    # Windows fix: ensure forward slashes for import path
    import_path = node['file_path'].replace(os.path.sep, '/').replace('.py', '')
    import_parts = import_path.split('/')
    relative_import_path = '.'.join(import_parts)
    
    import_statement = f"from .{relative_import_path} import {class_name_str}"
    mapping_entry = f'    "{mapping_key}": {class_name_str}'
    
    # Check if this is a testing/private node 
    # We check if 'testing/' is at the start of the path
    is_testing = node['file_path'].startswith("testing/") or node['file_path'].startswith("testing\\")

    if is_testing:
      private_imports.add(import_statement)
      private_mappings.append(mapping_entry)
      private_display[mapping_key] = node_title
      private_count += 1
    else:
      public_imports.add(import_statement)
      public_mappings.append(mapping_entry)
      public_display[mapping_key] = node_title
      public_count += 1

  # Sort everything for cleanliness
  public_imports_str = "\n".join(sorted(list(public_imports)))
  public_mappings_str = "{\n" + ",\n".join(sorted(public_mappings)) + "\n}"
  public_display_str = json.dumps(public_display, indent=4, sort_keys=True)

  private_imports_str = "\n    ".join(sorted(list(private_imports))) # Indented for try block
  private_mappings_str = "{\n    " + ",\n    ".join(sorted(private_mappings)) + "\n    }" # Indented
  # JSON dump for private display needs indentation adjustment or manual formatting, 
  # but for simplicity we will just inject the dict structure
  private_display_str = json.dumps(private_display, indent=4, sort_keys=True)

  # Dynamic Template
  content = f'''"""
ComfyUI Custom Nodes - Auto-generated __init__.py
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
import os
import sys
import importlib.util

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
if PACKAGE_DIR not in sys.path:
  sys.path.insert(0, PACKAGE_DIR)

# --- Public Node Imports ---
{public_imports_str}

# --- Main Mapping (Public Nodes) ---
NODE_CLASS_MAPPINGS = {public_mappings_str}

NODE_DISPLAY_NAME_MAPPINGS = {public_display_str}

# --- Private / Testing Node Imports ---
# These will only load if the 'testing' folder exists locally.
try:
    {private_imports_str}

    # Update Mappings if import succeeds
    NODE_CLASS_MAPPINGS.update({private_mappings_str})

    NODE_DISPLAY_NAME_MAPPINGS.update({private_display_str})
except ImportError:
    pass  # Testing folder missing (standard user install), skip these nodes.

# WEB_DIRECTORY = "./web"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

print(f"[ComfyUI_MD_Nodes] Initialized ({{len(NODE_CLASS_MAPPINGS)}} nodes)")
'''
  
  if dry_run: 
    print(f" (Dry run: would write {len(parsed_nodes)} nodes to __init__.py)")
    print(f"   - Public: {public_count}")
    print(f"   - Private/Testing: {private_count}")
    return

  try:
    with open(file_path, 'w', encoding='utf-8') as f: f.write(content)
    print(f" ✓ __init__.py updated.")
    print(f"   - Standard Nodes: {public_count}")
    print(f"   - Testing/Private Nodes: {private_count}")
  except Exception as e:
    print(f" ❌ Error writing __init__.py: {e}")


def update_requirements_txt(project_root: Path, dependencies: List[str], dry_run: bool, pin_versions: bool):
  print("✏️ Updating requirements.txt...")
  file_path = project_root / "requirements.txt"
  existing_versions = {}
  if pin_versions and file_path.exists():
    try:
      with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
          line = line.strip()
          if line and not line.startswith('#'):
            if '==' in line: pkg, ver = line.split('==', 1); existing_versions[pkg.strip()] = f"=={ver.strip()}"
            elif '>=' in line: pkg, ver = line.split('>=', 1); existing_versions[pkg.strip()] = f">={ver.strip()}"
            elif '<=' in line: pkg, ver = line.split('<=', 1); existing_versions[pkg.strip()] = f"<={ver.strip()}"
            elif '~=' in line: pkg, ver = line.split('~=', 1); existing_versions[pkg.strip()] = f"~={ver.strip()}"
            elif '>' in line: pkg, ver = line.split('>', 1); existing_versions[pkg.strip()] = f">{ver.strip()}"
            elif '<' in line: pkg, ver = line.split('<', 1); existing_versions[pkg.strip()] = f"<{ver.strip()}"
            else: existing_versions[line] = ""
    except Exception as e: print(f" ⚠️ Could not parse existing requirements.txt: {e}")

  content_lines = []
  for dep in sorted(dependencies):
    if pin_versions and dep in existing_versions:
      content_lines.append(f"{dep}{existing_versions[dep]}")
    else:
      content_lines.append(dep)

  content = "\n".join(content_lines) + "\n"

  if dry_run: print(f" (Dry run: would write {len(dependencies)} dependencies to requirements.txt)"); return
  try:
    with open(file_path, 'w', encoding='utf-8') as f: f.write(content)
    print(f" ✓ requirements.txt updated with {len(dependencies)} dependencies.")
  except Exception as e:
    print(f" ❌ Error writing requirements.txt: {e}")

def update_comfyui_json(project_root: Path, parsed_nodes: List[Dict[str, Any]], dependencies: List[str], config: Dict[str, Any], dry_run: bool):
  print("✏️ Updating comfyui.json...")
  file_path = project_root / "comfyui.json"
  data = {}
  if file_path.exists():
    try:
      with open(file_path, 'r', encoding='utf-8') as f: data = json.load(f)
    except json.JSONDecodeError: print(" ⚠️ Could not parse existing comfyui.json, creating fresh file.")

  data['version'] = config.get('project_version', '1.0.0')
  data['nodes'] = parsed_nodes
  data['dependencies'] = dependencies

  if 'name' not in data: data['name'] = project_root.name
  if 'author' not in data: data['author'] = "Unknown"
  if 'description' not in data: data['description'] = "Custom nodes for ComfyUI"

  if dry_run: print(f" (Dry run: would write {len(parsed_nodes)} nodes to comfyui.json)"); return
  try:
    with open(file_path, 'w', encoding='utf-8') as f: json.dump(data, f, indent=4)
    print(f" ✓ comfyui.json updated.")
  except Exception as e:
    print(f" ❌ Error writing comfyui.json: {e}")

def update_pyproject_toml(project_root: Path, dependencies: List[str], version: str, dry_run: bool):
  if not HAS_TOML: print(" ⚠️ tomli/tomli_w not installed. Skipping pyproject.toml update."); return
  print("✏️ Updating pyproject.toml...")
  file_path = project_root / "pyproject.toml"
  if file_path.exists():
    try:
      with open(file_path, 'rb') as f: data = tomli.load(f)
    except Exception as e: print(f" ⚠️ Could not parse pyproject.toml: {e}"); return
  else: data = {"project": {}}
  if "project" not in data: data["project"] = {}
  data["project"]["version"] = version
  data["project"]["dependencies"] = sorted(dependencies)
  if dry_run: print(f" (Dry run: would update pyproject.toml)"); return
  try:
    with open(file_path, 'wb') as f: tomli_w.dump(data, f)
    print(f" ✓ pyproject.toml updated.")
  except Exception as e:
    print(f" ❌ Error writing pyproject.toml: {e}")


def update_gitignore(project_root: Path, config: Dict[str, Any], dry_run: bool):
  print("✏️ Updating .gitignore...")
  file_path = project_root / ".gitignore"
  existing = set()
  if file_path.exists():
    try:
      with open(file_path, 'r', encoding='utf-8') as f: existing = {line.strip() for line in f if line.strip()}
    except Exception as e:
      print(f" ⚠️ Could not read existing .gitignore: {e}")
      return

  entries_to_add = config.get('gitignore_entries', [])
  new_entries = [e for e in entries_to_add if e not in existing]

  if not new_entries:
    print(" ✓ .gitignore is up to date."); return

  if dry_run: print(f" (Dry run: would add {len(new_entries)} entries to .gitignore)"); return

  try:
    with open(file_path, 'a', encoding='utf-8') as f:
      if existing and not file_path.read_text(encoding='utf-8').endswith('\n'):
        f.write("\n")
      f.write("\n".join(new_entries) + "\n")
    print(f" ✓ .gitignore updated with {len(new_entries)} new entries.")
  except Exception as e:
    print(f" ❌ Error writing .gitignore: {e}")


# =================================================================================
# --- NODE PARSING ---
# =================================================================================

def extract_node_mappings_dynamic(file_path: Path, debug: bool) -> tuple[dict, dict]:
  """
  Extract NODE_CLASS_MAPPINGS by actually executing the file in a sandbox.
  Uses mock modules for comfy imports.
  Returns mappings where values are class name STRINGS.
  """
  from io import StringIO

  mappings = {}
  display_names = {}
  module_name = f"temp_node_module_{file_path.stem}_{hashlib.md5(str(file_path).encode()).hexdigest()[:8]}"

  # --- Start Mock Module Setup v4.3.6 ---
  class MockTorch:
    Tensor = type('Tensor', (object,), {})
    def __getattr__(self, name):
      if name in ('float32', 'float64', 'int32', 'int64'): return name
      if name in ('zeros', 'ones', 'empty', 'tensor', 'linspace', 'randn_like', 'rand_like', 'mean', 'max', 'abs', 'sqrt', 'exp', 'sin', 'cos', 'acos', 'where', 'clamp', 'norm', 'sum', 'from_numpy', 'isinf', 'isnan'):
        return lambda *args, **kwargs: f"mock_tensor_func_{name}"
      if name == 'nn':
        mock_nn = ModuleType('torch.nn')
        mock_nn.functional = ModuleType('torch.nn.functional') # Mock torch.nn.functional
        return mock_nn
      if name == 'cuda':
        mock_cuda = ModuleType('torch.cuda')
        mock_cuda.is_available = lambda: False
        mock_cuda.synchronize = lambda: None
        mock_cuda.memory_allocated = lambda: 0
        return mock_cuda
      if name == 'hub': # Fix for torchaudio
        mock_hub = ModuleType('torch.hub')
        mock_hub.download_url_to_file = lambda *a, **k: None
        mock_hub.load_state_dict_from_url = lambda *a, **k: {}
        return mock_hub
      return lambda *args, **kwargs: f"mock_torch_attr_{name}"

  mock_torch_instance = MockTorch()

  mock_modules = {
    'comfy': ModuleType('comfy'),
    'comfy.model_management': ModuleType('comfy.model_management'),
    'comfy.utils': ModuleType('comfy.utils'),
    'comfy.samplers': ModuleType('comfy.samplers'),
    'comfy.sample': ModuleType('comfy.sample'),
    'comfy.sd': ModuleType('comfy.sd'),
    'comfy.cli_args': ModuleType('comfy.cli_args'),
    'comfy.k_diffusion': ModuleType('comfy.k_diffusion'),
    'comfy.k_diffusion.sampling': ModuleType('comfy.k_diffusion.sampling'),
    'comfy.text_encoders': ModuleType('comfy.text_encoders'), # Fix for ACE_T5
    'comfy.text_encoders.t5': ModuleType('comfy.text_encoders.t5'), # Fix for ACE_T5
    'nodes': ModuleType('nodes'),
    'folder_paths': ModuleType('folder_paths'),
    'torch': mock_torch_instance,
    'torch.nn': mock_torch_instance.nn,
    'torch.nn.functional': mock_torch_instance.nn.functional,
    'torch.hub': mock_torch_instance.hub,
  }
 
  setattr(mock_modules['comfy'], '__path__', ['mock_comfy_path']) # Make comfy a package

  # Add dummy attributes
  mock_modules['comfy.model_management'].get_torch_device = lambda: 'cpu'
  mock_modules['comfy.model_management'].soft_empty_cache = lambda: None
  mock_modules['comfy.model_management'].is_device_mps = lambda: False
  mock_modules['comfy.model_management'].is_device_cpu = lambda: True
  mock_modules['comfy.model_management'].is_device_cuda = lambda: False
  mock_modules['comfy.model_management'].supports_fp16 = lambda: False
  mock_modules['comfy.model_management'].supports_bf16 = lambda: False
 
  mock_modules['folder_paths'].get_input_directory = lambda: '.'
  mock_modules['folder_paths'].get_output_directory = lambda: '.'
  mock_modules['folder_paths'].get_temp_directory = lambda: '.'
  mock_modules['folder_paths'].get_filename_list = lambda _: []
  mock_modules['folder_paths'].get_folder_paths = lambda _: []

  # Mock specific classes/objects
  mock_modules['comfy.cli_args'].args = type('MockArgs', (object,), {'disable_metadata': False})()
  mock_modules['comfy.samplers'].KSAMPLER = type('MockKSAMPLER', (object,), {})
  mock_modules['comfy.samplers'].CFGGuider = type('MockCFGGuider', (object,), {})
  mock_modules['comfy.samplers'].cfg_function = lambda *args, **kwargs: None # Fix for FBG
  mock_modules['comfy.k_diffusion.sampling'].CFGDenoiser = type('MockCFGDenoiser', (object,), {})
  mock_modules['comfy.k_diffusion.sampling'].get_sigmas_karras = lambda *args, **kwargs: "mock_sigmas"
  mock_modules['comfy'].model_sampling = ModuleType('comfy.model_sampling')
  setattr(mock_modules['comfy'].model_sampling, 'CONST', type('MockCONST', (object,), {}))
  mock_modules['comfy'].model_patcher = ModuleType('comfy.model_patcher')
  mock_modules['comfy'].model_patcher.set_model_options_post_cfg_function = lambda *args, **kwargs: args[0] if args else {}
  mock_modules['comfy'].sd1_clip = ModuleType('comfy.sd1_clip')
  # --- End Mock Module Setup ---

  original_modules = {}
  for name, mock in mock_modules.items():
    if name in sys.modules:
      original_modules[name] = sys.modules[name]
    sys.modules[name] = mock

  found_class_mapping = False
  found_display_mapping = False
  class_mapping_type = "N/A"
  class_mapping_keys = "N/A"
  display_mapping_type = "N/A"

  try:
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
      if debug: print(f"  Warning: Could not create module spec for {file_path.name}")
      return extract_node_mappings_static(file_path, debug)

    module = importlib.util.module_from_spec(spec)
   
    # --- Fix for NameError: name 'torch' is not defined ---
    module.__dict__['torch'] = mock_torch_instance
    # --- End Fix ---
   
    sys.modules[module_name] = module

    old_stdout, old_stderr = sys.stdout, sys.stderr
    redirect_out = StringIO()
    sys.stdout, sys.stderr = redirect_out, redirect_out

    file_dir = str(file_path.parent)
    is_path_added = False

    try:
      if file_dir not in sys.path:
        sys.path.insert(0, file_dir); is_path_added = True

      spec.loader.exec_module(module)

      found_class_mapping = hasattr(module, 'NODE_CLASS_MAPPINGS')
      found_display_mapping = hasattr(module, 'NODE_DISPLAY_NAME_MAPPINGS')
      if found_class_mapping:
        try:
          raw_map = getattr(module, 'NODE_CLASS_MAPPINGS', None)
          class_mapping_type = str(type(raw_map))
          if isinstance(raw_map, dict): class_mapping_keys = str(list(raw_map.keys()))
          else: class_mapping_keys = f"Not a dict (value: {str(raw_map)[:100]}...)"
        except Exception as e_get: class_mapping_type = f"Error inspecting: {e_get}"
      if found_display_mapping:
        try: display_mapping_type = str(type(getattr(module, 'NODE_DISPLAY_NAME_MAPPINGS', None)))
        except Exception: display_mapping_type = "Error inspecting"

      if found_class_mapping:
        raw_mappings = getattr(module, 'NODE_CLASS_MAPPINGS', {})
        if isinstance(raw_mappings, dict):
          for key, value in raw_mappings.items():
            # Store the class name string, which is the dictionary value
            if isinstance(value, type): mappings[key] = value.__name__
            elif isinstance(value, str): mappings[key] = value
            else:
              if debug: print(f"  Warning: Skipping mapping '{key}'. Value '{value}' is not a string or class.")
        elif debug:
          print(f"  Warning: NODE_CLASS_MAPPINGS in {file_path.name} is not a dictionary (Type: {type(raw_mappings)}).")

      if found_display_mapping:
        raw_display_mappings = getattr(module, 'NODE_DISPLAY_NAME_MAPPINGS', {})
        if isinstance(raw_display_mappings, dict): display_names = raw_display_mappings
        elif debug:
          print(f"  Warning: NODE_DISPLAY_NAME_MAPPINGS in {file_path.name} is not a dictionary (Type: {type(raw_display_mappings)}).")

    except Exception as e:
      if debug:
        print(f"  ❌ ERROR: Failed to dynamically execute {file_path.name}: {e}")
        import traceback
        print(f"   Full Traceback:\n{traceback.format_exc()}")
      
      sys.stdout, sys.stderr = old_stdout, old_stderr
      print(f"  ❌ ERROR: Failed to dynamically execute {file_path.name}: {e}")
      if debug:
        import traceback
        print(f"   Full Traceback:\n{traceback.format_exc()}")
      print(f"  Attempting static fallback parsing for {file_path.name}...")
      sys.stdout, sys.stderr = redirect_out, redirect_out
      
      for name in mock_modules.keys():
        if name in original_modules: sys.modules[name] = original_modules[name]
        elif name in sys.modules and sys.modules[name] == mock_modules[name]:
          del sys.modules[name]
      
      sys.stdout, sys.stderr = old_stdout, old_stderr
      if is_path_added and file_dir in sys.path: sys.path.remove(file_dir)
      if module_name in sys.modules: del sys.modules[module_name]
      
      # --- NOW CALL THE *IMPROVED* STATIC PARSER ---
      return extract_node_mappings_static(file_path, debug)

    finally:
      sys.stdout, sys.stderr = old_stdout, old_stderr
      if is_path_added and file_dir in sys.path: sys.path.remove(file_dir)
      if module_name in sys.modules: del sys.modules[module_name]

    if debug:
      print(f"  !!! DEBUG Post-Exec {file_path.name}:")
      print(f"    Found NODE_CLASS_MAPPINGS attribute: {found_class_mapping}")
      if found_class_mapping:
        print(f"    Mappings Attr Type = {class_mapping_type}")
        print(f"    Mappings Keys = {class_mapping_keys}")
      print(f"    Found NODE_DISPLAY_NAME_MAPPINGS attribute: {found_display_mapping}")
      if found_display_mapping:
        print(f"    Display Mappings Attr Type = {display_mapping_type}")

  finally:
    for name in mock_modules.keys():
      if name in original_modules: sys.modules[name] = original_modules[name]
      elif name in sys.modules and sys.modules[name] == mock_modules[name]:
        del sys.modules[name]

  return mappings, display_names


def extract_node_mappings_static(file_path: Path, debug: bool) -> tuple[dict, dict]:
  """
  Fallback: Extract using static AST parsing.
  v4.3.7: Now supports type-hinted assignments (ast.AnnAssign).
  """
  mappings = {}
  display_names = {}
  if debug: print(f"  [Static] Running fallback parser for {file_path.name}...")
  try:
    tree = ast.parse(file_path.read_text(encoding="utf-8"))
   
    # --- START v4.3.7 STATIC PARSER FIX ---
    # Iterate over all top-level nodes in the file
    for node in tree.body:
      target_name = None
      value_node = None

      if isinstance(node, ast.Assign):
        # Handles: NODE_CLASS_MAPPINGS = ...
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
          target_name = node.targets[0].id
          value_node = node.value
      elif isinstance(node, ast.AnnAssign):
        # Handles: NODE_CLASS_MAPPINGS: Dict = ...
        if isinstance(node.target, ast.Name):
          target_name = node.target.id
          value_node = node.value
      # --- END v4.3.7 STATIC PARSER FIX ---

      if target_name in ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"] and isinstance(value_node, ast.Dict):
        temp_dict = {}
        for key_node, val_node in zip(value_node.keys, value_node.values):
          key_val = None
          # Try to evaluate simple key (string constant)
          if isinstance(key_node, ast.Constant): key_val = key_node.value
          elif hasattr(ast, 'Str') and isinstance(key_node, ast.Str): key_val = key_node.s # Older Python

          if key_val is not None:
            val_val = None
            # Try to evaluate simple value (string constant or variable name)
            if isinstance(val_node, ast.Constant): val_val = val_node.value
            elif hasattr(ast, 'Str') and isinstance(val_node, ast.Str): val_val = val_node.s # Older Python
            elif isinstance(val_node, ast.Name): val_val = val_node.id
           
            if val_val is not None:
              temp_dict[key_val] = val_val
            elif debug:
              print(f"  [Static] Skipping key '{key_val}' in {target_name}: value is not a simple type.")
          elif debug:
            print(f"  [Static] Skipping key in {target_name}: key is not a simple string.")
       
        if target_name == "NODE_CLASS_MAPPINGS": mappings = temp_dict
        else: display_names = temp_dict
       
  except Exception as e:
    if debug: print(f"  Warning: Static parsing failed for {file_path.name}: {e}")
 
  if debug and (mappings or display_names):
    print(f"  [Static] Fallback parser found {len(mappings)} class mappings and {len(display_names)} display mappings for {file_path.name}.")
   
  return mappings, display_names

def discover_python_files(project_root: Path, config: Dict[str, Any], debug: bool) -> List[Path]:
  print("🔍 Scanning for Python files...")
  py_files, exclude_dirs, exclude_files = [], set(config.get('exclude_dirs', [])), set(config.get('exclude_files', []))
  exclude_exts = set(config.get('exclude_extensions', []))

  all_files = sorted(project_root.rglob("*"))
  print(f" Found {len(all_files)} total items in directory tree.")

  skipped_count = 0
  for path in all_files:
    if not path.is_file():
      skipped_count += 1
      continue

    relative_path = path.relative_to(project_root)

    if path.suffix in exclude_exts:
      if debug: print(f" - Skipping '{relative_path}' (excluded extension)")
      skipped_count += 1
      continue

    if any(part.startswith('.') for part in relative_path.parts):
      if debug: print(f" - Skipping '{relative_path}' (hidden directory or file)")
      skipped_count += 1
      continue

    excluded_by_dir = False
    for pattern in exclude_dirs:
      for parent in relative_path.parents:
        if parent.match(pattern):
          excluded_by_dir = True; break
      if excluded_by_dir: break
    if excluded_by_dir:
      if debug: print(f" - Skipping '{relative_path}' (excluded directory pattern match)")
      skipped_count += 1
      continue

    excluded_by_file = False
    for pattern in exclude_files:
      if path.match(pattern):
        excluded_by_file = True; break
    if excluded_by_file:
      if debug: print(f" - Skipping '{relative_path}' (excluded file pattern match)")
      skipped_count += 1
      continue

    if path.suffix == ".py":
      py_files.append(path)
      if debug: print(f" + Including '{relative_path}'")
    else:
      skipped_count +=1
      if debug: print(f" - Skipping '{relative_path}' (not a python file)")

  print(f" Skipped {skipped_count} non-python or excluded items.")
  print(f"✅ Found {len(py_files)} Python files to analyze.")
  return py_files


def find_project_dependencies(project_root: Path, all_py_files: List[Path], config: Dict[str, Any], debug: bool = False) -> List[str]:
  print("📦 Scanning for dependencies...")
  found, std_lib = set(), set(sys.builtin_module_names)

  try:
    import stdlib_list
    std_lib.update(stdlib_list.stdlib_list(f"{sys.version_info.major}.{sys.version_info.minor}"))
  except ImportError:
    if debug: print(" `stdlib-list` not found, using basic stdlib detection.")
    std_lib_paths = [os.path.dirname(os.__file__), os.path.dirname(hashlib.__file__)]
    for module_info in pkgutil.iter_modules():
      is_std_lib_module = False
      if module_info.module_finder and hasattr(module_info.module_finder, 'path'):
        module_path = module_info.module_finder.path
        is_std_lib_module = any(module_path.startswith(p) for p in std_lib_paths if p) or \
                  ('site-packages' not in module_path and 'dist-packages' not in module_path)
      elif not module_info.ispkg:
        is_std_lib_module = True

      if is_std_lib_module:
        std_lib.add(module_info.name)

  std_lib.update(['os', 'sys', 'json', 'ast', 'shutil', 'subprocess', 'pathlib', 'datetime', 'pkgutil', 'argparse', 'hashlib', 'importlib', 'types', 'io', 're', 'logging', 'math', 'time', 'traceback', 'contextlib'])

  local_modules = {p.stem for p in all_py_files}
  local_packages = {p.name for p in project_root.iterdir() if p.is_dir() and (p / "__init__.py").exists()}
  local = local_modules | local_packages

  denylist = set(config.get('denylist_packages', []))
  import_to_package_map = config.get('import_to_package_map', {})
  manual_imports = set(config.get('manual_imports', []))

  if debug: print(f" Stdlib detected: {len(std_lib)} modules")
  if debug: print(f" Local modules/packages: {local}")

  for file in all_py_files:
    if debug: print(f" Parsing imports in: {file.name}")
    try:
      tree = ast.parse(file.read_text(encoding="utf-8"))
      for node in ast.walk(tree):
        module_name = None
        if isinstance(node, ast.Import):
          for alias in node.names:
            module_name = alias.name.split('.')[0]
            if module_name: found.add(module_name)
        elif isinstance(node, ast.ImportFrom) and node.module:
          if node.level == 0:
            module_name = node.module.split('.')[0]
            if module_name: found.add(module_name)
    except Exception as e:
      if debug: print(f"  Warning: Could not parse imports from {file.name}: {e}")

  if debug: print(f" All imports found (raw): {found}")
  external = {imp for imp in found if imp not in std_lib and imp not in local}
  if debug: print(f" External imports (filtered): {external}")
  mapped = {import_to_package_map.get(imp, imp) for imp in external}
  if debug: print(f" Mapped package names: {mapped}")
  final = {dep for dep in mapped if dep not in denylist}
  if debug: print(f" After denylist: {final}")
  final.update(manual_imports)
  if debug: print(f" After adding manual: {final}")
  final = {dep for dep in final if dep not in denylist}
  if debug: print(f" Final dependencies (before sort): {final}")

  print(f"✅ Found {len(final)} external dependencies.")
  return sorted(list(final))


def parse_node_files(project_root: Path, file_paths: List[Path], debug: bool, cache: Dict[str, Any]) -> tuple[List[Dict[str, Any]], List[Dict[str, str]], Dict[str, Any]]:
  all_nodes, errors, new_cache, cached_count = [], [], {}, 0
  print("📋 Parsing node metadata...")

  iterable = tqdm(file_paths, desc="Parsing nodes", unit="file") if HAS_TQDM and len(file_paths) > 5 else file_paths
  for path in iterable:
    rel_path = str(path.relative_to(project_root))
    try:
      current_hash = get_file_hash(path)
      if rel_path in cache and cache[rel_path]['hash'] == current_hash:
        if debug: print(f" - Skipping '{rel_path}' (cached)")
        cached_nodes = cache[rel_path].get('nodes', [])
        if isinstance(cached_nodes, list): all_nodes.extend(cached_nodes)
        else:
          if debug: print(f"  Warning: Invalid node cache format for {rel_path}, re-parsing.")
          cache.pop(rel_path)
        new_cache[rel_path] = cache[rel_path]; cached_count += 1
        continue

      if debug: print(f" - Analyzing '{rel_path}'")
      mappings, display_names = extract_node_mappings_dynamic(path, debug)

      file_nodes = []
      if mappings:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for key, mapped_name in mappings.items(): # key is the mapping key (e.g., "APGGuiderForked"), mapped_name is the class name (e.g., "APGGuiderNode")
          class_info = None
          for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == mapped_name:
              class_info = node; break

          display_name_key = key
          node_title = display_names.get(display_name_key, mapped_name)

          if class_info:
            info = {
              "key": key, # CRITICAL: Store the original mapping key for __init__.py generation
              "type": mapped_name, "title": node_title,
              "description": ast.get_docstring(class_info) or "",
              "category": "Uncategorized",
              "file_path": rel_path.replace(os.path.sep, "/")
            }
            for item in class_info.body:
              if isinstance(item, ast.Assign) and len(item.targets) == 1:
                target = item.targets[0]
                if isinstance(target, ast.Name) and target.id == "CATEGORY":
                  if isinstance(item.value, ast.Constant): info["category"] = item.value.value
                  elif hasattr(ast, 'Str') and isinstance(item.value, ast.Str): info["category"] = item.value.s
                  break
            file_nodes.append(info)
          else:
            if debug: print(f"  Warning: Class AST node '{mapped_name}' not found in {path.name}. Using basic info.")
            file_nodes.append({
              "key": key, # CRITICAL: Still need to store key in fallback
              "type": mapped_name, "title": node_title, "description": "",
              "category": "Uncategorized", "file_path": rel_path.replace(os.path.sep, "/")
            })

      new_cache[rel_path] = {'hash': current_hash, 'nodes': file_nodes}
      all_nodes.extend(file_nodes)

    except Exception as e:
      errors.append({"file": rel_path, "error": str(e)})
      if debug:
        import traceback
        print(f"  ❌ Error processing {rel_path}:")
        print(f"   {traceback.format_exc()}")

  found_nodes_count = len(all_nodes)
  if debug and cached_count > 0 and len(file_paths) > 0:
    cache_hit_rate = (cached_count / len(file_paths)) * 100
    print(f" 📊 Cache hit rate: {cache_hit_rate:.1f}% ({cached_count}/{len(file_paths)} files)")

  print(f"✅ Found a total of {found_nodes_count} nodes ({len(errors)} errors during parsing).")
  return sorted(all_nodes, key=lambda x: x['title']), errors, new_cache

def validate_node_metadata(parsed_nodes: List[Dict[str, Any]]):
  warnings = []
  titles = set()
  duplicates = set()

  for n in parsed_nodes:
    if not n.get('description'):
      warnings.append(f" - Node '{n['title']}' is missing a description.")
    title = n['title']
    if title in titles: duplicates.add(title)
    else: titles.add(title)

  if duplicates:
    warnings.append("\n --- Duplicate Display Titles Found ---")
    for title in sorted(list(duplicates)):
      warnings.append(f" - Title '{title}' used by multiple nodes.")

  if warnings:
    print_section_header("METADATA WARNINGS", "⚠️")
    for w in warnings: print(w)


# =================================================================================
# --- MAIN EXECUTION ---
# =================================================================================

def main():
  args = parse_arguments()
  print_header()
 
  # This check is no longer needed here, as it's in the __name__ == "__main__" block
  # check_environment_and_warn()

  has_non_interactive_args = any(getattr(args, arg) for arg in vars(args) if arg != 'interactive')
  is_interactive_run = args.interactive or not has_non_interactive_args

  updater_settings = load_updater_settings()

  if args.project_root:
    project_root = Path(args.project_root).resolve()
    print(f"📁 Using specified project root: {project_root}")
  elif is_interactive_run:
    project_root = prompt_for_project_root(updater_settings)
  else: # Non-interactive fallback
    last_path = updater_settings.get('last_project_path')
    if last_path and Path(last_path).is_dir():
      project_root = Path(last_path).resolve()
      print(f"📁 Using last known project root: {project_root}")
    else:
      project_root = Path.cwd().resolve()
      print(f"📁 Using current directory as project root: {project_root}")

  if not project_root.is_dir():
    print(f"\n❌ Error: Project root '{project_root}' not found or is not a directory.")
    sys.exit(1)

  project_config = load_or_create_project_config(project_root)
  config_errors = validate_config(project_config)
  if config_errors:
    print_section_header("CONFIG ERRORS", "⚠️")
    for e in config_errors: print(f" - {e}")
    sys.exit(1)

  if is_interactive_run:
    menu_options = interactive_menu(project_config)
    for key, value in menu_options.items(): setattr(args, key, value)

  if args.dry_run:
    print("\n" + "=" * 70 + "\n🔍 DRY RUN MODE IS ACTIVE\n" + "="*70 + "\n No files will be modified.\n" + "="*70)

  if args.clear_cache:
    print("\n🗑️ Clearing file cache...")
    updater_settings['file_cache'] = {}; print(" ✓ Cache cleared")

  new_version = handle_versioning(args, project_config.get('project_version', '1.0.0'))
  project_config['project_version'] = new_version

  print_section_header("RUNNING UPDATES", "⚙️")
  if not args.no_backup: create_backup(project_root, args.dry_run)

  all_py_files = discover_python_files(project_root, project_config, args.debug)

  parsed_nodes, parsing_errors, new_cache = parse_node_files(
    project_root, all_py_files, args.debug, updater_settings.get('file_cache', {})
  )

  updater_settings['last_project_path'] = str(project_root)
  updater_settings['file_cache'] = new_cache

  dependencies = find_project_dependencies(project_root, all_py_files, project_config, args.debug)
  validate_node_metadata(parsed_nodes)

  print_section_header("WRITING FILES", "✏️")
  update_init_py(project_root, parsed_nodes, args.dry_run)
  update_requirements_txt(project_root, dependencies, args.dry_run, args.pin_versions)
  update_comfyui_json(project_root, parsed_nodes, dependencies, project_config, args.dry_run)
  update_pyproject_toml(project_root, dependencies, new_version, args.dry_run)
  update_gitignore(project_root, project_config, args.dry_run)

  if args.git_commit: git_commit_changes(project_root, new_version, args.dry_run)

  if parsing_errors:
    print_section_header("PARSING ERRORS", "⚠️")
    print(" Some files could not be parsed:")
    for err in parsing_errors: print(f" ❌ {err['file']}: {err['error']}")

  if not args.dry_run:
    try:
      current_config_content = ""
      config_path = project_root / PROJECT_CONFIG_FILENAME
      if config_path.exists():
        try: current_config_content = config_path.read_text(encoding='utf-8')
        except Exception: pass
      new_config_content = json.dumps(project_config, indent=4)
      if new_config_content != current_config_content:
        with open(config_path, 'w', encoding='utf-8') as f: f.write(new_config_content)
        print("\n✅ Project config updated.")

      save_updater_settings(updater_settings)
      print("✅ Updater settings saved.")
    except Exception as e:
      print(f"\n⚠️ Error saving config/settings: {e}")

  print_section_header("COMPLETE", "🎉")
  if args.dry_run:
    print("🔍 Dry Run Finished. No changes were made to any files.")
  else:
    print(f"✅ Project update to version {new_version} complete!")
    print(f"\n📊 Summary:")
    print(f" • Nodes found: {len(parsed_nodes)} ({len(parsing_errors)} errors during parsing)")
    print(f" • Dependencies: {len(dependencies)}")
    print(f" • Python files scanned: {len(all_py_files)}")

  print()

if __name__ == "__main__":
  # --- v4.3.6 Re-launch Guard Check ---
  is_debug_arg = "--debug" in sys.argv[1:]
 
  if not is_in_correct_venv(debug=is_debug_arg):
    print("█" * 70, file=sys.stderr)
    print("█" + "   !! INCORRECT PYTHON ENVIRONMENT DETECTED !!   ".center(68), file=sys.stderr)
    print("█" + " This script is not running in the correct ComfyUI venv. ".center(68), file=sys.stderr)
    print("█" * 70, file=sys.stderr)
   
    if not Path(VENV_PYTHON_EXE).exists():
      print(f"\n❌ FATAL: Cannot re-launch. The target Python venv was not found at:")
      print(f" {VENV_PYTHON_EXE}")
      print(f"\nPlease check this path in the 'VENV_PYTHON_EXE' variable at the top of the script.")
      print("Press any key to exit...")
      try: os.system("pause > nul") # Windows
      except Exception: pass # Non-windows
      sys.exit(1)
     
    print(f"\nAttempting to re-launch using the correct venv Python:")
    print(f" {VENV_PYTHON_EXE}")
   
    script_path = Path(__file__).resolve()
    original_args = sys.argv[1:]
   
    try:
      result = subprocess.run(
        [VENV_PYTHON_EXE, str(script_path)] + original_args,
        check=False
      )
      print(f"\nRe-launch complete. Exiting original script.")
      sys.exit(result.returncode)
    except Exception as e:
      print(f"\n❌ FATAL: Failed to re-launch: {e}")
      print("Please run this script using the 'run_updater.bat' file.")
      print("Press any key to exit...")
      try: os.system("pause > nul") # Windows
      except Exception: pass # Non-windows
      sys.exit(1)
 
  # --- If we are here, we are in the correct environment ---
  main()