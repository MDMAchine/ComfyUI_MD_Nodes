#!/usr/bin/env python3
"""
ComfyUI Launcher v3.0.0

A lightweight launcher for daily ComfyUI usage with argument configuration,
preset management, and update functionality.

For heavy lifting (PyTorch installation, wheel management, venv creation),
use the ComfyUI Master Suite instead.

Author: MDMAchine
"""

import os
import sys
import platform
import subprocess
import configparser
import json
import time
from pathlib import Path
from typing import List, Optional, Tuple, Any, Callable
from datetime import datetime

# ============================================================================
# CONSTANTS
# ============================================================================
APP_VERSION = "3.0.0"
CUSTOM_PRESETS_FILE = 'comfyui_custom_presets.json'
CONFIG_FILE = 'comfyui_launcher_config.ini'
MAX_PARENT_SEARCH_DEPTH = 4

# ============================================================================
# COLORS
# ============================================================================
class Colors:
    """ANSI color codes with Windows compatibility."""
    _enabled = True
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'

    @classmethod
    def disable(cls):
        cls._enabled = False

    @classmethod
    def _get(cls, code: str) -> str:
        return code if cls._enabled else ''

    def __getattribute__(self, name: str):
        if name.startswith('_') or name in ('disable', 'init'):
            return object.__getattribute__(self, name)
        return Colors._get(object.__getattribute__(self, name))

    @staticmethod
    def init():
        if platform.system() == "Windows":
            try:
                import ctypes
                kernel32 = ctypes.windll.kernel32
                kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
            except Exception:
                Colors.disable()

Colors.init()
C = Colors()

# ============================================================================
# PRESETS
# ============================================================================
PRESETS = {
    "high_performance": {
        "name": "🚀 High Performance (16GB+ VRAM)",
        "description": "Maximum speed for powerful GPUs",
        "args": {"--highvram": "true", "--use-flash-attention": "true"}
    },
    "balanced": {
        "name": "⚖️ Balanced (8-16GB VRAM)",
        "description": "Good balance of speed and compatibility",
        "args": {"--normalvram": "true", "--use-flash-attention": "true"}
    },
    "low_vram": {
        "name": "💾 Low VRAM (4-8GB)",
        "description": "Optimized for limited VRAM",
        "args": {"--lowvram": "true", "--disable-smart-memory": "true"}
    },
    "network": {
        "name": "🌐 Network Accessible",
        "description": "Allow access from other devices",
        "args": {"--listen": "0.0.0.0"}
    },
}

# ============================================================================
# UTILITIES
# ============================================================================
def get_script_dir() -> Path:
    """Get directory containing this script."""
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd()

def clear_screen():
    os.system('cls' if platform.system() == 'Windows' else 'clear')

def print_header(text: str):
    print(f"\n{C.BOLD}{C.CYAN}{'═' * 70}{C.ENDC}")
    print(f"{C.BOLD}{C.CYAN}  {text}{C.ENDC}")
    print(f"{C.BOLD}{C.CYAN}{'═' * 70}{C.ENDC}")

def run_command(command: List[str], cwd: Optional[str] = None, capture_output: bool = False):
    """Run a shell command with real-time output or capture."""
    if not capture_output:
        print(f"\n{C.BOLD}--- Running: {' '.join(command)} ---{C.ENDC}")
    
    process = None
    try:
        if capture_output:
            result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, errors='replace')
            return result.returncode == 0, result.stdout, result.stderr
        else:
            process = subprocess.Popen(
                command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, errors='replace', bufsize=1
            )
            for line in process.stdout:
                print(f"{C.DIM}{line.rstrip()}{C.ENDC}", flush=True)
            process.wait()
            success = process.returncode == 0
            status = f"{C.GREEN}✓ Success{C.ENDC}" if success else f"{C.RED}✗ Failed (exit {process.returncode}){C.ENDC}"
            print(f"\n{status}")
            return success
    except Exception as e:
        print(f"{C.RED}Error: {e}{C.ENDC}")
        return False if not capture_output else (False, "", str(e))
    finally:
        if process and process.poll() is None:
            process.terminate()

def find_comfyui_roots(start_dir: Path) -> List[Path]:
    """Find potential ComfyUI root directories."""
    roots = set()
    search_dirs = [start_dir] + list(start_dir.parents)[:MAX_PARENT_SEARCH_DEPTH]

    for directory in search_dirs:
        if not directory.is_dir():
            continue
        
        # Check if directory itself is ComfyUI
        if (directory / 'main.py').exists() and (directory / 'custom_nodes').is_dir():
            roots.add(directory)
        
        # Check subdirectories
        try:
            for item in directory.iterdir():
                if item.is_dir() and (item / 'main.py').exists() and (item / 'custom_nodes').is_dir():
                    roots.add(item)
        except PermissionError:
            continue

    return sorted(roots)

def find_python_executables(comfyui_root: Path) -> List[Tuple[str, Path]]:
    """Find Python executables in venvs folder and system."""
    executables = []
    seen = set()

    # Check sibling venvs folder
    venvs_dir = comfyui_root.parent / 'venvs'
    if venvs_dir.is_dir():
        for venv in venvs_dir.iterdir():
            if venv.is_dir() and (venv / 'pyvenv.cfg').exists():
                py = venv / 'Scripts' / 'python.exe' if platform.system() == "Windows" else venv / 'bin' / 'python'
                if py.exists() and py.resolve() not in seen:
                    executables.append((f"Venv: {venv.name}", py.resolve()))
                    seen.add(py.resolve())

    # Add system Python
    sys_py = Path(sys.executable).resolve()
    if sys_py not in seen:
        executables.append(("System Python", sys_py))

    return executables

def select_from_list(items: list, title: str, display_func: Callable[[Any], str]) -> Optional[Any]:
    """Generic selection menu."""
    if not items:
        print(f"\n{C.YELLOW}No items found for '{title}'.{C.ENDC}")
        return None
    
    if len(items) == 1:
        print(f"\n{C.GREEN}Auto-selected:{C.ENDC} {display_func(items[0])}")
        time.sleep(1)
        return items[0]

    while True:
        clear_screen()
        print_header(f"Select {title}")
        for i, item in enumerate(items, 1):
            print(f"  [{i}] {display_func(item)}")
        print(f"\n  [B] Back")
        
        choice = input(f"\n{C.BOLD}Choice (1-{len(items)} or B): {C.ENDC}").strip()
        
        if choice.upper() == 'B':
            return None
        
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(items):
                return items[idx]
        except (ValueError, IndexError):
            pass
        
        print(f"{C.RED}Invalid choice{C.ENDC}")
        input(f"{C.BOLD}Press Enter to retry...{C.ENDC}")

# ============================================================================
# GIT OPERATIONS
# ============================================================================
def get_git_commit(repo_path: str) -> Optional[str]:
    """Get current git commit hash."""
    success, stdout, _ = run_command(["git", "rev-parse", "HEAD"], cwd=repo_path, capture_output=True)
    return stdout.strip() if success else None

def git_stash_if_needed(repo_path: str) -> bool:
    """Stash uncommitted changes if any."""
    success, stdout, _ = run_command(["git", "status", "--porcelain"], cwd=repo_path, capture_output=True)
    if success and stdout.strip():
        print(f"{C.YELLOW}Stashing uncommitted changes...{C.ENDC}")
        return run_command(["git", "stash", "push", "-m", f"Auto-stash {datetime.now().isoformat()}"], cwd=repo_path)
    return True

def fix_detached_head(repo_path: str) -> bool:
    """Try to switch from detached HEAD to a branch."""
    success, stdout, _ = run_command(["git", "symbolic-ref", "--short", "HEAD"], cwd=repo_path, capture_output=True)
    if success and stdout.strip():
        return True  # Already on a branch
    
    print(f"{C.YELLOW}Detected detached HEAD, attempting to fix...{C.ENDC}")
    for branch in ["master", "main"]:
        # Check if branch exists
        exists, _, _ = run_command(["git", "show-ref", "--verify", f"refs/heads/{branch}"], cwd=repo_path, capture_output=True)
        if exists:
            if run_command(["git", "checkout", branch], cwd=repo_path):
                print(f"{C.GREEN}Switched to '{branch}'{C.ENDC}")
                return True
    
    print(f"{C.RED}Could not fix detached HEAD{C.ENDC}")
    return False

def update_repository(repo_path: str, repo_name: str, target_version: Optional[str] = None) -> bool:
    """Pull updates for a git repository."""
    print(f"\n{C.BOLD}Updating {repo_name}...{C.ENDC}")
    
    if not os.path.exists(os.path.join(repo_path, ".git")):
        print(f"{C.YELLOW}Not a git repository{C.ENDC}")
        return False
    
    current = get_git_commit(repo_path)
    if current:
        print(f"{C.DIM}Current: {current[:7]}{C.ENDC}")
    
    if not git_stash_if_needed(repo_path):
        return False
    
    if not fix_detached_head(repo_path):
        return False
    
    print(f"{C.CYAN}Fetching...{C.ENDC}")
    if not run_command(["git", "fetch", "--all"], cwd=repo_path):
        return False
    
    if target_version:
        print(f"{C.CYAN}Checking out {target_version}...{C.ENDC}")
        if not run_command(["git", "checkout", target_version], cwd=repo_path):
            return False
    else:
        if not run_command(["git", "pull", "--ff-only"], cwd=repo_path):
            print(f"{C.YELLOW}Fast-forward failed, trying merge...{C.ENDC}")
            if not run_command(["git", "pull"], cwd=repo_path):
                return False
    
    new_commit = get_git_commit(repo_path)
    if new_commit and current and new_commit != current:
        print(f"{C.GREEN}Updated: {current[:7]} → {new_commit[:7]}{C.ENDC}")
    elif new_commit:
        print(f"{C.GREEN}At: {new_commit[:7]}{C.ENDC}")
    
    return True

def update_requirements(repo_path: str, python_path: str, upgrade: bool = True) -> bool:
    """Install packages from requirements.txt."""
    req_file = os.path.join(repo_path, "requirements.txt")
    if not os.path.exists(req_file):
        print(f"{C.YELLOW}No requirements.txt found{C.ENDC}")
        return True
    
    cmd = [str(python_path), "-m", "pip", "install", "-r", req_file]
    if upgrade:
        cmd.append("--upgrade")
    
    return run_command(cmd, cwd=repo_path)

def get_custom_nodes(root_dir: str) -> List[Tuple[str, str]]:
    """Get list of custom node directories."""
    cn_dir = os.path.join(root_dir, "custom_nodes")
    if not os.path.exists(cn_dir):
        return []
    return sorted([
        (item, os.path.join(cn_dir, item))
        for item in os.listdir(cn_dir)
        if os.path.isdir(os.path.join(cn_dir, item))
    ])

# ============================================================================
# UPDATE MENUS
# ============================================================================
def update_menu(root_dir: str, python_path: str):
    """Update submenu."""
    while True:
        clear_screen()
        print_header("Update ComfyUI & Custom Nodes")
        print(f"\n  [1] Update ComfyUI (latest)")
        print(f"  [2] Update ComfyUI (specific version)")
        print(f"  [3] Update All Custom Nodes")
        print(f"  [4] Update Specific Custom Node")
        print(f"  [5] Update Everything")
        print(f"\n  [B] Back")
        
        choice = input(f"\n{C.BOLD}Choice: {C.ENDC}").strip().upper()
        
        if choice == 'B':
            break
        elif choice == '1':
            update_comfyui(root_dir, python_path)
        elif choice == '2':
            version = input(f"{C.BOLD}Version (commit/tag/branch): {C.ENDC}").strip()
            if version:
                update_comfyui(root_dir, python_path, version)
        elif choice == '3':
            update_all_nodes(root_dir, python_path)
        elif choice == '4':
            update_specific_node(root_dir, python_path)
        elif choice == '5':
            update_everything(root_dir, python_path)
        else:
            print(f"{C.RED}Invalid choice{C.ENDC}")
            time.sleep(1)

def update_comfyui(root_dir: str, python_path: str, target_version: Optional[str] = None):
    """Update ComfyUI core."""
    clear_screen()
    print_header("Update ComfyUI")
    
    if update_repository(root_dir, "ComfyUI", target_version):
        if input(f"\n{C.BOLD}Run pip install? (y/N): {C.ENDC}").strip().lower() == 'y':
            update_requirements(root_dir, python_path, upgrade=(target_version is None))
        print(f"\n{C.GREEN}✓ ComfyUI updated{C.ENDC}")
    else:
        print(f"\n{C.RED}✗ Update failed{C.ENDC}")
    
    input(f"\n{C.BOLD}Press Enter...{C.ENDC}")

def update_all_nodes(root_dir: str, python_path: str):
    """Update all git-based custom nodes."""
    clear_screen()
    print_header("Update All Custom Nodes")
    
    nodes = get_custom_nodes(root_dir)
    git_nodes = [(n, p) for n, p in nodes if os.path.exists(os.path.join(p, ".git"))]
    
    if not git_nodes:
        print(f"{C.YELLOW}No git-based custom nodes found{C.ENDC}")
        input(f"\n{C.BOLD}Press Enter...{C.ENDC}")
        return
    
    print(f"\nFound {len(git_nodes)} git-based nodes:")
    for name, _ in git_nodes:
        print(f"  • {name}")
    
    if input(f"\n{C.BOLD}Proceed? (y/N): {C.ENDC}").strip().lower() != 'y':
        return
    
    # Ask about pip install
    pip_mode = input(f"\n{C.BOLD}Run pip install? [1] Yes to all, [2] No to all, [3] Ask each (default: 2): {C.ENDC}").strip()
    
    success, fail = 0, 0
    for name, path in git_nodes:
        print(f"\n{C.CYAN}{'─' * 60}{C.ENDC}")
        if update_repository(path, name):
            success += 1
            if pip_mode == '1':
                update_requirements(path, python_path)
            elif pip_mode == '3':
                if input(f"{C.BOLD}Pip install for {name}? (y/N): {C.ENDC}").strip().lower() == 'y':
                    update_requirements(path, python_path)
        else:
            fail += 1
    
    print(f"\n{C.CYAN}{'─' * 60}{C.ENDC}")
    print(f"{C.GREEN}Success: {success}{C.ENDC} | {C.RED}Failed: {fail}{C.ENDC}")
    input(f"\n{C.BOLD}Press Enter...{C.ENDC}")

def update_specific_node(root_dir: str, python_path: str):
    """Update a specific custom node."""
    nodes = get_custom_nodes(root_dir)
    git_nodes = [(n, p) for n, p in nodes if os.path.exists(os.path.join(p, ".git"))]
    
    if not git_nodes:
        print(f"{C.YELLOW}No git-based custom nodes found{C.ENDC}")
        input(f"\n{C.BOLD}Press Enter...{C.ENDC}")
        return
    
    selected = select_from_list(git_nodes, "Custom Node", lambda x: x[0])
    if not selected:
        return
    
    name, path = selected
    version = input(f"\n{C.BOLD}Version (blank for latest): {C.ENDC}").strip() or None
    
    clear_screen()
    print_header(f"Update {name}")
    
    if update_repository(path, name, version):
        if input(f"\n{C.BOLD}Run pip install? (y/N): {C.ENDC}").strip().lower() == 'y':
            update_requirements(path, python_path, upgrade=(version is None))
    
    input(f"\n{C.BOLD}Press Enter...{C.ENDC}")

def update_everything(root_dir: str, python_path: str):
    """Update ComfyUI and all custom nodes."""
    clear_screen()
    print_header("Update Everything")
    
    nodes = get_custom_nodes(root_dir)
    git_nodes = [(n, p) for n, p in nodes if os.path.exists(os.path.join(p, ".git"))]
    
    print(f"This will update:\n  • ComfyUI\n  • {len(git_nodes)} custom nodes")
    
    if input(f"\n{C.BOLD}Proceed? (y/N): {C.ENDC}").strip().lower() != 'y':
        return
    
    comfy_pip = input(f"{C.BOLD}Pip install for ComfyUI? (y/N): {C.ENDC}").strip().lower() == 'y'
    nodes_pip = input(f"{C.BOLD}Pip install for nodes? [1] Yes, [2] No (default): {C.ENDC}").strip()
    
    # Update ComfyUI
    print(f"\n{C.CYAN}{'═' * 60}{C.ENDC}")
    if update_repository(root_dir, "ComfyUI") and comfy_pip:
        update_requirements(root_dir, python_path)
    
    # Update nodes
    for name, path in git_nodes:
        print(f"\n{C.CYAN}{'─' * 60}{C.ENDC}")
        if update_repository(path, name) and nodes_pip == '1':
            update_requirements(path, python_path)
    
    print(f"\n{C.GREEN}✓ Update complete{C.ENDC}")
    input(f"\n{C.BOLD}Press Enter...{C.ENDC}")

def install_node_requirements(root_dir: str, python_path: str):
    """Install requirements for a specific node."""
    clear_screen()
    print_header("Install Node Requirements")
    
    nodes = get_custom_nodes(root_dir)
    req_nodes = [(n, p) for n, p in nodes if os.path.exists(os.path.join(p, "requirements.txt"))]
    
    if not req_nodes:
        print(f"{C.YELLOW}No nodes with requirements.txt found{C.ENDC}")
        input(f"\n{C.BOLD}Press Enter...{C.ENDC}")
        return
    
    selected = select_from_list(req_nodes, "Node", lambda x: x[0])
    if not selected:
        return
    
    name, path = selected
    upgrade = input(f"{C.BOLD}Use --upgrade? (y/N): {C.ENDC}").strip().lower() == 'y'
    
    if update_requirements(path, python_path, upgrade):
        print(f"\n{C.GREEN}✓ Requirements installed for {name}{C.ENDC}")
    else:
        print(f"\n{C.RED}✗ Installation failed{C.ENDC}")
    
    input(f"\n{C.BOLD}Press Enter...{C.ENDC}")

# ============================================================================
# CONFIGURATION
# ============================================================================
def get_argument_definitions() -> dict:
    """Return available ComfyUI arguments with metadata."""
    return {
        "--listen": {"title": "Listen IP", "help": "IP for server (0.0.0.0 for network)", "type": "value", "category": "Network"},
        "--port": {"title": "Port", "help": "Server port (default: 8188)", "type": "value", "category": "Network"},
        "--highvram": {"title": "High VRAM", "help": "Optimize for >16GB VRAM", "type": "choice", "category": "VRAM"},
        "--normalvram": {"title": "Normal VRAM", "help": "Default mode for >8GB VRAM", "type": "choice", "category": "VRAM"},
        "--lowvram": {"title": "Low VRAM", "help": "For <=8GB VRAM (slower)", "type": "choice", "category": "VRAM"},
        "--disable-smart-memory": {"title": "Disable Smart Memory", "help": "Can help stability", "type": "boolean", "category": "VRAM"},
        "--use-flash-attention": {"title": "Flash Attention", "help": "Enable Flash Attention", "type": "boolean", "category": "Performance"},
        "--auto-launch": {"title": "Auto-Launch Browser", "help": "Open browser on start", "type": "boolean", "category": "UI"},
        "--custom-args": {"title": "Custom Arguments", "help": "Additional args (space-separated)", "type": "value", "category": "Advanced"},
    }

def load_presets() -> dict:
    """Load built-in and custom presets."""
    all_presets = {"built_in": PRESETS}
    custom_path = get_script_dir() / CUSTOM_PRESETS_FILE
    if custom_path.exists():
        try:
            with open(custom_path, 'r') as f:
                all_presets["custom"] = json.load(f)
        except Exception:
            pass
    return all_presets

def save_custom_preset(config: configparser.ConfigParser):
    """Save current config as a custom preset."""
    print_header("Save Custom Preset")
    
    if not config.has_section('Arguments') or not list(config.items('Arguments')):
        print(f"{C.YELLOW}No arguments configured{C.ENDC}")
        input(f"\n{C.BOLD}Press Enter...{C.ENDC}")
        return
    
    name = input(f"\n{C.BOLD}Preset name: {C.ENDC}").strip()
    if not name:
        return
    
    presets = load_presets().get("custom", {})
    key = name.lower().replace(" ", "_")
    presets[key] = {
        "name": f"⭐ {name}",
        "description": "Custom preset",
        "args": dict(config.items('Arguments'))
    }
    
    try:
        with open(get_script_dir() / CUSTOM_PRESETS_FILE, 'w') as f:
            json.dump(presets, f, indent=2)
        print(f"\n{C.GREEN}✓ Preset '{name}' saved{C.ENDC}")
    except Exception as e:
        print(f"\n{C.RED}Error: {e}{C.ENDC}")
    
    input(f"\n{C.BOLD}Press Enter...{C.ENDC}")

def apply_preset_menu(config: configparser.ConfigParser):
    """Menu to apply a preset."""
    print_header("Apply Preset")
    
    all_presets = load_presets()
    preset_list = []
    
    for category, presets in all_presets.items():
        if presets:
            print(f"\n{C.BOLD}{category.replace('_', ' ').title()}:{C.ENDC}")
            for key, preset in presets.items():
                preset_list.append(preset)
                print(f"  [{len(preset_list)}] {preset['name']}")
                print(f"      {C.DIM}{preset['description']}{C.ENDC}")
    
    choice = input(f"\n{C.BOLD}Select (or B to cancel): {C.ENDC}").strip()
    if choice.upper() == 'B':
        return
    
    try:
        preset = preset_list[int(choice) - 1]
        if input(f"{C.YELLOW}Replace current settings? (y/N): {C.ENDC}").lower() == 'y':
            if config.has_section('Arguments'):
                config.remove_section('Arguments')
            config.add_section('Arguments')
            for arg, val in preset['args'].items():
                config.set('Arguments', arg, val)
            
            config_path = config.get('DEFAULT', 'config_file_path')
            with open(config_path, 'w') as f:
                config.write(f)
            print(f"\n{C.GREEN}✓ Preset applied{C.ENDC}")
    except (ValueError, IndexError):
        print(f"{C.RED}Invalid selection{C.ENDC}")
    
    input(f"\n{C.BOLD}Press Enter...{C.ENDC}")

def edit_arguments_menu(config: configparser.ConfigParser):
    """Menu to edit launch arguments."""
    arg_defs = get_argument_definitions()
    
    while True:
        clear_screen()
        print_header("Edit Launch Arguments")
        
        # Group by category
        categories = {}
        for arg, detail in arg_defs.items():
            cat = detail.get('category', 'Other')
            categories.setdefault(cat, []).append((arg, detail))
        
        all_args = []
        for cat in sorted(categories.keys()):
            print(f"\n{C.BOLD}--- {cat} ---{C.ENDC}")
            for arg, detail in sorted(categories[cat]):
                all_args.append((arg, detail))
                current = config.get('Arguments', arg, fallback='DISABLED')
                
                if current == 'DISABLED':
                    status = f"{C.DIM}DISABLED{C.ENDC}"
                elif current == 'true':
                    status = f"{C.GREEN}✓ ENABLED{C.ENDC}"
                else:
                    status = f"{C.YELLOW}{current}{C.ENDC}"
                
                print(f"  [{len(all_args):2}] {detail['title']}: {status}")
        
        print(f"\n  [B] Back")
        
        choice = input(f"\n{C.BOLD}Select (1-{len(all_args)} or B): {C.ENDC}").strip().upper()
        
        if choice == 'B':
            return
        
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(all_args):
                arg, detail = all_args[idx]
                edit_single_argument(config, arg, detail)
        except ValueError:
            print(f"{C.RED}Invalid choice{C.ENDC}")
            time.sleep(1)

def edit_single_argument(config: configparser.ConfigParser, arg: str, detail: dict):
    """Edit a single argument."""
    clear_screen()
    print_header(f"Configure: {detail['title']}")
    
    current = config.get('Arguments', arg, fallback=None)
    print(f"\n{C.DIM}Help: {detail.get('help', 'N/A')}{C.ENDC}")
    print(f"Current: {C.YELLOW}{current or 'DISABLED'}{C.ENDC}")
    
    arg_type = detail.get('type', 'boolean')
    
    if arg_type in ('boolean', 'choice'):
        choice = input(f"\n{C.BOLD}Enable (Y) / Disable (D) / Cancel (B): {C.ENDC}").strip().upper()
        if choice == 'Y':
            config.set('Arguments', arg, 'true')
            print(f"{C.GREEN}Enabled{C.ENDC}")
        elif choice == 'D' and config.has_option('Arguments', arg):
            config.remove_option('Arguments', arg)
            print(f"{C.GREEN}Disabled{C.ENDC}")
    else:
        value = input(f"\n{C.BOLD}Value (D to disable, B to cancel): {C.ENDC}").strip()
        if value.upper() == 'D' and config.has_option('Arguments', arg):
            config.remove_option('Arguments', arg)
            print(f"{C.GREEN}Disabled{C.ENDC}")
        elif value.upper() != 'B' and value:
            config.set('Arguments', arg, value)
            print(f"{C.GREEN}Set to: {value}{C.ENDC}")
    
    # Save
    config_path = config.get('DEFAULT', 'config_file_path')
    if config.has_section('Arguments'):
        with open(config_path, 'w') as f:
            config.write(f)
    
    input(f"\n{C.BOLD}Press Enter...{C.ENDC}")

def launch_comfyui(config: configparser.ConfigParser, root_dir: str, python_path: str):
    """Launch ComfyUI with configured arguments."""
    clear_screen()
    print_header("Launching ComfyUI")
    
    args = []
    if config.has_section('Arguments'):
        custom_args = config.get('Arguments', '--custom-args', fallback=None)
        for key, value in config.items('Arguments'):
            if key == '--custom-args' and custom_args:
                args.extend(custom_args.split())
            elif key != '--custom-args':
                if value == 'true':
                    args.append(key)
                else:
                    args.extend([key, value])
    
    command = [str(python_path), "main.py"] + args
    run_command(command, cwd=root_dir)
    input(f"\n{C.BOLD}Press Enter to return...{C.ENDC}")

# ============================================================================
# MAIN
# ============================================================================
def main():
    """Main application entry point."""
    script_dir = get_script_dir()
    config_file = script_dir / CONFIG_FILE
    config = configparser.ConfigParser()
    config.read(config_file)
    
    def save_paths():
        with open(config_file, 'w') as f:
            config.write(f)
    
    # Load saved paths
    comfyui_root = config.get('Paths', 'comfyui_root', fallback=None)
    python_exec = config.get('Paths', 'python_executable', fallback=None)
    
    # Validate paths
    if not comfyui_root or not Path(comfyui_root, "main.py").exists():
        comfyui_root = None
    if not python_exec or not Path(python_exec).exists():
        python_exec = None
    
    # Main loop
    while True:
        # Select ComfyUI root if needed
        if not comfyui_root:
            roots = find_comfyui_roots(script_dir)
            selected = select_from_list(roots, "ComfyUI Root", str)
            if not selected:
                print(f"{C.RED}No ComfyUI root selected. Exiting.{C.ENDC}")
                sys.exit(1)
            comfyui_root = str(selected)
            if not config.has_section('Paths'):
                config.add_section('Paths')
            config.set('Paths', 'comfyui_root', comfyui_root)
            save_paths()
            python_exec = None  # Force re-selection
        
        # Select Python if needed
        if not python_exec:
            pythons = find_python_executables(Path(comfyui_root))
            selected = select_from_list(pythons, "Python Environment", lambda x: f"{x[0]} ({x[1]})")
            if not selected:
                print(f"{C.RED}No Python selected. Exiting.{C.ENDC}")
                sys.exit(1)
            python_exec = str(selected[1])
            if not config.has_section('Paths'):
                config.add_section('Paths')
            config.set('Paths', 'python_executable', python_exec)
            save_paths()
        
        # Ensure Arguments section exists
        if not config.has_section('Arguments'):
            config.add_section('Arguments')
        config.set('DEFAULT', 'config_file_path', str(config_file))
        
        # Show main menu
        clear_screen()
        print(f"{C.BOLD}{C.CYAN}╔══════════════════════════════════════════════════════════════╗{C.ENDC}")
        print(f"{C.BOLD}{C.CYAN}║           🚀 ComfyUI Launcher v{APP_VERSION} 🚀                    ║{C.ENDC}")
        print(f"{C.BOLD}{C.CYAN}╚══════════════════════════════════════════════════════════════╝{C.ENDC}")
        print(f"{C.DIM}Root: {comfyui_root}")
        print(f"Python: {python_exec}{C.ENDC}\n")
        
        print(f"  {C.HEADER}{C.BOLD}[L] Launch ComfyUI{C.ENDC}")
        
        print(f"\n{C.BOLD}Configuration:{C.ENDC}")
        print(f"  [1] Apply Preset")
        print(f"  [2] Save Current as Preset")
        print(f"  [3] Edit Arguments")
        
        print(f"\n{C.BOLD}Maintenance:{C.ENDC}")
        print(f"  [4] Update Menu")
        print(f"  [5] Install Node Requirements")
        print(f"  [S] Select Root/Python")
        
        print(f"\n  [Q] Quit")
        
        choice = input(f"\n{C.BOLD}Choice: {C.ENDC}").strip().upper()
        
        if choice == 'L':
            launch_comfyui(config, comfyui_root, python_exec)
        elif choice == '1':
            apply_preset_menu(config)
        elif choice == '2':
            save_custom_preset(config)
        elif choice == '3':
            edit_arguments_menu(config)
        elif choice == '4':
            update_menu(comfyui_root, python_exec)
        elif choice == '5':
            install_node_requirements(comfyui_root, python_exec)
        elif choice == 'S':
            comfyui_root = None
            python_exec = None
            if config.has_section('Paths'):
                config.remove_section('Paths')
                save_paths()
        elif choice == 'Q':
            print(f"\n{C.CYAN}Goodbye!{C.ENDC}")
            sys.exit(0)
        else:
            print(f"{C.RED}Invalid choice{C.ENDC}")
            time.sleep(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{C.YELLOW}Interrupted by user{C.ENDC}")
        sys.exit(0)
    except Exception as e:
        print(f"\n{C.RED}Error: {type(e).__name__}: {e}{C.ENDC}")
        import traceback
        traceback.print_exc()
        input("\nPress Enter to exit...")
        sys.exit(1)
