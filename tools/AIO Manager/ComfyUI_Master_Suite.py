#!/usr/bin/env python3
"""
ComfyUI Master Management Suite v3.0.0

A comprehensive tool for managing ComfyUI installations, virtual environments,
PyTorch versions, and accelerator packages.

Features:
- Setup Wizard for new ComfyUI installations
- PyTorch Manager with GPU detection and version selection
- Requirements Scanner & Installer
- Virtual Environment Management
- Accelerator Manager (Flash Attention, Sage Attention, Triton, xFormers)
- Snapshot System for backup/restore
- External wheels.json database for easy updates

Author: MDMAchine
"""

import os
import sys
import platform
import subprocess
import json
import shutil
import re
import logging
import time
import concurrent.futures
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict

# ============================================================================
# AUTO-INSTALL DEPENDENCIES
# ============================================================================
def ensure_packages():
    """Ensure required packages are installed."""
    required = [('packaging', 'packaging'), ('requests', 'requests')]
    for import_name, pip_name in required:
        try:
            __import__(import_name)
        except ImportError:
            print(f"Installing required '{pip_name}' library...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name, "-q"])

ensure_packages()
from packaging.version import parse as parse_version
import requests

# ============================================================================
# CONSTANTS
# ============================================================================
APP_VERSION = "3.0.0"
SNAPSHOT_DIR_NAME = ".comfyui_snapshots"
WHEELS_DB_FILENAME = "wheels.json"

# ============================================================================
# COLORS
# ============================================================================
class Colors:
    """ANSI color codes with automatic Windows compatibility."""
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
    def _get(cls, code: str) -> str:
        return code if cls._enabled else ''

    def __getattribute__(self, name: str):
        if name.startswith('_') or name in ('disable', 'enable', 'init'):
            return object.__getattribute__(self, name)
        return Colors._get(object.__getattribute__(self, name))

    @classmethod
    def disable(cls):
        cls._enabled = False

    @classmethod
    def enable(cls):
        cls._enabled = True

    @staticmethod
    def init():
        if platform.system() == "Windows":
            try:
                import ctypes
                kernel32 = ctypes.windll.kernel32
                handle = kernel32.GetStdHandle(-11)
                mode = ctypes.c_ulong()
                if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                    kernel32.SetConsoleMode(handle, mode.value | 0x0004)
            except Exception:
                Colors.disable()
        elif not sys.stdout.isatty():
            Colors.disable()

Colors.init()
C = Colors()

# ============================================================================
# LOGGING
# ============================================================================
def setup_logging(log_dir: Path) -> logging.Logger:
    """Configure logging with file and console handlers."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"comfyui_manager_{datetime.now():%Y%m%d_%H%M%S}.log"

    logger = logging.getLogger('comfyui_manager')
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    try:
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'))
        logger.addHandler(fh)
    except Exception as e:
        print(f"Warning: Could not set up file logging: {e}", file=sys.stderr)

    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    ch.setFormatter(logging.Formatter(f'{C.RED}%(levelname)s:{C.ENDC} %(message)s'))
    logger.addHandler(ch)
    logger.propagate = False
    return logger

# ============================================================================
# DATA MODELS
# ============================================================================
@dataclass
class EnvironmentSnapshot:
    """Represents a snapshot of installed packages."""
    timestamp: str
    packages: Dict[str, str]
    description: str
    python_version: str

# ============================================================================
# UTILITIES
# ============================================================================
def clear_screen():
    os.system('cls' if platform.system() == 'Windows' else 'clear')

def print_header(text: str):
    print(f"\n{C.BOLD}{C.CYAN}{'=' * 80}\n  {text}\n{'=' * 80}{C.ENDC}")

def print_subheader(text: str):
    print(f"\n{C.BOLD}{C.BLUE}{'─' * 80}\n  {text}\n{'─' * 80}{C.ENDC}")

def print_success(text: str):
    print(f"{C.GREEN}✓ {text}{C.ENDC}")

def print_warning(text: str):
    print(f"{C.YELLOW}⚠ {text}{C.ENDC}")

def print_error(text: str):
    print(f"{C.RED}✗ {text}{C.ENDC}")

def print_info(text: str):
    print(f"{C.CYAN}ℹ {text}{C.ENDC}")

def confirm_action(prompt: str, default: bool = False) -> bool:
    """Prompt user for confirmation."""
    hint = "Y/n" if default else "y/N"
    response = input(f"{C.BOLD}{prompt} [{hint}]: {C.ENDC}").strip().lower()
    return response in ('y', 'yes') if response else default

def get_script_dir() -> Path:
    """Get the directory containing this script."""
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd()

def find_comfyui_root() -> Optional[str]:
    """Auto-detect ComfyUI root directory."""
    for directory in [get_script_dir()] + list(get_script_dir().parents)[:3]:
        if directory.is_dir() and (directory / "main.py").exists() and (directory / "custom_nodes").is_dir():
            return str(directory)
    return None

# ============================================================================
# CONFIG MANAGER
# ============================================================================
class ConfigManager:
    """Manages application configuration with JSON persistence."""
    
    def __init__(self, config_path: Path, logger: logging.Logger):
        self.config_path = config_path
        self.logger = logger
        self.config = self._load_config()

    def _get_default_config(self) -> Dict:
        """Return default configuration."""
        return {
            'version': APP_VERSION,
            'active_comfyui_root': '',
            'custom_venv_paths': [],
            'pytorch': {
                'default_version': 'stable_cu129',
                'available_versions': {
                    'stable_cu129': {
                        'name': 'Stable CUDA 12.9 (Recommended)',
                        'index_url': 'https://download.pytorch.org/whl/cu129',
                        'use_extra_index': True
                    },
                    'stable_cu130': {
                        'name': 'Stable CUDA 13.0',
                        'index_url': 'https://download.pytorch.org/whl/cu130',
                        'use_extra_index': True
                    },
                    'nightly_cu130': {
                        'name': 'Nightly CUDA 13.0 (Experimental)',
                        'index_url': 'https://download.pytorch.org/whl/nightly/cu130',
                        'use_extra_index': False
                    },
                    'cpu': {
                        'name': 'CPU Only',
                        'index_url': 'https://download.pytorch.org/whl/cpu',
                        'use_extra_index': False
                    },
                    'pt_2_8_0_cu129': {
                        'name': 'PyTorch 2.8.0 (CUDA 12.9)',
                        'index_url': 'https://download.pytorch.org/whl/cu129',
                        'packages': ['torch==2.8.0', 'torchvision==0.23.0', 'torchaudio==2.8.0'],
                        'python_versions': ['3.11', '3.12', '3.13'],
                        'use_extra_index': False
                    },
                    'pt_2_7_1_cu128': {
                        'name': 'PyTorch 2.7.1 (CUDA 12.8)',
                        'index_url': 'https://download.pytorch.org/whl/cu128',
                        'packages': ['torch==2.7.1', 'torchvision==0.22.1', 'torchaudio==2.7.1'],
                        'python_versions': ['3.11', '3.12'],
                        'use_extra_index': False
                    },
                }
            },
            'requirements': {
                'ignore_patterns': ['.venv', 'venv', 'python_embeded', '__pycache__', '.git'],
                'ignore_packages': [
                    'comfyui-frontend-package', 'comfyui-workflow-templates',
                    'pytest', 'pytest-aiohttp', 'pytest-asyncio', 'websocket-client',
                    'comfy', 'folder_paths', 'nodes', 'models'
                ],
                'upgrade_only': True
            },
            'snapshots': {'enabled': True, 'max_snapshots': 10, 'auto_create': True},
            'ui': {'colors_enabled': True, 'show_pip_output': False},
            'structure': {
                'stable_diffusion_root': '',
                'comfyui_base_name': 'ComfyUI_Base',
                'venvs_dir_name': 'venvs'
            },
            'last_active_python': ''
        }

    def _load_config(self) -> Dict:
        """Load config from file, merging with defaults."""
        default = self._get_default_config()
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                self._deep_merge(default, loaded)
                self.logger.info(f"Config loaded from {self.config_path}")
            except (json.JSONDecodeError, Exception) as e:
                self.logger.error(f"Error loading config: {e}. Using defaults.")
        return default

    def _deep_merge(self, base: Dict, override: Dict):
        """Recursively merge override into base."""
        for key, value in override.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    def save_config(self) -> bool:
        """Save configuration to file."""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Config saved to {self.config_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save config: {e}")
            return False

    def get(self, key_path: str, default=None):
        """Get nested config value using dot notation."""
        try:
            value = self.config
            for key in key_path.split('.'):
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

    def set(self, key_path: str, value):
        """Set nested config value using dot notation."""
        keys = key_path.split('.')
        d = self.config
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        d[keys[-1]] = value

# ============================================================================
# SNAPSHOT MANAGER
# ============================================================================
class SnapshotManager:
    """Manages environment snapshots for backup/restore."""
    
    def __init__(self, comfyui_root: Path, logger: logging.Logger, config: ConfigManager):
        self.snapshot_dir = comfyui_root / SNAPSHOT_DIR_NAME
        self.snapshot_dir.mkdir(exist_ok=True)
        self.logger = logger
        self.config = config

    def create_snapshot(self, python_exec: str, description: str = "") -> Optional[str]:
        """Create a snapshot of current packages."""
        if not self.config.get('snapshots.enabled', True):
            return None
        try:
            result = subprocess.run(
                [python_exec, "-m", "pip", "list", "--format=json"],
                capture_output=True, text=True, check=True, timeout=30
            )
            packages = {pkg['name'].lower(): pkg['version'] for pkg in json.loads(result.stdout)}
            py_version = subprocess.check_output([python_exec, "--version"], text=True, timeout=5).strip()
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            snapshot = EnvironmentSnapshot(
                timestamp=timestamp,
                packages=packages,
                description=description or f"Snapshot at {datetime.now():%Y-%m-%d %H:%M}",
                python_version=py_version
            )
            
            with open(self.snapshot_dir / f"snapshot_{timestamp}.json", 'w', encoding='utf-8') as f:
                json.dump(asdict(snapshot), f, indent=2)
            
            print_success(f"Snapshot '{timestamp}' created")
            self._cleanup_old_snapshots()
            return timestamp
        except Exception as e:
            self.logger.error(f"Failed to create snapshot: {e}")
            print_warning(f"Could not create snapshot: {e}")
            return None

    def list_snapshots(self) -> List[EnvironmentSnapshot]:
        """List all available snapshots."""
        snapshots = []
        for f in sorted(self.snapshot_dir.glob("snapshot_*.json"), reverse=True):
            try:
                with open(f, 'r', encoding='utf-8') as fp:
                    data = json.load(fp)
                snapshots.append(EnvironmentSnapshot(**data))
            except Exception as e:
                self.logger.warning(f"Failed to load {f.name}: {e}")
        return snapshots

    def restore_snapshot(self, timestamp: str, python_exec: str) -> bool:
        """Restore packages from a snapshot."""
        snapshot_file = self.snapshot_dir / f"snapshot_{timestamp}.json"
        if not snapshot_file.exists():
            print_error(f"Snapshot not found: {timestamp}")
            return False
        
        try:
            with open(snapshot_file, 'r', encoding='utf-8') as f:
                snapshot = EnvironmentSnapshot(**json.load(f))
            
            print_info(f"Restoring: {snapshot.description} ({len(snapshot.packages)} packages)")
            if not confirm_action("This will reinstall packages. Continue?", False):
                return False
            
            to_install = [f"{name}=={version}" for name, version in snapshot.packages.items()]
            print_info(f"Installing {len(to_install)} packages...")
            subprocess.run(
                [python_exec, "-m", "pip", "install", "--force-reinstall", "--no-cache-dir"] + to_install,
                check=True
            )
            print_success("Snapshot restored!")
            return True
        except Exception as e:
            self.logger.error(f"Restore failed: {e}")
            print_error(f"Restore failed: {e}")
            return False

    def _cleanup_old_snapshots(self):
        """Remove old snapshots beyond max limit."""
        max_keep = self.config.get('snapshots.max_snapshots', 10)
        snapshots = sorted(self.snapshot_dir.glob("snapshot_*.json"), key=os.path.getmtime)
        for f in snapshots[:-max_keep]:
            try:
                f.unlink()
            except Exception:
                pass

# ============================================================================
# PYTORCH MANAGER
# ============================================================================
class PyTorchManager:
    """Manages PyTorch installation with GPU detection."""
    
    def __init__(self, python_exec: str, config: ConfigManager, logger: logging.Logger):
        self.python_exec = python_exec
        self.config = config
        self.logger = logger

    def get_python_version(self) -> Optional[str]:
        """Get Python version in X.Y format."""
        try:
            output = subprocess.check_output([self.python_exec, "--version"], text=True, timeout=5)
            match = re.search(r'Python (\d+\.\d+)', output)
            return match.group(1) if match else None
        except Exception:
            return None

    def check_torch_status(self) -> Tuple[Optional[str], bool, Optional[str], Optional[Dict]]:
        """Check current PyTorch installation status."""
        code = """
import json, torch
try:
    cuda = torch.cuda.is_available()
    dev = {}
    if cuda and torch.cuda.device_count() > 0:
        p = torch.cuda.get_device_properties(0)
        dev = {'count': torch.cuda.device_count(), 'name': p.name, 'capability': f'{p.major}.{p.minor}'}
    print(json.dumps({'version': torch.__version__, 'cuda': cuda, 'cuda_ver': torch.version.cuda, 'dev': dev}))
except Exception as e:
    print(json.dumps({'error': str(e)}))
"""
        try:
            result = subprocess.run([self.python_exec, "-c", code], capture_output=True, text=True, timeout=20)
            data = json.loads(result.stdout)
            return data.get('version'), data.get('cuda', False), data.get('cuda_ver'), data.get('dev')
        except Exception:
            return None, False, None, None

    def interactive_install(self, snapshot_mgr: Optional[SnapshotManager] = None):
        """Interactive PyTorch installation menu."""
        clear_screen()
        print_header("PyTorch Installation Manager")
        
        py_ver = self.get_python_version()
        if py_ver:
            print(f"{C.DIM}Active Python: {py_ver}{C.ENDC}")
        
        version, has_cuda, cuda_ver, dev_info = self.check_torch_status()
        if version:
            print_subheader("Current Installation")
            print(f"  PyTorch: {C.CYAN}{version}{C.ENDC}")
            if has_cuda and dev_info:
                print(f"  CUDA: {C.GREEN}Enabled ({cuda_ver}){C.ENDC}")
                print(f"  GPU: {C.CYAN}{dev_info.get('name', 'Unknown')}{C.ENDC}")
            else:
                print(f"  CUDA: {C.YELLOW}Disabled{C.ENDC}")
            if not confirm_action("\nReinstall PyTorch?", False):
                return
        else:
            print_warning("PyTorch not detected")
        
        # Build version menu
        print_subheader("Available Versions")
        versions = self.config.get('pytorch.available_versions', {})
        version_list = []
        for key, info in versions.items():
            supported_py = info.get('python_versions')
            if not supported_py or not py_ver or py_ver in supported_py:
                version_list.append((key, info))
        
        if not version_list:
            print_error("No compatible versions found")
            input(f"\n{C.BOLD}Press Enter...{C.ENDC}")
            return
        
        default_key = self.config.get('pytorch.default_version')
        for i, (key, info) in enumerate(version_list, 1):
            rec = f" {C.GREEN}(Recommended){C.ENDC}" if key == default_key else ""
            print(f"  [{i}] {C.CYAN}{info['name']}{C.ENDC}{rec}")
        print(f"  [B] Back")
        
        choice = input(f"\n{C.BOLD}Choice: {C.ENDC}").strip()
        if choice.upper() == 'B':
            return
        
        try:
            idx = int(choice) - 1
            if not (0 <= idx < len(version_list)):
                raise ValueError()
        except ValueError:
            print_error("Invalid choice")
            input(f"\n{C.BOLD}Press Enter...{C.ENDC}")
            return
        
        key, info = version_list[idx]
        
        # Create snapshot
        if snapshot_mgr and self.config.get('snapshots.auto_create'):
            print_info("Creating safety snapshot...")
            snapshot_mgr.create_snapshot(self.python_exec, f"Before PyTorch '{info['name']}'")
        
        # Build command
        packages = info.get('packages', ['torch', 'torchvision', 'torchaudio'])
        index_url = info['index_url']
        use_extra = info.get('use_extra_index', False)
        
        cmd = [self.python_exec, "-m", "pip", "install", "--force-reinstall", "--no-cache-dir"]
        if 'nightly' in key:
            cmd += ["--pre"] + packages + ["--index-url", index_url]
        elif use_extra:
            cmd += packages + ["--extra-index-url", index_url]
        else:
            cmd += packages + ["--index-url", index_url]
        
        print_subheader("Installing PyTorch")
        print_warning("This may take a long time...")
        
        try:
            show_output = self.config.get('ui.show_pip_output', False)
            subprocess.run(cmd, check=True, capture_output=not show_output)
            print_success("PyTorch installed!")
            
            # Verify
            new_ver, new_cuda, new_cuda_ver, _ = self.check_torch_status()
            if new_ver:
                cuda_str = f"CUDA {new_cuda_ver}" if new_cuda else "CPU"
                print_success(f"Verified: PyTorch {new_ver} ({cuda_str})")
        except subprocess.CalledProcessError as e:
            print_error(f"Installation failed (exit {e.returncode})")
            self.logger.error(f"PyTorch install failed: {e}")
        
        input(f"\n{C.BOLD}Press Enter...{C.ENDC}")

# ============================================================================
# REQUIREMENTS MANAGER
# ============================================================================
class RequirementsManager:
    """Scans and installs requirements from custom nodes."""
    
    def __init__(self, python_exec: str, comfyui_root: Path, config: ConfigManager, logger: logging.Logger):
        self.python_exec = python_exec
        self.comfyui_root = comfyui_root
        self.config = config
        self.logger = logger

    def find_requirements_files(self) -> List[Path]:
        """Find all requirements.txt files."""
        ignore = self.config.get('requirements.ignore_patterns', [])
        return [p for p in self.comfyui_root.rglob('requirements.txt')
                if not any(pat in str(p) for pat in ignore)]

    def get_installed_packages(self) -> Dict[str, str]:
        """Get currently installed packages."""
        try:
            result = subprocess.run(
                [self.python_exec, "-m", "pip", "list", "--format=json"],
                capture_output=True, text=True, check=True, timeout=30
            )
            return {pkg['name'].lower(): pkg['version'] for pkg in json.loads(result.stdout)}
        except Exception:
            return {}

    def scan_and_install(self, snapshot_mgr: Optional[SnapshotManager] = None):
        """Scan and install missing requirements."""
        clear_screen()
        print_header("Requirements Scanner & Installer")
        
        print_info("Scanning for requirements...")
        req_files = self.find_requirements_files()
        if not req_files:
            print_warning("No requirements files found")
            input(f"\n{C.BOLD}Press Enter...{C.ENDC}")
            return
        
        # Consolidate requirements
        ignore_pkgs = set(self.config.get('requirements.ignore_packages', []))
        all_reqs = {}
        for f in req_files:
            try:
                source = f.relative_to(self.comfyui_root / 'custom_nodes').parts[0]
            except ValueError:
                source = f.parent.name
            
            with open(f, 'r', encoding='utf-8', errors='replace') as fp:
                for line in fp:
                    line = line.strip()
                    if not line or line.startswith('#') or line.startswith('-'):
                        continue
                    match = re.match(r'^([a-zA-Z0-9_.-]+)', line.split(';')[0].split('#')[0])
                    if match:
                        name = match.group(1).lower()
                        if name not in ignore_pkgs:
                            all_reqs.setdefault(name, {'line': line, 'sources': set()})
                            all_reqs[name]['sources'].add(source)
        
        installed = self.get_installed_packages()
        missing = {k: v for k, v in all_reqs.items() if k not in installed}
        
        print(f"\nTotal: {C.CYAN}{len(all_reqs)}{C.ENDC} | Installed: {C.GREEN}{len(all_reqs) - len(missing)}{C.ENDC} | Missing: {C.YELLOW}{len(missing)}{C.ENDC}")
        
        if not missing:
            print_success("All requirements satisfied!")
            input(f"\n{C.BOLD}Press Enter...{C.ENDC}")
            return
        
        print(f"\n{C.BOLD}Missing packages:{C.ENDC}")
        for name, data in list(missing.items())[:15]:
            print(f"  • {name} {C.DIM}[{', '.join(data['sources'])}]{C.ENDC}")
        if len(missing) > 15:
            print(f"  {C.DIM}... and {len(missing) - 15} more{C.ENDC}")
        
        if not confirm_action(f"\nInstall {len(missing)} packages?", True):
            return
        
        if snapshot_mgr and self.config.get('snapshots.auto_create'):
            print_info("Creating snapshot...")
            snapshot_mgr.create_snapshot(self.python_exec, "Before requirements install")
        
        print_subheader("Installing")
        success, failed = 0, []
        for name, data in missing.items():
            print(f"Installing {C.CYAN}{name}{C.ENDC}...", end=' ', flush=True)
            try:
                subprocess.run(
                    [self.python_exec, "-m", "pip", "install", "--upgrade", "--no-cache-dir", data['line']],
                    check=True, capture_output=True, timeout=300
                )
                print(f"{C.GREEN}✓{C.ENDC}")
                success += 1
            except Exception as e:
                print(f"{C.RED}✗{C.ENDC}")
                failed.append(name)
                self.logger.error(f"Failed to install {name}: {e}")
        
        print(f"\n{C.GREEN}Installed: {success}{C.ENDC}")
        if failed:
            print(f"{C.RED}Failed: {len(failed)}{C.ENDC} - {', '.join(failed)}")
        
        input(f"\n{C.BOLD}Press Enter...{C.ENDC}")

# ============================================================================
# ACCELERATOR MANAGER
# ============================================================================
class AcceleratorManager:
    """Manages accelerator packages (Flash Attention, Triton, etc.)."""
    
    def __init__(self, python_exec: str, config: ConfigManager, logger: logging.Logger):
        self.python_exec = python_exec
        self.config = config
        self.logger = logger
        self.platform = sys.platform
        self.wheels_db = self._load_wheels_db()

    def _load_wheels_db(self) -> Dict:
        """Load wheels database from external JSON."""
        db_path = get_script_dir() / WHEELS_DB_FILENAME
        if db_path.exists():
            try:
                with open(db_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Failed to load wheels.json: {e}")
        return {}

    def get_python_cpversion(self) -> Optional[str]:
        """Get Python version as cpXY format."""
        try:
            output = subprocess.check_output([self.python_exec, "--version"], text=True, timeout=5)
            match = re.search(r'Python (\d+)\.(\d+)', output)
            return f"cp{match.group(1)}{match.group(2)}" if match else None
        except Exception:
            return None

    def _pip_install(self, args: List[str]) -> bool:
        """Run pip install with given args."""
        cmd = [self.python_exec, "-m", "pip", "install"] + args
        print_info(f"Running: pip install {' '.join(args)}")
        try:
            show = self.config.get('ui.show_pip_output', False)
            subprocess.run(cmd, check=True, capture_output=not show)
            print_success("Installation successful!")
            return True
        except subprocess.CalledProcessError as e:
            print_error("Installation failed!")
            self.logger.error(f"pip install failed: {e}")
            return False

    def install_wheels(self):
        """Install pre-compiled wheels for current setup."""
        clear_screen()
        print_header("Install Pre-compiled Wheels")
        
        if not self.wheels_db:
            print_error("wheels.json not found. Please ensure it's in the same directory.")
            input(f"\n{C.BOLD}Press Enter...{C.ENDC}")
            return
        
        py_ver = self.get_python_cpversion()
        if not py_ver:
            print_error("Could not detect Python version")
            input(f"\n{C.BOLD}Press Enter...{C.ENDC}")
            return
        
        print_info(f"Python: {C.CYAN}{py_ver}{C.ENDC}")
        torch_ver = input(f"\n{C.BOLD}PyTorch version (e.g., 2.8.0): {C.ENDC}").strip()
        
        if torch_ver not in self.wheels_db:
            print_error(f"No wheels for PyTorch {torch_ver}. Available: {', '.join(self.wheels_db.keys())}")
            input(f"\n{C.BOLD}Press Enter...{C.ENDC}")
            return
        
        if py_ver not in self.wheels_db[torch_ver]:
            print_error(f"No wheels for {py_ver}. Available: {', '.join(self.wheels_db[torch_ver].keys())}")
            input(f"\n{C.BOLD}Press Enter...{C.ENDC}")
            return
        
        platform_key = 'win32' if self.platform == 'win32' else 'linux'
        wheels = self.wheels_db[torch_ver][py_ver]
        
        available = []
        for name, platforms in wheels.items():
            if platform_key in platforms:
                available.append((name, platforms[platform_key]))
        
        if not available:
            print_warning("No wheels available for your platform")
            input(f"\n{C.BOLD}Press Enter...{C.ENDC}")
            return
        
        print(f"\n{C.BOLD}Available wheels:{C.ENDC}")
        for i, (name, _) in enumerate(available, 1):
            print(f"  [{i}] {name}")
        print(f"  [A] Install All")
        print(f"  [B] Back")
        
        choice = input(f"\n{C.BOLD}Choice: {C.ENDC}").strip().upper()
        
        if choice == 'B':
            return
        elif choice == 'A':
            to_install = available
        else:
            try:
                idx = int(choice) - 1
                to_install = [available[idx]]
            except (ValueError, IndexError):
                print_error("Invalid choice")
                input(f"\n{C.BOLD}Press Enter...{C.ENDC}")
                return
        
        for name, url in to_install:
            print_info(f"\nInstalling {name}...")
            self._pip_install(["--no-cache-dir", url])
        
        input(f"\n{C.BOLD}Press Enter...{C.ENDC}")

    def install_xformers(self):
        """Install xFormers from PyPI."""
        clear_screen()
        print_header("Install xFormers")
        if confirm_action("Install xformers from PyPI?", True):
            self._pip_install(["xformers"])
        input(f"\n{C.BOLD}Press Enter...{C.ENDC}")

    def install_triton(self):
        """Install Triton (Windows version)."""
        clear_screen()
        print_header("Install Triton (Windows)")
        if self.platform != 'win32':
            print_info("On Linux, Triton is typically installed via pip or PyTorch.")
        if confirm_action("Install triton-windows?", True):
            self._pip_install(["-U", "triton-windows<3.6"])
        input(f"\n{C.BOLD}Press Enter...{C.ENDC}")

    def run(self):
        """Accelerator manager menu."""
        while True:
            clear_screen()
            print_header("Accelerator Manager")
            print(f"{C.DIM}Python: {self.python_exec}{C.ENDC}\n")
            print(f"  {C.CYAN}[1]{C.ENDC} Install Pre-compiled Wheels (Flash Attn, Mamba, etc.)")
            print(f"  {C.CYAN}[2]{C.ENDC} Install Triton")
            print(f"  {C.CYAN}[3]{C.ENDC} Install xFormers")
            print(f"  {C.CYAN}[B]{C.ENDC} Back")
            
            choice = input(f"\n{C.BOLD}Choice: {C.ENDC}").strip().upper()
            
            if choice == '1':
                self.install_wheels()
            elif choice == '2':
                self.install_triton()
            elif choice == '3':
                self.install_xformers()
            elif choice == 'B':
                break

# ============================================================================
# ENVIRONMENT MANAGER
# ============================================================================
class EnvironmentManager:
    """Manages Python virtual environments."""
    
    def __init__(self, root_dir: Path, logger: logging.Logger, config: ConfigManager):
        self.root_dir = root_dir
        self.logger = logger
        self.config = config

    def get_python_info(self, path: str) -> Dict:
        """Get info about a Python executable."""
        if not Path(path).is_file():
            return {'version': 'Not Found', 'path': path, 'pip': 'N/A'}
        try:
            version = subprocess.check_output([path, "--version"], text=True, timeout=5).strip()
            try:
                pip_out = subprocess.check_output([path, "-m", "pip", "--version"], text=True, timeout=5)
                pip_ver = pip_out.split()[1]
            except Exception:
                pip_ver = "N/A"
            return {'version': version, 'path': str(Path(path).resolve()), 'pip': pip_ver}
        except Exception as e:
            return {'version': 'Error', 'path': path, 'pip': 'Error', 'error': str(e)}

    def detect_environments(self) -> List[Dict]:
        """Detect available Python environments."""
        envs = []
        seen = set()
        
        def add(info: Dict, env_type: str, name: str):
            path = str(Path(info['path']).resolve())
            if path not in seen:
                seen.add(path)
                envs.append({'type': env_type, 'name': name, **info})
        
        # System Python
        add(self.get_python_info(sys.executable), 'system', 'System Python')
        
        # Embedded Python
        embedded = self.root_dir / "python_embeded" / "python.exe"
        if embedded.exists():
            add(self.get_python_info(str(embedded)), 'embedded', 'Embedded Python')
        
        # Venvs in root
        for item in self.root_dir.iterdir():
            if item.is_dir() and (item / 'pyvenv.cfg').exists():
                py = item / ("Scripts/python.exe" if platform.system() == "Windows" else "bin/python")
                if py.is_file():
                    add(self.get_python_info(str(py)), 'venv', f'Venv: {item.name}')
        
        # Structured venvs
        sd_root = self.config.get('structure.stable_diffusion_root')
        venvs_name = self.config.get('structure.venvs_dir_name')
        if sd_root and venvs_name:
            venvs_path = Path(sd_root) / venvs_name
            if venvs_path.is_dir():
                for item in venvs_path.iterdir():
                    if item.is_dir() and (item / 'pyvenv.cfg').exists():
                        py = item / ("Scripts/python.exe" if platform.system() == "Windows" else "bin/python")
                        if py.is_file():
                            add(self.get_python_info(str(py)), 'venv', f'Structured: {item.name}')
        
        # Custom paths
        for path in self.config.get('custom_venv_paths', []):
            if Path(path).is_file():
                add(self.get_python_info(path), 'custom', f'Custom: {Path(path).parent.parent.name}')
        
        return envs

    def detect_system_pythons(self) -> List[Dict]:
        """Detect system Python installations."""
        pythons = []
        candidates = ['python', 'python3', 'python3.11', 'python3.12', 'python3.13']
        
        if platform.system() == "Windows":
            # Check common Windows paths
            for ver in ['311', '312', '313']:
                for path in [
                    Path(os.environ.get('LOCALAPPDATA', '')) / f'Programs/Python/Python{ver}/python.exe',
                    Path(f'C:/Python{ver}/python.exe'),
                ]:
                    if path.exists():
                        candidates.append(str(path))
        
        seen = set()
        for cmd in candidates:
            exe = shutil.which(cmd) if not Path(cmd).is_absolute() else cmd
            if exe and Path(exe).exists():
                resolved = str(Path(exe).resolve())
                if resolved not in seen:
                    seen.add(resolved)
                    pythons.append(self.get_python_info(resolved))
        
        return [p for p in pythons if 'Error' not in p.get('version', '')]

    def create_venv(self, base_python: str, target_dir: Path):
        """Create a new virtual environment."""
        name = input(f"\n{C.BOLD}Venv name: {C.ENDC}").strip()
        if not name or not re.match(r'^[a-zA-Z0-9_.-]+$', name):
            print_error("Invalid name")
            return
        
        venv_path = target_dir / name
        if venv_path.exists():
            if not confirm_action(f"'{name}' exists. Overwrite?", False):
                return
            shutil.rmtree(venv_path)
        
        print_info(f"Creating venv at {venv_path}...")
        try:
            subprocess.run([base_python, "-m", "venv", str(venv_path)], check=True, capture_output=True)
            py_exe = venv_path / ("Scripts/python.exe" if platform.system() == "Windows" else "bin/python")
            if py_exe.exists():
                print_info("Upgrading pip...")
                subprocess.run([str(py_exe), "-m", "pip", "install", "--upgrade", "pip"], capture_output=True)
            print_success(f"Venv '{name}' created!")
        except subprocess.CalledProcessError as e:
            print_error(f"Failed: {e.stderr.decode() if e.stderr else e}")
        
        input(f"\n{C.BOLD}Press Enter...{C.ENDC}")

# ============================================================================
# SETUP MANAGER
# ============================================================================
class SetupManager:
    """Wizard for setting up new ComfyUI installations."""
    
    def __init__(self, app, config: ConfigManager, logger: logging.Logger, env_mgr: EnvironmentManager):
        self.app = app
        self.config = config
        self.logger = logger
        self.env_mgr = env_mgr
        self.sd_root = None
        self.comfy_base = None
        self.venvs_dir = None

    def run_wizard(self):
        """Run the setup wizard."""
        clear_screen()
        print_header("ComfyUI Structure Setup Wizard")
        print("This wizard helps set up a folder structure for ComfyUI.")
        print(f"\nExample:\n  {C.CYAN}C:\\Stable Diffusion\\{C.ENDC}")
        print(f"  {C.DIM}├── ComfyUI_Base/\n  └── venvs/{C.ENDC}")
        
        if not confirm_action("\nContinue?", True):
            return
        
        # Step 1: Get paths
        print_subheader("Step 1: Define Paths")
        default = "C:\\Stable Diffusion" if platform.system() == "Windows" else str(Path.home() / "Stable_Diffusion")
        root_str = input(f"{C.BOLD}Root folder [{default}]: {C.ENDC}").strip() or default
        self.sd_root = Path(root_str).resolve()
        
        base_name = input(f"{C.BOLD}ComfyUI folder name [ComfyUI_Base]: {C.ENDC}").strip() or "ComfyUI_Base"
        self.comfy_base = self.sd_root / base_name
        
        venvs_name = input(f"{C.BOLD}Venvs folder name [venvs]: {C.ENDC}").strip() or "venvs"
        self.venvs_dir = self.sd_root / venvs_name
        
        print(f"\nStructure:\n  Root: {C.CYAN}{self.sd_root}{C.ENDC}")
        print(f"  ComfyUI: {C.CYAN}{self.comfy_base}{C.ENDC}")
        print(f"  Venvs: {C.CYAN}{self.venvs_dir}{C.ENDC}")
        
        if not confirm_action("\nCorrect?", True):
            return
        
        # Step 2: Create directories
        print_subheader("Step 2: Creating Folders")
        try:
            for path in [self.sd_root, self.comfy_base, self.venvs_dir]:
                path.mkdir(parents=True, exist_ok=True)
                print_success(f"Created: {path}")
        except Exception as e:
            print_error(f"Failed: {e}")
            input(f"\n{C.BOLD}Press Enter...{C.ENDC}")
            return
        
        # Step 3: Clone ComfyUI
        print_subheader("Step 3: Install ComfyUI")
        if shutil.which('git'):
            if confirm_action("Clone from GitHub?", True):
                try:
                    subprocess.run(
                        ["git", "clone", "https://github.com/comfyanonymous/ComfyUI.git", str(self.comfy_base)],
                        check=True
                    )
                    print_success("ComfyUI cloned!")
                except subprocess.CalledProcessError:
                    print_error("Clone failed")
        else:
            print_warning("Git not found. Please install ComfyUI manually.")
        
        # Save config
        self.config.set('structure.stable_diffusion_root', str(self.sd_root))
        self.config.set('structure.venvs_dir_name', venvs_name)
        self.config.set('active_comfyui_root', str(self.comfy_base))
        self.config.save_config()
        
        print_header("Setup Complete!")
        print_success(f"ComfyUI root: {self.comfy_base}")
        input(f"\n{C.BOLD}Press Enter...{C.ENDC}")

# ============================================================================
# MAIN APPLICATION
# ============================================================================
class ComfyUIManager:
    """Main application class."""
    
    def __init__(self):
        self.comfyui_root: Optional[Path] = None
        self.active_env_path: str = sys.executable
        self.config: Optional[ConfigManager] = None
        self.logger: Optional[logging.Logger] = None
        self.snapshot_mgr: Optional[SnapshotManager] = None
        self.env_mgr: Optional[EnvironmentManager] = None
        self.setup_mgr: Optional[SetupManager] = None
        self.accelerator_mgr: Optional[AcceleratorManager] = None

    def initialize(self):
        """Initialize the application."""
        clear_screen()
        print_header(f"ComfyUI Master Management Suite v{APP_VERSION}")
        
        script_dir = get_script_dir()
        self.logger = setup_logging(script_dir / "logs")
        self.config = ConfigManager(script_dir / "manager_config.json", self.logger)
        
        # Initialize root path
        self._init_root_path()
        self._init_active_python()
        
        if not self.config.get('ui.colors_enabled', True):
            Colors.disable()
        
        self._init_managers()
        self.logger.info(f"Initialized. Root: {self.comfyui_root}, Python: {self.active_env_path}")
        print_success(f"Target: {self.comfyui_root}")
        time.sleep(1)

    def _init_root_path(self):
        """Initialize ComfyUI root path."""
        # Try config
        if config_root := self.config.get('active_comfyui_root'):
            path = Path(config_root)
            if path.is_dir() and (path / "main.py").exists():
                self.comfyui_root = path
                print_info(f"Using saved root: {path}")
                return
        
        # Try auto-detect
        if auto_root := find_comfyui_root():
            self.comfyui_root = Path(auto_root)
            print_info(f"Auto-detected: {self.comfyui_root}")
            self.config.set('active_comfyui_root', str(self.comfyui_root))
            self.config.save_config()
            return
        
        # Manual input
        while not self.comfyui_root:
            print_warning("ComfyUI root not found")
            path_str = input(f"\n{C.BOLD}Enter path (or 'q' to quit): {C.ENDC}").strip().strip('"\'')
            if path_str.lower() == 'q':
                sys.exit(0)
            path = Path(path_str).expanduser().resolve()
            if path.is_dir() and (path / "main.py").is_file():
                self.comfyui_root = path
                self.config.set('active_comfyui_root', str(path))
                self.config.save_config()
            else:
                print_error("Invalid path (must contain main.py)")

    def _init_active_python(self):
        """Initialize active Python path."""
        if saved := self.config.get('last_active_python'):
            if Path(saved).is_file():
                self.active_env_path = saved
                return
        self.active_env_path = sys.executable

    def _init_managers(self):
        """Initialize all manager instances."""
        self.snapshot_mgr = SnapshotManager(self.comfyui_root, self.logger, self.config)
        self.env_mgr = EnvironmentManager(self.comfyui_root, self.logger, self.config)
        self.setup_mgr = SetupManager(self, self.config, self.logger, self.env_mgr)
        self.accelerator_mgr = AcceleratorManager(self.active_env_path, self.config, self.logger)

    def update_active_python(self, new_path: str):
        """Switch active Python environment."""
        self.active_env_path = new_path
        self.config.set('last_active_python', new_path)
        self.config.save_config()
        self.accelerator_mgr = AcceleratorManager(new_path, self.config, self.logger)
        print_success("Active Python switched")

    def show_main_menu(self):
        """Display main menu."""
        clear_screen()
        print_header(f"ComfyUI Master Management Suite v{APP_VERSION}")
        
        py_info = self.env_mgr.get_python_info(self.active_env_path) if self.env_mgr else {}
        py_display = py_info.get('version', Path(self.active_env_path).name)
        
        print(f"{C.DIM}Target: {self.comfyui_root}")
        print(f"Python: {py_display} ({self.active_env_path}){C.ENDC}\n")
        
        print(f"{C.BOLD}Main Menu:{C.ENDC}")
        print(f"  {C.YELLOW}[0]{C.ENDC}{C.BOLD} Setup Wizard (New Installation){C.ENDC}")
        print(f"  {C.CYAN}[1]{C.ENDC} PyTorch Manager")
        print(f"  {C.CYAN}[2]{C.ENDC} Requirements Scanner & Installer")
        print(f"  {C.CYAN}[3]{C.ENDC} Virtual Environment Manager")
        print(f"  {C.CYAN}[4]{C.ENDC} Snapshot Manager")
        print(f"  {C.CYAN}[5]{C.ENDC} Accelerator Manager")
        print(f"  {C.CYAN}[6]{C.ENDC} Settings")
        print(f"  {C.CYAN}[T]{C.ENDC} Set Target ComfyUI Root")
        print(f"  {C.CYAN}[Q]{C.ENDC} Quit")

    def run(self):
        """Main application loop."""
        self.initialize()
        
        while True:
            try:
                self.show_main_menu()
                choice = input(f"\n{C.BOLD}Choice: {C.ENDC}").strip().upper()
                
                if choice == '0':
                    self.setup_mgr.run_wizard()
                elif choice == '1':
                    PyTorchManager(self.active_env_path, self.config, self.logger).interactive_install(self.snapshot_mgr)
                elif choice == '2':
                    RequirementsManager(self.active_env_path, self.comfyui_root, self.config, self.logger).scan_and_install(self.snapshot_mgr)
                elif choice == '3':
                    self._handle_venv_menu()
                elif choice == '4':
                    self._handle_snapshots()
                elif choice == '5':
                    self.accelerator_mgr.run()
                elif choice == '6':
                    self._handle_settings()
                elif choice == 'T':
                    self._handle_set_root()
                elif choice == 'Q':
                    if confirm_action("Exit?", True):
                        print(f"\n{C.CYAN}Goodbye!{C.ENDC}")
                        break
                else:
                    print_error("Invalid choice")
                    time.sleep(1)
            except KeyboardInterrupt:
                print(f"\n\n{C.YELLOW}Interrupted{C.ENDC}")
            except Exception as e:
                self.logger.critical(f"Error: {e}", exc_info=True)
                print_error(f"Error: {e}")
                input(f"\n{C.BOLD}Press Enter...{C.ENDC}")

    def _handle_venv_menu(self):
        """Virtual environment submenu."""
        while True:
            clear_screen()
            print_header("Virtual Environment Manager")
            print(f"  {C.CYAN}[1]{C.ENDC} Switch Active Environment")
            print(f"  {C.CYAN}[2]{C.ENDC} Create New Venv")
            print(f"  {C.CYAN}[3]{C.ENDC} List All Environments")
            print(f"  {C.CYAN}[4]{C.ENDC} Add Custom Path")
            print(f"  {C.CYAN}[B]{C.ENDC} Back")
            
            choice = input(f"\n{C.BOLD}Choice: {C.ENDC}").strip().upper()
            
            if choice == '1':
                self._switch_environment()
            elif choice == '2':
                self._create_venv()
            elif choice == '3':
                self._list_environments()
            elif choice == '4':
                self._add_custom_path()
            elif choice == 'B':
                break

    def _switch_environment(self):
        """Switch active Python environment."""
        envs = self.env_mgr.detect_environments()
        if not envs:
            print_warning("No environments found")
            input(f"\n{C.BOLD}Press Enter...{C.ENDC}")
            return
        
        clear_screen()
        print_header("Switch Environment")
        current = str(Path(self.active_env_path).resolve())
        
        for i, env in enumerate(envs, 1):
            active = f" {C.GREEN}✓{C.ENDC}" if env['path'] == current else ""
            print(f"  [{i}] {C.CYAN}{env['name']}{C.ENDC}{active}")
            print(f"      {C.DIM}{env['version']} | {env['path']}{C.ENDC}\n")
        
        choice = input(f"{C.BOLD}Select (1-{len(envs)}) or [B]ack: {C.ENDC}").strip()
        if choice.upper() == 'B':
            return
        
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(envs):
                self.update_active_python(envs[idx]['path'])
        except ValueError:
            print_error("Invalid choice")
        
        input(f"\n{C.BOLD}Press Enter...{C.ENDC}")

    def _create_venv(self):
        """Create new virtual environment."""
        pythons = self.env_mgr.detect_system_pythons()
        if not pythons:
            print_warning("No system Python found")
            input(f"\n{C.BOLD}Press Enter...{C.ENDC}")
            return
        
        clear_screen()
        print_header("Create Virtual Environment")
        
        for i, py in enumerate(pythons, 1):
            print(f"  [{i}] {C.CYAN}{py['version']}{C.ENDC}")
            print(f"      {C.DIM}{py['path']}{C.ENDC}")
        
        choice = input(f"\n{C.BOLD}Select base Python (1-{len(pythons)}): {C.ENDC}").strip()
        try:
            base_py = pythons[int(choice) - 1]['path']
        except (ValueError, IndexError):
            print_error("Invalid choice")
            input(f"\n{C.BOLD}Press Enter...{C.ENDC}")
            return
        
        # Determine target directory
        sd_root = self.config.get('structure.stable_diffusion_root')
        venvs_name = self.config.get('structure.venvs_dir_name')
        if sd_root and venvs_name:
            target = Path(sd_root) / venvs_name
        else:
            target = self.comfyui_root
        
        target.mkdir(parents=True, exist_ok=True)
        self.env_mgr.create_venv(base_py, target)

    def _list_environments(self):
        """List all detected environments."""
        clear_screen()
        print_header("Detected Environments")
        
        envs = self.env_mgr.detect_environments()
        current = str(Path(self.active_env_path).resolve())
        
        for env in envs:
            active = f" {C.GREEN}✓ ACTIVE{C.ENDC}" if env['path'] == current else ""
            print(f"\n{C.BOLD}{env['name']}{C.ENDC}{active}")
            print(f"  Type: {C.CYAN}{env['type']}{C.ENDC}")
            print(f"  Version: {C.CYAN}{env['version']}{C.ENDC}")
            print(f"  Pip: {C.CYAN}{env['pip']}{C.ENDC}")
            print(f"  Path: {C.DIM}{env['path']}{C.ENDC}")
        
        input(f"\n{C.BOLD}Press Enter...{C.ENDC}")

    def _add_custom_path(self):
        """Add custom Python path."""
        clear_screen()
        print_header("Add Custom Python Path")
        
        path_str = input(f"{C.BOLD}Full path to python executable: {C.ENDC}").strip().strip('"\'')
        if not path_str:
            return
        
        path = Path(path_str).resolve()
        if path.is_file() and 'python' in path.name.lower():
            paths = self.config.get('custom_venv_paths', [])
            if str(path) not in paths:
                paths.append(str(path))
                self.config.set('custom_venv_paths', paths)
                self.config.save_config()
                print_success(f"Added: {path}")
            else:
                print_warning("Path already in list")
        else:
            print_error("Invalid Python executable")
        
        input(f"\n{C.BOLD}Press Enter...{C.ENDC}")

    def _handle_snapshots(self):
        """Snapshot management submenu."""
        while True:
            clear_screen()
            print_header("Snapshot Manager")
            
            enabled = self.config.get('snapshots.enabled', True)
            status = f"{C.GREEN}ENABLED{C.ENDC}" if enabled else f"{C.YELLOW}DISABLED{C.ENDC}"
            print(f"Status: {status}\n")
            
            print(f"  {C.CYAN}[1]{C.ENDC} Create Snapshot")
            print(f"  {C.CYAN}[2]{C.ENDC} List Snapshots")
            print(f"  {C.CYAN}[3]{C.ENDC} Restore Snapshot")
            print(f"  {C.CYAN}[4]{C.ENDC} Delete Snapshot")
            print(f"  {C.CYAN}[B]{C.ENDC} Back")
            
            choice = input(f"\n{C.BOLD}Choice: {C.ENDC}").strip().upper()
            
            if choice == '1':
                if not enabled:
                    print_warning("Snapshots disabled")
                    time.sleep(1.5)
                    continue
                desc = input(f"\n{C.BOLD}Description (optional): {C.ENDC}").strip()
                self.snapshot_mgr.create_snapshot(self.active_env_path, desc)
                input(f"\n{C.BOLD}Press Enter...{C.ENDC}")
            elif choice == '2':
                self._list_snapshots()
            elif choice == '3':
                self._restore_snapshot()
            elif choice == '4':
                self._delete_snapshot()
            elif choice == 'B':
                break

    def _list_snapshots(self):
        """List all snapshots."""
        clear_screen()
        print_subheader("Available Snapshots")
        
        snapshots = self.snapshot_mgr.list_snapshots()
        if not snapshots:
            print_warning("No snapshots found")
        else:
            for snap in snapshots:
                print(f"\n  {C.CYAN}{snap.timestamp}{C.ENDC}")
                print(f"  {snap.description}")
                print(f"  Python: {snap.python_version} | Packages: {len(snap.packages)}")
        
        input(f"\n{C.BOLD}Press Enter...{C.ENDC}")

    def _restore_snapshot(self):
        """Restore from snapshot."""
        snapshots = self.snapshot_mgr.list_snapshots()
        if not snapshots:
            print_warning("No snapshots to restore")
            input(f"\n{C.BOLD}Press Enter...{C.ENDC}")
            return
        
        clear_screen()
        print_subheader("Restore Snapshot")
        
        for i, snap in enumerate(snapshots, 1):
            print(f"  [{i}] {C.CYAN}{snap.timestamp}{C.ENDC} - {snap.description}")
        
        choice = input(f"\n{C.BOLD}Select (1-{len(snapshots)}) or [B]ack: {C.ENDC}").strip()
        if choice.upper() == 'B':
            return
        
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(snapshots):
                self.snapshot_mgr.restore_snapshot(snapshots[idx].timestamp, self.active_env_path)
        except ValueError:
            print_error("Invalid choice")
        
        input(f"\n{C.BOLD}Press Enter...{C.ENDC}")

    def _delete_snapshot(self):
        """Delete a snapshot."""
        snapshots = self.snapshot_mgr.list_snapshots()
        if not snapshots:
            print_warning("No snapshots to delete")
            input(f"\n{C.BOLD}Press Enter...{C.ENDC}")
            return
        
        clear_screen()
        print_subheader("Delete Snapshot")
        
        for i, snap in enumerate(snapshots, 1):
            print(f"  [{i}] {C.CYAN}{snap.timestamp}{C.ENDC} - {snap.description}")
        
        choice = input(f"\n{C.BOLD}Select (1-{len(snapshots)}) or [B]ack: {C.ENDC}").strip()
        if choice.upper() == 'B':
            return
        
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(snapshots):
                ts = snapshots[idx].timestamp
                if confirm_action(f"Delete '{ts}'?", False):
                    (self.snapshot_mgr.snapshot_dir / f"snapshot_{ts}.json").unlink()
                    print_success("Deleted")
        except (ValueError, OSError) as e:
            print_error(f"Error: {e}")
        
        input(f"\n{C.BOLD}Press Enter...{C.ENDC}")

    def _handle_settings(self):
        """Settings submenu."""
        while True:
            clear_screen()
            print_header("Settings")
            
            settings = [
                ('snapshots.enabled', "Snapshots Enabled", bool),
                ('snapshots.auto_create', "Auto-create Snapshots", bool),
                ('snapshots.max_snapshots', "Max Snapshots", int),
                ('ui.show_pip_output', "Show Pip Output", bool),
                ('ui.colors_enabled', "Console Colors", bool),
            ]
            
            print(f"\n{C.BOLD}Current Settings:{C.ENDC}\n")
            for i, (key, name, typ) in enumerate(settings, 1):
                val = self.config.get(key)
                display = "Enabled" if typ is bool and val else "Disabled" if typ is bool else val
                print(f"  [{i}] {name}: {C.CYAN}{display}{C.ENDC}")
            
            print(f"\n  [S] Save")
            print(f"  [B] Back")
            
            choice = input(f"\n{C.BOLD}Choice: {C.ENDC}").strip().upper()
            
            if choice == 'B':
                break
            elif choice == 'S':
                if self.config.save_config():
                    print_success("Settings saved!")
                else:
                    print_error("Failed to save")
                input(f"\n{C.BOLD}Press Enter...{C.ENDC}")
            else:
                try:
                    idx = int(choice) - 1
                    if 0 <= idx < len(settings):
                        key, name, typ = settings[idx]
                        if typ is bool:
                            self.config.set(key, not self.config.get(key))
                            if key == 'ui.colors_enabled':
                                Colors.enable() if self.config.get(key) else Colors.disable()
                        elif typ is int:
                            new_val = input(f"New value for {name}: ").strip()
                            self.config.set(key, int(new_val))
                        print_success(f"{name} updated")
                        time.sleep(0.5)
                except (ValueError, IndexError):
                    print_error("Invalid choice")
                    time.sleep(1)

    def _handle_set_root(self):
        """Set target ComfyUI root."""
        clear_screen()
        print_header("Set Target ComfyUI Root")
        print(f"Current: {C.CYAN}{self.comfyui_root}{C.ENDC}")
        
        path_str = input(f"\n{C.BOLD}New path (blank to cancel): {C.ENDC}").strip().strip('"\'')
        if not path_str:
            return
        
        path = Path(path_str).resolve()
        if path.is_dir() and (path / "main.py").is_file():
            self.comfyui_root = path
            self.config.set('active_comfyui_root', str(path))
            self.config.save_config()
            self._init_managers()
            print_success(f"Updated to: {path}")
        else:
            print_error("Invalid path (must contain main.py)")
        
        input(f"\n{C.BOLD}Press Enter...{C.ENDC}")

# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    app = None
    try:
        app = ComfyUIManager()
        app.run()
    except KeyboardInterrupt:
        print(f"\n\n{C.YELLOW}Interrupted by user{C.ENDC}")
        sys.exit(0)
    except Exception as e:
        print(f"\n{C.RED}Fatal error: {type(e).__name__}: {e}{C.ENDC}")
        if app and app.logger:
            app.logger.critical("Fatal error", exc_info=True)
        else:
            import traceback
            traceback.print_exc()
        input("\nPress Enter to exit...")
        sys.exit(1)
    finally:
        if app and app.logger:
            logging.shutdown()
