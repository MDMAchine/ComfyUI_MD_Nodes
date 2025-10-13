# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# ████ ComfyUI Project Updater v4.2.1 - Final Bugfix          ████▓▒░
# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
# ░▒▓ ORIGIN & DEV:
#   • Crafted with care by MDMAchine / MD_Nodes
#   • License: Apache 2.0 — do what you want, just give credit
# ░▒▓ DESCRIPTION:
#   A professional-grade CLI tool for ComfyUI project management. This
#   version includes a critical bugfix for the dependency scanner.
# ░▒▓ NEW IN v4.2.1:
#   ✓ BUGFIX: Corrected an `AttributeError` in the dependency scanner
#     caused by an overwritten variable during tuple unpacking.
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
from typing import Any, Dict, List, Optional, Set

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
    "exclude_dirs": [".git", "__pycache__", "venv", ".venv", "env", ".env", "node_modules", ".ipynb_checkpoints", ".project_backups", "misc", "Internal", "docs", "tests", "tools"],
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
    with open(config_path, 'r', encoding='utf-8') as f: config = json.load(f)
    for key, value in DEFAULT_PROJECT_CONFIG.items():
        if key not in config: config[key] = value
    return config

def validate_config(config: Dict[str, Any]) -> List[str]:
    issues = []
    for key, expected_type in [("denylist_packages", list), ("exclude_dirs", list), ("exclude_files", list), ("exclude_extensions", list)]:
        if key in config and not isinstance(config.get(key), expected_type):
            issues.append(f"Config Error: '{key}' must be a list.")
    if not issues:
        print("  ✓ Configuration validated successfully")
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
    parser.add_argument("--project-root", type=str); parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--git-commit", action="store_true"); parser.add_argument("--no-backup", action="store_true")
    parser.add_argument("--interactive", action="store_true"); parser.add_argument("--debug", action="store_true")
    parser.add_argument("--set-version", type=str); parser.add_argument("--bump", choices=['major', 'minor', 'patch'])
    parser.add_argument("--pin-versions", action="store_true", help="Preserve existing version numbers in requirements.txt")
    return parser.parse_args()

def print_header():
    print("\n" + "█" * 70)
    print("█" + "         ComfyUI Project Updater v4.2.1 - BY MDMACHINE         ".center(68) + "█")
    print("█" + "       Automate Your Node Management & Release Workflow        ".center(68) + "█")
    print("█" * 70 + "\n")

def print_section_header(title: str, icon: str = "▓"):
    print("\n" + "▓" * 70); print(f"▓▓▓ {icon} {title}"); print("▓" * 70)

def print_option(number: int, title: str, description: str, selected: bool = False):
    marker = "▶" if selected else " "; status = "✓" if selected else " "
    print(f"  {marker} [{number}] {status} {title}\n      └─ {description}")

def prompt_for_project_root(updater_settings: dict) -> Path:
    print_section_header("SELECT PROJECT DIRECTORY", "📂")
    last_path_str = updater_settings.get("last_project_path")
    default_path = Path(last_path_str) if last_path_str and Path(last_path_str).is_dir() else Path.cwd()
    user_input = input(f"\n  ▶ Enter path or press Enter to use default:\n    [{default_path}] > ").strip()
    if user_input.startswith('"') and user_input.endswith('"'): user_input = user_input[1:-1]
    if not user_input: return default_path
    chosen_path = Path(user_input).expanduser().resolve()
    if chosen_path.is_dir(): return chosen_path
    print(f"  ❌ Error: Path '{chosen_path}' not found. Reverting to default."); return default_path

def interactive_menu(config: dict) -> Dict[str, Any]:
    options = {'dry_run': False, 'no_backup': False, 'git_commit': False, 'debug': False, 'pin_versions': False}
    while True:
        print_section_header("MAIN MENU", "⚡")
        print_option(1, "Preview Mode (Dry Run)", "See changes without modifying files", options['dry_run'])
        print_option(2, "Backup Files", "Create timestamped backups", not options['no_backup'])
        print_option(3, "Auto-Commit to Git", "Automatically commit changes", options['git_commit'])
        print_option(4, "Preserve Versions", "Keep existing versions in requirements.txt", options['pin_versions'])
        print_option(5, "Debug Mode", "Show detailed parsing info", options['debug'])
        choice = input("\n  ▶ Enter command (go, quit): ").strip().lower()
        if choice in ['go', 'run']: return options
        if choice in ['quit', 'exit', 'q']: sys.exit(0)
        if choice == '1': options['dry_run'] = not options['dry_run']
        if choice == '2': options['no_backup'] = not options['no_backup']
        if choice == '3': options['git_commit'] = not options['git_commit']
        if choice == '4': options['pin_versions'] = not options['pin_versions']
        if choice == '5': options['debug'] = not options['debug']

def update_file(path: Path, content: str, message: str, dry_run: bool):
    if dry_run: print(f"🔍 [DRY RUN] {message}"); return
    path.write_text(content, encoding="utf-8"); print(f"✅ {path.name} updated.")

def create_backup(project_root: Path, dry_run: bool):
    backup_dir = project_root / ".project_backups"
    if dry_run: print(f"🔍 [DRY RUN] Would create backup in: {backup_dir.relative_to(project_root.parent)}"); return
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / timestamp
    files_to_backup = ["__init__.py", "requirements.txt", "comfyui.json", "pyproject.toml", ".gitignore"]
    backed_up = [f for f in files_to_backup if (project_root / f).exists()]
    if backed_up:
        backup_path.mkdir(parents=True, exist_ok=True)
        for filename in backed_up: shutil.copy2(project_root / filename, backup_path / filename)
        print(f"💾 Backed up {len(backed_up)} files to: {backup_path.relative_to(project_root)}")

def update_init_py(project_root: Path, parsed_nodes: List[Dict[str, Any]], dry_run: bool):
    if not parsed_nodes: return
    files_with_nodes = sorted({project_root / node["file_path"] for node in parsed_nodes})
    init_content = ["# This file is auto-generated by update_project.py\n", "NODE_CLASS_MAPPINGS = {}", "NODE_DISPLAY_NAME_MAPPINGS = {}\n", "# Import and register all nodes"]
    for file_path in files_with_nodes:
        module_name = ".".join(file_path.relative_to(project_root).with_suffix("").parts)
        alias = module_name.replace('.', '_').upper()
        init_content.extend([f"try:", f"    from .{module_name} import NODE_CLASS_MAPPINGS as C_{alias}, NODE_DISPLAY_NAME_MAPPINGS as D_{alias}", f"    NODE_CLASS_MAPPINGS.update(C_{alias})", f"    NODE_DISPLAY_NAME_MAPPINGS.update(D_{alias})", f"except ImportError as e:", f"    print(f'Could not import nodes from {module_name}: {{e}}')\n"])
    init_content.append("__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']")
    update_file(project_root / "__init__.py", "\n".join(init_content), f"Would update __init__.py with imports from {len(files_with_nodes)} files.", dry_run)

def preserve_versions(old_reqs_path: Path, new_deps: List[str]) -> List[str]:
    if not old_reqs_path.exists(): return new_deps
    existing_versions = {}
    for line in old_reqs_path.read_text(encoding="utf-8").splitlines():
        if '==' in line:
            pkg, ver = line.split('==', 1); existing_versions[pkg.strip()] = ver.strip()
    return [f"{dep}=={existing_versions[dep]}" if dep in existing_versions else dep for dep in new_deps]

def update_requirements_txt(project_root: Path, dependencies: List[str], dry_run: bool, pin_versions: bool):
    if not dependencies: return
    req_path = project_root / "requirements.txt"
    if pin_versions: print("  📌 Preserving existing version pins...")
    final_deps = preserve_versions(req_path, dependencies) if pin_versions else dependencies
    content = "# Auto-generated by update_project.py\n" + "\n".join(final_deps)
    update_file(req_path, content, f"Would write {len(final_deps)} dependencies to requirements.txt.", dry_run)

def update_comfyui_json(project_root: Path, parsed_nodes: List[Dict[str, Any]], dependencies: List[str], config: Dict[str, Any], dry_run: bool):
    json_path = project_root / "comfyui.json"
    data = json.loads(json_path.read_text(encoding="utf-8")) if json_path.exists() else {"name": project_root.name}
    exclude_dirs, exclude_ext = set(config.get('exclude_dirs', [])), set(config.get('exclude_extensions', []))
    all_files = sorted([str(p.relative_to(project_root)).replace("\\", "/") for p in project_root.rglob("*") if p.is_file() and not any(part in exclude_dirs for part in p.parts) and p.suffix not in exclude_ext])
    data.update({"version": config['project_version'], "dependencies": dependencies, "nodes": parsed_nodes, "files": all_files})
    content = json.dumps(data, indent=2, ensure_ascii=False)
    update_file(json_path, content, f"Would update comfyui.json with {len(parsed_nodes)} nodes and {len(all_files)} files.", dry_run)

def update_pyproject_toml(project_root: Path, dependencies: List[str], new_version: str, dry_run: bool):
    toml_path = project_root / "pyproject.toml"
    if not toml_path.exists(): return
    if HAS_TOML:
        with open(toml_path, 'rb') as f: data = tomli.load(f)
        if 'project' not in data: data['project'] = {}
        data['project']['version'], data['project']['dependencies'] = new_version, dependencies
        content_bytes = tomli_w.dumps(data).encode('utf-8')
        if dry_run: print("🔍 [DRY RUN] Would update pyproject.toml.")
        else:
            with open(toml_path, 'wb') as f: f.write(content_bytes)
            print("✅ pyproject.toml updated.")
    else: # Fallback
        lines = toml_path.read_text(encoding="utf-8").splitlines()
        new_lines = [f'version = "{new_version}"' if line.strip().startswith("version =") else line for line in lines]
        content = "\n".join(new_lines)
        update_file(toml_path, content, "Would update pyproject.toml (fallback mode).", dry_run)

def update_gitignore(project_root: Path, config: Dict[str, Any], dry_run: bool):
    path = project_root / ".gitignore"
    existing_content = path.read_text(encoding="utf-8") if path.exists() else ""
    existing_patterns = {line.strip() for line in existing_content.splitlines() if line.strip() and not line.strip().startswith('#')}
    default_patterns = {e for e in config['gitignore_entries'] if e and not e.startswith('#')}
    new_patterns = default_patterns - existing_patterns
    if not new_patterns: return
    new_block = "\n# Added by updater\n" + "\n".join(sorted(list(new_patterns)))
    content = existing_content.rstrip() + new_block
    update_file(path, content, f"Would add {len(new_patterns)} entries to .gitignore.", dry_run)

def git_commit_changes(project_root: Path, new_version: str, dry_run: bool):
    if dry_run: print("🔍 [DRY RUN] Would commit changes to git."); return
    try:
        if not (project_root / ".git").is_dir(): return
        files_to_add = ["__init__.py", "requirements.txt", "comfyui.json", "pyproject.toml", ".gitignore", PROJECT_CONFIG_FILENAME]
        subprocess.run(["git", "add", *[f for f in files_to_add if (project_root / f).exists()]], cwd=project_root, check=True, capture_output=True)
        commit_message = f"chore: auto-update project to v{new_version}"
        subprocess.run(["git", "commit", "-m", commit_message], cwd=project_root, check=True, capture_output=True)
        print(f"✅ Changes committed to git: {commit_message}")
    except (subprocess.CalledProcessError, FileNotFoundError) as e: print(f"⚠️ Git commit failed: {e}")

# =================================================================================
# --- CORE LOGIC ---
# =================================================================================

def get_file_hash(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()

class NodeParser(ast.NodeVisitor):
    def __init__(self, debug: bool = False):
        self.mappings, self.display_names, self.debug = {}, {}, debug
    def _eval_dict_value(self, value_node: ast.expr) -> Optional[str]:
        if isinstance(value_node, ast.Constant): return value_node.value
        if isinstance(value_node, ast.Name): return value_node.id
        return None
    def visit_Assign(self, node: ast.Assign):
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target_name = node.targets[0].id
            if target_name in ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"] and isinstance(node.value, ast.Dict):
                new_dict = {key_node.value: self._eval_dict_value(value_node) for key_node, value_node in zip(node.value.keys, node.value.values) if isinstance(key_node, ast.Constant)}
                if target_name == "NODE_CLASS_MAPPINGS": self.mappings = new_dict
                else: self.display_names = new_dict
        self.generic_visit(node)

def discover_python_files(project_root: Path, config: Dict[str, Any], debug: bool) -> List[Path]:
    print("🔍 Scanning for Python files...")
    py_files, exclude_dirs, exclude_files = [], set(config.get('exclude_dirs', [])), set(config.get('exclude_files', []))
    for path in sorted(project_root.rglob("*.py")):
        relative_path = path.relative_to(project_root)
        if any(part in exclude_dirs for part in relative_path.parts) or path.name in exclude_files:
            if debug: print(f"  - Skipping '{relative_path}'")
            continue
        py_files.append(path);
        if debug: print(f"  + Including '{relative_path}'")
    print(f"✅ Found {len(py_files)} Python files to analyze.")
    return py_files

def find_project_dependencies(project_root: Path, all_py_files: List[Path], config: Dict[str, Any]) -> List[str]:
    print("📦 Scanning for dependencies...")
    found, std_lib = set(), {name for module_finder, name, ispkg in pkgutil.iter_modules() if 'site-packages' not in (module_finder.path or '')}
    local = {p.stem for p in all_py_files} | {p.name for p in project_root.iterdir() if p.is_dir() and (p / "__init__.py").exists()}
    denylist = set(config.get('denylist_packages', []))
    for file in all_py_files:
        try:
            tree = ast.parse(file.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names: found.add(alias.name.split('.')[0])
                elif isinstance(node, ast.ImportFrom) and node.module: found.add(node.module.split('.')[0])
        except Exception: pass
    external = {imp for imp in found if imp not in std_lib and imp not in local}
    mapped = {config['import_to_package_map'].get(imp, imp) for imp in external}
    final = {dep for dep in mapped if dep not in denylist}; final.update(config.get('manual_imports', []))
    final = {dep for dep in final if dep not in denylist}
    print(f"✅ Found {len(final)} external dependencies.")
    return sorted(list(final))

def parse_node_files(project_root: Path, file_paths: List[Path], debug: bool, cache: Dict[str, Any]) -> (List[Dict[str, Any]], List[Dict[str, str]], Dict[str, Any]):
    all_nodes, errors, new_cache, cached_count = [], [], {}, 0
    print("📋 Parsing node metadata...")
    
    iterable = tqdm(file_paths, desc="Parsing nodes") if HAS_TQDM and len(file_paths) > 10 else file_paths
    for path in iterable:
        rel_path = str(path.relative_to(project_root))
        try:
            current_hash = get_file_hash(path)
            if rel_path in cache and cache[rel_path]['hash'] == current_hash:
                if debug: print(f"  - Skipping '{rel_path}' (cached)")
                all_nodes.extend(cache[rel_path]['nodes']); new_cache[rel_path] = cache[rel_path]; cached_count += 1
                continue
            if debug: print(f"  - Analyzing '{rel_path}'")
            tree = ast.parse(path.read_text(encoding="utf-8")); parser = NodeParser(debug); parser.visit(tree)
            file_nodes = []
            if parser.mappings:
                for class_def in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
                    for key, mapped_name in parser.mappings.items():
                        if class_def.name == mapped_name:
                            info = {"type": class_def.name, "title": parser.display_names.get(key, key), "description": ast.get_docstring(class_def) or "", "category": "Uncategorized", "file_path": rel_path.replace("\\", "/")}
                            for item in class_def.body:
                                if isinstance(item, ast.Assign) and len(item.targets) == 1 and isinstance(item.targets[0], ast.Name) and item.targets[0].id == "CATEGORY" and isinstance(item.value, ast.Constant):
                                    info["category"] = item.value.value
                            file_nodes.append(info)
            new_cache[rel_path] = {'hash': current_hash, 'nodes': file_nodes}; all_nodes.extend(file_nodes)
        except Exception as e: errors.append({"file": rel_path, "error": str(e)})

    found_nodes_count = len(all_nodes)
    if debug and cached_count > 0 and len(file_paths) > 0:
        cache_hit_rate = (cached_count / len(file_paths)) * 100
        print(f"  📊 Cache hit rate: {cache_hit_rate:.1f}%")

    print(f"✅ Found a total of {found_nodes_count} nodes ({cached_count}/{len(file_paths)} files from cache).")
    return sorted(all_nodes, key=lambda x: x['title']), errors, new_cache

def validate_node_metadata(parsed_nodes: List[Dict[str, Any]]):
    warnings = [f"  - Node '{n['title']}' is missing a description." for n in parsed_nodes if not n.get('description')]
    if warnings: print_section_header("METADATA WARNINGS", "⚠️"); [print(w) for w in warnings]

# =================================================================================
# --- MAIN EXECUTION ---
# =================================================================================

def main():
    args = parse_arguments()
    print_header()
    is_interactive_run = not any(arg for arg in sys.argv[1:] if not arg.startswith('--interactive')) or args.interactive
    updater_settings = load_updater_settings()
    project_root = Path(args.project_root).resolve() if args.project_root else prompt_for_project_root(updater_settings) if is_interactive_run else Path.cwd().resolve()
    print(f"📁 Operating on project root: {project_root}")
    if not project_root.is_dir(): print(f"❌ Error: Project root '{project_root}' not found."); sys.exit(1)
    
    project_config = load_or_create_project_config(project_root)
    config_errors = validate_config(project_config)
    if config_errors: print_section_header("CONFIG ERRORS", "⚠️"); [print(f"  - {e}") for e in config_errors]; sys.exit(1)

    if is_interactive_run and not any([args.project_root, args.bump, args.set_version]):
        menu_options = interactive_menu(project_config)
        for key, value in menu_options.items(): setattr(args, key, value)

    if args.dry_run: print("\n   -------------------------------------------------\n   🔍 DRY RUN MODE IS ACTIVE. No files will be changed.\n   -------------------------------------------------")

    new_version = handle_versioning(args, project_config.get('project_version', '1.0.0'))
    project_config['project_version'] = new_version
    
    print_section_header("RUNNING UPDATES", "⚙️")
    if not args.no_backup: create_backup(project_root, args.dry_run)
    all_py_files = discover_python_files(project_root, project_config, args.debug)
    
    parsed_nodes, parsing_errors, new_cache = parse_node_files(project_root, all_py_files, args.debug, updater_settings.get('file_cache', {}))
    updater_settings['last_project_path'] = str(project_root)
    updater_settings['file_cache'] = new_cache

    dependencies = find_project_dependencies(project_root, all_py_files, project_config)
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
        for err in parsing_errors: print(f"  ❌ {err['file']}: {err['error']}")

    if not args.dry_run:
        with open(project_root / PROJECT_CONFIG_FILENAME, 'w', encoding='utf-8') as f: json.dump(project_config, f, indent=4)
        save_updater_settings(updater_settings)
        print("\n✅ Config files updated.")

    print_section_header("COMPLETE", "🎉")
    if args.dry_run: print("🔍 Dry Run Finished. No changes were made to any files.")
    else: print(f"✅ Project update to version {new_version} complete!")

if __name__ == "__main__":
    main()