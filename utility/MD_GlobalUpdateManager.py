# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# █▓▒░                                                                     ░▒▓█
# █▓▒░         MD_Nodes/Utilities – Global Update Architect v2.0.0         ░▒▓█
# █▓▒░                                                                     ░▒▓█
# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
# ╠═ © 2026 MDMAchine
# ╠═ License: GNU General Public License v3.0 (GPL v3)
# ║
# ║  This program is free software: you can redistribute it and/or modify
# ║  it under the terms of the GNU General Public License as published by
# ║  the Free Software Foundation, either version 3 of the License, or
# ║  (at your option) any later version.
# ║
# ║  This program is distributed in the hope that it will be useful,
# ║  but WITHOUT ANY WARRANTY; without even the implied warranty of
# ║  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# ║  GNU General Public License for more details.
# ║
# ║  You should have received a copy of the GNU General Public License
# ║  along with this program. If not, see <https://www.gnu.org/licenses/>.
# ╠════════════════════════════════════════════════════════════════════════════
# ║ ░▒▓ ORIGIN & DEV:
# ║   • Cast into the void by: MDMAchine
# ║   • Enhanced by: Gemini
# ║
# ║ ░▒▓ DESCRIPTION:
# ║   The Architect is the ultimate command center. It combines the Fixer's
# ║   repair tools, the Sniper's targeting precision, and the Diplomat's
# ║   dependency analysis into one unified node.
# ║   NOTE: As an admin utility, this runs entirely in the public wrapper domain.
# ║
# ║ ░▒▓ FEATURES:
# ║   ✓ Sniper Mode: Update a single node by name ('Target Single Node')
# ║   ✓ Repair Mode: Convert manual installs to Git repos ('Inject Git')
# ║   ✓ Atomic Rollback: Cleans up failed git inits automatically
# ║   ✓ Full Lifecycle: Scan -> Repair -> Update -> Verify
# ║
# ║ ░▒▓ CHANGELOG:
# ║   - v2.0.0 (Enterprise Standards - Feb 2026):
# ║       • ADDED: PerformanceProfiler class (v1.5.3 standard).
# ║       • ADDED: debug_mode parameter for git tracking.
# ║       • REFACTOR: Tooltips strictly updated to 5-part v1.5.4 standard.
# ║   - v1.9.0 (The Architect):
# ║       • MERGED: All features from Sniper and Fixer.
# ║       • ADDED: 'Update Single Node' mode.
# ╚════════════════════════════════════════════════════════════════════════════

# =================================================================================
# == Standard Library Imports
# =================================================================================
VERSION = "v2.0.0"  # UPS v1.5.8


import os
import sys
import subprocess
import time
import shutil
import datetime
import glob
import re
import json
import logging

# =================================================================================
# == Configuration Constants
# =================================================================================

NODE_VERSION = "2.0.0"
CONST_GIT_TIMEOUT = 45
CONST_BACKUP_DIR_NAME = "_Global_Backups"
CONST_STATE_FILE = "_md_batch_state.json"
CONST_REPORT_FILE = "md_global_report.json"
CONST_MAX_RETRIES = 3
CONST_JS_MAX_SAFE_INTEGER = 9007199254740991
CONST_SEED_MIN = 0

CONST_INTERNAL_IGNORE = [
    "MD_Nodes", "ComfyUI-Manager", 
    ".git", "__pycache__", ".disabled", 
    ".vscode", ".idea"
]

CONST_IGNORE_EXT = [".rar", ".zip", ".7z", ".json", ".example", ".txt", ".bat", ".sh"]

CONST_MODES = [
    "📊 Global Status Check",
    "🎯 Update Single Node (Target ID)",
    "🛠️ Inject Git (Convert Manual to Repo)",
    "🔍 Analyze Dependency Conflicts",
    "📦 Create Global Snapshot",
    "🚀 Batch Update (Stateful - Resume/Next Chunk)",
    "⚠️ Update ALL + Install Deps (High Risk)"
]

# Colors
CONST_CLR_GREEN = "\033[92m"
CONST_CLR_RED = "\033[91m"
CONST_CLR_YELLOW = "\033[93m"
CONST_CLR_CYAN = "\033[96m"
CONST_CLR_RESET = "\033[0m"

# =================================================================================
# == Helper Classes (Enterprise Standards)
# =================================================================================

class PerformanceProfiler:
    """Standard performance profiler for MD_Nodes."""
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.timings = {}
        self.start_times = {}
    
    def start(self, operation_name):
        if not self.enabled: return
        self.start_times[operation_name] = time.perf_counter()
    
    def stop(self, operation_name):
        if not self.enabled: return
        if operation_name in self.start_times:
            elapsed = time.perf_counter() - self.start_times[operation_name]
            if operation_name not in self.timings:
                self.timings[operation_name] = []
            self.timings[operation_name].append(elapsed)
            del self.start_times[operation_name]
    
    def get_total_time(self):
        if not self.enabled or not self.timings: return 0.0
        return sum(sum(times) for times in self.timings.values())
    
    def print_report(self):
        if not self.enabled or not self.timings: return
        logging.info("\n⏱️  PERFORMANCE (SYS/IO):")
        total = self.get_total_time()
        logging.info(f"    • Total Time: {total:.2f}s")
        for op_name, times in sorted(self.timings.items()):
            avg = sum(times) / len(times)
            if len(times) == 1:
                logging.info(f"    • {op_name}: {avg:.3f}s")
            else:
                logging.info(f"    • {op_name}: {avg:.3f}s avg ({len(times)}x)")

# =================================================================================
# == Utility Functions
# =================================================================================

def log_console(msg, color=CONST_CLR_RESET):
    logging.info(f"{color}[MD_Architect] {msg}{CONST_CLR_RESET}")

def run_git_command(args, cwd):
    for attempt in range(CONST_MAX_RETRIES):
        try:
            result = subprocess.run(
                ["git"] + args,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=CONST_GIT_TIMEOUT,
                check=False
            )
            if result.returncode == 0:
                return (True, result.stdout.strip(), result.stderr.strip())
            time.sleep(1 * (attempt + 1))
        except Exception as e:
            if attempt == CONST_MAX_RETRIES - 1:
                return (False, "", str(e))
    return (False, "", result.stderr.strip())

def detect_branch(repo_path):
    s, out, _ = run_git_command(["symbolic-ref", "--short", "HEAD"], repo_path)
    if s and out: return out
    s, out, _ = run_git_command(["branch", "--show-current"], repo_path)
    if s and out: return out
    return None

def parse_ignore_list(ignore_string):
    if not ignore_string: return []
    clean = ignore_string.replace("\n", ",").replace(";", ",")
    user_list = [x.strip() for x in clean.split(",") if x.strip()]
    return list(set(user_list + CONST_INTERNAL_IGNORE))

def scan_custom_nodes_folder():
    current_dir = os.path.dirname(os.path.realpath(__file__))
    candidate = current_dir
    for _ in range(4):
        if os.path.basename(candidate) == "custom_nodes":
            return candidate
        parent = os.path.dirname(candidate)
        if parent == candidate: break
        candidate = parent
    return os.path.dirname(os.path.dirname(current_dir))

def backup_node_folder(node_path, backup_root, tag="PRE_UPDATE"):
    node_name = os.path.basename(node_path)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_name = f"{node_name}_{tag}_{timestamp}"
    save_path = os.path.join(backup_root, zip_name)
    try:
        shutil.make_archive(save_path, 'zip', node_path)
        return True, f"{zip_name}.zip"
    except Exception as e:
        return False, str(e)

def perform_single_update(nodes_root, target_id, global_backup_dir, dry_run, do_backup):
    target_path = os.path.join(nodes_root, target_id)
    
    if not os.path.exists(target_path):
        return f"❌ Error: Folder '{target_id}' not found."
    if not os.path.exists(os.path.join(target_path, ".git")):
        return f"❌ Error: '{target_id}' is not a Git repo. Use 'Inject Git' first."

    log_console(f"Targeting: {target_id}", CONST_CLR_YELLOW)

    if dry_run:
        return f"🟦 [Dry Run] Would backup and pull '{target_id}'."

    if do_backup:
        os.makedirs(global_backup_dir, exist_ok=True)
        bs, bmsg = backup_node_folder(target_path, global_backup_dir, tag="SNIPER_UPDATE")
        if not bs: return f"❌ Backup Failed: {bmsg}"
        log_console("Backup Secure.", CONST_CLR_GREEN)

    s, o, e = run_git_command(["pull"], target_path)
    if s:
        log_console("Update Successful.", CONST_CLR_GREEN)
        return f"✅ '{target_id}' Updated Successfully.\n{o}\n⚠️ Restart ComfyUI."
    else:
        return f"❌ Git Pull Failed: {e}"

def repair_manual_install(nodes_root, target_folder, repo_url, global_backup_dir, dry_run):
    target_path = os.path.join(nodes_root, target_folder)
    
    if not os.path.exists(target_path): return f"❌ Folder not found: {target_folder}"
    if os.path.exists(os.path.join(target_path, ".git")): return "⚠️ Already a Git repository."
    if not repo_url.startswith("http"): return "❌ Invalid URL."

    log_console(f"Repairing {target_folder} -> {repo_url}", CONST_CLR_YELLOW)
    if dry_run: return "🟦 [Dry Run] Would inject git."

    os.makedirs(global_backup_dir, exist_ok=True)
    bs, bmsg = backup_node_folder(target_path, global_backup_dir, tag="PRE_REPAIR")
    if not bs: return f"❌ Backup Failed: {bmsg}"

    git_dir = os.path.join(target_path, ".git")
    try:
        s, o, e = run_git_command(["init"], target_path)
        if not s: raise Exception(f"Init: {e}")
        s, o, e = run_git_command(["remote", "add", "origin", repo_url], target_path)
        if not s: raise Exception(f"Remote: {e}")
        
        log_console("Fetching remote...", CONST_CLR_CYAN)
        s, o, e = run_git_command(["fetch", "origin"], target_path)
        if not s: raise Exception(f"Fetch: {e}")

        branch = "main"
        s, _, _ = run_git_command(["rev-parse", "--verify", "origin/main"], target_path)
        if not s:
            branch = "master"
            s, _, _ = run_git_command(["rev-parse", "--verify", "origin/master"], target_path)
            if not s: raise Exception("Remote branch not found.")

        log_console(f"Aligning to origin/{branch}...", CONST_CLR_YELLOW)
        s, o, e = run_git_command(["reset", "--hard", f"origin/{branch}"], target_path)
        if not s: raise Exception(f"Reset: {e}")
        
        run_git_command(["branch", "--set-upstream-to=origin/" + branch, branch], target_path)
        return f"✅ SUCCESS: {target_folder} repaired on '{branch}'."

    except Exception as e:
        if os.path.exists(git_dir):
            time.sleep(0.5)
            try:
                def onerror(func, path, exc_info):
                    import stat
                    if not os.access(path, os.W_OK):
                        os.chmod(path, stat.S_IWUSR)
                        func(path)
                shutil.rmtree(git_dir, onerror=onerror)
                log_console("♻️ Rolled back partial git initialization.", CONST_CLR_RED)
            except Exception: pass
        return f"❌ Repair Failed & Rolled Back: {str(e)}"

def analyze_conflicts(nodes_root, ignored_folders):
    pkg_map = {}
    subdirs = [d for d in os.listdir(nodes_root) if os.path.isdir(os.path.join(nodes_root, d))]
    for folder in subdirs:
        if folder in ignored_folders or folder.startswith("."): continue
        req_path = os.path.join(nodes_root, folder, "requirements.txt")
        if not os.path.exists(req_path): continue
        try:
            with open(req_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"): continue
                    match = re.match(r'^([a-zA-Z0-9_\-\[\]]+)(.*)', line.split(';')[0])
                    if match:
                        pkg = match.group(1).lower()
                        ver = match.group(2).strip() or "any"
                        if pkg not in pkg_map: pkg_map[pkg] = []
                        pkg_map[pkg].append((folder, ver))
        except Exception: pass

    report = ["🔍 DEPENDENCY CONFLICT REPORT", "="*60]
    conflict_count = 0
    for pkg, consumers in pkg_map.items():
        if len(consumers) < 2: continue
        versions = set()
        has_strict = False
        for node, ver in consumers:
            versions.add(ver)
            if "==" in ver: has_strict = True
        if has_strict and len(versions) > 1:
            conflict_count += 1
            report.append(f"\n🔴 CONFLICT: '{pkg}'")
            for node, ver in consumers:
                report.append(f"   • {node:<30} requires {ver}")
    return "\n".join(report) if conflict_count > 0 else "✅ No strict conflicts found."

def create_global_snapshot(nodes_root):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    parent_dir = os.path.dirname(nodes_root)
    zip_path = os.path.join(parent_dir, f"ComfyUI_Global_Snapshot_{timestamp}")
    try:
        shutil.make_archive(zip_path, 'zip', nodes_root)
        return True, f"{zip_path}.zip"
    except Exception as e:
        return False, str(e)

def save_json_report(nodes_root, data):
    path = os.path.join(nodes_root, CONST_REPORT_FILE)
    try:
        with open(path, 'w') as f: json.dump(data, f, indent=4)
        return True
    except Exception: return False

# =================================================================================
# == Core Node Class
# =================================================================================

class MD_GlobalUpdateManager:
    """The Architect. Targeted updates, batch processing, and full repair suite."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (CONST_MODES, {
                    "default": "📊 Global Status Check",
                    "tooltip": (
                        "OPERATION MODE\n"
                        "• Purpose: Select the maintenance action.\n"
                        "• Status Check: View current versions safely.\n"
                        "• Update Single: Target specific node (Requires ID).\n"
                        "• Inject Git: Fix manual installs (Requires URL).\n"
                        "• Batch Update: Process multiple nodes safely.\n"
                        "\n⭐ Recommended: Status Check before any updates."
                    )
                }),
                "batch_size": ("INT", {
                    "default": 10, "min": 1, "max": 200,
                    "tooltip": (
                        "BATCH SIZE\n"
                        "• Purpose: Number of nodes to process per run.\n"
                        "• Why: Prevents ComfyUI UI timeouts during massive global updates.\n"
                        "\n⭐ Recommended: 10."
                    )
                }),
                "dry_run": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "DRY RUN (SIMULATION)\n"
                        "• Purpose: Test actions without making permanent changes.\n"
                        "• Effect: Prints what *would* happen instead of doing it.\n"
                        "\n⭐ Recommended: True when testing new modes."
                    )
                }),
                "do_backup": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "AUTO-BACKUP\n"
                        "• Purpose: Create a zip snapshot before modifying a folder.\n"
                        "• Trade-offs: Slower execution, but 100% safe rollback if update breaks.\n"
                        "\n⭐ Recommended: True."
                    )
                }),
                "ignore_list": ("STRING", {
                    "default": "", "multiline": True,
                    "tooltip": (
                        "IGNORE LIST\n"
                        "• Purpose: Comma-separated list of folder names to skip.\n"
                        "• Note: 'MD_Nodes' and 'ComfyUI-Manager' are ignored by default."
                    )
                }),
                "target_node_ID": ("STRING", {
                    "default": "", 
                    "tooltip": (
                        "TARGET NODE FOLDER (ID)\n"
                        "• Purpose: The exact folder name for 'Single Node' or 'Inject Git' modes.\n"
                        "• Example: 'ComfyUI-VideoHelperSuite'"
                    )
                }),
                "repair_url": ("STRING", {
                    "default": "", 
                    "tooltip": (
                        "REPAIR URL\n"
                        "• Purpose: The GitHub repository URL used for 'Inject Git' mode.\n"
                        "• Example: 'https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite'"
                    )
                }),
            },
            "optional": {
                "trigger": ("INT", {
                    "default": 0,
                    "min": CONST_SEED_MIN,
                    "max": CONST_JS_MAX_SAFE_INTEGER,
                    "tooltip": "EXECUTION TRIGGER\n• Connect a button or changing seed to force execution."
                }),
                "debug_mode": (["0 - Silent", "1 - Info"], {
                    "default": "0 - Silent",
                    "tooltip": "LOGGING VERBOSITY\n• Controls console output and System/IO profiling."
                }),
            }
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("report_text", "updates_pending")
    FUNCTION = "execute_executive_scan"
    CATEGORY = "MD_Nodes/Utility"

    def execute_executive_scan(self, mode, batch_size, dry_run, do_backup, ignore_list, target_node_ID, repair_url, trigger=0, debug_mode="0 - Silent"):
        debug_level = int(debug_mode.split(" ")[0])
        profiler = PerformanceProfiler(enabled=(debug_level >= 1))
        profiler.start("total_execution")
        
        nodes_root = scan_custom_nodes_folder()
        ignored_folders = parse_ignore_list(ignore_list)
        state_file_path = os.path.join(nodes_root, CONST_STATE_FILE)
        global_backup_dir = os.path.join(nodes_root, CONST_BACKUP_DIR_NAME)

        if not nodes_root or not os.path.exists(nodes_root):
            return ("❌ Error: 'custom_nodes' not found.", 0)

        # --- TARGETED MODES ---
        profiler.start("targeted_action")
        if "Inject Git" in mode:
            res = repair_manual_install(nodes_root, target_node_ID.strip(), repair_url.strip(), global_backup_dir, dry_run)
            profiler.stop("targeted_action")
            return (res, 0)

        if "Target Single Node" in mode:
            if not target_node_ID: return ("❌ Missing 'target_node_ID'.", 0)
            res = perform_single_update(nodes_root, target_node_ID.strip(), global_backup_dir, dry_run, do_backup)
            profiler.stop("targeted_action")
            return (res, 0)

        if "Analyze Dependency" in mode:
            res = analyze_conflicts(nodes_root, ignored_folders)
            profiler.stop("targeted_action")
            return (res, 0)
        
        if "Create Global Snapshot" in mode:
            if dry_run: return ("🟦 Dry Run: Global Zip", 0)
            log_console("Creating Global Snapshot...", CONST_CLR_YELLOW)
            _, path = create_global_snapshot(nodes_root)
            profiler.stop("targeted_action")
            return (f"📦 Global Snapshot: {path}", 0)
        profiler.stop("targeted_action")

        # --- BATCH LOGIC ---
        is_batch_mode = "Batch Update" in mode
        pending_nodes = []
        processed_log = []
        loose_files_count = 0
        ignored_files_count = 0
        
        if do_backup and not dry_run and "Update" in mode:
            os.makedirs(global_backup_dir, exist_ok=True)

        if is_batch_mode and os.path.exists(state_file_path):
            try:
                with open(state_file_path, 'r') as f:
                    state_data = json.load(f)
                    pending_nodes = state_data.get("pending", [])
                    processed_log = state_data.get("log", [])
                    log_console(f"Resume: {len(pending_nodes)} left.", CONST_CLR_CYAN)
            except Exception: log_console("State corrupt.", CONST_CLR_RED)

        profiler.start("directory_scan")
        if not pending_nodes and (not is_batch_mode or not os.path.exists(state_file_path)):
            log_console("Scanning...", CONST_CLR_YELLOW)
            try:
                for item in sorted(os.listdir(nodes_root)):
                    path = os.path.join(nodes_root, item)
                    if os.path.isfile(path):
                        if os.path.splitext(item)[1].lower() in CONST_IGNORE_EXT or item == "md_global_report.json":
                            ignored_files_count += 1
                        else: loose_files_count += 1
                        continue
                    if item in ignored_folders or item.startswith("."): continue
                    pending_nodes.append(item)
            except Exception as e: return (f"Scan Error: {e}", 0)
        profiler.stop("directory_scan")

        target_chunk = pending_nodes[:batch_size] if is_batch_mode else pending_nodes
        remaining_nodes = pending_nodes[batch_size:] if is_batch_mode else []
        
        log = [f"🌍 Architect v{NODE_VERSION}", f"📂 Root: {nodes_root}"]
        if dry_run: log.append("🟦 DRY RUN")
        
        updates_found = 0
        current_run_report = []
        
        profiler.start("git_operations")
        for i, folder in enumerate(target_chunk):
            repo_path = os.path.join(nodes_root, folder)
            is_git = os.path.exists(os.path.join(repo_path, ".git"))
            
            if not is_git:
                branch, status, action = "-", "⚠️ ZIP", "Skip"
            else:
                branch = detect_branch(repo_path)
                if not branch:
                    branch, status, action = "HEAD?", "⚠️ DETACHED", "Skip"
                else:
                    s, _, _ = run_git_command(["fetch", "origin"], repo_path)
                    if not s: status, action = "❌ Err", "Retry"
                    else:
                        s2, count, _ = run_git_command(["rev-list", "--count", f"HEAD..origin/{branch}"], repo_path)
                        commits = int(count) if s2 and count.isdigit() else 0
                        if commits == 0: status, action = "✅ Sync", "-"
                        else:
                            status = f"⬇️ {commits}"
                            updates_found += 1
                            if "Update" in mode:
                                if dry_run: action = "🟦 Pull"
                                else:
                                    if do_backup: backup_node_folder(repo_path, global_backup_dir)
                                    ps, _, _ = run_git_command(["pull"], repo_path)
                                    if ps:
                                        action = "🚀 Done"
                                        if "Install Deps" in mode:
                                            req = os.path.join(repo_path, "requirements.txt")
                                            if os.path.exists(req):
                                                try:
                                                    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req], cwd=repo_path)
                                                    action += "+Deps"
                                                except Exception: action += "+Fail"
                                    else: action = "❌ Fail"

            entry = f"{folder:<30} | {branch:<10} | {status:<10} | {action}"
            current_run_report.append(entry)
            log_console(f"[{i+1}/{len(target_chunk)}] {folder}: {status}", CONST_CLR_CYAN)
        profiler.stop("git_operations")

        processed_log.extend(current_run_report)
        
        if is_batch_mode:
            if remaining_nodes:
                state = {"pending": remaining_nodes, "log": processed_log, "timestamp": time.time()}
                with open(state_file_path, 'w') as f: json.dump(state, f)
                log.append(f"\n⏸️ PAUSED. {len(remaining_nodes)} left. Queue again.")
            else:
                if os.path.exists(state_file_path): os.remove(state_file_path)
                log.append("\n✅ BATCH DONE.")
        
        save_json_report(nodes_root, {"version": NODE_VERSION, "log": processed_log})
        
        footer = []
        if loose_files_count: footer.append(f"ℹ️ {loose_files_count} loose files.")
        header = f"{'NODE':<30} | {'BRANCH':<10} | {'STATUS':<10} | {'ACTION'}\n" + ("-"*80)
        
        profiler.stop("total_execution")
        if debug_level >= 1: profiler.print_report()
        
        return ("\n".join(log + [header] + processed_log + footer), updates_found)

NODE_CLASS_MAPPINGS = {"MD_GlobalUpdateManager": MD_GlobalUpdateManager}
NODE_DISPLAY_NAME_MAPPINGS = {"MD_GlobalUpdateManager": "MD: Global Update Architect"}

# =================================================================================
# == Development & Testing
# =================================================================================

if __name__ == "__main__":
    logging.info("🧪 Running Self-Tests for MD_GlobalUpdateManager...")
    
    try:
        assert CONST_JS_MAX_SAFE_INTEGER == 9007199254740991
        logging.info("✅ Constants: PASSED")
        
        test_str = "NodeA, NodeB"
        ignored = parse_ignore_list(test_str)
        assert "NodeA" in ignored
        assert "MD_Nodes" in ignored 
        logging.info("✅ Ignore List Logic: PASSED")
        
        s, o, e = run_git_command(["--version"], ".")
        logging.info(f"✅ Git Wrapper: {'Available' if s else 'Not Found'}")
        
    except Exception as e:
        logging.error(f"❌ Test Failed: {e}")