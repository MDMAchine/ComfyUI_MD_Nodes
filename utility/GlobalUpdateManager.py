# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
# ████ MD_Nodes/Utilities – Global Update Fixer (v1.7.1) ███████████▓▒░
# © 2025 MDMAchine
# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
# ░▒▓ ORIGIN & DEV:
#   • Cast into the void by: MDMAchine
#   • Enhanced by: Gemini
#   • License: Apache 2.0
#
# ░▒▓ DESCRIPTION:
#   The Fixer adds surgical repair capabilities with Atomic Rollback.
#   It converts "Zip Installed" folders into proper Git repositories and
#   cleans up after itself if the conversion fails.
#
# ░▒▓ FEATURES:
#   ✓ Atomic Repair: Deletes corrupt .git folder if conversion fails
#   ✓ Smart Branching: Auto-detects 'main' vs 'master' during repair
#   ✓ Safety Backups: Mandatory snapshots before any conversion logic
#   ✓ Full Lifecycle: Scan -> Repair -> Update -> Verify
#
# ░▒▓ CHANGELOG:
#   - v1.7.1 (Atomic Fixer):
#       • CRITICAL: Added cleanup logic. Failed repairs no longer leave 'zombie' .git folders.
#   - v1.7.0 (The Fixer):
#       • ADDED: 'Inject Git' mode

# =================================================================================
# == Standard Library Imports
# =================================================================================
import os
import sys
import subprocess
import time
import shutil
import datetime
import glob
import re
import json

# =================================================================================
# == Configuration Constants
# =================================================================================

NODE_VERSION = "1.7.1"
CONST_GIT_TIMEOUT = 45
CONST_BACKUP_DIR_NAME = "_Global_Backups"
CONST_STATE_FILE = "_md_batch_state.json"
CONST_REPORT_FILE = "md_global_report.json"
CONST_MAX_RETRIES = 3

CONST_INTERNAL_IGNORE = [
    "MD_Nodes", "ComfyUI-Manager", 
    ".git", "__pycache__", ".disabled", 
    ".vscode", ".idea"
]

CONST_IGNORE_EXT = [".rar", ".zip", ".7z", ".json", ".example", ".txt", ".bat", ".sh"]

CONST_MODES = [
    "📊 Global Status Check",
    "🔍 Analyze Dependency Conflicts",
    "🛠️ Inject Git (Convert Manual to Repo)",
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
# == Utility Functions
# =================================================================================

def log_console(msg, color=CONST_CLR_RESET):
    print(f"{color}[MD_Fixer] {msg}{CONST_CLR_RESET}")

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

def repair_manual_install(nodes_root, target_folder, repo_url, global_backup_dir, dry_run):
    """
    Converts a standard folder into a git repo linked to repo_url.
    Includes Atomic Rollback (deletes .git on failure).
    """
    target_path = os.path.join(nodes_root, target_folder)
    
    # 1. Validation
    if not os.path.exists(target_path):
        return f"❌ Folder not found: {target_folder}"
    if os.path.exists(os.path.join(target_path, ".git")):
        return f"⚠️ Skipped: {target_folder} is already a Git repository."
    if not repo_url.startswith("http"):
        return "❌ Invalid URL. Must start with http/https."

    log_console(f"Repairing {target_folder} -> {repo_url}", CONST_CLR_YELLOW)

    if dry_run:
        return "🟦 [Dry Run] Would init git, add remote, fetch, and checkout."

    # 2. Safety Backup (Mandatory)
    os.makedirs(global_backup_dir, exist_ok=True)
    bs, bmsg = backup_node_folder(target_path, global_backup_dir, tag="PRE_REPAIR")
    if not bs: return f"❌ Backup Failed: {bmsg}"
    log_console(f"Safety Backup Created: {bmsg}", CONST_CLR_GREEN)

    # 3. Git Operations (With Rollback)
    git_dir = os.path.join(target_path, ".git")
    
    try:
        # Init
        s, o, e = run_git_command(["init"], target_path)
        if not s: raise Exception(f"Init Failed: {e}")

        # Remote
        s, o, e = run_git_command(["remote", "add", "origin", repo_url], target_path)
        if not s: raise Exception(f"Remote Add Failed: {e}")

        # Fetch
        log_console("Fetching remote history...", CONST_CLR_CYAN)
        s, o, e = run_git_command(["fetch", "origin"], target_path)
        if not s: raise Exception(f"Fetch Failed (Bad URL?): {e}")

        # Detect Remote Branch
        branch = "main"
        s, _, _ = run_git_command(["rev-parse", "--verify", "origin/main"], target_path)
        if not s:
            branch = "master"
            s, _, _ = run_git_command(["rev-parse", "--verify", "origin/master"], target_path)
            if not s: raise Exception("Could not detect 'main' or 'master'.")

        # Checkout / Reset
        log_console(f"Aligning to origin/{branch}...", CONST_CLR_YELLOW)
        s, o, e = run_git_command(["reset", "--hard", f"origin/{branch}"], target_path)
        if not s: raise Exception(f"Reset Failed: {e}")

        # Set Upstream
        run_git_command(["branch", "--set-upstream-to=origin/" + branch, branch], target_path)

        return f"✅ SUCCESS: {target_folder} is now a Git Repo on branch '{branch}'."

    except Exception as e:
        # ATOMIC ROLLBACK: Delete the .git folder so the user can try again clean
        if os.path.exists(git_dir):
            try:
                # On Windows, git keeps files locked sometimes, wait a split second
                time.sleep(0.5)
                # Need shell=True or specialized handler for readonly files sometimes
                # Simple shutil usually works for .git
                def onerror(func, path, exc_info):
                    # Force write permission if needed
                    import stat
                    if not os.access(path, os.W_OK):
                        os.chmod(path, stat.S_IWUSR)
                        func(path)
                shutil.rmtree(git_dir, onerror=onerror)
                log_console("♻️ Rolled back partial git initialization.", CONST_CLR_RED)
            except Exception as cleanup_err:
                log_console(f"⚠️ Rollback failed: {cleanup_err}", CONST_CLR_RED)
        
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
        except: pass

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
    
    if conflict_count == 0: report.append("\n✅ No obvious strict version conflicts found.")
    else: report.append(f"\n⚠️ Found {conflict_count} potential conflicts.")
    return "\n".join(report)

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
    except: return False

# =================================================================================
# == Core Node Class
# =================================================================================

class MD_GlobalUpdateManager:
    """
    Manages custom nodes. Includes Repair Tool for converting manual installs to Git repos.
    """
    DESCRIPTION = "The Fixer. Convert Zip installs to Git repos, update everything, and resolve conflicts."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (CONST_MODES, {
                    "default": "📊 Global Status Check",
                    "tooltip": "Select action. Use 'Inject Git' to repair a specific manual install."
                }),
                "batch_size": ("INT", {
                    "default": 10, "min": 1, "max": 200,
                    "tooltip": "Nodes to process per run."
                }),
                "dry_run": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "SIMULATION MODE."
                }),
                "do_backup": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Backup nodes before updating? (Mandatory for Repair Mode)"
                }),
                "ignore_list": ("STRING", {
                    "default": "", 
                    "multiline": True,
                    "tooltip": "Folders to skip."
                }),
                "target_node_ID": ("STRING", {
                    "default": "", 
                    "multiline": False,
                    "tooltip": "FOLDER NAME of the node to repair (e.g. 'LanPaint'). Only used in 'Inject Git' mode."
                }),
                "repair_url": ("STRING", {
                    "default": "", 
                    "multiline": False,
                    "tooltip": "GitHub URL for the repair target (e.g. 'https://github.com/User/Repo')."
                }),
            },
            "optional": {
                "trigger": ("INT", {
                    "default": 0, "min": 0, "max": 9007199254740991
                }),
            }
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("report_text", "updates_pending")
    FUNCTION = "execute_executive_scan"
    CATEGORY = "MD_Nodes/Utilities"

    def execute_executive_scan(self, mode, batch_size, dry_run, do_backup, ignore_list, target_node_ID, repair_url, trigger=0):
        start_time = time.time()
        nodes_root = scan_custom_nodes_folder()
        ignored_folders = parse_ignore_list(ignore_list)
        state_file_path = os.path.join(nodes_root, CONST_STATE_FILE)
        global_backup_dir = os.path.join(nodes_root, CONST_BACKUP_DIR_NAME)

        if not nodes_root or not os.path.exists(nodes_root):
            return ("❌ Error: 'custom_nodes' not found.", 0)

        # 1. SPECIAL MODES
        
        # --- REPAIR MODE ---
        if "Inject Git" in mode:
            if not target_node_ID or not repair_url:
                return ("❌ Error: You must provide 'target_node_ID' and 'repair_url' for Repair Mode.", 0)
            
            result_msg = repair_manual_install(nodes_root, target_node_ID.strip(), repair_url.strip(), global_backup_dir, dry_run)
            return (result_msg, 0)
        # -------------------

        if "Analyze Dependency" in mode:
            return (analyze_conflicts(nodes_root, ignored_folders), 0)
        
        if "Create Global Snapshot" in mode:
            if dry_run: return ("🟦 Dry Run: Global Zip", 0)
            log_console("Creating Global Snapshot...", CONST_CLR_YELLOW)
            _, path = create_global_snapshot(nodes_root)
            return (f"📦 Global Snapshot: {path}", 0)

        # 2. STATEFUL BATCH LOGIC
        is_batch_mode = "Batch Update" in mode
        pending_nodes = []
        processed_log = []
        loose_files_count = 0
        ignored_files_count = 0
        
        # Setup Backup Dir
        if do_backup and not dry_run and "Update" in mode:
            os.makedirs(global_backup_dir, exist_ok=True)

        # A. Load State
        if is_batch_mode and os.path.exists(state_file_path):
            try:
                with open(state_file_path, 'r') as f:
                    state_data = json.load(f)
                    pending_nodes = state_data.get("pending", [])
                    processed_log = state_data.get("log", [])
                    log_console(f"Resume: {len(pending_nodes)} nodes remaining.", CONST_CLR_CYAN)
            except:
                log_console("State file corrupt, restarting scan.", CONST_CLR_RED)

        # B. Discovery
        if not pending_nodes and (not is_batch_mode or not os.path.exists(state_file_path)):
            log_console("Scanning directory...", CONST_CLR_YELLOW)
            try:
                all_items = sorted(os.listdir(nodes_root))
                for item in all_items:
                    item_path = os.path.join(nodes_root, item)
                    if os.path.isfile(item_path):
                        ext = os.path.splitext(item)[1].lower()
                        if ext in CONST_IGNORE_EXT or item == "md_global_report.json":
                            ignored_files_count += 1
                        else: loose_files_count += 1
                        continue
                    if item in ignored_folders or item.startswith("."): continue
                    pending_nodes.append(item)
            except Exception as e: return (f"Scan Error: {e}", 0)

        # 3. PROCESSING
        target_chunk = pending_nodes[:batch_size] if is_batch_mode else pending_nodes
        remaining_nodes = pending_nodes[batch_size:] if is_batch_mode else []
        
        log = [f"🌍 Fixer v{NODE_VERSION}", f"📂 Scanning: {nodes_root}"]
        if dry_run: log.append("🟦 DRY RUN ACTIVE")
        
        updates_found = 0
        current_run_report = []
        
        for i, folder in enumerate(target_chunk):
            repo_path = os.path.join(nodes_root, folder)
            is_git_repo = os.path.exists(os.path.join(repo_path, ".git"))
            
            if not is_git_repo:
                branch = "-"
                status = "⚠️ ZIP/MANUAL"
                action = "Use 'Inject Git' Mode"
            else:
                branch = detect_branch(repo_path)
                if not branch:
                    branch = "HEAD?"
                    status = "⚠️ DETACHED"
                    action = "Skip"
                else:
                    s, _, _ = run_git_command(["fetch", "origin"], repo_path)
                    if not s:
                        status = "❌ Net Err"
                        action = "Retry?"
                    else:
                        s2, count, _ = run_git_command(["rev-list", "--count", f"HEAD..origin/{branch}"], repo_path)
                        commits = int(count) if s2 and count.isdigit() else 0
                        
                        if commits == 0:
                            status = "✅ Synced"
                            action = "-"
                        else:
                            status = f"⬇️ Behind {commits}"
                            updates_found += 1
                            if "Update" in mode:
                                if dry_run:
                                    action = "🟦 Would Pull"
                                else:
                                    if do_backup: backup_node_folder(repo_path, global_backup_dir)
                                    ps, _, _ = run_git_command(["pull"], repo_path)
                                    if ps:
                                        action = "🚀 Updated"
                                        if "Install Deps" in mode:
                                            req = os.path.join(repo_path, "requirements.txt")
                                            if os.path.exists(req):
                                                try:
                                                    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req], cwd=repo_path)
                                                    action += "+Deps"
                                                except: action += "+DepsFail"
                                    else: action = "❌ Fail"

            entry = f"{folder:<30} | {branch:<10} | {status:<15} | {action}"
            current_run_report.append(entry)
            log_console(f"[{i+1}/{len(target_chunk)}] {folder}: {status}", CONST_CLR_CYAN)

        # 4. STATE SAVE
        processed_log.extend(current_run_report)
        
        if is_batch_mode:
            if remaining_nodes:
                state = {"pending": remaining_nodes, "log": processed_log, "timestamp": time.time()}
                with open(state_file_path, 'w') as f: json.dump(state, f)
                log.append(f"\n⏸️ BATCH PAUSED. {len(remaining_nodes)} nodes remaining.")
                log.append("👉 Queue this node again to continue.")
            else:
                if os.path.exists(state_file_path): os.remove(state_file_path)
                log.append("\n✅ BATCH COMPLETE.")
        
        full_data = {"version": NODE_VERSION, "date": str(datetime.datetime.now()), "log": processed_log}
        save_json_report(nodes_root, full_data)

        # Footer Stats
        footer = []
        if loose_files_count > 0: footer.append(f"ℹ️ Found {loose_files_count} unknown loose files.")
        if ignored_files_count > 0: footer.append(f"ℹ️ Hiding {ignored_files_count} archives/system files.")
        
        header = f"{'NODE':<30} | {'BRANCH':<10} | {'STATUS':<15} | {'ACTION'}\n" + ("-"*80)
        full_text = "\n".join(log + [header] + processed_log + footer)
        return (full_text, updates_found)

# =================================================================================
# == Node Registration
# =================================================================================

NODE_CLASS_MAPPINGS = {
    "MD_GlobalUpdateManager": MD_GlobalUpdateManager
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MD_GlobalUpdateManager": "MD: Global Update Fixer (v1.7.1)"
}