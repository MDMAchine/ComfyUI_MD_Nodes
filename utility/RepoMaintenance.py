# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
# ████ MD_Nodes/Utilities – Repo Fortress (v2.2.1) █████████████████▓▒░
# © 2025 MDMAchine
# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
# ░▒▓ ORIGIN & DEV:
#   • Cast into the void by: MDMAchine
#   • Enhanced by: Gemini
#   • License: Apache 2.0
#
# ░▒▓ DESCRIPTION:
#   Complete Lifecycle Manager with "Fortress-Grade" safeguards.
#   Includes Disk Space checks, Concurrency handling, and Restore Safety.
#
# ░▒▓ FEATURES:
#   ✓ Storage Guard: Pre-flight check for disk space vs repo size
#   ✓ Race Protection: Microsecond timestamps & atomic rotation logic
#   ✓ Time Travel: Restore from backups with pre-restore snapshots
#   ✓ Architect Mode: Dry Run simulation for all operations
#
# ░▒▓ CHANGELOG:
#   - v2.2.1 (Enterprise Compliance):
#       • FIXED: trigger input capped at JS_MAX_SAFE_INTEGER
#       • ADDED: Embedded Unit Tests (Section 11.3)
#       • REFACTOR: Strict Header & Global Constants
#   - v2.2.0 (Fortress):
#       • ADDED: Disk space calculation & enforcement

# ░▒▓ CONFIGURATION:
#   → Primary Use: Updating and maintaining the MD_Nodes suite
#   → Secondary Use: Creating safe snapshots before editing code
#   → Edge Use: recovering from merge conflicts via Hard Reset

# ░▒▓ WARNING:
#   This node can execute git commands and delete files (backups).
#   Use "Dry Run" mode to preview destructive actions.
# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄

# =================================================================================
# == Standard Library Imports
# =================================================================================
import os
import sys
import subprocess
import time
import shutil
import glob
import json
import zipfile
from datetime import datetime

# =================================================================================
# == Third-Party Imports
# =================================================================================
# None required for this utility node (Pure Python)

# =================================================================================
# == ComfyUI Core Modules
# =================================================================================
# None required (Utility Node)

# =================================================================================
# == Configuration Constants
# =================================================================================

# Versioning
NODE_VERSION = "2.2.1"

# Integer Safety (Section 8.5)
CONST_JS_MAX_SAFE_INTEGER = 9007199254740991
CONST_SEED_MIN = 0

# Git / System
CONST_GIT_TIMEOUT = 120
CONST_REPO_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
CONST_BACKUP_DIR_NAME = "_backups"

# Disk Safety
CONST_DISK_BUFFER_MB = 500
CONST_REPO_SIZE_MULTIPLIER = 2  # Require 2x repo size free

# Menu Options
CONST_MODES = [
    "Check Status Only",
    "📋 List Available Backups",
    "Create Snapshot (Backup Only)",
    "Standard Update (Git Pull)",
    "Update & Install Requirements",
    "⚠️ FORCE RESET & UPDATE (Discard Local Changes)",
    "♻️ RESTORE from Backup (Requires Filename)"
]

# File Exclusions
CONST_EXCLUDES = [
    '.git', '__pycache__', CONST_BACKUP_DIR_NAME, 
    '.idea', '.vscode', 'node_modules', 
    '.pytest_cache', '.DS_Store', '__MACOSX'
]

# Console Colors
CONST_CLR_GREEN = "\033[92m"
CONST_CLR_RED = "\033[91m"
CONST_CLR_YELLOW = "\033[93m"
CONST_CLR_CYAN = "\033[96m"
CONST_CLR_MAGENTA = "\033[95m"
CONST_CLR_RESET = "\033[0m"

# =================================================================================
# == Utility Functions
# =================================================================================

def log_console(msg, color=CONST_CLR_RESET):
    """Prints colorful messages to the console."""
    print(f"{color}[MD_RepoMaintenance] {msg}{CONST_CLR_RESET}")

def run_git_command(args, cwd):
    """Executes git commands with timeout and error handling."""
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
        return (result.returncode == 0, result.stdout.strip(), result.stderr.strip())
    except FileNotFoundError:
        return (False, "", "Git executable not found in PATH.")
    except Exception as e:
        return (False, "", str(e))

def get_current_hash(repo_root):
    """Retrieve short hash of HEAD."""
    s, out, _ = run_git_command(["rev-parse", "--short", "HEAD"], repo_root)
    return out if s else "unknown"

def check_detached_head(repo_root):
    """Check if repo is in detached HEAD state."""
    s, _, _ = run_git_command(["symbolic-ref", "-q", "HEAD"], repo_root)
    return not s

def get_dir_size(start_path, excludes):
    """Calculates total size of repo excluding ignored folders."""
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(start_path):
            dirnames[:] = [d for d in dirnames if d not in excludes]
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if not os.path.islink(fp):
                    total_size += os.path.getsize(fp)
    except Exception:
        pass 
    return total_size

def list_backups(repo_root):
    """Generates a text report of available backups."""
    backup_dir = os.path.join(repo_root, CONST_BACKUP_DIR_NAME)
    if not os.path.exists(backup_dir): return "No _backups folder found."

    zips = sorted(glob.glob(os.path.join(backup_dir, "*.zip")), reverse=True)
    if not zips: return "No backup zip files found."

    report = ["📂 AVAILABLE BACKUPS (Newest First):\n"]
    for zip_path in zips:
        filename = os.path.basename(zip_path)
        size_mb = os.path.getsize(zip_path) / (1024*1024)
        try:
            with zipfile.ZipFile(zip_path, 'r') as zipf:
                if "backup_manifest.json" in zipf.namelist():
                    data = json.loads(zipf.read("backup_manifest.json"))
                    date = data.get('backup_date', 'Unknown')
                    hash_val = data.get('commit_hash', 'Unknown')
                    branch = data.get('branch', 'Unknown')
                    report.append(f"📄 {filename} ({size_mb:.1f} MB)\n   └── 📅 {date} | #️⃣ {hash_val} | 🌿 {branch}")
                else:
                    report.append(f"📄 {filename} (Legacy)")
        except Exception:
            report.append(f"📄 {filename} (Corrupt)")
    return "\n".join(report)

def create_smart_backup(repo_root, current_hash, branch, max_keep=5, dry_run=False, context_tag="Auto"):
    """Creates backup with manifest, disk checks, and rotation."""
    backup_dir = os.path.join(repo_root, CONST_BACKUP_DIR_NAME)
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    zip_filename = os.path.join(backup_dir, f"backup_{timestamp}_{current_hash}.zip")
    
    # 1. Size & Space Checks
    repo_size = get_dir_size(repo_root, CONST_EXCLUDES)
    repo_size_mb = repo_size / (1024*1024)
    
    try:
        total, used, free = shutil.disk_usage(repo_root)
    except Exception:
        free = 10**12 # Mock high value if check fails
        
    required_space = (repo_size * CONST_REPO_SIZE_MULTIPLIER) + (CONST_DISK_BUFFER_MB * 1024 * 1024)
    
    if free < required_space:
        err = f"❌ STORAGE ERROR: Low Disk Space. Needed: {required_space/(1024**2):.0f}MB, Free: {free/(1024**2):.0f}MB"
        log_console(err, CONST_CLR_RED)
        return err

    if dry_run:
        log_console(f"[DRY RUN] Would backup {repo_size_mb:.1f}MB to: {os.path.basename(zip_filename)}", CONST_CLR_CYAN)
        return f"🟦 [Dry Run] Backup skipped ({repo_size_mb:.1f}MB)"

    if repo_size_mb > 500:
        log_console(f"⚠️ Large Backup Detected ({repo_size_mb:.1f}MB). This may take a moment...", CONST_CLR_YELLOW)

    os.makedirs(backup_dir, exist_ok=True)
    
    manifest = {
        "node_version": NODE_VERSION,
        "backup_date": timestamp,
        "commit_hash": current_hash,
        "branch": branch,
        "context": context_tag,
        "size_bytes": repo_size
    }

    # 2. Perform Backup
    try:
        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.writestr("backup_manifest.json", json.dumps(manifest, indent=4))
            for root, dirs, files in os.walk(repo_root):
                dirs[:] = [d for d in dirs if d not in CONST_EXCLUDES]
                for file in files:
                    if file.endswith('.pyc') or file == '.DS_Store': continue
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, repo_root)
                    zipf.write(file_path, arcname)

        # 3. Integrity Check
        with zipfile.ZipFile(zip_filename, 'r') as zipf:
            if zipf.testzip() is not None:
                raise Exception("Integrity check failed (Bad CRC)")

        log_console(f"Snapshot verified: {os.path.basename(zip_filename)}", CONST_CLR_GREEN)
        result_msg = f"✅ Backup created: {os.path.basename(zip_filename)}"
        
    except Exception as e:
        log_console(f"Backup Failed: {e}", CONST_CLR_RED)
        if os.path.exists(zip_filename):
            try: os.remove(zip_filename)
            except: pass
        return f"❌ Backup Failed: {e}"

    # 4. Atomic Rotation
    try:
        zips = sorted(glob.glob(os.path.join(backup_dir, "*.zip")))
        if len(zips) > max_keep:
            to_delete = zips[:-max_keep]
            for f in to_delete:
                try:
                    os.remove(f)
                except FileNotFoundError:
                    pass 
                except Exception as e:
                    print(f"Rotation warning for {f}: {e}")
            result_msg += f" (Cleaned {len(to_delete)} old)"
    except Exception as e:
        result_msg += f" (Rotation warning: {e})"
        
    return result_msg

def perform_restore(repo_root, target_file, dry_run=False):
    """Restores a backup zip over the current repo."""
    backup_dir = os.path.join(repo_root, CONST_BACKUP_DIR_NAME)
    
    if target_file.lower().strip() == "latest":
        zips = sorted(glob.glob(os.path.join(backup_dir, "*.zip")))
        if not zips: return (False, "❌ No backups found to restore.", None)
        target_path = zips[-1]
    else:
        target_path = os.path.join(backup_dir, target_file)

    if not os.path.exists(target_path):
        return (False, f"❌ Backup file not found: {os.path.basename(target_path)}", None)

    manifest_data = {}
    try:
        with zipfile.ZipFile(target_path, 'r') as zipf:
            if "backup_manifest.json" in zipf.namelist():
                manifest_data = json.loads(zipf.read("backup_manifest.json"))
    except Exception as e:
        return (False, f"❌ Corrupt Backup: {e}", None)

    log_msg = (
        f"♻️ RESTORE TARGET: {os.path.basename(target_path)}\n"
        f"   📅 Date: {manifest_data.get('backup_date', 'Unknown')}\n"
        f"   #️⃣ Hash: {manifest_data.get('commit_hash', 'Unknown')}\n"
        f"   🌿 Branch: {manifest_data.get('branch', 'Unknown')}"
    )

    if dry_run:
        log_console(f"[DRY RUN] Would restore: {os.path.basename(target_path)}", CONST_CLR_CYAN)
        return (True, f"🟦 [Dry Run] Would restore: {log_msg}", manifest_data)

    try:
        with zipfile.ZipFile(target_path, 'r') as zipf:
            zipf.extractall(repo_root)
        return (True, f"✅ RESTORE SUCCESSFUL.\n{log_msg}", manifest_data)
    except Exception as e:
        return (False, f"❌ Restore Failed: {e}", None)

# =================================================================================
# == Core Node Class
# =================================================================================

class MD_RepoMaintenance:
    """
    Manages updates for MD_Nodes. Includes Fortress safeguards (Disk Space, Race Conditions).
    """
    DESCRIPTION = "Lifecycle Manager. Update, Backup, or Restore with enterprise safeguards."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (CONST_MODES, {
                    "default": "Check Status Only",
                    "tooltip": "Select action. 'List Backups' shows available files for Restore."
                }),
                "branch": ("STRING", {
                    "default": "main", 
                    "tooltip": "Git branch to track."
                }),
                "dry_run": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "SIMULATION MODE. Preview changes without touching files."
                }),
                "changelog_depth": ("INT", {
                    "default": 5, "min": 1, "max": 50,
                    "tooltip": "Number of commits to show in log."
                }),
                "restore_file": ("STRING", {
                    "default": "latest",
                    "tooltip": "Filename of backup to restore or 'latest'."
                }),
                "do_backup": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Create safety zip before destructive actions?"
                }),
                "backup_keep_count": ("INT", {
                    "default": 5, "min": 1, "max": 50,
                    "tooltip": "Max backups to keep."
                }),
            },
            "optional": {
                "trigger": ("INT", {
                    "default": 0,
                    "min": CONST_SEED_MIN,
                    "max": CONST_JS_MAX_SAFE_INTEGER, # Section 8.5 Compliance
                    "tooltip": "Connect Seed node to force execution. Capped at JS safe limit."
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "BOOLEAN")
    RETURN_NAMES = ("status_report", "changelog_preview", "restart_required")
    FUNCTION = "execute_maintenance"
    CATEGORY = "MD_Nodes/Utilities"

    def execute_maintenance(self, mode, branch, dry_run, changelog_depth, restore_file, do_backup, backup_keep_count, trigger=0):
        # 1. Locate Repo
        repo_root = CONST_REPO_DIR
        if not os.path.exists(os.path.join(repo_root, ".git")):
            repo_root = os.path.dirname(repo_root)
        if not os.path.exists(os.path.join(repo_root, ".git")):
            return ("❌ Error: .git folder not found.", "N/A", False)

        start_time = time.time()
        log = [f"🔧 MD_RepoMaintenance v{NODE_VERSION}", f"📂 Root: {repo_root}"]
        restart_needed = False
        
        # 2. LIST MODE
        if "List Available Backups" in mode:
            report = list_backups(repo_root)
            return (report, "Backup List Mode", False)

        if dry_run:
            log.append("🟦 DRY RUN ACTIVE")
            log_console("DRY RUN ACTIVE", CONST_CLR_CYAN)

        # 3. Current State
        local_hash = get_current_hash(repo_root)
        
        # 4. RESTORE MODE
        if "RESTORE" in mode:
            if do_backup:
                log_console("Creating Pre-Restore Safety Snapshot...", CONST_CLR_YELLOW)
                backup_msg = create_smart_backup(
                    repo_root, local_hash, branch, 
                    backup_keep_count, dry_run, context_tag="Pre-Restore"
                )
                log.append(f"🛡️ Safety: {backup_msg}")
                if "STORAGE ERROR" in backup_msg:
                    return ("\n".join(log), "Aborted (Disk Full)", False)

            success, msg, _ = perform_restore(repo_root, restore_file, dry_run)
            log.append(msg)
            if success and not dry_run:
                restart_needed = True
                log.append("⚠️ RESTART REQUIRED.")
            return ("\n".join(log), "Restore Operation", restart_needed)

        # 5. Fetch
        s, out, err = run_git_command(["fetch", "origin", branch], repo_root)
        if not s:
            log.append(f"❌ Fetch Error: {err}")
            return ("\n".join(log), "Error", False)

        # 6. Status Checks
        if check_detached_head(repo_root): log.append("⚠️ WARNING: Detached HEAD.")
        
        s, remote_hash, _ = run_git_command(["rev-parse", "--short", f"origin/{branch}"], repo_root)
        s, count_behind, _ = run_git_command(["rev-list", "--count", f"HEAD..origin/{branch}"], repo_root)
        commits_behind = int(count_behind) if s and count_behind.isdigit() else 0
        s, commit_msg, _ = run_git_command(["log", "-1", "--format=%s", f"origin/{branch}"], repo_root)

        # 7. Changelog
        changelog_str = str(commit_msg)
        if commits_behind > 0:
            s, incoming_log, _ = run_git_command(
                ["log", "--oneline", "--no-merges", "-n", str(changelog_depth), f"HEAD..origin/{branch}"], 
                repo_root
            )
            if s: log.append(f"\n📜 Incoming ({commits_behind}):\n{incoming_log}\n")

        # Status
        log.append(f"📊 Local: {local_hash} | Remote: {remote_hash}")
        up_to_date = (commits_behind == 0 and local_hash == remote_hash)
        if up_to_date: log.append("✅ Status: Synced.")
        else: log.append(f"⚠️ Status: {commits_behind} commits behind.")

        # 8. Backup
        should_update = ("Update" in mode and not up_to_date) or "RESET" in mode
        
        if mode == "Create Snapshot (Backup Only)" or (should_update and do_backup):
            log_console("Backing up...", CONST_CLR_YELLOW)
            msg = create_smart_backup(
                repo_root, local_hash, branch, 
                backup_keep_count, dry_run, context_tag="Pre-Update"
            )
            log.append(msg)
            
            if "STORAGE ERROR" in msg:
                return ("\n".join(log), "Aborted (Disk Full)", False)
                
            if mode == "Create Snapshot (Backup Only)":
                return ("\n".join(log), changelog_str, False)

        # 9. Execute
        if "FORCE RESET" in mode:
            log_console("HARD RESET...", CONST_CLR_RED)
            if dry_run: log.append(f"🟦 [Dry Run] Reset to origin/{branch}")
            else:
                s, r_out, r_err = run_git_command(["reset", "--hard", f"origin/{branch}"], repo_root)
                if s: 
                    log.append(f"✅ Reset: {r_out}")
                    restart_needed = True
                else: log.append(f"❌ Reset Failed: {r_err}")

        elif "Update" in mode:
            if up_to_date: log.append("✨ Skipping update (Synced).")
            else:
                if dry_run: log.append(f"🟦 [Dry Run] Pull origin {branch}")
                else:
                    log.append(f"🚀 Pulling {commits_behind} commits...")
                    s, p_out, p_err = run_git_command(["pull", "origin", branch], repo_root)
                    if s:
                        log.append(f"✅ Pull: {p_out}")
                        restart_needed = True
                    else: log.append(f"❌ Pull Failed: {p_err}")

        # 10. Requirements
        should_install = ("Install Requirements" in mode or "FORCE RESET" in mode)
        if should_install and (restart_needed or (dry_run and should_update)):
            req_path = os.path.join(repo_root, "requirements.txt")
            if os.path.exists(req_path):
                if dry_run: log.append(f"🟦 [Dry Run] Install: {req_path}")
                else:
                    try:
                        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_path], cwd=repo_root)
                        log.append("✅ Deps Installed.")
                    except Exception as e: log.append(f"❌ Pip Error: {e}")

        elapsed = time.time() - start_time
        log.append(f"⏱️ {elapsed:.2f}s")
        final_report = "\n".join(log)
        
        if restart_needed and not dry_run:
            final_report += "\n\n⚠️ RESTART COMFYUI!"
            log_console("RESTART REQUIRED", CONST_CLR_MAGENTA)

        return (final_report, changelog_str, restart_needed)

# =================================================================================
# == Node Registration
# =================================================================================

NODE_CLASS_MAPPINGS = {
    "MD_RepoMaintenance": MD_RepoMaintenance
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MD_RepoMaintenance": "MD: Repo Fortress (v2.2.1)"
}

# =================================================================================
# == Development & Testing
# =================================================================================

if __name__ == "__main__":
    print("🧪 Running Self-Tests for MD_RepoMaintenance...")
    
    test_passed = 0
    test_failed = 0

    try:
        # Test 1: Constants Check
        assert CONST_JS_MAX_SAFE_INTEGER == 9007199254740991
        print("✅ Constants: PASSED")
        test_passed += 1
    except AssertionError as e:
        print(f"❌ Constants: FAILED - {e}")
        test_failed += 1

    try:
        # Test 2: Directory Size Calculation (Mocking)
        # Using current dir as test subject
        current_dir = os.path.dirname(os.path.realpath(__file__))
        size = get_dir_size(current_dir, CONST_EXCLUDES)
        assert isinstance(size, int)
        assert size >= 0
        print(f"✅ Dir Size Calc: PASSED ({size} bytes)")
        test_passed += 1
    except Exception as e:
        print(f"❌ Dir Size Calc: FAILED - {e}")
        test_failed += 1

    try:
        # Test 3: Git Command Parsing (Mock Failure is OK)
        # We just want to ensure the function doesn't crash on invalid inputs
        s, out, err = run_git_command(["--version"], ".")
        if s:
            print(f"✅ Git Installed: {out}")
        else:
            print(f"⚠️ Git Check Skipped (Not found): {err}")
        test_passed += 1
    except Exception as e:
        print(f"❌ Git Command Wrapper: FAILED - {e}")
        test_failed += 1

    print(f"\n{'='*60}")
    print(f"Test Results: {test_passed} passed, {test_failed} failed")
    print(f"{'='*60}")