# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# █▓▒░                                                                     ░▒▓█
# █▓▒░              MD_Nodes/Utilities – Repo Fortress v2.3.0              ░▒▓█
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
# ║   Complete Lifecycle Manager with "Fortress-Grade" safeguards.
# ║   Includes Disk Space checks, Concurrency handling, and Restore Safety.
# ║   NOTE: As a system utility, this runs entirely in the public wrapper domain.
# ║
# ║ ░▒▓ FEATURES:
# ║   ✓ Storage Guard: Pre-flight check for disk space vs repo size
# ║   ✓ Race Protection: Microsecond timestamps & atomic rotation logic
# ║   ✓ Time Travel: Restore from backups with pre-restore snapshots
# ║   ✓ Architect Mode: Dry Run simulation for all operations
# ║
# ║ ░▒▓ CHANGELOG:
# ║   - v2.3.0 (Enterprise Standards - Feb 2026):
# ║       • ADDED: PerformanceProfiler class (v1.5.3 standard).
# ║       • ADDED: debug_mode parameter to track heavy IO operations.
# ║       • REFACTOR: Tooltips strictly updated to 5-part v1.5.4 standard.
# ║   - v2.2.1 (Enterprise Compliance):
# ║       • FIXED: trigger input capped at JS_MAX_SAFE_INTEGER
# ║       • ADDED: Embedded Unit Tests 
# ╚════════════════════════════════════════════════════════════════════════════

# =================================================================================
# == Standard Library Imports
# =================================================================================
VERSION = "v2.3.0"  # UPS v1.5.8


import os
import sys
import subprocess
import time
import shutil
import glob
import json
import zipfile
from datetime import datetime
import logging

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
        logging.info(f"    • Total Time: {total:.4f}s")
        for op_name, times in sorted(self.timings.items()):
            avg = sum(times) / len(times)
            if len(times) == 1:
                logging.info(f"    • {op_name}: {avg:.4f}s")
            else:
                logging.info(f"    • {op_name}: {avg:.4f}s avg ({len(times)}x)")

# =================================================================================
# == Configuration Constants
# =================================================================================

NODE_VERSION = "2.3.0"

CONST_JS_MAX_SAFE_INTEGER = 9007199254740991
CONST_SEED_MIN = 0

CONST_GIT_TIMEOUT = 120
CONST_REPO_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
CONST_BACKUP_DIR_NAME = "_backups"

CONST_DISK_BUFFER_MB = 500
CONST_REPO_SIZE_MULTIPLIER = 2  

CONST_MODES = [
    "Check Status Only",
    "📋 List Available Backups",
    "Create Snapshot (Backup Only)",
    "Standard Update (Git Pull)",
    "Update & Install Requirements",
    "⚠️ FORCE RESET & UPDATE (Discard Local Changes)",
    "♻️ RESTORE from Backup (Requires Filename)"
]

CONST_EXCLUDES = [
    '.git', '__pycache__', CONST_BACKUP_DIR_NAME, 
    '.idea', '.vscode', 'node_modules', 
    '.pytest_cache', '.DS_Store', '__MACOSX'
]

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
    logging.info(f"{color}[MD_RepoMaintenance] {msg}{CONST_CLR_RESET}")

def run_git_command(args, cwd):
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
    s, out, _ = run_git_command(["rev-parse", "--short", "HEAD"], repo_root)
    return out if s else "unknown"

def check_detached_head(repo_root):
    s, _, _ = run_git_command(["symbolic-ref", "-q", "HEAD"], repo_root)
    return not s

def get_dir_size(start_path, excludes):
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

def create_smart_backup(repo_root, current_hash, branch, max_keep=5, dry_run=False, context_tag="Auto", profiler=None):
    if profiler: profiler.start("backup_prep")
    backup_dir = os.path.join(repo_root, CONST_BACKUP_DIR_NAME)
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    zip_filename = os.path.join(backup_dir, f"backup_{timestamp}_{current_hash}.zip")
    
    repo_size = get_dir_size(repo_root, CONST_EXCLUDES)
    repo_size_mb = repo_size / (1024*1024)
    
    try:
        total, used, free = shutil.disk_usage(repo_root)
    except Exception:
        free = 10**12 
        
    required_space = (repo_size * CONST_REPO_SIZE_MULTIPLIER) + (CONST_DISK_BUFFER_MB * 1024 * 1024)
    if profiler: profiler.stop("backup_prep")
    
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

    if profiler: profiler.start("zip_write")
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

        with zipfile.ZipFile(zip_filename, 'r') as zipf:
            if zipf.testzip() is not None:
                raise Exception("Integrity check failed (Bad CRC)")

        log_console(f"Snapshot verified: {os.path.basename(zip_filename)}", CONST_CLR_GREEN)
        result_msg = f"✅ Backup created: {os.path.basename(zip_filename)}"
        
    except Exception as e:
        log_console(f"Backup Failed: {e}", CONST_CLR_RED)
        if os.path.exists(zip_filename):
            try: os.remove(zip_filename)
            except Exception: pass
        if profiler: profiler.stop("zip_write")
        return f"❌ Backup Failed: {e}"
    if profiler: profiler.stop("zip_write")

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
                    logging.info(f"Rotation warning for {f}: {e}")
            result_msg += f" (Cleaned {len(to_delete)} old)"
    except Exception as e:
        result_msg += f" (Rotation warning: {e})"
        
    return result_msg

def perform_restore(repo_root, target_file, dry_run=False, profiler=None):
    if profiler: profiler.start("restore_prep")
    backup_dir = os.path.join(repo_root, CONST_BACKUP_DIR_NAME)
    
    if target_file.lower().strip() == "latest":
        zips = sorted(glob.glob(os.path.join(backup_dir, "*.zip")))
        if not zips: 
            if profiler: profiler.stop("restore_prep")
            return (False, "❌ No backups found to restore.", None)
        target_path = zips[-1]
    else:
        target_path = os.path.join(backup_dir, target_file)

    if not os.path.exists(target_path):
        if profiler: profiler.stop("restore_prep")
        return (False, f"❌ Backup file not found: {os.path.basename(target_path)}", None)

    manifest_data = {}
    try:
        with zipfile.ZipFile(target_path, 'r') as zipf:
            if "backup_manifest.json" in zipf.namelist():
                manifest_data = json.loads(zipf.read("backup_manifest.json"))
    except Exception as e:
        if profiler: profiler.stop("restore_prep")
        return (False, f"❌ Corrupt Backup: {e}", None)

    log_msg = (
        f"♻️ RESTORE TARGET: {os.path.basename(target_path)}\n"
        f"   📅 Date: {manifest_data.get('backup_date', 'Unknown')}\n"
        f"   #️⃣ Hash: {manifest_data.get('commit_hash', 'Unknown')}\n"
        f"   🌿 Branch: {manifest_data.get('branch', 'Unknown')}"
    )

    if dry_run:
        log_console(f"[DRY RUN] Would restore: {os.path.basename(target_path)}", CONST_CLR_CYAN)
        if profiler: profiler.stop("restore_prep")
        return (True, f"🟦 [Dry Run] Would restore: {log_msg}", manifest_data)
        
    if profiler: profiler.stop("restore_prep")
    
    if profiler: profiler.start("zip_extract")
    try:
        with zipfile.ZipFile(target_path, 'r') as zipf:
            zipf.extractall(repo_root)
        if profiler: profiler.stop("zip_extract")
        return (True, f"✅ RESTORE SUCCESSFUL.\n{log_msg}", manifest_data)
    except Exception as e:
        if profiler: profiler.stop("zip_extract")
        return (False, f"❌ Restore Failed: {e}", None)

# =================================================================================
# == Core Node Class
# =================================================================================

class MD_RepoMaintenance:
    """Manages updates for MD_Nodes. Includes Fortress safeguards."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (CONST_MODES, {
                    "default": "Check Status Only",
                    "tooltip": (
                        "OPERATION MODE\n"
                        "• Purpose: Select the maintenance action to perform.\n"
                        "• Options:\n"
                        "  - Check Status: View git hash and sync state safely.\n"
                        "  - List Backups: Show available zip restore points.\n"
                        "  - Standard Update: Pull latest changes.\n"
                        "  - FORCE RESET: Discard local changes and reset to origin.\n"
                        "\n⭐ Recommended: Always Check Status before Updating."
                    )
                }),
                "branch": ("STRING", {
                    "default": "main", 
                    "tooltip": (
                        "GIT BRANCH\n"
                        "• Purpose: Target branch for git updates/resets.\n"
                        "\n⭐ Recommended: 'main'. Do not change unless testing."
                    )
                }),
                "dry_run": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "DRY RUN (SIMULATION)\n"
                        "• Purpose: Simulate actions without modifying files.\n"
                        "• Effect: Prints what WOULD happen to the console.\n"
                        "\n⭐ Recommended: True when previewing destructive actions."
                    )
                }),
                "changelog_depth": ("INT", {
                    "default": 5, "min": 1, "max": 50,
                    "tooltip": (
                        "CHANGELOG DEPTH\n"
                        "• Purpose: Number of recent commit messages to display in the status report."
                    )
                }),
                "restore_file": ("STRING", {
                    "default": "latest",
                    "tooltip": (
                        "RESTORE FILENAME\n"
                        "• Purpose: Exact backup zip filename to restore.\n"
                        "• Note: 'latest' automatically picks the newest file in the backup dir."
                    )
                }),
                "do_backup": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "AUTO-BACKUP\n"
                        "• Purpose: Create a full zip snapshot before modifying code.\n"
                        "\n⭐ Recommended: True. Essential for atomic rollbacks."
                    )
                }),
                "backup_keep_count": ("INT", {
                    "default": 5, "min": 1, "max": 50,
                    "tooltip": (
                        "BACKUP RETENTION\n"
                        "• Purpose: Number of recent backups to keep on disk.\n"
                        "• Effect: Older backups are automatically rotated/deleted to save space."
                    )
                }),
            },
            "optional": {
                "trigger": ("INT", {
                    "default": 0,
                    "min": CONST_SEED_MIN,
                    "max": CONST_JS_MAX_SAFE_INTEGER, 
                    "tooltip": (
                        "EXECUTION TRIGGER\n"
                        "• Purpose: Connect a Seed node to force execution on queue.\n"
                        "• Note: Any change in value triggers the node."
                    )
                }),
                "debug_mode": (["0 - Silent", "1 - Info"], {
                    "default": "0 - Silent",
                    "tooltip": "LOGGING VERBOSITY\n• Controls console output and enables System/IO profiling."
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "BOOLEAN")
    RETURN_NAMES = ("status_report", "changelog_preview", "restart_required")
    FUNCTION = "execute_maintenance"
    CATEGORY = "MD_Nodes/Utility"

    def execute_maintenance(self, mode, branch, dry_run, changelog_depth, restore_file, do_backup, backup_keep_count, trigger=0, debug_mode="0 - Silent"):
        
        debug_level = int(debug_mode.split(" ")[0])
        profiler = PerformanceProfiler(enabled=(debug_level >= 1))
        profiler.start("total_execution")
        
        repo_root = CONST_REPO_DIR
        if not os.path.exists(os.path.join(repo_root, ".git")):
            repo_root = os.path.dirname(repo_root)
        if not os.path.exists(os.path.join(repo_root, ".git")):
            return ("❌ Error: .git folder not found.", "N/A", False)

        start_time = time.time()
        log = [f"🔧 MD_RepoMaintenance v{NODE_VERSION}", f"📂 Root: {repo_root}"]
        restart_needed = False
        
        if "List Available Backups" in mode:
            report = list_backups(repo_root)
            profiler.stop("total_execution")
            return (report, "Backup List Mode", False)

        if dry_run:
            log.append("🟦 DRY RUN ACTIVE")
            log_console("DRY RUN ACTIVE", CONST_CLR_CYAN)

        profiler.start("git_status")
        local_hash = get_current_hash(repo_root)
        profiler.stop("git_status")
        
        if "RESTORE" in mode:
            if do_backup:
                log_console("Creating Pre-Restore Safety Snapshot...", CONST_CLR_YELLOW)
                backup_msg = create_smart_backup(
                    repo_root, local_hash, branch, 
                    backup_keep_count, dry_run, context_tag="Pre-Restore", profiler=profiler
                )
                log.append(f"🛡️ Safety: {backup_msg}")
                if "STORAGE ERROR" in backup_msg:
                    profiler.stop("total_execution")
                    return ("\n".join(log), "Aborted (Disk Full)", False)

            success, msg, _ = perform_restore(repo_root, restore_file, dry_run, profiler=profiler)
            log.append(msg)
            if success and not dry_run:
                restart_needed = True
                log.append("⚠️ RESTART REQUIRED.")
            
            profiler.stop("total_execution")
            if debug_level >= 1: profiler.print_report()
            return ("\n".join(log), "Restore Operation", restart_needed)

        profiler.start("git_fetch")
        s, out, err = run_git_command(["fetch", "origin", branch], repo_root)
        profiler.stop("git_fetch")
        
        if not s:
            log.append(f"❌ Fetch Error: {err}")
            profiler.stop("total_execution")
            return ("\n".join(log), "Error", False)

        if check_detached_head(repo_root): log.append("⚠️ WARNING: Detached HEAD.")
        
        s, remote_hash, _ = run_git_command(["rev-parse", "--short", f"origin/{branch}"], repo_root)
        s, count_behind, _ = run_git_command(["rev-list", "--count", f"HEAD..origin/{branch}"], repo_root)
        commits_behind = int(count_behind) if s and count_behind.isdigit() else 0
        s, commit_msg, _ = run_git_command(["log", "-1", "--format=%s", f"origin/{branch}"], repo_root)

        changelog_str = str(commit_msg)
        if commits_behind > 0:
            s, incoming_log, _ = run_git_command(
                ["log", "--oneline", "--no-merges", "-n", str(changelog_depth), f"HEAD..origin/{branch}"], 
                repo_root
            )
            if s: log.append(f"\n📜 Incoming ({commits_behind}):\n{incoming_log}\n")

        log.append(f"📊 Local: {local_hash} | Remote: {remote_hash}")
        up_to_date = (commits_behind == 0 and local_hash == remote_hash)
        if up_to_date: log.append("✅ Status: Synced.")
        else: log.append(f"⚠️ Status: {commits_behind} commits behind.")

        should_update = ("Update" in mode and not up_to_date) or "RESET" in mode
        
        if mode == "Create Snapshot (Backup Only)" or (should_update and do_backup):
            log_console("Backing up...", CONST_CLR_YELLOW)
            msg = create_smart_backup(
                repo_root, local_hash, branch, 
                backup_keep_count, dry_run, context_tag="Pre-Update", profiler=profiler
            )
            log.append(msg)
            
            if "STORAGE ERROR" in msg:
                profiler.stop("total_execution")
                return ("\n".join(log), "Aborted (Disk Full)", False)
                
            if mode == "Create Snapshot (Backup Only)":
                profiler.stop("total_execution")
                if debug_level >= 1: profiler.print_report()
                return ("\n".join(log), changelog_str, False)

        profiler.start("git_execute")
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
        profiler.stop("git_execute")

        profiler.start("pip_install")
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
        profiler.stop("pip_install")

        elapsed = time.time() - start_time
        log.append(f"⏱️ {elapsed:.2f}s")
        final_report = "\n".join(log)
        
        if restart_needed and not dry_run:
            final_report += "\n\n⚠️ RESTART COMFYUI!"
            log_console("RESTART REQUIRED", CONST_CLR_MAGENTA)

        profiler.stop("total_execution")
        if debug_level >= 1: profiler.print_report()
        
        return (final_report, changelog_str, restart_needed)

# =================================================================================
# == Node Registration
# =================================================================================

NODE_CLASS_MAPPINGS = {
    "MD_RepoMaintenance": MD_RepoMaintenance
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MD_RepoMaintenance": "MD: Repo Fortress"
}


# ==============================================================================
# == Unit Tests (smoke — runs without ComfyUI)
# ==============================================================================

if __name__ == "__main__":
    print("\n🧪 Smoke tests: MD_RepoMaintenance")
    print("   VERSION :", VERSION)
    _pass = _fail = 0

    def _check(label, expr):
        global _pass, _fail
        if expr:
            print(f"  ✅  {label}")
            _pass += 1
        else:
            print(f"  ❌  {label}")
            _fail += 1

    _check("VERSION defined",    VERSION == "v2.3.0")
    _check("CONST CONST_JS_MAX_SAFE_INTEGER defined", CONST_JS_MAX_SAFE_INTEGER is not None)
    _check("CONST CONST_SEED_MIN defined", CONST_SEED_MIN is not None)
    _check("CONST CONST_GIT_TIMEOUT defined", CONST_GIT_TIMEOUT is not None)
    _check("CONST CONST_BACKUP_DIR_NAME defined", CONST_BACKUP_DIR_NAME is not None)
    _check("NODE_CLASS_MAPPINGS defined",
           isinstance(NODE_CLASS_MAPPINGS, dict) and len(NODE_CLASS_MAPPINGS) > 0)
    _check("  class MD_RepoMaintenance in map", "MD_RepoMaintenance" in NODE_CLASS_MAPPINGS)

    print(f"\n  {_pass} passed, {_fail} failed")
    if _fail == 0:
        print("  🎉 All good.")
