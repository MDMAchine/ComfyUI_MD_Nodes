# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# █▓▒░                                                                     ░▒▓█
# █▓▒░          MD_Nodes/LLMVRAMManager – LLM VRAM Control v1.6.1          ░▒▓█
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
# ║   • Cast into the void by: Gemini (Google AI)
# ║   • Enhanced by: MDMAchine
# ║
# ║ ░▒▓ DESCRIPTION:
# ║   A utility node to prevent VRAM conflicts between ComfyUI and local LLM
# ║   servers (Ollama, LM Studio, llama-swap). Unloads models via API or force-stops
# ║   processes to free up GPU memory on demand directly from your workflow.
# ║   NOTE: As a system utility, this runs entirely in the public wrapper domain.
# ║
# ║ ░▒▓ FEATURES:
# ║   ✓ Unloads Ollama models gracefully using its API (`requests`).
# ║   ✓ Unloads llama-swap models gracefully using its API (`requests`).
# ║   ✓ Force-stops the LM Studio process (`taskkill`/`pkill`) to reclaim VRAM.
# ║   ✓ Force-stops the Ollama process as a fallback.
# ║
# ║ ░▒▓ CHANGELOG:
# ║   - v1.6.1 (2026-04-16) - Public Release Cleanup:
# ║       • FIX: llama-swap host now defaults to localhost:11435 instead of
# ║         host.docker.internal. Override via MD_LLAMA_SWAP_HOST env var.
# ║       • FIX: ollama host also respects MD_OLLAMA_HOST env var override.
# ║   - v1.6.0 (Enterprise Standards - Feb 2026):
# ║       • ADDED: PerformanceProfiler class (v1.5.3 standard).
# ║       • ADDED: debug_mode parameter.
# ║       • REFACTOR: Tooltips strictly updated to 5-part v1.5.4 standard.
# ╚════════════════════════════════════════════════════════════════════════════

# =================================================================================
# == Standard Library Imports                                                    ==
# =================================================================================
VERSION = "v1.6.1"  # UPS v1.5.8


import os
import sys
import subprocess
import json
import secrets
import logging
import traceback
import time

# =================================================================================
# == Third-Party Imports                                                         ==
# =================================================================================
try:
    import requests
except ImportError:
    logging.info("-----------------------------------------------------------")
    logging.warning("WARNING: [LLM VRAM Manager] 'requests' library not found.")
    logging.info("Please install it: pip install requests")
    logging.info("Ollama API and llama-swap API actions will not work without it.")
    logging.info("-----------------------------------------------------------")
    requests = None

try:
    import lmstudio
except ImportError:
    lmstudio = None

# =================================================================================
# == Helper Classes (Enterprise Standards)                                       ==
# =================================================================================

logger = logging.getLogger("LLMVRAMManager")

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
# == Core Node Class                                                             ==
# =================================================================================

class LLMVRAMManager:
    """
    MD: LLM VRAM Manager
    Utility node to manage VRAM conflicts with local LLM servers.
    """
    def __init__(self):
        self.ollama_host = os.environ.get("MD_OLLAMA_HOST", "http://localhost:11434")
        # Default is localhost. If running llama-swap inside Docker, set the
        # MD_LLAMA_SWAP_HOST environment variable to http://host.docker.internal:11435
        self.llama_swap_host = os.environ.get("MD_LLAMA_SWAP_HOST", "http://localhost:11435")
        self.is_windows = sys.platform == "win32"

    @classmethod
    def INPUT_TYPES(cls):
        actions = [
            "None (Do Nothing)",
            "Unload Ollama Models (API)",
            "Unload llama-swap Models (API)",
            "Unload LM Studio Model (SDK) [DISABLED]",
            "Unload ALL (Ollama/llama-swap/LM Studio) [LM Studio DISABLED]", 
            "---FORCE STOP (USE WITH CAUTION)---", 
            "Stop Ollama Process (Force)",
            "Stop LM Studio Process (Force)",
            "Stop BOTH Ollama & LM Studio (Force)"
        ]
        return {
            "required": {
                "action": (actions, {
                    "default": "None (Do Nothing)",
                    "tooltip": (
                        "ACTION TO PERFORM\n"
                        "• Purpose: Select the operation to free up LLM VRAM.\n"
                        "• Options: API unloads (graceful) or Process Kills (forceful).\n"
                        "• Trade-offs: Force stops are abrupt but guaranteed to clear memory.\n"
                        "\n⭐ Recommended: Try API methods first before resorting to force stops."
                    )
                }),
                "trigger": ("BOOLEAN", {
                    "default": True, "label_on": "ACTION ENABLED", "label_off": "ACTION DISABLED",
                    "tooltip": (
                        "ENABLE ACTION TRIGGER\n"
                        "• Purpose: Master switch for the node's destructive actions.\n"
                        "• Options: True executes the action; False makes the node a passive pass-through.\n"
                        "\n⭐ Recommended: True."
                    )
                }),
            },
            "optional": {
                "pass_through": ("*", {
                    "tooltip": (
                        "PASS-THROUGH (Generic)\n"
                        "• Purpose: Connect any generic data type here to enforce workflow execution order."
                    )
                }),
                "seed_in": ("INT", {
                    "default": 0, "min": 0, "max": 0xffffffffffffffff, 
                    "tooltip": (
                        "PASS-THROUGH (Seed)\n"
                        "• Purpose: Pass an integer/seed safely through the node to control execution order."
                    )
                }),
                "image_in": ("IMAGE", {
                    "tooltip": (
                        "PASS-THROUGH (Image)\n"
                        "• Purpose: Pass an image tensor safely through the node."
                    )
                }),
                "latent_in": ("LATENT", {
                    "tooltip": (
                        "PASS-THROUGH (Latent)\n"
                        "• Purpose: Pass a latent dictionary safely through the node."
                    )
                }),
                "debug_mode": (["0 - Silent", "1 - Info"], {
                    "default": "0 - Silent",
                    "tooltip": "LOGGING VERBOSITY\n• Controls console output and System/IO profiling."
                }),
            }
        }

    RETURN_TYPES = ("*", "STRING", "INT", "IMAGE", "LATENT")
    RETURN_NAMES = ("pass_through", "status", "seed_out", "image_out", "latent_out")
    FUNCTION = "execute"
    CATEGORY = "MD_Nodes/Utility" 
    OUTPUT_NODE = True 

    @classmethod
    def IS_CHANGED(cls, action, trigger, **kwargs): 
        if trigger and action != "None (Do Nothing)":
            return secrets.token_hex(16)
        return "static"

    def _unload_ollama_models(self):
        if not requests:
            msg = "SKIPPED: 'requests' library not installed. Cannot use Ollama API."
            logger.warning(msg)
            return msg
        try:
            api_url_ps = f"{self.ollama_host}/api/ps"
            res_ps = requests.get(api_url_ps, timeout=5)
            res_ps.raise_for_status() 
            models_data = res_ps.json()
            models_to_unload = [model.get("name") for model in models_data.get("models", []) if model.get("name")]

            if not models_to_unload:
                msg = "SUCCESS: No Ollama models reported as running."
                logger.info(f"[LLM VRAM Manager] {msg}")
                return msg

            unloaded_count = 0
            errors = []
            api_url_generate = f"{self.ollama_host}/api/generate" 
            for model_name in models_to_unload:
                try:
                    unload_payload = {"model": model_name, "keep_alive": "0s"} 
                    res_unload = requests.post(api_url_generate, json=unload_payload, timeout=10) 
                    if res_unload.status_code == 200:
                         logger.info(f"[LLM VRAM Manager] Unload request successful for Ollama model: {model_name}")
                         unloaded_count += 1
                    else:
                         logger.warning(f"[LLM VRAM Manager] Unload request for {model_name} returned status {res_unload.status_code}.")
                         errors.append(f"{model_name} (status {res_unload.status_code})")

                except requests.RequestException as unload_e:
                    err_msg = f"Warning: Could not unload Ollama model {model_name}. Reason: {unload_e}"
                    logger.warning(f"[LLM VRAM Manager] {err_msg}")
                    errors.append(f"{model_name} ({unload_e.__class__.__name__})")

            if unloaded_count == len(models_to_unload):
                msg = f"SUCCESS: Sent unload request for {unloaded_count} Ollama model(s)."
            elif unloaded_count > 0:
                 msg = f"PARTIAL SUCCESS: Sent unload for {unloaded_count}/{len(models_to_unload)} models. Errors: {', '.join(errors)}"
            elif errors:
                 msg = f"ERROR: Failed to send unload requests. Errors: {', '.join(errors)}"
            else: 
                 msg = "INFO: No models needed unloading or encountered issues."

            logger.info(f"[LLM VRAM Manager] {msg}")
            return msg

        except requests.ConnectionError:
            msg = "SKIPPED: Ollama server connection failed. Is it running?"
            logger.warning(msg)
            return msg
        except requests.Timeout:
             msg = "ERROR: Ollama API request timed out."
             logger.error(msg)
             return msg
        except Exception as e:
            msg = f"ERROR: Ollama API unload failed unexpectedly: {e}"
            logger.error(msg, exc_info=True) 
            return msg

    def _unload_llama_swap_models(self):
        if not requests:
            msg = "SKIPPED: 'requests' library not installed. Cannot use llama-swap API."
            logger.warning(msg)
            return msg
        try:
            api_url_unload = f"{self.llama_swap_host}/models/unload"
            res_unload = requests.post(api_url_unload, timeout=10)
            
            if res_unload.status_code == 200:
                msg = "SUCCESS: llama-swap models unloaded successfully."
                logger.info(f"[LLM VRAM Manager] {msg}")
                return msg
            else:
                msg = f"WARNING: llama-swap unload returned status {res_unload.status_code}."
                logger.warning(f"[LLM VRAM Manager] {msg}")
                return msg

        except requests.ConnectionError:
            msg = "SKIPPED: llama-swap server connection failed. Is it running?"
            logger.warning(msg)
            return msg
        except requests.Timeout:
            msg = "ERROR: llama-swap API request timed out."
            logger.error(msg)
            return msg
        except Exception as e:
            msg = f"ERROR: llama-swap API unload failed unexpectedly: {e}"
            logger.error(msg, exc_info=True)
            return msg

    def _unload_lm_studio_model(self):
        msg = "SKIPPED: LM Studio SDK/API unload is disabled (v3.3). No reliable method found. Use 'Stop LM Studio Process (Force)' action instead."
        logger.info(f"[LLM VRAM Manager] {msg}")
        return msg

    def _stop_process(self, process_name, is_service=False):
        status_message = f"Attempting force-stop for '{process_name}'..."
        logger.info(f"[LLM VRAM Manager] {status_message}")
        success = False
        try:
            startupinfo = None
            if self.is_windows:
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                if is_service and "ollama" in process_name.lower():
                    try:
                        subprocess.run(["net", "stop", "Ollama"], check=False, capture_output=True, startupinfo=startupinfo, timeout=10)
                        logger.info("[LLM VRAM Manager] Attempted 'net stop Ollama'.")
                        time.sleep(1)
                    except Exception as service_e:
                        logger.warning(f"[LLM VRAM Manager] Failed to run 'net stop Ollama': {service_e}")

                command = ["taskkill", "/F", "/IM", process_name]
                result = subprocess.run(command, check=False, capture_output=True, text=True, startupinfo=startupinfo)

                if result.returncode == 0 or "SUCCESS" in result.stdout:
                    status_message = f"SUCCESS: Sent force-stop command for process '{process_name}'."
                    success = True
                elif "could not find" in result.stderr.lower() or "not found" in result.stdout.lower():
                     status_message = f"INFO: Process '{process_name}' was not found running."
                     success = True 
                else:
                    status_message = f"WARNING: 'taskkill' command for '{process_name}' failed. Stderr: {result.stderr.strip()}"

            else: 
                if is_service and "ollama" in process_name.lower():
                     try:
                         subprocess.run(["sudo", "systemctl", "stop", "ollama"], check=False, capture_output=True, timeout=10)
                         logger.info("[LLM VRAM Manager] Attempted 'systemctl stop ollama'.")
                         time.sleep(1)
                     except FileNotFoundError: 
                          try:
                               subprocess.run(["sudo", "service", "ollama", "stop"], check=False, capture_output=True, timeout=10)
                               logger.info("[LLM VRAM Manager] Attempted 'service ollama stop'.")
                               time.sleep(1)
                          except Exception as service_e:
                               logger.warning(f"[LLM VRAM Manager] Failed to stop Ollama service: {service_e}")
                     except Exception as service_e:
                          logger.warning(f"[LLM VRAM Manager] Failed to stop Ollama service: {service_e}")

                command = ["pkill", "-f", process_name]
                result = subprocess.run(command, check=False, capture_output=True)

                if result.returncode == 0:
                    status_message = f"SUCCESS: Sent force-stop command for process matching '{process_name}'."
                    success = True
                else:
                    check_cmd = ["pgrep", "-f", process_name]
                    check_result = subprocess.run(check_cmd, check=False, capture_output=True)
                    if check_result.returncode != 0:
                        status_message = f"INFO: Process matching '{process_name}' was not found running."
                        success = True 
                    else:
                        status_message = f"WARNING: 'pkill' command failed for '{process_name}'. Stderr: {result.stderr.decode().strip()}"

            if success:
                 time.sleep(1)

            logging.info(f"[LLM VRAM Manager] {status_message}")
            return status_message
        except Exception as e:
            error_msg = f"ERROR: Failed during force-stop for '{process_name}': {e}"
            logger.error(f"[LLM VRAM Manager] {error_msg}", exc_info=True)
            logging.info(f"[LLM VRAM Manager] {error_msg}")
            return error_msg

    def execute(self, action, trigger, pass_through=None, seed_in=0, image_in=None, latent_in=None, debug_mode="0 - Silent"):
        debug_level = int(debug_mode.split(" ")[0])
        profiler = PerformanceProfiler(enabled=(debug_level >= 1))
        profiler.start("total_execution")
        
        status = "DISABLED: Action trigger was off."
        output_seed = seed_in
        output_image = image_in
        output_latent = latent_in

        try:
            if not trigger:
                logger.info("[LLM VRAM Manager] Action disabled by trigger.")
                profiler.stop("total_execution")
                return (pass_through, status, output_seed, output_image, output_latent)

            status = "No action performed." 

            logger.info(f"[LLM VRAM Manager] Executing action: {action}")

            profiler.start("vram_action")
            if action == "None (Do Nothing)":
                status = "INFO: No action selected."
            elif action == "Unload Ollama Models (API)":
                status = self._unload_ollama_models()
            elif action == "Unload llama-swap Models (API)":
                status = self._unload_llama_swap_models()
            elif action == "Unload LM Studio Model (SDK) [DISABLED]":
                status = self._unload_lm_studio_model() 
            elif action == "Unload ALL (Ollama/llama-swap/LM Studio) [LM Studio DISABLED]":
                ollama_status = self._unload_ollama_models()
                llama_swap_status = self._unload_llama_swap_models()
                lm_studio_status = self._unload_lm_studio_model()
                status = f"Ollama: {ollama_status}\nllama-swap: {llama_swap_status}\nLM Studio: {lm_studio_status}"
            elif action == "Stop Ollama Process (Force)":
                process = "ollama.exe" if self.is_windows else "ollama"
                status = self._stop_process(process, is_service=True) 
            elif action == "Stop LM Studio Process (Force)":
                process = "LM Studio.exe" if self.is_windows else "lm-studio" 
                status = self._stop_process(process)
            elif action == "Stop BOTH Ollama & LM Studio (Force)":
                ollama_process = "ollama.exe" if self.is_windows else "ollama"
                lm_studio_process = "LM Studio.exe" if self.is_windows else "lm-studio"
                ollama_status = self._stop_process(ollama_process, is_service=True)
                time.sleep(0.5)
                lm_studio_status = self._stop_process(lm_studio_process)
                status = f"Ollama: {ollama_status}\nLM Studio: {lm_studio_status}"
            else:
                 status = f"WARNING: Unknown action selected: {action}"
                 logger.warning(status)
            profiler.stop("vram_action")

            logger.info(f"[LLM VRAM Manager] Action '{action}' completed. Status: {status.splitlines()[0]}") 
            
            profiler.stop("total_execution")
            if debug_level >= 1: profiler.print_report()
            
            return (pass_through, status, output_seed, output_image, output_latent)

        except Exception as e:
            logger.error(f"[LLM VRAM Manager] Unexpected error during execute: {e}", exc_info=True)
            error_status = f"FATAL ERROR: {e}\n{traceback.format_exc()}"
            logging.error(f"ERROR: [LLM VRAM Manager] {error_status}")
            return (pass_through, error_status, output_seed, output_image, output_latent)


# =================================================================================
# == Node Registration                                                           ==
# =================================================================================

NODE_CLASS_MAPPINGS = {
    "LLMVRAMManager": LLMVRAMManager
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LLMVRAMManager": "MD: LLM VRAM Manager"
}


# ==============================================================================
# == Unit Tests (smoke — runs without ComfyUI)
# ==============================================================================

if __name__ == "__main__":
    print("\n🧪 Smoke tests: MD_LLM_VRAMManager")
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

    _check("VERSION defined",    VERSION == "v1.6.1")
    _check("NODE_CLASS_MAPPINGS defined",
           isinstance(NODE_CLASS_MAPPINGS, dict) and len(NODE_CLASS_MAPPINGS) > 0)
    _check("  class LLMVRAMManager in map", "LLMVRAMManager" in NODE_CLASS_MAPPINGS)

    print(f"\n  {_pass} passed, {_fail} failed")
    if _fail == 0:
        print("  🎉 All good.")
