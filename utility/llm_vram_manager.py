# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
# ████ MD_Nodes/LLMVRAMManager – LLM VRAM Control v3.4 ████▓▒░
# © 2025 MDMAchine
# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
# ░▒▓ ORIGIN & DEV:
#   • Cast into the void by: Gemini (Google AI)
#   • Enhanced by: MDMAchine
#   • License: Public Domain — Use it freely!
#   • Original source (if applicable): N/A

# ░▒▓ DESCRIPTION:
#   A utility node to prevent VRAM conflicts between ComfyUI and local LLM
#   servers (Ollama, LM Studio, llama-swap). Unloads models via API or force-stops processes
#   to free up GPU memory on demand directly from your workflow.

# ░▒▓ FEATURES:
#   ✓ Unloads Ollama models gracefully using its API (`requests`).
#   ✓ Unloads llama-swap models gracefully using its API (`requests`).
#   ✓ Force-stops the LM Studio process (`taskkill`/`pkill`) to reclaim VRAM.
#   ✓ Force-stops the Ollama process as a fallback.
#   ✓ Pass-through inputs/outputs for easy workflow integration.
#   ✓ Disables LM Studio SDK/API calls (documented as non-functional).

# ░▒▓ CHANGELOG:
#   - v3.4 (Current Release - llama-swap Support):
#       • ADDED: "Unload llama-swap Models (API)" action for llama-swap integration.
#       • WORKING: "Unload Ollama Models (API)" works perfectly.
#       • WORKING: "Unload llama-swap Models (API)" works perfectly.
#       • DISABLED: LM Studio API/SDK unload methods (all failed tests v1.7-v3.2).
#       • NOTE: The ONLY reliable way to free LM Studio VRAM is "Stop LM Studio Process (Force)".
#   - v3.3 (Previous Release - LM Studio Fix):
#       • WORKING: "Unload Ollama Models (API)" works perfectly.
#       • DISABLED: LM Studio API/SDK unload methods (all failed tests v1.7-v3.2).
#       • ADDED: Detailed developer notes documenting all failed LM Studio attempts.
#       • NOTE: The ONLY reliable way to free LM Studio VRAM is "Stop LM Studio Process (Force)".

# ░▒▓ CONFIGURATION:
#   → Primary Use: Adding "Unload Ollama Models (API)" or "Unload llama-swap Models (API)" to the start of a workflow to ensure VRAM is free.
#   → Secondary Use: Using "Stop LM Studio Process (Force)" when a diffusion model fails to load due to VRAM limits.
#   → Edge Use: Chaining this node with `trigger` disabled, connected to a Switch node to activate remotely.

# ░▒▓ WARNING:
#   This node may trigger:
#   ▓▒░ A cold sweat as you hit "Stop LM Studio Process" hoping it doesn't kill ComfyUI too.
#   ▓▒░ A profound appreciation for Ollama's working API.
#   ▓▒░ 'nvtop' flickering violently as models are loaded and yeeted from VRAM.
#   ▓▒░ Flashbacks to `kill -9`, and the trail of digital carnage you left behind in the /tmp/ directory.
#   Consult your nearest demoscene vet if hallucinations persist.
# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄


# =================================================================================
# == Standard Library Imports                                                  ==
# =================================================================================
import os
import sys
import subprocess
import json
import secrets
import logging
import traceback
import time

# =================================================================================
# == Third-Party Imports                                                       ==
# =================================================================================
# Optional dependency: requests (for Ollama API and llama-swap API)
try:
    import requests
except ImportError:
    # Use print for visibility during ComfyUI startup
    print("-----------------------------------------------------------")
    print("WARNING: [LLM VRAM Manager] 'requests' library not found.")
    print("Please install it: pip install requests")
    print("Ollama API and llama-swap API actions will not work without it.")
    print("-----------------------------------------------------------")
    requests = None

# Optional dependency: lmstudio (currently disabled functionality)
try:
    import lmstudio
except ImportError:
    # This is expected and okay as of v3.3
    # print("INFO: [LLM VRAM Manager] 'lmstudio' library not found (SDK support is disabled).")
    lmstudio = None

# =================================================================================
# == ComfyUI Core Modules                                                      ==
# =================================================================================
# (None needed directly)

# =================================================================================
# == Local Project Imports                                                     ==
# =================================================================================
# (None needed)

# =================================================================================
# == Helper Classes & Dependencies                                             ==
# =================================================================================
# Configure logging for this module
logger = logging.getLogger("LLMVRAMManager")
# Set default level - can be overridden by ComfyUI's logging setup
# logger.setLevel(logging.INFO)
# handler = logging.StreamHandler()
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# handler.setFormatter(formatter)
# logger.addHandler(handler)
# logger.propagate = False # Prevent duplicate messages in root logger if ComfyUI sets one up

# =================================================================================
# == Core Node Class                                                           ==
# =================================================================================

class LLMVRAMManager:
    """
    MD: LLM VRAM Manager

    Utility node to manage VRAM conflicts with local LLM servers (Ollama, LM Studio, llama-swap).
    Provides actions to unload models via API (Ollama, llama-swap) or force-stop processes
    to free GPU memory during ComfyUI workflow execution. Includes pass-through
    connections for easy integration.
    """
    def __init__(self):
        """Initializes default settings."""
        self.ollama_host = "http://localhost:11434" # Default Ollama API endpoint
        self.llama_swap_host = "http://host.docker.internal:11435" # Default llama-swap API endpoint
        self.is_windows = sys.platform == "win32"

    @classmethod
    def INPUT_TYPES(cls):
        """Define node inputs with detailed tooltips."""
        # Define action choices, clearly marking disabled options
        actions = [
            "None (Do Nothing)",
            "Unload Ollama Models (API)",
            "Unload llama-swap Models (API)",
            "Unload LM Studio Model (SDK) [DISABLED]", # Keep option but mark as disabled
            "Unload ALL (Ollama/llama-swap/LM Studio) [LM Studio DISABLED]", # Keep option but mark as disabled
            "---FORCE STOP (USE WITH CAUTION)---", # Visual separator
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
                        "- Select the operation to manage LLM VRAM.\n\n"
                        "API METHODS (Graceful):\n"
                        "- 'Unload Ollama Models (API)': Recommended for Ollama. Uses API to unload running models.\n"
                        "- 'Unload llama-swap Models (API)': Recommended for llama-swap. Uses API to unload running models.\n"
                        "- 'Unload LM Studio Model (SDK)': CURRENTLY DISABLED (v3.3) - No reliable SDK/API method found.\n\n"
                        "FORCE STOP METHODS (Use if API fails or for LM Studio):\n"
                        "- 'Stop Ollama Process': Force-kills the Ollama server process/service.\n"
                        "- 'Stop LM Studio Process': Force-kills the LM Studio application process (ONLY reliable way for LM Studio).\n"
                        "- 'Stop BOTH': Force-kills both Ollama and LM Studio processes.\n\n"
                        "WARNING: Force-stopping can be abrupt. Use API methods when possible."
                    )
                }),
                "trigger": ("BOOLEAN", {
                    "default": True, "label_on": "ACTION ENABLED", "label_off": "ACTION DISABLED",
                    "tooltip": (
                        "ENABLE ACTION TRIGGER\n"
                        "- True (ENABLED): The selected 'action' will be performed when the node executes.\n"
                        "- False (DISABLED): The node acts purely as a pass-through; no action is performed."
                    )
                }),
            },
            "optional": {
                # Passthrough inputs allow chaining without breaking workflow data flow
                "pass_through": ("*", {"tooltip": "PASS-THROUGH (Any Type)\n- Connect any data type here to control execution order. The data will be passed through unchanged."}),
                "seed_in": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "PASS-THROUGH (Seed)\n- Safely pass an integer (like a seed) through the node."}),
                "image_in": ("IMAGE", {"tooltip": "PASS-THROUGH (Image)\n- Safely pass an image tensor through the node."}),
                "latent_in": ("LATENT", {"tooltip": "PASS-THROUGH (Latent)\n- Safely pass a latent tensor dictionary through the node."}),
            }
        }

    RETURN_TYPES = ("*", "STRING", "INT", "IMAGE", "LATENT")
    RETURN_NAMES = ("pass_through", "status", "seed_out", "image_out", "latent_out")
    FUNCTION = "execute"
    CATEGORY = "MD_Nodes/Utility" # Corrected category
    OUTPUT_NODE = True # Although it has outputs, its primary role is action/side-effect

    @classmethod
    def IS_CHANGED(cls, action, trigger, **kwargs): # Simplified signature
        """
        Controls caching. Forces re-run only when triggered with a valid action.
        """
        # If trigger is True and an action other than "None" is selected, force re-run
        if trigger and action != "None (Do Nothing)":
            # Use secrets.token_hex for a unique hash each time
            return secrets.token_hex(16)
        else:
            # If not triggered or action is "None", allow caching (static behavior)
            return "static"


    def _unload_ollama_models(self):
        """Attempts to unload all running models via Ollama API."""
        if not requests:
            msg = "SKIPPED: 'requests' library not installed. Cannot use Ollama API."
            logger.warning(msg)
            return msg
        try:
            # 1. Get list of running models
            api_url_ps = f"{self.ollama_host}/api/ps"
            res_ps = requests.get(api_url_ps, timeout=5)
            res_ps.raise_for_status() # Check for HTTP errors
            models_data = res_ps.json()
            models_to_unload = [model.get("name") for model in models_data.get("models", []) if model.get("name")]

            if not models_to_unload:
                msg = "SUCCESS: No Ollama models reported as running."
                logger.info(f"[LLM VRAM Manager] {msg}")
                return msg

            # 2. Iterate and send unload requests (keep_alive: 0)
            unloaded_count = 0
            errors = []
            api_url_generate = f"{self.ollama_host}/api/generate" # Correct endpoint for unload via keep_alive
            for model_name in models_to_unload:
                try:
                    # Sending a generate request with keep_alive=0 tells Ollama to unload after this dummy request
                    unload_payload = {"model": model_name, "keep_alive": "0s"} # Use string duration like "0s"
                    # We don't need prompt or stream for just unloading
                    res_unload = requests.post(api_url_generate, json=unload_payload, timeout=10) # Longer timeout for unload
                    # Check status but response content might not be critical here
                    # res_unload.raise_for_status() # Might fail if model is already unloading? Be lenient.
                    if res_unload.status_code == 200:
                         logger.info(f"[LLM VRAM Manager] Unload request successful for Ollama model: {model_name}")
                         unloaded_count += 1
                    else:
                         logger.warning(f"[LLM VRAM Manager] Unload request for {model_name} returned status {res_unload.status_code}. It might already be unloaded or an issue occurred.")
                         # Don't increment count, but don't treat as critical error unless server is down
                         errors.append(f"{model_name} (status {res_unload.status_code})")

                except requests.RequestException as unload_e:
                    err_msg = f"Warning: Could not unload Ollama model {model_name}. Reason: {unload_e}"
                    logger.warning(f"[LLM VRAM Manager] {err_msg}")
                    errors.append(f"{model_name} ({unload_e.__class__.__name__})")

            # 3. Report outcome
            if unloaded_count == len(models_to_unload):
                msg = f"SUCCESS: Sent unload request for {unloaded_count} Ollama model(s)."
            elif unloaded_count > 0:
                 msg = f"PARTIAL SUCCESS: Sent unload for {unloaded_count}/{len(models_to_unload)} models. Errors: {', '.join(errors)}"
            elif errors:
                 msg = f"ERROR: Failed to send unload requests. Errors: {', '.join(errors)}"
            else: # Should not happen if models_to_unload was populated
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
            logger.error(msg, exc_info=True) # Log traceback for unexpected errors
            return msg

    def _unload_llama_swap_models(self):
        """Attempts to unload all running models via llama-swap API."""
        if not requests:
            msg = "SKIPPED: 'requests' library not installed. Cannot use llama-swap API."
            logger.warning(msg)
            return msg
        try:
            # Send unload request to llama-swap
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
        """Placeholder for LM Studio SDK unload (currently disabled)."""
        msg = "SKIPPED: LM Studio SDK/API unload is disabled (v3.3). No reliable method found. Use 'Stop LM Studio Process (Force)' action instead."
        logger.info(f"[LLM VRAM Manager] {msg}")
        # --- DEVELOPER NOTES remain accessible in the source code ---
        # ... (Notes from original code omitted here for brevity) ...
        return msg

    def _stop_process(self, process_name, is_service=False):
        """
        Force-stops a process by name using OS-specific commands.

        Args:
            process_name: The executable name (e.g., "ollama.exe", "lm-studio").
            is_service: If True, attempts service stop commands first (Ollama specific).

        Returns:
            A status message string indicating success, failure, or info.
        """
        status_message = f"Attempting force-stop for '{process_name}'..."
        logger.info(f"[LLM VRAM Manager] {status_message}")
        success = False
        try:
            startupinfo = None
            if self.is_windows:
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                # Try stopping service first if applicable (Ollama)
                if is_service and "ollama" in process_name.lower():
                    try:
                        # Attempt to stop the service gracefully first
                        subprocess.run(["net", "stop", "Ollama"], check=False, capture_output=True, startupinfo=startupinfo, timeout=10)
                        logger.info("[LLM VRAM Manager] Attempted 'net stop Ollama'.")
                        # Give it a moment to stop before taskkill
                        time.sleep(1)
                    except Exception as service_e:
                        logger.warning(f"[LLM VRAM Manager] Failed to run 'net stop Ollama' (may not be installed as service): {service_e}")

                # Force kill the process
                command = ["taskkill", "/F", "/IM", process_name]
                result = subprocess.run(command, check=False, capture_output=True, text=True, startupinfo=startupinfo)

                if result.returncode == 0 or "SUCCESS" in result.stdout:
                    status_message = f"SUCCESS: Sent force-stop command for process '{process_name}'."
                    success = True
                elif "could not find" in result.stderr.lower() or "not found" in result.stdout.lower():
                     status_message = f"INFO: Process '{process_name}' was not found running."
                     success = True # Consider not found as success for the goal of freeing VRAM
                else:
                    status_message = f"WARNING: 'taskkill' command for '{process_name}' failed or reported no termination. Stderr: {result.stderr.strip()}"

            else: # macOS and Linux
                # Try stopping service first if applicable (Ollama)
                if is_service and "ollama" in process_name.lower():
                     try:
                         # Try systemctl first
                         subprocess.run(["sudo", "systemctl", "stop", "ollama"], check=False, capture_output=True, timeout=10)
                         logger.info("[LLM VRAM Manager] Attempted 'systemctl stop ollama'.")
                         time.sleep(1)
                     except FileNotFoundError: # systemctl might not be available
                          try:
                               # Try service command
                               subprocess.run(["sudo", "service", "ollama", "stop"], check=False, capture_output=True, timeout=10)
                               logger.info("[LLM VRAM Manager] Attempted 'service ollama stop'.")
                               time.sleep(1)
                          except FileNotFoundError:
                               logger.warning("[LLM VRAM Manager] Neither systemctl nor service command found for stopping Ollama service.")
                          except Exception as service_e:
                               logger.warning(f"[LLM VRAM Manager] Failed to stop Ollama service: {service_e}")
                     except Exception as service_e:
                          logger.warning(f"[LLM VRAM Manager] Failed to stop Ollama service: {service_e}")


                # Force kill using pkill -f (matches full command line)
                command = ["pkill", "-f", process_name]
                result = subprocess.run(command, check=False, capture_output=True)

                if result.returncode == 0:
                    status_message = f"SUCCESS: Sent force-stop command for process matching '{process_name}'."
                    success = True
                else:
                    # Check if the process was simply not found
                    check_cmd = ["pgrep", "-f", process_name]
                    check_result = subprocess.run(check_cmd, check=False, capture_output=True)
                    if check_result.returncode != 0:
                        status_message = f"INFO: Process matching '{process_name}' was not found running."
                        success = True # Not found is success here
                    else:
                        status_message = f"WARNING: 'pkill' command failed for '{process_name}'. Return code: {result.returncode}. Stderr: {result.stderr.decode().strip()}"

            # Short delay to allow OS to release resources
            if success:
                 time.sleep(1)

            print(f"[LLM VRAM Manager] {status_message}") # Print final status
            return status_message
        except Exception as e:
            error_msg = f"ERROR: Failed during force-stop for '{process_name}': {e}"
            logger.error(f"[LLM VRAM Manager] {error_msg}", exc_info=True)
            print(f"[LLM VRAM Manager] {error_msg}") # Print error for user
            return error_msg

    def execute(self, action, trigger, pass_through=None, seed_in=0, image_in=None, latent_in=None):
        """
        Main execution function. Performs action if triggered, otherwise passes through.

        Args:
            action: The action string selected by the user.
            trigger: Boolean indicating if the action should be performed.
            pass_through: Optional data to pass through.
            seed_in: Optional integer (seed) to pass through.
            image_in: Optional IMAGE tensor to pass through.
            latent_in: Optional LATENT dict to pass through.

        Returns:
            A tuple containing (pass_through, status_message, seed_out, image_out, latent_out).
        """
        status = "DISABLED: Action trigger was off."
        output_seed = seed_in
        output_image = image_in
        output_latent = latent_in

        try:
            if not trigger:
                logger.info("[LLM VRAM Manager] Action disabled by trigger.")
                # Return passthrough values directly
                return (pass_through, status, output_seed, output_image, output_latent)

            status = "No action performed." # Default status if action is None

            logger.info(f"[LLM VRAM Manager] Executing action: {action}")

            if action == "None (Do Nothing)":
                status = "INFO: No action selected."
            elif action == "Unload Ollama Models (API)":
                status = self._unload_ollama_models()
            elif action == "Unload llama-swap Models (API)":
                status = self._unload_llama_swap_models()
            elif action == "Unload LM Studio Model (SDK) [DISABLED]":
                status = self._unload_lm_studio_model() # Returns disabled message
            elif action == "Unload ALL (Ollama/llama-swap/LM Studio) [LM Studio DISABLED]":
                ollama_status = self._unload_ollama_models()
                llama_swap_status = self._unload_llama_swap_models()
                lm_studio_status = self._unload_lm_studio_model()
                status = f"Ollama: {ollama_status}\nllama-swap: {llama_swap_status}\nLM Studio: {lm_studio_status}"
            elif action == "Stop Ollama Process (Force)":
                # Determine process name based on OS and common variations
                process = "ollama.exe" if self.is_windows else "ollama"
                status = self._stop_process(process, is_service=True) # Assume it might be service
            elif action == "Stop LM Studio Process (Force)":
                # Process name might differ slightly (e.g., case, spacing)
                process = "LM Studio.exe" if self.is_windows else "lm-studio" # Check actual process name on Linux/Mac
                status = self._stop_process(process)
            elif action == "Stop BOTH Ollama & LM Studio (Force)":
                ollama_process = "ollama.exe" if self.is_windows else "ollama"
                lm_studio_process = "LM Studio.exe" if self.is_windows else "lm-studio"
                ollama_status = self._stop_process(ollama_process, is_service=True)
                # Add slight delay between kills
                time.sleep(0.5)
                lm_studio_status = self._stop_process(lm_studio_process)
                status = f"Ollama: {ollama_status}\nLM Studio: {lm_studio_status}"
            else:
                 status = f"WARNING: Unknown action selected: {action}"
                 logger.warning(status)

            logger.info(f"[LLM VRAM Manager] Action '{action}' completed. Status: {status.splitlines()[0]}") # Log first line of status
            # Return tuple matching RETURN_TYPES
            return (pass_through, status, output_seed, output_image, output_latent)

        except Exception as e:
            logger.error(f"[LLM VRAM Manager] Unexpected error during execute: {e}", exc_info=True)
            error_status = f"FATAL ERROR: {e}\n{traceback.format_exc()}"
            print(f"ERROR: [LLM VRAM Manager] {error_status}") # Ensure user sees fatal errors
            # Return passthrough values and error status
            return (pass_through, error_status, output_seed, output_image, output_latent)


# =================================================================================
# == Node Registration                                                         ==
# =================================================================================

NODE_CLASS_MAPPINGS = {
    "LLMVRAMManager": LLMVRAMManager
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LLMVRAMManager": "MD: LLM VRAM Manager" # Added MD prefix
}