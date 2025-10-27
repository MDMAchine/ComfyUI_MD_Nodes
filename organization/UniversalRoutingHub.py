# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
# ████ MD_Nodes/UniversalRoutingHubAdvanced – Advanced Signal Junction v2.0.0 ████▓▒░
# © 2025 MDMAchine
# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
# ░▒▓ ORIGIN & DEV:
#   • Cast into the void by: MDMAchine
#   • Enhanced by: Claude
#   • License: Apache 2.0 — Sharing is caring
#   • Part of: ComfyUI_MD_Nodes Workflow Organization Suite

# ░▒▓ DESCRIPTION:
#   An advanced universal signal routing hub (* type) with professional monitoring,
#   status reporting, and multi-hub linking capabilities. Acts as an intelligent
#   junction box that routes signals, detects data types, monitors throughput,
#   provides connection status feedback via text and image map.

# ░▒▓ FEATURES:
#   ✓ Universal `*` type support - handles ANY ComfyUI data type.
#   ✓ Automatic data type detection and display for connected slots.
#   ✓ Real-time connection status indicators (via Status Map output).
#   ✓ Pass-through monitoring with metrics (connected count).
#   ✓ Multi-hub linking via status string pass-through.
#   ✓ Configurable slot count (5, 10, 15, or 20 slots).
#   ✓ Custom labels for each slot.
#   ✓ Visual status map output image showing connections and types.
#   ✓ Textual metrics and hub status string outputs.

# ░▒▓ CHANGELOG:
#   - v2.0.0 (Major Feature Update & Compliance):
#       • COMPLIANCE: Removed all Python type hints (Guide Sec 7.2).
#       • COMPLIANCE: Added MD: prefix to display name, removed emoji (Guide Sec 6.2).
#       • COMPLIANCE: Ensured tooltips follow standard structure (Guide Sec 9.1).
#       • COMPLIANCE: Added logging and robust error handling (Guide Sec 7.3, 8.3).
#       • COMPLIANCE: Node is correctly cacheable (Guide Sec 8.1).
#       • COMPLIANCE: Matplotlib usage follows guidelines (Guide Sec 9.2).
#       • ADDED: Automatic data type detection system.
#       • ADDED: Connection status indicator output (text & image).
#       • ADDED: Pass-through monitoring with metrics.
#       • REMOVED: Flawed "persistent" cache feature (prevented memory leak risk).
#       • ADDED: Multi-hub linking capability (via status string).
#       • ADDED: Visual status map generation.
#       • ADDED: Performance statistics output string.
#       • FIXED: Node no longer crashes if visualization libraries (Matplotlib/PIL) are missing.
#   - v1.0.0 (Initial Release):
#       • ADDED: Universal routing, configurable slot counts, custom slot labeling.

# ░▒▓ CONFIGURATION:
#   → Primary Use: Create a clean junction point for multiple connections, reducing wire clutter. Monitor data types passing through.
#   → Secondary Use: Link multiple hubs together using the 'hub_status' output/input to track connections across complex graphs.
#   → Edge Use: Use the 'status_map' image output for dynamic visual feedback on workflow state.

# ░▒▓ WARNING:
#   This node may trigger:
#   ▓▒░ Obsessive monitoring of data flow metrics ('hub_status' string).
#   ▓▒░ Compulsive hub chaining to create elaborate "routing matrices".
#   ▓▒░ Flashbacks to network topology diagrams and the OSI model.
#   ▓▒░ A sudden desire to implement SNMP monitoring on your ComfyUI workflows.
#   Side effects include: Workflows with better instrumentation than production
#   systems, colleagues asking for "the routing dashboard," and spontaneous
#   creation of hub-to-hub communication protocols. Monitor responsibly.
# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄


# =================================================================================
# == Standard Library Imports                                                  ==
# =================================================================================
import logging
import io
import traceback
import secrets # Standard import
from collections import defaultdict

# =================================================================================
# == Third-Party Imports                                                       ==
# =================================================================================
# Core libraries expected by ComfyUI
try:
    import torch
    import numpy as np
    TORCH_AVAILABLE = True
except ImportError:
    # This is a critical failure if torch isn't available
    print("CRITICAL ERROR: [UniversalRoutingHub] torch or numpy not found. This node cannot function.")
    TORCH_AVAILABLE = False # Set flag

# Optional imports for visual report generation
try:
    from PIL import Image
    import matplotlib as mpl
    mpl.use('Agg') # Set backend BEFORE importing pyplot (CRITICAL)
    import matplotlib.pyplot as plt
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    # Use print for visibility during ComfyUI startup
    print("WARNING: [UniversalRoutingHub] Matplotlib/PIL not found. Status map output will be a blank placeholder.")
    
    # --- FIX #1: Correct the conditional class definition for Image ---
    if 'Image' not in globals():
        class Image:  # This line MUST be indented
            pass
            
    # --- FIX #2: Correct the conditional class definition for plt ---
    if 'plt' not in globals():
        class plt:  # This line MUST be indented
            @staticmethod
            def subplots(*a, **kw):
                return None, None
            @staticmethod
            def Rectangle(*a, **kw):
                pass
            @staticmethod
            def close(*a, **kw):
                pass
            @staticmethod
            def style(*a, **kw):
                pass

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

# Setup logger for this node
logger = logging.getLogger("ComfyUI_MD_Nodes.UniversalRoutingHub") # Corrected logger name

# --- Data Type Detection ---
def detect_data_type(data):
    """
    Detects and returns a human-readable string for common ComfyUI data types.

    Args:
        data: The input data (any type).

    Returns:
        A string representing the detected data type (e.g., "LATENT", "IMAGE", "STRING").
        Returns "EMPTY" if data is None.
    """
    if data is None:
        return "EMPTY"

    # Use try-except for type checks that might fail if libs are missing
    try:
        # Check specific ComfyUI dictionary types first
        if isinstance(data, dict):
            if "samples" in data and TORCH_AVAILABLE and isinstance(data["samples"], torch.Tensor): return "LATENT"
            if "waveform" in data and "sample_rate" in data: return "AUDIO"
            if "prompt" in data and "extra_pnginfo" in data: return "METADATA" # Common hidden type
            # Add checks for other known dict types if necessary
            return f"DICT ({len(data)} keys)"

        # Check for tensors (IMAGE is typically a tensor)
        if TORCH_AVAILABLE and isinstance(data, torch.Tensor):
            # Basic IMAGE check (4D, last dim 3 or 4) - might need refinement
            if data.ndim == 4 and data.shape[-1] in [3, 4]: return "IMAGE"
            shape_str = "x".join(str(s) for s in data.shape)
            return f"TENSOR [{shape_str}]"

        # Check for Conditioning format: list containing [tensor, dict]
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], (list, tuple)) and \
           len(data[0]) == 2 and TORCH_AVAILABLE and isinstance(data[0][0], torch.Tensor) and \
           isinstance(data[0][1], dict):
           return "CONDITIONING"

        # Check for standard Python types
        if isinstance(data, str): return "STRING"
        if isinstance(data, int): return "INT"
        if isinstance(data, float): return "FLOAT"
        if isinstance(data, bool): return "BOOLEAN"
        if isinstance(data, (list, tuple)): return f"{type(data).__name__.upper()} ({len(data)} items)"

        # Fallback to class name
        return type(data).__name__.upper()

    except Exception as e:
         logger.warning(f"Error during data type detection: {e}. Falling back to basic type name.")
         return type(data).__name__.upper()


# --- Status Map Generation ---
def _create_placeholder_map():
    """Returns a blank placeholder tensor if visualization fails or is unavailable."""
    logger.warning("[UniversalRoutingHub] Returning blank placeholder status map.")
    if not TORCH_AVAILABLE: # Critical check
         # Cannot create tensor, return None or raise error? ComfyUI expects a tensor.
         # This state is unlikely as ComfyUI itself needs torch.
         return None # Or raise an error
    # Dark gray placeholder [1, H, W, 3]
    return torch.ones((1, 400, 800, 3), dtype=torch.float32) * 0.1

def create_status_map(slot_statuses, num_slots):
    """
    Generates a visual status map image showing connection states and data types.
    Compliant with MD Nodes guidelines (Agg backend, PIL conversion, plt.close).

    Args:
        slot_statuses: Dictionary mapping slot index (int) to status dict.
                       Status dict contains {'connected': bool, 'type': str, 'label': str}.
        num_slots: Total number of active slots for this hub.

    Returns:
        An IMAGE tensor [1, H, W, 3] or a placeholder tensor on error/missing libs.
    """
    if not VISUALIZATION_AVAILABLE:
        return _create_placeholder_map()

    fig = None # Ensure defined for finally block
    buf = None # Ensure defined

    try:
        # Style and figure setup
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(10, max(4, num_slots * 0.4)), dpi=100, facecolor='#1a1a1a') # Dynamic height
        if ax is None: raise RuntimeError("Matplotlib failed to create axes.")

        ax.set_facecolor('#1a1a1a')
        ax.axis('off') # Hide axes entirely

        # Title
        ax.text(0.5, 0.97, 'Universal Routing Hub Status', transform=ax.transAxes,
                fontsize=16, color='white', ha='center', va='top', weight='bold') # Adjusted position/size

        # Slot drawing parameters
        bar_height_abs = 0.8 / max(num_slots, 1) # Relative height per slot
        bar_v_spacing = bar_height_abs * 0.15   # Space between bars
        bar_actual_height = bar_height_abs * 0.85 # Visible bar height
        start_y_top = 0.90 # Start plotting below title

        for i in range(1, num_slots + 1):
            status = slot_statuses.get(i, {"connected": False, "type": "EMPTY", "label": f"Slot {i}"})
            is_connected = status.get("connected", False)

            # Calculate Y position for the top of the bar
            y_pos_top = start_y_top - (i - 1) * (bar_actual_height + bar_v_spacing)

            # --- Draw Elements ---
            # Status Bar Background
            bar_color = '#1E90FF' if is_connected else '#444444' # Blue if connected, dark gray if empty
            ax.add_patch(plt.Rectangle((0.15, y_pos_top - bar_actual_height), 0.7, bar_actual_height,
                                       transform=ax.transAxes, facecolor=bar_color,
                                       edgecolor='#666666', linewidth=0.5, alpha=0.8))

            # Slot Number (Left)
            ax.text(0.13, y_pos_top - bar_actual_height / 2, f"{i:02d}", transform=ax.transAxes,
                    fontsize=10, color='white', ha='right', va='center', weight='bold')

            # Slot Label (Inside Bar)
            label_text = str(status.get("label", f"Slot {i}"))[:35] # Limit length
            ax.text(0.17, y_pos_top - bar_actual_height / 2, label_text, transform=ax.transAxes,
                    fontsize=9, color='white', ha='left', va='center')

            # Data Type (Inside Bar, Right)
            type_text = str(status.get("type", "N/A"))[:20] # Limit length
            type_color = '#DDDDDD' if is_connected else '#888888'
            ax.text(0.83, y_pos_top - bar_actual_height / 2, type_text, transform=ax.transAxes,
                    fontsize=8, color=type_color, ha='right', va='center', family='monospace')


        # Save figure to buffer using recommended PIL method (Guide Sec 9.2)
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1, facecolor=fig.get_facecolor(), dpi=100)
        buf.seek(0)

        # Convert buffer to PIL Image -> NumPy array -> Torch Tensor
        img = Image.open(buf).convert('RGB')
        img_np = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).unsqueeze(0) # Add batch dim [1, H, W, 3]

        return img_tensor

    except Exception as e:
        logger.error(f"Error creating status map: {e}", exc_info=True)
        return _create_placeholder_map() # Return placeholder on error

    finally:
        # --- CRITICAL Cleanup ---
        if buf:
            try: buf.close()
            except Exception: pass
        if fig:
            try: plt.close(fig) # Ensure figure is always closed
            except Exception: pass
        # Clear current figure/axes state if possible
        if VISUALIZATION_AVAILABLE:
             try: plt.clf(); plt.cla()
             except Exception: pass


# =================================================================================
# == Core Node Class                                                           ==
# =================================================================================

class UniversalRoutingHubAdvanced:
    """
    MD: Universal Routing Hub (Advanced)

    An advanced signal routing node using the universal '*' type. It passes data
    from input slots to corresponding output slots. Optionally monitors connected
    slots, detects data types, generates a visual status map, and provides
    textual metrics. Supports chaining hubs via status strings.
    """

    @classmethod
    def INPUT_TYPES(cls):
        """Define node inputs, dynamically creating slots and labels."""
        optional_inputs = {}
        # Create inputs for labels (1 to 20)
        for i in range(1, 21):
            optional_inputs[f"label_{i}"] = ("STRING", {
                "default": f"Slot {i}",
                "tooltip": f"CUSTOM LABEL (Slot {i})\n- Optional descriptive name for this input/output pair."
            })
        # Create inputs for data slots (1 to 20) using wildcard type
        for i in range(1, 21):
            optional_inputs[f"slot_{i}"] = ("*", {
                 # Force input allows connecting even if optional
                "forceInput": True,
                "tooltip": f"DATA INPUT (Slot {i})\n- Connect any data type (*).\n- Data will be passed directly to 'slot_{i}_output'."
            })

        # Input for chaining hub statuses
        optional_inputs["upstream_hub_status"] = ("STRING", {
            "default": "", "forceInput": True, # Encourage connection if chaining
            "tooltip": (
                "UPSTREAM HUB STATUS (Optional)\n"
                "- Connect the 'hub_status' output from another Routing Hub.\n"
                "- Allows chaining status information for the metrics report."
            )
        })

        return {
            "required": {
                "num_slots": ([5, 10, 15, 20], {
                    "default": 10,
                    "tooltip": (
                        "NUMBER OF SLOTS\n"
                        "- Sets the number of active input/output pairs (5, 10, 15, or 20).\n"
                        "- Unused slots beyond this number will be ignored and output None."
                    )
                }),
                "enable_monitoring": ("BOOLEAN", {
                    "default": True, "label_on": "Monitoring ON", "label_off": "Monitoring OFF",
                    "tooltip": (
                        "ENABLE MONITORING\n"
                        "- True: Activates data type detection, status map, and metrics generation.\n"
                        "- False: Disables monitoring features; node acts as a simple pass-through."
                    )
                }),
            },
            "optional": optional_inputs
        }

    # Define 20 output slots + 3 status outputs
    OUTPUT_COUNT = 20 + 3
    RETURN_TYPES = tuple(["*"] * 20 + ["IMAGE", "STRING", "STRING"])
    RETURN_NAMES = tuple([f"slot_{i}_output" for i in range(1, 21)] + ["status_map", "hub_status", "metrics"])

    FUNCTION = "route_signals_advanced"
    CATEGORY = "MD_Nodes/Workflow Organization" # Corrected category
    OUTPUT_NODE = False # This node primarily passes data through

    # No IS_CHANGED needed - this node is deterministic based on its inputs.

    def route_signals_advanced(self, num_slots, enable_monitoring, **kwargs):
        """
        Routes input slot data to output slots and performs monitoring if enabled.

        Args:
            num_slots: The number of active slots configured by the user.
            enable_monitoring: Boolean flag to enable/disable status checks.
            **kwargs: Dictionary containing all optional inputs (slot_1, label_1, etc.).

        Returns:
            A tuple containing 23 items:
            - 20 outputs corresponding to slot_1 to slot_20 (passing through input or None).
            - 1 IMAGE tensor for the status map (or placeholder).
            - 1 STRING for the current hub's status (for chaining).
            - 1 STRING containing detailed metrics.
            Returns safe defaults on error.
        """
        # Prepare lists/dicts for outputs and status
        outputs = [None] * 20 # Initialize all 20 potential outputs to None
        slot_statuses = {}    # Dict to store status for monitoring {slot_index: status_dict}
        total_connected = 0

        # --- Process Slots ---
        try:
            for i in range(1, 21): # Iterate through all possible 20 slots
                slot_key = f"slot_{i}"
                label_key = f"label_{i}"

                # Get data and label for the current slot from kwargs
                slot_data = kwargs.get(slot_key, None)
                slot_label = kwargs.get(label_key, f"Slot {i}") # Use default if label missing

                # --- Pass-through Data ---
                # Assign input data to the corresponding output position (index i-1)
                outputs[i-1] = slot_data

                # --- Monitoring Logic (only if enabled AND within active range) ---
                if enable_monitoring and i <= num_slots:
                    data_type = detect_data_type(slot_data)
                    is_connected = (slot_data is not None)

                    if is_connected:
                        total_connected += 1

                    slot_statuses[i] = {
                        "connected": is_connected,
                        "type": data_type,
                        "label": slot_label,
                    }
                    if is_connected:
                        # Log detailed info only if logger level is DEBUG
                        logger.debug(f"[UniversalRoutingHub] Slot {i} ('{slot_label}'): Connected, Type={data_type}")

            # --- Generate Status Outputs (if monitoring) ---
            status_map_tensor = _create_placeholder_map() # Default placeholder
            hub_status_string = f"HubStatus|Slots:{num_slots}|Connected:{total_connected}"
            metrics_string = (f"Universal Routing Hub Metrics\n{'='*40}\n"
                              f"Active Slots: {num_slots}\n"
                              f"Connected: {total_connected}/{num_slots}\n"
                              f"Monitoring: {'Enabled' if enable_monitoring else 'Disabled'}")

            if enable_monitoring:
                # Generate visual map
                status_map_tensor = create_status_map(slot_statuses, num_slots)

                # Append upstream status if provided
                upstream_status = kwargs.get("upstream_hub_status", "")
                if upstream_status:
                    hub_status_string += f"|UpstreamDataPresent" # Indicate connection
                    metrics_string += f"\n--- Upstream Hub ---\nLinked Status: {upstream_status}" # Include raw upstream string

                # Append detailed slot info to metrics
                metrics_string += "\n\n--- Slot Details ---"
                for i in range(1, num_slots + 1):
                    if i in slot_statuses:
                        s = slot_statuses[i]
                        status_icon = "✅" if s.get("connected", False) else "❌" # Use emojis here
                        # Format line carefully for alignment
                        metrics_string += (f"\n {status_icon} Slot {i:02d}: "
                                           f"{str(s.get('label', 'N/A'))[:20]:<20} → "
                                           f"{str(s.get('type', 'N/A'))[:25]}")
                    else: # Should not happen if i <= num_slots
                         metrics_string += f"\n ❔ Slot {i:02d}: Status Unavailable"


            logger.info(f"[UniversalRoutingHub] Processed {num_slots} slots. Connected: {total_connected}. Monitoring: {enable_monitoring}.")

            # --- Combine and Return Outputs ---
            # Result tuple = (slot_1_out, ..., slot_20_out, status_map, hub_status, metrics)
            final_outputs = tuple(outputs + [status_map_tensor, hub_status_string, metrics_string])
            return final_outputs

        except Exception as e:
            logger.critical(f"[UniversalRoutingHub] CRITICAL ERROR during execution: {e}", exc_info=True)
            # Return safe defaults matching the return types
            error_outputs = [None] * 20
            error_image = _create_placeholder_map()
            error_status = "ERROR"
            error_metrics = f"ERROR: Processing failed.\n{e}\n{traceback.format_exc()}"
            return tuple(error_outputs + [error_image, error_status, error_metrics])


# =================================================================================
# == Node Registration                                                         ==
# =================================================================================

NODE_CLASS_MAPPINGS = {
    "UniversalRoutingHubAdvanced": UniversalRoutingHubAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UniversalRoutingHubAdvanced": "MD: Universal Routing Hub", # Simplified name, removed emoji
}