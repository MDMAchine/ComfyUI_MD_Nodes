# C:\Stable Diffusion\ComfyUI\custom_nodes\MD_Nodes\__init__.py

# Initialize empty dictionaries to collect all mappings
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# --- DIAGNOSTIC PRINTS (Keep these for now to confirm behavior) ---
#import sys
#print(f"\n--- MD_Nodes __init__.py Diagnostics ---")
#print(f"__name__: {__name__}")
#print(f"__package__: {__package__}")
#print(f"sys.path (relevant part for MD_Nodes):")
#for p in sys.path:
#    if "ComfyUI" in p and "MD_Nodes" not in p:
#        print(f"  - {p}")
#print(f"--- End MD_Nodes __init__.py Diagnostics ---\n")
# --- END DIAGNOSTIC PRINTS ---


# --- Import and add mappings from node files ---
# Using absolute imports starting with 'MD_Nodes' as the base.
# making 'MD_Nodes' discoverable as a top-level module relative to 'custom_nodes'.
try:
    from MD_Nodes.APG_Guider_Forked import NODE_CLASS_MAPPINGS as APG_CLASS_MAPPINGS
    from MD_Nodes.APG_Guider_Forked import NODE_DISPLAY_NAME_MAPPINGS as APG_DISPLAY_MAPPINGS
    NODE_CLASS_MAPPINGS.update(APG_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(APG_DISPLAY_MAPPINGS)
except Exception as e:
    print(f"WARNING: Could not load APG_Guider_Forked: {e}")

try:
    from MD_Nodes.NoiseDecayScheduler_Custom import NODE_CLASS_MAPPINGS as ND_CLASS_MAPPINGS
    from MD_Nodes.NoiseDecayScheduler_Custom import NODE_DISPLAY_NAME_MAPPINGS as ND_DISPLAY_MAPPINGS
    NODE_CLASS_MAPPINGS.update(ND_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(ND_DISPLAY_MAPPINGS)
except Exception as e:
    print(f"WARNING: Could not load NoiseDecayScheduler_Custom: {e}")

try:
    from MD_Nodes.PingPongSampler_Custom import NODE_CLASS_MAPPINGS as PPS_CLASS_MAPPINGS
    from MD_Nodes.PingPongSampler_Custom import NODE_DISPLAY_NAME_MAPPINGS as PPS_DISPLAY_MAPPINGS
    NODE_CLASS_MAPPINGS.update(PPS_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(PPS_DISPLAY_MAPPINGS)
except Exception as e:
    print(f"WARNING: Could not load PingPongSampler_Custom: {e}")

try:
    from MD_Nodes.SCENE_GENIUS_AUTOCREATOR import NODE_CLASS_MAPPINGS as SGC_CLASS_MAPPINGS
    from MD_Nodes.SCENE_GENIUS_AUTOCREATOR import NODE_DISPLAY_NAME_MAPPINGS as SGC_DISPLAY_MAPPINGS
    NODE_CLASS_MAPPINGS.update(SGC_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(SGC_DISPLAY_MAPPINGS)
except Exception as e:
    print(f"WARNING: Could not load SCENE_GENIUS_AUTOCREATOR: {e}")

try:
    from MD_Nodes.Hybrid_Sigma_Scheduler import NODE_CLASS_MAPPINGS as HSS_CLASS_MAPPINGS
    from MD_Nodes.Hybrid_Sigma_Scheduler import NODE_DISPLAY_NAME_MAPPINGS as HSS_DISPLAY_MAPPINGS
    NODE_CLASS_MAPPINGS.update(HSS_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(HSS_DISPLAY_MAPPINGS)
except Exception as e:
    print(f"WARNING: Could not load Hybrid_Sigma_Scheduler: {e}")

# --- Import and add mappings from the 'latent' subdirectory ---
try:
    from MD_Nodes.latent.ACE_LATENT_VISUALIZER import NODE_CLASS_MAPPINGS as ALV_CLASS_MAPPINGS
    from MD_Nodes.latent.ACE_LATENT_VISUALIZER import NODE_DISPLAY_NAME_MAPPINGS as ALV_DISPLAY_MAPPINGS
    NODE_CLASS_MAPPINGS.update(ALV_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(ALV_DISPLAY_MAPPINGS)
except Exception as e:
    print(f"WARNING: Could not load ACE_LATENT_VISUALIZER: {e}")

# --- Import and add mappings from the 'audio' subdirectory ---
try:
    from MD_Nodes.audio.AdvancedAudioPreviewAndSave import NODE_CLASS_MAPPINGS as AAPS_CLASS_MAPPINGS
    from MD_Nodes.audio.AdvancedAudioPreviewAndSave import NODE_DISPLAY_NAME_MAPPINGS as AAPS_DISPLAY_MAPPINGS
    NODE_CLASS_MAPPINGS.update(AAPS_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(AAPS_DISPLAY_MAPPINGS)
except Exception as e:
    print(f"WARNING: Could not load AdvancedAudioPreviewAndSave: {e}")

try:
    from MD_Nodes.audio.mastering_chain_node import NODE_CLASS_MAPPINGS as MCN_CLASS_MAPPINGS
    from MD_Nodes.audio.mastering_chain_node import NODE_DISPLAY_NAME_MAPPINGS as MCN_DISPLAY_MAPPINGS
    NODE_CLASS_MAPPINGS.update(MCN_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(MCN_DISPLAY_MAPPINGS)
except Exception as e:
    print(f"WARNING: Could not load mastering_chain_node: {e}")

# Define __all__ to explicitly expose what should be available when the package is imported
__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]