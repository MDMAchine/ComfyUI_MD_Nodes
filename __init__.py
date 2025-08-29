# C:\Stable Diffusion\ComfyUI\custom_nodes\ComfyUI_MD_Nodes\__init__.py

# Initialize empty dictionaries to collect all mappings
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# --- Import and add mappings from node files ---
# Using relative imports since this __init__.py is now at the package root.
try:
    from .AdvancedMediaSave import NODE_CLASS_MAPPINGS as AMS_CLASS_MAPPINGS
    from .AdvancedMediaSave import NODE_DISPLAY_NAME_MAPPINGS as AMS_DISPLAY_MAPPINGS
    NODE_CLASS_MAPPINGS.update(AMS_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(AMS_DISPLAY_MAPPINGS)
except Exception as e:
    print(f"WARNING: Could not load AdvancedMediaSave: {e}")

try:
    from .APG_Guider_Forked import NODE_CLASS_MAPPINGS as APG_CLASS_MAPPINGS
    from .APG_Guider_Forked import NODE_DISPLAY_NAME_MAPPINGS as APG_DISPLAY_MAPPINGS
    NODE_CLASS_MAPPINGS.update(APG_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(APG_DISPLAY_MAPPINGS)
except Exception as e:
    print(f"WARNING: Could not load APG_Guider_Forked: {e}")

try:
    from .NoiseDecayScheduler_Custom import NODE_CLASS_MAPPINGS as ND_CLASS_MAPPINGS
    from .NoiseDecayScheduler_Custom import NODE_DISPLAY_NAME_MAPPINGS as ND_DISPLAY_MAPPINGS
    NODE_CLASS_MAPPINGS.update(ND_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(ND_DISPLAY_MAPPINGS)
except Exception as e:
    print(f"WARNING: Could not load NoiseDecayScheduler_Custom: {e}")

try:
    from .PingPongSampler_Custom import NODE_CLASS_MAPPINGS as PPS_CLASS_MAPPINGS
    from .PingPongSampler_Custom import NODE_DISPLAY_NAME_MAPPINGS as PPS_DISPLAY_MAPPINGS
    NODE_CLASS_MAPPINGS.update(PPS_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(PPS_DISPLAY_MAPPINGS)
    print("INFO: Successfully loaded original PingPong Sampler (Custom).")
except Exception as e:
    print(f"WARNING: Could not load PingPongSampler_Custom: {e}")

try:
    from .PingPongSampler_Custom_FBG import NODE_CLASS_MAPPINGS as PPS_FBG_CLASS_MAPPINGS
    from .PingPongSampler_Custom_FBG import NODE_DISPLAY_NAME_MAPPINGS as PPS_FBG_DISPLAY_MAPPINGS
    NODE_CLASS_MAPPINGS.update(PPS_FBG_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(PPS_FBG_DISPLAY_MAPPINGS)
    print("INFO: Successfully loaded FBG-integrated PingPong Sampler Custom.")
except Exception as e:
    print(f"WARNING: Could not load FBG-integrated PingPong Sampler Custom: {e}")

try:
    from .SCENE_GENIUS_AUTOCREATOR import NODE_CLASS_MAPPINGS as SGC_CLASS_MAPPINGS
    from .SCENE_GENIUS_AUTOCREATOR import NODE_DISPLAY_NAME_MAPPINGS as SGC_DISPLAY_MAPPINGS
    NODE_CLASS_MAPPINGS.update(SGC_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(SGC_DISPLAY_MAPPINGS)
except Exception as e:
    print(f"WARNING: Could not load SCENE_GENIUS_AUTOCREATOR: {e}")

try:
    from .Hybrid_Sigma_Scheduler import NODE_CLASS_MAPPINGS as HSS_CLASS_MAPPINGS
    from .Hybrid_Sigma_Scheduler import NODE_DISPLAY_NAME_MAPPINGS as HSS_DISPLAY_MAPPINGS
    NODE_CLASS_MAPPINGS.update(HSS_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(HSS_DISPLAY_MAPPINGS)
except Exception as e:
    print(f"WARNING: Could not load Hybrid_Sigma_Scheduler: {e}")

try:
    from .latent.ACE_LATENT_VISUALIZER import NODE_CLASS_MAPPINGS as ALV_CLASS_MAPPINGS
    from .latent.ACE_LATENT_VISUALIZER import NODE_DISPLAY_NAME_MAPPINGS as ALV_DISPLAY_MAPPINGS
    NODE_CLASS_MAPPINGS.update(ALV_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(ALV_DISPLAY_MAPPINGS)
except Exception as e:
    print(f"WARNING: Could not load ACE_LATENT_VISUALIZER: {e}")

try:
    from .audio.AdvancedAudioPreviewAndSave import NODE_CLASS_MAPPINGS as AAPS_CLASS_MAPPINGS
    from .audio.AdvancedAudioPreviewAndSave import NODE_DISPLAY_NAME_MAPPINGS as AAPS_DISPLAY_MAPPINGS
    NODE_CLASS_MAPPINGS.update(AAPS_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(AAPS_DISPLAY_MAPPINGS)
except Exception as e:
    print(f"WARNING: Could not load AdvancedAudioPreviewAndSave: {e}")

try:
    from .audio.mastering_chain_node import NODE_CLASS_MAPPINGS as MCN_CLASS_MAPPINGS
    from .audio.mastering_chain_node import NODE_DISPLAY_NAME_MAPPINGS as MCN_DISPLAY_MAPPINGS
    NODE_CLASS_MAPPINGS.update(MCN_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(MCN_DISPLAY_MAPPINGS)
except Exception as e:
    print(f"WARNING: Could not load mastering_chain_node: {e}")

try:
    from .seed_saver_node import NODE_CLASS_MAPPINGS as SS_CLASS_MAPPINGS
    from .seed_saver_node import NODE_DISPLAY_NAME_MAPPINGS as SS_DISPLAY_MAPPINGS
    NODE_CLASS_MAPPINGS.update(SS_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(SS_DISPLAY_MAPPINGS)
    print("INFO: Successfully loaded MD Seed Saver node.")
except Exception as e:
    print(f"WARNING: Could not load MD Seed Saver node: {e}")

try:
    from .UniversalGuardian import NODE_CLASS_MAPPINGS as UG_CLASS_MAPPINGS
    from .UniversalGuardian import NODE_DISPLAY_NAME_MAPPINGS as UG_DISPLAY_MAPPINGS
    NODE_CLASS_MAPPINGS.update(UG_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(UG_DISPLAY_MAPPINGS)
    print("INFO: Successfully loaded Universal Guardian node.")
except Exception as e:
    print(f"WARNING: Could not load Universal Guardian node: {e}")

# Define __all__ to explicitly expose what should be available when the package is imported
__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
