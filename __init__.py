"""
ComfyUI Custom Nodes - Auto-generated __init__.py
Generated: 2025-11-24 00:48:17
"""
import os
import sys
import importlib.util

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
if PACKAGE_DIR not in sys.path:
  sys.path.insert(0, PACKAGE_DIR)

# --- Public Node Imports ---
from .APG_Guider_Forked import APGGuiderNode
from .AdvancedMediaSave import AdvancedMediaSave
from .Hybrid_Sigma_Scheduler import HybridAdaptiveSigmas
from .Hybrid_Sigma_Scheduler import SigmaConcatenate
from .Hybrid_Sigma_Scheduler import SigmaSmooth
from .NoiseDecayScheduler_Custom import NoiseDecayScheduler_Custom
from .PingPongSampler_Custom import PingPongSamplerNode
from .PingPongSampler_Custom_FBG import PingPongSamplerNodeFBG
from .SCENE_GENIUS_AUTOCREATOR import SceneGeniusAutocreator
from .WildcardPromptBuilder import WildcardPromptBuilder
from .audio.AdvancedAudioPreviewAndSave import AdvancedAudioPreviewAndSave
from .audio.MD_AutoMasterNode import MD_AutoMasterNode
from .audio.auto_eq import MD_AudioAutoEQ
from .audio.mastering_chain_node import MasteringChainNode
from .conditioning.ACE_T5_Conditioning_Node import AceT5ConditioningAnalyzer
from .conditioning.ACE_T5_Conditioning_Node import AceT5ConditioningScheduled
from .conditioning.ACE_T5_Conditioning_Node import AceT5ModelLoader
from .guiders.MD_ApplyTPG import MD_ApplyTPG
from .latent.ACE_LATENT_VISUALIZER import ACELatentVisualizer
from .organization.AutoLayoutOptimizer import AutoLayoutOptimizer
from .organization.EnhancedAnnotation import EnhancedAnnotationNode
from .organization.SmartColorPaletteManager import SmartColorPaletteManager
from .organization.UniversalRoutingHub import UniversalRoutingHubAdvanced
from .organization.WorkflowSectionOrganizer import WorkflowSectionOrganizer
from .seed_saver_node import EnhancedSeedSaverNode
from .text.TextInputNode import AdvancedTextNode
from .text.TextInputNode import TextFileLoader
from .utility.GPUTemperatureProtection import GPUTemperatureProtectionEnhanced
from .utility.SmartFilenameBuilder import FilenameCounterNode
from .utility.SmartFilenameBuilder import FilenameTokenReplacer
from .utility.SmartFilenameBuilder import SmartFilenameBuilder
from .utility.llm_vram_manager import LLMVRAMManager

# --- Main Mapping (Public Nodes) ---
NODE_CLASS_MAPPINGS = {
    "ACE_LatentVisualizer": ACELatentVisualizer,
    "APGGuiderForked": APGGuiderNode,
    "AceT5ConditioningAnalyzer": AceT5ConditioningAnalyzer,
    "AceT5ConditioningScheduled": AceT5ConditioningScheduled,
    "AceT5ModelLoader": AceT5ModelLoader,
    "AdvancedAudioPreviewAndSave": AdvancedAudioPreviewAndSave,
    "AdvancedMediaSave": AdvancedMediaSave,
    "AdvancedTextNode": AdvancedTextNode,
    "AutoLayoutOptimizer": AutoLayoutOptimizer,
    "EnhancedAnnotationNode": EnhancedAnnotationNode,
    "EnhancedSeedSaver": EnhancedSeedSaverNode,
    "FilenameCounterNode": FilenameCounterNode,
    "FilenameTokenReplacer": FilenameTokenReplacer,
    "GPUTemperatureProtectionEnhanced": GPUTemperatureProtectionEnhanced,
    "HybridAdaptiveSigmas": HybridAdaptiveSigmas,
    "LLMVRAMManager": LLMVRAMManager,
    "MD_ApplyTPG": MD_ApplyTPG,
    "MD_AudioAutoEQ": MD_AudioAutoEQ,
    "MD_AutoMasterNode": MD_AutoMasterNode,
    "MasteringChainNode": MasteringChainNode,
    "NoiseDecayScheduler_Custom": NoiseDecayScheduler_Custom,
    "PingPongSamplerNodeFBG": PingPongSamplerNodeFBG,
    "PingPongSampler_Custom_Lite": PingPongSamplerNode,
    "SceneGeniusAutocreator": SceneGeniusAutocreator,
    "SigmaConcatenate": SigmaConcatenate,
    "SigmaSmooth": SigmaSmooth,
    "SmartColorPaletteManager": SmartColorPaletteManager,
    "SmartFilenameBuilder": SmartFilenameBuilder,
    "TextFileLoader": TextFileLoader,
    "UniversalRoutingHubAdvanced": UniversalRoutingHubAdvanced,
    "WildcardPromptBuilder": WildcardPromptBuilder,
    "WorkflowSectionOrganizer": WorkflowSectionOrganizer
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ACE_LatentVisualizer": "MD: ACE Latent Visualizer",
    "APGGuiderForked": "MD: APG Guider",
    "AceT5ConditioningAnalyzer": "MD: ACE T5 Conditioning Analyzer",
    "AceT5ConditioningScheduled": "MD: ACE T5 Conditioning Scheduled",
    "AceT5ModelLoader": "MD: ACE T5 Model Loader",
    "AdvancedAudioPreviewAndSave": "MD: Advanced Audio Preview & Save",
    "AdvancedMediaSave": "MD: Advanced Media Save",
    "AdvancedTextNode": "MD: Advanced Text Input",
    "AutoLayoutOptimizer": "MD: Auto-Layout Optimizer",
    "EnhancedAnnotationNode": "MD: Enhanced Annotation",
    "EnhancedSeedSaver": "MD: Enhanced Seed Saver",
    "FilenameCounterNode": "MD: Filename Counter",
    "FilenameTokenReplacer": "MD: Filename Token Replacer",
    "GPUTemperatureProtectionEnhanced": "MD: GPU Temp Protect (Enhanced)",
    "HybridAdaptiveSigmas": "Hybrid Sigma Scheduler",
    "LLMVRAMManager": "MD: LLM VRAM Manager",
    "MD_ApplyTPG": "MD: Apply TPG (Token Perturbation)",
    "MD_AudioAutoEQ": "MD: Audio Auto EQ",
    "MD_AutoMasterNode": "MD: Audio Auto Master Pro",
    "MasteringChainNode": "MD: Mastering Chain",
    "NoiseDecayScheduler_Custom": "MD: Noise Decay Scheduler (Advanced)",
    "PingPongSamplerNodeFBG": "MD: PingPong Sampler (FBG)",
    "PingPongSampler_Custom_Lite": "MD: PingPong Sampler (Lite+)",
    "SceneGeniusAutocreator": "MD: Scene Genius Autocreator",
    "SigmaConcatenate": "Sigma Concatenate",
    "SigmaSmooth": "Sigma Smooth",
    "SmartColorPaletteManager": "MD: Smart Color Palette Manager",
    "SmartFilenameBuilder": "MD: Smart Filename Builder",
    "TextFileLoader": "MD: Text File Loader",
    "UniversalRoutingHubAdvanced": "MD: Universal Routing Hub",
    "WildcardPromptBuilder": "MD: Wildcard Prompt Builder",
    "WorkflowSectionOrganizer": "MD: Workflow Section Organizer"
}

# --- Private / Testing Node Imports ---
# These will only load if the 'testing' folder exists locally.
try:
    from .testing.Aesthetic_VLM_Pack.Aesthetic_VLM_Pack_Node1_Transformer import Aesthetic_Transformer_Node
    from .testing.Aesthetic_VLM_Pack.Aesthetic_VLM_Pack_Node2_ConfirmationGate import VLM_Confirmation_Gate_Node
    from .testing.Aesthetic_VLM_Pack.Aesthetic_VLM_Pack_Node3_QwenReplacer import MD_Qwen_Image_Edit_Object_Replacer_Node
    from .testing.MD_CLIPTokenFinder import MD_CLIPTokenFinder
    from .testing.MD_InitNoiseOptimizer import MD_InitNoiseOptimizer
    from .testing.MD_SelfCrossGuider import MD_SelfCrossGuider

    # Update Mappings if import succeeds
    NODE_CLASS_MAPPINGS.update({
        "Aesthetic_Transformer_Node_MD": Aesthetic_Transformer_Node,
        "MD_CLIPTokenFinder": MD_CLIPTokenFinder,
        "MD_InitNoiseOptimizer": MD_InitNoiseOptimizer,
        "MD_Qwen_Image_Edit_Object_Replacer_Node": MD_Qwen_Image_Edit_Object_Replacer_Node,
        "MD_SelfCrossGuider": MD_SelfCrossGuider,
        "VLM_Confirmation_Gate_Node_MD": VLM_Confirmation_Gate_Node
    })

    NODE_DISPLAY_NAME_MAPPINGS.update({
    "Aesthetic_Transformer_Node_MD": "MD: Aesthetic Transformer",
    "MD_CLIPTokenFinder": "MD: CLIP Token Finder (DIAGNOSTIC)",
    "MD_InitNoiseOptimizer": "MD: Init Noise Optimizer",
    "MD_Qwen_Image_Edit_Object_Replacer_Node": "MD: Qwen Object Replacer",
    "MD_SelfCrossGuider": "MD: Self-Cross Separation Guider",
    "VLM_Confirmation_Gate_Node_MD": "MD: VLM Confirmation Gate"
})
except ImportError:
    pass  # Testing folder missing (standard user install), skip these nodes.

# WEB_DIRECTORY = "./web"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

print(f"[ComfyUI_MD_Nodes] Initialized ({len(NODE_CLASS_MAPPINGS)} nodes)")
