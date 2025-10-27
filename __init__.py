"""
ComfyUI Custom Nodes - Auto-generated __init__.py
Generated: 2025-10-26 01:46:50
"""
import os
import sys
import importlib.util

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
if PACKAGE_DIR not in sys.path:
  sys.path.insert(0, PACKAGE_DIR)

# --- Node Imports ---
from .APG_Guider_Forked import APGGuiderNode
from .AdvancedMediaSave import AdvancedMediaSave
from .Hybrid_Sigma_Scheduler import HybridAdaptiveSigmas
from .NoiseDecayScheduler_Custom import NoiseDecayScheduler_Custom
from .PingPongSampler_Custom import PingPongSamplerNode
from .PingPongSampler_Custom_FBG import PingPongSamplerNodeFBG
from .SCENE_GENIUS_AUTOCREATOR import SceneGeniusAutocreator
from .audio.AdvancedAudioPreviewAndSave import AdvancedAudioPreviewAndSave
from .audio.MD_AutoMasterNode import MD_AutoMasterNode
from .audio.auto_eq import MD_AudioAutoEQ
from .audio.mastering_chain_node import MasteringChainNode
from .conditioning.ACE_T5_Conditioning_Node import AceT5ConditioningAnalyzer
from .conditioning.ACE_T5_Conditioning_Node import AceT5ConditioningScheduled
from .conditioning.ACE_T5_Conditioning_Node import AceT5ModelLoader
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
# --- End Node Imports ---

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
  "MD_AudioAutoEQ": MD_AudioAutoEQ,
  "MD_AutoMasterNode": MD_AutoMasterNode,
  "MasteringChainNode": MasteringChainNode,
  "NoiseDecayScheduler_Custom": NoiseDecayScheduler_Custom,
  "PingPongSamplerNodeFBG": PingPongSamplerNodeFBG,
  "PingPongSampler_Custom_Lite": PingPongSamplerNode,
  "SceneGeniusAutocreator": SceneGeniusAutocreator,
  "SmartColorPaletteManager": SmartColorPaletteManager,
  "SmartFilenameBuilder": SmartFilenameBuilder,
  "TextFileLoader": TextFileLoader,
  "UniversalRoutingHubAdvanced": UniversalRoutingHubAdvanced,
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
    "HybridAdaptiveSigmas": "MD: Hybrid Sigma Scheduler",
    "LLMVRAMManager": "MD: LLM VRAM Manager",
    "MD_AudioAutoEQ": "MD: Audio Auto EQ",
    "MD_AutoMasterNode": "MD: Audio Auto Master Pro",
    "MasteringChainNode": "MD: Mastering Chain",
    "NoiseDecayScheduler_Custom": "MD: Noise Decay Scheduler (Advanced)",
    "PingPongSamplerNodeFBG": "MD: PingPong Sampler (FBG)",
    "PingPongSampler_Custom_Lite": "MD: PingPong Sampler (Lite+)",
    "SceneGeniusAutocreator": "MD: Scene Genius Autocreator",
    "SmartColorPaletteManager": "MD: Smart Color Palette Manager",
    "SmartFilenameBuilder": "MD: Smart Filename Builder",
    "TextFileLoader": "MD: Text File Loader",
    "UniversalRoutingHubAdvanced": "MD: Universal Routing Hub",
    "WorkflowSectionOrganizer": "MD: Workflow Section Organizer"
}

# WEB_DIRECTORY = "./web"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

print(f"[ComfyUI_MD_Nodes] Initialized (28 nodes)")
