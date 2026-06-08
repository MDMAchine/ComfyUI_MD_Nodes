"""
ComfyUI Custom Nodes - Auto-generated __init__.py
Generated: 2026-06-07 23:51:45
"""
# Copyright (C) 2026 Alexander Allan (MDMAchine) | A&E Concepts
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
VERSION = "v1.0.0"  # UPS v1.5.8


import os
import sys
import importlib.util

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
if PACKAGE_DIR not in sys.path:
  sys.path.insert(0, PACKAGE_DIR)

# --- Public Node Imports ---
from .ACE_Engine.MD_ACE_SigmaDenoisePatcher import MD_ACE_SigmaDenoisePatcher
from .ACE_Engine.MD_AceStepInpaint import MD_ACE_LatentInpaintMask
from .audio.MD_AdvancedAudioPreviewAndSave import AdvancedAudioPreviewAndSave
from .audio.MD_AudioAutoEQ import MD_AudioAutoEQ_Adaptive
from .audio.MD_AudioSimpleEditor import MD_AudioSimpleEditor
from .audio.MD_AutoMasterNode import MD_AutoMasterNode
from .audio.MD_BroadcastTools import AudioSpectrumAnalyzer_Enhanced
from .audio.MD_BroadcastTools import AudioSpectrumVisualizer
from .audio.MD_BroadcastTools import MD_LUFS_Normalizer
from .audio.MD_BroadcastTools import MD_Stereo_Width_Controller
from .audio.MD_MasteringChain import MasteringChainNode
from .audio.MD_MasteringChain import MasteringCompressorNode
from .audio.MD_MasteringChain import MasteringEQNode
from .audio.MD_MasteringChain import MasteringGainNode
from .audio.MD_MasteringChain import MasteringLimiterNode
from .guiders.MD_APGGuiderForked import APGGuiderNode
from .latent.MD_LatentVisualizer import ACELatentVisualizer
from .loaders.MD_ACE_XLLoader import MD_ACE_XLLatentProcessor
from .loaders.MD_ACE_XLLoader import MD_ACE_XLLoader
from .logic.MD_Logic_Switch import MD_AnySwitch
from .logic.MD_Logic_Switch import MD_MultiSwitch
from .logic.MD_String_Logic import MD_String_Logic
from .lora.MD_DynamicLoRAStacker import MD_DynamicLoRAStacker
from .maintenance.MD_LatentSanitizer import MD_LatentSanitizer
from .maintenance.MD_ModelStateReset import MD_ModelStateReset
from .masking.MD_Latent_Time_Mask import MD_Latent_Time_Mask
from .math.MD_BasicMath import MD_Math_Add
from .math.MD_BasicMath import MD_Math_Divide
from .math.MD_BasicMath import MD_Math_Multiply
from .math.MD_BasicMath import MD_Math_Subtract
from .modulation.MD_LFO_Generator import MD_LFO_Generator
from .noise.MD_CustomNoiseGenerator import MD_CustomNoiseGenerator
from .noise.MD_CustomNoiseGenerator import MD_MultiNoiseBlender
from .optimization.MD_ApplyTPG import MD_ApplyTPG
from .optimization.MD_FSampler import FSampler
from .samplers.MD_AMED_Sampler import MD_AMED_Sampler
from .samplers.MD_PingPongSamplerFBG_Legacy import PingPongSamplerNodeBasic
from .samplers.MD_PingPongSamplerFBG_Legacy import PingPongSamplerNodeFBG
from .samplers.MD_PingPongSamplerFBG_Legacy import PingPongSamplerNodeLite
from .save.MD_AdvancedMediaSave import AdvancedMediaSave
from .save.MD_SeedSaver import EnhancedSeedSaverNode
from .schedulers.MD_GITS_Scheduler import MD_GITS_Scheduler
from .schedulers.MD_HybridSigmaScheduler import HybridAdaptiveSigmas_Advanced
from .schedulers.MD_HybridSigmaScheduler import HybridAdaptiveSigmas_Basic
from .schedulers.MD_HybridSigmaScheduler import HybridAdaptiveSigmas_Lite
from .schedulers.MD_HybridSigmaScheduler import SigmaConcatenate
from .schedulers.MD_HybridSigmaScheduler import SigmaSmooth
from .schedulers.MD_NoiseDecayScheduler import NoiseDecayScheduler_Custom
from .text.MD_TextInput import AdvancedTextNode
from .text.MD_TextInput import TextFileLoader
from .utility.MD_AdvancedSeedGenerator import MD_AdvancedSeedGenerator
from .utility.MD_ConditioningCacheNodes import MD_LoadConditioning
from .utility.MD_ConditioningCacheNodes import MD_SaveConditioning
from .utility.MD_EmptyLatentRatioSelector import MD_EmptyLatentRatioSelector
from .utility.MD_GPUTemperatureProtection import GPUTemperatureProtectionEnhanced
from .utility.MD_GlobalUpdateManager import MD_GlobalUpdateManager
from .utility.MD_GuardianSuite import MD_Audio_Guardian
from .utility.MD_GuardianSuite import MD_Image_Guardian
from .utility.MD_GuardianSuite import MD_NaN_Guardian
from .utility.MD_LLM_VRAMManager import LLMVRAMManager
from .utility.MD_RepoMaintenance import MD_RepoMaintenance
from .utility.MD_SceneGeniusAutocreator import MD_WorkflowContextBus
from .utility.MD_SceneGeniusAutocreator import SceneGeniusAutocreator
from .utility.MD_SmartFilenameBuilder import FilenameCounterNode
from .utility.MD_SmartFilenameBuilder import FilenameTokenReplacer
from .utility.MD_SmartFilenameBuilder import SmartFilenameBuilder
from .utility.MD_UniversalWildcardOrchestrator import UniversalWildcardOrchestrator
from .utility.MD_VRAMCanary import MD_VRAMCanary
from .utility.MD_WildcardPromptBuilder import WildcardPromptBuilder
from .utility.MD_YAML_Node import MD_YAML_Generator
from .utility.MD_YAML_Utils import MD_YAML_Utils

# --- Main Mapping (Public Nodes) ---
NODE_CLASS_MAPPINGS = {
    "ACE_LatentVisualizer": ACELatentVisualizer,
    "APGGuiderForked": APGGuiderNode,
    "AdvancedAudioPreviewAndSave": AdvancedAudioPreviewAndSave,
    "AdvancedMediaSave": AdvancedMediaSave,
    "AdvancedTextNode": AdvancedTextNode,
    "EnhancedSeedSaver": EnhancedSeedSaverNode,
    "FSampler": FSampler,
    "FilenameCounterNode": FilenameCounterNode,
    "FilenameTokenReplacer": FilenameTokenReplacer,
    "GPUTemperatureProtectionEnhanced": GPUTemperatureProtectionEnhanced,
    "HybridAdaptiveSigmas": HybridAdaptiveSigmas_Advanced,
    "HybridAdaptiveSigmas_Basic": HybridAdaptiveSigmas_Basic,
    "HybridAdaptiveSigmas_Lite": HybridAdaptiveSigmas_Lite,
    "LLMVRAMManager": LLMVRAMManager,
    "MD_ACE_LatentInpaintMask": MD_ACE_LatentInpaintMask,
    "MD_ACE_SigmaDenoisePatcher": MD_ACE_SigmaDenoisePatcher,
    "MD_ACE_XLLatentProcessor": MD_ACE_XLLatentProcessor,
    "MD_ACE_XLLoader": MD_ACE_XLLoader,
    "MD_AMED_Sampler": MD_AMED_Sampler,
    "MD_AdvancedSeedGenerator": MD_AdvancedSeedGenerator,
    "MD_AnySwitch": MD_AnySwitch,
    "MD_ApplyTPG": MD_ApplyTPG,
    "MD_AudioAutoEQ_Adaptive": MD_AudioAutoEQ_Adaptive,
    "MD_AudioSimpleEditor": MD_AudioSimpleEditor,
    "MD_Audio_Guardian": MD_Audio_Guardian,
    "MD_Audio_Spectrum_Analyzer_Enhanced": AudioSpectrumAnalyzer_Enhanced,
    "MD_Audio_Spectrum_Visualizer": AudioSpectrumVisualizer,
    "MD_AutoMasterNode": MD_AutoMasterNode,
    "MD_CustomNoiseGenerator": MD_CustomNoiseGenerator,
    "MD_DynamicLoRAStacker": MD_DynamicLoRAStacker,
    "MD_EmptyLatentRatioSelector": MD_EmptyLatentRatioSelector,
    "MD_GITS_Scheduler": MD_GITS_Scheduler,
    "MD_GlobalUpdateManager": MD_GlobalUpdateManager,
    "MD_Image_Guardian": MD_Image_Guardian,
    "MD_LFO_Generator": MD_LFO_Generator,
    "MD_LUFS_Normalizer": MD_LUFS_Normalizer,
    "MD_LatentSanitizer": MD_LatentSanitizer,
    "MD_Latent_Time_Mask": MD_Latent_Time_Mask,
    "MD_LoadConditioning": MD_LoadConditioning,
    "MD_Mastering_Compressor": MasteringCompressorNode,
    "MD_Mastering_EQ": MasteringEQNode,
    "MD_Mastering_Gain": MasteringGainNode,
    "MD_Mastering_Limiter": MasteringLimiterNode,
    "MD_Math_Add": MD_Math_Add,
    "MD_Math_Divide": MD_Math_Divide,
    "MD_Math_Multiply": MD_Math_Multiply,
    "MD_Math_Subtract": MD_Math_Subtract,
    "MD_ModelStateReset": MD_ModelStateReset,
    "MD_MultiNoiseBlender": MD_MultiNoiseBlender,
    "MD_MultiSwitch": MD_MultiSwitch,
    "MD_NaN_Guardian": MD_NaN_Guardian,
    "MD_RepoMaintenance": MD_RepoMaintenance,
    "MD_SaveConditioning": MD_SaveConditioning,
    "MD_Stereo_Width_Controller": MD_Stereo_Width_Controller,
    "MD_String_Logic": MD_String_Logic,
    "MD_VRAMCanary": MD_VRAMCanary,
    "MD_WorkflowContextBus": MD_WorkflowContextBus,
    "MD_YAML_Generator": MD_YAML_Generator,
    "MD_YAML_Utils": MD_YAML_Utils,
    "MasteringChainNode": MasteringChainNode,
    "NoiseDecayScheduler_Custom": NoiseDecayScheduler_Custom,
    "PingPongSamplerNodeBasic": PingPongSamplerNodeBasic,
    "PingPongSamplerNodeFBG": PingPongSamplerNodeFBG,
    "PingPongSamplerNodeLite": PingPongSamplerNodeLite,
    "SceneGeniusAutocreator": SceneGeniusAutocreator,
    "SigmaConcatenate": SigmaConcatenate,
    "SigmaSmooth": SigmaSmooth,
    "SmartFilenameBuilder": SmartFilenameBuilder,
    "TextFileLoader": TextFileLoader,
    "UniversalWildcardOrchestrator": UniversalWildcardOrchestrator,
    "WildcardPromptBuilder": WildcardPromptBuilder
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ACE_LatentVisualizer": "MD: Latent Visualizer",
    "APGGuiderForked": "MD: APG Guider",
    "AdvancedAudioPreviewAndSave": "MD: Advanced Audio Preview & Save",
    "AdvancedMediaSave": "MD: Advanced Media Save",
    "AdvancedTextNode": "MD: Advanced Text Input",
    "EnhancedSeedSaver": "MD: Enhanced Seed Saver",
    "FSampler": "MD: FSampler (Speed Patcher)",
    "FilenameCounterNode": "MD: Filename Counter",
    "FilenameTokenReplacer": "MD: Filename Token Replacer",
    "GPUTemperatureProtectionEnhanced": "MD: GPU Temp Protect",
    "HybridAdaptiveSigmas": "MD: Hybrid Scheduler (Advanced)",
    "HybridAdaptiveSigmas_Basic": "MD: Hybrid Scheduler (Basic)",
    "HybridAdaptiveSigmas_Lite": "MD: Hybrid Scheduler (Lite)",
    "LLMVRAMManager": "MD: LLM VRAM Manager",
    "MD_ACE_LatentInpaintMask": "MD: AceStep Audio Generative Fill \ud83d\udd8c\ufe0f",
    "MD_ACE_SigmaDenoisePatcher": "MD: ACE Sigma Denoise Patcher \u2702\ufe0f",
    "MD_ACE_XLLatentProcessor": "MD: ACE-Step XL Latent Processor \ud83c\udf9b\ufe0f",
    "MD_ACE_XLLoader": "MD: ACE-Step XL Loader \ud83c\udfb5",
    "MD_AMED_Sampler": "MD: AMED Solver (Corrected Euler)",
    "MD_AdvancedSeedGenerator": "MD: Advanced Seed Generator",
    "MD_AnySwitch": "MD: Any Switch (Boolean)",
    "MD_ApplyTPG": "MD: Apply TPG (Token Perturbation)",
    "MD_AudioAutoEQ_Adaptive": "MD: Audio Auto EQ (Adaptive)",
    "MD_AudioSimpleEditor": "MD: Audio Simple Editor \u2702\ufe0f",
    "MD_Audio_Guardian": "MD: Audio Guardian",
    "MD_Audio_Spectrum_Analyzer_Enhanced": "MD: Audio Analyzer (Report + LUFS)",
    "MD_Audio_Spectrum_Visualizer": "MD: Audio Spectrum Visualizer (Plot)",
    "MD_AutoMasterNode": "MD: Audio Auto Master Pro",
    "MD_CustomNoiseGenerator": "MD: Custom Noise Generator",
    "MD_DynamicLoRAStacker": "MD: Dynamic LoRA Stacker (Style Butler)",
    "MD_EmptyLatentRatioSelector": "MD: Empty Latent Ratio Select",
    "MD_GITS_Scheduler": "MD: GITS Scheduler (Boomerang)",
    "MD_GlobalUpdateManager": "MD: Global Update Architect",
    "MD_Image_Guardian": "MD: Image Guardian",
    "MD_LFO_Generator": "MD: LFO Generator (Automator)",
    "MD_LUFS_Normalizer": "MD: LUFS Normalizer",
    "MD_LatentSanitizer": "MD: Universal Latent Sanitizer (Audio/Video/Image)",
    "MD_Latent_Time_Mask": "MD: Latent Time Mask (Timeline Director)",
    "MD_LoadConditioning": "MD: Load Conditioning \ud83d\udcc2",
    "MD_Mastering_Compressor": "MD: Mastering Compressor",
    "MD_Mastering_EQ": "MD: Mastering EQ",
    "MD_Mastering_Gain": "MD: Mastering Gain",
    "MD_Mastering_Limiter": "MD: Mastering Limiter",
    "MD_Math_Add": "MD: Math Add (Int/Float)",
    "MD_Math_Divide": "MD: Math Divide (Int/Float)",
    "MD_Math_Multiply": "MD: Math Multiply (Int/Float)",
    "MD_Math_Subtract": "MD: Math Subtract (Int/Float)",
    "MD_ModelStateReset": "MD: Model State Reset (Anti-Static)",
    "MD_MultiNoiseBlender": "MD: Noise Blender (5-Layer)",
    "MD_MultiSwitch": "MD: Multi-Way Switch (5-Path)",
    "MD_NaN_Guardian": "MD: NaN Guardian",
    "MD_RepoMaintenance": "MD: Repo Fortress",
    "MD_SaveConditioning": "MD: Save Conditioning \ud83d\udcbe",
    "MD_Stereo_Width_Controller": "MD: Stereo Width Controller",
    "MD_String_Logic": "MD: String Logic (Router)",
    "MD_VRAMCanary": "MD: VRAM Canary (Memory Guardian)",
    "MD_WorkflowContextBus": "MD: Universal Context Bus",
    "MD_YAML_Generator": "MD: YAML Configuration Tool",
    "MD_YAML_Utils": "MD: YAML Utils (Architect)",
    "MasteringChainNode": "MD: Mastering Chain (Full)",
    "NoiseDecayScheduler_Custom": "MD: Noise Decay Scheduler (Advanced)",
    "PingPongSamplerNodeBasic": "MD: PingPong Basic (Presets)",
    "PingPongSamplerNodeFBG": "MD: PingPong FBG (Full Control)",
    "PingPongSamplerNodeLite": "MD: PingPong Lite (Classic)",
    "SceneGeniusAutocreator": "MD: Scene Genius Autocreator",
    "SigmaConcatenate": "MD: Sigma Concatenate",
    "SigmaSmooth": "MD: Sigma Smooth",
    "SmartFilenameBuilder": "MD: Smart Filename Builder",
    "TextFileLoader": "MD: Text File Loader",
    "UniversalWildcardOrchestrator": "MD: Universal Wildcard Orchestrator",
    "WildcardPromptBuilder": "MD: Wildcard Prompt Builder"
}

# --- Private / Testing Node Imports ---
try:
    

    NODE_CLASS_MAPPINGS.update({
    
    })
    NODE_DISPLAY_NAME_MAPPINGS.update({})
except ImportError:
    pass  

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

print(f"[ComfyUI_MD_Nodes] Initialized ({len(NODE_CLASS_MAPPINGS)} nodes)")


# ==============================================================================
# == Unit Tests (smoke — runs without ComfyUI)
# ==============================================================================

if __name__ == "__main__":
    print("\n🧪 Smoke tests: __init__")
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

    _check("VERSION defined",    VERSION == "v1.0.0")
    _check("NODE_CLASS_MAPPINGS is dict",
           isinstance(NODE_CLASS_MAPPINGS, dict))
    _check("NODE_CLASS_MAPPINGS not empty",
           len(NODE_CLASS_MAPPINGS) > 0)

    print(f"\n  {_pass} passed, {_fail} failed")
    if _fail == 0:
        print("  🎉 All good.")
