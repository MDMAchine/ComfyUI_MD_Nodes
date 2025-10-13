# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# ████ ACE T5 Conditioning Node v4.1 – Production Edition ████▓▒░
# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
# ░▒▓ CHANGELOG:
#    - v4.1 (Production Edition Update):
#        • FIXED: Capped `vocal_weight` at 0.95 to prevent distortion at 1.0+.
#        • IMPROVED: Blend logic now always uses normalization for stability.
#        • UPDATED: Tooltips now correctly guide users on `vocal_weight`.
#    - v4.0 (Production Edition):
#        • CLEANED: Removed all non-working experimental nodes
#        • WORKING: AceT5ConditioningScheduled with dual CLIP support
#        • WORKING: Native CLIP support for LoRA compatibility
#        • WORKING: Normalized merging prevents distortion
#        • WORKING: Proper conditioning_lyrics metadata preservation
#        • PRODUCTION READY and TESTED!
# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄

from comfy import sd1_clip
import comfy.text_encoders.t5
import comfy.utils
from .spiece_tokenizer import SPieceTokenizer
import os
import re
import torch
import logging
import copy
import folder_paths
from typing import List, Dict, Any, Optional, Tuple

from tokenizers import Tokenizer
from .ace_text_cleaners import multilingual_cleaners, japanese_to_romaji

# Safer import for built-in nodes
try:
    from nodes import ConditioningAverage
except ImportError:
    import nodes
    ConditioningAverage = nodes.NODE_CLASS_MAPPINGS.get("ConditioningAverage")
    if ConditioningAverage is None:
        raise ImportError("Could not import the ConditioningAverage node.")

# =================================================================================
# == Helper Classes & Dependencies                                               ==
# =================================================================================

SUPPORT_LANGUAGES = {"en": 259, "de": 260, "fr": 262, "es": 284, "it": 285, "pt": 286, "pl": 294, "tr": 295, "ru": 267, "cs": 293, "nl": 297, "ar": 5022, "zh": 5023, "ja": 5412, "hu": 5753, "ko": 6152, "hi": 6680}
structure_pattern = re.compile(r"\[.*?\]")

COMFY_BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
DEFAULT_VOCAB_FILE = os.path.join(COMFY_BASE, "comfy", "text_encoders", "ace_lyrics_tokenizer", "vocab.json")

class VoiceBpeTokenizer:
    START_TOKEN, END_TOKEN = 261, 2
    def __init__(self, vocab_file=DEFAULT_VOCAB_FILE):
        self.tokenizer = Tokenizer.from_file(vocab_file) if vocab_file and os.path.exists(vocab_file) else None
        if not self.tokenizer: logging.warning(f"VoiceBpeTokenizer vocab file not found: {vocab_file}")

    def encode(self, txt: str, lang: str = 'en') -> List[int]:
        if not self.tokenizer: return []
        txt = multilingual_cleaners(txt, lang)
        txt = f"[{'zh-cn' if lang == 'zh' else lang}]{txt}".replace(" ", "[SPACE]")
        return self.tokenizer.encode(txt).ids

    def __call__(self, string: str) -> Dict[str, Any]:
        lyric_token_idx = [self.START_TOKEN]
        for line in string.split("\n"):
            line = line.strip()
            if not line: lyric_token_idx.append(self.END_TOKEN); continue
            lang, line_text = ("en", line)
            if line.startswith("[") and line[3:4] == ']':
                lang_code = line[1:3].lower()
                if lang_code in SUPPORT_LANGUAGES: lang, line_text = lang_code, line[4:]
            try:
                romaji = japanese_to_romaji(line_text)
                if romaji != line_text: lang, line_text = "ja", romaji
            except Exception: pass
            try:
                token_idx = self.encode(line_text, "en" if structure_pattern.match(line_text) else lang)
                lyric_token_idx.extend(token_idx + [self.END_TOKEN])
            except Exception as e: logging.warning(f"Tokenize error {e} for line '{line_text}'")
        return {"input_ids": lyric_token_idx}
        
    @staticmethod
    def from_pretrained(path, **kwargs):
        return VoiceBpeTokenizer(path, **kwargs)

    def get_vocab(self):
        return {}

class UMT5BaseModel(sd1_clip.SDClipModel):
    def __init__(self, device="cpu", layer="last", layer_idx=None, dtype=None, model_options={}):
        json_config = os.path.join(COMFY_BASE, "comfy", "text_encoders", "umt5_config_base.json")
        super().__init__(device=device, layer=layer, layer_idx=layer_idx, textmodel_json_config=json_config, dtype=dtype, special_tokens={"end": 1, "pad": 0}, model_class=comfy.text_encoders.t5.T5, enable_attention_masks=True, zero_out_masked=False, model_options=model_options)

class UMT5BaseTokenizer(sd1_clip.SDTokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        spiece_path = tokenizer_data.get("spiece_model")
        if not spiece_path:
            spiece_path = os.path.join(os.path.dirname(__file__), "spiece.model")
        
        super().__init__(
            spiece_path, 
            pad_with_end=False, 
            embedding_size=768, 
            embedding_key='umt5base', 
            tokenizer_class=SPieceTokenizer, 
            has_start_token=False, 
            pad_to_max_length=False, 
            max_length=99999999, 
            min_length=1, 
            pad_token=0,
            end_token=1,
            tokenizer_data=tokenizer_data
        )
    
    def state_dict(self):
        return {"spiece_model": self.tokenizer.serialize_model()}

class LyricsTokenizer(sd1_clip.SDTokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        path = os.path.join(COMFY_BASE, "comfy", "text_encoders", "ace_lyrics_tokenizer", "vocab.json")
        super().__init__(path, pad_with_end=False, embedding_size=1024, embedding_key='lyrics', tokenizer_class=VoiceBpeTokenizer, has_start_token=True, pad_to_max_length=False, max_length=99999999, min_length=1, pad_token=2, has_end_token=False, tokenizer_data=tokenizer_data)

class AceT5Tokenizer:
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        self.voicebpe = LyricsTokenizer(embedding_directory, tokenizer_data)
        self.umt5base = UMT5BaseTokenizer(embedding_directory, tokenizer_data)
    
    def tokenize_with_weights(self, text:str, return_word_ids=False, **kwargs):
        return {
            "lyrics": self.voicebpe.tokenize_with_weights(kwargs.get("lyrics", ""), return_word_ids, **kwargs),
            "umt5base": self.umt5base.tokenize_with_weights(text, return_word_ids, **kwargs)
        }
        
    def untokenize(self, token_weight_pair):
        return self.umt5base.untokenize(token_weight_pair)
    
    def state_dict(self):
        return self.umt5base.state_dict()

# =================================================================================
# == Core Model Class                                                            ==
# =================================================================================

class AceT5Model(torch.nn.Module):
    def __init__(self, device="cpu", dtype=None, model_options={}):
        super().__init__()
        self.umt5base = UMT5BaseModel(device=device, dtype=dtype, model_options=model_options)
        self.dtypes = set()
        if dtype is not None:
            self.dtypes.add(dtype)

    def set_clip_options(self, options): 
        self.umt5base.set_clip_options(options)
    
    def reset_clip_options(self): 
        self.umt5base.reset_clip_options()
    
    def load_sd(self, sd): 
        return self.umt5base.load_sd(sd)

    def encode_token_weights(self, token_weight_pairs):
        t5_out, t5_pooled = self.umt5base.encode_token_weights(token_weight_pairs["umt5base"])
        lyrics_embeds = torch.tensor(list(map(lambda a: a[0], token_weight_pairs["lyrics"][0]))).unsqueeze(0)
        return t5_out, t5_pooled, {"conditioning_lyrics": lyrics_embeds}

    def encode_separate(self, token_weight_pairs):
        t5_out, t5_pooled = self.umt5base.encode_token_weights(token_weight_pairs["umt5base"])
        
        if t5_pooled is None:
            t5_pooled = torch.zeros(t5_out.shape[0], t5_out.shape[-1], device=t5_out.device, dtype=t5_out.dtype)
        
        lyrics_tokens = token_weight_pairs["lyrics"]
        lyrics_embeds = torch.tensor(list(map(lambda a: a[0], lyrics_tokens[0]))).unsqueeze(0) if lyrics_tokens and any(lyrics_tokens) else torch.zeros(1, 1, dtype=torch.long, device=t5_out.device)
        
        cond_tags = [[t5_out, {"pooled_output": t5_pooled, "conditioning_lyrics": lyrics_embeds}]]
        
        if not lyrics_tokens or not any(lyrics_tokens):
            lyrics_cond_tensor = torch.zeros_like(t5_out)
            lyrics_pooled = torch.zeros_like(t5_pooled)
        else:
            device = t5_out.device
            ids = torch.tensor([x[0] for x in lyrics_tokens[0]], device=device).unsqueeze(0)
            
            seq_len = t5_out.shape[1]
            if ids.shape[1] < seq_len:
                padding = torch.zeros(ids.shape[0], seq_len - ids.shape[1], device=device, dtype=ids.dtype)
                ids = torch.cat([ids, padding], dim=1)
            elif ids.shape[1] > seq_len:
                ids = ids[:, :seq_len]
            
            with torch.no_grad():
                embeds = self.umt5base.transformer.shared
                lyrics_cond_tensor = embeds(ids).to(device)
                
                num_tokens = min(len(lyrics_tokens[0]), lyrics_cond_tensor.shape[1])
                for i in range(num_tokens):
                    lyrics_cond_tensor[:, i] *= lyrics_tokens[0][i][1]
            
            lyrics_pooled = torch.zeros_like(t5_pooled)
        
        cond_lyrics = [[lyrics_cond_tensor, {"pooled_output": lyrics_pooled, "conditioning_lyrics": lyrics_embeds}]]
        return (cond_tags, cond_lyrics)

# =================================================================================
# == ComfyUI Custom Nodes                                                        ==
# =================================================================================

class AceT5ModelLoader:
    """
    Loads the ACE T5 model with proper tokenizer initialization.
    
    This is a custom CLIP loader specifically for ACE Step audio generation models.
    It provides the highest quality conditioning but does NOT support LoRAs.
    
    For LoRA support, use the native DualCLIPLoader instead (slightly lower quality).
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip_name": (folder_paths.get_filename_list("clip"), {
                    "tooltip": "Select your ACE T5 model file (.safetensors).\n\n"
                               "📁 Location: ComfyUI/models/clip/\n"
                               "📝 Example: umt5_with_tokenizer.safetensors\n\n"
                               "This custom loader provides:\n"
                               "✅ Best quality conditioning\n"
                               "✅ Proper lyrics token encoding\n"
                               "✅ Optimal for ACE Step audio models\n"
                               "❌ Does NOT support LoRAs\n\n"
                               "For LoRA support, use native DualCLIPLoader instead."
                })
            }
        }
    
    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip"
    CATEGORY = "MD_Nodes/Loaders"
    DESCRIPTION = "Loads ACE T5 models with custom tokenizers for optimal audio generation quality."

    def load_clip(self, clip_name):
        """Load ACE T5 CLIP model with custom tokenizer."""
        clip_path = folder_paths.get_full_path("clip", clip_name)
        if not clip_path:
            raise FileNotFoundError(f"ACE T5 model not found: {clip_name}")
        
        clip_data = comfy.utils.load_torch_file(clip_path, safe_load=True)
        
        class AceT5ClipWrapper:
            def __init__(self, model_data):
                self.cond_stage_model = AceT5Model()
                self.tokenizer = AceT5Tokenizer()
                self.cond_stage_model.load_sd(model_data)
                self.patcher = None
                self.layer_idx = None
                
            def tokenize(self, text, return_word_ids=False, **kwargs):
                return self.tokenizer.tokenize_with_weights(text, return_word_ids, **kwargs)
            
            def encode_from_tokens(self, tokens, return_pooled=False, return_dict=False):
                cond, pooled, extra = self.cond_stage_model.encode_token_weights(tokens)
                if return_dict:
                    return {"cond": cond, "pooled_output": pooled}
                if return_pooled:
                    return cond, pooled
                return cond
            
            def encode_separate(self, tokens):
                return self.cond_stage_model.encode_separate(tokens)
            
            def load_sd(self, sd):
                return self.cond_stage_model.load_sd(sd)
            
            def get_sd(self):
                return self.cond_stage_model.umt5base.state_dict()
            
            def clone(self):
                n = AceT5ClipWrapper.__new__(AceT5ClipWrapper)
                n.cond_stage_model = self.cond_stage_model
                n.tokenizer = self.tokenizer
                n.patcher = self.patcher
                n.layer_idx = self.layer_idx
                return n
            
            def set_clip_options(self, options):
                self.cond_stage_model.set_clip_options(options)
            
            def reset_clip_options(self):
                self.cond_stage_model.reset_clip_options()
        
        clip = AceT5ClipWrapper(clip_data)
        return (clip,)

class AceT5ConditioningScheduled:
    """
    Production-ready ACE T5 conditioning with vocal control.
    
    This node creates high-quality audio conditioning with separate control over:
    - Genre/style tags (base_tags)
    - Vocal characteristics (vocal_tags)
    - Lyrical content (lyrics)
    
    Works with BOTH:
    • Custom ACE T5 Model Loader (best quality, no LoRAs)
    • Native DualCLIPLoader (LoRA support, slightly lower quality)
    
    RECOMMENDED: Use 'vocal_cond' output for best results!
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP", {
                    "tooltip": "Connect ACE T5 CLIP model here.\n\n"
                               "Options:\n"
                               "1️⃣ ACE T5 Model Loader (recommended)\n"
                               "   • Best quality\n"
                               "   • Clean conditioning\n"
                               "   • No LoRA support\n\n"
                               "2️⃣ Native DualCLIPLoader\n"
                               "   • LoRA support!\n"
                               "   • Slightly lower quality\n"
                               "   • May need more steps (80-100)\n\n"
                               "Both work - choose based on your needs!"
                }),
                
                "base_tags": ("STRING", {
                    "multiline": True, 
                    "default": "dubstep, synthwave, minimal, melodic",
                    "tooltip": "🎸 GENRE & STYLE TAGS\n\n"
                               "Describe the overall sound, genre, mood, and instrumentation.\n"
                               "These tags form the foundation of your track.\n\n"
                               "Examples:\n"
                               "• 'trap, dark, heavy 808s, aggressive'\n"
                               "• 'ambient, ethereal, soft pads, peaceful'\n"
                               "• 'rock, electric guitar, energetic, drums'\n"
                               "• 'orchestral, epic, strings, brass'\n\n"
                               "💡 TIP: Use base_tag_boost=2 or 3 if genre isn't coming through!"
                }),
                
                "vocal_tags": ("STRING", {
                    "multiline": True, 
                    "default": "soft female vocals, breathy, reverb",
                    "tooltip": "🎤 VOCAL CHARACTERISTICS\n\n"
                               "Describe HOW the vocals should sound (timbre, processing, style).\n"
                               "This is separate from the actual words being sung.\n\n"
                               "Examples:\n"
                               "• 'aggressive male rap vocals, doubled, distorted'\n"
                               "• 'ethereal female choir, harmonized, cathedral reverb'\n"
                               "• 'breathy whisper vocals, intimate, close-mic'\n"
                               "• 'powerful opera vocals, vibrato, dramatic'\n\n"
                               "💡 TIP: Leave empty for instrumental tracks!"
                }),
                
                "lyrics": ("STRING", {
                    "multiline": True, 
                    "default": "[en]oh oh oh",
                    "tooltip": "🎵 LYRICAL CONTENT\n\n"
                               "The actual words/syllables to be sung.\n"
                               "ALWAYS start each line with a language code!\n\n"
                               "Supported languages:\n"
                               "[en] English  [ja] Japanese  [zh] Chinese\n"
                               "[de] German   [fr] French    [es] Spanish\n"
                               "[it] Italian  [pt] Portuguese [ru] Russian\n"
                               "[ko] Korean   [hi] Hindi     [ar] Arabic\n\n"
                               "Examples:\n"
                               "[en]never gonna give you up\n"
                               "[en]never gonna let you down\n\n"
                               "[ja]さくら さくら\n\n"
                               "⚠️ IMPORTANT: Keep lyrics SHORT (under 20 words) for best quality!\n"
                               "⚠️ Longer lyrics = more tokens = potential distortion"
                }),
            },
            "optional": {
                "base_tag_boost": ("INT", {
                    "default": 2, 
                    "min": 1, 
                    "max": 10,
                    "tooltip": "🔊 GENRE EMPHASIS\n\n"
                               "Repeats your base_tags to strengthen genre adherence.\n\n"
                               "Values:\n"
                               "• 1 = Normal (no repetition)\n"
                               "• 2 = RECOMMENDED (balanced)\n"
                               "• 3 = Strong genre emphasis\n"
                               "• 4-5 = Very strong (if genre not coming through)\n\n"
                               "💡 Use 2-3 for most tracks.\n"
                               "💡 Increase to 4-5 if output ignores your genre!"
                }),
                
                "vocal_tag_boost": ("INT", {
                    "default": 1, 
                    "min": 1, 
                    "max": 5,
                    "tooltip": "🎙️ VOCAL EMPHASIS\n\n"
                               "Repeats your vocal_tags to strengthen vocal characteristics.\n\n"
                               "Values:\n"
                               "• 1 = RECOMMENDED (normal)\n"
                               "• 2 = Stronger vocal characteristics\n"
                               "• 3+ = Very pronounced vocal style\n\n"
                               "💡 Usually keep at 1.\n"
                               "💡 Increase only if vocals are too subtle."
                }),
                
                "vocal_weight": ("FLOAT", {
                    "default": 0.5, 
                    "min": 0.0, 
                    "max": 0.95,
                    "step": 0.05,
                    "tooltip": "🎚️ VOCAL INFLUENCE\n\n"
                               "Controls lyrics influence (NOT volume).\n\n"
                               "Values:\n"
                               "• 0.0 = Tags only (but tags may describe vocals!)\n"
                               "• 0.3 = Subtle lyrical influence\n"
                               "• 0.5 = RECOMMENDED (balanced)\n"
                               "• 0.7 = Strong lyrical influence\n"
                               "• 0.9 = Maximum safe value\n\n"
                               "💡 For TRUE instrumental, use 'base_cond' output!\n"
                               "💡 This blends tag conditioning with lyrics embeddings"
                }),
            }
        }
    
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("base_cond", "vocal_cond", "tags_only_vocal")
    FUNCTION = "encode"
    CATEGORY = "MD_Nodes/Conditioning"
    DESCRIPTION = (
        "Creates ACE audio conditioning with separate genre, vocal, and lyrics control.\n\n"
        "OUTPUTS:\n"
        "• base_cond: Instrumental only (no vocals)\n"
        "• vocal_cond: ⭐ RECOMMENDED - Full track with vocals (USE THIS!)\n"
        "• tags_only_vocal: Alternative version (tags + lyrics metadata only)\n\n"
        "Connect 'vocal_cond' to your sampler's positive input for best results!"
    )
    
    def encode(self, clip, base_tags, vocal_tags, lyrics, base_tag_boost, vocal_tag_boost, vocal_weight, **kwargs):
        """Encode with support for both custom and native CLIP."""
        
        # Boost tags
        if base_tag_boost > 1:
            base_tags = ", ".join([base_tags] * base_tag_boost)
        if vocal_tag_boost > 1:
            vocal_tags = ", ".join([vocal_tags] * vocal_tag_boost)
        
        # Detect CLIP type
        has_encode_separate = hasattr(clip, 'encode_separate')
        
        if has_encode_separate:
            return self._encode_custom(clip, base_tags, vocal_tags, lyrics, vocal_weight)
        else:
            return self._encode_native(clip, base_tags, vocal_tags, lyrics, vocal_weight)
    
    def _encode_custom(self, clip, base_tags, vocal_tags, lyrics, vocal_weight):
        """Custom ACE T5 Model Loader (best quality)."""
        tokens_base = clip.tokenize(base_tags, lyrics="")
        base_cond, _ = clip.encode_separate(tokens_base)
        
        combined_tags = f"{base_tags}, {vocal_tags}"
        tokens_vocal = clip.tokenize(combined_tags, lyrics=lyrics)
        vocal_tags_cond, vocal_lyrics_cond = clip.encode_separate(tokens_vocal)
        
        vocal_tags_tensor, vocal_tags_extras = vocal_tags_cond[0]
        vocal_lyrics_tensor, vocal_lyrics_extras = vocal_lyrics_cond[0]
        
        original_conditioning_lyrics = vocal_tags_extras.get("conditioning_lyrics", torch.tensor([[261]]))
        
        # Tags only version
        vocal_cond_simple = [[vocal_tags_tensor, {
            "pooled_output": vocal_tags_extras["pooled_output"],
            "conditioning_lyrics": original_conditioning_lyrics
        }]]
        
        # Clamp vocal_weight to safe range
        vocal_weight = max(0.0, min(vocal_weight, 0.95))  # Cap at 0.95 instead of allowing 1.0+
        
        if vocal_weight == 0:
            vocal_cond_merged = vocal_cond_simple
        else:
            # Always blend, never use pure lyrics
            tags_norm = torch.norm(vocal_tags_tensor)
            lyrics_norm = torch.norm(vocal_lyrics_tensor)
            
            if lyrics_norm > 0:
                lyrics_scaled = vocal_lyrics_tensor * (tags_norm / lyrics_norm)
            else:
                lyrics_scaled = vocal_lyrics_tensor
            
            # alpha ranges from 0 to ~0.49 (never reaches 0.5+)
            alpha = vocal_weight / (1.0 + vocal_weight)
            merged_tensor = vocal_tags_tensor * (1 - alpha) + lyrics_scaled * alpha
            merged_pooled = vocal_tags_extras["pooled_output"] * (1 - alpha) + vocal_lyrics_extras["pooled_output"] * alpha
            
            merged_extras = {
                "pooled_output": merged_pooled,
                "conditioning_lyrics": original_conditioning_lyrics
            }
            
            vocal_cond_merged = [[merged_tensor, merged_extras]]
        
        return (base_cond, vocal_cond_merged, vocal_cond_simple)
    
    def _encode_native(self, clip, base_tags, vocal_tags, lyrics, vocal_weight):
        """Native ComfyUI CLIP (enables LoRA support)."""
        tokens_base = clip.tokenize(base_tags, lyrics="")
        cond_base_tensor, pooled_base = clip.encode_from_tokens(tokens_base, return_pooled=True)
        
        tokens_for_lyrics = clip.tokenize(base_tags, lyrics=lyrics)
        if hasattr(clip, 'cond_stage_model') and hasattr(clip.cond_stage_model, 'encode_token_weights'):
            _, _, extra = clip.cond_stage_model.encode_token_weights(tokens_for_lyrics)
            original_conditioning_lyrics = extra.get("conditioning_lyrics", torch.tensor([[261]]))
        else:
            original_conditioning_lyrics = torch.tensor([[261]])
        
        base_cond = [[cond_base_tensor, {
            "pooled_output": pooled_base if pooled_base is not None else torch.zeros(1, 768),
            "conditioning_lyrics": original_conditioning_lyrics
        }]]
        
        combined_tags = f"{base_tags}, {vocal_tags}"
        tokens_vocal = clip.tokenize(combined_tags, lyrics=lyrics)
        cond_vocal_tensor, pooled_vocal = clip.encode_from_tokens(tokens_vocal, return_pooled=True)
        
        vocal_cond = [[cond_vocal_tensor, {
            "pooled_output": pooled_vocal if pooled_vocal is not None else torch.zeros(1, 768),
            "conditioning_lyrics": original_conditioning_lyrics
        }]]
        
        return (base_cond, vocal_cond, vocal_cond)

class AceT5ConditioningAnalyzer:
    """
    Debugging tool to inspect conditioning tensors.
    
    Connect any conditioning output to see detailed information about:
    - Tensor shape and statistics
    - Lyrics token count
    - Conditioning metadata
    - Potential quality issues
    
    Useful for troubleshooting distortion or checking if lyrics are properly encoded.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING", {
                    "tooltip": "🔍 CONDITIONING ANALYZER\n\n"
                               "Connect ANY conditioning output here to inspect:\n\n"
                               "✅ What it checks:\n"
                               "• Tensor dimensions & data type\n"
                               "• Mean, Std, Min, Max values\n"
                               "• Lyrics token count (should be 400+)\n"
                               "• Pooled output status\n"
                               "• Timestep range (if set)\n\n"
                               "⚠️ Look for:\n"
                               "• Std > 1.0 = May have distortion\n"
                               "• Lyrics tokens = [261] = No lyrics encoded!\n"
                               "• Extreme Min/Max values = Potential artifacts\n\n"
                               "💡 Good values:\n"
                               "• Std: 0.15-0.25\n"
                               "• Min/Max: -2 to +2\n"
                               "• Lyrics: 400+ tokens"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("analysis",)
    FUNCTION = "analyze"
    CATEGORY = "MD_Nodes/Conditioning"
    OUTPUT_NODE = True
    DESCRIPTION = "Analyzes conditioning tensors to help debug quality issues and verify proper encoding."
    
    def analyze(self, conditioning):
        tensor, extras = conditioning[0]
        
        analysis = []
        analysis.append("=== ACE T5 Conditioning Analysis ===\n")
        analysis.append(f"Tensor Shape: {tensor.shape}")
        analysis.append(f"Tensor Device: {tensor.device}")
        analysis.append(f"Tensor Dtype: {tensor.dtype}")
        analysis.append(f"Tensor Mean: {tensor.mean().item():.6f}")
        analysis.append(f"Tensor Std: {tensor.std().item():.6f}")
        analysis.append(f"Tensor Min: {tensor.min().item():.6f}")
        analysis.append(f"Tensor Max: {tensor.max().item():.6f}")
        
        # Quality assessment
        std_val = tensor.std().item()
        min_val = tensor.min().item()
        max_val = tensor.max().item()
        
        analysis.append("\n=== Quality Assessment ===")
        if std_val < 0.1:
            analysis.append("⚠️  Std too low - may lack detail")
        elif std_val > 1.0:
            analysis.append("⚠️  Std too high - may have distortion!")
        else:
            analysis.append("✅ Std in good range (0.1-1.0)")
        
        if abs(min_val) > 5 or abs(max_val) > 5:
            analysis.append("⚠️  Extreme values detected - potential artifacts!")
        else:
            analysis.append("✅ Value range looks good")
        
        analysis.append("\n=== Extras Dictionary ===")
        for key, value in extras.items():
            if isinstance(value, torch.Tensor):
                analysis.append(f"{key}: Tensor {value.shape}")
            else:
                analysis.append(f"{key}: {type(value).__name__}")
        
        if "conditioning_lyrics" in extras:
            lyrics = extras["conditioning_lyrics"]
            analysis.append(f"\n=== Lyrics Analysis ===")
            analysis.append(f"Lyrics Tokens Shape: {lyrics.shape}")
            
            try:
                lyrics_list = lyrics.squeeze().tolist()
                if isinstance(lyrics_list, (int, float)):
                    lyrics_list = [lyrics_list]
                    
                token_count = len(lyrics_list)
                display_tokens = lyrics_list[:20]
                
                analysis.append(f"Token Count: {token_count}")
                analysis.append(f"First 20 Tokens: {display_tokens}...")
                
                # Lyrics quality check
                if token_count == 1 and lyrics_list[0] == 261:
                    analysis.append("⚠️  WARNING: Only START token! No lyrics encoded!")
                elif token_count < 50:
                    analysis.append("✅ Short lyrics (good for quality)")
                elif token_count < 200:
                    analysis.append("✅ Medium lyrics length")
                elif token_count < 400:
                    analysis.append("⚠️  Long lyrics - may affect quality")
                else:
                    analysis.append("⚠️  Very long lyrics - consider shortening!")
                    
            except Exception as e:
                analysis.append(f"⚠️  Could not parse lyrics: {e}")
        else:
            analysis.append("\n⚠️  No conditioning_lyrics found in metadata!")
        
        if "pooled_output" in extras:
            pooled = extras["pooled_output"]
            analysis.append(f"\n=== Pooled Output ===")
            analysis.append(f"Shape: {pooled.shape}")
            analysis.append(f"Mean: {pooled.mean().item():.6f}")
            
            if pooled.mean().item() == 0:
                analysis.append("ℹ️  Pooled is zeros (normal for T5 models)")
        
        if "start_percent" in extras:
            analysis.append(f"\n=== Timestep Scheduling ===")
            analysis.append(f"Active Range: {extras['start_percent']:.2f} - {extras['end_percent']:.2f}")
            analysis.append(f"({extras['start_percent']*100:.0f}% to {extras['end_percent']*100:.0f}% of generation)")
        
        return ("\n".join(analysis),)

# =================================================================================
# == Node Registration                                                           ==
# =================================================================================

NODE_CLASS_MAPPINGS = {
    "AceT5ModelLoader": AceT5ModelLoader,
    "AceT5ConditioningScheduled": AceT5ConditioningScheduled,
    "AceT5ConditioningAnalyzer": AceT5ConditioningAnalyzer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AceT5ModelLoader": "ACE T5 Model Loader 🎵",
    "AceT5ConditioningScheduled": "ACE T5 Conditioning ⚡",
    "AceT5ConditioningAnalyzer": "ACE T5 Analyzer 🔍",
}