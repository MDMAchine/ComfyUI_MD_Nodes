# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
# ████ MD_Nodes/AceT5Conditioning – T5 Conditioning Suite v4.1 ████▓▒░
# © 2025 MDMAchine
# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
# ░▒▓ ORIGIN & DEV:
#   • Cast into the void by: MDMAchine
#   • Enhanced by: Gemini, Claude, etc.
#   • License: Apache 2.0 — Sharing is caring

# ░▒▓ DESCRIPTION:
#   A suite of nodes for advanced T5-based conditioning in audio generation.
#   Includes a custom model loader (`AceT5ModelLoader`), a powerful conditioning
#   blender (`AceT5ConditioningScheduled`), and a debugging tool
#   (`AceT5ConditioningAnalyzer`).

# ░▒▓ FEATURES:
#   ✓ `AceT5ModelLoader`: Loads T5 models with custom tokenizers (Best Quality).
#   ✓ `AceT5ConditioningScheduled`: Blends base tags, vocal tags, and lyrics.
#   ✓ Dual-Loader Support: Works with custom loader (quality) or native CLIP (LoRA).
#   ✓ Preserves and utilizes critical `conditioning_lyrics` metadata.
#   ✓ `AceT5ConditioningAnalyzer`: Debugs conditioning tensor stats and lyric encoding.

# ░▒▓ CHANGELOG:
#   - v4.1 (Current Release - Stability Fix):
#       • FIXED: Capped `vocal_weight` at 0.95 to prevent distortion at 1.0+.
#       • IMPROVED: Blend logic now always uses normalization for stability.
#       • UPDATED: Tooltips now correctly guide users on `vocal_weight`.
#   - v4.0 (Production Edition):
#       • ADDED: `AceT5ConditioningScheduled` with dual CLIP support.
#       • ADDED: Native CLIP support for LoRA compatibility.
#       • ADDED: Normalized merging & `conditioning_lyrics` metadata preservation.
#       • CLEANED: Removed all non-working experimental nodes.

# ░▒▓ CONFIGURATION:
#   → Primary Use: `AceT5ModelLoader` -> `AceT5ConditioningScheduled` for top-quality audio conditioning.
#   → Secondary Use: `Native_CLIP_Loader` -> `AceT5ConditioningScheduled` to enable LoRA support.
#   → Edge Use: Piping `vocal_cond` output into `AceT5ConditioningAnalyzer` to debug lyric token counts.

# ░▒▓ WARNING:
#   This node may trigger:
#   ▓▒░ The cold realization that your `vocal_tags` and `lyrics` are in a fistfight for tensor dominance.
#   ▓▒░ A 2-hour loop of just checking the `AceT5ConditioningAnalyzer` output for `Std: 0.16`.
#   ▓▒░ A crippling fear of `vocal_weight` values above 0.9.
#   ▓▒░ Suddenly understanding why your "epic choir" prompt sounds like a 2400 baud modem handshake.
#   Consult your nearest demoscene vet if hallucinations persist.
# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄


# =================================================================================
# == Standard Library Imports                                                  ==
# =================================================================================
import os
import re
import logging
import copy
import traceback

# =================================================================================
# == Third-Party Imports                                                       ==
# =================================================================================
import torch
try:
    from tokenizers import Tokenizer
except ImportError:
    # Use print for visibility during ComfyUI startup
    print("WARNING: [ACE T5 Nodes] 'tokenizers' library not found. Lyrics BPE tokenizer will not function.")
    Tokenizer = None

# Local project imports (relative paths)
from .spiece_tokenizer import SPieceTokenizer
from .ace_text_cleaners import multilingual_cleaners, japanese_to_romaji

# =================================================================================
# == ComfyUI Core Modules                                                      ==
# =================================================================================
from comfy import sd1_clip
import comfy.text_encoders.t5
import comfy.utils
import folder_paths
# Safer import for built-in nodes
try:
    # Attempt direct import first (newer ComfyUI versions)
    from nodes import ConditioningAverage
except ImportError:
    # Fallback to importing the main nodes module and accessing mappings
    try:
        import nodes
        ConditioningAverage = nodes.NODE_CLASS_MAPPINGS.get("ConditioningAverage")
        if ConditioningAverage is None:
             # Use logging here as it's less critical than tokenizer import
             logging.error("[ACE T5 Nodes] Could not import the built-in ConditioningAverage node.")
             # Define a dummy class to prevent crashes if it's somehow used
             class ConditioningAverage:
                 @classmethod
                 def INPUT_TYPES(s): return {"required": {}}
                 RETURN_TYPES = ("CONDITIONING",)
                 FUNCTION = "execute"
                 CATEGORY = "_DISABLED_"
                 def execute(self, **kwargs): return ([],)

    except ImportError:
         logging.error("[ACE T5 Nodes] Failed to import ComfyUI's 'nodes' module.")
         ConditioningAverage = None # Set to None if even `nodes` fails

# =================================================================================
# == Local Project Imports (Already handled above with relative paths)          ==
# =================================================================================
# (None)

# =================================================================================
# == Helper Classes & Dependencies                                             ==
# =================================================================================

SUPPORT_LANGUAGES = {"en": 259, "de": 260, "fr": 262, "es": 284, "it": 285, "pt": 286, "pl": 294, "tr": 295, "ru": 267, "cs": 293, "nl": 297, "ar": 5022, "zh": 5023, "ja": 5412, "hu": 5753, "ko": 6152, "hi": 6680}
structure_pattern = re.compile(r"\[.*?\]")

COMFY_BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
DEFAULT_VOCAB_FILE = os.path.join(COMFY_BASE, "comfy", "text_encoders", "ace_lyrics_tokenizer", "vocab.json")

class VoiceBpeTokenizer:
    """Tokenizer for lyrics using HuggingFace tokenizers library."""
    START_TOKEN, END_TOKEN = 261, 2
    def __init__(self, vocab_file=DEFAULT_VOCAB_FILE):
        """Initializes the BPE tokenizer from a vocab file."""
        self.tokenizer = None
        if Tokenizer is None:
             logging.error("VoiceBpeTokenizer cannot initialize: 'tokenizers' library is missing.")
             return

        if vocab_file and os.path.exists(vocab_file):
            try:
                self.tokenizer = Tokenizer.from_file(vocab_file)
            except Exception as e:
                logging.error(f"Failed to load VoiceBpeTokenizer vocab file '{vocab_file}': {e}")
        else:
            logging.error(f"VoiceBpeTokenizer vocab file not found or path invalid: {vocab_file}")


    def encode(self, txt, lang='en'):
        """
        Encodes cleaned text into token IDs.

        Args:
            txt: The input text string.
            lang: The language code (e.g., 'en', 'ja').

        Returns:
            A list of integer token IDs. Returns empty list on failure.
        """
        if not self.tokenizer: return []
        try:
            txt = multilingual_cleaners(txt, lang)
            # Format with language tag and replace spaces
            txt = f"[{'zh-cn' if lang == 'zh' else lang}]{txt}".replace(" ", "[SPACE]")
            return self.tokenizer.encode(txt).ids
        except Exception as e:
             logging.warning(f"VoiceBpeTokenizer encode failed for text '{txt[:50]}...': {e}")
             return []


    def __call__(self, string):
        """
        Tokenizes a multi-line string containing lyrics with language tags.

        Args:
            string: The input lyrics string. Lines can optionally start with '[lang]' (e.g., '[en]Hello').

        Returns:
            A dictionary suitable for transformers models, e.g., {'input_ids': [261, ..., 2]}.
            Returns {'input_ids': [261]} if tokenizer failed to load or on error.
        """
        if not self.tokenizer: return {"input_ids": [self.START_TOKEN]}

        lyric_token_idx = [self.START_TOKEN]
        for line in string.split("\n"):
            line = line.strip()
            if not line:
                lyric_token_idx.append(self.END_TOKEN)
                continue

            lang, line_text = ("en", line) # Default language
            # Check for language tag prefix '[xx]'
            if len(line) > 3 and line.startswith("[") and line[3:4] == ']':
                lang_code = line[1:3].lower()
                if lang_code in SUPPORT_LANGUAGES:
                    lang, line_text = lang_code, line[4:]

            # Attempt Japanese Romaji conversion
            try:
                romaji = japanese_to_romaji(line_text)
                if romaji != line_text:
                    lang = "ja" # Override language if conversion happened
                    line_text = romaji
            except Exception:
                pass # Ignore errors during romaji conversion

            try:
                # Use 'en' if the line looks like a structure tag (e.g., '[Verse]')
                current_lang = "en" if structure_pattern.match(line_text) else lang
                token_idx = self.encode(line_text, current_lang)
                if token_idx: # Only extend if encoding was successful
                     lyric_token_idx.extend(token_idx + [self.END_TOKEN])
                else:
                     logging.warning(f"Skipping line due to encode error: '{line_text}'")
            except Exception as e:
                logging.warning(f"Tokenize error {e} for line '{line_text}'")

        # Ensure there's at least the start token if all lines failed
        if len(lyric_token_idx) == 1:
             lyric_token_idx.append(self.END_TOKEN) # Add end token if only start exists

        return {"input_ids": lyric_token_idx}


    @staticmethod
    def from_pretrained(path, **kwargs):
        """Class method for compatibility, loads from vocab file path."""
        return VoiceBpeTokenizer(path, **kwargs)

    def get_vocab(self):
        """Returns vocab (required by SDTokenizer interface, but not used here)."""
        return {}

class UMT5BaseModel(sd1_clip.SDClipModel):
    """Wrapper for the UMT5 Base model using ComfyUI's T5 implementation."""
    def __init__(self, device="cpu", layer="last", layer_idx=None, dtype=None, model_options={}):
        """Initializes the T5 model configuration."""
        json_config = os.path.join(COMFY_BASE, "comfy", "text_encoders", "umt5_config_base.json")
        super().__init__(
            device=device, layer=layer, layer_idx=layer_idx,
            textmodel_json_config=json_config, dtype=dtype,
            special_tokens={"end": 1, "pad": 0}, # Define special tokens for T5
            model_class=comfy.text_encoders.t5.T5, # Use ComfyUI's T5 implementation
            enable_attention_masks=True, zero_out_masked=False, model_options=model_options
        )

class UMT5BaseTokenizer(sd1_clip.SDTokenizer):
    """Tokenizer for UMT5 Base using SentencePiece."""
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        """Initializes the SentencePiece tokenizer."""
        # Find the spiece.model file
        spiece_path = tokenizer_data.get("spiece_model")
        if not spiece_path or not os.path.exists(spiece_path):
             # Default path relative to this file's directory if not provided in data
             default_spiece_path = os.path.join(os.path.dirname(__file__), "spiece.model")
             if os.path.exists(default_spiece_path):
                 spiece_path = default_spiece_path
             else:
                  logging.error("UMT5BaseTokenizer: spiece.model file not found in provided data or default location.")
                  # Set path to None to indicate failure
                  spiece_path = None

        super().__init__(
            tokenizer_path=spiece_path, # Pass the found path
            pad_with_end=False,
            embedding_size=768,
            embedding_key='umt5base', # Key for embeddings if used
            tokenizer_class=SPieceTokenizer, # Use the custom SentencePiece wrapper
            has_start_token=False,
            pad_to_max_length=False, # Do not pad automatically
            max_length=99999999, # Effectively unlimited length
            min_length=1,
            pad_token=0, # T5 pad token ID
            end_token=1, # T5 end token ID
            tokenizer_data=tokenizer_data
        )

    def state_dict(self):
        """Returns tokenizer state (e.g., model path or content)."""
        # Ensure tokenizer exists before accessing serialize_model
        if hasattr(self.tokenizer, 'serialize_model'):
            return {"spiece_model": self.tokenizer.serialize_model()}
        return {}


class LyricsTokenizer(sd1_clip.SDTokenizer):
    """Wrapper for the VoiceBpeTokenizer adhering to SDTokenizer interface."""
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        """Initializes the Lyrics BPE tokenizer."""
        # Path to the vocab.json file
        path = tokenizer_data.get("vocab_file", DEFAULT_VOCAB_FILE) # Use data if provided, else default
        super().__init__(
            tokenizer_path=path, # Pass vocab file path
            pad_with_end=False,
            embedding_size=1024, # Expected embedding size for lyrics
            embedding_key='lyrics', # Key for embeddings if used
            tokenizer_class=VoiceBpeTokenizer, # Use the BPE wrapper
            has_start_token=True, # VoiceBPE adds start token
            pad_to_max_length=False,
            max_length=99999999,
            min_length=1,
            pad_token=2, # Specific pad token for lyrics if needed
            has_end_token=False, # VoiceBPE adds end tokens per line
            tokenizer_data=tokenizer_data
        )

class AceT5Tokenizer:
    """Combines UMT5 Base and Lyrics tokenizers."""
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        """Initializes both underlying tokenizers."""
        self.voicebpe = LyricsTokenizer(embedding_directory, tokenizer_data)
        self.umt5base = UMT5BaseTokenizer(embedding_directory, tokenizer_data)

    def tokenize_with_weights(self, text, return_word_ids=False, **kwargs):
        """
        Tokenizes text for UMT5 and lyrics separately.

        Args:
            text: The main input text (for UMT5).
            return_word_ids: Flag (unused here, kept for compatibility).
            **kwargs: Expected to contain a 'lyrics' key with the lyrics string.

        Returns:
            A dictionary containing tokenized results for 'lyrics' and 'umt5base'.
        """
        # Get lyrics from kwargs, default to empty string if not provided
        lyrics_text = kwargs.get("lyrics", "")
        return {
            "lyrics": self.voicebpe.tokenize_with_weights(lyrics_text, return_word_ids, **kwargs),
            "umt5base": self.umt5base.tokenize_with_weights(text, return_word_ids, **kwargs)
        }

    def untokenize(self, token_weight_pair):
        """Untokenizes UMT5 tokens (lyrics untokenization not directly supported)."""
        # Assuming untokenization primarily applies to the main text part (umt5base)
        return self.umt5base.untokenize(token_weight_pair)

    def state_dict(self):
        """Returns the state dictionary for the UMT5 tokenizer."""
        return self.umt5base.state_dict()

# =================================================================================
# == Core Model Class                                                          ==
# =================================================================================

class AceT5Model(torch.nn.Module):
    """Main ACE T5 model combining UMT5 Base and handling lyrics metadata."""
    def __init__(self, device="cpu", dtype=None, model_options={}):
        """Initializes the UMT5 base model."""
        super().__init__()
        self.umt5base = UMT5BaseModel(device=device, dtype=dtype, model_options=model_options)
        self.dtypes = set() # Track data types used
        if dtype is not None:
            self.dtypes.add(dtype)

    def set_clip_options(self, options):
        """Sets options for the underlying CLIP model."""
        self.umt5base.set_clip_options(options)

    def reset_clip_options(self):
        """Resets options for the underlying CLIP model."""
        self.umt5base.reset_clip_options()

    def load_sd(self, sd):
        """Loads the state dictionary into the UMT5 base model."""
        return self.umt5base.load_sd(sd)

    def encode_token_weights(self, token_weight_pairs):
        """
        Encodes token weights, extracting lyrics tokens for metadata.

        Args:
            token_weight_pairs: Dictionary from AceT5Tokenizer with 'umt5base' and 'lyrics' keys.

        Returns:
            Tuple: (umt5_embeddings, umt5_pooled_output, extra_metadata_dict)
                   where extra_metadata_dict contains 'conditioning_lyrics'.
        """
        umt5_tokens = token_weight_pairs.get("umt5base")
        lyrics_tokens_info = token_weight_pairs.get("lyrics", ([],)) # Default to empty list in tuple

        if umt5_tokens is None:
             raise ValueError("Missing 'umt5base' tokens in input.")

        # Encode UMT5 tokens
        t5_out, t5_pooled = self.umt5base.encode_token_weights(umt5_tokens)

        # Prepare lyrics metadata tensor
        # Extract just the token IDs from the lyrics token/weight pairs
        lyrics_ids = []
        if lyrics_tokens_info and lyrics_tokens_info[0]: # Check if list is not empty
            try:
                 # lyrics_tokens_info is like ([(id1, w1), (id2, w2)],)
                 lyrics_ids = [pair[0] for pair in lyrics_tokens_info[0]]
            except (IndexError, TypeError) as e:
                 logging.warning(f"Could not parse lyrics tokens for metadata: {e}")

        # Ensure lyrics_embeds is always a 2D tensor [1, num_tokens]
        if not lyrics_ids:
             lyrics_embeds = torch.tensor([[VoiceBpeTokenizer.START_TOKEN]], dtype=torch.long) # Default to just START token
        else:
             lyrics_embeds = torch.tensor([lyrics_ids], dtype=torch.long)

        # Store lyrics tensor in the metadata dictionary
        extra_metadata = {"conditioning_lyrics": lyrics_embeds}

        return t5_out, t5_pooled, extra_metadata

    def encode_separate(self, token_weight_pairs):
        """
        Encodes UMT5 and lyrics parts separately for blending.

        Args:
            token_weight_pairs: Dictionary from AceT5Tokenizer.

        Returns:
            Tuple: (conditioning_tags, conditioning_lyrics)
                   Each is a ComfyUI conditioning list: [[tensor, metadata_dict]]
        """
        umt5_tokens = token_weight_pairs.get("umt5base")
        lyrics_tokens_info = token_weight_pairs.get("lyrics", ([],))

        if umt5_tokens is None:
             raise ValueError("Missing 'umt5base' tokens in input.")

        # --- Encode UMT5 Tags ---
        t5_out, t5_pooled = self.umt5base.encode_token_weights(umt5_tokens)

        # Ensure pooled output is always present, even if None is returned
        if t5_pooled is None and t5_out is not None:
             t5_pooled = torch.zeros(t5_out.shape[0], t5_out.shape[-1], device=t5_out.device, dtype=t5_out.dtype)
        elif t5_pooled is None: # Handle case where t5_out might also be None (error?)
             # Cannot determine shape, create a fallback small tensor
             logging.error("UMT5 encoding returned None for embeddings and pooled output.")
             t5_pooled = torch.zeros(1, 768) # Assuming batch size 1, embed dim 768
             t5_out = torch.zeros(1, 1, 768) # Placeholder

        # Extract lyrics token IDs for metadata (consistent across both outputs)
        lyrics_ids = []
        if lyrics_tokens_info and lyrics_tokens_info[0]:
            try: lyrics_ids = [pair[0] for pair in lyrics_tokens_info[0]]
            except (IndexError, TypeError): pass
        if not lyrics_ids: lyrics_ids = [VoiceBpeTokenizer.START_TOKEN]
        lyrics_metadata_tensor = torch.tensor([lyrics_ids], dtype=torch.long, device=t5_out.device)

        # Create conditioning list for tags
        cond_tags = [[t5_out, {"pooled_output": t5_pooled, "conditioning_lyrics": lyrics_metadata_tensor}]]


        # --- Encode Lyrics Separately (using embedding lookup) ---
        lyrics_cond_tensor = None
        lyrics_pooled = torch.zeros_like(t5_pooled) # Lyrics part doesn't have a meaningful pooled output

        if lyrics_tokens_info and lyrics_tokens_info[0]: # If lyrics tokens exist
             try:
                 device = t5_out.device
                 # Get just the IDs and weights
                 lyrics_ids_weights = lyrics_tokens_info[0]
                 ids_only = torch.tensor([pair[0] for pair in lyrics_ids_weights], device=device).unsqueeze(0) #[1, seq_len]

                 # Pad or truncate lyrics IDs to match the sequence length of UMT5 output
                 target_seq_len = t5_out.shape[1]
                 current_seq_len = ids_only.shape[1]

                 if current_seq_len < target_seq_len:
                     # Pad with the specific pad token ID (check T5 config, usually 0)
                     padding = torch.full((1, target_seq_len - current_seq_len), 0, device=device, dtype=ids_only.dtype)
                     ids_padded = torch.cat([ids_only, padding], dim=1)
                 elif current_seq_len > target_seq_len:
                     ids_padded = ids_only[:, :target_seq_len]
                 else:
                     ids_padded = ids_only

                 # Lookup embeddings directly from the transformer's shared embedding layer
                 with torch.no_grad():
                     if hasattr(self.umt5base, 'transformer') and hasattr(self.umt5base.transformer, 'shared'):
                          embed_layer = self.umt5base.transformer.shared
                          lyrics_cond_tensor = embed_layer(ids_padded).to(device) # Get embeddings

                          # Apply weights (if necessary, though often handled by sampler)
                          # Ensure weights list matches the original (non-padded) length
                          num_tokens_original = len(lyrics_ids_weights)
                          num_tokens_to_weight = min(num_tokens_original, target_seq_len) # Apply weights only up to original length or target length
                          for i in range(num_tokens_to_weight):
                               lyrics_cond_tensor[:, i] *= lyrics_ids_weights[i][1]
                     else:
                          logging.error("Could not find embedding layer ('transformer.shared') for lyrics encoding.")
                          lyrics_cond_tensor = torch.zeros_like(t5_out) # Fallback

             except Exception as e:
                  logging.error(f"Error during separate lyrics embedding lookup: {e}", exc_info=True)
                  lyrics_cond_tensor = torch.zeros_like(t5_out) # Fallback

        else: # No lyrics provided
             lyrics_cond_tensor = torch.zeros_like(t5_out)

        # Create conditioning list for lyrics part
        cond_lyrics = [[lyrics_cond_tensor, {"pooled_output": lyrics_pooled, "conditioning_lyrics": lyrics_metadata_tensor}]]

        return (cond_tags, cond_lyrics)


# =================================================================================
# == ComfyUI Custom Nodes                                                      ==
# =================================================================================

class AceT5ModelLoader:
    """
    MD: ACE T5 Model Loader

    Loads the custom ACE T5 model and its specialized tokenizers.
    Provides the highest quality conditioning for ACE Step audio models.
    NOTE: This custom loader does *not* support LoRAs. Use the native
    CLIPLoader/DualCLIPLoader for LoRA compatibility (may have slightly
    different conditioning characteristics).
    """
    @classmethod
    def INPUT_TYPES(cls):
        """Define node inputs."""
        return {
            "required": {
                "clip_name": (folder_paths.get_filename_list("clip"), {
                    "tooltip": (
                        "ACE T5 MODEL FILE\n"
                        "- Select your ACE T5 model file (.safetensors).\n"
                        "- Location: ComfyUI/models/clip/\n"
                        "- Example: umt5_with_tokenizer.safetensors\n\n"
                        "LOADER COMPARISON:\n"
                        "1. THIS Loader (AceT5ModelLoader):\n"
                        "  ✅ Best conditioning quality\n"
                        "  ✅ Correct lyrics token handling\n"
                        "  ❌ NO LoRA Support\n"
                        "2. Native CLIPLoader/DualCLIPLoader:\n"
                        "  ✅ LoRA Support\n"
                        "  ⚠️ Slightly different quality\n"
                        "  ⚠️ May require more steps (80-100)\n\n"
                        "Choose based on LoRA needs."
                    )
                })
            }
        }

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip"
    CATEGORY = "MD_Nodes/Loaders"
    DESCRIPTION = "Loads ACE T5 models with custom tokenizers for optimal audio generation quality (No LoRA)."

    def load_clip(self, clip_name):
        """
        Loads the specified ACE T5 CLIP model file and initializes the custom wrapper.

        Args:
            clip_name: The filename of the model within the ComfyUI/models/clip directory.

        Returns:
            A tuple containing the custom ACE T5 CLIP wrapper object.
            Returns a tuple containing None on failure.
        """
        try:
            clip_path = folder_paths.get_full_path("clip", clip_name)
            if not clip_path or not os.path.exists(clip_path):
                raise FileNotFoundError(f"ACE T5 model file not found: {clip_name} at path {clip_path}")

            logging.info(f"[AceT5Loader] Loading model from: {clip_path}")
            # Use safe_load=True for security with safetensors
            clip_data = comfy.utils.load_torch_file(clip_path, safe_load=True)

            # --- Internal Wrapper Class ---
            class AceT5ClipWrapper:
                """Wraps the AceT5Model and AceT5Tokenizer for ComfyUI compatibility."""
                def __init__(self, model_data):
                    self.cond_stage_model = AceT5Model() # Initialize the combined model
                    self.tokenizer = AceT5Tokenizer()    # Initialize the combined tokenizer
                    # Load the state dictionary into the UMT5 part
                    self.cond_stage_model.load_sd(model_data)
                    # Standard CLIP wrapper attributes (may not be fully used but needed for interface)
                    self.patcher = None
                    self.layer_idx = None

                def tokenize(self, text, return_word_ids=False, **kwargs):
                    """Tokenizes using the combined AceT5Tokenizer."""
                    return self.tokenizer.tokenize_with_weights(text, return_word_ids, **kwargs)

                def encode_from_tokens(self, tokens, return_pooled=False, return_dict=False):
                    """Encodes tokens using the AceT5Model, returning embeddings and metadata."""
                    cond, pooled, extra = self.cond_stage_model.encode_token_weights(tokens)
                    # Add lyrics metadata to the standard conditioning format
                    # ComfyUI expects conditioning to be a list of lists: [[tensor, dict]]
                    output_cond = [[cond, {"pooled_output": pooled, **extra}]]
                    if return_dict: # Unused by ComfyUI sampler but kept for interface?
                         # This return format might not be standard for ComfyUI
                         return {"cond": output_cond, "pooled_output": pooled}
                    if return_pooled:
                         # Return format expected by some parts of ComfyUI
                         return cond, pooled # Return raw tensors
                    # Standard return for conditioning nodes
                    # Return just the conditioning tensor list
                    # This might be incorrect - ComfyUI usually expects the list of lists
                    # Let's return the standard format expected by samplers:
                    return output_cond # Return [[tensor, dict]]

                def encode_separate(self, tokens):
                    """Encodes tags and lyrics separately using AceT5Model."""
                    return self.cond_stage_model.encode_separate(tokens)

                def load_sd(self, sd):
                    """Loads state dict into the underlying model."""
                    return self.cond_stage_model.load_sd(sd)

                def get_sd(self):
                    """Gets state dict from the underlying UMT5 model."""
                    return self.cond_stage_model.umt5base.state_dict()

                def clone(self):
                    """Creates a shallow clone (shares model and tokenizer instances)."""
                    n = AceT5ClipWrapper.__new__(AceT5ClipWrapper)
                    n.cond_stage_model = self.cond_stage_model
                    n.tokenizer = self.tokenizer
                    n.patcher = self.patcher # Copy reference
                    n.layer_idx = self.layer_idx # Copy value
                    return n

                def set_clip_options(self, options):
                    """Passes options to the underlying model."""
                    self.cond_stage_model.set_clip_options(options)

                def reset_clip_options(self):
                    """Resets options on the underlying model."""
                    self.cond_stage_model.reset_clip_options()
            # --- End Wrapper Class ---

            clip = AceT5ClipWrapper(clip_data)
            logging.info(f"[AceT5Loader] Successfully loaded and initialized '{clip_name}'.")
            # Always return a tuple
            return (clip,)

        except FileNotFoundError as e:
            logging.error(f"[AceT5Loader] Error: {e}")
            print(f"ERROR: [AceT5Loader] {e}") # Print for visibility
            return (None,) # Return None in tuple on error
        except Exception as e:
            logging.error(f"[AceT5Loader] Failed to load model '{clip_name}': {e}")
            logging.debug(traceback.format_exc())
            print(f"ERROR: [AceT5Loader] Failed to load model '{clip_name}': {e}")
            return (None,) # Return None in tuple on error


class AceT5ConditioningScheduled:
    """
    MD: ACE T5 Conditioning Scheduled

    Creates high-quality audio conditioning using ACE T5, separating genre/style tags,
    vocal characteristics, and lyrical content. Supports both the custom AceT5ModelLoader
    (recommended for quality) and native ComfyUI CLIP loaders (for LoRA support).
    Outputs multiple conditioning tensors for different use cases.
    """
    @classmethod
    def INPUT_TYPES(cls):
        """Define node inputs."""
        return {
            "required": {
                "clip": ("CLIP", {
                    "tooltip": (
                        "CLIP MODEL INPUT\n"
                        "- Connect the loaded ACE T5 CLIP model here.\n\n"
                        "Options:\n"
                        "1. Use output from 'MD: ACE T5 Model Loader':\n"
                        "   ✅ Best quality & proper lyrics handling.\n"
                        "   ❌ No LoRA support.\n"
                        "2. Use output from native 'CLIPLoader' or 'DualCLIPLoader':\n"
                        "   ✅ Enables LoRA support.\n"
                        "   ⚠️ Slightly different conditioning quality.\n"
                        "   ⚠️ Lyrics might be less distinct.\n\n"
                        "Choose the loader based on whether you need LoRAs."
                    )
                }),
                "base_tags": ("STRING", {
                    "multiline": True,
                    "default": "dubstep, synthwave, minimal, melodic",
                    "tooltip": (
                        "BASE TAGS (Genre/Style)\n"
                        "- Describe the overall sound, genre, mood, instrumentation.\n"
                        "- Forms the foundation of the track.\n"
                        "- Examples: 'trap, dark, heavy 808s', 'ambient, ethereal, pads', 'rock, electric guitar, drums'."
                    )
                }),
                "vocal_tags": ("STRING", {
                    "multiline": True,
                    "default": "soft female vocals, breathy, reverb",
                    "tooltip": (
                        "VOCAL TAGS (Characteristics)\n"
                        "- Describe HOW the vocals sound (timbre, processing, style), NOT the words.\n"
                        "- Leave empty for purely instrumental tracks.\n"
                        "- Examples: 'aggressive male rap', 'ethereal female choir', 'breathy whisper'."
                    )
                }),
                "lyrics": ("STRING", {
                    "multiline": True,
                    "default": "[en]oh oh oh",
                    "tooltip": (
                        "LYRICS\n"
                        "- The actual words/syllables to be sung.\n"
                        "- **MUST** start each line with a language code (e.g., '[en]', '[ja]', '[zh]').\n"
                        "- Supported: en, ja, zh, de, fr, es, it, pt, ru, ko, hi, ar, etc.\n"
                        "- **Keep lyrics SHORT (< 20 words total) for best quality.** Longer lyrics increase distortion risk."
                    )
                }),
            },
            "optional": {
                "base_tag_boost": ("INT", {
                    "default": 2, "min": 1, "max": 10,
                    "tooltip": (
                        "BASE TAG BOOST\n"
                        "- Repeats base_tags to strengthen genre adherence.\n"
                        "- 1: No boost.\n"
                        "- 2-3: Recommended range.\n"
                        "- 4+: Use if genre is weak or ignored."
                    )
                }),
                "vocal_tag_boost": ("INT", {
                    "default": 1, "min": 1, "max": 5,
                    "tooltip": (
                        "VOCAL TAG BOOST\n"
                        "- Repeats vocal_tags to strengthen vocal style.\n"
                        "- 1: Recommended (usually sufficient).\n"
                        "- 2+: Use if vocal characteristics are too subtle."
                    )
                }),
                "vocal_weight": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 0.95, "step": 0.05, # Capped max at 0.95
                    "tooltip": (
                        "VOCAL WEIGHT (Lyrics Influence)\n"
                        "- Controls the blend between tag conditioning and lyrics embeddings (0.0 to 0.95).\n"
                        "- **This is NOT vocal volume.**\n"
                        "- 0.0: Uses only tags (base + vocal).\n"
                        "- 0.5: Balanced mix (Recommended).\n"
                        "- 0.95: Strongest safe lyrics influence.\n"
                        "- **Values near 1.0 can cause distortion.**\n"
                        "- For truly instrumental tracks, use the 'base_cond' output directly."
                    )
                }),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("base_cond (Instrumental)", "vocal_cond (Tags+Lyrics)", "tags_only_vocal (No Blend)")
    FUNCTION = "encode"
    CATEGORY = "MD_Nodes/Conditioning"
    DESCRIPTION = ( # Kept description concise
        "Creates ACE audio conditioning blending genre, vocal style, and lyrics. Supports custom and native CLIP loaders."
    )

    def encode(self, clip, base_tags, vocal_tags="", lyrics="", base_tag_boost=1, vocal_tag_boost=1, vocal_weight=0.5, **kwargs):
        """
        Encodes text and lyrics using the provided CLIP model (custom or native).

        Args:
            clip: The loaded CLIP object (either custom wrapper or native).
            base_tags: String for genre/style.
            vocal_tags: String for vocal characteristics.
            lyrics: String for lyrical content with language tags.
            base_tag_boost: Integer multiplier for base_tags.
            vocal_tag_boost: Integer multiplier for vocal_tags.
            vocal_weight: Float (0.0-0.95) controlling lyrics blend influence.

        Returns:
            Tuple of three conditioning lists: (base_cond, vocal_cond_merged, vocal_cond_simple)
            Returns default empty conditioning on critical errors.
        """
        # Define default empty conditioning structure
        empty_cond = [[torch.zeros((1, 1, 768)), {"pooled_output": torch.zeros((1, 768))}]]

        try:
            # --- Input Preparation ---
            # Ensure tags are strings, default to empty if None
            base_tags = base_tags or ""
            vocal_tags = vocal_tags or ""
            lyrics = lyrics or ""

            # Apply boosts
            if base_tag_boost > 1:
                base_tags = ", ".join([base_tags.strip()] * base_tag_boost).strip(', ')
            if vocal_tag_boost > 1 and vocal_tags: # Only boost if vocal tags exist
                vocal_tags = ", ".join([vocal_tags.strip()] * vocal_tag_boost).strip(', ')

            # --- CLIP Type Detection ---
            # Check for the unique method of the custom wrapper
            is_custom_loader = hasattr(clip, 'encode_separate')
            logging.debug(f"[AceT5Conditioning] Detected CLIP type: {'Custom AceT5Loader' if is_custom_loader else 'Native ComfyUI Loader'}")

            # --- Encoding Logic ---
            if is_custom_loader:
                # Use the custom loader's optimized methods
                base_cond, vocal_cond_merged, vocal_cond_simple = self._encode_custom(
                    clip, base_tags, vocal_tags, lyrics, vocal_weight
                )
            else:
                # Use methods compatible with native ComfyUI CLIPLoader
                base_cond, vocal_cond_merged, vocal_cond_simple = self._encode_native(
                    clip, base_tags, vocal_tags, lyrics, vocal_weight
                )

            # --- Validation & Return ---
            # Basic validation of outputs
            if not base_cond or not base_cond[0][0].numel():
                 logging.error("[AceT5Conditioning] Base conditioning encoding failed or produced empty tensor.")
                 return (empty_cond, empty_cond, empty_cond)
            if not vocal_cond_merged or not vocal_cond_merged[0][0].numel():
                 logging.error("[AceT5Conditioning] Merged vocal conditioning encoding failed or produced empty tensor.")
                 return (base_cond, empty_cond, empty_cond) # Return base at least

            logging.info("[AceT5Conditioning] Encoding successful.")
            return (base_cond, vocal_cond_merged, vocal_cond_simple)

        except Exception as e:
            logging.error(f"[AceT5Conditioning] Failed during encode: {e}", exc_info=True)
            print(f"ERROR: [AceT5Conditioning] Failed during encode: {e}") # Print for visibility
            # Return default empty conditioning on any error
            return (empty_cond, empty_cond, empty_cond)


    def _encode_custom(self, clip, base_tags, vocal_tags, lyrics, vocal_weight):
        """Encoding logic using the custom AceT5Loader wrapper."""
        logging.debug("[AceT5Conditioning] Using _encode_custom path.")
        # 1. Encode base tags only (for instrumental output)
        tokens_base = clip.tokenize(base_tags, lyrics="") # Ensure lyrics are empty
        # encode_separate returns ([cond_list], [lyrics_cond_list])
        base_cond_list, _ = clip.encode_separate(tokens_base)
        base_cond = base_cond_list # [[tensor, dict]]

        # 2. Encode combined tags + lyrics
        # Combine tags carefully, handle empty vocal tags
        combined_tags = f"{base_tags}, {vocal_tags}".strip(', ') if vocal_tags else base_tags
        tokens_vocal = clip.tokenize(combined_tags, lyrics=lyrics)
        vocal_tags_cond_list, vocal_lyrics_cond_list = clip.encode_separate(tokens_vocal)

        # Extract tensors and metadata dictionaries
        vocal_tags_tensor, vocal_tags_extras = vocal_tags_cond_list[0]
        vocal_lyrics_tensor, vocal_lyrics_extras = vocal_lyrics_cond_list[0]

        # Ensure crucial metadata exists
        original_conditioning_lyrics = vocal_tags_extras.get("conditioning_lyrics")
        if original_conditioning_lyrics is None:
            logging.warning("Missing 'conditioning_lyrics' metadata from custom loader tags output.")
            original_conditioning_lyrics = torch.tensor([[VoiceBpeTokenizer.START_TOKEN]], dtype=torch.long) # Fallback

        pooled_tags = vocal_tags_extras.get("pooled_output")
        pooled_lyrics = vocal_lyrics_extras.get("pooled_output")
        if pooled_tags is None or pooled_lyrics is None:
             logging.warning("Missing 'pooled_output' metadata from custom loader.")
             # Create fallback zeros matching tensor shape if possible
             fallback_pooled_shape = (vocal_tags_tensor.shape[0], vocal_tags_tensor.shape[-1]) if vocal_tags_tensor is not None else (1, 768)
             pooled_tags = pooled_tags if pooled_tags is not None else torch.zeros(fallback_pooled_shape, device=vocal_tags_tensor.device, dtype=vocal_tags_tensor.dtype)
             pooled_lyrics = pooled_lyrics if pooled_lyrics is not None else torch.zeros_like(pooled_tags)


        # 3. Create "tags_only_vocal" output (using combined tags embedding but original lyrics metadata)
        vocal_cond_simple = [[vocal_tags_tensor, {
            "pooled_output": pooled_tags,
            "conditioning_lyrics": original_conditioning_lyrics # Use lyrics metadata from the combined run
        }]]

        # 4. Create merged "vocal_cond" output (blending tags and lyrics embeddings)
        # Clamp vocal_weight to safe range [0.0, 0.95]
        vocal_weight_clamped = max(0.0, min(float(vocal_weight), 0.95))

        if vocal_weight_clamped == 0.0 or not lyrics: # If weight is zero or no lyrics provided, use tags only
            logging.debug("Vocal weight is 0 or no lyrics, using tags_only conditioning for merged output.")
            vocal_cond_merged = vocal_cond_simple
        else:
            logging.debug(f"Blending tags and lyrics with weight {vocal_weight_clamped:.2f}")
            # Normalize lyrics tensor magnitude relative to tags tensor magnitude for stable blending
            tags_norm = torch.norm(vocal_tags_tensor, p=2, dim=-1, keepdim=True).mean() + 1e-6 # Add epsilon
            lyrics_norm = torch.norm(vocal_lyrics_tensor, p=2, dim=-1, keepdim=True).mean() + 1e-6 # Add epsilon

            # Scale lyrics tensor to match average magnitude of tags tensor
            lyrics_scaled = vocal_lyrics_tensor * (tags_norm / lyrics_norm)

            # Blend using normalized weight (alpha) ensures tags dominate when weight approaches 1
            # alpha = vocal_weight / (1.0 + vocal_weight) -> This might overly weaken lyrics?
            # Let's try direct linear interpolation with the clamped weight
            alpha = vocal_weight_clamped

            # Interpolate tensors
            merged_tensor = torch.lerp(vocal_tags_tensor, lyrics_scaled, alpha)
            # Interpolate pooled outputs (though lyrics pooled is often zeros)
            merged_pooled = torch.lerp(pooled_tags, pooled_lyrics, alpha)

            merged_extras = {
                "pooled_output": merged_pooled,
                "conditioning_lyrics": original_conditioning_lyrics # Crucially, preserve original lyrics metadata
            }
            vocal_cond_merged = [[merged_tensor, merged_extras]]

        return (base_cond, vocal_cond_merged, vocal_cond_simple)

    def _encode_native(self, clip, base_tags, vocal_tags, lyrics, vocal_weight):
        """Encoding logic using native ComfyUI CLIPLoader."""
        logging.debug("[AceT5Conditioning] Using _encode_native path.")

        # --- Get Conditioning Tensors using standard encode ---
        # 1. Base conditioning (instrumental)
        tokens_base = clip.tokenize(base_tags) # Native tokenizer doesn't use lyrics kwarg
        cond_base_tensor, pooled_base = clip.encode_from_tokens(tokens_base, return_pooled=True)
        # Ensure pooled exists
        if pooled_base is None: pooled_base = torch.zeros_like(cond_base_tensor[:, 0, :]) # Use shape from cond

        # 2. Vocal conditioning (combined tags)
        # Combine tags carefully, handle empty vocal tags
        combined_tags = f"{base_tags}, {vocal_tags}".strip(', ') if vocal_tags else base_tags
        tokens_vocal = clip.tokenize(combined_tags)
        cond_vocal_tensor, pooled_vocal = clip.encode_from_tokens(tokens_vocal, return_pooled=True)
        if pooled_vocal is None: pooled_vocal = torch.zeros_like(cond_vocal_tensor[:, 0, :])

        # --- Synthesize Lyrics Metadata (Crucial Step) ---
        # We need to manually create the 'conditioning_lyrics' metadata
        # because the native loader doesn't know about it.
        # We use the separate VoiceBpeTokenizer for this.
        lyrics_tokenizer = VoiceBpeTokenizer() # Instantiated temporarily
        if lyrics and lyrics_tokenizer.tokenizer:
             lyric_tokens_dict = lyrics_tokenizer(lyrics) # Get {'input_ids': [...]}
             lyrics_ids = lyric_tokens_dict.get('input_ids', [VoiceBpeTokenizer.START_TOKEN])
             # Ensure it's not just the start token if lyrics were provided but failed
             if len(lyrics_ids) <= 1 and lyrics:
                  logging.warning("Lyrics tokenization for metadata failed, using default START token.")
                  lyrics_ids = [VoiceBpeTokenizer.START_TOKEN]
        else:
             lyrics_ids = [VoiceBpeTokenizer.START_TOKEN] # Default if no lyrics or tokenizer failed

        # Ensure device matches conditioning tensors
        target_device = cond_base_tensor.device
        lyrics_metadata_tensor = torch.tensor([lyrics_ids], dtype=torch.long, device=target_device)
        logging.debug(f"Native mode generated lyrics metadata tensor: shape={lyrics_metadata_tensor.shape}, count={lyrics_metadata_tensor.numel()}")


        # --- Format Outputs ---
        # 1. base_cond (Instrumental) - Add synthesized lyrics metadata
        base_cond = [[cond_base_tensor, {
            "pooled_output": pooled_base,
            "conditioning_lyrics": lyrics_metadata_tensor # Add it here too for consistency
        }]]

        # 2. vocal_cond (Combined Tags + Lyrics Metadata) - This is the primary output
        # In native mode, we don't blend embeddings, we rely on the combined tag encoding.
        # The crucial part is adding the correct lyrics metadata.
        vocal_cond_merged = [[cond_vocal_tensor, {
            "pooled_output": pooled_vocal,
            "conditioning_lyrics": lyrics_metadata_tensor # Add the essential metadata
        }]]

        # 3. tags_only_vocal (Same as vocal_cond_merged in native mode)
        # The concept of a separate "tags only" blend doesn't apply cleanly here.
        # We return the same conditioning as vocal_cond_merged for interface consistency.
        vocal_cond_simple = copy.deepcopy(vocal_cond_merged)

        # Note: vocal_weight is IGNORED in native mode as there's no embedding blend.
        if vocal_weight != 0.5: # Use default as reference
             logging.warning("Native CLIP Loader mode ignores 'vocal_weight'. Conditioning uses combined tags directly.")

        return (base_cond, vocal_cond_merged, vocal_cond_simple)


class AceT5ConditioningAnalyzer:
    """
    MD: ACE T5 Conditioning Analyzer

    Debugging tool to inspect conditioning tensor statistics, lyrics token count,
    and metadata. Helps diagnose quality issues related to conditioning.
    """
    @classmethod
    def INPUT_TYPES(cls):
        """Define node inputs."""
        return {
            "required": {
                "conditioning": ("CONDITIONING", {
                    "tooltip": (
                        "CONDITIONING INPUT\n"
                        "- Connect ANY conditioning output (e.g., from AceT5ConditioningScheduled).\n\n"
                        "ANALYSIS INCLUDES:\n"
                        "- Tensor Shape, Device, Dtype\n"
                        "- Statistics (Mean, Std, Min, Max)\n"
                        "- Lyrics Token Count (from metadata)\n"
                        "- Pooled Output Info\n"
                        "- Timestep Info (if present)\n\n"
                        "LOOK FOR:\n"
                        "- Std > 1.0 (Potential distortion)\n"
                        "- Lyrics Tokens = [261] (No lyrics were encoded)\n"
                        "- Extreme Min/Max (e.g., > 5 or < -5)\n\n"
                        "TYPICAL GOOD VALUES:\n"
                        "- Std: ~0.15 - 0.25\n"
                        "- Min/Max: ~-2.0 to +2.0\n"
                        "- Lyrics Tokens: > 1 (ideally > 50 if lyrics used)"
                    )
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("analysis_report",) # Changed name for clarity
    FUNCTION = "analyze"
    CATEGORY = "MD_Nodes/Conditioning"
    OUTPUT_NODE = True # This node outputs information, doesn't modify data flow
    DESCRIPTION = "Analyzes conditioning tensors for debugging quality issues and verifying encoding."

    def analyze(self, conditioning):
        """
        Analyzes the provided conditioning list.

        Args:
            conditioning: A ComfyUI conditioning list (e.g., [[tensor, dict]]).

        Returns:
            A tuple containing a single string with the analysis report.
            Returns an error message string in the tuple on failure.
        """
        analysis = ["=== ACE T5 Conditioning Analysis ==="]
        try:
            if not conditioning or not isinstance(conditioning, list) or not conditioning[0]:
                raise ValueError("Input is not a valid ComfyUI conditioning list.")

            # Analyze the first item in the conditioning list
            cond_item = conditioning[0]
            if not isinstance(cond_item, list) or len(cond_item) < 2:
                 raise ValueError("Conditioning item format is incorrect. Expected [tensor, dict].")

            tensor, extras = cond_item[0], cond_item[1]

            if not isinstance(tensor, torch.Tensor):
                 raise ValueError("First element of conditioning item is not a tensor.")
            if not isinstance(extras, dict):
                 raise ValueError("Second element of conditioning item is not a dictionary.")

            # --- Tensor Analysis ---
            analysis.append(f"\n--- Tensor Stats ---")
            analysis.append(f"Shape: {tensor.shape}")
            analysis.append(f"Device: {tensor.device}")
            analysis.append(f"Dtype: {tensor.dtype}")
            if tensor.numel() > 0: # Avoid stats on empty tensor
                 mean_val = tensor.mean().item()
                 std_val = tensor.std().item()
                 min_val = tensor.min().item()
                 max_val = tensor.max().item()
                 analysis.append(f"Mean: {mean_val:.6f}")
                 analysis.append(f"Std Dev: {std_val:.6f}")
                 analysis.append(f"Min: {min_val:.6f}")
                 analysis.append(f"Max: {max_val:.6f}")

                 # --- Quality Assessment ---
                 analysis.append("\n--- Quality Assessment ---")
                 if std_val < 0.1: analysis.append("⚠️ Std Dev < 0.1: May lack detail or be silent.")
                 elif std_val > 1.0: analysis.append("🔥 Std Dev > 1.0: HIGH RISK OF DISTORTION!")
                 else: analysis.append("✅ Std Dev looks good (0.1 - 1.0).")

                 if abs(min_val) > 5 or abs(max_val) > 5: analysis.append("⚠️ Extreme Min/Max values (> +/-5): Potential for artifacts/clipping.")
                 elif abs(min_val) > 2 or abs(max_val) > 2: analysis.append("ℹ️ Moderate Min/Max values (> +/-2): Check output quality.")
                 else: analysis.append("✅ Value range looks good (-2.0 to +2.0).")
            else:
                 analysis.append("Tensor is empty.")


            # --- Extras Dictionary Analysis ---
            analysis.append("\n--- Metadata ('extras') ---")
            for key, value in extras.items():
                if isinstance(value, torch.Tensor):
                    analysis.append(f"- {key}: Tensor {value.shape}, Dtype={value.dtype}, Device={value.device}")
                else:
                    analysis.append(f"- {key}: Type={type(value).__name__}") # Display type for non-tensors

            # --- Lyrics Analysis (Specific Key) ---
            if "conditioning_lyrics" in extras:
                lyrics_tensor = extras["conditioning_lyrics"]
                analysis.append(f"\n--- Lyrics Analysis ('conditioning_lyrics') ---")
                if isinstance(lyrics_tensor, torch.Tensor):
                    analysis.append(f"Shape: {lyrics_tensor.shape}")
                    analysis.append(f"Device: {lyrics_tensor.device}")
                    analysis.append(f"Dtype: {lyrics_tensor.dtype}")
                    if lyrics_tensor.numel() > 0:
                        try:
                            # Convert tensor to list, handle potential nesting/squeezing
                            lyrics_list = lyrics_tensor.long().cpu().flatten().tolist()
                            token_count = len(lyrics_list)
                            display_tokens = lyrics_list[:20] # Show first few tokens

                            analysis.append(f"Token Count: {token_count}")
                            analysis.append(f"First <=20 Tokens: {display_tokens}{'...' if token_count > 20 else ''}")

                            # Lyrics quality check based on tokens
                            if token_count == 1 and lyrics_list[0] == VoiceBpeTokenizer.START_TOKEN:
                                analysis.append("❌ NO LYRICS ENCODED! Only START token found.")
                            elif token_count < 50: analysis.append("✅ Short lyrics (good for quality).")
                            elif token_count < 200: analysis.append("ℹ️ Medium lyrics length.")
                            elif token_count < 400: analysis.append("⚠️ Long lyrics - may impact quality/coherence.")
                            else: analysis.append("🔥 Very long lyrics - HIGHLY recommend shortening!")
                        except Exception as parse_e:
                            analysis.append(f"⚠️ Could not parse lyrics tensor content: {parse_e}")
                    else:
                        analysis.append("⚠️ Lyrics tensor is empty.")
                else:
                    analysis.append("⚠️ 'conditioning_lyrics' is not a tensor!")
            else:
                analysis.append("\n--- Lyrics Analysis ---")
                analysis.append("❌ 'conditioning_lyrics' key NOT FOUND in metadata!")

            # --- Pooled Output Analysis ---
            if "pooled_output" in extras:
                pooled = extras["pooled_output"]
                analysis.append(f"\n--- Pooled Output ---")
                if isinstance(pooled, torch.Tensor):
                    analysis.append(f"Shape: {pooled.shape}")
                    analysis.append(f"Device: {pooled.device}")
                    analysis.append(f"Dtype: {pooled.dtype}")
                    if pooled.numel() > 0:
                         mean_val = pooled.mean().item()
                         analysis.append(f"Mean: {mean_val:.6f}")
                         # Check if pooled output is essentially zero (common for T5)
                         if abs(mean_val) < 1e-5 and pooled.std().item() < 1e-5:
                              analysis.append("ℹ️ Pooled output is near zero (typical for T5 models).")
                    else: analysis.append("⚠️ Pooled output tensor is empty.")
                else: analysis.append("⚠️ 'pooled_output' is not a tensor!")
            else:
                 analysis.append("\n--- Pooled Output ---")
                 analysis.append("ℹ️ 'pooled_output' key not found (may be normal depending on CLIP model).")

            # --- Timestep Analysis ---
            if "start_percent" in extras and "end_percent" in extras:
                analysis.append(f"\n--- Timestep Scheduling ---")
                try:
                    start_p = float(extras['start_percent'])
                    end_p = float(extras['end_percent'])
                    analysis.append(f"Active Range: {start_p:.2f} - {end_p:.2f} ({start_p*100:.0f}% to {end_p*100:.0f}%)")
                except (ValueError, TypeError):
                     analysis.append("⚠️ Could not parse start/end percent values.")

            analysis.append("\n=== End of Analysis ===")
            return ("\n".join(analysis),)

        except Exception as e:
            logging.error(f"[AceT5Analyzer] Analysis failed: {e}", exc_info=True)
            error_report = f"❌ ANALYSIS FAILED:\n{e}\n\n{traceback.format_exc()}"
            # Always return a tuple matching RETURN_TYPES
            return (error_report,)

# =================================================================================
# == Node Registration                                                         ==
# =================================================================================

NODE_CLASS_MAPPINGS = {
    "AceT5ModelLoader": AceT5ModelLoader,
    "AceT5ConditioningScheduled": AceT5ConditioningScheduled,
    "AceT5ConditioningAnalyzer": AceT5ConditioningAnalyzer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AceT5ModelLoader": "MD: ACE T5 Model Loader",            # Added MD: prefix, removed emoji
    "AceT5ConditioningScheduled": "MD: ACE T5 Conditioning Scheduled", # Added MD: prefix, removed emoji
    "AceT5ConditioningAnalyzer": "MD: ACE T5 Conditioning Analyzer",    # Added MD: prefix, removed emoji
}