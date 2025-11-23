#!/usr/bin/env python3

import argparse
import json
import os
import sys
import subprocess
import torch
import torchaudio
from pathlib import Path
from loguru import logger

# Configure loguru
logger.remove()
logger.add(lambda msg: print(msg, end=""), format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")

def check_and_install_ollama_package():
    """Check if ollama package is installed and install it if missing"""
    try:
        import ollama
        logger.info("Ollama Python package already installed")
        return True
    except ImportError:
        logger.info("Ollama Python package not found, attempting to install...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "ollama"])
            logger.info("Successfully installed ollama package")
            # Try to import again
            import ollama
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install ollama package: {e}")
            return False
        except Exception as e:
            logger.error(f"Error importing ollama after installation: {e}")
            return False

# Try to install ollama if needed
OLLAMA_AVAILABLE = check_and_install_ollama_package()

try:
    from gptqmodel import GPTQModel
    from gptqmodel.models.auto import MODEL_MAP, SUPPORTED_MODELS
    from gptqmodel.models.base import BaseGPTQModel
    from huggingface_hub import snapshot_download
    from qwen_omni_utils import process_mm_info
    from transformers import Qwen2_5OmniProcessor
    from modeling_qwen2_5_omni_low_VRAM_mode import Qwen2_5OmniForConditionalGeneration
    QWEN_AVAILABLE = True
except ImportError as e:
    QWEN_AVAILABLE = False
    logger.warning(f"Qwen dependencies not available: {e}")

# Import ollama after potential installation
if OLLAMA_AVAILABLE:
    try:
        import ollama
    except ImportError:
        OLLAMA_AVAILABLE = False
        logger.error("Failed to import ollama even after installation attempt")

QWEN_SAMPLE_RATE = 16000

QWEN_SYSTEM_PROMPT = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."

# Roughly based on the official prompt https://github.com/ace-step/ACE-Step/blob/main/TRAIN_INSTRUCTION.md  
PROMPT = r"""Analyze the input audio:
1. `genre`: Most representative genres of the audio.
2. `subgenre`: Three or more tags of specific sub-genres and techniques.
3. `instrument`: All audibly present instruments in the audio, except vocal.
4. `tempo`: Tags describing the tempo of the audio. Do not use number or BPM.
5. `mood`: Tags describing the mood of the audio.
6. `has_vocal`: Whether there is any vocal in the audio.
7. `vocal`: If there is any vocal, then output a list of tags describing the vocal timbre. Otherwise, output an empty list.

Output format:
```json
{
  "genre": <str list>,
  "subgenre": <str list>,
  "instrument": <str list>,
  "tempo": <str list>,
  "mood": <str list>,
  "has_vocal": <bool>,
  "vocal": <str list>
}
```"""

PROMPT_LYRICS = r"""Analyze the input audio:
1. `genre`: Most representative genres of the audio.
2. `subgenre`: Three or more tags of specific sub-genres and techniques.
3. `instrument`: All audibly present instruments in the audio, except vocal.
4. `tempo`: Tags describing the tempo of the audio. Do not use number or BPM.
5. `mood`: Tags describing the mood of the audio.
6. `has_vocal`: Whether there is any vocal in the audio.
7. `vocal`: If there is any vocal, then output a list of tags describing the vocal timbre. Otherwise, output an empty list.
8. `lyrics`: If there is any vocal, then transcribe the lyrics and output at most 1000 characters. Otherwise, output an empty string. Use \n after each sentence.

Output format:
```json
{
  "genre": <str list>,
  "subgenre": <str list>,
  "instrument": <str list>,
  "tempo": <str list>,
  "mood": <str list>,
  "has_vocal": <bool>,
  "vocal": <str list>,
  "lyrics": <str>
}
```"""

if QWEN_AVAILABLE:
    @classmethod
    def patched_from_config(cls, config, *args, **kwargs):
        kwargs.pop("trust_remote_code", None)
        model = cls._from_config(config, **kwargs)
        return model

    Qwen2_5OmniForConditionalGeneration.from_config = patched_from_config

    class Qwen2_5OmniThinkerGPTQ(BaseGPTQModel):
        loader = Qwen2_5OmniForConditionalGeneration
        base_modules = [
            "thinker.model.embed_tokens",
            "thinker.model.norm",
            "thinker.audio_tower",
            "thinker.model.rotary_emb",
        ]
        pre_lm_head_norm_module = "thinker.model.norm"
        require_monkeypatch = False
        layers_node = "thinker.model.layers"
        layer_type = "Qwen2_5OmniDecoderLayer"
        layer_modules = [
            ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
            ["self_attn.o_proj"],
            ["mlp.up_proj", "mlp.gate_proj"],
            ["mlp.down_proj"],
        ]

        def pre_quantize_generate_hook_start(self):
            self.thinker.audio_tower = self.thinker.audio_tower.to(
                self.quantize_config.device
            )

        def pre_quantize_generate_hook_end(self):
            self.thinker.audio_tower = self.thinker.audio_tower.to("cpu")

        def preprocess_dataset(self, sample):
            return sample

    MODEL_MAP["qwen2_5_omni"] = Qwen2_5OmniThinkerGPTQ
    SUPPORTED_MODELS.extend(["qwen2_5_omni"])

def load_model(model_path: str):
    if not QWEN_AVAILABLE:
        raise RuntimeError("Qwen dependencies not available")
        
    if not os.path.exists(model_path):
        model_path = snapshot_download(repo_id=model_path)

    device_map = {
        "thinker.model": "cuda",
        "thinker.lm_head": "cuda",
        # "thinker.visual": "cpu",
        "thinker.audio_tower": "cpu",
        "talker": "cpu",
        "token2wav": "cpu",
    }

    model = GPTQModel.load(
        model_path,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model.disable_talker()

    processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
    return model, processor

def read_audio(file_path):
    try:
        audio, sr = torchaudio.load(file_path)
        audio = audio[:, : sr * 360]  # Limit to 6 minutes
        if sr != QWEN_SAMPLE_RATE:
            audio = torchaudio.functional.resample(audio, sr, QWEN_SAMPLE_RATE)
            sr = QWEN_SAMPLE_RATE
        audio = audio.mean(dim=0, keepdim=True)
        return audio, sr
    except Exception as e:
        logger.error(f"Error reading audio file {file_path}: {e}")
        raise

def check_and_pull_ollama_model(model_name):
    """Check if Ollama model exists and pull it if not"""
    if not OLLAMA_AVAILABLE:
        raise RuntimeError("Ollama not available")
    
    try:
        # List available models
        models = ollama.list()
        model_names = [model['name'] for model in models['models']]
        
        # Check if model exists (exact match or with ":latest")
        if model_name in model_names or f"{model_name}:latest" in model_names:
            logger.info(f"Ollama model '{model_name}' already available")
            return True
        else:
            logger.info(f"Ollama model '{model_name}' not found, pulling...")
            ollama.pull(model_name)
            logger.info(f"Successfully pulled Ollama model '{model_name}'")
            return True
    except ollama.ResponseError as e:
        logger.error(f"Ollama API error while checking/pulling model: {e}")
        return False
    except Exception as e:
        logger.error(f"Error checking/pulling Ollama model: {e}")
        return False

def inference_qwen(file_path, model, processor, do_lyrics):
    try:
        audio, _ = read_audio(file_path)
        audio = audio.numpy().squeeze(axis=0)

        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": PROMPT_LYRICS if do_lyrics else PROMPT},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio},
                ],
            },
        ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        audios, images, videos = process_mm_info(messages, use_audio_in_video=False)
        inputs = processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=False,
        )

        # Copy tensors to GPU and match dtypes
        ks = list(inputs.keys())
        for k in ks:
            if hasattr(inputs[k], "to"):
                inputs[k] = inputs[k].to("cuda")
                if inputs[k].dtype.is_floating_point:
                    inputs[k] = inputs[k].to(model.dtype)

        output_ids = model.thinker.generate(
            **inputs,
            max_new_tokens=1000,
            use_audio_in_video=False,
        )

        generate_ids = output_ids[:, inputs.input_ids.shape[1] :]
        response = processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return response
    except Exception as e:
        logger.error(f"Error during Qwen inference for {file_path}: {e}")
        raise

def inference_ollama(file_path, model_name, do_lyrics):
    if not OLLAMA_AVAILABLE:
        raise RuntimeError("Ollama not available")
        
    try:
        prompt_text = PROMPT_LYRICS if do_lyrics else PROMPT
        
        response = ollama.chat(
            model=model_name,
            messages=[{
                'role': 'user',
                'content': prompt_text,
            }],
            audio=file_path,
            options={
                'temperature': 0.7,
                'num_predict': 1500
            }
        )
        
        return response['message']['content']
    except Exception as e:
        logger.error(f"Error during Ollama inference for {file_path}: {e}")
        raise

def parse_response(content, do_lyrics):
    try:
        # Clean up the response
        content = content.replace("```json", "")
        content = content.replace("```", "")
        content = content.strip()

        data = json.loads(content)
        
        # Extract tags
        tags = []
        tags += data.get("genre", [])
        tags += data.get("subgenre", [])
        tags += data.get("instrument", [])
        tags += data.get("tempo", [])
        tags += data.get("mood", [])
        vocal_tags = data.get("vocal", [])
        tags += [x.strip() + " vocal" for x in vocal_tags if x and x != "vocal"]

        tags = [x.strip().lower() for x in tags if x]
        # The order of tags does not matter, so we sort them here
        # Tags will be shuffled in training
        tags = sorted(set(tags))
        prompt = ", ".join(tags)

        lyrics = ""
        if do_lyrics:
            lyrics = data.get("lyrics", "")
            if not lyrics:
                lyrics = "[instrumental]"
                logger.info("No lyrics found in response")

        return prompt, lyrics
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON response: {e}")
        logger.error(f"Response content: {content}")
        raise
    except KeyError as e:
        logger.error(f"Missing key in JSON response: {e}")
        logger.error(f"Response content: {content}")
        raise
    except Exception as e:
        logger.error(f"Failed to parse response: {e}")
        logger.error(f"Response content: {content}")
        raise

def process_file(file_path, model_type, model, processor, ollama_model, do_lyrics, overwrite):
    """Process a single audio file"""
    stem = Path(file_path).stem
    parent_dir = Path(file_path).parent
    prompt_path = parent_dir / f"{stem}_prompt.txt"
    lyrics_path = parent_dir / f"{stem}_lyrics.txt"

    need_prompt = overwrite or (not prompt_path.exists())
    need_lyrics = do_lyrics and (overwrite or (not lyrics_path.exists()))

    if not (need_prompt or need_lyrics):
        logger.debug(f"Skipping {file_path} - files already exist")
        return True

    logger.info(f"Processing: {Path(file_path).name}")
    
    try:
        # Perform inference
        if model_type == "qwen":
            if not QWEN_AVAILABLE:
                raise RuntimeError("Qwen model selected but dependencies not available")
            content = inference_qwen(file_path, model, processor, do_lyrics)
        else:  # ollama
            if not OLLAMA_AVAILABLE:
                raise RuntimeError("Ollama model selected but ollama not available")
            content = inference_ollama(file_path, ollama_model, do_lyrics)

        # Parse response
        prompt, lyrics = parse_response(content, do_lyrics)

        # Save results
        if need_prompt:
            with open(prompt_path, "w", encoding="utf-8", newline="\n") as f:
                f.write(prompt)
            logger.debug(f"Saved prompt to {prompt_path}")

        if need_lyrics:
            with open(lyrics_path, "w", encoding="utf-8", newline="\n") as f:
                f.write(lyrics)
            logger.debug(f"Saved lyrics to {lyrics_path}")

        # Clear GPU memory for Qwen
        if model_type == "qwen":
            torch.cuda.empty_cache()
            
        return True
    except Exception as e:
        logger.error(f"Failed to process {file_path}: {e}")
        return False

def process_directory(data_dir, model_type, qwen_model_path, ollama_model, do_lyrics, overwrite, recursive=False, auto_pull=True, auto_install=True):
    """Process all audio files in directory"""
    
    # Check and install ollama package if needed
    if model_type == "ollama" and auto_install:
        global OLLAMA_AVAILABLE
        if not OLLAMA_AVAILABLE:
            logger.info("Attempting to install ollama package...")
            OLLAMA_AVAILABLE = check_and_install_ollama_package()
            if OLLAMA_AVAILABLE:
                try:
                    global ollama
                    import ollama
                except ImportError:
                    logger.error("Failed to import ollama after installation")
                    OLLAMA_AVAILABLE = False
    
    # Check and pull Ollama model if needed
    if model_type == "ollama" and auto_pull and OLLAMA_AVAILABLE:
        if not check_and_pull_ollama_model(ollama_model):
            logger.warning(f"Failed to pull Ollama model '{ollama_model}', continuing anyway...")
    
    # Initialize model if needed
    model = None
    processor = None
    if model_type == "qwen":
        if not QWEN_AVAILABLE:
            raise RuntimeError("Qwen dependencies not available")
        logger.info("Loading Qwen model...")
        model, processor = load_model(qwen_model_path)
    
    # Supported audio formats
    extensions = {".aac", ".flac", ".m4a", ".mp3", ".ogg", ".wav"}
    
    # Get list of files
    if recursive:
        audio_files = []
        for ext in extensions:
            audio_files.extend(Path(data_dir).rglob(f"*{ext}"))
            audio_files.extend(Path(data_dir).rglob(f"*{ext.upper()}"))
    else:
        audio_files = []
        for item in Path(data_dir).iterdir():
            if item.is_file() and item.suffix.lower() in extensions:
                audio_files.append(item)
    
    logger.info(f"Found {len(audio_files)} audio files to process")
    
    # Process files
    failed_files = []
    successful_files = 0
    
    for file_path in sorted(audio_files):
        try:
            success = process_file(
                file_path=str(file_path),
                model_type=model_type,
                model=model,
                processor=processor,
                ollama_model=ollama_model,
                do_lyrics=do_lyrics,
                overwrite=overwrite
            )
            if success:
                successful_files += 1
            else:
                failed_files.append(str(file_path))
        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}")
            failed_files.append(str(file_path))
            continue
    
    logger.info(f"Processing complete! Success: {successful_files}, Failed: {len(failed_files)}")
    
    if failed_files:
        logger.warning(f"Failed to process {len(failed_files)} files:")
        for file in failed_files[:10]:  # Show first 10 failures
            logger.warning(f"  - {file}")
        if len(failed_files) > 10:
            logger.warning(f"  ... and {len(failed_files) - 10} more")

@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description="Generate lyrics and genre tags from audio files")
    parser.add_argument("--data_dir", type=str, default="./audio", help="Directory containing audio files")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    parser.add_argument("--lyrics", action="store_true", help="Include lyrics transcription")
    parser.add_argument("--model_type", type=str, default="qwen", choices=["qwen", "ollama"], 
                       help="Model to use for processing")
    parser.add_argument("--qwen_model", type=str, default="Qwen/Qwen2.5-Omni-7B-GPTQ-Int4", 
                       help="Qwen model path or HuggingFace ID")
    parser.add_argument("--ollama_model", type=str, default="qwen2.5", 
                       help="Ollama model name")
    parser.add_argument("--recursive", action="store_true", 
                       help="Process subdirectories recursively")
    parser.add_argument("--no_auto_pull", action="store_true",
                       help="Disable automatic pulling of Ollama models")
    parser.add_argument("--no_auto_install", action="store_true",
                       help="Disable automatic installation of ollama package")
    
    args = parser.parse_args()
    
    # Validate model availability
    if args.model_type == "qwen" and not QWEN_AVAILABLE:
        logger.error("Qwen model selected but required dependencies are not available")
        return 1
        
    if args.model_type == "ollama" and not args.no_auto_install:
        global OLLAMA_AVAILABLE
        if not OLLAMA_AVAILABLE:
            logger.info("Attempting to install ollama package...")
            OLLAMA_AVAILABLE = check_and_install_ollama_package()
            if OLLAMA_AVAILABLE:
                try:
                    global ollama
                    import ollama
                except ImportError:
                    logger.error("Failed to import ollama after installation")
                    OLLAMA_AVAILABLE = False
    
    if args.model_type == "ollama" and not OLLAMA_AVAILABLE:
        logger.error("Ollama model selected but ollama is not available")
        logger.info("Try running: pip install ollama")
        return 1
    
    # Check if data directory exists
    if not Path(args.data_dir).exists():
        logger.error(f"Data directory does not exist: {args.data_dir}")
        return 1
    
    try:
        process_directory(
            data_dir=args.data_dir,
            model_type=args.model_type,
            qwen_model_path=args.qwen_model,
            ollama_model=args.ollama_model,
            do_lyrics=args.lyrics,
            overwrite=args.overwrite,
            recursive=args.recursive,
            auto_pull=not args.no_auto_pull,
            auto_install=not args.no_auto_install
        )
        return 0
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())