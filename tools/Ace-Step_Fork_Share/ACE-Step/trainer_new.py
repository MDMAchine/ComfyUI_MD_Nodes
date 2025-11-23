#!/usr/bin/env python3
import argparse
import json
import os
import shutil
from glob import glob
# import time # Original did not have time, but it was added in your modified version. Keeping for now if it's used elsewhere.
import h5py
import hdf5plugin
import numpy as np
import safetensors.torch
import torch
import torch.nn.functional as F
import torch.utils.data
from natsort import natsorted
from peft import LoraConfig
from prodigyopt import Prodigy
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core import LightningModule
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, Dataset
from typing import Any, Dict # Added for type hinting on_save_checkpoint
from pytorch_lightning.utilities import rank_zero_info # Ensure this is imported for logging messages
from acestep.pipeline_ace_step import ACEStepPipeline
from acestep.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)

# --- IMPACTFUL OPTIMIZATION 1: Improved Data Loading Defaults ---
# Ensure num_workers is set to a reasonable value based on CPU cores if not explicitly provided.
# This logic should ideally be handled by the calling script (lora_magician_pro.py) or
# by ensuring a good default is passed. However, we can add a safeguard here.
# A better approach is to calculate a default in main() if num_workers is 0 or negative.
# We'll implement the calculation in main().

if torch.cuda.is_bf16_supported():
    torch.set_float32_matmul_precision("high") # Original used "high"
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
# torch._dynamo.config.recompile_limit = 64

def augment_tags(text_token_ids, mask, shuffle, dropout):
    if not shuffle and not dropout:
        return text_token_ids, mask
    COMMA = 275
    bos = text_token_ids[-1:]
    text_token_ids = text_token_ids[:-1]
    tags = []
    start_idx = 0
    _len = len(text_token_ids)
    for idx in range(_len):
        if text_token_ids[idx] == COMMA:
            if start_idx < idx:
                tags.append(text_token_ids[start_idx:idx])
            start_idx = idx + 1
    if start_idx < _len:
        tags.append(text_token_ids[start_idx:_len])
    if shuffle:
        # Shuffle tags using torch's random seed
        perm = torch.randperm(len(tags))
        tags = [tags[i] for i in perm]
    if dropout:
        tags = [x for x in tags if torch.rand(()) > dropout]
    comma = torch.tensor([COMMA], dtype=text_token_ids.dtype)
    tags_and_commas = []
    for x in tags:
        tags_and_commas.append(x)
    tags_and_commas.append(comma) # This comma is actually the last comma, before BOS
    if tags_and_commas:
        tags_and_commas[-1] = bos # Replace last comma with BOS
    else:
        tags_and_commas.append(bos)
    text_token_ids = torch.cat(tags_and_commas)
    mask = mask[: len(text_token_ids)]
    return text_token_ids, mask

# RESTORED: pytree_to_dtype function from the original file
def pytree_to_dtype(x, dtype):
    if isinstance(x, list):
        return [pytree_to_dtype(y, dtype) for y in x]
    elif isinstance(x, dict):
        return {k: pytree_to_dtype(v, dtype) for k, v in x.items()}
    elif isinstance(x, torch.Tensor) and x.dtype.is_floating_point:
        return x.to(dtype)
    else:
        return x

class HDF5Dataset(Dataset):
    def __init__(self, dataset_path, dtype, tag_shuffle, tag_dropout):
        self.dataset_path = dataset_path
        self.dtype = dtype
        self.tag_shuffle = tag_shuffle
        self.tag_dropout = tag_dropout
        self.filenames = sorted(os.listdir(dataset_path))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        file_path = os.path.join(self.dataset_path, self.filenames[idx])
        # --- Data loading: Using numpy.asarray is generally faster than torch.tensor(h5py.Dataset) ---
        with h5py.File(file_path, "r") as f:
            # torch.tensor(f[k]) is slow
            sample = {
                k: torch.from_numpy(np.asarray(f[k])) for k in f.keys() if k != "keys"
            }
        sample["text_token_ids"], sample["text_attention_mask"] = augment_tags(
            sample["text_token_ids"],
            sample["text_attention_mask"],
            self.tag_shuffle,
            self.tag_dropout,
        )
        sample["text_attention_mask"] = sample["text_attention_mask"].float()
        sample = pytree_to_dtype(sample, self.dtype)
        return sample

class Pipeline(LightningModule):
    def __init__(
        self,
        # Model
        checkpoint_dir: str = None,
        T: int = 1000,
        shift: float = 3.0,
        timestep_densities_type: str = "logit_normal",
        logit_mean: float = 0.0,
        logit_std: float = 1.0,
        lora_config_path: str = None,
        last_lora_path: str = None, # This is used by the original to load a specific LoRA
        # Data
        dataset_path: str = "./data/your_dataset_path",
        batch_size: int = 1,
        num_workers: int = 0, # Default remains 0, but main() will optimize it
        tag_dropout: float = 0.5,
        speaker_dropout: float = 0.0,
        lyrics_dropout: float = 0.0,
        # Optimizer
        ssl_coeff: float = 1.0,
        optimizer: str = "adamw",
        learning_rate: float = 1e-4,
        beta1: float = 0.9,
        beta2: float = 0.99,
        weight_decay: float = 1e-2,
        max_steps: int = 2000,
        warmup_steps: int = 10,
        # Others
        adapter_name: str = "lora_adapter", # Original used adapter_name here
        save_last: int = 5,
        # --- IMPACTFUL OPTIMIZATION 4: Make Prodigy slice_p configurable ---
        prodigy_slice_p: int = 11, # Added argument for Prodigy slice_p
        # output_dir and wandb_run_id are NOT part of the original Pipeline __init__
        # They will be handled in main() and passed to Trainer/WandbLogger
    ):
        super().__init__()
        self.save_hyperparameters()
        if torch.cuda.is_bf16_supported():
            self.to_dtype = torch.bfloat16
        else:
            self.to_dtype = torch.float16
        self.to_device = torch.device("cuda:0")
        # Initialize scheduler
        self.scheduler = self.get_scheduler()
        # Load model
        acestep_pipeline = ACEStepPipeline(checkpoint_dir)
        acestep_pipeline.load_checkpoint(acestep_pipeline.checkpoint_dir)
        self.transformer = acestep_pipeline.ace_step_transformer.to(
            self.to_device, self.to_dtype
        )
        self.transformer.eval()
        self.transformer.requires_grad_(False)
        self.transformer.enable_gradient_checkpointing()
        self.text_encoder_model = acestep_pipeline.text_encoder_model.to(
            self.to_device, self.to_dtype
        )
        self.text_encoder_model.eval()
        self.text_encoder_model.requires_grad_(False)
        del acestep_pipeline
        # Load LoRA
        assert lora_config_path, "Please provide a LoRA config path"
        with open(lora_config_path, encoding="utf-8") as f:
            lora_config = json.load(f)
        lora_config = LoraConfig(**lora_config)
        self.transformer.add_adapter(
            adapter_config=lora_config, adapter_name=adapter_name
        )
        # Original LoRA loading logic
        if last_lora_path:
            state_dict = safetensors.torch.load_file(last_lora_path)
            state_dict = {
                k.replace(".weight", f".{adapter_name}.weight"): v
                for k, v in state_dict.items()
            }
            self.transformer.load_state_dict(state_dict, strict=False)
        for module in self.transformer.projectors:
            module.forward = torch.compile(module.forward, dynamic=True)
        self.transformer.encode = torch.compile(self.transformer.encode, dynamic=True)
        self.text_encoder_model = torch.compile(self.text_encoder_model, dynamic=True)

    def get_scheduler(self):
        return FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=self.hparams.T,
            shift=self.hparams.shift,
        )

    def configure_optimizers(self):
        trainable_params = [
            p for name, p in self.transformer.named_parameters() if p.requires_grad
        ]
        if self.hparams.optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                params=trainable_params,
                lr=self.hparams.learning_rate,
                betas=(self.hparams.beta1, self.hparams.beta2),
                weight_decay=self.hparams.weight_decay,
            )
            lr_multiplier_final = 0
        elif self.hparams.optimizer == "prodigy":
            # --- IMPACTFUL OPTIMIZATION 3: Enhanced Prodigy Warnings ---
            recommended_lr = 1.0
            recommended_warmup = 0
            if abs(self.hparams.learning_rate - recommended_lr) > 1e-6: # Check if not close to 1.0
                print(
                    f"Warning: With Prodigy optimizer, lr={recommended_lr} is often recommended. "
                    f"Current lr={self.hparams.learning_rate}. "
                    f"You can adjust with --learning_rate {recommended_lr}"
                )
            if self.hparams.warmup_steps != recommended_warmup:
                print(
                    f"Warning: With Prodigy optimizer, warmup_steps={recommended_warmup} is often recommended. "
                    f"Current warmup_steps={self.hparams.warmup_steps}. "
                    f"You can adjust with --warmup_steps {recommended_warmup}"
                )
            # --- END OF ENHANCED WARNINGS ---
            optimizer = Prodigy(
                params=trainable_params,
                lr=self.hparams.learning_rate,
                betas=(self.hparams.beta1, self.hparams.beta2),
                weight_decay=self.hparams.weight_decay,
                use_bias_correction=True,
                safeguard_warmup=True,
                # --- IMPACTFUL OPTIMIZATION 4: Configurable slice_p ---
                slice_p=self.hparams.prodigy_slice_p, # Use the configurable argument
            )
            lr_multiplier_final = 0.1
        else:
            raise ValueError(f"Unknown optimizer: {self.hparams.optimizer}")
        max_steps = self.hparams.max_steps
        warmup_steps = self.hparams.warmup_steps  # New hyperparameter for warmup steps
        # Create a scheduler that first warms up linearly, then decays linearly
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                # Linear warmup from 0 to learning_rate
                return float(current_step) / float(max(1, warmup_steps))
            else:
                # Linear decay from learning_rate to 0
                progress = float(current_step - warmup_steps) / float(
                    max(1, max_steps - warmup_steps)
                )
                multiplier = max(0.0, 1.0 - progress)
                multiplier = (
                    multiplier * (1 - lr_multiplier_final) + lr_multiplier_final
                )
                return multiplier
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda, last_epoch=-1
        )
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]

    def train_dataloader(self):
        ds = HDF5Dataset(
            dataset_path=self.hparams.dataset_path,
            dtype=self.to_dtype,
            tag_shuffle=True,
            tag_dropout=self.hparams.tag_dropout,
        )
        return DataLoader(
            ds,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            persistent_workers=True, # Keep workers alive between epochs for efficiency
        )

    def get_sd3_sigmas(self, timesteps, device, n_dim, dtype):
        sigmas = self.scheduler.sigmas.to(device=device, dtype=dtype)
        schedule_timesteps = self.scheduler.timesteps.to(device)
        timesteps = timesteps.to(device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def get_timestep(self, bsz, device):
        if self.hparams.timestep_densities_type == "logit_normal":
            # See 3.1 in the SD3 paper ($rf/lognorm(0.00,1.00)$).
            # In practice, we sample the random variable u from a normal distribution u ∼ N (u; m, s)
            # and map it through the standard logistic function
            u = torch.normal(
                mean=self.hparams.logit_mean,
                std=self.hparams.logit_std,
                size=(bsz,),
                device="cpu",
            )
            u = torch.nn.functional.sigmoid(u)
            indices = (u * self.scheduler.config.num_train_timesteps).long()
            indices = torch.clamp(
                indices, 0, self.scheduler.config.num_train_timesteps - 1
            )
            timesteps = self.scheduler.timesteps[indices].to(device)
        else:
            raise ValueError(
                f"Unknown timestep_densities_type: {self.hparams.timestep_densities_type}"
            )
        return timesteps

    def run_step(self, batch, batch_idx):
        target_latents = batch["target_latents"]
        attention_mask = batch["attention_mask"]
        text_token_ids = batch["text_token_ids"]
        text_attention_mask = batch["text_attention_mask"]
        speaker_embds = batch["speaker_embds"]
        lyric_token_ids = batch["lyric_token_ids"]
        lyric_mask = batch["lyric_mask"]
        mert_ssl_hidden_states = batch["mert_ssl_hidden_states"]
        mhubert_ssl_hidden_states = batch["mhubert_ssl_hidden_states"]
        target_image = target_latents
        device = self.to_device
        dtype = self.to_dtype
        with torch.no_grad():
            outputs = self.text_encoder_model(
                input_ids=text_token_ids, attention_mask=text_attention_mask
            )
            encoder_text_hidden_states = outputs.last_hidden_state
        if (
            self.hparams.speaker_dropout
            and torch.rand(()) < self.hparams.speaker_dropout
        ):
            speaker_embds = torch.zeros_like(speaker_embds)
        if self.hparams.lyrics_dropout and torch.rand(()) < self.hparams.lyrics_dropout:
            lyric_token_ids = torch.zeros_like(lyric_token_ids)
            lyric_mask = torch.zeros_like(lyric_mask)
        # Step 1: Generate random noise, initialize settings
        noise = torch.randn_like(target_image)
        bsz = target_image.shape[0]
        timesteps = self.get_timestep(bsz, device)
        # Add noise according to flow matching.
        sigmas = self.get_sd3_sigmas(
            timesteps=timesteps, device=device, n_dim=target_image.ndim, dtype=dtype
        )
        noisy_image = sigmas * noise + (1.0 - sigmas) * target_image
        # This is the flow-matching target for vanilla SD3.
        target = target_image
        # SSL constraints for CLAP and vocal_latent_channel2
        all_ssl_hiden_states = []
        if mert_ssl_hidden_states is not None:
            all_ssl_hiden_states.append(mert_ssl_hidden_states)
        if mhubert_ssl_hidden_states is not None:
            all_ssl_hiden_states.append(mhubert_ssl_hidden_states)
        # N x H -> N x c x W x H
        x = noisy_image
        # Step 5: Predict noise
        transformer_output = self.transformer(
            hidden_states=x,
            attention_mask=attention_mask,
            encoder_text_hidden_states=encoder_text_hidden_states,
            text_attention_mask=text_attention_mask,
            speaker_embeds=speaker_embds,
            lyric_token_idx=lyric_token_ids,
            lyric_mask=lyric_mask,
            timestep=timesteps.to(device, dtype),
            ssl_hidden_states=all_ssl_hiden_states,
        )
        model_pred = transformer_output.sample
        proj_losses = transformer_output.proj_losses
        # Follow: Section 5 of https://arxiv.org/abs/2206.00364.
        # Preconditioning of the model outputs.
        model_pred = model_pred * (-sigmas) + noisy_image
        # Compute loss. Only calculate loss where chunk_mask is 1 and there is no padding
        # N x T x 64
        # N x T -> N x c x W x T
        mask = (
            attention_mask.unsqueeze(1)
            .unsqueeze(1)
            .expand(-1, target_image.shape[1], target_image.shape[2], -1)
        )

        # --- IMPACTFUL OPTIMIZATION 2: Improved and Potentially More Correct Loss Calculation ---
        # Ensure mask is float for multiplication
        # Reshape mask to match model_pred/target dimensions for element-wise multiplication
        # mask is [B, 1, 1, T], model_pred/target are [B, C, W, T]
        # Element-wise multiply to zero out masked areas
        masked_model_pred = model_pred * mask
        masked_target = target * mask

        # Calculate MSE loss, keeping reduction='none' to apply mask correctly
        # loss_per_element is [B, C, W, T]
        loss_per_element = F.mse_loss(masked_model_pred, masked_target, reduction='none')

        # Apply the mask to the loss (zero out loss where mask is 0)
        # masked_loss is [B, C, W, T]
        masked_loss = loss_per_element * mask

        # Sum the loss over all dimensions except batch to get loss per sample
        # loss_per_sample is [B]
        loss_per_sample = masked_loss.sum(dim=(1, 2, 3)) # Sum over C, W, T

        # Count the number of valid (non-masked) elements per sample
        # valid_elements_per_sample is [B]
        valid_elements_per_sample = mask.sum(dim=(1, 2, 3)) # Sum over C, W, T

        # Avoid division by zero: if no valid elements, loss should be 0 for that sample
        # Add a small epsilon or use torch.where
        epsilon = 1e-8
        # mean_loss_per_sample is [B]
        mean_loss_per_sample = loss_per_sample / (valid_elements_per_sample + epsilon)

        # Final loss is the mean across the batch
        loss = mean_loss_per_sample.mean()
        # --- END OF NEW LOSS CALCULATION ---

        prefix = "train"
        self.log(f"{prefix}/denoising_loss", loss, on_step=True, on_epoch=False)
        total_proj_loss = 0.0
        for k, v in proj_losses:
            self.log(f"{prefix}/{k}_loss", v, on_step=True, on_epoch=False)
            total_proj_loss += v
        if len(proj_losses) > 0:
            total_proj_loss = total_proj_loss / len(proj_losses)
        if self.hparams.ssl_coeff:
            loss += total_proj_loss * self.hparams.ssl_coeff
        self.log(f"{prefix}/loss", loss, on_step=True, on_epoch=False)
        # Log learning rate if scheduler exists
        if self.lr_schedulers() is not None:
            learning_rate = self.lr_schedulers().get_last_lr()[0]
            self.log(
                f"{prefix}/learning_rate", learning_rate, on_step=True, on_epoch=False
            )
        if self.hparams.optimizer == "prodigy":
            prodigy_d = self.optimizers().param_groups[0]["d"]
            self.log(f"{prefix}/prodigy_d", prodigy_d, on_step=True, on_epoch=False)
        return loss

    def training_step(self, batch, batch_idx):
        return self.run_step(batch, batch_idx)

    # Updated on_save_checkpoint logic
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if self.trainer.is_global_zero: # Ensure only rank 0 saves
            # Define the base directory for all LoRA checkpoints for this run
            # self.trainer.log_dir is set by WandbLogger to something like 'your_output_dir/wandb/run-YYYYMMDD_HHMMSS-run_id'
            # If using ModelCheckpoint's dirpath directly, use self.trainer.checkpoint_callback.dirpath
            # Since we want to save LoRA adapters in a specific structure relative to the run,
            # using self.trainer.log_dir from WandbLogger is appropriate here.
            # OR, if you want LoRA saved strictly where ModelCheckpoint saves, use:
            # lora_base_save_dir = os.path.join(self.trainer.checkpoint_callback.dirpath, "lora_adapters")
            # Using WandbLogger's save_dir for the LoRA adapters, which will be inside the wandb run folder
            lora_base_save_dir = os.path.join(self.trainer.log_dir, "lora_checkpoints")
            # Ensure the base directory for LoRA checkpoints exists
            os.makedirs(lora_base_save_dir, exist_ok=True)
            epoch = self.current_epoch
            step = self.global_step
            lora_name_folder = f"epoch={epoch}-step={step}_lora"
            # Full path where the current LoRA adapter will be saved
            current_lora_path = os.path.join(lora_base_save_dir, lora_name_folder)
            # Create the specific folder for this LoRA checkpoint
            os.makedirs(current_lora_path, exist_ok=True)
            # Save the LoRA adapter
            self.transformer.save_lora_adapter(
                current_lora_path, adapter_name=self.hparams.adapter_name
            )
            rank_zero_info(f"LoRA checkpoint saved to: {current_lora_path}") # Log the save path
            # Clean up old loras and only save the last few loras
            lora_paths = glob(os.path.join(lora_base_save_dir, "epoch=*-step=*_lora"))
            lora_paths = natsorted(lora_paths)
            if len(lora_paths) > self.hparams.save_last:
                # Remove the oldest LoRA checkpoint folder
                oldest_lora_to_remove = lora_paths[0]
                shutil.rmtree(oldest_lora_to_remove)
                rank_zero_info(f"Removed oldest LoRA checkpoint: {oldest_lora_to_remove}")
            # This line prevents PyTorch Lightning from saving the full model state
            # by clearing the checkpoint dictionary. Only the custom LoRA save will occur.
            # checkpoint.clear()
            # Instead of checkpoint.clear(), filter the state_dict
            state_dict = checkpoint.get("state_dict", {})
            keys_to_remove = [k for k in state_dict.keys() if "lora" not in k and "adapter" not in k]
            for k in keys_to_remove:
                del state_dict[k]

        # Now checkpoint['state_dict'] only contains LoRA weights.
        # The optimizer state remains in checkpoint['optimizer_states'], allowing proper resuming.
        # return checkpoint is not needed as of PyTorch Lightning 2.0+ as the hook return type is None
        # However, keeping it doesn't cause issues for older versions.

def main(args):
    # --- IMPACTFUL OPTIMIZATION 1: Optimize num_workers if not set appropriately ---
    # Ensure output_dir is created if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Override num_workers if it's still the problematic default (0) or too low
    # Only if it hasn't been explicitly set to a good value by the user/lora_magician_pro
    # A common heuristic is number of CPU cores, or a fraction thereof.
    # Let's assume if it's 0 (the old default), we change it.
    # If lora_magician_pro sets it, we respect that.
    if args.num_workers <= 0:
        try:
            import multiprocessing
            # Use a fraction of cores to leave some for the main process/GPU work
            # Aim for 2-4 workers per GPU core is common, but depends on I/O speed.
            # For local SSDs, higher num_workers often help. For network/HDD, might plateau.
            # Start with a simple heuristic.
            calculated_workers = max(1, min(8, multiprocessing.cpu_count() - 2)) # Example: cap at 8, min 1
            print(f"Info: num_workers was {args.num_workers}. Setting to {calculated_workers} based on CPU count.")
            args.num_workers = calculated_workers
        except Exception as e:
            print(f"Warning: Could not determine CPU count for num_workers. Keeping num_workers={args.num_workers}. Error: {e}")
            # Keep the original value (likely 0) if calculation fails
    else:
        print(f"Info: Using provided num_workers: {args.num_workers}")
    # --- END OF IMPACTFUL OPTIMIZATION 1 ---

    model = Pipeline(
        # Model
        checkpoint_dir=args.checkpoint_dir,
        shift=args.shift,
        lora_config_path=args.lora_config_path,
        last_lora_path=args.last_lora_path,
        # Data
        dataset_path=args.dataset_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers, # Pass the (potentially optimized) num_workers
        tag_dropout=args.tag_dropout,
        speaker_dropout=args.speaker_dropout,
        lyrics_dropout=args.lyrics_dropout,
        # Optimizer
        ssl_coeff=args.ssl_coeff,
        optimizer=args.optimizer,
        learning_rate=args.learning_rate,
        beta1=args.beta1,
        beta2=args.beta2,
        weight_decay=args.weight_decay, # Ensure this is passed
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        # Others
        adapter_name=args.exp_name, # Pass exp_name as adapter_name to Pipeline
        save_last=args.save_last,
        # --- IMPACTFUL OPTIMIZATION 4: Pass configurable slice_p ---
        prodigy_slice_p=args.prodigy_slice_p, # Pass the configurable argument
        # output_dir and wandb_run_id are not passed to Pipeline init
    )
    # Generate a unique run name for the specific experiment run
    from datetime import datetime
    import uuid
    run_name = f"{args.exp_name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:8]}"
    # --- START OF CRITICAL CHANGES ---
    # 1. Define ModelCheckpoint with a proper dirpath
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.output_dir, args.exp_name, run_name, "checkpoints"),
        filename="lora-full-model-{step:06}", # Name for the full model checkpoint (if not cleared in on_save_checkpoint)
        every_n_train_steps=args.save_every_n_train_steps, # UPDATED
        save_top_k=-1, # Saves all full model checkpoints (if not cleared)
        save_on_train_epoch_end=False, # UPDATED
        save_last=True
    )
    # --- END OF CRITICAL CHANGES ---
    logger_callback = WandbLogger(
        project="ace_step_lora",
        name=run_name, # Use the generated run_name for Wandb
        save_dir=args.output_dir, # Tell WandbLogger where to save its base directory
        id=args.wandb_run_id, # Pass the run ID to WandbLogger for resuming
        resume="allow" if args.wandb_run_id else None # Allow resuming if ID is provided
    )
    trainer = Trainer(
        accelerator="gpu",
        # strategy="ddp_find_unused_parameters_true", # Uncomment if needed for DDP
        # devices=args.devices, # Uncomment and configure if using specific devices
        # num_nodes=args.num_nodes, # Uncomment and configure if using multiple nodes
        precision=args.precision,
        log_every_n_steps=1,
        logger=logger_callback,
        # --- START OF CRITICAL CHANGES ---
        callbacks=[checkpoint_callback], # Pass the ModelCheckpoint callback here
        max_epochs=args.epochs,
        max_steps=args.max_steps,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val,
        gradient_clip_algorithm=args.gradient_clip_algorithm,
        enable_checkpointing=True, # CRITICAL: Re-enable PyTorch Lightning's default checkpointing
        default_root_dir=args.output_dir # Set trainer's root dir to your desired output
        # --- END OF CRITICAL CHANGES ---
    )
    # --- START OF CRITICAL CHANGE FOR RESUMING ---
    trainer.fit(model, ckpt_path=args.resume_full_ckpt_path)
    # --- END OF CRITICAL CHANGE FOR RESUMING ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--shift", type=float, default=3.0)
    parser.add_argument("--lora_config_path", type=str, default="./config/lora_config_transformer_only.json")
    parser.add_argument("--last_lora_path", type=str, default=None,
                        help="Path to a specific last_lora.safetensors to resume from. This is for initial loading, not continuous saving.")
    # --- START OF CRITICAL CHANGE FOR RESUMING ---
    parser.add_argument("--resume_full_ckpt_path", type=str, default=None,
                        help="Path to a full PyTorch Lightning checkpoint (.ckpt) to resume training from. This checkpoint saves the full training state (model, optimizer, scheduler, epoch, step).")
    # --- END OF CRITICAL CHANGE FOR RESUMING ---
    # Data
    parser.add_argument("--dataset_path", type=str, default=r"C:\data\audio_prep")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0) # Default remains 0, but main() optimizes it
    parser.add_argument("--tag_dropout", type=float, default=0.5)
    parser.add_argument("--speaker_dropout", type=float, default=0.05) # UPDATED
    parser.add_argument("--lyrics_dropout", type=float, default=0.05) # UPDATED
    # Optimizer
    parser.add_argument("--ssl_coeff", type=float, default=1.0)
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.99)
    parser.add_argument("--weight_decay", type=float, default=1e-2) # Added missing default for weight_decay
    parser.add_argument("--epochs", type=int, default=-1)
    parser.add_argument("--max_steps", type=int, default=5000) # UPDATED
    parser.add_argument("--warmup_steps", type=int, default=100) # UPDATED
    parser.add_argument("--accumulate_grad_batches", type=int, default=2) # UPDATED
    parser.add_argument("--gradient_clip_val", type=float, default=1.0)
    parser.add_argument("--gradient_clip_algorithm", type=str, default="norm")
    # Others
    # parser.add_argument("--devices", type=int, default=1) # Keep commented unless explicitly setting
    # parser.add_argument("--num_nodes", type=int, default=1) # Keep commented unless explicitly setting
    parser.add_argument("--exp_name", type=str, default="ace_step_lora")
    parser.add_argument("--precision", type=str, default="bf16-mixed")
    parser.add_argument("--save_every_n_train_steps", type=int, default=500) # UPDATED
    parser.add_argument("--save_last", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="./runs",
                        help="Base directory for all outputs (Wandb logs, LoRA checkpoints).")
    parser.add_argument("--wandb_run_id", type=str, default=None,
                        help="Wandb run ID to resume from. If provided, WandB will attempt to resume this run.")
    # --- IMPACTFUL OPTIMIZATION 4: Add argument for Prodigy slice_p ---
    parser.add_argument("--prodigy_slice_p", type=int, default=11, help="Slice parameter 'p' for Prodigy optimizer.")
    # --- END OF IMPACTFUL OPTIMIZATION 4 ---
    args = parser.parse_args()
    main(args)