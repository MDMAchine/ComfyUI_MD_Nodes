***

## **Master Technical Manual Template (v2.0 - Exhaustive)**

### **Node Name: `MD_ApplyTPG`**
**Display Name:** `MD: Apply TPG (Token Perturbation)`
**Category:** `MD_Nodes/Guidance`
**Version:** `1.3.0 (Release Candidate)`
**Last Updated:** `June 2025`

---

### **Table of Contents**

1.  **Introduction**
    1.1. Executive Summary
    1.2. Conceptual Category
    1.3. Problem Domain & Intended Application
    1.4. Key Features (Functional & Technical)
2.  **Core Concepts & Theory**
    2.1. Theoretical Background
    2.2. Mathematical & Algorithmic Formulation
    2.3. Data I/O Deep Dive
    2.4. Strategic Role in the ComfyUI Graph
3.  **Node Architecture & Workflow**
    3.1. Node Interface & Anatomy
    3.2. Input Port Specification
    3.3. Output Port Specification
    3.4. Workflow Schematics (Minimal & Advanced)
4.  **Parameter Specification**
    4.1. Core Activation Parameters
    4.2. Perturbation Control Parameters
    4.3. Heuristic & Stability Parameters
    4.4. Debugging & Determinism Parameters
5.  **Applied Use-Cases & Recipes**
    5.1. Recipe 1: High-Fidelity Generation (Paper Standard)
    5.2. Recipe 2: Rescuing "Fried" Images at High CFG
    5.3. Recipe 3: Troubleshooting Inverted Outputs
6.  **Implementation Deep Dive**
    6.1. Source Code Walkthrough
    6.2. Dependencies & External Calls
    6.3. Performance & Resource Analysis
    6.4. Tensor Lifecycle Analysis
7.  **Troubleshooting & Diagnostics**
    7.1. Error Code Reference
    7.2. Unexpected Visual Artifacts & Mitigation

---

### **1. Introduction**

#### **1.1. Executive Summary**
The **MD: Apply TPG** node is a model-patching utility that implements **Token Perturbation Guidance (TPG)**, a technique derived from the research paper *"Token Perturbation Guidance for Diffusion Models" (Rajabi et al., 2025)*. It intercepts the Self-Attention (`attn1`) layers of a diffusion model during the generation process. By selectively identifying and shuffling the tokens of the "Unconditional" (negative) prompt embedding, it forces the model to generate a more robust negative score, thereby improving image fidelity, reducing artifacts at high guidance scales, and better adhering to the positive prompt.

#### **1.2. Conceptual Category**
**Model Patcher / Guidance Modifier**
This node does not alter the model weights permanently. Instead, it injects a dynamic runtime patch (a "hook") into the model's execution graph, modifying how internal tensors are processed during inference.

#### **1.3. Problem Domain & Intended Application**
* **Problem Domain:** Standard Classifier-Free Guidance (CFG) relies on the difference between a Conditional prediction and an Unconditional prediction. Often, the Unconditional prediction retains too much semantic structure, causing the guidance to "fight" itself, leading to burn artifacts or rigid composition.
* **Intended Application (Use-Cases):**
    * **High CFG Scales:** Allowing users to push CFG > 7.0 without texture burnout.
    * **Enhanced Realism:** Improving FID (Fréchet Inception Distance) scores by smoothing out the negative guidance manifold.
    * **General Purpose:** A "set and forget" enhancement for SDXL, SD1.5, and DiT-based models.
* **Non-Application (Anti-Use-Cases):**
    * **LCM / Turbo:** Models that use extremely low step counts (1-4 steps) may not benefit significantly as the perturbation requires iterative refinement to manifest correctly.

#### **1.4. Key Features (Functional & Technical)**
* **Functional Features:**
    * **Targeted Layer Patching:** Allows application to specific UNet blocks (Down, Mid, Up) to balance structural vs. detailed guidance.
    * **Blendable Perturbation:** Offers a strength slider to interpolate between the original unconditional tokens and the fully shuffled version.
    * **Heuristic Fallbacks:** Includes manual overrides for batch splitting when metadata is missing.
* **Technical Features:**
    * **Deterministic Permutation:** Uses a bitwise XOR hashing strategy (`Seed ^ LayerID ^ TimeStep`) to ensure every layer and step gets a unique, reproducible shuffle.
    * **Type-Safe Sigma Extraction:** Capable of extracting diffusion timesteps (sigmas) from various ComfyUI wrapper objects without forcing unnecessary CPU syncs.
    * **Device Fallback:** Automatically degrades to CPU-based random generation if CUDA/MPS generators fail (crucial for Apple Silicon compatibility).

### **2. Core Concepts & Theory**

#### **2.1. Theoretical Background**
In standard diffusion, the Unconditional estimate ($\epsilon_\theta(x_t, \emptyset)$) serves as a baseline. However, this baseline often contains specific semantic layouts that shouldn't be there. TPG posits that the "Unconditional" signal should represent pure noise statistics without semantic coherence.
By **shuffling the tokens** along the sequence dimension, TPG destroys the semantic relationships (e.g., "cat" is next to "sitting") while preserving the statistical distribution of the tokens. This creates a "harder" negative sample, making the CFG vector point more strongly toward the desired image.

#### **2.2. Mathematical & Algorithmic Formulation**
Let $Q_{uncond}$ be the Query tensor in the self-attention mechanism for the unconditional batch.
The operation is defined as:
$$ Q_{shuffled} = \text{Permute}(Q_{uncond}, \text{dim}=SequenceLength) $$
The final modified query $Q'_{uncond}$ is a linear interpolation:
$$ Q'_{uncond} = (1 - \alpha) \cdot Q_{uncond} + \alpha \cdot Q_{shuffled} $$
Where $\alpha$ is the `perturbation_strength`.

#### **2.3. Data I/O Deep Dive**
* **Input `model`:** A ComfyUI Model Wrapper (class `comfy.model_base.BaseModel`).
* **Output `patched_model`:** A shallow copy (`clone`) of the input model with a function appended to its internal `set_model_attn1_patch` list.
* **Internal Tensor Flow:**
    * **Input:** $Q$ tensor `[Batch, Tokens, Channels]`.
    * **Operation:** Split Batch $\rightarrow$ Identify Uncond $\rightarrow$ Shuffle Tokens $\rightarrow$ Recombine.
    * **Output:** Modified $Q$ tensor `[Batch, Tokens, Channels]`.

#### **2.4. Strategic Role in the ComfyUI Graph**
* **Placement Context:** Must be placed **before** the Sampler (KSampler/SamplerCustom).
* **Synergistic Nodes:** Works well with `PerturbedAttentionGuidance` (PAG) if tuned carefully, though they touch similar mechanisms.
* **Conflicting Nodes:** Other nodes that patch `attn1` specifically to modify the query geometry might conflict if chained improperly.

### **3. Node Architecture & Workflow**

#### **3.1. Node Interface & Anatomy**
The node is designed to be inserted into the main model link.

#### **3.2. Input Port Specification**
* **`model` (MODEL)** - (Required)
    * **Description:** The diffusion model to be patched.
    * **Impact:** The patch is applied to a clone; the original model connection remains untouched elsewhere.
* **`enable_tpg` (BOOLEAN)** - (Required)
    * **Default:** `True`
    * **Description:** Master toggle switch.
    * **Impact:** If False, passes the model through without cloning or patching (0 overhead).
* **`target_layers` (COMBO)** - (Required)
    * **Options:** `Down (Encoder)`, `Mid (Bottleneck)`, `Up (Decoder)`, `All`.
    * **Default:** `Down (Encoder)`.
    * **Impact:** Determines which UNet/DiT blocks receive the patch. The paper suggests **Down** blocks are most critical for setting the semantic structure.
* **`perturbation_strength` (FLOAT)** - (Required)
    * **Range:** 0.0 to 1.0.
    * **Default:** 1.0.
    * **Impact:** Controls the mix ratio. 1.0 is full replacement with shuffled tokens. Lower values retain some original semantic structure in the negative prompt.
* **`split_mode` (COMBO)** - (Required)
    * **Options:** `Uncond First (Standard)`, `Cond First (Inverted)`.
    * **Default:** `Uncond First`.
    * **Description:** Used only if the internal metadata mask is missing. Determines which half of the batch is treated as the "Negative" to be shuffled.
* **`seed` (INT)** - (Required)
    * **Description:** Base seed for the random permutation generator.
* **`debug_mode` (BOOLEAN)** - (Required)
    * **Default:** `False`.
    * **Description:** Enables console logging of sigma values, layer IDs, and batch split statistics.

#### **3.3. Output Port Specification**
* **`patched_model` (MODEL)**
    * **Description:** The model with the TPG hook registered.

#### **3.4. Workflow Schematics**

**Minimal Functional Graph:**
`[Load Checkpoint] -> [MD_ApplyTPG] -> [KSampler]`

**Advanced Standard Graph:**
`[Load Checkpoint] -> [LoRA Loader] -> [MD_ApplyTPG (Down, Str: 1.0)] -> [KSampler]`

```json
{
  "class_type": "MD_ApplyTPG",
  "inputs": {
    "model": ["10", 0],
    "enable_tpg": true,
    "target_layers": "Down (Encoder)",
    "perturbation_strength": 1.0,
    "seed": 0
  }
}
```

### **4. Parameter Specification**

#### **4.1. Core Activation Parameters**
* **`enable_tpg`**:
    * **Algorithmic Impact:** Immediate circuit break. Returns input `model` tuple if False.
* **`target_layers`**:
    * **Map:**
        * "Down (Encoder)" $\rightarrow$ blocks named `input`.
        * "Mid (Bottleneck)" $\rightarrow$ blocks named `middle`.
        * "Up (Decoder)" $\rightarrow$ blocks named `output`.
    * **Rationale:** The TPG paper demonstrates that perturbing the Encoder (Down) blocks yields the highest FID improvement because these layers interpret the prompt concepts.

#### **4.2. Perturbation Control Parameters**
* **`perturbation_strength`**:
    * **Internal Variable:** `alpha` (implicit in `torch.lerp`).
    * **Default:** 1.0.
    * **Behavior:**
        * At 1.0: $Q'_{uncond} = Q_{shuffled}$.
        * At 0.5: Average of original and shuffled.
    * **Visual-Technical Correlation:** Higher values produce stronger adherence to the positive prompt and higher contrast, potentially at the cost of "smoothness".

#### **4.3. Heuristic & Stability Parameters**
* **`split_mode`**:
    * **Purpose:** ComfyUI usually sends batches as `[Cond, Uncond]` or `[Uncond, Cond]`. The node *tries* to read a metadata mask to know for sure. If that mask is missing (common in some advanced samplers), it assumes the batch is 50/50.
    * **Logic:**
        * "Uncond First": Assumes the *first* half of the batch is the negative prompt (Target for shuffling).
        * "Cond First": Assumes the *second* half is the negative prompt.
    * **Critical Note:** If you select the wrong one, you shuffle the **Positive** prompt, destroying your image.

#### **4.4. Debugging & Determinism Parameters**
* **`seed`**:
    * **Bitwise Logic:** To ensure the shuffle changes *every step* and *every layer*, the code calculates:
        `current_seed = (base_seed ^ layer_hash ^ (sigma * 1000))`
    * **Constraint:** Capped at $2^{63}-1$ to prevent PyTorch C++ overflow errors.

### **5. Applied Use-Cases & Recipes**

#### **5.1. Recipe: High-Fidelity Generation (Paper Standard)**
* **Objective:** Maximize image quality and prompt adherence.
* **Configuration:**
    * `enable_tpg`: True
    * `target_layers`: "Down (Encoder)"
    * `perturbation_strength`: 1.0
    * `split_mode`: "Uncond First"
* **Rationale:** Perturbing the Down blocks prevents the model from latching onto semantic structures in the empty prompt, allowing the positive prompt to guide the structure more purely.

#### **5.2. Recipe: Rescuing "Fried" Images at High CFG**
* **Objective:** Use a CFG scale of 15.0 or 20.0 without the image turning into deep-fried noise.
* **Configuration:**
    * `target_layers`: "All" (Stronger intervention)
    * `perturbation_strength`: 0.8 (Slight blend to maintain stability)
* **Rationale:** High CFG amplifies artifacts in the negative prediction. Shuffling the negative tokens "whitewashes" these artifacts, making the negative prediction a smoother uniform noise floor.

#### **5.3. Recipe: Troubleshooting Inverted Outputs**
* **Objective:** Fix generation if the image looks like noise or completely unrelated concepts.
* **Configuration:**
    * `split_mode`: Switch from "Uncond First" to "Cond First".
* **Rationale:** Some custom sampling nodes invert the batch order. If TPG shuffles the wrong half, it destroys the user's prompt.

### **6. Implementation Deep Dive**

#### **6.1. Source Code Walkthrough**
The patch is implemented in the `tpg_attn1_patch` closure.

```python
import torch
import logging
import comfy.model_management

class MD_ApplyTPG:
    """
    Applies Token Perturbation Guidance (TPG) by patching the model's Self-Attention.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The diffusion model to patch."}),
                "enable_tpg": ("BOOLEAN", {"default": True}),
                "target_layers": (["Down (Encoder)", "Mid (Bottleneck)", "Up (Decoder)", "All"], {
                    "default": "Down (Encoder)",
                    "tooltip": "Paper recommends 'Down (Encoder)' for best FID scores."
                }),
                "perturbation_strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Blend factor: 1.0 = Full Shuffle, 0.0 = No effect."
                }),
                "split_mode": (["Uncond First (Standard)", "Cond First (Inverted)"], {
                    "default": "Uncond First (Standard)",
                    "tooltip": "HEURISTIC FALLBACK:\nUse if TPG seems to invert/break generation.\nMost samplers are [Uncond, Cond] (Select 'Uncond First')."
                }),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xffffffffffffffff,
                    "tooltip": "Base seed. Actual permutation varies deterministically by step and layer."
                }),
                "debug_mode": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Print tensor shapes and split logic to console (Rate Limited)."
                }),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("patched_model",)
    FUNCTION = "apply_tpg"
    CATEGORY = "MD_Nodes/Guidance"

    def apply_tpg(self, model, enable_tpg, target_layers, perturbation_strength, split_mode, seed, debug_mode):
        
        if not enable_tpg or perturbation_strength <= 0.0:
            return (model,)

        m = model.clone()

        # --- Parse Layer Targets ---
        # ComfyUI Block Names: 'input' (Down), 'middle' (Mid), 'output' (Up)
        target_blocks = []
        if "Down" in target_layers or "All" in target_layers: target_blocks.append("input")
        if "Mid" in target_layers or "All" in target_layers:  target_blocks.append("middle")
        if "Up" in target_layers or "All" in target_layers:   target_blocks.append("output")

        # --- Patch Function ---
        def tpg_attn1_patch(q, k, v, extra_options):
            try:
                # 1. Safety Validation
                if q.ndim != 3:
                    # TPG expects [Batch, Tokens, Dim]. If 4D/5D, skip.
                    return q, k, v

                # 2. Layer Filtering
                block_info = extra_options.get("block", None) # Tuple: ("input", 1)
                
                if block_info and len(block_info) > 1:
                    layer_id_salt = block_info[1]
                    block_type = block_info[0]
                elif block_info:
                    layer_id_salt = hash(str(block_info)) % 1000
                    block_type = str(block_info)
                else:
                    layer_id_salt = 0
                    block_type = "unknown"

                if block_type not in target_blocks:
                    return q, k, v 

                # 3. Debug Rate Limiting (Early init for tracking)
                should_print = False
                if debug_mode:
                    if not hasattr(tpg_attn1_patch, "_call_count"):
                        tpg_attn1_patch._call_count = 0
                    tpg_attn1_patch._call_count += 1
                    if tpg_attn1_patch._call_count % 20 == 0:
                        should_print = True

                # 4. Robust Sigma/Time Extraction
                transformer_options = extra_options.get("transformer_options", {})
                current_sigma = 0.0
                
                sigmas = transformer_options.get("sigmas", None)
                if sigmas is not None:
                    if isinstance(sigmas, torch.Tensor) and sigmas.numel() > 0:
                        # Optimization: Avoid CPU transfer if possible, but float cast usually forces it.
                        # Accessing flattened index 0 is safe.
                        current_sigma = float(sigmas.flatten()[0])
                    elif isinstance(sigmas, (list, tuple)) and len(sigmas) > 0:
                        current_sigma = float(sigmas[0])
                
                if current_sigma == 0.0 and "step" in transformer_options:
                    current_sigma = float(transformer_options["step"])

                # 5. Bitwise Seeding (Overflow Safe)
                time_salt = int(current_sigma * 1000)
                limit = 2**63 - 1
                current_seed = (seed ^ layer_id_salt ^ time_salt) & limit

                # 6. Batch Identification (Uncond vs Cond)
                cond_map = transformer_options.get("cond_or_uncond", None)
                q_out = q
                
                # --- Path A: Precise Masking (Preferred) ---
                if cond_map is not None and len(cond_map) == q.shape[0]:
                    if isinstance(cond_map, torch.Tensor):
                        is_uncond = (cond_map == 0).to(q.device)
                    else:
                        is_uncond = torch.tensor([x == 0 for x in cond_map], device=q.device)
                    
                    if is_uncond.any():
                        q_uncond = q[is_uncond]
                        
                        # Device-Safe Generator
                        gen = torch.Generator(device=q.device)
                        try:
                            gen.manual_seed(current_seed)
                            B_u, T, C = q_uncond.shape
                            perm_indices = torch.randperm(T, generator=gen, device=q.device)
                        except RuntimeError:
                            # Fallback for older PyTorch/MPS where Generator(device=cuda) might fail
                            gen = torch.Generator(device='cpu')
                            gen.manual_seed(current_seed)
                            perm_indices = torch.randperm(T, generator=gen).to(q.device)

                        q_shuffled = q_uncond[:, perm_indices, :]
                        
                        if perturbation_strength < 1.0:
                            q_uncond_final = torch.lerp(q_uncond, q_shuffled, perturbation_strength)
                        else:
                            q_uncond_final = q_shuffled
                        
                        # Selective Clone for Memory Safety
                        q_out = q.clone()
                        q_out[is_uncond] = q_uncond_final
                        
                        if should_print:
                             print(f"[MD_TPG] Masked | Block: {block_type}_{layer_id_salt} | "
                                   f"Sigma: {current_sigma:.3f} | Uncond: {is_uncond.sum().item()}/{q.shape[0]}")

                # --- Path B: Heuristic Chunking (Fallback) ---
                else:
                    if q.shape[0] % 2 == 0:
                        chunk_size = q.shape[0] // 2
                        q1, q2 = q.chunk(2, dim=0)
                        
                        if split_mode == "Uncond First (Standard)":
                            target_q = q1
                            other_q = q2
                            is_q1_target = True
                        else: 
                            target_q = q2
                            other_q = q1
                            is_q1_target = False
                        
                        # Device-Safe Generator (Repeated logic)
                        gen = torch.Generator(device=q.device)
                        try:
                            gen.manual_seed(current_seed)
                            B_u, T, C = target_q.shape
                            perm_indices = torch.randperm(T, generator=gen, device=q.device)
                        except RuntimeError:
                            gen = torch.Generator(device='cpu')
                            gen.manual_seed(current_seed)
                            perm_indices = torch.randperm(T, generator=gen).to(q.device)

                        q_shuffled = target_q[:, perm_indices, :]
                        
                        if perturbation_strength < 1.0:
                            target_final = torch.lerp(target_q, q_shuffled, perturbation_strength)
                        else:
                            target_final = q_shuffled
                        
                        if is_q1_target:
                            q_out = torch.cat([target_final, other_q], dim=0)
                        else:
                            q_out = torch.cat([other_q, target_final], dim=0)
                        
                        if should_print:
                             print(f"[MD_TPG] Heuristic | Block: {block_type}_{layer_id_salt} | "
                                   f"Sigma: {current_sigma:.3f} | Mode: {split_mode}")
                    else:
                        if should_print:
                            print(f"[MD_TPG] Odd batch {q.shape[0]} - Skipping.")
                        return q, k, v

                return q_out, k, v

            except Exception as e:
                print(f"[MD_ApplyTPG] Error in patch: {e}")
                return q, k, v

        m.set_model_attn1_patch(tpg_attn1_patch)
        
        logging.info(f"[MD_ApplyTPG] Patch active. Layers: {target_layers}, Str: {perturbation_strength}")
        return (m,)

NODE_CLASS_MAPPINGS = {
    "MD_ApplyTPG": MD_ApplyTPG,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MD_ApplyTPG": "MD: Apply TPG (Token Perturbation)",
}
```

#### **6.2. Dependencies & External Calls**
* **`torch`**: Essential for tensor manipulation, generators, and linear interpolation.
* **`comfy.model_management`**: Imported but primarily used for context; the patching mechanism relies on the standard `comfy.model_base.BaseModel.set_model_attn1_patch` API.

#### **6.3. Performance & Resource Analysis**
* **Execution Target:** GPU (same device as the model).
* **Benchmarking:**
    * The overhead is negligible. `torch.randperm` is extremely fast.
    * Memory usage increases slightly during the operation due to `q.clone()` required to safely mutate a subset of the tensor (Path A) or `torch.cat` (Path B).
* **VRAM Scaling:** No permanent VRAM increase. Transient VRAM usage is roughly equal to the size of the Query tensor $Q$ for the current layer.

#### **6.4. Tensor Lifecycle Analysis**
1.  **Stage 1 (Intercept):** `q`, `k`, `v` arrive at `attn1`. `q` shape: `[Batch, Tokens, Dim]`.
2.  **Stage 2 (Identification):** The code checks `cond_or_uncond`.
    * *Path A:* Boolean mask identifies `is_uncond`.
    * *Path B:* Tensor is chunked `q1, q2`.
3.  **Stage 3 (Perturbation):**
    * `perm_indices` generated via `torch.randperm` (seeded by step/layer).
    * `q_shuffled = q_target[:, perm_indices, :]`.
4.  **Stage 4 (Recombination):**
    * The perturbed `q` is merged back into the full batch.
5.  **Stage 5 (Return):** The modified `q` is returned to the attention mechanism. $K$ and $V$ are untouched.

### **7. Troubleshooting & Diagnostics**

#### **7.1. Error Code Reference**
* **`RuntimeError: Generator device not found`**:
    * **Root Cause:** Older PyTorch versions or specific MPS (Mac) environments often fail when initializing a `torch.Generator` directly on the device.
    * **Solution:** The code (v1.3.0) includes a `try/except` block to fall back to a CPU generator automatically. No user action required.
* **`[MD_TPG] Odd batch X - Skipping`**:
    * **Root Cause:** The batch size is odd (e.g., 1), meaning there is no clear Cond/Uncond pair, and metadata was missing.
    * **Solution:** TPG disables itself for this step to prevent breaking the single generation.

#### **7.2. Unexpected Visual Artifacts & Mitigation**
* **Artifact:** Image becomes pure noise / "Fried".
    * **Likely Cause:** `split_mode` is incorrect. You are shuffling the Positive prompt instead of the Negative.
    * **Correction:** Toggle `split_mode`.
* **Artifact:** Composition becomes disjointed or "scrambled".
    * **Likely Cause:** `perturbation_strength` is 1.0 on `All` layers.
    * **Correction:** Set `target_layers` to `Down (Encoder)` only, or reduce strength to 0.7.