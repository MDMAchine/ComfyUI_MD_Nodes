PingPong Sampler (Custom V0.8.15) ComfyUI Node Manual
Welcome to the manual for the PingPong Sampler, a custom ComfyUI node specifically optimized for Ace-Step audio and video diffusion models. While its primary purpose is high-quality audio/video generation, it's also a powerful tool for visual experimentation. This manual will guide you through its features, parameters, and inner workings.

1. Understanding the PingPong Sampler Node
The PingPong Sampler is a specialized sampling node for ComfyUI. Unlike standard samplers that often follow a fixed path through the denoising process, the PingPong Sampler employs a "ping-pong" strategy, intelligently alternating between denoising your latent data and injecting carefully controlled ancestral noise. This nuanced approach is designed to produce results with distinct characteristics, especially beneficial for time-series data like audio and video.

How it Works
At its core, a diffusion model refines noisy data step by step, gradually transforming it into a clean output. Samplers dictate how this transformation occurs. The PingPong Sampler enhances this process by:

Denoising: At each step, it asks the underlying diffusion model to predict a cleaner version of the current noisy latent.

Ancestral Noise Injection (Ping-Pong Action): For a configurable range of steps (the "ancestral steps"), instead of just using the denoised prediction directly, it strategically adds a fresh burst of random noise back into the latent. This "ping-pong" effect creates a dynamic interaction between the model's prediction and new randomness, influencing the texture and detail of the final output.

Non-Ancestral Steps: Outside of this "ping-pong" range, it progresses more like a standard denoising sampler, smoothly interpolating towards the denoised state.

This unique noise handling is why it excels with Ace-Step models, as it closely aligns with how they expect noise to be introduced and processed across frames or audio segments.

What it Does
This node helps you generate:

High-Quality Audio/Video: Its primary strength lies in creating stable and coherent audio and video sequences using models like Ace-Step diffusion and likley video models as well. Expect intricate textures and smooth transitions.

Experimental Images: While not its primary focus, you can certainly experiment with image generation. However, due to its specialized noise handling, results might be unexpected or uniquely stylized compared to samplers designed purely for still images.

How to Use It
To use the PingPong Sampler, you'll typically integrate it into your ComfyUI workflow like any other sampler:

Add the Node: Find "PingPong Sampler (Custom V0.8.15)" in the sampling/custom_sampling/samplers category.

Connect Inputs:

Connect your MODEL to the model input.

Connect your POSITIVE and NEGATIVE conditionings.

Connect a LATENT_IMAGE to the latent_image input.

Crucially, connect a SCHEDULER node (like a KSamplerScheduler) to the scheduler input. This defines the noise decay curve and is essential for proper sampling progression.

Configure Parameters: Adjust the various parameters described in the next section to fine-tune the sampling behavior.

Connect to KSampler: The output of this node (SAMPLER) will then connect to the sampler input of a KSampler node.

By adjusting the "ping-pong" range, random modes, and blending options, you can profoundly influence the character of your generated outputs, whether they are auditory or visual.

2. Detailed Parameter Information
The PingPong Sampler node exposes several parameters in the ComfyUI interface, allowing you to fine-tune its behavior.

Parameter

Type

Default

Description & Use Cases

step_random_mode

STRING

"block"

Controls how the RNG seed varies per sampling step.

step_size

INT

4

Used by block and reset step random modes to define the block/reset interval for the seed.

seed

INT

80085

Base random seed. The cosmic initializer. Change it for new universes, keep it for deja vu.

first_ancestral_step

INT

0

The sampler step index (0-based) to begin ancestral noise injection (ping-pong behavior). Use -1 to effectively disable ancestral noise if last_ancestral_step is also -1.

last_ancestral_step

INT

-1

The sampler step index (0-based) to end ancestral noise injection (ping-pong behavior). Use -1 to extend ancestral noise to the end of the sampling process.

start_sigma_index

INT

0

The index in the sigma array (denoising schedule) to begin sampling from. Allows skipping initial high-noise steps, potentially speeding up generation or changing visual character.

end_sigma_index

INT

-1

The index in the sigma array to end sampling at. -1 means sample all steps. To the bitter end, or a graceful exit? You decide.

enable_clamp_output

BOOLEAN

False

If true, clamps the final output latent to the range [-1.0, 1.0]. Useful for preventing extreme values that might lead to artifacts during decoding.

scheduler

SCHEDULER

(Required connection)

Connect a ComfyUI scheduler node (e.g., KSamplerScheduler) to define the noise decay curve for the sampler. Essential for proper sampling progression. It's the tempo track for your pixels!

blend_mode

STRING

"lerp"

Blend mode to use for blending noise in ancestral steps. Defaults to 'lerp' (linear interpolation). Choose your flavor: 'lerp' (smooth blend), 'a_only' (take noise as is), 'b_only' (take other input as is).

step_blend_mode

STRING

"lerp"

Blend mode to use for non-ancestral steps (regular denoising progression). Changing this from 'lerp' is generally not recommended unless you're feeling particularly chaotic.

yaml_settings_str

STRING

""

Optional YAML string to configure sampler parameters. Parameters provided directly via the ComfyUI node's inputs will now be OVERRIDDEN by any corresponding values set in the YAML string. If the YAML is empty, node inputs are used. YAML is the boss now, respect its authority!

Parameter Deep Dive:

Randomness (step_random_mode, step_size, seed): These parameters give you granular control over how the random noise (used in ancestral steps) behaves across your generation.

off: Provides perfect reproducibility for a given seed. Use this when you want identical results every time.

block: Creates patterns that repeat every step_size frames. Useful for rhythmic or structured variations.

reset: Offers larger, more unpredictable jumps in randomness. Good for breaking up visual monotony or injecting bursts of varied detail.

step: Introduces subtle, incremental changes in noise per step. Ideal for adding a touch of organic variation without drastic shifts.

Ancestral Control (first_ancestral_step, last_ancestral_step): These define the "ping-pong" window.

Setting a specific range allows you to apply the unique noise injection only during certain phases of the denoising process. For example, you might want more ancestral noise at the beginning for overall texture, or at the end for fine detail.

Using -1 for last_ancestral_step extends the ancestral behavior to the very end of the sampling, while -1 for first_ancestral_step (when last_ancestral_step is also -1) can effectively disable ancestral noise entirely.

Sampling Range (start_sigma_index, end_sigma_index): These are powerful for advanced control over the denoising schedule.

start_sigma_index: Skip initial steps. This can speed up generation by cutting out very noisy early steps, or shift the overall "feel" by starting the process from a less noisy state.

end_sigma_index: Stop early. Useful for deliberately creating "under-denoised" results, which can have a more abstract or painterly quality.

Output Control (enable_clamp_output): Prevents extreme pixel values. Diffusion models can sometimes produce values outside the standard [-1, 1] range, which can lead to visual artifacts or clipping when decoded. Clamping ensures your output stays within expected bounds.

Blending (blend_mode, step_blend_mode): These define how the sampler combines information at each step.

lerp (Linear Interpolation): This is the standard, smooth way to blend between the denoised prediction and the current latent state (or noise in ancestral steps). It's generally recommended for most stable results.

a_only: For ancestral steps, this would mean primarily taking the denoised output, largely ignoring the added noise. For non-ancestral, it would lean heavily on the denoised prediction.

b_only: For ancestral steps, this would mean primarily taking the raw noise. For non-ancestral, it would retain much of the previous noisy state.

Experimenting with a_only or b_only for blend_mode (ancestral steps) can lead to unique, often more chaotic or abstract results, as it changes how the new noise interacts. Changing step_blend_mode is generally not recommended as it controls the fundamental progression of non-ancestral denoising.

YAML Settings (yaml_settings_str): This is for advanced users who prefer programmatic control or want to easily share complex configurations.

You can paste a YAML string directly into this field. Any parameter specified in the YAML will override the corresponding setting directly on the node. This allows for powerful "preset" management or more intricate configurations not easily exposed via individual sliders.

3. In-depth Nerd Technical Information
For those who like to peek under the hood and understand the silicon-level operations, here's a deeper dive into the PingPong Sampler's technical architecture.

The Core: PingPongSampler Class
The actual sampling logic resides within the PingPongSampler class. This class is instantiated internally by the ComfyUI node wrapper (PingPongSamplerNode) and then executed by ComfyUI's KSAMPLER mechanism.

__init__ (Initialization):

It receives the core model, initial x (latent), and sigmas (noise schedule) from ComfyUI.

It carefully parses parameters from kwargs (keyword arguments), which are populated from the ComfyUI node's inputs. This pop() operation is crucial to avoid argument duplication errors.

Ancestral Step Boundaries: first_ancestral_step and last_ancestral_step are robustly handled: they are internally clamped to ensure they are always within the valid range of available steps (0 to num_steps_available - 1), even if the user inputs them in inverted order or outside bounds.

is_rf (CONST Model Detection): The sampler attempts to detect if the underlying diffusion model is a "CONST" type (e.g., from certain Reflow or Consistency models, often found in Stable Audio). This detection involves traversing inner_model wrappers. If model_sampling is an instance of model_sampling.CONST, self.is_rf is set to True. This flag dictates a slightly different ancestral update formula.

Noise Sampler: A default_noise_sampler is defined if no external one is provided. Crucially, this default now always returns raw, unconditioned torch.randn_like noise, ensuring unit variance. All specific scaling for ancestral steps happens after this raw noise generation.

Noise Decay: It integrates with external SCHEDULER nodes. If a scheduler is provided and has a get_decay method, it fetches a pre-calculated noise_decay array. Otherwise, it defaults to a zero array, effectively disabling external noise decay.

_model_denoise: A simple wrapper that prepares the sigma scalar into a tensor and combines extra_args before calling the core diffusion model_ to get the denoised prediction.

_do_callback: This method is vital for ComfyUI's user experience, feeding i (step index), x (current latent), sigma (current noise level), and denoised (model's prediction) back to the UI for progress updates and intermediate image previews.

__call__ (The Main Sampling Loop): This is where the magic happens.

It iterates through the sigmas schedule using tqdm.auto.trange for a progress bar.

Step Skipping: It respects start_sigma_index and end_sigma_index, allowing for truncated sampling runs.

Denoising: denoised_sample = self._model_denoise(x_current, sigma_current) is called in every iteration to get the model's prediction.

Final Clamping Logic: enable_clamp_output is applied only at very low sigma_next values (currently < 1e-3). This prevents premature clamping of noisy intermediate latents, preserving important detail throughout most of the sampling process. If sigma_next is effectively zero (<= 1e-6), the loop breaks, as the final denoised state is reached.

Ancestral Step Determination (use_anc): It checks if the current idx falls within the first_ancestral_step and last_ancestral_step range to decide whether to apply ancestral noise.

Non-Ancestral Branch: If use_anc is False, it performs a simple interpolation between the denoised sample and the current noisy state, using self.step_blend_function (usually torch.lerp). This is akin to a DDIM-like step.

Ancestral Branch (Noise Injection):

Stepped Seed: _stepped_seed(idx) is called to determine the RNG seed based on step_random_mode and step_size, ensuring controlled variability in noise. torch.manual_seed() is applied locally for this step.

Raw Noise Generation: noise_sample = self.noise_sampler(sigma_current, sigma_next) is called to generate raw, unit-variance Gaussian noise.

Critical Noise Application: Based on extensive user feedback for Ace-Step compatibility, the code explicitly removes s_noise and dynamic_noise scaling from ancestral steps. The final_noise_to_add is simply the raw noise_sample. The required scaling for the ancestral process is now handled by the sigma_next term in the ancestral update formulas themselves, or by the blend function, ensuring precise noise magnitudes as expected by Ace-Step models.

Ancestral Update Formula:

If self.is_rf is True (CONST model): x_current = self.step_blend_function(denoised_sample, final_noise_to_add, sigma_next)

If self.is_rf is False (Other models): x_current = denoised_sample + final_noise_to_add * sigma_next
This adheres to the original blepping sampler's behavior, which was found to work best with Ace-Step.

The Node Wrapper: PingPongSamplerNode Class
This class is the interface between the core PingPongSampler logic and the ComfyUI environment.

INPUT_TYPES: Defines all the parameters you see in the ComfyUI node. It specifies their data types, default values, min/max ranges, and tooltips.

get_sampler: This is the method ComfyUI calls.

It first gathers all the direct inputs from the node's UI.

YAML Override Logic: It then attempts to parse the yaml_settings_str. If valid YAML is found and contains parameters that also exist as direct node inputs, the YAML values take precedence and override the direct node inputs. This means you can paste a complex YAML configuration to instantly change multiple settings, acting as a powerful preset manager. Boolean values from YAML strings are automatically converted to Python booleans.

It resolves the blend_mode and step_blend_mode strings into actual torch functions (torch.lerp, or custom lambda functions for a_only/b_only) using the _INTERNAL_BLEND_MODES dictionary.

Finally, it constructs and returns a KSAMPLER object, passing PingPongSampler.go as the sampling function and all the resolved parameters (after YAML merging) as extra_options (which are then unpacked as **kwargs into PingPongSampler.go).

Internal Blend Modes
The _INTERNAL_BLEND_MODES dictionary is a simple lookup table that maps user-friendly string names ("lerp", "a_only", "b_only") to their corresponding Python functions. This allows for flexible blending behavior without requiring complex logic directly in the UI.

In summary, the PingPong Sampler is a finely tuned instrument for diffusion sampling, particularly adept at handling the temporal coherence required by audio and video models. Its internal logic, especially regarding noise injection and the "ping-pong" ancestral steps, is meticulously crafted to produce consistent and high-quality results.