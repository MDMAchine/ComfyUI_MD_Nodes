Advanced Audio Preview & Save (AAPS) ComfyUI Node Manual
1. What is the Advanced Audio Preview & Save Node?
The Advanced Audio Preview & Save (AAPS) node is your all-in-one solution for managing audio within ComfyUI workflows. It acts as a digital sound engineer, allowing you to process, preview, and save generated audio with advanced controls.

How it Works:
The AAPS node takes raw audio data from other ComfyUI nodes (like VAEAUDIO_Decode or any other audio-producing node) through a standardized "AUDIO" input. Once it receives the audio, it performs several key functions:

Real-time Audio Preview: It outputs the processed audio in a format compatible with ComfyUI's built-in "Preview Audio" node, allowing you to listen to your creations directly within your workflow.

Intelligent Audio Saving: You can save your audio masterpieces to disk in various high-quality formats: FLAC (lossless), MP3 (universally compatible), or OPUS (efficient and high-quality for web/streaming).

Workflow Metadata Embedding: A powerful feature of AAPS is its ability to automatically embed your entire ComfyUI workflow (the "blueprint" of how you created the audio) directly into the audio file's hidden metadata. This means you can later drag the saved audio file back into ComfyUI, and your workflow will magically reappear, enabling seamless recreation or sharing of your audio generation process.

Visual Waveform Generation: The node also generates a beautiful waveform image of your audio. This visual representation of your sound is outputted as a standard ComfyUI "IMAGE" tensor, which you can connect to any "Preview Image" or "Save Image" node to visually inspect your sonic art. This is incredibly useful for debugging or simply admiring your sound waves.

What it Does (Key Features):

Audio Output for Preview: Connects directly to ComfyUI's 'Preview Audio' node.

Waveform Image Output: Generates a visual waveform ready for 'Preview Image' or 'Save Image' nodes.

Save to Disk Control: Toggle saving on or off, with format choices (FLAC, MP3, OPUS).

Metadata Privacy Filter: A dedicated toggle to control embedding of all metadata, including workflow, prompt, and custom notes.

Custom Notes: Add personal text notes directly into the audio file's metadata.

Audio Normalization: Automatically adjusts audio loudness to prevent clipping and ensure consistent volume.

Customizable Waveform Appearance: Choose your preferred line and background colors for the waveform image.

Customizable Waveform Dimensions: Control the exact width and height of the generated waveform image in pixels.

How to Use It:

Connect Audio Input: Drag a connection from the 'AUDIO' output of your audio source node (e.g., VAEAUDIO_Decode, EQ node) to the audio_input of the AAPS node.

Connect for Preview: Connect the AAPS node's 'AUDIO' output to a 'Preview Audio' node to listen to your sound.

Connect Waveform Image: Connect the AAPS node's 'WAVEFORM_IMAGE' output to a 'Preview Image' node (or 'Save Image') to see your waveform.

Configure Settings: Adjust the parameters (explained in the next section) such as filename_prefix, save_format, save_to_disk, and waveform visualization options to your liking.

Queue Prompt: Hit 'Queue Prompt' in ComfyUI, and the node will process, preview, and save your audio and its corresponding waveform image.

This node is particularly useful for audio diffusion models like Ace-Step, ensuring your generated audio is properly handled and archived. While designed for audio, its ability to save workflows and generate visuals might also be beneficial for video and image generation workflows, though more testing would be needed in those areas.

2. Detailed Parameter Information
The AAPS node provides a comprehensive set of parameters to give you fine-grained control over your audio processing, saving, and visualization. Here’s a breakdown of each one:

audio_input

Type: AUDIO (Required)

Description: This is the primary input for the audio data you want to process. It expects a standardized audio dictionary containing the raw waveform tensor and its sample rate from preceding nodes in your ComfyUI workflow.

Use & Why: This is where the sound comes in! Without it, the node has nothing to process. It ensures compatibility and smooth data transfer from other audio-generating or modifying nodes.

filename_prefix

Type: STRING (Required)

Default: "ComfyUI_generated_audio"

Description: A text string that will form the beginning of your saved audio file's name. A timestamp and a sequential counter will be appended to ensure uniqueness. You can also include subdirectories here (e.g., my_project/my_song).

Use & Why: Helps you organize and identify your saved audio files. Using subdirectories in the prefix automatically creates those folders within the ComfyUI_AdvancedAudioOutputs directory.

save_format

Type: ENUM (Required)

Options: ["MP3", "FLAC", "OPUS"]

Description: Choose the desired audio file format for saving to disk.

Use & Why:

MP3: Universally compatible, good for sharing and web use. Quality is lossy (some data is removed to reduce file size).

FLAC: Free Lossless Audio Codec. Perfect for audiophiles or archiving, as it retains all original audio information. Files are larger than MP3/OPUS.

OPUS: Highly efficient and high-quality codec, excellent for web streaming and modern applications. Offers great quality at smaller file sizes than MP3, especially at lower bitrates.

save_to_disk

Type: BOOLEAN (Optional)

Default: False

Description: If checked (True), the processed audio will be saved to your specified output directory. If unchecked (False), the node will only process the audio for preview and waveform generation, but no file will be written to disk.

Use & Why: This is a crucial toggle for workflow efficiency. Defaulting to False prevents your disk from being cluttered with intermediate audio files when you only need to preview or debug. Set to True only when you intend to keep the final output.

save_metadata

Type: BOOLEAN (Optional)

Default: True

Description: If checked (True), all available metadata (including the ComfyUI workflow, prompt details, and any custom notes) will be embedded into the saved audio file. If unchecked (False), no metadata will be saved with the audio, prioritizing privacy and smaller file sizes.

Use & Why: This is your privacy control. Keeping it True is highly recommended for reproducibility, as dragging the audio file back into ComfyUI will reload your entire workflow. Set to False if you want to share audio without revealing how it was created or if you are concerned about file size for maximum compatibility. Note: This can be overridden by a global ComfyUI flag (--disable-metadata).

custom_notes

Type: STRING (Optional)

Default: "" (empty string)

Description: Allows you to add your own arbitrary text notes directly into the audio file's metadata.

Use & Why: Great for adding reminders, specific parameters used, creative ideas, or any other relevant information that doesn't fit into the workflow JSON. These notes will be saved if save_metadata is True.

normalize_audio

Type: BOOLEAN (Optional)

Default: True

Description: If checked (True), the audio's volume will be automatically scaled to a peak value of approximately 0.99 (relative to the maximum possible amplitude, usually 1.0). This prevents digital clipping (distortion from audio being too loud) and ensures consistent loudness.

Use & Why: Prevents harsh, distorted sounds by automatically adjusting the volume. It also helps in achieving a consistent output level across different generations. If your audio is intentionally very quiet or has specific dynamic range requirements, you might consider unchecking this.

waveform_color

Type: STRING (Optional)

Default: "hotpink"

Description: The color of the main line in the generated waveform visualization. Accepts HTML color names (e.g., "red", "blue", "green") or hexadecimal color codes (e.g., "#FF0000" for red, "#00FFFF" for cyan).

Use & Why: Customize the aesthetic of your waveform image to match your personal preference or workflow theme.

waveform_background_color

Type: STRING (Optional)

Default: "black"

Description: The background color of the waveform visualization. Accepts HTML color names or hexadecimal color codes.

Use & Why: Further customize the visual appearance of your waveform image. A good contrast between waveform and background color improves readability.

mp3_quality

Type: ENUM (Optional)

Options: ["V0", "128k", "320k"]

Default: "V0"

Description: Specifies the quality setting for MP3 files when save_format is set to MP3.

Use & Why:

V0 (Variable Bitrate): Generally considered the highest quality for MP3s using variable bitrate encoding. It optimizes file size while striving for near-transparent audio quality. Often the best choice for overall quality and efficiency.

128k (Constant Bitrate): A common standard for good quality MP3s, suitable for general listening and streaming.

320k (Constant Bitrate): The highest constant bitrate for MP3s, offering excellent quality but resulting in larger file sizes.

opus_quality

Type: ENUM (Optional)

Options: ["64k", "96k", "128k", "192k", "320k"]

Default: "128k"

Description: Specifies the bitrate and thus quality for OPUS files when save_format is set to OPUS.

Use & Why: Higher bitrates generally mean higher audio quality but larger file sizes. OPUS is known for achieving very good quality even at lower bitrates compared to other codecs.

64k / 96k: Good for voice or general web use where file size is critical.

128k: A good balance of quality and file size for most music and general audio.

192k / 320k: Higher quality, approaching transparency for critical listening.

waveform_width

Type: INT (Optional)

Default: 800

Min/Max: 100 to 2048 (step 16)

Description: The desired width (in pixels) of the generated waveform image.

Use & Why: Control the horizontal resolution and detail of your waveform visualization. A wider image can show more of the waveform over time.

waveform_height

Type: INT (Optional)

Default: 150

Min/Max: 50 to 1024 (step 16)

Description: The desired height (in pixels) of the generated waveform image.

Use & Why: Control the vertical resolution of your waveform image. A taller image can emphasize amplitude variations more clearly.

3. In-Depth Nerd Technical Information
The Advanced Audio Preview & Save (AAPS) node is built upon a robust stack of Python libraries, leveraging their capabilities to provide a seamless and powerful audio processing experience within ComfyUI.

Core Audio Handling (torchaudio & md_io):
The node interacts with audio data primarily as torch.Tensor objects. It utilizes torchaudio for key audio operations, particularly for its ability to handle audio tensors efficiently and for resampling.

md_io.audio_from_comfy_3d: This custom utility from the ..core.io module is central to receiving audio from ComfyUI's standard AUDIO input. It's designed to robustly extract the waveform tensor (expected to be in (batch, channels, samples) format) and its sample rate, including handling potential GPU tensors by moving them to the CPU for processing.

Normalization Logic: The normalize_audio feature performs a simple peak normalization. It calculates the absolute maximum value (peak_val) in the audio tensor and then scales the entire waveform by 0.99 / peak_val. This ensures that the audio does not exceed a normalized maximum amplitude of 0.99, providing crucial "headroom" to prevent digital clipping when played back or further processed, while avoiding division by zero for silent inputs.

Robust Audio Saving (PyAV/FFmpeg):
The most critical component for flexible and metadata-rich audio saving is av, which is a Pythonic binding for FFmpeg.

In-Memory Processing: Instead of directly saving torch.Tensor to disk, the node first saves the audio tensor into an in-memory WAV buffer using torchaudio.save. This WAV buffer then acts as the input source for PyAV. This two-step process allows PyAV to leverage FFmpeg's robust codec support and metadata handling capabilities without needing direct filesystem access for intermediate steps.

Codec Selection and Quality: PyAV streams are configured based on the save_format (mp3, flac, opus).

For MP3, the libmp3lame encoder is used, with V0 quality mapping to qscale = 1 (a high-quality variable bitrate setting) and 128k/320k mapping to fixed bitrates.

For OPUS, the libopus encoder is used, and the node intelligently handles Opus's strict sample rate requirements (8000, 12000, 16000, 24000, 48000 Hz) by resampling the audio using torchaudio.functional.resample if the input sample rate is not supported or exceeds Opus's maximum.

FLAC uses the flac encoder, which is inherently lossless and thus doesn't require specific quality settings.

Metadata Embedding: PyAV allows direct manipulation of container metadata. The metadata dictionary (populated with prompt, extra_pnginfo workflow, and custom_notes) is directly assigned to output_container.metadata. PyAV handles the underlying FFmpeg commands to embed this information into the supported audio file formats. The save_metadata boolean acts as a master switch, preventing any metadata from being passed to PyAV if unchecked, effectively creating a "privacy filter". This also respects the ComfyUI global --disable-metadata flag.

Error Handling: The _save_audio_with_av function includes a try-except block to gracefully handle potential errors during audio encoding or saving, printing detailed error messages to the console and returning an error status to ComfyUI.

Waveform Visualization (Matplotlib & PIL):
The node generates a visual waveform image using matplotlib and PIL (Pillow).

Matplotlib Backend: plt.switch_backend('Agg') is crucial for server-side environments like ComfyUI. The 'Agg' backend renders figures to an array (or file) without requiring a graphical user interface (GUI) backend, preventing unexpected pop-up windows.

Direct Pixel Control: The figsize and dpi parameters for plt.subplots are carefully calculated (waveform_width / dpi_val, waveform_height / dpi_val) to ensure the output image directly matches the requested pixel dimensions (waveform_width, waveform_height).

Aesthetic Control: ax.axis("off"), ax.set_position([0, 0, 1, 1]), plt.tight_layout(pad=0) are used to strip away all default Matplotlib elements (axes, padding, titles) and ensure the plot perfectly fills the figure, allowing only the waveform and chosen background color to be visible. This is critical for generating a clean, isolated waveform image.

Image Conversion: The Matplotlib figure is saved to an in-memory BytesIO buffer as a PNG. PIL.Image.open then reads this buffer, converts it to "RGB" format, and Image.LANCZOS is used for high-quality resizing to the final waveform_width and waveform_height.

ComfyUI Tensor Format: Finally, the PIL image is converted to a NumPy array, normalized to [0, 1] float values, and then converted to a torch.Tensor with an added batch dimension (.unsqueeze(0)). This results in a (1, Height, Width, Channels) tensor, which is the expected format for ComfyUI's IMAGE output, making it directly compatible with Preview Image and Save Image nodes.

Resource Management: A finally block ensures plt.close(fig) is called, which is vital for preventing memory leaks in long-running workflows where many plots might be generated.

File Management and Uniqueness:

Dedicated Output Directory: All saved audio files are placed into a ComfyUI_AdvancedAudioOutputs directory within ComfyUI's main output folder, ensuring organized storage.

Unique Filenaming: File names are constructed using the filename_prefix, a timestamp modulo 100000 (for a unique, shorter number), and a 5-digit zero-padded counter (_counter:05). This robust naming convention helps prevent overwriting files and makes them easily identifiable.

This node exemplifies best practices in Python-based media processing, integrating powerful libraries like torchaudio, PyAV (FFmpeg), Matplotlib, and PIL to provide a flexible and user-friendly experience within the ComfyUI ecosystem.