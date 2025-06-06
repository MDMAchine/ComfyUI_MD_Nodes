# ComfyUI Mastering Chain Node Manual

This manual provides a comprehensive guide to the ComfyUI Mastering Chain Node, a powerful tool designed to refine and enhance your audio. While primarily built for use with audio diffusion models, its capabilities may extend to other generative tasks involving time-series data, such as video or image processing, though further experimentation would be required.

---

## 1. Understanding the Mastering Chain Node

### What is it?

The Mastering Chain Node is a custom component for ComfyUI that acts as a digital audio workstation (DAW) mastering suite in a single node. It's designed to take your raw audio output from other nodes (like audio diffusion models) and apply a series of common audio processing effects to make it sound louder, clearer, and more balanced. Think of it as the final polish before your audio is ready for prime time.

### How it Works

The node processes your audio through a sequential "chain" of effects, much like how professional audio engineers master tracks. These effects are applied in a specific order to achieve optimal results:

* **Global Gain**: First, it applies an overall volume adjustment.
* **Equalization (EQ)**: Next, it sculpts the frequency content, boosting or cutting specific ranges (e.g., adding clarity to highs, cleaning up muddy lows).
* **Compression**: Then, it controls the dynamic range of the audio, making quiet parts louder and loud parts quieter, resulting in a more consistent and impactful sound. You can choose between single-band (affecting all frequencies uniformly) or multi-band (affecting different frequency ranges independently) compression.
* **Limiting**: Finally, a "brickwall" limiter is used to prevent the audio from exceeding a certain maximum level, ensuring no harsh digital clipping occurs and maximizing overall loudness without distortion.

Throughout this process, the node internally converts the audio data into a format suitable for digital signal processing (DSP), applies the chosen effects, and then converts it back for ComfyUI. It also generates visual representations of your audio before and after processing, allowing you to see the impact of your mastering settings.

### What it Does

* **Enhances Audio Quality**: Makes your audio sound more professional, balanced, and loud.
* **Prevents Clipping**: Ensures your audio doesn't exceed 0 dBFS (decibels full scale), preventing digital distortion.
* **Provides Control**: Offers detailed parameters for fine-tuning each mastering stage.
* **Visual Feedback**: Generates waveform images to visually compare the original and processed audio.
* **Supports Mono and Stereo**: Handles both single-channel (mono) and two-channel (stereo) audio seamlessly.

### How to Use It

1.  **Connect Audio Input**: Drag a connection from the `AUDIO` output of your audio source node (e.g., an audio diffusion model) to the `audio` input of the Mastering Chain Node.
2.  **Set Sample Rate**: Ensure the `sample_rate` input matches the sample rate of your input audio. This is crucial for correct DSP operation.
3.  **Adjust Parameters**: Use the various sliders and dropdowns in the node's properties panel to configure the Global Gain, EQ, Compression, and Limiter settings to your desired sound. Experimentation is key!
4.  **Connect Outputs**:
    * The `AUDIO` output provides the fully processed audio.
    * The first `IMAGE` output provides a waveform visualization of the audio before processing.
    * The second `IMAGE` output provides a waveform visualization of the audio after processing.
    You can connect these `IMAGE` outputs to a `Preview Image` node in ComfyUI to view them.

---

## 2. Detailed Information about the Parameters

The Mastering Chain Node offers a comprehensive set of controls to shape your audio. Here's a breakdown of each parameter, its purpose, and how to use it.

### Core Inputs

* **audio**
    * **Type**: `AUDIO`
    * **Description**: The input audio waveform. This should come from another audio-generating node.

* **sample_rate**
    * **Type**: `INT`
    * **Range**: `8000 Hz` to `192000 Hz` (default: `44100 Hz`).
    * **What it does**: Specifies the number of samples per second in your audio (e.g., 44100 Hz for CD quality).
    * **Use & Why**: This must accurately match the sample rate of your input audio. All internal digital signal processing (DSP) calculations, like filter frequencies and time-based effects (attack/release times), rely on this value. Incorrect `sample_rate` will lead to incorrect or distorted audio.

### Global Gain

* **master_gain_db**
    * **Type**: `FLOAT`
    * **Range**: `-20.0 dB` to `20.0 dB` (default: `0.0 dB`).
    * **What it does**: Applies a static volume increase or decrease to the entire audio signal, expressed in decibels (dB).
    * **Use & Why**: Use this as a foundational level adjustment. If your audio is consistently too quiet or too loud before any other processing, this is the place to make a broad change. Positive values boost, negative values cut.

### Equalization (EQ)

* **enable_eq**
    * **Type**: `BOOLEAN`
    * **Default**: `True`.
    * **What it does**: A master switch to enable or bypass the entire Equalizer section.
    * **Use & Why**: If you don't need any frequency adjustments, or want to hear the effect of other mastering stages in isolation, set this to `False`.

#### High-Shelf EQ

* **eq_high_shelf_gain_db**
    * **Type**: `FLOAT`
    * **Range**: `-12.0 dB` to `12.0 dB` (default: `1.5 dB`).
    * **What it does**: Boosts or cuts all frequencies above the `eq_high_shelf_freq`.
    * **Use & Why**: Used to add "air" and clarity to the high end (positive gain) or to tame harshness and sibilance (negative gain).

* **eq_high_shelf_freq**
    * **Type**: `FLOAT`
    * **Range**: `1000.0 Hz` to `20000.0 Hz` (default: `12000.0 Hz`).
    * **What it does**: The "corner" frequency where the high-shelf filter begins to have its effect.
    * **Use & Why**: Determines where the high-end shaping starts. A higher frequency affects only the very top end, while a lower frequency will impact more of the upper mids.

* **eq_high_shelf_q**
    * **Type**: `FLOAT`
    * **Range**: `0.1` to `5.0` (default: `0.707`).
    * **What it does**: Q factor for the high-shelf filter.
    * **Use & Why**: Note: This parameter is exposed but fixed internally at `0.707` for standard, predictable shelf behavior. While you can adjust it in the UI, its effective value remains constant for stability and predictable audio characteristics. `0.707` represents a gentle slope.

#### Low-Shelf EQ

* **enable_low_shelf_eq**
    * **Type**: `BOOLEAN`
    * **Default**: `True`.
    * **What it does**: Enables or bypasses the Low-Shelf Equalizer.
    * **Use & Why**: Allows you to isolate the effect of this specific filter.

* **eq_low_shelf_gain_db**
    * **Type**: `FLOAT`
    * **Range**: `-12.0 dB` to `12.0 dB` (default: `-5.0 dB`).
    * **What it does**: Boosts or cuts all frequencies below the `eq_low_shelf_freq`.
    * **Use & Why**: Used to add warmth and weight to the low end (positive gain) or to reduce muddiness and rumble (negative gain).

* **eq_low_shelf_freq**
    * **Type**: `FLOAT`
    * **Range**: `20.0 Hz` to `500.0 Hz` (default: `55.0 Hz`).
    * **What it does**: The "corner" frequency where the low-shelf filter begins to have its effect.
    * **Use & Why**: Determines where the low-end shaping starts. A lower frequency impacts only the very bottom end, while a higher frequency will affect more of the lower mids.

* **eq_low_shelf_q**
    * **Type**: `FLOAT`
    * **Range**: `0.1` to `5.0` (default: `0.707`).
    * **What it does**: Q factor for the low-shelf filter.
    * **Use & Why**: Note: Similar to the high-shelf Q, this parameter is exposed but fixed internally at `0.707` for standard, predictable shelf behavior. While you can adjust it in the UI, its effective value remains constant.

#### Parametric EQ Band 1

* **enable_param_eq1**
    * **Type**: `BOOLEAN`
    * **Default**: `True`.
    * **What it does**: Enables or bypasses Parametric EQ Band 1.
    * **Use & Why**: Allows you to isolate the effect of this specific filter.

* **param_eq1_gain_db**
    * **Type**: `FLOAT`
    * **Range**: `-20.0 dB` to `20.0 dB` (default: `-8.0 dB`).
    * **What it does**: Boosts or cuts frequencies around the `param_eq1_freq`.
    * **Use & Why**: Useful for targeted cuts (negative gain) to remove resonant frequencies or boosts (positive gain) to highlight specific elements. Default is a cut at 77Hz, often used to clean up sub-bass.

* **param_eq1_freq**
    * **Type**: `FLOAT`
    * **Range**: `20.0 Hz` to `20000.0 Hz` (default: `77.0 Hz`).
    * **What it does**: The center frequency of the EQ band.
    * **Use & Why**: Determines where the boost or cut occurs.

* **param_eq1_q**
    * **Type**: `FLOAT`
    * **Range**: `0.1` to `20.0` (default: `9.0`).
    * **What it does**: The "Q" (Quality factor) determines the width of the EQ band. A higher Q means a narrower, more precise adjustment. A lower Q means a wider, broader adjustment.
    * **Use & Why**: For targeted cuts (like removing hum or resonance), use a high Q. For broader tonal shaping, use a lower Q. Default is `9.0`, which is quite narrow, good for surgical cuts.

#### Parametric EQ Band 2

* **enable_param_eq2**
    * **Type**: `BOOLEAN`
    * **Default**: `True`.
    * **What it does**: Enables or bypasses Parametric EQ Band 2.
    * **Use & Why**: Allows you to isolate the effect of this specific filter.

* **param_eq2_gain_db**
    * **Type**: `FLOAT`
    * **Range**: `-20.0 dB` to `20.0 dB` (default: `3.5 dB`).
    * **What it does**: Boosts or cuts frequencies around the `param_eq2_freq`.
    * **Use & Why**: Similar to Band 1, but for a second, independent frequency adjustment. Default is a boost at 130Hz, often used to add punch to bass and drums.

* **param_eq2_freq**
    * **Type**: `FLOAT`
    * **Range**: `20.0 Hz` to `20000.0 Hz` (default: `130.0 Hz`).
    * **What it does**: The center frequency of the second EQ band.
    * **Use & Why**: Determines where the second boost or cut occurs.

* **param_eq2_q**
    * **Type**: `FLOAT`
    * **Range**: `0.1` to `20.0` (default: `1.5`).
    * **What it does**: The Q factor for the second EQ band.
    * **Use & Why**: Controls the width of the second EQ band. Default is `1.5`, which is a fairly broad boost.

### Compression

* **enable_comp**
    * **Type**: `BOOLEAN`
    * **Default**: `True`.
    * **What it does**: A master switch to enable or bypass the entire Compressor section.
    * **Use & Why**: If you don't need dynamic range control, or want to hear the effect of other mastering stages in isolation, set this to `False`.

* **comp_type**
    * **Type**: Dropdown (`"Single-Band"`, `"Multiband"`)
    * **Default**: `"Single-Band"`.
    * **What it does**: Selects between a single-band compressor (applies compression to the whole frequency spectrum) or a 3-band multiband compressor (splits audio into low, mid, and high bands and compresses each independently).
    * **Use & Why**:
        * **Single-Band**: Simpler, often used for general dynamic glue or overall loudness.
        * **Multiband**: Offers much finer control over different frequency ranges. For example, you can compress bass heavily without squashing the vocals in the mid-range. This is more complex but more powerful for professional mastering.

#### Single-Band Compressor Parameters (Active when `comp_type` is "Single-Band")

* **comp_threshold_db**
    * **Type**: `FLOAT`
    * **Range**: `-60.0 dB` to `0.0 dB` (default: `-14.0 dB`).
    * **What it does**: The input level (in dB) above which compression begins.
    * **Use & Why**: Signals exceeding this level will be reduced. A lower (more negative) threshold means more of the signal will be compressed, leading to more "loudness."

* **comp_ratio**
    * **Type**: `FLOAT`
    * **Range**: `1.0` to `20.0` (default: `1.5`).
    * **What it does**: The ratio of input level change to output level change once the signal is above the threshold (e.g., 4:1 means that for every 4dB the input signal goes over the threshold, the output will only increase by 1dB).
    * **Use & Why**: Higher ratios result in more aggressive compression and a flatter dynamic range. Lower ratios are more subtle.

* **comp_attack_ms**
    * **Type**: `FLOAT`
    * **Range**: `1.0 ms` to `500.0 ms` (default: `25.0 ms`).
    * **What it does**: The time (in milliseconds) it takes for the compressor to reach full gain reduction after the signal crosses the threshold.
    * **Use & Why**: A fast attack preserves initial transients (like drum hits) less, making them sound less punchy but more controlled. A slow attack allows transients to pass through before compression kicks in, preserving punch.

* **comp_release_ms**
    * **Type**: `FLOAT`
    * **Range**: `10.0 ms` to `2000.0 ms` (default: `300.0 ms`).
    * **What it does**: The time (in milliseconds) it takes for the compressor to return to unity gain (no compression) after the signal falls below the threshold.
    * **Use & Why**: A fast release can sound "pumped" or "breathing" with the music, sometimes desirable for effect. A slow release can create a more transparent, sustained sound.

* **comp_makeup_gain_db**
    * **Type**: `FLOAT`
    * **Range**: `0.0 dB` to `12.0 dB` (default: `6.0 dB`).
    * **What it does**: Adds gain after the compression stage to compensate for the volume reduction caused by compression.
    * **Use & Why**: Compression reduces the overall loudness. Makeup gain allows you to bring the level back up, often making the audio sound louder and denser.

#### Multiband Compressor Parameters (Active when `comp_type` is "Multiband")

* **low_band_comp_threshold_db**
    * **Type**: `FLOAT`
    * **Range**: `-60.0 dB` to `0.0 dB` (default: `-12.0 dB`).
    * **What it does**: Threshold for the low-frequency band.
    * **Use & Why**: Allows independent compression of the bass and sub-bass frequencies. Useful for tightening up muddy lows or adding punch to the kick drum without affecting other elements.

* **low_band_comp_ratio**
    * **Type**: `FLOAT`
    * **Range**: `1.0` to `20.0` (default: `2.0`).
    * **What it does**: Ratio for the low-frequency band.

* **low_band_comp_attack_ms**
    * **Type**: `FLOAT`
    * **Range**: `1.0 ms` to `500.0 ms` (default: `10.0 ms`).
    * **What it does**: Attack time for the low-frequency band.

* **low_band_comp_release_ms**
    * **Type**: `FLOAT`
    * **Range**: `10.0 ms` to `2000.0 ms` (default: `200.0 ms`).
    * **What it does**: Release time for the low-frequency band.

* **low_band_comp_makeup_gain_db**
    * **Type**: `FLOAT`
    * **Range**: `0.0 dB` to `12.0 dB` (default: `4.0 dB`).
    * **What it does**: Makeup gain for the low-frequency band.

* **mid_band_comp_threshold_db**
    * **Type**: `FLOAT`
    * **Range**: `-60.0 dB` to `0.0 dB` (default: `-14.0 dB`).
    * **What it does**: Threshold for the mid-frequency band.
    * **Use & Why**: Controls dynamics in the vocal and instrument range. Can be used to bring out details or control harshness.

* **mid_band_comp_ratio**
    * **Type**: `FLOAT`
    * **Range**: `1.0` to `20.0` (default: `1.5`).

* **mid_band_comp_attack_ms**
    * **Type**: `FLOAT`
    * **Range**: `1.0 ms` to `500.0 ms` (default: `25.0 ms`).

* **mid_band_comp_release_ms**
    * **Type**: `FLOAT`
    * **Range**: `10.0 ms` to `2000.0 ms` (default: `300.0 ms`).

* **mid_band_comp_makeup_gain_db**
    * **Type**: `FLOAT`
    * **Range**: `0.0 dB` to `12.0 dB` (default: `3.0 dB`).

* **high_band_comp_threshold_db**
    * **Type**: `FLOAT`
    * **Range**: `-60.0 dB` to `0.0 dB` (default: `-16.0 dB`).
    * **What it does**: Threshold for the high-frequency band.
    * **Use & Why**: Manages dynamics in the high-end, controlling sibilance or adding sparkle.

* **high_band_comp_ratio**
    * **Type**: `FLOAT`
    * **Range**: `1.0` to `20.0` (default: `1.8`).

* **high_band_comp_attack_ms**
    * **Type**: `FLOAT`
    * **Range**: `1.0 ms` to `500.0 ms` (default: `5.0 ms`).

* **high_band_comp_release_ms**
    * **Type**: `FLOAT`
    * **Range**: `10.0 ms` to `2000.0 ms` (default: `400.0 ms`).

* **high_band_comp_makeup_gain_db**
    * **Type**: `FLOAT`
    * **Range**: `0.0 dB` to `12.0 dB` (default: `2.0 dB`).

* **low_mid_crossover_freq**
    * **Type**: `FLOAT`
    * **Range**: `20.0 Hz` to `1000.0 Hz` (default: `250.0 Hz`).
    * **What it does**: Defines the crossover frequency between the low and mid bands in multiband compression.
    * **Use & Why**: Determines where the low-frequency compression stops and mid-frequency compression begins.

* **mid_high_crossover_freq**
    * **Type**: `FLOAT`
    * **Range**: `1000.0 Hz` to `10000.0 Hz` (default: `2500.0 Hz`).
    * **What it does**: Defines the crossover frequency between the mid and high bands.
    * **Use & Why**: Determines where the mid-frequency compression stops and high-frequency compression begins.

### Limiter

* **enable_limiter**
    * **Type**: `BOOLEAN`
    * **Default**: `True`.
    * **What it does**: Enables or bypasses the Limiter section.
    * **Use & Why**: The limiter is crucial for preventing digital clipping and maximizing loudness. Only disable if you have other limiting stages in your workflow.

* **limiter_ceiling_amplitude**
    * **Type**: `FLOAT`
    * **Range**: `0.1` to `1.0` (default: `0.98`).
    * **What it does**: Sets the maximum allowed peak amplitude for the audio, where `1.0` is 0 dBFS (full scale).
    * **Use & Why**: This is your "brickwall." No audio signal will exceed this level. Setting it slightly below `1.0` (e.g., `0.98` or `-0.1 dBFS`) is common practice to leave a tiny bit of "headroom" for digital-to-analog conversion.

* **limiter_release_ms**
    * **Type**: `FLOAT`
    * **Range**: `10.0 ms` to `2000.0 ms` (default: `200.0 ms`).
    * **What it does**: The time (in milliseconds) it takes for the limiter to release its gain reduction after the signal drops below the ceiling.
    * **Use & Why**: A faster release can make the limiter more reactive but might introduce pumping artifacts. A slower release is more transparent but might hold down the audio longer than necessary.

* **limiter_lookahead_ms**
    * **Type**: `FLOAT`
    * **Range**: `0.0 ms` to `50.0 ms` (default: `5.0 ms`).
    * **What it does**: The time (in milliseconds) the limiter "looks ahead" into the audio signal to detect upcoming peaks.
    * **Use & Why**: Essential for "brickwall" limiting. By knowing peaks are coming, the limiter can apply gain reduction *before* the peak hits, preventing overshoots and distortion. A `0.0` lookahead means no lookahead, which can lead to minor overshoots.

### Visualization Parameters

* **waveform_height**
    * **Type**: `INT`
    * **Range**: `64` to `2048` (default: `256`).
    * **What it does**: Sets the height of the generated waveform image in pixels.
    * **Use & Why**: Affects the visual resolution of the waveform plots. Higher values mean more detail, but larger image files.

* **waveform_width**
    * **Type**: `INT`
    * **Range**: `256` to `4096` (default: `1024`).
    * **What it does**: Sets the width of the generated waveform image in pixels.
    * **Use & Why**: Similar to height, affects visual resolution and file size.

* **plot_color_primary**
    * **Type**: `COLOR` (RGB hex string, e.g., `#FFFFFF` for white)
    * **Default**: `#00FF00` (green).
    * **What it does**: Defines the primary color for the waveform plot line.
    * **Use & Why**: Customize the appearance of your plots.

* **plot_color_background**
    * **Type**: `COLOR` (RGB hex string)
    * **Default**: `#000000` (black).
    * **What it does**: Defines the background color of the waveform plot.

* **plot_color_grid**
    * **Type**: `COLOR` (RGB hex string)
    * **Default**: `#888888` (dark gray).
    * **What it does**: Defines the color of the grid lines on the plot.

* **plot_color_text**
    * **Type**: `COLOR` (RGB hex string)
    * **Default**: `#FFFFFF` (white).
    * **What it does**: Defines the color of any text (like axis labels, though mostly hidden) on the plot.

* **plot_grid**
    * **Type**: `BOOLEAN`
    * **Default**: `True`.
    * **What it does**: Toggles the visibility of grid lines on the plot.
    * **Use & Why**: Grid lines can help with visual alignment and reading values.

* **plot_axis**
    * **Type**: `BOOLEAN`
    * **Default**: `False`.
    * **What it does**: Toggles the visibility of the X and Y axes on the plot.
    * **Use & Why**: Often set to `False` for cleaner, minimalist waveform visualizations where only the signal matters.

---

## 3. In-Depth Technical Information

This section provides a deeper dive into the algorithms and internal workings of the Mastering Chain Node.

### Digital Signal Processing (DSP) Core

The node extensively uses `numpy` for efficient numerical operations and `scipy.signal` for digital filter design and application.

* **Audio Data Format**: Audio is typically represented as `torch.Tensor` within ComfyUI. The node converts this to `numpy.ndarray` (float32) for DSP operations and then back to `torch.Tensor` for output.
* **Mono Conversion**: For stereo inputs, processing (especially for EQ and Compression) is often done on a mono sum of the channels (`(left + right) / 2`) to ensure consistent dynamic and tonal shaping across the stereo field. The gain changes are then applied back to the original stereo channels.

### Global Gain Implementation

Simple multiplication of the audio signal by a linear gain factor derived from the `master_gain_db` parameter.

`linear_gain = 10^(master_gain_db / 20)`

### Equalization (EQ) Implementation

The EQ section uses `scipy.signal.iirfilter` to design biquadratic (second-order) IIR filters.

* **High-Shelf and Low-Shelf Filters**: These are implemented as `highshelf` and `lowshelf` filters, respectively. The `Q` factor for these is internally fixed to `0.707` (approximately 1/√2), which corresponds to a "Butterworth" or maximally flat response, preventing undesirable peaks or dips near the shelf frequency.
* **Parametric EQ (Peaking Filters)**: These are implemented as `peak` filters, allowing a boost or cut at a specific center frequency (`param_eq_freq`) with a variable bandwidth controlled by the `param_eq_q` factor. A higher `Q` makes the band narrower and more selective.

All filters are applied using `scipy.signal.lfilter`, which is a causal (real-time capable) IIR filter. To achieve phase-linear (and thus more transparent) processing, the filters are applied forward and backward (`filtfilt`), effectively doubling the filter order and removing phase distortion.

### Compression Implementation

Both single-band and multiband compressors are implemented as feed-forward RMS compressors.

* **RMS Detection**: The `envelope` function calculates the Root Mean Square (RMS) level of the audio signal over a short window, providing a smoothed representation of loudness.
* **Gain Reduction Calculation**: Based on the detected RMS level, `comp_threshold_db`, and `comp_ratio`, the required gain reduction is calculated.
* **Attack and Release**:
    * **Attack**: When the signal exceeds the threshold, the gain reduction quickly increases based on the `comp_attack_ms` parameter. This determines how fast the compressor "clamps down."
    * **Release**: When the signal falls below the threshold, the gain reduction smoothly decreases back to unity gain based on the `comp_release_ms` parameter. This determines how fast the compressor "lets go."
    * These attack and release curves are typically exponential, using coefficients derived from the `ms` values and `sample_rate`.
* **Makeup Gain**: Applied after compression to restore overall loudness.

### Multiband Crossover Filters

For multiband compression, `scipy.signal.butter` is used to design Butterworth filters for frequency splitting:

* **Low-Pass Filter**: For the low band.
* **Band-Pass Filter**: For the mid band (using two cutoff frequencies).
* **High-Pass Filter**: For the high band.

These filters sum to a "perfect reconstruction" system (or close to it), ensuring that when the bands are recombined, the original frequency response is largely preserved, preventing phase issues or gaps.

### Limiter Implementation

The limiter is a "brickwall" peak limiter with a lookahead.

* **Lookahead Buffer**: The audio is buffered by a certain number of samples (calculated from `limiter_lookahead_ms` and `sample_rate`). This allows the limiter to "see" upcoming peaks.
* **Peak Detection**: For each sample, the limiter examines a window of audio (current_sample + lookahead_samples) to find the maximum absolute peak.
* **Instantaneous Attack**: If a peak is detected that would exceed the `limiter_ceiling_amplitude`, the gain reduction is applied immediately (a "brickwall" action). The `current_gain_reduction_factor` is set to the minimum of its current value and the `desired_gain_reduction_factor` for the detected peak, ensuring the ceiling is never breached.
* **Exponential Release**: After the peak passes, the gain reduction smoothly decays back to unity gain (no reduction) using an exponential release coefficient, similar to the compressor's release stage.

### Waveform Plotting

* **Matplotlib**: `matplotlib.pyplot` is used to generate waveform plots.
* **Mono Mix**: For stereo audio, channels are averaged to create a mono mix for a single waveform visualization.
* **Image Conversion**: The plot is saved to an in-memory `BytesIO` buffer as a PNG, then opened with `PIL.Image`, converted to RGB, and finally transformed into a `NumPy` array (`float32`) normalized to 0-1, and then into a `PyTorch` tensor with a batch dimension (`1, height, width, channels`) for ComfyUI's `IMAGE` output.

This node leverages robust scientific computing libraries (`numpy`, `scipy`) and standard DSP algorithms to provide a powerful and flexible audio mastering solution within the ComfyUI environment.
