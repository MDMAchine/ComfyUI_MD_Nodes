ComfyUI Mastering Chain Node Manual
This manual provides a comprehensive guide to the ComfyUI Mastering Chain Node, a powerful tool designed to refine and enhance your audio. While primarily built for use with audio diffusion models, its capabilities may extend to other generative tasks involving time-series data, such as video or image processing, though further experimentation would be required.

1. Understanding the Mastering Chain Node
What is it?
The Mastering Chain Node is a custom component for ComfyUI that acts as a digital audio workstation (DAW) mastering suite in a single node. It's designed to take your raw audio output from other nodes (like audio diffusion models) and apply a series of common audio processing effects to make it sound louder, clearer, and more balanced. Think of it as the final polish before your audio is ready for prime time.

How it Works
The node processes your audio through a sequential "chain" of effects, much like how professional audio engineers master tracks. These effects are applied in a specific order to achieve optimal results:

Global Gain: First, it applies an overall volume adjustment.

Equalization (EQ): Next, it sculpts the frequency content, boosting or cutting specific ranges (e.g., adding clarity to highs, cleaning up muddy lows).

Compression: Then, it controls the dynamic range of the audio, making quiet parts louder and loud parts quieter, resulting in a more consistent and impactful sound. You can choose between single-band (affecting all frequencies uniformly) or multi-band (affecting different frequency ranges independently) compression.

Limiting: Finally, a "brickwall" limiter is used to prevent the audio from exceeding a certain maximum level, ensuring no harsh digital clipping occurs and maximizing overall loudness without distortion.

Throughout this process, the node internally converts the audio data into a format suitable for digital signal processing (DSP), applies the chosen effects, and then converts it back for ComfyUI. It also generates visual representations of your audio before and after processing, allowing you to see the impact of your mastering settings.

What it Does
Enhances Audio Quality: Makes your audio sound more professional, balanced, and loud.

Prevents Clipping: Ensures your audio doesn't exceed 0 dBFS (decibels full scale), preventing digital distortion.

Provides Control: Offers detailed parameters for fine-tuning each mastering stage.

Visual Feedback: Generates waveform images to visually compare the original and processed audio.

Supports Mono and Stereo: Handles both single-channel (mono) and two-channel (stereo) audio seamlessly.

How to Use It
Connect Audio Input: Drag a connection from the AUDIO output of your audio source node (e.g., an audio diffusion model) to the audio input of the Mastering Chain Node.

Set Sample Rate: Ensure the sample_rate input matches the sample rate of your input audio. This is crucial for correct DSP operation.

Adjust Parameters: Use the various sliders and dropdowns in the node's properties panel to configure the Global Gain, EQ, Compression, and Limiter settings to your desired sound. Experimentation is key!

Connect Outputs:

The AUDIO output provides the fully processed audio.

The first IMAGE output provides a waveform visualization of the audio before processing.

The second IMAGE output provides a waveform visualization of the audio after processing.
You can connect these IMAGE outputs to a Preview Image node in ComfyUI to view them.

2. Detailed Information about the Parameters
The Mastering Chain Node offers a comprehensive set of controls to shape your audio. Here's a breakdown of each parameter, its purpose, and how to use it.

Core Inputs
audio (AUDIO): The input audio waveform. This should come from another audio-generating node.

sample_rate (INT):

What it does: Specifies the number of samples per second in your audio (e.g., 44100 Hz for CD quality).

Use & Why: This must accurately match the sample rate of your input audio. All internal digital signal processing (DSP) calculations, like filter frequencies and time-based effects (attack/release times), rely on this value. Incorrect sample_rate will lead to incorrect or distorted audio.

Range: 8000 Hz to 192000 Hz (default: 44100 Hz).

Global Gain
master_gain_db (FLOAT):

What it does: Applies a static volume increase or decrease to the entire audio signal, expressed in decibels (dB).

Use & Why: Use this as a foundational level adjustment. If your audio is consistently too quiet or too loud before any other processing, this is the place to make a broad change. Positive values boost, negative values cut.

Range: -20.0 dB to 20.0 dB (default: 0.0 dB).

Equalization (EQ)
enable_eq (BOOLEAN):

What it does: A master switch to enable or bypass the entire Equalizer section.

Use & Why: If you don't need any frequency adjustments, or want to hear the effect of other mastering stages in isolation, set this to False.

Default: True.

High-Shelf EQ
eq_high_shelf_gain_db (FLOAT):

What it does: Boosts or cuts all frequencies above the eq_high_shelf_freq.

Use & Why: Used to add "air" and clarity to the high end (positive gain) or to tame harshness and sibilance (negative gain).

Range: -12.0 dB to 12.0 dB (default: 1.5 dB).

eq_high_shelf_freq (FLOAT):

What it does: The "corner" frequency where the high-shelf filter begins to have its effect.

Use & Why: Determines where the high-end shaping starts. A higher frequency affects only the very top end, while a lower frequency will impact more of the upper mids.

Range: 1000.0 Hz to 20000.0 Hz (default: 12000.0 Hz).

eq_high_shelf_q (FLOAT):

What it does: Q factor for the high-shelf filter.

Use & Why: Note: This parameter is exposed but fixed internally at 0.707 for standard, predictable shelf behavior. While you can adjust it in the UI, its effective value remains constant for stability and predictable audio characteristics. 0.707 represents a gentle slope.

Range: 0.1 to 5.0 (default: 0.707).

Low-Shelf EQ
enable_low_shelf_eq (BOOLEAN):

What it does: Enables or bypasses the Low-Shelf Equalizer.

Use & Why: Allows you to isolate the effect of this specific filter.

Default: True.

eq_low_shelf_gain_db (FLOAT):

What it does: Boosts or cuts all frequencies below the eq_low_shelf_freq.

Use & Why: Used to add warmth and weight to the low end (positive gain) or to reduce muddiness and rumble (negative gain).

Range: -12.0 dB to 12.0 dB (default: -5.0 dB).

eq_low_shelf_freq (FLOAT):

What it does: The "corner" frequency where the low-shelf filter begins to have its effect.

Use & Why: Determines where the low-end shaping starts. A lower frequency impacts only the very bottom end, while a higher frequency will affect more of the lower mids.

Range: 20.0 Hz to 500.0 Hz (default: 55.0 Hz).

eq_low_shelf_q (FLOAT):

What it does: Q factor for the low-shelf filter.

Use & Why: Note: Similar to the high-shelf Q, this parameter is exposed but fixed internally at 0.707 for standard, predictable shelf behavior. While you can adjust it in the UI, its effective value remains constant.

Range: 0.1 to 5.0 (default: 0.707).

Parametric EQ Band 1
enable_param_eq1 (BOOLEAN):

What it does: Enables or bypasses Parametric EQ Band 1.

Use & Why: Allows you to isolate the effect of this specific filter.

Default: True.

param_eq1_gain_db (FLOAT):

What it does: Boosts or cuts frequencies around the param_eq1_freq.

Use & Why: Useful for targeted cuts (negative gain) to remove resonant frequencies or boosts (positive gain) to highlight specific elements. Default is a cut at 77Hz, often used to clean up sub-bass.

Range: -20.0 dB to 20.0 dB (default: -8.0 dB).

param_eq1_freq (FLOAT):

What it does: The center frequency of the EQ band.

Use & Why: Determines where the boost or cut occurs.

Range: 20.0 Hz to 20000.0 Hz (default: 77.0 Hz).

param_eq1_q (FLOAT):

What it does: The "Q" (Quality factor) determines the width of the EQ band. A higher Q means a narrower, more precise adjustment. A lower Q means a wider, broader adjustment.

Use & Why: For targeted cuts (like removing hum or resonance), use a high Q. For broader tonal shaping, use a lower Q. Default is 9.0, which is quite narrow, good for surgical cuts.

Range: 0.1 to 20.0 (default: 9.0).

Parametric EQ Band 2
enable_param_eq2 (BOOLEAN):

What it does: Enables or bypasses Parametric EQ Band 2.

Use & Why: Allows you to isolate the effect of this specific filter.

Default: True.

param_eq2_gain_db (FLOAT):

What it does: Boosts or cuts frequencies around the param_eq2_freq.

Use & Why: Similar to Band 1, but for a second, independent frequency adjustment. Default is a boost at 130Hz, often used to add punch to bass and drums.

Range: -20.0 dB to 20.0 dB (default: 3.5 dB).

param_eq2_freq (FLOAT):

What it does: The center frequency of the second EQ band.

Use & Why: Determines where the second boost or cut occurs.

Range: 20.0 Hz to 20000.0 Hz (default: 130.0 Hz).

param_eq2_q (FLOAT):

What it does: The Q factor for the second EQ band.

Use & Why: Controls the width of the second EQ band. Default is 1.5, which is a fairly broad boost.

Range: 0.1 to 20.0 (default: 1.5).

Compression
enable_comp (BOOLEAN):

What it does: A master switch to enable or bypass the entire Compressor section.

Use & Why: If you don't need dynamic range control, or want to hear the effect of other mastering stages in isolation, set this to False.

Default: True.

comp_type (Dropdown: "Single-Band", "Multiband"):

What it does: Selects between a single-band compressor (applies compression to the whole frequency spectrum) or a 3-band multiband compressor (splits audio into low, mid, and high bands and compresses each independently).

Use & Why:

Single-Band: Simpler, often used for general dynamic glue or overall loudness.

Multiband: Offers much finer control over different frequency ranges. For example, you can compress bass heavily without squashing the vocals in the mid-range. This is more complex but more powerful for professional mastering.

Default: "Single-Band".

Single-Band Compressor Parameters (Active when comp_type is "Single-Band")
comp_threshold_db (FLOAT):

What it does: The input level (in dB) above which compression begins.

Use & Why: Signals exceeding this level will be reduced. A lower (more negative) threshold means more of the signal will be compressed, leading to more "loudness."

Range: -60.0 dB to 0.0 dB (default: -14.0 dB).

comp_ratio (FLOAT):

What it does: The ratio of input level change to output level change once the signal is above the threshold (e.g., 4:1 means that for every 4dB the input signal goes over the threshold, the output will only increase by 1dB).

Use & Why: Higher ratios result in more aggressive compression and a flatter dynamic range. Lower ratios are more subtle.

Range: 1.0 to 20.0 (default: 1.5).

comp_attack_ms (FLOAT):

What it does: The time (in milliseconds) it takes for the compressor to reach full gain reduction after the signal crosses the threshold.

Use & Why: A fast attack preserves initial transients (like drum hits) less, making them sound less punchy but more controlled. A slow attack allows transients to pass through before compression kicks in, preserving punch.

Range: 1.0 ms to 500.0 ms (default: 25.0 ms).

comp_release_ms (FLOAT):

What it does: The time (in milliseconds) it takes for the compressor to return to unity gain (no compression) after the signal falls below the threshold.

Use & Why: A fast release can sound "pumped" or "breathing" with the music, sometimes desirable for effect. A slow release can create a more transparent, sustained sound.

Range: 10.0 ms to 2000.0 ms (default: 300.0 ms).

comp_makeup_gain_db (FLOAT):

What it does: Adds gain after the compression stage to compensate for the volume reduction caused by compression.

Use & Why: Compression often reduces overall loudness, so makeup gain is used to bring the level back up. This is essential for achieving a "loud" master without clipping.

Range: -12.0 dB to 12.0 dB (default: 0.0 dB).

comp_soft_knee_db (FLOAT):

What it does: Defines a range (in dB) around the threshold where the compression ratio is gradually applied, rather than instantly (a "hard knee").

Use & Why: A soft knee provides a smoother, more transparent compression effect, as it eases into the gain reduction. A value of 0.0 creates a hard knee.

Range: 0.0 dB to 12.0 dB (default: 3.0 dB).

Multiband Compressor Crossover Frequencies (Active when comp_type is "Multiband")
mb_crossover_low_mid_hz (FLOAT):

What it does: The frequency where the low band transitions into the mid band.

Use & Why: Defines the boundary between the low and mid-frequency ranges for independent compression. Lower values keep more of the fundamental bass in the low band.

Range: 20.0 Hz to 1000.0 Hz (default: 250.0 Hz).

mb_crossover_mid_high_hz (FLOAT):

What it does: The frequency where the mid band transitions into the high band.

Use & Why: Defines the boundary between the mid and high-frequency ranges.

Range: 1000.0 Hz to 15000.0 Hz (default: 4000.0 Hz).

Multiband Compressor Parameters (Each band has its own independent settings. Active when comp_type is "Multiband")
mb_low_threshold_db, mb_low_ratio, mb_low_attack_ms, mb_low_release_ms, mb_low_makeup_gain_db, mb_low_soft_knee_db:

What they do: These parameters control the compression for the low-frequency band (below mb_crossover_low_mid_hz).

Use & Why: Allows you to control the punch, sustain, and density of the bass independently from other frequencies. For instance, you might use a higher ratio and slower release here to get a tighter, more sustained bass.

Defaults: -16.0 dB, 2.0, 40.0 ms, 350.0 ms, 0.0 dB, 0.0 dB.

mb_mid_threshold_db, mb_mid_ratio, mb_mid_attack_ms, mb_mid_release_ms, mb_mid_makeup_gain_db, mb_mid_soft_knee_db:

What they do: These parameters control the compression for the mid-frequency band (between mb_crossover_low_mid_hz and mb_crossover_mid_high_hz).

Use & Why: Crucial for vocals, guitars, and most musical elements. Allows for balancing clarity and presence without affecting the low end or harshness in the highs.

Defaults: -14.0 dB, 1.5, 25.0 ms, 200.0 ms, 0.0 dB, 0.0 dB.

mb_high_threshold_db, mb_high_ratio, mb_high_attack_ms, mb_high_release_ms, mb_high_makeup_gain_db, mb_high_soft_knee_db:

What they do: These parameters control the compression for the high-frequency band (above mb_crossover_mid_high_hz).

Use & Why: Used to manage sibilance, cymbals, and overall brightness. Gentle compression here can smooth out harshness.

Defaults: -12.0 dB, 1.5, 10.0 ms, 120.0 ms, 0.0 dB, 0.0 dB.

Limiting
enable_limiter (BOOLEAN):

What it does: Enables or bypasses the Limiter.

Use & Why: Always recommended to have the limiter enabled in a mastering chain to prevent digital clipping and maximize loudness without distortion.

Default: True.

limiter_ceiling_db (FLOAT):

What it does: The absolute maximum peak output level (in dBFS) the limiter will allow. No signal will exceed this level.

Use & Why: Typically set just below 0.0 dB (e.g., -0.1 dB or -0.3 dB) to provide a small safety margin before the absolute digital limit.

Range: -6.0 dB to 0.0 dB (default: -0.1 dB).

limiter_lookahead_ms (FLOAT):

What it does: The time (in milliseconds) the limiter "looks ahead" into the audio signal to detect incoming peaks.

Use & Why: This is a crucial feature for limiters. By detecting peaks before they occur, the limiter can apply gain reduction preemptively, preventing the audio from ever hitting the ceiling and causing distortion. Without lookahead, a limiter would react only after a peak, potentially causing clicks or overs.

Range: 0.1 ms to 10.0 ms (default: 2.0 ms).

limiter_release_ms (FLOAT):

What it does: The time (in milliseconds) it takes for the limiter to smoothly return to unity gain after reducing a peak.

Use & Why: Controls how quickly the limiter recovers. A very fast release can cause "pumping" artifacts, while a very slow release might keep the signal too quiet for too long.

Range: 1.0 ms to 500.0 ms (default: 50.0 ms).

3. In-Depth Nerd Technical Information
This section dives into the digital signal processing (DSP) techniques and implementation details within the Mastering Chain Node, for those who want to understand what's happening under the hood.

Audio Data Handling and Normalization
The node is designed to be robust with various audio input formats from ComfyUI.

Input Flexibility: It attempts to extract the waveform from common dictionary keys ('waveform', 'audio', 'samples') or directly use a tensor if passed.

Dimensionality: Internally, it converts the input PyTorch tensor into a NumPy array and normalizes its dimensionality.

A 3D tensor (batch_size, num_channels, num_samples) is reduced to (num_channels, num_samples), assuming a batch size of 1.

A 1D tensor (num_samples,) (mono audio) is reshaped to (1, num_samples) to ensure consistent channel-based processing throughout the chain.

Output Conversion: After all DSP steps, the processed NumPy array is converted back to a PyTorch tensor, reshaped to (1, num_channels, num_samples), and wrapped in a dictionary for ComfyUI's AUDIO output type.

Clipping: Aggressive clipping to [-1.0, 1.0] is applied after each major stage (Global Gain, EQ, Compression, Limiting) to prevent values from exceeding the valid digital audio range, which could lead to hard clipping or NaN/Inf propagation.

Global Gain
Conversion: Gain in dB is converted to an amplitude multiplier using the formula 10 
( 
20.0
dB
​
 )
 .

Application: Simple multiplication of the audio samples by this linear gain factor.

Equalization (EQ)
The EQ section utilizes Second-Order Sections (SOS) filters from scipy.signal for numerical stability, especially when cascading multiple filters. All filter coefficients are calculated with numpy.float64 precision.

High-Shelf Filter: Implemented as a biquad filter. While a eq_high_shelf_q input is provided, the internal Q factor for this filter is fixed at 0.707 for standard Butterworth-like shelf behavior, which provides a smooth, predictable slope.

Low-Shelf Filter: Also a biquad filter, fixed internal Q of 0.707. The _design_low_shelf_filter method correctly applies formulas for both boost and cut scenarios.

Parametric (Bell) Filters (_design_peaking_filter): These are classic biquad peaking filters. The gain (A), center frequency (ω), and Q factor are directly translated from the user inputs to calculate the filter coefficients based on common digital biquad filter design equations (often derived from the Audio EQ Cookbook).

Filter Application: The _apply_filters_to_audio method iteratively applies each designed SOS filter to the audio data using scipy.signal.sosfilt. A small epsilon (1e-12) is added to all-zero audio to prevent NaN propagation in filter computations.

Compression
Both single-band and multiband compressors are implemented as feed-forward designs, meaning the gain reduction is calculated based on the input signal.

RMS Envelope Detection: Rather than simple peak detection, the compressor uses an RMS (Root Mean Square) envelope follower. This is achieved by smoothing the squared absolute value of the audio signal using an IIR (Infinite Impulse Response) filter (a simple one-pole low-pass filter with attack/release coefficients). This provides a more natural and musical response than peak-based detection, as it reacts to the average power of the signal.

attack_coeff = np.exp(-1 / attack_samples)

release_coeff = np.exp(-1 / release_samples)

Gain Computer with Soft Knee:

The input level (from the RMS envelope) is converted to dB.

Hard Knee: If soft_knee_db is 0.0, gain reduction is applied instantly once the signal exceeds the threshold_db.

Soft Knee: If soft_knee_db > 0, the compression ratio is gradually introduced over a specified dB range around the threshold. This reduces audible artifacts associated with abrupt gain changes.

Gain Smoother: The calculated target gain reduction is smoothed over time using separate attack and release coefficients. This prevents instantaneous (and audible) gain changes, ensuring smooth transitions.

Makeup Gain: Applied as a simple linear multiplication after gain reduction.

Multiband Compression
Linkwitz-Riley Crossovers (_design_linkwitz_riley_crossover):

The node uses 4th-order Linkwitz-Riley filters for band splitting. These are preferred in professional audio for their phase coherence (summing the low-pass and high-pass outputs results in a flat frequency response with no phase distortion) and gentle 24 dB/octave slopes.

A 4th-order Linkwitz-Riley filter is derived by cascading two 2nd-order Butterworth filters of the same type (e.g., two 2nd-order low-pass Butterworth filters for a 4th-order Linkwitz-Riley low-pass).

Zero-Phase Filtering: Critically, the scipy.signal.sosfiltfilt function is used. This applies the filter forward and then backward, effectively creating a zero-phase filter. This is essential for crossovers to ensure that the different frequency bands remain perfectly aligned in time, preventing smearing or comb-filtering artifacts when recombined.

Band Splitting:

Low band: Input audio filtered with a low-pass at mb_crossover_low_mid_hz.

High band: Input audio filtered with a high-pass at mb_crossover_mid_high_hz.

Mid band: Input audio first high-pass filtered at mb_crossover_low_mid_hz, then low-pass filtered at mb_crossover_mid_high_hz.

Independent Processing: Each band (low, mid, high) is then fed into its own instance of the _apply_single_band_compression function, allowing for independent dynamic control.

Recombination: The processed bands are simply summed together to reconstruct the full-range audio.

Limiting
Brickwall Peak Limiter with Lookahead: This is a crucial final stage to prevent overs.

Lookahead Buffer: The input audio is padded at the end by limiter_lookahead_samples (calculated from limiter_lookahead_ms and sample_rate). This allows the limiter to "see" upcoming peaks.

Peak Detection: For each sample, the limiter examines a window of audio (current_sample + lookahead_samples) to find the maximum absolute peak.

Instantaneous Attack: If a peak is detected that would exceed the limiter_ceiling_amplitude, the gain reduction is applied immediately (a "brickwall" action). The current_gain_reduction_factor is set to the minimum of its current value and the desired_gain_reduction_factor for the detected peak, ensuring the ceiling is never breached.

Exponential Release: After the peak passes, the gain reduction smoothly decays back to unity gain (no reduction) using an exponential release coefficient, similar to the compressor's release stage.

Waveform Plotting
Matplotlib: matplotlib.pyplot is used to generate waveform plots.

Mono Mix: For stereo audio, channels are averaged to create a mono mix for a single waveform visualization.

Image Conversion: The plot is saved to an in-memory BytesIO buffer as a PNG, then opened with PIL.Image, converted to RGB, and finally transformed into a NumPy array (float32) normalized to 0-1, and then into a PyTorch tensor with a batch dimension (1, height, width, channels) for ComfyUI's IMAGE output.

This node leverages robust scientific computing libraries (numpy, scipy) and standard DSP algorithms to provide a powerful and flexible audio mastering solution within the ComfyUI environment.