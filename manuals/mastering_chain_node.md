# Comprehensive Manual: Custom Audio Mastering Chain (v1.2a)

Welcome to the complete guide for the **Custom Audio Mastering Chain** node, a professional-grade audio finishing toolkit for ComfyUI. This manual details every feature, from basic gain staging to advanced multiband compression, empowering you to sculpt and polish your generated audio with precision.

---

### **Table of Contents**

1.  **Introduction**
    * What is the Mastering Chain Node?
    * Who is this Node For?
    * Key Features in Version 1.2a
2.  **Installation**
3.  **Core Concepts: The Art of Mastering**
    * The Mastering Chain: Order Matters
    * Equalization (EQ): Sculpting the Tone
    * Dynamic Range Compression: Controlling the Power
    * The Lookahead Limiter: The Final Safety Net
4.  **Node Setup and Workflow**
    * How to Use It: A Step-by-Step Guide
5.  **Parameter Deep Dive**
    * Primary & Global Controls
    * Equalizer Controls (Shelves & Parametric)
    * Compressor Controls (Single-Band & Multiband)
    * Limiter Controls
6.  **Practical Recipes & Use Cases**
    * Recipe 1: Gentle Mastering for a Full Music Track
    * Recipe 2: Surgical Fix for a Boomy Kick Drum
    * Recipe 3: Adding Clarity and "Air" to Vocals or Synths
7.  **Technical Deep Dive**
    * The Order of Operations
8.  **Troubleshooting & FAQ**

---

## 1. Introduction

### What is the Mastering Chain Node?

The **Custom Audio Mastering Chain** is an all-in-one audio finalization node that brings the essential tools of a professional mastering studio into your ComfyUI workflow. It's designed to be the last step in your audio generation process, taking a raw sound and transforming it into a polished, balanced, and loud final product. It provides sequential processing for gain, equalization, compression, and limiting in a single, powerful interface.

### Who is this Node For?

* **AI Musicians & Producers:** Anyone generating full music tracks who needs to make them sound cohesive, loud, and commercially competitive.
* **Sound Designers:** Creators who need to shape the frequency content and dynamic impact of sound effects to fit a specific context.
* **Video and Animation Creators:** Users who need to process dialogue or background music to ensure clarity and consistent volume.
* **Experimental Audio Artists:** Anyone looking for a powerful tool to radically reshape and color their sounds.

### Key Features in Version 1.2a

* **Complete Processing Chain:** Four essential mastering stages (Gain, EQ, Compression, Limiter) in one node.
* **Advanced Multi-Band EQ:** A high-shelf, a low-shelf, and four fully parametric bands for precise tonal sculpting.
* **Dual-Mode Compression:** Choose between a straightforward **Single-Band** compressor or a powerful **3-Band Multiband** compressor for transparent dynamic control.
* **High-Quality Crossovers:** Multiband mode uses Linkwitz-Riley filters to split audio with minimal phase distortion, preserving audio integrity.
* **Lookahead Peak Limiter:** A "brickwall" limiter that prevents digital clipping and maximizes loudness without audible distortion.
* **Instant Visual Feedback:** Get immediate "before" and "after" waveform visualizations to see the impact of your processing.

---

## 2. 🧰 INSTALLATION: JACK INTO THE MATRIX

This node is part of the **MD Nodes** package. All required Python libraries, including `pedalboard` for the limiter, are listed in the `requirements.txt` and should be installed automatically.

### Method 1: ComfyUI Manager (Recommended)

1.  Open the **ComfyUI Manager**.
2.  Click "Install Custom Nodes".
3.  Search for `MD Nodes` and click "Install".
4.  The manager will download the package and automatically install its dependencies.
5.  **Restart ComfyUI.**

### Method 2: Manual Installation (Git)

1.  Open a terminal or command prompt.
2.  Navigate to your `ComfyUI/custom_nodes/` directory.
3.  Run the following command to clone the repository:
    ```bash
    git clone [https://github.com/MDMAchine/ComfyUI_MD_Nodes.git](https://github.com/MDMAchine/ComfyUI_MD_Nodes.git)
    ```
4.  Install the required dependencies by running:
    ```bash
    pip install -r ComfyUI_MD_Nodes/requirements.txt
    ```
5.  **Restart ComfyUI.**

After restarting, the node and all its features should be fully available. Don’t forget, even gods need to reboot.

---

## 3. Core Concepts: The Art of Mastering

### The Mastering Chain: Order Matters

A mastering chain is a sequence of audio processors. The order is critical because each stage affects the input to the next. This node uses a classic, proven order: **Gain → EQ → Compression → Limiter**. This ensures you can balance the initial level, shape the tone, control the dynamics, and finally, maximize the loudness without distortion.

### Equalization (EQ): Sculpting the Tone

EQ is the art of adjusting the volume of specific frequencies to change the character of a sound. Think of it as a highly advanced tone control.
* **Shelf Filters (Low & High):** These are broad-stroke tools. A **low-shelf** turns down (or up) all the bass frequencies below a certain point, while a **high-shelf** does the same for the treble. They are great for general brightening or warming.
* **Parametric Filters:** These are surgical tools. They target a very specific frequency and can create a narrow "peak" or "dip." This is perfect for fixing problems, like cutting a muddy frequency in a bassline or boosting the "snap" of a snare drum.

### Dynamic Range Compression: Controlling the Power

Dynamic range is the difference between the quietest and loudest parts of your audio. Compression reduces this range, making the overall volume more consistent.
* **Single-Band Compressor:** This is the classic approach. It listens to the entire audio signal and turns the volume down whenever it gets too loud. It's simple and effective for adding "glue" to a mix.
* **Multiband Compressor:** This is a more advanced and transparent tool. It splits the audio into frequency bands (in this case, Low, Mid, and High) and compresses each one independently. This allows you to control the dynamics of the bass without affecting the crispness of the cymbals, leading to a much cleaner and more powerful result.

### The Lookahead Limiter: The Final Safety Net

After EQ and compression have shaped your sound, the limiter's job is to make it loud without distortion. It's a "brickwall" that prevents the audio signal from ever crossing the maximum digital level (0 dBFS). By using **lookahead**, it can see peaks coming a few milliseconds in advance and smoothly turn them down, which is far more transparent than simply clipping them off.

---

## 4. Node Setup and Workflow

### How to Use It: A Step-by-Step Guide

1.  **Connect Audio Source:** Connect the `AUDIO` output from your audio generation or loading node to the `audio` input of the Mastering Chain Node.
2.  **Set Initial Gain:** Use `master_gain_db` to set a good starting level. You want the loudest parts to be active but not constantly hitting the maximum.
3.  **Enable and Configure EQ:** Check `enable_eq`. Use the shelf and parametric bands to shape the tone. Small boosts and cuts often work best.
4.  **Enable and Configure Compression:**
    * Check `enable_comp` and choose your `comp_type` (`Single-Band` or `Multiband`).
    * Set the `threshold` to determine when the compressor starts working.
    * Adjust the `ratio`, `attack`, and `release` to control how the compression sounds.
    * Use `makeup_gain` to restore any volume lost during compression.
5.  **Enable and Configure Limiter:** Check `enable_limiter`. Set the `limiter_ceiling_db` to just below zero (e.g., -0.1) to prevent any possibility of clipping.
6.  **Connect Outputs:**
    * The `AUDIO` output provides the fully processed audio stream, ready to be saved or previewed.
    * The two `IMAGE` outputs show the waveform *before* and *after* processing. Connect these to "Preview Image" nodes to see the effects of your work.
7.  **Queue Prompt:** Run the workflow. The output images will give you immediate feedback on how the dynamics have changed.

---

## 5. Parameter Deep Dive

### Primary & Global Controls

* **`audio`** (Required): The input audio waveform.
* **`sample_rate`** (`INT`): The sample rate of the input audio (e.g., 44100).
* **`master_gain_db`** (`FLOAT`): The overall volume adjustment applied *before* all other processing.

### Equalizer Controls (Shelves & Parametric)

* **`enable_eq`** (`BOOLEAN`): Master switch for the entire EQ section.
* **Low-Shelf:**
    * `enable_low_shelf_eq`: Toggles the low-shelf filter.
    * `eq_low_shelf_gain_db`: Boosts or cuts the bass frequencies.
    * `eq_low_shelf_freq`: Sets the corner frequency where the shelf begins.
* **High-Shelf:**
    * `eq_high_shelf_gain_db`: Boosts or cuts the treble frequencies.
    * `eq_high_shelf_freq`: Sets the corner frequency where the shelf begins.
* **Parametric Bands (1-4):**
    * `enable_param_eqX`: Toggles the specific parametric band.
    * `param_eqX_gain_db`: Boosts or cuts the targeted frequency.
    * `param_eqX_freq`: The center frequency of the band.
    * `param_eqX_q`: The "Q factor" or bandwidth. A high Q is very narrow and surgical; a low Q is broad.

### Compressor Controls (Single-Band & Multiband)

* **`enable_comp`** (`BOOLEAN`): Master switch for the entire compression section.
* **`comp_type`** (`ENUM`): Selects `Single-Band` or `Multiband` operation.

* **Single-Band Parameters:**
    * `comp_threshold_db`: The volume level (in dB) above which compression is applied.
    * `comp_ratio`: The amount of compression. A 4:1 ratio means for every 4dB the signal goes over the threshold, the output only rises by 1dB.
    * `comp_attack_ms`: How quickly the compressor reacts (in milliseconds).
    * `comp_release_ms`: How quickly the compressor stops reacting.
    * `comp_makeup_gain_db`: A final gain stage to compensate for volume lost during compression.
    * `comp_soft_knee_db`: Creates a smoother transition into compression.

* **Multiband Parameters:**
    * `mb_crossover_low_mid_hz` / `mb_crossover_mid_high_hz`: These set the frequencies that divide the audio into Low, Mid, and High bands.
    * Each band (Low, Mid, High) has its own independent set of `threshold`, `ratio`, `attack`, `release`, `makeup_gain`, and `soft_knee` controls.

### Limiter Controls

* **`enable_limiter`** (`BOOLEAN`): Toggles the final limiter stage.
* **`limiter_ceiling_db`** (`FLOAT`): The absolute maximum output level. Set this to -0.1 or -0.3 to prevent inter-sample peaks on consumer devices.
* **`limiter_lookahead_ms`** (`FLOAT`): How far ahead (in ms) the limiter "looks" to anticipate peaks for cleaner processing.
* **`limiter_release_ms`** (`FLOAT`): How quickly the limiter recovers after reducing a peak.

---

## 6. Practical Recipes & Use Cases

### Recipe 1: Gentle Mastering for a Full Music Track

Goal: Add punch, clarity, and competitive loudness to a finished song without sounding over-processed.

* **`master_gain_db`**: `-3.0` (to create headroom before compression).
* **EQ:**
    * **Low-Shelf:** `enable`, `gain=-1dB`, `freq=100Hz` (to gently tighten the low end).
    * **High-Shelf:** `gain=+1.5dB`, `freq=12000Hz` (to add some "air").
* **Compressor (`Multiband`):**
    * **Low Band:** `threshold=-20dB`, `ratio=3:1`, `attack=50ms`, `release=400ms` (to control the bass).
    * **Mid Band:** `threshold=-18dB`, `ratio=2:1`, `attack=25ms`, `release=200ms` (gentle glue for the main instruments).
    * **High Band:** `threshold=-15dB`, `ratio=1.5:1`, `attack=10ms`, `release=120ms` (to lightly tame cymbals).
* **Limiter:** `enable`, `ceiling=-0.1dB`, `lookahead=2.0ms`.

### Recipe 2: Surgical Fix for a Boomy Kick Drum

Goal: Reduce the "mud" from a kick drum around 150Hz without losing its low-end punch.

* **EQ:**
    * **Parametric EQ 1:** `enable`, `gain=-4.0dB`, `freq=150Hz`, `q=5.0` (a sharp cut to remove the unwanted resonance).
    * **Parametric EQ 2:** `enable`, `gain=+2.0dB`, `freq=60Hz`, `q=2.0` (a broad boost to restore the deep punch).
* **Compressor (`Single-Band`):** `enable`, `threshold=-10dB`, `ratio=4:1`, `attack=10ms`, `release=100ms` (to even out the kick's dynamics).
* **Limiter:** `enable`.

### Recipe 3: Adding Clarity and "Air" to Vocals or Synths

Goal: Make a lead instrument or vocal cut through the mix and sound brighter.

* **EQ:**
    * **Parametric EQ 1:** `enable`, `gain=-2.0dB`, `freq=300Hz`, `q=1.5` (to remove any boxiness).
    * **Parametric EQ 2:** `enable`, `gain=+2.5dB`, `freq=5000Hz`, `q=2.0` (to boost presence and clarity).
    * **High-Shelf:** `gain=+2.0dB`, `freq=10000Hz` (to add sparkle and air).
* **Compressor (`Single-Band`):** Use very light settings (`ratio=2:1`) just to even out the level slightly.
* **Limiter:** `enable`.

---

## 7. Technical Deep Dive

### The Order of Operations

Understanding the signal flow is crucial. The audio passes through the enabled modules in this fixed sequence:

1.  **Global Gain:** Sets the initial level for the entire chain.
2.  **Equalizer:** The audio is tonally shaped *before* dynamic processing. This is standard practice, as it allows you to control which frequencies trigger the compressor.
3.  **Compressor:** The dynamics of the EQ'd signal are controlled.
4.  **Limiter:** The final processed signal is brought up to its maximum loudness without clipping.

The "Before" waveform is captured right at the input, and the "After" waveform is captured at the final output, showing the cumulative effect of all enabled stages.

---

## 8. Troubleshooting & FAQ

* **"My audio sounds distorted."**
    * This is likely digital clipping. The most important fix is to **enable the limiter** and set its ceiling to -0.1dB. Also, check your `master_gain_db` and compressor `makeup_gain_db` settings; you may be sending too much level into the next stage.
* **"The sound is 'pumping' or 'breathing' unnaturally."**
    * This is a classic sign of over-compression. Try using a lower `ratio`, a slower `attack` time, or a faster `release` time on your compressor. If using a single-band compressor on a full mix, the bass might be triggering the compression for the whole track; this is a perfect reason to switch to `Multiband` mode.
* **"My audio sounds thin after I used the EQ."**
    * You might have cut too much in the low or mid-range frequencies. EQ is powerful, and small changes go a long way. Try reducing the amount of your cuts (e.g., -2dB instead of -6dB) or using a lower Q value for a broader, more natural shape.
* **"The compressor isn't doing anything."**
    * The compressor only acts on signals that go *above* the `threshold`. If your input audio is too quiet, it may never cross the threshold. Try lowering the `threshold` value or increasing the `master_gain_db` before the compressor stage.
* **"What's the real advantage of Multiband compression?"**
    * It gives you independent dynamic control over different parts of the sound. It lets you clamp down on a boomy bass guitar without making the vocals sound squashed, or de-ess a vocal (reduce sibilance) without affecting the midrange warmth. It's a more transparent and powerful approach for complex audio like a full music track.