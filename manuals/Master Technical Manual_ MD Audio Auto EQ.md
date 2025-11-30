***

## **Master Technical Manual Template (v2.0 - Exhaustive)**

### **Node Name: `MD_AudioAutoEQ`**
**Display Name:** `MD: Audio Auto EQ`
**Category:** `MD_Nodes/Audio`
**Version:** `1.1.0`
**Last Updated:** `Nov 2025`

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
    4.1. Audio & Profile Selection
    4.2. EQ Strength & Filtering
    4.3. Debugging
5.  **Applied Use-Cases & Recipes**
    5.1. Recipe 1: Podcast Voice Enhancement
    5.2. Recipe 2: Lo-Fi/Vintage Effect
    5.3. Recipe 3: Corrective De-Muddying
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
The **MD: Audio Auto EQ** is an intelligent spectral shaping node for ComfyUI. It automates the equalization process by first analyzing the frequency spectrum of the input audio and then constructing a custom parametric EQ chain to match a user-selected target profile. Unlike static EQ presets, this node adapts its processing based on the specific energy distribution of the incoming signal, applying cuts or boosts only where necessary to achieve the desired tonal balance.

#### **1.2. Conceptual Category**
**Adaptive Spectral Processor**
This node bridges the gap between raw analysis tools and audio effect processors. It functions as an "AI Mixing Engineer" that listens to the track (via FFT analysis) and makes decisions on how to eq it.

#### **1.3. Problem Domain & Intended Application**
* **Problem Domain:** Raw audio from generative models (TTS, MusicGen) often suffers from "spectral masking" (muddy low-mids), harsh resonances, or a lack of presence. Manually fixing these issues requires a trained ear and complex plugin chains.
* **Intended Application (Use-Cases):**
    * **TTS Post-Processing:** Automatically cleaning up "boxy" or muffled synthetic voices.
    * **Creative Styling:** Forcing a "Vintage" or "Radio" tonality onto pristine audio.
    * **Mix Balancing:** Quickly identifying and taming dominant frequency bands in musical generations.
* **Non-Application:**
    * **Surgical Noise Removal:** While it includes high/low pass filters, it is not a spectral denoiser or background noise remover.

#### **1.4. Key Features (Functional & Technical)**
* **Functional Features:**
    * **Adaptive Profiling:** Contains 18+ targeted profiles (e.g., "Vocal Clarity", "Warm & Smooth", "EDM/Electronic") that dictate the target tonal curve.
    * **Visual Feedback:** Outputs high-resolution waveform plots for both the pre-processed and post-processed signals.
    * **Detailed Reporting:** Returns a string output detailing the energy percentage of 7 distinct frequency bands (Sub-Bass to Brilliance).
* **Technical Features:**
    * **FFT Analysis:** Utilizes `librosa.stft` to compute precise spectral energy density.
    * **VST-Quality DSP:** Leverages Spotify's `pedalboard` library for high-quality, artifact-free filtering (Highpass, Lowpass, Peak).
    * **Headless Plotting:** Implements a thread-safe `matplotlib` backend (`Agg`) to ensure stability in server environments without display servers.

### **2. Core Concepts & Theory**

#### **2.1. Theoretical Background**
The node operates on the principle of **Spectral Energy Analysis**. Audio signals are composed of various frequencies vibrating at different amplitudes. By dividing the audible spectrum (20Hz - 20kHz) into perceptual bands (Sub-Bass, Bass, Low-Mid, Mid, High-Mid, Presence, Brilliance), the node calculates the relative power of each band.

If a specific band's energy deviates significantly from the expected norm of the selected profile (e.g., too much Low-Mid energy in a "Vocal Clarity" profile), the node constructs a corrective filter (a "Peak Filter") at the center frequency of that band to attenuate or boost it.

#### **2.2. Mathematical & Algorithmic Formulation**
**Spectral Energy Calculation:**
The average magnitude $M_{band}$ for a frequency band defined by $[f_{low}, f_{high})$ is calculated as:
$$M_{band} = \frac{1}{N} \sum_{k \in K} |X(k)|$$
Where $X(k)$ is the magnitude of the Short-Time Fourier Transform (STFT) bin $k$ corresponding to frequencies within the band, and $N$ is the number of bins.

**Filter Application Logic:**
The node applies filters conditionally. For example, in the "Flat/Neutral" profile:
$$ \text{If } \text{Ratio}_{bass} > 1.3 \times \text{AverageRatio} \implies \text{Apply Cut: } -3\text{dB} \times \text{Strength} \text{ @ } 150\text{Hz} $$
This ensures processing is dynamic; if the input is already balanced, minimal filtering occurs.

#### **2.3. Data I/O Deep Dive**
* **Input `audio`:** Standard ComfyUI Audio Dict (`waveform`: Tensor `[B, C, N]`, `sample_rate`: Int). The node automatically handles batch dimension extraction and mono downmixing for analysis.
* **Output `waveform_before` / `waveform_after`:**
    * **Tensor Shape:** `[1, 64, 64, 3]` (Note: The visual container size is dictated by ComfyUI, but the underlying plot is generated at 10x3 inches @ 96 DPI).
    * **Data Type:** `torch.float32`, normalized `[0.0, 1.0]`.
    * **Visuals:** Blue line = waveform; Red dashed line = Peak amplitude; Green dotted line = RMS level.

#### **2.4. Strategic Role in the ComfyUI Graph**
* **Placement Context:** Best placed immediately after audio generation or loading, but *before* final mastering limiters or reverb effects.
* **Synergistic Nodes:**
    * **Input:** `VHS_Audio_Load`, `MusicGen`.
    * **Output:** `MD_AutoMasterNode` (place AutoEQ *before* AutoMaster for best results).

### **3. Node Architecture & Workflow**

#### **3.1. Node Interface & Anatomy**
The node features a straightforward interface with the audio input on the left and the profile/strength controls in the center.

#### **3.2. Input Port Specification**
* **`audio` (AUDIO)** - (Required)
    * **Description:** The raw audio signal to be equalized.
    * **Impact:** Empty inputs trigger a warning and return blank/zero outputs.

#### **3.3. Output Port Specification**
* **`audio` (AUDIO)**
    * **Description:** The processed audio signal.
* **`analysis_report` (STRING)**
    * **Description:** A text summary containing the sample rate, selected profile, applied strength, and the calculated energy distribution for all 7 frequency bands.
* **`waveform_before` (IMAGE)**
    * **Description:** Visual plot of the source audio.
* **`waveform_after` (IMAGE)**
    * **Description:** Visual plot of the equalized audio, useful for verifying dynamic range changes or peak reduction.

#### **3.4. Workflow Schematics**

**Minimal Functional Graph:**
`[Audio Loader] -> [MD_AudioAutoEQ (Profile: Vocal Clarity)] -> [Audio Save]`

```json
{
  "nodes": [
    {
      "id": 15,
      "type": "MD_AudioAutoEQ",
      "inputs": {
        "audio": ["10", 0],
        "target_profile": "Vocal Clarity",
        "strength": 0.8
      }
    }
  ]
}
```

### **4. Parameter Specification**

#### **4.1. Audio & Profile Selection**
* **`target_profile` (COMBO)**
    * **Options:** `Flat/Neutral`, `Vocal Clarity`, `Vocal De-esser`, `Podcast/Speech`, `Music Master`, `Warm & Smooth`, `Bright & Airy`, `De-muddy`, `Bass Boost`, `Bass Reduce`, `Treble Boost`, `Treble Reduce`, `Lo-Fi/Vintage`, `Modern/Crisp`, `EDM/Electronic`, `Acoustic/Natural`, `Radio Voice`, `ASMR/Whisper Enhance`.
    * **Algorithmic Impact:** Selects the specific set of conditional logic statements (thresholds and filter frequencies) used in `create_eq_chain`. For example, `Radio Voice` aggressively boosts 1kHz and cuts bass/treble, while `Vocal Clarity` focuses on boosting 3kHz presence.

#### **4.2. EQ Strength & Filtering**
* **`strength` (FLOAT)**
    * **Default:** `0.7` | **Min:** `0.0` | **Max:** `1.0` | **Step:** `0.05`
    * **Function:** A global scaler for all gain adjustments.
    * **Logic:** If the profile dictates a -4dB cut, and strength is 0.5, the actual cut applied is -2dB.
    * **Use:** Lower this value to blend the effect (similar to a Dry/Wet knob, but applied to filter gains).
* **`highpass_freq` (FLOAT)**
    * **Default:** `80.0` | **Min:** `20.0` | **Max:** `500.0`
    * **Function:** Sets the cutoff for the Highpass Filter. Frequencies below this are removed. Essential for cleaning up "rumble".
* **`lowpass_freq` (FLOAT)**
    * **Default:** `15000.0` | **Min:** `8000.0` | **Max:** `20000.0`
    * **Function:** Sets the cutoff for the Lowpass Filter. Frequencies above this are attenuated. Useful for removing digital aliasing or hiss.

#### **4.3. Debugging**
* **`debug` (BOOLEAN)**
    * **Default:** `False`
    * **Function:** If enabled, prints verbose information to the console, including input tensor shapes, calculated band ratios, and the exact count of filters applied to the chain.

### **5. Applied Use-Cases & Recipes**

#### **5.1. Recipe: Podcast Voice Enhancement**
* **Objective:** Make a TTS voice sound professional and broadcast-ready.
* **Settings:**
    * `target_profile`: "Podcast/Speech"
    * `strength`: 0.8
    * `highpass_freq`: 100 Hz (Removes plosives/pops)
* **Result:** Reduced proximity effect (muddy bass), boosted presence (1.2kHz - 3kHz), and rolled-off highs for a focused sound.

#### **5.2. Recipe: Lo-Fi/Vintage Effect**
* **Objective:** Simulate an old recording or telephone effect.
* **Settings:**
    * `target_profile`: "Lo-Fi/Vintage"
    * `strength`: 1.0 (Full intensity)
    * `highpass_freq`: 300 Hz
    * `lowpass_freq`: 6000 Hz
* **Result:** Narrow bandwidth audio with a distinct mid-range "honk" characteristic of old hardware.

#### **5.3. Recipe: Corrective De-Muddying**
* **Objective:** Clean up a music track that sounds muffled or undefined.
* **Settings:**
    * `target_profile`: "De-muddy"
    * `strength`: 0.6
    * `highpass_freq`: 40 Hz
* **Result:** Surgical cuts are applied to the 200Hz - 500Hz region, clearing space for the kick drum and vocals to sit better in the mix.

### **6. Implementation Deep Dive**

#### **6.1. Source Code Walkthrough**
The core logic resides in `process_audio` and `create_eq_chain`.
1.  **Analysis (`analyze_spectrum`):**
    * Computes STFT using `librosa.stft`.
    * Defines frequency band limits (e.g., Bass: 60-250Hz).
    * Masks the frequency array to calculate mean magnitude per band.
    * Normalizes these values into a ratio (percentage of total energy).
2.  **Filter Construction (`create_eq_chain`):**
    * Initializes a `Pedalboard()` object.
    * Adds global Highpass/Lowpass filters first.
    * Evaluates the `target_profile`.
    * *Example Logic:*
        ```python
        if target_profile == "Flat/Neutral":
            avg_ratio = 1.0 / 7
            # If Bass energy is 30% higher than the average band energy...
            if band_ratios['bass'] > avg_ratio * 1.3:
                # ...Add a cut at 150Hz, scaled by strength.
                board.append(PeakFilter(cutoff_frequency_hz=150, gain_db=-3 * strength, q=1.0))
        ```
3.  **Processing:**
    * The constructed `board` processes the audio array: `processed_audio = eq_board(audio_np_float, sample_rate)`.
4.  **Visualization:**
    * Both input and output arrays are passed to `_plot_waveform_to_tensor`, which generates Matplotlib figures, saves them to a memory buffer, and converts them to tensors for ComfyUI preview.

#### **6.2. Dependencies & External Calls**
* **`librosa`**: Used for FFT/STFT spectrum analysis.
* **`pedalboard`**: Provides the DSP implementation for EQ filters. It is highly optimized (C++ backend) and preferred over `scipy.signal` for audio effect chains due to ease of chaining.
* **`matplotlib`**: Used with the `Agg` backend for generating waveform images without a GUI.
* **`PIL` (Pillow)**: Converts memory buffers to image objects.

#### **6.3. Performance & Resource Analysis**
* **Execution Target:** CPU.
* **Memory:** Moderate. STFT analysis requires creating a complex spectrogram array. For very long audio files (>10 mins), this may consume significant RAM.
* **Speed:** Analysis is fast (FFT is efficient). Pedalboard processing is real-time capable. The slowest operation is typically the generation of the Matplotlib plots (Visuals).

#### **6.4. Tensor Lifecycle Analysis**
1.  **Input:** `torch.Tensor` (GPU/CPU) $\rightarrow$ `numpy.ndarray` (CPU).
2.  **Analysis:** `numpy.ndarray` $\rightarrow$ `librosa.stft` (Complex64) $\rightarrow$ `numpy.float32` (Magnitude).
3.  **Processing:** `numpy.ndarray` (Float32) $\rightarrow$ `Pedalboard` $\rightarrow$ `numpy.ndarray` (Float32).
4.  **Output:** `numpy.ndarray` $\rightarrow$ `torch.Tensor` (CPU) $\rightarrow$ Moved to original device.

### **7. Troubleshooting & Diagnostics**

#### **7.1. Error Code Reference**
* **`Zero total energy detected in spectrum analysis`**
    * **Cause:** Input audio is purely silent (all zeros).
    * **Behavior:** The node logs a warning and sets band ratios to 0.0.
* **`[MD_AudioAutoEQ] Skipping plot: Empty audio data`**
    * **Cause:** The input audio tensor existed but had a size of 0.
    * **Behavior:** Returns a blank black image to prevent the workflow from crashing.

#### **7.2. Unexpected Visual Artifacts & Mitigation**
* **Artifact:** Visual plots appear completely flat/empty.
    * **Cause:** Matplotlib backend conflict or extremely low amplitude audio (< 1e-6).
    * **Mitigation:** Ensure the input audio is normalized before entering the EQ node, or check console logs for Matplotlib errors.
* **Artifact:** Audio sounds "phasey" or thin after EQ.
    * **Cause:** Extreme EQ cuts in adjacent bands (e.g., cutting Low-Mid and Mid heavily).
    * **Mitigation:** Reduce the `strength` parameter (try 0.4 - 0.5) to lessen the phase shift introduced by the IIR filters.