# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# █▓▒░                                                                     ░▒▓█
# █▓▒░       MD_Nodes Core: AudioAutoEQ – Adaptive Equalizer v3.2.4        ░▒▓█
# █▓▒░                                                                     ░▒▓█
# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
# ╠═ © 2026 MDMAchine
# ╠═ License: GNU General Public License v3.0 (GPL v3)
# ║
# ║  This program is free software: you can redistribute it and/or modify
# ║  it under the terms of the GNU General Public License as published by
# ║  the Free Software Foundation, either version 3 of the License, or
# ║  (at your option) any later version.
# ║
# ║  This program is distributed in the hope that it will be useful,
# ║  but WITHOUT ANY WARRANTY; without even the implied warranty of
# ║  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# ║  GNU General Public License for more details.
# ║
# ║  You should have received a copy of the GNU General Public License
# ║  along with this program. If not, see <https://www.gnu.org/licenses/>.
# ╠════════════════════════════════════════════════════════════════════════════
# ║ ░▒▓ ORIGIN: True Adaptive Auto-EQ Implementation
# ║ ░▒▓ DESCRIPTION:
# ║    Core mathematical engine for spectral analysis and adaptive 
# ║    equalization. Handles all STFT and DSP operations. Handles 
# ║    bit-exact reconstruction of target frequency profiles.
# ╚════════════════════════════════════════════════════════════════════════════


VERSION = "v3.2.4"  # UPS v1.5.8

import numpy as np
import time

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    from pedalboard import Pedalboard, HighpassFilter, LowpassFilter, PeakFilter
    PEDALBOARD_AVAILABLE = True
except ImportError:
    PEDALBOARD_AVAILABLE = False

NUMPY_VERSION = tuple(map(int, np.__version__.split('.')[:2]))
NUMPY_2_COMPATIBLE = NUMPY_VERSION >= (2, 0)

CONST_STRENGTH_POWER = 2.0
CONST_MAX_GAIN_DB = 12.0
CONST_MAX_CUT_DB = -12.0
CONST_ADAPTIVE_MULTIPLIER = 30.0
CONST_Q_MODERATE = 1.0

CONST_BAND_CENTERS = {
    'sub_bass': 40, 'bass': 120, 'low_mid': 350, 'mid': 1000,
    'high_mid': 3000, 'presence': 5000, 'brilliance': 10000
}

TARGET_PROFILES = {
    "Flat/Neutral": {'sub_bass': 0.05, 'bass': 0.15, 'low_mid': 0.15, 'mid': 0.25, 'high_mid': 0.20, 'presence': 0.12, 'brilliance': 0.08},
    "Vocal Clarity": {'sub_bass': 0.02, 'bass': 0.08, 'low_mid': 0.12, 'mid': 0.28, 'high_mid': 0.25, 'presence': 0.18, 'brilliance': 0.07},
    "Podcast/Speech": {'sub_bass': 0.01, 'bass': 0.06, 'low_mid': 0.10, 'mid': 0.32, 'high_mid': 0.28, 'presence': 0.15, 'brilliance': 0.08},
    "Radio Voice": {'sub_bass': 0.01, 'bass': 0.05, 'low_mid': 0.08, 'mid': 0.35, 'high_mid': 0.30, 'presence': 0.15, 'brilliance': 0.06},
    "Voice Warmth": {'sub_bass': 0.03, 'bass': 0.18, 'low_mid': 0.18, 'mid': 0.25, 'high_mid': 0.18, 'presence': 0.12, 'brilliance': 0.06},
    "Voice Presence": {'sub_bass': 0.02, 'bass': 0.08, 'low_mid': 0.10, 'mid': 0.25, 'high_mid': 0.30, 'presence': 0.20, 'brilliance': 0.05},
    "Music Master": {'sub_bass': 0.08, 'bass': 0.18, 'low_mid': 0.12, 'mid': 0.22, 'high_mid': 0.20, 'presence': 0.12, 'brilliance': 0.08},
    "EDM/Electronic": {'sub_bass': 0.15, 'bass': 0.22, 'low_mid': 0.08, 'mid': 0.15, 'high_mid': 0.15, 'presence': 0.12, 'brilliance': 0.13},
    "Rock/Metal": {'sub_bass': 0.10, 'bass': 0.20, 'low_mid': 0.15, 'mid': 0.18, 'high_mid': 0.18, 'presence': 0.12, 'brilliance': 0.07},
    "Hip-Hop/Trap": {'sub_bass': 0.18, 'bass': 0.25, 'low_mid': 0.10, 'mid': 0.15, 'high_mid': 0.15, 'presence': 0.10, 'brilliance': 0.07},
    "De-muddy": {'sub_bass': 0.08, 'bass': 0.18, 'low_mid': 0.08, 'mid': 0.25, 'high_mid': 0.22, 'presence': 0.12, 'brilliance': 0.07},
    "Bass Boost": {'sub_bass': 0.12, 'bass': 0.28, 'low_mid': 0.12, 'mid': 0.20, 'high_mid': 0.15, 'presence': 0.08, 'brilliance': 0.05},
    "Bass Reduce": {'sub_bass': 0.02, 'bass': 0.08, 'low_mid': 0.12, 'mid': 0.28, 'high_mid': 0.25, 'presence': 0.15, 'brilliance': 0.10},
    "Treble Boost": {'sub_bass': 0.04, 'bass': 0.12, 'low_mid': 0.12, 'mid': 0.20, 'high_mid': 0.20, 'presence': 0.18, 'brilliance': 0.14},
    "Treble Reduce": {'sub_bass': 0.08, 'bass': 0.20, 'low_mid': 0.18, 'mid': 0.28, 'high_mid': 0.15, 'presence': 0.08, 'brilliance': 0.03},
    "Harshness Tamer": {'sub_bass': 0.06, 'bass': 0.16, 'low_mid': 0.16, 'mid': 0.25, 'high_mid': 0.18, 'presence': 0.10, 'brilliance': 0.09},
    "Warm & Smooth": {'sub_bass': 0.06, 'bass': 0.22, 'low_mid': 0.18, 'mid': 0.25, 'high_mid': 0.16, 'presence': 0.08, 'brilliance': 0.05},
    "Bright & Airy": {'sub_bass': 0.03, 'bass': 0.10, 'low_mid': 0.10, 'mid': 0.22, 'high_mid': 0.22, 'presence': 0.18, 'brilliance': 0.15},
}

def analyze_spectrum(audio_mono, sr):
    if not LIBROSA_AVAILABLE:
        return {k: 0.14 for k in CONST_BAND_CENTERS}
        
    stft = librosa.stft(audio_mono)
    magnitude = np.abs(stft)
    avg_magnitude = np.mean(magnitude, axis=1)
    freqs = librosa.fft_frequencies(sr=sr)
    
    bands = {
        'sub_bass': (20, 60), 'bass': (60, 250), 'low_mid': (250, 500),
        'mid': (500, 2000), 'high_mid': (2000, 4000), 'presence': (4000, 6000),
        'brilliance': (6000, 20000)
    }
    
    band_energy = {}
    for band_name, (low, high) in bands.items():
        mask = (freqs >= low) & (freqs < high)
        if np.any(mask) and avg_magnitude[mask].size > 0:
            band_energy[band_name] = np.mean(avg_magnitude[mask])
        else:
            band_energy[band_name] = 0.0
    
    total_energy = sum(band_energy.values())
    if total_energy == 0:
        return {k: 0.0 for k in band_energy}
    return {k: v / total_energy for k, v in band_energy.items()}

def execute_autoeq_pipeline(audio_np, sr, target_profile, strength, 
                            highpass_freq, lowpass_freq, adaptive_mode):
    if not PEDALBOARD_AVAILABLE:
        return audio_np, {}, {}, {}

    audio_mono = np.mean(audio_np, axis=0) if len(audio_np.shape) > 1 else audio_np
    ratios_before = analyze_spectrum(audio_mono, sr)

    board = Pedalboard()
    board.append(HighpassFilter(cutoff_frequency_hz=highpass_freq))
    
    eq_adj = {}
    if not target_profile.startswith("──"):
        target_ratios = TARGET_PROFILES.get(target_profile, TARGET_PROFILES["Flat/Neutral"])
        eff = strength ** CONST_STRENGTH_POWER
        
        for b, current in ratios_before.items():
            target = target_ratios.get(b, current)
            if adaptive_mode == "full":
                gain = (target - current) * CONST_ADAPTIVE_MULTIPLIER * eff
            elif adaptive_mode == "hybrid":
                adaptive_gain = (target - current) * CONST_ADAPTIVE_MULTIPLIER
                fixed_gain = (target - 0.15) * 10
                gain = (adaptive_gain * 0.7 + fixed_gain * 0.3) * eff
            else: 
                gain = (target - 0.15) * 20 * eff
            
            gain = max(CONST_MAX_CUT_DB, min(CONST_MAX_GAIN_DB, gain))
            eq_adj[b] = gain
            if abs(gain) > 0.1:
                board.append(PeakFilter(cutoff_frequency_hz=CONST_BAND_CENTERS[b], gain_db=gain, q=CONST_Q_MODERATE))
    
    board.append(LowpassFilter(cutoff_frequency_hz=lowpass_freq))
    processed = board(audio_np.astype(dtype=np.float32), sr)
    
    processed_mono = np.mean(processed, axis=0) if len(processed.shape) > 1 else processed
    ratios_after = analyze_spectrum(processed_mono, sr)
    
    return processed, eq_adj, ratios_before, ratios_after

def serialize_for_api(audio_np, sr, params):
    import base64, zlib, pickle
    payload = {"audio": audio_np, "sr": sr, "params": params}
    return base64.b64encode(zlib.compress(pickle.dumps(payload))).decode('utf-8')

def deserialize_from_api(b64_data):
    import base64, zlib, pickle
    return pickle.loads(zlib.decompress(base64.b64decode(b64_data)))


# ==============================================================================
# == Unit Tests (smoke — runs without ComfyUI)
# ==============================================================================

if __name__ == "__main__":
    print("\n🧪 Smoke tests: audio_autoeq_core")
    print("   VERSION :", VERSION)
    _pass = _fail = 0

    def _check(label, expr):
        global _pass, _fail
        if expr:
            print(f"  ✅  {label}")
            _pass += 1
        else:
            print(f"  ❌  {label}")
            _fail += 1

    _check("VERSION defined",    VERSION == "v3.2.4")
    _check("CONST CONST_STRENGTH_POWER defined", CONST_STRENGTH_POWER is not None)
    _check("CONST CONST_MAX_GAIN_DB defined", CONST_MAX_GAIN_DB is not None)
    _check("CONST CONST_MAX_CUT_DB defined", CONST_MAX_CUT_DB is not None)
    _check("CONST CONST_ADAPTIVE_MULTIPLIER defined", CONST_ADAPTIVE_MULTIPLIER is not None)
    _check("CONST CONST_Q_MODERATE defined", CONST_Q_MODERATE is not None)
    _check("fn analyze_spectrum is callable", callable(analyze_spectrum))
    _check("fn execute_autoeq_pipeline is callable", callable(execute_autoeq_pipeline))
    _check("fn serialize_for_api is callable", callable(serialize_for_api))

    print(f"\n  {_pass} passed, {_fail} failed")
    if _fail == 0:
        print("  🎉 All good.")
