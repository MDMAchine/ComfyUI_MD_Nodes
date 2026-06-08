# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# █▓▒░                                                                     ░▒▓█
# █▓▒░          AutoMaster Core Engine – v6.32.1 (Pro-Audio Tilt)          ░▒▓█
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
# ║ ░▒▓ PURPOSE: Fix Tilt Artifacts via Pedalboard / Parallel Blend
# ╚════════════════════════════════════════════════════════════════════════════


VERSION = "v6.32.1"  # UPS v1.5.8

import numpy as np
from scipy import signal
from scipy.signal import lfilter, sosfilt
import pickle
import zlib
import base64
import math

try:
    import pyloudnorm as pln
    PYLOUDNORM_AVAILABLE = True
except ImportError:
    PYLOUDNORM_AVAILABLE = False

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    from pedalboard import Pedalboard, Compressor, Limiter, HighShelfFilter, LowShelfFilter
    PEDALBOARD_AVAILABLE = True
except ImportError:
    PEDALBOARD_AVAILABLE = False

# =================================================================================
# == Configuration Constants
# =================================================================================

CONST_TARGET_SR = 44100
CONST_LIMITER_RELEASE_MS = 100.0     
CONST_SOFT_CLIP_THRESHOLD = 0.99
CONST_BASS_TOLERANCE = 0.5
CONST_HIGH_TOLERANCE = 0.2
CONST_SPIKE_THRESHOLD = 8.0
CONST_EQ_BASS_FREQ = 120
CONST_EQ_HIGH_FREQ = 8000

# Safety Limits
CONST_MAX_SIGNAL_VAL = 32.0          
CONST_SILENCE_THRESH = 1e-7          

# Harmonic Exciter & Safety
CONST_HARMONIC_EXCITER_FREQ = 6000
CONST_HARMONIC_SOOTHE_FREQ = 9000
CONST_SAFETY_SHELF_FREQ = 10000
CONST_SAFETY_SHELF_CUT = -1.5
CONST_SAFETY_SHELF_THRESHOLD_DB = 3.0

_FILTER_CACHE = {}

# =================================================================================
# == DSP Helper Functions
# =================================================================================

def _sanitize(audio):
    if not np.all(np.isfinite(audio)):
        audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)
    return np.clip(audio, -CONST_MAX_SIGNAL_VAL, CONST_MAX_SIGNAL_VAL)

def _is_dead(audio):
    return np.max(np.abs(audio)) < CONST_SILENCE_THRESH

def _get_cached_coeffs(sr, freq, gain_db, Q):
    cache_id = f"{sr}_{freq}_{gain_db}_{Q}"
    if cache_id in _FILTER_CACHE:
        return _FILTER_CACHE[cache_id]
        
    amplitude = math.pow(10, gain_db / 40.0)
    omega = 2.0 * math.pi * (freq / sr)
    sn = math.sin(omega)
    cs = math.cos(omega)
    alpha = sn / (2.0 * Q)
    
    b_terms = [1.0 + alpha * amplitude, -2.0 * cs, 1.0 - alpha * amplitude]
    a_terms = [1.0 + alpha / amplitude, -2.0 * cs, 1.0 - alpha / amplitude]
    
    b_array = np.array(b_terms) / a_terms[0]
    a_array = np.array(a_terms) / a_terms[0]
    
    _FILTER_CACHE[cache_id] = (b_array, a_array)
    return _FILTER_CACHE[cache_id]

def _apply_bell_filter(audio, sr, freq, gain_db, Q=1.2):
    if abs(gain_db) < 0.05 or freq <= 0 or freq >= sr/2: return audio
    b, a = _get_cached_coeffs(sr, freq, gain_db, Q)
    return lfilter(b, a, audio, axis=0)

def _design_shelf_sos(gain_db, freq, sr, shelf_type='low'):
    if abs(gain_db) < 0.001: return None
    A = 10**(gain_db/40.0)
    w0 = 2*np.pi*freq/sr
    alpha = np.sin(w0)/2.0*np.sqrt(2)
    cos_w0 = np.cos(w0)
    
    if shelf_type == 'low':
        b0 = A*((A+1)-(A-1)*cos_w0+2*np.sqrt(A)*alpha)
        b1 = 2*A*((A-1)-(A+1)*cos_w0)
        b2 = A*((A+1)-(A-1)*cos_w0-2*np.sqrt(A)*alpha)
        a0 = (A+1)+(A-1)*cos_w0+2*np.sqrt(A)*alpha
        a1 = -2*((A-1)+(A+1)*cos_w0)
        a2 = (A+1)+(A-1)*cos_w0-2*np.sqrt(A)*alpha
    else:
        b0 = A*((A+1)+(A-1)*cos_w0+2*np.sqrt(A)*alpha)
        b1 = -2*A*((A-1)+(A+1)*cos_w0)
        b2 = A*((A+1)-(A-1)*cos_w0-2*np.sqrt(A)*alpha)
        a0 = (A+1)-(A-1)*cos_w0+2*np.sqrt(A)*alpha
        a1 = 2*((A-1)+(A+1)*cos_w0)
        a2 = (A+1)-(A-1)*cos_w0-2*np.sqrt(A)*alpha
    return np.array([[b0/a0, b1/a0, b2/a0, 1.0, a1/a0, a2/a0]])

def _apply_high_shelf(audio, sr, freq, gain_db, Q=0.7):
    sos = _design_shelf_sos(gain_db, freq, sr, 'high')
    if sos is not None:
        return sosfilt(sos, audio, axis=0)
    return audio

def _apply_spectral_tilt(audio, sr, tilt_db):
    if abs(tilt_db) < 0.01: return audio
    gain = tilt_db / 2.0
    
    if PEDALBOARD_AVAILABLE:
        board = Pedalboard([
            HighShelfFilter(cutoff_frequency_hz=1000, gain_db=gain),
            LowShelfFilter(cutoff_frequency_hz=1000, gain_db=-gain)
        ])
        return board(audio, sr)
        
    sos_lp = signal.butter(2, 1000, btype='lowpass', fs=sr, output='sos')
    sos_hp = signal.butter(2, 1000, btype='highpass', fs=sr, output='sos')
    
    lows = signal.sosfiltfilt(sos_lp, audio, axis=0)
    highs = signal.sosfiltfilt(sos_hp, audio, axis=0)
    
    low_gain = 10**(-gain/20.0)
    high_gain = 10**(gain/20.0)
    
    return (lows * low_gain) + (highs * high_gain)

def _apply_filters(audio, sr, highpass_freq, lowpass_freq):
    if highpass_freq > 0:
        sos = signal.butter(4, highpass_freq, btype='highpass', fs=sr, output='sos')
        audio = signal.sosfiltfilt(sos, audio, axis=0)
    if lowpass_freq > 0 and lowpass_freq < sr/2:
        sos = signal.butter(4, lowpass_freq, btype='lowpass', fs=sr, output='sos')
        audio = signal.sosfiltfilt(sos, audio, axis=0)
    return audio

def _apply_harmonic_exciter(audio, sr, drive):
    # Exit early if drive is negligible
    if drive < 0.051:
        return audio
    
    # 1. Isolate the high-frequency band
    highpass_filter = signal.butter(4, CONST_HARMONIC_EXCITER_FREQ, 'highpass', fs=sr, output='sos')
    hf_content = signal.sosfiltfilt(highpass_filter, audio, axis=0)
    
    # 2. Generate harmonics using a wave folder/saturator approximation
    drive_factor = drive * 5.0
    generated_harmonics = np.tanh(hf_content * drive_factor)
    
    # 3. Tame harshness (soothe)
    lowpass_tamer = signal.butter(1, CONST_HARMONIC_SOOTHE_FREQ, 'lowpass', fs=sr, output='sos')
    tamed_harmonics = signal.sosfiltfilt(lowpass_tamer, generated_harmonics, axis=0)
    
    # 4. Clean up low-end bleed from the saturation process
    cleanup_filter = signal.butter(4, CONST_HARMONIC_EXCITER_FREQ, 'highpass', fs=sr, output='sos')
    pure_harmonics = signal.sosfiltfilt(cleanup_filter, tamed_harmonics, axis=0)
    
    # 5. Blend back into original signal
    blend_amount = 0.25
    return audio + (pure_harmonics * blend_amount)

def _design_linkwitz_riley_crossover(cutoff_freq, sample_rate, order=8):
    nyquist = 0.5 * sample_rate
    normal_cutoff = np.clip(cutoff_freq / nyquist, 0.01, 0.99)
    sos_lp = signal.butter(order // 2, normal_cutoff, btype='lowpass', output='sos')
    sos_hp = signal.butter(order // 2, normal_cutoff, btype='highpass', output='sos')
    return [sos_lp, sos_lp], [sos_hp, sos_hp]

def _apply_multiband_compression(audio, sr, x_l, x_h, order, l_p, m_p, h_p):
    if not PEDALBOARD_AVAILABLE: return audio
    
    is_stereo = audio.ndim > 1
    audio_ch = audio.T if is_stereo else audio[np.newaxis, :]
    out = np.zeros_like(audio_ch)
    
    sos_l_lp, sos_l_hp = _design_linkwitz_riley_crossover(x_l, sr, order)
    sos_h_lp, sos_h_hp = _design_linkwitz_riley_crossover(x_h, sr, order)

    for c in range(audio_ch.shape[0]):
        ch = audio_ch[c]
        low = signal.sosfiltfilt(sos_l_lp[0], ch)
        low = signal.sosfiltfilt(sos_l_lp[1], low)
        high = signal.sosfiltfilt(sos_h_hp[0], ch)
        high = signal.sosfiltfilt(sos_h_hp[1], high)
        mid = ch - low - high 
        
        low = Compressor(threshold_db=l_p['threshold_db'], ratio=l_p['ratio'])(low, sample_rate=sr)
        mid = Compressor(threshold_db=m_p['threshold_db'], ratio=m_p['ratio'])(mid, sample_rate=sr)
        high = Compressor(threshold_db=h_p['threshold_db'], ratio=h_p['ratio'])(high, sample_rate=sr)
        out[c] = low + mid + high
        
    return out.T if is_stereo else out[0]

def _apply_stereo_width(audio, width):
    if width == 1.0 or audio.ndim < 2: return audio
    mid = (audio[:, 0] + audio[:, 1]) / 2.0
    side = (audio[:, 0] - audio[:, 1]) / 2.0 * width
    processed = np.column_stack([mid + side, mid - side])
    return processed

def _apply_eq(audio, sr, adjustments, adaptive=False):
    processed = audio.copy()
    if "bass_cut_db" in adjustments:
        cut = adjustments["bass_cut_db"] * (adjustments.get("bass_scale", 1.0) if adaptive else 1.0)
        sos = _design_shelf_sos(cut, CONST_EQ_BASS_FREQ, sr, 'low')
        if sos is not None: processed = sosfilt(sos, processed, axis=0)
    if "high_cut_db" in adjustments:
        cut = adjustments["high_cut_db"] * (adjustments.get("high_scale", 1.0) if adaptive else 1.0)
        sos = _design_shelf_sos(cut, CONST_EQ_HIGH_FREQ, sr, 'high')
        if sos is not None: processed = sosfilt(sos, processed, axis=0)
    return processed

# =================================================================================
# == Analysis Helpers
# =================================================================================

def _normalize(audio, sr, meter_obj, target):
    audio = _sanitize(audio)
    loudness = meter_obj.integrated_loudness(audio)
    if loudness == -float('inf'): return audio, loudness
    return pln.normalize.loudness(audio, loudness, target), loudness

def _analyze(audio, sr):
    if not LIBROSA_AVAILABLE: return {"bass": 0, "high": 0, "spike": 0}
    audio = _sanitize(audio)
    analysis_channel = audio[:, 0] if audio.ndim > 1 else audio
    if analysis_channel.size == 0: return {"bass": np.nan, "high": np.nan, "spike": 0.0}
    stft = np.abs(librosa.stft(analysis_channel))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=stft.shape[0]*2-2)
    bass = np.mean(stft[freqs < 100])
    high = np.mean(stft[freqs > 10000])
    mid_slice = stft[(freqs >= 1000) & (freqs <= 3000)]
    spike = np.max(mid_slice) / (np.mean(mid_slice) + 1e-6)
    return {"bass": bass, "high": high, "spike": spike}

def _calculate_hf_energy(audio, sr, freq_threshold=8000):
    if not LIBROSA_AVAILABLE:
        return 0.0
    try:
        channel_data = audio[:, 0] if len(audio.shape) > 1 else audio
        spectra = np.abs(librosa.stft(channel_data))
        bin_frequencies = librosa.fft_frequencies(sr=sr)
        high_freq_bins = spectra[bin_frequencies >= freq_threshold]
        return float(np.mean(high_freq_bins))
    except Exception:
        return 0.0

# =================================================================================
# == Main Pipeline Execution
# =================================================================================

def execute_pipeline(audio_data, sr, params, log_callback=None):
    """Main AutoMaster DSP Pipeline."""
    processed = _sanitize(audio_data.copy())
    original_data = audio_data.copy()
    history = {'lufs': {}, 'peak': {}}
    
    meter = None
    if PYLOUDNORM_AVAILABLE:
        meter = pln.Meter(sr)
        history['lufs']['Input'] = meter.integrated_loudness(audio_data)
        history['peak']['Input'] = np.max(np.abs(audio_data))
    else:
        history['lufs']['Input'] = 0.0
        history['peak']['Input'] = 0.0

    def run_step(step_name, func, *args):
        nonlocal processed
        safe_state = np.copy(processed)
        
        try:
            # Execute the DSP function
            new_state = func(processed, *args)
            # Clean the output
            clean_state = _sanitize(new_state)
            
            # Verify we didn't mute the track
            max_amplitude = np.max(np.abs(clean_state))
            if max_amplitude < CONST_SILENCE_THRESH:
                if log_callback:
                    log_callback(f"⚠️ {step_name} resulted in silence. Reverting to previous state.")
                processed = safe_state
            else:
                processed = clean_state
                
        except BaseException as error_msg:
            if log_callback:
                log_callback(f"⚠️ DSP Error in {step_name}: {str(error_msg)}. Reverting.")
            processed = safe_state

    # 1. Input Gain
    if params.get("input_gain_db", 0) != 0: 
        run_step("Input Gain", lambda x: x * 10**(params["input_gain_db"]/20))

    # 2. Spectral Tilt
    if params.get('tilt', 0) != 0: 
        run_step("Spectral Tilt", _apply_spectral_tilt, sr, params['tilt'])
        if log_callback: log_callback(f"  ⚖️ Applied Spectral Tilt: {params['tilt']:+.2f} dB")

    # 3. Harmonic Exciter
    if params.get('exciter', 0) > 0:
        run_step("Harmonic Exciter", _apply_harmonic_exciter, sr, params['exciter'])
        if log_callback: log_callback(f"  ✨ Harmonic Exciter: Drive {params['exciter']:.2f}")

    # 4. Repair
    if params.get('tamer', 0) > 0:
        cut = -6.0 * params['tamer']
        run_step("Vocal Tamer", _apply_bell_filter, sr, 1000, cut, 0.7)
        
    if params.get('mud', 0) != 0:
        sos = _design_shelf_sos(params['mud'], 75, sr, 'low')
        if sos is not None:
            run_step("Mud Fix", lambda x: sosfilt(sos, x, axis=0))
            
    if params.get('thump', 0) != 0:
        run_step("Thump Fix", _apply_bell_filter, sr, 90, params['thump'], 2.0)
    
    if meter:
        processed = _sanitize(processed)
        history['lufs']['Repair'] = meter.integrated_loudness(processed)
        history['peak']['Repair'] = np.max(np.abs(processed))

    # 5. Filters
    if params.get('hp', 0) > 0 or params.get('lp', 0) > 0: 
        run_step("Filters", _apply_filters, sr, params.get('hp', 0), params.get('lp', 0))
    
    # Normalization (Guarded)
    if meter and params.get('target_lufs'):
        try:
            processed = _sanitize(processed)
            if not _is_dead(processed):
                processed, _ = _normalize(processed, sr, meter, params['target_lufs'])
        except Exception: pass 

    # 6. Adaptive EQ
    if params.get('do_eq', True):
        for i in range(params.get("max_iterations_eq", 5)):
            processed = _sanitize(processed)
            an = _analyze(processed, sr)
            
            bass_ok = an["bass"] <= (params['eq_bass'] + CONST_BASS_TOLERANCE)
            high_ok = an["high"] <= (params['eq_high'] + CONST_HIGH_TOLERANCE)
            
            if bass_ok and high_ok:
                if i == 0 and log_callback: log_callback(f"  ✅ EQ: Spectrum Balanced (Bass:{an['bass']:.1f}, High:{an['high']:.1f})")
                break
            
            adj = {}
            if not bass_ok and an["bass"] > params['eq_bass']: 
                if params.get('mud', 0) > -1.0: 
                    adj["bass_cut_db"] = -2.0
                    if log_callback: log_callback(f"  🎛️ EQ Iter {i+1}: Cutting Bass (Current: {an['bass']:.1f})")
            
            if not high_ok and an["high"] > params['eq_high']: 
                adj["high_cut_db"] = -1.5
                if log_callback: log_callback(f"  🎛️ EQ Iter {i+1}: Cutting Highs (Current: {an['high']:.1f})")
            
            run_step(f"EQ Iter {i}", _apply_eq, sr, adj, params.get('eq_adaptive', True))
        
        if not params.get("fast_mode") and meter:
            try: processed, _ = _normalize(processed, sr, meter, params['target_lufs'])
            except Exception: pass
            history['lufs']['EQ'] = meter.integrated_loudness(processed)
            history['peak']['EQ'] = np.max(np.abs(processed))

    # 7. Dynamics
    if params.get('do_deess', False): 
        run_step("DeEsser", _apply_bell_filter, sr, 7500, params['deess_amount'], 2.5)
        
    if params.get('width', 1.0) != 1.0:
        run_step("Stereo Width", _apply_stereo_width, params['width'])

    # 8. MBC
    if params.get('do_mbc', True):
        mid_ratio = params['mbc_mid_ratio']
        processed = _sanitize(processed)
        if not _is_dead(processed):
            mono_check = processed[:,0] if processed.ndim > 1 else processed
            rms_check = np.sqrt(np.mean(mono_check**2))
            if rms_check > 1e-9:
                crest = 20 * np.log10(np.max(np.abs(mono_check)) / rms_check)
                if crest < 8.0: mid_ratio = 1.2
        
        l_p = {'threshold_db': params['mbc_low_thresh'], 'ratio': params['mbc_low_ratio']}
        m_p = {'threshold_db': params['mbc_mid_thresh'], 'ratio': mid_ratio}
        h_p = {'threshold_db': params['mbc_high_thresh'], 'ratio': params['mbc_high_ratio']}
        
        run_step("MBC", _apply_multiband_compression, sr, 
                 params['x_low'], params['x_high'], params['x_order'], 
                 l_p, m_p, h_p)
        
        if meter:
            try: processed, _ = _normalize(processed, sr, meter, params['target_lufs'])
            except Exception: pass
            history['lufs']['MBC'] = meter.integrated_loudness(processed)
            history['peak']['MBC'] = np.max(np.abs(processed))

    # 9. Limiter
    if params.get('do_limiter', True):
        clip_drive = params.get('soft_clip_drive', 1.0)
        run_step("Soft Clip", lambda x: np.tanh(x * clip_drive) / clip_drive)
        
        if PEDALBOARD_AVAILABLE:
            board = Pedalboard([
                Limiter(threshold_db=params.get('lim_db', -1.0), release_ms=CONST_LIMITER_RELEASE_MS)
            ])
            run_step("Limiter", board, sr)

    # 10. Safety Shelf
    if params.get('exciter', 0) > 0 and LIBROSA_AVAILABLE:
        input_hf = _calculate_hf_energy(original_data, sr, 8000)
        output_hf = _calculate_hf_energy(processed, sr, 8000)
        in_db = 20 * np.log10(input_hf + 1e-9)
        out_db = 20 * np.log10(output_hf + 1e-9)
        
        if (out_db - in_db) > CONST_SAFETY_SHELF_THRESHOLD_DB:
            run_step("Safety Shelf", _apply_high_shelf, sr, CONST_SAFETY_SHELF_FREQ, CONST_SAFETY_SHELF_CUT)
            if log_callback: log_callback(f"  🛡️ Safety Shelf Triggered")

    # Final Stats
    processed = _sanitize(processed)
    if meter:
        final_lufs = meter.integrated_loudness(processed)
        history['lufs']['Final'] = final_lufs
        history['peak']['Final'] = np.max(np.abs(processed))
    
    return processed, history

# =================================================================================
# == API Serialization Helpers
# =================================================================================

def serialize_for_api(audio_np, sr, params):
    payload = {"audio": audio_np, "sr": sr, "params": params}
    return base64.b64encode(zlib.compress(pickle.dumps(payload))).decode('utf-8')

def deserialize_from_api(b64_data):
    return pickle.loads(zlib.decompress(base64.b64decode(b64_data)))


# ==============================================================================
# == Unit Tests (smoke — runs without ComfyUI)
# ==============================================================================

if __name__ == "__main__":
    print("\n🧪 Smoke tests: automaster_core")
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

    _check("VERSION defined",    VERSION == "v6.32.1")
    _check("CONST CONST_TARGET_SR defined", CONST_TARGET_SR is not None)
    _check("CONST CONST_LIMITER_RELEASE_MS defined", CONST_LIMITER_RELEASE_MS is not None)
    _check("CONST CONST_SOFT_CLIP_THRESHOLD defined", CONST_SOFT_CLIP_THRESHOLD is not None)
    _check("CONST CONST_BASS_TOLERANCE defined", CONST_BASS_TOLERANCE is not None)
    _check("CONST CONST_HIGH_TOLERANCE defined", CONST_HIGH_TOLERANCE is not None)
    _check("fn execute_pipeline is callable", callable(execute_pipeline))
    _check("fn serialize_for_api is callable", callable(serialize_for_api))
    _check("fn deserialize_from_api is callable", callable(deserialize_from_api))

    print(f"\n  {_pass} passed, {_fail} failed")
    if _fail == 0:
        print("  🎉 All good.")
