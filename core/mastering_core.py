# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# █▓▒░                                                                     ░▒▓█
# █▓▒░            MD_Nodes Core: Mastering Suite (IP-Protected)            ░▒▓█
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
# ║ ░▒▓ DESCRIPTION:
# ║   Core DSP math for mastering (EQ, Multiband Comp, Limiter).
# ║   Pure stateless tensor/numpy operations.
# ╚════════════════════════════════════════════════════════════════════════════


VERSION = "v3.0.0"  # UPS v1.5.8

import numpy as np
import pickle
import zlib
import base64

# =================================================================================
# == Dependency Management
# =================================================================================

try:
    from scipy import signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import pedalboard
    PEDALBOARD_AVAILABLE = True
except ImportError:
    PEDALBOARD_AVAILABLE = False

# =================================================================================
# == Global Constants (Bit-Exact Parity)
# =================================================================================

CONST_MIN_FREQ = 0.0001
CONST_EPSILON = 1e-6

# =================================================================================
# == DSP Helper Functions (Stateless & Pure)
# =================================================================================

def _db_to_amplitude(db):
    """Converts dB to linear amplitude."""
    return 10**(db / 20.0)

def _soft_clip(audio_data):
    """Analog-style tanh soft clipping."""
    return np.clip(np.tanh(audio_data), -1.0, 1.0)

def _apply_gain(audio_data, gain_db):
    if gain_db == 0.0: return audio_data
    return audio_data * _db_to_amplitude(gain_db)

# --- Filter Design ---

def _design_lowpass_filter(cutoff_freq, sample_rate, order=4):
    if not SCIPY_AVAILABLE: return None
    nyquist = 0.5 * sample_rate
    normal_cutoff = np.clip(cutoff_freq / nyquist, CONST_MIN_FREQ, 0.99)
    b, a = signal.butter(order, normal_cutoff, btype='lowpass', analog=False)
    return signal.tf2sos(b, a)

def _design_highpass_filter(cutoff_freq, sample_rate, order=4):
    if not SCIPY_AVAILABLE: return None
    nyquist = 0.5 * sample_rate
    normal_cutoff = np.clip(cutoff_freq / nyquist, CONST_MIN_FREQ, 0.99)
    b, a = signal.butter(order, normal_cutoff, btype='highpass', analog=False)
    return signal.tf2sos(b, a)

def _design_peaking_filter(gain_db, freq, Q, sample_rate):
    if abs(gain_db) < 0.001 or Q <= 0 or freq <= 0 or freq >= sample_rate / 2: return None
    if not SCIPY_AVAILABLE: return None
    
    A = 10**(gain_db / 40.0)
    omega = 2 * np.pi * freq / sample_rate
    sn = np.sin(omega); cs = np.cos(omega)
    alpha = sn / (2 * Q)
    a0 = 1 + alpha / A
    b0 = 1 + alpha * A
    b1 = -2 * cs
    b2 = 1 - alpha * A
    a1 = -2 * cs
    a2 = 1 - alpha / A
    
    b = np.array([b0/a0, b1/a0, b2/a0], dtype=np.float64)
    a = np.array([1.0, a1/a0, a2/a0], dtype=np.float64)
    return signal.tf2sos(b, a)

def _design_low_shelf_filter(gain_db, freq, sample_rate):
    if abs(gain_db) < 0.001 or freq <= 0 or freq >= sample_rate / 2: return None
    if not SCIPY_AVAILABLE: return None
    
    A = 10**(gain_db / 40.0)
    w0 = 2 * np.pi * freq / sample_rate
    cos_w0 = np.cos(w0); sin_w0 = np.sin(w0)
    alpha = sin_w0 / (2 * 0.707)
    a0 = (A+1) + (A-1)*cos_w0 + 2*np.sqrt(A)*alpha
    
    if abs(a0) < 1e-9: return None
    
    b0 = A*((A+1) - (A-1)*cos_w0 + 2*np.sqrt(A)*alpha)
    b1 = 2*A*((A-1) - (A+1)*cos_w0)
    b2 = A*((A+1) - (A-1)*cos_w0 - 2*np.sqrt(A)*alpha)
    a1 = -2*((A-1) + (A+1)*cos_w0)
    a2 = (A+1) + (A-1)*cos_w0 - 2*np.sqrt(A)*alpha
    
    b = np.array([b0/a0, b1/a0, b2/a0], dtype=np.float64)
    a = np.array([1.0, a1/a0, a2/a0], dtype=np.float64)
    return signal.tf2sos(b, a)

def _design_high_shelf_filter(gain_db, freq, sample_rate):
    if abs(gain_db) < 0.001 or freq <= 0 or freq >= sample_rate / 2: return None
    if not SCIPY_AVAILABLE: return None
    
    A = 10**(gain_db / 40.0)
    w0 = 2 * np.pi * freq / sample_rate
    cos_w0 = np.cos(w0); sin_w0 = np.sin(w0)
    alpha = sin_w0 / (2 * 0.707)
    a0 = (A+1) - (A-1)*cos_w0 + 2*np.sqrt(A)*alpha
    
    if abs(a0) < 1e-9: return None
    
    b0 = A*((A+1) + (A-1)*cos_w0 + 2*np.sqrt(A)*alpha)
    b1 = -2*A*((A-1) + (A+1)*cos_w0)
    b2 = A*((A+1) + (A-1)*cos_w0 - 2*np.sqrt(A)*alpha)
    a1 = 2*((A-1) - (A+1)*cos_w0)
    a2 = (A+1) - (A-1)*cos_w0 - 2*np.sqrt(A)*alpha
    
    b = np.array([b0/a0, b1/a0, b2/a0], dtype=np.float64)
    a = np.array([1.0, a1/a0, a2/a0], dtype=np.float64)
    return signal.tf2sos(b, a)

def _design_linkwitz_riley_crossover(cutoff_freq, sample_rate, order=8):
    if not SCIPY_AVAILABLE: return None, None
    if order % 2 != 0: order = max(4, order + 1)
    
    nyquist = 0.5 * sample_rate
    normal_cutoff = np.clip(cutoff_freq / nyquist, CONST_MIN_FREQ, 0.99)
    b_lp, a_lp = signal.butter(order // 2, normal_cutoff, btype='lowpass')
    b_hp, a_hp = signal.butter(order // 2, normal_cutoff, btype='highpass')
    sos_lp = signal.tf2sos(b_lp, a_lp)
    sos_hp = signal.tf2sos(b_hp, a_hp)
    
    return np.vstack([sos_lp, sos_lp]), np.vstack([sos_hp, sos_hp])

def _apply_filters_to_audio(audio_data, sos_filters):
    if not sos_filters or all(s is None for s in sos_filters) or not SCIPY_AVAILABLE: 
        return audio_data
        
    filtered = audio_data.copy()
    for sos in sos_filters:
        if sos is not None:
            try: 
                filtered = signal.sosfiltfilt(sos, filtered, axis=-1)
            except ValueError: 
                continue
    # Apply soft clip at the end of EQ stage to prevent internal clipping
    return _soft_clip(np.nan_to_num(filtered, nan=0.0, posinf=1.0, neginf=-1.0))

# =================================================================================
# == Core Processing Logic
# =================================================================================

def process_gain(audio_np, gain_db):
    """Core logic for Gain."""
    processed = _apply_gain(audio_np, gain_db)
    return _soft_clip(processed)

def process_eq(audio_np, sample_rate, params):
    """Core logic for EQ."""
    sos_filters = []
    
    # Lowpass / Highpass
    if params.get('enable_lowpass', False):
        sos_filters.append(_design_lowpass_filter(params.get('lowpass_freq', 18000.0), sample_rate, params.get('lowpass_order', 4)))
    if params.get('enable_highpass', False):
        sos_filters.append(_design_highpass_filter(params.get('highpass_freq', 20.0), sample_rate, params.get('highpass_order', 4)))
    
    # Shelves
    sos_filters.append(_design_high_shelf_filter(params.get('eq_high_shelf_gain_db', 0.0), params.get('eq_high_shelf_freq', 12000.0), sample_rate))
    if params.get('enable_low_shelf_eq', False):
        sos_filters.append(_design_low_shelf_filter(params.get('eq_low_shelf_gain_db', 0.0), params.get('eq_low_shelf_freq', 75.0), sample_rate))
    
    # Parametric Bands
    for i in range(1, 5):
        if params.get(f'enable_param_eq{i}', False):
            sos_filters.append(_design_peaking_filter(
                params.get(f'param_eq{i}_gain_db', 0.0), params.get(f'param_eq{i}_freq', 1000.0),
                params.get(f'param_eq{i}_q', 1.0), sample_rate
            ))
            
    return _apply_filters_to_audio(audio_np, sos_filters)

def process_compression(audio_np, sample_rate, params):
    """Core logic for Compressor."""
    if not PEDALBOARD_AVAILABLE or not params.get('enable_comp', False): 
        return audio_np
        
    # Ensure float32 for Pedalboard
    audio_float = audio_np.astype(np.float32)
    
    # Single-Band
    if params.get('comp_type', "Multiband") == "Single-Band":
        comp = pedalboard.Compressor(
            threshold_db=params.get('comp_threshold_db', -8.0), 
            ratio=params.get('comp_ratio', 2.5),
            attack_ms=params.get('comp_attack_ms', 20.0), 
            release_ms=params.get('comp_release_ms', 250.0)
        )
        needs_transpose = audio_float.ndim == 2 and audio_float.shape[0] <= 2
        res = comp(audio_float.T, sample_rate=sample_rate).T if needs_transpose else comp(audio_float, sample_rate=sample_rate)
        
        if params.get('comp_makeup_gain_db', 0.0) != 0:
            gain = pedalboard.Gain(gain_db=params.get('comp_makeup_gain_db'))
            res = gain(res.T, sample_rate=sample_rate).T if needs_transpose else gain(res, sample_rate=sample_rate)
        return res
    
    # Multiband
    else:
        try:
            sos_lm_lp, sos_lm_hp = _design_linkwitz_riley_crossover(params.get('mb_crossover_low_mid_hz', 250.0), sample_rate, params.get('mb_crossover_order', 8))
            sos_mh_lp, sos_mh_hp = _design_linkwitz_riley_crossover(params.get('mb_crossover_mid_high_hz', 4000.0), sample_rate, params.get('mb_crossover_order', 8))
        except Exception:
            return audio_np 

        if sos_lm_lp is None or sos_mh_lp is None: return audio_np

        bands = {
            'low': signal.sosfiltfilt(sos_lm_lp, audio_float, axis=-1),
            'high': signal.sosfiltfilt(sos_mh_hp, audio_float, axis=-1)
        }
        temp_mid = signal.sosfiltfilt(sos_lm_hp, audio_float, axis=-1)
        bands['mid'] = signal.sosfiltfilt(sos_mh_lp, temp_mid, axis=-1)
        
        acc = np.zeros_like(audio_float, dtype=np.float32)
        
        for band, p_prefix in [('low', 'mb_low'), ('mid', 'mb_mid'), ('high', 'mb_high')]:
            comp = pedalboard.Compressor(
                threshold_db=params.get(f'{p_prefix}_threshold_db', -10.0), 
                ratio=params.get(f'{p_prefix}_ratio', 3.0),
                attack_ms=params.get(f'{p_prefix}_attack_ms', 30.0), 
                release_ms=params.get(f'{p_prefix}_release_ms', 300.0)
            )
            d = bands[band]
            needs_transpose = d.ndim == 2 and d.shape[0] <= 2
            res = comp(d.T, sample_rate=sample_rate).T if needs_transpose else comp(d, sample_rate=sample_rate)
            
            if params.get(f'{p_prefix}_makeup_gain_db', 0.0) != 0:
                g = pedalboard.Gain(gain_db=params.get(f'{p_prefix}_makeup_gain_db'))
                res = g(res.T, sample_rate=sample_rate).T if needs_transpose else g(res, sample_rate=sample_rate)
            acc += res
            
        return _soft_clip(np.nan_to_num(acc))

def process_limiting(audio_np, sample_rate, params):
    """Core logic for Limiter."""
    if not PEDALBOARD_AVAILABLE or not params.get('enable_limiter', False): return audio_np
    
    audio_float = audio_np.astype(np.float32)
    try:
        lim = pedalboard.Limiter(
            threshold_db=params.get('limiter_ceiling_db', -0.1), 
            release_ms=params.get('limiter_release_ms', 50.0)
        )
        needs_transpose = audio_float.ndim == 2 and audio_float.shape[0] <= 2
        limited = lim(audio_float.T, sample_rate=sample_rate).T if needs_transpose else lim(audio_float, sample_rate=sample_rate)
        return np.clip(limited, -1.0, 1.0)
    except Exception:
        return _soft_clip(audio_float)

def execute_full_chain(audio_np, sample_rate, master_gain_db, params):
    """Core logic for the Full Mastering Chain (Gain -> EQ -> Comp -> Limit)."""
    processed = process_gain(audio_np, master_gain_db)
    processed = process_eq(processed, sample_rate, params)
    processed = process_compression(processed, sample_rate, params)
    processed = process_limiting(processed, sample_rate, params)
    return processed

# =================================================================================
# == API Serialization Helpers
# =================================================================================

def serialize_for_api(audio_np, sr, operation, params, gain_db=0.0):
    payload = {
        "audio": audio_np,
        "sr": sr,
        "op": operation,
        "params": params,
        "gain": gain_db
    }
    return base64.b64encode(zlib.compress(pickle.dumps(payload))).decode('utf-8')

def deserialize_from_api(b64_data):
    return pickle.loads(zlib.decompress(base64.b64decode(b64_data)))


# ==============================================================================
# == Unit Tests (smoke — runs without ComfyUI)
# ==============================================================================

if __name__ == "__main__":
    print("\n🧪 Smoke tests: mastering_core")
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

    _check("VERSION defined",    VERSION == "v3.0.0")
    _check("CONST CONST_MIN_FREQ defined", CONST_MIN_FREQ is not None)
    _check("fn process_gain is callable", callable(process_gain))
    _check("fn process_eq is callable", callable(process_eq))
    _check("fn process_compression is callable", callable(process_compression))

    print(f"\n  {_pass} passed, {_fail} failed")
    if _fail == 0:
        print("  🎉 All good.")
