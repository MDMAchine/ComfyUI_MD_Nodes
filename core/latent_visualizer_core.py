# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# █▓▒░                                                                     ░▒▓█
# █▓▒░          latent_visualizer_core.py - Core Algorithm v1.6.1          ░▒▓█
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
# ║ CORE RESPONSIBILITIES:
# ║   • Tensor slicing and 1D signal extraction
# ║   • Fast Fourier Transform (FFT) math for spectrum/phase
# ║   • Statistical calculations (Mean, Std, Min, Max)
# ║   • Scipy-based Peak Detection algorithms
# ║   • Stateless processing (pure data math, no UI)
# ╚════════════════════════════════════════════════════════════════════════════


VERSION = "v1.6.1"  # UPS v1.5.8

import torch
import numpy as np
import logging

# Robust Dependency Fallback: Scipy
try:
    from scipy import signal
    CONST_SCIPY_AVAILABLE = True
except ImportError:
    CONST_SCIPY_AVAILABLE = False
    logging.warning("[ACELatentCore] Scipy not found. Peak detection disabled.")

# Core Constants
CONST_MIN_SIGNAL_VARIANCE = 1e-6
CONST_LOG_EPSILON = 1e-8
CONST_FLAT_SIGNAL_OFFSET = 0.5
CONST_PEAK_MIN_DISTANCE = 10

def extract_1d_signal(data_tensor, normalize=True):
    """Extracts 1D signal from tensor slice and applies optional normalization."""
    if not isinstance(data_tensor, torch.Tensor):
         return np.array([])

    signal_np = data_tensor.detach().cpu().numpy().flatten()

    if signal_np.size == 0:
         return np.array([])
    
    # Check if flat
    if np.ptp(signal_np) < CONST_MIN_SIGNAL_VARIANCE:
        if normalize:
            return np.zeros_like(signal_np) if signal_np.mean() < 0.5 else np.ones_like(signal_np)
        return np.full_like(signal_np, CONST_FLAT_SIGNAL_OFFSET)

    if normalize:
        min_val, max_val = np.min(signal_np), np.max(signal_np)
        range_val = max_val - min_val
        if range_val > CONST_LOG_EPSILON:
            signal_np = (signal_np - min_val) / range_val
        else:
             signal_np = np.full_like(signal_np, CONST_FLAT_SIGNAL_OFFSET)

    return signal_np

def compute_spectrum(raw_signal, log_scale=False):
    """Computes FFT spectrum magnitude."""
    if raw_signal.size < 2: 
        return np.array([]), np.array([])
    
    spectrum = np.fft.rfft(raw_signal)
    freqs = np.fft.rfftfreq(len(raw_signal))
    magnitude = np.abs(spectrum)
    
    if log_scale:
        plot_data = 20 * np.log10(np.maximum(magnitude, CONST_LOG_EPSILON))
    else:
        plot_data = magnitude
        
    return freqs, plot_data

def compute_phase(raw_signal):
    """Computes unwrapped phase spectrum."""
    if raw_signal.size < 2: 
        return np.array([]), np.array([])
        
    spectrum = np.fft.rfft(raw_signal)
    phase = np.unwrap(np.angle(spectrum))
    freqs = np.fft.rfftfreq(len(raw_signal))
    return freqs, phase

def compute_statistics(data_chw):
    """Calculates per-channel statistics for the first 16 channels."""
    num_channels = min(data_chw.shape[0], 16)
    means, stds, mins, maxs = [], [], [], []
    
    for i in range(num_channels):
        d = data_chw[i].detach().cpu().numpy().flatten()
        means.append(d.mean().item())
        stds.append(d.std().item())
        mins.append(d.min().item())
        maxs.append(d.max().item())
        
    x_ticks = list(range(num_channels))
    return np.array(means), np.array(stds), np.array(mins), np.array(maxs), x_ticks

def detect_peaks(signal_data, threshold):
    """Detects peaks in a 1D signal using Scipy."""
    if not CONST_SCIPY_AVAILABLE or signal_data.size < 3:
        return np.array([]), np.array([])

    min_val, max_val = np.min(signal_data), np.max(signal_data)
    peak_height_abs = threshold * (max_val - min_val) + min_val
    
    peaks, _ = signal.find_peaks(
        signal_data, 
        height=peak_height_abs,
        distance=CONST_PEAK_MIN_DISTANCE
    )
    
    if len(peaks) > 0:
        return peaks, signal_data[peaks]
    return np.array([]), np.array([])

def compute_difference(s1, s2):
    """Computes the difference between two signals with dynamic padding."""
    max_len = max(len(s1), len(s2))
    if len(s1) < max_len: s1 = np.pad(s1, (0, max_len - len(s1)))
    if len(s2) < max_len: s2 = np.pad(s2, (0, max_len - len(s2)))
    return s1 - s2


# ==============================================================================
# == Unit Tests (smoke — runs without ComfyUI)
# ==============================================================================

if __name__ == "__main__":
    print("\n🧪 Smoke tests: latent_visualizer_core")
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

    _check("VERSION defined",    VERSION == "v1.6.1")
    _check("CONST CONST_FLAT_SIGNAL_OFFSET defined", CONST_FLAT_SIGNAL_OFFSET is not None)
    _check("CONST CONST_PEAK_MIN_DISTANCE defined", CONST_PEAK_MIN_DISTANCE is not None)
    _check("fn extract_1d_signal is callable", callable(extract_1d_signal))
    _check("fn compute_spectrum is callable", callable(compute_spectrum))
    _check("fn compute_phase is callable", callable(compute_phase))

    print(f"\n  {_pass} passed, {_fail} failed")
    if _fail == 0:
        print("  🎉 All good.")
