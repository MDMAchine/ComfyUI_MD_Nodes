# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# █▓▒░                                                                     ░▒▓█
# █▓▒░              fsampler_core.py - Core Algorithm v1.6.1               ░▒▓█
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
# ║   • Linear and Quadratic Extrapolation Math
# ║   • Sigma-based Progress Tracking
# ║   • Skip Interval Strategy Logic (Conservative/Aggressive)
# ║   • Model Patcher Hook Implementation
# ╚════════════════════════════════════════════════════════════════════════════


VERSION = "v1.6.1"  # UPS v1.5.8

import torch
import logging
from collections import deque

CONST_EPSILON = 1e-6
CONST_SIGMA_JUMP_THRESHOLD = 10.0
CONST_MIN_HISTORY_FOR_QUAD = 3
CONST_MIN_HISTORY_FOR_LINEAR = 2

logger = logging.getLogger("ComfyUI_MD_Nodes.FSamplerCore")

class FSamplerHook:
    """
    The core logic that intercepts UNet calls.
    Maintains a history buffer and performs the math extrapolation.
    """
    def __init__(self, mode, history_depth, skip_interval, start_percent, end_percent, skip_strategy):
        self.mode = mode
        self.history_depth = history_depth
        self.skip_interval = skip_interval
        self.start_percent = start_percent
        self.end_percent = end_percent
        self.skip_strategy = skip_strategy
        
        # State tracking
        self.history = deque(maxlen=history_depth)
        self.step_counter = 0
        self.last_sigma = None
        self.initial_sigma = None 
        
        # Stats
        self.skipped_steps = 0
        self.executed_steps = 0

    def reset_state(self):
        """Clears history for a new generation run."""
        self.history.clear()
        self.step_counter = 0
        self.last_sigma = None
        self.initial_sigma = None
        self.skipped_steps = 0
        self.executed_steps = 0
        
    def extrapolate(self, target_sigma):
        """
        Performs the mathematical prediction of the next tensor.
        """
        # Linear Extrapolation (Requires 2 points)
        if self.mode == "linear" or len(self.history) < 3:
            s1, y1 = self.history[-1]
            s0, y0 = self.history[-2]
            
            denom = s1 - s0
            if abs(denom) < CONST_EPSILON:
                return y1
                
            slope = (y1 - y0) / denom
            y_pred = y1 + slope * (target_sigma - s1)
            return y_pred

        # Quadratic Extrapolation (Requires 3 points)
        elif self.mode == "quadratic":
            s2, y2 = self.history[-1]
            s1, y1 = self.history[-2]
            s0, y0 = self.history[-3]
            
            x = target_sigma
            
            denom0 = (s0 - s1) * (s0 - s2)
            denom1 = (s1 - s0) * (s1 - s2)
            denom2 = (s2 - s0) * (s2 - s1)
            
            if abs(denom0) < CONST_EPSILON or abs(denom1) < CONST_EPSILON or abs(denom2) < CONST_EPSILON:
                return y2 
                
            L0 = ((x - s1) * (x - s2)) / denom0
            L1 = ((x - s0) * (x - s2)) / denom1
            L2 = ((x - s0) * (x - s1)) / denom2
            
            y_pred = (y0 * L0) + (y1 * L1) + (y2 * L2)
            return y_pred
            
        return self.history[-1][1]

    def __call__(self, apply_model_func, args):
        """
        The wrapper function executed by ComfyUI's ModelPatcher.
        """
        input_x = args['input']
        timestep = args['timestep']
        c = args['c']
        
        # 1. Calculate Sigma
        current_sigma = timestep.max().item() if isinstance(timestep, torch.Tensor) else timestep

        # 2. Jump Detection (Reset if we started a new generation)
        if self.last_sigma is not None:
            if abs(current_sigma - self.last_sigma) > CONST_SIGMA_JUMP_THRESHOLD:
                self.reset_state()
        
        self.last_sigma = current_sigma
        
        if self.initial_sigma is None:
            self.initial_sigma = current_sigma

        # 3. Progress Estimation (Sigma-based)
        if self.initial_sigma is not None and self.initial_sigma > CONST_EPSILON:
            progress = 1.0 - (current_sigma / self.initial_sigma)
            progress = max(0.0, min(1.0, progress))
        else:
            progress = 0.0

        # 4. Determine Action
        can_extrapolate = False
        
        min_history = CONST_MIN_HISTORY_FOR_QUAD if self.mode == "quadratic" else CONST_MIN_HISTORY_FOR_LINEAR
        has_history = len(self.history) >= min_history
        
        step_mod = self.step_counter % self.skip_interval
        
        is_skip_slot = False
        if self.skip_strategy == "conservative":
            is_skip_slot = (step_mod == (self.skip_interval - 1))
        elif self.skip_strategy == "aggressive":
            is_skip_slot = (step_mod != 0)

        in_start_zone = progress < self.start_percent
        in_end_zone = progress > self.end_percent
        
        should_run_anyway = in_start_zone or in_end_zone

        if has_history and is_skip_slot and not should_run_anyway:
            can_extrapolate = True

        # 5. Execute
        if can_extrapolate:
            if self.step_counter % 10 == 0:
                 logger.debug(f"[FSampler] Extrapolating step {self.step_counter} (Progress: {progress:.2f})")
            
            out = self.extrapolate(current_sigma)
            self.skipped_steps += 1
            self.step_counter += 1
            return out
        else:
            out = apply_model_func(input_x, timestep, **c)
            self.executed_steps += 1
            
            self.history.append((current_sigma, out))
            self.step_counter += 1
            return out


# ==============================================================================
# == Unit Tests (smoke — runs without ComfyUI)
# ==============================================================================

if __name__ == "__main__":
    print("\n🧪 Smoke tests: fsampler_core")
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
    _check("CONST CONST_SIGMA_JUMP_THRESHOLD defined", CONST_SIGMA_JUMP_THRESHOLD is not None)
    _check("CONST CONST_MIN_HISTORY_FOR_QUAD defined", CONST_MIN_HISTORY_FOR_QUAD is not None)
    _check("CONST CONST_MIN_HISTORY_FOR_LINEAR defined", CONST_MIN_HISTORY_FOR_LINEAR is not None)

    print(f"\n  {_pass} passed, {_fail} failed")
    if _fail == 0:
        print("  🎉 All good.")
