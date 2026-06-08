# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# █▓▒░                                                                     ░▒▓█
# █▓▒░ MD_Nodes/GPUTemperatureProtectionEnhanced – GPU Temp Protect v2.3.0 ░▒▓█
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
# ║ ░▒▓ ORIGIN & DEV:
# ║   • Cast into the void by: w-e-w (Original Concept), meap158 (ComfyUI Adapt.)
# ║   • Enhanced by: MDMAchine, Gemini, Claude
# ║
# ║ ░▒▓ DESCRIPTION:
# ║   An enhanced GPU temperature protection node for ComfyUI. Monitors GPU
# ║   temperature, VRAM, and utilization via `nvidia-smi`, pausing the queue if a
# ║   threshold is exceeded.
# ║   NOTE: As a hardware monitoring utility, this runs entirely in the public wrapper domain.
# ║
# ║ ░▒▓ FEATURES:
# ║   ✓ Multi-GPU monitoring and protection (`gpu_id`).
# ║   ✓ Temperature, VRAM, and GPU utilization monitoring via nvidia-smi.
# ║   ✓ CSV logging (`gpu_temp_logs/`) with timestamps & stats.
# ║   ✓ Configurable cooling profiles (balanced, aggressive, conservative).
# ║   ✓ Adaptive sleep intervals based on temperature trend (°C/min).
# ║   ✓ Color-coded console output for temperature status.
# ║   ✓ Specific pass-through for LATENT, IMAGE, AUDIO, VIDEO.
# ║
# ║ ░▒▓ CHANGELOG:
# ║   - v2.3.0 (Enterprise Standards - Feb 2026):
# ║       • ADDED: PerformanceProfiler class (v1.5.3 standard).
# ║       • ADDED: debug_mode parameter to profile nvidia-smi calls.
# ║       • REFACTOR: Tooltips strictly updated to 5-part v1.5.4 standard.
# ║   - v2.2.0 (Input/Output Update):
# ║       • ADDED: Specific optional pass-through inputs/outputs.
# ║       • CHANGED: Temperature output changed from INT to formatted STRING.
# ╚════════════════════════════════════════════════════════════════════════════

# =================================================================================
# == Standard Library Imports                                                    ==
# =================================================================================
VERSION = "v2.3.0"  # UPS v1.5.8


import subprocess
import time
import os
from datetime import datetime
from pathlib import Path
import logging
import traceback
import secrets

# =================================================================================
# == Helper Classes (Enterprise Standards)                                       ==
# =================================================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GPUTempProtect")

LOG_DIR_NAME = "gpu_temp_logs"

class PerformanceProfiler:
    """Standard performance profiler for MD_Nodes."""
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.timings = {}
        self.start_times = {}
    
    def start(self, operation_name):
        if not self.enabled: return
        self.start_times[operation_name] = time.perf_counter()
    
    def stop(self, operation_name):
        if not self.enabled: return
        if operation_name in self.start_times:
            elapsed = time.perf_counter() - self.start_times[operation_name]
            if operation_name not in self.timings:
                self.timings[operation_name] = []
            self.timings[operation_name].append(elapsed)
            del self.start_times[operation_name]
    
    def get_total_time(self):
        if not self.enabled or not self.timings: return 0.0
        return sum(sum(times) for times in self.timings.values())
    
    def print_report(self):
        if not self.enabled or not self.timings: return
        logging.info("\n⏱️  PERFORMANCE (Hardware I/O):")
        total = self.get_total_time()
        logging.info(f"    • Total Time: {total:.4f}s")
        for op_name, times in sorted(self.timings.items()):
            avg = sum(times) / len(times)
            if len(times) == 1:
                logging.info(f"    • {op_name}: {avg:.4f}s")
            else:
                logging.info(f"    • {op_name}: {avg:.4f}s avg ({len(times)}x)")

# =================================================================================
# == Core Node Class                                                             ==
# =================================================================================

class GPUTemperatureProtectionEnhanced:
    """
    MD: GPU Temperature Protection (Enhanced)
    Monitors GPU temperature, VRAM, and utilization via nvidia-smi.
    Pauses the ComfyUI queue if temperature exceeds a set threshold.
    """

    last_call_time = 0.0
    session_stats = {
        'max_temp': 0, 'min_temp': 999, 'total_checks': 0,
        'total_sleeps': 0, 'total_sleep_duration': 0.0
    }

    def __init__(self):
        self.temp_history = []
        self.sleep_count = 0
        self.total_sleep_time = 0.0

    # --- Static methods for GPU info (nvidia-smi calls) ---
    @staticmethod
    def get_gpu_temperature(gpu_id=0):
        try:
            cmd = ['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits', f'--id={gpu_id}']
            startupinfo = None
            if os.name == 'nt':
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            result = subprocess.check_output(cmd, stderr=subprocess.STDOUT, startupinfo=startupinfo)
            return int(result.decode().strip())
        except subprocess.CalledProcessError as e: logger.error(f"[GPU Temp Protect] nvidia-smi error (GPU {gpu_id}): {e.output.decode('utf-8').strip()}")
        except FileNotFoundError: logger.error("[GPU Temp Protect] nvidia-smi command not found.")
        except ValueError: logger.error(f"[GPU Temp Protect] Could not parse temperature output for GPU {gpu_id}.")
        except Exception as e: logger.error(f'[GPU Temp Protect] Error getting temperature for GPU {gpu_id}: {e}')
        return None

    @staticmethod
    def get_all_gpu_temperatures():
        temps = {}
        try:
            cmd = ['nvidia-smi', '--query-gpu=index,temperature.gpu', '--format=csv,noheader,nounits']
            startupinfo = None
            if os.name == 'nt':
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            result = subprocess.check_output(cmd, stderr=subprocess.STDOUT, startupinfo=startupinfo)
            for line in result.decode().strip().split('\n'):
                if line.strip():
                    try: 
                        gpu_id, temp = line.split(',')
                        temps[int(gpu_id.strip())] = int(temp.strip())
                    except ValueError: 
                        logger.warning(f"[GPU Temp Protect] Could not parse multi-GPU temp line: '{line}'")
            return temps
        except Exception as e: 
            logger.error(f'[GPU Temp Protect] Could not get all GPU temps: {e}')
            return {}

    @staticmethod
    def get_gpu_memory_usage(gpu_id=0):
        try:
            cmd = ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits', f'--id={gpu_id}']
            startupinfo = None
            if os.name == 'nt':
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            result = subprocess.check_output(cmd, stderr=subprocess.STDOUT, startupinfo=startupinfo)
            used, total = result.decode().strip().split(',')
            used_mb = int(used.strip()); total_mb = int(total.strip())
            percent = (used_mb / total_mb * 100.0) if total_mb > 0 else 0.0
            return used_mb, total_mb, percent
        except Exception as e: 
            logger.error(f'[GPU Temp Protect] Could not get memory usage for GPU {gpu_id}: {e}')
            return None, None, None

    @staticmethod
    def get_gpu_utilization(gpu_id=0):
        try:
            cmd = ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits', f'--id={gpu_id}']
            startupinfo = None
            if os.name == 'nt':
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            result = subprocess.check_output(cmd, stderr=subprocess.STDOUT, startupinfo=startupinfo)
            return int(result.decode().strip())
        except Exception as e: 
            logger.error(f'[GPU Temp Protect] Could not get utilization for GPU {gpu_id}: {e}')
            return None

    # --- Formatting and Logging ---
    @staticmethod
    def format_temp_colored(temp, sleep_temp, wake_temp):
        if temp >= sleep_temp: color = '\033[91m'; status = "🔥 CRITICAL"
        elif temp > wake_temp: color = '\033[93m'; status = "⚠️  WARNING"
        else: color = '\033[92m'; status = "✅ NORMAL"
        reset = '\033[0m'
        return f"{color}{status}: {temp}°C{reset}"

    def log_to_csv(self, temp, status, message=""):
        if not hasattr(self, 'log_enabled') or self.log_enabled != 'True': return
        try:
            log_dir = Path(LOG_DIR_NAME); log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / f"gpu_temp_log_{datetime.now().strftime('%Y%m%d')}.csv"
            file_exists = log_file.exists()
            mem_used, mem_total, _ = self.get_gpu_memory_usage(self.gpu_id)
            utilization = self.get_gpu_utilization(self.gpu_id)
            with open(log_file, 'a', newline='', encoding='utf-8') as f:
                if not file_exists: f.write("timestamp,gpu_id,temperature,status,message,memory_used_mb,memory_total_mb,utilization_percent\n")
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                mem_used_str = str(mem_used) if mem_used is not None else "N/A"
                mem_total_str = str(mem_total) if mem_total is not None else "N/A"
                util_str = str(utilization) if utilization is not None else "N/A"
                safe_message = message.replace(',', ';').replace('\n', ' ').replace('"', '""')
                f.write(f'"{timestamp}",{self.gpu_id},{temp},"{status}","{safe_message}",{mem_used_str},{mem_total_str},{util_str}\n')
        except IOError as e: logger.error(f"[GPU Temp Protect] File I/O error writing log: {e}")
        except Exception as e: logger.error(f"[GPU Temp Protect] Could not write to log: {e}")

    # --- Statistics and Trend Analysis ---
    def update_statistics(self, temp):
        GPUTemperatureProtectionEnhanced.session_stats['total_checks'] += 1
        if temp > GPUTemperatureProtectionEnhanced.session_stats['max_temp']: GPUTemperatureProtectionEnhanced.session_stats['max_temp'] = temp
        if temp < GPUTemperatureProtectionEnhanced.session_stats['min_temp']: GPUTemperatureProtectionEnhanced.session_stats['min_temp'] = temp
        self.temp_history.append((time.time(), temp))
        if len(self.temp_history) > 50: self.temp_history.pop(0)

    def get_temp_trend(self):
        if len(self.temp_history) < 2: return 0.0
        recent = self.temp_history[-5:]
        if len(recent) < 2: recent = self.temp_history[-2:]
        time_diff = recent[-1][0] - recent[0][0]
        temp_diff = recent[-1][1] - recent[0][1]
        if time_diff > 1.0: return round((temp_diff / time_diff) * 60.0, 2)
        else: return 0.0

    def print_status(self, temp, status_msg=""):
        if not hasattr(self, 'print_enabled') or self.print_enabled != 'True': return
        timestamp = datetime.now().strftime('%H:%M:%S')
        temp_str_colored = self.format_temp_colored(temp, self.sleep_temp, self.wake_temp)
        status_parts = [f"[{timestamp}] GPU {self.gpu_id}: {temp_str_colored}"]
        if status_msg: status_parts.append(f"- {status_msg}")
        trend = self.get_temp_trend()
        if abs(trend) > 0.1: status_parts.append(f"({'↑' if trend > 0 else '↓'} {abs(trend):.1f}°C/min)")
        if hasattr(self, 'monitor_memory') and self.monitor_memory == 'True':
            mem_used, mem_total, mem_percent = self.get_gpu_memory_usage(self.gpu_id)
            if mem_used is not None and mem_total is not None: status_parts.append(f"| VRAM: {mem_used}/{mem_total} MB ({mem_percent:.1f}%)")
        utilization = self.get_gpu_utilization(self.gpu_id)
        if utilization is not None: status_parts.append(f"| Util: {utilization}%")
        logging.info(" ".join(status_parts))

    def print_statistics(self):
        if not hasattr(self, 'print_enabled') or self.print_enabled != 'True': return
        stats = GPUTemperatureProtectionEnhanced.session_stats
        if stats['total_checks'] == 0: return
        logging.info("\n" + "="*70 + "\n GPU TEMPERATURE PROTECTION - SESSION STATISTICS\n" + "="*70)
        logging.info(f" Total Temperature Checks: {stats['total_checks']}")
        min_temp_display = stats['min_temp'] if stats['min_temp'] < 999 else "N/A"
        logging.info(f" Temperature Range Seen: {min_temp_display}°C - {stats['max_temp']}°C")
        logging.info(f" Total Cooling Cycles Initiated: {stats['total_sleeps']}")
        logging.info(f" Total Time Spent Cooling: {stats['total_sleep_duration']:.1f} seconds")
        if stats['total_sleeps'] > 0: print(f" Average Cooling Cycle Duration: {stats['total_sleep_duration'] / stats['total_sleeps']:.1f} seconds")
        logging.info("="*70 + "\n")

    # --- Core Protection Logic ---
    def run_protection_logic(self, profiler):
        profiler.start("temperature_read")
        current_temp = self.get_gpu_temperature(self.gpu_id)
        profiler.stop("temperature_read")
        
        if current_temp is None:
            logger.error(f"Failed to read temperature for GPU {self.gpu_id}. Protection inactive.")
            self.log_to_csv(-1, "error", "Failed temp read")
            return 0
            
        self.update_statistics(current_temp)
        
        if hasattr(self, 'show_all_gpus') and self.show_all_gpus == 'True':
            profiler.start("read_all_gpus")
            all_temps = self.get_all_gpu_temperatures()
            profiler.stop("read_all_gpus")
            if all_temps and hasattr(self, 'print_enabled') and self.print_enabled == 'True': 
                logging.info(f"[All GPUs] {', '.join([f'GPU {gid}: {t}°C' for gid, t in sorted(all_temps.items())])}")
                
        self.print_status(current_temp)
        
        profiler.start("csv_logging")
        self.log_to_csv(current_temp, "check", "Regular check")
        profiler.stop("csv_logging")
        
        if hasattr(self, 'enabled') and self.enabled == 'True':
            current_time = time.time()
            if current_time - GPUTemperatureProtectionEnhanced.last_call_time > self.min_interval:
                if current_temp > self.sleep_temp:
                    self.print_status(current_temp, "Threshold exceeded, starting cooling cycle...")
                    self.log_to_csv(current_temp, "cooling_start", f"Temp > {self.sleep_temp}C")
                    sleep_start_time = time.time()
                    self.sleep_count = 0
                    GPUTemperatureProtectionEnhanced.session_stats['total_sleeps'] += 1
                    
                    while True:
                        self.sleep_count += 1
                        cooling_cycle_elapsed = time.time() - sleep_start_time
                        
                        if self.max_sleep_time > 0 and cooling_cycle_elapsed > self.max_sleep_time: 
                            logger.warning(f"Max cool time exceeded ({self.max_sleep_time}s). Resuming...")
                            self.log_to_csv(current_temp, "cooling_abort", f"Max sleep {self.max_sleep_time}s")
                            break
                            
                        base_sleep = float(self.sleep_time)
                        sleep_multiplier = 1.0
                        if hasattr(self, 'cooling_profile'):
                            if self.cooling_profile == 'aggressive': sleep_multiplier = 0.5
                            elif self.cooling_profile == 'conservative': sleep_multiplier = 2.0
                            
                        sleep_duration = base_sleep * sleep_multiplier
                        
                        if hasattr(self, 'adaptive_cooling') and self.adaptive_cooling == 'True':
                            trend = self.get_temp_trend()
                            if trend < -2.0: sleep_duration *= 0.75
                            elif trend > 0.5: sleep_duration *= 1.25
                            sleep_duration = max(1.0, sleep_duration)
                            
                        logger.debug(f"Cooling cycle {self.sleep_count}, sleeping for {sleep_duration:.1f}s")
                        
                        profiler.start("hardware_sleep")
                        time.sleep(sleep_duration)
                        profiler.stop("hardware_sleep")
                        
                        current_temp_check = self.get_gpu_temperature(self.gpu_id)
                        if current_temp_check is None: 
                            logger.error("Read fail during cooling. Aborting.")
                            break
                            
                        current_temp = current_temp_check
                        self.update_statistics(current_temp)
                        self.print_status(current_temp, f"Cooling... (cycle {self.sleep_count})")
                        self.log_to_csv(current_temp, "cooling_check", f"Cycle {self.sleep_count}")
                        
                        if current_temp <= self.wake_temp: break
                        
                    cycle_total_time = time.time() - sleep_start_time
                    self.total_sleep_time += cycle_total_time
                    GPUTemperatureProtectionEnhanced.session_stats['total_sleep_duration'] += cycle_total_time
                    status_msg = f"Cooling complete ({cycle_total_time:.1f}s / {self.sleep_count} cycles). Resuming."
                    self.print_status(current_temp, status_msg)
                    self.log_to_csv(current_temp, "cooling_end", status_msg)
                    
                    if hasattr(self, 'show_stats') and self.show_stats == 'True': self.print_statistics()
                    GPUTemperatureProtectionEnhanced.last_call_time = time.time()
                else: 
                    GPUTemperatureProtectionEnhanced.last_call_time = current_time
        return current_temp if current_temp is not None else 0

    # --- Node Interface ---
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enabled": (["True", "False"], {
                    "default": "True",
                    "tooltip": (
                        "ENABLE PROTECTION\n"
                        "• Purpose: Master switch for hardware protection logic.\n"
                        "\n⭐ Recommended: True"
                    )
                }),
                "gpu_id": ("INT", {
                    "default": 0, "min": 0, "max": 15, "step": 1,
                    "tooltip": (
                        "GPU ID\n"
                        "• Purpose: Target specific GPU via nvidia-smi indexing.\n"
                        "• Range: 0 to 15.\n"
                        "\n⭐ Recommended: 0 for single-GPU systems."
                    )
                }),
                "sleep_temp": ("INT", {
                    "default": 79, "min": 40, "max": 110, "step": 1,
                    "tooltip": (
                        "SLEEP THRESHOLD\n"
                        "• Purpose: Temperature (°C) that triggers a queue pause.\n"
                        "• Trade-offs: Lower values protect hardware but interrupt renders more often.\n"
                        "\n⭐ Recommended: 79"
                    )
                }),
                "wake_temp": ("INT", {
                    "default": 65, "min": 30, "max": 100, "step": 1,
                    "tooltip": (
                        "WAKE THRESHOLD\n"
                        "• Purpose: Temperature (°C) at which the queue resumes.\n"
                        "• Note: Must be lower than sleep_temp.\n"
                        "\n⭐ Recommended: 65"
                    )
                }),
                "min_interval": ("INT", {
                    "default": 5, "min": 1, "max": 300, "step": 1,
                    "tooltip": (
                        "CHECK INTERVAL\n"
                        "• Purpose: Minimum seconds between temperature checks.\n"
                        "• Trade-offs: Frequent checks increase CPU overhead slightly.\n"
                        "\n⭐ Recommended: 5"
                    )
                }),
                "sleep_time": ("FLOAT", {
                    "default": 5.0, "min": 0.5, "max": 60.0, "step": 0.5,
                    "tooltip": (
                        "COOLING DURATION\n"
                        "• Purpose: Base duration for each cooling sleep cycle (s).\n"
                        "\n⭐ Recommended: 5.0"
                    )
                }),
                "max_sleep_time": ("INT", {
                    "default": 180, "min": 0, "max": 3600, "step": 10,
                    "tooltip": (
                        "MAX COOLING TIME\n"
                        "• Purpose: Force resume after this duration (s), even if still hot.\n"
                        "• Note: Prevents permanent workflow lockups.\n"
                        "\n⭐ Recommended: 180"
                    )
                }),
            },
             "optional": {
                "latent_in": ("LATENT", {
                    "forceInput": True,
                    "tooltip": "PASS-THROUGH (LATENT)\n• Connect LATENT here to enforce execution order."
                }),
                "image_in": ("IMAGE", {
                    "forceInput": True,
                    "tooltip": "PASS-THROUGH (IMAGE)\n• Connect IMAGE here to enforce execution order."
                }),
                "audio_in": ("AUDIO", {
                    "forceInput": True,
                    "tooltip": "PASS-THROUGH (AUDIO)\n• Connect AUDIO here to enforce execution order."
                }),
                "pass_through_generic": ("*", {
                    "forceInput": True,
                    "tooltip": "PASS-THROUGH (GENERIC)\n• Connect any other data type here to enforce order."
                }),
                "print_enabled": (["True", "False"], {
                    "default": "True",
                    "tooltip": "PRINT TO CONSOLE\n• Purpose: Show status messages in ComfyUI console."
                }),
                "show_all_gpus": (["True", "False"], {
                    "default": "False",
                    "tooltip": "SHOW ALL GPUS\n• Purpose: List temps for all detected cards in console."
                }),
                "monitor_memory": (["True", "False"], {
                    "default": "True",
                    "tooltip": "MONITOR VRAM\n• Purpose: Show VRAM usage stats in console."
                }),
                "show_stats": (["True", "False"], {
                    "default": "False",
                    "tooltip": "SHOW STATS\n• Purpose: Print session summary after cooling cycles."
                }),
                "cooling_profile": (["balanced", "aggressive", "conservative"], {
                    "default": "balanced",
                    "tooltip": (
                        "COOLING PROFILE\n"
                        "• Purpose: Adjusts sleep cycle aggressiveness.\n"
                        "• Balanced: Standard behavior.\n"
                        "• Aggressive: Faster checks (0.5x sleep).\n"
                        "• Conservative: Slower checks (2.0x sleep)."
                    )
                }),
                "adaptive_cooling": (["True", "False"], {
                    "default": "True",
                    "tooltip": (
                        "ADAPTIVE COOLING\n"
                        "• Purpose: Adjust sleep time based on temp trend.\n"
                        "• Note: Cooling fast? Sleep less. Cooling slow? Sleep more."
                    )
                }),
                "log_enabled": (["True", "False"], {
                    "default": "False",
                    "tooltip": "CSV LOGGING\n• Purpose: Save temp history to /gpu_temp_logs directory."
                }),
                "debug_mode": (["0 - Silent", "1 - Info"], {
                    "default": "0 - Silent",
                    "tooltip": "LOGGING VERBOSITY\n• Controls hardware I/O profiling display."
                }),
            }
        }

    RETURN_TYPES = ("LATENT", "IMAGE", "AUDIO", "*", "STRING",) 
    RETURN_NAMES = ("latent_out", "image_out", "audio_out", "passthrough_generic_out", "temp_status_string",)
    FUNCTION = "execute"
    OUTPUT_NODE = True
    CATEGORY = "MD_Nodes/Utility"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return secrets.token_hex(16)

    def execute(self, enabled="True", gpu_id=0, sleep_temp=79, wake_temp=65,
                min_interval=5, sleep_time=5.0, max_sleep_time=180,
                latent_in=None, image_in=None, audio_in=None, pass_through_generic=None,
                debug_mode="0 - Silent", **kwargs):
        
        debug_level = int(debug_mode.split(" ")[0])
        profiler = PerformanceProfiler(enabled=(debug_level >= 1))
        profiler.start("total_execution")
        
        current_temp_int = 0 
        temp_status_string = "N/A" 

        try:
            self.enabled = enabled
            self.gpu_id = int(gpu_id)
            self.sleep_temp = int(sleep_temp)
            self.wake_temp = int(wake_temp)
            self.min_interval = int(min_interval)
            self.sleep_time = float(sleep_time)
            self.max_sleep_time = int(max_sleep_time)
            
            for key, value in kwargs.items():
                setattr(self, key, value)

            connected_passthroughs = {
                "latent": latent_in, "image": image_in, "audio": audio_in, "generic": pass_through_generic
            }
            active_passthroughs = {k: v for k, v in connected_passthroughs.items() if v is not None}
            if len(active_passthroughs) == 0:
                 logger.warning("[GPU Temp Protect] No pass-through input connected. Node may not execute in correct order.")
            elif len(active_passthroughs) > 1:
                 logger.warning(f"[GPU Temp Protect] Multiple pass-through inputs connected ({list(active_passthroughs.keys())}). Passing all through.")

            current_temp_int = self.run_protection_logic(profiler)
            temp_status_string = f"{current_temp_int}°C" if current_temp_int is not None else "Error Reading Temp"

            profiler.stop("total_execution")
            if debug_level >= 1: profiler.print_report()

            return (latent_in, image_in, audio_in, pass_through_generic, temp_status_string)

        except Exception as e:
            logger.error(f"[GPU Temp Protect] Unexpected error in execute(): {e}", exc_info=True)
            logging.error(f"ERROR: [GPU Temp Protect] Node execution failed: {e}")
            error_string = f"ERROR: {e}"
            return (latent_in, image_in, audio_in, pass_through_generic, error_string)

# =================================================================================
# == Node Registration                                                           ==
# =================================================================================

NODE_CLASS_MAPPINGS = {
    "GPUTemperatureProtectionEnhanced": GPUTemperatureProtectionEnhanced
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GPUTemperatureProtectionEnhanced": "MD: GPU Temp Protect"
}


# ==============================================================================
# == Unit Tests (smoke — runs without ComfyUI)
# ==============================================================================

if __name__ == "__main__":
    print("\n🧪 Smoke tests: MD_GPUTemperatureProtection")
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

    _check("VERSION defined",    VERSION == "v2.3.0")
    _check("NODE_CLASS_MAPPINGS defined",
           isinstance(NODE_CLASS_MAPPINGS, dict) and len(NODE_CLASS_MAPPINGS) > 0)
    _check("  class GPUTemperatureProtectionEnhanced in map", "GPUTemperatureProtectionEnhanced" in NODE_CLASS_MAPPINGS)

    print(f"\n  {_pass} passed, {_fail} failed")
    if _fail == 0:
        print("  🎉 All good.")
