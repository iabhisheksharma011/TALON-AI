"""
Hardware detection — CPU cores, GPU info (NVIDIA via nvidia-smi, AMD via rocm-smi).
Works on Windows and Linux.
"""

import subprocess
import os
import platform
import re
from typing import Dict, List


def _run(cmd: List[str], timeout: int = 5) -> str:
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            creationflags=0x08000000 if platform.system() == "Windows" else 0  # NO_WINDOW on Windows
        )
        return result.stdout.strip()
    except Exception:
        return ""


def get_cpu_info() -> Dict:
    cpu_count = os.cpu_count() or 4
    # Suggested thread count: leave 1-2 cores for OS
    suggested_threads = max(1, cpu_count - 2)

    name = ""
    # Windows
    out = _run(["wmic", "cpu", "get", "Name", "/value"])
    if out:
        m = re.search(r"Name=(.+)", out)
        if m:
            name = m.group(1).strip()

    # Linux fallback
    if not name:
        out = _run(["grep", "-m1", "model name", "/proc/cpuinfo"])
        if out:
            name = out.split(":", 1)[-1].strip()

    return {
        "name": name or "Unknown CPU",
        "cores": cpu_count,
        "suggested_threads": suggested_threads,
    }


def get_gpu_info() -> List[Dict]:
    gpus = []

    # ── NVIDIA via nvidia-smi ─────────────────────────────
    nvidia_out = _run([
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits"
    ])
    if nvidia_out:
        for i, line in enumerate(nvidia_out.strip().splitlines()):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                try:
                    mem_total_mb = int(parts[1])
                    mem_free_mb  = int(parts[2]) if len(parts) > 2 else 0
                    util         = int(parts[3]) if len(parts) > 3 else 0
                except ValueError:
                    mem_total_mb = mem_free_mb = util = 0

                gpus.append({
                    "index": i,
                    "name": parts[0],
                    "vendor": "NVIDIA",
                    "vram_total_mb": mem_total_mb,
                    "vram_free_mb": mem_free_mb,
                    "utilization_pct": util,
                    "available": True,
                })

    # ── AMD via rocm-smi ──────────────────────────────────
    if not gpus:
        rocm_out = _run(["rocm-smi", "--showmeminfo", "vram", "--csv"])
        if rocm_out and "GPU" in rocm_out:
            lines = rocm_out.strip().splitlines()
            for i, line in enumerate(lines[1:], 0):
                parts = [p.strip() for p in line.split(",")]
                try:
                    used = int(parts[1]) if len(parts) > 1 else 0
                    total = int(parts[2]) if len(parts) > 2 else 0
                except ValueError:
                    used = total = 0
                gpus.append({
                    "index": i,
                    "name": f"AMD GPU {i}",
                    "vendor": "AMD",
                    "vram_total_mb": total // 1024,
                    "vram_free_mb": (total - used) // 1024,
                    "utilization_pct": 0,
                    "available": True,
                })

    # ── Windows WMIC fallback (basic) ─────────────────────
    if not gpus:
        wmic_out = _run(["wmic", "path", "win32_VideoController", "get", "Name,AdapterRAM", "/value"])
        if wmic_out:
            names = re.findall(r"Name=(.+)", wmic_out)
            rams  = re.findall(r"AdapterRAM=(\d+)", wmic_out)
            for i, name in enumerate(names):
                ram_mb = int(rams[i]) // (1024 * 1024) if i < len(rams) and rams[i].isdigit() else 0
                vendor = "NVIDIA" if "nvidia" in name.lower() else ("AMD" if "amd" in name.lower() else "Unknown")
                gpus.append({
                    "index": i,
                    "name": name.strip(),
                    "vendor": vendor,
                    "vram_total_mb": ram_mb,
                    "vram_free_mb": ram_mb,
                    "utilization_pct": 0,
                    "available": vendor in ("NVIDIA", "AMD"),
                })

    return gpus


def get_hardware_summary() -> Dict:
    cpu = get_cpu_info()
    gpus = get_gpu_info()
    return {
        "cpu": cpu,
        "gpus": gpus,
        "has_gpu": len(gpus) > 0,
    }
