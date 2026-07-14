"""
Launcher shim for vLLM 0.19+ under Singularity --nv on clusters that do not
set LD_LIBRARY_PATH for NVIDIA libs.

vLLM 0.19+ uses pynvml (CDLL("libnvidia-ml.so.1")) at argument-parse time to
detect the device type.  Ubuntu-based containers don't ship libnvidia-ml, and
some Singularity/Apptainer configurations bind-mount it into a non-standard
path rather than injecting it via LD_LIBRARY_PATH.

This script finds the library, patches os.environ, then os.execvp's into the
real vLLM server so the fix is in place before any vLLM import occurs.
All CLI arguments passed to this script are forwarded unchanged.
"""
from __future__ import annotations

import ctypes
import glob
import os
import subprocess
import sys
from pathlib import Path

_SEARCH_PATHS = [
    "/.singularity.d/libs",
    "/usr/local/nvidia/lib64",
    "/usr/local/nvidia/lib",
    "/run/opengl-driver/lib",
    "/usr/lib/x86_64-linux-gnu/nvidia/current",
    "/usr/lib/x86_64-linux-gnu",
    "/usr/lib64",
    "/usr/lib",
]


def _find_nvml_dir() -> str | None:
    # Already loadable — nothing to do.
    try:
        ctypes.CDLL("libnvidia-ml.so.1")
        return ""
    except OSError:
        pass

    # Check well-known paths first (fast).
    for path in _SEARCH_PATHS:
        if glob.glob(os.path.join(path, "libnvidia-ml.so*")):
            return path

    # Ask ldconfig (covers bind-mounts that updated the ld cache).
    try:
        out = subprocess.check_output(["ldconfig", "-p"], text=True, stderr=subprocess.DEVNULL)
        for line in out.splitlines():
            if "libnvidia-ml" in line and "=>" in line:
                lib_path = line.split("=>")[-1].strip()
                return os.path.dirname(lib_path)
    except Exception:
        pass

    return None


def _sync_hf_env() -> None:
    """Sync HF offline/cache env vars before vLLM imports huggingface_hub.

    vLLM 0.19.1 reads huggingface_hub.constants.HF_HUB_OFFLINE (computed at
    import time) to decide whether to attempt network access.  If any of the
    HF offline env vars didn't reach this process (e.g. because Singularity's
    SINGULARITYENV_* translation was skipped), this function recovers them from
    whatever IS available and ensures the constants are set consistently.
    """
    # If SINGULARITYENV_* vars weren't stripped (unusual Singularity config),
    # manually translate the ones we care about.
    _singularity_map = {
        "SINGULARITYENV_HF_HOME": "HF_HOME",
        "SINGULARITYENV_HUGGINGFACE_HUB_CACHE": "HUGGINGFACE_HUB_CACHE",
        "SINGULARITYENV_HF_HUB_OFFLINE": "HF_HUB_OFFLINE",
        "SINGULARITYENV_TRANSFORMERS_OFFLINE": "TRANSFORMERS_OFFLINE",
        "SINGULARITYENV_HF_OFFLINE": "HF_OFFLINE",
    }
    for s_key, key in _singularity_map.items():
        if s_key in os.environ and key not in os.environ:
            os.environ[key] = os.environ[s_key]

    # Propagate HF_OFFLINE → HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE.
    if os.environ.get("HF_OFFLINE") == "1":
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    # Derive HUGGINGFACE_HUB_CACHE from HF_HOME when not explicitly set.
    if not os.environ.get("HUGGINGFACE_HUB_CACHE"):
        hf_home = os.environ.get("HF_HOME")
        if hf_home:
            os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(hf_home, "hub")

    # Last resort: infer cache from project layout (<workdir>/scripts/vllm_launcher.py).
    # This covers clusters where the SINGULARITYENV_* mechanism is disabled by SLURM and
    # none of the HF env vars make it into the container.
    if not os.environ.get("HUGGINGFACE_HUB_CACHE"):
        candidate = Path(__file__).resolve().parent.parent / ".cache" / "huggingface" / "hub"
        if candidate.is_dir():
            os.environ["HUGGINGFACE_HUB_CACHE"] = str(candidate)
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    print(
        f"[vllm_launcher] HF_HUB_OFFLINE={os.environ.get('HF_HUB_OFFLINE', '<unset>')!r}"
        f" HUGGINGFACE_HUB_CACHE={os.environ.get('HUGGINGFACE_HUB_CACHE', '<unset>')!r}",
        flush=True,
    )


def main() -> None:
    nvml_dir = _find_nvml_dir()

    if nvml_dir is None:
        print("[vllm_launcher] WARNING: libnvidia-ml.so.1 not found; "
              "vLLM platform detection will likely fail.", flush=True)
    elif nvml_dir == "":
        print("[vllm_launcher] libnvidia-ml.so.1 already loadable.", flush=True)
    else:
        current_ld = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = f"{nvml_dir}:{current_ld}" if current_ld else nvml_dir
        print(f"[vllm_launcher] Found libnvidia-ml in {nvml_dir!r}; "
              f"LD_LIBRARY_PATH={os.environ['LD_LIBRARY_PATH']}", flush=True)

    _sync_hf_env()

    # Prepend our patches/ dir so sitecustomize.py runs at vLLM startup and
    # patches prometheus_fastapi_instrumentator before vLLM imports it.
    patches_dir = str(Path(__file__).parent / "patches")
    current_pythonpath = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = f"{patches_dir}:{current_pythonpath}" if current_pythonpath else patches_dir

    # Replace this process with the real vLLM server.
    # sys.argv[1:] contains the vLLM flags (--model, --port, …).
    os.execvp(sys.executable,
              [sys.executable, "-m", "vllm.entrypoints.openai.api_server"] + sys.argv[1:])


if __name__ == "__main__":
    main()
