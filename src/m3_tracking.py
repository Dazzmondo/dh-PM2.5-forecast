"""
MLflow tracking helpers for the Milestone 3 experiments. Not a pipeline step: it is only imported (by train.py, ...).

OUTPUT: an MLflow store in <repo>/mlruns/ (a SQLite file mlflow.db plus an artifacts/ folder). The folder is local; it must not
        be committed (add `mlruns/` to .gitignore). View it with:  mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db

What is recorded for every run (so another student can compare runs and see exactly what produced a number):
  lineage tags   git commit (+ "-dirty"), SHA-256 of the features and block files, dataset run ID and Docker image digest
                 (from the environment variables DATASET_RUN_ID and IMAGE_DIGEST when set), Python and library versions,
                 CPU model, physical cores / threads, RAM, GPU (if any), machine name
  parameters     the full configuration
  artefacts      the configuration (YAML), `pip freeze`
Metrics, timings and per-station tables are logged by train.py.

Requires: pip install mlflow pyyaml numpy
"""
import hashlib
import os
import platform
import sys
from importlib import metadata

import mlflow

from config import REPO_ROOT, git_commit
from m3_data import BLOCKS_PATH, FEATURES_PATH

STORE_DIR = REPO_ROOT / "mlruns"
TRACKING_URI = f"sqlite:///{(STORE_DIR / 'mlflow.db').as_posix()}"
ARTIFACT_URI = (STORE_DIR / "artifacts").as_uri()


def setup_experiment(name: str) -> None:
    """Point MLflow at the local store and select the experiment (created with its artefacts under mlruns/artifacts)."""
    STORE_DIR.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(TRACKING_URI)
    if mlflow.get_experiment_by_name(name) is None:
        mlflow.create_experiment(name, artifact_location=f"{ARTIFACT_URI}/{name}")
    mlflow.set_experiment(name)


def sha256_of_file(path) -> str:
    """Fingerprint of a file's contents, read in 1 MiB chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def peak_ram_mb() -> float:
    """Peak memory (resident set) of this process so far, in MB; None if it cannot be read on this system."""
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            class Counters(ctypes.Structure):
                """The Windows PROCESS_MEMORY_COUNTERS structure."""
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                            ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                            ("PeakPagefileUsage", ctypes.c_size_t)]
            counters = Counters()
            counters.cb = ctypes.sizeof(Counters)
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            if ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
                return counters.PeakWorkingSetSize / 1e6
            return None
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss   # kilobytes on Linux, bytes on macOS
        return peak / 1e3 if sys.platform != "darwin" else peak / 1e6
    except Exception:
        return None


def library_versions() -> dict:
    """Versions of the libraries that decide a result."""
    names = ["numpy", "pandas", "pyarrow", "scikit-learn", "xgboost", "torch", "mlflow", "optuna", "pyyaml"]
    versions = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def machine_description() -> dict:
    """CPU, cores, RAM and GPU of this machine (for the compute comparison)."""
    info = {"machine": platform.node(), "os": platform.platform(), "cpu": platform.processor(), "logical_cores": os.cpu_count()}
    try:
        import torch
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
        info["cuda"] = torch.version.cuda or "none"
    except ImportError:
        info["gpu"] = "unknown"
    return info


def lineage_tags() -> dict:
    """Everything that identifies the code, data, environment and machine behind a run."""
    versions = library_versions()
    tags = {"git_commit": os.environ.get("GIT_COMMIT") or git_commit(),      # inside Docker there is no .git: the commit is a build argument
            "features_file_sha256": sha256_of_file(FEATURES_PATH),
            "block_table_sha256": sha256_of_file(BLOCKS_PATH),
            "dataset_run_id": os.environ.get("DATASET_RUN_ID", "not set"),
            "image_digest": os.environ.get("IMAGE_DIGEST", "not in a container"),
            "python": platform.python_version()}
    tags.update({f"lib_{name}": version for name, version in versions.items()})
    tags.update({f"machine_{key}": str(value) for key, value in machine_description().items()})
    return tags


def pip_freeze_text() -> str:
    """The installed packages and versions, one per line (a `pip freeze` without calling pip)."""
    return "\n".join(sorted(f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions()))
