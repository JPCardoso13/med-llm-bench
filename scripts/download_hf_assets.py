#!/usr/bin/env python3
"""Download a fixed set of Hugging Face models and datasets into the local cache."""

import os
from pathlib import Path


MODELS = [
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "microsoft/Phi-3-mini-4k-instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen3-32B-AWQ",
]

DATASETS = [
    {
        "repo_id": "GBaker/MedQA-USMLE-4-options",
        "subset": None,
        "splits": ["test", "train"],
    },
    {
        "repo_id": "TsinghuaC3I/MedXpertQA",
        "subset": "Text",
        "splits": ["test", "dev"],
    },
    {
        "repo_id": "nsk7153/MedCalc-Bench-Verified",
        "subset": None,
        "splits": ["test", "one_shot"],
    },
    {
        "repo_id": "zou-lab/MedCaseReasoning",
        "subset": None,
        "splits": ["test", "val"],
    },
    {
        "repo_id": "ccdv/pubmed-summarization",
        "subset": "document",
        "splits": ["test[:1000]", "train[:1000]"],
    },
]


def main() -> int:
    os.environ.setdefault("TQDM_DISABLE", "1")
    os.environ.setdefault("TQDM_MONITOR_INTERVAL", "0")
    os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    from datasets import load_dataset
    from datasets.utils.logging import disable_progress_bar
    from huggingface_hub import snapshot_download

    disable_progress_bar()

    workdir = Path(__file__).resolve().parents[1]
    hf_home = Path(os.environ.get("HF_HOME", workdir / ".cache" / "huggingface"))
    hub_cache = hf_home / "hub"
    datasets_cache = hf_home / "datasets"

    hub_cache.mkdir(parents=True, exist_ok=True)
    datasets_cache.mkdir(parents=True, exist_ok=True)

    print("HF_HOME: {0}".format(hf_home))
    print("HF hub cache: {0}".format(hub_cache))
    print("HF datasets cache: {0}".format(datasets_cache))

    failures = []

    print("=== Downloading models ===")
    for repo_id in MODELS:
        try:
            local_path = snapshot_download(repo_id=repo_id, cache_dir=str(hub_cache), repo_type="model")
            print("{0} -> {1}".format(repo_id, local_path))
        except Exception as exc:
            failures.append((repo_id, str(exc)))
            print("ERROR model {0}: {1}".format(repo_id, exc))

    print("=== Downloading datasets ===")
    for dataset_cfg in DATASETS:
        repo_id = dataset_cfg["repo_id"]
        subset = dataset_cfg["subset"]
        splits = dataset_cfg["splits"]
        try:
            loaded_splits = []
            for split_name in splits:
                if subset is None:
                    load_dataset(repo_id, split=split_name, cache_dir=str(datasets_cache))
                else:
                    load_dataset(repo_id, subset, split=split_name, cache_dir=str(datasets_cache))
                loaded_splits.append(split_name)

            print("{0} -> splits: {1}".format(repo_id, ", ".join(loaded_splits)))
        except Exception as exc:
            failures.append((repo_id, str(exc)))
            print("ERROR dataset {0}: {1}".format(repo_id, exc))

    if failures:
        print("\nSummary: {0} failure(s)".format(len(failures)))
        for repo_id, message in failures:
            print(" - {0}: {1}".format(repo_id, message))
        return 1

    print("\nDownload complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())