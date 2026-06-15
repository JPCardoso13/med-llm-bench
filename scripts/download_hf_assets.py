import os
from pathlib import Path

import yaml


def _load_model_ids(models_dir: Path) -> list[str]:
    model_ids = []
    for yaml_file in sorted(models_dir.glob("*.yaml")):
        cfg = yaml.safe_load(yaml_file.read_text())
        model_ids.append(cfg["model_id"])
    return model_ids


def _load_dataset_configs(datasets_dir: Path) -> list[dict]:
    configs = []
    for yaml_file in sorted(datasets_dir.glob("*.yaml")):
        if yaml_file.stem == "template":
            continue
        cfg = yaml.safe_load(yaml_file.read_text())
        source = cfg.get("source", {})
        if "hub_path" not in source:
            continue
        splits = list(cfg.get("splits", {}).values())
        configs.append({
            "repo_id": source["hub_path"],
            "subset": source.get("subset"),
            "splits": splits,
        })
    return configs


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

    models_dir = workdir / "configs" / "models"
    datasets_dir = workdir / "configs" / "datasets"

    models = _load_model_ids(models_dir)
    dataset_configs = _load_dataset_configs(datasets_dir)

    failures = []

    print("=== Downloading models ===")
    for repo_id in models:
        try:
            local_path = snapshot_download(repo_id=repo_id, cache_dir=str(hub_cache), repo_type="model")
            print("{0} -> {1}".format(repo_id, local_path))
        except Exception as exc:
            failures.append((repo_id, str(exc)))
            print("ERROR model {0}: {1}".format(repo_id, exc))

    print("=== Downloading datasets ===")
    for dataset_cfg in dataset_configs:
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
