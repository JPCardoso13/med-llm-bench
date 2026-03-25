import yaml
import importlib.util
import logging
from typing import List, Dict, Any
from datasets import load_dataset
from jinja2 import Template
from pydantic import ValidationError

from .base_loader import BaseLoader


logger = logging.getLogger(__name__)


class YamlLoader(BaseLoader):
    def __init__(self, config_path: str):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        super().__init__(schema_type=self.config.get("schema"))

        source_cfg = self.config.get("source", {})
        self.source_name = source_cfg.get("name")
        self.hub_path = source_cfg.get("hub_path")
        self.subset = source_cfg.get("subset")
        self.data_files = source_cfg.get("data_files")

        splits_cfg = self.config.get("splits", {})
        self.eval_split = splits_cfg.get("eval")
        self.fewshot_split = splits_cfg.get("fewshot")

        if self.hub_path and self.data_files:
            raise ValueError("Configuration cannot contain both 'hub_path' and 'data_files'. Choose one.")
        if isinstance(self.data_files, dict) and (self.eval_split or self.fewshot_split):
            raise ValueError("Cannot define 'splits' slicing when using explicit dictionary paths in 'data_files'.")

        self.mapping = self.config.get("mapping", {})
        self.preprocess = None

        if script_path := self.config.get("preprocess_script"):
            spec = importlib.util.spec_from_file_location("preprocess", script_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.preprocess = getattr(module, "preprocess")

    def load(self) -> Dict[str, List[Any]]:
        result = {"eval": [], "fewshot": []}

        ds_name = self.source_name or (self.hub_path or str(self.data_files or "local")).split('/')[-1]

        if self.eval_split or (isinstance(self.data_files, dict) and "eval" in self.data_files):
            result["eval"] = self._load_split(self.eval_split, ds_name, is_fewshot=False)

        if self.fewshot_split or (isinstance(self.data_files, dict) and "fewshot" in self.data_files):
            result["fewshot"] = self._load_split(self.fewshot_split, ds_name, is_fewshot=True)
            
        return result

    def _load_split(self, split_name: str, ds_name: str, is_fewshot: bool) -> List[Any]:
        if self.hub_path:
            raw_data = load_dataset(self.hub_path, name=self.subset, split=split_name)
        elif isinstance(self.data_files, dict):
            target_path = self.data_files.get("fewshot" if is_fewshot else "eval")
            if not target_path: return []
            ext = "json" if target_path.endswith("jsonl") else target_path.split('.')[-1]
            raw_data = load_dataset(ext, data_files=target_path, split="train")
        else:
            ext = "json" if str(self.data_files).endswith("jsonl") else str(self.data_files).split('.')[-1]
            if ext not in ["json", "csv", "parquet", "txt"]: ext = "parquet" if "parquet" in str(self.data_files) else "json"
            raw_data = load_dataset(ext, data_files=self.data_files, split=split_name)

        samples = []
        for idx, row in enumerate(raw_data):
            if self.preprocess:
                row = self.preprocess(row)
                
            mapped_data = self._apply_mapping(row)

            if "id" not in mapped_data:
                mapped_data["id"] = f"{ds_name}_{idx}"
                
            mapped_data["source"] = ds_name
            
            # Pydantic handles validation
            try:
                samples.append(self.schema_class(**mapped_data))
            except ValidationError as exc:
                logger.warning(
                    "Skipping invalid sample for source=%s split=%s idx=%s id=%s: %s",
                    ds_name,
                    split_name,
                    idx,
                    mapped_data.get("id", "<missing>"),
                    exc,
                )
                continue

        return samples

    def _apply_mapping(self, row: Dict[str, Any]) -> Dict[str, Any]:
        mapped = {}
        for target_field, source_field in self.mapping.items():
            if isinstance(source_field, dict):
                mapped[target_field] = {
                    k: [row.get(v)] if target_field == "grouping" and isinstance(row.get(v), str) else row.get(v)
                    for k, v in source_field.items()
                }
            elif isinstance(source_field, str) and ("{{" in source_field or "{%" in source_field):
                mapped[target_field] = Template(source_field).render(**row)
            else:
                mapped[target_field] = row.get(source_field)
        return mapped
