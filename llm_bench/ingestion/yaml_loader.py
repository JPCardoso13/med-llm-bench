import yaml
import hashlib
import logging
import importlib.util
from typing import List, Dict, Any, Union
from datasets import load_dataset
from jinja2 import Template

from llm_bench.schemas import MCQSample, GenerativeSample
from .base_loader import BaseLoader, BenchmarkSample


logger = logging.getLogger(__name__)


class YamlLoader(BaseLoader):
    def __init__(self, config_path: str):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        super().__init__(task_type=self.config.get("task_type", "mcq"))

        self.dataset_source: Union[str, Dict[str, str]] = self.config.get("dataset")
        self.subset = self.config.get("subset")
        self.eval_split = self.config.get("eval_split", "test")
        self.fewshot_split = self.config.get("fewshot_split")
        self.schema_type = self.config.get("schema")
        self.mapping = self.config.get("mapping", {})
        self.preprocess = None

        preprocess_script = self.config.get("preprocess_script")
        if preprocess_script:
            spec = importlib.util.spec_from_file_location("yaml_loader_preprocess", preprocess_script)
            if spec is None or spec.loader is None:
                raise RuntimeError(f"Failed to load preprocess script from: {preprocess_script}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.preprocess = getattr(module, "preprocess", None)

    def load(self) -> Dict[str, List[BenchmarkSample]]:
        result = {"eval": [], "fewshot": []}
        
        # Determine dataset name for ID prefixing
        ds_name = self.dataset_source if isinstance(self.dataset_source, str) else "local_dataset"
        ds_name = ds_name.split('/')[-1]

        result["eval"] = self._load_split(self.eval_split, ds_name, is_fewshot=False)

        if self.fewshot_split:
            result["fewshot"] = self._load_split(self.fewshot_split, ds_name, is_fewshot=True)
            
        return result

    def _load_split(self, split_name: str, ds_name: str, is_fewshot: bool) -> List[BenchmarkSample]:
        # Handle Local Files vs HuggingFace Hub
        if isinstance(self.dataset_source, dict):
            # The YAML defined local paths for eval/fewshot (e.g., dataset: {eval: "path.jsonl"})
            target_path = self.dataset_source.get("fewshot" if is_fewshot else "eval")
            if not target_path:
                return []
            
            # Use 'json' or 'csv' loader based on extension
            ext = target_path.split('.')[-1]
            ext = "json" if ext == "jsonl" else ext
            raw_data = load_dataset(ext, data_files=target_path, split=split_name)
        else:
            # Hugging Face Hub
            raw_data = load_dataset(self.dataset_source, name=self.subset, split=split_name)

        samples: List[BenchmarkSample] = []
        schema_class = MCQSample if self.schema_type == "MCQSample" else GenerativeSample
        total_count = 0
        success_count = 0
        error_count = 0
        
        for idx, row in enumerate(raw_data):
            total_count += 1
            try:
                if self.preprocess:
                    row = self.preprocess(row)
                mapped_data = self._apply_mapping(row)
                
                # Deterministic ID Fallback
                if "id" not in mapped_data:
                    q_text = mapped_data.get("question", str(idx))
                    hash_val = hashlib.md5(q_text.encode('utf-8')).hexdigest()[:10]
                    mapped_data["id"] = f"{ds_name}_{hash_val}"
                    
                mapped_data["source"] = ds_name
                samples.append(schema_class(**mapped_data))
                success_count += 1

            except Exception as e:
                error_count += 1
                logger.warning(f"Error processing entry at index {idx}: {e}")
                if total_count > 50 and error_count > 0.1 * total_count:
                    raise RuntimeError(
                        f"Aborting load for split '{split_name}' due to high error rate: "
                        f"{error_count}/{total_count} entries failed"
                    )

        logger.info(f"Loaded {success_count}/{total_count} samples successfully for {split_name}.")
        return samples

    def _apply_mapping(self, row: Dict[str, Any]) -> Dict[str, Any]:
        mapped = {}
        for target_field, source_field in self.mapping.items():
            if isinstance(source_field, dict):
                nested_map = {}
                for k, v in source_field.items():
                    val = row.get(v)
                    # Smart coercion: string to list for grouping
                    if target_field == "grouping" and isinstance(val, str):
                        nested_map[k] = [val]
                    else:
                        nested_map[k] = val
                mapped[target_field] = nested_map
            else:
                if isinstance(source_field, str) and ("{{" in source_field or "{%" in source_field):
                    mapped[target_field] = Template(source_field).render(**row)
                else:
                    mapped[target_field] = row.get(source_field)
        return mapped
