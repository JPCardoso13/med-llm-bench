from llm_bench.ingestion.yaml_loader import YamlLoader
from llm_bench.backends.openai_backend import OpenAIBackend
from llm_bench.prompt.mcq_formatter import MCQFormatter
from llm_bench.runner.sequential_runner import SequentialRunner
from jinja2 import Environment, BaseLoader
import yaml
from pathlib import Path


def load_prompt_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# Load data
loader = YamlLoader("configs/datasets/medqa.yaml")
data = loader.load()
eval_samples = data["eval"]
fewshot_samples = data["fewshot"]

# Load prompt config
prompt_cfg = load_prompt_config("configs/prompts/mcq_default.yaml")

# Build formatter
formatter = MCQFormatter(
    system_prompt=prompt_cfg["system_prompt"],
    user_turn_template=prompt_cfg["user_turn_template"],
    fewshot_template=prompt_cfg.get("fewshot_template"),
)

# Build backend
backend = OpenAIBackend(
    model_id="your-model-id-here",
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
    temperature=0.0,
    max_tokens=256,
)

# Build runner
runner = SequentialRunner(
    backend=backend,
    formatter=formatter,
    task_name="closed_domain_knowledge_retrieval",
    dataset_name="medqa",
    output_path="outputs/test_run.jsonl",
    num_fewshot=3,
    fewshot_pool=fewshot_samples,
    flush_every=5,
)

# Run on a small slice first
results = runner.run(eval_samples[:10])

# Sanity check
for r in results[:2]:
    print(r.model_dump_json(indent=2))