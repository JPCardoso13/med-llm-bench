#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_REPORTS = [
    Path("outputs/reports/cdkr_llama3_8b_instruct_systems_summary.json"),
    Path("outputs/reports/cdkr_qwen3_32b_awq_systems_summary.json"),
]


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _format_number(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (int, float)):
        return f"{value:,.2f}" if isinstance(value, float) else str(value)
    return str(value)


def _print_group_summary(group: Dict[str, Any]) -> None:
    key = group.get("group_key", {})
    latency = group.get("metrics", {}).get("latency", {})
    throughput = group.get("metrics", {}).get("throughput", {})
    total_latency = latency.get("total_latency_ms", {})
    ttft = latency.get("ttft_ms", {})
    e2e_throughput = throughput.get("e2e_throughput", {})
    prefill_throughput = throughput.get("prefill_throughput", {})
    decoding_throughput = throughput.get("decoding_throughput", {})

    dataset = key.get("dataset", "unknown")
    model_id = key.get("model_id", "unknown")
    sample_count = group.get("sample_count", "n/a")

    print(f"  - {dataset} | {model_id} | samples={sample_count}")
    print(
        "      total latency mean: "
        f"{_format_number(total_latency.get('mean'))} ms | "
        f"p50: {_format_number(total_latency.get('p50'))} ms | "
        f"p95: {_format_number(total_latency.get('p95'))} ms"
    )
    print(
        "      ttft mean: "
        f"{_format_number(ttft.get('mean'))} ms | "
        f"p50: {_format_number(ttft.get('p50'))} ms | "
        f"p95: {_format_number(ttft.get('p95'))} ms"
    )
    print(
        "      throughput e2e mean: "
        f"{_format_number(e2e_throughput.get('mean'))} tok/s | "
        f"p50: {_format_number(e2e_throughput.get('p50'))} tok/s | "
        f"p95: {_format_number(e2e_throughput.get('p95'))} tok/s"
    )
    print(
        "      throughput prefill mean: "
        f"{_format_number(prefill_throughput.get('mean'))} tok/s | "
        f"p50: {_format_number(prefill_throughput.get('p50'))} tok/s | "
        f"p95: {_format_number(prefill_throughput.get('p95'))} tok/s"
    )
    print(
        "      throughput decoding mean: "
        f"{_format_number(decoding_throughput.get('mean'))} tok/s | "
        f"p50: {_format_number(decoding_throughput.get('p50'))} tok/s | "
        f"p95: {_format_number(decoding_throughput.get('p95'))} tok/s"
    )
    print()


def _print_file(path: Path, full_json: bool = False) -> None:
    data = _load_json(path)

    print("=" * 96)
    print(path)
    print("=" * 96)
    print(
        f"profile={data.get('profile_id', 'n/a')} | scope={data.get('scope', 'n/a')} | "
        f"sample_count={data.get('sample_count', 'n/a')} | groups={len(data.get('groups', []))}"
    )

    if full_json:
        print()
        print(json.dumps(data, indent=2, ensure_ascii=False))

    print()
    print("summary:")
    for group in data.get("groups", []):
        _print_group_summary(group)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pretty-print the CDKR systems summary JSON files with a compact latency/TTFT summary."
    )
    parser.add_argument(
        "files",
        nargs="*",
        type=Path,
        default=DEFAULT_REPORTS,
        help="Systems summary JSON files to print. Defaults to the two CDKR report files.",
    )
    parser.add_argument(
        "--full-json",
        action="store_true",
        help="Also print the full pretty-printed JSON contents for each file.",
    )
    args = parser.parse_args()

    for file_path in args.files:
        _print_file(file_path, full_json=args.full_json)


if __name__ == "__main__":
    main()