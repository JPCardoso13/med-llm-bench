from __future__ import annotations

from pathlib import Path

import pandas as pd


def pivot_headline_table(headline_df: pd.DataFrame, task_id: str, dataset: str) -> pd.DataFrame:
    """Wide table for one task+dataset: rows=model, columns=metric, values=score."""
    subset = headline_df[(headline_df["task_id"] == task_id) & (headline_df["dataset"] == dataset)]
    if subset.empty:
        return pd.DataFrame()

    wide = subset.pivot_table(index="model_name", columns="metric_name", values="value")
    wide = wide.sort_index()
    wide.index.name = "model"
    return wide


def pivot_group_by_table(group_df: pd.DataFrame, task_id: str, dataset: str, group_field: str, metric_name: str) -> pd.DataFrame:
    """Wide table for one task+dataset+grouping field+metric: rows=model, columns=group value."""
    subset = group_df[
        (group_df["task_id"] == task_id)
        & (group_df["dataset"] == dataset)
        & (group_df["group_field"] == group_field)
        & (group_df["metric_name"] == metric_name)
    ]
    if subset.empty:
        return pd.DataFrame()

    wide = subset.pivot_table(index="model_name", columns="group_value", values="value")
    wide = wide.sort_index()
    wide.index.name = "model"
    return wide


def pivot_judge_table(judge_df: pd.DataFrame, task_id: str, dataset: str, rubric_item: str) -> pd.DataFrame:
    """Wide table for one task+dataset+rubric item: rows=model, columns=judge label."""
    subset = judge_df[
        (judge_df["task_id"] == task_id) & (judge_df["dataset"] == dataset) & (judge_df["rubric_item"] == rubric_item)
    ]
    if subset.empty:
        return pd.DataFrame()

    label_order = subset.sort_values("label_order")["label"].unique().tolist()
    wide = subset.pivot_table(index="model_name", columns="label", values="percentage")
    wide = wide.reindex(columns=label_order).sort_index()
    wide.index.name = "model"
    return wide


def pivot_reliability_table(reliability_df: pd.DataFrame, task_id: str, dataset: str) -> pd.DataFrame:
    """Wide table for one task+dataset: rows=model, columns=failure-rate metric."""
    subset = reliability_df[(reliability_df["task_id"] == task_id) & (reliability_df["dataset"] == dataset)]
    if subset.empty:
        return pd.DataFrame()

    wide = subset.pivot_table(index="model_name", columns="metric_name", values="value")
    wide = wide.sort_index()
    wide.index.name = "model"
    return wide


def qualitative_examples_table(examples_df: pd.DataFrame, task_id: str, dataset: str, model_name: str) -> pd.DataFrame:
    """Best/worst example rows for one task+dataset+model, ready to write as-is."""
    subset = examples_df[
        (examples_df["task_id"] == task_id) & (examples_df["dataset"] == dataset) & (examples_df["model_name"] == model_name)
    ][["rank", "sample_id", "composite_score", "response_preview", "reference_answer"]]
    return subset.sort_values(["rank", "composite_score"])


def save_qualitative_examples(table: pd.DataFrame, out_dir: str | Path, name: str) -> tuple[Path, Path]:
    """CSV + Markdown, not CSV + LaTeX - long free-text preview columns don't
    fit a rigid LaTeX tabular grid without column-width/escaping work that
    isn't worth it for a quick-reference table meant for picking out a quote,
    not pasting the whole thing into a thesis as-is."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"{name}.csv"
    table.to_csv(csv_path, index=False)

    md_path = out_dir / f"{name}.md"
    md_path.write_text(table.to_markdown(index=False))

    return csv_path, md_path


def save_table(table: pd.DataFrame, out_dir: str | Path, name: str, caption: str | None = None) -> tuple[Path, Path]:
    """Write both a .csv and a .tex twin of the same table, same base name."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"{name}.csv"
    table.to_csv(csv_path, float_format="%.4f")

    tex_path = out_dir / f"{name}.tex"
    # Clear both axis names before styling - pandas otherwise renders them as
    # an extra header row (e.g. "metric_name" / blank "model" row), which is
    # noise in a table meant to be pasted straight into a thesis document.
    latex_table = table.copy()
    latex_table.index.name = None
    latex_table.columns.name = None
    tex_path.write_text(
        latex_table.style.format(precision=4).to_latex(
            caption=caption or name.replace("_", " "),
            label=f"tab:{name}",
            hrules=True,
        )
    )

    return csv_path, tex_path
