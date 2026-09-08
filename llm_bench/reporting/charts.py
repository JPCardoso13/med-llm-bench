from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

# Validated categorical palette (fixed order - this IS the CVD-safety mechanism,
# never cycle or generate past slot 8). Light-mode steps from the dataviz skill's
# reference palette; this is a print/static-figure context, so only light mode
# is used, not the dark-mode pairing. Used as the fallback for any task_id not
# in TASK_CATEGORICAL_PALETTES below.
CATEGORICAL_PALETTE = [
    "#2a78d6",  # 1 blue
    "#008300",  # 2 green
    "#e87ba4",  # 3 magenta
    "#eda100",  # 4 yellow
    "#1baf7a",  # 5 aqua
    "#eb6834",  # 6 orange
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
]
FOLD_COLOR = "#898781"  # muted gray, for any model past the 8-slot ceiling

# Sequential blue ramp (light -> dark), for magnitude encoding (heatmaps).
# Fallback for any task_id not in TASK_SEQUENTIAL_RAMPS below.
SEQUENTIAL_BLUE_STEPS = [
    "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b",
]

# Per-task single-hue-family palettes (2026-09-04) - every chart for a given
# task uses shades of the same hue, so a reader scanning a paper full of
# figures can tell which task a figure belongs to at a glance, independent
# of reading its title. Within a task, shades vary in lightness (not hue) to
# stay distinguishable as categorical identifiers - CVD-safety here relies on
# the per-bar/per-cell value labels already drawn on every chart, not on hue
# separation. Add an entry here for any new task; anything absent falls back
# to CATEGORICAL_PALETTE/SEQUENTIAL_BLUE_STEPS.
TASK_CATEGORICAL_PALETTES: dict[str, list[str]] = {
    "cdkr": [
        "#1c5aa8", "#2a78d6", "#6ea8e8", "#123f7a",
        "#4fa8d8", "#0d3b66", "#7fc4e8", "#2f4f8f",
    ],
    "oecr": [
        "#1a7a1a", "#4caf50", "#0d5c0d", "#7fc97f",
        "#2e8b57", "#0a4d0a", "#66bb6a", "#3d8b40",
    ],
    "src": [
        "#6a3fa0", "#4a3aa7", "#9b7fd4", "#3a2570",
        "#8055b8", "#2d1a52", "#b39ddb", "#5c3d99",
    ],
}

TASK_SEQUENTIAL_RAMPS: dict[str, list[str]] = {
    "cdkr": SEQUENTIAL_BLUE_STEPS,
    "oecr": ["#d4edda", "#a8dab5", "#7bc788", "#4caf50", "#2e8b3d", "#1a6b1a", "#0d4d0d"],
    "src": ["#e6d9f5", "#c9a8e8", "#ab7fd4", "#8a5cc0", "#6a3fa0", "#4a2570", "#2d1550"],
}

PRIMARY_INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED_INK = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"


def _sequential_cmap(task_id: str | None = None):
    steps = TASK_SEQUENTIAL_RAMPS.get(task_id, SEQUENTIAL_BLUE_STEPS)
    return LinearSegmentedColormap.from_list(f"seq_{task_id or 'default'}", steps)


def model_color_map(model_names: list[str], task_id: str | None = None) -> dict[str, str]:
    """Fixed model -> color assignment, stable across figures and across runs.

    Sorted alphabetically (not by rank/value) so a model keeps its color even
    if scores change between runs - recoloring on re-sort is exactly the
    anti-pattern this avoids. Past 8 models, extras fold to a shared muted
    gray rather than generating indistinguishable new hues. task_id picks the
    task's single-hue-family palette (TASK_CATEGORICAL_PALETTES); an unknown
    or missing task_id falls back to the original multi-hue CATEGORICAL_PALETTE.
    """
    palette = TASK_CATEGORICAL_PALETTES.get(task_id, CATEGORICAL_PALETTE)
    ordered = sorted(model_names)
    colors: dict[str, str] = {}
    for i, name in enumerate(ordered):
        colors[name] = palette[i] if i < len(palette) else FOLD_COLOR
    return colors


def metric_color_map(metric_names: list[str], task_id: str | None = None) -> dict[str, str]:
    """Fixed metric -> color assignment - a SEPARATE namespace from
    model_color_map. Used only in charts where metric (not model) is the
    identity being distinguished by color, so there's no collision with the
    model-color convention used everywhere else.
    """
    palette = TASK_CATEGORICAL_PALETTES.get(task_id, CATEGORICAL_PALETTE)
    ordered = sorted(metric_names)
    colors: dict[str, str] = {}
    for i, name in enumerate(ordered):
        colors[name] = palette[i] if i < len(palette) else FOLD_COLOR
    return colors


def _style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=SECONDARY_INK, labelsize=9)
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)


def plot_headline_bar_chart(
    headline_df: pd.DataFrame,
    task_id: str,
    dataset: str,
    metric_name: str,
    out_path: str | Path,
    title: str | None = None,
) -> Path:
    """One bar per model for a single task+dataset+metric - the core comparison chart."""
    subset = headline_df[
        (headline_df["task_id"] == task_id)
        & (headline_df["dataset"] == dataset)
        & (headline_df["metric_name"] == metric_name)
    ].sort_values("model_name")

    if subset.empty:
        raise ValueError(f"No data for {task_id}/{dataset}/{metric_name}")

    colors_by_model = model_color_map(subset["model_name"].tolist(), task_id=task_id)
    colors = [colors_by_model[m] for m in subset["model_name"]]

    fig, ax = plt.subplots(figsize=(max(4, 0.9 * len(subset)), 4))
    fig.patch.set_facecolor(SURFACE)
    _style_axes(ax)

    bars = ax.bar(subset["model_name"], subset["value"], color=colors, width=0.6, zorder=3)

    for bar, value in zip(bars, subset["value"]):
        ax.annotate(
            f"{value:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            color=SECONDARY_INK,
        )

    ax.set_ylabel(metric_name, color=PRIMARY_INK, fontsize=10)
    ax.set_title(title or f"{task_id}/{dataset} - {metric_name}", color=PRIMARY_INK, fontsize=12, loc="left")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_grouped_metric_bar_chart(
    headline_df: pd.DataFrame,
    task_id: str,
    dataset: str,
    metric_names: list[str],
    out_path: str | Path,
    title: str | None = None,
) -> Path:
    """Multiple metrics side by side per model - one cluster of bars per
    model, one color per metric (not per model, since here the metrics are
    what's being distinguished, not the models). Intended for a small metric
    set (comfortably legible up to ~3, per the dataviz skill's series-count
    ladder) - accuracy/precision/recall, not an open-ended list.
    """
    subset = headline_df[
        (headline_df["task_id"] == task_id)
        & (headline_df["dataset"] == dataset)
        & (headline_df["metric_name"].isin(metric_names))
    ]
    if subset.empty:
        raise ValueError(f"No data for {task_id}/{dataset}/{metric_names}")

    models = sorted(subset["model_name"].unique())
    colors_by_metric = metric_color_map(metric_names, task_id=task_id)
    n_metrics = len(metric_names)
    bar_width = 0.8 / n_metrics
    x = range(len(models))

    fig, ax = plt.subplots(figsize=(max(5, 1.4 * len(models)), 4.5))
    fig.patch.set_facecolor(SURFACE)
    _style_axes(ax)

    for i, metric_name in enumerate(metric_names):
        values = []
        for model in models:
            row = subset[(subset["model_name"] == model) & (subset["metric_name"] == metric_name)]
            values.append(float(row["value"].iloc[0]) if not row.empty else 0.0)
        offsets = [xi + (i - (n_metrics - 1) / 2) * bar_width for xi in x]
        bars = ax.bar(offsets, values, width=bar_width * 0.9, color=colors_by_metric[metric_name], label=metric_name, zorder=3)
        for bar, value in zip(bars, values):
            ax.annotate(f"{value:.2f}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                        fontsize=7, color=SECONDARY_INK)

    ax.set_xticks(list(x))
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylabel("score", color=PRIMARY_INK, fontsize=10)
    ax.set_title(title or f"{task_id}/{dataset} - {', '.join(metric_names)}", color=PRIMARY_INK, fontsize=12, loc="left")
    # Legend goes to the right of the axes, not below - rotated x-tick labels
    # (model names, which can be long) sit in that lower region, and a
    # below-axes legend collides with them regardless of vertical offset.
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=9)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _ordinal_ramp(n: int, task_id: str | None = None) -> list[str]:
    """n evenly-spaced steps from the task's sequential ramp, light (worst) -> dark (best).

    Judge labels (Poor/Fair/.../Excellent) are an ordered quality ladder, not
    a symmetric agree<->disagree scale - a single-hue ordinal ramp is the
    honest encoding here, not a diverging pair (which implies a neutral
    midpoint this data doesn't have).
    """
    steps = TASK_SEQUENTIAL_RAMPS.get(task_id, SEQUENTIAL_BLUE_STEPS)
    if n <= len(steps):
        step = len(steps) / n
        return [steps[int(i * step)] for i in range(n)]
    cmap = _sequential_cmap(task_id)
    return [cmap(i / max(1, n - 1)) for i in range(n)]


def plot_judge_distribution(
    judge_df: pd.DataFrame,
    task_id: str,
    dataset: str,
    rubric_item: str,
    out_path: str | Path,
    title: str | None = None,
) -> Path:
    """100%-stacked horizontal bar: one bar per model, segments = judge label share."""
    subset = judge_df[
        (judge_df["task_id"] == task_id)
        & (judge_df["dataset"] == dataset)
        & (judge_df["rubric_item"] == rubric_item)
    ]
    if subset.empty:
        raise ValueError(f"No judge data for {task_id}/{dataset}/{rubric_item}")

    labels = subset.sort_values("label_order")["label"].unique().tolist()
    models = sorted(subset["model_name"].unique())
    ramp = _ordinal_ramp(len(labels), task_id=task_id)

    fig, ax = plt.subplots(figsize=(7, max(2.5, 0.5 * len(models) + 1)))
    fig.patch.set_facecolor(SURFACE)
    _style_axes(ax)
    ax.xaxis.grid(True, color=GRIDLINE, linewidth=0.8, linestyle="-")
    ax.yaxis.grid(False)

    left = [0.0] * len(models)
    for label, color in zip(labels, ramp):
        values = []
        for model in models:
            row = subset[(subset["model_name"] == model) & (subset["label"] == label)]
            values.append(float(row["percentage"].iloc[0]) if not row.empty else 0.0)
        ax.barh(models, values, left=left, color=color, height=0.6, label=label, zorder=3)
        for i, (value, l) in enumerate(zip(values, left)):
            if value >= 0.08:
                text_color = "#ffffff" if labels.index(label) >= len(labels) / 2 else PRIMARY_INK
                ax.text(l + value / 2, i, f"{value:.0%}", ha="center", va="center", fontsize=8, color=text_color)
        left = [l + v for l, v in zip(left, values)]

    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=len(labels), frameon=False, fontsize=9)
    ax.set_title(title or f"{task_id}/{dataset} - judge: {rubric_item}", color=PRIMARY_INK, fontsize=12, loc="left")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_tradeoff_scatter(
    systems_df: pd.DataFrame,
    headline_df: pd.DataFrame,
    task_id: str,
    dataset: str,
    x_metric: str,
    y_metric: str,
    out_path: str | Path,
    title: str | None = None,
) -> Path:
    """One point per model: x = a systems metric (latency/throughput), y = a quality metric."""
    x_subset = systems_df[
        (systems_df["task_id"] == task_id) & (systems_df["dataset"] == dataset) & (systems_df["metric_name"] == x_metric)
    ][["model_name", "value"]].rename(columns={"value": "x"})
    y_subset = headline_df[
        (headline_df["task_id"] == task_id) & (headline_df["dataset"] == dataset) & (headline_df["metric_name"] == y_metric)
    ][["model_name", "value"]].rename(columns={"value": "y"})

    merged = x_subset.merge(y_subset, on="model_name", how="inner")
    if merged.empty:
        raise ValueError(f"No overlapping data for {task_id}/{dataset}: {x_metric} vs {y_metric}")

    colors_by_model = model_color_map(merged["model_name"].tolist(), task_id=task_id)

    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor(SURFACE)
    _style_axes(ax)
    ax.xaxis.grid(True, color=GRIDLINE, linewidth=0.8, linestyle="-")

    for _, row in merged.iterrows():
        ax.scatter(row["x"], row["y"], s=90, color=colors_by_model[row["model_name"]], zorder=3, edgecolors=SURFACE, linewidths=1.5)
        ax.annotate(row["model_name"], xy=(row["x"], row["y"]), xytext=(6, 4), textcoords="offset points",
                    fontsize=8, color=SECONDARY_INK)

    ax.set_xlabel(x_metric, color=PRIMARY_INK, fontsize=10)
    ax.set_ylabel(y_metric, color=PRIMARY_INK, fontsize=10)
    ax.set_title(title or f"{task_id}/{dataset} - {y_metric} vs {x_metric}", color=PRIMARY_INK, fontsize=12, loc="left")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_group_by_heatmap(
    group_df: pd.DataFrame,
    task_id: str,
    dataset: str,
    group_field: str,
    metric_name: str,
    out_path: str | Path,
    title: str | None = None,
) -> Path:
    """Model x subgroup-value grid, cell color = metric magnitude (sequential blue)."""
    subset = group_df[
        (group_df["task_id"] == task_id)
        & (group_df["dataset"] == dataset)
        & (group_df["group_field"] == group_field)
        & (group_df["metric_name"] == metric_name)
    ]
    if subset.empty:
        raise ValueError(f"No group_by data for {task_id}/{dataset}/{group_field}/{metric_name}")

    wide = subset.pivot_table(index="model_name", columns="group_value", values="value").sort_index()

    fig, ax = plt.subplots(figsize=(max(5, 1.1 * len(wide.columns)), max(3, 0.6 * len(wide.index) + 1.5)))
    fig.patch.set_facecolor(SURFACE)

    im = ax.imshow(wide.values, cmap=_sequential_cmap(task_id), vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(wide.columns)))
    ax.set_xticklabels(wide.columns, rotation=30, ha="right", color=SECONDARY_INK, fontsize=9)
    ax.set_yticks(range(len(wide.index)))
    ax.set_yticklabels(wide.index, color=SECONDARY_INK, fontsize=9)

    for i in range(len(wide.index)):
        for j in range(len(wide.columns)):
            value = wide.values[i, j]
            if pd.isna(value):
                continue
            # Light cells need dark text and vice versa - readability over a
            # fixed ink color across the whole magnitude range.
            text_color = PRIMARY_INK if value < 0.6 else "#ffffff"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8, color=text_color)

    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors=SECONDARY_INK, labelsize=8)
    cbar.outline.set_visible(False)

    ax.set_title(
        title or f"{task_id}/{dataset} - {metric_name} by {group_field}",
        color=PRIMARY_INK, fontsize=12, loc="left",
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out_path
