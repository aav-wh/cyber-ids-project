"""
ids.visualization.plots
-----------------------
Core plotting functions used across all notebooks.

All functions return matplotlib Figure objects and optionally save to disk.
The style is consistent: dark background (#0f1117), accent colours matching
the dashboard (red for ATTACK, green for BENIGN, blue for neutral).
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.metrics import roc_curve, auc


# ── Style ──────────────────────────────────────────────────────────────────────
ATTACK_COLOR = "#e74c3c"
BENIGN_COLOR = "#2ecc71"
ACCENT_COLOR = "#3498db"
MUTED_COLOR  = "#8892a4"


def set_style() -> None:
    """Apply project-wide matplotlib style."""
    plt.rcParams.update({
        "figure.facecolor": "#0f1117",
        "axes.facecolor":   "#1a1d27",
        "axes.edgecolor":   "#2a2d3e",
        "axes.labelcolor":  "#e0e0e0",
        "xtick.color":      "#8892a4",
        "ytick.color":      "#8892a4",
        "text.color":       "#e0e0e0",
        "grid.color":       "#2a2d3e",
        "grid.linestyle":   "--",
        "grid.alpha":       0.5,
    })


def plot_class_distribution(
    y: np.ndarray | pd.Series,
    title: str = "Class Distribution",
    label_names: list[str] | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """Bar chart of class label counts and percentages."""
    labels, counts = np.unique(y, return_counts=True)
    names  = label_names or [str(l) for l in labels]
    colors = [ATTACK_COLOR if "ATTACK" in n.upper() else BENIGN_COLOR for n in names]
    pcts   = counts / counts.sum() * 100

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(names, counts, color=colors, edgecolor="#2a2d3e", linewidth=0.8)

    for bar, pct in zip(bars, pcts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + counts.max() * 0.01,
            f"{pct:.1f}%",
            ha="center", fontsize=10, color="#e0e0e0",
        )

    ax.set_title(title, fontsize=13, pad=12)
    ax.set_ylabel("Count")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_feature_importance(
    importances: np.ndarray,
    feature_names: list[str],
    top_n: int = 20,
    title: str = "Feature Importance (RF MDI)",
    save_path: str | None = None,
) -> plt.Figure:
    """Horizontal bar chart of top-N RF feature importances."""
    order = np.argsort(importances)[::-1][:top_n]
    names  = [feature_names[i] for i in order]
    values = importances[order]

    fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.38)))
    bars = ax.barh(range(len(names)), values[::-1], color=ACCENT_COLOR, edgecolor="#2a2d3e")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names[::-1], fontsize=9)
    ax.set_xlabel("Importance (MDI)")
    ax.set_title(title, fontsize=13, pad=12)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_roc_comparison(
    models: list[tuple[str, np.ndarray, np.ndarray]],
    save_path: str | None = None,
) -> plt.Figure:
    """
    Overlay multiple ROC curves on one plot.

    Parameters
    ----------
    models : list of (name, y_true, y_proba) tuples
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = [ATTACK_COLOR, BENIGN_COLOR, ACCENT_COLOR, "#f39c12", "#9b59b6"]

    for (name, y_true, y_proba), color in zip(models, colors):
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc_score   = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, color=color,
                label=f"{name} (AUC={auc_score:.4f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve Comparison", fontsize=13, pad=12)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_per_day_metrics(
    per_day_df: pd.DataFrame,
    metrics: list[str] | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """Line chart of metrics across days (cross-day drift visualisation)."""
    metrics = metrics or ["f1", "recall", "precision"]
    colors  = [ACCENT_COLOR, BENIGN_COLOR, ATTACK_COLOR]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(per_day_df))

    for metric, color in zip(metrics, colors):
        if metric in per_day_df.columns:
            ax.plot(x, per_day_df[metric], marker="o", lw=2,
                    color=color, label=metric.upper())

    ax.set_xticks(x)
    ax.set_xticklabels(per_day_df["day"].tolist(), rotation=20, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Across Days", fontsize=13, pad=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    label_names: list[str],
    title: str = "Confusion Matrix",
    normalise: bool = True,
    save_path: str | None = None,
) -> plt.Figure:
    """Seaborn heatmap confusion matrix with raw counts overlaid."""
    if normalise:
        cm_plot = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt     = ".2%"
    else:
        cm_plot = cm
        fmt     = "d"

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm_plot, annot=True, fmt=fmt, cmap="Blues",
                xticklabels=label_names, yticklabels=label_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title, fontsize=12, pad=10)

    if normalise:
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j + 0.5, i + 0.75, f"n={cm[i,j]:,}",
                        ha="center", va="center", fontsize=8, color="grey")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
