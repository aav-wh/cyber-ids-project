"""
ids.models.evaluate
-------------------
Evaluation utilities: metrics, classification reports, confusion matrices.

Extracted from notebooks/04_evaluation.ipynb.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

_DEFAULT_RESULTS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "results",
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    label_names: list[str] | None = None,
) -> dict:
    """
    Compute a standard suite of binary classification metrics.

    Parameters
    ----------
    y_true     : true integer labels
    y_pred     : predicted integer labels
    y_proba    : predicted probabilities for positive class (optional, for AUC)
    label_names: class names for the report

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1, auc (if y_proba given),
                    report (full sklearn classification report string)
    """
    metrics = {
        "accuracy":  float(np.mean(y_true == y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "report":    classification_report(y_true, y_pred, target_names=label_names, zero_division=0),
    }

    if y_proba is not None:
        try:
            metrics["auc"] = float(roc_auc_score(y_true, y_proba))
        except ValueError:
            metrics["auc"] = None

    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str],
    title: str = "Confusion Matrix",
    save_path: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """
    Plot a labelled, normalised confusion matrix.

    Parameters
    ----------
    y_true, y_pred : integer label arrays
    label_names    : list of class name strings
    title          : plot title
    save_path      : if given, save figure to this path
    ax             : optional existing matplotlib Axes to draw on

    Returns
    -------
    matplotlib Figure
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.get_figure()

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2%",
        cmap="Blues",
        xticklabels=label_names,
        yticklabels=label_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)

    # Overlay raw counts in smaller text
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j + 0.5, i + 0.7,
                f"n={cm[i, j]:,}",
                ha="center", va="center",
                fontsize=8, color="grey",
            )

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    model_name: str = "Model",
    save_path: str | None = None,
) -> plt.Figure:
    """
    Plot a ROC curve with AUC annotation.

    Parameters
    ----------
    y_true     : true binary labels (0/1)
    y_proba    : predicted probability of the positive class
    model_name : legend label
    save_path  : optional path to save the figure

    Returns
    -------
    matplotlib Figure
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, lw=2, label=f"{model_name} (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
