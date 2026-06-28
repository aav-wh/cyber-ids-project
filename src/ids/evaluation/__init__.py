# ids.evaluation — evaluation metrics, reports, cross-day analysis, bootstrap CI
from ids.evaluation.metrics import compute_full_metrics
from ids.evaluation.reports import classification_report_df

__all__ = ["compute_full_metrics", "classification_report_df"]
