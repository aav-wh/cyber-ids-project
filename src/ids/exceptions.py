"""
ids.exceptions
--------------
Custom exception hierarchy for the AI-IDS project.

Catching IDSError catches all project-specific errors.
Catching a subclass catches only that category.

Usage
-----
    from ids.exceptions import FeatureValidationError, ModelNotFoundError

    raise FeatureValidationError("Expected 78 features, got 42")
    raise ModelNotFoundError("random_forest.pkl not found")
"""


class IDSError(Exception):
    """Base class for all AI-IDS project exceptions."""


# ── Data errors ───────────────────────────────────────────────────────────────

class DataLoadError(IDSError):
    """Raised when CSV files cannot be loaded or are malformed."""


class DataCleaningError(IDSError):
    """Raised when data cleaning fails (e.g. all rows dropped)."""


class DataSplitError(IDSError):
    """Raised when train/test splitting produces an invalid split."""


# ── Feature errors ────────────────────────────────────────────────────────────

class FeatureValidationError(IDSError):
    """Raised when a feature vector fails validation (wrong shape, NaN, Inf)."""


class FeatureDriftError(IDSError):
    """Raised when significant covariate drift is detected."""


class FeatureEngineeringError(IDSError):
    """Raised when a feature engineering step fails."""


# ── Model errors ──────────────────────────────────────────────────────────────

class ModelNotFoundError(IDSError, FileNotFoundError):
    """Raised when a required model artefact (.pkl) does not exist."""


class ModelTrainingError(IDSError):
    """Raised when model training fails."""


class ModelInferenceError(IDSError):
    """Raised when inference fails (e.g. unexpected input shape)."""


class EnsembleError(IDSError):
    """Raised when the ensemble logic encounters an inconsistency."""


# ── API errors ────────────────────────────────────────────────────────────────

class APIValidationError(IDSError):
    """Raised when an API request payload fails validation."""


class APIRateLimitError(IDSError):
    """Raised when an API rate limit is exceeded."""


# ── Monitoring errors ─────────────────────────────────────────────────────────

class DriftDetectionError(IDSError):
    """Raised when the drift monitor cannot compute statistics."""


class PerformanceDegradationError(IDSError):
    """Raised when live model performance falls below a threshold."""


# ── Config errors ─────────────────────────────────────────────────────────────

class ConfigurationError(IDSError):
    """Raised when the project configuration is invalid."""
