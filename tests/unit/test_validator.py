"""
Unit tests for ids.data.validator
"""

import numpy as np
import pytest

from ids.data.validator import validate_feature_vector, validate_batch
from ids.exceptions import FeatureValidationError


class TestValidateFeatureVector:
    def test_valid_vector_passes(self):
        fv = [0.0] * 78
        result = validate_feature_vector(fv, 78)
        assert result.shape == (78,)

    def test_wrong_length_raises(self):
        with pytest.raises(FeatureValidationError, match="Expected 78"):
            validate_feature_vector([0.0] * 10, 78)

    def test_nan_raises(self):
        fv = [0.0] * 78
        fv[0] = float("nan")
        with pytest.raises(FeatureValidationError, match="NaN"):
            validate_feature_vector(fv, 78)

    def test_inf_raises(self):
        fv = [0.0] * 78
        fv[1] = float("inf")
        with pytest.raises(FeatureValidationError, match="Inf"):
            validate_feature_vector(fv, 78)

    def test_nan_allowed_when_flag_set(self):
        fv = [float("nan")] * 78
        # Should not raise
        validate_feature_vector(fv, 78, allow_nan=True, allow_inf=True)

    def test_non_numeric_raises(self):
        fv = ["not_a_number"] + [0.0] * 77
        with pytest.raises(FeatureValidationError):
            validate_feature_vector(fv, 78)

    def test_output_dtype_float32(self):
        fv = [1.0] * 78
        result = validate_feature_vector(fv, 78)
        assert result.dtype == np.float32


class TestValidateBatch:
    def test_valid_batch_passes(self):
        flows = [[0.0] * 78] * 5
        # Should not raise
        validate_batch(flows, 78)

    def test_empty_list_raises(self):
        with pytest.raises(FeatureValidationError, match="non-empty"):
            validate_batch([], 78)

    def test_oversized_batch_raises(self):
        flows = [[0.0] * 78] * 1001
        with pytest.raises(FeatureValidationError, match="1000"):
            validate_batch(flows, 78, max_size=1000)
