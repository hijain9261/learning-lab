import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from linalg import Matrix


@pytest.fixture
def assert_matrix_close():
    def _assert(actual, expected, tol=1e-9):
        assert isinstance(actual, Matrix)
        assert actual.rows == len(expected)
        assert actual.cols == len(expected[0])

        for row_actual, row_expected in zip(actual.data, expected):
            assert row_actual == pytest.approx(row_expected, abs=tol)

    return _assert