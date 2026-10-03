import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
from linalg import Matrix
from linalg.operations import (
    determinant,
    gaussian_elimination,
    inverse,
    is_orthogonal,
    is_symmetric,
    matmul,
    norm,
    rank,
    shape,
    trace,
    transpose,
)


class TestSingleElementMatrix:
    @pytest.fixture
    def m(self):
        return Matrix([[4]])

    def test_shape(self, m):
        assert shape(m) == (1, 1)

    def test_transpose(self, m):
        assert transpose(m) == m

    def test_trace(self, m):
        assert trace(m) == 4.0

    def test_determinant(self, m):
        assert determinant(m) == pytest.approx(4.0)

    def test_inverse(self, m):
        assert inverse(m) == Matrix([[0.25]])

    def test_rank(self, m):
        assert rank(m) == 1

    def test_norm(self, m):
        assert norm(m) == pytest.approx(4.0)

    def test_solve(self):
        result = gaussian_elimination(Matrix([[2]]), Matrix([[6]]))
        assert result == Matrix([[3]])

    def test_zero_element_is_singular(self):
        zero = Matrix([[0]])
        assert rank(zero) == 0
        with pytest.raises(ValueError):
            inverse(zero)


class TestZeroMatrix:
    @pytest.fixture
    def zero(self):
        return Matrix([[0, 0], [0, 0]])

    def test_norm_is_zero(self, zero):
        assert norm(zero) == 0.0

    def test_rank_is_zero(self, zero):
        assert rank(zero) == 0

    def test_determinant_is_zero(self, zero):
        assert determinant(zero) == pytest.approx(0.0)

    def test_inverse_raises(self, zero):
        with pytest.raises(ValueError, match="Singular"):
            inverse(zero)

    def test_adding_zero_changes_nothing(self, zero):
        a = Matrix([[1, 2], [3, 4]])
        assert a + zero == a


class TestVectors:
    def test_row_vector_transpose_is_column(self):
        row = Matrix([[1, 2, 3]])
        assert shape(transpose(row)) == (3, 1)

    def test_row_times_column_is_one_by_one(self):
        row = Matrix([[1, 2, 3]])
        col = Matrix([[4], [5], [6]])
        assert matmul(row, col) == Matrix([[32]])

    def test_column_times_row_is_outer_product(self):
        col = Matrix([[1], [2]])
        row = Matrix([[3, 4]])
        assert matmul(col, row) == Matrix([[3, 4], [6, 8]])

    def test_column_vector_is_not_square(self):
        with pytest.raises(ValueError):
            trace(Matrix([[1], [2]]))


class TestRectangularMatrices:
    def test_product_shape(self):
        a = Matrix([[1, 2, 3], [4, 5, 6]])
        b = Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
        assert shape(matmul(a, b)) == (2, 4)

    def test_transpose_of_product_reverses_order(self):
        a = Matrix([[1, 2, 3], [4, 5, 6]])
        b = Matrix([[7, 8], [9, 10], [11, 12]])
        assert transpose(matmul(a, b)) == matmul(transpose(b), transpose(a))

    def test_non_square_is_not_symmetric(self):
        assert is_symmetric(Matrix([[1, 2, 3], [4, 5, 6]])) is False

    @pytest.mark.parametrize("func", [determinant, inverse, trace])
    def test_square_only_operations_reject_rectangular(self, func):
        with pytest.raises(ValueError):
            func(Matrix([[1, 2, 3], [4, 5, 6]]))


class TestNumericalBehaviour:
    def test_negative_and_fractional_entries(self):
        assert determinant(Matrix([[-1, -2], [-3, -4]])) == pytest.approx(-2.0)
        assert determinant(Matrix([[0.5, 0], [0, 0.5]])) == pytest.approx(0.25)

    def test_large_entries(self):
        m = Matrix([[1000, 0, 0], [0, 1000, 0], [0, 0, 1000]])
        assert determinant(m) == pytest.approx(1e9)

    def test_tiny_pivot_is_handled_by_partial_pivoting(self):
        A = Matrix([[1e-13, 1], [1, 1]])
        b = Matrix([[1], [2]])
        assert gaussian_elimination(A, b) == Matrix([[1], [1]])

    def test_rank_ignores_floating_point_noise(self):
        assert rank(Matrix([[0.1, 0.2], [0.3, 0.6]])) == 1

    def test_rank_deficient_matrix_has_zero_determinant(self):
        assert determinant(Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])) == pytest.approx(0.0)

    def test_rank_deficient_matrix_has_no_inverse(self):
        with pytest.raises(ValueError, match="Singular"):
            inverse(Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))


class TestPermutationMatrices:
    def test_swap_is_orthogonal_and_symmetric(self):
        swap = Matrix([[0, 1], [1, 0]])
        assert is_orthogonal(swap) is True
        assert is_symmetric(swap) is True

    def test_cyclic_permutation_is_orthogonal_but_not_symmetric(self):
        cycle = Matrix([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
        assert is_orthogonal(cycle) is True
        assert is_symmetric(cycle) is False
        assert determinant(cycle) == pytest.approx(1.0)

    def test_orthogonal_inverse_equals_transpose(self):
        rotation = Matrix([[0, -1], [1, 0]])
        assert inverse(rotation) == transpose(rotation)


class TestInputsStayUntouched:
    def test_operations_do_not_mutate_their_arguments(self):
        original = [[2, 1, 1], [1, 3, 2], [1, 0, 0]]
        m = Matrix(original)
        determinant(m)
        rank(m)
        inverse(m)
        transpose(m)
        assert m == Matrix(original)
