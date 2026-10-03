import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


import math
import pytest

from linalg import Matrix
from linalg.operations import (
    determinant,
    gaussian_elimination,
    identity,
    inverse,
    is_orthogonal,
    is_symmetric,
    matmul,
    matvec,
    norm,
    rank,
    shape,
    trace,
    transpose,
    vector_to_matrix,
)


class TestShape:
    def test_square(self):
        assert shape(Matrix([[1, 2], [3, 4]])) == (2, 2)

    def test_rectangular(self):
        assert shape(Matrix([[1, 2, 3], [4, 5, 6]])) == (2, 3)


class TestTranspose:
    def test_square(self):
        assert transpose(Matrix([[1, 2], [3, 4]])) == Matrix([[1, 3], [2, 4]])

    def test_rectangular(self):
        result = transpose(Matrix([[1, 2, 3], [4, 5, 6]]))
        assert result == Matrix([[1, 4], [2, 5], [3, 6]])
        assert shape(result) == (3, 2)

    def test_double_transpose_returns_original(self):
        m = Matrix([[1, 2, 3], [4, 5, 6]])
        assert transpose(transpose(m)) == m


class TestMatmul:
    def test_square_product(self):
        a = Matrix([[1, 2], [3, 4]])
        b = Matrix([[5, 6], [7, 8]])
        assert matmul(a, b) == Matrix([[19, 22], [43, 50]])

    def test_rectangular_product(self):
        a = Matrix([[1, 2, 3], [4, 5, 6]])
        b = Matrix([[7, 8], [9, 10], [11, 12]])
        assert matmul(a, b) == Matrix([[58, 64], [139, 154]])

    def test_identity_is_neutral(self):
        a = Matrix([[1, 2], [3, 4]])
        eye = Matrix([[1, 0], [0, 1]])
        assert matmul(a, eye) == a
        assert matmul(eye, a) == a

    def test_not_commutative(self):
        a = Matrix([[1, 2], [3, 4]])
        b = Matrix([[0, 1], [1, 0]])
        assert matmul(a, b) != matmul(b, a)

    def test_incompatible_dimensions_raise(self):
        with pytest.raises(TypeError):
            matmul(Matrix([[1, 2, 3]]), Matrix([[1, 2, 3]]))

    def test_non_matrix_operand_raises(self):
        with pytest.raises(ValueError):
            matmul(Matrix([[1, 2]]), [[1], [2]])


class TestMatvec:
    def test_multiplies_column_vector(self):
        result = matvec(Matrix([[1, 2], [3, 4]]), [1, 1])
        assert result == Matrix([[3], [7]])

    def test_rectangular_matrix(self):
        result = matvec(Matrix([[1, 0, 2], [0, 1, 3]]), [1, 2, 3])
        assert result == Matrix([[7], [11]])

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="Dimension mismatch"):
            matvec(Matrix([[1, 2], [3, 4]]), [1, 2, 3])


class TestIdentityCheck:
    def test_identity_matrix(self):
        assert identity(Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])) is True

    def test_non_identity_diagonal(self):
        assert identity(Matrix([[1, 0], [0, 2]])) is False

    def test_non_zero_off_diagonal(self):
        assert identity(Matrix([[1, 1], [0, 1]])) is False

    def test_non_square_raises(self):
        with pytest.raises(ValueError, match="Square"):
            identity(Matrix([[1, 0, 0], [0, 1, 0]]))


class TestTrace:
    def test_sum_of_diagonal(self):
        assert trace(Matrix([[1, 2], [3, 4]])) == 5.0

    def test_three_by_three(self):
        assert trace(Matrix([[2, 0, 0], [0, 3, 0], [0, 0, 4]])) == 9.0

    def test_non_square_raises(self):
        with pytest.raises(ValueError, match="Square"):
            trace(Matrix([[1, 2, 3], [4, 5, 6]]))


class TestNorm:
    def test_default_is_frobenius(self):
        assert norm(Matrix([[3, 4]])) == pytest.approx(5.0)

    def test_frobenius_string_alias(self):
        m = Matrix([[1, 2], [3, 4]])
        assert norm(m, "fro") == pytest.approx(math.sqrt(30))
        assert norm(m, "fro") == norm(m, 2)

    def test_l1_sums_absolute_values(self):
        assert norm(Matrix([[1, -2], [3, -4]]), 1) == pytest.approx(10.0)

    @pytest.mark.parametrize("p", [float("inf"), "inf"])
    def test_inf_returns_largest_absolute_value(self, p):
        assert norm(Matrix([[1, -7], [3, 4]]), p) == pytest.approx(7.0)

    def test_unsupported_p_raises(self):
        with pytest.raises(ValueError, match="Unsupported"):
            norm(Matrix([[1, 2]]), 3)


class TestVectorToMatrix:
    def test_builds_column_matrix(self):
        assert vector_to_matrix([1, 2, 3]) == Matrix([[1], [2], [3]])

    def test_rounds_to_two_decimals(self):
        assert vector_to_matrix([1.234, 5.678]) == Matrix([[1.23], [5.68]])


class TestSymmetry:
    def test_symmetric(self):
        assert is_symmetric(Matrix([[1, 2], [2, 1]])) is True

    def test_not_symmetric(self):
        assert is_symmetric(Matrix([[1, 2], [3, 4]])) is False

    def test_identity_is_symmetric(self):
        assert is_symmetric(Matrix([[1, 0], [0, 1]])) is True


class TestOrthogonality:
    def test_identity_is_orthogonal(self):
        assert is_orthogonal(Matrix([[1, 0], [0, 1]])) is True

    def test_quarter_turn_rotation(self):
        assert is_orthogonal(Matrix([[0, -1], [1, 0]])) is True

    def test_permutation_matrix(self):
        assert is_orthogonal(Matrix([[0, 1], [1, 0]])) is True

    def test_shear_is_not_orthogonal(self):
        assert is_orthogonal(Matrix([[1, 1], [0, 1]])) is False


class TestGaussianElimination:
    def test_two_by_two_system(self, assert_matrix_close):
        A = Matrix([[2, 1], [1, 3]])
        b = Matrix([[3], [5]])
        assert_matrix_close(gaussian_elimination(A, b), [[0.8], [1.4]])

    def test_three_by_three_system(self, assert_matrix_close):
        A = Matrix([[2, 1, -1], [-3, -1, 2], [-2, 1, 2]])
        b = Matrix([[8], [-11], [-3]])
        assert_matrix_close(gaussian_elimination(A, b), [[2], [3], [-1]])

    def test_zero_on_diagonal_needs_pivoting(self, assert_matrix_close):
        A = Matrix([[0, 1], [1, 0]])
        b = Matrix([[2], [3]])
        assert_matrix_close(gaussian_elimination(A, b), [[3], [2]])

    def test_singular_matrix_raises(self):
        with pytest.raises(ValueError, match="singular"):
            gaussian_elimination(Matrix([[1, 2], [2, 4]]), Matrix([[1], [2]]))

    def test_row_mismatch_raises(self):
        with pytest.raises(ValueError, match="Row mismatch"):
            gaussian_elimination(Matrix([[1, 0], [0, 1]]), Matrix([[1], [2], [3]]))

    def test_inputs_are_not_modified(self):
        A = Matrix([[2, 1], [1, 3]])
        b = Matrix([[3], [5]])
        gaussian_elimination(A, b)
        assert A == Matrix([[2, 1], [1, 3]])
        assert b == Matrix([[3], [5]])


class TestDeterminant:
    def test_two_by_two(self):
        assert determinant(Matrix([[1, 2], [3, 4]])) == pytest.approx(-2.0)

    def test_three_by_three(self):
        assert determinant(Matrix([[6, 1, 1], [4, -2, 5], [2, 8, 7]])) == pytest.approx(-306.0)

    def test_identity(self):
        assert determinant(Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])) == pytest.approx(1.0)

    def test_row_swap_flips_sign(self):
        assert determinant(Matrix([[0, 1], [1, 0]])) == pytest.approx(-1.0)

    def test_singular_matrix_is_zero(self):
        assert determinant(Matrix([[1, 2], [2, 4]])) == pytest.approx(0.0)

    def test_non_square_raises(self):
        with pytest.raises(ValueError, match="square"):
            determinant(Matrix([[1, 2, 3], [4, 5, 6]]))

    def test_input_is_not_modified(self):
        m = Matrix([[1, 2], [3, 4]])
        determinant(m)
        assert m == Matrix([[1, 2], [3, 4]])


class TestInverse:
    def test_two_by_two(self, assert_matrix_close):
        result = inverse(Matrix([[4, 7], [2, 6]]))
        assert_matrix_close(result, [[0.6, -0.7], [-0.2, 0.4]])

    def test_three_by_three(self, assert_matrix_close):
        result = inverse(Matrix([[1, 2, 3], [0, 1, 4], [5, 6, 0]]))
        assert_matrix_close(result, [[-24, 18, 5], [20, -15, -4], [-5, 4, 1]])

    def test_product_with_original_is_identity(self, assert_matrix_close):
        A = Matrix([[4, 7], [2, 6]])
        assert_matrix_close(matmul(A, inverse(A)), [[1, 0], [0, 1]], tol=1e-6)

    def test_singular_matrix_raises(self):
        with pytest.raises(ValueError, match="Singular"):
            inverse(Matrix([[1, 2], [2, 4]]))

    def test_non_square_raises(self):
        with pytest.raises(ValueError, match="Square"):
            inverse(Matrix([[1, 2, 3], [4, 5, 6]]))

    def test_non_matrix_raises(self):
        with pytest.raises(TypeError):
            inverse([[1, 0], [0, 1]])


class TestRank:
    def test_full_rank_identity(self):
        assert rank(Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])) == 3

    def test_rank_deficient_square(self):
        assert rank(Matrix([[1, 2], [2, 4]])) == 1

    def test_classic_rank_two_three_by_three(self):
        assert rank(Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])) == 2

    def test_zero_matrix(self):
        assert rank(Matrix([[0, 0], [0, 0]])) == 0

    def test_wide_matrix(self):
        assert rank(Matrix([[1, 2, 3], [4, 5, 6]])) == 2

    def test_tall_matrix(self):
        assert rank(Matrix([[1, 2], [3, 4], [5, 6]])) == 2

    def test_input_is_not_modified(self):
        m = Matrix([[1, 2], [2, 4]])
        rank(m)
        assert m == Matrix([[1, 2], [2, 4]])