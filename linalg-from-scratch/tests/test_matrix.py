import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
from linalg import Matrix


class TestConstruction:
    def test_stores_dimensions(self):
        m = Matrix([[1, 2, 3], [4, 5, 6]])
        assert m.rows == 2
        assert m.cols == 3

    def test_converts_entries_to_float(self):
        m = Matrix([[1, 2], [3, 4]])
        assert m.data == [[1.0, 2.0], [3.0, 4.0]]
        assert all(isinstance(v, float) for row in m.data for v in row)

    def test_does_not_alias_input_list(self):
        raw = [[1, 2], [3, 4]]
        m = Matrix(raw)
        raw[0][0] = 99
        assert m[0, 0] == 1.0

    @pytest.mark.parametrize("bad_input", [[], "abc", 5, None, (1, 2), [1, 2, 3], [(1, 2), (3, 4)]])
    def test_rejects_invalid_input(self, bad_input):
        with pytest.raises(ValueError):
            Matrix(bad_input)

    def test_rejects_ragged_rows(self):
        with pytest.raises(ValueError, match="same length"):
            Matrix([[1, 2], [3]])

    def test_rejects_non_list_row(self):
        with pytest.raises(ValueError, match="must be list"):
            Matrix([[1, 2], (3, 4)])


class TestIndexing:
    @pytest.fixture
    def m(self):
        return Matrix([[1, 2, 3], [4, 5, 6]])

    def test_tuple_index_returns_element(self, m):
        assert m[1, 2] == 6.0

    def test_single_index_returns_row(self, m):
        assert m[0] == [1.0, 2.0, 3.0]

    def test_chained_index_returns_element(self, m):
        assert m[1][0] == 4.0

    def test_negative_index(self, m):
        assert m[-1, -1] == 6.0

    @pytest.mark.parametrize("index", [(2, 0), (0, 3), 5])
    def test_out_of_range_raises(self, m, index):
        with pytest.raises(IndexError):
            m[index]


class TestAssignment:
    @pytest.fixture
    def m(self):
        return Matrix([[1, 2], [3, 4]])

    def test_set_single_element(self, m):
        m[0, 1] = 9
        assert m[0, 1] == 9.0
        assert isinstance(m[0, 1], float)

    def test_set_whole_row(self, m):
        m[1] = [7, 8]
        assert m.data[1] == [7.0, 8.0]

    def test_row_of_wrong_length_is_ignored(self, m):
        m[0] = [1, 2, 3]
        assert m.data == [[1.0, 2.0], [3.0, 4.0]]

    def test_non_list_row_is_ignored(self, m):
        m[0] = 5
        assert m.data == [[1.0, 2.0], [3.0, 4.0]]


class TestAddition:
    def test_adds_elementwise(self):
        a = Matrix([[1, 2], [3, 4]])
        b = Matrix([[5, 6], [7, 8]])
        assert a + b == Matrix([[6, 8], [10, 12]])

    def test_is_commutative(self):
        a = Matrix([[1, 2], [3, 4]])
        b = Matrix([[0, -1], [2.5, 1]])
        assert a + b == b + a

    def test_does_not_modify_operands(self):
        a = Matrix([[1, 2], [3, 4]])
        b = Matrix([[1, 1], [1, 1]])
        a + b
        assert a.data == [[1.0, 2.0], [3.0, 4.0]]
        assert b.data == [[1.0, 1.0], [1.0, 1.0]]

    def test_dimension_mismatch_raises(self):
        with pytest.raises(ValueError, match="Dimension mismatch"):
            Matrix([[1, 2]]) + Matrix([[1], [2]])

    def test_non_matrix_raises(self):
        with pytest.raises(TypeError):
            Matrix([[1, 2]]) + [[1, 2]]


class TestSubtraction:
    def test_subtracts_elementwise(self):
        a = Matrix([[5, 6], [7, 8]])
        b = Matrix([[1, 2], [3, 4]])
        assert a - b == Matrix([[4, 4], [4, 4]])

    def test_subtracting_self_gives_zero_matrix(self):
        a = Matrix([[1, -2], [3.5, 4]])
        assert a - a == Matrix([[0, 0], [0, 0]])

    def test_dimension_mismatch_raises(self):
        with pytest.raises(ValueError):
            Matrix([[1, 2, 3]]) - Matrix([[1, 2]])

    def test_non_matrix_raises(self):
        with pytest.raises(TypeError):
            Matrix([[1, 2]]) - 3


class TestScalarMultiplication:
    def test_multiply_by_int(self):
        assert Matrix([[1, 2], [3, 4]]) * 2 == Matrix([[2, 4], [6, 8]])

    def test_multiply_by_float(self):
        assert Matrix([[2, 4]]) * 0.5 == Matrix([[1, 2]])

    def test_scalar_on_the_left(self):
        assert 3 * Matrix([[1, 2], [3, 4]]) == Matrix([[3, 6], [9, 12]])

    def test_multiply_by_zero(self):
        assert Matrix([[1, 2], [3, 4]]) * 0 == Matrix([[0, 0], [0, 0]])

    def test_multiply_by_negative(self):
        assert Matrix([[1, -2]]) * -1 == Matrix([[-1, 2]])

    @pytest.mark.parametrize("bad", ["2", [1, 2], None, Matrix([[1, 2]])])
    def test_non_scalar_raises(self, bad):
        with pytest.raises(TypeError):
            Matrix([[1, 2]]) * bad


class TestEquality:
    def test_equal_matrices(self):
        assert Matrix([[1, 2], [3, 4]]) == Matrix([[1, 2], [3, 4]])

    def test_different_values(self):
        assert Matrix([[1, 2], [3, 4]]) != Matrix([[1, 2], [3, 5]])

    def test_different_shapes(self):
        assert Matrix([[1, 2]]) != Matrix([[1], [2]])

    def test_int_and_float_entries_compare_equal(self):
        assert Matrix([[1, 2]]) == Matrix([[1.0, 2.0]])

    @pytest.mark.parametrize("other", [[[1, 2]], "matrix", None, 5])
    def test_non_matrix_is_never_equal(self, other):
        assert Matrix([[1, 2]]) != other


class TestStringRepresentation:
    def test_str_multi_row(self):
        assert str(Matrix([[1, 2], [3, 4]])) == "[[1.0, 2.0],\n [3.0, 4.0]]"

    def test_str_single_row(self):
        assert str(Matrix([[1, 2]])) == "[[1.0, 2.0]]"

    def test_repr(self):
        assert repr(Matrix([[1, 2], [3, 4]])) == "Matrix([[1.0, 2.0], [3.0, 4.0]])"