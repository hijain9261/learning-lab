import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from linalg import Matrix
from linalg.operations import (
    determinant,
    gaussian_elimination,
    inverse,
    matmul,
    matvec,
    rank,
)


def column_to_list(column):
    return [row[0] for row in column.data]


def tidy(matrix, digits=4):
    return Matrix([[round(v, digits) + 0.0 for v in row] for row in matrix.data])


def solve_with_elimination():
    print("=== Solving Ax = b with Gaussian elimination ===")
    A = Matrix([[2, 1, -1], [-3, -1, 2], [-2, 1, 2]])
    b = Matrix([[8], [-11], [-3]])

    print("A =")
    print(A)
    # A =
    # [[2.0, 1.0, -1.0],
    #  [-3.0, -1.0, 2.0],
    #  [-2.0, 1.0, 2.0]]
    print("b =")
    print(b)
    # b =
    # [[8.0],
    #  [-11.0],
    #  [-3.0]]

    x = gaussian_elimination(A, b)
    print("x =")
    print(x)
    # x =
    # [[2.0],
    #  [3.0],
    #  [-1.0]]

    residual = matvec(A, column_to_list(x)) - b
    print("\nresidual (Ax - b) =")
    print(tidy(residual))
    # residual (Ax - b) =
    # [[0.0],
    #  [0.0],
    #  [0.0]]


def solve_with_inverse():
    print("\n=== Solving Ax = b with the inverse ===")
    A = Matrix([[4, 7], [2, 6]])
    b = Matrix([[1], [2]])

    x = tidy(matmul(inverse(A), b))
    print("inverse(A) x b =")
    print(x)
    # inverse(A) x b =
    # [[-0.8],
    #  [0.6]]

    print("Gaussian elimination gives:")
    print(gaussian_elimination(A, b))
    # Gaussian elimination gives:
    # [[-0.8],
    #  [0.6]]


def solve_needing_row_swap():
    print("\n=== System with a zero in the pivot position ===")
    A = Matrix([[0, 1], [1, 0]])
    b = Matrix([[2], [3]])

    print("A =")
    print(A)
    # A =
    # [[0.0, 1.0],
    #  [1.0, 0.0]]
    print("b = ")
    print(b)
    print("x =")
    print(gaussian_elimination(A, b))
    # x =
    # [[3.0],
    #  [2.0]]


def solve_singular_system():
    print("\n=== Singular system ===")
    A = Matrix([[1, 2], [2, 4]])
    b = Matrix([[1], [2]])
    print("A = ")
    print(A)
    print("determinant(A) =", determinant(A) + 0.0)
    # determinant(A) = 0.0
    print("rank(A) =", rank(A))
    # rank(A) = 1

    try:
        gaussian_elimination(A, b)
    except ValueError as error:
        print("Could not solve:", error)
        # Could not solve: Matrix is singular; no unique solution exists.


def main():
    solve_with_elimination()
    solve_with_inverse()
    solve_needing_row_swap()
    solve_singular_system()


if __name__ == "__main__":
    main()