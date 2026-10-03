import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from linalg import Matrix
from linalg.operations import (
    determinant,
    inverse,
    is_orthogonal,
    is_symmetric,
    matmul,
    norm,
    rank,
    shape,
    trace,
    transpose,
    inverse
)


def section(title):
    print(f"\n=== {title} ===")


def tidy(matrix, digits=4):
    return Matrix([[round(v, digits) + 0.0 for v in row] for row in matrix.data])


def main():
    A = Matrix([[2, 1], [1, 3]])
    B = Matrix([[0, 1], [1, 0]])
    R = Matrix([[1, 2, 3], [4, 5, 6]])

    section("Creating matrices")
    print("A = ")
    print(A)
    print("B = ")
    print(B)
    print("R = ")
    print(R)
    print("shape of R:", shape(R))

    section("Element access")
    print("A[0, 1] =", A[0, 1])
    print("first row of R:", R[0])

    section("Arithmetic")
    print("A + B = ")
    print(A + B)
    print("A - B = ")
    print(A - B)
    print("3 * A = ")
    print(3 * A)

    section("Matrix multiplication")
    print("A x B = ")
    print(matmul(A, B))
    print("R x transpose(R) = ")
    print(matmul(R, transpose(R)))

    section("Transpose")
    print("transpose(R)")
    print(transpose(R))
    print("shape after transpose:", shape(transpose(R)))

    section("Scalar properties")
    print("trace(A) =", trace(A))
    print("determinant(A) =", determinant(A))
    print("rank(R) =", rank(R))

    section("Norms")
    print("Frobenius norm of A:", round(norm(A), 4))
    print("L1 norm of A:", norm(A, 1))
    print("Infinity norm of A:", norm(A, "inf"))

    section("Inverse")
    A_inv = inverse(A)
    print("inverse(A) =")
    print(A_inv)
    print("A x inverse(A) =")
    print(tidy(matmul(A, A_inv)))

    section("Structure checks")
    print("A symmetric:", is_symmetric(A))
    print("R symmetric:", is_symmetric(R))
    print("B orthogonal:", is_orthogonal(B))
    print("A orthogonal:", is_orthogonal(A))


if __name__ == "__main__":
    main()
