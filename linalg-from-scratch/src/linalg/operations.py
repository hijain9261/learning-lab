from matrix import Matrix 
import math

def shape(self: Matrix) -> tuple:

    """return tuple Shape(rows, colms)"""
    return (self.rows, self.cols)

def transpose(self: Matrix) -> Matrix:
    result = [[self[j][i] for j in range(self.cols)] for i in range(self.rows)]
    return Matrix(result)

def matmul(self: Matrix, other: Matrix) -> Matrix:
    if not isinstance(other, Matrix):
        raise ValueError("Only operates on Matrices")
    
    if self.cols != other.rows:
        raise TypeError("Not Compatible for Matrix Multiplication")

    C = []
    for i in range(self.rows):
        result = []
        for j in range(other.cols):
            prod = 0
            for k in range(self.cols):
                prod = prod + self[i, k] * other[k, j]
            result.append(prod)
        C.append(result)
        del(result)   

    return Matrix(C)     

def matvec(A: Matrix, x: list) -> list:

    """Performing A.x (A matrix, x row vector)"""
    if A.cols != len(x):
        raise ValueError(
            f"Dimension mismatch: matrix columns ({A.cols}) != Vector length ({len(x)})"
        )

    x_cols = Matrix([[val] for val in x])
    return matmul(A, x_cols)

def identity(self: Matrix):

    # Checking if Square Matrix
    if self.rows != self.cols:
        raise ValueError("Not a Square Matrix")
    
    n = self.rows
    for i in range(self.rows):
        for j in range(self.cols):
            if i != j and self[i][j] != 0:
                return False
            
            if i == j and self[i][j] != 1:
                return False
    return True

def trace(self: Matrix):

    # Checking if Square Matrix
    if self.rows != self.cols:
        raise ValueError("Not a Square Matrix")

    trace = 0
    for i in range(self.rows):
        trace += self[i][i]
    
    return trace


def norm(self: Matrix, p=2):
    def _compute_l2_norm(self):
        m,n = shape(self)
        sum_of_squares = 0
        for i in range(self.rows):
            for j in range(self.cols):
                sum_of_squares += (self[i][j])**2
        return sum_of_squares ** 0.5
    
    def _compute_l1_norm(self):
        sum_abs = 0
        m,n = shape(self)
        for i in range(self.rows):
            for j in range(self.cols):
                sum_abs += abs(self[i][j])
        return sum_abs

    def _compute_inf_norm(self):
        max_val = abs(self[0][0])
        for i in range(self.rows):
            for j in range(self.cols):
                max_val = max(max_val, abs(self[i][j]))
        return max_val
    
    if p == 2 or p == "fro":
        return _compute_l2_norm(self)
    elif p == 1:
        return _compute_l1_norm(self)
    elif p == float("inf") or p == "inf":
        return _compute_inf_norm(self)
    else:
        raise ValueError(f"Unsupported norm parameter p={p}")


def vector_to_matrix(vector: list) -> Matrix:
    mat = [[round(v, 2)] for v in vector]
    return Matrix(mat)

def is_symmetric(self):
    return A == transpose(A)

def is_orthogonal(self):
    result = matmul(self, transpose(self))
    return identity(result)
    
def gaussian_elimination(A, b):

    if A.rows != b.rows:
        raise ValueError(
            f"Row mismatch for augmentation: A has {A.rows} rows, b has {b.rows}"
            " rows."
        )

    aug_matrix = Matrix([list(A.data[i]) + list(b.data[i]) for i in range(A.rows)])
    def _apply_partial_pivoting(matrix, k):
        max_val = abs(matrix[k, k])
        max_row = k

        for r in range(k+1, matrix.rows):
            val = abs(matrix[r, k])
            if val > max_val:
                max_val = val
                max_row = r  
        
        if max_row != k:
            matrix.data[k], matrix.data[max_row] = (matrix.data[max_row], matrix.data[k])
        return matrix


    # forward elimination
    for i in range(aug_matrix.rows):
        aug_matrix = _apply_partial_pivoting(aug_matrix, i)

        if abs(aug_matrix[i, i]) < 1e-12:
            raise ValueError("Matrix is singular; no unique solution exists.")

        for j in range(i+1, aug_matrix.rows):
            fact = aug_matrix[j,i]/ aug_matrix[i, i]
            for col_idx in range(aug_matrix.cols):
                aug_matrix[j, col_idx] -= fact * aug_matrix[i, col_idx]

    # back substitution 
    m, n = shape(aug_matrix)
    x = [0] * (n-1)
    x[-1] = aug_matrix[m-1, n-1]/aug_matrix[m-1, n-2]
    for i in range(m-2, -1, -1):
        lc = 0
        for j in range(i+1, n-1):
            lc += x[j] * aug_matrix[i, j]
        x[i] = (aug_matrix[i, n-1] - lc)/aug_matrix[i, i]

    return vector_to_matrix(x)

def determinant(self: matrix) -> float:
    if self.rows != self.cols:
        raise ValueError("Matrix must be a square Matrix")
    aug_matrix = Matrix([list(self.data[i]) for i in range(A.rows)])
    total_swaps = 0
    def _apply_partial_pivoting(matrix, k):
        nonlocal total_swaps
        max_val = abs(matrix[k, k])
        max_row = k

        for r in range(k+1, matrix.rows):
            val = abs(matrix[r, k])
            if val > max_val:
                max_val = val
                max_row = r
        if max_row != k:
            total_swaps += 1
            matrix.data[k], matrix.data[max_row] = (matrix.data[max_row], matrix.data[k])
        return matrix

    # Creating an upper Triangular Matrix
    for i in range(aug_matrix.rows):
        aug_matrix = _apply_partial_pivoting(aug_matrix, i)
        if abs(aug_matrix[i, i]) < 1e-12:
            continue
        for j in range(i+1, aug_matrix.rows):
            fact = aug_matrix[j,i]/ aug_matrix[i, i]
            for col_idx in range(aug_matrix.cols):
                aug_matrix[j, col_idx] -= fact * aug_matrix[i, col_idx]

    # finding determinant = product of principle diagonal elements
    det = 1
    for i in range(aug_matrix.rows):
        det *= aug_matrix[i][i]
    
    return round(det * math.pow(-1, total_swaps), 2)

def inverse(self: Matrix) -> Matrix:
    if not isinstance(self, Matrix):
        raise TypeError("Content must be of Matrix Type")
    
    if self.rows != self.cols:
        raise ValueError(f"Matrix must be Square ({self.rows} x {self.cols})")
    
    # determinant condition
    if determinant(self) == 0:
        raise ValueError(f"Inverse doesn't exist because Matrix is Singular")

    def create_identity_mat(n: int):
        k = 0
        I_n = []
        while (k < n):
            temp = [0] * n 
            temp[k] = 1
            I_n.append(temp) 
            k += 1
        
        return Matrix(I_n)

    def column_slicing(self, j):
        v = []
        for iter in range(self.cols):
            v.append(self[iter, j])
        return vector_to_matrix(v)

    inverse_mat = []
    I_n = create_identity_mat(self.rows)
    for i in range(self.cols):
        b = column_slicing(I_n, i)
        x = gaussian_elimination(self, b)
        x.data = [row[0] for row in x.data]
        inverse_mat.append(x.data)
    
    return transpose(Matrix(inverse_mat))

def rank(self):
    aug_matrix = Matrix([list(self.data[i]) for i in range(A.rows)])
    def _apply_partial_pivoting(matrix, k):
        max_val = abs(matrix[k, k])
        max_row = k

        for r in range(k+1, matrix.rows):
            val = abs(matrix[r, k])
            if val > max_val:
                max_val = val
                max_row = r
        if max_row != k:
            matrix.data[k], matrix.data[max_row] = (matrix.data[max_row], matrix.data[k])
        return matrix

    # Creating an upper Triangular Matrix
    for i in range(aug_matrix.rows):

        if abs(aug_matrix[i, i]) < 1e-12:
            continue
        aug_matrix = _apply_partial_pivoting(aug_matrix, i)
        for j in range(i+1, aug_matrix.rows):
            fact = aug_matrix[j,i]/ aug_matrix[i, i]
            for col_idx in range(aug_matrix.cols):
                aug_matrix[j, col_idx] -= fact * aug_matrix[i, col_idx]

    # finding total pivots 
    rank = 0
    for i in range(aug_matrix.rows):
        if aug_matrix[i, i] != 0:
            rank += 1
    
    return rank


