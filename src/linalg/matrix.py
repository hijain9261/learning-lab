class Matrix: 

    # param constructor 
    def __init__(self, data):
        # Ensure input is list and not empty
        if not isinstance(data, list) or not data:
            raise ValueError("Matric data must be a non-empty list of lists")

        # Checking 2D list capability
        if not isinstance(data[0], list):
            raise ValueError("Matrix data must be a 2D list of lists.") 

        self.rows = len(data)
        self.cols = len(data[0])

        for row in data:
            if not isinstance(row, list):
                raise ValueError("All rows must be list")

            if len(row) != self.cols:
                raise ValueError("All rows must have same length (rectangular)")

        # final self.data initilization
        self.data = [[float(val) for val in row] for row in data] 
    
    # String form of matrix
    def __str__(self):
        if not self.data:
            return "[]"
        
        # Converting each row to string
        row_string = [str(row) for row in self.data]

        # Intended Nice Output representation
        return "[" + ",\n ".join(row_string) + "]"

    # Representation of matrix
    def __repr__(self):
        return f"Matrix({self.data})"


    # 1D, 2D indexing 
    def __getitem__(self, index):
        if isinstance(index, tuple):
            i, j = index
            return self.data[i][j]
        
        return self.data[index]
    
    # Set Item 
    def __setitem__(self, index, value):
        if isinstance(index, int):

            # Only added when Value is a List of length equal to col_length
            if isinstance(value, list) and len(value) == self.cols:
                self.data[index] = [float(v) for v in value] 
            else:
                print("Not compatible")

        if isinstance(index, tuple):
            i, j = index
            self.data[i][j] = float(value)

    # Add two matrix 
    def __add__(self, other):
        if not isinstance(other, Matrix):
            raise TypeError("Can only add another Matrix instance")
        
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError(
                f"Dimension mismatch for addition: ({self.rows}x{self.cols}) vs ({other.rows}x{other.cols})"
            )

        new_data = [
            [a+b for a, b in zip(row_a, row_b)]
            for row_a, row_b in zip(self.data, other.data)
        ]

        return Matrix(new_data)

    # Sub two matrix 
    def __sub__(self, other):
        if not isinstance(other, Matrix):
            raise TypeError("Can only add another Matrix instance")
        
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError(
                f"Dimension mismatch for addition: ({self.rows}x{self.cols}) vs ({other.rows}x{other.cols})"
            )

        new_data = [
            [a-b for a, b in zip(row_a, row_b)]
            for row_a, row_b in zip(self.data, other.data)
        ]

        return Matrix(new_data)

    
    # left_multiplication A.c (scalar only)
    def __mul__(self, scalar):
        if not isinstance(scalar, (int, float)):
            raise TypeError("Can Only Multiply Matrix by a scalar")
        
        new_data = [[val * scalar for val in row] for row in self.data]
        return Matrix(new_data)


    # right_multiplication (c.A)
    def __rmul__(self, scalar):
        return self.__mul__(scalar)


    # equal check
    def __eq__(self, other):
        if not isinstance(other, Matrix):
            return False
        
        if self.rows != other.rows or self.cols != other.cols:
            return False
        
        for row_a, row_b in zip(self.data, other.data):
            for val_a, val_b in zip(row_a, row_b):
                if val_a != val_b:
                    return False
        
        return True