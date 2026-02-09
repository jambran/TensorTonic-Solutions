import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    A = np.array(A)
    num_rows, num_cols = A.shape
    B = np.zeros((num_cols, num_rows))

    for i in range(num_rows):
        for j in range(num_cols):
            B[j][i] = A[i][j]

    return B
