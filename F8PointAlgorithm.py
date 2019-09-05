import numpy as np
import ToolsP2

# Compute the fundamental matrix F using the 8-point algorithm.
# X1: (3, npoints). The coordinates of the points in one image, at least 8.
# X2: (3, npoints). The coordinates of the points in the second image, at least 8.
def EightPointAlgorithm(X1, X2):
    assert X1.shape == X2.shape, 'The shape of the two point sets should be equal.'
    _, npoints = X1.shape
    assert npoints >= 8, 'At least 8 points are needed to use the 8-point algorithm.'
    # Set last coordinate to 1.
    # Normalization.
    T1 = ToolsP2.Normalization(X1)
    X1norm = np.matmul(T1, X1)
    T2 = ToolsP2.Normalization(X2)
    X2norm = np.matmul(T2, X2)
    # Linear solution.
    A = np.ones(shape=(npoints, 9), dtype=np.float32)
    for i in range(npoints):
        A[i, 0] = X1norm[0, i] * X2norm[0, i]
        A[i, 1] = X1norm[0, i] * X2norm[1, i]
        A[i, 2] = X1norm[0, i]
        A[i, 3] = X1norm[1, i] * X2norm[0, i]
        A[i, 4] = X1norm[1, i] * X2norm[1, i]
        A[i, 5] = X1norm[1, i]
        A[i, 6] = X2norm[0, i]
        A[i, 7] = X2norm[1, i]
    U, s, Vt = np.linalg.svd(A)
    nullvector = np.array(Vt)[-1, :]
    Fnorm = np.array([[nullvector[0], nullvector[1], nullvector[2]],
                      [nullvector[3], nullvector[4], nullvector[5]],
                      [nullvector[6], nullvector[7], nullvector[8]]], dtype=np.float32)
    # Constraint enforcement.
    U, s, Vt = np.linalg.svd(Fnorm)
    D = np.diag(s)
    D[2, 2] = 0
    Fnorm = np.matmul(U, np.matmul(D, Vt))
    # Denormalization.
    F = np.matmul(np.transpose(T2), np.matmul(Fnorm, T1))
    return F


def EightPointAlgorithm2(x1, x2, additional_enforcement=False):
    assert x1.shape[0] == 2, 'The points should be 2 dimensional, not in homogeneous coordinates.'
    assert x1.shape == x2.shape, 'The shape of the two point sets should be equal.'
    _, npoints = x1.shape
    assert npoints >= 8, 'At least 8 points are needed to use the 8-point algorithm.'
    # Homogeneous coordinates:
    ones = np.ones(shape=(1, npoints), dtype=np.float32)
    x1_hom = np.concatenate([x1, ones], axis=0)
    x2_hom = np.concatenate([x2, ones], axis=0)
    # Normalization.
    T1 = ToolsP2.Normalization(x1_hom)
    x1norm = np.matmul(T1, x1_hom)
    T2 = ToolsP2.Normalization(x2_hom)
    x2norm = np.matmul(T2, x2_hom)
    # Linear solution.
    A = np.ones(shape=(npoints, 9), dtype=np.float32)
    for i in range(npoints):
        x = x1norm[0, i]
        y = x1norm[1, i]
        xp = x2norm[0, i]
        yp = x2norm[1, i]
        A[i, 0] = xp * x
        A[i, 1] = xp * y
        A[i, 2] = xp
        A[i, 3] = yp * x
        A[i, 4] = yp * y
        A[i, 5] = yp
        A[i, 6] = x
        A[i, 7] = y
        # A[i, 0] = x2norm[0, i] * x1norm[0, i]
        # A[i, 1] = x2norm[0, i] * x1norm[1, i]
        # A[i, 2] = x2norm[0, i]
        # A[i, 3] = x2norm[1, i] * x1norm[0, i]
        # A[i, 4] = x2norm[1, i] * x1norm[1, i]
        # A[i, 5] = x1norm[0, i]
        # A[i, 6] = x1norm[1, i]
        # A[i, 7] = 1
        # A[i, 0] = x1norm[0, i] * x2norm[0, i]
        # A[i, 1] = x1norm[0, i] * x2norm[1, i]
        # A[i, 2] = x1norm[0, i]
        # A[i, 3] = x1norm[1, i] * x2norm[0, i]
        # A[i, 4] = x1norm[1, i] * x2norm[1, i]
        # A[i, 5] = x1norm[1, i]
        # A[i, 6] = x2norm[0, i]
        # A[i, 7] = x2norm[1, i]
    U, s, Vt = np.linalg.svd(A)
    nullvector = np.array(Vt)[-1, :]
    Fnorm = np.array([[nullvector[0], nullvector[1], nullvector[2]],
                      [nullvector[3], nullvector[4], nullvector[5]],
                      [nullvector[6], nullvector[7], nullvector[8]]], dtype=np.float32)
    # Constraint enforcement.
    U, s, Vt = np.linalg.svd(Fnorm)
    D = np.diag(s)
    D[2, 2] = 0
    Fnorm = np.matmul(U, np.matmul(D, Vt))
    # Denormalization.
    F = np.matmul(np.transpose(T2), np.matmul(Fnorm, T1))
    # Additional constraint enforcement.
    if additional_enforcement:
        U, s, Vt = np.linalg.svd(F)
        D = np.diag(s)
        D[2, 2] = 0
        F = np.matmul(U, np.matmul(D, Vt))
    # Set last element to 1:
    F = F / F[2, 2]
    return F

