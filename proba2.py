import numpy as np

def ProjectPointsIntoCamera(P, X):
    expansion = np.ones(shape=(1, X.shape[1]), dtype=np.float32)
    X_hom = np.concatenate([X, expansion], axis=0)
    x = np.matmul(P, X_hom)
    for i in range(x.shape[1]):
        x[:, i] = x[:, i] / x[2, i]
    return x[:2, :]

def GetCameraMatrixFromParameters(theta, position, K):
    C = np.reshape(np.array(position), newshape=(3, 1))
    R = np.array([[np.cos(theta), -np.sin(theta), 0],
                   [np.sin(theta), np.cos(theta), 0],
                   [0, 0, 1]])  # rotation over the Z axis.
    t = -np.matmul(R, C)
    Rt = np.concatenate([R, t], axis=1)
    P = np.matmul(K, Rt)
    return P


npoints = 1
X = np.zeros(shape=(3, npoints), dtype=np.float32)
X[1, 0] = 1

# Camera parameters:
K = np.array([[0.5, 0, 0],
              [0, 0.5, 0],
              [0, 0, 1]])
P = GetCameraMatrixFromParameters(0, [0, 0, -5], K)

x = ProjectPointsIntoCamera(P, X)



