import numpy as np


def GetCameraMatrixFromParameters(thetas, position, K):
    # thetas: [thetaX, thetaY, thetaZ] rotation angles on each axis.
    # position: [posX, posY, posZ] position of the camera.
    C = np.reshape(np.array(position), newshape=(3, 1))
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(thetas[0]), -np.sin(thetas[0])],
                   [0, np.sin(thetas[0]), np.cos(thetas[0])]])  # rotation over the X axis.
    Ry = np.array([[np.cos(thetas[1]), -np.sin(thetas[1]), 0],
                   [np.sin(thetas[1]), np.cos(thetas[1]), 0],
                   [0, 0, 1]])  # rotation over the Y axis.
    Rz = np.array([[np.cos(thetas[2]), -np.sin(thetas[2]), 0],
                   [0, 1, 0],
                   [0, np.sin(thetas[2]), np.cos(thetas[2])]])  # rotation over the Z axis.
    R = np.matmul(Rz, np.matmul(Ry, Rx))
    t = -np.matmul(R, C)
    Rt = np.concatenate([R, t], axis=1)
    P = np.matmul(K, Rt)
    return P


# Project points into cameras:
def ProjectPointsIntoCamera(P, X3D):
    expansion = np.ones(shape=(1, X3D.shape[1]), dtype=np.float32)
    X_hom = np.concatenate([X3D, expansion], axis=0)
    x_proj = np.matmul(P, X_hom)
    for i in range(x_proj.shape[1]):
        x_proj[:, i] = x_proj[:, i] / x_proj[2, i]
    return x_proj[:2, :]



if __name__ == '__main__':
    npoints = 1
    X = np.zeros(shape=(3, npoints), dtype=np.float32)
    X[1, 0] = 1
    # Define the cameras:
    f = 0.095
    x0 = 0
    y0 = 0
    K = np.array([[f, 0, x0],
                  [0, f, y0],
                  [0, 0, 1]])
    P1 = GetCameraMatrixFromParameters([np.pi, 0, 0], [0, 0, -5], K)
    P2 = GetCameraMatrixFromParameters([-np.pi * 0.25, 0, 0], [0, 2.5, -2.5], K)
    P3 = GetCameraMatrixFromParameters([np.pi * 0.5, 0, 0], [0, 5, 0], K)
    # Project points into cameras:
    x1 = ProjectPointsIntoCamera(P1, X)
    x2 = ProjectPointsIntoCamera(P2, X)
    x3 = ProjectPointsIntoCamera(P3, X)
    print('x1')
    print(x1)
    print('x2')
    print(x2)
    print('x3')
    print(x3)



