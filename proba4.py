import numpy as np
from MakeUpGeometry import ProjectPointsIntoCamera, GetCameraMatrixFromParameters

npoints = 8
# Random points, with coordinates between -1 and 1:
X = np.zeros(shape=(3, npoints), dtype=np.float32)
for i in range(npoints):
    X[0, i] = np.random.rand() * 2 - 1
    X[1, i] = np.random.rand() * 2 - 1
    X[2, i] = np.random.rand() * 2 - 1

# Define the cameras:
# Internal parameters:
f = 0.5
x0 = 0.5
y0 = 0.5
K = np.array([[f, 0, x0],
              [0, f, y0],
              [0, 0, 1]])

P1 = GetCameraMatrixFromParameters([np.pi, 0, 0], [0, 0, -2], K)
P2 = GetCameraMatrixFromParameters([-np.pi * 0.25, 0, 0], [0, np.sqrt(2), -np.sqrt(2)], K)
P3 = GetCameraMatrixFromParameters([np.pi * 0.5, 0, 0], [0, 2, 0], K)

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



