import numpy as np
from MakeUpGeometry import ProjectPointsIntoCamera, GetCameraMatrixFromParameters
# from FundamentalMatrixComputation import ComputeF
from F8PointAlgorithm import EightPointAlgorithm2
import cv2

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


#########################################################################################
#########################################################################################

ones = np.ones(shape=(1, npoints), dtype=np.float32)
x1_hom = np.concatenate([x1, ones], axis=0)
x2_hom = np.concatenate([x2, ones], axis=0)
x3_hom = np.concatenate([x3, ones], axis=0)

# myF = EightPointAlgorithm2(x1, x2, additional_enforcement=False)
# myF = EightPointAlgorithm2(x1, x2, additional_enforcement=True)
res = cv2.findFundamentalMat(np.transpose(x1), np.transpose(x2), method=cv2.FM_8POINT)
F = res[0]
print('F')
print(F)

E = np.matmul(np.transpose(K), np.matmul(F, K))
print('E')
print(E)

U, s, Vt = np.linalg.svd(E)
print('s')
print(s)

print('Dividing by first singular value')

E = E / s[0]
print('E')
print(E)
U, s, Vt = np.linalg.svd(E)
print('s')
print(s)

Z = np.array([[0, 1, 0],
              [-1, 0, 0],
              [0, 0, 0]])
W = np.array([[0, -1, 0],
              [1, 0, 0],
              [0, 0, 1]])

S = np.matmul(U, np.matmul(Z, np.transpose(U)))
R1 = np.matmul(U, np.matmul(W, Vt))
R2 = np.matmul(U, np.matmul(np.transpose(W), Vt))

print('np.matmul(S, R1)')
print(np.matmul(S, R1))
print('np.matmul(S, R2)')
print(np.matmul(S, R2))

t = np.reshape(U[:, 2], newshape=(3, 1))
print('t')
print(t)

PI = np.concatenate([np.eye(3), np.zeros(shape=(3, 1), dtype=np.float32)], axis=1)
Pp1 = np.concatenate([R1, t], axis=1)
Pp2 = np.concatenate([R1, -t], axis=1)
Pp3 = np.concatenate([R2, t], axis=1)
Pp4 = np.concatenate([R2, -t], axis=1)

