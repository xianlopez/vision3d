import F8PointAlgorithm
import numpy as np
import cv2


def compute_inliers(F, X1, X2, max_distance):
    inliers = []
    for i in range(X1.shape[1]):
        distance = np.abs(np.matmul(np.transpose(X2[:, i]), np.matmul(F, X1[:, i])))
        if distance <= max_distance:
            inliers.append(i)
    return inliers


# X1: (3, npoints)
# X2: (3, npoints)
def ComputeF(X1, X2, ntrials=50, max_distance=1):
    _, npoints = X1.shape
    assert X1.shape == X2.shape, 'The shape of the two point sets should be equal.'
    all_indices = np.arange(npoints)
    inliers_list = []
    for k in range(ntrials):
        indices = np.random.choice(all_indices, size=8, replace=False)
        X1sub = X1[:, indices]
        X2sub = X2[:, indices]
        F = F8PointAlgorithm.EightPointAlgorithm(X1sub, X2sub)
        inliers = compute_inliers(F, X1, X2, max_distance)
        inliers_list.append(inliers)
    max_inliers = -1
    best_case = -1
    for k in range(ntrials):
        if len(inliers_list[k]) > max_inliers:
            best_case = k
            max_inliers = len(inliers_list[k])
    # Recompute F with all the inliers:
    inliers = inliers_list[best_case]
    X1_in = X1[:, inliers]
    X2_in = X2[:, inliers]
    F = F8PointAlgorithm.EightPointAlgorithm(X1_in, X2_in)
    inliers = compute_inliers(F, X1, X2, max_distance)
    return F, inliers




