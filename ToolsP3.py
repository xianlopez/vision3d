import numpy as np


# Center the set of points and scale them to have RMS distance = sqrt(3)
# X: (4, npoints)
def Normalization(X, last_coords_is_1=False):
    coords = X[:3, :]  # (3, npoints)
    if not last_coords_is_1:
        last_cooord = X[3, :]  # (1, npoints)
        divisor = np.expand_dims(last_cooord, axis=0)  # (1, npoints)
        divisor = np.tile(divisor, reps=(3, 1))  # (3, npoints)
        coords = np.divide(coords, divisor)  # (3, npoints)
    # Centroid of the points:
    centroid = np.mean(coords, axis=-1)  # (3)
    # Center points:
    coords = coords - np.expand_dims(centroid, axis=-1)  # (3, npoints)
    # Distance to center:
    distance = np.sqrt(np.sum(np.square(coords), axis=0))  # (npoints)
    RMS = np.mean(distance)
    assert RMS > 1e-6, 'RMS too close to 0.'
    scaling_factor = np.sqrt(3.0) / RMS
    # Translation and scaling transformation:
    T = np.diag([scaling_factor, scaling_factor, scaling_factor, 1])
    T[0, 3] = scaling_factor * (-centroid[0])
    T[1, 3] = scaling_factor * (-centroid[1])
    T[2, 3] = scaling_factor * (-centroid[2])
    return T


if __name__ == '__main__':
    X = np.ones(shape=(4, 4))
    X[0, 0] = 1 + 2.5
    X[1, 0] = np.sqrt(2) - 3
    X[2, 0] = 0 + 1.2
    X[0, 1] = -1 + 2.5
    X[1, 1] = 0 - 3
    X[2, 1] = np.sqrt(2) + 1.2
    X[0, 2] = 1 + 2.5
    X[1, 2] = -np.sqrt(2) - 3
    X[2, 2] = 0 + 1.2
    X[0, 3] = -1 + 2.5
    X[1, 3] = 0 - 3
    X[2, 3] = -np.sqrt(2) + 1.2
    X[:3, :] = X[:3, :] * 3.2
    X[3, :] = 2
    T = Normalization(X)
    Xnorm = np.matmul(T, X)
    expected_points = [(1, 1, 1, 1), (np.sqrt(2), 1, 0, 1), (-1, 0, np.sqrt(2), 1)]
    for i in range(3):
        print('Expected point: ' + str(expected_points[i]))
        print('Obtained point: ' + str(Xnorm[:, i]))
        diff = np.mean(np.abs(Xnorm[:, i] - expected_points[i]))
