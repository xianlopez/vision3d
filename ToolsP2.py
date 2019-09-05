import numpy as np

# Get the intersection point of two lines in P2.
# The lines are in homogeneous coordinates, as well as the point.
def IntersectionOf2Lines(line1, line2):
    return np.cross(line1, line2)

# Get the line defined by two points in P2.
# The lines are in homogeneous coordinates, as well as the point.
def LineFrom2Points(point1, point2):
    return np.cross(point1, point2)

def TransformImageByHomography(image, H):
    pass

def CenterImage(image, H, maxSide):
    origHeight, origWidth, _ = image.shape
    # Compute the position of the image vertices under the transformation H:
    new00 = np.matmul(H, (0, 0, 1))
    neww0 = np.matmul(H, (origWidth - 1, 0, 1))
    new0h = np.matmul(H, (0, origHeight - 1, 1))
    newwh = np.matmul(H, (origWidth - 1, origHeight - 1, 1))
    # Set the last coordinate to 1:
    new00 = new00 / new00[2]
    neww0 = neww0 / neww0[2]
    new0h = new0h / new0h[2]
    newwh = newwh / newwh[2]
    # Build the matrix of the transformation that will scale and translate the image:
    minx = min(new00[0], neww0[0], new0h[0], newwh[0])
    miny = min(new00[1], neww0[1], new0h[1], newwh[1])
    maxx = max(new00[0], neww0[0], new0h[0], newwh[0])
    maxy = max(new00[1], neww0[1], new0h[1], newwh[1])
    assert maxx > minx, 'maxx not greater than minx (' + str(maxx) + ', ' + str(minx) + ')'
    assert maxy > miny, 'maxy not greater than miny (' + str(maxy) + ', ' + str(miny) + ')'
    sx = maxSide / (maxx - minx)
    sy = maxSide / (maxy - miny)
    s = max(sx, sy)
    tx = -s * minx
    ty = -s * miny
    Hcenter = np.eye(3)
    Hcenter[0, 0] = s
    Hcenter[1, 1] = s
    Hcenter[0, 2] = tx
    Hcenter[1, 2] = ty
    # Combine the scaling and translation with the previous transformation:
    Hdst = np.matmul(Hcenter, H)
    # Obtain the width and height of the image adjusted to the corners:
    newWidth = int(s * maxx + tx)
    newHeight = int(s * maxy + ty)
    return Hdst, newWidth, newHeight


# Center the set of points and scale them to have RMS distance = sqrt(2)
# X: (3, npoints)
def Normalization(X, last_coords_is_1=False):
    coords = X[:2, :]  # (3, npoints)
    if not last_coords_is_1:
        last_cooord = X[2, :]  # (1, npoints)
        divisor = np.expand_dims(last_cooord, axis=0)  # (1, npoints)
        divisor = np.tile(divisor, reps=(2, 1))  # (2, npoints)
        coords = np.divide(coords, divisor)  # (2, npoints)
    # Centroid of the points:
    centroid = np.mean(coords, axis=-1)  # (2)
    # Center points:
    coords = coords - np.expand_dims(centroid, axis=-1)  # (2, npoints)
    # Distance to center:
    distance = np.sqrt(np.sum(np.square(coords), axis=0))  # (npoints)
    RMS = np.mean(distance)
    assert RMS > 1e-6, 'RMS too close to 0.'
    scaling_factor = np.sqrt(2.0) / RMS
    # Translation and scaling transformation:
    T = np.diag([scaling_factor, scaling_factor, 1])
    T[0, 2] = scaling_factor * (-centroid[0])
    T[1, 2] = scaling_factor * (-centroid[1])
    return T


