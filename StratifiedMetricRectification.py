## Rectification of a planar image to recover metric properties.
## Done in two steps: Affine rectification and metric rectification.

import ToolsP2
import numpy as np
import cv2
import Draw2PairsOfParallelLines
import Draw2PairsOfOrthogonalLines

# This function receives and image and two pairs of parallel lines
# (lines that we know should be parallel, but they need not to be
# in the image).
# The lines are given in homogeneous coordinates in P2.
def AffineRectification(image, pair1, pair2, maxSide=300, center=True):
    # Compute vanishing points and line at infinity:
    vanishingPoint1 = ToolsP2.IntersectionOf2Lines(pair1[0], pair1[1])
    vanishingPoint1 = vanishingPoint1 / vanishingPoint1[2]
    print('vanishingPoint1')
    print(vanishingPoint1.__class__.__name__)
    print(vanishingPoint1)
    print(vanishingPoint1 / vanishingPoint1[2])
    vanishingPoint2 = ToolsP2.IntersectionOf2Lines(pair2[0], pair2[1])
    vanishingPoint2 = vanishingPoint2 / vanishingPoint2[2]
    print('vanishingPoint2')
    print(vanishingPoint2.__class__.__name__)
    print(vanishingPoint2)
    print(vanishingPoint2 / vanishingPoint2[2])
    lineAtInfinity = ToolsP2.LineFrom2Points(vanishingPoint1, vanishingPoint2)
    lineAtInfinity = lineAtInfinity / lineAtInfinity[2]
    print('Image of the line at infinity:')
    print(lineAtInfinity.__class__.__name__)
    print(lineAtInfinity)
    print(lineAtInfinity / lineAtInfinity[2])
    # Compute the homography:
    H = np.eye(3)
    H[2, 0] = lineAtInfinity[0]
    H[2, 1] = lineAtInfinity[1]
    H[2, 2] = lineAtInfinity[2]
    if center:
        H, newWidth, newHeight = ToolsP2.CenterImage(image, H, 300)
    else:
        newWidth = maxSide
        newHeight = maxSide
    # Transform image:
    newImage = cv2.warpPerspective(image, H, (newWidth, newHeight))
    return newImage

# This function receives and image and two pairs of orthogonal lines
# (lines that we know should be perpendicular, but they need not to be
# in the image).
# The lines are given in homogeneous coordinates in P2.
def MetricRectification(image, pair1, pair2, maxSide=300, center=True):
    # Build the system of equations over the elements of matrix S = KK^t
    A = np.array([[pair1[0][0] * pair1[1][0], pair1[0][1] * pair1[1][0] + pair1[0][0] * pair1[1][1],
                   pair1[0][1] * pair1[1][1]],
                  [pair2[0][0] * pair2[1][0], pair2[0][1] * pair2[1][0] + pair2[0][0] * pair2[1][1],
                   pair2[0][1] * pair2[1][1]]])
    # Compute the SVD of A:
    U, s, Vt = np.linalg.svd(A)
    # Obtain its null vector:
    nullvector = np.array(Vt)[2, :]
    x = np.matmul(A, nullvector)
    assert np.sqrt(np.sum(x * x)) < 1e-8, 'Product of A and null vector is too big: ' + str(x)
    # Build up matrix S from its elements:
    S = np.array([[nullvector[0], nullvector[1]], [nullvector[1], nullvector[2]]])
    # Obtain K using Cholesky factorization:
    K = np.linalg.cholesky(S)
    # Make det(K) = 1:
    K = K / np.sqrt(np.linalg.det(K))
    # Invert K (because we want the opposite mapping):
    K = np.linalg.inv(K)
    # Homography H:
    H = np.eye(3)
    H[0:2, 0:2] = K
    if center:
        H, newWidth, newHeight = ToolsP2.CenterImage(image, H, 300)
    else:
        newWidth = maxSide
        newHeight = maxSide
    # Transform image:
    newImage = cv2.warpPerspective(image, H, (newWidth, newHeight))
    return newImage


def InteractiveStratifiedReconstruction(image):
    # Affine rectification:
    parallelPairs = Draw2PairsOfParallelLines.Draw2PairsOfParallelLines(image)
    affineImage = AffineRectification(image, parallelPairs[0], parallelPairs[1])
    # Metric rectification:
    pairs = Draw2PairsOfOrthogonalLines.Draw2PairsOfOrthogonalLines(affineImage)
    metricImage = MetricRectification(affineImage, pairs[0], pairs[1])
    # Show all the images:
    cv2.imshow('Original image', image)
    cv2.imshow('Affine rectification', affineImage)
    cv2.imshow('Metric rectification', metricImage)
    cv2.waitKey()


if __name__ == '__main__':
    #img_path = r'C:\Users\Nelaalvarez\Documents\Xian\vision3d\chess.jpg'
    img_path = r'C:\Users\Nelaalvarez\Documents\Xian\vision3d\cangas.jpg'
    image = cv2.imread(img_path)
    #image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
    image = cv2.resize(image, (0, 0), fx=0.2, fy=0.2)
    InteractiveStratifiedReconstruction(image)

