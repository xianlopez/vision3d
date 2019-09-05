import cv2
import FundamentalMatrixComputation
import numpy as np

#img_path_1 = r'C:\Users\Nelaalvarez\Documents\Xian\vision3d\mesa1.jpg'
#img_path_2 = r'C:\Users\Nelaalvarez\Documents\Xian\vision3d\mesa2.jpg'

img_path_1 = r'C:\Users\Nelaalvarez\Documents\Xian\vision3d\mesa3.jpg'
img_path_2 = r'C:\Users\Nelaalvarez\Documents\Xian\vision3d\mesa4.jpg'

img1 = cv2.imread(img_path_1)
img2 = cv2.imread(img_path_2)

img1 = cv2.resize(img1, (0, 0), fx=0.15, fy=0.15, interpolation=cv2.INTER_AREA)
img2 = cv2.resize(img2, (0, 0), fx=0.15, fy=0.15, interpolation=cv2.INTER_AREA)

cv2.imshow('img1', img1)
cv2.imshow('img2', img2)

cv2.waitKey()

sift = cv2.xfeatures2d.SIFT_create()

kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key= lambda x:x.distance)

img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=2)
cv2.imshow('matches', img3)
cv2.waitKey()

npoints = len(matches)
X1 = np.ones(shape=(3, npoints), dtype=np.float32)
X2 = np.ones(shape=(3, npoints), dtype=np.float32)
for i in range(npoints):
    #X1[0, i] = kp1[matches[i].trainIdx].pt[0]
    #X1[1, i] = kp1[matches[i].trainIdx].pt[1]
    #X2[0, i] = kp2[matches[i].queryIdx].pt[0]
    #X2[1, i] = kp2[matches[i].queryIdx].pt[1]

    X1[0, i] = kp1[matches[i].queryIdx].pt[0]
    X1[1, i] = kp1[matches[i].queryIdx].pt[1]
    X2[0, i] = kp2[matches[i].trainIdx].pt[0]
    X2[1, i] = kp2[matches[i].trainIdx].pt[1]

F, inliers = FundamentalMatrixComputation.ComputeF(X1, X2, max_distance=0.5)

remaining_matches = []
discarded_matches = []
for i in range(npoints):
    if i in inliers:
        remaining_matches.append(matches[i])
    else:
        discarded_matches.append(matches[i])

img_remaining = cv2.drawMatches(img1, kp1, img2, kp2, remaining_matches, None, flags=2)
img_discarded = cv2.drawMatches(img1, kp1, img2, kp2, discarded_matches, None, flags=2)
cv2.imshow('remaining matches', img_remaining)
cv2.imshow('discarded matches', img_discarded)
cv2.waitKey()

