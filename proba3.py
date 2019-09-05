import cv2
import numpy as np

img = cv2.imread(r'C:\Users\Nelaalvarez\Documents\Xian\vision3d\sudoku.jpg')
#img = cv2.resize(img, (0, 0), fx=1.5, fy=1.5)
cv2.imshow('src', img)
rows,cols,ch = img.shape
#pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
#pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])
pts1 = np.float32([[28, 33],[183, 28],[13, 190],[193, 194]])
pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])
M = cv2.getPerspectiveTransform(pts1,pts2)
dst = cv2.warpPerspective(img,M,(300,300))

cv2.imshow('dst', dst)
cv2.waitKey()