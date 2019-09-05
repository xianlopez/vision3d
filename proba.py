import Draw2PairsOfParallelLines
import Draw2PairsOfOrthogonalLines
import StratifiedMetricRectification
import cv2
import numpy as np

img_path = r'C:\Users\Nelaalvarez\Documents\Xian\vision3d\chess.jpg'
image = cv2.imread(img_path)
image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
cv2.imshow('original image', image)
#pairs = Draw2PairsOfParallelLines.Draw2PairsOfParallelLines(image)
line1 = np.array((164, 248, -141272))
line2 = np.array((95, 242, -45974))
line3 = np.array((-139, 245, 30124))
line4 = np.array((-207, 228, -17331))
line1 = line1 / line1[2]
line2 = line2 / line2[2]
line3 = line3 / line3[2]
line4 = line4 / line4[2]
pair1 = [line1, line2]
pair2 = [line3, line4]
pairs = [pair1, pair2]
affineImage = StratifiedMetricRectification.AffineRectification(image, pairs[0], pairs[1])
cv2.imshow('Affine rectification', affineImage)
cv2.waitKey()



#pairs = Draw2PairsOfOrthogonalLines.Draw2PairsOfOrthogonalLines(affineImage)
line1 = np.array((-132, 179, 42870))
line2 = np.array((-93, -368, 57440))
line3 = np.array((-33, 441, -63378))
line4 = np.array((-187, -155, 82872))
line1 = line1 / line1[2]
line2 = line2 / line2[2]
line3 = line3 / line3[2]
line4 = line4 / line4[2]
pair1 = [line1, line2]
pair2 = [line3, line4]
pairs = [pair1, pair2]
metricImage = StratifiedMetricRectification.MetricRectification(affineImage, pairs[0], pairs[1])
cv2.imshow('Metric rectification', metricImage)
cv2.waitKey()

