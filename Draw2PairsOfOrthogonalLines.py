import LineDrawer
import cv2


def Draw2PairsOfOrthogonalLines(image):
    print('Start by drawing two perpendicular lines. Press "d" after drawing each line.')
    print('Line 1...')
    drawer = LineDrawer.LineDrawer(image)
    p1l1, point1, point2 = drawer.Start()
    cv2.line(image, point1, point2, (0, 0, 255), 2)
    print('Line 2...')
    drawer = LineDrawer.LineDrawer(image)
    p1l2, point1, point2 = drawer.Start()
    cv2.line(image, point1, point2, (0, 0, 255), 2)
    print('Draw the second pair of perpendicular lines. Press "d" after drawing each line.')
    print('Line 1...')
    drawer = LineDrawer.LineDrawer(image)
    p2l1, point1, point2 = drawer.Start()
    cv2.line(image, point1, point2, (255, 0, 128), 2)
    print('Line 2...')
    drawer = LineDrawer.LineDrawer(image)
    p2l2, point1, point2 = drawer.Start()
    cv2.line(image, point1, point2, (255, 0, 128), 2)
    #cv2.imshow('FinalImage', image)
    #cv2.waitKey()
    return [p1l1, p1l2], [p2l1, p2l2]

