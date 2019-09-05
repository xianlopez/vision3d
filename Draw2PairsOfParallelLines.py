import LineDrawer
import cv2


def Draw2PairsOfParallelLines(image):
    print('Start by drawing two parallel lines. Press "d" after drawing each line.')
    print('Line 1...')
    drawer = LineDrawer.LineDrawer(image)
    p1l1, point1, point2 = drawer.Start()
    cv2.line(image, point1, point2, (0, 0, 255), 2)
    print('Line 2...')
    drawer = LineDrawer.LineDrawer(image)
    p1l2, point1, point2 = drawer.Start()
    cv2.line(image, point1, point2, (0, 0, 255), 2)
    print('Draw the second pair of parallel lines. Press "d" after drawing each line.')
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

if __name__ == '__main__':
    img_path = r'C:\development\vision3d\chess.jpg'
    image = cv2.imread(img_path)
    Draw2PairsOfParallelLines(image)
