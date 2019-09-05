import cv2
import numpy as np


class LineDrawer:
    def __init__(self, baseImage):
        self.image = baseImage.copy()
        self.tmpImage = baseImage.copy()
        self.point1 = None
        self.point2 = None
        self.dragging = False

    def ClickAndCrop(self, event, x, y, flags, param):
        if event == cv2.EVENT_RBUTTONDOWN:
            self.tmpImage = self.image.copy()
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.point1 = (x, y)
            self.dragging = True
        elif event == cv2.EVENT_LBUTTONUP:
            self.point2 = (x, y)
            self.dragging = False
        elif self.dragging:
            tmpPoint = (x, y)
            self.tmpImage = self.image.copy()
            cv2.line(self.tmpImage, self.point1, tmpPoint, (255, 0, 0), 2)

    def Start(self):
        cv2.namedWindow('image')
        cv2.setMouseCallback('image', self.ClickAndCrop)

        while True:
            cv2.imshow('image', self.tmpImage)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('d'):  # Done
                if self.point1 is None or self.point2 is None:
                    print('Please draw a line before trying to save it.')
                else:
                    cv2.line(self.image, self.point1, self.point2, (0, 255, 0), 2)
                    # Extend the points to homogeneous coordinates, and then compute the line
                    # passing through both of them:
                    point1H = (self.point1[0], self.point1[1], 1)
                    point2H = (self.point2[0], self.point2[1], 1)
                    line = np.cross(point1H, point2H)
                    line = line / line[2]
                    print('Line coordinates: (' + str(line[0]) + ', ' + str(line[1]) + ', ' + str(line[2]) + ')')
                    cv2.destroyWindow('image')
                    return line, self.point1, self.point2


if __name__ == '__main__':
    img_path = r'C:\development\vision3d\chess.jpg'
    image = cv2.imread(img_path)
    drawer = LineDrawer(image)
    line = drawer.Start()