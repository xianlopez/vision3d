import cv2

refPt = []
point1 = None
point2 = None
tmpPoint = None
cropping = False
# img_path = r'C:\development\vision3d\images_img40_box237.png'
# img_path = r'C:\development\vision3d\chess.jpg'
img_path = r'C:\Users\Nelaalvarez\Documents\Xian\vision3d\chess.jpg'
image = cv2.imread(img_path)
baseImage = image.copy()
tempImage = image.copy()

def click_and_crop(event, x, y, flags, param):
    global point1, point2, tmpPoint, cropping, image, tempImage
    if event == cv2.EVENT_RBUTTONDOWN:
        tempImage = image.copy()
    elif event == cv2.EVENT_LBUTTONDOWN:
        point1 = (x, y)
        cropping = True
    elif event == cv2.EVENT_LBUTTONUP:
        point2 = (x, y)
        cropping = False
    elif cropping:
        tmpPoint = (x, y)
        tempImage = image.copy()
        cv2.line(tempImage, point1, tmpPoint, (255, 0, 0), 2)


cv2.namedWindow('image')
cv2.setMouseCallback('image', click_and_crop)

while True:
    cv2.imshow('image', tempImage)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('r'):  # Reset
        image = baseImage.copy()
        tempImage = baseImage.copy()
    elif key == ord('s'):  # Save line
        if point1 is None or point2 is None:
            print('Please draw a line before trying to save it.')
        else:
            cv2.line(image, point1, point2, (0, 255, 0), 2)
            tempImage = image.copy()
            point1 = None
            point2 = None
    elif key == ord('q'):  # Quit
        break



