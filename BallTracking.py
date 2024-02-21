import cv2
import numpy as np
from time import time

# make a ball path list
ball_path = ['Images/ball1.jpg'
    , 'Images/ball2.jpg'
    , 'Images/ball3.jpg'
    , 'Images/ball4.jpg'
    , 'Images/ball5.jpg'
    , 'Images/ball6.jpg'
             ]

# Yellow HSV threshold
hsv_lower = np.array([15, 100, 100])  # Yellow lower HSV threshold
hsv_upper = np.array([35, 255, 255])  # Yellow upper HSV threshold


def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def trackBall(image):
    # Resize the image
    image = cv2.resize(image, (640, 480))
    origin = image.copy()

    # Convert to HSV and apply mask
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_image, hsv_lower, hsv_upper)

    # Blurring and edge detection
    blurred = cv2.GaussianBlur(mask, (5, 5), 0)

    # Morphological closing, erosion
    kernel = np.ones((5, 5), np.uint8)
    # Can't directly use opening or erosion, because the ball would be smaller due to its shadow
    # Can't directly use dilation, because there     cv_show('thresh', thresh)are noise in ball5 ball6.
    closing = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel)

    # THRESH_OTSU will automatically find a suitable threshold, suitable for double peaks
    # , and the threshold parameter needs to be set to 0
    thresh = cv2.threshold(closing, 0, 255,
                           cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Approximate the contour to a polygon
        perimeter = cv2.arcLength(contour, True)
        epsilon = 0.005 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # calculate circularity and area
        area = cv2.contourArea(contour)
        if perimeter == 0:  # Avoid divide zero error
            continue
        circularity = 4 * np.pi * (area / (perimeter * perimeter))

        if 3000 <= area and 0.6 <= circularity <= 1.0:
            # Draw the contour (in green) and bounding box (in blue)
            cv2.drawContours(image, [approx], 0, (0, 255, 0), 3)
            x, y, w, h = cv2.boundingRect(approx)
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)
            cv2.circle(image, center, radius, (0, 0, 255), 2)

    return origin, mask, blurred, closing, thresh, image


def main():
    # ball tracking in sample images
    for p in ball_path:
        # Read image
        image = cv2.imread(p)
        origin, mask, blurred, closing, thresh, detected = trackBall(image)
        cv_show('origin', origin)
        cv_show('mask', mask)
        cv_show('blurred', blurred)
        cv_show('closing', closing)
        cv_show('thresh', thresh)
        cv_show('detected', detected)
        cv2.imwrite(p[:-4] + '_origin.jpg', origin)
        cv2.imwrite(p[:-4] + '_mask.jpg', mask)
        cv2.imwrite(p[:-4] + '_blurred.jpg', blurred)
        cv2.imwrite(p[:-4] + '_closing.jpg', closing)
        cv2.imwrite(p[:-4] + '_thresh.jpg', thresh)
        cv2.imwrite(p[:-4] + '_detected.jpg', detected)

    # ball tracking in webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Webcam not accessible.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        detected = trackBall(frame)[-1]
        cv2.imshow('ball tracking', detected)

        # keyboard setting
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            t = int(time())
            cv2.imwrite('Images/ball_snapshot_%s.jpg' % t, detected)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
