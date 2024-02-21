import cv2
import numpy as np
from time import time

# make a marker path list
marker_path = ['Images/marker1.jpg'
    , 'Images/marker2.jpg'
    , 'Images/marker3.jpg'
    , 'Images/marker4.jpg'
               ]


def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def order_points(pts):
    # A total of 4 coordinate points
    rect = np.zeros((4, 2), dtype="float32")

    # Find the corresponding coordinates 0123 in order: upper left, upper right, lower right, lower left
    # Calculate upper left, lower right
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # Calculate upper right and lower left
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def four_point_transform(image, pts):
    # Get the input coordinate point
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Calculate the input w and h values
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # Corresponding coordinate position after transformation
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # Calculate transformation matrix
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # Return the transformed result
    return warped


def get_geofeature(corners, error_threshold):
    corners = corners.reshape(-1, 2)
    # Calculate the centroid of all corners
    centroid = np.mean(corners, axis=0)
    centroid = np.int0(centroid)

    # find line with most corners
    max_count = 0
    best_line = None
    for i, start in enumerate(corners[:-1]):
        for end in corners[i + 1:]:
            # Filter for approximately vertical or horizontal endpoints
            if (np.abs(start[0] - end[0]) <= 2 * error_threshold
                    or np.abs(start[1] - end[1]) <= 2 * error_threshold):
                line_vector = end - start
                line_length = np.linalg.norm(line_vector)
                if line_length == 0:
                    continue
                line_unit_vector = line_vector / line_length
                count = 0
                # Calculate whether other corner points are on the same straight line
                for corner in corners:
                    if np.array_equal(corner, start) or np.array_equal(corner, end):
                        continue
                    corner_vector = corner - start
                    distance_along_line = np.dot(corner_vector, line_unit_vector)
                    closest_point_on_line = start + distance_along_line * line_unit_vector
                    distance_from_line = np.linalg.norm(corner - closest_point_on_line)
                    # Check if it is within the error range
                    if distance_from_line <= error_threshold:
                        count += 1
                # Update the best line segment
                if count >= 2 and count > max_count:  # 4 points for a line
                    max_count = count
                    best_line = (start, end)

    if not best_line:
        return None
    return centroid, best_line


def determine_arrow_direction(best_line, centroid):
    if best_line is not None:
        start, end = best_line
        line_vector = end - start
        line_length = np.linalg.norm(line_vector)
        line_unit_vector = line_vector / line_length
        centroid_vector = centroid - start
        projection_length = np.dot(centroid_vector, line_unit_vector)
        closest_point_on_line = start + projection_length * line_unit_vector
        direction_vector = centroid - closest_point_on_line
        direction_angle = np.arctan2(direction_vector[1], direction_vector[0])
        return np.degrees(direction_angle)
    return None


def map_angle_to_direction(angle):
    # Angle maps to direction
    if -135 <= angle < -45:
        return "Down"
    elif -45 <= angle < 45:
        return "Left"
    elif 45 <= angle < 135:
        return "Up"
    else:
        return "Right"


def trackArrows(image):
    # Resize the image
    image = cv2.resize(image, (640, 480))
    origin = image.copy()

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Blurring and edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 100, 200)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    doCnt = dict()
    # Find contours
    for contour in contours:
        # Approximate the contour to a polygon
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        area = cv2.contourArea(contour)

        # If the polygon has 4 vertices, it could be a rectangle, area must reasonable
        if len(approx) == 4 and area >= 10000:
            doCnt[area] = approx

    # Use the minimum rectangular
    if not doCnt:
        return None
    roi = doCnt[min(doCnt)]
    # Draw the contour (in green) and bounding box (in blue)
    contour_image = image.copy()
    cv2.drawContours(contour_image, [roi], 0, (0, 255, 0), 3)
    # Perform perspective transformation
    warped = four_point_transform(gray, roi.reshape(4, 2))
    # Blur again for Anti-aliasing
    warped = cv2.GaussianBlur(warped, (5, 5), 0)

    # Otsu's thresholding
    thresh = cv2.threshold(warped, 0, 255,
                           cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # corner detection
    corners = cv2.goodFeaturesToTrack(thresh, 10, 0.3, 3)
    corners = np.int0(corners)
    # Convert gray to RGB for color marking
    corner_image = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
    # Draw corners in red
    for i in corners:
        x, y = i.ravel()
        cv2.circle(corner_image, (x, y), 3, (0, 0, 255), -1)

    # Get geometric features
    if not get_geofeature(corners, 3):
        return None
    centroid, best_line = get_geofeature(corners, 3)

    geo_feature = thresh.copy()
    geo_feature = cv2.cvtColor(geo_feature, cv2.COLOR_GRAY2RGB)
    cv2.circle(geo_feature, tuple(centroid), 3, (0, 0, 255), -1)

    cv2.line(geo_feature, tuple(best_line[0]), tuple(best_line[1]), (0, 0, 255), 2)

    # Determine the direction of the arrow
    arrow_direction_angle = determine_arrow_direction(best_line, centroid)
    arrow_direction = map_angle_to_direction(arrow_direction_angle)
    cv2.putText(image, arrow_direction, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return origin, blurred, edges, contour_image, warped, thresh, corner_image, geo_feature, arrow_direction, image


def main():
    # arrows tracking in sample images
    for p in marker_path:
        # Read image
        image = cv2.imread(p)
        if not trackArrows(image):
            continue
        origin, blurred, edges, contour_image, warped, thresh, corner_image, geo_feature, arrow_direction, detected = trackArrows(image)
        cv_show('origin', origin)
        cv_show('blurred', blurred)
        cv_show('edges', edges)
        cv_show('contour_image', contour_image)
        cv_show('warped', warped)
        cv_show('thresh', thresh)
        cv_show('corner_image', corner_image)
        cv_show('geo_feature', geo_feature)
        cv_show('detected', detected)

        cv2.imwrite(p[:-4] + '_origin.jpg', origin)
        cv2.imwrite(p[:-4] + '_blurred.jpg', blurred)
        cv2.imwrite(p[:-4] + '_edges.jpg', edges)
        cv2.imwrite(p[:-4] + '_contour_image.jpg', contour_image)
        cv2.imwrite(p[:-4] + '_warped.jpg', warped)
        cv2.imwrite(p[:-4] + '_thresh.jpg', thresh)
        cv2.imwrite(p[:-4] + '_corner_image.jpg', corner_image)
        cv2.imwrite(p[:-4] + '_geo_feature.jpg', geo_feature)
        cv2.imwrite(p[:-4] + '_detected.jpg', detected)

    # ball tracking in webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Webcam not accessible.")
        return

    last_detected_direction = None
    last_update_time = time()  # Record the time when the text was last updated
    direction_display_duration = 5.0  # Text display duration in seconds
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if not trackArrows(frame):
            detected = cv2.resize(frame, (640, 480))
        else:
            geo_feature, direction, detected = trackArrows(frame)[-3:]
            last_detected_direction = direction
            last_update_time = time()  # Update time
            cv2.imshow('geo_feature', geo_feature)

        # If within the specified time, continue to display the last detected direction
        if last_detected_direction and time() - last_update_time < direction_display_duration:
            cv2.putText(detected, last_detected_direction, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('arrows tracking', detected)

        # keyboard setting
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            t = int(time())
            cv2.imwrite('Images/arrow_snapshot_%s.jpg' % t, detected)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
