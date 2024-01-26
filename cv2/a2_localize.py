
import cv2
import math
import numpy as np
import pandas as pd
from scipy.spatial import distance


def localize_iris(images):
    """
    function called localize_iris() that takes an enhanced image as input and returns the location of the iris in the image.
    """

    # Convert image to grayscale
    gray_images = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images]

    # Initialize empty lists to store boundaries and centers
    boundaries = []
    centers = []

    for gray_image in gray_images:
        # Remove noise by bilateral filtering
        filtered_image = cv2.bilateralFilter(gray_image, 9, 75, 75)
        img = filtered_image
        
        # Estimate center of pupil
        center_x = np.mean(filtered_image, 0).argmin()
        center_y = np.mean(filtered_image, 1).argmin()
        
        #recalculate of pupil by concentrating on a 120X120 area
        centered_cropped_image_x = filtered_image[center_x - 60:center_x + 60]
        centered_cropped_image_y = filtered_image[center_y - 60:center_y + 60]
        new_center_x = np.mean(centered_cropped_image_x, 0).argmin()
        new_center_y = np.mean(centered_cropped_image_y, 0).argmin()

        # Draw circle around pupil center
        drawing_image = filtered_image.copy()
        cv2.circle(drawing_image, (new_center_x, new_center_y), 1, (255, 0, 0), 2)

        # Apply Canny edge detection on masked image
        masked_image = cv2.inRange(filtered_image, 0, 70)
        output_image = cv2.bitwise_and(filtered_image, masked_image)
        edges = cv2.Canny(output_image, 100, 220)

        # Find potential boundaries of pupil using Hough Circles
        circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 10, 100)

        # Define center of pupil
        pupil_center = (new_center_x, new_center_y)

        # Find circle whose center is closest to approximated center
        min_distance = math.inf
        for circle in circles[0]:
            circle_center = (circle[0], circle[1])
            dist = distance.euclidean(pupil_center, circle_center)
            if dist < min_distance:
                min_distance = dist
                best_circle = circle

        # Draw inner and outer boundaries
        cv2.circle(filtered_image, (int(best_circle[0]), int(best_circle[1])), int(best_circle[2]), (255, 0, 0), 3)
        cv2.circle(filtered_image, (int(best_circle[0]), int(best_circle[1])), int(best_circle[2]) + 53, (255, 0, 0), 3)

        # Append boundaries and centers to respective lists
        boundaries.append(filtered_image)
        centers.append((best_circle[0], best_circle[1], best_circle[2]))

    return boundaries, centers
