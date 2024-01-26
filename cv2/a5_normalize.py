
import cv2
import numpy as np


def normalize_iris(boundary, centers):
    '''
    #function called normalize_iris() that takes an iris image as input and returns a normalized version of the iris.
    '''
    # Define a list to store the normalized iris images
    normalized = []
    target_images = [images for images in boundary]

    # Iterate over the boundary images
    cent_img = 0
    for img in target_images:
        # Load the pupil center and radius of the inner circle
        center_x = centers[cent_img][0]
        center_y = centers[cent_img][1]
        radius_pupil = int(centers[cent_img][2])

        # Define the iris radius and spacing for sampling
        iris_radius = 53
        nsamples = 360
        samples = np.linspace(0, 2 * np.pi, nsamples)[:-1]

        # Create a polar coordinate grid for the iris
        polar = np.zeros((iris_radius, nsamples))

        for r in range(iris_radius):
            for theta in samples:
                # Calculate the x and y coordinates for the inner boundary
                x = (r + radius_pupil) * np.cos(theta) + center_x
                y = (r + radius_pupil) * np.sin(theta) + center_y
                x = int(x)
                y = int(y)

                # Try to convert the coordinates (ignore out-of-bounds values)
                try:
                    polar[r][int((theta * nsamples) / (2 * np.pi))] = img[y][x]
                except IndexError:
                    pass
                continue

        # Resize the polar grid to a 512x64 image
        result = cv2.resize(polar, (512, 64))

        # Add the normalized image to the list
        normalized.append(result)

        # Increment the center index
        cent_img += 1

    # Return the list of normalized iris images
    return normalized
