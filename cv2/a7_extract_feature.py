
from scipy.signal import convolve2d
from a6_filter import *


def feature_extraction(enhanced): 
    """Extracts features from enhanced images."""

    feature_vectors = []
    first_filter = spatial(0.67,3,1.5)
    second_filter = spatial(0.67,4,1.5) 

    for image in range(len(enhanced)):
        image = enhanced[image]
        image_roi = image[:48, :]

        first_filtered = convolve2d(image_roi, first_filter, mode='same')
        second_filtered = convolve2d(image_roi, second_filter, mode='same')
        
        feature_vector = get_vec(first_filtered, second_filtered)
        feature_vectors.append(feature_vector)

    return feature_vectors
