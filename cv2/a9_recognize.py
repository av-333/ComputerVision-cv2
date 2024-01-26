
import cv2
import glob
import pandas as pd
import matplotlib.pyplot as plt

from a2_localize import localize_iris
from a5_normalize import normalize_iris
from a1_enhance import enhance_images
from a7_extract_feature import feature_extraction
from a4_matching import matching_iris
from a8_eval_perf import performance_evaluation

import warnings
warnings.filterwarnings("ignore")


# Read training images from CASIA dataset
training_images = [cv2.imread(file) for file in sorted(glob.glob('../CASIA Iris Image Database (version 1.0)/*/1/*.bmp'))]

# Run localization, normalization, enhancement, and feature extraction on training images
boundary, centers = localize_iris(training_images)
normalized = normalize_iris(boundary, centers)
enhanced = enhance_images(normalized)
training_feature_vector = feature_extraction(enhanced)
print("Training complete.")


# Read testing images from CASIA dataset
testing_images = [cv2.imread(file) for file in sorted(glob.glob('../CASIA Iris Image Database (version 1.0)/*/2/*.bmp'))]

# Run localization, normalization, enhancement, and feature extraction on testing images
boundary_1, centers_1 = localize_iris(testing_images)
normalized_1 = normalize_iris(boundary_1, centers_1)
enhanced_1 = enhance_images(normalized_1)
testing_feature_vector = feature_extraction(enhanced_1)
print("Testing complete.")


# Define lists to store CRR scores and matching results
recognition_L1 = []
recognition_L2 = []
recognition_cosine = []
matching_cosine = []
matching_cosine_ROC = []

# Perform matching and calculate CRR scores for various dimensionalities
components = [10, 40, 60, 80, 90, 107]

print("Beginning matching testing with training data...")
for component in components:
    # Run matching for current dimensionality
    component_matching_L1, component_matching_L2, component_matching_cosine, component_matching_cosine_ROC = matching_iris(training_feature_vector, testing_feature_vector, component, 0)

    # Calculate CRR scores for current dimensionality
    component_recognition_L1, component_recognition_L2, component_recognition_cosine = performance_evaluation(component_matching_L1, component_matching_L2, component_matching_cosine)

    # Append results to lists
    recognition_L1.append(component_recognition_L1)
    recognition_L2.append(component_recognition_L2)
    recognition_cosine.append(component_recognition_cosine)
    matching_cosine.append(component_matching_cosine)
    matching_cosine_ROC.append(component_matching_cosine_ROC)

# Perform matching and calculate CRR scores for original feature vector
true_matching_L1, true_matching_L2, true_matching_cosine, true_matching_cosine_ROC = matching_iris(training_feature_vector, testing_feature_vector, 0, 1)
true_recognition_L1, true_recognition_L2, true_recognition_cosine = performance_evaluation(true_matching_L1, true_matching_L2, true_matching_cosine)

print("Matching complete.", "\n")


# Orginal and reduced feature set Table for CRR rates
print("\n")
dict = {'Similarity Metrics':['L1','L2','Cosine Distance'],'CRR for Original Feature Set':[true_recognition_L1, true_recognition_L2, true_recognition_cosine],'CRR for Reduced Feature Set (107)':[recognition_L1[5],recognition_L2[5],recognition_cosine[5]]}
dictionary = pd.DataFrame(dict)
print("Comparative Analysis of Recognition Results Using Various Similarity Metrics : \n")
print(dictionary.iloc[0],"\n")
print(dictionary.iloc[1],"\n")
print(dictionary.iloc[2],"\n")


#Visualizing the Increasing Cosine Similarity-Dimensionality Relationship 
plt.plot(components, recognition_cosine)
plt.axis([10, 107, 0, 100])
plt.ylabel("Correct Recognition Rate (Cosine)")
plt.xlabel("Feature Vector Dim")
plt.title("Result CRR and Dimensionality")
plt.show()

# false positive and true positive rates
false_positive_rates = []
true_positive_rates = []

thresholds = [0.4, 0.5, 0.6]

for a in range(0, 3):
    acceptance_false=0
    rejection_false=0
    num_1 = len([i for i in matching_cosine_ROC[5][a] if i == 1])
    num_0 = len([i for i in matching_cosine_ROC[5][a] if i == 0])

    for b in range(0,len(matching_cosine[5])):
        if matching_cosine[5][b] == 0 and matching_cosine_ROC[5][a][b] == 1:
            acceptance_false += 1
        if matching_cosine[5][b] == 1 and matching_cosine_ROC[5][a][b] == 0:
            rejection_false += 1
    
    false_positive = acceptance_false / num_1
    true_positive = rejection_false / num_0

    threshold=[0.4,0.5,0.6]
    false_positive_rates.append(false_positive)
    true_positive_rates.append(true_positive)

roc_dictionary = pd.DataFrame({"Threshold": threshold, "false_positive": false_positive_rates, "true_positive": true_positive_rates})
print("ROC Metrics : \n")
print(roc_dictionary.iloc[0], "\n")
print(roc_dictionary.iloc[1], "\n")
print(roc_dictionary.iloc[2], "\n")

# Plotting the ROC Curve
plt.plot(true_positive_rates, false_positive_rates)
plt.title("ROC Curve")
plt.ylabel("Not Matching False Rate")
plt.xlabel("Matching False Rate")
plt.show()
