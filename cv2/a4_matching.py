
import math
import numpy as np
from a3_reduce_dim import dimensionality_reduction


def matching_iris(train_features, test_features, components, flag):
    """
    Performs matching between train and test sets using dist_L1, dist_L2, and dist_cosine distances.
    """

    #Dimensionality reduction If flag is 1, we don't need to reduce dimensionality, otherwise, we call the dimensionality_reduction function
    if flag == 1:
        reduced_train = train_features
        reduced_test = test_features
        
    elif flag == 0:
        reduced_train, reduced_test = dimensionality_reduction(train_features, test_features, components)


    x_1 = reduced_test
    x_2 = reduced_train

    # Initialize empty lists to store the matching results
    new_cosine = []
    ind_cosine = []
    ind_L1 = []
    ind_L2 = []
    
    # Loop over each test image
    for a in range(0, len(x_1)):
        # Calculate L1, L2, and cosine distances between the test image and each train image
        dist_L1 = []
        dist_L2 = []
        dist_cosine = []
        
        # All training image ompared to each test image
        for b in range(0, len(x_2)):
            testing_data = x_1[a]
            training_data = x_2[b]
            sum_L1 = 0
            sum_L2 = 0
            sum_cos1 = 0
            sum_cos2 = 0
            cosine_dist=0
            
            # Calculate L1, L2 distances and sum of squares of all features
            for c in range(0, len(testing_data)):
                sum_L1 += abs(testing_data[c] - training_data[c])
                sum_L2 += math.pow((testing_data[c] - training_data[c]), 2)
            
            for d in range(0, len(testing_data)):
                sum_cos1 += math.pow(testing_data[d], 2)
                sum_cos2 += math.pow(training_data[d], 2)
                
            
            # Calculate cosine distance using sum_cos1 and sum_cos2 calculated above
            cosine_dist = 1 - ((np.matmul(np.transpose(testing_data), training_data)) / (math.pow(sum_cos1, 0.5) * math.pow(sum_cos2, 0.5)))
            
            dist_cosine.append(cosine_dist)
            dist_L1.append(sum_L1)
            dist_L2.append(sum_L2)

        # Get the indices of the closest matches for each distance metric
        new_cosine.append(min(dist_cosine))
        ind_cosine.append(dist_cosine.index(min(dist_cosine)))
        ind_L1.append(dist_L1.index(min(dist_L1)))
        ind_L2.append(dist_L2.index(min(dist_L2)))
        

    # Store final matching results
    matching_cosine = []
    matching_cosine_ROC = []
    matching_L1 = []
    matching_L2 = []
    
    # Calculate matching according to ROC thresholds
    threshold = [0.4,0.5,0.6]
    count = 0
    match = 0
    
    for m in range(0, len(threshold)):
        matching_ROC = []
        for n in range(0, len(new_cosine)):
            if new_cosine[n] <= threshold[m]:
                matching_ROC.append(1)
            else:
                matching_ROC.append(0)
        matching_cosine_ROC.append(matching_ROC)
        
    # Iterate through test set and update matching arrays
    for o in range(0, len(ind_L1)):
        if count < 4:
            count += 1
        else:
            match += 3
            count = 1
            
        if ind_L1[o] in range(match, match+3):
                matching_L1.append(1)
        else:
            matching_L1.append(0)
        
        if ind_L2[o] in range(match, match+3):
            matching_L2.append(1)
        else:
            matching_L2.append(0)
        
        if ind_cosine[o] in range(match, match+3):
            matching_cosine.append(1)
        else:
            matching_cosine.append(0)
    
    return matching_L1,matching_L2,matching_cosine,matching_cosine_ROC
