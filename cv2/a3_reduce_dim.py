
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


def dimensionality_reduction(training_data, test_data, num_components):
    """
    Perform dimensionality reduction on the given data using Latent Dirichlet Allocation (LDA).
    """
    # Training data
    X_train = training_data

    y_train = []
    for a in range(0,108):
        for b in range(0,3):
            y_train.append(a+1)
    y_train = np.array(y_train)
    
    # Fit LDA model on training data
    lda = LDA(n_components=num_components)
    lda.fit(X_train, y_train)
    
    # Transform training data
    X_train_reduced = lda.transform(X_train)
    
    # Testing data
    X_test = test_data
    
    # Transform testing data
    X_test_reduced = lda.transform(X_test)
    
    # Predict classes for testing data
    y_pred = lda.predict(X_test)
    
    #return transformed training and testing data, and the testing classes and predicted values for ROC
    return X_train_reduced, X_test_reduced
