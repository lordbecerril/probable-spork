#AUTHOR: Eric Becerril-Blas
import pandas as pd
import numpy as np
import math
import random
import statistics


# Datasets given to me from course web page
MNIST15_15_df = pd.read_csv("MNIST_15_15.csv", header = None)
MNIST15_15_labels = pd.read_csv("MNIST_LABEL.csv", header = None)



def SolverLinearRegression(X, y):
    """
        Does Magic
    """
    XTdotX = np.dot(X.transpose(), X)
    #print("np.dot(X.transpose(), X) is:")
    #print(XTdotX)
    #Using pinv for psuedo inverse since most of these give me vectors with determinant equal to 0
    return np.dot(np.dot(np.linalg.pinv(XTdotX), X.transpose()), y)


def CreateDataFrame(data, Normalized):
    """
        Combines the labels and the normalized data into one data frame and returns it
    """
    # Verify Length of data
    #print(len(data.values))

    # Create numpy array of data and print to verify
    dummy = data.values
    #print(dummy)

    # Create 2 empty lists. 1 for labelz and the other for pixel data
    labelz = []
    picz = []

    # Loop through indices of K Cross validation Vector and put them in assigned vector
    for i in range(len(dummy)):
        labelz.append(MNIST15_15_labels.iloc[dummy[i]].values)
        picz.append(Normalized.iloc[dummy[i]].values)

    #print(labelz)
    labelz_df = pd.DataFrame(labelz)
    pixels_df = pd.DataFrame(picz)
    df = pd.concat([labelz_df, pixels_df], axis=1) # Make a new dataframe with the labels now added to it
    df.columns = range(df.shape[1])
    return(df)



def LinearRegressionClassification(cluster, Normalized):
    """
        Does the linear regression from scratch
    """
    # Create an empty vector which will hold Accuracies of each experiment
    Accuracies = []

    for k in range(len(cluster.index)):
        # Create a vector call subAccuracies which will append to Accuracies
        subAccuracies=[]

        # This list will hold TPR values
        TPR_arr = []

        # This list will hold FPR values
        FPR_arr = []

        # This will let you know what experiment you are on
        print("Starting Experiment "+str(k+1))

        # Gets the k row of the cluster data (33 random indices)
        test_data = cluster.iloc[k]

        # Gets everything else but the k row
        train_data = cluster.drop(cluster.index[k])

        # Create data frame with actual data
        TEST_df = CreateDataFrame(test_data, Normalized)
        # Print Test data labels to assure yourself, we are indeed changing test data each time
        print("Test data set labels are:")
        # Values are located in first column of data
        print(TEST_df[0].values)

        # Iterate through rows of the train data
        for i in range(len(train_data.index)):

            # Create dataframe of training data
            dummy = train_data.iloc[i]
            TRAIN_df = CreateDataFrame(dummy, Normalized)

            # the magic
            n, p = TRAIN_df.shape # number of samples and features

            y = np.zeros(n)
            y[TRAIN_df[0] > 5] = 1
            # data preparation for training data
            TRAIN_df.drop(TRAIN_df[0])
            X = TRAIN_df.iloc[:, 0:-1]
            X = pd.DataFrame(np.c_[np.ones(n), X])

            # data preparation for test data
            y_groundtruth = np.zeros(TEST_df.shape[0])
            y_groundtruth[TEST_df[0] > 5] = 1
            TEST_df.drop(TEST_df[0])
            X_test = TEST_df.iloc[:, 0:-1]
            X_test = pd.DataFrame(np.c_[np.ones(n), X_test])


            b_opt = SolverLinearRegression(X, y)
            # Append TPR values to corresponding array
            TPR_arr.append(float(sum(np.array(np.dot(X_test, b_opt) > 0.5) == y_groundtruth))/33)
            # Append FPR values to corresponding array
            FPR_arr.append( float(33 - sum(np.array(np.dot(X_test, b_opt) > 0.5) == y_groundtruth))/33)
            # Append Accuracies to corresponding array
            subAccuracies.append(sum(np.array(np.dot(X_test, b_opt) > 0.5) == y_groundtruth) / len(y_groundtruth))

        # Make a dictionary of data
        data = {'TPR':TPR_arr, 'FPR':FPR_arr, 'Accuracies':subAccuracies}
        # Create dataframe with the data
        TprFprFrame = pd.DataFrame(data)
        # Print it
        print(TprFprFrame)
        # Add a line feed for readability
        print("\n")
        # Append to the bigger Accuracies vector
        Accuracies.append(subAccuracies)

    # Once done looping, crete a dataframe with Accuracies
    df = pd.DataFrame(Accuracies)

    # aoee stands for Accuracies Of Each Experiment (AOEE)
    aoee = []
    # Loop through dataframe of Accuracies
    for i in range(len(df.index)):
        # Get the Mean of each row... which holds Accuracies of each sub experiment
        a = statistics.mean(df.iloc[i].values)
        # Append to Aoee
        aoee.append(a)
    # Cteate another data dict
    data = {'Averaged Accuracies':aoee,'Experiment': [1, 2,3,4,5,6,7,8,9,10]}
    # Create dataframe
    averages_of_experiments = pd.DataFrame(data)
    # Set index to experiments... so it looks nicer when opened in excel or numbers
    averages_of_experiments = averages_of_experiments.set_index('Experiment')
    # Show yourself what you gout.
    print("Averages of each experiment are:")
    print(averages_of_experiments)

def main():
    """
    Purpose: Download the MNIST data on the course web page. There are two files: MNIST_15_15.csv and
    MNIST_LABEL.csv. The former file contains hand-written digit image data (n = 335, p=15*15 pixel
    values) and the latter has the corresponding label of digit 5 or 6. Normalize the data (by min-max
    normalization, i.e. divide by 255) and train a linear model for classification (use a threshold of
    0.5). 10-fold cross-validation will be applied. Show a table of TPR, FPR, and accuracy for each
    experiment, and compute the average accuracy.
    """
    print("Hello World from LinRegScratch.py!")

    # Normalize da data
    Normalized = MNIST15_15_df/255
    #print(Normalized) # TEST PRINT

    # Get ze length of Normalized data
    len_Normalized = len(Normalized.index)
    #print(len_Normalized) # TEST Print

    k = 10

    members_in_cross_validation = int(len_Normalized/k)
    #print(members_in_cross_validation) # TEST PRINT should be 33

    # Create an Array of size 330 of Randomized indices from 0 to 334
    randomized_indices = list(random.sample(range(len_Normalized), (members_in_cross_validation*k)))

    # Split the array randomized_indices into k numpy arrays where k =10 in this case
    cross_validation_vectors = np.array_split(np.array(randomized_indices),k)

    # Create Dataframe of randomized indices called df
    df = pd.DataFrame(cross_validation_vectors)


    # Let the classification  begin
    LinearRegressionClassification(df,Normalized)


if __name__== "__main__":
    main()
