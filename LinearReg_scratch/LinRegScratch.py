#AUTHOR: Eric Becerril-Blas
import pandas as pd
import numpy as np
import math
import random
#import matplotlib.pyplot as plt
import statistics


# Datasets given to me from course web page
MNIST15_15_df = pd.read_csv("MNIST_15_15.csv", header = None)
MNIST15_15_labels = pd.read_csv("MNIST_LABEL.csv", header = None)


def generateImages(data):
    #print(data.loc[0:0,1:])
    data[0].to_csv("./images/labels")
    for i in range(len(data.index)):
        img = np.array(data.loc[i:i,1:]).reshape(15, 15)
        #plt.imshow(img)
        #plt.show()
        #plt.savefig("./images/"+str(i)+".png")

def SolverLinearRegression(X, y):
    XTdotX = np.dot(X.transpose(), X)
    #print("np.dot(X.transpose(), X) is:")
    #print(XTdotX)
    #Using pinv for psuedo inverse since most of these give me vectors with determinant equal to 0
    return np.dot(np.dot(np.linalg.pinv(XTdotX), X.transpose()), y)


def CreateDataFrame(data, Normalized):
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
    Accuracies = []
    for k in range(len(cluster.index)):
        subAccuracies=[]
        TPR_arr = []
        FPR_arr = []
        #f = open("experiment"+str(k+1)+".txt", "w")
        print("Starting Experiment "+str(k+1))

        # Gets the k row of the cluster data (33 random indices)
        test_data = cluster.iloc[k]
        #print("Test data is:")
        #print(test_data)

        # Gets everything else but the k row
        train_data = cluster.drop(cluster.index[k])
        #print("Train Data is:")
        #print(train_data)

        # Create data frame with actual data
        TEST_df = CreateDataFrame(test_data, Normalized)
        print("Test data set labels are:")
        print(TEST_df[0].values)
        #f.write("Test dataframe being used is:\n")
        #f.write(str(TEST_df.values)+"\n")
        #print("Test dataframe with Normalized pixel values:")
        #print(TEST_df)

        # I was curious to see what 15x15 image looked like
        #generateImages(TEST_df)

        # Iterate through rows of
        for i in range(len(train_data.index)):
            #f.write("Training with \n")

            # Create dataframe of training data
            dummy = train_data.iloc[i]
            TRAIN_df = CreateDataFrame(dummy, Normalized)
            #print("Train dataframe with normalized pixel values")
            #print(TRAIN_df)
            #f.write(str(TRAIN_df.values)+"\n")

            # FROM KANGS NOTES ... Aka magic
            n, p = TRAIN_df.shape # number of samples and features

            # if y is greater than 5, class will be 1 otherwise 0
            # we convert a regression problem to a classification problem (discretization)
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
            #print("bopt is ")
            #print(b_opt)

            #print("np.dot(X_test, b_opt) is")
            #print(np.dot(X_test, b_opt))

            #print("np.array(np.dot(X_test, b_opt) > 0.5) is")
            #print(np.array(np.dot(X_test, b_opt) > 0.5))

            #print("sum(np.array(np.dot(X_test, b_opt) > 0.5) == y_groundtruth) is")
            #print(sum(np.array(np.dot(X_test, b_opt) > 0.5) == y_groundtruth))
            TPR_arr.append(float(sum(np.array(np.dot(X_test, b_opt) > 0.5) == y_groundtruth))/33)
            FPR_arr.append( float(33 - sum(np.array(np.dot(X_test, b_opt) > 0.5) == y_groundtruth))/33)
            SUMMER = sum(np.array(np.dot(X_test, b_opt) > 0.5) == y_groundtruth) / len(y_groundtruth)
            subAccuracies.append(SUMMER)
            #print("Accuracy is "+str(SUMMER))
            #f.write("Accuracy is "+str(SUMMER)+"\n")
            # visualization of the result
#            plt.plot(np.dot(X_test, b_opt), 'ro', y, 'bo')
            #plt.show()
        # intialise data of lists.
        data = {'TPR':TPR_arr, 'FPR':FPR_arr, 'Accuracies':subAccuracies}
        TprFprFrame = pd.DataFrame(data)
        print(TprFprFrame)
        print("\n")
        Accuracies.append(subAccuracies)
        #f.close()
    print("Accuracies of each experiment are: ")
    df = pd.DataFrame(Accuracies)
    print(df)
    aoee = []
    for i in range(len(df.index)):
        a = statistics.mean(df.iloc[i].values)
        aoee.append(a)
    #print(aoee)
    data = {'Averaged Accuracies':aoee,'Experiment': [1, 2,3,4,5,6,7,8,9,10]}
    averages_of_experiments = pd.DataFrame(data)
    averages_of_experiments = averages_of_experiments.set_index('Experiment')
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

    threshold = 0.5
    k = 10

    members_in_cross_validation = int(len_Normalized/k)
    #print(members_in_cross_validation) # TEST PRINT should be 33

    # Create an Array of size 330 of Randomized indices from 0 to 334
    randomized_indices = list(random.sample(range(len_Normalized), (members_in_cross_validation*k)))

    # Split the array randomized_indices into k numpy arrays where k =10 in this case
    cross_validation_vectors = np.array_split(np.array(randomized_indices),k)

    # Create Dataframe of randomized indices called df
    df = pd.DataFrame(cross_validation_vectors)

    #print(df) # TEST PRINT

    # Let the classification  begin
    LinearRegressionClassification(df,Normalized)



if __name__== "__main__":
    main()
