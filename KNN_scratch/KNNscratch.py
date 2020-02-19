#AUTHOR: Eric Becerril-Blas <becere1@unlv.nevada.edu>
import pandas as pd
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from math import sqrt
from mpl_toolkits.mplot3d import Axes3D
from statistics import mode
import operator
from scipy.spatial import distance
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection, neighbors)
#Global Variables
# Create Dataframe for MNIST training data-------------
MNISTtrain_df = pd.read_csv("MNIST_training.csv")

# Create Dataframe for MNIST test data-----------------
MNISTtest_df = pd.read_csv("MNIST_test.csv")

def get_eucladian_distance(data1, data2):
    """
        Returns eucaladian distance of the data
            data1: np.array of row 1 of the test data
            data2: np.array of a row in the train data
    """
    dist = np.linalg.norm(data1 - data2) # using numpy linalg function
    return(dist)


def get_neighbors(train, row, k, dimensionality):
    """
        Locate the most similar neighbors
            train: the reduced training data set
            row: The row of interest
            k: the k value
            dimensionality: the dimension I am dealing with
    """
    distances = []
    labels = []
    neighbors =[]
    # 3 dimensions
    if dimensionality == 3:
        for label,x1,y1,z1 in zip(train['label'],train['x'],train['y'],train['z']):
            # zip is super quick to itereate through dataframes
            dist = get_eucladian_distance(row, np.array((x1 ,y1, z1))) # get euclidean distance
            distances.append(dist) # Append the label and the distance to distances list
            labels.append(label)
    # 2 dimensions
    else:
        for label,x1,y1 in zip(train['label'],train['x'],train['y']):
            # zip is supah quick
            dist = get_eucladian_distance(row, np.array((x1 ,y1))) # get euclideandistance
            distances.append(dist) # Append the label and the distance to distances list
            labels.append(label)
    data = {'label': labels, 'distance': distances}
    df = pd.DataFrame(data=data)
    # sort the data frame in ascending order
    df = df.sort_values(by=['distance'], ascending=True)
    df = df.reset_index()
    for i in range(k): # for range in the k value
        neighbors.append(df['label'][i]) # append the label to the neighbors list
    return neighbors #return the neighbors

def predict_classification(train, row, k, file, dimensionality):
    """
        Makes a classification prediction with neighbors
            train: the entirety of the reduced training data set
            row: a single row in the test dataset
            k: the k value
            file: the output.txt file
            dimensionality: the dimensions
    """
    neighbors = get_neighbors(train, row, k, dimensionality)

    try:
        # using the built in stat mode function, return the neighbor label that shows up the most
        return(mode(neighbors))
    except:
        # if there was no unique mode, return the smaller number
        return(compute_mode(neighbors))

def compute_mode(numbers):
    """
        statistic mode function throws an exception so built this to handle that
    """
    counts = {}
    maxcount = 0
    for number in numbers:
        if number not in counts:
            counts[number] = 0
        counts[number] += 1
        if counts[number] > maxcount:
            maxcount = counts[number]

    for number, count in counts.items():
        if count == maxcount:
            return(number)

def KNN(train, test, dimensionality, fname):
    """
        KNN written from scratch homie
            train: the reduced training data in its entirety
            test: the reduced test data in its entirety
            dimensionality: either a 2 or 3. Is it in 3D or 2D
            fname: the filenname of the output .txt file to see results
    """
    file = open(fname, "w+") # Create an output .txt file
    # We wanna see accuracy for various values for K so here he have a loop to go through many values
    for k in range(1,91):
        file.write("For k = "+str(k)+"\n") # Write to output file
        # If statement below For 3 Dimensions
        if dimensionality == 3:
            num_correct = 0 # variable used to count the number of correct predictions
            for label,x1,y1,z1 in zip(test['label'], test['x'], test['y'], test['z']):
                # For loops using zip are super fast, thats why I used it
                prediction = predict_classification(train, np.array((x1 ,y1, z1)), k, file, dimensionality)
                file.write('Expected '+str(label)+', Got '+str(prediction)+'.\n') #write to file
                if label == prediction:
                    # if the label equals the prediction label then you increment the number of correct
                    num_correct = num_correct + 1
            accur = (num_correct/50) * 100 # Calculate the accuracy
            file.write('Accuracy of KNN was '+str(accur)+'% with k equal to '+str(k)+'\n') # Write to output file total accuracy
            file.write('\n') # linefeed to output file to make a bit more readable
            print('Accuracy of KNN was '+str(accur)+'% with k equal to '+str(k)) # Print to terminal
        # Else statement For 2 Dimensions
        else:
            num_correct = 0 # variable used to count the number of correct predictions
            for label,x1,y1 in zip(test['label'], test['x'], test['y']):
                # zip is supah quick
                prediction = predict_classification(train, np.array((x1 ,y1)), k, file, dimensionality)
                file.write('Expected '+str(label)+', Got '+str(prediction)+'.\n') # Write to output file
                if label == prediction:
                    # if label matches prediction, increment the number of correct
                    num_correct = num_correct + 1
            accur = (num_correct/50) * 100 # Calculate the accuracy
            file.write('Accuracy of KNN was '+str(accur)+'% with k equal to '+str(k)+'\n') # output it to .txt
            file.write('\n') # linefeed to .txt
            print('Accuracy of KNN was '+str(accur)+'% with k equal to '+str(k)) # Print to command line
    file.close() # Close the outuput file

def reduce_data_to_2d(data, type):
    """
        Reduces the data to 2 dimensions and returns dataframe with labels and reduced data
        Takes 2 variables: data, type
            # data is the actual data
            # if type == "PCA", do PCA in 2D
            # if type == "t-SNE", do using t-SNE in 2D
    """
    # make two vectors from the MNIST Train data
    labels = data.iloc[:, 0] # label data
    images = data.drop('label', axis=1) # pixel values for image
    if type == "PCA":
        # Reduce the dimensionality to 2 dimensions using PCA
        pca = PCA(n_components=2)
        pca.fit(images)
        PCAX = pca.transform(images)
        #PCAX = decomposition.TruncatedSVD(n_components=2).fit_transform(images) #accuracy was lower with this :(
        columns = [ 'x', 'y'] # define column names
        df1 = pd.DataFrame(data=PCAX, columns=columns) # Make reduced data into a dataframe
        df = pd.concat([labels, df1], axis=1) # Make a new dataframe with the labels now added to it
    if type == "t-SNE":
        # Reduce the dimensionality to 2 dimensions using t-SNE
        #tsne = TSNE(n_components=2)
        #tsneX = tsne.fit_transform(images)
        tsne = manifold.TSNE(n_components = 2, init='pca', random_state=0)
        tsneX = tsne.fit_transform(images)
        columns = [ 'x', 'y'] # define column names
        df1 = pd.DataFrame(data=tsneX, columns=columns) # Make reduced data into a dataframe
        df = pd.concat([labels, df1], axis=1) # Make a new dataframe with the labels now added to it
    return df


def reduce_data_to_3d(data, type):
    """
        Reduces the data to 3 dimensions and returns dataframe with labels and reduced data
        Takes 2 variables: data, type
            # data is the actual data
            # if type == "PCA", do PCA in 3D
            # if type == "t-SNE", do using t-SNE in 3D
    """
    # make two vectors from the MNIST Train data
    labels = data.iloc[:, 0] # label data
    images = data.drop('label', axis=1) # pixel values for image
    if type == "PCA":
        # Reduce the dimensionality to 3 dimensions using PCA
        #pca = PCA(n_components=3)
        #pca.fit(images)
        #PCAX = pca.transform(images)
        PCAX = decomposition.TruncatedSVD(n_components=3).fit_transform(images)
        columns = [ 'x', 'y','z'] # declare column names
        df1 = pd.DataFrame(data=PCAX, columns=columns) # Make reduced data into a dataframe
        df = pd.concat([labels, df1], axis=1) # Make a new dataframe with the labels now added to it
    if type == "t-SNE":
        # Reduce the dimensionality to 3 dimensions using t-SNE
        #tsne = TSNE(n_components=3)
        #tsneX = tsne.fit_transform(images)
        tsne = manifold.TSNE(n_components=3, init='pca', random_state=0)
        tsneX = tsne.fit_transform(images)
        columns = [ 'x', 'y','z'] # declare column names
        df1 = pd.DataFrame(data=tsneX, columns=columns) # Make reduced data to dataframe
        df = pd.concat([labels, df1], axis=1) # Merge labels an
    return df

def reduction_2D(reduc_method, to_csv_name_train, to_csv_name_test, master_txt_file):
    """
        Reduces the Dataset to 2 dimensions
            reduc_method: is the method used to reduce, either PSA or t-SNE
            to_csv_name_train: is the name of the of the CSV file with train variables reduced
            to_csv_name_test: is the name of the of the CSV file with test variables reduced
            master_txt_file: text file where output will be outputted to
    """
    reduced_df_train = reduce_data_to_2d(MNISTtrain_df,reduc_method) # Reduce train data to 2 Dimensions
    reduced_df_train.to_csv(to_csv_name_train, index = False) # Write to a CSV file

    reduced_df_test = reduce_data_to_2d(MNISTtest_df,reduc_method) # reduce the dimensionality of test datato 2 dimensions
    reduced_df_test.to_csv(to_csv_name_test, index = False) # Write to CSV to see output of dimensionality reduction

    KNN(reduced_df_train, reduced_df_test, 2, master_txt_file) # KNN time

def reduction_3D(reduc_method, to_csv_name_train, to_csv_name_test, master_txt_file):
    """
        Reduces the Dataset to 3 dimensions
            reduc_method: is the method used to reduce, either PSA or t-SNE
            to_csv_name_train: is the name of the of the CSV file with train variables reduced
            to_csv_name_test: is the name of the of the CSV file with test variables reduced
            master_txt_file: text file where output will be outputted to
    """
    reduced_df_train = reduce_data_to_3d(MNISTtrain_df,reduc_method) # reduce the dimensionality of train data to 3 dimensions
    reduced_df_train.to_csv(to_csv_name_train, index = False) # Write to CSV to see output of dimensionality reduction

    reduced_df_test = reduce_data_to_3d(MNISTtest_df,reduc_method) # reduce the dimensionality of test data to 3 dimensions
    reduced_df_test.to_csv(to_csv_name_test, index = False) # Write to CSV to see output of dimensionality reduction

    KNN(reduced_df_train, reduced_df_test, 3, master_txt_file) #KNN time


def main():
    """
        Implement KNN from scratch using Python. You are given two data sets:
        MNIST_training.csv and MNIST_test.csv, where “MNIST_training.csv” contains training data
        that you will find the K-nearest neighbors, whereas “MNIST_test.csv” consists of test data that you need
        to predict labels. The training data contains 10 classes (i.e., 0, 1, 2, …, 9), each of which has 95 samples,
        while there are 5 samples on each class in the test data set
    """
    print("Hello World from KNNscratch.py Script!\n")

    # ---------------
    print("Commencing KNN using MNIST reduced to 2D by PCA")
    reduction_2D("PCA", "2dPCA_MNISTtrain_data.csv", "2dPCA_MNISTtest_data.csv", "AccuracyWithPCA2D.txt")
    print("2D PCA finished\n")

    # ---------------
    print("Commencing KNN using MNIST reduced to 3D by PCA")
    reduction_3D("PCA", "3dPCA_MNISTtrain_data.csv", "3dPCA_MNISTtest_data.csv", "AccuracyWithPCA3D.txt")
    print("3D PCA finished\n")

    # ---------------
    print("Commencing KNN using MNIST reduced to 2D by t-SNE")
    reduction_2D("t-SNE", "2dtsne_MNISTtrain_data.csv", "2dtsne_MNISTtest_data.csv",  "AccuracyWithtSNE2D.txt")
    print("2D t-SNE finished\n")

    # ---------------
    print("Commencing KNN using MNIST reduced to 3D by t-SNE")
    reduction_3D("t-SNE", "3dtsne_MNISTtrain_data.csv", "3dtsne_MNISTtest_data.csv","AccuracyWithtSNE3D.txt")
    print("3D t-SNE finished\n")

    # ---------------
    print("KNNscratch.py is finished, have a great day! :)")


if __name__== "__main__":
    main()
