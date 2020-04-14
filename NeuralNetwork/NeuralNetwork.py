#AUTHOR: Eric Becerril-Blas
from __future__ import absolute_import, division, print_function, unicode_literals
import functools
import numpy as np
import tensorflow as tf
import pandas as pd
import tensorflow.keras as keras
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold # import KFold
import random
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import RMSprop
from keras import backend as K
from keras.utils import to_categorical
from tensorflow.keras.losses import categorical_crossentropy


# declaring hyper parameters
BATCH_SIZE = 100
EPOCHS = 200
DIMENSION = 28


def CreateDataFrame(data, MNIST):
    """
        Combines the labels and the normalized data into one data frame and returns it
    """
    # Verify Length of data
    #print(len(data.values))

    # Create numpy array of data and print to verify
    dummy = data.values
    #print(dummy)

    # Create 2 empty lists. 1 for labelz and the other for pixel data
    picz = []

    # Loop through indices of K Cross validation Vector and put them in assigned vector
    for i in range( len(dummy) ):
        picz.append(MNIST.iloc[dummy[i]].values)

    #print(labelz)
    pixels_df = pd.DataFrame(picz)
    #pixels_df.columns = range(pixels_df.shape[1])
    return(pixels_df)


def MNIST_classification(cluster, MNIST):

    for k in range(len(cluster.index)):

        # This will let you know what experiment you are on
        print("Starting Experiment "+str(k+1))

        # Gets the k row of the cluster data (200 random indices)
        test_data = cluster.iloc[k]
        #print("Test data is")
        #print(test_data)

        # Gets everything else but the k row
        train_data = cluster.drop(cluster.index[k])
        #print("train data is")
        #print(train_data)

        # Create data frame with actual data
        TEST_df = CreateDataFrame(test_data, MNIST)
        #print("Test DF is")
        #print(TEST_df)

        # Print Test data labels to assure yourself, we are indeed changing test data each time
        #print("Test data set labels are:")
        test_labels = TEST_df[0].values
        test_labels = to_categorical(test_labels)

        # Values are located in first column of data
        #print(TEST_df[0].values)
        #print("Test Labels are")
        #print(test_labels)

        TEST_df = TEST_df.drop(columns=[0])
        TEST_df = TEST_df.values
        TEST_df = TEST_df.reshape((199, 28, 28, 1))
        TEST_df = TEST_df.astype('float32') / 255
        #print("New test df")
        #print(TEST_df)

        # Iterate through rows of the train data
        for i in range(len(train_data.index)):

            # Create dataframe of training data
            dummy = train_data.iloc[i]
            TRAIN_df = CreateDataFrame(dummy, MNIST)

            #print("Train dataframe is")
            #print(TRAIN_df)

            # Print Train data labels to assure yourself, we are indeed changing test data each time
            #print("Train data set labels are:")
            train_labels = TRAIN_df[0].values
            # Values are located in first column of data
            #print(TEST_df[0].values)
            #print("Train Labels are")
            #print(train_labels)

            TRAIN_df = TRAIN_df.drop(columns=[0])
            TRAIN_df = TRAIN_df.values

            TRAIN_df = TRAIN_df.reshape((199, 28, 28, 1))
            TRAIN_df = TRAIN_df.astype('float32') / 255

            train_labels = to_categorical(train_labels)

            model = Sequential()

            model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(28, 28, 1)))

            model.add(MaxPooling2D((2, 2)))

            #model.add(Dropout(0.25))

            model.add(Conv2D(64, (5, 5), activation='relu'))

            model.add(MaxPooling2D((2, 2)))

            model.add(Flatten())

            model.add(Dense(10, activation='softmax'))

            model.summary()

            model.compile(loss= categorical_crossentropy, optimizer='sgd',metrics=['accuracy'])

            history = model.fit(TRAIN_df, train_labels, epochs=EPOCHS, batch_size = BATCH_SIZE, verbose = 1, validation_split=0.1)

            val_loss, val_acc = model.evaluate(TEST_df, test_labels)  # evaluate the out of sample data with model
            print('Test loss',val_loss)  # model's loss (error)
            print('Test accuracy',val_acc)  # model's accuracy
    print("\n")
    print(history.history.keys())

    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def main():
    """
        implement a neural network for the handwritten digit classification
        problem with the MNIST data. Please use the MNIST data for HW4 that includes 100 images on
        each label of 0 – 9.

        You should implement a neural network (NN) and compute accuracy using 5-fold CV to
        compare their performance. You can design the network by yourself. You must clearly explain
        the architecture of your neural network. You can implement the neural network using any deep
        learning frameworks (e.g., keras, pytorch, tensorflow)
    """

    print("Hello World from NeuralNetwork.py!")
    print(tf.__version__)

    # Create Dataframe from given dataset
    MNIST = pd.read_csv("MNIST_HW4.csv")

    # Test outputs
    #print(MNIST)

    # Get ze length of the data
    length_of_data = len(MNIST.index)
    #print(length_of_data)

    k = 5

    members_in_cross_validation = int(length_of_data/k)
    #print(members_in_cross_validation) # TEST PRINT should be 999

    # Create an Array of size 199 of Randomized indices from 0 to 990
    randomized_indices = list(random.sample(range(length_of_data), (members_in_cross_validation*k)))

    # Split the array randomized_indices into k numpy arrays where k = 5 in this case
    cross_validation_vectors = np.array_split(np.array(randomized_indices),k)
    #print(cross_validation_vectors)

    # Create Dataframe of randomized indices called df
    df = pd.DataFrame(cross_validation_vectors)
    #print(df)

    # Let the classification  begin
    MNIST_classification(df,MNIST)




if __name__== "__main__":
    main()
