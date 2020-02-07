#AUTHOR: Eric Becerril-Blas <becere1@unlv.nevada.edu>
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

def main():
    '''
        The main body of the program
    '''
    print("Hello World from KNNscratch.py Script!")

    # Create Dataframe for MNIST test data
    MNISTtest_df = pd.read_csv("MNIST_test.csv")

    # Create Dataframe for MNIST training data
    MNISTtrain_df = pd.read_csv("MNIST_train.csv")


if __name__== "__main__":
    main()
