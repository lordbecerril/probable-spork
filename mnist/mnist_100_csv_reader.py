import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():
    '''
        Following function is the main function that contains the essence of what this script will be doing. Reading the MNIST_100.csv
    '''
    print("Hello from MNIST_100.csv reader!")
    # Create Dataframe
    df = pd.read_csv("MNIST_100.csv")
    print(df)
    #Create axis
    y = df.iloc[:, 0]
    X = df.drop('label', axis=1)
    img = np.array(X[0:1]).reshape(28, 28) / 255
    plt.imshow(img)
    plt.show()

if __name__== "__main__":
    main()
