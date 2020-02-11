#AUTHOR: Eric Becerril-Blas <becere1@unlv.nevada.edu>
#ML Housing dataset (UCI Machine Learning database): https://github.com/rupakc/UCI-Data-Analysis/blob/master/Boston%20Housing%20Dataset/Boston%20Housing/UCI%20Machine%20Learning%20Repository_%20Housing%20Data%20Set.pdf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from time import time
from matplotlib.ticker import NullFormatter
from sklearn import manifold, datasets
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE

def plot_2d_data(data, title, pathname):
    '''
        Plots the data in 2d with given data from task 1 and task 2 and then use plt.clf() to clear current figure
    '''
    i = 0
    k = 0
    while i < 1000:
        j = i + 100
        plt.scatter(data[i:j, 0], data[i:j, 1], marker='$'+str(k)+'$')
        k = k + 1
        i = i + 100
    plt.title(title)
    plt.savefig(pathname)
    plt.clf()

def plot_3d_data(data, title, pathname):
    '''
        Plots the data in 3d with given data from task 1 and task 2 and then use plt.clf() to clear current figure
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    i = 0
    k = 0
    while i < 1000:
        j = i + 100
        ax.scatter(data[i:j, 0], data[i:j, 1], data[i:j, 2], marker='$'+str(k)+'$')
        k = k + 1
        i = i + 100
    plt.title(title)
    plt.savefig(pathname)
    plt.clf()


def task1(data, labels, images):
    '''
        Visualize the MNIST data using PCA. Reduce the data dimension to two or three and plot the
        data of reduced dimension. Must plot all the data of ten groups (0 to 9)
    '''
    pca = PCA(n_components=2)
    pca.fit(images)
    PCAX = pca.transform(images)
    plot_2d_data(PCAX, "2D PCA of MNIST",'MNISTpca2D.png')
    pca = PCA(n_components=3)
    pca.fit(images)
    PCAX = pca.transform(images)
    plot_3d_data(PCAX, "3D PCA of MNIST",'MNISTpca3D.png')


def task2(data, labels, images):
    '''
        Visualize the MNIST data using t-SNE library.
    '''
    tsne = TSNE(n_components=2)
    tsneX = tsne.fit_transform(images)
    plot_2d_data(tsneX, "2D t-SNE of MNIST",'MNISTt_SNE2D.png')
    tsne = TSNE(n_components=3)
    tsneX = tsne.fit_transform(images)
    plot_3d_data(tsneX, "3D t-SNE of MNIST",'MNISTt_SNE3D.png')

def task3(data):
    '''
        Visualize the housing data using violin plot.
        1. CRIM      per capita crime rate by town
        2. ZN        proportion of residential land zoned for lots over
                     25,000 sq.ft.
        3. INDUS     proportion of non-retail business acres per town
        4. CHAS      Charles River dummy variable (= 1 if tract bounds
                     river; 0 otherwise)
        5. NOX       nitric oxides concentration (parts per 10 million)
        6. RM        average number of rooms per dwelling
        7. AGE       proportion of owner-occupied units built prior to 1940
        8. DIS       weighted distances to five Boston employment centres
        9. RAD       index of accessibility to radial highways
        10. TAX      full-value property-tax rate per $10,000
        11. PTRATIO  pupil-teacher ratio by town
        12. B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks
                     by town
        13. LSTAT    % lower status of the population
        14. MEDV     Median value of owner-occupied homes in $1000's
    '''
    # array variable with each column of housing data
    all_data = [data["CRIM"], data["ZN"], data["INDUS"], data["CHAS"], data["NOX"], data["RM"], data["AGE"], data["DIS"], data["RAD"],data["TAX"],data["PTRATIO"],data["B"],data["LSTAT"],data["MEDV"]]

    # The labels for the x-axis
    x_labels = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT","MEDV"]

    plt.xticks(ticks = [y + 1 for y in range(len(all_data))],labels = x_labels)

    # Create the violin plots
    plt.violinplot(all_data,
                   showmeans=True,
                   showmedians=True)
    plt.title("Violin Plot of Boston Housing Data")
    #plt.show() #uncomment to see Violin Plot
    plt.tight_layout()
    plt.savefig('ViolinPlot.png')


def main():
    '''
        The main body of the program which runs all 3 tasks

        sample output should look as follows

            Hello World from HW1.py Script!

            2D PCA created and can be viewed in MNISTpca2D.png

            3D PCA created and saved as MNISTpca3D.png

            2D t-SNE created and can be viewed in MNISTt_SNE2D.png

            3D t-SNE created and saved as MNISTt_SNE3D.png

            Violin Plot created and can be viewed in ViolinPlot.png

            HW1.py is finished. Have a good day! :)

    '''
    print("Hello World from HW1.py Script!\n")

    # Create Dataframe for MNIST data
    MNIST_df = pd.read_csv("MNIST_100.csv")

    # make two variables - X and y for the MNIST dataset
    y = MNIST_df.iloc[:, 0] # label data
    X = MNIST_df.drop('label', axis=1) # pixel values for image

    # Create Dataframe for housing dataset
    HOUSING_df = pd.read_csv("housing_training.csv",names=["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT","MEDV"])

    # Call functions for each task
    task1(MNIST_df,y,X)
    print("2D PCA created and saved as MNISTpca2D.png\n")
    print("3D PCA created and saved as MNISTpca3D.png\n")

    task2(MNIST_df,y,X)
    print("2D t-SNE created and saved as MNISTt_SNE2D.png\n")
    print("3D t-SNE created and saved as MNISTt_SNE3D.png\n")


    task3(HOUSING_df)
    print("Violin Plot created and saved as ViolinPlot.png\n")

    print("HW1.py is finished. Have a good day! :)")


if __name__== "__main__":
    main()
