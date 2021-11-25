import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist


def readData():
    x = []
    y = []
    with open("data/input.data", "r") as file:
        line = file.readline()
        while line:
            tmp1, tmp2 = line.strip().split(" ")
            x.append(float(tmp1))
            y.append(float(tmp2))
            line = file.readline()
    x = np.array(x)
    x = x[:,np.newaxis]
    y = np.array(y)
    y = y[:,np.newaxis]
    return x,y

def rationalKernel(xn, xm, l = 1.0, alpha = 1.0 ):
    return (1+ cdist(xn,xm,"sqeuclidean")/(2*alpha*(l**2)))**(-alpha)

if __name__ == "__main__":
    train_x, train_y = readData()
    z = rationalKernel(train_x, train_x)
    print(z)
