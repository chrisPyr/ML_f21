import numpy as np
import matplotlib.pyplot as plt
from HW3_1 import PolyLinearData as polyData



import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-b", help = "precision", type = float, default = 1)
parser.add_argument("-a", help = "variance", type = float, default = 1)
parser.add_argument("-w", help = "paramete", nargs="+", type = float, default = 1)
parser.add_argument("-n", help = "fit n polynomial", type = int, default = 4)

args = parser.parse_args()

def calDesignMat(x, n):
    deMatrix = np.zeros((1,n))
    for k in range(n):
        deMatrix[0, k] = x**k

    return deMatrix

def printPosMean(pos_mean):
    for i in range(pos_mean.shape[0]):
        print("{:15.10f}".format(pos_mean[i,0]))

def printPosVar(pos_var):
    for i in range(pos_var.shape[0]):
        for j in range(pos_var.shape[1]-1):
            print("{:15.10f}".format(pos_var[i, j]), end =", ")
        print("{:15.10f}".format(pos_var[i, n-1]))

def BaysianLinearRegression(b, a, w, n):

    data_x = []
    data_y = []
    print_var = []
    print_var.append(a)
    print_mean = []
    print_mean.append(w)
    itr = 0
    prior_cov, pos_cov = np.identity(n),np.identity(n)
    prior_mean, pos_mean = np.zeros((n,1)), np.zeros((n,1))
    while True:
        x, y = polyData(n,a,w)
        data_x.append(x)
        data_y.append(y)
        designMatrix = calDesignMat(x, n)
        if itr == 0:
            # cov = aX^TX + bI
            pos_cov = a * np.matmul(np.transpose(designMatrix),designMatrix) + b*np.identity(n)
            # mean = a cov^-1 X^T y
            pos_mean = a * np.matmul(np.linalg.inv(pos_cov), np.transpose(designMatrix)) * y
        else:
            # cov = a X^T X + S
            pos_cov = a * np.matmul(designMatrix.transpose(),designMatrix) + prior_cov
            # mean = cov^-1 ( a X^T y + Sm)
            pos_mean = np.matmul(np.linalg.inv(pos_cov), a * designMatrix.transpose() * y + np.matmul(prior_cov, prior_mean))
        print("Add data point ({}, {}):".format(x,y))
        print()
        print("Posterior mean:")
        printPosMean(pos_mean)
        print()
        pos_var = np.linalg.inv(pos_cov)
        print("Posterior variance:")
        printPosVar(pos_var)
        predictive_mean = np.matmul(designMatrix, prior_mean)
        #predictive_var = 1/a + np.matmul(designMatrix, np.matmul(pos_var,designMatrix.transpose()))
        predictive_var = 1/a + np.matmul(designMatrix, np.matmul(np.linalg.inv(prior_cov),designMatrix.transpose()))
        print()
        print("Predictive distribution ~ N({:.5f}, {:.5f})".format(predictive_mean[0,0], predictive_var[0,0]))
        print("------------------------------------------------------")
        print()

        if itr == 10 or itr == 50:
            print_var.append(pos_var)
            print_mean.append(pos_mean)

        if (abs(pos_mean - prior_mean) < 0.00001).all() and  abs((pos_var - np.linalg.inv(prior_cov)) < 0.00001).all() and itr > 50:
            print_var.append(pos_var)
            print_mean.append(pos_mean)
            break
        prior_cov = pos_cov
        prior_mean = pos_mean
        itr+=1

    visualize(print_var,print_mean, data_x, data_y)

def calResulty(x,m):
    y = []
    mean = np.array(m)
    for k in range(100):
        func = calDesignMat(x[k],n)
        y.append(np.matmul(func, mean)[0][0])
    y = np.array(y)
    return y

def calVaronx(print_var, i, deMats):
    tmp_var = np.zeros(100)
    for k in range(100):
        tmp_var[k] = print_var[0]+ deMats[k].dot(print_var[i].dot(deMats[k].T))
    return tmp_var


def visualize(print_var,print_mean, data_x, data_y):
    plt.subplot(221)
    plt.title("Ground Truth")
    x = np.linspace(-2.0,2.0,100)
    deMats = []
    for k in range(100):
        deMats.append(calDesignMat(x[k], n))

    y = []
    mean = np.array(print_mean[0])
    for k in range(100):
        func = calDesignMat(x[k],n)
        y.append(np.matmul(func, mean.transpose()))
    y = np.array(y)
    var = print_var[0]
    drawResult(x,y,var)

    plt.subplot(222)
    plt.title("Predictive Result")
    y = calResulty(x, print_mean[3])
    var = calVaronx(print_var,3,deMats)
    plt.scatter(data_x, data_y, s = 7.0)
    drawResult(x,y,var)

    plt.subplot(223)
    plt.title("After 10 incomes")
    y = calResulty(x, print_mean[1])
    var = calVaronx(print_var,1,deMats)
    plt.scatter(data_x[:10], data_y[:10], s = 7.0)
    drawResult(x,y,var)

    plt.subplot(224)
    plt.title("After 50 incomes")
    y = calResulty(x, print_mean[2])
    var = calVaronx(print_var,2,deMats)
    plt.scatter(data_x[:50], data_y[:50], s = 7.0)
    drawResult(x,y,var)
    plt.show()

def drawResult(x,y,var):
    plt.plot(x,y, color = 'black')
    plt.plot(x,y+var, color = 'red')
    plt.plot(x,y-var, color = 'red')
    plt.xlim(-2,2)
    plt.ylim(-15,25)



if __name__ == "__main__":
    b = args.b
    a = args.a
    w = args.w
    n = args.n

    BaysianLinearRegression(b, a, w, n)
