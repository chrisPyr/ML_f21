import matplotlib.pyplot as plt
import numpy as np
import sys
import random
import argparse
from scipy.special import expit
sys.path.append("../HW3/")
from HW3_1 import UniGaussian as UniGau

parser = argparse.ArgumentParser()
parser.add_argument("-N", help = "number of data", type = int, default = 50)
parser.add_argument("-mx1", help = "mean of x1", type = float, required = True)
parser.add_argument("-vx1", help = "variance of x1", type = float, required = True)
parser.add_argument("-mx2", help = "mean of x2", type = float, required = True)
parser.add_argument("-vx2", help = "variance of x2", type = float, required = True)
parser.add_argument("-my1", help = "mean of y1", type = float, required = True)
parser.add_argument("-vy1", help = "variance of y1", type = float, required = True)
parser.add_argument("-my2", help = "mean of y2", type = float, required = True)
parser.add_argument("-vy2", help = "variance of y2", type = float, required = True)
args = parser.parse_args()


def genData():
    x1_data=[]
    y1_data=[]
    x2_data=[]
    y2_data=[]
    for i in range(50):
        x1_data.append(UniGau(args.mx1,args.vx1))
        y1_data.append(UniGau(args.my1,args.vy1))
        x2_data.append(UniGau(args.mx2,args.vx2))
        y2_data.append(UniGau(args.my2,args.vy2))
    return x1_data, y1_data, x2_data, y2_data

def GradientDescent(Z,Y):
    w = np.random.rand(3,1)
    count = 0
    while True:
        count += 1
        dJ = np.matmul(Z.transpose(),(Y - expit(np.matmul(Z, w))) )
        if (abs(dJ) < 0.000001).all() or count >= 10000000:
            return w
        w = w + dJ

def NewtonMethod(Z,Y):
    w =  np.random.rand(3,1)
    count = 0
    while True:
        count+=1
        prev_w = w
        dJ = np.matmul(Z.transpose(),(Y - expit(np.matmul(Z, w))) )
        diagonal = np.zeros((2*args.N,2*args.N))
        for i in range(2*args.N):
            p = np.matmul(Z[i],w)
            diagonal[i,i] = p*(1-p)
        Hessian = np.matmul(Z.transpose(),np.matmul(diagonal,Z))
        try:
            w = w - np.matmul(np.linalg.inv(Hessian), dJ)
        except:
            w = w + dJ
        if (abs(prev_w - w)< 1e-10).all() or count>=10000:
            return w


def calConfusionMatrix(Z,Y,w,mode):
    if mode == 1:
        print("Gradient Descent:\n")
        print("w:")
        for i in range(3):
            print("{:.10f}".format(w[i][0]))
    else:
        print("Newton Method:\n")
        print("w:")
        for i in range(3):
            print("{:.10f}".format(w[i][0]))
    print()
    predict = np.zeros((2*args.N,1))
    for i in range(2*args.N):
        if np.matmul(Z[i],w) > 0:
            predict[i] = 1
        else:
            predict[i] = 0
    #cal TP, FN, TN, FP
    TP, FN, TN, FP = 0, 0 ,0 ,0
    for i in range(2*args.N):
        if Y[i] == 0 and predict[i] == 0:
            TN += 1
        elif Y[i] == 1 and predict[i] == 0:
            FN += 1
        elif Y[i] == 0 and predict[i] == 1:
            FP += 1
        else:
            TP += 1

    print("Confusion Matrix:")
    print("{:14}Predict Cluster 1   Predict Cluster 2".format(" "))
    print("In cluster 1 {:6}{}{:19}{}".format(" ",TN," ",FN))
    print("In cluster 2 {:6}{}{:19}{}".format(" ",FP," ",TP))
    #cal sensitivity and specificity
    print("Sensitivity (successfully predict cluster 1):{:.5f}".format(TN/(TN + FP)))
    print("Specificity (successfully predict cluster 2):{:.5f}".format(TP/(TP + FN)))
    return predict

def Visualization(x1,x2,y1,y2,pred_grad):
    plt.subplot(131)
    plt.title("Ground Truth")
    plt.scatter(x1,y1,color = "red")
    plt.scatter(x2,y2,color = "blue")
    plt.subplot(132)
    plt.title("Gradient Descent")
    col = np.where(pred_grad[0:args.N,0]==0, "red", "blue")
    plt.scatter(x1,y1, color = col)
    col = np.where(pred_grad[args.N:args.N*2,0]==0, "red", "blue")
    plt.scatter(x2,y2, color = col)
    plt.show()






if __name__ == "__main__":
    x1, y1, x2, y2 = genData()
    #Xw = w1x + w2y + w3
    Z = np.zeros((2*args.N, 3), dtype = float)
    Z[0:args.N,0] = x1
    Z[0:args.N,1] = y1
    Z[args.N:2*args.N,0] = x2
    Z[args.N:2*args.N,1] = y2
    Z[:,2] = 1

    # y = Bernoulli(f(Z))
    Y = np.zeros((2*args.N,1), dtype = int)
    Y[args.N:2*args.N, 0] = 1
    w_gradient = GradientDescent(Z,Y)
    w_newton = NewtonMethod(Z,Y)
    print(w_newton)
    pred_grad = calConfusionMatrix(Z,Y,w_gradient,1)
    print()
    pred_newton = calConfusionMatrix(Z,Y,w_newton,2)
    Visualization(x1,x2,y1,y2,pred_grad)


