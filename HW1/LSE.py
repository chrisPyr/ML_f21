#!/usr/bin/env python3

import sys
import numpy as np
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="HW1 program description")
parser.add_argument("--file", type = str, default = "test.txt")
parser.add_argument("--n", type = int,required = True, help="require input n(basis number)")
parser.add_argument("--lamb", type = float, required = True, help = "require input lambda")
args = parser.parse_args()
#np.set_printoptions(precision = 20)

def print_ans(ans,err):

    print("Fitting", end=" ")
    for x in range(len(ans)-1):
        if x < len(ans):
            sign = ""
            if float(ans[x+1]) > 0:
                sign = "+"
            print("%.11f"%float(ans[x])+"X^"+str(len(ans)-x-1)+sign, end=" ")
    sign = ""
    if float(ans[len(ans)-1]) > 0:
        sign = "+"
    print(sign+"%.11f"%float(ans[len(ans)-1]))

    print("Total Error: ", float(err))
    return



def matmul(A, B):
    row = len(A)
    col = len(B[0])
    col_A = len(A[0])
    assert col_A==len(B)

    C = np.zeros((row, col))
    for k in range(col_A):
        for i in range(row):
            for j in range(col):
                C[i,j]=C[i,j]+A[i,k]*B[k,j]
    return C

def readdata(fname):
    file = open(fname,'r')
    A = []
    B = []
    x = file.readline()
    while x:
        a, b = x.strip().split(",")
        A.append(list([float(a)]))
        B.append(list([float(b)]))
        x = file.readline()
    file.close()

    return  A, B

def identity(dim):
    id = (dim,dim)
    idma = np.zeros(id)
    for x in range(dim):
        idma[x,x] = 1
    return idma

def LU_decomposition(matrix):
    num_row,num_col = matrix.shape
    rows = np.size(matrix,0)
    L = identity(num_row)
    U = matrix.copy()

    for x in range(rows-1):
        base = U[x,x]
        for y in range(x,rows-1):
            z = y+1
            re = (U[z,x] / base)
            L[z,x] = L[x,x] * re + L[z,x]
            U[z,:] = U[x,:] * -re + U[z,:]
    return L,U

def LSE(AT,A,B, lamb):
    matrix =  matmul(AT,A)
    for x in range(matrix.shape[0]):
        matrix[x,x] += lamb
    L, U = LU_decomposition(matrix)

    ans = LU_inverse(L, U, matmul(AT,B))
    err = error_sum(ans, A, B)
    print("LSE:")
    print_ans(ans,err)

    return ans

def Newton_method(AT,A,B,lamb):

    x = np.random.rand(A.shape[1],1)
    #H(f(x))
    Hessian = 2* matmul(AT,A)
    #gradient
    target = matmul(Hessian,x) - 2 * matmul(AT,B)
    L,U = LU_decomposition(Hessian)
    ans = x - LU_inverse(L,U,target)
    err = error_sum(ans, A, B)
    print("Newton's method:")
    print_ans(ans,err)
    return ans

'''
def LU_inverse(L, U):
    L_in = identity(L.shape[0])
    L_tmp = L.copy()
    U_in = identity(U.shape[0])
    U_tmp = U.copy()
    for x in range(L.shape[0]-1):
        basey = U_tmp[U.shape[0]-1-x,U.shape[0]-1-x]
        U_in[U.shape[0]-1-x,:] = U_in[U.shape[0]-1-x,:] * (1/basey)
        for y in range(x, L.shape[0]-1):
            zx = y+1
            zy = U.shape[0]-1-(y+1)
            rex = (L_tmp[zx,x])
            rey = (U_tmp[zy,U.shape[0]-1-x])
            L_in[zx,:] = L_in[x,:] * -rex + L_in[zx,:]
            U_in[zy,:] = U_in[U.shape[0]-1-x,:] * -rey + U_in[zy,:]
    U_in[0,:] = U_in[0,:] / U_tmp[0,0]
    return L_in, U_in
'''

def LU_inverse(L,U,B):
    # Ax = b, L(Ux) = B, Ly =B
    B_derive = B.copy()
    for x in range(B.shape[0]-1):
        for y in range(x,B.shape[0]-1):
            z = y+1
            re = L[z,x]
            B_derive[z,:] = B_derive[x,:] * -re + B_derive[z,:]

    # Ux = y
    U_tmp = U.copy()
    for x in range(U.shape[0]-1,-1,-1):
        base = U_tmp[x,x]
        B_derive[x,:] = B_derive[x,:] / base
        for y in range(x-1,-1,-1):
            re = U_tmp[y,x]
            B_derive[y,:] = B_derive[x,:] * -re + B_derive[y,:]

    return B_derive

def error_sum(para, A,B):
    sum = 0
    estimate = matmul(A,para)
    for x in range(A.shape[0]):
        sum += (estimate[x] - B[x])**2

    return sum



def plot(A, B, A_extend, ans, ans_new):
    plt.figure()
    xmin = min(A)
    xmax = max(A)
    t = 0
    x_label = np.arange(xmin-1,xmax+1,0.1)
    plt.xlim(xmin-1,xmax+1)
    plt.subplot(211)
    plt.scatter(A,B)
    for x in range(len(ans)):
        t += ((x_label)**x)*ans[len(ans)-x-1]

    plt.plot(x_label,t, color = 'red')
    plt.subplot(212)
    plt.scatter(A,B)
    t = 0
    for x in range(len(ans_new)):
        t += ((x_label)**x)*ans_new[len(ans_new)-x-1]
    plt.plot(x_label,t, color = 'red')
    plt.show()
    return

if __name__ == '__main__':
    A, B = readdata(args.file)
    A = np.asarray(A)
    A_extend = np.zeros((A.shape[0],args.n))
    for z in range(A.shape[0]):
        for k in range(args.n):
            A_extend[z,k] = A[z]**(args.n-1-k)
    lamb = args.lamb
    AT = np.transpose(A_extend)
    B = np.asarray(B)
    ans = LSE(AT,A_extend,B,lamb)
    ans_new = Newton_method(AT,A_extend,B,lamb)
    plot(A, B,A_extend, ans, ans_new)
