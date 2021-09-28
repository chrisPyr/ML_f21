#!/usr/bin/env python3

import sys
import numpy as np

#np.set_printoptions(precision = 20)

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

def LSE(AT,A,B):
    matrix =  matmul(AT,A)
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

    ans = LU_inverse(L, U, matmul(AT,B))
    err = error_sum(ans, A, B)
    print(ans)
    print("Fitting", end=" ")
    for x in range(len(ans)-1):
        if(ans[x])
        print("%.11f"%float(ans[x])+"X^"+str(len(ans)-x-1)+" + ", end=" ")

    print("Total Error: ", err)


    return
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



def plot():
    return

if __name__ == '__main__':
    #A = np.matrix([[1,4,-3],[-2,8,5],[3,4,7]])

    if len(sys.argv) < 2:
        print("Please provide data file\n")
        exit()

    A, B = readdata(sys.argv[1])
    A = np.asarray(A)
    A_extend = np.zeros((A.shape[0],int(sys.argv[2])))
    for z in range(A.shape[0]):
        for k in range(int(sys.argv[2])):
            A_extend[z,k] = A[z]**(int(sys.argv[2])-1-k)

    #Amatrix = np.matrix(A)
    AT = np.transpose(A_extend)
    B = np.asarray(B)
    #print(arg)
    #print(AT * A)
    LSE(AT,A_extend,B)
    #Bmatrix = np.matrix(B)
    plot()
