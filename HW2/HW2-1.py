#!/usr/bin/env python3

import struct
import sys
import numpy as np
import argparse
import matplotlib.pyplot as plt
import pathlib

NUM_OF_BINS = 32
NUM_OF_LABELS = 10
PI = 3.14159265
cu_dir = pathlib.Path().resolve()
parser = argparse.ArgumentParser()
parser.add_argument("--re_train", help = "true:1, false:0, recalculate or not(take time)", type = int, default = 0 )
parser.add_argument("--mode", help = "discrete:0 ,continous:1", type = int, required = True)
args = parser.parse_args()

def readtest():
    with open(str(cu_dir)+"/data/t10k-images-idx3-ubyte",'rb') as data:
        magic = int.from_bytes(data.read(4), byteorder='big')
        if(magic != 2051):
            print("not MNIST format")
            exit()
        num_of_images = int.from_bytes(data.read(4), byteorder='big')
        row_of_image = int.from_bytes(data.read(4),byteorder='big')
        col_of_image = int.from_bytes(data.read(4),byteorder='big')
        images_data = np.fromfile(data, dtype = np.ubyte, count = -1)
        images = images_data.reshape(num_of_images, row_of_image*col_of_image)
    with open(str(cu_dir)+"/data/t10k-labels-idx1-ubyte",'rb') as label:
        magic = int.from_bytes(label.read(4), byteorder='big')
        if(magic != 2049):
            print("not MNIST format")
            exit()
        num_of_images = int.from_bytes(label.read(4), byteorder='big')
        labels_data = np.fromfile(label, dtype = np.ubyte, count = -1)
        labels = labels_data.reshape(num_of_images,1)
    return images, labels

def readdata():
    with open(str(cu_dir)+"/data/train-images-idx3-ubyte",'rb') as data:
        magic = int.from_bytes(data.read(4), byteorder='big')
        if(magic != 2051):
            print("not MNIST format")
            exit()
        num_of_images = int.from_bytes(data.read(4), byteorder='big')
        row_of_image = int.from_bytes(data.read(4),byteorder='big')
        col_of_image = int.from_bytes(data.read(4),byteorder='big')
        images_data = np.fromfile(data, dtype = np.ubyte, count = -1)
        images = images_data.reshape(num_of_images, row_of_image*col_of_image)
    with open(str(cu_dir)+"/data/train-labels-idx1-ubyte",'rb') as label:
        magic = int.from_bytes(label.read(4), byteorder='big')
        if(magic != 2049):
            print("not MNIST format")
            exit()
        num_of_images = int.from_bytes(label.read(4), byteorder='big')
        labels_data = np.fromfile(label, dtype = np.ubyte, count = -1)
        labels = labels_data.reshape(num_of_images,1)
    return images, labels

def cal_prior(labels):
    prior = np.zeros(10, dtype= float)
    for x in range(len(labels)):
        prior[labels[x]]+=1

    prior = np.divide(prior, len(labels))
    return prior

def cal_likelihood(images, labels):

    likelihood = np.zeros((NUM_OF_LABELS,len(images[0]), NUM_OF_BINS), dtype = float)

    for x in range(len(labels)):
        for y in range(len(images[1])):
            Bin = images[x][y] //8
            likelihood[labels[x,0]][y][Bin]+=1


    min = np.amin(likelihood[np.nonzero(likelihood)])
    for x in range(likelihood.shape[0]):
        for y in range(likelihood.shape[1]):
            for z in range(likelihood.shape[2]):
                if likelihood[x,y,z] == 0:
                    likelihood[x,y,z] = min

    sum = np.sum(likelihood,axis=2)
    for x in range(likelihood.shape[0]):
        for y in range(likelihood.shape[1]):
            likelihood[x][y][:] /= sum[x][y]


    return likelihood

def cal_posterior(prior, likelihood, test_images, mean, var, mode):
    posteriors = np.zeros((test_images.shape[0],NUM_OF_LABELS), dtype = float)

    if mode == 0:
        for x in range(test_images.shape[0]):
            posteriors[x,:] += np.log(prior)
            for y in range(NUM_OF_LABELS):
                for z in range(test_images.shape[1]):
                    test_bin = test_images[x,z] //8
                    posteriors[x,y] += np.log(likelihood[y,z,test_bin])
            posteriors[x] /= sum(posteriors[x,:])
        return posteriors
    else:
        for x in range(test_images.shape[0]):
            posteriors[x,:]+=np.log(prior)
            for y in range(NUM_OF_LABELS):
                for z in range(test_images.shape[1]):
                    if var[y,z] == 0:
                        continue
                    posteriors[x,y] -= np.log(2.0*PI*var[y,z]) /2.0
                    posteriors[x,y] -= (((test_images[x,z] - mean[y,z])**2)/(2.0*var[y,z]))
            posteriors[x] /= sum(posteriors[x,:])
        print(posteriors)
        return posteriors

def cal_error_rate(posteriors, labels):

    error_rate = 0
    for x in range(posteriors.shape[0]):
        print("Posterior (in log scale)")
        for y in range(posteriors.shape[1]):
            print("%s: %s" %(y,posteriors[x,y]))
        predict = np.argmin(posteriors[x])
        if predict != labels[x]:
            error_rate += 1
        print("Prediction: %s, Ans: %s" %(predict, int(labels[x])))
        print()
    error_rate/= posteriors.shape[0]
    return error_rate

def cal_mean_var(images, labels, prior):
    mean = np.zeros((NUM_OF_LABELS, images.shape[1]), dtype = float)
    var = np.zeros((NUM_OF_LABELS, images.shape[1]), dtype = float)

    for x in range(images.shape[0]):
        for y in range(images.shape[1]):
            mean[labels[x],y] += images[x,y]

    for x in range(NUM_OF_LABELS):
        mean[x,:] /= (prior[x]*images.shape[0])

    for x in range(images.shape[0]):
        for y in range(images.shape[1]):
            var[labels[x], y] += (images[x,y] - mean[labels[x], y])**2

    for x in range(NUM_OF_LABELS):
        var[x,:] /= (prior[x]*images.shape[0])

    return mean, var



def printPredictDiscrete(likelihood, error_rate, mean, mode):
    print("imagination of numbers in Bayesian classifier:")
    if mode == 0:
        for x in range(likelihood.shape[0]):
            print("%s:"%(x))
            imagination = np.zeros(likelihood.shape[1])
            for y in range(likelihood.shape[1]):
                max = np.argmax(likelihood[x,y])
                imagination[y] = max
            imagination = np.reshape(imagination, (28,28))
            for row in range(28):
                for col in range(28):
                    if imagination[row,col] >= 17:
                        print("1", end = ' ')
                    else:
                        print("0", end = ' ')
                print()
            print()

        print("Error rate: %s" %(error_rate))
    else:
        for x in range(mean.shape[0]):
            print("%s:"%(x))
            imagination = np.zeros(mean.shape[1])
            for y in range(mean.shape[1]):
                imagination[y] = mean[x,y]
            imagination = np.reshape(imagination, (28,28))
            for row in range(28):
                for col in range(28):
                    if imagination[row,col] >= 128:
                        print("1", end = ' ')
                    else:
                        print("0", end = ' ')
                print()
            print()
        print("Error rate: %s" %(error_rate))




if __name__ == "__main__":
    train_images,train_labels = readdata()
    train_prior = cal_prior(train_labels)
    test_images, test_labels = readtest()

    if args.re_train == 0:
        train_likelihood = np.load("train_likelihood.npy")
        mean = np.load("mean_continous.npy")
        var = np.load("var_continous.npy")
    elif args.re_train == 1:
        if args.mode == 0:
            train_likelihood = cal_likelihood(train_images,train_labels)
            np.save("train_likelihood.npy",train_likelihood)
            posteriors = cal_posterior(train_prior, train_likelihood, test_images, 0 ,0, 0)
            np.save("posteriors_discrete.npy",posteriors)
        elif args.mode == 1:
            mean, var = cal_mean_var(train_images, train_labels, train_prior)
            np.save("mean_continous.npy", mean)
            np.save("var_continous.npy", var)
            posteriors = cal_posterior(train_prior, 0 , test_images, mean, var, 1)
            np.save("posteriors_continous.npy", posteriors)
    else:
        print("value should be 0 or 1")
        exit()

    if args.mode == 0:
        posteriors = np.load("posteriors_discrete.npy")
        error_rate = cal_error_rate(posteriors, test_labels)
        printPredictDiscrete(train_likelihood, error_rate,0 ,0)
    elif args.mode == 1:
        posteriors = np.load("posteriors_continous.npy")
        error_rate = cal_error_rate(posteriors, test_labels)
        printPredictDiscrete(0, error_rate, mean, 1)
    else:
        print("wrong mode type")
        exit()


