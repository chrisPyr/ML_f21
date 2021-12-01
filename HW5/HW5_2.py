import sys
import numpy as np
#from libsvm.svmutil import *
import libsvm.svmutil as svm
import time
from scipy.spatial.distance import cdist
from enum import Enum

kernel_types = Enum("ktypes", [("linear", 0), ("polynomial", 1), ("RBF", 2)])


def readData(train=True):
    if train == True:
        x = np.genfromtxt("./data/X_train.csv", delimiter=',')
        y = np.genfromtxt("./data/Y_train.csv", delimiter=',')
    else:
        x = np.genfromtxt("./data/X_test.csv", delimiter=',')
        y = np.genfromtxt("./data/Y_test.csv", delimiter=',')

    return x, y


def linearKernel(u, v):
    return u.dot(v.T)


def SVM(types, train_x, train_y, test_x, test_y):
    print("=========part1============")
    if types.name == "linear":
        print("linear")
    elif types.name == "polynomial":
        print("polynomial")
    elif types.name == "RBF":
        print("RBF")
    else:
        print("unknown type")
        exit()

    '''
    -t to select kernel type
        0: linear
        1: polynomial
        2: RBF
        3: sigmoid
        4: precomputed kernel
    -q to mute output
    '''
    parameters = svm.svm_parameter('-t ' + str(types.value) + ' -q')
    time_start = time.time()
    dataset = svm.svm_problem(train_y, train_x)
    model = svm.svm_train(dataset, parameters)
    _, _, _, = svm.svm_predict(test_y, test_x, model)
    time_end = time.time()
    print("cost time: {}".format(time_end-time_start))
    return


def SVMGridSearch(types, train_x, train_y, test_x, test_y):
    print("=========part2============")
    if types.name == "linear":
        print("linear")
    elif types.name == "polynomial":
        print("polynomial")
    elif types.name == "RBF":
        print("RBF")
    else:
        print("unknown type")
        exit()
    k_fold = 3
    max_accuracy = 0
    opt_paras = {}
    if types.name == "linear":
        start_time = time.time()
        '''
        In linear kernel, there's no parameters, so only need to tune C for cost function in C-SVC
        '''
        for c in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
            parameters = svm.svm_parameter(
                '-t '+str(types.value) + ' -q -c ' + str(c) + ' -v ' + str(k_fold))
            dataset = svm.svm_problem(train_y, train_x)
            sys.stdout = None
            model_acc = svm.svm_train(dataset, parameters)
            sys.stdout = sys.__stdout__
            if model_acc > max_accuracy:
                max_accuracy = model_acc
                opt_paras = {'C': c}
        end_time = time.time()
        print(opt_paras)
        print("tune cost time: {}".format(end_time-start_time))
        parameters = svm.svm_parameter(
            '-t '+str(types.value) + ' -q -c ' + str(opt_paras["C"]))
        dataset = svm.svm_problem(train_y, train_x)
        model = svm.svm_train(dataset, parameters)
        _, _, _, = svm.svm_predict(test_y, test_x, model)
    elif types.name == "polynomial":
        '''
        Paras:
            C: cost function penalty
            d: order of polynomial
            g: gamma
            r: coef
        '''
        start_time = time.time()
        cost = [np.power(10.0, i) for i in range(-1, 3)]
        gamma = [1.0/784] + [np.power(10.0, i) for i in range(-1, 2)]
        coef0 = [np.power(10.0, i) for i in range(-1, 2)]
        degree = [i for i in range(0, 4)]

        for c in cost:
            for g in gamma:
                for coef in coef0:
                    for d in degree:
                        parameters = svm.svm_parameter('-t '+str(types.value) + ' -c ' + str(c) + ' -v ' + str(k_fold) +
                                                       ' -g '+str(g) + ' -r '+str(coef) + ' -d '+str(d) + ' -q')
                        dataset = svm.svm_problem(train_y, train_x)
                        sys.stdout = None
                        model_acc = svm.svm_train(dataset, parameters)
                        sys.stdout = sys.__stdout__
                        if model_acc > max_accuracy:
                            max_accuracy = model_acc
                            opt_paras = {'C': c, 'gamma': g,
                                         'coef': coef, 'degree': d}
        end_time = time.time()
        print(opt_paras)
        print("tune cost time: {}".format(end_time-start_time))
        parameters = svm.svm_parameter('-t '+str(types.value) + ' -c ' + str(opt_paras["C"]) +
                                       ' -g '+str(opt_paras["gamma"]) + ' -r '+str(opt_paras["coef"]) +
                                       ' -d '+str(opt_paras["degree"]) + ' -q')

        dataset = svm.svm_problem(train_y, train_x)
        model = svm.svm_train(dataset, parameters)
        _, _, _, = svm.svm_predict(test_y, test_x, model)
    elif types.name == "RBF":
        '''
        Paras:
            C: cost function penalty
            g: gamma
        '''

        start_time = time.time()
        cost = [np.power(10.0, i) for i in range(-1, 2)]
        gamma = [1.0/784] + [np.power(10.0, i) for i in range(-1, 2)]

        for c in cost:
            for g in gamma:
                parameters = svm.svm_parameter(
                    '-t '+str(types.value)+' -c ' + str(c)+' -v '+str(k_fold) + ' -g '+str(g) + ' -q')
                dataset = svm.svm_problem(train_y, train_x)
                sys.stdout = None
                model_acc = svm.svm_train(dataset, parameters)
                sys.stdout = sys.__stdout__
                if model_acc > max_accuracy:
                    max_accuracy = model_acc
                    opt_paras = {'C': c, 'gamma': g}
        end_time = time.time()
        print(opt_paras)
        print("tune cost time: {}".format(end_time-start_time))
        parameters = svm.svm_parameter(
            '-t '+str(types.value)+' -c ' + str(opt_paras["C"]) + ' -g '+str(opt_paras["gamma"]) + ' -q')
        dataset = svm.svm_problem(train_y, train_x)
        model = svm.svm_train(dataset, parameters)
        _, _, _, = svm.svm_predict(test_y, test_x, model)

    else:
        print("unkown types")
        exit(-1)

    return


def RBFKernel(u, v, gamma=1.0/784):
    return np.exp(-gamma * cdist(u, v, "sqeuclidean"))


def SVMLinearRBF(train_x, train_y, test_x, test_y):
    print("=========part3============")
    linear = linearKernel(train_x, train_x)
    RBF = RBFKernel(train_x, train_x)
    LinearRBF_kernel = np.hstack(
        (np.arange(1, train_x.shape[0]+1).reshape(-1, 1), linear + RBF))
    linearTest = linearKernel(test_x, test_x)
    rbfTest = RBFKernel(test_x, test_x)
    new_test_x = np.hstack(
        (np.arange(1, test_x.shape[0]+1).reshape(-1, 1), linearTest+rbfTest))
    start_time = time.time()
    parameters = svm.svm_parameter('-t 4 -q')
    problem = svm.svm_problem(train_y, LinearRBF_kernel, isKernel=True)
    model = svm.svm_train(problem, parameters)

    _, model_acc, _ = svm.svm_predict(test_y, new_test_x, model)
    end_time = time.time()
    print("cost time: {}".format(end_time - start_time))

    return


if __name__ == "__main__":
    train_x, train_y = readData(True)
    test_x, test_y = readData(False)
    # part1
    #SVM(kernel_types.linear, train_x, train_y, test_x, test_y)
    #SVM(kernel_types.polynomial, train_x, train_y, test_x, test_y)
    #SVM(kernel_types.RBF, train_x, train_y, test_x, test_y)

    # part2
    #SVMGridSearch(kernel_types.linear, train_x, train_y, test_x, test_y)
    #SVMGridSearch(kernel_types.polynomial, train_x, train_y, test_x, test_y)
    #SVMGridSearch(kernel_types.RBF, train_x, train_y, test_x, test_y)

    # part3
    SVMLinearRBF(train_x, train_y, test_x, test_y)
