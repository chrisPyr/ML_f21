import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from scipy.spatial.distance import cdist
import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument("--k", default = 5, type = int, help = "Number of nearest neighbors")
parser.add_argument("--gamma", default = 0.000001, type = float)
args = parser.parse_args()

def readData(type = "Training"):
    path = "data/Yale_Face_Database/" + type
    files = os.listdir(path)
    images = np.zeros((len(files), 98 * 116))
    labels = np.zeros(len(files), dtype = int)
    for i, file in enumerate(files):
        with Image.open(os.path.join(path, file)) as f:
            images[i] = np.array(f.resize((98,116))).reshape(1,-1)
            labels[i] = int(file[7:9])
    return images, labels

def PCA(train_images_data, train_images_label, test_images_data, test_images_label, mode = 0, kernel_type = 0):
    """
    Simple PCA
    """
    mean = (np.sum(train_images_data,axis = 0) / len(train_images_data)).flatten()
    if not mode:
        # Compute Covariance
        #train_images_data_trans = train_images_data.T
        #mean = (np.sum(train_images_data_trans, axis = 1) / len(train_images_data))
        #mean = np.tile(mean.T,(len(train_images_data),1)).T
        #diff = train_images_data_trans - mean
        #Cov = diff.dot(diff.T)/ len(train_images_data)
        diff = train_images_data - mean
        Cov = diff.T.dot(diff)
    else:
        if not kernel_type:
            kernel = train_images_data.T.dot(train_images_data)
        else:
            #RBF
            kernel = np.exp(-args.gamma * cdist(train_images_data.T, train_images_data.T, 'sqeuclidean'))
        matrix_n = np.ones((116*98,116*98), dtype= float) / (116*98)
        Cov = kernel - matrix_n.dot(kernel)  - kernel.dot(matrix_n) + matrix_n.dot(kernel).dot(matrix_n)

    # find eigenvectors and eigenvalues
    #eigenvalues, eigenvectors = np.linalg.eig(Cov)
    #np.save("eigenvalues_PCA"+ ("_simple" if not mode else "_kernel_linear" if not kernel_type else "_kernel_RBF")  + ".npy", eigenvalues)
    #np.save("eigenvectors_PCA"+ ("_simple" if not mode else "_kernel_linear" if not kernel_type else "_kernel_RBF")  + ".npy", eigenvectors)
    eigenvalues = np.load("eigenvalues_PCA"+ ("_simple" if not mode else "_kernel_linear" if not kernel_type else "_kernel_RBF") + ".npy")
    eigenvectors = np.load("eigenvectors_PCA"+ ("_simple" if not mode else "_kernel_linear" if not kernel_type else "_kernel_RBF") + ".npy")

    #find first 25 eigenvectors
    sort_index = np.argsort(-eigenvalues)
    eigenvectors = eigenvectors[:,sort_index[0:25]].real

    #transform eigenvectors to fisher faces
    faces = eigenvectors.T.reshape((25, 116, 98))
    fig = plt.figure(1)
    for idx in range(25):
        plt.subplot(5,5,idx+1)
        plt.axis('off')
        plt.imshow(faces[idx,:,:], cmap = 'gray')

    #reconstruct faces
    chosen_idx = random.sample(range(len(train_images_data)), 10)
    #re_faces = train_images_data[chosen_idx].dot(eigenvectors).dot(eigenvectors.T)
    weight = train_images_data[chosen_idx].dot(eigenvectors)
    re_faces = mean+weight.dot(eigenvectors.T)
    fig = plt.figure(1)
    for idx in range(10):
        plt.subplot(10,2,idx*2+1)
        plt.axis('off')
        plt.imshow(train_images_data[chosen_idx[idx], :].reshape((116,98)), cmap = 'gray')
        plt.subplot(10,2,idx*2+2)
        plt.axis('off')
        plt.imshow(re_faces[idx,:].reshape((116,98)), cmap = 'gray')


    #decorrelate
    deco_train = train_images_data.dot(eigenvectors)
    deco_test = test_images_data.dot(eigenvectors)
    face_classify(deco_train, deco_test, train_images_label, test_images_label)

def LDA(train_images_data, train_images_label, test_images_datal, test_images_label, mode =0, kernel_type = 0):
    if not mode:
        """
        simple LDA
        """
        mean = (np.sum(train_images_data, axis = 0) / len(train_images_data)).flatten()

        num_of_class = 15
        matrix_Sw = np.zeros((98*116,98*116))
        matrix_Sb = np.zeros((98*116,98*116))
        for i in range(1, num_of_class+1):
            idx = np.where(train_images_label == i)[0]
            faces_i = train_images_data[idx]
            mean_class_i = (np.sum(faces_i, axis = 0) / len(faces_i)).flatten()
            diff_class_i = faces_i - mean_class_i
            diff_class = mean_class_i - mean
            matrix_Sw += diff_class_i.T.dot(diff_class_i)
            matrix_Sb += len(idx) * diff_class.T.dot(diff_class)
        S = np.linalg.inv(matrix_Sw).dot(matrix_Sb)
    else:
        if not kernel_type:
            kernel = train_images_data.T.dot(train_images_data)
        else:
            #RBF
            kernel = np.exp(-args.gamma * cdist(train_images_data.T, train_images_data.T, 'sqeuclidean'))
        mean = np.sum(train_images_data, axis = 0) / len(train_images_data)
        matrix_M_star = np.sum(kernel, axis = 0) / len(train_images_data)
        num_of_class = 15
        matrix_N = np.zeros((98*116, 98*116))
        matrix_M = np.zeros((98*116, 98*116))
        matrix_1_lj = np.ones((9,9)) / 9
        for i in range(1, num_of_class+1):
            idx = np.where(train_images_label == i)[0]
            matrix_Kj = kernel[idx]
            lj = len(idx)
            matrix_Mj = np.sum(matrix_Kj, axis = 0)/ len(matrix_Kj)
            diff_class = matrix_Mj - matrix_M_star
            matrix_N += matrix_Kj.T.dot((np.identity(lj)- matrix_1_lj).dot(matrix_Kj))
            matrix_M += lj * diff_class.T.dot(diff_class)
        S = np.linalg.pinv(matrix_N).dot(matrix_M)


    #eigenvalues, eigenvectors = np.linalg.eig(S)
    #np.save("eigenvalues_LDA"+ ("_simple" if not mode else "_kernel_linear" if not kernel_type else "_kernel_RBF")  + ".npy", eigenvalues)
    #np.save("eigenvectors_LDA"+ ("_simple" if not mode else "_kernel_linear" if not kernel_type else "_kernel_RBF")  + ".npy", eigenvectors)
    eigenvalues = np.load("eigenvalues_LDA"+ ("_simple" if not mode else "_kernel_linear" if not kernel_type else "_kernel_RBF")  + ".npy")
    eigenvectors = np.load("eigenvectors_LDA"+ ("_simple" if not mode else "_kernel_linear" if not kernel_type else "_kernel_RBF")  + ".npy")

    #find first 25 eigenvectors
    sort_index = np.argsort(-eigenvalues)
    eigenvectors = eigenvectors[:,sort_index[0:25]].real
    eigenvectors_pca = np.load("eigenvectors_PCA"+ ("_simple" if not mode else "_kernel_linear" if not kernel_type else "_kernel_RBF")  + ".npy").real
    fisherfaces = eigenvectors_pca.dot(eigenvectors).T

    #transform eigenvectors to fisher faces
    #faces = eigenvectors.T.reshape((25, 116, 98))
    fisherfaces_re = fisherfaces.reshape((25,116,98))
    fig = plt.figure(1)
    for idx in range(25):
        plt.subplot(5,5,idx+1)
        plt.axis('off')
        plt.imshow(fisherfaces_re[idx], cmap = 'gray')
    plt.show()

    #reconstruct faces
    chosen_idx = random.sample(range(len(train_images_data)), 10)
    weight = train_images_data[chosen_idx].dot(fisherfaces.T)
    re_faces = mean + weight.dot(fisherfaces)
    fig = plt.figure(1)
    for idx in range(10):
        plt.subplot(10,2,idx*2+1)
        plt.axis('off')
        plt.imshow(train_images_data[chosen_idx[idx], :].reshape((116,98)), cmap = 'gray')
        plt.subplot(10,2,idx*2+2)
        plt.axis('off')
        plt.imshow(re_faces[idx,:].reshape((116,98)), cmap = 'gray')
    plt.show()

    #decorrelate
    deco_train = train_images_data.dot(fisherfaces.T)
    deco_test = test_images_data.dot(fisherfaces.T)
    face_classify(deco_train, deco_test, train_images_label, test_images_label)

def face_classify(deco_train, deco_test, train_images_label, test_images_label):
    error = 0
    dist = np.zeros(len(deco_train))
    for i in range(len(deco_test)):
        for j in range(len(deco_train)):
            dist[j] = cdist([deco_test[i]], [deco_train[j]], 'sqeuclidean')
        k_nearest = np.argsort(dist)[0:args.k]
        predict = np.argmax(np.bincount(train_images_label[k_nearest]))
        if test_images_label[i] != predict:
            error+=1
    print("Error rate: {}".format((error/len(test_images_label))))






if __name__ == "__main__":
    train_images_data, train_images_label = readData()
    test_images_data, test_images_label = readData("Testing")

    #PCA(train_images_data, train_images_label, test_images_data, test_images_label, 0, 0)
    #PCA(train_images_data, train_images_label, test_images_data, test_images_label, 1, 0)
    #PCA(train_images_data, train_images_label, test_images_data, test_images_label, 1, 1)

    LDA(train_images_data, train_images_label, test_images_data, test_images_label, 0, 0)
    LDA(train_images_data, train_images_label, test_images_data, test_images_label, 1, 0)
    LDA(train_images_data, train_images_label, test_images_data, test_images_label, 1, 1)
