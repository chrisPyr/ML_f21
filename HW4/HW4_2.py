import argparse
import matplotlib.pyplot as plt
import numpy as np
import pathlib
from numba import jit

cu_dir = pathlib.Path().resolve()
NUM_OF_LABELS = 10
PI = 3.1415926535

def readdata(train):
    if train == True:
        images_path = str(cu_dir)+"/data/train-images-idx3-ubyte"
        labels_path = str(cu_dir)+"/data/train-labels-idx1-ubyte"
    else:
        images_path = str(cu_dir)+"/data/t10k-images-idx3-ubyte"
        labels_path = str(cu_dir)+"/data/t10k-labels-idx1-ubyte"

    with open(images_path,'rb') as data:
        magic = int.from_bytes(data.read(4), byteorder='big')
        if(magic != 2051):
            print("not MNIST format")
            exit()
        num_of_images = int.from_bytes(data.read(4), byteorder='big')
        row_of_image = int.from_bytes(data.read(4),byteorder='big')
        col_of_image = int.from_bytes(data.read(4),byteorder='big')
        images_data = np.fromfile(data, dtype = np.ubyte, count = -1)
        images = images_data.reshape(num_of_images, row_of_image*col_of_image)
    with open(labels_path,'rb') as label:
        magic = int.from_bytes(label.read(4), byteorder='big')
        if(magic != 2049):
            print("not MNIST format")
            exit()
        num_of_images = int.from_bytes(label.read(4), byteorder='big')
        labels_data = np.fromfile(label, dtype = np.ubyte, count = -1)
        labels = labels_data.reshape(num_of_images,1)
    return images, labels

@jit
def EStep(train_images, lambd, proba_image, proba_pixels):

    for i in range(train_images.shape[0]):
        for j in range(NUM_OF_LABELS):
            proba_image[i,j] = lambd[j]
            for k in range(train_images.shape[1]):
                if train_images[i,k] == 1:
                    proba_image[i,j] *= proba_pixels[j,k]
                else:
                    proba_image[i,j] *= (1-proba_pixels[j,k])
        if np.sum(proba_image[i,:]):
            proba_image[i,:] /= np.sum(proba_image[i,:])

    return proba_image

@jit
def MStep(train_images, lambd, proba_image, proba_pixels ):
    lambd = np.sum(proba_image, axis = 0)
    for i in range(NUM_OF_LABELS):
        for j in range(train_images.shape[1]):
            proba_pixels[i,j] = 0
            for k in range(train_images.shape[0]):
                proba_pixels[i,j] += proba_image[k,i]*train_images[k,j]
            proba_pixels[i,j] = (proba_pixels[i,j] + 1e-10) / (lambd[i] + 1e-10*784)
        lambd[i] = (lambd[i] + 1e-10) / (np.sum(lambd) + 1e-10*10)
    return lambd, proba_pixels

def imagination(proba_pixels, count, diff):
    for i in range(NUM_OF_LABELS):
        print("class {}:".format(i))
        for j in range(proba_pixels.shape[1]):
            if proba_pixels[i,j] > 0.5:
                print("1", end = " ")
            else:
                print("0", end = " ")
            if (j+1)%28 == 0:
                print()
        print()

    print("No. of Iteration: {}, Difference: {}".format(count, diff))

# predict classification - label match
# we only classify as 10 class but we don't know class belongs to which label
@jit
def predictLabel(proba_pixels, train_images, train_labels,lambd):
    class_label_matrix = np.zeros((NUM_OF_LABELS,NUM_OF_LABELS))
    p_label = np.zeros(NUM_OF_LABELS)

    for i in range(train_images.shape[0]):
        for j in range(NUM_OF_LABELS):
            p_label[j] = lambd[j]
            for k in range(train_images.shape[1]):
                if train_images[i,k]:
                    p_label[j] *= proba_pixels[j,k]
                else:
                    p_label[j] *= (1-proba_pixels[j,k])
        class_label_matrix[train_labels[i], np.argmax(p_label)] +=1
    return class_label_matrix

def assignLabel(confusion_matrix):

    P_label = np.zeros((NUM_OF_LABELS, NUM_OF_LABELS))
    for i in range(NUM_OF_LABELS):
        P_label[i,:] = confusion_matrix[i,:]/np.sum(confusion_matrix[i,:])
    P_label = P_label.ravel()

    count = 0
    class_match_label = np.full(10,-1)
    label_match_class = np.full(10,-1)
    while count < 10:
        match = np.argmax(P_label)
        if P_label[match] == 0:
            break
        else:
            P_label[match] = 0
            if class_match_label[(int)(match/NUM_OF_LABELS)] == -1 and label_match_class[(match%NUM_OF_LABELS)] == -1:
                class_match_label[(int)(match/NUM_OF_LABELS)] = match % NUM_OF_LABELS
                label_match_class[(match%NUM_OF_LABELS)] = (int)(match/NUM_OF_LABELS)
                count +=1

    return class_match_label


def labelImagination(proba_pixels, assign_label):
    proba_pixels = (proba_pixels >= 0.5) * 1
    for i in range(NUM_OF_LABELS):
        id = assign_label[i]
        print("labeled class {}:".format(i))
        for j in range(proba_pixels.shape[1]):
            print(proba_pixels[id,j],end = " ")
            if (j+1) % 28 == 0:
                print()
        print()
    return

def printConfusionMatrix(confusion_matrix, assign_label):
    for i in range(NUM_OF_LABELS):
        TP = confusion_matrix[i, assign_label[i]]
        FN = np.sum(confusion_matrix[:,assign_label[i]]) - TP
        FP = np.sum(confusion_matrix[i]) - TP
        TN = 60000-TP-FP-FN

        print("----------------------------------")
        print("Confusion Matrix {}:".format(i))
        print("{:20}Predict number {}  Predict not number {}".format(" ",i,i))
        print("Is number {} {:12}{}{:12}{}".format(i," ",TP, " ",FN))
        print("Isn't number {} {:9}{}{:11}{}".format(i," ",FP," ",TN))
        print()
        print("Sensitivity (Successfully predict number {}    ): {}".format(i, TP/(TP+FN)))
        print("Specificity (Successfully predict not number {}): {}".format(i, TN/(FP+TN)))





if __name__=="__main__":
    train_images, train_labels = readdata(True)
    test_images, test_labels = readdata(False)
    #convert to 0 1 bin
    train_images = (train_images > 127) * 1
    # probability of each pixels in each images (0 or 1)
    proba_pixels = np.random.uniform(0.0,1.0,(NUM_OF_LABELS, train_images.shape[1]))
    # probability of which label of each image (the probability of showing which label)
    proba_image = np.zeros((train_images.shape[0],NUM_OF_LABELS))
    lambd = np.full(NUM_OF_LABELS,0.1)

    count = 0
    while True:
        count+=1
        prev_P = np.copy(proba_pixels)
        proba_image = EStep(train_images,lambd,proba_image,proba_pixels)
        lambd, proba_pixels = MStep(train_images, lambd, proba_image, proba_pixels)

        diff = np.linalg.norm(prev_P - proba_pixels)
        imagination(proba_pixels, count, diff)
        if (abs(prev_P - proba_pixels) < 1e-5).all() or count >= 20:
            break;

    confusion_matrix = predictLabel(proba_pixels, train_images, train_labels, lambd)
    assign_label = assignLabel(confusion_matrix)
    labelImagination(proba_pixels, assign_label)
    printConfusionMatrix(confusion_matrix, assign_label)




