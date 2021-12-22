import os
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import imageio
from PIL import Image
from scipy.spatial.distance import pdist, cdist

parser = argparse.ArgumentParser()
parser.add_argument("--k", type = int, default = 3)
parser.add_argument("--input_image", type = str, default = "data/image1.png")
parser.add_argument("--output_image", type = str, default = "output", help = "name of output image")
parser.add_argument("--gamma_s", type = float, default = 0.0001)
parser.add_argument("--gamma_c", type = float, default = 0.001)
parser.add_argument("--mode", type = int, default = 0, help = "choose center method, 0 for random, 1 for k means++")
args = parser.parse_args()

num_of_points = 10000

def readImage():
    image = Image.open(args.input_image)
    data = np.array(image.getdata())
    return data

def outputImage(cluster, img_type, num_of_img):
    colors = np.array([[255,0,0], [0,255,0], [0,0,255], [0,215,175], [95,0,135], [255,255,0],[255,175,0]])
    point_color = np.zeros((num_of_points,3))
    for i in range(num_of_points):
        point_color[i,:] = colors[cluster[i],:]

    image = point_color.reshape((100,100,3))
    image = Image.fromarray(np.uint8(image))
    #image.save(os.path.join(args.output_image, img_type+str(num_of_img)+".png"))
    return image

def createGIF(images, gif_name, iter):
    #images = []
    #for i in range(iter):
    #    images.append(imageio.imread("output/kmeans"+str(i)+".png"))
    #imageio.mimsave("output/kmeans.gif", images)
    images[0].save("output/kmeans.gif", save_all = True, append_images=images[1:], optimize= False, loop = 0, duration=100)
    return

def distance(x, y, gram_matrix):
    return gram_matrix[x,x] + gram_matrix[y,y] -2* gram_matrix[x,y]


def computeKernel(data_x):
    """
    Kernel formula according to HW spec
    """
    #color distance
    color_dist = cdist(data_x, data_x, 'sqeuclidean')
    grid = np.indices((100,100)).reshape(2,10000,1)

    #pixel coordinate
    coordinate_repre = np.hstack((grid[0], grid[1]))

    #spatial distance
    spatial_dist = cdist(coordinate_repre, coordinate_repre, 'sqeuclidean')
    gram_matrix = np.multiply(np.exp(-args.gamma_s * spatial_dist), np.exp(-args.gamma_c * color_dist))

    return gram_matrix

def initialCluster(x,num_of_cluster, kernel, mode):
    """
    initialize cluster
    mode 0 for first randomly choose centers
    mode 1 for k-means++
    """
    images = []
    centers= chooseCenters(x, mode, kernel)
    cluster = np.zeros(num_of_points, dtype = int)

    for i in range(num_of_points):
        min_dist = np.full(num_of_cluster, np.inf)
        for j in range(num_of_cluster):
            min_dist[j] = distance(i, centers[j], gram_matrix)
        cluster[i] = np.argmin(min_dist)
    images.append(outputImage(cluster,"kmeans",0))
    return cluster, images

def chooseCenters(x,mode, kernel):
    if 0 == mode:
        return np.random.choice(10000, (args.k,1))
    else:
        centers = []
        centers = list(random.sample(range(0,10000), 1))
        for number_center in range(1, args.k):
            min_dist = np.full(num_of_points, np.inf)
            for i in range(num_of_points):
                for j in range(number_center):
                    #dist = distance(i, centers[j], kernel)
                    dist = np.linalg.norm(i - centers[j])
                    if dist < min_dist[i]:
                        min_dist[i] = dist
            min_dist /= np.sum(min_dist)
            centers.append(np.random.choice(np.arange(10000), 1, p=min_dist)[0])
        return centers




def KernelKmeans(num_of_cluster, cluster, kernel, images):
    current_cluster = cluster.copy()
    for i in range(1,100):
        print("Iteration #{}".format(i))
        new_cluster = np.zeros(num_of_points, dtype = int)

        _, cluster_size = np.unique(current_cluster, return_counts=True)

        k_pq = np.zeros(num_of_cluster)
        #if xn belongs to cluster k then akn = 1, otherwise 0
        for k in range(num_of_cluster):
            temp = kernel.copy()
            for j in range(num_of_points):
                if current_cluster[j] != k:
                    temp[j,:] = 0
                    temp[:,j] = 0
            k_pq[k] = np.sum(temp)

        #compute new centers
        for j in range(num_of_points):
            min_dist = np.zeros(num_of_cluster)
            for k in range(num_of_cluster):
                k_jn = np.sum(kernel[j,:][np.where(current_cluster == k)])

                min_dist[k] = kernel[j,j] -2/cluster_size[k]*k_jn + (k_pq[k]/cluster_size[k]**2)
            new_cluster[j] = np.argmin(min_dist)

        images.append(outputImage(new_cluster,"kmeans",i))
        if np.linalg.norm((new_cluster - current_cluster), ord=2)<1e-3:
            break
        current_cluster = new_cluster.copy()

    createGIF(images, "kmeans", i)
    return



if __name__ == "__main__":
    # dimension of data (100000,3) RGB
    data_points = readImage()
    gram_matrix = computeKernel(data_points)

    print("==========K-means==========")
    #initial clustering
    init_cluster, images = initialCluster(data_points, args.k, gram_matrix, args.mode)
    KernelKmeans(args.k,init_cluster, gram_matrix, images)
