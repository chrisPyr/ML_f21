import os
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import imageio
from PIL import Image
from scipy.spatial.distance import pdist, cdist

parser = argparse.ArgumentParser()
parser.add_argument("--input_image", type=str,
                    default="data/image1.png", help="path of input image")
parser.add_argument("--k", type=int, default=3, help="number of cluster")
parser.add_argument("--gamma_s", type=float, default=0.0001,
                    help="hyperparamter in kernel")
parser.add_argument("--gamma_c", type=float, default=0.001,
                    help="hyperparamter in kernel")
parser.add_argument("--cut_type", type=int, default=1,
                    help="1 for ratio cut, 2 for normalized cut")
parser.add_argument("--mode", type=int, default=1,
                    help="method of choosing initial centers, 1 for random, 2 for kmeans++")
args = parser.parse_args()

num_of_points = 10000


def readImage():
    image = Image.open(args.input_image)
    data = np.array(image.getdata())
    return data


def outputImage(cluster, img_type, num_of_img):
    colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [0, 215, 175], [
                      95, 0, 135], [255, 255, 0], [255, 175, 0]])
    point_color = np.zeros((num_of_points, 3))
    for i in range(num_of_points):
        point_color[i, :] = colors[cluster[i], :]

    image = point_color.reshape((100, 100, 3))
    image = Image.fromarray(np.uint8(image))
    #image.save(os.path.join(args.output_image, img_type+str(num_of_img)+".png"))
    return image


def createGIF(images):
    if args.cut_type == 1:
        cut = "ratio"
    else:
        cut = "normalized"
    if args.mode == 1:
        mode = "random"
    else:
        mode = "kmeans++"
    file_name = "output/spetral_clustering_" + \
        str(pic)+"_"+str(args.k)+"_"+cut+"_"+mode+".gif"
    images[0].save(file_name, save_all=True, append_images=images[1:],
                   optimize=False, loop=0, duration=100)
    png_name = "spetral_clustering_" + \
        str(pic)+"_"+str(args.k)+"_"+cut+"_"+mode
    images[0].save(png_name+".png")
    images[len(images)-1].save(png_name+str(len(images)-1)+".png")
    return


def printEigen(matrix_U, clusters):
    if args.cut_type == 1:
        cut = "ratio"
    else:
        cut = "normalized"
    if args.mode == 1:
        mode = "random"
    else:
        mode = "kmeans++"

    matrix_U = np.real(matrix_U)
    colors = ["r", "g", "b"]
    plt.clf()
    for idx in range(len(matrix_U)):
        plt.scatter(matrix_U[idx, 0], matrix_U[idx, 1],
                    color=colors[clusters[idx]])
        if idx % 1000 == 0:
            print("hihi")
    plt.savefig(str(pic)+"_eigen_coordinate_"+str(cut)+"_"+str(mode)+".png")
    return


def computeKernel(x):
    # color distance
    color_dist = cdist(x, x, 'sqeuclidean')
    grid = np.indices((100, 100)).reshape(2, 10000, 1)

    # pixel coordinate
    coordinate_repre = np.hstack((grid[0], grid[1]))

    # spatial distance
    spatial_dist = cdist(coordinate_repre, coordinate_repre, 'sqeuclidean')
    gram_matrix = np.multiply(
        np.exp(-args.gamma_s * spatial_dist), np.exp(-args.gamma_c * color_dist))
    return gram_matrix


def computeMatrixU(matrix_W, cut_type, num_of_cluster):
    """
    matrix_w: weight matrix
    """
    if cut_type == 1:
        cut = "ratio"
    else:
        cut = "normalized"
    # Graph Laplacian L and degree matrix D
#    matrix_D, matrix_L = Laplacian(matrix_W)
#    if cut_type == 2:
#        matrix_D_sym = np.zeros((matrix_D.shape))
#        for i in range(len(matrix_D)):
#            matrix_D_sym[i,i] = 1.0 / np.sqrt(matrix_D[i,i])
#        matrix_L = matrix_D_sym.dot(matrix_L).dot(matrix_D_sym)
#
#    eigenvalues, eigenvectors = np.linalg.eig(matrix_L)
#    eigenvectors = eigenvectors.T
#    np.save(cut+"eigenvalues"+str(pic)+".npy", eigenvalues)
#    np.save(cut+"eigenvectors"+str(pic)+".npy", eigenvectors)
    eigenvalues = np.load(cut+"eigenvalues"+str(pic)+".npy")
    eigenvectors = np.load(cut+"eigenvectors"+str(pic)+".npy")

    # sort eigenvalues and find indices of nonzero eigenvalues
    sort_idx = np.argsort(eigenvalues)
    mask = eigenvalues[sort_idx] > 0
    idx = sort_idx[mask][0:num_of_cluster]
    matrix_U = eigenvectors[idx].T
    if cut_type == 2:
        T = matrix_U.copy()
        temp = np.sum(matrix_U, axis=1)
        for i in range(len(T)):
            T[i] /= temp[i]
        matrix_U = T
#    return eigenvectors[sort_idx[:num_of_cluster]].T
    return matrix_U


def Laplacian(matrix_W):
    matrix_D = np.zeros_like(matrix_W)
    for idx, row in enumerate(matrix_W):
        matrix_D[idx, idx] += np.sum(row)
    matrix_L = matrix_D - matrix_W
    return matrix_D, matrix_L


def chooseCenters(matrix_U, num_of_cluster, method):
    if 0 == method:
        return matrix_U[np.random.choice(10000, args.k)]
    else:
        centers = []
        centers = list(random.sample(range(0, 10000), 1))
        for number_center in range(1, args.k):
            min_dist = np.full(num_of_points, np.inf)
            for i in range(num_of_points):
                for j in range(number_center):
                    dist = np.linalg.norm(i - centers[j])
                    if dist < min_dist[i]:
                        min_dist[i] = dist
            min_dist /= np.sum(min_dist)
            centers.append(np.random.choice(
                np.arange(10000), 1, p=min_dist)[0])
        U_centers = []
        for i in range(num_of_cluster):
            U_centers.append(matrix_U[centers[i]])
        U_centers = np.array(U_centers)
        return U_centers


def kmeansClustering(matrix_U, U_centers,  num_of_cluster):
    new_clusters = np.zeros(num_of_points, dtype=int)
    for i in range(num_of_points):
        dist = np.zeros(num_of_cluster)
        for j, cen in enumerate(U_centers):
            dist[j] = np.linalg.norm((matrix_U[i]-cen), ord=2)
        new_clusters[i] = np.argmin(dist)
    return new_clusters


def kmeansCenters(matrix_U, new_clusters, num_of_cluster):
    """
    Recompute new centers
    """
    new_centers = []
    for i in range(num_of_cluster):
        points_in_c = matrix_U[new_clusters == i]
        new_center = np.average(points_in_c, axis=0)
        new_centers.append(new_center)
    return np.array(new_centers)


def kmeans(matrix_U, U_centers, num_of_cluster, cut_type):
    images = []
    old_centers = U_centers.copy()
    new_clusters = np.zeros(num_of_points, dtype=int)
    for i in range(100):
        print("Iteration #{}".format(i))
        new_clusters = kmeansClustering(matrix_U, old_centers, num_of_cluster)
        new_centers = kmeansCenters(matrix_U, new_clusters, num_of_cluster)
        image = outputImage(new_clusters, "spectral_clustering", i)
        images.append(image)
        if np.linalg.norm((new_centers - old_centers), ord=2) < 1e-2:
            break
        old_centers = new_centers.copy()
    return new_clusters, images


def spectralClustering(kernel, num_of_cluster, cut_type, method):
    matrix_U = computeMatrixU(kernel, cut_type, num_of_cluster)
    U_centers = chooseCenters(matrix_U, num_of_cluster, method)
    clusters, images = kmeans(matrix_U, U_centers, num_of_cluster, cut_type)
    if num_of_cluster == 2:
        printEigen(matrix_U, clusters)
    return images


if __name__ == "__main__":
    data_points = readImage()
    gram_matrix = computeKernel(data_points)
    pic = 0
    if args.input_image == "data/image1.png":
        pic = 1
    else:
        pic = 2

    print("========Spectral clustering=======")
    images = spectralClustering(gram_matrix, args.k, args.cut_type, args.mode)
    createGIF(images)
