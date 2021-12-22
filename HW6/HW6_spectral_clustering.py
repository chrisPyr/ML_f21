import os
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import imageio
from PIL import Image
from scipy.spatial.distance import pdist, cdist

parser = argparse.ArgumentParser()
parser.add_argument("--k", type = int, default = 3, help = "number of cluster")
parser.add_argument("--gamma_s", type = float, default = 0.0001, help = "hyperparamter in kernel")
parser.add_argument("--gamma_c", type = float, default = 0.001, help = "hyperparamter in kernel")
