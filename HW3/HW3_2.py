import numpy as np
from HW3_1 import UniGaussian as gaussian
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--m", help = "initial mean", type = float, default = 0)
parser.add_argument("--s", help = "initial variance", type = float, default = 1)
args = parser.parse_args()

def SeqEsti(m,s):
    print("Data point source function: N({}, {})".format(m,s))
    p_mean, mean, p_var, var = 0.0, 0.0, 0.0, 0.0
    itr = 0
    sum, sumsq = 0.0 , 0.0

    while True:
        itr+=1
        data = gaussian(m,s)
        sum+=data
        sumsq += data**2
        print("Add data point: {}".format(data))
        # update mean for new data
        mean = sum / itr
        #update variance for new data
        if itr == 1:
            var = 0
        else:
            var = (sumsq - (sum**2) / itr) / (itr-1)
        print("Mean: {}".format(mean), end = ' ')
        print("Variance: {}".format(var))
        if (abs(p_mean - mean) <= 0.001) and (abs(p_var - var)<=0.001):
            break
        p_mean = mean
        p_var = var



if __name__ == "__main__":
    m = args.m
    s = args.s

    SeqEsti(m,s)




