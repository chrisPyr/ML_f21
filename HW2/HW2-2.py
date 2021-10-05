#!/usr/bin/env python3

import struct
import sys
import numpy as np
import argparse
import pathlib
import math

cu_dir = pathlib.Path().resolve()

parser = argparse.ArgumentParser()
parser.add_argument("-a", help = "initial a value", default = 0)
parser.add_argument("-b", help = "initial b value", default = 0)
args = parser.parse_args()

if __name__ == "__main__":
    data = []
    with open(str(cu_dir)+"/data/testfile.txt") as file:
        line = file.readline().strip()
        while line:
            data.append(line)
            line = file.readline().strip()

    global_a_prior = int(args.a)
    global_b_prior = int(args.b)
    global_a_posterior = int(args.a)
    global_b_posterior = int(args.b)
    for x in range(len(data)):
        a = 0
        b = 0
        print("case: %s: %s"%(x, data[x]))
        for pos in range(len(data[x])):
            if float(data[x][pos]) == 1:
                a+=1
                global_a_posterior+=1
            else:
                b+=1
                global_b_posterior+=1
        theta = a/(a+b)
        likelihood = math.factorial(a+b)/(math.factorial(a)*math.factorial(b)) * (theta**a) * ((1-theta)**b)
        print("likelihood: ",likelihood)
        print("Beta prior: a:%s, b:%s"%(global_a_prior,global_b_prior))
        print("Beta posterior: a:%s, b:%s"%(global_a_posterior,global_b_posterior))
        global_a_prior = global_a_posterior
        global_b_prior = global_b_posterior
        print()
