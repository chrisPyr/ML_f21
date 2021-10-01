#!/usr/bin/env python3

import struct
import sys
import numpy as np
import argparse
import matplotlib.pyplot as plt
import pathlib

cu_dir = pathlib.Path().resolve()

def readdata():
    with open(str(cu_dir)+"/data/t10k-images-idx3-ubyte",'rb') as data:
        magic = data.read(4)
        magic = int.from_bytes(magic, byteorder='big')
        print(magic)
        if(magic != 2051):
            print("not MNIST format")
            exit()


if __name__ == "__main__":
    readdata()
