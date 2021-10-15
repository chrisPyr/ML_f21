import numpy as np

PI = 3.14159265

def uniformrd(down,up):
    return np.random.uniform(down,up)

def UniGaussian(m,s):
    x = (-2 * np.log(uniformrd(0.0,1.0)))**0.5 * np.cos(2*PI*uniformrd(0.0,1.0))
    # x~N(0,1) => sx+m~N(b,s^2)
    return s**0.5*x + m

def PolyLinearData(n,a,w):

    x = uniformrd(-1.0,1.0)
    y = 0
    for k in range(n):
        y+=w[k] * (x**k)
    y+=UniGaussian(0,a)
    return x,y




if __name__ == "__main__":
    n = 4
    a = 1
    w = [1,2,3,4]
    x, y = PolyLinearData(n,a,w)
    print(x,y)



