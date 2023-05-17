import numpy as np
import matplotlib.pylab as plt

def function_1(x):
    return 0.01*x**2+0.1*x

def function_2(x):
    return x[0]**2+x[1]**2

if __name__ == '__main__': 
    x = np.arange(0.0, 20.0, 0.1)
    x = np.outer(x, x)
    print(x)
    y = function_2(x)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.plot(x, y)
    plt.show()
