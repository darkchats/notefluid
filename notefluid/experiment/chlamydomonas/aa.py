import numpy as np


def fun(dd, r=0.2):
    d0 = 0.548
    return (dd - d0) / r / 2


print(fun(np.array([0.703, 0.377])))
print(fun(np.array([0.728, 0.306]), r=0.25))
