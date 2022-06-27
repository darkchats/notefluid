import numpy as np
from scipy.optimize import curve_fit


def func(x, a, b):
    return a * x ** b


def cul_popt(xd, yd):
    popt, _ = curve_fit(func, np.array(xd), np.array(yd))
    return popt
