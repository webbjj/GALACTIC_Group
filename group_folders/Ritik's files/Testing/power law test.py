import numpy as np


def _power_law_distribution_function(n, alpha, xmin, xmax):

    eta = alpha + 1.

    if xmin == xmax:
        x = xmin
    elif alpha == 0:
        x = xmin + np.random.random(n) * (xmax - xmin)
    elif alpha > 0:
        x = xmin + np.random.power(eta, n) * (xmax - xmin)
    elif alpha < 0 and alpha != -1.:
        x = (xmin ** eta + (xmax ** eta - xmin ** eta) * np.random.rand(
            n)) ** (1. / eta)
    elif alpha == -1:
        x = np.log10(xmin) + np.random.random(n) * (
                np.log10(xmax) - np.log10(xmin))
        x = 10.0 ** x

    if n == 1:
        return x
    else:
        return np.array(x)


msun = 1.9891e30

mlow = 0.8 * msun
mhigh = 1.4 * msun

lst = [2, 1, 4, 3]
m = min(lst)
lst.remove(min(lst))
a, b, c = lst
print(m, a, b, c)
