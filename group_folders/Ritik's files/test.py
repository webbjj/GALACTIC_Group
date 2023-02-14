import numpy as np

a = 1
b = 2
c = a < b
print(c)

alpha = np.random.rand(10) * 10
beta = a > alpha
if np.sum(beta) > 0:
    print(np.sum(beta))
    print(beta)









