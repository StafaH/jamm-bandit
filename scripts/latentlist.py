import numpy as np
latentlist = []

for i in range(6600, 9600):
    np.random.seed(i)
    latentlist.append(np.random.randn(1, 512))
