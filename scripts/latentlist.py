import numpy as np
import pandas as pd
latentlist = []

for i in range(6600, 9600):
    np.random.seed(i)
    latentlist.append(np.random.randn(1, 512))
latentlist = np.asarray(latentlist)
latentlist = np.reshape(latentlist, [3000, 512])
    
df = pd.DataFrame(latentlist)
df.to_csv('latentlist.csv')