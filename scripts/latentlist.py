import numpy as np
import pandas as pd
latentlist = []
seeds = []

for i in range(6600, 9600):
    np.random.seed(i)
    latentlist.append(np.random.randn(1, 512))
    seeds.append(i)
    
latentlist = np.asarray(latentlist)
latentlist = np.reshape(latentlist, [3000, 512])
    
df = pd.DataFrame(latentlist)
seedseries = pd.Series(seeds)
df.insert(loc=0, column='seed', value=seedseries)
df.to_csv('latentlist.csv')