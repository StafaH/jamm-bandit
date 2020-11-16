import pandas as pd
import numpy as np
import os

dict = {}
dir = 'stylegan2directions'
for filename in os.listdir(dir):
    if filename[-4:] == '.npy':
        trait = filename.split('.')[0]
        dict[trait] = np.load(dir+'/'+filename)[0]
        
df = pd.DataFrame.from_dict(dict, orient='index')
df.to_csv('luxemberg.csv')