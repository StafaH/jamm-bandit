import numpy as np
import statsmodels.api as sm

def random_latents():
    weights = np.random.normal(loc=0, scale=1.0, size=(512,))
    return weights

def thompson_sample(observation, prior):
    weights = np.random.normal(loc=0, scale=1.0, size=(512,))
    return weights

def hill_climbing():
    weights = np.random.normal(loc=0, scale=1.0, size=(512,))
    return weights

def limited_thompson_sample():
    weights = np.random.normal(loc=0, scale=1.0, size=(512,))
    return weights

#features = 20 # would normally by 512
#trials = 1000 
#coef = np.zeros(features,) # this is the prior

def logistic_reg(coef):
    
    def rand_bin_array(weights, N):
        arr = np.zeros(N)
        for i in range(0, len(weights)):
            if weights[i][features-1] > 2:
                arr[i] = 1
        return arr
    
    weights = np.random.normal(loc=coef[0], scale=1.0, size=(trials,1)) # weights sampled from a normal dist
    for c in coef[1:]:
        weight = np.random.normal(loc=c, scale=1.0, size=(trials,1))
        weights = np.hstack((weights, weight))
    weights = np.clip(weights, -3, 3)
    labels = rand_bin_array(weights, trials)
    
    result = sm.Logit(labels,weights).fit()
    coef = result.params
    p_vals = result.pvalues
    #print(result.summary())
    
    return coef # updates coefficients
    