import streamlit as st
from streamlit.report_thread import get_report_ctx
from streamlit.server.server import Server
from streamlit.hashing import _CodeHasher

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

def logistic_reg(weights, observations):
    
    '''
    weights = np.random.normal(loc=coef[0], scale=1.0, size=(trials,1)) # weights sampled from a normal dist
    for c in coef[1:]:
        weight = np.random.normal(loc=c, scale=1.0, size=(trials,1))
        weights = np.hstack((weights, weight))
    '''
    #labels = rand_bin_array(weights, trials)
    
    result = sm.Logit(observations, weights).fit()
    weights = result.params
    st.text(result.params)
    weights = np.clip(weights, -3, 3)
    st.text(result.summary())
    #p_vals = result.pvalues
    #print(result.summary())
    
    return weights # updates coefficients
    