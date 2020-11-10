import streamlit as st
from streamlit.report_thread import get_report_ctx
from streamlit.server.server import Server
from streamlit.hashing import _CodeHasher

import numpy as np
import statsmodels.api as sm
import math

from nig_normal import *

def random_latents():
    weights = np.random.normal(loc=0, scale=1.0, size=(512,))
    return weights

def thompson_sample(observation, prior):
    weights = np.random.normal(loc=0, scale=1.0, size=(512,))
    return weights

def magnitude_shift(means, observation):
    # means = list(np.random.rand(512)) #temp
    weights = []
    for mean in means:
        weight = float(np.random.normal(loc=mean, scale=1.0, size=(1,)))
        weights.append(weight)
    
    new_means = []
    new_weights = []
    if observation == 1: # yes
        for i in range(0, 512):
            residual = abs(weights[i] - means[i])
            if weights[i] > means[i]:
                new_mean = means[i] + math.log(residual+1) # increases mean by an amount that's log-proportional to the residual (will at higher residuals)
            elif weights[i] < means[i]:
                new_mean = means[i] - math.log(residual+1)
            new_means.append(new_mean)
    elif observation == 0: # no
        for i in range(0, 512):
            residual = abs(weights[i] - means[i])
            if weights[i] > means[i]:
                new_mean = means[i] - math.log(residual+1)
            elif weights[i] < means[i]:
                new_mean = means[i] + math.log(residual+1)    
            new_means.append(new_mean)

    for new_mean in new_means:
        new_weight = float(np.random.normal(loc=new_mean, scale=1.0, size=(1,)))
        new_weight = np.clip(new_weight, -3, 3)
        new_weights.append(new_weight)
    
    return new_weights
            
    # normal dist with mean --> sample from it
    # input is the mean and yes/no reward
    # if yes:
        # if weight > mean, increase mean by ___
        # elif weight < mean, decrease mean by ___
    # if no:
        # if weight > mean, decrease mean by ___
        # elif weight < mean, increase mean by ___
    # sample again and return those weights

def hill_climbing():
    weights = np.random.normal(loc=0, scale=1.0, size=(512,))
    return weights

def limited_thompson_sample():
    weights = np.random.normal(loc=0, scale=1.0, size=(512,))
    return weights

def logistic_reg(weights, observations):
    '''
    THIS WAS A FAILURE. THIS IS NOT GOING TO WORK! :(:(:(
    '''
    result = sm.Logit(observations, weights).fit()
    weights = result.params
    st.text(result.params)
    weights = np.clip(weights, -3, 3)
    st.text(result.summary())
    
    return weights
    