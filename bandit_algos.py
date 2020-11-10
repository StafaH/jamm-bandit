import streamlit as st
from streamlit.report_thread import get_report_ctx
from streamlit.server.server import Server
from streamlit.hashing import _CodeHasher

import numpy as np
import statsmodels.api as sm

def random_latents():
    weights = np.random.normal(loc=0, scale=1.0, size=(512,))
    return weights

def bayesian_linear_reg(observation, prior):
    weights = np.random.normal(loc=0, scale=1.0, size=(512,))
    return weights

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
    