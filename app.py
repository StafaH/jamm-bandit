import warnings
import argparse
import os

from PIL import Image
import numpy as np
import torch

import stylegan2
from stylegan2 import utils

from flask import Flask, request, make_response, render_template, send_from_directory

IMAGES_FOLDER = os.path.join('static', 'images')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = IMAGES_FOLDER

# Load the StyleGAN2 Model
G = stylegan2.models.load('Gs.pth')
G.eval()

def generate_image(G):
    latent_size, label_size = G.latent_size, G.label_size

    device = torch.device('cpu')
    if device.index is not None:
        torch.cuda.set_device(device.index)
    G.to(device)

    G.set_truncation(0.5)
    noise_reference = G.static_noise()
    noise_tensors = [[] for _ in noise_reference]

    latents = []
    labels = []
    rnd = np.random.RandomState(6600)
    latents.append(torch.from_numpy(rnd.randn(latent_size)))
            

    for i, ref in enumerate(noise_reference):
        noise_tensors[i].append(torch.from_numpy(rnd.randn(*ref.size()[1:])))

    if label_size:
        labels.append(torch.tensor([rnd.randint(0, label_size)]))
    
    latents = torch.stack(latents, dim=0).to(device=device, dtype=torch.float32)
    if labels:
        labels = torch.cat(labels, dim=0).to(device=device, dtype=torch.int64)
    else:
        labels = None
    
    noise_tensors = [torch.stack(noise, dim=0).to(device=device, dtype=torch.float32) for noise in noise_tensors]

    if noise_tensors is not None:
        G.static_noise(noise_tensors=noise_tensors)
    with torch.no_grad():
        generated = G(latents, labels=labels)
    
    images = utils.tensor_to_PIL(
        generated, pixel_min=-1, pixel_max=1)
    
    for img in images:
        img.save('static/images/seed6600.png')
        #return img
    print("DONE!")
    
    
    
import arviz as az
import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm
import statsmodels.formula.api as sm
import pandas as pd, numpy as np, re
from contextualbandits.online import LinTS

alpha, sigma = 1, 1
beta = [1, 2.5]

size = 100

X1 = np.random.randn(size)
X2 = np.random.randn(size) * 0.2
Y = alpha + beta[0] * X1 + beta[1] * X2 + np.random.randn(size) * sigma

basic_model = pm.Model()

with basic_model:

    # Priors for unknown model parameters
    alpha = pm.Normal("alpha", mu=0, sigma=10)
    beta = pm.Normal("beta", mu=0, sigma=10, shape=2)
    sigma = pm.HalfNormal("sigma", sigma=1)

    # Expected value of outcome
    mu = alpha + beta[0] * X1 + beta[1] * X2

    # Likelihood (sampling distribution) of observations
    Y_obs = pm.Normal("Y_obs", mu=mu, sigma=sigma, observed=Y)
    
    

# class to implement our contextual bandit setting
class ContextualMAB:
    
    # initialization
    def __init__(self):
        
        # we build two bandits
        self.weights = {}
        self.weights[0] = [0.0, 1.6]
        self.weights[1] = [0.0, 0.4]
    
    # method for acting on the bandits
    def draw(self, k, x):
        
        # probability dict
        prob_dict = {}
        
        # loop for each bandit
        for bandit in self.weights.keys():
        
            # linear function of external variable
            f_x = self.weights[bandit][0] + self.weights[bandit][1]*x

            # generate reward with probability given by the logistic
            probability = 1/(1 + np.exp(-f_x))
            
            # appending to dict
            prob_dict[bandit] = probability
        
        # give reward according to probability
        return np.random.choice([0,1], p=[1 - prob_dict[k], prob_dict[k]]), max(prob_dict.values()) - prob_dict[k], prob_dict[k]


from thompson_sampling.bernoulli import BernoulliExperiment
from thompson_sampling.priors import BetaPrior

pr = BetaPrior()
pr.add_one(mean=0.5, variance=0.1, effective_size=512, label="option1")
pr.add_one(mean=0.5, variance=0.1, effective_size=512, label="option2")
pr.add_one(mean=0.5, variance=0.1, effective_size=512, label="option3")
pr.add_one(mean=0.5, variance=0.1, effective_size=512, label="option4")
experiment = BernoulliExperiment(priors=pr)

experiment.choose_arm()

rewards = [{"label":"option1", "reward":1}, {"label":"option2", "reward":1}, 
           {"label":"option3", "reward":1}, {"label":"option4", "reward":1}]
experiment.add_rewards(rewards)    


from contextualbandits.online import ParametricTS


@app.route("/")
@app.route('/index')
def gen_image():
    generate_image(G)
    full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'seed6600.png')
    return render_template('webUI.html', image = full_filename)
