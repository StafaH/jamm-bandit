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

@app.route("/")
@app.route('/index')
def gen_image():
    generate_image(G)
    full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'seed6600.png')
    return render_template('webUI.html', image = full_filename)
