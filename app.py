import warnings
import argparse
import os

from PIL import Image
import numpy as np
import torch

#import stylegan2
#from stylegan2 import utils

from flask import Flask, request, make_response, render_template, send_from_directory
IMAGES_FOLDER = os.path.join('static', 'images')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = IMAGES_FOLDER

# Load the StyleGAN2 Model
#G = stylegan2.models.load('Gs.pth')
#G.eval()
# generate_images(G, args)

'''
def generate_image(G, output = '/images', truncation_psi = 1, gpu = [], seeds = [6600], batch_size = 1, pixel_min = 1, pixel_max = 1):
    latent_size, label_size = G.latent_size, G.label_size
    #device = torch.device(args.gpu[0] if args.gpu else 'cpu')
    device = torch.device('cpu')
    if device.index is not None:
        torch.cuda.set_device(device.index)
    G.to(device)
    if truncation_psi != 1:
        G.set_truncation(truncation_psi=truncation_psi)
    if len(gpu) > 1:
        warnings.warn(
            'Noise can not be randomized based on the seed ' + \
            'when using more than 1 GPU device. Noise will ' + \
            'now be randomized from default random state.'
        )
        G.random_noise()
        G = torch.nn.DataParallel(G, device_ids=gpu)
    else:
        noise_reference = G.static_noise()

    def get_batch(seeds):
        latents = []
        labels = []
        if len(gpu) <= 1:
            noise_tensors = [[] for _ in noise_reference]
        for seed in seeds:
            rnd = np.random.RandomState(seed)
            latents.append(torch.from_numpy(rnd.randn(latent_size)))
            #latents.append(torch.from_numpy(np.zeros(512)))
            # Get the Latents from file
            #latents.append(torch.from_numpy(np.genfromtxt('seed6600.txt', delimiter=" ")))
            
            if len(gpu) <= 1:
                for i, ref in enumerate(noise_reference):
                    noise_tensors[i].append(torch.from_numpy(rnd.randn(*ref.size()[1:])))
            if label_size:
                labels.append(torch.tensor([rnd.randint(0, label_size)]))
        latents = torch.stack(latents, dim=0).to(device=device, dtype=torch.float32)
        if labels:
            labels = torch.cat(labels, dim=0).to(device=device, dtype=torch.int64)
        else:
            labels = None
        if len(gpu) <= 1:
            noise_tensors = [
                torch.stack(noise, dim=0).to(device=device, dtype=torch.float32)
                for noise in noise_tensors
            ]
        else:
            noise_tensors = None
        return latents, labels, noise_tensors

    #progress = utils.ProgressWriter(len(args.seeds))
    #progress.write('Generating images...', step=False)

    for i in range(0, len(seeds), batch_size):
        latents, labels, noise_tensors = get_batch(seeds[i: i + batch_size])
        if noise_tensors is not None:
            G.static_noise(noise_tensors=noise_tensors)
        with torch.no_grad():
            generated = G(latents, labels=labels)

        images = utils.tensor_to_PIL(
            generated, pixel_min=pixel_min, pixel_max=pixel_max)
        for seed, img in zip(seeds[i: i + batch_size], images):
            #img.save(os.path.join(output, 'seed%04d.png' % seed))
            print("DONE!")
            return img
            #progress.step()
            # Output Latents as txt file
            # with open('seed%04d.txt' % seed, 'w') as f:
            #     for item in latents:
            #         f.write("%s, " % item)
            #     f.write("\n" % item)

            #np.savetxt('seed%04d.txt' % seed, latents.numpy())
            
            # Output Latents as tensor
            #torch.save(latents, 'seed%04d.pt' % seed)

    #progress.write('Done!', step=False)
    #progress.close()
'''

@app.route("/")
@app.route('/index')
def gen_image():
    #image = generate_image(G)
    full_filename = os.path.join(app.config['UPLOAD_FOLDER'], '1.png')
    return render_template('webUI.html', image = full_filename)