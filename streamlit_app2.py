import numpy as np
import os
import urllib
from pathlib import Path
from datetime import datetime
import torch
import stylegan2
from stylegan2 import utils
import streamlit as st
from streamlit.report_thread import get_report_ctx
from streamlit.server.server import Server
from streamlit.hashing import _CodeHasher
from pymongo import MongoClient

import bandit_algos
import SessionState

def main():

    session = SessionState.get(initialized = False, submitted = False, first_random = False)
    
    if not session.initialized:
        initialize_thompson_sampling(session)
        session.initialized = True
        session.controls = get_control_latent_vectors('stylegan2directions/')

    if not session.submitted:
        display_intro_page(session)
    else:
        display_feature_sidebar(session)
        display_faces_page(session)


def display_intro_page(session):
    #Instructions
    st.title("Thank you for your interest in our app!")
    st.title("Before you get a chance to look at the different faces, you will first be asked to fill out some demographic questions.")
    st.title("After answering the demographic question you will then be able to look at different faces. Please select the face that appears to be more aggressive to you by pressing either the X or Y button.")
    
    #Demographics
    st.header('Please fill this out before starting!')
    session.username = st.text_input('Enter username')
    session.age = st.number_input('Age', min_value=18, max_value=100)
    session.gender = st.selectbox('Gender', ('Male', 'Female', 'Other'))
    session.ethnicity = st.selectbox('Ethnicity', ('White', 'Hispanic', 'Black', 'Middle Eastern', 'South Asian', 'South-East Asian', 'East Asian', 'Pacific Islander', 'Native American/Indigenous'))
    session.politics = st.selectbox('Political Orientation', ('Very Liberal', 'Moderately Liberal', 'Slightly Liberal', 'Neither Liberal or Conservative', 'Very Conservative', 'Moderately Conservative', 'Slightly Conservative'))
    session.images_seen = []
    
    if st.button('Submit'):
        add_user_to_database(session)

        # TODO: Check if username is empty also check if the demographic of this user changed 
        session.submitted = True

def display_faces_page(session):
    
    st.header('Which face is more assertive?')

    client = get_database_connection()
    results = client.results
    images = results['images']
    users = results['users']

    myquery = { "username": session.username }
    user_dict = users.find_one(myquery)
    
    
    rewards_list = list(user_dict['rewards'])
    weights_list = list(user_dict['weights'])
    
    # Completely random sampling
    if not session.first_random:
        weights = bandit_algos.random_latents()
        session.weights = weights
        session.first_random = True
    
    # Output the image
    col1, col2 = st.beta_columns(2)
    col1.image(image_out, use_column_width=True)
    col2.image(image_out2, use_column_width=True)

    if col1.button('Left'):
        rewards_list.append(0)
        weights_list.append(list(weights))
        basic.update_one({'username': session.username}, {'$set':{'rewards': rewards_list}})
        basic.update_one({'username': session.username}, {'$set':{'weights': weights_list}})
        
    
    if col2.button('Right'):
        rewards_list.append(1)
        weights_list.append(list(weights))
        basic.update_one({'username': session.username}, {'$set':{'rewards': rewards_list}})
        basic.update_one({'username': session.username}, {'$set':{'weights': weights_list}})
        
    if st.button('There is something wrong with this picture!'):
        pass
        

def add_user_to_database(session): 
    client = get_database_connection()
    
    results = client.results
    users = results['users']
    myquery = { 'username': session.username }
    user_dict = users.find(myquery)
    user_dict = list(user_dict)
    
    if not user_dict:
        new_user = {
            'username': session.username,
            'age': session.age,
            'gender': session.gender,
            'ethnicity': session.ethnicity,
            'politics': session.politics,  
            'images seen': session.images_seen
        }
        users.insert_one(new_user)

def pregen_images(n):
    # Download the model file
    download_file('Gs.pth')
    
    # Load the StyleGAN2 Model
    G = load_model()
    G.eval()
    
    client = get_database_connection()
    results = client.results
    images = results['images']
    
    for i in range(0, n):
        weights = np.random.normal(loc=0, scale=1.0, size=(512,))
        image = generate_image(G, weights)
        new_image = {
            'image': image,
            'weights': weights,
            'id': i,
            'alpha': 1,
            'beta': 1,
            'n_wins': 0,
            'n_losses': 0,
            'win_usernames': [],
            'loss_usernames': [],
            'living': True
        }
        images.insert_one(new_image)


@st.cache(allow_output_mutation=True, hash_funcs={MongoClient: id})
def get_database_connection():
    return MongoClient("mongodb+srv://jammadmin:jamm2020@cluster0.qch9t.mongodb.net/jamm?retryWrites=true&w=majority")

@st.cache
def get_control_latent_vectors(path):
    files = [x for x in Path(path).iterdir() if str(x).endswith('.npy')]
    latent_vectors = {f.name[:-4]:np.load(f) for f in files}
    return latent_vectors

@st.cache(show_spinner=False)
def generate_image(G, weights):
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
    latents.append(torch.from_numpy(weights))
            
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
        return img

@st.cache(allow_output_mutation=True)
def load_model():
    return stylegan2.models.load('Gs.pth')
    
@st.cache(suppress_st_warning=True)
def download_file(file_path):
    # Don't download the file twice. (If possible, verify the download using the file length.)
    if os.path.exists(file_path):
        return
    
    # These are handles to two visual elements to animate.
    weights_warning, progress_bar = None, None
    try:
        weights_warning = st.warning("Downloading %s..." % file_path)
        progress_bar = st.progress(0)
        with open(file_path, "wb") as output_file:
            with urllib.request.urlopen("https://d1p4vo8bv9dco3.cloudfront.net/Gs.pth") as response:
                length = int(response.info()["Content-Length"])
                counter = 0.0
                MEGABYTES = 2.0 ** 20.0
                while True:
                    data = response.read(8192)
                    if not data:
                        break
                    counter += len(data)
                    output_file.write(data)
                    # We perform animation by overwriting the elements.
                    weights_warning.warning("Downloading %s... (%6.2f/%6.2f MB)" %
                        (file_path, counter / MEGABYTES, length / MEGABYTES))
                    progress_bar.progress(min(counter / length, 1.0))
    # Finally, we remove these visual elements by calling .empty().
    
    finally:
        if weights_warning is not None:
            weights_warning.empty()
        if progress_bar is not None:
            progress_bar.empty()


if __name__ == "__main__":
    main()