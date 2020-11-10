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
import pickle

import bandit_algos
from nig_normal import *
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
    
    if st.button('Submit'):
        add_user_to_database(session)

        # TODO: Check if username is empty also check if the demographic of this user changed 
        session.submitted = True

def display_faces_page(session):
    
    st.header('Which face is more aggressive?')

    # Download the model file
    download_file('Gs.pth')
    
    # Load the StyleGAN2 Model
    G = load_model()
    G.eval()
    
    client = get_database_connection()
    results = client.results
    basic = results['basic']

    myquery = { "username": session.username }
    user_dict = basic.find_one(myquery)
    
    rewards_list = list(user_dict['rewards'])
    weights_list = list(user_dict['weights'])
    final_list = list(user_dict['final_dist'])
    
    # Completely random sampling
    if not session.first_random:
        weights = bandit_algos.random_latents()
        session.weights = weights
        session.first_random = True
    
    # Magnitude shift sampling
    # if database for username is empty, all means are 0
    # if len(rewards_list) > 0:
    #     means = []
    #     for i in range(0, len(final_list)):
    #         means.append(final_list[i][0])
    #     weights, means = bandit_algos.magnitude_shift(means, rewards_list[-1])
    #     weights = np.asarray(weights)
    
    

    # Thompson Sampling
    
    x = 0

    # Generate the image
    weights = session.weights
    weights = weights + (session.age_magnitude * session.controls['age'])
    weights = weights + (session.gender_magnitude * session.controls['gender'])
    weights = weights + (session.smile_magnitude * session.controls['smile'])
    weights = weights + (session.pitch_magnitude * session.controls['pitch'])
    weights = weights + (session.roll_magnitude * session.controls['roll'])
    weights = weights + (session.yaw_magnitude * session.controls['yaw'])
    weights = weights + (session.eyebrow_magnitude * session.controls['eye_eyebrow_distance'])
    weights = weights + (session.eyedist_magnitude * session.controls['eye_distance'])
    weights = weights + (session.eyeratio_magnitude * session.controls['eye_ratio'])
    weights = weights + (session.eyeopen_magnitude * session.controls['eyes_open'])
    weights = weights + (session.noseratio_magnitude * session.controls['nose_ratio'])
    weights = weights + (session.nosetip_magnitude * session.controls['nose_tip'])
    weights = weights + (session.nousemouthdist_magnitude * session.controls['nose_mouth_distance'])
    weights = weights + (session.mouthratio_magnitude * session.controls['mouth_ratio'])
    weights = weights + (session.mouthopen_magnitude * session.controls['mouth_open'])
    weights = weights + (session.lipratio_magnitude * session.controls['lip_ratio'])
    
    image_out = generate_image(G, weights)
    
    # Output the image
    col1, col2 = st.beta_columns(2)
    st.image(image_out, use_column_width=True)
    #col2.image(image_out2, use_column_width=True)

    if col1.button('No'):
        rewards_list.append(0)
        weights_list.append(list(weights))
        basic.update_one({'username': session.username}, {'$set':{'rewards': rewards_list}})
        basic.update_one({'username': session.username}, {'$set':{'weights': weights_list}})
        
    
    if col2.button('Yes'):
        rewards_list.append(1)
        weights_list.append(list(weights))
        basic.update_one({'username': session.username}, {'$set':{'rewards': rewards_list}})
        basic.update_one({'username': session.username}, {'$set':{'weights': weights_list}})
        
    if st.button('There is something wrong with this picture!'):
        pass
    
    st.markdown(f'Faces Viewed = {len(rewards_list)} times.')
    
    # final_params = []
    # for i in range(0, len(means)):
    #     params = [means[i], 1, 1, 1]
    #     final_params.append(params)
    # basic.update_one({'username': session.username}, {'$set':{'final_dist': final_params}})   
        
    #params = list(user_dict['final_dist'])
    #final_params = list(user_dict['final_dist'])
    '''
    final_params = []
    for model in state.models:
        model.update_posterior(x, rewards_list[-1])
        params = [model.mu, model.v, model.alpha, model.beta]
        final_params.append(params)
    basic.update_one({'username': state.username}, {'$set':{'final_dist': final_params}})
    '''

def display_feature_sidebar(session):
    session.age_magnitude = st.sidebar.slider('age', -10.00, 10.00, 0.0, 0.1)
    session.gender_magnitude = st.sidebar.slider('gender', -10.00, 10.00, 0.0, 0.1)
    session.smile_magnitude = st.sidebar.slider('smile', -10.00, 10.00, 0.0, 0.1)
    session.pitch_magnitude = st.sidebar.slider('pitch', -10.00, 10.00, 0.0, 0.1)
    session.roll_magnitude = st.sidebar.slider('roll', -10.00, 10.00, 0.0, 0.1)
    session.yaw_magnitude = st.sidebar.slider('yaw', -10.00, 10.00, 0.0, 0.1)
    session.eyebrow_magnitude = st.sidebar.slider('eye-eyebrow distance', -10.00, 10.00, 0.0, 0.1)
    session.eyedist_magnitude = st.sidebar.slider('eye distance', -10.00, 10.00, 0.0, 0.1)
    session.eyeratio_magnitude = st.sidebar.slider('eye ratio', -10.00, 10.00, 0.0, 0.1)
    session.eyeopen_magnitude = st.sidebar.slider('eyes open', -10.00, 10.00, 0.0, 0.1)
    session.noseratio_magnitude = st.sidebar.slider('nose ratio', -10.00, 10.00, 0.0, 0.1)
    session.nosetip_magnitude = st.sidebar.slider('nose tip', -1.00, 1.00)
    session.nousemouthdist_magnitude = st.sidebar.slider('nose-mouth distance', -10.00, 10.00, 0.0, 0.1)
    session.mouthratio_magnitude = st.sidebar.slider('mouth ratio', -10.00, 10.00, 0.0, 0.1)
    session.mouthopen_magnitude = st.sidebar.slider('mouth open', -10.00, 10.00, 0.0, 0.1)
    session.lipratio_magnitude = st.sidebar.slider('lip ratio', -10.00, 10.00, 0.0, 0.1)


def initialize_thompson_sampling(session):
    session.models = [NIGNormal(mu=0, v=1, alpha=1, beta=1) for latent in range(512)]

def add_user_to_database(session): 
    client = get_database_connection()

    results = client.results
    basic = results['basic']

    myquery = { "username": session.username }
    user_dict = basic.find(myquery)
    user_dict = list(user_dict)

    if not user_dict:
        new_user = {
            'username': session.username,
            'age': session.age,
            'gender': session.gender,
            'ethnicity': session.ethnicity,
            'politics': session.politics,
            'rewards': [],
            'weights': np.zeros((1,512)).tolist(),
            'final_dist': np.zeros((512,4)).tolist() # mu, sigma, alpha, beta
        }
        basic.insert_one(new_user)
    

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