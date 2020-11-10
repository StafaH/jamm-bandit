import numpy as np
import os
import urllib
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

def main():

    # Session state for persistent values
    state = _get_state()

    if not state.rewards:
        state.rewards = []

    if not state.submitted:
        display_intro_page(state)
    else:
        display_faces_page(state)
        
    # Mandatory to avoid rollbacks with widgets, must be called at the end of your app
    state.sync()

def display_intro_page(state):

    #Instructions
    st.title("Thank you for your interest in our app!")
    st.title("Before you get a chance to look at the different faces, you will first be asked to fill out some demographic questions.")
    st.title("After answering the demographic question you will then be able to look at different faces. Please select the face that appears to be more aggressive to you by pressing either the X or Y button.")
    
    #Demographics
    st.header('Please fill this out before starting!')
    state.username = st.text_input('Enter username')
    state.age = st.number_input('Age', min_value=18, max_value=100)
    state.gender = st.selectbox('Gender', ('Male', 'Female', 'Other'))
    state.ethnicity = st.selectbox('Ethnicity', ('White', 'Hispanic', 'Black', 'Middle Eastern', 'South Asian', 'South-East Asian', 'East Asian', 'Pacific Islander', 'Native American/Indigenous'))
    state.politics = st.selectbox('Political Orientation', ('Very Liberal', 'Moderately Liberal', 'Slightly Liberal', 'Neither Liberal or Conservative', 'Very Conservative', 'Moderately Conservative', 'Slightly Conservative'))
    
    if st.button('Submit'):
        add_user_to_database(state)
        state.means = np.zeros(512)

        # TODO: Check if username is empty also check if the demographic of this user changed 
        state.submitted = True

def display_faces_page(state):
    
    st.header('Which face is more aggressive?')

    st.sidebar.slider('age', -1.00, 1.00)
    st.sidebar.slider('gender', -1.00, 1.00)
    st.sidebar.slider('smile', -1.00, 1.00)
    st.sidebar.slider('pitch', -1.00, 1.00)
    st.sidebar.slider('roll', -1.00, 1.00)
    st.sidebar.slider('yaw', -1.00, 1.00)
    st.sidebar.slider('eye-eyebrow distance', -1.00, 1.00)
    st.sidebar.slider('eye distance', -1.00, 1.00)
    st.sidebar.slider('eye ratio', -1.00, 1.00)
    st.sidebar.slider('eyes open', -1.00, 1.00)
    st.sidebar.slider('nose ratio', -1.00, 1.00)
    st.sidebar.slider('nose tip', -1.00, 1.00)
    st.sidebar.slider('nose-mouth distance', -1.00, 1.00)
    st.sidebar.slider('mouth ratio', -1.00, 1.00)
    st.sidebar.slider('mouth open', -1.00, 1.00)
    st.sidebar.slider('lip ratio', -1.00, 1.00)

    # Download the model file
    download_file('Gs.pth')
    
    # Load the StyleGAN2 Model
    G = load_model()
    G.eval()
    
    client = get_database_connection()
    results = client.results
    basic = results['basic']

    myquery = { "username": state.username }
    user_dict = basic.find_one(myquery)
    
    rewards_list = list(user_dict['rewards'])
    weights_list = list(user_dict['weights'])
    
    # Completely random sampling
    weights = bandit_algos.random_latents()
    
    # Magnitude shift sampling
    # if database for username is empty, all means are 0
    if len(rewards_list) > 0:
        weights, state.means = bandit_algos.magnitude_shift(state.means, rewards_list[-1])
        weights = np.asarray(weights)
    
    # Thompson Sampling 
    state.models = [NIGNormal(mu=0, v=1, alpha=1, beta=1) for latent in range(512)]
    x = 0

    #weights = np.asarray([model.draw_expected_value(x) for model in state.models])

    # Generate the image
    image_out = generate_image(G, weights)

    #weights = bandit_algos.random_latents()
    #image_out2 = generate_image(G, weights)
    
    # Output the image
    col1, col2 = st.beta_columns(2)
    st.image(image_out, use_column_width=True)
    #col2.image(image_out2, use_column_width=True)
    if col1.button('No'):
        rewards_list.append(0)
        weights_list.append(list(weights))
        basic.update_one({'username': state.username}, {'$set':{'rewards': rewards_list}})
        basic.update_one({'username': state.username}, {'$set':{'weights': weights_list}})
        
    
    if col2.button('Yes'):
        rewards_list.append(1)
        weights_list.append(list(weights))
        basic.update_one({'username': state.username}, {'$set':{'rewards': rewards_list}})
        basic.update_one({'username': state.username}, {'$set':{'weights': weights_list}})
        
    if st.button('There is something wrong with this picture!'):
        pass
    
    st.markdown(f'Faces Viewed = {len(rewards_list)} times.')
    
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


def add_user_to_database(state): 
    client = get_database_connection()

    results = client.results
    basic = results['basic']

    myquery = { "username": state.username }
    user_dict = basic.find(myquery)
    user_dict = list(user_dict)

    if not user_dict:
        new_user = {
            'username': state.username,
            'age': state.age,
            'gender': state.gender,
            'ethnicity': state.ethnicity,
            'politics': state.politics,
            'rewards': [],
            'weights': np.zeros((1,512)).tolist(),
            'final_dist': np.zeros((512,4)).tolist() # mu, sigma, alpha, beta
        }
        basic.insert_one(new_user)
    

@st.cache(allow_output_mutation=True, hash_funcs={MongoClient: id})
def get_database_connection():
    return MongoClient("mongodb+srv://jammadmin:jamm2020@cluster0.qch9t.mongodb.net/jamm?retryWrites=true&w=majority")


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
        #img.save('static/images/seed6600.png')
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


class _SessionState:
    
    def __init__(self, session, hash_funcs):
        """Initialize SessionState instance."""
        self.__dict__["_state"] = {
            "data": {},
            "hash": None,
            "hasher": _CodeHasher(hash_funcs),
            "is_rerun": False,
            "session": session,
        }
        
    def __call__(self, **kwargs):
        """Initialize state data once."""
        for item, value in kwargs.items():
            if item not in self._state["data"]:
                self._state["data"][item] = value
                
    def __getitem__(self, item):
        """Return a saved state value, None if item is undefined."""
        return self._state["data"].get(item, None)
        
    def __getattr__(self, item):
        """Return a saved state value, None if item is undefined."""
        return self._state["data"].get(item, None)
    
    def __setitem__(self, item, value):
        """Set state value."""
        self._state["data"][item] = value
        
    def __setattr__(self, item, value):
        """Set state value."""
        self._state["data"][item] = value
    
    def clear(self):
        """Clear session state and request a rerun."""
        self._state["data"].clear()
        self._state["session"].request_rerun()
    
    def sync(self):
        """Rerun the app with all state values up to date from the beginning to fix rollbacks."""
        # Ensure to rerun only once to avoid infinite loops
        # caused by a constantly changing state value at each run.
        #
        # Example: state.value += 1
        if self._state["is_rerun"]:
            self._state["is_rerun"] = False
        
        elif self._state["hash"] is not None:
            if self._state["hash"] != self._state["hasher"].to_bytes(self._state["data"], None):
                self._state["is_rerun"] = True
                self._state["session"].request_rerun()
        self._state["hash"] = self._state["hasher"].to_bytes(self._state["data"], None)

def _get_session():
    session_id = get_report_ctx().session_id
    session_info = Server.get_current()._get_session_info(session_id)
    if session_info is None:
        raise RuntimeError("Couldn't get your Streamlit Session object.")
    
    return session_info.session

def _get_state(hash_funcs=None):
    session = _get_session()
    if not hasattr(session, "_custom_session_state"):
        session._custom_session_state = _SessionState(session, hash_funcs)
    return session._custom_session_state

if __name__ == "__main__":
    main()