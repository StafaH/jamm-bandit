import os
import urllib
from pathlib import Path

import numpy as np
import torch
from operator import itemgetter
import stylegan2
from stylegan2 import utils
from pymongo import MongoClient

import streamlit as st
from streamlit.hashing import _CodeHasher
from streamlit.report_thread import get_report_ctx
from streamlit.server.server import Server

def main():

    state = _get_state()
    
    if not state.initialized:
        state.initialized = True
        state.controls = get_control_latent_vectors('stylegan2directions/')

    if not state.submitted:
        display_intro_page(state)
    else:
        display_faces_page(state)

    state.sync()


def display_intro_page(state):
    # Text Instructions for users
    st.title("Thank you for your interest in our app!")
    st.title("Before you get a chance to look at the different faces, you will first be asked to fill out some demographic questions.")
    st.title("After answering the demographic question you will then be able to look at different faces. Please select the face that appears to be more aggressive to you by pressing either the X or Y button.")
    
    # Collect Demographic Information
    st.header('Please fill this out before starting!')
    state.username = st.text_input('Enter username')
    state.age = st.number_input('Age', min_value=18, max_value=100)
    state.gender = st.selectbox('Gender', ('Male', 'Female', 'Other'))
    state.ethnicity = st.selectbox('Ethnicity', ('White', 'Hispanic', 'Black', 'Middle Eastern', 'South Asian', 'South-East Asian', 'East Asian', 'Pacific Islander', 'Native American/Indigenous'))
    state.politics = st.selectbox('Political Orientation', ('Very Liberal', 'Moderately Liberal', 'Slightly Liberal', 'Neither Liberal or Conservative', 'Very Conservative', 'Moderately Conservative', 'Slightly Conservative'))
    state.images_seen = []
    
    # Add user to the database using demographic information (if they do not exist)
    if st.button('Submit'):
        add_user_to_database(state)
        state.submitted = True

def display_faces_page(state):
    
    st.header('Which face is more assertive?')
    
    client = get_database_connection()
    results = client.results
    arms = results['mortalbandit']
    users = results['users']
    
    # Query mongodb collection for the document with this username, it retusn a dictionary of keys and values
    myquery = { 'username': state.username }
    user_dict = users.find_one(myquery)
    
    samples = []
    for i in range(0, len(arms)):
        arm = arms[i]
        if arm['living'] == True and arm['id'] not in user_dict['images_seen']:
            sample = (np.random.beta(arm['alpha'], arm['beta']), i)
            samples.append(sample)
    largest = max(samples, key=itemgetter(0))
    samples.remove(largest)
    secondlargest = max(samples, key=itemgetter(0))
                        
    # image1 = get image from AWS idk how to do that lol - this one is the largest
    # image2 = get image from AWS idk how to do that lol - this one is the second largest
    
    # Output the image
    col1, col2 = st.beta_columns(2)
    col1.image(image1, use_column_width=True)
    col2.image(image2, use_column_width=True)
    
    users.update_one(myquery, { '$push': {'images seen': largest[1] } })
    users.update_one(myquery, { '$push': {'images seen': secondlargest[1] } })
    
    query1 = { 'id': largest[1] }
    query2 = { 'id': secondlargest[1] }
        
    if col1.button('Left'):
        arms.update_one(query1, { '$inc' : { 'alpha' : 1 } } )
        arms.update_one(query1, { '$inc' : { 'n_wins' : 1 } } ) 
        
        arms.update_one(query2, { '$inc' : { 'beta' : 1 } } )
        arms.update_one(query2, { '$inc' : { 'n_losses' : 1 } } )
        
    if col2.button('Right'):
        arms.update_one(query2, { '$inc' : { 'alpha' : 1 } } )
        arms.update_one(query2, { '$inc' : { 'n_wins' : 1 } } )
    
        arms.update_one(query1, { '$inc' : { 'beta' : 1 } } )
        arms.update_one(query1, { '$inc' : { 'n_losses' : 1 } } )
     
    image_dict1 = arms.find_one(query1)
    image_dict2 = arms.find_one(query2)
    
    if image_dict1['n_losses'] > 10:
        arms.update_one(query1, { '$set' : { 'living' : False } } )
    if image_dict2['n_losses'] > 10:
        arms.update_one(query2, { '$set' : { 'living' : False } } )

def add_user_to_database(state): 
    client = get_database_connection()
    
    results = client.results
    users = results['users']
    myquery = { 'username': state.username }
    user_dict = users.find(myquery)
    user_dict = list(user_dict)
    
    if not user_dict:
        new_user = {
            'username': state.username,
            'age': state.age,
            'gender': state.gender,
            'ethnicity': state.ethnicity,
            'politics': state.politics,  
            'images seen': state.images_seen
        }
        users.insert_one(new_user)


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