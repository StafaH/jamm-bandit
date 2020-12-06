import os
import urllib
from pathlib import Path

from PIL import Image
import requests
from io import BytesIO
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
    
    if not state.page:
        state.page = 0
    
    if not state.initialized:
        state.initialized = True
        state.controls = get_control_latent_vectors('stylegan2directions/')

    if state.page == 0:
        display_intro_page(state)
    elif state.page == 1:
        display_faces_page(state)
    elif state.page == 2:
        display_exit_page(state)

    state.sync()


def display_intro_page(state):
    # Text Instructions for users
    st.title("Thank you for your interest in our app!")
    st.title("You will be shown two different faces at a time. Please select the face that answers the question best by pressing the appropriate buttons.")
    
    # Collect Demographic Information
    st.header('Please enter a username and the type of experiment before starting!')
    state.username = st.text_input('Enter username')
    state.age = 'N/A'
    state.gender = 'N/A'
    state.ethnicity = 'N/A'
    state.politics = 'N/A'
    state.images_seen = []
    
    state.experiment = st.selectbox('Experiment', ('Random', 'Thompson Sampling'))#, 'Mortal Thompson Sampling'))
    
    # Add user to the database using demographic information (if they do not exist)
    if st.button('Submit'):
        add_user_to_database(state)
        state.page = 1

def display_exit_page(state):
    st.title("Thank you for participating!")
    st.title("Before you leave, please fill out the following demographic questions.")
    state.age = st.number_input('Age', min_value=18, max_value=100)
    state.gender = st.selectbox('Gender', ('Male', 'Female', 'Other'))
    state.ethnicity = st.selectbox('Ethnicity', ('White', 'Hispanic', 'Black', 'Middle Eastern', 'South Asian', 'South-East Asian', 'East Asian', 'Pacific Islander', 'Native American/Indigenous'))
    state.politics = st.selectbox('Political Orientation', ('Very Liberal', 'Moderately Liberal', 'Slightly Liberal', 'Neither Liberal or Conservative', 'Very Conservative', 'Moderately Conservative', 'Slightly Conservative'))
    if st.button('Submit'):
        st.text('Thank you for submitting!')
        update_user_db(state)
    if st.button('Go back'):
        state.page = 1
    
def display_faces_page(state):
    
    st.header('Which face is more dominant?')
    
    st.text('Please try to attempt viewing at least 50 image pairs for each experiment trial!')
    
    client = get_database_connection()
    if state.experiment == "Random":
        results = client.resultsRandom
    elif state.experiment == "Thompson Sampling":
        results = client.resultsTS
    elif state.experiment == "Mortal Thompson Sampling":
        results = client.resultsMortalTS
    arms = results['arms']
    users = results['users']
    
    # Query mongodb collection for the document with this username, it returns a dictionary of keys and values
    myquery = { 'username': state.username }
    user_dict = users.find_one(myquery)
    
    # seen = len(list(user_dict['images seen']))
    # if seen < 400:
    
    samples = []
    for arm in arms.find():
        if arm['living'] == True and arm['id'] not in user_dict['images seen']:
            sample = (np.random.beta(arm['alpha'], arm['beta']), arm['id'])
            samples.append(sample)
    
    largest = max(samples, key=itemgetter(0))
    samples.remove(largest)
    secondlargest = max(samples, key=itemgetter(0))
    
    users.update_one(myquery, { '$push': {'images seen': largest[1] } })
    users.update_one(myquery, { '$push': {'images seen': secondlargest[1] } })
    
    if np.random.randint(1, 10) > 5:
        temp = largest
        largest = secondlargest
        secondlargest = temp
    
    query1 = { 'id': largest[1] }
    query2 = { 'id': secondlargest[1] }
    
    seed1 = str(arms.find_one(query1)['seed']) + '.png'
    seed2 = str(arms.find_one(query2)['seed']) + '.png'
    
    response1 = requests.get('https://stylegan2-pytorch-ffhq-config-f.s3.ca-central-1.amazonaws.com/data/images/'+str(seed1))
    response2 = requests.get('https://stylegan2-pytorch-ffhq-config-f.s3.ca-central-1.amazonaws.com/data/images/'+str(seed2))

    image1 = Image.open(BytesIO(response1.content))
    image2 = Image.open(BytesIO(response2.content))
    
    # Output the image
    col1, col2 = st.beta_columns(2)
    col1.image(image1, use_column_width=True)
    col2.image(image2, use_column_width=True)
    #col1.text(largest[1])
    #col2.text(secondlargest[1])
        
    if col1.button('Left'):
        if state.experiment == "Thompson Sampling" or state.experiment == "Mortal Thompson Sampling":
            arms.update_one(query1, { '$inc' : { 'alpha' : 1 } } )
            arms.update_one(query2, { '$inc' : { 'beta' : 1 } } )
       
        arms.update_one(query1, { '$inc' : { 'n_wins' : 1 } } ) 
        arms.update_one(query2, { '$inc' : { 'n_losses' : 1 } } )
        
    if col2.button('Right'):
        if state.experiment == "Thompson Sampling" or state.experiment == "Mortal Thompson Sampling":
            arms.update_one(query2, { '$inc' : { 'alpha' : 1 } } )
            arms.update_one(query1, { '$inc' : { 'beta' : 1 } } )
        
        arms.update_one(query2, { '$inc' : { 'n_wins' : 1 } } )
        arms.update_one(query1, { '$inc' : { 'n_losses' : 1 } } )
     
    if state.experiment == "Mortal Thompson Sampling":
        image_dict1 = arms.find_one(query1)
        image_dict2 = arms.find_one(query2)
        
        if image_dict1['n_losses'] > 10:
            arms.update_one(query1, { '$set' : { 'living' : False } } )
        if image_dict2['n_losses'] > 10:
            arms.update_one(query2, { '$set' : { 'living' : False } } )
    
    st.text("")
    st.text("You have seen " + str(int(len(list(user_dict['images seen']))/2)) + " pairs of faces")
    if st.button("Finish"):
        state.page = 2
    #else:
    #    st.title("You've reached the end! Thank you for participating :)")

def add_user_to_database(state): 
    client = get_database_connection()
    
    if state.experiment == "Random":
        results = client.resultsRandom
    elif state.experiment == "Thompson Sampling":
        results = client.resultsTS
    elif state.experiment == "Mortal Thompson Sampling":
        results = client.resultsMortalTS

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

def update_user_db(state):
    client = get_database_connection()
    
    if state.experiment == "Random":
        results = client.resultsRandom
    elif state.experiment == "Thompson Sampling":
        results = client.resultsTS
    elif state.experiment == "Mortal Thompson Sampling":
        results = client.resultsMortalTS
        
    users = results['users']
    myquery = { 'username': state.username }
    users.update_one(myquery, { '$set' : { 'age' : state.age } } )
    users.update_one(myquery, { '$set' : { 'gender' : state.gender } } )    
    users.update_one(myquery, { '$set' : { 'ethnicity' : state.ethnicity } } )    
    users.update_one(myquery, { '$set' : { 'politics' : state.politics } } )    
    

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
def download_file(file_path, url):
    # Don't download the file twice. (If possible, verify the download using the file length.)
    if os.path.exists(file_path):
        return
    
    # These are handles to two visual elements to animate.
    weights_warning, progress_bar = None, None
    try:
        weights_warning = st.warning("Downloading %s..." % file_path)
        progress_bar = st.progress(0)
        with open(file_path, "wb") as output_file:
            with urllib.request.urlopen(url) as response:
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