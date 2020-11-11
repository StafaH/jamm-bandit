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

def pregen_arms(n):
    # Download the model file
    download_file('Gs.pth')
    
    # Load the StyleGAN2 Model
    G = load_model()
    G.eval()
    
    client = get_database_connection()
    results = client.results
    arm = results['arms']
   
    for i in range(0, n):
        new_arm = {
            'id': i,
            'alpha': 1,
            'beta': 1,
            'n_wins': 0,
            'n_losses': 0,
            'win_usernames': [],
            'loss_usernames': [],
            'living': True
        }
        arm.insert_one(new_arm)