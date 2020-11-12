import os
import urllib
from pathlib import Path
import streamlit as st

import numpy as np
from pymongo import MongoClient

resultsList = ["resultsRandom", "resultsTS", "resultsMortalTS"]

def main():

    client = MongoClient("mongodb+srv://jammadmin:jamm2020@cluster0.qch9t.mongodb.net/jamm?retryWrites=true&w=majority")

    results = client.resultsMortalTS
    arm = results['arms']
    
    direc = '../../../imgs/'
    i = 0
    for filename in os.listdir(direc): 
        print(filename)
        new_arm = {
            'id': i,
            'seed': filename.split('.')[0],
            'alpha': 1,
            'beta': 1,
            'n_wins': 0,
            'n_losses': 0,
            'living': True
        }
        arm.insert_one(new_arm)
        i += 1

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


if __name__ == "__main__":
    main()