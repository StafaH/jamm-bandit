import os
import urllib
from pathlib import Path

import numpy as np
from pymongo import MongoClient


def main():

    client = MongoClient("mongodb+srv://jammadmin:jamm2020@cluster0.qch9t.mongodb.net/jamm?retryWrites=true&w=majority")
    results = client.results
    arm = results['arms']
   
    for i in range(0, 3000):
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

        
if __name__ == "__main__":
    main()