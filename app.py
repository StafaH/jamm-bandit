import warnings
import argparse
import os

#from PIL import Image
import numpy as np

#import stylegan2
#from stylegan2 import utils

from flask import Flask, request, make_response, render_template, send_from_directory
IMAGES_FOLDER = os.path.join('static', 'images')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = IMAGES_FOLDER


@app.route("/")
@app.route('/index')
def gen_image():
    full_filename = os.path.join(app.config['UPLOAD_FOLDER'], '2.png')
    return render_template('webUI.html', image = full_filename)
