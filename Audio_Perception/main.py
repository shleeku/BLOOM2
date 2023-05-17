import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
import argparse
import numpy as np
from moviepy.editor import *
from flask import Flask, request, jsonify
from interact import main


app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def main() :
    # parsing
    if request.method == 'POST' :
        print("POST")
        file = request.files['video']
        os.makedirs('./temp', exist_ok = True)
        file.save('./temp/video.avi')

    elif request.method == 'GET' :
        print("GET")
        file = request.files['video']
        os.makedirs('./temp', exist_ok = True)
        file.save('./temp/video.avi')
    else :
        raise ValueError("Bad Request")

    # configuration here
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", dest="name", type=str, default="VIP701_audio_perception", help="NAME")
    args = parser.parse_args()

    # implementation here
    videoclip = VideoFileClip("./temp/video.avi")
    video = np.array(list(videoclip.iter_frames()))
    audio = videoclip.audio.to_soundarray()

    # chatbot

    interact.main()

    # output here
    return jsonify({'msg' : video.shape})

if __name__ == "__main__" :
    app.run(host='0.0.0.0', port=8080)