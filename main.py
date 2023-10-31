# all the common libraries
import gc
from PIL import Image
import numpy as np
import json
import os
import streamlit as st
import torch

# all the image models are imported here
from models.image.clip import *
from models.image.blip import *
from models.image.align import *
from models.image.blip2 import *


# all the video models are imported here
from models.video.clip4clip import *
from models.video.xclip import *


# all the common variables are imported here
models_config = json.load(open('config.json', 'r'))
device = "cuda" if torch.cuda.is_available() else "cpu"


cache_dir = '/ssd_scratch/cvit/hardk/cache/'
print("Setting hugging face transformers cache folder to : ", cache_dir)
os.environ['TRANSFORMERS_CACHE'] = cache_dir
torch.hub.set_dir(cache_dir)

gc.enable()

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("# Welcome to Katha AI! 👋")


st.markdown(
    """
    Katha AI is a research group led by Makarand Tapaswi at IIIT Hyderabad. 
    We are interested in building machines that can understand the world around them, especially by learning from movies and videos.

    **👈 Select Model Library from the sidebar** to play around with a range of image and video based models.

    ### Want to learn more?
    - Check out [our github](https://github.com/katha-ai)

    ### See Papers from Katha AI
    - How you feelin' ? [Learning Emotions and Mental States in Movie Scenes](https://katha-ai.github.io/projects/emotx/)
    
"""
)