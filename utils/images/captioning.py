import gc
from PIL import Image
import streamlit as st

from models.image.blip import *

def clear_ada_cache():
    """
    Function to clear cache in ada on task change in different models
    """
    print("Clearing cache in ada")

    # use gc to clear all the dereferenced versions of the variable model
    gc.collect()
def captioning_wrapper(image, model_type):

    caption = image_captioning_models(image, model_type)
    print_content = f'''
    ### Caption Generated
    {caption}
    '''

    st.code(print_content, language='markdown')

def image_captioning_models(image, model_type):
    gc.enable()
    model = None
    processor = None

    clear_ada_cache()
    # image = Image.open(image)

    if model_type == "BLIP":
        print("Loading BLIP captioning model")
        with st.spinner("Loading BLIP captioning model"):
            return BLIP_captioning_model(image) 



