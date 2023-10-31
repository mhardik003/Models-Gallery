import gc
from PIL import Image
import streamlit as st


from models.image.blip2 import *
from models.image.blip import *

def clear_ada_cache():
    """
    Function to clear cache in ada on task change in different models
    """
    print("Clearing cache in ada")

    # use gc to clear all the dereferenced versions of the variable model
    gc.collect()


def get_prompted_caption(uploaded_file, model_type):
    """
    Function to get the caption
    """
    prompt = st.text_input(
        'Enter prompt', value="A photo of a ", key=model_type+"prompt")

    print("> Prompt Entered : ", prompt)
    if (prompt != ""):
        caption = prompted_captioning_models(uploaded_file, prompt, model_type)
        print_content = f'''
        ### Caption Generated
        {caption}
        '''
        st.code(print_content, language='markdown')
    else:
        st.code(":red[Please enter the prompt]")


def prompted_captioning_models( image, prompt, model_type):
    """
    Function to run the classification models
    """
    gc.enable()
    model = None
    processor = None
    clear_ada_cache()
    # st.write("prompt", prompt)
    # st.write(model_type)

    if model_type=="BLIP":
        print("Loading BLIP prompted captioning model")
        with st.spinner("Loading BLIP prompted captioning model"):
            return BLIP_captioning_model(image, prompt)
        
    if model_type=="BLIPv2":
        print("Loading BLIPv2 prompted captioning model")
        with st.spinner("Loading BLIPv2 prompted captioning model"):
            return BLIP2_prompted_captioning_Model(image, prompt)