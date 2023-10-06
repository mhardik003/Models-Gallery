import streamlit as st
import gc
from PIL import Image
from models.clip import *
from models.blip import *
from models.align import *

def clear_ada_cache(model, processor):
    """
    Function to clear cache in ada on task change in different models
    """

    print("Clearing cache in ada")
    # use gc to clear all the dereferenced versions of the variable model
    gc.collect()
    # del model
    # gc.disable()
    # return model, processor

def choose_task(uploaded_file, model_type):
    """
    Function to choose the task
    """
    task = st.selectbox("Select the task",
                        ("None", "Image Classification", "Image Captioning"), key=model_type+"_task")
    if task == "Image Classification":
        # print("Classification using "+model_type)
        get_classification_prompts(uploaded_file, model_type)
    elif task == "Image Captioning":
        print("Captioning using "+model_type)
        st.write("Image Captioning")

def get_classification_prompts(uploaded_file, model_type):
    """
    Function to get the classification prompts
    """
    prompt1 = st.text_input(
        'Enter prompt 1', value="A photo of a ", key=model_type+"prompt1")
    prompt2 = st.text_input(
        'Enter prompt 2', value="A photo of a ", key=model_type+"prompt2")
    prompt = [prompt1, prompt2]
    
    image = Image.open(uploaded_file)
        
    if (prompt1 != "" and prompt2 != "" and prompt1 != "A photo of a " and prompt2 != "A photo of a "):
        probs = classification_models(image, prompt, model_type)
        st.write("Probability of prompt 1: ", probs.detach().numpy()[0][0])
        st.write("Probability of prompt 2: ", probs.detach().numpy()[0][1])
    else:
        st.markdown(":red[Please enter both the prompts]")



def classification_models(image, prompt, model_type):
    """
    Function to run the classification models
    """
    gc.enable()
    model = None
    processor = None
    clear_ada_cache(model, processor)

    if model_type=="ALIGN":
        print("Loading ALIGN classification model")
        return ALIGN_classification_model(image, prompt)
    
    elif model_type=="CLIP":
        print ("Loading CLIP classification model")
        st.spinner("Loading CLIP classification model")
        return CLIP_classification_model(image, prompt)
    
    elif model_type=="BLIP":
        print ("Loading BLIP classification model")
        return BLIP_classification_model(image, prompt)
    