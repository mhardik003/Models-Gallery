import streamlit as st
import gc
from PIL import Image
from models.image.clip import *
from models.image.eva import *


def clear_ada_cache():
    """
    Function to clear cache in ada on task change in different models
    """
    print("Clearing cache in ada")

    # use gc to clear all the dereferenced versions of the variable model
    gc.collect()


def zero_shot_prediction_wrapper( uploaded_file, model_type):
    num_classes = st.number_input("Enter the number of classes", min_value=1, max_value=100, value=5, step=1)

    results = zero_shot_prediction_models(uploaded_file, model_type, num_classes)  
    st.write(f'The top {len(results)} classes are : ')


    answer_output=f'## The top {num_classes} classes are : \n'

    for i in range(len(results)):
        answer_output += f'{results[i][0]} : {results[i][1]} \n'    

    st.code(answer_output, language="markdown")

def zero_shot_prediction_models( image, model_type, num_classes):
    """
    Function to run the classification models
    """
    gc.enable()

    model = None
    processor = None
    clear_ada_cache()

    
    # st.write("prompt", prompt)
    # st.write(model_type)

    if model_type=="CLIP":
        print ("Loading CLIP captioning model")
        with st.spinner(f'Loading CLIP captioning model with {num_classes} classes'):
            return CLIP_prediction_model(image, num_classes)


    # elif model_type=="EVA":
    #     print ("Loading EVA captioning model")
    #     with st.spinner("Loading EVA captioning model"):
    #         return EVA_zero_shot_prediction(image, num_classes)