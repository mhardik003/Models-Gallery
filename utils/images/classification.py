import gc
import streamlit as st
from PIL import Image

from models.image.align import *
from models.image.clip import *
from models.image.blip import *
from models.image.blip2 import *
from models.image.vilt import *

# Initialize a session state variable to store the refresh trigger
if 'refresh' not in st.session_state:
    st.session_state['refresh'] = False

# Define a function to update the session state variable
def refresh():
    st.session_state['refresh'] = not st.session_state['refresh']

def clear_ada_cache():
    """
    Function to clear cache in ada on task change in different models
    """
    print("Clearing cache in ada")

    # use gc to clear all the dereferenced versions of the variable model
    gc.collect()


def get_classification_prompts(uploaded_file, model_type):
    """
    Function to get the classification prompts
    """

    # prompts=['A photo of a ', 'A photo of a '] #default prompts
    
    prompt1 = st.text_input(
        "Enter prompt 1", value="A photo of a ", key=model_type + "prompt1"
    )
    prompt2 = st.text_input(
        "Enter prompt 2", value="A photo of a ", key=model_type + "prompt2"
    )
    prompt = [prompt1, prompt2]
    # for i in range(len(prompts)):
    #     prompt = st.text_input(
    #         f'Enter prompt {i+1}', value=prompts[i], key=model_type+f"prompt{i+1}"
    #     )
    #     prompts[i]=prompt

    

    # if(st.button("Add another prompt")):
    #     # remove the button from the UI
        
    #     prompts.append('A photo of a ')
    #     print(prompts)
    #     refresh()
        
    # st.write(prompt)


    # IF NO PROMPT IS empty then
    if prompt[0] != "" and prompt[1] != "" and prompt[0] != "A photo of a " and prompt[1] != "A photo of a ":
        probs = classification_models(uploaded_file, prompt, model_type)
        # st.write(probs)
        for i in range(len(probs[0])):
            st.write(f"Probability of prompt {i+1}: ", probs[0][i])
    
    else:
        st.markdown(":red[Please enter both the prompts]")



def classification_models( image, prompt, model_type):
    """
    Function to run the classification models
    """
    gc.enable()
    model = None
    processor = None
    clear_ada_cache()

    
    
    # st.write("prompt", prompt)
    # st.write(model_type)

    if model_type=="ALIGN":
        print("Loading ALIGN classification model")
        with st.spinner("Loading ALIGN classification model"):
            return ALIGN_classification_model(image, prompt)
    
    elif model_type=="CLIP":
        print ("Loading CLIP classification model")
        with st.spinner("Loading CLIP classification model"):
            return CLIP_classification_model(image, prompt)
    
    elif model_type=="BLIP":
        print ("Loading BLIP classification model")
        with st.spinner("Loading BLIP classification model"):
            return BLIP_classification_model(image, prompt)

    # elif model_type=="BLIPv2":
    #     print ("Loading BLIPv2 classification model")
    #     return BLIP2_classification_Model(image, prompt)

    elif model_type=="ViLT":
        print ("Loading ViLT classification model")
        with st.spinner("Loading ViLT classification model"):
            return ViLT_classification_model(image, prompt)


    else :
        print("Model not found")
        st.write("Model not found")
        return [[0,0]]