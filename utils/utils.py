import streamlit as st
import gc
from PIL import Image
import numpy as np
import json
from memory_profiler import profile

# Importing the files for image related tasks
from utils.images.classification import *
from utils.images.captioning import *
from utils.images.prompted_captioning import *
from utils.images.question_answering import *
from utils.images.zero_shot_prediction import *


# Importing the files for video related tasks
from models.video.clip4clip import *
from models.video.xclip import *


models_config = json.load(open("config.json", "r"))

# @profile
def choose_model(modality, task, uploaded_file):
    # print("Task: ", task)
    # st.write("Task: ", task)

    # st.write("The models available for this task are: ", models_config[modality][task])
    st.write(
        "<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: left; }div.st-ag{padding-left:3px;} </style>",
        unsafe_allow_html=True,
    )
    st.write(
        "<style>div.st-bf{flex-direction:column;} div.st-ag{font-weight:bold;padding-left:2px;}</style>",
        unsafe_allow_html=True,
    )

    model_type = st.radio(
        "Select the model",
        ["None"] + models_config[modality][task],
        key=modality + "_" + task,
    )

    if model_type != "None" and modality == "image":
        if task == "Image Classification":
            print("Classification using " + model_type)
            get_classification_prompts(uploaded_file, model_type)

        if task == "Image Question Answering":
            print("Question Answering using " + model_type)
            st.write("Question Answering")
            get_answer(uploaded_file, model_type)

        if task == "Prompted Image Captioning":
            print("Prompted Image Captioning using " + model_type)
            get_prompted_caption(uploaded_file, model_type)

        if task == "Image Captioning":
            print("Captioning using " + model_type)
            # st.write("Image Captioning")
            captioning_wrapper(uploaded_file, model_type)

        if(task == "Zero Shot Class Prediction"):
            # st.write("Zero Shot Prediction")
            zero_shot_prediction_wrapper(uploaded_file, model_type)

    if model_type != "None" and modality == "video":
        choose_vid_task(uploaded_file, model_type)


# ------------------------------------------------------------------------------------------------------------------------
# VIDEO HELPERS


def choose_vid_task(uploaded_file, model_type):
    task = st.selectbox(
        "Select the task",
        ("None", "Video Classification", "Video Captioning"),
        key=model_type + "_task",
    )
    if task == "Video Classification":
        # print("Classification using "+model_type)
        get_vid_cls_prompts(uploaded_file, model_type)
    elif task == "Image Captioning":
        print("Captioning using " + model_type)
        st.write("Image Captioning")


def get_vid_cls_prompts(uploaded_file, model_type):
    prompt1 = st.text_input(
        "Enter prompt 1", value="A video of a ", key=model_type + "prompt1"
    )
    prompt2 = st.text_input(
        "Enter prompt 2", value="A video of a ", key=model_type + "prompt2"
    )
    prompt = [prompt1, prompt2]

    #####reading video inpt
    # image = Image.open(uploaded_file)

    if (
        prompt1 != ""
        and prompt2 != ""
        and prompt1 != "A video of a "
        and prompt2 != "A video of a "
    ):
        probs = vid_cls_models(uploaded_file, prompt, model_type)
        st.write("Probability of prompt 1: ", probs.detach().numpy()[0][0])
        st.write("Probability of prompt 2: ", probs.detach().numpy()[0][1])
    else:
        st.markdown(":red[Please enter both the prompts]")


def vid_cls_models(video, prompt, model_type):
    """
    Function to run the classification models
    """
    gc.enable()
    model = None
    processor = None
    clear_ada_cache()

    if model_type == "CLIP4CLIP":
        print("Loading CLIP4CLIP classification model")
        st.spinner("Loading CLIP4CLIP classification model")
        return CLIP4CLIP_classification_model(video, prompt)

    elif model_type == "XCLIP":
        print("Loading XCLIP classification model")
        st.spinner("Loading XCLIP classification model")
        return XCLIP_classification_model(video, prompt)

    elif model_type == "BLIP":
        print("Loading BLIP classification model")
        return BLIP_classification_model(video, prompt)
