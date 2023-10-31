import streamlit as st
import gc
from PIL import Image

from models.image.clip import *
from models.image.blip2 import *
from models.image.blip import *
from models.image.vilt import *


def clear_ada_cache():
    """
    Function to clear cache in ada on task change in different models
    """
    print("Clearing cache in ada")

    # use gc to clear all the dereferenced versions of the variable model
    gc.collect()


def get_answer(uploaded_file, model_type):
    """
    Function to get the answer
    """
    question = st.text_input(
        "Enter question",
        value="What is present in the image ",
        key=model_type + "question",
    )

    # st.write("Question: ", question)
    # image = Image.open(uploaded_file)

    if question != "" and question != "What is the name of the ":
        answer = question_answering_models(uploaded_file, question, model_type)
        answer_output = f'''### Answer Generated \n{answer}'''
        st.code(answer_output, language="markdown")
    else:
        st.markdown(":red[Please enter the question]")


def question_answering_models(image, question, model_type):
    """
    Function to run the classification models
    """
    gc.enable()
    model = None
    processor = None
    clear_ada_cache()

    if model_type == "BLIPv2":
        print("Loading BLIPv2 question answering model")
        with st.spinner("Loading BLIPv2 question answering model"):
            return BLIP2_question_answering_Model(image, question)

    if model_type == "BLIP":
        print("Loading BLIP question answering model")
        with st.spinner("Loading BLIP question answering model"):
            return BLIP_question_answering_model(image, question)

    if model_type=="ViLT":
        print ("Loading ViLT question answering model")
        with st.spinner("Loading ViLT question answering model"):
            return ViLT_question_answering_model(image, question)