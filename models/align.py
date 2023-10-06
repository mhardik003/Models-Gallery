import requests
import torch
from PIL import Image
from transformers import AlignProcessor, AlignModel
from utils.utils import *
import streamlit as st

def ALIGN_wrapper(uploaded_file, model, processor):
    task = st.selectbox("Select the task",
                        ("None", "Image Classification", "Image Captioning"), on_change=clear_ada_cache(model, processor), key="align_task")
    if task == "Image Classification":
        print("Classification using ALIGN")
        ALIGN_classification_st(uploaded_file, model, processor)
    elif task == "Image Captioning":
        print("Captioning using ALIGN")
        st.write("Image Captioning")

def ALIGN_classification_st(uploaded_file, model, processor):
    prompt1 = st.text_input(
        'Enter prompt 1', value="A photo of a ", key="prompt1_align")
    prompt2 = st.text_input(
        'Enter prompt 2', value="A photo of a ", key="prompt2_align")
    prompt = [prompt1, prompt2]
    image = Image.open(uploaded_file)
    
    if (prompt1 != "" and prompt2 != "" and prompt1 != "A photo of a " and prompt2 != "A photo of a "):
        print('Sending the image to ALIGN model')
        probs = ALIGN_classification_model(image, prompt, model, processor)

        st.write("Probability of prompt 1: ", probs.detach().numpy()[0][0])
        st.write("Probability of prompt 2: ", probs.detach().numpy()[0][1])
    else:
        st.markdown(":red[Please enter both the prompts]")


def ALIGN_classification_model(image, prompt, model, processor):
    processor = AlignProcessor.from_pretrained("kakaobrain/align-base")
    model = AlignModel.from_pretrained("kakaobrain/align-base")

    inputs = processor(text=prompt, images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    # this is the image-text similarity score
    logits_per_image = outputs.logits_per_image

    # we can take the softmax to get the label probabilities
    probs = logits_per_image.softmax(dim=1)
    return probs