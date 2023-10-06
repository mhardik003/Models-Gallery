from PIL import Image
from transformers import AutoProcessor, BlipModel
from utils.utils import *
import streamlit as st

def BLIP_wrapper(uploaded_file, model, processor):
    task = st.selectbox("Select the task",
                        ("None","Image Classification", "Image Captioning"), on_change=clear_ada_cache(model, processor), key="blip_task")
    if task == "Image Classification":
        print("Classification using BLIP")
        BLIP_classification_model_st(uploaded_file, model, processor)

    elif task == "Image Captioning":
        print("Captioning using BLIP")
        st.write("Image Captioning")

def BLIP_classification_model_st(uploaded_file, prompt, model, processor):
    prompt1 = st.text_input(
        'Enter prompt 1', value="A photo of a ", key="prompt1_blip")
    prompt2 = st.text_input(
        'Enter prompt 2', value="A photo of a ", key="prompt2_blip")
    prompt = [prompt1, prompt2]
    image = Image.open(uploaded_file)
    # if st.button('Send to BLIP model'):
    if (prompt1 != "" and prompt2 != "" and prompt1 != "A photo of a " and prompt2 != "A photo of a "):
        print('Sending the image to BLIP model')
        probs = BLIP_classification_model(image, prompt, model, processor)

        st.write("Probability of prompt 1: ", probs.detach().numpy()[0][0])
        st.write("Probability of prompt 2: ", probs.detach().numpy()[0][1])
    else:
        st.markdown(":red[Please enter both the prompts]")


def BLIP_classification_model(image, prompt, model, processor):
    print("hello")
    print(prompt)
    model = BlipModel.from_pretrained("Salesforce/blip-image-captioning-base")
    processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    inputs = processor(text=prompt, images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    return probs




if __name__ == "__main__":
    image_inp = input("Enter the path to the image: ")
    image = Image.open(image_inp)
    prompt1 = input("Enter prompt 1: ")
    prompt2 = input("Enter prompt 2: ")
    prompt = [prompt1, prompt2]
    model = BLIP_model(image, prompt)