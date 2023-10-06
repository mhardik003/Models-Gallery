from PIL import Image
import streamlit as st
from Utils.utils import *

from transformers import CLIPProcessor, CLIPModel

def CLIP_wrapper(uploaded_file, model):

    task = st.selectbox("Select the task",
                        ("None","Image Classification", "Image Captioning"), on_change=clear_ada_cache(model), key="clip_task")
    if task == "Image Classification":
        print("Classification using CLIP")
        CLIP_classification_st(uploaded_file, model)
    elif task == "Image Captioning":
        print("Captioning using CLIP")
        st.write("Image Captioning")

def CLIP_classification_st(uploaded_file, model):
    from transformers import CLIPProcessor, CLIPModel
    prompt1 = st.text_input(
        'Enter prompt 1', value="A photo of a ", key="prompt1_clip")
    prompt2 = st.text_input(
        'Enter prompt 2', value="A photo of a ", key="prompt2_clip")
    prompt = [prompt1, prompt2]
    image = Image.open(uploaded_file)
    
    if (prompt1 != "" and prompt2 != "" and prompt1 != "A photo of a " and prompt2 != "A photo of a "):
        probs = CLIP_classfication_model(image, prompt, model)
        print('Sending the image to CLIP model')

        st.write("Probability of prompt 1: ", probs.detach().numpy()[0][0])
        st.write("Probability of prompt 2: ", probs.detach().numpy()[0][1])
    else:
        st.markdown(":red[Please enter both the prompts]")


def CLIP_classfication_model(image, prompt, model):
    print("hello")
    print(prompt)
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    inputs = processor(text=prompt, images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)

    
    return probs


if __name__ == "__main__":
    image_inp = input("Enter the path to the image: ")
    image = Image.open(image_inp)
    task = input("Choose 1 for classification and 2 for captioning: ")
    if task == "1":
        CLIP_classification_st(image)
        prompt1 = input("Enter prompt 1: ")
        prompt2 = input("Enter prompt 2: ")
        prompt = [prompt1, prompt2]
    elif task == "2":
        print("Captioning using CLIP")
        # st.write("Image Captioning")
    # model = CLIP_model(image, prompt)