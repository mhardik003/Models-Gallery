import streamlit as st
from io import StringIO
import pandas as pd
import sys
from Models.clip import CLIP_model
from Models.blip import BLIP_model
from PIL import Image

def clear_ada_cache():
    """
    Functoin to clear the model's
    """

def CLIP_wrapper(image):
    from transformers import CLIPProcessor, CLIPModel
    prompt1 = st.text_input(
        'Enter prompt 1', value="A photo of a ", key="prompt1_clip")
    prompt2 = st.text_input(
        'Enter prompt 2', value="A photo of a ", key="prompt2_clip")
    prompt = [prompt1, prompt2]
    image = Image.open(uploaded_file)
    # if st.button('Send to CLIP model'):
    if (prompt1 != "" and prompt2 != "" and prompt1 != "A photo of a " and prompt2 != "A photo of a "):
        clear_ada_cache()
        probs = CLIP_model(image, prompt)
        print('Sending the image to CLIP model')

        st.write("Probability of prompt 1: ", probs.detach().numpy()[0][0])
        st.write("Probability of prompt 2: ", probs.detach().numpy()[0][1])
    else:
        st.markdown(":red[Please enter both the prompts]")


def BLIP_wrapper(image):
    from transformers import AutoProcessor, BlipModel
    prompt1 = st.text_input(
        'Enter prompt 1', value="A photo of a ", key="prompt1_blip")
    prompt2 = st.text_input(
        'Enter prompt 2', value="A photo of a ", key="prompt2_blip")
    prompt = [prompt1, prompt2]
    image = Image.open(uploaded_file)
    # if st.button('Send to BLIP model'):
    if (prompt1 != "" and prompt2 != "" and prompt1 != "A photo of a " and prompt2 != "A photo of a "):
        print('Sending the image to BLIP model')
        probs = BLIP_model(image, prompt)

        st.write("Probability of prompt 1: ", probs.detach().numpy()[0][0])
        st.write("Probability of prompt 2: ", probs.detach().numpy()[0][1])
    else:
        st.markdown(":red[Please enter both the prompts]")


st.set_page_config(page_title="Katha AI Model Library", layout="wide", page_icon="💻", menu_items={
    'About': 'https://github.com/katha-ai',
    'Get Help': 'https://github.com/katha-ai',
    'Report a bug': "mailto:hardik.mittal@research.iiit.ac.in "
})
st.title("KATHA AI's Model Library")


uploaded_file = None


pic_col, vid_col = st.tabs(["Image", "Video"])

# print(pic_col, vid_col)

with pic_col:

    uploaded_file = st.file_uploader("Choose an image", type=[
        'png', 'jpg'], accept_multiple_files=False,  help='Upload an image')

    if uploaded_file is not None:
        left_col, right_col = st.columns(2, gap="large")
        with left_col:
            with st.spinner('Loading the image'):
                st.image(uploaded_file, caption='Uploaded Image',
                     use_column_width=True)
        with right_col:

            st.subheader("Select the model to run the image on")

            ALIGN_col, CLIP_col, BLIP_col, ALBEF_col, BLIPv2_col, GLIP_col = st.tabs(
                ["ALIGN", "CLIP", "BLIP", "ALBEF", "BLIPv2", "GLIP"])


            with ALIGN_col:
                print("hello align")
                # clear cache from ada for all other models by checking if they are in the memory first
                # 
                st.write('Sending the image to ALIGN model')
            with CLIP_col:


                task = st.selectbox("Select the task",
                                    ("None","Image Classification", "Image Captioning"), on_change=print("Changing task"))
                if task == "Image Classification":
        
                    print("Classification using CLIP")
                    CLIP_wrapper(uploaded_file)
                elif task == "Image Captioning":
                    print("Captioning using CLIP")
                    st.write("Image Captioning")

            with BLIP_col:
                BLIP_wrapper(uploaded_file)

            with ALBEF_col:
                st.write('Sending the image to ALBEF model')
            with BLIPv2_col:
                st.write('Sending the image to BLIPv2 model')
            with GLIP_col:
                st.write('Sending the image to GLIP model')

with vid_col:
    uploaded_file = st.file_uploader("Choose a video", type=[
        'mp4'], accept_multiple_files=False,  help='Upload a video')

    if uploaded_file is not None:
        left_col, right_col = st.columns(2, gap="large")
        with left_col:
            with st.spinner('Loading the video'):
                st.video(uploaded_file)
        with right_col:
            st.subheader("Select the model to run the video on")

            # VideoCLIP, CLIP4Clip, XCLIP, CLIP-ViP, ViFi-CLIP

            VideoCLIP_col, CLIP4Clip_col, XCLIP_col, CLIP_ViP_col, ViFi_CLIP_col = st.tabs(
                ["VideoCLIP", "CLIP4Clip", "XCLIP", "CLIP-ViP", "ViFi-CLIP"])

            with VideoCLIP_col:
                st.write('Sending the video to VideoCLIP model')
            with CLIP4Clip_col:
                st.write('Sending the video to CLIP4Clip model')
            with XCLIP_col:
                st.write('Sending the video to XCLIP model')
            with CLIP_ViP_col:
                st.write('Sending the video to CLIP_ViP model')
            with ViFi_CLIP_col:
                st.write('Sending the video to ViFi_CLIP model')

# st.write("Probability of prompt 1: ", probs.detach().numpy()[0][0])
