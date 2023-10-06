import streamlit as st
from io import StringIO
import pandas as pd
import sys
from Models.clip import *
from Models.blip import *
from PIL import Image

uploaded_file = st.file_uploader("Choose an image", type=[
    'png', 'jpg', 'jpeg'], accept_multiple_files=False,  help='Upload an image')

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
            CLIP_wrapper(uploaded_file)


        with BLIP_col:
            BLIP_wrapper(uploaded_file)

        with ALBEF_col:
            st.write('Sending the image to ALBEF model')
        with BLIPv2_col:
            st.write('Sending the image to BLIPv2 model')
        with GLIP_col:
            st.write('Sending the image to GLIP model')
