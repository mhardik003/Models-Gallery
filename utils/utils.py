from global_vars import *
import streamlit as st

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
    
#------------------------------------------------------------------------------------------------------------------------
             
                                           #IMAGE HELPER    


def choose_model( modality, task, uploaded_file):
    print("Task: ", task)
    # st.write("Task: ", task)

    # st.write("The models available for this task are: ", models_config[modality][task])
    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: left; }div.st-ag{padding-left:3px;} </style>', unsafe_allow_html=True)
    # st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
    # st.write('<style>div.st-bf{flex-direction:column;} div.st-ag{font-weight:bold;padding-left:2px;}</style>', unsafe_allow_html=True)

    model_type = st.radio("Select the model", ["None"]+models_config[modality][task], key=modality+"_"+task)
    
    if(model_type!="None"):
        
        if(task=="Image Classification"):
            get_classification_prompts(modality,uploaded_file, model_type)
        
        if (task=="Question Answering"):
            # get_answer(uploaded_file, model_type)
            st.write("Question Answering")

        if (task=="Image Captioning"):
            st.write("Image Captioning")


# TODO : Add chat like interface  for question answering
# def get_answer(uploaded_file, model_type):
#     """
#     Function to get the answer
#     """
#     question = st.text_input(
#         'Enter question', value="What is the name of the ", key=model_type+"question")
#     image = Image.open(uploaded_file)
        
#     if (question != "" and question != "What is the name of the "):
#         answer = question_answering_models(image, question, model_type)
#         st.write("Answer: ", answer)
#     else:
#         st.markdown(":red[Please enter the question]")

# def question_answering_models(image, question, model_type):
#     """
#     Function to run the classification models
#     """
#     gc.enable()
#     model = None
#     processor = None
#     clear_ada_cache(model, processor)

#     if model_type=="BLIPv2":
#         print("Loading BLIPv2 classification model")
#         return BLIP2_Question_Answering_Model(image, question)



def get_classification_prompts(modality,uploaded_file, model_type):
    """
    Function to get the classification prompts
    """

    prompt1 = st.text_input(
        'Enter prompt 1', value="A photo of a ", key=model_type+"prompt1")
    prompt2 = st.text_input(
        'Enter prompt 2', value="A photo of a ", key=model_type+"prompt2")
    
    prompt = [prompt1, prompt2]
    # st.write(prompt)


    if (prompt1 != "" and prompt2 != "" and prompt1 != "A photo of a " and prompt2 != "A photo of a "):
        probs = classification_models(modality,uploaded_file, prompt, model_type)
        # st.write(probs)
        st.write("Probability of prompt 1: ", probs[0][0])
        st.write("Probability of prompt 2: ", probs[0][1])
    else:
        st.markdown(":red[Please enter both the prompts]")



def classification_models(modality, image, prompt, model_type):
    """
    Function to run the classification models
    """
    gc.enable()
    model = None
    processor = None
    clear_ada_cache(model, processor)

    if modality=="image":

        image = Image.open(image)
        # st.write("prompt", prompt)
        # st.write(model_type)

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
        
        elif model_type=="BLIPv2":
            print ("Loading BLIPv2 classification model")
            return BLIP2_classification_Model(image, prompt)
    

#------------------------------------------------------------------------------------------------------------------------
                                            #VIDEO HELPERS

def choose_vid_task(uploaded_file, model_type):

    task = st.selectbox("Select the task",
                        ("None", "Video Classification", "Video Captioning"), key=model_type+"_task")
    if task == "Video Classification":
        # print("Classification using "+model_type)
        get_vid_cls_prompts(uploaded_file, model_type)
    elif task == "Image Captioning":
        print("Captioning using "+model_type)
        st.write("Image Captioning")        

def get_vid_cls_prompts(uploaded_file, model_type):

    prompt1 = st.text_input(
        'Enter prompt 1', value="A video of a ", key=model_type+"prompt1")
    prompt2 = st.text_input(
        'Enter prompt 2', value="A video of a ", key=model_type+"prompt2")
    prompt = [prompt1, prompt2]

    #####reading video inpt
    # image = Image.open(uploaded_file)
        
    if (prompt1 != "" and prompt2 != "" and prompt1 != "A video of a " and prompt2 != "A video of a "):
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
    clear_ada_cache(model, processor)

    if model_type=="CLIP4CLIP":
        print("Loading CLIP4CLIP classification model")
        st.spinner("Loading CLIP4CLIP classification model")
        return CLIP4CLIP_classification_model(video, prompt)
    
    elif model_type=="XCLIP":
        print ("Loading XCLIP classification model")
        st.spinner("Loading XCLIP classification model")
        return XCLIP_classification_model(video, prompt)
    
    elif model_type=="BLIP":
        print ("Loading BLIP classification model")
        return BLIP_classification_model(video, prompt)
    
