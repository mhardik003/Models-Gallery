from utils.utils import *
from PIL import Image

from PIL import Image
import requests
import torch


def BLIP_question_answering_model(image, question):
    from transformers import AutoProcessor, BlipForQuestionAnswering
    model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
    processor = AutoProcessor.from_pretrained("Salesforce/blip-vqa-base")

    
    image = Image.open(image)

    # training
    # text = "How many cats are in the picture?"
    # label = "2"
    # inputs = processor(images=image, text=text, return_tensors="pt")
    # labels = processor(text=label, return_tensors="pt").input_ids

    # inputs["labels"] = labels
    # outputs = model(**inputs)
    # loss = outputs.loss
    # loss.backward()

    # inference
    text = question
    inputs = processor(images=image, text=text, return_tensors="pt")
    outputs = model.generate(**inputs)
    answer = processor.decode(outputs[0], skip_special_tokens=True)
    
    print("answer: ", answer)

    return answer
    
    

def BLIP_captioning_model(image, prompt=None):
    # Load the processor and model
    from transformers import BlipProcessor, BlipForConditionalGeneration

    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    # Load the image from the web
    raw_image = Image.open(image)

    # Preprocess the image and use detection model
    # BLIP uses a processor to prepare inputs for the model, so we do not need to specify image size
    if prompt is None:
        inputs = processor(raw_image, return_tensors="pt")
        
    else:
        inputs = processor(raw_image,prompt, return_tensors="pt")
    # text = "a photography of"

    # Generate image caption using beam search
    max_length =50
    if prompt is not None:
        max_length = max_length + len(prompt)
        
    out = model.generate(**inputs, max_new_tokens=max_length, num_beams=3, min_length=5)
    caption = processor.decode(out[0], skip_special_tokens=True)

    print("caption: ", caption)
    return caption


def BLIP_classification_model(image, prompt):
    # print("hello")
    # print(prompt)
    import torch
    from transformers import AutoProcessor, BlipModel

    image = Image.open(image)
    model = BlipModel.from_pretrained("Salesforce/blip-image-captioning-base")
    processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    inputs = processor(text=prompt, images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    probs = probs.detach().numpy()
    return probs


if __name__ == "__main__":
    print("BLIP")
    # image_inp = input("Enter the path to the image: ")
    # image = Image.open(image_inp)
    # prompt1 = input("Enter prompt 1: ")
    # prompt2 = input("Enter prompt 2: ")
    # prompt = [prompt1, prompt2]
    # model = BLIP_model(image, prompt)
