from utils.utils import *
from PIL import Image

def softmax_calc(x):
    import numpy as np
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def ViLT_question_answering_model(image, question):
    from transformers import ViltProcessor, ViltForQuestionAnswering
    import torch

    image = Image.open(image)
    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    model.eval()
    # prepare inputs
    encoding = processor(image, question, return_tensors="pt")

    # forward pass
    outputs = model(**encoding)
    logits = outputs.logits
    # print(logits)
    idx = logits.argmax(-1).item()
    answer = model.config.id2label[idx]


    # top 5 labels
    idxs = logits.topk(5).indices[0].tolist()
    answers = [model.config.id2label[idx] for idx in idxs]

    # top 5 probabilities
    probs = logits.softmax(-1).topk(5).values[0].tolist()

    print("answers : ", answers)
    print("probs : ", probs)
    
    # covert to a dictionary
    answer = ''

    for i in range(len(answers)):
        answer += answers[i] + " : " + str(probs[i]) + "\n"

    return answer


def ViLT_classification_model(image, prompts):
    from transformers import ViltProcessor, ViltForImageAndTextRetrieval
    from PIL import Image


    image = Image.open(image)
    

    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-coco")
    model = ViltForImageAndTextRetrieval.from_pretrained("dandelin/vilt-b32-finetuned-coco")

    # forward pass
    scores = dict()
    for text in prompts:
        # prepare inputs
        encoding = processor(image, text, return_tensors="pt")
        outputs = model(**encoding)
        scores[text] = outputs.logits[0, :].item()

    # print(scores)
    scores_probs =[]

    # store only the probabilities in a list
    for key in scores.keys():
        scores_probs.append(scores[key])    
    
    # do softmax
    scores_probs = softmax_calc(scores_probs)

    # print(scores_probs)

    return [scores_probs]
