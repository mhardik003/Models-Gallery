from utils.utils import *
from PIL import Image


def BLIP2_captioning_Model(image):
    import torch
    from transformers import AutoProcessor, Blip2ForConditionalGeneration

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
    )
    model.to(device)

    inputs = processor(image, return_tensors="pt").to(device, torch.float16)

    generated_ids = model.generate(**inputs, max_new_tokens=20)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[
        0
    ].strip()

    print(generated_text)
    return generated_text


def BLIP2_prompted_captioning_Model(image, prompt):
    import torch
    from transformers import AutoProcessor, Blip2ForConditionalGeneration

    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
    # by default `from_pretrained` loads the weights in float32
    # we load in float16 instead to save memory
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
    )
    print("meow1")
    model.to(device)
    image = Image.open(image)


    inputs = processor(image, text=prompt, return_tensors="pt").to(
        device, torch.float16
    )

    generated_ids = model.generate(**inputs, max_new_tokens=20)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[
        0
    ].strip()

    print(generated_text)
    return generated_text


def BLIP2_question_answering_Model(image, prompt):
    import torch
    from transformers import AutoProcessor, Blip2ForConditionalGeneration

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
    )
    model.to(device)

    prompt = "Question: " + prompt + " Answer: "

    inputs = processor(image, text=prompt, return_tensors="pt").to(
        device, torch.float16
    )

    generated_ids = model.generate(**inputs, max_new_tokens=10)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[
        0
    ].strip()
    print(generated_text)


def BLIP2_chatbased_prompting_Model(image, context, question):
    import torch
    from transformers import AutoProcessor, Blip2ForConditionalGeneration

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
    )
    model.to(device)

    """
    context = [
    ("which city is this?", "singapore"),
    ("why?", "it has a statue of a merlion"),
    ]
    question = "where is the name merlion coming from?"
    """

    template = "Question: {} Answer: {}."

    prompt = (
        " ".join(
            [template.format(context[i][0], context[i][1]) for i in range(len(context))]
        )
        + " Question: "
        + question
        + " Answer:"
    )

    print(prompt)

    inputs = processor(image, text=prompt, return_tensors="pt").to(
        device, torch.float16
    )

    generated_ids = model.generate(**inputs, max_new_tokens=10)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[
        0
    ].strip()
    print(generated_text)

    return generated_text, prompt

    # def BLIP2_classification_Model(image, prompt):
    #     import torch
    #     from transformers import AutoProcessor, Blip2Model, AutoTokenizer
    #     device = "cuda" if torch.cuda.is_available() else "cpu"

    #     model  = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
    #     model.to(device)

    #     tokenizer = AutoTokenizer.from_pretrained("Salesforce/blip2-opt-2.7b")

    #     inputs = tokenizer(prompt, padding=True, return_tensors="pt").to(device)

    #     # text_features = model.get_text_features(**inputs)
    #     with torch.no_grad():
    #         outputs = model(**inputs)

    #     # this is the image-text similarity score
    #     logits_per_image = outputs.logits_per_image

    #     # we can take the softmax to get the label probabilities
    #     probs = logits_per_image.softmax(dim=1)
    #     probs = probs.detach().numpy()
    #     return probs


if __name__ == "__main__":
    print("BLIP2")
    # image_inp = input("Enter the path to the image: ")
    # image = Image.open(image_inp)
    # task = input("Choose 1 for classification and 2 for captioning: ")
    # if task == "1":
    #     BLIP2_classification_Model(image)
    #     prompt1 = input("Enter prompt 1: ")
    #     prompt2 = input("Enter prompt 2: ")
    #     prompt = [prompt1, prompt2]
    # elif task == "2":
    #     print("Captioning using CLIP")
    # st.write("Image Captioning")
    # model = CLIP_model(image, prompt)
