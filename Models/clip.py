from PIL import Image

from transformers import CLIPProcessor, CLIPModel

def CLIP_model(image, prompt):
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
    prompt1 = input("Enter prompt 1: ")
    prompt2 = input("Enter prompt 2: ")
    prompt = [prompt1, prompt2]
    model = CLIP_model(image, prompt)