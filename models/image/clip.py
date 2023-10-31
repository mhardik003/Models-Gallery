from utils.utils import *
from PIL import Image



def CLIP_classification_model(image, prompt):
    import torch
    import clip
    from PIL import Image
    image = Image.open(image)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)


    image = preprocess(image).unsqueeze(0).to(device)
    text = clip.tokenize(prompt).to(device)
    
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=1).cpu().numpy()

    print("probs: ", probs)
    
    return probs

def CLIP_prediction_model(image, num_classes):
    import os
    import clip
    import torch
    from torchvision.datasets import CIFAR100

    image = Image.open(image)
    # print("hello")
    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-B/32', device)

    # Download the dataset
    cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)

    # Prepare the inputs
    # image, class_id = cifar100[3637]
    image_input = preprocess(image).unsqueeze(0).to(device)
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in cifar100.classes]).to(device)

    # Calculate features
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)

    # Pick the top 5 most similar labels for the image
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    values, indices = similarity[0].topk(num_classes)

    # Print the result
    
    # print("\nTop predictions:\n")
    # for value, index in zip(values, indices):
    #     print(f"{cifar100.classes[index]:>16s}: {100 * value.item():.2f}%")

    result = [(cifar100.classes[idx], value.item()) for value, idx in zip(values, indices)]
    return result


if __name__ == "__main__":
    image_inp = input("Enter the path to the image: ")
    image = Image.open(image_inp)
    task = input("Choose 1 for classification and 2 for captioning: ")
    if task == "1":
        prompt1 = input("Enter prompt 1: ")
        prompt2 = input("Enter prompt 2: ")
        prompt = [prompt1, prompt2]
        CLIP_classification_model(image, prompt)
    elif task == "2":
        print("Captioning using CLIP")
        # st.write("Image Captioning")
    # model = CLIP_model(image, prompt)