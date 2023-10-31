from PIL import Image

def EVA_zero_shot_prediction(image, num_classes):
    import timm
    import torch

    image = Image.open(image)
    model = timm.create_model('eva02_large_patch14_448.mim_m38m_ft_in1k', pretrained=True)
    model = model.eval()

    # get model specific transforms (normalization, resize)
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)

    output = model(transforms(image).unsqueeze(0))  # unsqueeze single image into batch of 1

    topk_probabilities, topk_class_indices = torch.topk(output.softmax(dim=1) * 100, k=num_classes)

    print("Top 5 predicted classes and probabilities")
    for i in range(topk_probabilities.size(1)):
        print(f"Class: {topk_class_indices[0, i]} - Probability: {topk_probabilities[0, i].item():.2f}%")

    results = [] # tuple of (class, probability)
    for i in range(topk_probabilities.size(1)):
        results.append((topk_class_indices[0, i], topk_probabilities[0, i].item()))

    return results