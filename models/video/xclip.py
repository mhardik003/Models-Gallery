from utils.utils import *
import os
import numpy as np

def import_modules_for_xclip():
    from huggingface_hub import hf_hub_download
    import torch
    from decord import VideoReader, cpu
    from transformers import XCLIPProcessor, XCLIPModel


def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices


def XCLIP_classification_model(video, prompt):
    """
    args : video, list of prompts
    returns : prob for each prompt
    """

    import_modules_for_xclip()

    #encoding text and vid
    vr = VideoReader(video, num_threads=1, ctx=cpu(0))
    indices = sample_frame_indices(clip_len=8, frame_sample_rate=1, seg_len=len(vr))
    video = vr.get_batch(indices).asnumpy()
    model_name = "microsoft/xclip-base-patch32"
    processor = XCLIPProcessor.from_pretrained(model_name)
    model = XCLIPModel.from_pretrained(model_name)
    inputs = processor(text=prompt, videos=list(video), return_tensors="pt", padding=True)

    # forward pass
    with torch.no_grad():
        outputs = model(**inputs)

    probs = outputs.logits_per_video.softmax(dim=1)    
    return probs



if __name__ == "__main__":
    vid_input = input("Enter the path to the video: ")

    prompt1 = input("Enter prompt 1: ")
    prompt2 = input("Enter prompt 2: ")
    prompt = [prompt1, prompt2]
    print(XCLIP_classification_model(vid_input, prompt))
