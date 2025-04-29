import av
import numpy as np
import torch
import pandas as pd
from transformers import VivitImageProcessor, VivitForVideoClassification
from huggingface_hub import hf_hub_download
from tqdm import tqdm
np.random.seed(0)


def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    '''
    Sample a given number of frame indices from the video.
    Args:
        clip_len (`int`): Total number of frames to sample.
        frame_sample_rate (`int`): Sample every n-th frame.
        seg_len (`int`): Maximum allowed index of sample's last frame.
    Returns:
        indices (`List[int]`): List of sampled frame indices
    '''
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices

def calculate_top_k_accuracy(logits, true_label, k=1):
    # Get the top-k predictions
    top_k_values, top_k_indices = logits.topk(k, dim=-1)
    
    # Check if the true label is in the top-k predictions
    return true_label in top_k_indices.squeeze().tolist()

# Initialize counters
top_1_correct = 0
top_5_correct = 0
total_samples = 0
# video clip consists of 300 frames (10 seconds at 30 FPS)
df = pd.read_csv("/home/jackson-devworks/Desktop/VAC/test.csv", sep=" ", header=None, names=["path", "label"])

for index, row in tqdm(df.iterrows(), total=len(df)):
    try:
        path = row["path"]
        label = row["label"]
        container = av.open(path)

        # sample 32 frames
        indices = sample_frame_indices(clip_len=32, frame_sample_rate=4, seg_len=container.streams.video[0].frames)
        video = read_video_pyav(container=container, indices=indices)

        image_processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")
        model = VivitForVideoClassification.from_pretrained("google/vivit-b-16x2-kinetics400")

        inputs = image_processor(list(video), return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        # model predicts one of the 400 Kinetics-400 classes
        predicted_label = logits.argmax(-1).item()
        top_1_correct += (predicted_label == label)
        top_5_correct += calculate_top_k_accuracy(logits, label, k=5)

        total_samples += 1
    except Exception as e:
        print(f"Error processing video {path}: {e}")
    # print(f"Predicted: {predicted_label}, Ground Truth: {label}")

# Calculate final accuracies
top_1_accuracy = top_1_correct / total_samples
top_5_accuracy = top_5_correct / total_samples

print(f"Top-1 Accuracy: {top_1_accuracy:.4f}")
print(f"Top-5 Accuracy: {top_5_accuracy:.4f}")