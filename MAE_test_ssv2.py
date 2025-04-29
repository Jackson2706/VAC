import av
import numpy as np
import torch
import pandas as pd
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
from huggingface_hub import hf_hub_download
from tqdm import tqdm
import imageio

np.random.seed(0)
def load_and_process_video(video_path, num_frames=16, size=(224, 224)):
    reader = imageio.get_reader(video_path, 'ffmpeg')
    

    frames = []
    for frame in reader:
        img = torch.tensor(frame)  # Tensor (3, H, W), resized
        frames.append(img)

    # Bước: lấy đúng 16 frame (hoặc lặp lại nếu thiếu)
    if len(frames) < num_frames:
        repeat_factor = (num_frames + len(frames) - 1) // len(frames)
        frames = (frames * repeat_factor)[:num_frames]
    else:
        frames = frames[:num_frames]

    # Bước: chuyển về list numpy
    video = [frame.numpy() for frame in frames]
    return video

def calculate_top_k_accuracy(logits, true_label, k=1):
    """Check if the true label is in top-k predictions."""
    top_k_values, top_k_indices = logits.topk(k, dim=-1)
    return int(true_label) in top_k_indices.squeeze().tolist()

# Initialize counters
top_1_correct = 0
top_5_correct = 0
total_samples = 0

# Load labels
df = pd.read_csv("/home/jackson-devworks/Desktop/VAC/video_labels.csv", header=None, names=["path", "label"])

# Load model and processor once (outside the loop)
processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-ssv2")
model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-ssv2")

model.eval()

# Loop through video entries
for index, row in tqdm(df.iterrows(), total=len(df)):
    try:
        # print(row)
        path = row["path"]
        label = int(row["label"])
        # print(path, label)
        # Read video frames
        video = load_and_process_video(path)
        # Preprocess video
        inputs = processor(video, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits  # Shape: (1, num_classes)

        predicted_label = logits.argmax(-1).item()
        # print(f"Predicted: {predicted_label}, True: {label}")
        top_1_correct += (predicted_label == label)
        top_5_correct += calculate_top_k_accuracy(logits, label, k=5)
        total_samples += 1

    except Exception as e:
        print(f"Error processing video {path}: {e}")
    # if index % 100 == 0:
    #     break

# Final metrics
if total_samples > 0:
    top_1_accuracy = top_1_correct / total_samples
    top_5_accuracy = top_5_correct / total_samples

    print(f"Top-1 Accuracy: {top_1_accuracy:.4f}")
    print(f"Top-5 Accuracy: {top_5_accuracy:.4f}")
else:
    print("No valid videos processed.")
