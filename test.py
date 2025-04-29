<<<<<<< HEAD
from datasets import Kinetics
from  utils_ import  load_config

cfg = load_config(cfg_file='/home/dung2/VAC/configs/default.yaml')
dataset = Kinetics(
    cfg=cfg,
    mode="val"
)

print(dataset[0][0].shape)
=======
import imageio
import torch
import numpy as np
# import torchvision.transforms as T

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

if __name__ == "__main__":
    video_path = "/home/jackson-devworks/Desktop/SSv2/ssv2test/Approaching_something_with_your_camera/9415.webm"
    video = load_and_process_video(video_path)
    print(f"Length: {len(video)}, shape of first: {video[0].shape}")
>>>>>>> e4975e1 (update)
