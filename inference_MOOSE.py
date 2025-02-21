import torch
from torch.utils.data import DataLoader
from torchvision.datasets.kinetics import Kinetics
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

from config import *
from model import *
from utils_ import *


test_dataset = Kinetics(
    root=DATASET_DIR, 
    frames_per_clip=FRAME_PER_CLIP, 
    split ="test", download=False, 
    num_workers=NUM_WORKER_LOADING, 
    transform = transforms.Lambda(
        lambda x: torch.stack([transforms.Resize((224, 224), interpolation=InterpolationMode.BILINEAR)(frame) for frame in x])
    )
)

test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False)

model = MOOSE(None)
model_infos = torch.load("/home/jackson-devworks/Desktop/VAC/weights/checkpoint_epoch_00030.pyth", weights_only=False)
model.load_state_dict(model_infos['model_state'])
# Kiểm tra mô hình trên tập test
test_result = test_cls_model(model, test_loader, device=DEVICE)
