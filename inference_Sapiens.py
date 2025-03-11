import torch
from torch.utils.data import DataLoader
from torchvision.datasets.kinetics import Kinetics
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

from configs import *
from model import SapiensCLS
from utils_ import *

train_dataset = Kinetics(
    root=DATASET_DIR, 
    frames_per_clip=FRAME_PER_CLIP, 
    split ="val", download=True, 
    num_workers=NUM_WORKER_LOADING, 
    transform = transforms.Lambda(
        lambda x: torch.stack([transforms.Resize((1024, 1024), interpolation=InterpolationMode.BILINEAR)(frame) for frame in x])
    )
)

val_dataset = Kinetics(
    root=DATASET_DIR, 
    frames_per_clip=FRAME_PER_CLIP, 
    split ="val", download=True, 
    num_workers=NUM_WORKER_LOADING, 
    transform = transforms.Lambda(
        lambda x: torch.stack([transforms.Resize((1024, 1024), interpolation=InterpolationMode.BILINEAR)(frame) for frame in x])
    )
)

test_dataset = Kinetics(
    root=DATASET_DIR, 
    frames_per_clip=FRAME_PER_CLIP, 
    split ="test", download=True, 
    num_workers=NUM_WORKER_LOADING, 
    transform = transforms.Lambda(
        lambda x: torch.stack([transforms.Resize((1024, 1024), interpolation=InterpolationMode.BILINEAR)(frame) for frame in x])
    )
)
train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=VAL_BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False)

model = SapiensCLS(encoder_weight_path=ENCODER_WEIGHT_PATH, num_classes=NUM_CLS)

# Huấn luyện mô hình
best_model = train_cls_model(model, train_loader, val_loader, num_epochs=NUM_EPOCH, lr=LR, device=DEVICE)

# Kiểm tra mô hình trên tập test
test_result = test_cls_model(best_model, test_loader, device=DEVICE)
