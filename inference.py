import torch
from torch.utils.data import Dataloader
from torchvision.datasets.kinetics import Kinetics

from config import *
from model import SapiensCLS

train_dataset = Kinetics(root="./k400", frames_per_clip=8, split ="val", download=False, num_workers=32)
val_dataset = Kinetics(root="./k400", frames_per_clip=8, split ="val", download=False, num_workers=32)
test_dataset = Kinetics(root="./k400", frames_per_clip=8, split ="val", download=False, num_workers=32)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = SapiensCLS(encoder_weight_path=ENCODER_WEIGHT_PATH, num_classes=NUM_CLS)

# Huấn luyện mô hình
best_model = train_cls_model(model, train_loader, val_loader, num_epochs=20, lr=0.001, device='cuda')

# Kiểm tra mô hình trên tập test
test_result = test_cls_model(best_model, test_loader, device='cuda')
print(f"Test result: \n {test_result}")