import torch
from torch.utils.data import DataLoader
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.core.xla_model as xm
from config import *
from model import DINO_CLS
from utils_ import *
from datasets import Kinetics
from utils_ import load_config

def train_fn(rank, cfg, num_epochs, lr):
    """
    Hàm train được chạy trên mỗi TPU core
    """
    # Chỉ khởi tạo XLA device trong hàm này
    device = xm.xla_device()
    
    # Load dataset và sampler
    train_dataset = Kinetics(cfg=cfg, mode="train")
    val_dataset = Kinetics(cfg=cfg, mode="val")

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal(), shuffle=True
    )
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal(), shuffle=False
    )

    train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, sampler=train_sampler)
    train_loader = pl.MpDeviceLoader(train_loader, device)
    
    val_loader = DataLoader(val_dataset, batch_size=VAL_BATCH_SIZE, sampler=val_sampler)
    val_loader = pl.MpDeviceLoader(val_loader, device)

    # Khởi tạo model và đưa lên TPU
    model = DINO_CLS(num_classes=NUM_CLS).to(device)
    
    # Train model
    best_model = train_cls_model(model, train_loader, val_loader, num_epochs, lr)

    # Lưu model trên TPU-0
    if xm.get_ordinal() == 0:
        torch.save(best_model.state_dict(), "best_model.pth")

if __name__ == "__main__":
    cfg = load_config(cfg_file='/home/dung2/VAC/configs/default.yaml')

    # Chạy train trên tất cả TPU cores
    xmp.spawn(train_fn, args=(cfg, NUM_EPOCH, LR), nprocs=None)

    # Load model đã lưu sau khi train xong
    device = xm.xla_device()
    model = DINO_CLS(num_classes=NUM_CLS).to(device)
    model.load_state_dict(torch.load("best_model.pth", map_location=device))

    # Load tập test
    test_dataset = Kinetics(cfg=cfg, mode="test")
    test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False)
    test_loader = pl.MpDeviceLoader(test_loader, device)

    # Chạy test
    test_result = test_cls_model(model, test_loader)
    print("Test Results:", test_result)
