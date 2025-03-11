import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from configs import  get_cfg
from _utils import checkpoint as cu
import torch_xla.core.xla_model as xm


def load_config(args = None, cfg_file = "/data2/hongn/TimeSformer/configs/Rtmri75s/simple_cfg.yaml"):
    """
    Given the arguemnts, load and initialize the configs.
    Args:
        args (argument): arguments includes `shard_id`, `num_shards`,
            `init_method`, `cfg_file`, and `opts`.
    """
    # Setup cfg.
    cfg = get_cfg()
    if(args is not None):
        # Load configs from cfg
        if args.cfg_file is not None:
            cfg.merge_from_file(args.cfg_file)
        # Load configs from command line, overwrite configs from opts.
        if args.opts is not None:
            cfg.merge_from_list(args.opts)

        # Inherit parameters from args.
        if hasattr(args, "num_shards") and hasattr(args, "shard_id"):
            cfg.NUM_SHARDS = args.num_shards
            cfg.SHARD_ID = args.shard_id
        if hasattr(args, "rng_seed"):
            cfg.RNG_SEED = args.rng_seed
        if hasattr(args, "output_dir"):
            cfg.OUTPUT_DIR = args.output_dir
    else:
        cfg.merge_from_file(cfg_file)

    # Create the checkpoint dir.
    cu.make_checkpoint_dir(cfg.OUTPUT_DIR)

    return cfg
def train_cls_model(model, train_loader, val_loader, num_epochs, lr):
    """
    Huấn luyện mô hình phân loại (train + validation).

    Args:
        model: Mô hình cần huấn luyện.
        train_loader: DataLoader cho tập huấn luyện.
        val_loader: DataLoader cho tập validation.
        num_epochs: Số epoch.
        lr: Learning rate.
        device: 'cuda' hoặc 'cpu'.

    Returns:
        Mô hình có kết quả tốt nhất trên tập validation.
    """
    device = xm.xla_device()
    model = model.to(device).to(torch.bfloat16)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.cls.parameters(), lr=lr)

    best_acc = 0.0
    best_model_state = None
    torch.cuda.empty_cache()
    for epoch in tqdm(range(num_epochs)):
        # Training phase
        torch.cuda.empty_cache()
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for  inputs, labels, _, _ in tqdm(train_loader):
            torch.cuda.empty_cache()
            inputs = inputs.float()
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            xm.optimizer_step(optimizer)
            xm.mark_step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total

        # Validation phase
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            torch.cuda.empty_cache()
            for  inputs, labels, _, _ in tqdm(val_loader):
                torch.cuda.empty_cache()

                inputs = inputs.float()
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.4f} - Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.4f}")

        # Lưu mô hình có độ chính xác validation tốt nhất
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")


    print(f"Best Validation Accuracy: {best_acc:.4f}")



def test_cls_model(model, test_loader):
    """
    Đánh giá mô hình phân loại trên tập test với Top-1, Top-5 Accuracy và F1-score.

    Args:
        model: Mô hình đã train xong.
        test_loader: DataLoader cho tập test.
        device: 'cuda' hoặc 'cpu'.

    Returns:
        Dictionary chứa các metric: Top-1 Acc, Top-5 Acc, F1-score.
    """
    device = xm.xla_device()
    model = model.to(device).to(torch.bfloat16)
    model.eval()

    correct_top1, correct_top5, total = 0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        torch.cuda.empty_cache()
        for inputs, labels, _, _ in tqdm(test_loader):
            torch.cuda.empty_cache()
            
            inputs = inputs.float()
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)  # Shape: [B, num_classes]
            _, predicted_top1 = torch.max(outputs, 1)  # Top-1 prediction
            _, predicted_top5 = torch.topk(outputs, 5, dim=1)  # Top-5 predictions
            
            # Top-1 accuracy
            correct_top1 += (predicted_top1 == labels).sum().item()

            # Top-5 accuracy
            correct_top5 += sum([labels[i] in predicted_top5[i] for i in range(labels.size(0))])

            # Collect predictions for F1-score
            all_preds.extend(predicted_top1.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            total += labels.size(0)

    top1_acc = correct_top1 / total
    top5_acc = correct_top5 / total
    f1 = f1_score(all_labels, all_preds, average="macro")  # Macro F1-score

    print(f"Top-1 Accuracy: {top1_acc:.4f}, Top-5 Accuracy: {top5_acc:.4f}, F1-score: {f1:.4f}")
    
    return {"Top-1 Accuracy": top1_acc, "Top-5 Accuracy": top5_acc, "F1-score": f1}
