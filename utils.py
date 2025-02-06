import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader


def train_cls_model(model, train_loader, val_loader, num_epochs, lr, device):
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
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_acc = 0.0
    best_model_state = None

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for inputs, _ ,labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total

        # Validation phase
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, _ , labels in val_loader:
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
            best_model_state = model.state_dict()

    print(f"Best Validation Accuracy: {best_acc:.4f}")
    
    # Trả về mô hình tốt nhất
    if best_model_state:
        model.load_state_dict(best_model_state)
    return model



def test_cls_model(model, test_loader, device):
    """
    Đánh giá mô hình phân loại trên tập test với Top-1, Top-5 Accuracy và F1-score.

    Args:
        model: Mô hình đã train xong.
        test_loader: DataLoader cho tập test.
        device: 'cuda' hoặc 'cpu'.

    Returns:
        Dictionary chứa các metric: Top-1 Acc, Top-5 Acc, F1-score.
    """
    model.to(device)
    model.eval()

    correct_top1, correct_top5, total = 0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, _, labels in test_loader:
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
