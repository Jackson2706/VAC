import torch
import torch.jit as jit
import torch.nn as nn
import torch.nn.functional as F


class SapiensCLS(nn.Module):
    def __init__(self, encoder_weight_path: str, num_classes: int):
        super(SapiensCLS, self).__init__()
        self.encoder = jit.load(encoder_weight_path)
        
        # Assuming the output from the encoder is [B, 1024, 64, 64]
        self.cls = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Reduces to [B, 1024, 1, 1]
            nn.Flatten(),            # Flattens to [B, 1024]
            nn.Linear(1, num_classes)  # Linear layer for classification
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)
        x = self.encoder(x)[0]
        x = x.reshape(B, -1)
        # print(x.shape)
        x = self.cls(x)
        return x
if __name__ == "__main__":
    # Example usage:
    model = SapiensCLS(encoder_weight_path='/home/jackson-devworks/Desktop/VAC/weights/sapiens_0.3b_epoch_1600_torchscript.pt2', num_classes=10)
    output = model(torch.randn(1, 1, 3, 1024, 1024))  # Assuming input image size is [B, 3, 256, 256]
    print(output.shape)