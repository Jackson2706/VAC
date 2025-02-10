from torch import  nn
import torch

class DINO_CLS(nn.Module):
    def __init__(self, backbone_name: str = "dino_vits16", num_classes: int = 400):
        super(DINO_CLS, self).__init__()

        self.encoder = torch.hub.load('facebookresearch/dino:main', backbone_name)
        self.cls = nn.Sequential(
            nn.Linear(384, num_classes)  # Linear layer for classification
        )
        
    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)
        x = self.encoder(x)
        x = self.cls(x)
        return x


if __name__ == "__main__":
    # Example usage:
    model = DINO_CLS()
    output = model(torch.randn(1, 1, 3, 224, 224))  # Assuming input image size is [B, 3, 256, 256]
    print(output.shape)