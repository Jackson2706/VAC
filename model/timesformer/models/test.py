import torch
from vit import MOOSE


model = MOOSE()
model = torch.load("/home/jackson-devworks/Desktop/VAC/weights/checkpoint_epoch_00030.pyth", weights_only=False)
model.load_state_dict(model_infos['model_state'])

print(model)