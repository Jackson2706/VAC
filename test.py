from datasets import Kinetics
from  utils_ import  load_config

cfg = load_config(cfg_file='/home/dung2/VAC/configs/default.yaml')
dataset = Kinetics(
    cfg=cfg,
    mode="val"
)

print(dataset[0][0].shape)