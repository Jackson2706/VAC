import sys
# sys.path.append('/data2/hongn/RAFT')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image
import torch.nn as nn
from raft.raft import RAFT
from raft.utils import flow_viz
from raft.utils.utils import InputPadder
from PIL import Image
from einops import rearrange
from .build import MODEL_REGISTRY
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as pth_transforms
import numpy as np
from PIL import Image
# sys.path.append('/data2/hongn/dino')
import utils_
import vision_transformer as vits

import torch.nn.functional as F

def cross_attention(Q, K, V, mask=None):
    # Compute the dot products between Q and K, then scale
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    
    # Apply mask if provided
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    # Softmax to normalize scores and get attention weights
    attention_weights = F.softmax(scores, dim=-1)
    
    # Weighted sum of values
    output = torch.matmul(attention_weights, V)
    return output, attention_weights

class CustomAttention(nn.Module):
    def __init__(self, embed_size, num_heads=1, attention_type="cross"):
        super(CustomAttention, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.attention_type = attention_type

        # Linear layers for Q, K, V
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)
        
        # Final linear layer after concatenating heads
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, target, source=None, mask=None):
        if self.attention_type == "self":
            Q = self.query(target)
            K = self.key(target)
            V = self.value(target)
        elif self.attention_type == "cross":
            assert source is not None, "Source input required for cross-attention"
            Q = self.query(target)
            K = self.key(source)
            V = self.value(source)

        # Perform attention calculation (self or cross)
        out, _ = cross_attention(Q, K, V, mask)
        return self.fc_out(out)

class CustomAttentionWithResidual(nn.Module):
    def __init__(self, embed_size, num_heads=1, attention_type="self"):
        super(CustomAttentionWithResidual, self).__init__()
        self.attention = CustomAttention(embed_size, num_heads, attention_type)
        self.norm = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, target, source=None, mask=None):
        attention_out = self.attention(target, source, mask)
        # Add residual connection and layer normalization
        out = self.norm(target + self.dropout(attention_out))
        return out

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None]


def load_image_from_uint8array(img):
    img = torch.from_numpy(img).permute(0, 3, 1, 2).float()
    return img[None]

def denomalizing_img(tensor_img, mean = [0.45, 0.45, 0.45], std = [0.225, 0.225, 0.225]): # tensor_img (c t w h)
    device = tensor_img.device
    if(len(tensor_img.shape) == 4):
        ret = []
        for i in range(tensor_img.shape[0]):
            unnormalize_img = tensor_img[i].permute(1, 2, 0) * torch.tensor(std).to(device) + torch.tensor(mean).to(device)
            unnormalize_img = unnormalize_img * 225
            # unnormalize_img = unnormalize_img.astype(np.uint8)
            ret.append(unnormalize_img.permute(2, 0, 1).float())
        ret = torch.stack(ret)
    return ret

class MOOSE_Encoder(nn.Module):
    """ Vision Transformer """
    def __init__(self, raft_args, num_classes=1000, embed_dim=768):
        super().__init__()

        # self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.motion_model = self._init_motion_model(raft_args)
        self.patch_embed = PatchEmbed(img_size=28, patch_size=2, in_chans=2, embed_dim=768) # Flow patches embedding
        self.visual_model = self._init_visual_model(dino_args)
        self.crossatt = CustomAttentionWithResidual(embed_size = embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def _init_motion_model(self, args):
        with torch.no_grad():
            motion_model = torch.nn.DataParallel(RAFT(args))
            for p in motion_model.parameters():
                p.requires_grad = False
        # motion_model.eval()
        # motion_model.to(device)    
        motion_model.load_state_dict(torch.load(args.model))
        motion_model = motion_model.module
        return motion_model

    def _init_visual_model(self, args):
        # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # build model
        with torch.no_grad():
            model = vits.__dict__['vit_base'](patch_size=16, num_classes=0)
            for p in model.parameters():
                p.requires_grad = False
        # model.eval()
        # model.to(device)
        print("Since no pretrained weights have been provided, we load the reference pretrained DINO weights.")
        url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
        model.load_state_dict(state_dict, strict=True)
        return model

    def motion_forward(self, x):
        '''
            Inputs should have shape of (b, c, t, w, h) Example: (4, 3, 8, 224, 224)
        '''
        with torch.no_grad():
            x = rearrange(x, 'b c t w h -> b t c w h')
            flow_embs_batches = []
            for inputs in x: # loop over batches
                # denomalized_imgs = denomalizing_img(inputs)
                # images = load_image_from_uint8array(denomalized_imgs)
                images = denomalizing_img(inputs)
                flow_embs = []
                for index in range(images.shape[0]-1): #loop over time
                    image1 = images[index].unsqueeze(0)
                    image2 = images[index+1].unsqueeze(0)
                    # print(image1.shape)
                    padder = InputPadder(image1.shape)
                    image1, image2 = padder.pad(image1, image2)

                    flow_low, flow_up = self.motion_model(image1, image2, iters=20, test_mode=True)
                    # flow_embedding = self.patch_embed(flow_low)
                    flow_embs.append(flow_low)
                    
                    # viz(image1, flow_up, index)
                flow_embs = torch.stack(flow_embs)
                flow_embs_batches.append(flow_embs)
            ret = torch.stack(flow_embs_batches)
        return ret

    def visual_forward(self, inputs):
        '''
            Inputs should have shape of (b, c, t, w, h) Example: (4, 3, 8, 224, 224)
        '''
        with torch.no_grad():
            index_of_time_to_get_principle_embeddings = 0
            img = inputs[:,:,index_of_time_to_get_principle_embeddings,:]
            features = self.visual_model.get_intermediate_layers(img, len(self.visual_model.blocks))
            features = features[-1]  # residual stream @ final block
            # features.requires_grad = False
        return features
        # print([features.shape])

    def forward(self, inputs):
        with torch.no_grad():
            visual_embeddings = self.visual_forward(inputs) # b x 1 x 197 x 768
            motion_embeddings = self.motion_forward(inputs) # b x t x 196 x 768
            # visual_embeddings.requires_grad = False
            # motion_embeddings.requires_grad = False
        # video_embeddings = visual_embeddings
        # for i in range(motion_embeddings.shape[1]):
        #     video_embeddings = self.crossatt(video_embeddings, motion_embeddings[:,i,:][1])
        # x = video_embeddings[:, 0]
        # x = self.head(x)
        return visual_embeddings, motion_embeddings
