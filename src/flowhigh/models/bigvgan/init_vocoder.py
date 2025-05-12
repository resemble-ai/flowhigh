import json
import torch

from .models import BigVGAN
from .env import AttrDict


def init_bigvgan(config, checkpoint, vocoder_freeze=False):

    with open(config) as f:
        h = AttrDict(json.load(f))

    vocoder = BigVGAN(h)
    checkpoint_dict = torch.load(checkpoint, map_location="cuda")
    vocoder.load_state_dict(checkpoint_dict['generator'])
    _ = vocoder.cuda().eval()
    vocoder.remove_weight_norm()

    if vocoder_freeze == True:
        for param in vocoder.parameters():
            param.requires_grad = False

    return vocoder
