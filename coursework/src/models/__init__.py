# Copyright (c) EEEM071, University of Surrey

from .resnet import (
    resnet18,
    resnet18_fc512,
    resnet34,
    resnet34_fc512,
    resnet50,
    resnet50_fc512,
)
from .clip_senet import clip_senet_v1_dual, clip_senet_v2_frozen
from .clip_senet_v3 import clip_senet_v3_ibn_supcon
from .tvmodels import mobilenet_v3_small
from .timm_model import swin_t_fc512 , swin_t_custom


__model_factory = {
    # image classification models
    "resnet18": resnet18,
    "resnet18_fc512": resnet18_fc512,
    "resnet34": resnet34,
    "resnet34_fc512": resnet34_fc512,
    "resnet50": resnet50,
    "resnet50_fc512": resnet50_fc512,
    "mobilenet_v3_small": mobilenet_v3_small,
    "swin_t_fc512" : swin_t_fc512,
    "swin_t_custom" : swin_t_custom,
    "clip_senet_v1_dual"  : clip_senet_v1_dual,
    "clip_senet_v2_frozen"  : clip_senet_v2_frozen,
    "clip_senet_v3_ibn_supcon"  : clip_senet_v3_ibn_supcon,
}


def get_names():
    return list(__model_factory.keys())


def init_model(name, *args, **kwargs):
    if name not in list(__model_factory.keys()):
        raise KeyError(f"Unknown model: {name}")
    return __model_factory[name](*args, **kwargs)
