import logging
from typing import Tuple
from PIL import Image
import timm
import clip
import open_clip
import torch
from torchvision import transforms as T
from os.path import join

from models_utils import build_alexnet_brainscore, build_ReAlnet_model

import os
# PNC edit: the original per-user gate (getpass.getuser(): tfel -> /n/home08/tfel/projects/prj_control_neuronsV2/assets,
# binxuwang -> /n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Projects/VVS_Accentuation/model_backbones,
# jacobprince -> /n/holylabs/LABS/alvarez_lab/Everyone/Accentuate_VVS/model_backbones, else ValueError) is replaced
# by a single lookup; the default is the directory the paper's runs used.
ckptroot = os.environ.get("PNC_MODEL_BACKBONES", "/n/holylabs/LABS/alvarez_lab/Everyone/Accentuate_VVS/model_backbones")  # was the per-user gate above

logger = logging.getLogger(__name__)

class ProbablyToTensor:
    def __call__(self, x):
        if isinstance(x, Image.Image):
            return T.ToTensor()(x)
        elif isinstance(x, torch.Tensor):
            return x
        else:
            raise TypeError(f"Unsupported type for ProbablyToTensor: {type(x)}")

DEFAULT_PREPROCESS = T.Compose([
    ProbablyToTensor(),
    T.Resize((224, 224)),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

def load_dinov2(device: str = "cuda") -> Tuple[torch.nn.Module, T.Compose]:
    logger.info("Loading DinoV2 model")
    try:
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
        model = model.eval().to(device)
        model.requires_grad_(False)
        return model, DEFAULT_PREPROCESS
    except Exception as e:
        logger.error(f"Error loading DinoV2 model: {e}")
        raise

def load_alexnet(device: str = "cuda") -> Tuple[torch.nn.Module, T.Compose]:
    logger.info("Loading AlexNet model")
    try:
        # tfel: clean preprocess
        model, _ = build_alexnet_brainscore(identifier="training_seed_01")
        model = model.eval().to(device)
        model.requires_grad_(False)
        return model, DEFAULT_PREPROCESS
    except Exception as e:
        logger.error(f"Error loading AlexNet: {e}")
        raise

def load_realnet(device: str = "cuda") -> Tuple[torch.nn.Module, T.Compose]:
    logger.info("Loading ReAlNet model")
    try:
        # tfel: clean preprocess
        model, _ = build_ReAlnet_model(identifier="ReAlnet01")
        model = model.eval().to(device)
        model.requires_grad_(False)
        return model, DEFAULT_PREPROCESS
    except Exception as e:
        logger.error(f"Error loading ReAlNet: {e}")
        raise

def load_regnety(device: str = "cuda") -> Tuple[torch.nn.Module, T.Compose]:
    logger.info("Loading RegNetY model")
    try:
        model = timm.create_model('regnety_640.seer', pretrained=True, num_classes=0)
        model = model.eval().to(device)
        model.requires_grad_(False)
        data_config = timm.data.resolve_model_data_config(model)
        transforms_pipeline = timm.data.create_transform(**data_config, is_training=False)
        transforms_pipeline = T.Compose([
            ProbablyToTensor(),
            T.Resize(size=232, interpolation=T.InterpolationMode.BICUBIC, 
                     max_size=None, antialias=True),
            T.CenterCrop(224),
            # get rid of the transform to rgb
            T.Normalize(mean=[0.4850, 0.4560, 0.4060],
                        std=[0.2290, 0.2240, 0.2250]),
        ])
        return model, transforms_pipeline
    except Exception as e:
        logger.error(f"Error loading RegNetY: {e}")
        raise

def load_radiov2_5(device: str = "cuda") -> Tuple[torch.nn.Module, T.Compose]:
    logger.info("Loading RadioV2.5 model")
    try:
        model = torch.hub.load('NVlabs/RADIO', 'radio_model', version="radio_v2.5-b", progress=True, skip_validation=True)
        model = model.eval().to(device)
        model.requires_grad_(False)
        preprocess = T.Compose([
            ProbablyToTensor(),
            T.Resize((224, 224)),
        ])
        return model, preprocess
    except Exception as e:
        logger.error(f"Error loading RadioV2.5 model: {e}")
        raise

def load_resnet50(device: str = "cuda") -> Tuple[torch.nn.Module, T.Compose]:
    logger.info("Loading ResNet-50")
    try:
        model = torch.hub.load('pytorch/vision', 'resnet50', pretrained=True)
        model = model.eval().to(device)
        model.requires_grad_(False)
        return model, DEFAULT_PREPROCESS
    except Exception as e:
        logger.error(f"Error loading ResNet-50: {e}")
        raise

def load_resnet50_robust(device: str = "cuda") -> Tuple[torch.nn.Module, T.Compose]:
    logger.info("Loading ResNet-50 Robust model")
    try:
        model = torch.hub.load('pytorch/vision', 'resnet50', pretrained=False)
        model.load_state_dict(torch.load(join(ckptroot, "imagenet_linf_8_pure.pt"), weights_only=False))
        model = model.eval().to(device)
        model.requires_grad_(False)
        return model, DEFAULT_PREPROCESS
    except Exception as e:
        logger.error(f"Error loading ResNet-50 Robust: {e}")
        raise

def load_resnet50_clip(device: str = "cuda") -> Tuple[torch.nn.Module, T.Compose]:
    logger.info("Loading ResNet-50 CLIP model")
    try:
        model_clip, _ = clip.load('RN50', device=device)
        model = model_clip.visual.eval().to(device)
        model.requires_grad_(False)
        return model, DEFAULT_PREPROCESS
    except Exception as e:
        logger.error(f"Error loading ResNet-50 CLIP: {e}")
        raise

def load_resnet50_dino(device: str = "cuda") -> Tuple[torch.nn.Module, T.Compose]:
    logger.info("Loading ResNet-50 DINO model")
    try:
        model = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
        model = model.eval().to(device)
        model.requires_grad_(False)
        return model, DEFAULT_PREPROCESS
    except Exception as e:
        logger.error(f"Error loading ResNet-50 DINO: {e}")
        raise

def load_clipag_vitb32(device: str = "cuda") -> Tuple[torch.nn.Module, T.Compose]:
    logger.info("Loading CLIPAG ViT-B/32 model")
    try:
        ckpt_path = join(ckptroot, "CLIPAG_ViTB32.pt")
        data = torch.load(ckpt_path, map_location='cpu', weights_only=False)['state_dict']
        data = {k.replace('module.', ''): v for k, v in data.items()}
        model_clip, _, transforms_pipeline = open_clip.create_model_and_transforms('ViT-B/32', device=device)
        model_clip.load_state_dict(data)
        model = model_clip.visual.eval().to(device)
        model.requires_grad_(False)
        transforms_pipeline = T.Compose([
            ProbablyToTensor(),
            T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC, 
                     max_size=None, antialias=True),
            T.CenterCrop(224),
            # get rid of the transform to rgb
            T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                        std=(0.26862954, 0.26130258, 0.27577711)),
        ])
        return model, transforms_pipeline
    except Exception as e:
        logger.error(f"Error loading CLIPAG ViT-B/32: {e}")
        raise

def load_siglip2_vitb16(device: str = "cuda") -> Tuple[torch.nn.Module, T.Compose]:
    logger.info("Loading SigLIP2 ViT-B/16 model")
    try:
        model, transforms_pipeline = open_clip.create_model_from_pretrained('hf-hub:timm/ViT-B-16-SigLIP2')
        model = model.visual.eval().to(device)
        model.requires_grad_(False)
        transforms_pipeline = T.Compose([
            ProbablyToTensor(),
            T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC, 
                     max_size=None, antialias=True),
            # we got rid of the transform to rgb
            T.Normalize(mean=[0.5, 0.5, 0.5],
                        std=[0.5, 0.5, 0.5]), 
        ])
        return model, transforms_pipeline
    except Exception as e:
        logger.error(f"Error loading SigLIP2 ViT-B/16: {e}")
        raise

MODEL_LOADERS = {
    "dinov2_vitb14_reg": load_dinov2,
    "radio_v2.5-b": load_radiov2_5,
    "regnety_640": load_regnety,
    "AlexNet_training_seed_01": load_alexnet,
    "ReAlnet01": load_realnet,
    "resnet50": load_resnet50,
    "resnet50_robust": load_resnet50_robust,
    "resnet50_clip": load_resnet50_clip,
    "resnet50_dino": load_resnet50_dino,
    "clipag_vitb32": load_clipag_vitb32,
    "siglip2_vitb16": load_siglip2_vitb16,
}

def load_model(model_name: str, device: str = "cuda") -> Tuple[torch.nn.Module, T.Compose]:
    logger.info(f"Dispatching model loader for: {model_name}")
    loader = MODEL_LOADERS.get(model_name)
    if loader is None:
        raise ValueError(f"Model '{model_name}' is not supported.")
    return loader(device)
