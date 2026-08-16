import torch as th
import torch
import torchvision.transforms as T
import torchvision.models as models
from torchvision.models import resnet50

def load_model_transform(modelname, device="cuda"):
    # Prepare model and transforms
    if modelname == "resnet50_robust":
        model = resnet50(pretrained=False)
        model.load_state_dict(th.load("/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Projects/VVS_Accentuation/model_backbones/imagenet_linf_8_pure.pt", weights_only=False))
        transforms_pipeline = T.Compose([
            T.ToTensor(),
            T.Resize((224, 224)),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    elif modelname == "resnet50":
        model = resnet50(pretrained=True)
        transforms_pipeline = T.Compose([
            T.ToTensor(),
            T.Resize((224, 224)),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    elif modelname == "resnet50_clip":
        import clip
        model_clip, preprocess = clip.load('RN50', device=device)
        model = model_clip.visual
        transforms_pipeline = preprocess
    elif modelname == "resnet50_dino":
        # https://github.com/facebookresearch/dino
        model = th.hub.load('facebookresearch/dino:main', 'dino_resnet50')
        transforms_pipeline = T.Compose([
            T.ToTensor(),
            T.Resize((224, 224)),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    elif modelname == "clipag_vitb32":
        import open_clip
        from os.path import join
        ckpt_dir = '/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Projects/VVS_Accentuation/model_backbones'
        data = torch.load(join(ckpt_dir, 'CLIPAG_ViTB32.pt'), map_location='cpu', weights_only=False)
        data = data['state_dict']
        data = {k.replace('module.', ''): v for k, v in data.items()}
        model_clip, _, transforms_pipeline = open_clip.create_model_and_transforms('ViT-B/32',device="cuda")
        model_clip.load_state_dict(data)
        model = model_clip.visual
        # tokenizer = open_clip.get_tokenizer('ViT-B-32')
    elif modelname == "siglip2_vitb16":
        import open_clip
        siglip_model, transforms_pipeline = open_clip.create_model_from_pretrained('hf-hub:timm/ViT-B-16-SigLIP2')
        model = siglip_model.visual
    elif modelname == "dinov2_vitb14_reg":
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
        transforms_pipeline = T.Compose([
            T.ToTensor(),
            T.Resize((224, 224)),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    elif modelname == "radio_v2.5-b":
        # for RADIOv2.5-B model (ViT-B/16)
        resolution = (224, 224)
        #model_version="e-radio_v2" # for E-RADIO
        model = torch.hub.load('NVlabs/RADIO', 'radio_model', version="radio_v2.5-b", progress=True, skip_validation=True)
        transforms_pipeline = T.Compose([
            T.ToTensor(),
            T.Resize((224, 224)),
            # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    elif modelname == "ReAlnet01":
        import sys
        sys.path.append("/n/home12/binxuwang/Github/Closed-loop-visual-insilico")
        from core.brainscore_model_utils import build_ReAlnet_model
        model, transforms_pipeline = build_ReAlnet_model(identifier="ReAlnet01")
    elif modelname == "AlexNet_training_seed_01":
        import sys
        sys.path.append("/n/home12/binxuwang/Github/Closed-loop-visual-insilico")
        from core.brainscore_model_utils import build_alexnet_brainscore
        model, transforms_pipeline = build_alexnet_brainscore(identifier="training_seed_01")
    elif modelname == "regnety_640":
        import timm
        model = timm.create_model(
            'regnety_640.seer',
            pretrained=True,
            num_classes=0,  # remove classifier nn.Linear
        )
        model = model.eval()
        # get model specific transforms (normalization, resize)
        data_config = timm.data.resolve_model_data_config(model)
        transforms_pipeline = timm.data.create_transform(**data_config, is_training=False)
    else:
        raise ValueError(f"Unknown model: {modelname}")
        # model = timm.create_model(modelname, pretrained=True).to(device).eval()
        # data_config = timm.data.resolve_model_data_config(model)
        # transforms_pipeline = timm.data.create_transform(**data_config, is_training=False)
    model = model.to(device).eval()
    model.requires_grad_(False)
    return model, transforms_pipeline


def make_keyword_filter(*substrs):
    return lambda name: any(s in name for s in substrs)


# define, once, the patterns you care about per model
MODEL_LAYER_FILTERS = {
    "siglip2_vitb16":      make_keyword_filter(".trunk.blocks.Block", ".trunk.AttentionPoolLatentattn_pool"),
    "dinov2_vitb14_reg":   make_keyword_filter(".blocks.NestedTensorBlock"),
    "radio_v2.5-b":        make_keyword_filter(".model.blocks.Block"),
    "clipag_vitb32":       make_keyword_filter("ResidualAttentionBlock"),
    # all the ResNet-50 variants use the same Bottleneck block
    "resnet50_clip":       make_keyword_filter("Bottleneck"),
    "resnet50_dino":       make_keyword_filter("Bottleneck"),
    "resnet50_robust":     make_keyword_filter("Bottleneck"),
    "resnet50":            make_keyword_filter("Bottleneck"),
    "AlexNet_training_seed_01": make_keyword_filter("ReLU", "Conv2d", "MaxPool"),
    "regnety_640":         make_keyword_filter("Bottleneck"),
    # ReAlnet
    # "ReAlnet01": 
    # "AlexNet_training_seed_01": 
    # "regnety_640": 
}


LAYER_ABBREVIATION_MAPS = {
    "siglip2_vitb16":           lambda layername: layername.replace(".trunk.blocks.Block", "B").replace(".trunk.AttentionPoolLatentattn_pool", "attnpool"),
    "dinov2_vitb14_reg":        lambda layername: layername.replace(".blocks.NestedTensorBlock", "B"),
    "radio_v2.5-b":             lambda layername: layername.replace(".model.blocks.Block", "B"),
    "clipag_vitb32":            lambda layername: layername.replace(".transformer.resblocks.ResidualAttentionBlock", "B"),
    "resnet50_clip":            lambda layername: layername.replace("Bottleneck", "B").replace(".layer", "L"),
    "resnet50_dino":            lambda layername: layername.replace("Bottleneck", "B").replace(".layer", "L"),
    "resnet50_robust":          lambda layername: layername.replace("Bottleneck", "B").replace(".layer", "L"),
    "resnet50":                 lambda layername: layername.replace("Bottleneck", "B").replace(".layer", "L"),
    "AlexNet_training_seed_01": lambda layername: alexnet_rename_dict[layername] if layername in alexnet_rename_dict else layername,
    "regnety_640":              lambda layername: layername.replace("Bottleneckb", "B").replace(".s", "S"),
    # AlexNet
    # ReAlnet
    # "ReAlnet01": 
    # "AlexNet_training_seed_01": 
    # "regnety_640": 
}


alexnet_rename_dict = {'.features.Conv2d0': 'conv1',
    '.features.ReLU1': 'relu1',
    '.features.MaxPool2d2': 'maxpool1',
    '.features.Conv2d3': 'conv2',
    '.features.ReLU4': 'relu2',
    '.features.MaxPool2d5': 'maxpool2',
    '.features.Conv2d6': 'conv3',
    '.features.ReLU7': 'relu3',
    '.features.Conv2d8': 'conv4',
    '.features.ReLU9': 'relu4',
    '.features.Conv2d10': 'conv5',
    '.features.ReLU11': 'relu5',
    '.features.MaxPool2d12': 'maxpool5',
    '.classifier.Conv2d0': 'fc6',
    '.classifier.ReLU1': 'relu6',
    '.classifier.Conv2d3': 'fc7',
    '.classifier.ReLU4': 'relu7',
    '.classifier.Conv2d6': 'fc8'}


# OLDER VERSION
# if modelname == "siglip2_vitb16":
#     layer_filter = lambda name: ".trunk.blocks.Block" in name or ".trunk.AttentionPoolLatentattn_pool" in name
#     # module_names = [name for name in fetcher.module_names.values()   if ".trunk.blocks.Block" in name or ".trunk.AttentionPoolLatentattn_pool" in name]
# elif modelname == "dinov2_vitb14_reg":
#     layer_filter = lambda name: ".blocks.NestedTensorBlock" in name
#     # module_names = [name for name in fetcher.module_names.values() if ".blocks.NestedTensorBlock" in name]
# elif modelname == "radio_v2.5-b":
#     layer_filter = lambda name: ".model.blocks.Block" in name
#     # module_names = [name for name in fetcher.module_names.values() if ".model.blocks.Block" in name]
# elif modelname == "clipag_vitb32":
#     layer_filter = lambda name: "ResidualAttentionBlock" in name
#     # module_names = [name for name in fetcher.module_names.values() if "ResidualAttentionBlock" in name]
# elif modelname == "resnet50_clip":
#     layer_filter = lambda name: "Bottleneck" in name
#     # module_names = [name for name in fetcher.module_names.values() if "Bottleneck" in name]
# elif modelname == "resnet50_dino":
#     layer_filter = lambda name: "Bottleneck" in name
#     # module_names = [name for name in fetcher.module_names.values() if "Bottleneck" in name]
# elif modelname == "resnet50_robust":
#     layer_filter = lambda name: "Bottleneck" in name
#     # module_names = [name for name in fetcher.module_names.values() if "Bottleneck" in name]
# elif modelname == "resnet50":
#     layer_filter = lambda name: "Bottleneck" in name
#     # module_names = [name for name in fetcher.module_names.values() if "Bottleneck" in name]
# else:
#     raise ValueError(f"Unknown model: {modelname}")







