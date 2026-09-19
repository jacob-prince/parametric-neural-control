
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from collections import OrderedDict
import torch
from torch import nn
from torchvision import transforms
import torch.utils.model_zoo
import os
import pandas as pd
import numpy as np
import torch.nn.functional as F
import time
from tqdm.auto import tqdm
import random
import os
import torch
from pathlib import Path
from torchvision import transforms
import boto3

#todo: clean!

# PNC edit: save_root (download cache for the Brain-Score AlexNet / ReAlnet weights) was a per-user
# getpass.getuser() table (tfel, binxuwang -> /n/netscratch/kempner_fellows/Lab/tfel/cache;
# jacobprince -> /n/holylabs/LABS/alvarez_lab/Everyone/Accentuate_VVS/model_backbones; else tfel's cache).
# Replaced by a single lookup with a user-writable default.
save_root = os.environ.get("PNC_CACHE_DIR", os.path.expanduser("~/.cache/parametric-neural-control"))  # was the per-user table above

def alexnet_v2_pytorch(num_classes=1000, dropout_keep_prob=0.5, global_pool=False):
    """
    Instantiate the AlexNetV2 model in PyTorch.
    """
    class AlexNetV2(nn.Module):
        def __init__(self, num_classes, dropout_keep_prob, global_pool):
            super(AlexNetV2, self).__init__()
            self.global_pool = global_pool

            # Convolutional layers
            self.features = nn.Sequential(
                #conv1
                nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=0),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),

                #conv2
                nn.Conv2d(64, 192, kernel_size=5, stride=1, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),

                #conv3
                nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),

                #conv4
                nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),

                #conv5
                nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2)
            )

            # Fully connected layers
            self.classifier = nn.Sequential(
                #fc6
                nn.Conv2d(256, 4096, kernel_size=5, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.Dropout(p=1 - dropout_keep_prob),

                #fc7
                nn.Conv2d(4096, 4096, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.Dropout(p=1 - dropout_keep_prob),

                #fc8
                nn.Conv2d(4096, num_classes, kernel_size=1, stride=1, padding=0)
            )

        def forward(self, x):
            x = self.features(x)
            if self.global_pool:
                x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
            x = self.classifier(x)
            x = torch.flatten(x, start_dim=1)
            return x

  # Instantiate and return the AlexNetV2 model
    return AlexNetV2(num_classes, dropout_keep_prob, global_pool)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


class Flatten(nn.Module):

    """
    Helper module for flattening input tensor to 1-D for the use in Linear modules
    """

    def forward(self, x):
        return x.view(x.size(0), -1)


class Identity(nn.Module):

    """
    Helper module that stores the current tensor. Useful for accessing by name
    """

    def forward(self, x):
        return x


class CORblock_S(nn.Module):

    scale = 4  # scale of the bottleneck convolution channels

    def __init__(self, in_channels, out_channels, times=1):
        super().__init__()

        self.times = times

        self.conv_input = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.skip = nn.Conv2d(out_channels, out_channels,
                              kernel_size=1, stride=2, bias=False)
        self.norm_skip = nn.BatchNorm2d(out_channels)

        self.conv1 = nn.Conv2d(out_channels, out_channels * self.scale,
                               kernel_size=1, bias=False)
        self.nonlin1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels * self.scale, out_channels * self.scale,
                               kernel_size=3, stride=2, padding=1, bias=False)
        self.nonlin2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(out_channels * self.scale, out_channels,
                               kernel_size=1, bias=False)
        self.nonlin3 = nn.ReLU(inplace=True)

        self.output = Identity()  # for an easy access to this block's output

        # need BatchNorm for each time step for training to work well
        for t in range(self.times):
            setattr(self, f'norm1_{t}', nn.BatchNorm2d(out_channels * self.scale))
            setattr(self, f'norm2_{t}', nn.BatchNorm2d(out_channels * self.scale))
            setattr(self, f'norm3_{t}', nn.BatchNorm2d(out_channels))

    def forward(self, inp):
        x = self.conv_input(inp)

        for t in range(self.times):
            if t == 0:
                skip = self.norm_skip(self.skip(x))
                self.conv2.stride = (2, 2)
            else:
                skip = x
                self.conv2.stride = (1, 1)

            x = self.conv1(x)
            x = getattr(self, f'norm1_{t}')(x)
            x = self.nonlin1(x)

            x = self.conv2(x)
            x = getattr(self, f'norm2_{t}')(x)
            x = self.nonlin2(x)

            x = self.conv3(x)
            x = getattr(self, f'norm3_{t}')(x)

            x += skip
            x = self.nonlin3(x)
            output = self.output(x)

        return output


def CORnet_S():
    model = nn.Sequential(OrderedDict([
        ('V1', nn.Sequential(OrderedDict([  # this one is custom to save GPU memory
            ('conv1', nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                            bias=False)),
            ('norm1', nn.BatchNorm2d(64)),
            ('nonlin1', nn.ReLU(inplace=True)),
            ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
            ('conv2', nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1,
                            bias=False)),
            ('norm2', nn.BatchNorm2d(64)),
            ('nonlin2', nn.ReLU(inplace=True)),
            ('output', Identity())
        ]))),
        ('V2', CORblock_S(64, 128, times=2)),
        ('V4', CORblock_S(128, 256, times=4)),
        ('IT', CORblock_S(256, 512, times=2)),
        ('decoder', nn.Sequential(OrderedDict([
            ('avgpool', nn.AdaptiveAvgPool2d(1)),
            ('flatten', Flatten()),
            ('linear', nn.Linear(512, 1000)),
            ('output', Identity())
        ])))
    ]))

    # weight initialization
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        # nn.Linear is missing here because I originally forgot
        # to add it during the training of this network
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    return model


class Encoder(nn.Module):
    def __init__(self, realnet, n_output):
        super(Encoder, self).__init__()

        # CORnet
        self.realnet = realnet

        # full connected layer
        self.fc_v1 = nn.Linear(200704, 128)
        self.fc_v2 = nn.Linear(100352, 128)
        self.fc_v4 = nn.Linear(50176, 128)
        self.fc_it = nn.Linear(25088, 128)
        self.fc = nn.Linear(512, n_output)
        self.activation = nn.ReLU()

    def forward(self, imgs):

        outputs = self.realnet(imgs)

        N = len(imgs)
        v1_outputs = self.realnet.module.V1(imgs) # N * 64 * 56 * 56
        v2_outputs = self.realnet.module.V2(v1_outputs) # N * 128 * 28 * 28
        v4_outputs = self.realnet.module.V4(v2_outputs) # N * 256 * 14 * 14
        it_outputs = self.realnet.module.IT(v4_outputs) # N * 512 * 7 * 7
        v1_features = self.fc_v1(v1_outputs.view(N, -1))
        v1_features = self.activation(v1_features)
        v2_features = self.fc_v2(v2_outputs.view(N, -1))
        v2_features = self.activation(v2_features)
        v4_features = self.fc_v4(v4_outputs.view(N, -1))
        v4_features = self.activation(v4_features)
        it_features = self.fc_it(it_outputs.view(N, -1))
        it_features = self.activation(it_features)
        features = torch.cat((v1_features, v2_features, v4_features, it_features), dim=1)
        features = self.fc(features)

        return outputs, features



def download_from_s3(
    bucket: str,
    key: str,
    local_path: Path,
    version_id: str = None,
):
    """
    Downloads an S3 object to local_path (creating parents if needed),
    optionally at a specific version.
    """
    s3 = boto3.client("s3")
    # ensure local directory exists
    local_path.parent.mkdir(parents=True, exist_ok=True)

    extra_args = {}
    if version_id:
        extra_args["VersionId"] = version_id

    s3.download_file(
        Bucket=bucket,
        Key=key,
        Filename=str(local_path),
        ExtraArgs=extra_args  # only passes VersionId if given
    )
    return local_path



def build_ReAlnet_model(identifier: str = "ReAlnet01",
                        save_root: str = save_root):
    """
    Build a model from a given identifier.
    """
    import sys

    local_weights_path = f"{save_root}/{identifier}_best_model_params.pt"
    realnet = CORnet_S()
    # (Optional) remove DataParallel if not needed for CPU
    # realnet = torch.nn.DataParallel(realnet)
    # Build encoder model
    encoder = Encoder(realnet, 340)

    weights_info = {
        "version_ids": {
        "ReAlnet01": "75oY3CnI17U5S1f_yrZxl1XGhRfJEG9N",
        "ReAlnet02": "TfGdm1CphJJ1vvkJGcm3n266PHvTuOaV",
        "ReAlnet03": "dmohrH_AHZzgL_o8Xd2SDp6XCnjPOdAu",
        "ReAlnet04": "45qJFXHihmIHdpHbjKWZco6STH1eh49p",
        "ReAlnet05": "nqvoYgiBTyWSskjnpF9YOK4yYQfOnc_H",
        "ReAlnet06": "6.cloFvnMihiicwQ0jkag8reEe4bVlxZ",
        "ReAlnet07": "WKJaiN4b1ttpbGYNn8yVjng4LjCqWdk.",
        "ReAlnet08": "vmouew6ePkPnKP.We8VnVxU7TifuhL.x",
        "ReAlnet09": "53gqQ2tgS.5MEoncipy9mrBEqCc5izw5",
        "ReAlnet10": "ZZFMhTm9KQYEXl8OGwKmnTr0S.pxkU0J"
        },
        "sha1s": {
        "ReAlnet01": "05e4e401e8734b97e561aad306fc584b7e027225",
        "ReAlnet02": "e85769fadb3c09ff88a7d73b01451b6bcccefd77",
        "ReAlnet03": "f32d01d73380374ae501a1504e9c8cd219e9f0bf",
        "ReAlnet04": "8062373fd6a74c52360420619235590d3688b4df",
        "ReAlnet05": "88ca110f6b6d225b7b4e7dca02d2e7a906f5a8ed",
        "ReAlnet06": "a1658c15a3c9d61262f87349c9fb7aa63854ac5b",
        "ReAlnet07": "6a1c260839c75f6e6c018e06830562cdcda877e5",
        "ReAlnet08": "1772211b27dd3a7d9255ac59d5f9b7e7cb6c3314",
        "ReAlnet09": "159d96f0433a87c7063259dac4527325a3c7b79a",
        "ReAlnet10": "dbdeaee9280267613ebce92dd5d515d89b544352"
        }
    }
    version_id = weights_info['version_ids'][identifier]
    sha1 = weights_info['sha1s'][identifier]
    if not os.path.exists(local_weights_path):
        key = f"brainscore-vision/models/ReAlnet/{identifier}_best_model_params.pt"
        download_from_s3(bucket="brainscore-storage", key=key,
                         local_path=Path(local_weights_path), version_id=version_id)

    weights_data = torch.load(local_weights_path, map_location="cpu", weights_only=False)
    new_state_dict = {}
    for key, val in weights_data.items():
        # remove "module." (if it exists) from the key
        new_key = key.replace("module.", "")
        new_state_dict[new_key] = val

    encoder.load_state_dict(new_state_dict)
    # Retrieve the realnet portion from the encoder
    realnet = encoder.realnet
    realnet.eval()
    realnet.requires_grad_(False)
    preprocess = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    )
    return realnet, preprocess


def build_alexnet_brainscore(identifier: str = "training_seed_01",
                             save_root: str = save_root):
    """
    Build a model from a given identifier.
    """

    model = alexnet_v2_pytorch()
    local_weights_path = f"{save_root}/alexnet_weights_{identifier}.pth"
    if not os.path.exists(local_weights_path):
        # sha1 = "4b1bb7810d5288631c04cf4cde882540d6ebee77"
        key = f"models/model_weights/{identifier}.pth"
        download_from_s3(bucket="brainscorevariability", key=key,
                        local_path=Path(local_weights_path), version_id=None)
    model_weights = torch.load(local_weights_path, map_location='cpu', weights_only=False)
    model.load_state_dict(model_weights)
    model.eval()
    model.requires_grad_(False)
    preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return model, preprocess