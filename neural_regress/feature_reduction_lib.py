"""Use this to handle common token selection operations, 
We output a dictionary of transforms (in nn.Module) format such that collaborators can use it in a unified way.
"""

import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.random_projection import SparseRandomProjection


# Base token selection classes
class TokenSelector(nn.Module):
    """Base class for token selection operations"""
    def __init__(self):
        super().__init__()
    
    def forward(self, X: torch.Tensor):
        raise NotImplementedError("Subclasses must implement forward method")


class SingleTokenSelector(TokenSelector):
    """Select a single token at a specific index"""
    def __init__(self, token_idx: int = 0):
        super().__init__()
        self.token_idx = token_idx
    
    def forward(self, X: torch.Tensor):
        # X: (B, T, C)
        return X[:, self.token_idx, :]


class TokenRangeFlatten(TokenSelector):
    """Select a range of tokens and flatten them"""
    def __init__(self, start_idx: int, end_idx: int):
        super().__init__()
        self.start_idx = start_idx
        self.end_idx = end_idx
    
    def forward(self, X: torch.Tensor):
        # X: (B, T, C)
        return X[:, self.start_idx:self.end_idx, :].flatten(1)


class TokenMeanPooling(TokenSelector):
    """Apply mean pooling to a range of tokens"""
    def __init__(self, start_idx: int = 0, end_idx: int = None):
        super().__init__()
        self.start_idx = start_idx
        self.end_idx = end_idx
    
    def forward(self, X: torch.Tensor):
        # X: (B, T, C)
        if self.end_idx is None:
            return X[:, self.start_idx:, :].mean(dim=1)
        else:
            return X[:, self.start_idx:self.end_idx, :].mean(dim=1)


class TokenMaxPooling(TokenSelector):
    """Apply max pooling to a range of tokens"""
    def __init__(self, start_idx: int = 0, end_idx: int = None):
        super().__init__()
        self.start_idx = start_idx
        self.end_idx = end_idx
    
    def forward(self, X: torch.Tensor):
        # X: (B, T, C)
        if self.end_idx is None:
            return X[:, self.start_idx:, :].max(dim=1)[0]
        else:
            return X[:, self.start_idx:self.end_idx, :].max(dim=1)[0]


class TokenConcatenator(TokenSelector):
    """Concatenate outputs from multiple token selectors"""
    def __init__(self, selectors):
        super().__init__()
        self.selectors = nn.ModuleList(selectors)
    
    def forward(self, X: torch.Tensor):
        # X: (B, T, C)
        return torch.cat([selector(X) for selector in self.selectors], dim=1)


# Legacy classes for backward compatibility
class SummaryFlatten(TokenRangeFlatten):
    def __init__(self, summary_idxs: torch.LongTensor):
        super().__init__(0, 0)  # Placeholder, will be overridden
        # register_idxs as a buffer so it moves with .to(device), .eval(), etc.
        self.register_buffer('summary_idxs', summary_idxs)
    
    def forward(self, X: torch.Tensor):
        # X: (B, T, C)
        return X[:, self.summary_idxs, :].flatten(1)


class ExtraClsFlatten(TokenRangeFlatten):
    def __init__(self, max_summary_idx: int, num_cls_tokens: int):
        super().__init__(max_summary_idx, num_cls_tokens)


class ClsWithoutSummaryFlatten(TokenRangeFlatten):
    def __init__(self, max_summary_idx: int, num_skip: int):
        super().__init__(max_summary_idx, num_skip)


class MaxPoolSpaceToken(TokenMaxPooling):
    def __init__(self, num_skip: int):
        super().__init__(start_idx=num_skip)


class AvgPoolSpaceToken(TokenMeanPooling):
    def __init__(self, num_skip: int):
        super().__init__(start_idx=num_skip)


# Model-specific transform dictionaries
def get_radio_transforms(model):
    # consider if we need to use the model argument or we can hard code the values
    summary_idxs = model.summary_idxs.cpu()  # torch.tensor([0, 1, 2], dtype=torch.long)
    max_summary_idx = summary_idxs.max() + 1  # 3
    num_skip = model.model.patch_generator.num_skip  # 8 
    num_cls_tokens = model.model.patch_generator.num_cls_tokens  # 4
    
    return {
        "summary_token_flatten": SummaryFlatten(summary_idxs),
        "extra_cls_token_flatten": TokenRangeFlatten(max_summary_idx, num_cls_tokens),
        "cls_without_summary_token_flatten": TokenRangeFlatten(max_summary_idx, num_skip),
        "maxpool_space_token": MaxPoolSpaceToken(num_skip),
        "avgpool_space_token": AvgPoolSpaceToken(num_skip),
    }


def get_siglip_transforms(model):
    return {
        "maxpool_space_token": MaxPoolSpaceToken(0),
        "avgpool_space_token": AvgPoolSpaceToken(0),
        "full": nn.Identity(),
    }


def get_dino_transforms(model):
    return {
        "cls_token": SingleTokenSelector(0),
        "mean_register_token": TokenMeanPooling(1, 5),
        "maxpool_space_token": MaxPoolSpaceToken(num_skip=5),
        "avgpool_space_token": AvgPoolSpaceToken(num_skip=5),
        "cls_cat_maxpool_space_token": TokenConcatenator([SingleTokenSelector(0), 
                                                          MaxPoolSpaceToken(num_skip=5)]),
    }


def get_clipag_transforms(model):
    return {
        # "pca750": PCA(n_components=750),
        "cls_token": SingleTokenSelector(0),
        "maxpool_space_token": MaxPoolSpaceToken(num_skip=1),
        "avgpool_space_token": AvgPoolSpaceToken(num_skip=1),
    }


# def get_resnet_transforms():
#     raise NotImplementedError("ResNet transforms are implemented, but we should use the more efficient method in regress_lib.py")
#     return {
#         "pca750": PCA(n_components=750),
#         "srp": SparseRandomProjection(),
#     }

FEATURE_REDUCTION_DEFAULTS = {
    "siglip2_vitb16":      get_siglip_transforms,
    "dinov2_vitb14_reg":   get_dino_transforms,
    "radio_v2.5-b":        get_radio_transforms,
    "clipag_vitb32":       get_clipag_transforms,
    # all the ResNet-50 variants use the same Bottleneck block
    # "resnet50_clip":       get_resnet_transforms,
    # "resnet50_dino":       get_resnet_transforms,
    # "resnet50_robust":     get_resnet_transforms,
    # "resnet50":            get_resnet_transforms,
    # AlexNet
    # ReAlnet
    # "ReAlnet01":           get_alexnet_transforms,
    # "AlexNet_training_seed_01": get_alexnet_transforms,
    # "regnety_640": get_regnety_transforms,
}


LAYER_TRANSFORM_FILTERS = {
    "siglip2_vitb16":      lambda layer, dimred_str: not (("attn_pool" in layer) != ("full" in dimred_str)), 
    # this is a special case for siglip2_vitb16, XOR, only apply FULL to attn_pool layers, and other transforms to other layers
    "dinov2_vitb14_reg":   lambda layer, dimred_str: True,
    "radio_v2.5-b":        lambda layer, dimred_str: True,
    "clipag_vitb32":       lambda layer, dimred_str: True,
}
