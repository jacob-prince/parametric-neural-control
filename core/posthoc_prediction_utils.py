import os
import sys
import re
import glob
import pickle as pkl
import yaml
from os.path import join
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

import torch
import torch as th
import torch.nn as nn
import timm
from torch.utils.data import DataLoader
from circuit_toolkit.layer_hook_utils import featureFetcher_module, featureFetcher, get_module_names
from circuit_toolkit.dataset_utils import ImagePathDataset
sys.path.append("/n/home12/binxuwang/Github/Closed-loop-visual-insilico")
from core.model_load_utils import load_model_transform


def check_gradient(objective_fn):
    """Check if gradients can flow through the objective function."""
    img_opt = th.randn(1, 3, 224, 224).cuda()
    img_opt.requires_grad_(True)
    resp = objective_fn(img_opt)
    resp.mean().backward()
    print(resp.shape)
    assert img_opt.grad is not None


def get_predictor_from_config(acc_config, device="cuda"):
    """Create a function that predicts neural population responses for images.
    
    Args:
        subject_id (str): ID of the subject
        modelname (str): Name of the model to use (e.g. "resnet50_robust") 
        layer_name (str): Name of layer to extract features from
        device (str): Device to run model on ("cuda" or "cpu")
        
    Returns:
        function: A function that takes images as input and returns predicted population responses
    """
    if isinstance(acc_config, str):
        acc_config = yaml.safe_load(open(acc_config))
        
    # Construct paths
    xtransform_path = acc_config['xtransform_path']
    meta_path = acc_config['meta_path']
    readout_path = acc_config['readout_path']
    model_name = acc_config['model_name']
    unit_ids = acc_config['unit_ids']
    layer_name = acc_config['layer_name']
    regressor = acc_config['fit_method_name']
    print(f"Loading model and set up feature extraction: model {model_name}, layer {layer_name}")
    print(f"Target unit ids: {unit_ids}")
    print(f"Loading readout layer and PCA transform: readout {readout_path}, xtransform {xtransform_path}")
    # Load model and set up feature extraction
    model, transforms_pipeline = load_model_transform(model_name, device=device)
    model = model.eval().to(device)
    model.requires_grad_(False)
    fetcher = featureFetcher(model, input_size=(3, 224, 224), print_module=False)
    fetcher.record(layer_name, ingraph=True, store_device=device)
    # Load readout layer and PCA transform
    readout = th.load(readout_path, weights_only=False).to(device)
    Xtransform = th.load(xtransform_path, weights_only=False).to(device)
    def predict_population_response(images):
        """Predict neural population responses for input images.
        
        Args:
            images (torch.Tensor): Input images of shape (batch_size, 3, 224, 224)
            
        Returns:
            torch.Tensor: Predicted neural responses
        """
        model(images)
        feat_tsr = fetcher[layer_name]
        feat_vec = Xtransform(feat_tsr)
        return readout(feat_vec)
    
    
    def predict_target_unit_response(images):
        """Predict neural population responses for input images.
        
        Args:
            images (torch.Tensor): Input images of shape (batch_size, 3, 224, 224)
            
        Returns:
            torch.Tensor: Predicted neural responses
        """
        model(images)
        feat_tsr = fetcher[layer_name]
        feat_vec = Xtransform(feat_tsr)
        return readout(feat_vec)[:, unit_ids]

    check_gradient(predict_population_response)
    print("Gradient check passed!")
    return predict_population_response, predict_target_unit_response, \
           model, transforms_pipeline, fetcher, Xtransform, readout
           

def get_unit_population_PCA_basis_predictor_from_config(acc_config, device="cuda"):
    """Create a function that predicts neural population responses for images.
    
    Args:
        subject_id (str): ID of the subject
        modelname (str): Name of the model to use (e.g. "resnet50_robust") 
        layer_name (str): Name of layer to extract features from
        device (str): Device to run model on ("cuda" or "cpu")
        
    Returns:
        function: A function that takes images as input and returns predicted population responses
    """
    if isinstance(acc_config, str):
        acc_config = yaml.safe_load(open(acc_config))
        
    # Construct paths
    xtransform_path = acc_config['xtransform_path']
    meta_path = acc_config['meta_path']
    readout_path = acc_config['readout_path']
    model_name = acc_config['model_name']
    unit_ids = acc_config['unit_ids']
    layer_name = acc_config['layer_name']
    regressor = acc_config['fit_method_name']
    print(f"Loading model and set up feature extraction: model {model_name}, layer {layer_name}")
    print(f"Target unit ids: {unit_ids}")
    print(f"Loading readout layer and PCA transform: readout {readout_path}, xtransform {xtransform_path}")
    # Load model and set up feature extraction
    model, transforms_pipeline = load_model_transform(model_name, device=device)
    model = model.eval().to(device)
    model.requires_grad_(False)
    fetcher = featureFetcher(model, input_size=(3, 224, 224), print_module=False)
    fetcher.record(layer_name, ingraph=True, store_device=device)
    # Load readout layer and PCA transform
    readout = th.load(readout_path, weights_only=False).to(device)
    Xtransform = th.load(xtransform_path, weights_only=False).to(device)
    def predict_population_response(images):
        """Predict neural population responses for input images.
        
        Args:
            images (torch.Tensor): Input images of shape (batch_size, 3, 224, 224)
            
        Returns:
            torch.Tensor: Predicted neural responses
        """
        model(images)
        feat_tsr = fetcher[layer_name]
        feat_vec = Xtransform(feat_tsr)
        return readout(feat_vec)
    
    
    def predict_target_unit_response(images):
        """Predict neural population responses for input images.
        
        Args:
            images (torch.Tensor): Input images of shape (batch_size, 3, 224, 224)
            
        Returns:
            torch.Tensor: Predicted neural responses
        """
        model(images)
        feat_tsr = fetcher[layer_name]
        feat_vec = Xtransform(feat_tsr)
        return readout(feat_vec)[:, unit_ids]


    def predict_reduced_feature(images):
        """Predict PCA features responses for input images.
        
        Args:
            images (torch.Tensor): Input images of shape (batch_size, 3, 224, 224)
            
        Returns:
            torch.Tensor: Predicted PCA features (no readout)
        """
        model(images)
        feat_tsr = fetcher[layer_name]
        feat_vec = Xtransform(feat_tsr)
        return feat_vec
    
    check_gradient(predict_reduced_feature)
    print("Gradient check passed!")
    return predict_reduced_feature, predict_population_response, predict_target_unit_response, \
           model, transforms_pipeline, fetcher, Xtransform, readout


def get_PCA_basis_predictor_from_config(acc_config, device="cuda"):
    """Create a function that predicts neural population responses for images.
    
    Args:
        subject_id (str): ID of the subject
        modelname (str): Name of the model to use (e.g. "resnet50_robust") 
        layer_name (str): Name of layer to extract features from
        device (str): Device to run model on ("cuda" or "cpu")
        
    Returns:
        function: A function that takes images as input and returns predicted population responses
    """
    if isinstance(acc_config, str):
        acc_config = yaml.safe_load(open(acc_config))
        
    # Construct paths
    xtransform_path = acc_config['xtransform_path']
    meta_path = acc_config['meta_path']
    readout_path = acc_config['readout_path']
    model_name = acc_config['model_name']
    unit_ids = acc_config['unit_ids']
    layer_name = acc_config['layer_name']
    regressor = acc_config['fit_method_name']
    print(f"Loading model and set up feature extraction: model {model_name}, layer {layer_name}")
    print(f"Target unit ids: {unit_ids}")
    print(f"Loading readout layer and PCA transform: readout {readout_path}, xtransform {xtransform_path}")
    # Load model and set up feature extraction
    model, transforms_pipeline = load_model_transform(model_name, device=device)
    model = model.eval().to(device)
    model.requires_grad_(False)
    fetcher = featureFetcher(model, input_size=(3, 224, 224), print_module=False)
    fetcher.record(layer_name, ingraph=True, store_device=device)
    # Load readout layer and PCA transform
    readout = th.load(readout_path, weights_only=False).to(device)
    Xtransform = th.load(xtransform_path, weights_only=False).to(device)
    def predict_reduced_feature(images):
        """Predict neural population responses for input images.
        
        Args:
            images (torch.Tensor): Input images of shape (batch_size, 3, 224, 224)
            
        Returns:
            torch.Tensor: Predicted neural responses
        """
        model(images)
        feat_tsr = fetcher[layer_name]
        feat_vec = Xtransform(feat_tsr)
        return feat_vec
    
    check_gradient(predict_reduced_feature)
    print("Gradient check passed!")
    return predict_reduced_feature, \
           model, transforms_pipeline, fetcher, Xtransform, readout
           

def get_prediction_responses(pred_fn, transforms_pipeline, image_fps, device="cuda", batch_size=100, num_workers=16):
    """
    Get the prediction responses for a set of images specified by image_fps (list of image file paths)
    
    Args:
        pred_fn (function): A function that takes images as input and returns predicted responses
                    `predict_reduced_feature`, `predict_population_response`, `predict_target_unit_response` are all valid functions. 
        transforms_pipeline (torchvision.transforms.Compose): A pipeline of transformations to apply to the images
        image_fps (list): List of image file paths
        device (str): Device to run model on ("cuda" or "cpu")
    """
    dataset = ImagePathDataset(image_fps, transform=transforms_pipeline)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    pred_resp = []
    for batch in tqdm(dataloader):
        batch_img = batch[0].to(device)
        with th.no_grad():
            batch_resp = pred_fn(batch_img).cpu()
        pred_resp.append(batch_resp)
    return torch.cat(pred_resp, dim=0)


def find_sort_png_files_by_subfolder(target_subfolder, suffix="png"):
    """
    List all png files in each subfolder (one level deep)
    """
    all_png_files = {}
    for subfolder in target_subfolder:
        subfolder_name = os.path.basename(subfolder)
        png_files = glob.glob(join(subfolder, f"*.{suffix}"))
        # Sort the files to ensure consistent order
        png_files.sort()
        all_png_files[subfolder_name] = png_files
        print(f"Found {len(png_files)} {suffix} files in {subfolder_name}")
        # Print first few files as example
        for file in png_files[:5]:
            print(f"  - {os.path.basename(file)}")
        if len(png_files) > 5:
            print(f"  - ... and {len(png_files)-5} more files")
        print()
    return all_png_files



def parse_accentuated_filename(filename):
    """
    Parse accentuated image filenames to extract metadata.
    
    Example filename format:
    radio_v2.5-b_RidgeCV_unit_0_img_0_level_0.580872533679007_score_0.5726814270019531.png
    
    Returns:
        dict: Dictionary containing extracted metadata (model_name, unit_id, img_id, level, score)
    """
    # Get just the filename without path
    basename = os.path.basename(filename)
    # Extract components using regex
    pattern = r'(.+?)_RidgeCV_unit_(\d+)_img_(\d+)_level_([-\d\.]+)_score_([-\d\.]+)\.png'
    match = re.search(pattern, basename)
    if match:
        model_name, unit_id, img_id, level, score = match.groups()
        return {
            'model_name': model_name,
            'unit_id': int(unit_id),
            'img_id': int(img_id),
            'level': float(level),
            'score': float(score),
            'filepath': filename
        }
    else:
        return None

def parse_accentuated_filenames_to_df(filenames, include_parse_fail=False):
    """
    Parse a list of accentuated image filenames and return as a DataFrame.
    
    Args:
        filenames (list): List of filenames to parse
        
    Returns:
        pd.DataFrame: DataFrame containing parsed metadata for all files
    """
    parsed_data = []
    for filename in filenames:
        parsed = parse_accentuated_filename(filename)
        if parsed:
            parsed_data.append(parsed)
        else:
            # this is critical since some files are not accentuated, if we don't include them the index will not match the order of response! @binxu Oct. 16th
            if include_parse_fail:
                parsed_data.append({
                    'filepath': filename,
                    'model_name': pd.NA,
                    'unit_id': pd.NA,
                    'img_id': pd.NA,
                    'level': pd.NA,
                    'score': pd.NA,
                })
    
    if parsed_data:
        return pd.DataFrame(parsed_data)
    else:
        return pd.DataFrame()

