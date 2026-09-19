#%%
import os
import sys
import yaml
import re
import glob
import timm
import torch
import torch as th
import torch.nn as nn
from tqdm.auto import tqdm
from os.path import join
import pickle as pkl
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.models import resnet50
from torchvision.models.feature_extraction import create_feature_extractor
import torchvision.transforms as T
from torchvision.transforms import ToPILImage, ToTensor, Normalize, Resize
# from circuit_toolkit.CNN_scorers import TorchScorer  # PNC: unused in this script; not vendored in core/
# from circuit_toolkit.GAN_utils import upconvGAN, Caffenet  # PNC: unused in this script; not vendored in core/
from core.plot_utils import to_imgrid, show_imgrid, save_imgrid, saveallforms
from core.layer_hook_utils import featureFetcher_module, featureFetcher, get_module_names
from core.dataset_utils import ImagePathDataset, DataLoader
from neural_regress.regress_lib import sweep_regressors, perform_regression_sweeplayer_RidgeCV, perform_regression_sweeplayer, record_features
from neural_regress.sklearn_torchify_lib import SRP_torch, PCA_torch, LinearRegression_torch, SpatialAvg_torch, LinearLayer_from_sklearn
from core.data_utils import load_neural_data, load_from_hdf5, load_neural_trial_resp_tensor, create_response_tensor, parse_image_fullpaths
from core.model_load_utils import load_model_transform
from horama import maco, plot_maco
from sklearn.metrics.pairwise import cosine_similarity
from core.posthoc_prediction_utils import find_sort_png_files_by_subfolder, parse_accentuated_filename, parse_accentuated_filenames_to_df 
from core.posthoc_prediction_utils import get_PCA_basis_predictor_from_config, get_predictor_from_config, get_unit_population_PCA_basis_predictor_from_config, get_prediction_responses

acc_stim_root = os.environ.get("PNC_ACCENT_OUTPUT_ROOT", "/n/holylabs/LABS/alvarez_lab/Everyone/Accentuate_VVS/accentuation_outputs")  # was r"/n/holylabs/LABS/alvarez_lab/Everyone/Accentuate_VVS/accentuation_outputs"
config_root = os.environ.get("PNC_ACCENT_CONFIG_ROOT", "/n/holylabs/LABS/alvarez_lab/Everyone/Accentuate_VVS/accentuation_configs")  # was r"/n/holylabs/LABS/alvarez_lab/Everyone/Accentuate_VVS/accentuation_configs"
subject_id = "red_20250428-20250430"
    
for subject_id, filename in [
    ("red_20250428-20250430", "red_20250428-20250430_vvs-encodingstimuli_z1_rw100-400.h5"), 
    ("paul_20250428-20250430", "paul_20250428-20250430_vvs-encodingstimuli_z1_rw100-400.h5"), 
    ("venus_250426-250429", "venus_250426-250429_vvs-encodingstimuli_z1_rw80-250.h5"),
    ("three0_250426-250501", "three0_250426-250501_vvs-encodingstimuli_z1_rw80-250.h5"),
    ("leap_250426-250501", "leap_250426-250501_vvs-encodingstimuli_z1_rw80-250.h5"),
]:
    posthoc_PCA_dir = join(os.environ.get("PNC_ENCODING_MODEL_ROOT", "/n/holylabs/LABS/alvarez_lab/Lab/VVS_Accentuation/Encoding_models"), subject_id, "posthoc_model_predict_PCA_popul_unit")  # was f"/n/holylabs/LABS/alvarez_lab/Lab/VVS_Accentuation/Encoding_models/{subject_id}/posthoc_model_predict_PCA_popul_unit"
    os.makedirs(posthoc_PCA_dir, exist_ok=True)
    # config_files
    target_subfolder = glob.glob(join(acc_stim_root, f"*{subject_id}*_accentuation"))
    all_png_files = find_sort_png_files_by_subfolder(target_subfolder)
    png_files_list = sum(all_png_files.values(), [])
    df_accentuated = parse_accentuated_filenames_to_df(png_files_list)
    df_accentuated.to_pickle(join(posthoc_PCA_dir, f"df_accentuated_{subject_id}.pkl"))
    config_dir = join(config_root, subject_id)
    config_files = sorted(glob.glob(join(config_dir, "*.yaml")))
    # chan_pattern = "_Ch19_"
    # config_pre_chan = [f for f in config_files if chan_pattern in f]
    # cosine_results = []
    for config_file in config_files:
        config_acc = yaml.load(open(config_file, "r"), Loader=yaml.FullLoader)
        model_name = config_acc["model_name"]
        unit_id = config_acc["unit_ids"][0]
        layer_name = config_acc["layer_name"]
        print(f"Subject: {subject_id}, Unit: {unit_id}, Model: {model_name}, Layer: {layer_name}")
        acc_split_df = df_accentuated.copy()
        # acc_split_df = df_accentuated.query("model_name == @model_name and unit_id == @unit_id")
        # assert len (acc_split_df) == 110 

        predict_PCA_feature, predict_population_response, predict_target_unit_response, \
            model, transforms_pipeline, fetcher, Xtransform, readout \
                = get_unit_population_PCA_basis_predictor_from_config(config_file, device="cuda")
        # accentuated_dataset = ImagePathDataset(acc_split_df["filepath"], transform=transforms_pipeline)
        # accentuated_dataloader = DataLoader(accentuated_dataset, batch_size=60, shuffle=False, num_workers=10)
        acc_img_PCA_resp = get_prediction_responses(predict_PCA_feature, transforms_pipeline, 
                                            acc_split_df["filepath"].tolist(), batch_size=120, num_workers=16)
        acc_img_population_resp = get_prediction_responses(predict_population_response, transforms_pipeline, 
                                            acc_split_df["filepath"].tolist(), batch_size=120, num_workers=16)
        acc_model_target_unit_resp = get_prediction_responses(predict_target_unit_response, transforms_pipeline, 
                                            acc_split_df["filepath"].tolist(), batch_size=120, num_workers=16)
        readout_vec = readout.weight.data[unit_id, :].cpu()
        readout_bias = readout.bias.data[unit_id].cpu()
        PCA_norm = acc_img_PCA_resp.norm(dim=1)
        cosine_sims = cosine_similarity(acc_img_PCA_resp.cpu().numpy(), readout_vec.numpy().reshape(1, -1))
        cosine_sims = cosine_sims.flatten() 
        acc_split_df_cos = acc_split_df.copy()
        acc_split_df_cos["cosine_similarity"] = cosine_sims
        acc_split_df_cos["PCA_norm"] = PCA_norm
        
        print(f"Min cosine similarity: {cosine_sims.min():.4f}")
        print(f"Max cosine similarity: {cosine_sims.max():.4f}")
        print(f"PCA norm max {PCA_norm.max():.4f}")
        print(f"PCA norm min {PCA_norm.min():.4f}")
        
        stats_dict = {"config": config_acc, 
                            "df": acc_split_df_cos.copy(), 
                            "PCA_resp": acc_img_PCA_resp, 
                            "population_resp": acc_img_population_resp,
                            "target_unit_resp": acc_model_target_unit_resp,
                            "readout_vec": readout_vec, 
                            "readout_bias": readout_bias,
                            "cosine_sims": cosine_sims, 
                            "PCA_norm": PCA_norm}
        pkl.dump(stats_dict, open(join(posthoc_PCA_dir, f"posthoc_prediction_PCA_pop_unit_{subject_id}_unit{unit_id}_{model_name}.pkl"), "wb"))
        print(f"Saved to {join(posthoc_PCA_dir, f'posthoc_prediction_PCA_pop_unit_{subject_id}_unit{unit_id}_{model_name}.pkl')} completed!")

        
        


# %%
