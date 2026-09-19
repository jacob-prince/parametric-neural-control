# %%

# %%
import os
import sys
import time
import re
import yaml
import glob
import timm
import torch
import torch as th
import torch.nn as nn
from tqdm.auto import tqdm
from os.path import join
import pickle as pkl
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from horama import maco, plot_maco
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
from neural_regress.posthoc_analysis_lib import parse_accentuated_filenames_to_df, \
    compute_prediction_responses, construct_predictor_from_config
# %% [markdown]
# ### Loading at scale

# %% [markdown]
# Check that all predictors can be loaded! 
config_root = os.environ.get("PNC_ACCENT_CONFIG_ROOT", "/n/holylabs/LABS/alvarez_lab/Everyone/Accentuate_VVS/accentuation_configs")  # was r"/n/holylabs/LABS/alvarez_lab/Everyone/Accentuate_VVS/accentuation_configs"
acc_stim_root = os.environ.get("PNC_ACCENT_OUTPUT_ROOT", "/n/holylabs/LABS/alvarez_lab/Everyone/Accentuate_VVS/accentuation_outputs")  # was r"/n/holylabs/LABS/alvarez_lab/Everyone/Accentuate_VVS/accentuation_outputs"

subject_ids = ["red_20250428-20250430",  "paul_20250428-20250430", "leap_250426-250501", "three0_250426-250501", "venus_250426-250429"]
for subject_id in subject_ids:
    posthoc_dir = join(os.environ.get("PNC_ENCODING_MODEL_ROOT", "/n/holylabs/LABS/alvarez_lab/Lab/VVS_Accentuation/Encoding_models"), subject_id, "posthoc_model_predict")  # was f"/n/holylabs/LABS/alvarez_lab/Lab/VVS_Accentuation/Encoding_models/{subject_id}/posthoc_model_predict"
    os.makedirs(posthoc_dir, exist_ok=True)
    # %% [markdown]
    # ### Construct the stimuli set
    encoding_stim_dir = os.environ.get("PNC_ENCODING_STIM_DIR", "/n/holylabs/LABS/alvarez_lab/Lab/VVS_Accentuation/Stimuli/encodingstimuli_apr2025")  # was r"/n/holylabs/LABS/alvarez_lab/Lab/VVS_Accentuation/Stimuli/encodingstimuli_apr2025"
    df_encoding = pd.read_csv(join(encoding_stim_dir, "encoding_stimuli_split_seed0_swapped.csv"))
    df_encoding['image_fps'] = parse_image_fullpaths(df_encoding["stimulus_name"], [encoding_stim_dir], arbitrary_format=True)
    # target_subfolder = sorted(glob.glob(join(acc_stim_root, f"*{subject_id}*_accentuation")))
    # # List all png files in each subfolder (one level deep)
    # all_png_files = {}
    # for subfolder in target_subfolder:
    #     subfolder_name = os.path.basename(subfolder)
    #     png_files = glob.glob(join(subfolder, "*.png"))
    #     # Sort the files to ensure consistent order
    #     png_files.sort()
    #     all_png_files[subfolder_name] = png_files
    #     print(f"Found {len(png_files)} PNG files in {subfolder_name}")
    #     # Print first few files as example
    #     for file in png_files[:5]:
    #         print(f"  - {os.path.basename(file)}")
    #     if len(png_files) > 5:
    #         print(f"  - ... and {len(png_files)-5} more files")
    #     print()

    # # Test the function on the file list
    # # Parse all files into a DataFrame
    # png_files_list = sum(all_png_files.values(), [])
    # df_accentuated = parse_accentuated_filenames_to_df(png_files_list)
    # Display info about the DataFrame
    print(f"DataFrame shape: {df_encoding.shape}")
    print("\nDataFrame columns:")
    print(df_encoding.columns.tolist())
    # Show unique models and units
    # print(f"\nUnique models: {df_accentuated['model_name'].nunique()}")
    # print(f"Unique units: {df_accentuated['unit_id'].nunique()}")
    # print(f"Unique image IDs: {df_accentuated['img_id'].nunique()}")
    # print(f"Unique levels: {df_accentuated['level'].nunique()}")
    # print(f"Unique scores: {df_accentuated['score'].nunique()}")

    # df_accentuated.to_pickle(join(posthoc_dir, f"accentuated_stim_info_{subject_id}.pkl"))
    # df_accentuated.to_csv(join(posthoc_dir, f"accentuated_stim_info_{subject_id}.csv"))

    # %% [markdown]
    # ### Mass produce cross prediction 

    # %%
    config_dir = join(config_root, subject_id)
    config_files = sorted(glob.glob(join(config_dir, "*.yaml")))
    # chan_pattern = "_Ch19_"
    # config_pre_chan = [f for f in config_files if chan_pattern in f]
    t0 = time.time()
    pred_resp_dict = {}
    df_encoding_w_pred_resp = df_encoding.copy()
    for config_file in config_files:
        acc_config = yaml.safe_load(open(config_file))
        unit_ids = acc_config['unit_ids']
        assert len(unit_ids) == 1, "Only one unit is supported for now"
        unit_ids = unit_ids[0]
        model_name = acc_config['model_name']
        population_predictor, target_unit_predictor, \
            model, transforms_pipeline, _, _, _ \
                = construct_predictor_from_config(config_file, device="cuda")
        acc_img_chan_resp = compute_prediction_responses(target_unit_predictor, transforms_pipeline, 
                                                df_encoding["image_fps"])
        pred_resp_dict[(model_name, unit_ids)] = acc_img_chan_resp.cpu().numpy()
        df_encoding_w_pred_resp[f"pred_resp_{model_name}_unit_{unit_ids}"] = acc_img_chan_resp.cpu().numpy()

    t1 = time.time()
    print(f"Time taken for all predictors: {t1 - t0} seconds")
    df_encoding_w_pred_resp.to_pickle(join(posthoc_dir, f"encoding_stim_info_w_pred_resp_{subject_id}.pkl"))
    pkl.dump(pred_resp_dict, open(join(posthoc_dir, f"encoding_stim_pred_resp_dict_{subject_id}.pkl"), "wb"))
    print(f"Saved to {posthoc_dir} complete!")


