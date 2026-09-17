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
from core.posthoc_prediction_utils import get_PCA_basis_predictor_from_config, get_predictor_from_config, get_unit_population_PCA_basis_predictor_from_config, get_prediction_responses
# %% [markdown]
# ### Loading at scale

# %% [markdown]
# Check that all predictors can be loaded! 
config_root = os.environ.get("PNC_ACCENT_CONFIG_ROOT", "/n/holylabs/LABS/alvarez_lab/Everyone/Accentuate_VVS/accentuation_configs")  # was r"/n/holylabs/LABS/alvarez_lab/Everyone/Accentuate_VVS/accentuation_configs"
acc_stim_root = os.environ.get("PNC_ACCENT_OUTPUT_ROOT", "/n/holylabs/LABS/alvarez_lab/Everyone/Accentuate_VVS/accentuation_outputs")  # was r"/n/holylabs/LABS/alvarez_lab/Everyone/Accentuate_VVS/accentuation_outputs"

subject_ids = ["red_20250428-20250430",  
               "paul_20250428-20250430", 
               "leap_250426-250501", 
               "three0_250426-250501", 
               "venus_250426-250429",
               "red_250426-250501"]

for subject_id in subject_ids:
    posthoc_PCA_dir = join(os.environ.get("PNC_ENCODING_MODEL_ROOT", "/n/holylabs/LABS/alvarez_lab/Lab/VVS_Accentuation/Encoding_models"), subject_id, "posthoc_model_predict_PCA_popul_unit")  # was f"/n/holylabs/LABS/alvarez_lab/Lab/VVS_Accentuation/Encoding_models/{subject_id}/posthoc_model_predict_PCA_popul_unit"
    os.makedirs(posthoc_PCA_dir, exist_ok=True)
    # %% [markdown]
    # ### Construct the stimuli set
    encoding_stim_dir = os.environ.get("PNC_ENCODING_STIM_DIR", "/n/holylabs/LABS/alvarez_lab/Lab/VVS_Accentuation/Stimuli/encodingstimuli_apr2025")  # was r"/n/holylabs/LABS/alvarez_lab/Lab/VVS_Accentuation/Stimuli/encodingstimuli_apr2025"
    df_encoding = pd.read_csv(join(encoding_stim_dir, "encoding_stimuli_split_seed0_swapped.csv"))
    df_encoding['image_fps'] = parse_image_fullpaths(df_encoding["stimulus_name"], [encoding_stim_dir], arbitrary_format=True)
    # Display info about the DataFrame
    print(f"DataFrame shape: {df_encoding.shape}")
    print("\nDataFrame columns:")
    print(df_encoding.columns.tolist())
    # %% [markdown]
    # ### Mass produce cross prediction 
    config_dir = join(config_root, subject_id)
    config_files = sorted(glob.glob(join(config_dir, "*.yaml")))
    # chan_pattern = "_Ch19_"
    # config_pre_chan = [f for f in config_files if chan_pattern in f]
    t0 = time.time()
    pred_resp_dict = {}
    for config_file in config_files:
        acc_config = yaml.safe_load(open(config_file))
        unit_ids = acc_config['unit_ids']
        assert len(unit_ids) == 1, "Only one unit is supported for now"
        unit_id = unit_ids[0]
        model_name = acc_config['model_name']
        predict_PCA_feature, predict_population_response, predict_target_unit_response, \
            model, transforms_pipeline, fetcher, Xtransform, readout \
                = get_unit_population_PCA_basis_predictor_from_config(config_file, device="cuda")
        enc_img_chan_resp = compute_prediction_responses(predict_target_unit_response, transforms_pipeline, 
                                                df_encoding["image_fps"])
        enc_img_population_resp = compute_prediction_responses(predict_population_response, transforms_pipeline, 
                                                df_encoding["image_fps"])
        enc_img_PCA_resp = compute_prediction_responses(predict_PCA_feature, transforms_pipeline, 
                                                df_encoding["image_fps"])
        print(f"enc_img_PCA_resp shape: {enc_img_PCA_resp.shape}")
        print(f"enc_img_population_resp shape: {enc_img_population_resp.shape}")
        print(f"enc_img_chan_resp shape: {enc_img_chan_resp.shape}")
        readout_vec = readout.weight.data[unit_id, :].cpu()
        readout_bias = readout.bias.data[unit_id].cpu()
        print(f"readout_vec shape: {readout_vec.shape}")
        print(f"readout_bias shape: {readout_bias.shape}")
        stats_dict = {"config": acc_config, 
                    "df": df_encoding.copy(), 
                    "PCA_resp": enc_img_PCA_resp, 
                    "population_resp": enc_img_population_resp,
                    "target_unit_resp": enc_img_chan_resp,
                    "readout_vec": readout_vec, 
                    "readout_bias": readout_bias,
                    }
                    # "cosine_sims": cosine_sims, 
                    # "PCA_norm": PCA_norm}
        pkl.dump(stats_dict, open(join(posthoc_PCA_dir, f"posthoc_prediction_NSDencimg_PCA_pop_unit_{subject_id}_unit{unit_id}_{model_name}.pkl"), "wb"))
        print(f"Saved to {join(posthoc_PCA_dir, f'posthoc_prediction_NSDencimg_PCA_pop_unit_{subject_id}_unit{unit_id}_{model_name}.pkl')} completed!")
        t1 = time.time()
        print(f"Time taken for {model_name} {unit_id} predictor: {t1 - t0} seconds")
    t1 = time.time()
    print(f"Time taken for all predictors: {t1 - t0} seconds")
    print(f"Saved to {posthoc_PCA_dir} complete!")


