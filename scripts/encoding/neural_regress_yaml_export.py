# %%

# %% [markdown]
# ### Model export to Thomas
import os
from os.path import join
import torch as th
import torch
from sklearn.decomposition import PCA
from sklearn.random_projection import SparseRandomProjection
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
import sys
from core.data_utils import load_from_hdf5, load_neural_data, load_neural_trial_resp_tensor, create_response_tensor, parse_image_fullpaths, extract_neural_data_dict_2025apr
from core.model_load_utils import load_model_transform, MODEL_LAYER_FILTERS, LAYER_ABBREVIATION_MAPS
from neural_regress.feature_reduction_lib import FEATURE_REDUCTION_DEFAULTS, LAYER_TRANSFORM_FILTERS
from neural_regress.regress_lib import record_features, perform_regression_sweeplayer, perform_regression_sweeplayer_RidgeCV
from neural_regress.regress_lib import sweep_regressors, transform_features2Xdict_new, transform_features2Xdict, RidgeCV, apply_feature_transforms
from neural_regress.regress_eval_lib import format_result_df, plot_result_df_per_layer, construct_result_df_masked, \
    compute_pred_dict_D2_per_unit, format_result_df_tuple_index
from neural_regress.sklearn_torchify_lib import SRP_torch, PCA_torch, LinearRegression_torch
from neural_regress.sklearn_torchify_lib import LinearLayer_from_sklearn, PCA_torch, SRP_torch


# %%
import yaml
import datetime
def create_accentuation_config(
    subject_id, 
    modelname, 
    unit_id, 
    regressor_name, 
    layername, 
    exportdir, 
    export_readout_path, 
    export_Xtransform_JIT_path, 
    export_meta_path, 
    date=None,
    template_path=None, 
):
    """
    Create an accentuation configuration by filling in a template with specific values.
    
    Args:
        template_path: Path to the YAML template file
        subject_id: Subject ID for the experiment
        modelname: Name of the model
        unit_id: ID of the unit to process
        regressor_name: Name of the regression method
        layername: Name of the layer
        exportdir: Directory to export results
        export_readout_path: Path to the exported readout
        export_Xtransform_JIT_path: Path to the exported transform JIT script
        export_meta_path: Path to the exported metadata
        
    Returns:
        dict: Loaded YAML configuration
    """
    import datetime
    import yaml
    if template_path is None:
        template_path = os.environ.get("PNC_ACCENT_TEMPLATE_YAML", join(os.path.dirname(os.path.abspath(__file__)), "accentuation_template.yaml"))  # was "/n/home12/binxuwang/Github/Closed-loop-visual-insilico/notebooks/accentuation_template.yaml" (template now shipped next to this script)
    with open(template_path, "r") as f:
        content = f.read()
    
    content = content.replace("{{subject_id}}", subject_id)
    content = content.replace("{{model_name}}", modelname)
    content = content.replace('"{{unit_ids}}"', str([unit_id]))
    content = content.replace("{{fit_method_name}}", regressor_name)
    content = content.replace("{{layer_name}}", layername)
    content = content.replace("{{outputdir}}", exportdir)
    content = content.replace("{{readout_path}}", export_readout_path)
    content = content.replace("{{xtransform_path}}", export_Xtransform_JIT_path)
    content = content.replace("{{meta_path}}", export_meta_path)
    if date is None:
        content = content.replace("{{date}}", datetime.datetime.now().strftime("%d-%m-%Y"))
    else:
        content = content.replace("{{date}}", date)
    
    return content, yaml.safe_load(content)


def save_and_verify_config(content, output_yaml_path, config=None):
    """
    Save the configuration to a file and verify it loads correctly.
    
    Args:
        content: YAML content as string
        config: Configuration dictionary
        exportdir: Directory to export the configuration
        subject_id: Subject ID for the experiment
        modelname: Name of the model
        unit_id: ID of the unit to process
        
    Returns:
        str: Path to the saved configuration file
    """
    with open(output_yaml_path, "w") as f:
        f.write(content)
    print(f"Configuration saved to {output_yaml_path}")
    
    # Verify that it loads correctly
    try:
        loaded_config = yaml.safe_load(open(output_yaml_path, "r"))
        print("Configuration loaded successfully")
        # check it's the same as config
        if config is not None:
            assert loaded_config == config, "Loaded configuration differs from original config"
            print("Loaded configuration matches the original config")
    except Exception as e:
        print(f"Error loading configuration: {e}")
    
    return output_yaml_path


# %% [markdown]
# ### Export for additional day

model_root = os.environ.get("PNC_ENCODING_MODEL_ROOT", "/n/holylabs/LABS/alvarez_lab/Lab/VVS_Accentuation/Encoding_models")  # was "/n/holylabs/LABS/alvarez_lab/Lab/VVS_Accentuation/Encoding_models/"
ephys_data_root = os.environ.get("PNC_EPHYS_DATA_ROOT", "/n/holylabs/LABS/alvarez_lab/Lab/VVS_Accentuation/Ephys_Data")  # was "/n/holylabs/LABS/alvarez_lab/Lab/VVS_Accentuation/Ephys_Data"
# yaml_root = f"/n/holylabs/LABS/alvarez_lab/Lab/VVS_Accentuation/accentuation_configs"
# exportroot = f"/n/holylabs/LABS/alvarez_lab/Lab/VVS_Accentuation/Encoding_model_outputs"
export_root = os.environ.get("PNC_READOUT_EXPORT_ROOT", "/n/holylabs/LABS/alvarez_lab/Everyone/Accentuate_VVS/Encoding_model_outputs")  # was f"/n/holylabs/LABS/alvarez_lab/Everyone/Accentuate_VVS/Encoding_model_outputs"
yaml_root = os.environ.get("PNC_ACCENT_CONFIG_ROOT", "/n/holylabs/LABS/alvarez_lab/Everyone/Accentuate_VVS/accentuation_configs")  # was "/n/holylabs/LABS/alvarez_lab/Everyone/Accentuate_VVS/accentuation_configs"
# subject_id = "red_20250428-20240429"
# subject_id, h5_filename = ("red_20250428-20250430", "red_20250428-20250430_vvs-encodingstimuli_z1_rw100-400.h5")
subject_id, h5_filename = ("paul_20250428-20250430", "paul_20250428-20250430_vvs-encodingstimuli_z1_rw100-400.h5")
subject_id, h5_filename = ("venus_250426-250429", "venus_250426-250429_vvs-encodingstimuli_z1_rw80-250.h5")
subject_id, h5_filename = ("three0_250426-250501", "three0_250426-250501_vvs-encodingstimuli_z1_rw80-250.h5")
# subject_id, h5_filename = ("leap_250426-250501", "leap_250426-250501_vvs-encodingstimuli_z1_rw80-250.h5")

raw_model_output_dir = join(model_root, subject_id, "model_outputs_pca4all")
yaml_exportdir = f"{yaml_root}/{subject_id}"
exportdir = f"{export_root}/{subject_id}"
os.makedirs(exportdir, exist_ok=True)
os.makedirs(yaml_exportdir, exist_ok=True)

# %%
encoding_stim_dir = os.environ.get("PNC_ENCODING_STIM_DIR", "/n/holylabs/LABS/alvarez_lab/Lab/VVS_Accentuation/Stimuli/encodingstimuli_apr2025")  # was r"/n/holylabs/LABS/alvarez_lab/Lab/VVS_Accentuation/Stimuli/encodingstimuli_apr2025"
data_path = join(ephys_data_root, h5_filename)
data = load_from_hdf5(data_path)
data_dict = extract_neural_data_dict_2025apr(data)
data_dict['image_fps'] = parse_image_fullpaths(data_dict["stimulus_names"], [encoding_stim_dir], arbitrary_format=True)
resp_mat = data_dict['resp_mat']
ncsnr = data_dict['ncsnr']
reliability = data_dict["reliability"]
# save resp statistics 
mean_resp = resp_mat.mean(0)
std_resp = resp_mat.std(0)
q05_resp = np.percentile(resp_mat, 5, axis=0)
q95_resp = np.percentile(resp_mat, 95, axis=0)
q01_resp = np.percentile(resp_mat, 1, axis=0)
q99_resp = np.percentile(resp_mat, 99, axis=0)
min_resp = resp_mat.min(0)
max_resp = resp_mat.max(0)
resp_stats = {
    "resp_mat": resp_mat,
    "mean_resp": mean_resp,
    "std_resp": std_resp,
    "q05_resp": q05_resp,
    "q95_resp": q95_resp,
    "q01_resp": q01_resp,
    "q99_resp": q99_resp,
    "min_resp": min_resp,
    "max_resp": max_resp,
}
np.savez(join(exportdir, f"{subject_id}_resp_stats.npz"), **resp_stats)

# %%
# Get indices of top 5 most reliable neurons
topk_reliable_chan_idx = [342, 282, 306, 286, 81]
topk_reliable_chan_idx = [204, 56, 168, 120, 74]
# topk_reliable_chan_idx = np.argsort(data_dict['reliability'])[-5:][::-1]
topk_reliability = data_dict['reliability'][topk_reliable_chan_idx]
print("Most reliable channels")
print(topk_reliability)
print(topk_reliable_chan_idx)
DRY_RUN = False
# %%
model_names = [
    "dinov2_vitb14_reg",
    "clipag_vitb32",
    "siglip2_vitb16",
    "radio_v2.5-b",
    "resnet50_robust",
    "resnet50_clip",
    "resnet50_dino",
    "resnet50",
    "regnety_640",
    "AlexNet_training_seed_01",
    # "ReAlnet01",
]
for modelname in [
    "dinov2_vitb14_reg",
    "clipag_vitb32",
    "siglip2_vitb16",
    "radio_v2.5-b",
    "resnet50_robust",
    "resnet50_clip",
    "resnet50_dino",
    "resnet50",
    "regnety_640",
    "AlexNet_training_seed_01",
]:
    abbrev_map = LAYER_ABBREVIATION_MAPS[modelname]
    Xtransform_path = join(raw_model_output_dir, f"{subject_id}_{modelname}_sweep_regressors_layers_Xtfmer_RidgeCV.pkl")
    # red_20250428-20240429_resnet50_robust_sweep_regressors_layers_Xtfmer_RidgeCV.pkl
    readout_path = join(raw_model_output_dir, f"{subject_id}_{modelname}_sweep_regressors_layers_fitmodels_RidgeCV.pth")
    meta_path = join(raw_model_output_dir, f"{subject_id}_{modelname}_sweep_regressors_layers_pred_meta.pkl")
    fit_models_lyrswp_RidgeCV = th.load(readout_path, weights_only=False)
    # Xtfmer_lyrswp_RidgeCV = th.load(Xtransform_path, weights_only=False)
    Xtfmer_lyrswp_RidgeCV = pkl.load(open(Xtransform_path, "rb"))
    pred_data = pkl.load(open(meta_path, "rb"))
    pred_dict = pred_data["pred_dict"]
    D2_per_unit_train_dict = pred_data["D2_per_unit_train_dict"]
    D2_per_unit_test_dict = pred_data["D2_per_unit_test_dict"]
    for unit_id in topk_reliable_chan_idx:
        single_chan_result_df = construct_result_df_masked(D2_per_unit_train_dict, D2_per_unit_test_dict, mask=unit_id)
        single_chan_result_df = format_result_df_tuple_index(single_chan_result_df, )
        # Filter out 'srp' dimension reduction and get the row with the best test score
        print(f"Best config for PCA | {modelname} | {subject_id} | Ch{unit_id:02d}")
        best_row = single_chan_result_df.query("dimred == 'pca750'").sort_values('test_score', ascending=False).iloc[0]
        print(f"Best config - layer: {abbrev_map(best_row['layer'])} ({best_row['layer']}), dimred: {best_row['dimred']}, regressor: {best_row['regressor']}, "+\
              f"train_score: {best_row['train_score']:.3f}, test_score: {best_row['test_score']:.3f}")
        layername = best_row['layer']
        dimred_str = best_row['dimred']
        regressor_name = best_row['regressor']
        if DRY_RUN:
            print(f"DRY RUN: Printing best layer only Skipping export ")
            continue
        export_Xtransform_path = join(exportdir, f"{subject_id}_{modelname}_Ch{unit_id:02d}_Xtfmer_{layername}_{dimred_str}_{regressor_name}.pth")
        export_Xtransform_JIT_path = export_Xtransform_path.replace('.pth', '_JITscript.pt')
        export_meta_path = join(exportdir, f"{subject_id}_{modelname}_Ch{unit_id:02d}_meta_{layername}_{dimred_str}_{regressor_name}.pkl")
        export_readout_path = join(exportdir, f"{subject_id}_{modelname}_Ch{unit_id:02d}_readout_{layername}_{dimred_str}_{regressor_name}.pth")
        key = ((layername, dimred_str), regressor_name)
        
        regressor = fit_models_lyrswp_RidgeCV[key]
        Xtfmer = Xtfmer_lyrswp_RidgeCV[key[0]]
        pred_rsp = pred_data["pred_dict"][key]
        D2_per_unit_test = pred_data["D2_per_unit_test_dict"][key]
        D2_per_unit_train = pred_data["D2_per_unit_train_dict"][key]
        readout = LinearLayer_from_sklearn(regressor)
        th.save(readout, export_readout_path)
        if isinstance(Xtfmer, PCA):
            Xtfmer = PCA_torch(Xtfmer)
        elif isinstance(Xtfmer, SparseRandomProjection):
            Xtfmer = SRP_torch(Xtfmer)
        th.save(Xtfmer, export_Xtransform_path)
        Xtfmer_script = torch.jit.script(Xtfmer)
        Xtfmer_script.save(export_Xtransform_JIT_path)
        th.save({
            "reliability": reliability,
            "ncsnr": ncsnr,
            "D2_per_unit_test": D2_per_unit_test,
            "D2_per_unit_train": D2_per_unit_train,
            **resp_stats
        }, export_meta_path)
        print(f"Saved {export_readout_path}")
        print(f"Saved {export_Xtransform_path}")
        print(f"JIT script saved to {export_Xtransform_JIT_path}")
        print(f"Saved {export_meta_path}")
        # export yaml to somewhere        
        content, config = create_accentuation_config(
            subject_id=subject_id,
            modelname=modelname,
            unit_id=unit_id,
            regressor_name=regressor_name,
            layername=layername,
            exportdir=exportdir,
            export_readout_path=export_readout_path,
            export_Xtransform_JIT_path=export_Xtransform_JIT_path,
            export_meta_path=export_meta_path
        )
        output_yaml_path = f"{yaml_exportdir}/{subject_id}_{modelname}_Ch{unit_id}_accentuation_config.yaml"
        save_and_verify_config(content, output_yaml_path, config)
        # raise Exception

# %%
