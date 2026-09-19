# %%


# %%
import os
import pickle as pkl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from core.plot_utils import saveallforms
import seaborn as sns
from os.path import join
import sys
from core.model_load_utils import load_model_transform, MODEL_LAYER_FILTERS, LAYER_ABBREVIATION_MAPS
from neural_regress.feature_reduction_lib import FEATURE_REDUCTION_DEFAULTS, LAYER_TRANSFORM_FILTERS
from neural_regress.regress_eval_lib import format_result_df, plot_result_df_per_layer, construct_result_df_masked, \
    compute_pred_dict_D2_per_unit, format_result_df_tuple_index
#%% Utility Functions
from core.data_utils import load_from_hdf5, load_neural_data, load_neural_trial_resp_tensor, create_response_tensor, parse_image_fullpaths, extract_neural_data_dict_2025apr

# %%

encoding_stim_dir = os.environ.get("PNC_ENCODING_STIM_DIR", "/n/holylabs/LABS/alvarez_lab/Lab/VVS_Accentuation/Stimuli/encodingstimuli_apr2025")  # was r"/n/holylabs/LABS/alvarez_lab/Lab/VVS_Accentuation/Stimuli/encodingstimuli_apr2025"
dataroot = os.environ.get("PNC_EPHYS_DATA_ROOT", "/n/holylabs/LABS/alvarez_lab/Lab/VVS_Accentuation/Ephys_Data")  # was r"/n/holylabs/LABS/alvarez_lab/Lab/VVS_Accentuation/Ephys_Data"
model_root = os.environ.get("PNC_ENCODING_MODEL_ROOT", "/n/holylabs/LABS/alvarez_lab/Lab/VVS_Accentuation/Encoding_models")  # was "/n/holylabs/LABS/alvarez_lab/Lab/VVS_Accentuation/Encoding_models/"
# %% [markdown]
# ### Function version of plots

def sweep_combine_result_df(model_output_dir, subject_id, channel_mask=None, model_names=None):
    if model_names is None:
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
    
    top5_chan_result_df_col = []
    for modelname in model_names:
        pred_meta_path = join(model_output_dir, f"{subject_id}_{modelname}_sweep_regressors_layers_pred_meta.pkl")
        if not os.path.exists(pred_meta_path):
            print(f"pred_meta_path {pred_meta_path} does not exist, skipping {modelname}")
            continue
        result_df_path = join(model_output_dir, f"{subject_id}_{modelname}_sweep_regressors_layers_sweep_RidgeCV_df.pkl")
        if not os.path.exists(result_df_path):
            print(f"result_df_path {result_df_path} does not exist, skipping {modelname}")
            continue
        
        pred_meta = pkl.load(open(pred_meta_path, "rb"))
        result_df = pd.read_pickle(result_df_path)
        D2_per_unit_train_dict = pred_meta["D2_per_unit_train_dict"]
        D2_per_unit_test_dict = pred_meta["D2_per_unit_test_dict"]
        top5_chan_result_df = construct_result_df_masked(D2_per_unit_train_dict, D2_per_unit_test_dict, mask=channel_mask)
        top5_chan_result_df = format_result_df_tuple_index(top5_chan_result_df, )
        top5_chan_result_df["modelname"] = modelname
        top5_chan_result_df["layer_abbrev"] = top5_chan_result_df["layer"].map(LAYER_ABBREVIATION_MAPS[modelname])
        top5_chan_result_df_col.append(top5_chan_result_df)
    
    if top5_chan_result_df_col:
        return pd.concat(top5_chan_result_df_col, axis=0)
    else:
        return pd.DataFrame()


# %%
# Create a function that returns a figure
def plot_best_per_model(best_per_model, subtitle=f'Best Layer Performance per Model', figsize=(12, 6), descending=True):
    if descending:
        best_per_model = best_per_model.sort_values('test_score', ascending=False)
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(best_per_model['modelname'], best_per_model['test_score'], color='skyblue')
    ax.set_xlabel('Model')
    ax.set_ylabel('Best Test Score')
    ax.set_title(subtitle)
    plt.xticks(rotation=30, ha='right')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    # Add the best layer/dimred info as text on top of each bar
    for i, bar in enumerate(bars):
        layer = best_per_model.iloc[i]['layer_abbrev']
        dimred = best_per_model.iloc[i]['dimred']
        score = best_per_model.iloc[i]['test_score']
        
        # Truncate layer name if too long
        if len(layer) > 20:
            layer = layer[:17] + "..."
            
        ax.text(
            bar.get_x() + bar.get_width()/2, 
            bar.get_height() + 0.01, 
            f"{layer}\n{dimred}\n{score:.3f}", 
            ha='center', va='bottom', 
            fontsize=8, rotation=0
        )
    fig.tight_layout()
    return fig

# %%
def plot_model_layer_comparison(result_df, suptitle, rows=3, cols=4, figsize=(20, 15)):
    """
    Create a multi-panel figure with one subplot per model showing test score as function of layer.
    
    Parameters:
    -----------
    result_df : pandas.DataFrame
        DataFrame containing the results with columns: modelname, layer_abbrev, test_score, dimred, regressor
    suptitle : str
        Suptitle for the figure
    rows : int, optional
        Number of rows in the subplot grid (default: 3)
    cols : int, optional
        Number of columns in the subplot grid (default: 4)
    figsize : tuple, optional
        Figure size in inches (width, height) (default: (20, 15))
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure
    """
    # Create a multi-panel figure with one subplot per model showing test score as function of layer
    fig, axes = plt.subplots(rows, cols, figsize=figsize, constrained_layout=True, sharey='all')
    axes = axes.flatten()
    
    # Get unique models
    models = result_df['modelname'].unique()
    
    # Plot each model in its own subplot
    for i, model in enumerate(models):
        if i >= len(axes):  # Skip if we have more models than subplots
            break
        ax = axes[i]
        sns.lineplot(
            data=result_df.query("modelname == @model"),
            x='layer_abbrev', y='test_score', hue='dimred', style='regressor',
            marker='o', markersize=8, linewidth=2,
            ax=ax, 
        )
        ax.grid(True)
        ax.set_title(model, fontsize=14)
        ax.set_xlabel('Layer')
        ax.set_ylabel('Test Score (R²)')
        # Ensure y-axis ticks are visible
        ax.tick_params(axis='y', which='both', left=True, labelleft=True)
        # ax.set_ylim(0, layer_scores['test_score'].max() * 1.1)  # Add 10% padding
        # rotate and align in one go
        ax.tick_params(axis='x', rotation=40)        # rotate them
        for lbl in ax.get_xticklabels():             # grab the Text objects
            lbl.set_ha('right')                      # set horizontal alignment

    # Hide any unused subplots
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(suptitle, 
                 fontsize=16, y=0.98)
    plt.tight_layout()
    
    return fig


# %% [markdown]
# ### Other monkeys

# %%
# Use the function with reliability threshold

subject_id, filename = "paul_20250428-20250430", "paul_20250428-20250430_vvs-encodingstimuli_z1_rw100-400.h5"

for subject_id, filename in [
    # ("red_20250428-20240429", "red_20250428-20240429_vvs-encodingstimuli_z1_rw100-400.h5"),
    # ("red_20250428-20250430", "red_20250428-20250430_vvs-encodingstimuli_z1_rw100-400.h5"), 
    # ("paul_20250428-20250430", "paul_20250428-20250430_vvs-encodingstimuli_z1_rw100-400.h5"), 
    # ("three0_250426-250430", "three0_250426-250430_vvs-encodingstimuli_z1_rw80-250.h5"), 
    # ("venus_250426-250429", "venus_250426-250429_vvs-encodingstimuli_z1_rw80-250.h5"), 
    ("three0_250426-250501", "three0_250426-250501_vvs-encodingstimuli_z1_rw80-250.h5"),
    ("leap_250426-250501", "leap_250426-250501_vvs-encodingstimuli_z1_rw80-250.h5"),
]:
    data_path = join(dataroot, filename)
    model_output_dir = join(model_root, subject_id, "model_outputs_pca4all")
    synopsis_dir = join(model_root, subject_id, "synopsis")
    os.makedirs(synopsis_dir, exist_ok=True)
    data_dict = extract_neural_data_dict_2025apr(load_from_hdf5(data_path))
    reliability = data_dict['reliability']
    # Get the top 5 channels with highest reliability
    topk = 5
    topk_indices = np.argsort(reliability)[-topk:]
    # Create a binary mask for these top 5 channels
    top5_channel_mask = np.zeros_like(reliability, dtype=bool)
    top5_channel_mask[topk_indices] = True
    all_channel_mask = np.ones_like(reliability, dtype=bool)
    for channel_mask, mask_label in [(top5_channel_mask, "top5_reliab"), 
                                    (all_channel_mask, "all")]:
        # Report the threshold as the lowest reliability value among the top 5
        threshold = reliability[channel_mask].min()
        print(f"Using top 5 channels with reliability threshold: {threshold:.3f}")
        top5_chan_result_df = sweep_combine_result_df(model_output_dir, subject_id, channel_mask)
        top5_chan_result_df.to_csv(join(synopsis_dir, f"{subject_id}_{mask_label}_chan_all_models_result_df_synopsis_reliability_{threshold}.csv"))

        # %
        # Find the best layer/dimred/regressor combination for each model
        best_per_model = top5_chan_result_df.query("'pca750' in dimred").groupby('modelname').apply(
            lambda x: x.loc[x['test_score'].idxmax()]
        ).reset_index(drop=True)
        # Generate and display the figure
        fig = plot_best_per_model(best_per_model, f"{subject_id} - best regressor test R2 - All layers PCA only | Reliability > {threshold:.3f} (N={channel_mask.sum()})")
        saveallforms(synopsis_dir, f"{subject_id}_{mask_label}_chan_model_comparison_best_pca750", fig)
        plt.show()

        # Find the best layer/dimred/regressor combination for each model
        best_per_model = top5_chan_result_df.groupby('modelname').apply(
            lambda x: x.loc[x['test_score'].idxmax()]
        ).reset_index(drop=True)
        # Generate and display the figure
        fig = plot_best_per_model(best_per_model, f"{subject_id} - best regressor test R2 - All layers any dimred | Reliability > {threshold:.3f} (N={channel_mask.sum()})")
        saveallforms(synopsis_dir, f"{subject_id}_{mask_label}_chan_model_comparison_best_anydimred", fig)
        plt.show()

        # Example usage:
        fig = plot_model_layer_comparison(top5_chan_result_df, f"{subject_id} - Test Score by Layer Across Models | Reliability > {threshold:.3f} (N={channel_mask.sum()})")
        saveallforms(synopsis_dir, f"{subject_id}_{mask_label}_chan_all_models_comparison_by_layer", fig)
        plt.show()
        

# %%
