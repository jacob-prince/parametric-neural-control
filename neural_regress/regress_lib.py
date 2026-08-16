import time
import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pylab as plt
from os.path import join
from tqdm.auto import tqdm
from collections import defaultdict
from copy import deepcopy
import torch
import torch as th
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToPILImage, ToTensor, Normalize, Resize
from scipy.stats import spearmanr, pearsonr
import sklearn
from sklearn.random_projection import johnson_lindenstrauss_min_dim, \
    SparseRandomProjection, GaussianRandomProjection
from torch.utils.data import Subset, SubsetRandomSampler
from sklearn.linear_model import LogisticRegression, LinearRegression, \
    Ridge, Lasso, PoissonRegressor, RidgeCV, LassoCV, MultiTaskLassoCV
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from core.plot_utils import show_imgrid
from core.layer_hook_utils import featureFetcher
from core.dataset_utils import ImagePathDataset, DataLoader


denormalizer = Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                         std=[1/0.229, 1/0.224, 1/0.225])
normalizer = Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
resizer = Resize(227, )

def calc_features(score_vect, imgfullpath_vect, net, featlayer,
                  batch_size=40, workers=6, img_dim=(227, 227)):
    """
    Calculate features for a set of images.
    :param score_vect: numpy vector of scores,
            if None, then it's default to be zeros. ImagePathDataset will handle None scores.
    :param imgfullpath_vect: a list full path to images
    :param net: net to extract features from
    :param featlayer: layer to extract features from
    :param batch_size: batch size for DataLoader
    :param workers: number of workers for DataLoader
    :param img_dim: image dimensions
    :return:
    """
    imgdata = ImagePathDataset(imgfullpath_vect, score_vect, img_dim=img_dim)
    imgloader = DataLoader(imgdata, batch_size=batch_size, shuffle=False, num_workers=workers)

    featFetcher = featureFetcher(net, print_module=False)
    featFetcher.record(featlayer, )
    feattsr_col = []
    for i, (imgtsr, score) in tqdm(enumerate(imgloader)):
        with torch.no_grad():
            net(imgtsr.cuda())
        feattsr = featFetcher[featlayer]
        feattsr_col.append(feattsr.cpu().numpy())

    feattsr_all = np.concatenate(feattsr_col, axis=0)
    print("feature tensor shape", feattsr_all.shape)
    print("score vector shape", score_vect.shape)
    del feattsr_col, featFetcher
    return feattsr_all


def calc_reduce_features(score_vect, imgfullpath_vect, feat_transformers, net, featlayer,
                  batch_size=40, workers=6, img_dim=(227, 227)):
    """Calculate reduced features for a set of images. (for memory saving)

    :param score_vect: numpy vector of scores,
            if None, then it's default to be zeros. ImagePathDataset will handle None scores.
    :param imgfullpath_vect: a list full path to images
    :param feattsr_reducer: a dict of functions that reduce a feature tensor to a vector
            Here we assume the input to each transformer is a numpy array (not torch tensor)
        Examples:
                    {"none": lambda x: x, }
        Examples:
            Xfeat_transformer = {'pca': lambda tsr: pca.transform(tsr.reshape(tsr.shape[0], -1)),
                     "srp": lambda tsr: srp.transform(tsr.reshape(tsr.shape[0], -1)),
                     "sp_rf": lambda tsr: tsr[:, :, 6, 6],
                     "sp_avg": lambda tsr: tsr.mean(axis=(2, 3))}
    :param net: net to extract features from
    :param featlayer: layer to extract features from
    :param batch_size: batch size for DataLoader
    :param workers: number of workers for DataLoader
    :param img_dim: image dimensions
    :return:
    """
    imgdata = ImagePathDataset(imgfullpath_vect, score_vect, img_dim=img_dim)
    imgloader = DataLoader(imgdata, batch_size=batch_size, shuffle=False, num_workers=workers)

    featFetcher = featureFetcher(net, print_module=False)
    featFetcher.record(featlayer, )
    feattsr_col = defaultdict(list)
    for i, (imgtsr, score) in tqdm(enumerate(imgloader)):
        with torch.no_grad():
            net(imgtsr.cuda())
        feattsr = featFetcher[featlayer]
        for tfmname, feat_transform in feat_transformers.items():
            feattsr_col[tfmname].append(feat_transform(feattsr.cpu().numpy()))
        # feattsr_col.append(feattsr.cpu().numpy())
    for tfmname in feattsr_col:
        feattsr_col[tfmname] = np.concatenate(feattsr_col[tfmname], axis=0)
        print(tfmname, "feature tensor shape", feattsr_col[tfmname].shape)
    # feattsr_all = np.concatenate(feattsr_col, axis=0)
    print("score vector shape", score_vect.shape)
    del featFetcher
    return feattsr_col


def calc_reduce_features_dataset(dataset, feat_transformers, net, featlayer,
                  batch_size=40, workers=6, img_dim=(227, 227), idx_range=None):
    """Calculate reduced features for a set of images. (for memory saving)

    :param dataset: Image Dataset
    :param feattsr_reducer: a dict of functions that reduce a feature tensor to a vector
            Here we assume the input to each transformer is a numpy array (not torch tensor)
        Examples:
                    {"none": lambda x: x, }
        Examples:
            Xfeat_transformer = {'pca': lambda tsr: pca.transform(tsr.reshape(tsr.shape[0], -1)),
                     "srp": lambda tsr: srp.transform(tsr.reshape(tsr.shape[0], -1)),
                     "sp_rf": lambda tsr: tsr[:, :, 6, 6],
                     "sp_avg": lambda tsr: tsr.mean(axis=(2, 3))}
    :param net: net to extract features from
    :param featlayer: layer to extract features from
    :param batch_size: batch size for DataLoader
    :param workers: number of workers for DataLoader
    :param img_dim: image dimensions
    :return:
    """
    if idx_range is None:
        imgloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    else:
        imgloader = DataLoader(Subset(dataset, idx_range), batch_size=batch_size,
                               shuffle=False, num_workers=workers,)
    # imgdata = dataset
    # imgloader = DataLoader(imgdata, batch_size=batch_size, shuffle=False, num_workers=workers)

    featFetcher = featureFetcher(net, print_module=False)
    featFetcher.record(featlayer, )
    feattsr_col = defaultdict(list)
    for i, (imgtsr, score) in tqdm(enumerate(imgloader)):
        with torch.no_grad():
            net(imgtsr.cuda())
        feattsr = featFetcher[featlayer]
        for tfmname, feat_transform in feat_transformers.items():
            feattsr_col[tfmname].append(feat_transform(feattsr.cpu().numpy()))
        # feattsr_col.append(feattsr.cpu().numpy())
    for tfmname in feattsr_col:
        feattsr_col[tfmname] = np.concatenate(feattsr_col[tfmname], axis=0)
        print(tfmname, "feature tensor shape", feattsr_col[tfmname].shape)
    # feattsr_all = np.concatenate(feattsr_col, axis=0)
    # print("score vector shape", score_vect.shape)
    del featFetcher
    return feattsr_col


def sweep_regressors(Xdict, y_all, regressors, regressor_names, verbose=True, n_jobs=-1, train_split_idx=None):
    """
    Sweep through a list of regressors (with cross validation), and input type (Xdict)
    For each combination of Xtype and regressor, return the best CVed regressor
        and its parameters, in a model_dict and a dataframe
    :param Xdict:
    :param y_all:
    :param regressors:
    :param regressor_names:
    :return:
        result_df: dataframe with the results of the regression
        models: dict of regressor objects
    """
    result_summary = {}
    models = {}
    if train_split_idx is None:
        idx_train, idx_test = train_test_split(
            np.arange(len(y_all)), test_size=0.2, random_state=42, shuffle=True
        )
    else:
        idx_train = train_split_idx
        idx_test = np.setdiff1d(np.arange(len(y_all)), train_split_idx)
        print(f"Using {len(idx_train)} training samples provided by train_split_idx and {len(idx_test)} testing samples")
    for xtype in Xdict:
        X_all = Xdict[xtype]  # score_vect
        y_train, y_test = y_all[idx_train], y_all[idx_test]
        X_train, X_test = X_all[idx_train], X_all[idx_test]
        nfeat = X_train.shape[1]
        for estim, label in zip(regressors, regressor_names):
            start_time = time.time()
            if isinstance(estim, RidgeCV) or isinstance(estim, LassoCV) or isinstance(estim, MultiTaskLassoCV):
                clf = estim.fit(X_train, y_train)
                clf = deepcopy(clf)
                alpha = estim.alpha_
            elif isinstance(estim, MultiOutputSeparateLassoCV):
                clf = estim.fit(X_train, y_train)
                clf = deepcopy(clf)
                alpha = estim.alpha_()
            elif hasattr(estim, "alpha"):
                clf = GridSearchCV(estimator=estim, n_jobs=n_jobs,
                                   param_grid=dict(alpha=[1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 1E4, 1E5, 1E6, 1E7, 1E8, 1E9], ),
                                   ).fit(X_train, y_train)
                alpha = clf.best_params_["alpha"]
            else:
                clf = estim.fit(X_train, y_train)
                clf = deepcopy(clf)
                alpha = np.nan
            D2_train = clf.score(X_train, y_train)
            D2_test = clf.score(X_test, y_test)
            end_time = time.time()
            # if isinstance(xtype, tuple):
            #     result_summary[(*xtype, label)] = \
            #         {"alpha": alpha, "train_score": D2_train, "test_score": D2_test, "n_feat": nfeat, "runtime": end_time - start_time}
            #     models[(*xtype, label)] = clf
            # elif isinstance(xtype, str):
            result_summary[(xtype, label)] = \
                {"alpha": alpha, "train_score": D2_train, "test_score": D2_test, "n_feat": nfeat, "runtime": end_time - start_time}
            models[(xtype, label)] = clf
            # else:
            #     raise ValueError(f"Unknown input type: {xtype}")
            if verbose:
                print(f"{xtype} {label} D2_train: {D2_train:.3f} D2_test: {D2_test:.3f} time: {end_time - start_time:.3f}")

        result_df = pd.DataFrame(result_summary)

    print(result_df.T)
    return result_df.T, models


def compare_activation_prediction(target_scores_natval, pred_scores_natval, exptitle="", savedir=""):
    result_col = {}
    for k in pred_scores_natval:
        rho_s = spearmanr(target_scores_natval, pred_scores_natval[k])
        rho_p = pearsonr(target_scores_natval, pred_scores_natval[k])
        R2 = 1 - np.var(pred_scores_natval[k] - target_scores_natval) / np.var(target_scores_natval)
        print(k, f"spearman: {rho_s.correlation:.3f} P={rho_s.pvalue:.1e}",
                 f"pearson: {rho_p[0]:.3f} P={rho_p[1]:.1e} R2={R2:.3f}")
        result_col[k] = {"spearman": rho_s.correlation, "pearson": rho_p[0],
                         "spearman_pval": rho_s.pvalue, "pearson_pval": rho_p[1],
                         "R2": R2, "dataset": exptitle, "n_sample": len(target_scores_natval)}

        plt.figure()
        plt.scatter(target_scores_natval, pred_scores_natval[k], s=16, alpha=0.5)
        plt.xlabel("Target scores")
        plt.ylabel("Predicted scores")
        plt.axis('equal')
        plt.title(f"{exptitle} {k}\n"
                  f"corr pearsonr {rho_p[0]:.3f} P={rho_p[1]:.1e}\n"
                  f"corr spearmanr {rho_s.correlation:.3f} P={rho_s.pvalue:.1e} R2={R2:.3f}")
        plt.tight_layout()
        plt.savefig(join(savedir, f"{exptitle}_{k}_regress.png"))
        plt.show()

    test_result_df = pd.DataFrame(result_col)
    test_result_df.T.to_csv(join(savedir, f"{exptitle}_regress_results.csv"))
    return test_result_df.T


def evaluate_prediction(fit_models, Xfeat_dict, y_true, label="", savedir=None):
    """
    Evaluate the prediction of a dict of models
    :param fit_models: dict of models
    :param Xfeat_dict: dict of feature tensors, input to the models
    :param y_true: true y values
    :param label: label for saving, saved to the dataframe.
    :param savedir: directory to save the results. If None, no saving.
        saved `df` `eval_dict` `y_pred_dict`
    :return:
        df: dataframe summarize evaluation statistics of each model
        eval_dict: same thing in a dict
        y_pred_dict: dict of predictions vectors, used to do other evaluation.

    Ported from ManifExp_regression_fit
    """
    print(label, f"  N imgs: {len(y_true)}")
    eval_dict = {}
    y_pred_dict = {}
    for (Xtype, regrname), regr in fit_models.items():
        try:
            y_pred = regr.predict(Xfeat_dict[Xtype])
            if isinstance(regr, PLSRegression):
                y_pred = y_pred[:, 0]
            D2 = regr.score(Xfeat_dict[Xtype], y_true)
            rho_p, pval_p = pearsonr(y_pred, y_true)
            rho_s, pval_s = spearmanr(y_pred, y_true)
            print(f"{Xtype} {regrname} Prediction Pearson: {rho_p:.3f} {pval_p:.1e} Spearman: {rho_s:.3f} {pval_s:.1e} D2: {D2:.3f}")
            eval_dict[(Xtype, regrname)] = {"rho_p":rho_p, "pval_p":pval_p, "rho_s":rho_s, "pval_s":pval_s, "D2":D2, "imgN":len(y_true)}
            y_pred_dict[(Xtype, regrname)] = y_pred
        except:
            continue
    parts = label.split("-")
    layer, datasetstr = parts[-2], parts[-1]
    df = pd.DataFrame(eval_dict).T
    df["label"] = label
    df["layer"] = layer
    df["img_space"] = datasetstr
    if savedir is not None:
        df.to_csv(join(savedir, f"eval_predict_{label}.csv"), index=True)
        pkl.dump(eval_dict, open(join(savedir, f"eval_stats_{label}.pkl"), "wb"))
        pkl.dump(y_pred_dict, open(join(savedir, f"eval_predvec_{label}.pkl"), "wb"))
    return df, eval_dict, y_pred_dict


def merge_dict_arrays(*dict_arrays):
    """
    Merge a list of dicts into a single dict.
    """
    # skip the empty dicts, use the dict with the most keys
    keys = []
    for d in dict_arrays:
        if len(d) != 0:
            keys = [*d.keys()]
            break
    # fill the empty dicts with the keys paired with empty arrays (this dataset has no images in this )
    for d in dict_arrays:
        if len(d) == 0:
            for k in keys:
                d[k] = np.array([])

    # TODO: solve variable array length problem. Every value in each array should be the same length.
    merged_dict = {}
    for key in keys:
        merged_dict[key] = np.concatenate([d[key] for d in dict_arrays], axis=0)

    return merged_dict


def merge_arrays(*arrays):
    """
    Merge a list of arrays into a single array.
    """
    return np.concatenate(arrays, axis=0)


def evaluate_dict(y_pred_dict, y_true, label, savedir=None):
    print(label, f"  N imgs: {len(y_true)}")
    eval_dict = {}
    for (Xtype, regrname), y_pred in y_pred_dict.items():
        D2 = 1 - np.var(y_pred - y_true) / np.var(y_true)  # regr.score(Xfeat_dict[Xtype], y_true)
        rho_p, pval_p = pearsonr(y_pred, y_true)
        rho_s, pval_s = spearmanr(y_pred, y_true)
        print(
            f"{Xtype} {regrname} Prediction Pearson: {rho_p:.3f} {pval_p:.1e} Spearman: {rho_s:.3f} {pval_s:.1e} D2: {D2:.3f}")
        eval_dict[(Xtype, regrname)] = {"rho_p": rho_p, "pval_p": pval_p, "rho_s": rho_s, "pval_s": pval_s,
                                        "D2": D2, "imgN": len(y_true)}
    parts = label.split("-")
    layer, datasetstr = parts[-2], parts[-1]
    df = pd.DataFrame(eval_dict).T
    df["label"] = label
    df["layer"] = layer
    df["img_space"] = datasetstr
    if savedir is not None:
        df.to_csv(join(savedir, f"eval_predict_{label}.csv"), index=True)
        pkl.dump(eval_dict, open(join(savedir, f"eval_stats_{label}.pkl"), "wb"))
        pkl.dump(y_pred_dict, open(join(savedir, f"eval_predvec_{label}.pkl"), "wb"))

    return df, eval_dict, y_pred_dict


def LinearLayer_from_sklearn(model):
    """Convert a sklearn linear model to a PyTorch Linear layer."""
    if isinstance(model, sklearn.model_selection._search.GridSearchCV):
        model = model.best_estimator_
    assert model.coef_.shape[1] == model.n_features_in_
    readout = nn.Linear(model.coef_.shape[1], model.coef_.shape[0], bias=True)
    readout.weight.data = th.tensor(model.coef_).float()
    readout.bias.data = th.tensor(model.intercept_).float()
    return readout


def _perform_regression_old(feat_dict, resp_mat, reliability, thresh=0.8, layerkey="last_block",):
    """Perform regression using extracted features and neural responses."""
    # TODO: add customizable feature transforms
    # TODO: add customizable regressors
    # Preprocess features
    feat_tsr = feat_dict[layerkey]
    print(feat_tsr.shape)
    featmat = feat_tsr.view(feat_tsr.shape[0], -1).numpy()
    featmat_avg = feat_tsr.mean(dim=(2, 3))  # B x C
    centpos = (feat_tsr.shape[2] // 2, feat_tsr.shape[3] // 2)
    featmat_rf = feat_tsr[:, :, centpos[0]:centpos[0]+1, centpos[1]:centpos[1]+1].mean(dim=(2,3))  # B x n_components
    # featmat_CLS = feat_tsr[:, 0, :].numpy()
    Xdict = {"sp_avg": featmat_avg, "sp_rf": featmat_rf, 'none': featmat} # "srp": srp_featmat, "pca": pca_featmat,
    # Xdict = {"sp_avg": featmat_avg, "cls": featmat_CLS}
    # Define regressors
    ridge = Ridge(alpha=1.0)
    kr_rbf = KernelRidge(alpha=1.0, kernel="rbf", gamma=None)
    # Mask reliable channels
    chan_mask = reliability > thresh
    print(f"Fitting models for reliable channels > {thresh} N={chan_mask.sum()}")
    regressors = [ridge, kr_rbf]
    regressor_names = ["Ridge", "KernelRBF"]
    result_df, fit_models = sweep_regressors(Xdict, resp_mat[:, chan_mask], regressors, regressor_names)
    return result_df, fit_models, chan_mask, Xdict


# Define explicit feature reduction methods. 
# Easier for exporting and serializing.
def avgtoken_transform(x):
    return x.mean(dim=(1))


def clstoken_transform(x):
    return x[:, 0, :]


def flatten_transform(x):
    return x.flatten(start_dim=1)


def sp_avg_transform(x):
    return x.mean(dim=(2,3))


def sp_cent_transform(x):
    centpos = (x.shape[2] // 2, x.shape[3] // 2)
    return x[:, :, centpos[0]:centpos[0]+1, centpos[1]:centpos[1]+1].mean(dim=(2,3))


def transform_features2Xdict(feat_dict, layer_names=None, 
                             dimred_list=["pca1000", "sp_cent", "sp_avg", "full",],
                             pretrained_Xtransforms={}, use_pca_dual=True, use_srp_torch=True,
                             train_split_idx=None):
    # now we use pca dual solver by default, changing default behavior. 
    Xdict = {}
    tfm_dict = {}
    time_start = time.time()
    for layerkey in feat_dict.keys() if layer_names is None else layer_names:
        feat_tsr = feat_dict[layerkey]
        time_feat_tsr = time.time()
        print(layerkey, feat_tsr.shape, )
        featmat = feat_tsr.flatten(start_dim=1)
        if train_split_idx is not None:
            featmat_train = featmat[train_split_idx, :]
        else:
            featmat_train = featmat
        
        for dimred in dimred_list:
            time_dimred = time.time()
            if dimred.startswith("pca"):
                # TODO: if there is multiple PCA dimred str, we can perform PCA only once and used the cached results and chuncage the columns. 
                n_components = int(dimred.split("pca")[-1])
                if f"{layerkey}_{dimred}" in pretrained_Xtransforms:
                    # use pretrained PCA transformer
                    pca_transformer = pretrained_Xtransforms[f"{layerkey}_{dimred}"]
                    featmat_pca = pca_transformer.transform(featmat)
                else:
                    # fit PCA transformer on training set
                    if use_pca_dual:
                        from neural_regress.PCA_dual_solver_lib import pca_dual_fit_transform, pca_dual_fit_transform_sep
                        # pca_transformer, featmat_pca_ = pca_dual_fit_transform(featmat_train, n_components, device='cuda')
                        # featmat_pca = pca_transformer.transform(featmat)
                        pca_transformer, _, featmat_pca = pca_dual_fit_transform_sep(featmat_train, featmat, n_components, device='cuda') # this is more efficient than above. this is transforming on GPU
                    else:
                        pca_transformer = PCA(n_components=n_components)
                        pca_transformer.fit(featmat_train)
                        featmat_pca = pca_transformer.transform(featmat)
                Xdict.update({f"{layerkey}_{dimred}": featmat_pca})
                tfm_dict.update({f"{layerkey}_{dimred}": pca_transformer})
                X_shape = featmat_pca.shape
            elif dimred.startswith("srp"):
                if dimred == "srp":
                    n_components = "auto"
                else:
                    n_components = int(dimred.split("srp")[-1])
                if f"{layerkey}_{dimred}" in pretrained_Xtransforms:
                    # use pretrained SRP transformer
                    srp_transformer = pretrained_Xtransforms[f"{layerkey}_{dimred}"]
                    featmat_srp = srp_transformer.transform(featmat)
                else:
                    # fit SRP transformer on training set
                    if use_srp_torch:
                        from neural_regress.SRP_torch_lib import SparseRandomProjection_fit_transform_torch
                        featmat_srp, srp_transformer = SparseRandomProjection_fit_transform_torch(featmat, 
                                n_components=n_components, eps=0.1, random_state=42, device="cuda")
                        if srp_transformer is None:
                            srp_transformer = nn.Identity()
                    else:
                        srp_transformer = SparseRandomProjection(n_components=n_components)
                        featmat_srp = srp_transformer.fit_transform(featmat)
                Xdict.update({f"{layerkey}_{dimred}": featmat_srp})
                tfm_dict.update({f"{layerkey}_{dimred}": srp_transformer})
                X_shape = featmat_srp.shape
            elif dimred == "sp_avg":
                featmat_avg = feat_tsr.mean(dim=(2, 3))  # B x C
                Xdict.update({f"{layerkey}_sp_avg": featmat_avg})
                tfm_dict.update({f"{layerkey}_sp_avg": sp_avg_transform})
                X_shape = featmat_avg.shape
            elif dimred == "sp_cent":
                centpos = (feat_tsr.shape[2] // 2, feat_tsr.shape[3] // 2)
                featmat_cent = feat_tsr[:, :, centpos[0]:centpos[0]+1, centpos[1]:centpos[1]+1].mean(dim=(2,3))  # B x n_components
                Xdict.update({f"{layerkey}_sp_cent": featmat_cent})
                tfm_dict.update({f"{layerkey}_sp_cent": sp_cent_transform})
                X_shape = featmat_cent.shape
            elif dimred == "avgtoken":
                featmat_avg = feat_tsr.mean(dim=(1))  # B x C
                Xdict.update({f"{layerkey}_avgtoken": featmat_avg})
                tfm_dict.update({f"{layerkey}_avgtoken": avgtoken_transform})
                X_shape = featmat_avg.shape
            elif dimred == "clstoken":
                featmat_cls = feat_tsr[:, 0, :]  # B x C
                Xdict.update({f"{layerkey}_clstoken": featmat_cls})
                tfm_dict.update({f"{layerkey}_clstoken": clstoken_transform})
                X_shape = featmat_cls.shape
            elif dimred == "full":
                Xdict.update({f"{layerkey}_full": featmat})
                tfm_dict.update({f"{layerkey}_full": flatten_transform})
                X_shape = featmat.shape
            else:
                raise ValueError(f"Unknown dimension reduction method: {dimred}")
            print(f"Time taken to transform {layerkey} {dimred} {list(X_shape)}: {time.time() - time_dimred:.3f}s")
        print(f"Time taken to transform {layerkey}: {time.time() - time_feat_tsr:.3f}s")
    print(f"Time taken to transform all features: {time.time() - time_start:.3f}s")
    return Xdict, tfm_dict



def transform_features2Xdict_new(feat_dict, layer_names=None, 
                             dimred_list=["pca1000", "sp_cent", "sp_avg", "full",],
                             pretrained_Xtransforms={}, use_pca_dual=True, use_srp_torch=True,
                             train_split_idx=None):
    # now we use pca dual solver by default, changing default behavior. 
    Xdict = {}
    tfm_dict = {}
    time_start = time.time()
    for layerkey in feat_dict.keys() if layer_names is None else layer_names:
        feat_tsr = feat_dict[layerkey]
        time_feat_tsr = time.time()
        print(layerkey, feat_tsr.shape, )
        featmat = feat_tsr.flatten(start_dim=1)
        if train_split_idx is not None:
            featmat_train = featmat[train_split_idx, :]
        else:
            featmat_train = featmat
        
        for dimred in dimred_list:
            time_dimred = time.time()
            if dimred.startswith("pca"):
                # TODO: if there is multiple PCA dimred str, we can perform PCA only once and used the cached results and chuncage the columns. 
                n_components = int(dimred.split("pca")[-1])
                if (layerkey, dimred) in pretrained_Xtransforms:
                    # use pretrained PCA transformer
                    pca_transformer = pretrained_Xtransforms[(layerkey, dimred)]
                    featmat_pca = pca_transformer.transform(featmat)
                else:
                    # fit PCA transformer on training set
                    if use_pca_dual:
                        from neural_regress.PCA_dual_solver_lib import pca_dual_fit_transform, pca_dual_fit_transform_sep
                        # pca_transformer, featmat_pca_ = pca_dual_fit_transform(featmat_train, n_components, device='cuda')
                        # featmat_pca = pca_transformer.transform(featmat)
                        pca_transformer, _, featmat_pca = pca_dual_fit_transform_sep(featmat_train, featmat, n_components, device='cuda') # this is more efficient than above. this is transforming on GPU
                    else:
                        pca_transformer = PCA(n_components=n_components)
                        pca_transformer.fit(featmat_train)
                        featmat_pca = pca_transformer.transform(featmat)
                Xdict.update({(layerkey, dimred): featmat_pca})
                tfm_dict.update({(layerkey, dimred): pca_transformer})
                X_shape = featmat_pca.shape
            elif dimred.startswith("srp"):
                if dimred == "srp":
                    n_components = "auto"
                else:
                    n_components = int(dimred.split("srp")[-1])
                if (layerkey, dimred) in pretrained_Xtransforms:
                    # use pretrained SRP transformer
                    srp_transformer = pretrained_Xtransforms[(layerkey, dimred)]
                    featmat_srp = srp_transformer.transform(featmat)
                else:
                    # fit SRP transformer on training set
                    if use_srp_torch:
                        from neural_regress.SRP_torch_lib import SparseRandomProjection_fit_transform_torch
                        featmat_srp, srp_transformer = SparseRandomProjection_fit_transform_torch(featmat, 
                                n_components=n_components, eps=0.1, random_state=42, device="cuda")
                    else:
                        srp_transformer = SparseRandomProjection(n_components=n_components)
                        featmat_srp = srp_transformer.fit_transform(featmat)
                Xdict.update({(layerkey, dimred): featmat_srp})
                tfm_dict.update({(layerkey, dimred): srp_transformer})
                X_shape = featmat_srp.shape
            elif dimred == "sp_avg":
                featmat_avg = feat_tsr.mean(dim=(2, 3))  # B x C
                Xdict.update({(layerkey, dimred): featmat_avg})
                tfm_dict.update({(layerkey, dimred): sp_avg_transform})
                X_shape = featmat_avg.shape
            elif dimred == "sp_cent":
                centpos = (feat_tsr.shape[2] // 2, feat_tsr.shape[3] // 2)
                featmat_cent = feat_tsr[:, :, centpos[0]:centpos[0]+1, centpos[1]:centpos[1]+1].mean(dim=(2,3))  # B x n_components
                Xdict.update({(layerkey, dimred): featmat_cent})
                tfm_dict.update({(layerkey, dimred): sp_cent_transform})
                X_shape = featmat_cent.shape
            elif dimred == "avgtoken":
                featmat_avg = feat_tsr.mean(dim=(1))  # B x C
                Xdict.update({(layerkey, dimred): featmat_avg})
                tfm_dict.update({(layerkey, dimred): avgtoken_transform})
                X_shape = featmat_avg.shape
            elif dimred == "clstoken":
                featmat_cls = feat_tsr[:, 0, :]  # B x C
                Xdict.update({(layerkey, dimred): featmat_cls})
                tfm_dict.update({(layerkey, dimred): clstoken_transform})
                X_shape = featmat_cls.shape
            elif dimred == "full":
                Xdict.update({(layerkey, dimred): featmat})
                tfm_dict.update({(layerkey, dimred): flatten_transform})
                X_shape = featmat.shape
            else:
                raise ValueError(f"Unknown dimension reduction method: {dimred}")
            print(f"Time taken to transform {layerkey} {dimred} {list(X_shape)}: {time.time() - time_dimred:.3f}s")
        print(f"Time taken to transform {layerkey}: {time.time() - time_feat_tsr:.3f}s")
    print(f"Time taken to transform all features: {time.time() - time_start:.3f}s")
    return Xdict, tfm_dict




def apply_feature_transforms(feat_dict_lyrswp, module_names2record, dimred_transform_dict, 
                            layer_transform_filter=None):
    """Apply feature transforms to each layer using the provided transform dictionary.
    
    Args:
        feat_dict_lyrswp: Dictionary of features for each layer
        module_names2record: List of layer names to process
        dimred_transform_dict: Dictionary of transform modules
        layer_transform_filter: Optional function that takes (layer_name, transform_name) 
                               and returns True if the transform should be applied to that layer
        
    Returns:
        Xdict_lyrswp: Dictionary of transformed features
        Xtfmer_lyrswp: Dictionary of transform modules used
    """
    Xdict_lyrswp = {}
    Xtfmer_lyrswp = {}
    
    # Default filter that allows all combinations
    if layer_transform_filter is None:
        # For backward compatibility with siglip2_vitb16 case
        layer_transform_filter = lambda layer, dimred_str: not (("attn_pool" in layer) != ("full" in dimred_str))
    
    for layer in module_names2record:
        for dimred_str, transform_module in dimred_transform_dict.items():
            # Skip if the filter returns False
            if not layer_transform_filter(layer, dimred_str):
                continue
            t0 = time.time()
            Xdict_lyrswp[(layer, dimred_str)] = transform_module(feat_dict_lyrswp[layer])
            Xtfmer_lyrswp[(layer, dimred_str)] = transform_module
            print(f"Time taken to transform {layer} x {dimred_str}: {time.time() - t0:.3f}s")
    
    return Xdict_lyrswp, Xtfmer_lyrswp




def perform_regression_sweeplayer(feat_dict, resp_mat, layer_names=None, 
                                  dimred_list=["pca1000", "sp_cent", "sp_avg", "full",],
                                  regressor_list=["Ridge",], verbose=True, use_pca_dual=False,
                                  pretrained_Xtransforms={}):
    """Perform regression using extracted features and neural responses."""
    # Preprocess features
    Xdict, tfm_dict = transform_features2Xdict(feat_dict, layer_names, 
                                    dimred_list, pretrained_Xtransforms, use_pca_dual=use_pca_dual)
    # Define regressors
    regressors = []
    for regressor_name in regressor_list:
        if regressor_name == "Ridge":
            regressors.append(Ridge(alpha=1.0))
        elif regressor_name == "Lasso":
            regressors.append(Lasso(alpha=1.0))
        elif regressor_name == "KernelRBF":
            regressors.append(KernelRidge(alpha=1.0, kernel="rbf", gamma=None))
        else:
            raise ValueError(f"Unknown regressor: {regressor_name}")
    regressor_names = regressor_list
    
    result_df, fit_models = sweep_regressors(Xdict, resp_mat, regressors, regressor_names, verbose=verbose)
    return result_df, fit_models, Xdict, tfm_dict


def perform_regression_sweeplayer_RidgeCV(feat_dict, resp_mat, layer_names=None, 
                                  dimred_list=["pca1000", "sp_cent", "sp_avg", "full",],
                                  alpha_list=[1E-4, 1E-3, 1E-2, 1E-1, 1, 10, 100, 1E3, 1E4, 1E5, 1E6, 1E7, 1E8, 1E9],
                                  alpha_per_target=True,
                                  pretrained_Xtransforms={},
                                  verbose=True, use_pca_dual=False):
    """Perform regression using extracted features and neural responses."""
    # Preprocess features
    Xdict, tfm_dict = transform_features2Xdict(feat_dict, layer_names, 
                                    dimred_list, pretrained_Xtransforms, use_pca_dual=use_pca_dual)
    # Define regressors
    regressors = [RidgeCV(alphas=alpha_list, alpha_per_target=alpha_per_target)]
    regressor_names = ["RidgeCV"]
    
    result_df, fit_models = sweep_regressors(Xdict, resp_mat, regressors, regressor_names, verbose=verbose)
    return result_df, fit_models, Xdict, tfm_dict


def perform_regression_sweeplayer_MultiTaskLassoCV(feat_dict, resp_mat, layer_names=None, 
                                  dimred_list=["pca1000", "sp_cent", "sp_avg", "full",],
                                  n_alphas=100, #alpha_list=[1E-4, 1E-3, 1E-2, 1E-1, 1, 10, 100, 1E3, 1E4, 1E5, 1E6, 1E7, 1E8, 1E9],
                                  alpha_per_target=True,
                                  pretrained_Xtransforms={},
                                  verbose=True, use_pca_dual=False):
    """Perform regression using extracted features and neural responses."""
    # Preprocess features
    Xdict, tfm_dict = transform_features2Xdict(feat_dict, layer_names, 
                                    dimred_list, pretrained_Xtransforms, use_pca_dual=use_pca_dual)
    # Define regressors
    regressors = [MultiTaskLassoCV(cv=5, n_alphas=n_alphas, n_jobs=-1)] # alpha_per_target=alpha_per_target RidgeCV(alphas=alpha_list, alpha_per_target=alpha_per_target)
    regressor_names = ["MultiTaskLassoCV"]
    result_df, fit_models = sweep_regressors(Xdict, resp_mat, regressors, regressor_names, verbose=verbose)
    return result_df, fit_models, Xdict, tfm_dict


def perform_regression_sweeplayer_LassoCV_sepchannel(feat_dict, resp_mat, layer_names=None, 
                                  dimred_list=["pca1000", "sp_cent", "sp_avg", "full",],
                                  n_alphas=100, #alpha_list=[1E-4, 1E-3, 1E-2, 1E-1, 1, 10, 100, 1E3, 1E4, 1E5, 1E6, 1E7, 1E8, 1E9],
                                  alpha_per_target=True,
                                  pretrained_Xtransforms={},
                                  verbose=True, use_pca_dual=False):
    """Perform regression using extracted features and neural responses."""
    # Preprocess features
    Xdict, tfm_dict = transform_features2Xdict(feat_dict, layer_names, 
                                    dimred_list, pretrained_Xtransforms, use_pca_dual=use_pca_dual)
    # Define regressors
    regressors = [MultiOutputSeparateLassoCV(cv=5, n_alphas=n_alphas, n_jobs=-1)] # alpha_per_target=alpha_per_target RidgeCV(alphas=alpha_list, alpha_per_target=alpha_per_target)
    regressor_names = ["MultiOutputSeparateLassoCV"]
    result_df, fit_models = sweep_regressors(Xdict, resp_mat, regressors, regressor_names, verbose=verbose)
    return result_df, fit_models, Xdict, tfm_dict


class MultiOutputSeparateLassoCV(BaseEstimator, RegressorMixin):
    """
    Custom estimator that fits a separate LassoCV model for each target 
    in a multi-output regression problem, in sequence.

    Parameters
    ----------
    lasso_cv_params : dict, optional
        Dictionary of parameters passed to the underlying LassoCV estimator.
        Examples include: 'alphas', 'cv', 'max_iter', etc.

    Attributes
    ----------
    models_ : list of LassoCV
        List of fitted LassoCV models, one per output target.
    """
    def __init__(self, **lasso_cv_params):
        # Store any parameters for LassoCV in a dictionary
        self.lasso_cv_params = lasso_cv_params

    def fit(self, X, Y, pbar=True):
        """
        Fit a separate LassoCV model for each column of Y.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input samples.
        Y : array-like of shape (n_samples, n_outputs)
            Training targets for multi-output regression.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X = np.asarray(X)
        Y = np.asarray(Y)
        
        # Make sure Y is 2D
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        
        # Fit one model per output target
        self.models_ = []
        for i in range(Y.shape[1]) if not pbar else tqdm(range(Y.shape[1])):
            model = LassoCV(**self.lasso_cv_params)
            model.fit(X, Y[:, i])
            self.models_.append(model)
        
        return self

    def predict(self, X):
        """
        Predict for each target using the corresponding LassoCV model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        Y_pred : ndarray of shape (n_samples, n_outputs)
            Predicted values for each target.
        """
        X = np.asarray(X)
        # Predict each target, then stack column-wise
        predictions = [model.predict(X) for model in self.models_]
        return np.column_stack(predictions)

    def alpha_per_target(self):
        return [model.alpha_ for model in self.models_]
    
    def coef_per_target(self):
        return [model.coef_ for model in self.models_]
    
    def intercept_per_target(self):
        return [model.intercept_ for model in self.models_]
    
    def alpha_(self):
        return self.alpha_per_target()
    
    def coef_(self):
        return self.coef_per_target()
    
    def intercept_(self):
        return self.intercept_per_target()
    

@th.no_grad()
def record_features(model, fetcher, dataset, batch_size=20, device="cuda"):
    """Record features from the model using the fetcher."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.to(device).eval()
    feat_col = {}
    for imgs, _ in tqdm(loader):
        model(imgs.to(device))
        for key in fetcher.activations.keys():
            if key not in feat_col:
                feat_col[key] = []
            feat_col[key].append(fetcher[key].cpu())
    for key in feat_col.keys():
        feat_col[key] = th.cat(feat_col[key], dim=0)
        print(key, feat_col[key].shape)
    return feat_col


def compute_D2_per_unit(rspavg_resp_peak, rspavg_pred):
    return 1 - np.square(rspavg_resp_peak - rspavg_pred).sum(axis=0) / np.square(rspavg_resp_peak - rspavg_resp_peak.mean(axis=0)).sum(axis=0)

compute_R2_per_unit = compute_D2_per_unit
