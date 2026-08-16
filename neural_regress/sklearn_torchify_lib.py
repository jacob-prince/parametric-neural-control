"""
Create torch version of sklearn models in order to pass gradient and feature visualization.
Used in `ManifExp_regression_featvisualize.py`
"""
from sklearn.linear_model import LogisticRegression, LinearRegression, \
    Ridge, Lasso, PoissonRegressor, RidgeCV, LassoCV
from sklearn.random_projection import johnson_lindenstrauss_min_dim, \
            SparseRandomProjection, GaussianRandomProjection
from sklearn.linear_model._base import LinearModel
from sklearn.cross_decomposition import PLSRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
import torch
import numpy as np
from typing import List, Union

import sklearn
import torch.nn as nn
import torch as th

def LinearLayer_from_sklearn(model):
    """Convert a sklearn linear model to a PyTorch Linear layer."""
    if isinstance(model, sklearn.model_selection._search.GridSearchCV):
        model = model.best_estimator_
    assert model.coef_.shape[1] == model.n_features_in_
    readout = nn.Linear(model.coef_.shape[1], model.coef_.shape[0], bias=True)
    readout.weight.data = th.tensor(model.coef_).float()
    readout.bias.data = th.tensor(model.intercept_).float()
    return readout


class SRP_torch(torch.nn.Module):
    def __init__(self, srp: SparseRandomProjection, device="cpu"):
        super(SRP_torch, self).__init__()
        matcoo = srp.components_.tocoo()
        self.register_buffer('components', torch.sparse.FloatTensor(
            torch.LongTensor([matcoo.row.tolist(), matcoo.col.tolist()]),
            torch.FloatTensor(matcoo.data.astype(np.float32))).to(device))

    def forward(self, X):
        if X.ndim > 2:
            X = X.flatten(start_dim=1)
        return torch.sparse.mm(self.components, X.T).T
    
    def to(self, device):
        self.components = self.components.to(device)
        return self


class PCA_torch(torch.nn.Module):
    def __init__(self, pca: PCA, device="cpu"):
        super(PCA_torch, self).__init__()
        self.n_features = pca.n_features_in_
        self.n_components = pca.n_components
        self.register_buffer('mean', torch.from_numpy(pca.mean_).float().to(device))  # (n_features,)
        self.register_buffer('components', torch.from_numpy(pca.components_).float().to(device))  # (n_components, n_features)

    def forward(self, X):
        if X.ndim > 2:
            X = X.flatten(start_dim=1)
        X = X - self.mean
        return torch.mm(X, self.components.T)
    
    def to(self, device):
        self.mean = self.mean.to(device)
        self.components = self.components.to(device)
        return self


class LinearRegression_torch(torch.nn.Module):
    def __init__(self, linear_regression: Union[LinearModel, GridSearchCV], device="cpu"):
        super(LinearRegression_torch, self).__init__()
        if isinstance(linear_regression, GridSearchCV):
            assert isinstance(linear_regression.estimator, LinearModel)
            coef = linear_regression.best_estimator_.coef_
            intercept = linear_regression.best_estimator_.intercept_
        else:
            coef = linear_regression.coef_
            intercept = linear_regression.intercept_
        
        coef = torch.from_numpy(coef).float()
        if coef.ndim == 1:
            coef = coef.unsqueeze(1)
            
        self.register_buffer('coef', coef.to(device))
        self.register_buffer('intercept', torch.tensor(intercept).float().to(device))

    def forward(self, X):
        return torch.mm(X, self.coef) + self.intercept
    
    def to(self, device):
        self.coef = self.coef.to(device)
        self.intercept = self.intercept.to(device)
        return self


class PLS_torch(torch.nn.Module):
    def __init__(self, pls: PLSRegression, device="cpu"):
        super(PLS_torch, self).__init__()
        self.n_components = pls.n_components
        self.n_features = pls.n_features_in_
        self.coef = torch.from_numpy(pls.coef_).to(device)  # (n_components, n_features)
        self.x_mean = torch.from_numpy(pls.x_mean_).to(device)  # (n_features,)
        self.x_std = torch.from_numpy(pls.x_std_).to(device)  # (n_features,)
        self.y_mean = torch.from_numpy(pls.y_mean_).to(device)  # (n_targets,)

    def forward(self, X):
        X = X - self.x_mean
        X = X / self.x_std
        Ypred = torch.mm(X, self.coef)
        return Ypred + self.y_mean
    
    def to(self, device):
        self.coef = self.coef.to(device)
        self.x_mean = self.x_mean.to(device)
        self.x_std = self.x_std.to(device)
        self.y_mean = self.y_mean.to(device)
        return self


class SpatialAvg_torch(torch.nn.Module):
    def __init__(self, device="cpu"):
        super(SpatialAvg_torch, self).__init__()

    def forward(self, X):
        if X.ndim == 4:
            return X.mean(dim=(2, 3))
        elif X.ndim == 3:
            return X.mean(dim=2)
        else:
            return X
    
    def to(self, device):
        return self  # No parameters to move


if __name__ == "__main__":
    import pickle as pkl
    from os.path import join
    saveroot = r"E:\OneDrive - Harvard University\Manifold_NeuralRegress"
    featlayer = ".layer3.Bottleneck5"
    Xtfm_fn = f"{featlayer}_regress_Xtransforms.pkl"
    data = pkl.load(open(join(saveroot, Xtfm_fn), "rb"))
    Animal, Expi = "Alfa", 3
    expdir = join(saveroot, f"{Animal}_{Expi:02d}")
    regr_data = pkl.load(open(join(expdir, f"{featlayer}_regress_models.pkl"), "rb"))
    #%% Test torch modules have same results as sklearn
    X = np.random.rand(10, 1024)
    pls = regr_data[('sp_avg', 'PLS')]
    y_pred = pls.predict(X)
    pls_th = PLS_torch(pls)
    y_pred_th = pls_th(torch.from_numpy(X))
    assert torch.allclose(torch.from_numpy(y_pred), y_pred_th)
    #%%
    X = np.random.rand(10, 1024).astype(np.float32)
    reg = regr_data[('sp_avg', 'Lasso')]
    y_pred = reg.predict(X)
    reg_th = LinearRegression_torch(reg)
    y_pred_th = reg_th(torch.from_numpy(X))
    assert torch.allclose(torch.from_numpy(y_pred[:, np.newaxis]), y_pred_th)
    #%%
    X = np.random.rand(10, 1024).astype(np.float32)
    reg = regr_data[('sp_avg', 'Ridge')]
    y_pred = reg.predict(X)
    reg_th = LinearRegression_torch(reg)
    y_pred_th = reg_th(torch.from_numpy(X))
    assert torch.allclose(torch.from_numpy(y_pred[:, np.newaxis]), y_pred_th)
    #%% PCA
    X = np.random.rand(10, 230400).astype(np.float32)
    pca = data["pca"]
    pca_th = PCA_torch(pca)
    X_red = pca.transform(X)
    X_red_th = pca_th(torch.from_numpy(X))
    assert torch.allclose(torch.from_numpy(X_red), X_red_th)
    #%% SRP
    X = np.random.rand(10, 230400).astype(np.float32)
    srp = data["srp"]
    srp_th = SRP_torch(srp)
    X_red = srp.transform(X)
    X_red_th = srp_th(torch.from_numpy(X))
    assert torch.allclose(torch.from_numpy(X_red).float(), X_red_th, atol=1E-5)
    
    
def eval_regressor(regressor, Xmat, ymat, idx_train, idx_test, target_idx=1):
    """Evaluate a regressor on the given data. speed and score are computed."""
    import time
    if idx_train is None or idx_test is None:
        idx_train, idx_test = train_test_split(np.arange(len(ymat)), test_size=0.2, random_state=42, shuffle=True)
    if target_idx is None:
        target_idx = slice(None)
    start = time.time()
    regressor.fit(Xmat[idx_train], ymat[idx_train, target_idx])
    train_score = regressor.score(Xmat[idx_train], ymat[idx_train, target_idx])
    test_score = regressor.score(Xmat[idx_test], ymat[idx_test, target_idx])
    print(f"train score: {train_score:.3f} test score: {test_score:.3f}")
    if hasattr(regressor, "best_estimator_"):
        coef = regressor.best_estimator_.coef_
    elif hasattr(regressor, "coef_"):
        coef = regressor.coef_
    else:
        raise ValueError("No coef attribute found")
    sparsity = (coef != 0).sum() / coef.size
    print(f"weight sparsity: {sparsity:.3f}")
    print(f"time elapsed for {regressor.__class__.__name__}: {time.time() - start:.3f} sec")
    return regressor, train_score, test_score
