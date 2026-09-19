#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Feature visualization module for neural networks.
This module provides classes and functions for visualizing features of neural network units.
"""

import os
import logging
import pickle
import dill
import numpy as np
import matplotlib.pyplot as plt
from functools import lru_cache
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Union, Tuple, Callable

from horama.fourier_fv import optimization_step, fft_2d_freq, get_fft_scale
from horama.common import recorrelate_colors, standardize

import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as T
from einops import rearrange

# Import models module for model loading
from models import load_model
from transforms_utils import *


class FeatureVisualizer:
    """Feature visualization for neural network units."""

    def __init__(
        self,
        model_name: str,
        readout_path: str,
        xtransform_path: str,
        meta_path: str,
        layer_name: str,
        use_bicubic: bool = False,
        device: str = "cuda",
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the feature visualizer.

        Args:
            model_name: Name of the model to use
            readout_path: Path to the readout weights
            xtransform_path: Path to the feature transform function
            meta_path: Path to the metadata
            layer_name: Name of the layer to visualize
            use_bicubic: Use bicubic interpolation for resizing
            device: Device to use (cuda or cpu)
            logger: Logger instance
        """
        self.model_name = model_name
        self.readout_path = readout_path
        self.xtransform_path = xtransform_path
        self.meta_path = meta_path
        self.layer_name = layer_name
        self.use_bicubic = use_bicubic
        self.device = device
        self.logger = logger or logging.getLogger(__name__)

        # Load model and feature extractor
        self.model, self.preprocess, self.fetcher = self._load_model_and_fetcher()

        # Load readout and transform
        self.readout, self.xtransform, self.meta = self._load_readout_and_transform()

        self.logger.info("Feature visualizer initialized successfully")

    def _load_model_and_fetcher(self):
        """Load the model and feature fetcher."""
        from core.layer_hook_utils import featureFetcher  # was circuit_toolkit.layer_hook_utils

        self.logger.info(f"Loading model: {self.model_name}")

        try:
            # Use the models module to load the model
            model, preprocess = load_model(self.model_name, device=self.device)
            model.requires_grad_(False)

            # Set up feature fetcher
            fetcher = featureFetcher(model, input_size=(3, 224, 224), print_module=False)
            fetcher.record(self.layer_name, ingraph=True, store_device=self.device)

            self.logger.info(f"Model loaded successfully: {self.model_name}")
            return model, preprocess, fetcher

        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise

    def _load_readout_and_transform(self):
        """Load the readout weights and transform function."""
        self.logger.info(f"Loading readout from {self.readout_path}")

        readout = torch.load(self.readout_path, weights_only=False).to(self.device)
        self.logger.info(f"Readout loaded successfully from {self.readout_path}")
        xtransform = torch.load(self.xtransform_path, weights_only=False).to(self.device)
        self.logger.info(f"Transform loaded successfully from {self.xtransform_path}")

        try:
            meta = torch.load(self.meta_path, weights_only=False)
            self.logger.info(f"Meta loaded successfully from {self.meta_path}")
        except Exception as e:
            self.logger.error(f"Error loading meta: {str(e)}")
            meta = None

        return readout, xtransform, meta

    def get_unit_prediction(self, images, unit_id):
        """Get prediction for a specific unit."""
        images = self.preprocess(images)
        self.model(images)
        feat_tsr = self.fetcher[self.layer_name]
        feat_vec = self.xtransform(feat_tsr)
        return self.readout(feat_vec)[:, unit_id].mean()

    def load_meta_statistics(self):
        # throw error if meta is not loaded
        if self.meta is None:
            raise ValueError("Meta statistics not loaded. Please load the meta statistics first.")
        # tfel: q = percetile here
        min_p = self.meta['q01_resp']
        max_p = self.meta['q99_resp']
        min_max = np.stack((min_p, max_p), axis=0)
        assert len(min_max.shape) == 2 and min_max.shape[0] == 2, "min_max should be of shape (2, num_units)"
        return min_max

    def get_all_unit_predictions(self, images_list):
        """
        Get predictions for all units across a list of images.

        Args:
            images_list: List of image tensors

        Returns:
            Array of predictions with shape (num_images, num_units)
        """
        # if the information are in meta just send them
        if self.meta is not None:
            return self.load_meta_statistics()
        else:
            # log that we are using the model
            raise ValueError("Meta statistics not loaded. Please load the meta statistics first.")
            #self.logger.info("!!! WARNING !!! Using model to get predictions")
            preds = []

            for batch_start in range(0, len(images_list), 10):
                batch_end = min(batch_start + 10, len(images_list))
                batch_images = torch.stack(images_list[batch_start:batch_end]).to(self.device)
                batch_images = self.preprocess(batch_images)

                self.model(batch_images)
                feat_tsr = self.fetcher[self.layer_name]
                feat_vec = self.xtransform(feat_tsr)
                predictions = self.readout(feat_vec)

                preds.append(predictions.cpu().detach().numpy())

            return np.vstack(preds)

    def feature_accentuation(
        self,
        unit_id,
        image,
        target_level,
        noise=0.1,
        decay=1.5,
        objective_function=None,
        total_steps=4096,
        learning_rate=3.0,
        image_size=1024,
        model_input_size=224,
        values_range=(0.0, 1.0),
        crops_per_iteration=8,
        box_size=(0.90, 0.95),
        penalty=0.0,
    ):
        """
        Perform feature accentuation for a specific unit.

        Args:
            unit_id: Unit ID to accentuate
            image: Seed image tensor
            target_level: Target activation level
            noise: Noise level for optimization
            decay: Decay power for frequency spectrum
            objective_function: Custom objective function (optional)
            total_steps: Number of optimization steps
            learning_rate: Learning rate for optimization
            image_size: Size of the image during optimization
            model_input_size: Size of the image for model input
            values_range: Range of values for the image
            crops_per_iteration: Number of crops per iteration
            box_size: Range of box sizes for cropping
            penalty: Penalty coefficient

        Returns:
            Tuple of (feature_viz, transparency_accumulator, loss_history, best_loss)
        """
        # Create a unit-specific objective function if none is provided
        if objective_function is None:
            def objective_function(images):
                return self.get_unit_prediction(images, unit_id)

        return self._fa(
            objective_function=objective_function,
            image_seed=image,
            decay_power=decay,
            total_steps=total_steps,
            learning_rate=learning_rate,
            image_size=image_size,
            model_input_size=model_input_size,
            noise=noise,
            values_range=values_range,
            crops_per_iteration=crops_per_iteration,
            box_size=box_size,
            penalty=penalty,
            device=self.device,
            objective_score=target_level
        )

    def save_visualization(self, visualization, filepath):
        """
        Save a visualization to a file.

        Args:
            visualization: Torch tensor to visualize
            filepath: Path to save the file
        """
        # tfel: no longer saving matplot figures
        # save torch tensor & pil
        torch.save(visualization, filepath.replace('.png', '.pt'))
        image = T.ToPILImage()(visualization)
        image.save(filepath)
        self.logger.info(f"Visualization saved to {filepath}")

    # Feature Accentuation implementation
    @lru_cache(maxsize=8)
    def get_color_correlation_svd_sqrt(self, device):
        return torch.tensor(
            [[0.56282854, 0.58447580, 0.58447580],
            [0.19482528, 0.00000000, -0.19482528],
            [0.04329450, -0.10823626, 0.06494176]],
            dtype=torch.float32, device=device
        )

    class SpectrumBackwardFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, spectrum_scaler):
            ctx.save_for_backward(spectrum_scaler)
            return x

        @staticmethod
        def backward(ctx, grad_output):
            (spectrum_scaler,) = ctx.saved_tensors
            g = grad_output * spectrum_scaler[None, :, :]
            return g, None

    class SpectrumBackwardScaler(nn.Module):
        def __init__(self, spectrum_scaler):
            super().__init__()
            self.register_buffer("spectrum_scaler", spectrum_scaler)

        def forward(self, x):
            return FeatureVisualizer.SpectrumBackwardFunction.apply(x, self.spectrum_scaler)

    def fa_preconditionner(self, spectrum, spectrum_backward_scaler, values_range, device):
        """Apply feature accentuation preconditioner."""
        assert spectrum.shape[0] == 3

        spec_scaled = spectrum_backward_scaler(spectrum)

        spatial_image = torch.fft.irfft2(spec_scaled)
        spatial_image = spatial_image - spatial_image.mean()

        color_recorrelated_image = recorrelate_colors(spatial_image, device)

        image = torch.sigmoid(color_recorrelated_image)
        image = image * (values_range[1] - values_range[0]) + values_range[0]

        return image

    def inverse_image_spectrum(self, image, device=None):
        if device is None:
            device = image.device

        img_max = torch.amax(image)
        img_min = torch.amin(image)

        image = (image - img_min) / (img_max - img_min)

        eps = 1e-3
        inv_sigmoid = torch.log((image+eps) / (1.0 - image + eps))

        inv_corr_matrix = torch.linalg.pinv(self.get_color_correlation_svd_sqrt(device))
        inv_sigmoid = rearrange(inv_sigmoid, 'c h w -> h w c')

        inv_recorrelate = torch.matmul(inv_sigmoid.contiguous().view(-1, 3),
                                    inv_corr_matrix)
        inv_recorrelate = rearrange(inv_recorrelate, '(h w) c -> c h w', c=inv_sigmoid.shape[-1], h=inv_sigmoid.shape[0], w=inv_sigmoid.shape[1])
        spectrum = torch.fft.rfft2(inv_recorrelate)

        return spectrum

    def _fa(
        self,
        objective_function,
        image_seed,
        decay_power=1.5,
        total_steps=1000,
        learning_rate=1.0,
        image_size=1280,
        model_input_size=224,
        noise=0.05,
        values_range=(0.0, 1.0),
        crops_per_iteration=6,
        box_size=(0.20, 0.25),
        penalty=1.0,
        device='cuda',
        objective_score=None
    ):
        """
        Perform the Feature Accentuation optimization process.

        Args:
            objective_function: Function to optimize
            image_seed: Seed image for optimization
            decay_power: Decay power for frequency spectrum
            total_steps: Number of optimization steps
            learning_rate: Learning rate for optimization
            image_size: Size of the image during optimization
            model_input_size: Size of the image for model input
            noise: Noise level for optimization
            values_range: Range of values for the image
            crops_per_iteration: Number of crops per iteration
            box_size: Range of box sizes for cropping
            penalty: Penalty coefficient
            device: Device to use
            objective_score: Target score value

        Returns:
            Tuple of (feature_viz, transparency_accumulator, loss_history, best_loss)
        """
        # Validate parameters
        assert values_range[1] >= values_range[0], "Invalid values range"
        assert box_size[1] >= box_size[0], "Invalid box size range"

        # Resize image if needed
        if image_seed.shape[1] != image_size:
            image_seed = torch.nn.functional.interpolate(
                image_seed.unsqueeze(0),
                size=(image_size, image_size),
                mode='bilinear',
                align_corners=False
            )[0]

        # Initialize spectrum
        spectrum = self.inverse_image_spectrum(image_seed, device)
        scaler = get_fft_scale(image_size, image_size, decay_power)

        # Set up spectrum backward scaler
        spectrum_scaler = self.SpectrumBackwardScaler(scaler).to(device)

        # Set up optimization
        spectrum = spectrum.to(device)
        spectrum.requires_grad = True

        optimizer = torch.optim.NAdam([spectrum], lr=learning_rate, betas=(0.95, 0.9))
        transparency_accumulator = torch.zeros((3, image_size, image_size)).to(device)

        # Initial image
        image_seed = self.fa_preconditionner(spectrum, spectrum_scaler, values_range, device).detach()

        # Track optimization progress
        accentuation_loss = []
        best_delta = float('inf')
        best_image = None
        best_loss = None

        # Main optimization loop
        for step_id in tqdm(range(total_steps)):
            optimizer.zero_grad()

            # Generate current image
            image = self.fa_preconditionner(spectrum, spectrum_scaler, values_range, device)

            # Compute loss
            loss, img = optimization_step(
                objective_function,
                image,
                box_size,
                noise,
                crops_per_iteration,
                model_input_size
            )

            # Evaluate on full image
            if self.use_bicubic:
                image_resized = torch.nn.functional.interpolate(
                    img.unsqueeze(0),
                    size=(model_input_size, model_input_size),
                    mode='bicubic',
                    align_corners=False
                )
            else:
                image_resized = img.unsqueeze(0)
            full_image_loss = objective_function(image_resized)

            # Track progress
            accentuation_loss.append(full_image_loss.item())

            # Track best image so far
            if objective_score is not None:
                delta = abs(full_image_loss.item() - objective_score)
                if delta < best_delta:
                    best_delta = delta
                    best_image = img
                    best_loss = full_image_loss.item()

                    # Early stopping if we're very close to target
                    if best_delta < 1e-2:
                        break

                # Adjust loss based on target
                if full_image_loss > objective_score:
                    loss = -0.1 * loss

            # Backward pass
            loss.backward()
            transparency_accumulator += torch.abs(img.grad)

            # Update spectrum
            optimizer.step()

        # Use last image if no best found
        if best_image is None:
            best_image = img
            best_loss = full_image_loss.item()

        return best_image, transparency_accumulator, np.array(accentuation_loss), best_loss