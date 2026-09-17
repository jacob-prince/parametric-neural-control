#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main script for running feature visualization with specified parameters.
This script processes unit IDs and generates visualization images.
"""

import os
import sys
import yaml
import logging
import argparse
from datetime import datetime
import traceback
from pathlib import Path
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from transforms_utils import *
from PIL import Image

# Import the feature visualization module
from feature_viz import FeatureVisualizer


def setup_logging(log_dir: str, log_level: int = logging.INFO) -> logging.Logger:
    """
    Set up logging configuration.

    Args:
        log_dir: Directory to store log files
        log_level: Logging level

    Returns:
        Logger instance
    """
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"feature_viz_{timestamp}.log")

    # Create logger
    logger = logging.getLogger("feature_viz")
    logger.setLevel(log_level)

    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)

    # Create formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def load_config(config_path: str) -> dict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Configuration as dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        raise RuntimeError(f"Failed to load config from {config_path}: {str(e)}")


def load_seed_images(image_paths, size=224, stimroot=""):
    """
    Load and preprocess seed images.

    Args:
        image_paths: List of paths to seed images
        size: Size to resize images to

    Returns:
        List of preprocessed torch tensors
    """
    imgs = []
    for p in image_paths:
        try:
            img = Image.open(os.path.join(stimroot, p))
            img = img.convert("RGB")
            img = img.resize((size, size), Image.BICUBIC)
            img = np.array(img).astype(np.float32)
            img = np.transpose(img, (2, 0, 1))
            img = torch.tensor(img / 255.0)
            imgs.append(img)
        except Exception as e:
            print(f"Error loading image {p}: {str(e)}")
    return imgs


def compute_unit_levels(lower, upper, extend_range=0.25, num_levels=11):
    """
    Compute min and max levels for all units based on binxu stastistic.

    Args:
        lower: Lower bound of the unit's activation range
        upper: Upper bound of the unit's activation range
        extend_range: Factor to extend range by (percentage)
        num_levels: Number of levels to compute

    Returns:
        Array of target levels
    """
    # tfel: we use extended range by 2 for the upper bound and 1 for the lower bound
    assert len(lower) == len(upper), "Lower and upper bounds must have the same length"

    bandwidth = upper - lower
    levels = np.linspace(
        lower - extend_range * bandwidth,
        upper + extend_range * bandwidth * 2,
        num_levels
    )
    assert levels.shape == (num_levels, len(upper)), "Levels shape mismatch"

    return levels


def truncate(f, n):
    """
    Truncates/pads a float f to n decimal places without rounding.

    Args:
        f: Float to truncate
        n: Number of decimal places

    Returns:
        Truncated float as string
    """
    s = '{}'.format(f)
    if 'e' in s or 'E' in s:
        return '{0:.{1}f}'.format(f, n)
    i, p, d = s.partition('.')
    return '.'.join([i, (d+'0'*n)[:n]])


def generate_gif(images, filename, output_dir, resize_to=(400, 400), max_colors=256):
    """
    Generate a GIF from a list of images.

    Args:
        images: List of NumPy array images
        filename: Base filename for the saved GIF
        output_dir: Directory to save GIFs
        resize_to: Desired size for the GIF (width, height)
        max_colors: Maximum number of colors in the GIF
    """
    from PIL import Image

    os.makedirs(output_dir, exist_ok=True)

    # Convert NumPy arrays to PIL Images and resize
    pil_images = [
        Image.fromarray(img.astype(np.uint8)).resize(resize_to)
        for img in images
    ]

    frames_forward = pil_images
    frames_backward = pil_images[::-1]
    pause_frame = pil_images[-1]

    # Duplicate pause frame for a longer pause
    pause_duration = 0.5  # Pause duration in seconds
    frame_duration = 0.1  # Frame duration in seconds
    pause_frames = [pause_frame] * int(pause_duration / frame_duration)

    # Combine frames: forward -> pause -> backward
    all_frames = frames_forward + pause_frames + frames_backward

    # Generate durations for all frames
    durations = [int(frame_duration * 1000)] * len(all_frames)

    # Ensure durations match the number of frames
    assert len(durations) == len(all_frames), "Mismatch between frame count and duration list."

    # Quantize each frame to reduce the number of colors
    quantized_frames = [frame.quantize(colors=max_colors) for frame in all_frames]

    # Save GIF using Pillow
    gif_path = os.path.join(output_dir, f"{filename}.gif")
    quantized_frames[0].save(
        gif_path,
        save_all=True,
        append_images=quantized_frames[1:],
        duration=durations,
        loop=0  # Infinite loop
    )
    return gif_path

def organize_results_for_gifs(result_folder):
    """
    Organize visualization results for GIF creation using a flexible filename parser.

    Args:
        result_folder: Folder containing visualization results

    Returns:
        Dictionary of organized results: mosaic[model_name][fitting][unit_id][img_id][target] = (img, score)
    """
    mosaic = {}

    for img_file in os.listdir(result_folder):
        if not img_file.endswith('.png') or 'controversial' in img_file:
            continue

        try:
            parts = img_file.replace('.png', '').split('_')

            # Locate the "unit" keyword as an anchor
            unit_idx = parts.index("unit")
            if unit_idx < 2:
                raise ValueError(f"Invalid filename structure: {img_file}")

            # Parse model name, fitting, unit ID, image ID, target, and score
            model_name = '_'.join(parts[:unit_idx - 1])
            fitting = parts[unit_idx - 1]
            unit_id = int(parts[unit_idx + 1])
            img_id = int(parts[unit_idx + 3])

            level_str = parts[unit_idx + 5]
            score_str = parts[unit_idx + 7]

            target = float(level_str)
            score = float(score_str)

            # Load image
            img_path = os.path.join(result_folder, img_file)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (512, 512))

            # Store in nested dictionary
            mosaic.setdefault(model_name, {}) \
                  .setdefault(fitting, {}) \
                  .setdefault(unit_id, {}) \
                  .setdefault(img_id, {})[target] = (img, score)

        except Exception as e:
            print(f"Error processing {img_file}: {e}")

    return mosaic


def create_gifs_from_results(mosaic, output_dir, logger=None):
    """
    Create GIFs from organized visualization results.

    Args:
        mosaic: Dictionary of organized results
        output_dir: Directory to save GIFs
        logger: Logger instance
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info(f"Creating GIFs in {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    for model_name in mosaic:
        for fitting in mosaic[model_name]:
            for unit_id in mosaic[model_name][fitting]:
                for img_id in mosaic[model_name][fitting][unit_id]:
                    # Get all target levels and sort them
                    targets = list(mosaic[model_name][fitting][unit_id][img_id].keys())
                    targets.sort()

                    # Only create GIF if we have multiple levels
                    if len(targets) <= 1:
                        logger.warning(f"Not enough levels for GIF: {model_name} {fitting} unit {unit_id} image {img_id}")
                        continue

                    # Collect images
                    images = []
                    for target in targets:
                        img, score = mosaic[model_name][fitting][unit_id][img_id][target]
                        # Convert BGR to RGB
                        images.append(img[..., ::-1])

                    try:
                        # Create and save GIF
                        filename = f"{model_name}_{fitting}_unit_{unit_id}_img_{img_id}"
                        gif_path = generate_gif(images, filename, output_dir)
                        logger.info(f"Created GIF at {gif_path}")
                    except Exception as e:
                        logger.error(f"Error creating GIF for {model_name} {fitting} unit {unit_id} image {img_id}: {str(e)}")
                        logger.error(traceback.format_exc())


def main():
    """Main function to run the feature visualization script."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate feature visualization images")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    parser.add_argument("--units", type=str, default=None, help="Comma-separated list of unit IDs to process")
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory to store logs")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--skip_existing", action="store_true", help="Skip existing images")
    parser.add_argument("--gifs_only", action="store_true", help="Only generate GIFs from existing images")
    parser.add_argument("--stimroot", type=str, default="", help="Root directory for loading seed images")
    args = parser.parse_args()

    # Set up logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logger = setup_logging(args.log_dir, log_level)

    logger.info("Starting feature visualization script")


    try:
        # Load configuration
        logger.info(f"Loading configuration from {args.config}")
        config = load_config(args.config)
        # Override unit IDs if specified in command line
        if args.units:
            unit_ids = [int(uid.strip()) for uid in args.units.split(",")]
            logger.info(f"Overriding unit IDs with command line values: {unit_ids}")
        else:
            unit_ids = config.get("unit_ids", [])
            logger.info(f"Using unit IDs from config: {unit_ids}")

        if not unit_ids and not args.gifs_only:
            raise ValueError("No unit IDs specified")

        # Set device
        device = args.device
        logger.info(f"Using device: {device}")

        # Create output directories
        result_folder = config.get("result_folder", f"results_{datetime.now().strftime('%d-%m-%Y')}")
        os.makedirs(result_folder, exist_ok=True)

        hp_tuning_folder = os.path.join(result_folder, "hp_tuning")
        os.makedirs(hp_tuning_folder, exist_ok=True)

        gifs_folder = config.get("gifs_folder", "gifs")
        os.makedirs(gifs_folder, exist_ok=True)

        # Save configuration for reproducibility
        config_save_path = os.path.join(result_folder, f"config_unit_{unit_ids[0]}.yaml")
        with open(config_save_path, 'w') as f:
            yaml.dump(config, f)
        logger.info(f"Saved configuration to {config_save_path}")

        # Generate GIFs only from existing results if requested
        if args.gifs_only:
            logger.info("Generating GIFs from existing results")
            mosaic = organize_results_for_gifs(result_folder)
            create_gifs_from_results(mosaic, gifs_folder, logger)
            logger.info("GIF generation completed")
            return

        use_bicubic = config.get("use_bicubic", False)
        logger.info(f"Using bicubic interpolation: {use_bicubic}")

        # Initialize feature visualizer
        logger.info("Initializing feature visualizer")
        visualizer = FeatureVisualizer(
            model_name=config.get("model_name"),
            readout_path=config.get("readout_path"),
            xtransform_path=config.get("xtransform_path"),
            meta_path=config.get("meta_path"),
            layer_name=config.get("layer_name"),
            use_bicubic=use_bicubic,
            device=device,
            logger=logger
        )

        # Load and preprocess seed images
        logger.info("Loading seed images")
        seed_images = load_seed_images(config.get("seed_image_paths"), stimroot=args.stimroot)
        if not seed_images:
            raise ValueError("No valid seed images found")
        logger.info(f"Loaded {len(seed_images)} seed images")

        # get unit stastistics and compute the bandwidth of each unit
        logger.info("Retrieving unit statistics")
        unit_statistics = visualizer.load_meta_statistics()
        logger.info(f"Unit statistics loaded: {unit_statistics.shape}")
        levels_per_unit = compute_unit_levels(unit_statistics[0, :], unit_statistics[1, :],
                                              extend_range=config.get("extend_range"), num_levels=config.get("num_levels"))
        logger.info(f"Computed levels with shape {levels_per_unit.shape}")
        # show levels of the units id 0
        logger.info(f"Levels for unit 0: {levels_per_unit[:, unit_ids[0]]}")
        # levels per unit is levels x num_units
        # check if all files exist, then quit early. 
        current_files = os.listdir(result_folder)
        all_files_exist = True
        existing_files = []
        for unit_id in unit_ids:
            levels = levels_per_unit[:, unit_id]
            for i_level, level in enumerate(levels):
                for img_id, img in enumerate(seed_images):
                    level_str = truncate(level, 1)
                    file_pattern = f"{config.get('model_name')}_{config.get('fit_method_name')}_unit_{unit_id}_img_{img_id}_level_{level_str}"
                    if any(f.startswith(file_pattern) for f in current_files):
                        logger.info(f"Skipping already processed level {level_str} for unit {unit_id}, image {img_id}")
                        existing_files.append(file_pattern)
                    else:
                        all_files_exist = False
        print(f"{len(existing_files)} files already exist out of {len(unit_ids) * len(seed_images) * len(levels)} total expected.")
        if all_files_exist:
            logger.info("All files already exist, skipping run, exiting")
            return
        # old code to get unit predictions for all images
        # Get unit predictions for all images to calculate levels
        #logger.info("Computing unit activations across seed images")
        #predictions = visualizer.get_all_unit_predictions(seed_images)
        #logger.info(f"Computed activations with shape {predictions.shape}")

        # First, perform hyperparameter tuning if enabled
        hp_tuning_enabled = config.get("hp_tuning", {}).get("enabled", False)

        if hp_tuning_enabled:
            # Use the first unit for hyperparameter tuning
            tuning_unit_id = unit_ids[0]
            logger.info(f"Running hyperparameter tuning with unit {tuning_unit_id}")

            hp_configs = config.get("hp_tuning", {}).get("configs")

            # Run hyperparameter tuning
            best_hp = run_hyperparameter_tuning(
                visualizer=visualizer,
                unit_id=tuning_unit_id,
                seed_images=seed_images[:3],  # Use first 3 images for tuning
                levels=levels_per_unit[:, tuning_unit_id],
                hp_configs=hp_configs,
                result_folder=hp_tuning_folder
            )

            # Save hyperparameter tuning results
            hp_result_path = os.path.join(result_folder, f"hp_tuning_results_unit_{tuning_unit_id}.yaml") # BW added unit id
            with open(hp_result_path, 'w') as f:
                yaml.dump(best_hp, f)

            logger.info(f"Saved hyperparameter tuning results to {hp_result_path}")

            # Use the best hyperparameters for all units
            noise = best_hp["noise"]
            decay = best_hp["decay"]
        else:
            # Use default hyperparameters from config
            fa_hyperparameters = config.get("fa_hyperparameters", {})
            noise = fa_hyperparameters.get("noise")
            decay = fa_hyperparameters.get("decay")

        logger.info(f"Using hyperparameters: noise={noise}, decay={decay}")

        # Get other hyperparameters from config
        total_steps = config.get("fa_hyperparameters", {}).get("total_steps")
        learning_rate = config.get("fa_hyperparameters", {}).get("learning_rate")
        image_size = config.get("fa_hyperparameters", {}).get("image_size")
        crops_per_iteration = config.get("fa_hyperparameters", {}).get("crops_per_iteration")
        box_size = config.get("fa_hyperparameters", {}).get("box_size")

        # Process all units with the optimized hyperparameters
        for unit_id in unit_ids:
            logger.info(f"Processing unit {unit_id}")

            # Compute target levels for this unit
            levels = levels_per_unit[:, unit_id]
            logger.info(f"Target levels for unit {unit_id}: {levels}")

            # Create objective function for this unit
            def objective_function(images, i=unit_id):
                images = visualizer.preprocess(images)
                visualizer.model(images)
                feat_tsr = visualizer.fetcher[visualizer.layer_name]
                feat_vec = visualizer.xtransform(feat_tsr)
                return visualizer.readout(feat_vec)[:, i].mean()

            # Process each seed image
            for img_id, img in enumerate(seed_images):
                logger.info(f"Processing image {img_id} for unit {unit_id}")

                # Check which levels have already been processed
                current_files = os.listdir(result_folder)
                img_results = []

                # Generate visualizations for each target level
                for i_level, level in enumerate(levels):
                    level_str = truncate(level, 1)
                    file_pattern = f"{config.get('model_name')}_{config.get('fit_method_name')}_unit_{unit_id}_img_{img_id}_level_{level_str}"

                    # Check if this level already exists and skip if requested
                    if args.skip_existing and any(f.startswith(file_pattern) for f in current_files):
                        logger.info(f"Skipping already processed level {level_str} for unit {unit_id}, image {img_id}")

                        # Find the existing file to include in GIF generation
                        existing_files = [f for f in current_files if f.startswith(file_pattern)]
                        if existing_files:
                            try:
                                existing_img = cv2.imread(os.path.join(result_folder, existing_files[0]))
                                existing_img = np.array(existing_img)
                                img_results.append((level, existing_img))
                            except Exception as e:
                                logger.error(f"Error loading existing image: {str(e)}")

                        continue

                    logger.info(f"Generating visualization for unit {unit_id}, image {img_id}, level {level}")

                    try:
                        # Generate feature visualization
                        fv, alpha, a_loss, best_loss = visualizer.feature_accentuation(
                            unit_id=unit_id,
                            image=img.to(device),
                            target_level=level,
                            noise=noise,
                            decay=decay,
                            objective_function=objective_function,
                            total_steps=total_steps,
                            learning_rate=learning_rate,
                            image_size=image_size,
                            crops_per_iteration=crops_per_iteration,
                            box_size=box_size
                        )

                        # Save visualization
                        fv = fv.double().detach().cpu()
                        fv_np = fv.numpy()
                        filename = f"{config.get('model_name')}_{config.get('fit_method_name')}_unit_{unit_id}_img_{img_id}_level_{level}_score_{best_loss}.png"
                        filepath = os.path.join(result_folder, filename)

                        # Save as image
                        visualizer.save_visualization(fv, filepath)

                        # Keep track of successful results for GIF generation
                        img_results.append((level, fv_np.transpose(1, 2, 0) * 255))
                        logger.info(f"Saved visualization to {filepath}")

                    except Exception as e:
                        logger.error(f"Error processing unit {unit_id}, image {img_id}, level {level}: {str(e)}")
                        logger.error(traceback.format_exc())

                # Generate GIF for this unit/image if we have results
                if img_results and config.get("generate_gifs", True):
                    img_results.sort(key=lambda x: x[0])  # Sort by level
                    images = [result[1].astype(np.uint8) for result in img_results]

                    try:
                        gif_filename = f"{config.get('model_name')}_{config.get('fit_method_name')}_unit_{unit_id}_img_{img_id}"
                        gif_path = generate_gif(images, gif_filename, gifs_folder)
                        logger.info(f"Saved GIF to {gif_path}")
                    except Exception as e:
                        logger.error(f"Error generating GIF for unit {unit_id}, image {img_id}: {str(e)}")
                        logger.error(traceback.format_exc())

        # Generate comprehensive GIFs from all results
        if config.get("generate_comprehensive_gifs", True):
            logger.info("Generating comprehensive GIFs from all results")
            mosaic = organize_results_for_gifs(result_folder)
            create_gifs_from_results(mosaic, gifs_folder, logger)

        logger.info("Feature visualization completed successfully")

    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)


def run_hyperparameter_tuning(visualizer, unit_id, seed_images, levels, hp_configs, result_folder):
    """
    Run hyperparameter tuning for feature accentuation.

    Args:
        visualizer: Feature visualizer instance
        unit_id: Unit ID to tune for
        seed_images: List of seed images
        levels: Target levels for the unit
        hp_configs: List of hyperparameter configurations
        result_folder: Folder to save results

    Returns:
        Best hyperparameter configuration
    """
    logger = visualizer.logger
    logger.info(f"Starting hyperparameter tuning for unit {unit_id}")
    os.makedirs(result_folder, exist_ok=True)

    # uncomment this if we want to keep only min and max levels to find the best hyperparameters
    #levels = np.array([levels[0], levels[-1]])
    logger.info(f"Target levels for unit {unit_id}: min={levels[0]}, max={levels[-1]}")

    # Test each hyperparameter configuration
    best_config = None
    best_success_rate = 0

    # Number of images to use for tuning
    nb_images = min(3, len(seed_images))

    for config in hp_configs:
        noise = config["noise"]
        decay = config["decay"]
        logger.info(f"Testing configuration: noise={noise}, decay={decay}")

        success = np.zeros((nb_images, len(levels)))

        for img_id in range(nb_images):
            for i_level, level in enumerate(levels):
                try:
                    img = seed_images[img_id].to(visualizer.device)

                    # Create unit-specific objective function
                    def objective_function(images, i=unit_id):
                        images = visualizer.preprocess(images)
                        visualizer.model(images)
                        feat_tsr = visualizer.fetcher[visualizer.layer_name]
                        feat_vec = visualizer.xtransform(feat_tsr)
                        return visualizer.readout(feat_vec)[:, i].mean()

                    # Run feature accentuation
                    fv, alpha, a_loss, best_loss = visualizer.feature_accentuation(
                        unit_id=unit_id,
                        image=img,
                        target_level=level,
                        noise=noise,
                        decay=decay,
                        objective_function=objective_function,
                        total_steps=config.get("total_steps"),
                        learning_rate=config.get("learning_rate"),
                        box_size=config.get("box_size"),
                    )

                    # Check if we successfully reached the target
                    success[img_id, i_level] = np.abs(best_loss - level) < 1e-1

                    # Save visualization and loss curve
                    plt.figure(figsize=(8, 4))
                    plt.subplot(1, 2, 1)
                    plt.imshow(fv.double().detach().cpu().numpy().transpose(1, 2, 0))
                    plt.title(f"Unit {unit_id}, Image {img_id}, Level {level:.2f}")

                    plt.subplot(1, 2, 2)
                    plt.plot(a_loss)
                    plt.hlines(level, 0, len(a_loss), color='green', linestyle='--')
                    plt.hlines(best_loss, 0, len(a_loss), color='grey', linestyle='--')
                    plt.title(f"Loss curve, final value: {best_loss:.4f}")

                    plt.tight_layout()
                    plt.savefig(os.path.join(
                        result_folder,
                        f"hp_unit_{unit_id}_img_{img_id}_level_{level:.2f}_noise_{noise}_decay_{decay}.png"
                    ))
                    plt.close()

                except Exception as e:
                    logger.error(f"Error in HP tuning for unit {unit_id}, image {img_id}, level {level}: {str(e)}")
                    logger.error(traceback.format_exc())

        # Calculate success rate for this configuration
        success_rate = np.mean(success)
        logger.info(f"Success rate for noise={noise}, decay={decay}: {success_rate:.2f}")

        # Track best configuration
        if success_rate > best_success_rate:
            best_success_rate = success_rate
            best_config = {
                "noise": noise,
                "decay": decay,
                "success_rate": float(success_rate)
            }

        # If we have a very good configuration, we can stop
        if success_rate > 0.95:
            logger.info(f"Found excellent configuration with success rate {success_rate:.2f}")
            break

    if best_config is None:
        # Default if no good configuration was found
        best_config = {"noise": 0.1, "decay": 1.5, "success_rate": 0.0}
        logger.warning("No good configuration found, using defaults")
    else:
        logger.info(f"Best hyperparameters: {best_config}")

    return best_config

if __name__ == "__main__":
    main()