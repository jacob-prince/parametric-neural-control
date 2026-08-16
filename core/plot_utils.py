import os.path
import numpy as np
import torch
from os.path import join
from PIL import Image
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid
from core.montage_utils import make_grid_T, make_grid_np
import matplotlib.pyplot as plt
import matplotlib as mpl
import core.colormap_matlab
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

def to_imgrid(img_tsr, *args, **kwargs):
    if type(img_tsr) is list:
        if img_tsr[0].ndim == 4:
            img_tsr = torch.cat(tuple(img_tsr), dim=0)
        elif img_tsr[0].ndim == 3:
            img_tsr = torch.stack(tuple(img_tsr), dim=0)
    PILimg = ToPILImage()(make_grid_T(img_tsr.cpu(), *args, **kwargs))
    return PILimg


def show_imgrid(img_tsr, *args, **kwargs):
    if type(img_tsr) is list:
        if img_tsr[0].ndim == 4:
            img_tsr = torch.cat(tuple(img_tsr), dim=0)
        elif img_tsr[0].ndim == 3:
            img_tsr = torch.stack(tuple(img_tsr), dim=0)
    PILimg = ToPILImage()(make_grid_T(img_tsr.cpu(), *args, **kwargs))
    PILimg.show()
    return PILimg


def save_imgrid(img_tsr, path, *args, **kwargs):
    if type(img_tsr) is list:
        if img_tsr[0].ndim == 4:
            img_tsr = torch.cat(tuple(img_tsr), dim=0)
        elif img_tsr[0].ndim == 3:
            img_tsr = torch.stack(tuple(img_tsr), dim=0)
    PILimg = ToPILImage()(make_grid_T(img_tsr.cpu(), *args, **kwargs))
    PILimg.save(path)
    return PILimg


def save_imgrid_by_row(img_tsr, path, n_row=5, *args, **kwargs):
    """Seperate img_tsr into rows and save them into different png files, with numbering 0-n."""
    if type(img_tsr) is list:
        if img_tsr[0].ndim == 4:
            img_tsr = torch.cat(tuple(img_tsr), dim=0)
        elif img_tsr[0].ndim == 3:
            img_tsr = torch.stack(tuple(img_tsr), dim=0)
    n_total = img_tsr.shape[0]
    row_num = np.ceil(n_total // n_row).astype(int)  # n_total // n_row + 1
    stem, ext = os.path.splitext(path)
    for i in range(row_num):
        PILimg = ToPILImage()(make_grid(img_tsr[i * n_row: (i + 1) * n_row].cpu(), n_row=5, *args, **kwargs))
        PILimg.save(stem + "_" + str(i) + ext)
    return


def saveallforms(figdirs, fignm, figh=None, fmts=("png", "pdf")):
    """Save all forms of a figure in an array of directories."""
    if type(figdirs) is str:
        figdirs = [figdirs]
    if figh is None:
        figh = plt.gcf()
    for figdir in figdirs:
        for sfx in fmts:
            figh.savefig(join(figdir, fignm+"."+sfx), bbox_inches='tight')


def showimg(ax, imgarr, cbar=False, ylabel=None):
    """Show an image in a given axis."""
    pcm = ax.imshow(imgarr)
    ax.set_ylabel(ylabel)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    if cbar:
        plt.colorbar(pcm, ax=ax)
    return pcm


def off_axes(axs):
    for ax in axs:
        ax.axis("off")


def show_image_without_frame(img):
    """matplotlib imshow an image without any frame.
    credit to ChatGPT4"""
    # Hide the axes and frame
    fig, ax = plt.subplots()
    ax.axis('off')
    # Show the image
    ax.imshow(img, aspect='auto')
    # Remove padding and margins from the figure and axes.
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    # Make the figure fit exactly the image size
    dpi = fig.get_dpi()
    fig.set_size_inches(img.shape[1] / dpi, img.shape[0] / dpi)
    # Show the plot window
    plt.show()


def create_image_grid(image_paths, grid_size, image_size=None, padding=0, bg_color=(255, 255, 255)):
    """
    Assembles multiple images into a grid and saves the result.

    Parameters:
    - image_paths: List of file paths to the images.
    - grid_size: Tuple (columns, rows) specifying the grid dimensions.
    - output_path: File path to save the resulting grid image.
    - image_size: Tuple (width, height) specifying the size to which each image should be resized. If None, original sizes are used.
    - padding: Integer specifying the number of pixels between images.
    - bg_color: Tuple specifying the RGB color for the background.
    """
    if isinstance(image_paths, str):
        image_paths = [image_paths]
    elif isinstance(image_paths, list):
        if isinstance(image_paths[0], str):
            images = [Image.open(path) for path in image_paths]
        else:
            images = image_paths
    else:
        raise ValueError("image_paths must be a list of file paths or a single file path")

    # Resize images if image_size is specified
    if image_size:
        images = [img.resize(image_size, Image.Resampling.LANCZOS) for img in images]
        image_width, image_height = image_size
    else:
        # Use the size of the first image
        image_width, image_height = images[0].size

    grid_columns, grid_rows = grid_size

    # Calculate grid dimensions including padding
    grid_width = grid_columns * image_width + (grid_columns - 1) * padding
    grid_height = grid_rows * image_height + (grid_rows - 1) * padding

    # Create a new blank image with the specified background color
    grid_image = Image.new('RGB', (grid_width, grid_height), color=bg_color)

    # Paste images into the grid
    for index, image in enumerate(images):
        row = index // grid_columns
        col = index % grid_columns
        x_offset = col * (image_width + padding)
        y_offset = row * (image_height + padding)
        grid_image.paste(image, (x_offset, y_offset))
    return grid_image
    # Save the resulting grid image
    grid_image.save(output_path)
    print(f"Image grid saved to {output_path}")
