import numpy as np
import matplotlib.pyplot as plt
import os
import itertools
import cv2
from PIL import Image, ImageDraw, ImageFont
import plotly.graph_objects as go
from skimage import color

from datasets.dataset_configurations import *
from datasets.dataset_utils import convert_data_file_to_numpy


####################
# 3D visualization #
####################

# Interactive Plot 3D
def interactive_plot_3d(data_3d: np.ndarray, version: int = 1, **kwargs):
    """
    Interactive 3D plot using Plotly or Matplotlib
    :param data_3d:
    :param version:
    :param kwargs: set_aspect_ratios, downsample_factor
    :return:
    """
    if version == 1:
        threshold = 0.5 * np.max(data_3d)
        x, y, z = np.where(data_3d > threshold)

        # Create the Plotly 3D scatter plot
        fig = go.Figure(data=[go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(
                size=2,
                color=data_3d[x, y, z],
                colorscale='Viridis',
                opacity=0.6
            )
        )])

        # aspect_ratios = data_3d.shape  # Lengths of each dimension
        fig.update_layout(scene=dict(
            xaxis_title='X-axis',
            yaxis_title='Y-axis',
            zaxis_title='Z-axis',
            # aspectmode="manual",  # Use manual aspect ratios
            # aspectratio=dict(
            #     x=aspect_ratios[0] / max(aspect_ratios),
            #     y=aspect_ratios[1] / max(aspect_ratios),
            #     z=aspect_ratios[2] / max(aspect_ratios),
            # )
        ), title="3D Volume Visualization")

        fig.show()

    elif version == 2:
        # Downsample the images
        downsample_factor = kwargs.get('downsample_factor', 1)
        data_downsampled = data_3d[::downsample_factor, ::downsample_factor, ::downsample_factor]

        # Get the indices of non-zero values in the downsampled array
        nonzero_indices = np.where(data_downsampled != 0)

        # Create a figure and a 3D axis
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Get the permutation
        if data_3d.max() > 1:
            color_mode = True
        else:
            color_mode = False

        array_value = [
            nonzero_indices[0],
            nonzero_indices[1],
            nonzero_indices[2]
        ]
        if color_mode is True:
            color_value = data_downsampled[
                nonzero_indices[0],
                nonzero_indices[1],
                nonzero_indices[2]
            ]
            color_value = color.label2rgb(label=color_value)
            ax.bar3d(*array_value, 1, 1, 1, color=color_value)
        else:
            ax.bar3d(*array_value, 1, 1, 1, color='b')

        # ax.bar3d(nonzero_indices[0], nonzero_indices[1], nonzero_indices[2], 1, 1, 1, color='b')

        # Set labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # Set aspect ratios
        set_aspect_ratios = kwargs.get('set_aspect_ratios', False)
        if set_aspect_ratios is True:
            aspect_ratios = np.array(
                [data_3d.shape[0], data_3d.shape[1], data_3d.shape[2]])  # Use the actual shape of the volume
            ax.set_box_aspect(aspect_ratios)

        # Display the plot
        plt.title('3d plot')
        plt.show()
    else:
        raise ValueError("Invalid version number. Please use either 1 or 2.")


# Matplotlib Plot 3D
def matplotlib_plot_3d(data_3d: np.ndarray, save_filename, set_aspect_ratios=False):
    print(
        f"Save filename: '{save_filename}*'\n"
        f"Data shape: '{data_3d.shape}'\n"
    )
    # Downsample the images
    downsample_factor = 1
    data_downsampled = data_3d[::downsample_factor, ::downsample_factor, ::downsample_factor]

    # Get the indices of non-zero values in the downsampled array
    nonzero_indices = np.where(data_downsampled != 0)

    if data_3d.max() > 1:
        color_mode = True
    else:
        color_mode = False

    # Plot the cubes based on the non-zero indices
    for permutation in list(itertools.permutations(iterable=[0, 1, 2], r=3)):
        # Create a figure and a 3D axis
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Get the permutation
        i, j, k = permutation
        array_value = [
            nonzero_indices[i],
            nonzero_indices[j],
            nonzero_indices[k]
        ]
        if color_mode is True:
            color_value = data_downsampled[
                nonzero_indices[0],
                nonzero_indices[1],
                nonzero_indices[2]
            ]
            color_value = color.label2rgb(label=color_value)
            ax.bar3d(*array_value, 1, 1, 1, color=color_value)
        else:
            ax.bar3d(*array_value, 1, 1, 1, color='b')

        # Set labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # Set aspect ratios
        if set_aspect_ratios:
            aspect_ratios = np.array([data_3d.shape[i],data_3d.shape[j], data_3d.shape[k]])  # Use the actual shape of the volume
            ax.set_box_aspect(aspect_ratios)

        # Display the plot
        permutation_save_filename = f"{save_filename}_({i},{j},{k})"
        plt.title('3d plot')
        plt.savefig(permutation_save_filename)
        plt.close('all')

        # merging_images(save_filename=save_filename)


def merging_images(save_filename):
    # List of saved image filenames and corresponding labels
    permutations = list(itertools.permutations(iterable=[0, 1, 2], r=3))  # Example list of permutations
    image_filenames = [f"{save_filename}_({i},{j},{k}).png" for (i, j, k) in permutations]
    image_labels = [f"Permutation: ({i}, {j}, {k})" for (i, j, k) in permutations]

    # Load all images
    images = [Image.open(img) for img in image_filenames]

    # Optional: Load a font, or use default
    try:
        font = ImageFont.truetype("arial.ttf", 20)  # Specify a TTF font file if available
    except IOError:
        font = ImageFont.load_default()  # Fallback to default font

    # Add text above each image
    labeled_images = []
    for img, label in zip(images, image_labels):
        # Create a new image with space for the text
        text_height = 30  # Height for the text area above the image
        new_img = Image.new('RGB', (img.width, img.height + text_height), (255, 255, 255))  # White background

        # Draw the text
        draw = ImageDraw.Draw(new_img)

        # Use textbbox to get the bounding box of the text (replaces textsize)
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]  # Width of the text
        text_position = ((img.width - text_width) // 2, 5)  # Center the text
        draw.text(text_position, label, font=font, fill=(0, 0, 0))  # Add text in black

        # Paste the original image below the text
        new_img.paste(img, (0, text_height))

        labeled_images.append(new_img)

    # Now, merge the labeled images into a single image
    widths, heights = zip(*(img.size for img in labeled_images))
    total_width = sum(widths)
    max_height = max(heights)

    # Create a blank image to merge all labeled images
    merged_image = Image.new('RGB', (total_width, max_height))

    # Paste each labeled image into the merged image
    x_offset = 0
    for img in labeled_images:
        merged_image.paste(img, (x_offset, 0))
        x_offset += img.width

    # Save the final merged image
    merged_image.save(f"{save_filename}.png")
    merged_image.show()


#####################
# 3D Core functions #
#####################
def single_plot_3d(data_3d_filepath, interactive_mode: bool = False, interactive_version: int = 1):
    numpy_3d_data = convert_data_file_to_numpy(data_filepath=data_3d_filepath)

    if interactive_mode is False:
        save_path = os.path.join(RESULTS_PATH, "single_predict")
        os.makedirs(name=save_path, exist_ok=True)
        save_name = os.path.join(save_path, "result")

        matplotlib_plot_3d(data_3d=numpy_3d_data, save_filename=save_name)
    else:
        interactive_plot_3d(data_3d=numpy_3d_data, version=interactive_version, set_aspect_ratios=True)


# TODO: support for any 3D data
def full_plot_3d(data_3d_basename: str, include_pipeline_results: bool = False):
    folder_paths = {
        "labels_3d": LABELS_3D,
        "labels_3d_reconstruct": LABELS_3D_RECONSTRUCT,

        "preds_3d": PREDS_3D,
        "preds_3d_reconstruct": PREDS_3D_RECONSTRUCT,
        "preds_3d_fusion": PREDS_3D_FUSION,

        "preds_fixed_3d": PREDS_FIXED_3D,
        "preds_fixed_3d_reconstruct": PREDS_FIXED_3D_RECONSTRUCT,
        "preds_fixed_3d_fusion": PREDS_FIXED_3D_FUSION
    }

    for key, value in folder_paths.items():
        save_path = os.path.join(VISUALIZATION_RESULTS_PATH, key)
        os.makedirs(name=save_path, exist_ok=True)
        save_filename = os.path.join(str(save_path), f"{data_3d_basename}")

        # data_3d_filepath = os.path.join(value, f"{data_3d_basename}.nii.gz")
        data_3d_filepath = os.path.join(value, f"{data_3d_basename}.*")
        numpy_3d_data = convert_data_file_to_numpy(data_filepath=data_3d_filepath)

        matplotlib_plot_3d(data_3d=numpy_3d_data, save_filename=save_filename)

    if include_pipeline_results is True:
        folder_path = os.path.join(PREDICT_PIPELINE_RESULTS_PATH, "output_3d")

        result_types = ["input", "output"]

        for result_type in result_types:
            save_path = os.path.join(VISUALIZATION_RESULTS_PATH, f"pipeline_{result_type}")
            os.makedirs(name=save_path, exist_ok=True)
            save_filename = os.path.join(str(save_path), f"{data_3d_basename}")

            # data_3d_filepath = os.path.join(folder_path, f"{data_3d_basename}_{result_type}.nii.gz")
            data_3d_filepath = os.path.join(folder_path, f"{data_3d_basename}_{result_type}.*")
            numpy_3d_data = convert_data_file_to_numpy(data_filepath=data_3d_filepath)

            matplotlib_plot_3d(data_3d=numpy_3d_data, save_filename=save_filename)


####################
# 2D visualization #
####################

# Interactive Plot 2D
def interactive_plot_2d(data_2d: np.ndarray, apply_label2rgb: bool = False):
    """
    Interactive 2D plot using Matplotlib
    :param data_2d:
    :param apply_label2rgb:
    :return:
    """
    if apply_label2rgb:
        colored_data = color.label2rgb(label=data_2d)
    else:
        colored_data = data_2d

    if colored_data.shape == 3:
        plt.imshow(colored_data)
    else:
        plt.imshow(colored_data, cmap='gray')
    plt.show()


# Matplotlib Plot 2D
def matplotlib_plot_2d(save_filepath, data_2d_list):
    columns = 6
    rows = 1
    fig = plt.figure(figsize=(columns + 0.5, rows + 0.5))
    ax = list()

    # 2D Input
    for j in range(columns):
        ax.append(fig.add_subplot(rows, columns, 0 * columns + j + 1))
        numpy_image = data_2d_list[j]
        plt.imshow(numpy_image)
        ax[j].set_title(f"View {IMAGES_6_VIEWS[j]}:")

    fig.tight_layout()
    plt.savefig(save_filepath)
    plt.close(fig)


#####################
# 2D Core functions #
#####################
def single_plot_2d(data_2d_filepath, interactive_mode: bool = False):
    numpy_2d_data = cv2.imread(data_2d_filepath, cv2.IMREAD_GRAYSCALE)

    if interactive_mode is False:
        pass
    else:
        interactive_plot_2d(data_2d=numpy_2d_data)


def full_plot_2d(data_3d_basename: str, plot_components: bool = False):
    folder_paths = {
        "labels_2d": LABELS_2D,
        "preds_2d": PREDS_2D,
        "preds_fixed_2d": PREDS_FIXED_2D,
    }

    if plot_components is True:
        folder_paths.update({
            "preds_components_2d": PREDS_COMPONENTS_2D,
            "preds_fixed_components_2d": PREDS_FIXED_COMPONENTS_2D
        })

    for key, value in folder_paths.items():
        format_of_2d_images = os.path.join(value, f"{data_3d_basename}_<VIEW>.png")
        # Projections 2D
        data_2d_list = list()
        for image_view in IMAGES_6_VIEWS:
            image_path = format_of_2d_images.replace("<VIEW>", image_view)
            image = cv2.imread(image_path)
            data_2d_list.append(image)

        save_path = os.path.join(VISUALIZATION_RESULTS_PATH, key)
        os.makedirs(name=save_path, exist_ok=True)
        save_filepath = os.path.join(str(save_path), f"{data_3d_basename}")
        matplotlib_plot_2d(save_filepath=save_filepath, data_2d_list=data_2d_list)


def main():
    # Single Test #

    # data_3d_filepath = os.path.join(RESULTS_PATH, "predict_pipeline", "output_3d", "PA000005_05352_input.nii.gz")
    # data_3d_filepath = os.path.join(TRAIN_CROPPED_PATH, "labels_3d_v6", "PA000005_05352.nii.gz")
    # data_3d_filepath = os.path.join(TRAIN_CROPPED_PATH, "preds_3d_v6", "PA000005_05352.nii.gz")
    # data_3d_filepath = os.path.join(DATA_PATH, "Pipes3DGeneratorTree", "labels", "01.npy")

    # Example 3D data
    # data_3d_filepath = os.path.join(TRAIN_CROPPED_PATH, "labels_3d_v6", "PA000078_11978.nii.gz")
    # data_3d_filepath = os.path.join(TRAIN_CROPPED_PATH, "preds_fixed_3d_v6", "PA000078_11978.nii.gz")
    data_3d_filepath = os.path.join(TRAIN_CROPPED_PATH, "preds_components_3d_v6", "PA000078_2020.nii.gz")
    single_plot_3d(data_3d_filepath=data_3d_filepath, interactive_mode=False, interactive_version=1)


    # Multiple Tests #

    # data_3d_basename = "PA000005_11899"
    # data_3d_basename = "PA000005_09039"
    # data_3d_basename = "PA000005_10017"
    # data_3d_basename = "PA000078_2020"

    # full_plot_3d(data_3d_basename=data_3d_basename)
    # full_plot_3d(data_3d_basename=data_3d_basename, include_pipeline_results=True)
    # full_plot_2d(data_3d_basename=data_3d_basename, plot_components=False)


if __name__ == "__main__":
    main()
