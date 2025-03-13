import numpy as np
import matplotlib.pyplot as plt
import torch
import random
from torchvision.utils import save_image


##################
# Train 2D Utils #
##################
def _zero_out_radius(tensor, point, radius):
    x, y = point[1], point[2]  # Get the coordinates
    for i in range(max(0, x - radius), min(tensor.size(1), x + radius + 1)):
        for j in range(max(0, y - radius), min(tensor.size(2), y + radius + 1)):
            if (i - x) ** 2 + (j - y) ** 2 <= radius ** 2:
                tensor[0, i, j] = 0


def create_2d_holes(input_data: torch.Tensor):
    # input_data: (batch_size, 1, 32, 32)
    for idx in range(len(input_data)):
        white_points = torch.nonzero(input_data[idx] > 0.6)

        if white_points.size(0) > 0:
            # Randomly select one of the non-zero points
            random_point = random.choice(white_points)
            radius = random.randint(3, 5)
            _zero_out_radius(tensor=input_data[idx], point=random_point, radius=radius)

            # plt.imshow(input_data[idx].permute(1, 2, 0))
            # save_image(input_data[idx], 'img1.png')


##################
# Train 3D Utils #
##################
def data_3d_to_2d_plot(data_3d: np.ndarray, save_filename, title: str = None):
    # Downsample the images
    downsample_factor = 1
    data_downsampled = data_3d[::downsample_factor, ::downsample_factor, ::downsample_factor]

    # Get the indices of non-zero values in the downsampled array
    nonzero_indices = np.where(data_downsampled != 0)

    # Plot the cubes based on the non-zero indices
    permutation = [0, 1, 2]

    # Create a figure and a 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Get the permutation
    i, j, k = permutation

    # Check that it is not empty
    if len(nonzero_indices[0]) > 0:
        ax.bar3d(nonzero_indices[i], nonzero_indices[j], nonzero_indices[k], 1, 1, 1, color='b')

        # Set labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # Display the plot
        if title is not None:
            plt.title(title)
        plt.savefig(save_filename)
        plt.close('all')

        return True
    else:
        return False
