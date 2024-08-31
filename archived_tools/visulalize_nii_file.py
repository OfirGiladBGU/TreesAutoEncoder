import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib


def convert_nii_to_numpy(data_file):
    ct_img = nib.load(data_file)
    ct_numpy = ct_img.get_fdata()
    return ct_numpy


def plot_3d_data(data_3d, idx):
    fig_name = f"3d_plot_{idx}"
    print(f"Data shape: {data_3d.shape}")

    # Downsample the images
    downsample_factor = 1
    data_downsampled = data_3d[::downsample_factor, ::downsample_factor, ::downsample_factor]

    # Get the indices of non-zero values in the downsampled array
    nonzero_indices = np.where(data_downsampled != 0)

    # Create a figure and a 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the cubes based on the non-zero indices
    ax.bar3d(nonzero_indices[2], nonzero_indices[0], nonzero_indices[1], 1, 1, 1, color='b')

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Display the plot
    plt.title('3d plot')
    plt.savefig(fig_name)
    plt.close('all')


def project_3d_to_2d(data_3d, apply_cropping=False):
    # Front projection (XY plane)
    front_image = np.max(data_3d, axis=2)

    # Up projection (XZ plane)
    up_image = np.max(data_3d, axis=1)

    # Left projection (YZ plane)
    left_image = np.max(data_3d, axis=0)

    # Apply Cropping
    if apply_cropping:
        front_image = front_image[0:64, 0:64]
        up_image = up_image[90:154, 0:64]
        left_image = left_image[0:64, 0:64]

    return front_image, up_image, left_image


def test_plot_3d_data():
    data_file = "../skel_np/PA000005.nii.gz"
    output_idx = 1
    ct_numpy = convert_nii_to_numpy(data_file=data_file)
    plot_3d_data(data_3d=ct_numpy, idx=output_idx)


def test_project_3d_to_2d():
    data_file = "../skel_np/PA000005.nii.gz"
    output_idx = 1
    ct_numpy = convert_nii_to_numpy(data_file=data_file)

    front_image, up_image, left_image = project_3d_to_2d(ct_numpy, apply_cropping=False)

    output_folder = "./"
    plt.imsave(f"{output_folder}/front_{output_idx}.png", front_image, cmap="gray")
    plt.imsave(f"{output_folder}/up_{output_idx}.png", up_image, cmap="gray")
    plt.imsave(f"{output_folder}/left_{output_idx}.png", left_image, cmap="gray")


def main():
    # test_plot_3d_data()
    test_project_3d_to_2d()


if __name__ == "__main__":
    # TODO:
    # 1. Make sure "skel_np" folder is present in the root directory
    main()
