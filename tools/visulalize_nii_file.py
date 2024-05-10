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


def main():
    data_file = r"C:\Users\Ofir Gilad\PycharmProjects\Auto Encoder\skel_np\PA000005.nii.gz"
    output_idx = 1
    ct_numpy = convert_nii_to_numpy(data_file=data_file)
    plot_3d_data(data_3d=ct_numpy, idx=output_idx)


if __name__ == "__main__":
    main()
