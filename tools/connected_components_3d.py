import numpy as np
import nibabel as nib
from scipy.ndimage import label


#########
# Utils #
#########
def convert_nii_to_numpy(data_file):
    ct_img = nib.load(data_file)
    ct_numpy = ct_img.get_fdata()
    return ct_numpy


def convert_numpy_to_nii_gz(numpy_array, save_name="", save=False):
    ct_nii_gz = nib.Nifti1Image(numpy_array, affine=np.eye(4))
    if save and save_name != "":
        nib.save(ct_nii_gz, f"{save_name}.nii.gz")
    return ct_nii_gz


def connected_components_3d(data_3d):
    # Define the structure for connectivity
    # Here, we use a structure that connects each voxel to its immediate neighbors
    structure = np.ones((3, 3, 3), dtype=np.int8)  # 26-connectivity

    # Label connected components
    labeled_array, num_features = label(data_3d, structure=structure)

    print("Labeled Array:")
    print(labeled_array)
    print("Number of features:", num_features)

    convert_numpy_to_nii_gz(labeled_array, "PA000005_components", save=True)


def main():
    data_filepath = r"C:\Users\ofirg\PycharmProjects\TreesAutoEncoder\parse2022\preds\PA000005_vessel.nii.gz"
    ct_numpy = convert_nii_to_numpy(data_file=data_filepath)
    data_3d = ct_numpy
    data_3d[data_3d > 0] = 1
    connected_components_3d(data_3d)


if __name__ == '__main__':
    main()
