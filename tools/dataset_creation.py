import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import cv2
import random
import os


# Utils
def convert_nii_to_numpy(data_file):
    ct_img = nib.load(data_file)
    ct_numpy = ct_img.get_fdata()
    return ct_numpy


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


def save_images(front_image, up_image, left_image, output_idx, output_folder="./"):
    plt.imsave(f"{output_folder}/front_{output_idx}.png", front_image, cmap="gray")
    plt.imsave(f"{output_folder}/up_{output_idx}.png", up_image, cmap="gray")
    plt.imsave(f"{output_folder}/left_{output_idx}.png", left_image, cmap="gray")


def randomly_remove_white_points(image, num_points=10):
    # Find the white pixels
    white_pixels = np.where(image == 255)

    # Randomly select 'num_points' indices
    selected_indices = random.sample(range(len(white_pixels[0])), num_points)

    # Change the selected white pixels to black (0)
    for idx in selected_indices:
        x, y = white_pixels[0][idx], white_pixels[1][idx]
        image[x, y] = 0

    return image


# Src Dataset
def create_dataset_src_images():
    folder_path = "../skel_np"
    data_filepaths = os.listdir(folder_path)

    output_folder = "./cropped_src_images"
    os.makedirs(output_folder, exist_ok=True)

    for data_filepath in data_filepaths:
        output_idx = data_filepath.split(".")[0]
        data_filepath = os.path.join(folder_path, data_filepath)
        ct_numpy = convert_nii_to_numpy(data_file=data_filepath)

        front_image, up_image, left_image = project_3d_to_2d(ct_numpy, apply_cropping=True)
        save_images(
            front_image=front_image,
            up_image=up_image,
            left_image=left_image,
            output_idx=output_idx,
            output_folder=output_folder
        )


# Dst Dataset
def create_dataset_dst_images():
    src_folder = "./cropped_src_images"
    dst_folder = "./cropped_dst_images"

    image_filepaths = os.listdir(src_folder)
    for image_filepath in image_filepaths:
        image_filepath = os.path.join(src_folder, image_filepath)
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        try:
            image = randomly_remove_white_points(image, num_points=150)
        except:  # If the image has less than 150 white points
            image = randomly_remove_white_points(image, num_points=50)

        image_output_filepath = image_filepath.replace(src_folder, dst_folder)
        plt.imsave(image_output_filepath, image, cmap="gray")


def main():
    create_dataset_src_images()
    create_dataset_dst_images()


if __name__ == "__main__":
    # TODO:
    # 1. Make sure "skel_np" folder is present in the root directory
    main()
