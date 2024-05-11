import numpy as np
import nibabel as nib
import cv2
import random
import os


#########
# Utils #
#########
def convert_nii_to_numpy(data_file):
    ct_img = nib.load(data_file)
    ct_numpy = ct_img.get_fdata()
    return ct_numpy


def project_3d_to_2d(data_3d):
    # Front projection (XY plane)
    front_image = np.max(data_3d, axis=2)

    # Up projection (XZ plane)
    up_image = np.max(data_3d, axis=1)

    # Left projection (YZ plane)
    left_image = np.max(data_3d, axis=0)

    return front_image, up_image, left_image


def crop_black_area(image):
    # Convert to Threshold image
    image = image.astype(dtype=np.uint8)
    image[image > 0] = 255

    # Find contours
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize variables for the bounding rectangle
    min_x, min_y, max_x, max_y = float('inf'), float('inf'), 0, 0

    # Iterate through contours and find the bounding rectangle
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_x = max(max_x, x + w)
        max_y = max(max_y, y + h)

    # Crop the image using the calculated bounding rectangle
    cropped_image = image[min_y:max_y, min_x:max_x]

    # Print the coordinates of the bounding rectangle
    print(f"Top-left: ({min_x}, {min_y})")
    print(f"Bottom-right: ({max_x}, {max_y})")

    return cropped_image


def crop_randomly(image, crop_size=64):
    # Get image dimensions
    height, width = image.shape

    # Randomly select top-left corner coordinates
    x = np.random.randint(0, width - crop_size)
    y = np.random.randint(0, height - crop_size)

    # Crop the 64x64 area
    cropped_area = image[y:y + crop_size, x:x + crop_size]

    return cropped_area


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


####################
# Original Dataset #
####################
def create_dataset_original_images():
    folder_path = "../skel_np"
    org_folder = "./cropped_original_images"

    os.makedirs(org_folder, exist_ok=True)
    data_filepaths = os.listdir(folder_path)
    for data_filepath in data_filepaths:
        output_idx = data_filepath.split(".")[0]
        data_filepath = os.path.join(folder_path, data_filepath)
        ct_numpy = convert_nii_to_numpy(data_file=data_filepath)

        front_image, up_image, left_image = project_3d_to_2d(ct_numpy)
        front_image = crop_black_area(front_image)
        up_image = crop_black_area(up_image)
        left_image = crop_black_area(left_image)

        cv2.imwrite(f"{org_folder}/front_{output_idx}.png", front_image)
        cv2.imwrite(f"{org_folder}/up_{output_idx}.png", up_image)
        cv2.imwrite(f"{org_folder}/left_{output_idx}.png", left_image)


###############
# Src Dataset #
###############
def create_dataset_src_images():
    random_crops_count = 10
    org_folder = "./cropped_original_images"
    src_folder = "./cropped_src_images"

    os.makedirs(src_folder, exist_ok=True)
    image_filepaths = os.listdir(org_folder)
    for image_filepath in image_filepaths:
        image_filepath = os.path.join(org_folder, image_filepath)
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        for idx in range(random_crops_count):
            image_i = crop_randomly(image, crop_size=64)

            image_output_filepath = image_filepath.replace(org_folder, src_folder)
            filename, ext = os.path.splitext(image_output_filepath)
            image_output_filepath = f"{filename}_{idx}{ext}"
            cv2.imwrite(image_output_filepath, image_i)

        break

###############
# Dst Dataset #
###############
def create_dataset_dst_images():
    src_folder = "./cropped_src_images"
    dst_folder = "./cropped_dst_images"

    os.makedirs(dst_folder, exist_ok=True)
    image_filepaths = os.listdir(src_folder)
    for image_filepath in image_filepaths:
        image_filepath = os.path.join(src_folder, image_filepath)
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        unique, counts = np.unique(image, return_counts=True)
        dict_result = dict(zip(unique, counts))
        num_points = int(dict_result[255] / 2)
        image = randomly_remove_white_points(image, num_points=num_points)

        image_output_filepath = image_filepath.replace(src_folder, dst_folder)
        cv2.imwrite(image_output_filepath, image)


def main():
    # create_dataset_original_images()
    # create_dataset_src_images()
    create_dataset_dst_images()


if __name__ == "__main__":
    # TODO:
    # 1. Make sure "skel_np" folder is present in the root directory
    main()
