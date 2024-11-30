import os
from datasets.dataset_visulalization import convert_data_file_to_numpy, matplotlib_plot_3d, RESULTS_PATH


def single_plot_3d():
    ########
    # MESH #
    ########
    data_3d_filepath = r"test_mesh.nii.gz"
    numpy_3d_data = convert_data_file_to_numpy(data_filepath=data_3d_filepath)

    save_path = os.path.join(RESULTS_PATH, "single_predict")
    os.makedirs(name=save_path, exist_ok=True)
    save_name = os.path.join(save_path, "mesh")

    matplotlib_plot_3d(data_3d=numpy_3d_data, save_filename=save_name)

    #######
    # PCD #
    #######
    data_3d_filepath = r"test_pcd.nii.gz"
    numpy_3d_data = convert_data_file_to_numpy(data_filepath=data_3d_filepath)

    save_path = os.path.join(RESULTS_PATH, "single_predict")
    os.makedirs(name=save_path, exist_ok=True)
    save_name = os.path.join(save_path, "pcd")

    matplotlib_plot_3d(data_3d=numpy_3d_data, save_filename=save_name, set_aspect_ratios=True)

single_plot_3d()
