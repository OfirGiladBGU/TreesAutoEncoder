from datasets.dataset_utils import convert_data_file_to_numpy, convert_numpy_to_data_file


data_filepath = r"..\data_results\PipeForge3DPCDCycles\45\45_output.npy"
points_scale = 0.25
voxel_size = 1.0

source_data_filepath = r"..\data\PipeForge3DPCD\originals\45.pcd"

numpy_data = convert_data_file_to_numpy(
    data_filepath=data_filepath,
    points_scale=points_scale,
    voxel_size=voxel_size
)

save_filename = data_filepath.replace(".npy", ".pcd")
convert_numpy_to_data_file(
    numpy_data=numpy_data,
    source_data_filepath=source_data_filepath,
    save_filename=save_filename,
    points_scale=points_scale,
    voxel_size=voxel_size
)