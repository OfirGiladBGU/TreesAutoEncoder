from datasets.dataset_utils import get_data_file_stem, convert_data_file_to_numpy, convert_numpy_to_data_file


if __name__ == '__main__':
    # Input
    data_filepath = r"..\data_results\PipeForge3DPCDCycles\45\45_output.npy"
    source_data_filepath = r"..\data\PipeForge3DPCD\originals\45.pcd"
    points_scale = 0.25
    voxel_size = 1.0

    numpy_data = convert_data_file_to_numpy(
        data_filepath=data_filepath,
        points_scale=points_scale,
        voxel_size=voxel_size
    )

    input_extention = get_data_file_stem(data_filepath=data_filepath)
    source_extention = get_data_file_stem(data_filepath=source_data_filepath)
    save_filename = data_filepath.replace(input_extention, source_extention)
    convert_numpy_to_data_file(
        numpy_data=numpy_data,
        source_data_filepath=source_data_filepath,
        save_filename=save_filename,
        points_scale=points_scale,
        voxel_size=voxel_size
    )