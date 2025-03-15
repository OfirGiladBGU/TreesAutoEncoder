from datasets.dataset_utils import get_data_file_extention, convert_data_file_to_numpy, convert_numpy_to_data_file


if __name__ == '__main__':
    # Input
    data_filepath = r".\46_009_output.npy"
    source_data_filepath = r"dummy.pcd"

    src_kwargs = dict(
        # Mesh
        # mesh_scale=1.0, voxel_size=1.0,

        # PCD
        points_scale=1.0, voxel_size=1.0
    )

    dst_kwargs = dict(
        # Mesh
        # mesh_scale=0.5, voxel_size=2.0

        # PCD
        points_scale=0.25, voxel_size=2.0
    )

    numpy_data = convert_data_file_to_numpy(
        data_filepath=data_filepath,
        **src_kwargs
    )

    input_extention = get_data_file_extention(data_filepath=data_filepath)
    source_extention = get_data_file_extention(data_filepath=source_data_filepath)
    save_filename = data_filepath.replace(input_extention, source_extention)
    convert_numpy_to_data_file(
        numpy_data=numpy_data,
        source_data_filepath=source_data_filepath,
        save_filename=save_filename,
        **dst_kwargs
    )
