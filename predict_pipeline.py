from models.ae_v2_model import Network
from models.ae_3d_v2_model import Network3D


def main():
    # 1. Use model 1 on the `parse_preds_mini_cropped_v5`
    # 2. Save the results in `parse_fixed_mini_cropped_v5`
    # 3. Perform direct `logical or` on `parse_fixed_mini_cropped_v5` to get `parse_prefixed_mini_cropped_3d_v5`
    # 4. Use model 2 on the `parse_prefixed_mini_cropped_3d_v5`
    # 5. Save the results in `parse_fixed_mini_cropped_3d_v5`
    # 6. Run steps 1-5 for mini cubes and combine all the results to get the final result
    pass


if __name__ == "__main__":
    main()