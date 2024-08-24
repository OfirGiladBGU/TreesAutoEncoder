import argparse

from models.ae_v2_model import Network
from models.ae_3d_v2_model import Network3D


def single_predict(image_path, model_path, model_type):
    model_2d = Network()
    model_3d = Network3D()

    # model.load_state_dict(torch.load(model_path))
    # model.eval()
    #
    # # Load the image
    # image = nib.load(image_path)
    # image_data = image.get_fdata()
    #
    # # Predict
    # with torch.no_grad():
    #     input_data = torch.tensor(image_data).unsqueeze(0).unsqueeze(0).float()
    #     output_data = model(input_data)
    #
    # return output_data

def main():
    # 1. Use model 1 on the `parse_preds_mini_cropped_v5`
    # 2. Save the results in `parse_fixed_mini_cropped_v5`
    # 3. Perform direct `logical or` on `parse_fixed_mini_cropped_v5` to get `parse_prefixed_mini_cropped_3d_v5`
    # 4. Use model 2 on the `parse_prefixed_mini_cropped_3d_v5`
    # 5. Save the results in `parse_fixed_mini_cropped_3d_v5`
    # 6. Run steps 1-5 for mini cubes and combine all the results to get the final result
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Main function to call training for different AutoEncoders')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--results-path', type=str, default='results/', metavar='N',
                        help='Where to store images')
    parser.add_argument('--dataset', type=str, default='MNIST', metavar='N',
                        help='Which dataset to use')
    parser.add_argument('--weights-filepath', type=str, default='./weights/Network.pth', metavar='N',
                        help='Which dataset to use')

    main()