import argparse
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib

from models.multi_view_model import MultiView3DReconstruction

from train_3d import Trainer


def train_model(model):
    trainer = Trainer(args, model)
    trainer.train()



def convert_numpy_to_nii_gz(numpy_array, save_name="", save=False):
    ct_nii_gz = nib.Nifti1Image(numpy_array, affine=np.eye(4))
    if save and save_name != "":
        nib.save(ct_nii_gz, f"{save_name}.nii.gz")
    return ct_nii_gz

def predict_model(model):
    try:
        os.stat(args.results_path)
    except:
        os.mkdir(args.results_path)

    if os.path.exists(args.weights_filepath):
        model.load_state_dict(torch.load(args.weights_filepath))
    trainer = Trainer(args=args, model=model)

    with torch.no_grad():
        for b in range(4):
            # Get the images from the test loader
            batch_num = b + 1
            data = iter(trainer.test_loader)
            for i in range(batch_num):
                input_images, target_3d_object = next(data)
            input_images = input_images.to(trainer.device)

            target_3d_object = target_3d_object.to(trainer.device)

            model.eval()
            output_3d_object = model(input_images)

            # TODO: Threshold
            trainer.apply_threshold(output_3d_object, 0.1)

            # Detach the images from the cuda and move them to CPU
            if trainer.args.cuda:
                input_images = input_images.cpu().detach()
                target_3d_object = target_3d_object.cpu().detach()
                output_3d_object = output_3d_object.cpu().detach()

            for idx in range(input_images.size(0)):
                target_3d_object_idx = target_3d_object[idx].squeeze().numpy()
                output_3d_object_idx = output_3d_object[idx].squeeze().numpy()

                convert_numpy_to_nii_gz(numpy_array=target_3d_object_idx,
                                        save_name=f"3d_results/target_{b}_{idx}", save=True)
                convert_numpy_to_nii_gz(numpy_array=output_3d_object_idx,
                                        save_name=f"3d_results/output_{b}_{idx}", save=True)

                # Create a grid of images
                columns = 6
                rows = 1
                fig = plt.figure(figsize=(columns + 0.5, rows + 0.5))
                ax = []

                for j in range(columns):
                    ax.append(fig.add_subplot(rows, columns, j + 1))
                    npimg = input_images[idx][j].numpy()
                    plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap='gray')

                    ax[j].set_title(f"View {j}:")

                fig.tight_layout()
                plt.savefig(os.path.join("3d_results", f"images_{b}_{idx}.png"))

                # only the first
                # exit()


def main():
    model = MultiView3DReconstruction(args)

    # Update save path
    filepath, ext = os.path.splitext(args.weights_filepath)
    args.weights_filepath = f"{filepath}_{model.model_name}{ext}"
    os.makedirs(name=os.path.dirname(args.weights_filepath), exist_ok=True)

    # model.load_state_dict(torch.load(args.weights_filepath))

    train_model(model=model)
    predict_model(model=model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Main function to call training for different AutoEncoders')
    parser.add_argument('--batch-size', type=int, default=21, metavar='N',
                        help='input batch size for training (default: 21)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--embedding-size', type=int, default=32, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--results-path', type=str, default='results/', metavar='N',
                        help='Where to store images')
    parser.add_argument('--dataset', type=str, default='MNIST', metavar='N',
                        help='Which dataset to use')
    parser.add_argument('--weights-filepath', type=str, default='./weights/Network.pth', metavar='N',
                        help='Which dataset to use')

    args = parser.parse_args()
    args.dataset = 'Trees3DV1'

    # args.batch_size = 3
    args.epochs = 3

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")
    torch.manual_seed(args.seed)

    if args.dataset == 'Trees3DV1':
        args.input_size = (6, 1, 32, 32)

    main()
