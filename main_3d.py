import argparse
import os
import torch

from models.multi_view_model import MultiView3DReconstruction


from train_3d import Trainer
import matplotlib.pyplot as plt
import numpy as np


def train_model(model):
    trainer = Trainer(args, model)
    trainer.train()


# def predict_model(model):
#     try:
#         os.stat(args.results_path)
#     except:
#         os.mkdir(args.results_path)
#
#     if os.path.exists(args.weights_filepath):
#         model.load_state_dict(torch.load(args.weights_filepath))
#     trainer = Trainer(args=args, model=model)
#
#     with torch.no_grad():
#         for b in range(4):
#             # Get the images from the test loader
#             batch_num = b + 1
#             data = iter(trainer.test_loader)
#             for i in range(batch_num):
#                 input_images, target_images = next(data)
#             input_images = input_images.to(trainer.device)
#             if args.dataset != 'TreesV1':
#                 target_images = input_images.clone().detach()
#             target_images = target_images.to(trainer.device)
#
#             model.eval()
#             output_images = model(input_images)
#
#             # TODO: Threshold
#             # trainer.apply_threshold(output_images, 0.5)
#
#             # Detach the images from the cuda and move them to CPU
#             if trainer.args.cuda:
#                 input_images = input_images.cpu().detach()
#                 target_images = target_images.cpu().detach()
#                 output_images = output_images.cpu().detach()
#
#             # Create a grid of images
#             columns = 3
#             rows = 25
#             fig = plt.figure(figsize=(columns + 0.5, rows + 0.5))
#             ax = []
#             for i in range(rows):
#                 # Input
#                 ax.append(fig.add_subplot(rows, columns, i * 3 + 1))
#                 npimg = input_images[i].numpy()
#                 plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap='gray')
#
#                 # Target
#                 ax.append(fig.add_subplot(rows, columns, i * 3 + 2))
#                 npimg = target_images[i].numpy()
#                 plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap='gray')
#
#                 # Output
#                 ax.append(fig.add_subplot(rows, columns, i * 3 + 3))
#                 npimg = output_images[i].numpy()
#                 plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap='gray')
#
#             ax[0].set_title("Input:")
#             ax[1].set_title("Target:")
#             ax[2].set_title("Output:")
#             fig.tight_layout()
#             plt.savefig(os.path.join(args.results_path, f'output_{args.dataset}_{model.model_name}_{b + 1}.png'))


def main():
    model = MultiView3DReconstruction(args)

    # Update save path
    filepath, ext = os.path.splitext(args.weights_filepath)
    args.weights_filepath = f"{filepath}_{model.model_name}{ext}"

    # model.load_state_dict(torch.load(args.weights_filepath))

    train_model(model=model)
    # predict_model(model=model)


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

    args.epochs = 10

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")
    torch.manual_seed(args.seed)

    if args.dataset == 'Trees3DV1':
        args.input_size = (6, 1, 1, 32, 32)

    main()
