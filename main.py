import argparse
import os
import torch

from models.ae_model import Network
# from models.vgg_demo_model import Network
# from models.gap_cnn import Network

from train import Trainer
import matplotlib.pyplot as plt


def train_model(model):
    trainer = Trainer(args, model)
    trainer.train()


def predict_model(model):
    try:
        os.stat(args.results_path)
    except:
        os.mkdir(args.results_path)

    model.load_state_dict(torch.load(args.weights_filepath))
    trainer = Trainer(args=args, model=model)

    # Set size
    image_size = args.input_size[1]

    with torch.no_grad():
        # Get the images from the test loader
        batch_num = 2
        data = iter(trainer.test_loader)
        for i in range(batch_num):
            input_images, target_images = next(data)
        input_images = input_images.to(trainer.device)
        if args.dataset != 'TreesV1':
            target_images = input_images.clone().detach()
        target_images = target_images.to(trainer.device)

        # TODO: Threshold
        # trainer.apply_threshold(input_images, 0.5)
        # trainer.apply_threshold(target_images, 0.5)

        # Fix for Trees dataset - Fixed problem
        # if input_images.dtype != torch.float32:
        #     input_images = input_images.float()
        # if target_images.dtype != torch.float32:
        #     target_images = target_images.float()

        # Create holes in the input images
        if args.dataset != 'TreesV1':
            trainer.create_holes(input_images)

        model.eval()
        output_images = model(input_images)

        # TODO: Threshold
        # trainer.apply_threshold(output_images, 0.5)

        # Detach the images from the cuda and move them to CPU
        if trainer.args.cuda:
            input_images = input_images.cpu().detach()
            target_images = target_images.cpu().detach()
            output_images = output_images.cpu().detach()

        # Create a grid of images
        columns = 3
        rows = 16
        fig = plt.figure(figsize=(columns + 0.5, rows + 0.5))
        ax = []
        for i in range(rows):
            # Input
            ax.append(fig.add_subplot(rows, columns, i * 3 + 1))
            plt.imshow(input_images[i].reshape(image_size, image_size), cmap="gray")

            # Target
            ax.append(fig.add_subplot(rows, columns, i * 3 + 2))
            plt.imshow(target_images[i].reshape(image_size, image_size), cmap="gray")

            # Output
            ax.append(fig.add_subplot(rows, columns, i * 3 + 3))
            plt.imshow(output_images[i].reshape(image_size, image_size), cmap="gray")

        ax[0].set_title("Input:")
        ax[1].set_title("Target:")
        ax[2].set_title("Output:")
        fig.tight_layout()
        plt.savefig(os.path.join(args.results_path, f'output_{args.dataset}_{model.model_name}.png'))


def main():
    model = Network(args)
    # model.load_state_dict(torch.load(args.weights_filepath))

    train_model(model=model)
    predict_model(model=model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Main function to call training for different AutoEncoders')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
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
    # parser.add_argument('--dataset', type=str, default='MNIST', metavar='N',
    #                     help='Which dataset to use')
    parser.add_argument('--dataset', type=str, default='TreesV1', metavar='N',
                        help='Which dataset to use')
    # parser.add_argument('--dataset', type=str, default='TreesV2', metavar='N',
    #                     help='Which dataset to use')
    parser.add_argument('--weights-filepath', type=str, default='Network.pth', metavar='N',
                        help='Which dataset to use')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")
    torch.manual_seed(args.seed)

    if args.dataset in ['MNIST', 'EMNIST', 'FashionMNIST']:
        args.input_size = (1, 28, 28)
    if args.dataset == 'TreesV1':
        args.input_size = (1, 64, 64)
    if args.dataset == 'TreesV2':
        args.input_size = (1, 28, 28)

    main()
