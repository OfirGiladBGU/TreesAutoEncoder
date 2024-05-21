import argparse
import os
import numpy as np
import imageio
from scipy import ndimage
import torch
from torchvision.utils import save_image

from model import Network
from train import Trainer
from utils import get_interpolations


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

    with torch.no_grad():
        images, _ = next(iter(trainer.test_input_loader))
        images = images.to(trainer.device)

        trainer.create_holes(images)

        if images.dtype != torch.float32:
            images = images.float()

        images_per_row = 16
        interpolations = get_interpolations(args, model, trainer.device, images, images_per_row)

        sample = torch.randn(64, args.embedding_size).to(trainer.device)
        sample = model.decode(sample).cpu()

        if args.dataset != 'Trees':
            save_image(sample.view(64, 1, 28, 28),
                       '{}/sample_{}.png'.format(args.results_path, args.dataset))
            save_image(interpolations.view(-1, 1, 28, 28),
                       '{}/interpolations_{}.png'.format(args.results_path, args.dataset), nrow=images_per_row)
            interpolations = interpolations.cpu()
            interpolations = np.reshape(interpolations.data.numpy(), (-1, 28, 28))
            interpolations = ndimage.zoom(interpolations, 5, order=1)
            interpolations *= 256
            imageio.mimsave('{}/animation_{}.gif'.format(args.results_path, args.dataset), interpolations.astype(np.uint8))
        else:
            save_image(sample.view(64, 1, 64, 64),
                       '{}/sample_{}.png'.format(args.results_path, args.dataset))
            save_image(interpolations.view(-1, 1, 64, 64),
                       '{}/interpolations_{}.png'.format(args.results_path, args.dataset), nrow=images_per_row)
            interpolations = interpolations.cpu()
            interpolations = np.reshape(interpolations.data.numpy(), (-1, 64, 64))
            interpolations = ndimage.zoom(interpolations, 5, order=1)
            interpolations *= 256
            imageio.mimsave('{}/animation_{}.gif'.format(args.results_path, args.dataset),
                            interpolations.astype(np.uint8))


def main():
    model = Network(args)

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
    parser.add_argument('--dataset', type=str, default='MNIST', metavar='N',
                        help='Which dataset to use')
    # parser.add_argument('--dataset', type=str, default='Trees', metavar='N',
    #                     help='Which dataset to use')
    parser.add_argument('--weights-filepath', type=str, default='Network.pth', metavar='N',
                        help='Which dataset to use')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    main()
