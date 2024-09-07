import argparse
import os
import torch

from datasets.dataset_list import init_dataset
from trainer.train_2d import Trainer

# TODO: Rename each model name to be more descriptive
# from models.ae import Network
from models.ae_2d_to_2d import Network
# from models.vgg_ae_demo import Network
# from models.gap_cnn import Network


def train_model(data, model):
    trainer = Trainer(args=args, data=data, model=model)
    trainer.train()


def predict_model(data, model):
    trainer = Trainer(args=args, data=data, model=model)
    trainer.predict()


def main():
    data = init_dataset(args)
    args.input_size = data.input_size

    model = Network(args)

    # Update save path
    filepath, ext = os.path.splitext(args.weights_filepath)
    args.weights_filepath = f"{filepath}_{model.model_name}{ext}"
    os.makedirs(name=os.path.dirname(args.weights_filepath), exist_ok=True)

    # model.load_state_dict(torch.load(args.weights_filepath))

    train_model(data=data, model=model)
    predict_model(data=data, model=model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Main function to call training for different AutoEncoders')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
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
    parser.add_argument('--results-path', type=str, default='./results/model_2d', metavar='N',
                        help='Where to store images')
    parser.add_argument('--dataset', type=str, default='MNIST', metavar='N',
                        help='Which dataset to use')
    parser.add_argument('--weights-filepath', type=str, default='./weights/Network.pth', metavar='N',
                        help='Which dataset to use')

    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")
    torch.manual_seed(args.seed)

    # args.dataset = 'MNIST'
    # args.dataset = 'CIFAR10'
    # args.dataset = 'TreesV0'
    args.dataset = 'TreesV1'
    # args.dataset = 'TreesV2'

    args.epochs = 20

    main()
