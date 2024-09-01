import argparse
import os
import torch

from trainer.train_3d import Trainer

from models.ae_6_2d_to_3d import MultiView3DReconstruction
from models.ae_3d_to_3d import Network3D


def train_model(model):
    trainer = Trainer(args=args, model=model)
    trainer.train()


def predict_model(model):
    trainer = Trainer(args=args, model=model)
    trainer.predict()


def main():
    # model = MultiView3DReconstruction(args)
    model = Network3D(args)

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
    parser.add_argument('--results-path', type=str, default='./results/model_3d', metavar='N',
                        help='Where to store images')
    parser.add_argument('--dataset', type=str, default='Trees3DV2', metavar='N',
                        help='Which dataset to use')
    parser.add_argument('--weights-filepath', type=str, default='./weights/Network.pth', metavar='N',
                        help='Which dataset to use')

    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")
    torch.manual_seed(args.seed)

    # args.dataset = 'Trees3DV1'
    args.dataset = 'Trees3DV2'

    # args.batch_size = 3
    args.epochs = 10

    if args.dataset == 'Trees3DV1':
        args.input_size = (6, 1, 32, 32)
    if args.dataset == 'Trees3DV2':
        args.input_size = (1, 32, 32, 32)

    main()
