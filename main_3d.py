import argparse
import os
import torch
import datetime
import wandb

from datasets.dataset_configurations import DATASET_INPUT_FOLDER, DATASET_OUTPUT_FOLDER, MODELS_RESULTS_PATH
from datasets.dataset_list import init_dataset
from trainer.train_3d import Trainer
from models.model_list import init_model

API_KEY = os.environ.get("WANDB_API_KEY")
wandb.login(key=API_KEY)


def train_model(dataset, model, use_weights: bool):
    trainer = Trainer(args=args, dataset=dataset, model=model)
    trainer.train(use_weights=use_weights)


def predict_model(dataset, model):
    trainer = Trainer(args=args, dataset=dataset, model=model)
    trainer.predict()


def main():
    dataset = init_dataset(args=args)
    args.input_size = dataset.input_size

    model = init_model(args=args)

    wandb_project = "TreesAutoEncoder"
    wandb_name = f"{model.model_name}>{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    wandb.init(
        project=wandb_project,
        name=wandb_name,
        tags=[args.dataset, DATASET_INPUT_FOLDER, f"{args.epochs} epochs", f"size {args.input_size}"]
    )

    # Update save path
    filepath, ext = os.path.splitext(args.weights_filepath)
    args.weights_filepath = f"{filepath}_{DATASET_OUTPUT_FOLDER}_{model.model_name}{ext}"
    os.makedirs(name=os.path.dirname(args.weights_filepath), exist_ok=True)

    # Update results path
    args.results_path = os.path.join(MODELS_RESULTS_PATH, model.model_name)

    use_weights = False
    train_model(dataset=dataset, model=model, use_weights=use_weights)
    predict_model(dataset=dataset, model=model)

    # wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Main function to run 3D models')
    parser.add_argument('--batch-size', type=int, default=20, metavar='N',
                        help='input batch size for training (default: 20)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--embedding-size', type=int, default=32, metavar='N',
                        help='embedding size for the model')
    parser.add_argument('--dataset', type=str, default='Trees3DV2', metavar='N',
                        help='Which dataset to use')
    parser.add_argument('--weights-filepath', type=str, default='./weights/Network.pth', metavar='N',
                        help='Which dataset to use')
    parser.add_argument('--model', type=str, default='ae_3d_to_3d', metavar='N',
                        help='Which model to use')

    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")
    torch.manual_seed(args.seed)

    # Custom Edit:

    # args.model = 'ae_6_2d_to_3d'
    # args.dataset = 'Trees3DV1'

    args.model = 'ae_3d_to_3d'
    # args.dataset = 'Trees3DV2'
    args.dataset = 'Trees3DV3'
    # args.dataset = 'Trees3DV4'

    args.epochs = 10

    # TODO: add connected components head, and use it to take only the "result" largest connected components

    main()
