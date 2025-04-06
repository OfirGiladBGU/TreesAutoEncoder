import argparse
import os
import torch
import datetime
import wandb

from datasets_forge.dataset_configurations import (ModelType, DATASET_INPUT_FOLDER, DATASET_OUTPUT_FOLDER,
                                                   MODELS_RESULTS_PATH)
from datasets.dataset_list import init_dataset
from models.model_list import init_model


def configure_wandb(args, model):
    API_KEY = os.environ.get("WANDB_API_KEY")
    wandb.login(key=API_KEY)

    wandb_project = "TreesAutoEncoder"
    init_timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    wandb_name = f"{model.model_name}>{init_timestamp}"
    wandb.init(
        project=wandb_project,
        name=wandb_name,
        tags=[args.dataset, DATASET_INPUT_FOLDER, f"{args.epochs} epochs", f"size {args.input_size}"]
    )


def run_main(args: argparse.Namespace, model_type: ModelType):
    # Import relvant Trainer
    if model_type == ModelType.Model_1D:
        from trainer.train_1d import Trainer
    elif model_type == ModelType.Model_2D:
        from trainer.train_2d import Trainer
    elif model_type == ModelType.Model_3D:
        from trainer.train_3d import Trainer
    else:
        raise ValueError(f"Model Type '{args.model_type}' is not supported.")

    dataset = init_dataset(args=args)
    args.input_size = dataset.input_size

    # TODO: Requires dataset
    model = init_model(args=args)

    if args.wandb:
        configure_wandb(args=args, model=model)

    # Update save path
    filepath, ext = os.path.splitext(args.weights_filepath)
    args.weights_filepath = f"{filepath}_{DATASET_OUTPUT_FOLDER}_{model.model_name}{ext}"
    os.makedirs(name=os.path.dirname(args.weights_filepath), exist_ok=True)

    # Update results path
    args.results_path = os.path.join(MODELS_RESULTS_PATH, model.model_name)

    trainer = Trainer(args=args, dataset=dataset, model=model)

    use_weights = False
    if args.train:
        trainer.train(use_weights=use_weights)
    if args.predict:
        trainer.predict(max_batches_to_plot=args.max_batches_to_plot)

    if args.wandb:
        # wandb.finish()
        pass
