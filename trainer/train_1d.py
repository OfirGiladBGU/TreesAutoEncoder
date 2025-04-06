import argparse
import torch
import torch.utils.data
from torch import optim
import copy
import os
import matplotlib.pyplot as plt
import numpy as np
import wandb
from tqdm import tqdm

from datasets.dataset_configurations import V1_1D_DATASETS, IMAGES_6_VIEWS
from datasets.dataset_utils import apply_threshold
from trainer import loss_functions
from trainer import train_utils

# TODO: remove later
import torch.nn.functional as F

class Trainer(object):
    def __init__(self, args: argparse.Namespace, dataset, model):
        self.args = args

        self.device = self.args.device
        self.dataset = dataset
        self.model = model
        self.model.to(self.device)

        # Get loaders
        self.train_loader = self.dataset.train_loader
        self.test_loader = self.dataset.test_loader

        # self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)

    def loss_function(self, output_data, target_data, input_data=None):
        """
        :param output_data: model output on the 'original' input
        :param target_data: the target data that the model should output
        :param input_data: the original input data for the model
        :return:
        """
        LOSS = F.binary_cross_entropy(input=output_data, target=target_data)

        return LOSS

    def _train(self, epoch):
        self.model.train()
        train_loss = 0
        correct = 0
        batch_idx = -1
        for batch_data in self.train_loader:
            batch_idx += 1
            (input_data, target_data) = batch_data

            input_data = input_data.to(self.device)
            target_data = target_data.to(self.device)

            self.optimizer.zero_grad()
            output_data = self.model(input_data)
            loss = self.loss_function(
                output_data=output_data,
                target_data=target_data,
                input_data=input_data
            )
            loss.backward()

            predicted = (output_data > 0.5).float()  # Convert logits to binary (0 or 1)
            correct += (predicted == target_data).sum().item()

            train_loss += loss.item()
            self.optimizer.step()
            if batch_idx % self.args.log_interval == 0:
                print(
                    '[Train] Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {}'.format(
                        epoch,
                        batch_idx * len(input_data),
                        len(self.train_loader.dataset),
                        100. * batch_idx / len(self.train_loader),
                        loss.item() / len(input_data)
                    )
                )

        train_avg_loss = train_loss / len(self.train_loader.dataset)
        train_accuracy = 100. * correct / len(self.train_loader.dataset)
        print(
            '> [Train] Epoch: {}, Average Loss: {}\n'.format(epoch, train_avg_loss) +
            '> [Train] Epoch: {}, Accuracy: {}/{} ({:.0f}%)'.format(
                epoch,
                correct,
                len(self.train_loader.dataset),
                train_accuracy
            )
        )
        return train_avg_loss, train_accuracy

    def _test(self, epoch):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            batch_idx = -1
            for batch_data in self.test_loader:
                batch_idx += 1
                (input_data, target_data) = batch_data

                input_data = input_data.to(self.device)
                target_data = target_data.to(self.device)

                output_data = self.model(input_data)
                test_loss += self.loss_function(
                    output_data=output_data,
                    target_data=target_data,
                    input_data=input_data
                ).item()

                predicted = (output_data > 0.5).float()  # Convert logits to binary (0 or 1)
                correct += (predicted == target_data).sum().item()

        test_avg_loss = test_loss / len(self.test_loader.dataset)
        test_accuracy = 100. * correct / len(self.test_loader.dataset)
        print(
            '> [Test] Epoch: {}, Average Loss: {}\n'.format(epoch, test_avg_loss) +
            '> [Test] Epoch: {}, Accuracy: {}/{} ({:.0f}%)'.format(
                epoch,
                correct,
                len(self.test_loader.dataset),
                test_accuracy
            )
        )
        return test_avg_loss, test_accuracy

    def train(self, use_weights=False):
        self.args.index_data = False
        if use_weights is True:
            print("Loading Model Weights")
            self.model.load_state_dict(torch.load(self.args.weights_filepath))

        print(f"[Model: '{self.model.model_name}'] Training...")
        try:
            for epoch in range(1, self.args.epochs + 1):
                train_avg_loss, train_accuracy = self._train(epoch=epoch)
                test_avg_loss, test_accuracy = self._test(epoch=epoch)
                if self.args.wandb:
                    wandb.log(
                        data={
                            "Train Loss": train_avg_loss, "Test Loss": test_avg_loss,
                            "Train Accuracy": train_accuracy, "Test Accuracy": test_accuracy
                        },
                        step=epoch
                    )
        except (KeyboardInterrupt, SystemExit):
            print("Manual Interruption")

        print("Saving Model Weights")
        model_parameters = copy.deepcopy(self.model.state_dict())
        torch.save(model_parameters, self.args.weights_filepath)

    # TODO: Handle V1 and V2 cases
    def predict(self, max_batches_to_plot=2):
        self.args.index_data = False
        print(f"[Model: '{self.model.model_name}'] Predicting...")
        os.makedirs(name=self.args.results_path, exist_ok=True)

        # Load model weights
        if os.path.exists(self.args.weights_filepath):
            self.model.load_state_dict(torch.load(self.args.weights_filepath))
        self.model.eval()

        with torch.no_grad():
            batches_to_plot = min(len(self.test_loader), max_batches_to_plot)
            z_fill_count = len(str(batches_to_plot))
            batch_idx = -1
            for batch_data in self.test_loader:
                batch_idx += 1
                batch_num = batch_idx + 1
                print(f"[Predict] Batch: {batch_num}/{batches_to_plot}")

                # Get the images from the test loader
                input_data, target_data = batch_data

                input_data = input_data.to(self.device)
                target_data = target_data.to(self.device)
                output_data = self.model(input_data)

                apply_threshold(tensor=output_data, threshold=0.5)

                # Detach the images from the cuda and move them to CPU
                if self.args.cuda is True:
                    input_data = input_data.cpu()
                    target_data = target_data.cpu()
                    output_data = output_data.cpu()

                img_size = (64, 64, 3)  # Image size (H, W, C)

                def create_color_image(value):
                    """Create a red or green image based on value."""
                    color = (0, 255, 0) if value.item() == 1.0 else (255, 0, 0)  # Green for 1, Red for 0
                    img = np.full(img_size, color, dtype=np.uint8)
                    return img

                # Convert `target_data` and `output_data` into colored images
                target_images = [create_color_image(target) for target in target_data]
                output_images = [create_color_image(output) for output in output_data]

                # Convert input images to NumPy for visualization
                input_images = input_data.numpy()

                #################
                # Visualization #
                #################

                plotting_data_list = [
                    {"Title": "Input", "Data": input_images},
                    {"Title": "Target", "Data": target_images},
                    {"Title": "Output", "Data": output_images}
                ]

                # Create a grid of images
                columns = len(plotting_data_list)
                rows = input_data.size(0)
                fig = plt.figure(figsize=(columns + 0.5, rows + 0.5))
                ax = []
                for row_idx in tqdm(range(rows)):
                    for col_idx in range(columns):
                        data: np.array = plotting_data_list[col_idx]["Data"]
                        ax.append(fig.add_subplot(rows, columns, row_idx * columns + col_idx + 1))
                        numpy_image = data[row_idx]

                        if col_idx == 0:
                            plt.imshow(np.transpose(numpy_image, (1, 2, 0)), cmap='gray')
                        else:
                            plt.imshow(numpy_image)

                for col_idx in range(columns):
                    title: str = plotting_data_list[col_idx]["Title"]
                    ax[col_idx].set_title(f"{title}:")

                fig.tight_layout()
                save_filename = os.path.join(self.args.results_path, f"{self.args.dataset}_{batch_num}.png")
                plt.savefig(save_filename)

                # Log the image to wandb
                if self.args.wandb:
                    batch_num_str = str(batch_num).zfill(z_fill_count)
                    wandb.log(
                        data={f"Batch {batch_num_str} - Predict Plots": wandb.Image(plt)}
                    )

                if batch_num == max_batches_to_plot:
                    break
