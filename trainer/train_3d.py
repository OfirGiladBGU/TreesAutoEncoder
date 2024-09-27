import argparse
import torch
import torch.utils.data
from torch import optim
import copy
import os
import matplotlib.pyplot as plt
import numpy as np
import wandb

from datasets.dataset_utils import convert_numpy_to_nii_gz, apply_threshold
from datasets.custom_datasets_3d import V1_3D_DATASETS, V2_3D_DATASETS
from trainer import loss_functions


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

        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

    def loss_function(self, output_data, target_data, input_data=None):
        """
        :param output_data: model output on the 'original' input
        :param target_data: the target data that the model should output
        :param input_data: the original input data for the model
        :return:
        """

        # LOSS = F.mse_loss(out, target, reduction='sum')
        # LOSS = loss_functions.bce_dice_loss(out, target)
        LOSS = loss_functions.weighted_bce_dice_loss(output_data, target_data)
        return LOSS

    def _train(self, epoch):
        self.model.train()
        train_loss = 0
        for batch_idx, batch_data in enumerate(self.train_loader):
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

            train_loss += loss.item()
            self.optimizer.step()
            if batch_idx % self.args.log_interval == 0:
                print('[Train] Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch,
                    batch_idx * len(input_data),
                    len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader),
                    loss.item() / len(input_data)
                ))

        train_avg_loss = train_loss / len(self.train_loader.dataset)
        print('> [Train] Epoch: {}, Average Loss: {:.4f}'.format(
            epoch,
            train_avg_loss
        ))
        return train_avg_loss

    def _test(self):
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(self.test_loader):
                (input_data, target_data) = batch_data
                input_data = input_data.to(self.device)
                target_data = target_data.to(self.device)

                output_data = self.model(input_data)
                test_loss += self.loss_function(
                    output_data=output_data,
                    target_data=target_data,
                    input_data=input_data
                ).item()

        test_avg_loss = test_loss / len(self.test_loader.dataset)
        print('> [Test] Average Loss: {:.4f}'.format(test_avg_loss))
        return test_avg_loss

    def train(self, use_weights=False):
        if use_weights is True:
            print("Loading Model Weights")
            self.model.load_state_dict(torch.load(self.args.weights_filepath))

        print(f"[Model: '{self.model.model_name}'] Training...")
        try:
            for epoch in range(1, self.args.epochs + 1):
                train_avg_loss = self._train(epoch=epoch)
                test_avg_loss = self._test()
                wandb.log(
                    data={"Train Loss": train_avg_loss, "Test Loss": test_avg_loss},
                    step=epoch
                )
        except (KeyboardInterrupt, SystemExit):
            print("Manual Interruption")

        print("Saving Model Weights")
        model_parameters = copy.deepcopy(self.model.state_dict())
        torch.save(model_parameters, self.args.weights_filepath)

    @staticmethod
    def _data_3d_to_2d_plot(data_3d: np.ndarray, save_filename):
        # Downsample the images
        downsample_factor = 1
        data_downsampled = data_3d[::downsample_factor, ::downsample_factor, ::downsample_factor]

        # Get the indices of non-zero values in the downsampled array
        nonzero_indices = np.where(data_downsampled != 0)

        # Plot the cubes based on the non-zero indices
        permutation = [0, 1, 2]

        # Create a figure and a 3D axis
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Get the permutation
        i, j, k = permutation
        ax.bar3d(nonzero_indices[i], nonzero_indices[j], nonzero_indices[k], 1, 1, 1, color='b')

        # Set labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # Display the plot
        permutation_save_filename = f"{save_filename}_({i},{j},{k})"
        plt.title('3d plot')
        plt.savefig(permutation_save_filename)
        plt.close('all')

    def predict(self):
        print(f"[Model: '{self.model.model_name}'] Predicting...")
        os.makedirs(name=self.args.results_path, exist_ok=True)

        # Load model weights
        if os.path.exists(self.args.weights_filepath):
            self.model.load_state_dict(torch.load(self.args.weights_filepath))

        with torch.no_grad():
            batches_to_plot = 4
            for batch_idx in range(batches_to_plot):
                # Get the images from the test loader
                batch_num = batch_idx + 1
                data = iter(self.test_loader)
                for i in range(batch_num):
                    input_data, target_data = next(data)
                input_data = input_data.to(self.device)
                target_data = target_data.to(self.device)

                self.model.eval()
                output_data = self.model(input_data)

                # TODO: Threshold
                apply_threshold(tensor=output_data, threshold=0.1)

                # Detach the images from the cuda and move them to CPU
                if self.args.cuda:
                    input_data = input_data.cpu()
                    target_data = target_data.cpu()
                    output_data = output_data.cpu()

                #################
                # Visualization #
                #################

                # columns = 3
                # rows = input_data.shape[0]
                data_3d_path = os.path.join(f"{self.args.results_path}", "data_3d")
                os.makedirs(data_3d_path, exist_ok=True)
                for idx in range(input_data.size(0)):
                    target_data_idx = target_data[idx].squeeze().numpy()
                    output_data_idx = output_data[idx].squeeze().numpy()

                    save_filename = os.path.join(data_3d_path, f"{batch_num}_{idx}_target")
                    convert_numpy_to_nii_gz(
                        numpy_data=target_data_idx,
                        save_filename=save_filename
                    )
                    save_filename = os.path.join(data_3d_path, f"{batch_num}_{idx}_output")
                    convert_numpy_to_nii_gz(
                        numpy_data=output_data_idx,
                        save_filename=save_filename
                    )

                    if self.args.dataset in V1_3D_DATASETS:
                        # Create a grid of images
                        columns = 6
                        rows = 1
                        fig = plt.figure(figsize=(columns + 0.5, rows + 0.5))
                        ax = []

                        for j in range(columns):
                            ax.append(fig.add_subplot(rows, columns, j + 1))
                            numpy_image = input_data[idx][j].numpy()
                            plt.imshow(np.transpose(numpy_image, (1, 2, 0)), cmap='gray')

                            ax[j].set_title(f"View {j}:")

                        fig.tight_layout()
                        save_filename = os.path.join(self.args.results_path, f"{batch_num}_{idx}_input.png")
                        plt.savefig(save_filename)

                    elif self.args.dataset in V2_3D_DATASETS:
                        input_data_idx = input_data[idx].squeeze().numpy()
                        save_filename = os.path.join(data_3d_path, f"{batch_num}_{idx}_input")
                        convert_numpy_to_nii_gz(
                            numpy_data=input_data_idx,
                            save_filename=save_filename
                        )
                    else:
                        raise ValueError("Invalid dataset")
