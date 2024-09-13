import argparse
import torch
import torch.utils.data
from torch import optim
import copy
import os
import matplotlib.pyplot as plt
import numpy as np

from datasets.dataset_utils import convert_numpy_to_nii_gz, apply_threshold
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

    def loss_function(self, out, target, original=None):
        """
        :param out: model output on the 'original' input
        :param target: the target data that the model should output
        :param original: the original input data for the model
        :return:
        """

        # LOSS = F.mse_loss(out, target, reduction='sum')
        # LOSS = loss_functions.bce_dice_loss(out, target)
        LOSS = loss_functions.weighted_bce_dice_loss(out, target)
        return LOSS

    def _train(self, epoch):
        self.model.train()
        train_loss = 0
        for batch_idx, batch_data in enumerate(self.train_loader):
            (input_data, target_data) = batch_data
            input_data = input_data.to(self.device)
            target_data = target_data.to(self.device)

            self.optimizer.zero_grad()
            out_data = self.model(input_data)
            loss = self.loss_function(out=out_data, target=target_data, original=input_data)
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

        print('> [Train] Epoch: {}, Average Loss: {:.4f}'.format(
            epoch,
            train_loss / len(self.train_loader.dataset)
        ))

    def _test(self):
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(self.test_loader):
                (input_data, target_data) = batch_data
                input_data = input_data.to(self.device)
                target_data = target_data.to(self.device)

                out_data = self.model(input_data)
                test_loss += self.loss_function(out=out_data, target=target_data, original=input_data).item()

        test_loss /= len(self.test_loader.dataset)
        print('> [Test] Average Loss: {:.4f}'.format(test_loss))

    def train(self):
        print(f"[Model: '{self.model.model_name}'] Training...")
        try:
            for epoch in range(1, self.args.epochs + 1):
                self._train(epoch=epoch)
                self._test()
        except (KeyboardInterrupt, SystemExit):
            print("Manual Interruption")

        print("Saving Model Weights")
        model_parameters = copy.deepcopy(self.model.state_dict())
        torch.save(model_parameters, self.args.weights_filepath)

    def predict(self):
        print(f"[Model: '{self.model.model_name}'] Predicting...")
        os.makedirs(name=self.args.results_path, exist_ok=True)

        # Load model weights
        if os.path.exists(self.args.weights_filepath):
            self.model.load_state_dict(torch.load(self.args.weights_filepath))

        with torch.no_grad():
            for b in range(4):
                # Get the images from the test loader
                batch_num = b + 1
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

                for idx in range(input_data.size(0)):
                    target_data_idx = target_data[idx].squeeze().numpy()
                    output_data_idx = output_data[idx].squeeze().numpy()

                    save_filename = f"{self.args.results_path}/{b}_{idx}_target"
                    convert_numpy_to_nii_gz(
                        numpy_data=target_data_idx,
                        save_filename=save_filename
                    )
                    save_filename = f"{self.args.results_path}/{b}_{idx}_output"
                    convert_numpy_to_nii_gz(
                        numpy_data=output_data_idx,
                        save_filename=save_filename
                    )

                    if self.args.dataset == 'Trees3DV1':
                        # Create a grid of images
                        columns = 6
                        rows = 1
                        fig = plt.figure(figsize=(columns + 0.5, rows + 0.5))
                        ax = []

                        for j in range(columns):
                            ax.append(fig.add_subplot(rows, columns, j + 1))
                            npimg = input_data[idx][j].numpy()
                            plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap='gray')

                            ax[j].set_title(f"View {j}:")

                        fig.tight_layout()
                        plt.savefig(os.path.join(self.args.results_path, f"{b}_{idx}_images.png"))

                        # only the first
                        # exit()

                    elif self.args.dataset in ['Trees3DV2', 'Trees3DV3']:
                        input_data_idx = input_data[idx].squeeze().numpy()
                        save_filename = f"{self.args.results_path}/{b}_{idx}_input"
                        convert_numpy_to_nii_gz(
                            numpy_data=input_data_idx,
                            save_filename=save_filename
                        )
                    else:
                        raise ValueError("Invalid dataset")
