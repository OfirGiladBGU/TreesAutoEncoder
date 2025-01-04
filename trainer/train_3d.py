import argparse
import torch
import torch.utils.data
from torch import optim
import copy
import os
import matplotlib.pyplot as plt
import numpy as np
import wandb
from PIL import Image, ImageDraw, ImageFont

from datasets.dataset_utils import apply_threshold, IMAGES_6_VIEWS
from datasets.custom_datasets_3d import V1_3D_DATASETS, V2_3D_DATASETS
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
        # LOSS = loss_functions.weighted_bce_dice_loss(output_data, target_data)

        holes_mask = ((target_data - input_data) > 0)  # area that should be filled
        black_mask = (target_data == 0)  # area that should stay black
        LOSS = (0.6 * F.l1_loss(output_data[holes_mask], target_data[holes_mask]) +
                0.4 * F.l1_loss(output_data[black_mask], target_data[black_mask]))

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
                print('[Train] Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {}'.format(
                    epoch,
                    batch_idx * len(input_data),
                    len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader),
                    loss.item() / len(input_data)
                ))

        train_avg_loss = train_loss / len(self.train_loader.dataset)
        print('> [Train] Epoch: {}, Average Loss: {}'.format(
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
        print('> [Test] Average Loss: {}'.format(test_avg_loss))
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

    def predict(self):
        print(f"[Model: '{self.model.model_name}'] Predicting...")
        os.makedirs(name=self.args.results_path, exist_ok=True)

        # Load model weights
        if os.path.exists(self.args.weights_filepath):
            self.model.load_state_dict(torch.load(self.args.weights_filepath))

        with torch.no_grad():
            batches_to_plot = 1
            for batch_idx in range(batches_to_plot):
                # Get the images from the test loader
                batch_num = batch_idx + 1
                data = iter(self.test_loader)
                for _ in range(batch_num):
                    input_data, target_data = next(data)

                input_data = input_data.to(self.device)
                target_data = target_data.to(self.device)

                self.model.eval()
                output_data = self.model(input_data)

                # TODO: Threshold
                apply_threshold(tensor=output_data, threshold=0.1)

                fusion_data = input_data + torch.where(input_data == 0, output_data, 0)

                # Detach the images from the cuda and move them to CPU
                if self.args.cuda:
                    input_data = input_data.cpu()
                    target_data = target_data.cpu()
                    output_data = output_data.cpu()
                    fusion_data = fusion_data.cpu()

                #################
                # Visualization #
                #################

                # Save 3d results and 2d results that will be used for the grid output
                images_info = list()
                data_3d_path = os.path.join(self.args.results_path, "data_3d")
                data_2d_path = os.path.join(self.args.results_path, "data_2d")
                os.makedirs(name=data_3d_path, exist_ok=True)
                os.makedirs(name=data_2d_path, exist_ok=True)
                for idx in range(input_data.size(0)):
                    images_info_idx = dict()

                    # Target
                    target_data_idx = target_data[idx].squeeze().numpy()
                    save_filename_3d = os.path.join(data_3d_path, f"{self.args.dataset}_{batch_num}_{idx}_target")
                    save_filename_2d = os.path.join(data_2d_path, f"{self.args.dataset}_{batch_num}_{idx}_target")
                    np.save(file=save_filename_3d, arr=target_data_idx)
                    # convert_numpy_to_nii_gz(
                    #     numpy_data=target_data_idx,
                    #     save_filename=save_filename_3d
                    # )
                    train_utils.data_3d_to_2d_plot(data_3d=target_data_idx, save_filename=save_filename_2d)
                    images_info_idx["target"] = save_filename_2d

                    # Output
                    output_data_idx = output_data[idx].squeeze().numpy()
                    save_filename_3d = os.path.join(data_3d_path, f"{self.args.dataset}_{batch_num}_{idx}_output")
                    save_filename_2d = os.path.join(data_2d_path, f"{self.args.dataset}_{batch_num}_{idx}_output")
                    np.save(file=save_filename_3d, arr=output_data_idx)
                    # convert_numpy_to_nii_gz(
                    #     numpy_data=output_data_idx,
                    #     save_filename=save_filename_3d
                    # )
                    train_utils.data_3d_to_2d_plot(data_3d=output_data_idx, save_filename=save_filename_2d)
                    images_info_idx["output"] = save_filename_2d

                    # Fusion
                    fusion_data_idx = fusion_data[idx].squeeze().numpy()
                    save_filename_3d = os.path.join(data_3d_path, f"{self.args.dataset}_{batch_num}_{idx}_fusion")
                    save_filename_2d = os.path.join(data_2d_path, f"{self.args.dataset}_{batch_num}_{idx}_fusion")
                    np.save(file=save_filename_3d, arr=fusion_data_idx)
                    # convert_numpy_to_nii_gz(
                    #     numpy_data=fusion_data_idx,
                    #     save_filename=save_filename_3d
                    # )
                    train_utils.data_3d_to_2d_plot(data_3d=fusion_data_idx, save_filename=save_filename_2d)
                    images_info_idx["fusion"] = save_filename_2d

                    # Input
                    save_filename_3d = os.path.join(data_3d_path, f"{self.args.dataset}_{batch_num}_{idx}_input")
                    save_filename_2d = os.path.join(data_2d_path, f"{self.args.dataset}_{batch_num}_{idx}_input")
                    if self.args.dataset in V1_3D_DATASETS:
                        # Create a grid of images
                        columns = 6
                        rows = 1
                        fig = plt.figure(figsize=(columns + 0.5, rows + 0.5))
                        ax = []
                        for view_idx, view_name in enumerate(IMAGES_6_VIEWS):
                            ax.append(fig.add_subplot(rows, columns, view_idx + 1))
                            numpy_image = input_data[idx][view_idx].numpy()
                            plt.imshow(np.transpose(numpy_image, (1, 2, 0)), cmap='gray')
                            ax[view_idx].set_title(f"{view_name} view:")

                        fig.tight_layout()
                        plt.savefig(save_filename_2d)

                    elif self.args.dataset in V2_3D_DATASETS:
                        input_data_idx = input_data[idx].squeeze().numpy()
                        np.save(file=save_filename_3d, arr=input_data_idx)
                        # convert_numpy_to_nii_gz(
                        #     numpy_data=input_data_idx,
                        #     save_filename=save_filename_3d
                        # )
                        train_utils.data_3d_to_2d_plot(data_3d=input_data_idx, save_filename=save_filename_2d)

                    else:
                        raise ValueError("Invalid dataset")

                    images_info_idx["input"] = save_filename_2d
                    images_info.append(images_info_idx)

                # Create a grid of images
                img_width = 0
                img_height = 0
                image_types = ["input", "target", "output", "fusion"]
                for image_type in image_types:
                    image_sample = Image.open(fp=f"{images_info[0][image_type]}.png")
                    img_width = max(img_width, image_sample.size[0])
                    img_height = max(img_height, image_sample.size[1])
                    image_sample.close()

                # Define a font size (can adjust based on preferences)
                font_size = 20

                # Define the text for each column header
                headers = [f"{image_type}:".title() for image_type in image_types]

                # Create a font object (You may need to specify the path to a TTF font on your system)
                try:
                    font = ImageFont.truetype("arial.ttf", font_size)
                except IOError:
                    font = ImageFont.load_default()

                # Number of columns and rows for the grid
                columns = len(headers)
                rows = input_data.shape[0]

                # Create a blank image for the grid, add extra space at the top for the headers
                header_height = font_size + 10  # Extra space for the header
                grid_img_width = columns * img_width
                grid_img_height = rows * img_height + header_height  # Add space for header
                grid_img = Image.new(
                    mode='RGB',
                    size=(grid_img_width, grid_img_height),
                    color=(255, 255, 255)
                )  # White background

                # Create a drawing context for the header
                draw = ImageDraw.Draw(grid_img)

                # Add the headers
                for col in range(columns):
                    header_text = headers[col]

                    # Calculate the bounding box of the text using textbbox
                    text_bbox = draw.textbbox((0, 0), header_text, font=font)
                    text_width = text_bbox[2] - text_bbox[0]

                    # Center text in the column
                    x_position = col * img_width + (img_width - text_width) // 2
                    draw.text((x_position, 5), header_text, font=font, fill="black")  # 5px padding from the top

                # Paste images into the grid (below the header)
                for i in range(rows):
                    j = -1

                    # Load Input image and paste into grid
                    j += 1
                    input_image = Image.open(fp=f"{images_info[i]['input']}.png")
                    grid_img.paste(input_image, (j * img_width, header_height + i * img_height))

                    # Load Target image and paste into grid
                    j += 1
                    target_image = Image.open(fp=f"{images_info[i]['target']}.png")
                    grid_img.paste(target_image, (j * img_width, header_height + i * img_height))

                    # Load Output image and paste into grid
                    j += 1
                    output_image = Image.open(fp=f"{images_info[i]['output']}.png")
                    grid_img.paste(output_image, (j * img_width, header_height + i * img_height))

                    # Load Fusion image and paste into grid
                    j += 1
                    fusion_image = Image.open(fp=f"{images_info[i]['fusion']}.png")
                    grid_img.paste(fusion_image, (j * img_width, header_height + i * img_height))

                # Save the combined image
                save_filename = os.path.join(self.args.results_path, f"{self.args.dataset}_{batch_num}.png")
                grid_img.save(save_filename)

                # Log the image to wandb
                wandb.log(
                    data={f"Batch {batch_num} - Predict Plots": wandb.Image(grid_img)}
                )
