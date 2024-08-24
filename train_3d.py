import torch
import torch.utils.data
from torch import optim
from torch.nn import functional as F
import copy

from datasets import TreesDataset3DV1, TreesDataset3DV2
import loss_functions


class Trainer(object):
    def __init__(self, args, model):
        self.args = args
        self.device = args.device
        self._init_dataset()

        # Get loaders
        self.train_loader = self.data.train_loader
        self.test_loader = self.data.test_loader

        self.model = model
        self.model.to(self.device)
        self.input_size = model.input_size

        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

    def _init_dataset(self):
        if self.args.dataset == 'Trees3DV1':
            self.data = TreesDataset3DV1(self.args)
        elif self.args.dataset == 'Trees3DV2':
            self.data = TreesDataset3DV2(self.args)
        else:
            raise Exception("Dataset not supported")

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

    @staticmethod
    def apply_threshold(tensor, threshold):
        tensor[tensor >= threshold] = 1.0
        tensor[tensor < threshold] = 0.0

    def _train(self, epoch):
        self.model.train()
        train_loss = 0
        for batch_idx, (input_data, target_data) in enumerate(self.train_loader):
            input_data = input_data.to(self.device)
            target_data = target_data.to(self.device)

            # TODO: temp fix
            input_data = input_data.to(torch.float32)
            target_data = target_data.to(torch.float32)

            self.optimizer.zero_grad()
            recon_batch = self.model(input_data)
            loss = self.loss_function(out=recon_batch, target=target_data, original=input_data)
            loss.backward()

            train_loss += loss.item()
            self.optimizer.step()
            if batch_idx % self.args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(input_data), len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader),
                    loss.item() / len(input_data)))

        print('====> Epoch: {} Average loss: {:.4f}'.format(
              epoch, train_loss / len(self.train_loader.dataset)))

    def _test(self):
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for i, (input_data, target_data) in enumerate(self.test_loader):
                input_data = input_data.to(self.device)
                target_data = target_data.to(self.device)

                # TODO: temp fix
                input_data = input_data.to(torch.float32)
                target_data = target_data.to(torch.float32)

                recon_batch = self.model(input_data)
                test_loss += self.loss_function(out=recon_batch, target=target_data, original=input_data).item()

        test_loss /= len(self.test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))

    def train(self):
        try:
            for epoch in range(1, self.args.epochs + 1):
                self._train(epoch)
                self._test()
        except (KeyboardInterrupt, SystemExit):
            print("Manual Interruption")

        print("Saving Model Weights")
        model_parameters = copy.deepcopy(self.model.state_dict())
        torch.save(model_parameters, self.args.weights_filepath)
