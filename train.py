import torch
import torch.utils.data
from torch import optim
from torch.nn import functional as F
import sys
import copy

from datasets import MNIST, EMNIST, FashionMNIST, TreesDataset


class Trainer(object):
    def __init__(self, args, model):
        self.args = args
        self.device = torch.device("cuda" if args.cuda else "cpu")
        self._init_dataset()

        if isinstance(self.data, TreesDataset):
            self.train_input_loader = self.data.train_input_loader
            self.train_target_loader = self.data.train_target_loader

            self.test_input_loader = self.data.test_input_loader
            self.test_target_loader = self.data.test_target_loader
        else:
            self.train_input_loader = self.data.train_loader
            self.train_target_loader = copy.deepcopy(self.data.train_loader)

            self.test_input_loader = self.data.test_loader
            self.test_target_loader = copy.deepcopy(self.data.test_loader)

        self.model = model
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

    def _init_dataset(self):
        if self.args.dataset == 'MNIST':
            self.data = MNIST(self.args)
        elif self.args.dataset == 'EMNIST':
            self.data = EMNIST(self.args)
        elif self.args.dataset == 'FashionMNIST':
            self.data = FashionMNIST(self.args)
        # Custom Dataset
        elif self.args.dataset == 'Trees':
            self.data = TreesDataset(self.args)
        else:
            print("Dataset not supported")
            sys.exit()

    def loss_function(self, recon_x, x, args):
        if args.dataset != 'Trees':
            BCE = F.binary_cross_entropy(recon_x, x.view(-1, 28 * 28), reduction='sum')
        else:
            BCE = F.binary_cross_entropy(recon_x, x.view(-1, 64 * 64), reduction='sum')
        return BCE

    def _train(self, epoch):
        self.model.train()
        train_loss = 0
        for batch_idx, ((input_data, _), (target_data, _)) in enumerate(zip(self.train_input_loader, self.train_target_loader)):
            input_data = input_data.to(self.device)
            target_data = target_data.to(self.device)

            # Fix for Trees dataset
            if input_data.dtype != torch.float32:
                input_data = input_data.float()
            if target_data.dtype != torch.float32:
                target_data = target_data.float()

            # # For Equality Check
            # res = torch.eq(input_data, target_data)
            # print(res.max())
            # print(res.min())

            self.optimizer.zero_grad()
            recon_batch = self.model(input_data)
            loss = self.loss_function(recon_batch, target_data, self.args)
            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()
            if batch_idx % self.args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(input_data), len(self.train_input_loader.dataset),
                    100. * batch_idx / len(self.train_input_loader),
                    loss.item() / len(input_data)))

        print('====> Epoch: {} Average loss: {:.4f}'.format(
              epoch, train_loss / len(self.train_input_loader.dataset)))

    def _test(self):
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for i, ((input_data, _), (target_data, _)) in enumerate(zip(self.test_input_loader, self.test_target_loader)):
                input_data = input_data.to(self.device)
                target_data = target_data.to(self.device)

                recon_batch = self.model(input_data)
                test_loss += self.loss_function(recon_batch, target_data, self.args).item()

        test_loss /= len(self.test_input_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))

    def train(self):
        try:
            for epoch in range(1, self.args.epochs + 1):
                self._train(epoch)
                self._test()
        except (KeyboardInterrupt, SystemExit):
            print("Manual Interruption")

        model_parameters = copy.deepcopy(self.model.state_dict())
        torch.save(model_parameters, self.args.weights_filepath)
