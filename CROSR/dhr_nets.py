import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .weibull_distribution_utils import calculate_openmax, calculate_openmax_threshold

MODEL_WEIGHTS_DIR_NAME = 'model_weights'


class DHRNet(nn.Module):
    def __init__(self, num_classes=10):
        super(DHRNet, self).__init__()

        self.num_classes = num_classes
        self._weibull_params = None

        # Main feature transformation blocks (fl) also known as the backbone layer
        # l1
        self.conv_l1_1 = nn.Conv2d(1, 100, kernel_size=3, stride=1, padding=1)
        self.conv_l1_2 = nn.Conv2d(100, 100, kernel_size=3, stride=1, padding=1)
        self.pool_l1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # l2
        self.conv_l2_1 = nn.Conv2d(100, 100, kernel_size=3, stride=1, padding=1)
        self.conv_l2_2 = nn.Conv2d(100, 100, kernel_size=3, stride=1, padding=1)
        self.pool_l2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # l3
        self.conv_l3 = nn.Conv2d(100, 100, kernel_size=3, stride=1, padding=1)
        self.pool_l3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Latent representation blocks (hl and h_l)
        self.hl1 = nn.Sequential(
            nn.Conv2d(100, 32, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )
        self.hl1_ = nn.Conv2d(32, 100, kernel_size=1, stride=1, padding=0)

        self.hl2 = nn.Sequential(
            nn.Conv2d(100, 32, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )
        self.hl2_ = nn.Conv2d(32, 100, kernel_size=1, stride=1, padding=0)

        self.hl3 = nn.Sequential(
            nn.Conv2d(100, 32, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )
        self.hl3_ = nn.Conv2d(32, 100, kernel_size=1, stride=1, padding=0)

        self.g_l1 = nn.ConvTranspose2d(100, 1, kernel_size=2, stride=2, padding=0)
        self.g_l2 = nn.ConvTranspose2d(100, 100, kernel_size=2, stride=2, padding=0)
        self.g_l3 = nn.ConvTranspose2d(100, 100, kernel_size=2, stride=2, padding=0, output_padding=1)

        # Fully connected layers
        self.fc1 = nn.Linear(100 * 3 * 3, 500)
        self.fc2 = nn.Linear(500, num_classes)

    def set_weibull_params(self, weibull_params):
        self._weibull_params = weibull_params

    def forward(self, x):
        # backbone run
        # First block with lateral connection
        x1 = F.relu(self.conv_l1_1(x))
        x1 = F.relu(self.conv_l1_2(x1))
        x1 = self.pool_l1(x1)
        x1 = F.dropout(x1, p=0.25)

        # Second block with lateral connection
        x2 = F.relu(self.conv_l2_1(x1))
        x2 = F.relu(self.conv_l2_2(x2))
        x2 = self.pool_l2(x2)
        x2 = F.dropout(x2, p=0.25)

        # Third block with lateral connection
        x3 = F.relu(self.conv_l3(x2))
        x3 = self.pool_l3(x3)
        x3 = F.dropout(x3, p=0.25)

        # Flatten and fully connected layers
        x4 = x3.view(-1, 100 * 3 * 3)
        x5 = F.relu(self.fc1(x4))
        y = self.fc2(x5)

        # Latent Representation Blocks
        z3 = self.hl3(x3)
        z2 = self.hl2(x2)
        z1 = self.hl1(x1)

        z1_ = F.adaptive_max_pool2d(z1, (1, 1))
        z2_ = F.adaptive_max_pool2d(z2, (1, 1))
        z3_ = F.adaptive_max_pool2d(z3, (1, 1))

        if self.training:
            h3 = self.hl1_(z3)
            h2 = self.hl1_(z2)
            h1 = self.hl1_(z1)

            g2 = F.relu(self.g_l3(h3))
            g1 = F.relu(self.g_l2(h2 + g2))
            g0 = F.relu(self.g_l1(h1 + g1))

            return y, g0, [z1_, z2_, z3_]

        else:
            if not self._weibull_params:
                raise Exception('No weibull params were set, exiting.')

            combined_logits_latent = [y, z1_, z2_, z3_]
            return calculate_openmax(combined_logits_latent, y, self._weibull_params)


class DHRNetThreshold(nn.Module):
    def __init__(self, num_classes=10, threshold=0.5):
        super(DHRNetThreshold, self).__init__()

        self.num_classes = num_classes
        self._weibull_params = None
        self._threshold = threshold

        # Main feature transformation blocks (fl) also known as the backbone layer
        # l1
        self.conv_l1_1 = nn.Conv2d(1, 100, kernel_size=3, stride=1, padding=1)
        self.conv_l1_2 = nn.Conv2d(100, 100, kernel_size=3, stride=1, padding=1)
        self.pool_l1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # l2
        self.conv_l2_1 = nn.Conv2d(100, 100, kernel_size=3, stride=1, padding=1)
        self.conv_l2_2 = nn.Conv2d(100, 100, kernel_size=3, stride=1, padding=1)
        self.pool_l2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # l3
        self.conv_l3 = nn.Conv2d(100, 100, kernel_size=3, stride=1, padding=1)
        self.pool_l3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Latent representation blocks (hl and h_l)
        self.hl1 = nn.Sequential(
            nn.Conv2d(100, 32, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )
        self.hl1_ = nn.Conv2d(32, 100, kernel_size=1, stride=1, padding=0)

        self.hl2 = nn.Sequential(
            nn.Conv2d(100, 32, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )
        self.hl2_ = nn.Conv2d(32, 100, kernel_size=1, stride=1, padding=0)

        self.hl3 = nn.Sequential(
            nn.Conv2d(100, 32, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )
        self.hl3_ = nn.Conv2d(32, 100, kernel_size=1, stride=1, padding=0)

        self.g_l1 = nn.ConvTranspose2d(100, 1, kernel_size=2, stride=2, padding=0)
        self.g_l2 = nn.ConvTranspose2d(100, 100, kernel_size=2, stride=2, padding=0)
        self.g_l3 = nn.ConvTranspose2d(100, 100, kernel_size=2, stride=2, padding=0, output_padding=1)

        # Fully connected layers
        self.fc1 = nn.Linear(100 * 3 * 3, 500)
        self.fc2 = nn.Linear(500, num_classes)

    def set_weibull_params(self, weibull_params):
        self._weibull_params = weibull_params

    def forward(self, x):
        # backbone run
        # First block with lateral connection
        x1 = F.relu(self.conv_l1_1(x))
        x1 = F.relu(self.conv_l1_2(x1))
        x1 = self.pool_l1(x1)
        x1 = F.dropout(x1, p=0.25)

        # Second block with lateral connection
        x2 = F.relu(self.conv_l2_1(x1))
        x2 = F.relu(self.conv_l2_2(x2))
        x2 = self.pool_l2(x2)
        x2 = F.dropout(x2, p=0.25)

        # Third block with lateral connection
        x3 = F.relu(self.conv_l3(x2))
        x3 = self.pool_l3(x3)
        x3 = F.dropout(x3, p=0.25)

        # Flatten and fully connected layers
        x4 = x3.view(-1, 100 * 3 * 3)
        x5 = F.relu(self.fc1(x4))
        y = self.fc2(x5)

        # Latent Representation Blocks
        z3 = self.hl3(x3)
        z2 = self.hl2(x2)
        z1 = self.hl1(x1)

        z1_ = F.adaptive_max_pool2d(z1, (1, 1))
        z2_ = F.adaptive_max_pool2d(z2, (1, 1))
        z3_ = F.adaptive_max_pool2d(z3, (1, 1))

        if self.training:
            h3 = self.hl1_(z3)
            h2 = self.hl1_(z2)
            h1 = self.hl1_(z1)

            g2 = F.relu(self.g_l3(h3))
            g1 = F.relu(self.g_l2(h2 + g2))
            g0 = F.relu(self.g_l1(h1 + g1))

            return y, g0, [z1_, z2_, z3_]

        else:
            if not self._weibull_params:
                raise Exception('No weibull params were set, exiting.')

            combined_logits_latent = [y, z1_, z2_, z3_]
            return calculate_openmax_threshold(combined_logits_latent, y, self._weibull_params, self._threshold)


class BaselineNet(nn.Module):
    def __init__(self, num_classes=10):
        super(BaselineNet, self).__init__()

        self.num_classes = num_classes

        # Main feature transformation blocks (fl) also known as the backbone layer
        # l1
        self.conv_l1_1 = nn.Conv2d(1, 100, kernel_size=3, stride=1, padding=1)
        self.conv_l1_2 = nn.Conv2d(100, 100, kernel_size=3, stride=1, padding=1)
        self.pool_l1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # l2
        self.conv_l2_1 = nn.Conv2d(100, 100, kernel_size=3, stride=1, padding=1)
        self.conv_l2_2 = nn.Conv2d(100, 100, kernel_size=3, stride=1, padding=1)
        self.pool_l2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # l3
        self.conv_l3 = nn.Conv2d(100, 100, kernel_size=3, stride=1, padding=1)
        self.pool_l3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Fully connected layers
        self.fc1 = nn.Linear(100 * 3 * 3, 500)
        self.fc2 = nn.Linear(500, num_classes)

    def forward(self, x):
        # First block with lateral connection
        x1 = F.relu(self.conv_l1_1(x))
        x1 = F.relu(self.conv_l1_2(x1))
        x1 = self.pool_l1(x1)
        x1 = F.dropout(x1, p=0.25)

        # Second block with lateral connection
        x2 = F.relu(self.conv_l2_1(x1))
        x2 = F.relu(self.conv_l2_2(x2))
        x2 = self.pool_l2(x2)
        x2 = F.dropout(x2, p=0.25)

        # Third block with lateral connection
        x3 = F.relu(self.conv_l3(x2))
        x3 = self.pool_l3(x3)
        x3 = F.dropout(x3, p=0.25)

        # Flatten and fully connected layers
        x4 = x3.view(-1, 100 * 3 * 3)
        x5 = F.relu(self.fc1(x4))
        y = self.fc2(x5)

        return y


def save_DHRnet(net):
    checkpoint = {
        'model_state_dict': net.state_dict(),  # The model's state dict containing weights
        'num_classes': net.num_classes,  # The number of classes
        'weibull_params': net._weibull_params  # The Weibull parameters
    }

    curr_folder = os.getcwd()
    save_folder_path = os.path.join(curr_folder, MODEL_WEIGHTS_DIR_NAME, 'dhrnet_checkpoint.pth')
    torch.save(checkpoint, save_folder_path)


def load_DHRNet():
    net = DHRNet()
    curr_folder = os.getcwd()
    save_folder_path = os.path.join(curr_folder, MODEL_WEIGHTS_DIR_NAME, 'dhrnet_checkpoint.pth')
    # Load the checkpoint
    checkpoint = torch.load(save_folder_path)

    # Restore the model's state dict
    net.load_state_dict(checkpoint['model_state_dict'])

    # Restore the attributes
    net.num_classes = checkpoint['num_classes']
    net.set_weibull_params(checkpoint['weibull_params'])
    return net


def save_baseline_model(baseline_model):
    curr_folder = os.getcwd()
    save_folder_path = os.path.join(curr_folder, MODEL_WEIGHTS_DIR_NAME, 'baseline_checkpoint.pth')
    # Save the model state_dict
    torch.save(baseline_model.state_dict(), save_folder_path)


def load_baseline_model():
    basaeline_model = BaselineNet()
    curr_folder = os.getcwd()
    save_folder_path = os.path.join(curr_folder, MODEL_WEIGHTS_DIR_NAME, 'baseline_checkpoint.pth')
    basaeline_model.load_state_dict(torch.load(save_folder_path))
    return basaeline_model

