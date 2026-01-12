import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader as DL
from torchvision import transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import numpy as np


from ml.model import GestureCNN
from ml import config


class Train():

    def __init__(self, model, loss_criterion, optimizer, num_epochs, num_batches, device):
        self.model = model
        self.loss_criterion = loss_criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.num_batches = num_batches
        self.device = device
    
    def set_transformations(self, augmentations_dict, img_size, mean_norm, std_norm):
        """
        Sets train and validation set transformations as fields of the class.
        """
        train_transformations = transforms.Compose(
            [transforms.RandomRotation(degrees=augmentations_dict["ROTATION_DEGREES"]),
            transforms.RandomHorizontalFlip(p=augmentations_dict["HORIZONTAL_FLIP_PROB"]),
            transforms.RandomCrop(size=img_size, padding=augmentations_dict["RANDOM_CROP_PADDING"]),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_norm, std=std_norm)])
        
        val_transformations = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_norm, std=std_norm)
        ])

        self.train_transform = train_transformations
        self.val_transform = val_transformations


