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

    def __init__(self, model, loss_criterion, optimizer, num_epochs, batch_size, device):
        self.model = model
        self.loss_criterion = loss_criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.batch_size = batch_size
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

    def set_data_loaders(self, path_to_train, path_to_val):
        """
        Sets the loaders for the training and validation sets and applies transformations.
        """
        training_dataset = ImageFolder(root=path_to_train, transform=self.train_transform)
        validation_dataset = ImageFolder(root=path_to_val, transform=self.val_transform)

        train_loader = DL(dataset=training_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DL(dataset=validation_dataset, batch_size=self.batch_size, shuffle=False)

        self.train_loader = train_loader
        self.val_loader = val_loader

        print(f"\nDataset Loaders defined.")
        print(f"Train Dataset:")
        print(f"Classes: {training_dataset.classes}")
        print(f"Class-to-idx Mapping: {training_dataset.class_to_idx}")
        print(f"Validation Dataset:")
        print(f"Classes: {validation_dataset.classes}")
        print(f"Class-to-idx Mapping: {validation_dataset.class_to_idx}")
    
    


