import numpy as np
import cv2
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10,ImageNet,ImageFolder
from torch.utils.data import DataLoader, Dataset
import torch
import os
from typing import Any, Callable, Optional, Tuple
from utils.cvdObserver import cvdSimulateNet
from PIL import Image
import os

class CVDcifar(CIFAR10):
    def __init__(        
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        patch_size = 4
    ) -> None:
        super().__init__(root,train,transform,target_transform,download)

        self.image_size = 32
        self.patch_size = patch_size
        self.my_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                # transforms.Resize(self.image_size),
            ]
        )
        self.cvd_observer = cvdSimulateNet('protan')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index] # names from CIFAR10

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        img = self.my_transform(img)
        img_target = img.clone()
        random_index = torch.randint(0,self.image_size//self.patch_size,size=(2,))
        patch_target = img[:, random_index[0]*self.patch_size:random_index[0]*self.patch_size+self.patch_size, 
                           random_index[1]*self.patch_size:random_index[1]*self.patch_size+self.patch_size]
        patch = self.cvd_observer(patch_target)
        img = self.cvd_observer(img)

        return img, patch, img_target, patch_target # CVD image, CVD patch, image target, patch target
    
class CVDImageNet(ImageFolder):
    def __init__(self, root: str, split: str = "train", patch_size=4, **kwargs: Any) -> None:
        target_path = os.path.join(root,split)
        super().__init__(target_path, **kwargs)
        self.image_size = 64
        self.patch_size = patch_size
        self.my_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((self.image_size,self.image_size)),
            ]
        )
        self.cvd_observer = cvdSimulateNet('protan')

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]  # names form ImageNet -> ImageFolder -> DatasetFolder
        sample = self.loader(path)

        img = self.my_transform(sample)
        img_target = img.clone()
        random_index = torch.randint(0,self.image_size//self.patch_size,size=(2,))
        patch_target = img[:, random_index[0]*self.patch_size:random_index[0]*self.patch_size+self.patch_size, 
                           random_index[1]*self.patch_size:random_index[1]*self.patch_size+self.patch_size]
        patch = self.cvd_observer(patch_target)
        img = self.cvd_observer(img)

        return img, patch, img_target, patch_target # CVD image, CVD patch, image target, patch target