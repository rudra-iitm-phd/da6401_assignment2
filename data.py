from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, ConcatDataset
import shared
import torch
import matplotlib.pyplot as plt
from copy import deepcopy
from torchvision.datasets import DatasetFolder
from PIL import Image



def pil_loader(path: str) -> Image.Image:
    """Load image using PIL"""
    return Image.open(path).convert("RGB")


class LazyLoader(DatasetFolder):
    def __init__(self, root, transform=None):
        super().__init__(
            root,
            loader=pil_loader,
            extensions=('jpg', 'jpeg', 'png'),
            transform=transform
        )
    
    def __getitem__(self, index):
        path, target = self.samples[index]
        image = self.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        return image, target

class Data:
      def __init__(self, train_path:str, test_path:str, resize:tuple, batch_size:int, train_test_split:float, augment:bool):

            self.transform = transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                             transforms.Resize(resize),   
                        ])

            self.augment_transform = transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                     transforms.Resize(resize),
                                     transforms.RandomRotation(20),
                                     transforms.RandomRotation(65),
                                     transforms.GaussianBlur((5,5)),
                                     transforms.RandomHorizontalFlip(0.5),
                                     transforms.ColorJitter(brightness=0.2, contrast=0.4, saturation=0.4, hue=0.2),
                                     
            ])


            self.train_path = train_path
            self.test_path = test_path

            # self.dataset = datasets.ImageFolder(train_path, transform=self.transform)
            # self.test_dataset = datasets.ImageFolder(test_path, transform=self.transform)

            self.dataset = LazyLoader(train_path, transform=self.transform)
            self.test_dataset = LazyLoader(test_path, transform=self.transform)


            self.classes = self.dataset.classes

            shared.classes = {i:j for i,j in enumerate(self.classes)}

            self.train_size = int(train_test_split * len(self.dataset))
            self.val_size = len(self.dataset) - self.train_size

            self.train_dataset, self.val_dataset = random_split(self.dataset, [self.train_size, self.val_size])

            if augment:
                  self.augment_dataset = deepcopy(self.train_dataset)
                  self.augment_dataset.dataset.transform = self.augment_transform
                  self.train_dataset = ConcatDataset([self.train_dataset, self.augment_dataset])


            self.train_dl = DataLoader(self.train_dataset, batch_size = batch_size, shuffle = True, num_workers = 4, pin_memory = True)
            self.val_dl = DataLoader(self.val_dataset, batch_size = batch_size, shuffle = True, num_workers = 8, pin_memory = True, persistent_workers = True)
            self.test_dl = DataLoader(self.test_dataset, batch_size = batch_size, shuffle = True, num_workers = 8, pin_memory = True, persistent_workers = True)


      def get_train_val_test_dataloaders(self):
            return self.train_dl, self.val_dl, self.test_dl

      def show_sample(self, x:torch.tensor, y:torch.tensor):

            x = x.permute(1, 2, 0)
            y = y.item()

            plt.figure()
            plt.imshow(x)
            plt.title(shared.classes[y])
            plt.show()






            

            






