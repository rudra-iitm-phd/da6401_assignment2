from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, ConcatDataset, Dataset
import shared
import torch
import matplotlib.pyplot as plt
from copy import deepcopy
from torchvision.datasets import DatasetFolder
from PIL import Image
from diskcache import Cache

torch.multiprocessing.set_sharing_strategy('file_system')





def pil_loader(path: str) -> Image.Image:
    """Load image using PIL"""
    return Image.open(path).convert("RGB")


class LazyLoader(DatasetFolder):
    def __init__(self, root, transform=None, cache_dir = "../cache"):
        super().__init__(
            root,
            loader=pil_loader,
            extensions=('jpg', 'jpeg', 'png'),
            transform=transform
        )
        self.cache = Cache(cache_dir)
    
    def __getitem__(self, index):
        path, target = self.samples[index]
        try:
            image = self.cache[path]
        except KeyError:
            image = self.loader(path)
            self.cache[path] = image
        if self.transform is not None:
            image = self.transform(image)
        return image, target

# class CacheDataset(Dataset):
#     def __init__(self, dataset):
#         self.dataset = dataset
#         self.cache = [None] * len(dataset)
#         # Handle different dataset types
#         if hasattr(dataset, 'dataset'):  # For Subset objects
#             self.file_paths = [sample[0] for sample in dataset.dataset.samples]
#             self.indices = dataset.indices
#         else:  # For raw ImageFolder
#             self.file_paths = [sample[0] for sample in dataset.samples]
#             self.indices = range(len(dataset))
#     def __getitem__(self, index):
#         if self.cache[index] is None:
#             # Use original index mapping
#             original_idx = self.indices[index]
#             with open(self.file_paths[original_idx], 'rb') as f:
#                 img = Image.open(f).convert('RGB')
#                 self.cache[index] = (self.dataset.transform(img), self.dataset[index][1])
#         return self.cache[index]

    # def __len__(self):
    #     return len(self.dataset)

class CacheDataset(Dataset):
    def __init__(self, dataset):
        """
        Wraps an existing dataset and caches its items in memory after first access.
        """
        self.dataset = dataset
        self.cache = [None] * len(dataset)

    def __getitem__(self, index):
        if self.cache[index] is None:
            self.cache[index] = self.dataset[index]
        return self.cache[index]

    def __len__(self):
        return len(self.dataset)

class Data:
      def __init__(self, train_path:str, test_path:str, resize:tuple, batch_size:int, train_test_split:float, augment:bool):

            self.transform = transforms.Compose([
                             transforms.Resize(resize),
                             transforms.ToTensor(),
                             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                
                        ])

            self.augment_transform = transforms.Compose([
                                     transforms.RandomResizedCrop(resize),  # Operates on PIL
                                     transforms.AutoAugment(),              # Operates on PIL
                                     transforms.ToTensor(),                 # Now converts to tensor
                                     transforms.Normalize(                  # Normalize after conversion
                                            mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225]
                                        )     
                        ])


            self.train_path = train_path
            self.test_path = test_path

            self.dataset = datasets.ImageFolder(train_path, transform=self.transform)
            self.test_dataset = datasets.ImageFolder(test_path, transform=self.transform)

            # self.dataset = LazyLoader(train_path, transform=self.transform)
            # self.test_dataset = LazyLoader(test_path, transform=self.transform)


            self.classes = self.dataset.classes

            shared.classes = {i:j for i,j in enumerate(self.classes)}

            self.train_size = int(train_test_split * len(self.dataset))
            self.val_size = len(self.dataset) - self.train_size

            self.train_dataset, self.val_dataset = random_split(self.dataset, [self.train_size, self.val_size])

            if augment:
                  self.augment_dataset = deepcopy(self.train_dataset)
                  self.augment_dataset.dataset.transform = self.augment_transform
                  self.train_dataset = ConcatDataset([self.train_dataset, self.augment_dataset])

            
            self.train_dataset = CacheDataset(self.train_dataset)
            self.val_dataset = CacheDataset(self.val_dataset)
            self.test_dataset = CacheDataset(self.test_dataset)

            self.train_dl = DataLoader(self.train_dataset, batch_size = batch_size, shuffle = True, num_workers = 4, pin_memory = False, prefetch_factor = 2)
            self.val_dl = DataLoader(self.val_dataset, batch_size = batch_size, shuffle = True, num_workers = 4, pin_memory = True, persistent_workers = False, prefetch_factor = 2)
            self.test_dl = DataLoader(self.test_dataset, batch_size = batch_size, shuffle = True, num_workers = 4, pin_memory = True, persistent_workers = False, prefetch_factor = 2)


      def get_train_val_test_dataloaders(self):
            return self.train_dl, self.val_dl, self.test_dl

      def show_sample(self, x:torch.tensor, y:torch.tensor):

            x = x.permute(1, 2, 0)
            y = y.item()

            plt.figure()
            plt.imshow(x)
            plt.title(shared.classes[y])
            plt.show()






            

            






