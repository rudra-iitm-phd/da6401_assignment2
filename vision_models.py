import torch 
import torch.nn as nn
import torchvision


class PretrainedResnet50:
      def __init__(self, output_size:int, optimizer, lr, trainable_layers = ['fc']):
            self.model = torchvision.models.resnet50(weights = 'IMAGENET1K_V2')
            self.model.fc = nn.Linear(self.model.fc._in_features, output_size)

            self.freeze_model()
            self.unfreeze_trainable_layers(trainable_layers)
            
      def freeze_model(self):
            for param in self.model.parameters():
                  param.requires_grad = False 

      def unfreeze_trainable_layers(self, layers:list[str]):
            for layer in layers:
                  for param in self.model[layer].parameters():
                        param.requires_grad = True 

      def get_model(self):
            return self.model

      

      
