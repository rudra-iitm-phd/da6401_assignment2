import torch 
import torch.nn as nn
import torchvision


class PretrainedResnet50:
      def __init__(self, output_size:int, trainable_layers = 1):
            self.model = torchvision.models.resnet50(weights = 'IMAGENET1K_V2')
            self.model.fc = nn.Linear(self.model.fc.in_features, output_size)
            self.k_layers = [self.model.fc, self.model.layer4, self.model.layer3, self.model.layer4]
            self.freeze_model()
            self.unfreeze_trainable_layers(trainable_layers)
            
      def freeze_model(self):
            for param in self.model.parameters():
                  param.requires_grad = False 

      def unfreeze_trainable_layers(self, layers):
            for layer in range(layers):
                  for param in self.k_layers[layer].parameters():
                        param.requires_grad = True

      def get_model(self):
            return self.model



      

      
