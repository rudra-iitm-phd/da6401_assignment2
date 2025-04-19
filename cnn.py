import torch 
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

class CNN(nn.Module):
      def __init__(self, input_size, output_size, filters:list, padding_config:list, stride_config:list, dense_config:list, conv_activation:list, dense_activation:list, kernel_config:list, batch_size:int, batch_norm:bool, dropout:float, xavier_init:bool):
            super(CNN, self).__init__()

            # Assign hyperparameters and configurations
            self.filter_config = filters 
            self.input_size = input_size 
            self.input_channel = input_size[1] 
            self.output_size = output_size 
            self.padding_config = padding_config 
            self.stride_config = stride_config 
            self.dense_config = dense_config 
            self.kernel_config = kernel_config 
            self.batch_size = batch_size 
            self.conv_activations = conv_activation 
            print(self.conv_activations)
            self.dense_activation = dense_activation 
            print(self.dense_activation)
            self.batch_norm = batch_norm 
            self.dropout = dropout 

            self.info = dict()

            # Supported activation functions
            self.activations = {
                  'relu':F.relu,
                  'gelu':F.gelu,
                  'sigmoid':F.sigmoid,
                  'silu':F.silu,
                  'mish':F.mish,
                  'tanh':F.tanh,
                  'relu6':F.relu6,
                  'leaky_relu':F.leaky_relu
            }

            # Generate CNN and dense layers
            self.cnn = self.generate_cnn() 
            self.dense = self.generate_dense() 

            # Initialize weights if selected
            if xavier_init:
                  self.cnn = self.xavier_init(self.cnn)
                  self.dense = self.xavier_init(self.dense)

      # Construct convolutional layers as per configuration
      def generate_cnn(self) -> dict:
            architecture = nn.ModuleDict()

            for i,j in enumerate(self.filter_config):
                  architecture[f'Block {i}'] = nn.ModuleDict()
                  self.info[i] = dict()

                  if i == 0:
                        architecture[f'Block {i}']['conv'] = nn.Conv2d(in_channels = self.input_channel, out_channels = j, stride = self.stride_config[i], padding = self.padding_config[i], kernel_size = self.kernel_config[i])

                        if self.batch_norm:
                              architecture[f'Block {i}']['batch_norm'] = nn.BatchNorm2d(j)

                        self.info[i]['op_size'] = ((self.input_size[-1] - self.kernel_config[i] + 2*self.padding_config[i])//self.stride_config[i] + 1)

                  else:
                        architecture[f'Block {i}']['conv'] = nn.Conv2d(in_channels = self.filter_config[i-1], out_channels = j, stride = self.stride_config[i], padding = self.padding_config[i], kernel_size = self.kernel_config[i])

                        if self.batch_norm:
                              architecture[f'Block {i}']['batch_norm'] = nn.BatchNorm2d(j)

                        self.info[i]['op_size'] = ((self.info[i-1]['op_size'] - self.kernel_config[i] + 2*self.padding_config[i])//self.stride_config[i] + 1)
                  
                  architecture[f'Block {i}']['pool'] = nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0)
                  self.info[i]['op_size'] = self.info[i]['op_size']//2

            return architecture

      # Construct fully connected layers as per configuration
      def generate_dense(self):
            architecture = nn.ModuleDict()
            cnn_output_shape = self.filter_config[-1]*(self.info[len(self.filter_config)-1]['op_size']**2)

            for i,j in enumerate(self.dense_config):
                  architecture[f'Layer {i}'] = nn.ModuleDict()
                  if i == 0:
                        architecture[f'Layer {i}']['linear'] = nn.Linear(cnn_output_shape, j) 
                  else:
                        architecture[f'Layer {i}']['linear'] = nn.Linear(self.dense_config[i-1], j)

                  if self.dropout > 0:
                        architecture[f'Layer {i}']['dropout'] = nn.Dropout(self.dropout)
                  if self.batch_norm:
                        architecture[f'Layer {i}']['batch_norm'] = nn.BatchNorm1d(j)
            
            architecture[f'Layer {i+1}'] = nn.ModuleDict()
            architecture[f'Layer {i+1}']['linear'] = nn.Linear(self.dense_config[-1], self.output_size)
                        
            return architecture

      # Apply Xavier initialization to weights
      def xavier_init(self, model):
            for p in model.parameters():
                  if len(p.shape) > 1:
                        nn.init.xavier_uniform_(p)
            return model

      # Forward pass through the CNN and dense layers
      def forward(self, x):
            for i,j in enumerate(self.cnn.keys()):
                  x = (self.cnn[j]['conv'](x))
                  if self.batch_norm:
                        x = self.cnn[j]['batch_norm'](x)

                  x = self.activations[self.conv_activations[i]](x)
                  x = self.cnn[j]['pool'](x)
                  
            x = x.view(x.size(0), -1)
            for i,j in enumerate(self.dense.keys()):
                  if i == len(self.dense.keys()) - 1:
                        x = self.dense[j]['linear'](x)
                  else:
                        x = self.activations[self.dense_activation[i]](self.dense[j]['linear'](x))
                  if self.dropout > 0 and 'dropout' in self.dense[j].keys():
                        x = self.dense[j]['dropout'](x)

            return x

      # Visualize intermediate layer outputs of CNN
      def layerwise_visualise(self, x):
            vis_images = []
            normalise = lambda x : (x - x.min())/(x.max() - x.min())

            for i,j in enumerate(self.cnn.keys()):
                  x = self.cnn[j]['conv'](x)
                  vis_images.append([x, f'Block{i} conv'])
                  x = self.activations[self.conv_activations[i]](x)
                  vis_images.append([x, f'Block{i} activation'])
                  if self.batch_norm:
                        x = self.cnn[j]['batch_norm'](x)
                  x = self.cnn[j]['pool'](x)
                  vis_images.append([x, f'Block{i} pool'])
            
            plt.figure()
            for img in vis_images:
                  if img[0][0].shape[1] > 3:
                        plt.imshow(normalise(img[0][0][:3].permute(1, 2, 0).detach().numpy()))
                  else:
                        plt.imshow(normalise(img[0][0].permute(1, 2, 0).detach().numpy()))
                  plt.title(f'{img[1]}')
                  plt.show()

      # Print detailed summary of the CNN and dense layers
      def view_model_summary(self):
            print("Convolutions\n")
            for i in self.cnn.keys():
                  print(i)
                  print(f'Convolution : {self.cnn[i]["conv"]}')
                  if self.batch_norm:
                        print(f'Batch Norm : {self.cnn[i]["batch_norm"]}' )
                  print(f'Activation : {self.conv_activations[int(i[-1])]})')
                  print(f'Pooling :{self.cnn[i]["pool"]}')
                  print(f'Output size : {self.info[int(i[-1])]["op_size"]} X {self.info[int(i[-1])]["op_size"]} X {self.filter_config[int(i[-1])]}\n')
                  
            print("Dense\n")
            for i in self.dense.keys():
                  print(f'Linear : {self.dense[i]["linear"]}')
                  if int(i[-1]) == len(self.dense.keys()) - 1:
                        print(f'Activation : Softmax\n')
                  else:
                        print(f'Activation : {self.dense_activation[int(i[-1])]}\n')
                  if self.dropout and 'dropout' in self.dense[i].keys():
                        print(f'Dropout : {self.dense[i]["dropout"]}')
                  if self.batch_norm and 'batch_norm' in self.dense[i].keys():
                        print(f'Batch Norm : {self.dense[i]["batch_norm"]}' )
