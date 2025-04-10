import shared
from cnn import CNN 
import torch

class Configure:
      def __init__(self):
            self.optimizers = {
                  'adam':torch.optim.Adam,
                  'sgd':torch.optim.SGD,
                  'nadam':torch.optim.NAdam,
                  'rmsprop':torch.optim.RMSprop,
                  'adamw':torch.optim.AdamW,
                  'adamax':torch.optim.Adamax
            }

      def configure(self, script):

            model = CNN(script['input_size'], script['output_size'], script['filters'], script['padding_config'], script['stride_config'], script['dense_config'], script['conv_activation'], script['dense_activation'], script['kernel_config'], script['batch_size'], script['batch_norm'], script['dropout'])

            optim = self.optimizers[script['optimizer']]

            loss = torch.nn.NLLLoss

            return model, optim, loss

           