import shared
from cnn import CNN 
import torch
from vision_models import PretrainedResnet50

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

            self.trainable = ['fc', 'layer3', 'layer2']

      def configure_optim(self, model, optim:str, lr:float, weight_decay:float, momentum=0.9):
            optimizer = self.optimizers[optim]
            if optim in ['adam', 'nadam', 'adamw', 'adamax']:
                  return optimizer(filter(lambda param : param.requires_grad, model.parameters()), lr = lr, betas = (momentum, 0.999), weight_decay = weight_decay)
            elif optim in ['sgd', 'rmsprop']:
                  return optimizer(filter(lambda param : param.requires_grad, model.parameters()), lr = lr, momentum = momentum, weight_decay = weight_decay)


      def configure(self, script):

            if not script['use_pretrained']:
                  model = CNN(script['input_size'], script['output_size'], script['filters'], script['padding_config'], script['stride_config'], script['dense_config'], script['conv_activation'], script['dense_activation'], script['kernel_config'], script['batch_size'], script['batch_norm'], script['dropout'], script['xavier_init'])
            else:

                  model = PretrainedResnet50(output_size = script['output_size'], trainable_layers = script['pk']).get_model()

            loss = torch.nn.CrossEntropyLoss

            return model, loss

           