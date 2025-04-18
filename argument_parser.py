import argparse


def genreate_filters(initial_filters, strategy, n_layers):
      if strategy == 'same':
            return [initial_filters for _ in range(n_layers)]
      elif strategy == 'doubled':
            return [initial_filters*(2**i) for i in range(n_layers)]
      elif strategy == 'halved':
            return [initial_filters//(2**i) for i in range(n_layers)]

def generate_activations(activation_function, n_layers):
      return [activation_function]*n_layers

def set_padding(kernel_config, enable_padding):
      padding = [0]*len(kernel_config)
      if enable_padding:
            for i,j in enumerate(kernel_config):
                  if j == 3:
                        padding[i] = 1
                  elif j == 5:
                        padding[i] = 2
      return padding


parser = argparse.ArgumentParser(description = "Train a Convolutional Neural Network for classifying images from the inaturalist dataset")

parser.add_argument('-b', '--batch_size', 
                  type = int, default = 32, 
                  help = 'Batch size')

parser.add_argument('-r_sz', '--resize', 
                  type = int, default = 128, 
                  help = 'Number of epochs to train the agent')

parser.add_argument('-f_a', '--filter_automated', 
                  type = bool, default = True, 
                  help = 'Boolean flag for indicating automatic filter design')

parser.add_argument('-f_s', '--filter_strategy', 
                  type = str, default = 'doubled', 
                  help = 'Choices for filter configuration strategy : same, doubled, halved')

parser.add_argument('-f_i', '--filter_initial', 
                  type = int, default = 32, 
                  help = 'Number of filters to be used in the first layer')

parser.add_argument('-n_c', '--n_convolutions', 
                  type = int, default = 5 , 
                  help = 'Number of Convolutional layers')

parser.add_argument('-f_m', '--filter_manual', 
                  type = int, default = [16, 32, 64, 128, 256], nargs = '+',
                  help = 'Number of filters for each layer')

parser.add_argument('-p', '--padding', 
                  type = bool, default = True,
                  help = 'Enable Padding')

parser.add_argument('-s', '--stride', 
                  type = int, default = [1, 1, 1, 1, 1], nargs = '+',
                  help = 'Stride to be performed in each layer')

parser.add_argument('-d', '--dense', 
                  type = int, default = [64], nargs = '+',
                  help = 'Number of neurons in each dense layer')

parser.add_argument('-k', '--kernel', 
                  type = int, default = [3, 3, 3, 3, 3], nargs = '+',
                  help = 'Size of the kernel for each layer')

parser.add_argument('-c_a', '--conv_activation', 
                  type = str, default = 'relu',
                  help = 'Choice of activation functions to be used for convolutions : relu ,gelu, sigmoid, silu, mish, tanh, relu6, leaky_relu')

parser.add_argument('-d_a', '--dense_activation', 
                  type = str, default = 'relu',
                  help = 'Choice of activation functions to be used for convolutions : relu ,gelu, sigmoid, silu, mish, tanh, relu6, leaky_relu')

parser.add_argument('-n_d', '--n_dense', 
                  type = int, default = 1,
                  help = 'Number of dense layers')

parser.add_argument('-o', '--optimizer', 
                  type = str, default = 'adam',
                  help = 'Choices for optimizers : adam, sgd, nadam, adamw, rmsprop, adamax')

parser.add_argument('-a', '--augment', 
                  type = bool, default = False,
                  help = 'Enable data augmentation')


parser.add_argument('-b_n', '--batch_norm', 
                  type = bool, default = False,
                  help = 'Enable Batch Normalisation')

parser.add_argument('-lr', '--learning_rate', 
                  type = float, default = 0.001,
                  help = 'Learning rate for optimizer')

parser.add_argument('-m', '--momentum', 
                  type = float, default = 0.9,
                  help = 'Momentum to be used by the optimizer')

parser.add_argument('-w_d', '--weight_decay', 
                  type = float, default = 0,
                  help = 'Value for weight decay or L2 Regularization')

parser.add_argument('-d_o', '--dropout', 
                  type = float, default = 0,
                  help = 'Drop out rate for the dense layer')

parser.add_argument('-xi', '--xavier_init', 
                  type = bool, default = False,
                  help = 'Xavier Initialization of the weights')

parser.add_argument('--wandb', action='store_true', help='Enable wandb logging')

parser.add_argument('-we', '--wandb_entity', 
                  type = str, default = 'da24d008-iit-madras' ,
                  help = 'Wandb Entity used to track experiments in the Weights & Biases dashboard')

parser.add_argument('-wp', '--wandb_project', 
                  type = str, default = 'da6401-assignment2' ,
                  help = 'Project name used to track experiments in Weights & Biases dashboard')

parser.add_argument('--wandb_sweep', action='store_true', help='Enable W&B sweep')

parser.add_argument('--sweep_id', type = str, help = "Sweep ID", default = None)

parser.add_argument('--use_pretrained', action='store_true', help='Use a pretrained resnet50 to train the network')

parser.add_argument('-pk', '--pretrained_k', 
                  type = int, default = 1,
                  help = 'The last k layers for pre-training')