import argparse


def genreate_filters(initial_filters, strategy, n_layers):
      if strategy == 'same':
            return [initial_filters for _ in range(n_layers)]
      elif strategy == 'doubled':
            return [initial_filters*(2**i) for i in range(n_layers)]
      elif strategy == 'halved':
            return [initial_filters//(2**i) for i in range(n_layers)]

def generate_activations(activation_function, n_layers):
      return [activation_function for _ in range(n_layers)]


parser = argparse.ArgumentParser(description = "Train a Convolutional Neural Network for classifying images from the inaturalist dataset")

parser.add_argument('-b', '--batch_size', 
                  type = int, default = 32, 
                  help = 'Batch size')

parser.add_argument('-r_sz', '--resize', 
                  type = int, default = 128, 
                  help = 'Number of epochs to train the agent')

parser.add_argument('-f_a', '--filter_automated', 
                  type = bool, default = False, 
                  help = 'Boolean flag for indicating automatic filter design')

parser.add_argument('-f_s', '--filter_strategy', 
                  type = str, default = 'doubled', 
                  help = 'Choices for filter configuration strategy : same, doubled, halved')

parser.add_argument('-f_i', '--filter_initial', 
                  type = int, default = 16, 
                  help = 'Number of filters to be used in the first layer')

parser.add_argument('-n_c', '--n_convolutions', 
                  type = int, default = 5 , 
                  help = 'Number of Convolutional layers')

parser.add_argument('-f_m', '--filter_manual', 
                  type = int, default = [16, 32, 64, 128, 256], nargs = '+',
                  help = 'Number of filters for each layer')

parser.add_argument('-p', '--padding', 
                  type = int, default = [1, 1, 1, 1, 1], nargs = '+',
                  help = 'Size of the padding to be used in each layer')

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
                  type = str, default = 'relu', nargs = '+',
                  help = 'Choice of activation functions to be used for convolutions : relu ,gelu, sigmoid, silu, mish, tanh, relu6, leaky_relu')

parser.add_argument('-d_a', '--dense_activation', 
                  type = str, default = 'relu', nargs = '+',
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

parser.add_argument('-w_d', '--weight_decay', 
                  type = float, default = 0,
                  help = 'Value for weight decay or L2 Regularization')

parser.add_argument('-d_o', '--dropout', 
                  type = bool, default = False,
                  help = 'Enable Dropout')