from data import Data
import shared
from argument_parser import parser, genreate_filters, generate_activations
import matplotlib.pyplot as plt
from cnn import CNN
from configuration import Configure
import torch
from tqdm.auto import tqdm




def train(train_dl, val_dl, test_dl, model, optim, criterion, epochs, device='cpu'):

      model = model.to(device)

      optimizer = optim(model.parameters())
      loss_fn = criterion()
      model.train()

      running_loss = 0
      acc = 0

      for epoch in range(epochs):
            for images, labels in tqdm(train_dl):

                  images = images.to(device)
                  labels = labels.to(device)

                  optimizer.zero_grad()
                  outputs = model(images)

                  loss = loss_fn(outputs, labels)

                  loss.backward()

                  optimizer.step()

                  
                  running_loss += loss.item()

            
            print(loss.item())
            running_loss = 0

            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                  for images, labels in tqdm(val_dl):
                        images = images.to(device)
                        labels = labels.to(device)
                        outputs = model(images)
                        _, preds = torch.max(outputs, dim=1)
                        correct += (labels == preds).sum().item()
                        total += labels.size(0)
            print(f'Validation accuracy : {round(correct/total*100, 2)}')
            model.train()

            
            




if __name__ == '__main__':

      TRAIN_PATH = '../inaturalist_12K/train'
      TEST_PATH = '../inaturalist_12K/val'

      DEVICE = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'

      args = parser.parse_args()

      config = args.__dict__

      print("Parsing DONE !!")

      d = Data(train_path=TRAIN_PATH, test_path=TEST_PATH, resize=(config['resize'],config['resize']), batch_size=config['batch_size'], train_test_split=0.8, augment=config['augment'])

      train_dl, val_dl, test_dl = d.get_train_val_test_dataloaders()

      x_train, y_train = next(iter(train_dl))

      print(x_train.shape, y_train.shape)

      x_val, y_val = next(iter(val_dl))

      x_test, y_test = next(iter(test_dl))

      script = {
            
            "input_size":x_train.shape, 
            "output_size":10, 

            "filters":genreate_filters(config['filter_initial'], config['filter_strategy'], config['n_convolutions']) if config['filter_automated'] else config['filter_manual'],

 
            "kernel_config":config['kernel'], 
            "padding_config":config['padding'], 
            "stride_config":config['stride'], 
            "conv_activation":generate_activations(config['conv_activation'], config['n_convolutions']), 

            "dense_activation":generate_activations(config['dense_activation'], config['n_dense']),
            "dense_config":config['dense'], 

            "batch_size":config['batch_size'],

            "optimizer":config['optimizer'], 

            "batch_norm":config['batch_norm'],

            "dropout":config['dropout']
      }


      c = Configure()

      print(train_dl.batch_size)

      model , optim, loss = c.configure(script)
      model.view_model_summary()
      pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
      print(f"Total number of parameters : {pytorch_total_params}")

      # model(x_train)

      

      # print(model.cnn)

      train(train_dl, val_dl, test_dl, model, optim, loss, 10, DEVICE)

      # model.layerwise_visualise(x_train)

      
      