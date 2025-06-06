from data import Data
import shared
from argument_parser import parser, genreate_filters, generate_activations, set_padding
import matplotlib.pyplot as plt
from cnn import CNN
from configuration import Configure
import torch
from tqdm.auto import tqdm
import wandb
import sweep_configuration
from torchvision.utils import make_grid

# Set multiprocessing strategy to avoid multiprocessing issues
torch.multiprocessing.set_sharing_strategy('file_system')

# Main training function

def train(log=True , sweep=True):
      if log:
            # Initialize W&B run
            run =  wandb.init(entity = config['wandb_entity'], project = config['wandb_project'], config = config)

            if log and sweep:
                  sweep_config = wandb.config
                  config.update(sweep_config)

            # Give a name to this W&B run based on config
            run.name = create_name(wandb.config)

      # Create data loaders
      d = Data(train_path=TRAIN_PATH, test_path=TEST_PATH, resize=(config['resize'], config['resize']), batch_size=config['batch_size'], train_test_split=0.8, augment=config['augment'])
      train_dl, val_dl, test_dl = d.get_train_val_test_dataloaders()

      # Get one batch to infer input size
      x_train, y_train = next(iter(train_dl))

      # Build configuration dictionary for the model
      script = {
            "input_size":x_train.shape, 
            "output_size":10, 
            "filters":genreate_filters(config['filter_initial'], config['filter_strategy'], config['n_convolutions']) if config['filter_automated'] else config['filter_manual'],
            "kernel_config":config['kernel'], 
            "padding_config":set_padding(config['kernel'], config['padding']) , 
            "stride_config":config['stride'], 
            "conv_activation":generate_activations(config['conv_activation'], config['n_convolutions']), 
            "dense_activation":generate_activations(config['dense_activation'], config['n_dense']),
            "dense_config":config['dense'], 
            "batch_size":config['batch_size'],
            "optimizer":config['optimizer'], 
            "batch_norm":config['batch_norm'],
            "dropout":config['dropout'], 
            "xavier_init":config['xavier_init'],
            "use_pretrained":config['use_pretrained'],
            "pk":config['pretrained_k']
      }

      # Initialize model and optimizer
      c = Configure()
      print(train_dl.batch_size)
      model, criterion = c.configure(script)

      # Optionally print model summary
      if config['use_pretrained']:
            print(model)
      else:
            model.view_model_summary()

      # Total trainable parameters
      pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
      print(f"Total number of parameters : {pytorch_total_params}")

      model = model.to(device)

      optimizer = c.configure_optim(model, config['optimizer'], lr=config['learning_rate'], momentum=config['momentum'], weight_decay=config['weight_decay'])
      loss_fn = criterion()
      model.train()

      running_loss = 0
      acc = 0

      # Training loop
      for epoch in range(15):
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
            train_acc = compute_accuracy(model, train_dl, device)
            val_acc = compute_accuracy(model, val_dl, device)

            # Optionally evaluate on test data
            if config['use_test']:
                  test_acc = compute_accuracy(model, test_dl, device)
                  if log:
                        wandb.log({"Test accuracy": test_acc})
                        infer(model, test_dl, device)

            # Log metrics to W&B
            if log:
                  wandb.log({
                              "Accuracy": val_acc,
                              "Train accuracy": train_acc,
                              "Train loss": round(loss.item(), 2)
                        })
            print(f'Train accuracy : {train_acc} Validation accuracy : {val_acc}')

# Log predictions and true labels for inference visualisation
def infer(model, dataloader, device='cpu'):
      model.eval()
      img_list, label_list, pred_list = [], [], []
      with torch.no_grad():
            images, labels = next(iter(dataloader))
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, dim=1)
            for img, label, pred in zip(images, labels, preds):
                  img_list.append(wandb.Image(img, caption=f"Pred: {shared.classes[pred.item()]} | True: {shared.classes[label.item()]}"))
                  if len(img_list) >= 30:  
                        break

      
      wandb.log({
                  "Sample images from Test data":img_list
                  })

# Accuracy computation helper
def compute_accuracy(model, data_loader, device='cpu'):
      model.eval()
      correct, total = 0, 0
      with torch.no_grad():
            for images, labels in tqdm(data_loader):
                  images = images.to(device)
                  labels = labels.to(device)
                  outputs = model(images)
                  _, preds = torch.max(outputs, dim=1)
                  correct += (labels == preds).sum().item()
                  total += labels.size(0)
      model.train()
      return round(correct/total * 100 , 2 )

# Generate a readable name for the W&B run
def create_name(configuration:dict):
      l = [f'{k}-{v}' for k,v in configuration.items() if k not in ['output_size', 'wandb_entity', 'wandb_project', 'wandb_sweep', 'sweep_id', 'wandb']]
      return '_'.join(l)

# Entry point of the script
if __name__ == '__main__':

      TRAIN_PATH = '../inaturalist_12K/train'
      TEST_PATH = '../inaturalist_12K/val'

      # Choose device
      device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'

      # Parse arguments
      args = parser.parse_args()
      config = args.__dict__

      print("Parsing DONE !!")

      # Handle W&B sweep mode
      if args.wandb_sweep:
            if args.use_pretrained:
                  sweep_config = sweep_configuration.sweep_config_resnet50
            else:
                  sweep_config = sweep_configuration.sweep_config
            if not args.sweep_id:
                  sweep_id = wandb.sweep(sweep_config, project=config['wandb_project'], entity=config['wandb_entity'])
            else:
                  sweep_id = args.sweep_id

            wandb.agent(sweep_id, function=train, count=20)
            wandb.finish()
      elif args.wandb:
            train(log=True, sweep=False)
      else:
            train(log=False, sweep=False)
