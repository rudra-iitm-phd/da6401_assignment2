# 🧠 Image Classification with Custom & Pretrained CNNs

Welcome to this all-in-one pipeline for image classification using PyTorch! Whether you're building from scratch 🛠️ or leveraging the power of pretrained ResNet50 ⚡, this project has you covered. Seamlessly configurable, robustly designed, and integrated with Weights & Biases for tracking 🔍 — it's your perfect companion for image classification projects!

---

## 🚀 Features

- 🧩 **Highly Modular Design**
- 🖼️ **Supports Custom CNNs & Pretrained ResNet50**
- ⚙️ **Flexible Hyperparameter Control via CLI**
- 🧪 **Data Augmentation, Normalization & Caching**
- 📈 **WANDB Integration for Logging & Sweeps**
- 🎛️ **Bayesian Optimization for Hyperparameter Tuning**
- 📊 **Visual Model Summary & Layer-wise Visualization**

---

## 📁 Project Structure

```bash
.
├── train.py                  # Main training script
├── cnn.py                   # Custom CNN architecture
├── configuration.py         # Model & optimizer configuration
├── data.py                  # Dataset loading & preprocessing
├── argument_parser.py       # Argument parsing & filter generation
├── sweep_configuration.py   # WandB sweep configs
├── vision_models.py         # Pretrained ResNet50 wrapper
├── shared.py                # Shared globals
└── README.md                # This file!

```
## 🧪 Getting Started
### 🔧 Installation
Make sure you have the required dependencies installed:

```bash
pip install torch torchvision matplotlib wandb
```
## ⚙️ Usage
### 🏋️ Train a Custom CNN
```bash
python train.py
```
## ⚙️ Command-Line Arguments Reference

The default values have been configured as per the results obtained from the sweep

| 🏷️ **Argument** | 🔤 **Type** | 🧰 **Default** | 📝 **Description** |
|-----------------|------------|----------------|---------------------|
| `--batch_size` `-b` | `int` | `32` | 📦 Batch size used for training |
| `--resize` `-r_sz` | `int` | `224` | 📐 Image resize dimension |
| `--filter_automated` `-f_a` | `bool` | `True` | 🧠 Enable automated filter configuration |
| `--filter_strategy` `-f_s` | `str` | `'doubled'` | 🔁 Filter config strategy: `same`, `doubled`, `halved` |
| `--filter_initial` `-f_i` | `int` | `16` | 🧱 Number of filters in first layer |
| `--n_convolutions` `-n_c` | `int` | `5` | 🏗️ Number of convolution layers |
| `--filter_manual` `-f_m` | `int list` | `[16, 32, 64, 128, 256]` | ⚙️ Manual filter config per layer |
| `--padding` `-p` | `bool` | `True` | 🧩 Enable padding in conv layers |
| `--stride` `-s` | `int list` | `[1, 1, 1, 1, 1]` | 🚶 Stride for each conv layer |
| `--dense` `-d` | `int list` | `[2048]` | 🧮 Neurons in dense layers |
| `--kernel` `-k` | `int list` | `[3, 3, 3, 3, 3]` | 🔬 Kernel size per conv layer |
| `--conv_activation` `-c_a` | `str` | `'leaky_relu'` | 🔥 Activation for conv: `relu`, `gelu`, `sigmoid`, etc. |
| `--dense_activation` `-d_a` | `str` | `'relu'` | 🌟 Activation for dense layers |
| `--n_dense` `-n_d` | `int` | `1` | 🧱 Number of dense layers |
| `--optimizer` `-o` | `str` | `'adamax'` | ⚙️ Optimizer: `adam`, `sgd`, `nadam`, etc. |
| `--augment` `-a` | `bool` | `True` | 🧪 Enable data augmentation |
| `--batch_norm` `-b_n` | `bool` | `True` | 🧼 Enable batch normalization |
| `--learning_rate` `-lr` | `float` | `0.0012` | 📉 Learning rate |
| `--momentum` `-m` | `float` | `0.849` | 🌀 Momentum for optimizer |
| `--weight_decay` `-w_d` | `float` | `0.0037` | 🏋️ Weight decay (L2 regularization) |
| `--dropout` `-d_o` | `float` | `0.2` | 🎯 Dropout rate in dense layers |
| `--xavier_init` `-xi` | `bool` | `False` | ✨ Use Xavier weight initialization |
| `--wandb` | `flag` | `False` | 📊 Enable logging to [Weights & Biases](https://wandb.ai) |
| `--wandb_entity` `-we` | `str` | `'da24d008-iit-madras'` | 🧪 W&B entity name |
| `--wandb_project` `-wp` | `str` | `'da6401-assignment2'` | 📁 W&B project name |
| `--wandb_sweep` | `flag` | `False` | 🎯 Enable W&B sweep optimization |
| `--sweep_id` | `str` | `None` | 🆔 Sweep ID to resume |
| `--use_pretrained` | `flag` | `False` | 🧠 Use pretrained ResNet50 |
| `--pretrained_k` `-pk` | `int` | `1` | 🔄 Number of fine-tuned layers in pretrained model |
| `--use_test` `-ut` | `bool` | `False` | 🧪 Evaluate on test set |


## 🛠️ Configuration Highlights
The script is fully CLI-driven using argparse. Here's a glimpse of what you can control:

- 🧱 Architecture: --filter_strategy, --n_convolutions, --dense

- 🔥 Activations: --conv_activation, --dense_activation

- 🎯 Optimizer: --optimizer, --learning_rate, --momentum

- 🎛️ Regularization: --dropout, --weight_decay, --batch_norm

- 🧪 Training Options: --augment, --xavier_init, --use_test

- 🖼️ Preprocessing: --resize, --stride, --padding

For a complete list run 

```bash
python train.py --help
```