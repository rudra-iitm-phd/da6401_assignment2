sweep_config = {
            "method": "bayes",  # Use Bayesian optimization for hyperparameter tuning
            "metric": {"name": "Accuracy", "goal": "maximize"},
            "parameters": {
                # List of activation functions to try for conv layers
                "conv_activation": {"values": ['relu','gelu','sigmoid','silu','mish','tanh','relu6','leaky_relu']},
                # Batch size options
                "batch_size": {"values": [32, 64, 128]},
                # Filter strategies to generate filter sizes
                "filter_strategy":{"values":["doubled", "halved", "same"]},
                # Initial number of filters in the first convolution layer
                "filter_initial":{"values":[16, 32, 64]},
                # Number of neurons in dense layers
                "dense":{"values":[[32], [64], [128], [256], [512], [1024], [2048]]},
                # Kernel size configurations
                "kernel":{"values":[[3]*5, [5]*5, [5]*2+[3]*3]},
                # Learning rate sampled logarithmically between min and max
                "learning_rate": {"distribution": "log_uniform_values", "min": 1e-4, "max": 1e-2},
                # Optimizer choices
                "optimizer": {"values": ['adam', 'nadam', 'adamw', 'adamax', 'sgd', 'rmsprop']},
                # Momentum value sampled uniformly between 0.8 and 0.99
                "momentum": {"distribution": "uniform", "min": 0.8, "max": 0.99},
                # Weight decay (L2 regularization)
                "weight_decay": {"distribution": "uniform", "min": 0, "max": 1e-2},
                # Dropout probabilities
                "dropout":{"values":[0, 0.1, 0.2, 0.3, 0.5]},
                # Toggle for enabling/disabling batch normalization
                "batch_norm":{"values":[True, False]},
                # Toggle for enabling/disabling data augmentation
                "augment":{"values":[True, False]},
                # Image resize dimensions
                "resize":{"values":[32, 64, 128, 224]},
                # List of activation functions to try for dense layers
                "dense_activation": {"values": ['relu','gelu','sigmoid','silu','mish','tanh','relu6','leaky_relu']}
            }
        }

# Sweep config for ResNet50 pretrained model
sweep_config_resnet50 = {
            "method": "bayes",  # Use Bayesian optimization
            "metric": {"name": "Accuracy", "goal": "maximize"},
            "parameters": {
                # Batch size options
                "batch_size": {"values": [32, 64, 128]},
                # Learning rate options
                "learning_rate": {"distribution": "log_uniform_values", "min": 1e-4, "max": 1e-2},
                # Optimizer choices
                "optimizer": {"values": ['adam', 'nadam', 'adamw', 'adamax', 'sgd', 'rmsprop']},
                # Momentum sampling range
                "momentum": {"distribution": "uniform", "min": 0.8, "max": 0.99},
                # L2 regularization
                "weight_decay": {"distribution": "uniform", "min": 0, "max": 1e-2},
                # Enable/disable data augmentation
                "augment":{"values":[True, False]},
                # Resize input images
                "resize":{"values":[224]},
                # Number of trainable layers from pretrained ResNet50
                "pretrained_k":{"values":[1, 2, 3]}
            }
        }
