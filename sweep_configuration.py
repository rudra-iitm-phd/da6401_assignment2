sweep_config = {
            "method": "bayes",
            "metric": {"name": "Accuracy", "goal": "maximize"},
            "parameters": {
                "conv_activation": {"values": ['relu','gelu','sigmoid','silu','mish','tanh','relu6','leaky_relu']},
                "batch_size": {"values": [32, 64, 128]},
                "filter_strategy":{"values":['doubled', 'halved', 'same']},
                "filter_initial":{"values":[16, 32, 64]},
                "dense":{"values":[[32], [64], [128], [256], [512], [1024], [2048]]},
                "kernel":{"values":[[3]*5, [5]*5, [5]*2+[3]*3]},
                "learning_rate": {"distribution": "log_uniform_values", "min": 1e-4, "max": 1e-2},
                "optimizer": {"values": ['adam', 'nadam', 'adamw', 'adamax', 'sgd', 'rmsprop']},
                "momentum": {"distribution": "uniform", "min": 0.8, "max": 0.99},
                "weight_decay": {"distribution": "uniform", "min": 0, "max": 1e-2},
                "dropout":{"values":[0, 0.1, 0.2, 0.3, 0.5]},
                "batch_norm":{"values":[True, False]},
                "augment":{"values":[True, False]},
                "resize":{"values":[32, 64, 128, 224]},
                "dense_activation": {"values": ['relu','gelu','sigmoid','silu','mish','tanh','relu6','leaky_relu']}
            }
        }