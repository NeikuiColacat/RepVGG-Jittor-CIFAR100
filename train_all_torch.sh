#!/bin/bash

for config_file in model_config_yamls/*.yaml; do
    if [[ $config_file == *torch* ]]; then
        python main_train_torch.py --config "$config_file"
    fi
done
