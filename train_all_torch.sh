#!/bin/bash

for config_file in *.yaml; do
    if [[ $config_file == *torch* ]]; then
        python main_train_torch.py --config "$config_file"
    fi
done
