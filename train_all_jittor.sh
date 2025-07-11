#!/bin/bash

for config_file in model_config_yamls/*.yaml; do
    if [[ $config_file == *jittor* ]]; then
        python main_train_jittor.py --config "$config_file"
    fi
done
