#!/bin/bash

for config_file in *.yaml; do
    if [[ $config_file == *jittor* ]]; then
        python main_train_jittor.py --config "$config_file"
    fi
done
