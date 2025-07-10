#!/bin/bash


for config_file in *.yaml; do
    if [[ $config_file == *torch* ]]; then
        python infer_test_torch.py --config "$config_file"
    else
        python infer_test_jittor.py --config "$config_file"
    fi
done
