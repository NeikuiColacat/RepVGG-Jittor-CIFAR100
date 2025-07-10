#!/bin/bash

rm -r /root/autodl-tmp/logs
rm -r /root/autodl-tmp/chk_points


for config_file in *.yaml; do
    if [[ $config_file == *torch* ]]; then
        python main_train_torch.py --config "$config_file"
    else
        python main_train_jittor.py --config "$config_file"
    fi
done
