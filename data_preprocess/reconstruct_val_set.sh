#!/bin/bash

cd ./imagenet/val
wget https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh
chmod +x valprep.sh
./valprep.sh 
rm valprep.sh