#!/bin/bash

conda create -n repvgg python=3.11

conda activate repvgg

pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

pip install -r requirements.txt