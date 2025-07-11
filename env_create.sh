#!/bin/bash

sudo apt update

sudo apt install -y python3.11 python3.11-venv python3.11-dev

python3.11 -m venv repvgg 

source repvgg/bin/activate

pip install -r requirements.txt