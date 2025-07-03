#!/bin/bash

aria2c -c -x 16 -s 16 \
  "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar"

aria2c -c -x 16 -s 16 \
  "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar"

