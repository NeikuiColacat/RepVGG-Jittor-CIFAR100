#!/bin/bash
set -e

mkdir -p imagenet/train

echo "extract train"

tar -xf ILSVRC2012_img_train.tar -C imagenet/train

cd imagenet/train

for f in *.tar; do
    d=$(basename "$f" .tar)
    mkdir -p "$d"
    tar -xf "$f" -C "$d"
    rm "$f"
done
cd ../..

mkdir -p imagenet/val
echo "解压验证集..."
tar -xf ILSVRC2012_img_val.tar -C imagenet/val

echo "extract done"