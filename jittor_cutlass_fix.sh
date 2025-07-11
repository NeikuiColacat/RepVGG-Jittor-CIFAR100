#!/bin/bash
cd ~/.cache/jittor/
mkdir cutlass
cd cutlass
wget https://cg.cs.tsinghua.edu.cn/jittor/assets/cutlass.zip
apt install unzip
unzip cutlass.zip