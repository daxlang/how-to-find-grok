#!/bin/bash
set -e; pip install torch numpy matplotlib -q
DEVICE="cpu"
for arg in "$@"; do case $arg in --gpu) DEVICE="cuda";; esac; done
echo "noise=0.4 wd 2.0-4.0  $DEVICE"
python run.py --device $DEVICE
echo "Done"
