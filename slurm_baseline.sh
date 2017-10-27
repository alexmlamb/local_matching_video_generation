#!/bin/bash

export HOME=`getent passwd $USER | cut -d':' -f6`
source ~/.bashrc
export PYTHONUNBUFFERED=1
echo Running on $HOSTNAME

python cifar_gan.py
