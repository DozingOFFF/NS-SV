#!/bin/bash

gpu=0

config=conf/ns_clean.yaml
trials_path=
ckpt_path=


echo "Evaluating $trials_path"
CUDA_VISIBLE_DEVICES=$gpu python main.py \
  --config $config \
  --evaluate \
  --checkpoint_path $ckpt_path \
  --trials_path $trials_path

