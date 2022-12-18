#!/bin/bash
mkdir data

# wlasl
wget --no-check-certificate -r "https://drive.google.com/uc?export=download&id=1QJo0T2q2EBaQQMyTFW98qhqErSyC_0rt&confirm=t" -O "data/train_data_joint.npy" 
wget --no-check-certificate -r "https://drive.google.com/uc?export=download&id=1GJJROIvHpX90kCBTRMfK2xdIk85S7IGI&confirm=t" -O "data/train_label.pkl" 
wget --no-check-certificate -r "https://drive.google.com/uc?export=download&id=1HsGKDepTEKVK7B9VBHmzLzGw_C9nb-oL&confirm=t" -O "data/val_data_joint.npy"
wget --no-check-certificate -r "https://drive.google.com/uc?export=download&id=1NgH_NL3sjcYApmkY02UAPZ77HLMzezu_&confirm=t" -O "data/val_label.pkl"

mkdir save_models
mkdir output