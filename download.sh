#!/bin/bash
mkdir data
mkdir data/wlasl
mkdir data/csl
mkdir data/autsl

# wlasl
wget --no-check-certificate -r "https://drive.google.com/uc?export=download&id=1QJo0T2q2EBaQQMyTFW98qhqErSyC_0rt&confirm=t" -O "data/wlasl/train_data_joint.npy" 
wget --no-check-certificate -r "https://drive.google.com/uc?export=download&id=1GJJROIvHpX90kCBTRMfK2xdIk85S7IGI&confirm=t" -O "data/wlasl/train_label.pkl" 
wget --no-check-certificate -r "https://drive.google.com/uc?export=download&id=1HsGKDepTEKVK7B9VBHmzLzGw_C9nb-oL&confirm=t" -O "data/wlasl/val_data_joint.npy"
wget --no-check-certificate -r "https://drive.google.com/uc?export=download&id=1NgH_NL3sjcYApmkY02UAPZ77HLMzezu_&confirm=t" -O "data/wlasl/val_label.pkl"

# autsl
wget --no-check-certificate -r "https://drive.google.com/uc?export=download&id=1kFqAQzrWP_b7fGvuBcnWjY3SGQJ62hsj&confirm=t" -O "data/autsl/train_data_joint.npy"
wget --no-check-certificate -r "https://drive.google.com/uc?export=download&id=1WY1XSvRPpGw5WgPVTcTMa3BCYx7aJ5DM&confirm=t" -O "data/autsl/train_label.pkl"
wget --no-check-certificate -r "https://drive.google.com/uc?export=download&id=1i41yLPIfM8ccX9uCtI2jVC0QS451hxkT&confirm=t" -O "data/autsl/val_data_joint.npy"
wget --no-check-certificate -r "https://drive.google.com/uc?export=download&id=1WeULp5U1OdwHfxqz02AE3MgaNkQcJx0M&confirm=t" -O "data/autsl/val_label.pkl"

# csl
wget --no-check-certificate -r "https://drive.google.com/uc?export=download&id=1AOAxcXxJpaoNLZYfy3Hn8BucQ7xPh9FR&confirm=t" -O "data/csl/train_data_joint.npy"
wget --no-check-certificate -r "https://drive.google.com/uc?export=download&id=1BVuK9M3MnPBtW7HGHJR79XycE1D2OR50&confirm=t" -O "data/csl/train_label.pkl"
wget --no-check-certificate -r "https://drive.google.com/uc?export=download&id=16jttL7AJJiMFi6qkeF8CXfi86RbFSsD1&confirm=t" -O "data/csl/val_data_joint.npy"
wget --no-check-certificate -r "https://drive.google.com/uc?export=download&id=1FnDSWbHAhrCfxlW7ESlS3_IHlMxNako_&confirm=t" -O "data/csl/val_label.pkl"

mkdir save_models
mkdir output