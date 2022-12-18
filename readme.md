# SLGTformer: An Attention Based Approach to Sign Language Recognition
## Data preparation

1. Please download and use the preprocessed skeleton data for WLASL by [Skeleton Aware Multi-modal Sign Language Recognition](https://arxiv.org/abs/2103.08833). Please be sure to follow their rules and agreements when using the preprocessed data.
    ```
    ./download.sh
    ```
    
## Pretrained models
Pretrained models are provided [here](https://utdallas.box.com/s/kbmzpjnvh3h1x6rc8gyg507b8vfjh721).

## Usage
### Train WLASL:
```
./train.sh
```

### Test:
```
./test.sh
```

## Acknowledgements

### SAM-SLR-v2
This code is based on [SAM-SLR-v2](https://github.com/jackyjsy/SAM-SLR-v2). Huge thank you to the authors for open sourcing their code.

### IRVL
Thank you to [@yuxng](https://github.com/yuxng) for his advice and guidance throughout this project. Shout-out to his lab [@IRVL](https://github.com/IRVLUTD) for the RTX A5000s and all the fun conversations while models were training.
