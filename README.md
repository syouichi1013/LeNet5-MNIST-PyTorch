# LeNet5-MNIST-PyTorch

##  Introduce 
A minimal LeNet5 implementation for MNIST handwritten digit classification with PyTorch.

## Data Preparation
MNIST dataset is auto-loaded via torchvision.datasets.

## Training
Run the training script directly:
```bashpython train.py```
## Inference
1.Place a test image named infer.jpg in the project root

2.Run the inference script:
```bashpython infer.py```

3.A result image with Pred (predicted digit) and Conf (confidence percentage) will be generated.(as below for example
![Prediction Image](infer/infer11.jpg)
