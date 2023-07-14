# Creating a CIFAR 10 Convolution Neural Network

    This repo contains a CNN for training CIFAR 10 Dataset.

`model.py` file contains the CNN Model. It has `BaseNet` which is the final lighter model with under 50k parameters and produces over 70% train and test accuracy for the CIFAR 10 Dataset.

Model Summary is as Follows -

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 64, 32, 32]           1,728
       BatchNorm2d-2           [-1, 64, 32, 32]             128
              ReLU-3           [-1, 64, 32, 32]               0
            Conv2d-4          [-1, 128, 32, 32]          73,728
         MaxPool2d-5          [-1, 128, 16, 16]               0
       BatchNorm2d-6          [-1, 128, 16, 16]             256
              ReLU-7          [-1, 128, 16, 16]               0
            Conv2d-8          [-1, 128, 32, 32]          73,728
       BatchNorm2d-9          [-1, 128, 32, 32]             256
             ReLU-10          [-1, 128, 32, 32]               0
           Conv2d-11          [-1, 128, 16, 16]         147,456
      BatchNorm2d-12          [-1, 128, 16, 16]             256
             ReLU-13          [-1, 128, 16, 16]               0
           Conv2d-14          [-1, 256, 16, 16]         294,912
        MaxPool2d-15            [-1, 256, 8, 8]               0
      BatchNorm2d-16            [-1, 256, 8, 8]             512
             ReLU-17            [-1, 256, 8, 8]               0
           Conv2d-18            [-1, 512, 8, 8]       1,179,648
        MaxPool2d-19            [-1, 512, 4, 4]               0
      BatchNorm2d-20            [-1, 512, 4, 4]           1,024
             ReLU-21            [-1, 512, 4, 4]               0
...
Forward/backward pass size (MB): 9.25
Params size (MB): 20.29
Estimated Total Size (MB): 29.56
----------------------------------------------------------------
```

Model Accuracy:
- Training Accuracy: 95.98%
- Test Accuracy: 92.30%

Training Set Images:
![Training Images](<CleanShot 2023-07-14 at 15.58.05@2x.png>)

Incorrect Predictions Images:
![Incorrect Predictions](<CleanShot 2023-07-14 at 15.57.14@2x.png>)