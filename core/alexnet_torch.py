
import torch
import torch.nn as nn
import torch.nn.functional as F

def alexnet_v2_pytorch(num_classes=1000, dropout_keep_prob=0.5, global_pool=False):
    """
    Instantiate the AlexNetV2 model in PyTorch.
    """
    class AlexNetV2(nn.Module):
        def __init__(self, num_classes, dropout_keep_prob, global_pool):
            super(AlexNetV2, self).__init__()
            self.global_pool = global_pool

            # Convolutional layers
            self.features = nn.Sequential(
                #conv1
                nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=0),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),

                #conv2
                nn.Conv2d(64, 192, kernel_size=5, stride=1, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),

                #conv3
                nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),

                #conv4
                nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),

                #conv5
                nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2)
            )

            # Fully connected layers
            self.classifier = nn.Sequential(
                #fc6
                nn.Conv2d(256, 4096, kernel_size=5, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.Dropout(p=1 - dropout_keep_prob),

                #fc7
                nn.Conv2d(4096, 4096, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.Dropout(p=1 - dropout_keep_prob),

                #fc8
                nn.Conv2d(4096, num_classes, kernel_size=1, stride=1, padding=0)
            )

        def forward(self, x):
            x = self.features(x)
            if self.global_pool:
                x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
            x = self.classifier(x)
            x = torch.flatten(x, start_dim=1)
            return x

  # Instantiate and return the AlexNetV2 model
    return AlexNetV2(num_classes, dropout_keep_prob, global_pool)
