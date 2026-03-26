import torch.nn as nn

import torchvision.models as models

model = models.resnet18(weights='IMAGENET1K_V1')

model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

model.maxpool = nn.Identity()

model.fc = nn.Linear(model.fc.in_features, 1)
