import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, 
                                  stride=stride, padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class CIFAR10Net3(nn.Module):
    def __init__(self):
        super().__init__()
        
        # C1 Block - RF: 3->5->9 (with dilation=2)
        self.c1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False),    # RF: 3
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(16, 24, kernel_size=3, padding=2, dilation=2, bias=False),  # RF: 3 + (2*2) = 7
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        # C2 Block - RF: 9->17->33 (with dilation=4,8)
        self.c2 = nn.Sequential(
            nn.Conv2d(24, 32, kernel_size=3, padding=4, dilation=4, bias=False),  # RF: 7 + (2*4) = 15
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.1),          
            nn.Conv2d(32, 48, kernel_size=3, padding=8, dilation=8, bias=False),  # RF: 15 + (2*8) = 31
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        # C3 Block - RF: 33->45->53 (with dilation=6,4)
        self.c3 = nn.Sequential(
            # Depthwise with dilation
            nn.Conv2d(48, 48, kernel_size=3, padding=6, dilation=6, groups=48, bias=False),  # RF: 31 + (2*6) = 43
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Dropout(0.1),            
            # Pointwise
            nn.Conv2d(48, 64, kernel_size=1, bias=False),    # RF: 43 (1x1 conv doesn't change RF)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(64, 80, kernel_size=3, padding=4, dilation=4, bias=False),  # RF: 43 + (2*4) = 51
            nn.BatchNorm2d(80),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        # C4 Block - RF: 53->57->61 (with dilation=2,2)
        self.c4 = nn.Sequential(
            nn.Conv2d(80, 90, kernel_size=3, padding=2, dilation=2, bias=False),  # RF: 51 + (2*2) = 55
            nn.BatchNorm2d(90),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(90, 72, kernel_size=3, padding=2, dilation=2, bias=False),  # RF: 55 + (2*2) = 59
        )
        # Final Receptive Field: 59

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(72, 10)  # Changed from 128 to 84

    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)
        x = self.gap(x)
        x = x.view(-1, 72)  # Changed from 128 to 84
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

class CIFAR10Net2(nn.Module):
    def __init__(self):
        super().__init__()
        
        # C1 Block - RF: 3->5->9 (with dilation=2)
        self.c1 = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size=3, padding=1, bias=False),    # RF: 3
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(12, 24, kernel_size=3, padding=2, dilation=2, bias=False),  # RF: 3 + (2*2) = 7
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        # C2 Block - RF: 9->17->33 (with dilation=4,8)
        self.c2 = nn.Sequential(
            nn.Conv2d(24, 32, kernel_size=3, padding=4, dilation=4, bias=False),  # RF: 7 + (2*4) = 15
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.1),          
            nn.Conv2d(32, 48, kernel_size=3, padding=8, dilation=8, bias=False),  # RF: 15 + (2*8) = 31
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        # C3 Block - RF: 33->45->53 (with dilation=6,4)
        self.c3 = nn.Sequential(
            # Depthwise with dilation
            nn.Conv2d(48, 48, kernel_size=3, padding=6, dilation=6, groups=48, bias=False),  # RF: 31 + (2*6) = 43
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Dropout(0.1),            
            # Pointwise
            nn.Conv2d(48, 64, kernel_size=1, bias=False),    # RF: 43 (1x1 conv doesn't change RF)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(64, 64, kernel_size=3, padding=4, dilation=4, bias=False),  # RF: 43 + (2*4) = 51
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        # C4 Block - RF: 53->57->61 (with dilation=2,2)
        self.c4 = nn.Sequential(
            nn.Conv2d(64, 96, kernel_size=3, padding=2, dilation=2, bias=False),  # RF: 51 + (2*2) = 55
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Conv2d(96, 96, kernel_size=3, padding=2, dilation=2, bias=False),  # RF: 55 + (2*2) = 59
        )
        # Final Receptive Field: 59

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(96, 10)  # Changed from 128 to 96

    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)
        x = self.gap(x)
        x = x.view(-1, 96)  # Changed from 128 to 96
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

class CIFAR10Net(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Regular Conv2d: RF: 3
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Depthwise Separable Conv: RF: 5
        self.conv2 = DepthwiseSeparableConv(32, 64, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Dilated Conv: RF: 13
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=2, dilation=2)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Regular Conv2d with stride: RF: 45
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Fully Connected Layer
        self.fc = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        
        x = self.gap(x)
        x = x.view(-1, 256)
        x = self.fc(x)
        return x 