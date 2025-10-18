import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch

class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        
        self.conv1 = nn.Sequential(
                            nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding=1 ,bias=False),
                            # GroupNorm with 1 group is LayerNorm
                            nn.GroupNorm(1,out_channels),
                            nn.LeakyReLU())
        
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding=1 ,bias=False),
                        # GroupNorm with 1 group is LayerNorm
                        nn.GroupNorm(1,out_channels))
            
            


        self.relu = nn.LeakyReLU()

        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels, 
                    out_channels, 
                    kernel_size=1, 
                    bias=False
                ),
                # GroupNorm with 1 group is LayerNorm
                nn.GroupNorm(1,out_channels)
            )

    def forward(self, x: Tensor) -> Tensor:
        
        identity = self.downsample(x) if self.downsample else x
        
        x = self.conv1(x)

        x = self.conv2(x)
        
        x = self.relu(x + identity)

        return x

class DownStepResBlock(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ResidualBlock(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)
