import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from sklearn.metrics import mutual_info_score
import numpy as np 
import torch

# Image sizees are (1024X1024)
#Spatial transformer network
    ## 3 parts:
    # 1. Localization network: takes the input image and outputs the parameters of the affine transformation
    # 2. Grid generator: generates a grid of (x,y) coordinates using the parameters of the affine transformation
    # 3. Sampler: uses the parameters of the affine transformation to sample the input image

class SpatialTransformer(nn.Module):
    def __init__(self,initial_theta=None):
        super(SpatialTransformer, self).__init__()
        
        # Localization network
        self.localization = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=7),  # Changed input channels to 2 for concatenated images
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )
        
        # Calculate the size of the feature map after the localization network
        self.feature_size = self._get_feature_size()
        
        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(self.feature_size, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )
        
        # Initialize the weights with identity transformation---> initial Affine matrix
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    
    def _get_feature_size(self):
        with torch.no_grad():
            x = torch.zeros(1, 2, 1024, 1024)  # Two channels for the concatenated input, fixed and moving images
            x = self.localization(x)
            return x.view(-1).size(0)
    
    def forward(self, fixed_image, moving_image):
        # Concatenate the fixed and moving images along the channel dimension
        x = torch.cat((fixed_image, moving_image), dim=1)
        
        # Localization network
        xs = self.localization(x)
        xs = xs.view(-1, self.feature_size)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        
        # Grid generator and sampler
        grid = F.affine_grid(theta, moving_image.size())
        aligned_moving_image = F.grid_sample(moving_image, grid)
        ncc_loss = self.calculate_ncc_loss(aligned_moving_image, fixed_image)
        
        return aligned_moving_image, theta, ncc_loss
    
    def calculate_ncc_loss(self, x, y): #normalized cross correlation 
        x_flat = x.view(x.size(0), -1)
        y_flat = y.view(y.size(0), -1)
        
        mean_x = torch.mean(x_flat, dim=1, keepdim=True)
        mean_y = torch.mean(y_flat, dim=1, keepdim=True)
        
        ncc = torch.sum((x_flat - mean_x) * (y_flat - mean_y), dim=1) / \
              (torch.sqrt(torch.sum((x_flat - mean_x)**2, dim=1)) * torch.sqrt(torch.sum((y_flat - mean_y)**2, dim=1)))
        
        return -torch.mean(ncc)