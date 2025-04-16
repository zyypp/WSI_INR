import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from experts import SineLayer

class PositionalEncoding(nn.Module):
    def __init__(self, num_encoding_functions=6, include_input=True, log_sampling=True, normalize=False,
                 input_dim=2, gaussian_pe=False, gaussian_variance=10):
        super().__init__()
        self.num_encoding_functions = num_encoding_functions
        self.include_input = include_input
        self.log_sampling = log_sampling
        self.normalize = normalize
        self.gaussian_pe = gaussian_pe
        self.normalization = None

        if self.gaussian_pe:
            # this needs to be registered as a parameter so that it is saved in the model state dict
            # and so that it is converted using .cuda(). Doesn't need to be trained though
            self.gaussian_weights = nn.Parameter(gaussian_variance * torch.randn(num_encoding_functions, input_dim),
                                                 requires_grad=False)
        else:
            self.frequency_bands = None
            if self.log_sampling:
                self.frequency_bands = 2.0 ** torch.linspace(
                    0.0,
                    self.num_encoding_functions - 1,
                    self.num_encoding_functions)
            else:
                self.frequency_bands = torch.linspace(
                    2.0 ** 0.0,
                    2.0 ** (self.num_encoding_functions - 1),
                    self.num_encoding_functions)

            if normalize:
                self.normalization = torch.tensor(1/self.frequency_bands)

    def forward(self, tensor):
        encoding = [tensor] if self.include_input else []
        if self.gaussian_pe:
            for func in [torch.sin, torch.cos]:
                encoding.append(func(torch.matmul(tensor, self.gaussian_weights.T)))
        else:
            for idx, freq in enumerate(self.frequency_bands):
                for func in [torch.sin, torch.cos]:
                    if self.normalization is not None:
                        encoding.append(self.normalization[idx]*func(tensor * freq))
                    else:
                        encoding.append(func(tensor * freq))

        # Special case, for no positional encoding
        if len(encoding) == 1:
            return encoding[0]
        else:
            return torch.cat(encoding, dim=-1)

class RegionFeatureExtractor(nn.Module):
    def __init__(self, feature_dim=16):
        super(RegionFeatureExtractor, self).__init__()
        self.feature_dim = feature_dim
        # This layer will learn to map coordinates to region features
        self.feature_net = nn.Sequential(
            SineLayer(2, 64),
            SineLayer(64, 32),
            nn.Linear(32, feature_dim)
        )
    
    def forward(self, coords):
        # Map coordinates to region features
        return self.feature_net(coords)

class ScaleEncoder(nn.Module):
    def __init__(self, scale_level, feature_dim=8):
        super(ScaleEncoder, self).__init__()
        self.scale_level = scale_level
        self.scale_factor = 2 ** scale_level  # Higher scale_level = more details
        
        # Network to compute scale-specific features
        self.net = nn.Sequential(
            SineLayer(2, 32, is_first=True, omega_0=30 / self.scale_factor),  # Lower frequency for lower scales
            SineLayer(32, feature_dim)
        )
    
    def forward(self, coords):
        # Transform coordinates based on scale
        scaled_coords = coords * self.scale_factor
        return self.net(scaled_coords)
    
    def get_relevance(self, coords):
        # Compute how relevant this scale is for the given coordinates
        # This is a simple heuristic that could be replaced with learning
        with torch.no_grad():
            features = self.forward(coords)
            # Measure the "activity" of the features
            relevance = torch.mean(torch.abs(features), dim=-1, keepdim=True)
            return relevance 