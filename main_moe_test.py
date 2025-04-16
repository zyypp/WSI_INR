"""
MoE Multiscale INR Model for Image Representation

This script implements a Mixture of Experts (MoE) based Implicit Neural Representation (INR) model
for image compression and reconstruction. The model uses 6 experts (2 of each complexity level)
and supports both cropped and full image processing.

Usage:
    python main_moe_multiscale.py [options]

Options:
    --use_crop          Use cropped image instead of full image (default: False)
    --crop_size SIZE    Size of cropped image (default: 512)
    --svs_path PATH     Path to SVS file (default: "/data/WRI_data/test.svs")
    --num_epochs N      Number of training epochs (default: 3000)
    --batch_size N      Batch size for training (default: 16384)
    --num_samples N     Number of samples for adaptive sampling (default: 200000)
    --level N           SVS pyramid level to use (default: 3)

Examples:
    1. Train on cropped image:
       python main_moe_multiscale.py --use_crop --crop_size 512
    
    2. Train on full image:
       python main_moe_multiscale.py
    
    3. Custom training parameters:
       python main_moe_multiscale.py --use_crop --crop_size 256 --num_epochs 5000 --batch_size 32768 --num_samples 400000

Model Architecture:
    - 6 Experts (2 simple, 2 medium, 2 complex)
    - 3 Scale encoders
    - Positional encoding with 6 encoding functions
    - Region feature extractor
    - Router network for expert selection

Output:
    - original_image.png: Original input image
    - complexity_map.png: Visualization of image complexity
    - reconstructed_image_*.png: Reconstructed images during training
    - moe_multiscale_checkpoint_*.pth: Model checkpoints
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import openslide
import os
import argparse
from skimage.io import imsave
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from utils.evaluate_metrics import evaluate_images, calculate_compression_ratio
from utils.data_utils import load_svs_image, compute_complexity_map, create_dataset, adaptive_sampling, SlideDataset
from utils.metrics import evaluate_model, visualize_expert_regions
from matplotlib.colors import LinearSegmentedColormap

class PositionalEncoding(nn.Module):
    def __init__(self, num_encoding_functions=6, include_input=True):
        super().__init__()
        self.num_encoding_functions = num_encoding_functions
        self.include_input = include_input
        
    def forward(self, x):
        if self.include_input:
            encoded = [x]
        else:
            encoded = []
            
        for i in range(self.num_encoding_functions):
            encoded.append(torch.sin(2**i * np.pi * x))
            encoded.append(torch.cos(2**i * np.pi * x))
            
        return torch.cat(encoded, dim=-1)

class RegionFeatureExtractor(nn.Module):
    def __init__(self, feature_dim=16):
        super().__init__()
        self.feature_dim = feature_dim
        
        # 使用多层感知机提取区域特征
        self.net = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, feature_dim)
        )
        
    def forward(self, x):
        return self.net(x)

class ScaleEncoder(nn.Module):
    def __init__(self, scale_level, feature_dim=8):
        super().__init__()
        self.scale_level = scale_level
        self.feature_dim = feature_dim
        
        # 使用多层感知机提取尺度特征
        self.net = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, feature_dim)
        )
        
        # 尺度相关性预测
        self.relevance = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
    def forward(self, x):
        return self.net(x)
        
    def get_relevance(self, x):
        return self.relevance(x)

class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

class ResBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ResBlock, self).__init__()
        self.net = nn.Sequential(
            SineLayer(in_features, out_features),
            SineLayer(out_features, out_features)
        )
        self.flag = (in_features != out_features)

        if self.flag:
            self.transform = SineLayer(in_features, out_features)

    def forward(self, features):
        outputs = self.net(features)
        if self.flag:
            features = self.transform(features)
        return 0.5 * (outputs + features)

class SimpleExpert(nn.Module):
    def __init__(self, input_dim, output_dim=3, hidden_dim=64, expert_type='local'):
        super(SimpleExpert, self).__init__()
        self.expert_type = expert_type
        if expert_type == 'local':
            # Focus on local features
            self.net = nn.Sequential(
                SineLayer(input_dim, hidden_dim, omega_0=30),  # Higher frequency for local details
                SineLayer(hidden_dim, hidden_dim),
                SineLayer(hidden_dim, output_dim)
            )
        elif expert_type == 'global':
            # Focus on global features
            self.net = nn.Sequential(
                SineLayer(input_dim, hidden_dim, omega_0=10),  # Lower frequency for global patterns
                SineLayer(hidden_dim, hidden_dim),
                SineLayer(hidden_dim, output_dim)
            )
        else:  # 'balanced'
            # Balanced approach
            self.net = nn.Sequential(
                SineLayer(input_dim, hidden_dim, omega_0=20),  # Medium frequency
                SineLayer(hidden_dim, hidden_dim),
                SineLayer(hidden_dim, output_dim)
            )
    
    def forward(self, x):
        return self.net(x)

class ComplexExpert(nn.Module):
    def __init__(self, input_dim, output_dim=3, hidden_dim=256, expert_type='structure'):
        super(ComplexExpert, self).__init__()
        self.expert_type = expert_type
        
        if expert_type == 'structure':
            # 结构专家：使用更深的网络和残差连接
            self.net = nn.Sequential(
                # 第一层：基础特征提取
                ResBlock(input_dim, hidden_dim),
                
                # 第二层：深层特征提取
                nn.Sequential(
                    ResBlock(hidden_dim, hidden_dim*2),
                    ResBlock(hidden_dim*2, hidden_dim*2),
                    ResBlock(hidden_dim*2, hidden_dim)
                ),
                
                # 第三层：结构特征增强
                nn.Sequential(
                    ResBlock(hidden_dim, hidden_dim*2),
                    ResBlock(hidden_dim*2, hidden_dim),
                    ResBlock(hidden_dim, hidden_dim)
                ),
                
                # 输出层
                SineLayer(hidden_dim, output_dim, omega_0=20)  # 中等频率
            )
            
        elif expert_type == 'detail':
            # 细节专家：使用更高频率的SineLayer
            self.net = nn.Sequential(
                # 第一层：高频特征提取
                SineLayer(input_dim, hidden_dim, omega_0=50),  # 高频率
                
                # 第二层：细节特征增强
                nn.Sequential(
                    SineLayer(hidden_dim, hidden_dim*2, omega_0=60),
                    SineLayer(hidden_dim*2, hidden_dim, omega_0=60)
                ),
                
                # 第三层：精细细节处理
                nn.Sequential(
                    SineLayer(hidden_dim, hidden_dim*2, omega_0=70),
                    SineLayer(hidden_dim*2, hidden_dim, omega_0=70)
                ),
                
                # 输出层
                SineLayer(hidden_dim, output_dim, omega_0=50)  # 高频率
            )
            
        else:  # 'hybrid'
            # 混合专家：结合结构和细节特征
            # 结构分支
            self.structure_branch = nn.Sequential(
                ResBlock(input_dim, hidden_dim),
                ResBlock(hidden_dim, hidden_dim*2),
                ResBlock(hidden_dim*2, hidden_dim)
            )
            
            # 细节分支
            self.detail_branch = nn.Sequential(
                SineLayer(input_dim, hidden_dim, omega_0=50),
                SineLayer(hidden_dim, hidden_dim*2, omega_0=60),
                SineLayer(hidden_dim*2, hidden_dim, omega_0=60)
            )
            
            # 特征融合
            self.fusion = nn.Sequential(
                nn.Linear(hidden_dim*2, hidden_dim),
                nn.ReLU(),
                SineLayer(hidden_dim, output_dim, omega_0=30)  # 平衡频率
            )
    
    def forward(self, x):
        if self.expert_type == 'hybrid':
            # 混合专家：融合结构和细节特征
            structure_features = self.structure_branch(x)
            detail_features = self.detail_branch(x)
            combined_features = torch.cat([structure_features, detail_features], dim=-1)
            return self.fusion(combined_features)
        else:
            # 结构专家和细节专家
            return self.net(x)

class Router(nn.Module):
    """Router network for MoE that selects experts based on input features."""
    
    def __init__(self, input_dim, num_experts, temperature=0.1):
        """Initialize router network.
        
        Args:
            input_dim (int): Input feature dimension
            num_experts (int): Number of experts
            temperature (float): Temperature for softmax
        """
        super(Router, self).__init__()
        self.temperature = temperature
        self.num_experts = num_experts
        
        # Main router network
        self.router = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_experts)
        )
        
        # Region feature router
        self.region_router = nn.Sequential(
            nn.Linear(16, 64),  # 16 is the region feature dimension
            nn.ReLU(),
            nn.Linear(64, num_experts)
        )
        
        # Scale feature router
        self.scale_router = nn.Sequential(
            nn.Linear(8, 32),  # 8 is the scale feature dimension
            nn.ReLU(),
            nn.Linear(32, num_experts)
        )
        
        # Expert performance tracking
        self.register_buffer('expert_performance', torch.zeros(num_experts, 3))
        self.register_buffer('expert_counts', torch.zeros(num_experts, 3))
        
        # 复杂度阈值
        self.simple_threshold = 0.01
        self.medium_threshold = 0.01
    
    def get_complexity_level(self, complexity):
        """Get complexity level based on thresholds.
        
        Args:
            complexity (torch.Tensor): Complexity values
            
        Returns:
            torch.Tensor: Complexity levels (0=simple, 1=medium, 2=complex)
        """
        simple_threshold = torch.tensor([self.simple_threshold]).to(complexity.device)
        medium_threshold = torch.tensor([self.medium_threshold]).to(complexity.device)
        
        # 0=simple, 1=medium, 2=complex
        return torch.where(
            complexity <= simple_threshold,
            torch.zeros_like(complexity),
            torch.where(
                complexity <= medium_threshold,
                torch.ones_like(complexity),
                torch.ones_like(complexity) * 2
            )
        ).long()
    
    def forward(self, x, complexity=None):
        """Forward pass with complexity-based expert selection"""
        # 基本路由logits
        base_logits = self.router(x)
        
        if complexity is not None:
            # 1. 计算复杂度相关的权重
            complexity_weights = torch.ones_like(base_logits)
            
            # 简单区域（复杂度最低）
            simple_mask = complexity <= self.simple_threshold
            if torch.any(simple_mask):
                complexity_weights[simple_mask, :2] *= 2.0  # 增加简单专家的权重
                complexity_weights[simple_mask, 2:4] *= 1.0  # 保持中等专家的权重
                complexity_weights[simple_mask, 4:] *= 0.5  # 降低复杂专家的权重
            
            # 中等复杂区域
            medium_mask = (complexity > self.simple_threshold) & (complexity <= self.medium_threshold)
            if torch.any(medium_mask):
                complexity_weights[medium_mask, :2] *= 0.5  # 降低简单专家的权重
                complexity_weights[medium_mask, 2:4] *= 2.0  # 增加中等专家的权重
                complexity_weights[medium_mask, 4:] *= 0.5  # 降低复杂专家的权重
            
            # 非常复杂区域
            complex_mask = complexity > self.medium_threshold
            if torch.any(complex_mask):
                complexity_weights[complex_mask, :2] *= 0.5  # 降低简单专家的权重
                complexity_weights[complex_mask, 2:4] *= 1.0  # 保持中等专家的权重
                complexity_weights[complex_mask, 4:] *= 2.0  # 增加复杂专家的权重
            
            # 2. 应用复杂度权重
            base_logits = base_logits * complexity_weights
        
        # 应用softmax得到最终权重
        weights = F.softmax(base_logits / self.temperature, dim=-1)
        return weights
    
    def update_expert_performance(self, expert_weights, complexity, loss):
        """Update expert performance tracking.
        
        Args:
            expert_weights (torch.Tensor): Expert weights
            complexity (torch.Tensor): Complexity values
            loss (torch.Tensor): Loss values
        """
        # Get complexity levels
        complexity_levels = self.get_complexity_level(complexity)
        
        # Update performance tracking
        for i in range(self.num_experts):
            for j in range(3):  # For each complexity level
                mask = (complexity_levels == j)
                if mask.any():
                    expert_loss = (expert_weights[:, i] * loss * mask.float()).sum() / (mask.float().sum() + 1e-6)
                    self.expert_performance[i, j] = self.expert_performance[i, j] * 0.9 + expert_loss * 0.1
                    self.expert_counts[i, j] += mask.float().sum()
    
    def calculate_region_losses(self, complexity, expert_weights, targets, experts, encoded_coords):
        """计算专家损失。
        
        Args:
            complexity (torch.Tensor): 复杂度值
            expert_weights (torch.Tensor): 专家权重 [batch_size, num_experts]
            targets (torch.Tensor): 目标值 [batch_size, 3]
            experts (nn.ModuleList): 专家列表
            encoded_coords (torch.Tensor): 编码后的坐标
            
        Returns:
            tuple: (total_loss, mse_loss, entropy_loss, final_outputs, expert_weights)
                - total_loss: 总损失
                - mse_loss: MSE损失
                - entropy_loss: 熵损失
                - final_outputs: 最终输出
                - expert_weights: 专家权重
        """
        device = complexity.device
        
        # 计算所有专家的输出
        expert_outputs = []
        for expert in experts:
            expert_output = expert(encoded_coords)  # [batch_size, 3]
            expert_outputs.append(expert_output)
        
        # Stack expert outputs
        expert_outputs = torch.stack(expert_outputs, dim=1)  # [batch_size, num_experts, 3]
        
        # 应用专家权重
        final_outputs = torch.sum(expert_weights.unsqueeze(-1) * expert_outputs, dim=1)  # [batch_size, 3]
        
        # 计算MSE损失
        mse_loss = F.mse_loss(final_outputs, targets)
        
        # 计算熵损失
        entropy_loss = -torch.mean(torch.sum(expert_weights * torch.log(expert_weights + 1e-10), dim=1))
        
        # 总损失 = MSE损失 + 0.1 * 熵损失
        total_loss = mse_loss + 0.1 * entropy_loss
        
        return total_loss, mse_loss, entropy_loss, final_outputs, expert_weights

class MoEMultiscaleINR(nn.Module):
    """MoE Multiscale INR model for image representation."""
    
    def __init__(self, input_dim=2, output_dim=3, num_experts=6, num_scales=3):
        """Initialize MoE Multiscale INR model.
        
        Args:
            input_dim (int): Input dimension (default: 2 for coordinates)
            output_dim (int): Output dimension (default: 3 for RGB)
            num_experts (int): Number of experts (default: 6)
            num_scales (int): Number of scales (default: 3)
        """
        super(MoEMultiscaleINR, self).__init__()
        
        # Create positional encoding
        self.pos_encoding = PositionalEncoding(num_encoding_functions=6, include_input=True)
        encoded_dim = input_dim + self.pos_encoding.num_encoding_functions * 2 * (1 + self.pos_encoding.include_input)
        
        # Region feature encoder
        self.region_encoder = RegionFeatureExtractor(feature_dim=16)
        region_feature_dim = self.region_encoder.feature_dim
        
        # Scale encoders
        self.scale_encoders = nn.ModuleList([
            ScaleEncoder(scale_level=i, feature_dim=8) for i in range(num_scales)
        ])
        scale_feature_dim = 8
        
        # Expert networks - 2 experts per complexity level
        self.experts = nn.ModuleList([
            SimpleExpert(input_dim=encoded_dim, output_dim=output_dim, hidden_dim=64, expert_type='local'),
            SimpleExpert(input_dim=encoded_dim, output_dim=output_dim, hidden_dim=64, expert_type='global'),
            SimpleExpert(input_dim=encoded_dim, output_dim=output_dim, hidden_dim=128, expert_type='balanced'),
            ComplexExpert(input_dim=encoded_dim, output_dim=output_dim, hidden_dim=256, expert_type='structure'),
            ComplexExpert(input_dim=encoded_dim, output_dim=output_dim, hidden_dim=256, expert_type='detail'),
            ComplexExpert(input_dim=encoded_dim, output_dim=output_dim, hidden_dim=256, expert_type='hybrid')
        ])
        
        # Router network
        router_input_dim = encoded_dim + region_feature_dim + scale_feature_dim
        self.router = Router(input_dim=router_input_dim, num_experts=num_experts)
        
        self.num_experts = num_experts
        self.num_scales = num_scales
    
    def forward(self, coords, complexity=None, scale_level=None):
        """Forward pass of the model.
        
        Args:
            coords (torch.Tensor): Input coordinates
            complexity (torch.Tensor, optional): Complexity values
            scale_level (int, optional): Scale level to use
            
        Returns:
            tuple: (output, expert_weights)
                - output: Model output
                - expert_weights: Expert weights
        """
        # Apply positional encoding
        encoded_coords = self.pos_encoding(coords)
        
        # Get region features
        region_features = self.region_encoder(coords)
        
        # Get scale features
        if scale_level is not None and 0 <= scale_level < self.num_scales:
            scale_features = self.scale_encoders[scale_level](coords)
        else:
            # Use all scales and weighted average
            scale_features_list = [encoder(coords) for encoder in self.scale_encoders]
            scale_weights = torch.cat([encoder.get_relevance(coords) for encoder in self.scale_encoders], dim=-1)
            scale_weights = F.softmax(scale_weights, dim=-1)
            
            # Weighted combination of scale features
            scale_features = torch.zeros_like(scale_features_list[0])
            for i, feat in enumerate(scale_features_list):
                scale_features += scale_weights[:, i:i+1] * feat
        
        # Combine features for routing
        routing_features = torch.cat([encoded_coords, region_features, scale_features], dim=-1)
        
        # Get expert weights with complexity information
        expert_weights = self.router(routing_features, complexity)
        
        # Calculate expert outputs
        expert_outputs = []
        for expert in self.experts:
            expert_output = expert(encoded_coords)
            expert_outputs.append(expert_output)
        
        # Stack expert outputs
        expert_outputs = torch.stack(expert_outputs, dim=1)  # [batch_size, num_experts, 3]
        
        # Apply expert weights
        outputs = torch.sum(expert_weights.unsqueeze(-1) * expert_outputs, dim=1)  # [batch_size, 3]
        
        return outputs, expert_weights

def crop_center(img, crop_size):
    """Crop the center of an image"""
    y, x = img.shape[:2]
    startx = x//2 - crop_size//2
    starty = y//2 - crop_size//2
    return img[starty:starty+crop_size, startx:startx+crop_size]

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train MoE Multiscale INR model')
    parser.add_argument('--use_crop', action='store_true', help='Use cropped image instead of full image')
    parser.add_argument('--crop_size', type=int, default=512, help='Size of cropped image')
    parser.add_argument('--svs_path', type=str, default="/data/WRI_data/test.svs", help='Path to SVS file')
    parser.add_argument('--num_epochs', type=int, default=3000, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16384, help='Batch size for training')
    parser.add_argument('--num_samples', type=int, default=200000, help='Number of samples for adaptive sampling')
    parser.add_argument('--level', type=int, default=3, help='SVS pyramid level to use')
    args = parser.parse_args()

    # Check for GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load SVS file and get image information
    slide = openslide.OpenSlide(args.svs_path)
    level_dimensions = slide.level_dimensions
    level_downsamples = slide.level_downsamples
    
    print("\nSVS Image Information:")
    print(f"Level dimensions: {level_dimensions}")
    print(f"Level downsamples: {level_downsamples}")
    
    # Calculate original image size in bytes (RGB, 8 bits per channel)
    original_size_bytes = level_dimensions[0][0] * level_dimensions[0][1] * 3
    print(f"Original image size: {original_size_bytes / (1024*1024):.2f} MB")
    
    # Load and process the image
    whole_image = np.array(slide.read_region((0, 0), args.level, level_dimensions[args.level]))[:, :, :3] / 255.0
    slide.close()
    
    if args.use_crop:
        # Sample an image
        image = crop_center(whole_image, crop_size=args.crop_size)
        print(f"Using cropped image of size {args.crop_size}x{args.crop_size}")
    else:
        image = whole_image
        print(f"Using level {args.level} image of size {image.shape[0]}x{image.shape[1]}")
    
    # Calculate model size
    model = MoEMultiscaleINR(input_dim=2, output_dim=3, num_experts=6, num_scales=3)
    model_size_bytes = sum(p.numel() * 4 for p in model.parameters())  # 4 bytes per parameter (float32)
    print(f"Model size: {model_size_bytes / (1024*1024):.2f} MB")
    
    # Calculate compression ratio
    compression_ratio = original_size_bytes / model_size_bytes
    print(f"Compression ratio: {compression_ratio:.2f}:1")
    
    # Save the original image
    imsave("original-moe-cx2/original_image.png", (image * 255).astype(np.uint8))

    # Create dataset with complexity information
    coordinates, pixels, complexity = create_dataset(image)
    
    # Normalize pixel values to [-1, 1]
    pixels = 2 * pixels - 1
    
    # Use adaptive sampling based on complexity
    sampled_coords, sampled_pixels, sampled_complexity = adaptive_sampling(
        coordinates, pixels, complexity, num_samples=args.num_samples
    )
    
    # Create dataset and dataloader
    dataset = SlideDataset(sampled_coords, sampled_pixels, sampled_complexity)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=8,
        pin_memory=True
    )
    
    # Create MoE Multiscale INR model with 6 experts
    model = MoEMultiscaleINR(input_dim=2, output_dim=3, num_experts=6, num_scales=3).to(device)
    
    # Setup loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-6)
    
    # Add learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=50, verbose=True
    )
    
    # Training loop
    best_psnr = 0
    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0.0
        
        for coord_batch, pixel_batch, complexity_batch in dataloader:
            coord_batch, pixel_batch, complexity_batch = coord_batch.to(device), pixel_batch.to(device), complexity_batch.to(device)
            
            optimizer.zero_grad()
            
            # 1. 获取所有必要的特征
            encoded_coords = model.pos_encoding(coord_batch)
            region_features = model.region_encoder(coord_batch)
            scale_features = model.scale_encoders[0](coord_batch)  # 使用第一个尺度
            
            # 2. 拼接特征作为 router 的输入
            routing_features = torch.cat([encoded_coords, region_features, scale_features], dim=-1)
            
            # 3. 使用完整的特征进行路由
            expert_weights = model.router(routing_features, complexity_batch)
            
            # 4. 计算损失
            loss, mse_loss, entropy_loss, output, expert_weights = model.router.calculate_region_losses(
                complexity_batch, expert_weights, pixel_batch, model.experts, encoded_coords
            )
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Update expert performance
            with torch.no_grad():
                model.router.update_expert_performance(expert_weights, complexity_batch, loss)
            
            
            epoch_loss += loss.item()
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{args.num_epochs}, Loss: {epoch_loss / len(dataloader)}")
        
        # Evaluate and save reconstructed image
        if (epoch + 1) % 100 == 0:
            model.eval()
            with torch.no_grad():
                # Process in chunks to avoid OOM
                chunk_size = 50000
                num_chunks = len(coordinates) // chunk_size + (1 if len(coordinates) % chunk_size != 0 else 0)
                
                reconstructed_pixels = []
                for i in range(num_chunks):
                    start_idx = i * chunk_size
                    end_idx = min((i + 1) * chunk_size, len(coordinates))
                    
                    coords_chunk = torch.tensor(coordinates[start_idx:end_idx]).to(device)
                    output_chunk = model(coords_chunk)[0].cpu().numpy()
                    reconstructed_pixels.append(output_chunk)
                
                reconstructed_pixels = np.vstack(reconstructed_pixels)
                reconstructed_image = reconstructed_pixels.reshape(image.shape)
                
                # Convert back to [0, 1] range
                reconstructed_image = (reconstructed_image + 1) / 2.0
                
                # Calculate PSNR
                original_image = (pixels.reshape(image.shape) + 1) / 2.0
                psnr_value = psnr(original_image, reconstructed_image)
                print(f"PSNR of reconstructed image: {psnr_value:.2f}")
                
                # Update learning rate based on PSNR
                scheduler.step(psnr_value)
                
                # Save best model
                if psnr_value > best_psnr:
                    best_psnr = psnr_value
                    checkpoint = {
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'loss': epoch_loss / len(dataloader),
                        'psnr': psnr_value
                    }
                    torch.save(checkpoint, "original-moe-cx2/best_model.pth")
                    print(f"New best PSNR: {psnr_value:.2f}, model saved")
                
                # Save the image
                imsave(f"original-moe-cx2/reconstructed_image_{epoch+1:06d}.png", 
                       (reconstructed_image * 255).astype(np.uint8))
            
            # Free GPU memory
            torch.cuda.empty_cache()
        
        # Save checkpoint
        if (epoch + 1) % 500 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': epoch_loss / len(dataloader)
            }
            torch.save(checkpoint, f"original-moe-cx2/moe_multiscale_checkpoint_{epoch+1}.pth")
            print(f"Checkpoint saved for epoch {epoch + 1}")

if __name__ == "__main__":
    main()