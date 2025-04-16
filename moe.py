import torch
import torch.nn as nn
import torch.nn.functional as F
from experts import SimpleExpert, ComplexExpert
from feature_extractors import PositionalEncoding, RegionFeatureExtractor, ScaleEncoder

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
        self.register_buffer('expert_performance', torch.zeros(num_experts, 2))
        self.register_buffer('expert_counts', torch.zeros(num_experts, 2))
        
        # 复杂度阈值
        self.complexity_threshold = 0.01
        
        # 损失权重
        self.simple_weight = 1.0
        self.complex_weight = 5.0
    
    def get_complexity_level(self, complexity):
        """Get complexity level based on threshold.
        
        Args:
            complexity (torch.Tensor): Complexity values
            
        Returns:
            torch.Tensor: Complexity levels (0 or 1)
        """
        threshold = torch.tensor([self.complexity_threshold]).to(complexity.device)
        return (complexity > threshold).long()
    
    def forward(self, x, complexity=None):
        """Forward pass with strict complexity-based expert category selection"""
        # 基本路由logits
        base_logits = self.router(x)
        
        if complexity is not None:
            # 1. 根据0.01阈值获取复杂度级别
            complexity_levels = self.get_complexity_level(complexity)
            
            # 2. 创建专家掩码：强制根据复杂度选择对应类别的专家
            batch_size = x.shape[0]
            expert_masks = torch.zeros(batch_size, self.num_experts, device=x.device)
            
            # 对于简单区域（复杂度级别0）：只允许使用前3个专家
            simple_indices = (complexity_levels == 0).nonzero().squeeze(-1)
            if len(simple_indices) > 0:
                # 将前3个专家的位置设为1，其他为0
                simple_mask = torch.zeros(1, self.num_experts, device=x.device)
                simple_mask[0, :3] = 1.0
                expert_masks[simple_indices] = simple_mask
                
                # 使用区域和尺度特征作为简单专家之间的选择依据
                region_weights_simple = self.region_router(x[simple_indices, :16])  # Ensure x has at least 16 features
                scale_weights_simple = self.scale_router(x[simple_indices, -8:])  # Ensure x has at least 8 features
                
                # 只保留前3个专家的权重，其他设为非常小的值
                region_weights_simple_filtered = torch.zeros_like(region_weights_simple)
                region_weights_simple_filtered[:, :3] = region_weights_simple[:, :3]
                
                scale_weights_simple_filtered = torch.zeros_like(scale_weights_simple)
                scale_weights_simple_filtered[:, :3] = scale_weights_simple[:, :3]
                
                # 结合权重（区域和尺度特征权重仅影响同类专家的选择）
                combined_weights_simple = (
                    region_weights_simple_filtered * 0.5 +
                    scale_weights_simple_filtered * 0.5
                )
                
                # 应用专家性能历史（只对简单专家）
                performance_weights_simple = F.softmax(self.expert_performance[:3, 0], dim=0)
                combined_weights_simple[:, :3] = combined_weights_simple[:, :3] * performance_weights_simple
                
                # 设置基本logits
                base_logits[simple_indices] = combined_weights_simple
            
            # 对于复杂区域（复杂度级别1）：只允许使用后3个专家
            complex_indices = (complexity_levels == 1).nonzero().squeeze(-1)
            if len(complex_indices) > 0:
                # 将后3个专家的位置设为1，其他为0
                complex_mask = torch.zeros(1, self.num_experts, device=x.device)
                complex_mask[0, 3:] = 1.0
                expert_masks[complex_indices] = complex_mask
                
                # 使用区域和尺度特征作为复杂专家之间的选择依据
                region_weights_complex = self.region_router(x[complex_indices, :16])
                scale_weights_complex = self.scale_router(x[complex_indices, -8:])
                
                # 只保留后3个专家的权重，其他设为非常小的值
                region_weights_complex_filtered = torch.zeros_like(region_weights_complex)
                region_weights_complex_filtered[:, 3:] = region_weights_complex[:, 3:]
                
                scale_weights_complex_filtered = torch.zeros_like(scale_weights_complex)
                scale_weights_complex_filtered[:, 3:] = scale_weights_complex[:, 3:]
                
                # 结合权重（区域和尺度特征权重仅影响同类专家的选择）
                combined_weights_complex = (
                    region_weights_complex_filtered * 0.5 +
                    scale_weights_complex_filtered * 0.5
                )
                
                # 应用专家性能历史（只对复杂专家）
                performance_weights_complex = F.softmax(self.expert_performance[3:, 1], dim=0)
                combined_weights_complex[:, 3:] = combined_weights_complex[:, 3:] * performance_weights_complex
                
                # 设置基本logits
                base_logits[complex_indices] = combined_weights_complex
            
            # 强制应用掩码，使得不相关的专家权重为0
            base_logits = base_logits * expert_masks
        
        # 应用softmax with temperature得到最终权重
        weights = F.softmax(base_logits / self.temperature, dim=-1)
        return weights
    
    def update_expert_performance(self, expert_weights, complexity, loss):
        """Update expert performance tracking.
        
        Args:
            expert_weights (torch.Tensor): Expert weights
            complexity (torch.Tensor): Complexity values
            loss (torch.Tensor): Loss values
        """
        # Get complexity levels with 0.01 threshold
        complexity_levels = self.get_complexity_level(complexity)
        
        # Update performance tracking
        for i in range(self.num_experts):
            for j in range(2):  # For each complexity level
                mask = (complexity_levels == j)
                if mask.any():
                    expert_loss = (expert_weights[:, i] * loss * mask.float()).sum() / (mask.float().sum() + 1e-6)
                    self.expert_performance[i, j] = self.expert_performance[i, j] * 0.9 + expert_loss * 0.1
                    self.expert_counts[i, j] += mask.float().sum()
    
    def calculate_region_losses(self, complexity, outputs, targets, experts, encoded_coords):
        """计算不同复杂度区域的专家损失。
        
        Args:
            complexity (torch.Tensor): 复杂度值
            outputs (torch.Tensor): 模型输出
            targets (torch.Tensor): 目标值
            experts (nn.ModuleList): 专家列表
            encoded_coords (torch.Tensor): 编码后的坐标
            
        Returns:
            tuple: (total_loss, simple_loss, complex_loss, final_outputs, expert_weights)
                - total_loss: 总损失
                - simple_loss: 简单区域损失
                - complex_loss: 复杂区域损失
                - final_outputs: 最终输出
                - expert_weights: 专家权重
        """
        device = complexity.device
        batch_size = complexity.shape[0]
        
        # 区分简单区域和复杂区域
        simple_mask = complexity <= self.complexity_threshold
        complex_mask = complexity > self.complexity_threshold
        
        # 初始化输出和损失值
        final_outputs = torch.zeros_like(targets)
        simple_loss = torch.tensor(0.0, device=device)
        complex_loss = torch.tensor(0.0, device=device)
        
        # 处理简单区域 - 只使用前3个简单专家
        if torch.any(simple_mask):
            simple_encoded = encoded_coords[simple_mask]
            simple_target = targets[simple_mask]
            simple_weights = outputs[simple_mask, :3]  # 只取前3个专家的权重
            
            # 计算简单区域的输出
            simple_output = torch.zeros_like(simple_target)
            for i in range(3):  # 前3个专家是简单专家
                expert = experts[i]
                expert_output = expert(simple_encoded)
                simple_output += simple_weights[:, i:i+1] * expert_output
            
            # 计算简单区域的MSE损失
            simple_loss = F.mse_loss(simple_output, simple_target)
            
            # 将简单区域的输出放入最终输出张量
            final_outputs[simple_mask] = simple_output
        
        # 处理复杂区域 - 只使用后3个复杂专家
        if torch.any(complex_mask):
            complex_encoded = encoded_coords[complex_mask]
            complex_target = targets[complex_mask]
            complex_weights = outputs[complex_mask, 3:]  # 只取后3个专家的权重
            
            # 计算复杂区域的输出
            complex_output = torch.zeros_like(complex_target)
            for i in range(3, 6):  # 后3个专家是复杂专家
                expert = experts[i]
                expert_output = expert(complex_encoded)
                complex_output += complex_weights[:, i-3:i-2] * expert_output
            
            # 计算复杂区域的MSE损失
            complex_loss = F.mse_loss(complex_output, complex_target)
            
            # 将复杂区域的输出放入最终输出张量
            final_outputs[complex_mask] = complex_output
        
        # 总损失是简单损失和复杂损失的和
        total_loss = simple_loss + 5 * complex_loss
        
        # 随机打印损失统计信息
        if torch.rand(1).item() < 0.01:  # 随机输出1%的批次信息
            simple_count = torch.sum(simple_mask).item()
            complex_count = torch.sum(complex_mask).item()
            print(f"Router区域损失 - 简单: {simple_loss.item():.6f}, 复杂: {complex_loss.item():.6f}, 总计: {total_loss.item():.6f}")
            print(f"样本分布 - 简单: {simple_count}/{batch_size} ({simple_count/batch_size:.1%}), 复杂: {complex_count}/{batch_size} ({complex_count/batch_size:.1%})")
        
        return total_loss, simple_loss, complex_loss, final_outputs, outputs

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
        
        # Expert networks - 3 experts per complexity level
        self.experts = nn.ModuleList([
            SimpleExpert(input_dim=encoded_dim, output_dim=output_dim, hidden_dim=64, expert_type='local'),
            SimpleExpert(input_dim=encoded_dim, output_dim=output_dim, hidden_dim=64, expert_type='global'),
            SimpleExpert(input_dim=encoded_dim, output_dim=output_dim, hidden_dim=64, expert_type='balanced')
        ] + [
            ComplexExpert(input_dim=encoded_dim, output_dim=output_dim, hidden_dim=256, expert_type='structure'),
            ComplexExpert(input_dim=encoded_dim, output_dim=output_dim, hidden_dim=256, expert_type='detail'),
            ComplexExpert(input_dim=encoded_dim, output_dim=output_dim, hidden_dim=256, expert_type='hybrid')
        ])
        
        # Router network
        router_input_dim = encoded_dim + region_feature_dim + scale_feature_dim  # 计算实际的路由器输入维度
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
        
        # Apply experts and combine their outputs
        outputs = torch.zeros(coords.shape[0], 3, device=coords.device)
        for i, expert in enumerate(self.experts):
            expert_output = expert(encoded_coords)
            outputs += expert_weights[:, i:i+1] * expert_output
        
        return outputs, expert_weights
    
