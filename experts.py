import torch
import torch.nn as nn
import numpy as np

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