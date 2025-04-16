import numpy as np
import openslide
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.io import imsave
from scipy import ndimage

def crop_center(image, crop_size=256):
    h, w, c = image.shape
    start_x = (w - crop_size) // 2
    start_y = (h - crop_size) // 2

    # Crop the image (handle the color channel by slicing the last dimension)
    cropped_image = image[start_y:start_y + crop_size, start_x:start_x + crop_size, :]
    return cropped_image.astype(np.float32)

# Positional Encoding (from existing code)
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

# SineLayer (from existing code)
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

# ResBlock (from existing code)
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

# Expert Networks with varying complexity
class SimpleExpert(nn.Module):
    def __init__(self, input_dim, output_dim=3, hidden_dim=64):
        super(SimpleExpert, self).__init__()
        self.net = nn.Sequential(
            SineLayer(input_dim, hidden_dim),
            SineLayer(hidden_dim, hidden_dim),
            SineLayer(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

class MediumExpert(nn.Module):
    def __init__(self, input_dim, output_dim=3, hidden_dim=128):
        super(MediumExpert, self).__init__()
        self.net = nn.Sequential(
            ResBlock(input_dim, hidden_dim),
            ResBlock(hidden_dim, hidden_dim),
            SineLayer(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

class ComplexExpert(nn.Module):
    def __init__(self, input_dim, output_dim=3, hidden_dim=256):
        super(ComplexExpert, self).__init__()
        self.net = nn.Sequential(
            ResBlock(input_dim, hidden_dim),
            ResBlock(hidden_dim, hidden_dim*2),
            ResBlock(hidden_dim*2, hidden_dim),
            SineLayer(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

# Region Feature Extractor
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

# Router Network for MoE
class Router(nn.Module):
    def __init__(self, input_dim, num_experts, temperature=0.1):
        super(Router, self).__init__()
        self.temperature = temperature
        self.router = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_experts)
        )
    
    def forward(self, x):
        # Get logits for each expert
        logits = self.router(x)
        # Apply softmax with temperature to get weights
        weights = F.softmax(logits / self.temperature, dim=-1)
        return weights

# Scale-aware encoder
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

# MoE Multiscale INR model
class MoEMultiscaleINR(nn.Module):
    def __init__(self, input_dim=2, output_dim=3, num_experts=9, num_scales=3):
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
            SimpleExpert(input_dim=encoded_dim, output_dim=output_dim, hidden_dim=64)
            for _ in range(num_experts // 3)
        ] + [
            MediumExpert(input_dim=encoded_dim, output_dim=output_dim, hidden_dim=128)
            for _ in range(num_experts // 3)
        ] + [
            ComplexExpert(input_dim=encoded_dim, output_dim=output_dim, hidden_dim=256)
            for _ in range(num_experts // 3)
        ])
        
        # Router network
        router_input_dim = encoded_dim + region_feature_dim + scale_feature_dim
        self.router = Router(input_dim=router_input_dim, num_experts=num_experts)
        
        self.num_experts = num_experts
        self.num_scales = num_scales
    
    def forward(self, coords, scale_level=None):
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
        
        # Get expert weights from router
        expert_weights = self.router(routing_features)
        
        # Apply experts and combine their outputs
        outputs = torch.zeros(coords.shape[0], 3, device=coords.device)
        for i, expert in enumerate(self.experts):
            expert_output = expert(encoded_coords)
            outputs += expert_weights[:, i:i+1] * expert_output
        
        return outputs

# Dataset for coordinate-pixel pairs
class SlideDataset(Dataset):
    def __init__(self, coordinates, pixels, region_complexity=None):
        self.coordinates = coordinates
        self.pixels = pixels
        self.region_complexity = region_complexity
    
    def __len__(self):
        return len(self.coordinates)
    
    def __getitem__(self, idx):
        if self.region_complexity is not None:
            return self.coordinates[idx], self.pixels[idx], self.region_complexity[idx]
        else:
            return self.coordinates[idx], self.pixels[idx]

# Function to load and process SVS image
def load_svs_image(svs_path, level=0):
    slide = openslide.OpenSlide(svs_path)
    image = np.array(slide.read_region((0, 0), level, slide.level_dimensions[level]))[:, :, :3] / 255.0
    slide.close()
    return image

# Function to compute complexity map from image
def compute_complexity_map(image, sigma=1.0):
    # Compute gradients in x and y direction
    grad_x = ndimage.sobel(image, axis=1)
    grad_y = ndimage.sobel(image, axis=0)
    
    # Compute gradient magnitude
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Convert to single channel if it's RGB
    if len(magnitude.shape) > 2:
        magnitude = np.mean(magnitude, axis=2)
    
    # Smooth the magnitude to reduce noise
    magnitude = ndimage.gaussian_filter(magnitude, sigma=sigma)
    
    # Normalize to [0, 1]
    magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)
    
    return magnitude

# Prepare coordinate-pixel dataset with region complexity
def create_dataset(image):
    h, w, _ = image.shape
    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    coordinates = np.stack((x.ravel(), y.ravel()), axis=-1) / np.array([w, h])  # Normalize
    pixels = image.reshape(-1, 3)  # RGB values
    
    # Compute complexity map
    complexity_map = compute_complexity_map(image)
    complexity = complexity_map.ravel()
    
    return coordinates.astype(np.float32), pixels.astype(np.float32), complexity.astype(np.float32)

# Adaptive sampling based on complexity
def adaptive_sampling(coordinates, pixels, complexity, num_samples=100000):
    # Convert complexity to sampling probability
    p = complexity / np.sum(complexity)
    
    # Sample indices based on complexity
    indices = np.random.choice(len(coordinates), size=num_samples, p=p, replace=True)
    
    return coordinates[indices], pixels[indices], complexity[indices]

# Main function
if __name__ == "__main__":
    # Check for GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load and process the image
    svs_path = "./data/test.svs"  # Replace with your file path
    whole_image = load_svs_image(svs_path, level=3)
    
    # Sample an image
    image = crop_center(whole_image, crop_size=512)

    # Save the original image
    imsave("original_image.png", (image * 255).astype(np.uint8))

    # Create dataset with complexity information
    coordinates, pixels, complexity = create_dataset(image)
    
    # Save complexity map visualization
    complexity_map = complexity.reshape(image.shape[0], image.shape[1])
    imsave("complexity_map.png", (complexity_map * 255).astype(np.uint8))
    
    # Normalize pixel values to [-1, 1]
    pixels = 2 * pixels - 1
    
    # Use adaptive sampling based on complexity
    sampled_coords, sampled_pixels, sampled_complexity = adaptive_sampling(
        coordinates, pixels, complexity, num_samples=200000
    )
    
    # Create dataset and dataloader
    dataset = SlideDataset(sampled_coords, sampled_pixels, sampled_complexity)
    dataloader = DataLoader(dataset, batch_size=16384, shuffle=True, num_workers=4)

    # Create MoE Multiscale INR model
    model = MoEMultiscaleINR(input_dim=2, output_dim=3, num_experts=9, num_scales=3).to(device)
    
    # Setup loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-6)
    
    # Training loop
    epochs = 3000
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for coord_batch, pixel_batch, _ in dataloader:
            coord_batch, pixel_batch = coord_batch.to(device), pixel_batch.to(device)
            
            optimizer.zero_grad()
            output = model(coord_batch)
            loss = criterion(output, pixel_batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(dataloader)}")
        
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
                    output_chunk = model(coords_chunk).cpu().numpy()
                    reconstructed_pixels.append(output_chunk)
                
                reconstructed_pixels = np.vstack(reconstructed_pixels)
                reconstructed_image = reconstructed_pixels.reshape(image.shape)
                
                # Convert back to [0, 1] range
                reconstructed_image = (reconstructed_image + 1) / 2.0
                
                # Calculate PSNR
                original_image = (pixels.reshape(image.shape) + 1) / 2.0
                psnr_value = psnr(original_image, reconstructed_image)
                print(f"PSNR of reconstructed image: {psnr_value:.2f}")
                
                # Save the image
                imsave(f"reconstructed_image_{epoch+1:06d}.png", 
                       (reconstructed_image * 255).astype(np.uint8))
            
            # Free GPU memory
            torch.cuda.empty_cache()
            
        # Save checkpoint
        if (epoch + 1) % 500 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss / len(dataloader)
            }
            torch.save(checkpoint, f"moe_multiscale_checkpoint_{epoch+1}.pth")
            print(f"Checkpoint saved for epoch {epoch + 1}")