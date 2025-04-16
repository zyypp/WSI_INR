import numpy as np
import openslide
from scipy import ndimage
import torch
from torch.utils.data import Dataset

def load_svs_image(svs_path, level=0):
    """Load and process SVS image from file.
    
    Args:
        svs_path (str): Path to the SVS file
        level (int): Pyramid level to load (default: 0)
        
    Returns:
        numpy.ndarray: Normalized image array in range [0, 1]
    """
    slide = openslide.OpenSlide(svs_path)
    image = np.array(slide.read_region((0, 0), level, slide.level_dimensions[level]))[:, :, :3] / 255.0
    slide.close()
    return image

def compute_complexity_map(image, sigma=1.0):
    """Compute complexity map from image using gradient magnitude.
    
    Args:
        image (numpy.ndarray): Input image
        sigma (float): Gaussian filter sigma for smoothing
        
    Returns:
        numpy.ndarray: Normalized complexity map in range [0, 1]
    """
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

def create_dataset(image):
    """Create coordinate-pixel dataset with region complexity.
    
    Args:
        image (numpy.ndarray): Input image
        
    Returns:
        tuple: (coordinates, pixels, complexity)
            - coordinates: Normalized coordinates in range [0, 1]
            - pixels: RGB values
            - complexity: Complexity values for each pixel
    """
    h, w, _ = image.shape
    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    coordinates = np.stack((x.ravel(), y.ravel()), axis=-1) / np.array([w, h])  # Normalize
    pixels = image.reshape(-1, 3)  # RGB values
    
    # Compute complexity map
    complexity_map = compute_complexity_map(image)
    complexity = complexity_map.ravel()
    
    return coordinates.astype(np.float32), pixels.astype(np.float32), complexity.astype(np.float32)

def adaptive_sampling(coordinates, pixels, complexity, num_samples=100000):
    """Perform adaptive sampling based on complexity.
    
    Args:
        coordinates (numpy.ndarray): Input coordinates
        pixels (numpy.ndarray): Input pixels
        complexity (numpy.ndarray): Complexity values
        num_samples (int): Number of samples to generate
        
    Returns:
        tuple: (sampled_coordinates, sampled_pixels, sampled_complexity)
    """
    # 确定复杂区域和简单区域
    complex_mask = complexity > 0.01
    simple_mask = ~complex_mask
    
    # 分配采样数量 - 复杂区域占75%的采样点，简单区域占25%
    complex_samples = int(num_samples * 0.75)
    simple_samples = num_samples - complex_samples
    
    # 分离复杂区域和简单区域的数据
    complex_coords = coordinates[complex_mask]
    complex_pixels = pixels[complex_mask]
    complex_complexity = complexity[complex_mask]
    
    simple_coords = coordinates[simple_mask]
    simple_pixels = pixels[simple_mask]
    simple_complexity = complexity[simple_mask]
    
    # 确保有足够的样本可供选择
    complex_samples = min(complex_samples, len(complex_coords))
    simple_samples = min(simple_samples, len(simple_coords))
    
    # 如果其中一个区域的样本不足，将剩余的样本分配给另一个区域
    if complex_samples < int(num_samples * 0.75):
        simple_samples = num_samples - complex_samples
    if simple_samples < num_samples - complex_samples:
        complex_samples = num_samples - simple_samples
    
    # 在复杂区域内均匀采样（不再使用复杂度作为权重）
    if len(complex_coords) > 0:
        complex_indices = np.random.choice(len(complex_coords), size=complex_samples, replace=True)
        sampled_complex_coords = complex_coords[complex_indices]
        sampled_complex_pixels = complex_pixels[complex_indices]
        sampled_complex_complexity = complex_complexity[complex_indices]
    else:
        sampled_complex_coords = np.array([])
        sampled_complex_pixels = np.array([])
        sampled_complex_complexity = np.array([])
    
    # 在简单区域内也均匀采样
    if len(simple_coords) > 0:
        simple_indices = np.random.choice(len(simple_coords), size=simple_samples, replace=True)
        sampled_simple_coords = simple_coords[simple_indices]
        sampled_simple_pixels = simple_pixels[simple_indices]
        sampled_simple_complexity = simple_complexity[simple_indices]
    else:
        sampled_simple_coords = np.array([])
        sampled_simple_pixels = np.array([])
        sampled_simple_complexity = np.array([])
    
    # 如果其中一个区域没有样本，全部使用另一个区域的样本
    if len(sampled_complex_coords) == 0:
        return sampled_simple_coords, sampled_simple_pixels, sampled_simple_complexity
    if len(sampled_simple_coords) == 0:
        return sampled_complex_coords, sampled_complex_pixels, sampled_complex_complexity
    
    # 合并两个区域的采样结果
    sampled_coords = np.vstack((sampled_complex_coords, sampled_simple_coords))
    sampled_pixels = np.vstack((sampled_complex_pixels, sampled_simple_pixels))
    sampled_complexity = np.concatenate((sampled_complex_complexity, sampled_simple_complexity))
    
    print(f"采样统计: 总样本 {len(sampled_coords)}, 复杂区域 {len(sampled_complex_coords)} ({len(sampled_complex_coords)/len(sampled_coords):.1%}), 简单区域 {len(sampled_simple_coords)} ({len(sampled_simple_coords)/len(sampled_coords):.1%})")
    
    return sampled_coords, sampled_pixels, sampled_complexity

class SlideDataset(Dataset):
    """Dataset for coordinate-pixel pairs with complexity information."""
    
    def __init__(self, coordinates, pixels, region_complexity=None):
        """Initialize dataset.
        
        Args:
            coordinates (numpy.ndarray): Input coordinates
            pixels (numpy.ndarray): Input pixels
            region_complexity (numpy.ndarray, optional): Complexity values
        """
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