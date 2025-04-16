"""
Image Quality Evaluation Metrics

This script provides functions to evaluate image quality using various metrics:
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- LPIPS (Learned Perceptual Image Patch Similarity)
- FID (Fréchet Inception Distance)

Usage:
    python evaluate_metrics.py --original_path PATH --reconstructed_path PATH
"""

import os
import torch
import numpy as np
import lpips
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from scipy import linalg
from torchvision import models, transforms
from PIL import Image
import argparse
import glob
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize LPIPS model
lpips_fn = lpips.LPIPS(net='alex')

# Initialize Inception model for FID
inception_model = models.inception_v3(pretrained=True, transform_input=False)
inception_model.fc = torch.nn.Identity()  # Remove final classification layer
inception_model.eval()

def load_image(image_path):
    """Load and preprocess image for evaluation"""
    if isinstance(image_path, str):
        image = Image.open(image_path).convert('RGB')
    else:
        # Assuming it's a numpy array
        if isinstance(image_path, np.ndarray):
            # If the data is in [0,1] range, convert to [0,255]
            if image_path.max() <= 1.0:
                image = (image_path * 255).astype(np.uint8)
            image = Image.fromarray(image).convert('RGB')
        else:
            raise TypeError("Image path must be a string or numpy array")
    
    transform = transforms.Compose([
        transforms.Resize((299, 299)),  # Inception model expects 299x299
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def calculate_psnr(original, reconstructed):
    """Calculate PSNR between original and reconstructed images"""
    # Make sure data is in valid range
    if original.max() > 1.0 or reconstructed.max() > 1.0:
        original = original / 255.0 if original.max() > 1.0 else original
        reconstructed = reconstructed / 255.0 if reconstructed.max() > 1.0 else reconstructed
    
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return 100  # Perfect match
    
    max_pixel = 1.0
    psnr_value = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr_value

def calculate_ssim(original, reconstructed):
    """
    Calculate SSIM between original and reconstructed images.
    Handles small images by adjusting window size.
    """
    # Ensure both images are in same range
    if original.max() > 1.0 or reconstructed.max() > 1.0:
        original = original / 255.0 if original.max() > 1.0 else original
        reconstructed = reconstructed / 255.0 if reconstructed.max() > 1.0 else reconstructed
    
    # Get image dimensions
    h, w = original.shape[:2]
    
    # Calculate appropriate window size
    win_size = min(7, min(h, w))
    if win_size % 2 == 0:  # Ensure window size is odd
        win_size -= 1
    
    # Calculate SSIM with adjusted parameters
    if len(original.shape) == 3:  # Color image
        return ssim(original, reconstructed, 
                   win_size=win_size,
                   channel_axis=2,  # Specify channel axis for color images
                   data_range=1.0,
                   gaussian_weights=True)
    else:  # Grayscale image
        return ssim(original, reconstructed,
                   win_size=win_size,
                   data_range=1.0,
                   gaussian_weights=True)

def calculate_lpips(original, reconstructed):
    """Calculate LPIPS between original and reconstructed images"""
    # Ensure inputs are torch tensors with correct shape
    if isinstance(original, np.ndarray):
        original = torch.from_numpy(original).float()
        if original.shape[-1] == 3:  # HWC to CHW
            original = original.permute(2, 0, 1)
        if original.max() > 1.0:
            original = original / 255.0
        original = original.unsqueeze(0)
    
    if isinstance(reconstructed, np.ndarray):
        reconstructed = torch.from_numpy(reconstructed).float()
        if reconstructed.shape[-1] == 3:  # HWC to CHW
            reconstructed = reconstructed.permute(2, 0, 1)
        if reconstructed.max() > 1.0:
            reconstructed = reconstructed / 255.0
        reconstructed = reconstructed.unsqueeze(0)
    
    return lpips_fn(original, reconstructed).mean().item()

def calculate_fid(original_features, reconstructed_features):
    """Calculate FID between original and reconstructed image features"""
    # Convert to numpy arrays if they're tensors
    if torch.is_tensor(original_features):
        original_features = original_features.cpu().numpy()
    if torch.is_tensor(reconstructed_features):
        reconstructed_features = reconstructed_features.cpu().numpy()
    
    # Ensure features are 2D arrays with shape (n_samples, n_features)
    original_features = np.atleast_2d(original_features)
    reconstructed_features = np.atleast_2d(reconstructed_features)
    
    # Ensure both feature matrices have the same number of features
    if original_features.shape[1] != reconstructed_features.shape[1]:
        min_features = min(original_features.shape[1], reconstructed_features.shape[1])
        original_features = original_features[:, :min_features]
        reconstructed_features = reconstructed_features[:, :min_features]
    
    # Calculate means
    mu1 = np.mean(original_features, axis=0)
    mu2 = np.mean(reconstructed_features, axis=0)
    
    # Calculate covariance matrices with safe handling for small sample sizes
    if original_features.shape[0] == 1:
        # If we have only one sample, use a small diagonal covariance
        sigma1 = np.eye(original_features.shape[1]) * 1e-6
    else:
        sigma1 = np.cov(original_features, rowvar=False)
    
    if reconstructed_features.shape[0] == 1:
        sigma2 = np.eye(reconstructed_features.shape[1]) * 1e-6
    else:
        sigma2 = np.cov(reconstructed_features, rowvar=False)
    
    # Add small constant to diagonal to ensure positive definiteness
    eps = 1e-6
    sigma1 = sigma1 + eps * np.eye(sigma1.shape[0])
    sigma2 = sigma2 + eps * np.eye(sigma2.shape[0])
    
    # Calculate squared sum of differences
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    
    # Calculate square root of product of covariances
    try:
        covmean = linalg.sqrtm(sigma1.dot(sigma2))
        # Ensure the result is real
        if np.iscomplexobj(covmean):
            covmean = covmean.real
    except (ValueError, np.linalg.LinAlgError):
        # If sqrtm fails, use a simplified calculation
        print("Warning: Using simplified FID calculation due to numerical issues")
        covmean = np.zeros_like(sigma1)
        for i in range(sigma1.shape[0]):
            for j in range(sigma1.shape[1]):
                covmean[i, j] = np.sqrt(max(sigma1[i, i] * sigma2[j, j], 0))
    
    # Calculate FID
    try:
        fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
        # Ensure FID is positive and not NaN
        if np.isnan(fid) or fid < 0:
            print("Warning: Invalid FID value detected, using fallback")
            fid = ssdiff + np.trace(sigma1 + sigma2)  # Simplified fallback
    except Exception as e:
        print(f"Error in FID calculation: {e}")
        fid = ssdiff + np.trace(sigma1 + sigma2)  # Simplified fallback
    
    return float(fid)

def extract_features(images):
    """Extract features using Inception model"""
    features = []
    with torch.no_grad():
        for img in images:
            # Ensure image is in correct format
            if isinstance(img, np.ndarray):
                # Convert from HWC to CHW format
                if img.shape[-1] == 3:
                    img = np.transpose(img, (2, 0, 1))
                # Scale to [0, 1] if needed
                if img.max() > 1.0:
                    img = img / 255.0
                img = torch.from_numpy(img).float()
            
            if len(img.shape) == 3:
                img = img.unsqueeze(0)
            
            # Move to device if needed
            if torch.cuda.is_available():
                img = img.cuda()
                inception_model.cuda()
            
            # Extract features
            try:
                feature = inception_model(img)
                # Handle both tuple and tensor outputs
                if isinstance(feature, tuple):
                    feature = feature[0]  # Take the first output if it's a tuple
                features.append(feature.squeeze().cpu().numpy())
            except Exception as e:
                print(f"Error extracting features: {e}")
                # Return empty feature as fallback
                features.append(np.zeros(2048))
    
    # Stack features into a single array
    if features:
        features = np.vstack(features)
    else:
        features = np.zeros((1, 2048))  # Fallback
    
    # Ensure features are in correct shape
    if len(features.shape) == 1:
        features = features.reshape(1, -1)
    
    return features

def evaluate_images(original_path, reconstructed_path):
    """Evaluate image quality metrics between original and reconstructed images"""
    # Load images with proper error handling
    try:
        original_img = load_image(original_path)
        reconstructed_img = load_image(reconstructed_path)
        
        # Convert to numpy arrays for PSNR and SSIM
        original_np = original_img.squeeze().permute(1, 2, 0).numpy()
        reconstructed_np = reconstructed_img.squeeze().permute(1, 2, 0).numpy()
        
        # Denormalize (assuming images were normalized with ImageNet stats)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        original_np = original_np * std + mean
        reconstructed_np = reconstructed_np * std + mean
        
        # Clip to valid range
        original_np = np.clip(original_np, 0, 1)
        reconstructed_np = np.clip(reconstructed_np, 0, 1)
        
        # Calculate metrics
        psnr_value = calculate_psnr(original_np, reconstructed_np)
        ssim_value = calculate_ssim(original_np, reconstructed_np)
        lpips_value = calculate_lpips(original_img, reconstructed_img)
        
        # Calculate FID with proper error handling
        try:
            original_features = extract_features([original_img])
            reconstructed_features = extract_features([reconstructed_img])
            fid_value = calculate_fid(original_features, reconstructed_features)
        except Exception as e:
            print(f"Error calculating FID: {e}")
            fid_value = float('inf')
        
        return {
            'psnr': float(psnr_value),
            'ssim': float(ssim_value),
            'lpips': float(lpips_value),
            'fid': float(fid_value)
        }
    
    except Exception as e:
        print(f"Error in evaluate_images: {e}")
        return {
            'psnr': 0.0,
            'ssim': 0.0,
            'lpips': 1.0,
            'fid': float('inf')
        }

def evaluate_directory(original_dir, reconstructed_dir):
    """Evaluate metrics for all images in directories"""
    original_files = sorted(glob.glob(os.path.join(original_dir, '*.png')))
    reconstructed_files = sorted(glob.glob(os.path.join(reconstructed_dir, '*.png')))
    
    if len(original_files) != len(reconstructed_files):
        print(f"Warning: Number of files mismatch - Original: {len(original_files)}, Reconstructed: {len(reconstructed_files)}")
    
    metrics = {
        'psnr': [],
        'ssim': [],
        'lpips': [],
        'fid': []
    }
    
    for orig_path, recon_path in zip(original_files, reconstructed_files):
        print(f"Evaluating {os.path.basename(orig_path)}...")
        result = evaluate_images(orig_path, recon_path)
        for metric in metrics:
            metrics[metric].append(result[metric])
    
    # Calculate averages
    avg_metrics = {metric: np.mean(values) for metric, values in metrics.items()}
    return avg_metrics

def calculate_compression_ratio(model, original_image, level_dimensions, level_downsamples, level=3):
    """Calculate and print compression ratio information.
    
    Args:
        model (nn.Module): The trained model
        original_image (numpy.ndarray): Original image array
        level_dimensions (list): List of level dimensions from openslide
        level_downsamples (list): List of level downsamples from openslide
        level (int): SVS pyramid level used for training
        
    Returns:
        float: Compression ratio
    """
    # Calculate original image size in bytes (RGB, 8 bits per channel)
    original_size_bytes = level_dimensions[0][0] * level_dimensions[0][1] * 3
    
    # Calculate model size in bytes (4 bytes per parameter for float32)
    model_size_bytes = sum(p.numel() * 4 for p in model.parameters())
    
    # Calculate compression ratio
    compression_ratio = original_size_bytes / model_size_bytes
    
    # Print information
    print("\nCompression Information:")
    print(f"Original image size (level 0): {original_size_bytes / (1024*1024):.2f} MB")
    print(f"Training image size (level {level}): {level_dimensions[level][0] * level_dimensions[level][1] * 3 / (1024*1024):.2f} MB")
    print(f"Model size: {model_size_bytes / (1024*1024):.2f} MB")
    print(f"Compression ratio: {compression_ratio:.2f}:1")
    
    return compression_ratio

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate image quality metrics')
    parser.add_argument('--original_path', type=str, required=True, help='Path to original image or directory')
    parser.add_argument('--reconstructed_path', type=str, required=True, help='Path to reconstructed image or directory')
    args = parser.parse_args()
    
    if os.path.isdir(args.original_path) and os.path.isdir(args.reconstructed_path):
        metrics = evaluate_directory(args.original_path, args.reconstructed_path)
    else:
        metrics = evaluate_images(args.original_path, args.reconstructed_path)
    
    print("\nEvaluation Results:")
    print(f"PSNR: {metrics['psnr']:.2f} dB")
    print(f"SSIM: {metrics['ssim']:.4f}")
    print(f"LPIPS: {metrics['lpips']:.4f}")
    print(f"FID: {metrics['fid']:.2f}") 