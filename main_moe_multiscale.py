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
    --batch_size N      Batch size for training (default: 131072)
    --num_samples N     Number of samples for adaptive sampling (default: 200000)
    --level N           SVS pyramid level to use (default: 3)

Examples:
    1. Train on cropped image:
       python main_moe_multiscale.py --use_crop --crop_size 512
    
    2. Train on full image:
       python main_moe_multiscale.py
    
    3. Custom training parameters:
       python main_moe_multiscale.py --use_crop --crop_size 256 --num_epochs 10000 --batch_size 32768 --num_samples 400000

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

import numpy as np
import openslide
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from skimage.io import imsave
import argparse
import lpips
import matplotlib.pyplot as plt
import seaborn as sns
from moe import MoEMultiscaleINR
from utils.evaluate_metrics import evaluate_images, calculate_compression_ratio
from utils.data_utils import load_svs_image, compute_complexity_map, create_dataset, adaptive_sampling, SlideDataset
from utils.metrics import  evaluate_model, visualize_expert_regions
from matplotlib.colors import LinearSegmentedColormap

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
    parser.add_argument('--batch_size', type=int, default=65536, help='Batch size for training')
    parser.add_argument('--num_samples', type=int, default=200000, help='Number of samples for adaptive sampling')
    parser.add_argument('--level', type=int, default=3, help='SVS pyramid level to use')
    args = parser.parse_args()

    # Check for GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Enable cuDNN benchmarking for faster training
    torch.backends.cudnn.benchmark = True
    
    # Load SVS file and get image information
    slide = openslide.OpenSlide(args.svs_path)
    level_dimensions = slide.level_dimensions
    level_downsamples = slide.level_downsamples
    
    print("\nSVS Image Information:")
    print(f"Level dimensions: {level_dimensions}")
    print(f"Level downsamples: {level_downsamples}")
    
    # Load and process the image
    whole_image = np.array(slide.read_region((0, 0), args.level, level_dimensions[args.level]))[:, :, :3] / 255.0
    slide.close()
    
    if args.use_crop:
        image = crop_center(whole_image, crop_size=args.crop_size)
        print(f"Using cropped image of size {args.crop_size}x{args.crop_size}")
    else:
        image = whole_image
        print(f"Using level {args.level} image of size {image.shape[0]}x{image.shape[1]}")
    
    # Create MoE Multiscale INR model
    model = MoEMultiscaleINR(input_dim=2, output_dim=3, num_experts=6, num_scales=3).to(device)
    
    # Calculate and print compression ratio
    calculate_compression_ratio(model, image, level_dimensions, level_downsamples, args.level)
    
    # Save the original image
    imsave("original_image.png", (image * 255).astype(np.uint8))

    # Create dataset with complexity information
    coordinates, pixels, complexity = create_dataset(image)
    
    # Normalize pixel values to [-1, 1]
    pixels = 2 * pixels - 1
    
    # Use adaptive sampling based on complexity
    sampled_coords, sampled_pixels, sampled_complexity = adaptive_sampling(
        coordinates, pixels, complexity, num_samples=args.num_samples
    )
    
    # # 可视化采样后的专家区域
    # print("\n生成采样后的专家区域可视化...")
    # visualize_expert_regions(sampled_coords, sampled_complexity, image, save_path="sampled_expert_regions.png", threshold=0.01)
    
    # Create dataset and dataloader
    dataset = SlideDataset(sampled_coords, sampled_pixels, sampled_complexity)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=8,
        pin_memory=True
    )
    
    # Setup loss and optimizer with increased learning rate
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-4, betas=(0.9, 0.999), weight_decay=1e-6)
    
    # Add learning rate scheduler with improved settings
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.7, patience=100, verbose=True
    )
    
    # Also add warmup and cosine annealing for better convergence
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs, eta_min=1e-5
    )
    
    # Training loop
    best_metrics = {
        'psnr': float('-inf'),
        'ssim': float('-inf'),
        'lpips': float('inf'),
        'fid': float('inf')
    }

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
            loss, simple_loss, complex_loss, output, expert_weights = model.router.calculate_region_losses(
                complexity_batch, expert_weights, pixel_batch, model.experts, encoded_coords
            )
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Update expert performance
            with torch.no_grad():
                # 传递损失值作为参数
                model.router.update_expert_performance(expert_weights, complexity_batch, loss)
            
            epoch_loss += loss.item()
        
        # Apply cosine annealing scheduler after each epoch
        cosine_scheduler.step()
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{args.num_epochs}, Loss: {epoch_loss / len(dataloader)}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Evaluate and save reconstructed image
        if (epoch + 1) % 100 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                metrics, reconstructed_image = evaluate_model(
                    model, dataloader, device, coordinates, image
                )
                print(f"\nEpoch {epoch + 1} Metrics:")
                print(f"PSNR: {metrics['psnr']:.2f} dB")
                print(f"SSIM: {metrics['ssim']:.4f}")
                print(f"LPIPS: {metrics['lpips']:.4f}")
                print(f"FID: {metrics['fid']:.2f}")
                
                # Update learning rate based on overall metrics
                combined_metric = metrics['psnr'] + 100 * metrics['ssim'] - 1000 * metrics['lpips']
                scheduler.step(combined_metric)
                
                # Save best model
                current_combined = metrics['psnr'] + 100 * metrics['ssim'] - 1000 * metrics['lpips']
                best_combined = best_metrics['psnr'] + 100 * best_metrics['ssim'] - 1000 * best_metrics['lpips']
                
                if current_combined > best_combined:
                    best_metrics = metrics.copy()
                    checkpoint = {
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'loss': epoch_loss / len(dataloader),
                        'metrics': metrics
                    }
                    torch.save(checkpoint, "best_model.pth")
                    print(f"New best model saved - PSNR: {metrics['psnr']:.2f}")
                
                # Save the reconstructed image
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
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': epoch_loss / len(dataloader)
            }
            torch.save(checkpoint, f"moe_multiscale_checkpoint_{epoch+1}.pth")
            print(f"Checkpoint saved for epoch {epoch + 1}")

if __name__ == "__main__":
    main() 