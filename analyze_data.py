"""
Data Distribution Analysis and Visualization

This script provides functions to analyze and visualize the distribution of:
1. Image pixel values
2. Coordinate normalization
3. Normalized pixel values
4. Complexity values

Usage:
    python analyze_data.py --svs_path PATH [--use_crop] [--crop_size SIZE] [--level N]
"""

import numpy as np
import openslide
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from scipy import ndimage

def crop_center(image, crop_size=256):
    """Crop the center of the image"""
    h, w, c = image.shape
    start_x = (w - crop_size) // 2
    start_y = (h - crop_size) // 2
    cropped_image = image[start_y:start_y + crop_size, start_x:start_x + crop_size, :]
    return cropped_image.astype(np.float32)

def compute_complexity_map(image, sigma=1.0):
    """Compute complexity map from image using gradient magnitude"""
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
    """Create coordinate-pixel dataset with complexity information"""
    h, w, _ = image.shape
    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    coordinates = np.stack((x.ravel(), y.ravel()), axis=-1) / np.array([w, h])  # Normalize
    pixels = image.reshape(-1, 3)  # RGB values
    
    # Compute complexity map
    complexity_map = compute_complexity_map(image)
    complexity = complexity_map.ravel()
    
    return coordinates.astype(np.float32), pixels.astype(np.float32), complexity.astype(np.float32)

def analyze_data_distribution(image, coordinates, pixels, complexity):
    """Analyze and visualize data distribution"""
    # Create a figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Check image value range
    axes[0, 0].hist(image.ravel(), bins=50, color='blue', alpha=0.7)
    axes[0, 0].set_title('Image Value Distribution')
    axes[0, 0].set_xlabel('Pixel Value')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].axvline(x=0, color='red', linestyle='--')
    axes[0, 0].axvline(x=1, color='red', linestyle='--')
    
    # 2. Check coordinate normalization
    axes[0, 1].scatter(coordinates[:, 0], coordinates[:, 1], alpha=0.1)
    axes[0, 1].set_title('Coordinate Distribution')
    axes[0, 1].set_xlabel('X Coordinate')
    axes[0, 1].set_ylabel('Y Coordinate')
    axes[0, 1].set_xlim(0, 1)
    axes[0, 1].set_ylim(0, 1)
    
    # 3. Check pixel value normalization
    axes[1, 0].hist(pixels.ravel(), bins=50, color='green', alpha=0.7)
    axes[1, 0].set_title('Normalized Pixel Value Distribution')
    axes[1, 0].set_xlabel('Pixel Value')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].axvline(x=-1, color='red', linestyle='--')
    axes[1, 0].axvline(x=1, color='red', linestyle='--')
    
    # 4. Check complexity distribution
    axes[1, 1].hist(complexity, bins=50, color='purple', alpha=0.7)
    axes[1, 1].set_title('Complexity Distribution')
    axes[1, 1].set_xlabel('Complexity Value (Higher = More Complex)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].axvline(x=0.5, color='red', linestyle='--')  # Current threshold
    
    plt.tight_layout()
    plt.savefig('data_distribution.png')
    plt.close()
    
    # Create complexity map highlighting regions with complexity > 0.01
    plt.figure(figsize=(12, 8))
    
    # Reshape complexity back to image shape
    h, w = image.shape[:2]
    complexity_map = complexity.reshape(h, w)
    
    # Create masks for different complexity levels
    low_complexity_mask = complexity_map > 0.01
    high_complexity_mask = complexity_map > 0.01
    
    # Create a colormap for visualization
    cmap = plt.cm.viridis
    cmap.set_bad('white', 1.0)  # Set masked values to white
    
    # Plot the complexity map, masking out regions with complexity <= 0.01
    plt.imshow(np.ma.masked_where(~low_complexity_mask, complexity_map), cmap=cmap)
    plt.colorbar(label='Complexity Value')
    
    # Overlay high complexity regions (complexity > 0.01) in red
    plt.imshow(np.ma.masked_where(~high_complexity_mask, high_complexity_mask), 
               cmap='Reds', alpha=0.3)
    
    plt.title('Complexity Map\nRegions with complexity > 0.01')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    
    # Add statistics to the plot
    complexity_percentage = np.sum(low_complexity_mask) / (h * w) * 100
    plt.text(0.02, 0.02, f'Complexity > 0.01: {complexity_percentage:.2f}%',
             transform=plt.gca().transAxes, color='black', fontsize=12,
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('high_complexity_map.png')
    plt.close()
    
    # Print statistics
    print("\nData Distribution Statistics:")
    print(f"Image value range: [{image.min():.3f}, {image.max():.3f}]")
    print(f"Coordinate range - X: [{coordinates[:, 0].min():.3f}, {coordinates[:, 0].max():.3f}]")
    print(f"Coordinate range - Y: [{coordinates[:, 1].min():.3f}, {coordinates[:, 1].max():.3f}]")
    print(f"Normalized pixel range: [{pixels.min():.3f}, {pixels.max():.3f}]")
    print(f"Complexity range: [{complexity.min():.3f}, {complexity.max():.3f}]")
    
    # Analyze complexity distribution
    print("\nComplexity Distribution Analysis:")
    thresholds = [0.01, 0.2, 0.4, 0.6, 0.8]
    for i in range(len(thresholds)):
        if i == 0:
            mask = complexity < thresholds[i]
            print(f"Complexity < {thresholds[i]}: {mask.sum()/len(complexity)*100:.2f}%")
        else:
            mask = (complexity >= thresholds[i-1]) & (complexity < thresholds[i])
            print(f"{thresholds[i-1]} <= Complexity < {thresholds[i]}: {mask.sum()/len(complexity)*100:.2f}%")
    mask = complexity >= thresholds[-1]
    print(f"Complexity >= {thresholds[-1]}: {mask.sum()/len(complexity)*100:.2f}%")
    
    # Print complexity interpretation
    print("\nComplexity Value Interpretation:")
    print("0.00-0.01: Very Simple regions (e.g., uniform background)")
    print("0.01-0.10: Simple regions (e.g., smooth gradients)")
    print("0.10-0.20: Moderately Simple regions (e.g., simple textures)")
    print("0.20-0.40: Medium complexity regions (e.g., detailed textures)")
    print("0.40-0.60: Complex regions (e.g., fine details)")
    print("0.60-0.80: Very Complex regions (e.g., sharp edges, high contrast)")
    print("0.80-1.00: Extremely Complex regions (e.g., fine structures, noise)")
    
    # Print high complexity region statistics
    print(f"\nHigh Complexity Regions (complexity > 0.01):")
    print(f"Total area: {complexity_percentage:.2f}%")
    print(f"Number of regions: {np.sum(low_complexity_mask)} pixels")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Analyze data distribution')
    parser.add_argument('--svs_path', type=str, default="/data/WRI_data/test.svs", help='Path to SVS file')
    parser.add_argument('--use_crop', action='store_true', help='Use cropped image instead of full image')
    parser.add_argument('--crop_size', type=int, default=512, help='Size of cropped image')
    parser.add_argument('--level', type=int, default=3, help='SVS pyramid level to use')
    args = parser.parse_args()

    # Load SVS file
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
    
    # Create dataset with complexity information
    coordinates, pixels, complexity = create_dataset(image)
    
    # Analyze data distribution
    analyze_data_distribution(image, coordinates, pixels, complexity)

if __name__ == "__main__":
    main() 