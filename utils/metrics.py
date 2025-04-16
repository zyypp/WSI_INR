import torch
import torch.nn.functional as F
import numpy as np
from skimage.io import imsave
import os
from utils.evaluate_metrics import evaluate_images
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap



def evaluate_model(model, dataloader, device, coordinates, image):
    """Evaluate model performance.
    
    Args:
        model (nn.Module): Model to evaluate
        dataloader (DataLoader): DataLoader for evaluation
        device (torch.device): Device to use
        coordinates (numpy.ndarray): Input coordinates
        image (numpy.ndarray): Original image
        
    Returns:
        tuple: (metrics, reconstructed_image)
    """
    model.eval()
    
    # Save temporary images for evaluation
    temp_original = "temp_original.png"
    temp_reconstructed = "temp_reconstructed.png"
    
    with torch.no_grad():
        # Process in chunks to avoid OOM
        chunk_size = 50000
        num_chunks = len(coordinates) // chunk_size + (1 if len(coordinates) % chunk_size != 0 else 0)
        
        reconstructed_pixels = []
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(coordinates))
            
            coords_chunk = torch.tensor(coordinates[start_idx:end_idx]).to(device)
            output_chunk, expert_weights_chunk = model(coords_chunk)
            reconstructed_pixels.append(output_chunk.cpu().numpy())
        
        reconstructed_pixels = np.vstack(reconstructed_pixels)
        reconstructed_image = reconstructed_pixels.reshape(image.shape)
        
        # Convert back to [0, 1] range
        reconstructed_image = (reconstructed_image + 1) / 2.0
        
        # Save images for evaluation
        imsave(temp_original, (image * 255).astype(np.uint8))
        imsave(temp_reconstructed, (reconstructed_image * 255).astype(np.uint8))
        
        # Evaluate using the metrics module
        metrics = evaluate_images(temp_original, temp_reconstructed)
        
        # Clean up temporary files
        os.remove(temp_original)
        os.remove(temp_reconstructed)
        
        return metrics, reconstructed_image

def visualize_expert_regions(coordinates, complexity, image, save_path="expert_regions.png", threshold=0.01):
    """可视化图像中由简单专家和复杂专家处理的区域。
    
    Args:
        coordinates (numpy.ndarray): 坐标数组，形状为 (N, 2)，值范围为 [0, 1]
        complexity (numpy.ndarray): 复杂度值，形状为 (N,)
        image (numpy.ndarray): 原始图像，形状为 (H, W, C)，值范围为 [0, 1]
        save_path (str): 保存可视化结果的路径
        threshold (float): 复杂度阈值，默认为0.01
        
    Returns:
        numpy.ndarray: 可视化的图像，简单区域为蓝色，复杂区域为红色
    """
    h, w = image.shape[:2]
    
    # 1. 创建掩码图像（而非热力图）- 这样可以清晰区分专家区域
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # 将坐标转换为图像索引
    y_indices = np.clip((coordinates[:, 1] * h).astype(int), 0, h-1)
    x_indices = np.clip((coordinates[:, 0] * w).astype(int), 0, w-1)
    
    # 简单区域掩码和复杂区域掩码
    simple_mask = complexity <= threshold
    complex_mask = complexity > threshold
    
    # 收集统计信息
    simple_count = np.sum(simple_mask)
    complex_count = np.sum(complex_mask)
    total_count = len(complexity)
    
    # 设置掩码值：1=简单区域，2=复杂区域
    mask[y_indices[simple_mask], x_indices[simple_mask]] = 1
    mask[y_indices[complex_mask], x_indices[complex_mask]] = 2
    
    # 2. 创建彩色覆盖层
    overlay = np.zeros((h, w, 4), dtype=np.float32)  # RGBA格式，带Alpha通道
    
    # 简单区域 - 蓝色 [0, 0, 1, 0.6]
    overlay[mask == 1] = [0, 0, 1, 0.6]
    
    # 复杂区域 - 红色 [1, 0, 0, 0.6]
    overlay[mask == 2] = [1, 0, 0, 0.6]
    
    # 3. 使用matplotlib进行正确的alpha混合（避免直接相加导致的变暗问题）
    plt.figure(figsize=(12, 10))
    plt.imshow(image)  # 首先显示原始图像
    plt.imshow(overlay[..., :3], alpha=overlay[..., 3])  # 然后用alpha通道叠加颜色
    plt.axis('off')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. 使用matplotlib创建更详细的可视化
    plt.figure(figsize=(16, 8))
    
    # 左图: 原始图像
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("原始图像")
    plt.axis('off')
    
    # 中图: 专家区域可视化 - 使用更清晰的方式
    plt.subplot(1, 3, 2)
    plt.imshow(image)  # 显示原始图像
    
    # 根据掩码创建蓝色和红色的散点标记
    y_simple = y_indices[simple_mask]
    x_simple = x_indices[simple_mask]
    y_complex = y_indices[complex_mask]
    x_complex = x_indices[complex_mask]
    
    # 如果样本太多，进行随机下采样以提高可视化清晰度
    max_points = 5000  # 最大可视化点数
    if len(y_simple) > max_points:
        idx = np.random.choice(len(y_simple), max_points, replace=False)
        y_simple = y_simple[idx]
        x_simple = x_simple[idx]
    
    if len(y_complex) > max_points:
        idx = np.random.choice(len(y_complex), max_points, replace=False)
        y_complex = y_complex[idx]
        x_complex = x_complex[idx]
    
    # 绘制散点
    plt.scatter(x_simple, y_simple, s=1, c='blue', alpha=0.5, label=f'简单区域 ({simple_count/total_count:.1%})')
    plt.scatter(x_complex, y_complex, s=1, c='red', alpha=0.5, label=f'复杂区域 ({complex_count/total_count:.1%})')
    
    plt.title("专家区域可视化")
    plt.legend(loc='upper right')
    plt.axis('off')
    
    # 右图: 复杂度热力图
    plt.subplot(1, 3, 3)
    complexity_map = np.zeros((h, w))
    for i in range(len(coordinates)):
        y, x = y_indices[i], x_indices[i]
        complexity_map[y, x] = complexity[i]
    
    # 使用插值使热力图更平滑
    plt.imshow(complexity_map, cmap='viridis', interpolation='bilinear')
    plt.colorbar(label='复杂度')
    plt.title("复杂度热力图")
    plt.axis('off')
    
    # 添加总标题
    plt.suptitle(f"专家处理区域可视化 (阈值: {threshold})", fontsize=16)
    
    # 保存详细可视化
    plt.tight_layout()
    detailed_path = save_path.replace('.png', '_detailed.png')
    plt.savefig(detailed_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"专家区域可视化已保存到: {save_path}")
    print(f"详细可视化已保存到: {detailed_path}")
    
    # 创建用于返回的混合图像
    output_image = np.copy(image) * 255
    output_image = output_image.astype(np.uint8)
    
    # 添加半透明的颜色覆盖
    simple_areas = mask == 1
    complex_areas = mask == 2
    
    # 将简单区域标记为蓝色，复杂区域标记为红色
    if np.any(simple_areas):
        output_image[simple_areas, 2] = np.minimum(255, output_image[simple_areas, 2] + 100)  # 增加蓝色
    if np.any(complex_areas):
        output_image[complex_areas, 0] = np.minimum(255, output_image[complex_areas, 0] + 100)  # 增加红色
    
    return output_image 