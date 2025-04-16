import numpy as np
import openslide
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.io import imsave

'''
# Function to load and process SVS image
def load_svs_image(svs_path, level=0):
    slide = openslide.OpenSlide(svs_path)
    image = np.array(slide.read_region((0, 0), level, slide.level_dimensions[level]))[:, :, :3] / 255.0
    slide.close()
    return image
'''

# Main function
if __name__ == "__main__":

    # Load and process the image
    svs_path = "./data/test.svs"  # Replace with your file path

    slide = openslide.OpenSlide(svs_path)

    for i in range(0, 4): 
        level = int(3-i)
        print('convert '.format(level))
        imsave("original_image_L{}.png".format(level), (np.array(slide.read_region((0, 0), level, slide.level_dimensions[level]))[:, :, :3]).astype(np.uint8))

    slide.close()