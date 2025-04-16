import numpy as np
import openslide
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.io import imsave

def crop_center(image, crop_size=256):
    h, w, c = image.shape
    start_x = (w - crop_size) // 2
    start_y = (h - crop_size) // 2

    # Crop the image (handle the color channel by slicing the last dimension)
    cropped_image = image[start_y:start_y + crop_size, start_x:start_x + crop_size, :]
    return cropped_image.astype(np.float32)

# Define INR Model
class INR(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=256, output_dim=3):
        super(INR, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

class Sine(nn.Module):
    def __init(self):
        super(Sine,self).__init__()

    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(30 * input)

class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the 
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a 
    # hyperparameter.
    
    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of 
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
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
    def __init__(self,in_features,out_features,nonlinearity='relu'):
        super(ResBlock,self).__init__()
        nls_and_inits = {'sine':Sine(),
                         'relu':nn.ReLU(inplace=True),
                         'sigmoid':nn.Sigmoid(),
                         'tanh':nn.Tanh(),
                         'selu':nn.SELU(inplace=True),
                         'softplus':nn.Softplus(),
                         'elu':nn.ELU(inplace=True)}

        self.nl = nls_and_inits[nonlinearity]
        self.net = []
        self.net.append(SineLayer(in_features,out_features))
        self.net.append(SineLayer(out_features,out_features))
        self.flag = (in_features!=out_features)

        if self.flag:
            self.transform = SineLayer(in_features,out_features)

        self.net = nn.Sequential(*self.net)
    
    def forward(self,features):
        outputs = self.net(features)
        if self.flag:
            features = self.transform(features)
        return 0.5*(outputs+features)

class PositionalEncoding(nn.Module):
    def __init__(self, num_encoding_functions=6, include_input=True, log_sampling=True, normalize=False,
                 input_dim=3, gaussian_pe=False, gaussian_variance=10):
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

    def forward(self, tensor) -> torch.Tensor:
        r"""Apply positional encoding to the input.
        Args:
            tensor (torch.Tensor): Input tensor to be positionally encoded.
            encoding_size (optional, int): Number of encoding functions used to compute
                a positional encoding (default: 6).
            include_input (optional, bool): Whether or not to include the input in the
                positional encoding (default: True).
        Returns:
        (torch.Tensor): Positional encoding of the input tensor.
        """

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

class CoordNet(nn.Module):
    def __init__(self, in_features, out_features, init_features=64,num_res=10,positional_encoding=None):
        super(CoordNet,self).__init__()
        self.positional_encoding = positional_encoding
        if positional_encoding:
            in_features = in_features + positional_encoding.num_encoding_functions * 2 * (1 + positional_encoding.include_input) 
            print(in_features)
            # positional_encoding.num_encoding_functions * 2 * (1 + positional_encoding.include_input)
        self.num_res = num_res
        self.net = []

        self.net.append(ResBlock(in_features,init_features))
        #self.net.append(nl)
        self.net.append(ResBlock(init_features,2*init_features))
        #self.net.append(nl)
        self.net.append(ResBlock(2*init_features,4*init_features))
        #self.net.append(nl)

        for i in range(self.num_res):
            self.net.append(ResBlock(4*init_features,4*init_features))

        self.net.append(ResBlock(4*init_features, out_features))

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        if self.positional_encoding:
            coords = self.positional_encoding(coords)
        output = self.net(coords)
        return output


# Dataset for coordinate-pixel pairs
class SlideDataset(Dataset):
    def __init__(self, coordinates, pixels):
        self.coordinates = coordinates
        self.pixels = pixels

    def __len__(self):
        return len(self.coordinates)

    def __getitem__(self, idx):
        return self.coordinates[idx], self.pixels[idx]

# Function to load and process SVS image
def load_svs_image(svs_path, level=0):
    slide = openslide.OpenSlide(svs_path)
    image = np.array(slide.read_region((0, 0), level, slide.level_dimensions[level]))[:, :, :3] / 255.0
    slide.close()
    return image

# Prepare coordinate-pixel dataset
def create_dataset(image):
    h, w, _ = image.shape
    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    coordinates = np.stack((x.ravel(), y.ravel()), axis=-1) / np.array([w, h])  # Normalize
    pixels = image.reshape(-1, 3)  # RGB values
    return coordinates.astype(np.float32), pixels.astype(np.float32)

# Main function
if __name__ == "__main__":
    # Check for GPU
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load and process the image
    svs_path = "./data/test.svs"  # Replace with your file path
    #whole_image = load_svs_image(svs_path,level=3)
    image = load_svs_image(svs_path,level=3)

    # sample an impage 
    #image = crop_center(whole_image, crop_size=512)

    # Save the original image
    imsave("original_image.png", (image * 255).astype(np.uint8))

    # Create dataset and dataloader
    coordinates, pixels = create_dataset(image)
    # p_min = np.min(pixels)
    # p_max = np.max(pixels)
    print(np.max(pixels), np.min(pixels))
    # pixels = 2*(pixels-p_min)/(p_max-p_min)-1 # normalize to [-1,1]
    pixels = 2*pixels-1
    dataset = SlideDataset(coordinates, pixels)
    dataloader = DataLoader(dataset, batch_size=32000, shuffle=True, num_workers=4)


    # Initialize Positional Encoding
    pos_encoding = PositionalEncoding(num_encoding_functions=6, include_input=True).to(device)

    # Initialize INR model, loss, and optimizer
    # model = INR().to(device)
    model = CoordNet(2,3,positional_encoding=pos_encoding).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5,betas=(0.9,0.999),weight_decay=1e-6)

    # Train the INR model
    epochs = 10000
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        iteration = 0
        for coord_batch, pixel_batch in dataloader:
            coord_batch, pixel_batch = coord_batch.to(device), pixel_batch.to(device)
            optimizer.zero_grad()
            output = model(coord_batch)
            loss = criterion(output, pixel_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            #if (iteration + 1) % 20 == 0:
            #    print(f"Epoch {epoch + 1}, Iteration {iteration + 1}, Loss: {loss.item():.6f}")
            iteration += 1 
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(dataloader)}")

        if (epoch + 1) % 2 == 0: 
            model.eval()
            with torch.no_grad():
                coords_tensor = torch.tensor(coordinates).to(device)
                reconstructed_pixels = model(coords_tensor).cpu().numpy()
                reconstructed_image = reconstructed_pixels.reshape(image.shape)
                psnr_value = psnr(image, reconstructed_image)
                print(f"PSNR of reconstructed image: {psnr_value:.2f}")

                reconstructed_image = (reconstructed_image+1)/2.0
                #reconstructed_image = ((reconstructed_image+1)/2.0)*(p_max-p_min)+p_min
                #print(reconstructed_image.shape, np.max(reconstructed_image), np.min(reconstructed_image))
                imsave("reconstructed_image_{:06}.png".format(epoch), (reconstructed_image * 255).astype(np.uint8))
            
            # free GPU memory 
            del coords_tensor, reconstructed_pixels 
            torch.cuda.empty_cache()

