import numpy as np
import torch
import random
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms
import torchvision
import colorspacious as cs
import cv2
from PIL import Image, ImageFilter
from utils.dataloader import datainfo

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def mixup_data(x, y, args):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if args.alpha > 0:
        lam = np.random.beta(args.alpha, args.alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    
    index = torch.randperm(batch_size).cuda()

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def Intensity_Mixup(x, y, args):
    if args.alpha > 0:
        lam = np.random.beta(args.alpha, args.alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    
    index = torch.randperm(batch_size).cuda()

    temp_x = lam * x + (1 - lam) * x[index, :]

    x_mag = torch.abs(torch.fft.rfftn(x, dim=[2, 3]))
    x_randperm_mag = torch.abs(torch.fft.rfftn(x[index, :], dim=[2, 3]))
    temp_x_mag = torch.abs(torch.fft.rfftn(temp_x, dim=[2, 3]))
    
    distance_temp = torch.norm(x_mag - temp_x_mag, p=2)
    distance_randperm = torch.norm(x_randperm_mag - temp_x_mag, p=2)

    weight_temp = 1.0 / distance_temp
    weight_randperm = 1.0 / distance_randperm

    total_weight = weight_temp + weight_randperm
    weight_temp_normalized = weight_temp / total_weight
    weight_randperm_normalized = weight_randperm / total_weight

    mixed_x = weight_temp_normalized * x + weight_randperm_normalized * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, weight_temp_normalized

def spectral_residual(image):
    x_mag = torch.abs(torch.fft.rfftn(image, dim=[2, 3]))
    log_x_mag = torch.log(x_mag + 1e-8)

    smooth_x_mag = torch.fft.irfftn(torch.fft.rfftn(log_x_mag, dim=[2, 3]).real, s=(log_x_mag.shape[2], log_x_mag.shape[3]), dim=[2, 3]).real

    spectral_residual_x_mag = log_x_mag - smooth_x_mag
    
    return x_mag * torch.exp(spectral_residual_x_mag)


def frequency_tuned_saliency(img):
    # Ensure the image tensor is on the GPU
    device = img.device

    # Convert from RGB to XYZ color space
    img = img / 255.0
    mask = (img > 0.04045)
    img = torch.where(mask, torch.pow((img + 0.055) / 1.055, 2.4), img / 12.92)
    img = img * 100.0

    # RGB to XYZ conversion matrix
    rgb_to_xyz_matrix = torch.tensor([[0.4124564, 0.3575761, 0.1804375],
                                      [0.2126729, 0.7151522, 0.0721750],
                                      [0.0193339, 0.1191920, 0.9503041]], device=device)
    xyz_img = torch.tensordot(img.permute(0, 2, 3, 1), rgb_to_xyz_matrix, dims=([-1], [-1]))

    # Convert from XYZ to LAB color space
    epsilon = 0.008856
    kappa = 903.3
    ref_white = torch.tensor([95.047, 100.000, 108.883], device=device)
    xyz_img = xyz_img / ref_white

    mask = (xyz_img > epsilon)
    xyz_img = torch.where(mask, torch.pow(xyz_img, 1/3), (kappa * xyz_img + 16) / 116)

    L = (116 * xyz_img[..., 1]) - 16
    a = 500 * (xyz_img[..., 0] - xyz_img[..., 1])
    B = 200 * (xyz_img[..., 1] - xyz_img[..., 2])
    lab_img = torch.stack([L, a, B], dim=-1)

    # Compute the mean color in LAB space
    mean_lab = lab_img.mean(dim=(1, 2), keepdim=True)

    # Compute the Euclidean distance between each pixel and the mean color
    saliency_map = torch.sqrt(((lab_img - mean_lab) ** 2).sum(dim=-1))

    # Normalize the saliency map
    min_val = saliency_map.view(saliency_map.size(0), -1).min(dim=1)[0].view(saliency_map.size(0), 1, 1)
    max_val = saliency_map.view(saliency_map.size(0), -1).max(dim=1)[0].view(saliency_map.size(0), 1, 1)
    saliency_map = (saliency_map - min_val) / (max_val - min_val + 1e-8)
    
    # Apply Gaussian blur using PyTorch's convolution function
    gaussian_kernel = torch.tensor([[1/16, 1/8, 1/16], 
                                    [1/8,  1/4, 1/8], 
                                    [1/16, 1/8, 1/16]], device=device).unsqueeze(0).unsqueeze(0)
    gaussian_kernel = gaussian_kernel.expand(saliency_map.size(1), 1, 3, 3)  # for each channel

    saliency_map = F.conv2d(saliency_map.unsqueeze(1), gaussian_kernel, padding=1, groups=1)
    saliency_map = saliency_map.squeeze(1)

    return saliency_map


def color_contrast_saliency(img):
    # Ensure img is on the GPU
    img = img.cuda()

    # Convert the image tensor to the CIELAB color space
    def rgb_to_lab(image):
        # Convert to numpy
        img_np = image.cpu().numpy().transpose(0, 2, 3, 1)  # Convert from (batch_size, channels, height, width) to (batch_size, height, width, channels)
        img_np = (img_np * 255).astype(np.uint8)
        lab_images = [cv2.cvtColor(img, cv2.COLOR_RGB2LAB) for img in img_np]
        return torch.tensor(np.stack(lab_images), dtype=torch.float32).cuda() / 255.0
    
    lab = rgb_to_lab(img)

    batch_size, height, width, _ = lab.shape
    saliency_maps = torch.zeros(batch_size, height, width, device='cuda', dtype=torch.float32)
    
    # Compute histograms for each image in the batch using advanced indexing
    def compute_histograms(lab_image):
        # Flatten the LAB channels
        l = lab_image[:, :, 0].long()
        a = lab_image[:, :, 1].long()
        b = lab_image[:, :, 2].long()
        
        hist_l = torch.zeros(256, device='cuda')
        hist_a = torch.zeros(256, device='cuda')
        hist_b = torch.zeros(256, device='cuda')
        
        # Compute histogram
        hist_l.scatter_add_(0, l.view(-1), torch.ones_like(l.view(-1), dtype=torch.float32))
        hist_a.scatter_add_(0, a.view(-1), torch.ones_like(a.view(-1), dtype=torch.float32))
        hist_b.scatter_add_(0, b.view(-1), torch.ones_like(b.view(-1), dtype=torch.float32))
        
        # Normalize histograms
        hist_l /= hist_l.sum()
        hist_a /= hist_a.sum()
        hist_b /= hist_b.sum()
        
        return hist_l, hist_a, hist_b

    # Compute saliency maps for the entire batch
    for i in range(batch_size):
        hist_l, hist_a, hist_b = compute_histograms(lab[i])
        
        l = lab[i][:, :, 0].long()
        a = lab[i][:, :, 1].long()
        b = lab[i][:, :, 2].long()
        
        saliency = hist_l[l] + hist_a[a] + hist_b[b]
        saliency_maps[i] = saliency
    
    # Normalize the saliency map
    saliency_maps -= saliency_maps.min()
    saliency_maps /= saliency_maps.max()
    saliency_maps *= 255
    saliency_maps = saliency_maps.byte()
    
    return saliency_maps


def gabor_kernel(frequency, orientation, size, device):
    x, y = np.meshgrid(np.linspace(-size/2, size/2, size), np.linspace(-size/2, size/2, size))
    x_theta = x * np.cos(orientation) + y * np.sin(orientation)
    y_theta = -x * np.sin(orientation) + y * np.cos(orientation)
    gb_real = np.exp(-0.5 * (x_theta**2 + y_theta**2)) * np.cos(2 * np.pi * frequency * x_theta)
    gb_imag = np.exp(-0.5 * (x_theta**2 + y_theta**2)) * np.sin(2 * np.pi * frequency * x_theta)
    return torch.tensor(gb_real, dtype=torch.float32, device=device), torch.tensor(gb_imag, dtype=torch.float32, device=device)

def gabor_filter(img, filters, img_size):
    num_filters = len(filters)
    num_channels = img.size(1)
    results = []
    for real, imag in filters:
        # Extend filter to multiple channels
        real = real.unsqueeze(0).repeat(num_channels, 1, 1, 1)
        imag = imag.unsqueeze(0).repeat(num_channels, 1, 1, 1)
        
        real = F.conv2d(img, real, padding=img_size//2, groups=num_channels)
        imag = F.conv2d(img, imag, padding=img_size//2, groups=num_channels)
        magnitude = torch.sqrt(real**2 + imag**2)
        results.append(magnitude)
    
    return torch.stack(results, dim=1)

def itti_koch_saliency(img):
    device = img.device  # Get the device of the input image
    img_size = img.shape[-1]  # Assuming square images

    # Define Gabor filter parameters
    frequencies = [0.1, 0.2, 0.3]
    orientations = [0, np.pi / 4, np.pi / 2]

    # Create Gabor filters
    filters = []
    for freq in frequencies:
        for ori in orientations:
            real, imag = gabor_kernel(freq, ori, img_size, device)
            filters.append((real, imag))

    # Apply Gabor filters
    gabor_features = gabor_filter(img, filters, img_size)

    # Combine features to get the saliency map
    saliency_map = torch.sum(gabor_features, dim=1)
    saliency_map = saliency_map - torch.min(saliency_map)
    saliency_map = saliency_map / torch.max(saliency_map)
    
    return saliency_map


def spectral_residual_saliency(img):
    device = img.device  # Get the device of the input image
    batch_size, channels, height, width = img.shape
    
    # Convert to grayscale using a simple average method (weights could be adjusted)
    if channels == 3:
        grayscale_weights = torch.tensor([0.2989, 0.5870, 0.1140], device=device).view(1, 3, 1, 1)
        gray = F.conv2d(img, grayscale_weights, stride=1, padding=0)
    else:
        gray = img  # Assuming already grayscale if channels != 3
    
    # Compute the Fourier transform of the image
    fft = torch.fft.fftn(gray, dim=(-2, -1))
    amplitude = torch.abs(fft)
    phase = torch.angle(fft)
    
    # Compute the log amplitude and the spectral residual
    log_amplitude = torch.log(amplitude + 1e-10)
    kernel = torch.ones(1, 1, 3, 3, device=device) / 9.0
    smooth_log_amplitude = F.conv2d(log_amplitude, kernel, stride=1, padding=1)
    spectral_residual = log_amplitude - smooth_log_amplitude
    
    # Compute the inverse Fourier transform
    amplitude = torch.exp(spectral_residual)
    saliency_map = torch.abs(torch.fft.ifftn(amplitude * torch.exp(1j * phase), dim=(-2, -1))) ** 2
    
    # Normalize the saliency map
    saliency_map = saliency_map - saliency_map.min()
    saliency_map = saliency_map / saliency_map.max()
    
    return saliency_map



def boolean_map_saliency(img, n_thresholds=8):
    # Convert the image tensor to numpy array
    img_np = img.cpu().numpy().transpose(0, 2, 3, 1)  # Convert from (batch_size, channels, height, width) to (batch_size, height, width, channels)
    
    # Ensure img_np is in float32
    img_np = img_np.astype(np.float32)
    
    # Initialize the list for saliency maps
    saliency_maps = []
    
    # Process each image in the batch
    for i in range(img_np.shape[0]):
        # Convert the image to grayscale
        gray = cv2.cvtColor(img_np[i], cv2.COLOR_RGB2GRAY)
        
        # Scale the grayscale image to [0, 255] range
        gray = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
        
        # Initialize the saliency map
        saliency_map = np.zeros_like(gray, dtype=np.float32)
        
        # Compute boolean maps
        for j in range(n_thresholds):
            threshold = 255 * (j + 1) / (n_thresholds + 1)
            _, binary_map = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
            binary_map = np.uint8(binary_map)  # Ensure binary_map is of type np.uint8
            
            # Detect regions
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_map, connectivity=8)
            
            # Compute region saliency
            region_saliency = np.zeros_like(binary_map, dtype=np.float32)
            for label in range(1, num_labels):
                region_saliency[labels == label] = stats[label, cv2.CC_STAT_AREA]
            
            saliency_map += region_saliency
        
        # Normalize the saliency map
        saliency_map = cv2.normalize(saliency_map, None, 0, 255, cv2.NORM_MINMAX)
        
        # Convert to uint8
        saliency_map = np.uint8(saliency_map)
        
        # Append to saliency maps list
        saliency_maps.append(saliency_map)
    
    # Convert the list of numpy arrays to a PyTorch tensor
    saliency_maps = torch.from_numpy(np.stack(saliency_maps)).float()
    
    return saliency_maps/255.
    

def multi_scale_grayscale_contrast(img, scales=[3, 5, 7]):
    # Convert the image tensor to numpy array
    img_np = img.cpu().numpy().transpose(0, 2, 3, 1)  # Convert from (batch_size, channels, height, width) to (batch_size, height, width, channels)
    
    # Ensure img_np is in float32
    img_np = img_np.astype(np.float32)
    
    # Initialize the list for saliency maps
    saliency_maps = []
    
    # Process each image in the batch
    for i in range(img_np.shape[0]):
        # Convert the image to grayscale
        gray = cv2.cvtColor(img_np[i], cv2.COLOR_RGB2GRAY)
        
        # Scale the grayscale image to [0, 255] range
        gray = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
        
        # Initialize the saliency map
        saliency_map = np.zeros_like(gray, dtype=np.float32)
        
        # Compute multi-scale contrast
        for scale in scales:
            blurred = cv2.GaussianBlur(gray, (scale, scale), 0)
            contrast = np.abs(gray - blurred)
            saliency_map += contrast
        
        # Normalize the saliency map
        saliency_map = cv2.normalize(saliency_map, None, 0, 255, cv2.NORM_MINMAX)
        
        # Convert to uint8
        saliency_map = np.uint8(saliency_map)
        
        # Append to saliency maps list
        saliency_maps.append(saliency_map)
    
    # Convert the list of numpy arrays to a PyTorch tensor
    saliency_maps = torch.from_numpy(np.stack(saliency_maps)).float()
    
    return saliency_maps/255.

def color_saliency(img):
    # Convert the image tensor to numpy array
    img_np = img.cpu().numpy().transpose(0, 2, 3, 1)  # Convert from (batch_size, channels, height, width) to (batch_size, height, width, channels)
    
    # Ensure img_np is in float32 and convert from normalized values to standard RGB range [0, 255]
    img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
    
    # Initialize the list for saliency maps
    saliency_maps = []
    
    # Process each image in the batch
    for i in range(img_np.shape[0]):
        # Convert the image to CIELAB color space
        lab = cv2.cvtColor(img_np[i], cv2.COLOR_RGB2LAB)
        
        # Compute the mean color of the image
        mean_lab = np.mean(lab, axis=(0, 1))
        
        # Compute the Euclidean distance between each pixel and the mean color
        saliency_map = np.sqrt(np.sum((lab - mean_lab) ** 2, axis=2))
        
        # Normalize the saliency map
        saliency_map = cv2.normalize(saliency_map, None, 0, 255, cv2.NORM_MINMAX)
        
        # Convert to uint8
        saliency_map = np.uint8(saliency_map)
        
        # Append to saliency maps list
        saliency_maps.append(saliency_map)
    
    # Convert the list of numpy arrays to a PyTorch tensor
    saliency_maps = torch.from_numpy(np.stack(saliency_maps)).float()
    
    return saliency_maps/255.


def contrast_based_saliency(img):
    # Convert the image tensor to numpy array
    img_np = img.cpu().numpy().transpose(0, 2, 3, 1)  # Convert from (batch_size, channels, height, width) to (batch_size, height, width, channels)
    
    # Ensure img_np is in float32 and convert from normalized values to standard RGB range [0, 255]
    img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
    
    # Initialize the list for saliency maps
    saliency_maps = []
    
    # Process each image in the batch
    for i in range(img_np.shape[0]):
        # Convert the image to grayscale
        gray = cv2.cvtColor(img_np[i], cv2.COLOR_RGB2GRAY)
        
        # Compute the local mean
        local_mean = cv2.blur(gray, (9, 9))
        
        # Compute the contrast
        contrast = np.abs(gray - local_mean)
        
        # Normalize the contrast map
        contrast = cv2.normalize(contrast, None, 0, 255, cv2.NORM_MINMAX)
        
        # Convert to uint8
        contrast = np.uint8(contrast)
        
        # Append to saliency maps list
        saliency_maps.append(contrast)
    
    # Convert the list of numpy arrays to a PyTorch tensor
    saliency_maps = torch.from_numpy(np.stack(saliency_maps)).float()
    
    return saliency_maps/255.

def edge_based_saliency(img):
    # Convert the image tensor to numpy array
    img_np = img.cpu().numpy().transpose(0, 2, 3, 1)  # Convert from (batch_size, channels, height, width) to (batch_size, height, width, channels)
    
    # Ensure img_np is in float32 and convert from normalized values to standard RGB range [0, 255]
    img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
    
    # Initialize the list for saliency maps
    saliency_maps = []
    
    # Process each image in the batch
    for i in range(img_np.shape[0]):
        # Convert the image to grayscale
        gray = cv2.cvtColor(img_np[i], cv2.COLOR_RGB2GRAY)
        
        # Compute the edges using Canny edge detector
        edges = cv2.Canny(gray, 100, 200)
        
        # Normalize the edges
        saliency_map = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX)
        
        # Convert to uint8
        saliency_map = np.uint8(saliency_map)
        
        # Append to saliency maps list
        saliency_maps.append(saliency_map)
    
    # Convert the list of numpy arrays to a PyTorch tensor
    saliency_maps = torch.from_numpy(np.stack(saliency_maps)).float()
    
    return saliency_maps/255.

def Cab_Mixup(x, y, args):
    if args.alpha > 0:
        lam = np.random.beta(args.alpha, args.alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    
    index = torch.randperm(batch_size).cuda()

    mixed_x = lam * x + (1 - lam) * x[index, :]

    if args.saliency == 'spectral_residual':
        saliency_function = spectral_residual
    elif args.saliency == 'frequency_tuned':
        saliency_function = frequency_tuned_saliency
    elif args.saliency == 'color_contrast':
        saliency_function = color_contrast_saliency
    elif args.saliency == 'itti_koch':
        saliency_function = itti_koch_saliency
    elif args.saliency == 'multi_scale_grayscale_contrast':
        saliency_function = multi_scale_grayscale_contrast
    elif args.saliency == 'color':
        saliency_function = color_saliency
    elif args.saliency == 'contrast_based':
        saliency_function = contrast_based_saliency
    elif args.saliency == 'edge_based':
        saliency_function = edge_based_saliency

    residual_image1 = saliency_function(x)
    residual_image2 = saliency_function(x[index, :])
    residual_mixed_image = saliency_function(mixed_x)

    diff1 = torch.abs(residual_image1-residual_mixed_image)

    diff2 = torch.abs(residual_image2-residual_mixed_image)

    cab_lam = (torch.sum(diff2.flatten(1),dim=(-1))/(torch.sum(diff1.flatten(1),dim=(-1)) + torch.sum(diff2.flatten(1),dim=(-1)))).cuda()

    cab_lam = torch.nan_to_num(cab_lam, nan=1e-7)

    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam, cab_lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def calibrate_cab_lam(cab_lam, lam,marginal=0.05):

    diff = lam - cab_lam
    
    abs_diff = torch.abs(diff)
    
    mask = abs_diff >= marginal
    
    calibrated_cab_lam = torch.where(
        mask, 
        cab_lam + torch.sign(diff) * marginal, 
        lam
    )
    
    return calibrated_cab_lam

def cab_mixup_criterion(criterion, pred, y_a, y_b, lam,cab_lam, args):
    return torch.mean(calibrate_cab_lam(cab_lam,lam,args.marginal*(1/datainfo(args)['n_classes']))* criterion(pred, y_a) +  (1 - calibrate_cab_lam(cab_lam,lam,args.marginal*(1/datainfo(args)['n_classes']))) *criterion(pred, y_b))
