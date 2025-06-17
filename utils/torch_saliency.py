import torch
import torch.nn.functional as F

def frequency_tuned_saliency(img):
    # img: (B, 3, H, W)
    mean_rgb = img.mean(dim=[2, 3], keepdim=True)
    saliency = ((img - mean_rgb) ** 2).sum(dim=1, keepdim=True)
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    return saliency

def spectral_residual_saliency(img):
    # img: (B, 3, H, W)
    gray = img.mean(dim=1, keepdim=True)
    fft = torch.fft.rfftn(gray, dim=[2, 3])
    amplitude = torch.abs(fft)
    log_amplitude = torch.log(amplitude + 1e-8)
    avg_log_amp = F.avg_pool2d(log_amplitude, 3, stride=1, padding=1)
    spectral_residual = log_amplitude - avg_log_amp
    fft_recon = torch.exp(spectral_residual) * torch.exp(1j * torch.angle(fft))
    recon = torch.fft.irfftn(fft_recon, s=gray.shape[-2:], dim=[2, 3])
    saliency = recon.abs()
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    return saliency

def edge_based_saliency(img):
    # img: (B, 3, H, W)
    gray = img.mean(dim=1, keepdim=True)
    sobel_x = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=img.dtype, device=img.device).view(1,1,3,3)
    sobel_y = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=img.dtype, device=img.device).view(1,1,3,3)
    edge_x = F.conv2d(gray, sobel_x, padding=1)
    edge_y = F.conv2d(gray, sobel_y, padding=1)
    edge = torch.sqrt(edge_x**2 + edge_y**2)
    edge = (edge - edge.min()) / (edge.max() - edge.min() + 1e-8)
    return edge

def contrast_based_saliency(img):
    # img: (B, 3, H, W)
    gray = img.mean(dim=1, keepdim=True)
    mean = F.avg_pool2d(gray, 11, stride=1, padding=5)
    contrast = (gray - mean).abs()
    contrast = (contrast - contrast.min()) / (contrast.max() - contrast.min() + 1e-8)
    return contrast

def color_saliency(img):
    # img: (B, 3, H, W)
    r, g, b = img[:,0:1], img[:,1:2], img[:,2:3]
    rg = (r - g).abs()
    yb = ((r + g)/2 - b).abs()
    saliency = rg + yb
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    return saliency

def multi_scale_grayscale_contrast(img):
    # img: (B, 3, H, W)
    gray = img.mean(dim=1, keepdim=True)
    contrast = torch.zeros_like(gray)
    for k in [3, 5, 9]:
        mean = F.avg_pool2d(gray, k, stride=1, padding=k//2)
        contrast += (gray - mean).abs()
    contrast = (contrast - contrast.min()) / (contrast.max() - contrast.min() + 1e-8)
    return contrast

def boolean_map_saliency(img):
    # img: (B, 3, H, W)
    gray = img.mean(dim=1, keepdim=True)
    th = gray.mean(dim=[2,3], keepdim=True)
    saliency = (gray > th).float()
    return saliency

def itti_koch_saliency(img):
    # img: (B, 3, H, W)
    gray = img.mean(dim=1, keepdim=True)
    g1 = F.avg_pool2d(gray, 3, stride=1, padding=1)
    g2 = F.avg_pool2d(gray, 7, stride=1, padding=3)
    g3 = F.avg_pool2d(gray, 15, stride=1, padding=7)
    saliency = (gray - g1).abs() + (gray - g2).abs() + (gray - g3).abs()
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    return saliency

def color_contrast_saliency(img):
    # img: (B, 3, H, W)
    mean_rgb = img.mean(dim=[2, 3], keepdim=True)
    saliency = ((img - mean_rgb) ** 2).sum(dim=1, keepdim=True)
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    return saliency