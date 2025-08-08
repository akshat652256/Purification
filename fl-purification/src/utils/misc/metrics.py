import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# images: input tensor (batch, 3, 28, 28)
# outputs: reformed tensor from your model (batch, 3, 28, 28)
# Assume both are in [0,1] range

def compute_psnr_ssim(images, outputs):
    images_np = images.cpu().numpy()
    outputs_np = outputs.cpu().numpy()
    n = images_np.shape[0]
    psnr_scores = []
    ssim_scores = []
    for i in range(n):
        x = images_np[i].transpose(1, 2, 0)
        y = outputs_np[i].transpose(1, 2, 0)
        x = np.clip(x, 0, 1)
        y = np.clip(y, 0, 1)
        # skimage wants shape (H, W, C) and float in [0,1]
        psnr_val = psnr(x, y, data_range=1.0)
        ssim_val = ssim(x, y, channel_axis=-1, data_range=1.0)
        psnr_scores.append(psnr_val)
        ssim_scores.append(ssim_val)
    print(f"Average PSNR: {np.mean(psnr_scores):.4f}")
    print(f"Average SSIM: {np.mean(ssim_scores):.4f}")
    

def jsd(p, q, eps=1e-6):
    m = 0.5 * (p + q)
    p = p + eps
    q = q + eps
    m = m + eps
    kld_pm = (p * (p / m).log()).sum(dim=1)
    kld_qm = (q * (q / m).log()).sum(dim=1)
    return 0.5 * (kld_pm + kld_qm)