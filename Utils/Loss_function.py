import kornia
import numpy as np
import torch
from skimage.io import imread
from torch import nn
from math import exp
import torch.nn.functional as F

from preprocessing import preprocessing

MSELoss = nn.MSELoss(reduction='mean')
Local_SSIMLoss = kornia.losses.SSIMLoss(3, reduction='mean')
Global_SSIMLoss = kornia.losses.SSIMLoss(11, reduction='mean')
Gradient_loss = nn.L1Loss(reduction='mean')
def ssim_loss (input_ir,input_vi,fused_result):
    ssim_loss=ssim(fused_result,torch.maximum(input_ir,input_vi))
    return ssim_loss
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window
def ssim(img1, img2, window_size=11, window=None, val_range=None):
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2
    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)
    ret = ssim_map.mean()
    return 1-ret

def cc(img1, img2):
    eps = torch.finfo(torch.float32).eps
    """Correlation coefficient for (N, C, H, W) image; torch.float32 [0.,1.]."""
    N, C, _, _ = img1.shape
    img1 = img1.reshape(N, C, -1)
    img2 = img2.reshape(N, C, -1)
    img1_mean = img1.mean(dim=-1, keepdim=True)
    img2_mean = img2.mean(dim=-1, keepdim=True)
    img1 = img1 - img1_mean
    img2 = img2 - img2_mean
    # Check for zero variance
    img1_var = torch.sum(img1 ** 2, dim=-1)
    img2_var = torch.sum(img2 ** 2, dim=-1)

    # Avoid division by zero
    cc = torch.zeros(N, C, device=img1.device)  # Initialize to zero
    non_zero_mask = (img1_var > 0) & (img2_var > 0)

    if non_zero_mask.any():
        cc[non_zero_mask] = (
                torch.sum(img1 * img2, dim=-1)[non_zero_mask] /
                (eps + torch.sqrt(img1_var[non_zero_mask]) * torch.sqrt(img2_var[non_zero_mask]))
        )
    cc = torch.clamp(cc, -1., 1.)
    return cc.mean()


class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)
class Fusionloss(nn.Module):
    def __init__(self):
        super(Fusionloss, self).__init__()
        self.sobelconv=Sobelxy()

    def forward(self,image_ir,image_vis,generate_img,L1_parametric,grad_parametric):
        image_y=image_vis[:,:1,:,:]
        x_in_max=torch.max(image_y,image_ir)
        loss_in=F.l1_loss(x_in_max,generate_img)
        y_grad=self.sobelconv(image_y)
        ir_grad=self.sobelconv(image_ir)
        generate_img_grad=self.sobelconv(generate_img)
        x_grad_joint=torch.max(y_grad,ir_grad)
        loss_grad=F.l1_loss(x_grad_joint,generate_img_grad)
        loss_total=L1_parametric*loss_in+grad_parametric*loss_grad
        return loss_total,loss_in,loss_grad
if __name__ == '__main__':
    ir_image = torch.tensor(imread('../ir.png').astype(np.float32)[None,None, :, :]/255.0)
    vi_image = np.transpose(imread('../vi.png').astype(np.float32), axes=(2, 0, 1)) / 255.0
    vi_image = torch.tensor(preprocessing.RGB_to_2Y(vi_image)[None,:,:,:])
    fusion_image = torch.tensor(imread('../Fusion_img.png').astype(np.float32)[None,None, :, :]/255.0)
    ir_fusion_mse_loss = MSELoss(ir_image,fusion_image)
    vi_fusion_mse_loss = MSELoss(vi_image,fusion_image)
    ir_fusion_local_ssim = Local_SSIMLoss(ir_image,fusion_image)
    vi_fusion_global_ssim = Local_SSIMLoss(vi_image,fusion_image)
    ir_fusion_gradient_loss = Gradient_loss(kornia.filters.SpatialGradient()(ir_image), kornia.filters.SpatialGradient()(fusion_image))
    vi_fusion_gradient_loss = Gradient_loss(kornia.filters.SpatialGradient()(vi_image), kornia.filters.SpatialGradient()(fusion_image))
    print('ir_fusion_mse_loss:', ir_fusion_mse_loss.item())
    print('vi_fusion_mse_loss:', vi_fusion_mse_loss.item())
    print('ir_fusion_local_ssim:', ir_fusion_local_ssim.item())
    print('vi_fusion_global_ssim:', vi_fusion_global_ssim.item())
    print('ir_fusion_gradient_loss:', ir_fusion_gradient_loss.item())
    print('vi_fusion_gradient_loss:', vi_fusion_gradient_loss.item())
