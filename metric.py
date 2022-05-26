
import argparse
import math
import os

import imageio
import lpips
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Mean Square Error
class MSE(object):
    def __call__(self, pred, gt):
        return torch.mean((pred - gt) ** 2)

# Peak Signal to Noise Ratio
class PSNR(object):
    def __call__(self, pred, gt):
        mse = torch.mean((pred - gt) ** 2)
        return 10 * torch.log10(1 / mse)


# structural similarity index
class SSIM(object):
    '''
    borrowed from https://github.com/huster-wgm/Pytorch-metrics/blob/master/metrics.py
    '''
    def gaussian(self, w_size, sigma):
        gauss = torch.Tensor([math.exp(-(x - w_size//2)**2/float(2*sigma**2)) for x in range(w_size)])
        return gauss/gauss.sum()

    def create_window(self, w_size, channel=1):
        _1D_window = self.gaussian(w_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, w_size, w_size).contiguous()
        return window

    def __call__(self, y_pred, y_true, w_size=11, size_average=True, full=False):
        """
        args:
            y_true : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
            y_pred : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
            w_size : int, default 11
            size_average : boolean, default True
            full : boolean, default False
        return ssim, larger the better
        """
        # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
        if torch.max(y_pred) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(y_pred) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val

        padd = 0
        (_, channel, height, width) = y_pred.size()
        window = self.create_window(w_size, channel=channel).to(y_pred.device)

        mu1 = F.conv2d(y_pred, window, padding=padd, groups=channel)
        mu2 = F.conv2d(y_true, window, padding=padd, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(y_pred * y_pred, window, padding=padd, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(y_true * y_true, window, padding=padd, groups=channel) - mu2_sq
        sigma12 = F.conv2d(y_pred * y_true, window, padding=padd, groups=channel) - mu1_mu2

        C1 = (0.01 * L) ** 2
        C2 = (0.03 * L) ** 2

        v1 = 2.0 * sigma12 + C2
        v2 = sigma1_sq + sigma2_sq + C2
        cs = torch.mean(v1 / v2)  # contrast sensitivity

        ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

        if size_average:
            ret = ssim_map.mean()
        else:
            ret = ssim_map.mean(1).mean(1).mean(1)

        if full:
            return ret, cs
        return ret



# Learned Perceptual Image Patch Similarity
class LPIPS(object):
    '''
    borrowed from https://github.com/huster-wgm/Pytorch-metrics/blob/master/metrics.py
    '''
    def __init__(self):
        self.model = lpips.LPIPS(net='vgg').cuda()

    def __call__(self, y_pred, y_true, normalized=True):
        """
        args:
            y_true : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
            y_pred : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
            normalized : change [0,1] => [-1,1] (default by LPIPS)
        return LPIPS, smaller the better
        """
        if normalized:
            y_pred = y_pred * 2.0 - 1.0
            y_true = y_true * 2.0 - 1.0
        error =  self.model.forward(y_pred, y_true)
        return torch.mean(error)


def read_images_in_dir(imgs_dir):
    imgs = []
    fnames = os.listdir(imgs_dir)
    fnames.sort()
    for fname in fnames:
        if fname.endswith(".mp4") == True:  # ignore canonical space, only evalute real scene
            continue
        if fname.endswith(".txt") == True:  # ignore canonical space, only evalute real scene
            continue

        img_path = os.path.join(imgs_dir, fname)
        # print(img_path)
        img = imageio.imread(img_path)
        img = (np.array(img) / 255.).astype(np.float32)
        img = np.transpose(img, (2, 0, 1))
        imgs.append(img)
    
    imgs = np.stack(imgs)       
    return imgs

def estim_error(estim, gt):
    errors = dict()
    metric = PSNR()
    errors["psnr"] = metric(estim, gt).item()
    metric = SSIM()
    errors["ssim"] = metric(estim, gt).item()
    metric = LPIPS()
    errors["lpips"] = metric(estim, gt).item()
    return errors


parser = argparse.ArgumentParser()
parser.add_argument('--estim_dir', type = str, default = None , help ='images path')
parser.add_argument('--gt_dir', type = str, default = None ,help ='GT path')
args = parser.parse_args()

psnr_cal = 0
ssim_cal = 0
lpips_cal = 0
scens = ['hellwarrior','mutant','hook','bouncingballs','lego','trex','standup','jumpingjacks']
for str in scens:
    estim_dir = args.estim_dir + '/dnerf_'+str+'-400/render_test_fine_last'
    gt_dir = args.gt_dir + '/'+str+'/renderonly_test_799999/gt'

    estim = read_images_in_dir(estim_dir)
    gt = read_images_in_dir(gt_dir)

    estim = torch.Tensor(estim).cuda()
    gt = torch.Tensor(gt).cuda()

    errors = estim_error(estim, gt)
    psnr_cal += errors["psnr"]
    ssim_cal += errors["ssim"]
    lpips_cal += errors["lpips"]
    print(str , errors)
print(psnr_cal/8 , ssim_cal/8 , lpips_cal/8)
