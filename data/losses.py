import torch
import torch.nn.functional as F
from math import exp
from torch.nn import MSELoss
from torch.autograd import Variable

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
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
    cs = torch.mean(v1 / v2)
    cs = (cs + 1) / 2

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
        ret = (ret + 1) / 2
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)
        ret = (ret + 1) / 2

    if full:
        return ret, cs
    return ret

def ssim_top(img1, img2, window_size=11, size_average=True, val_range=None, normalize=False):

    device = img1.device
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
    levels = weights.size()[0]
    mssim = []
    mcs = []

    for _ in range(levels):
        sim, cs = ssim(img1, img2, window_size=window_size, size_average=size_average, full=True, val_range=val_range)
        mssim.append(sim)
        mcs.append(cs)

        img1 = F.avg_pool2d(img1, (2, 2))
        img2 = F.avg_pool2d(img2, (2, 2))

    mssim = torch.stack(mssim)
    mcs = torch.stack(mcs)
    if normalize:
        mssim = (mssim + 1) / 2
        mcs = (mcs + 1) / 2

    pow1 = mcs ** weights
    pow2 = mssim ** weights
    output = torch.prod(pow1[:-1] * pow2[-1])
    return output

def loss_function(gt_pred, gt):
    l2_loss = torch.nn.MSELoss()

    msssim = ((1-ssim_top(gt,gt_pred)))/2
    f1 = l2_loss(gt_pred, gt)

    return msssim, f1

def cluster(feat_proto_all):
    crit = torch.nn.SmoothL1Loss(reduction='mean')
    length = len(feat_proto_all)
    B,C,H,W = feat_proto_all[0].detach().size()
    feat_proto_avg = torch.zeros((B,C,H,W)).cuda()

    for idx in range(length):
        feat_proto_avg += feat_proto_all[idx]
    feat_proto_avg = feat_proto_avg/length

    losses = 0
    for idx in range(length):
        losses += crit(feat_proto_avg,feat_proto_all[idx])
    losses = losses/length
    return losses,feat_proto_avg

def triplet(feat_proto_p,feat_proto_n,feat_proto_avg,thred = 1):
    crit = torch.nn.SmoothL1Loss(reduction='mean')
    length_p = len(feat_proto_p)
    length_n = len(feat_proto_n)

    list_n,list_p = [],[]
    for idx in range(length_n):
        losses = crit(feat_proto_avg,feat_proto_n[idx])
        list_n.append(losses)
    losses_n = max(list_n)

    for idx in range(length_p):
        losses = crit(feat_proto_avg,feat_proto_p[idx])
        list_p.append(losses)
    losses_p = min(list_p)
    device = losses_p.device
    zeros_0 = Variable(torch.tensor(0.0).to(device), requires_grad=False)

    losses = max(zeros_0,losses_n-losses_p+thred)
    return losses

def overall_discriminator_pass(discriminator, recon_batch, gt, valid, fake):
    adversarial_loss = MSELoss()
    gt = gt.cuda()
    real_loss = adversarial_loss(discriminator(gt), valid)
    fake_loss = adversarial_loss(discriminator(recon_batch.detach()), fake)
    d_loss = (real_loss + fake_loss) / 2
    return d_loss

def grad_loss(pred,gt):
    grad_loss0 = torch.nn.L1Loss()
    gt = gt.cuda()

    loss1 = grad_loss0((pred[:,:,1:]-pred[:,:,0:-1]),(gt[:,:,1:]-gt[:,:,0:-1]))
    loss2 = grad_loss0((pred[:,:,:,1:]-pred[:,:,:,0:-1]),(gt[:,:,:,1:]-gt[:,:,:,0:-1]))

    L_grad = (loss1+loss2)
    return L_grad

def kl_loss(x1,x2):
    kl = F.kl_div(x1.softmax(dim=-1).log(), x2.softmax(dim=-1), reduction='mean')
    return kl
