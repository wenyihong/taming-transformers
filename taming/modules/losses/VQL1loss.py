import torch
import torch.nn as nn
import torch.nn.functional as F

from taming.modules.losses.lpips import LPIPS



class VQL1loss(nn.Module):
    def __init__(self, codebook_weight=1.0, pixelloss_weight=1.0,
                 perceptual_weight=1.0):
        super().__init__()
        self.codebook_weight = codebook_weight
        self.pixel_weight = pixelloss_weight


    def forward(self, codebook_loss, targets, reconstructions,
                split="train"):
        rec_L1loss = torch.abs(targets.contiguous() - reconstructions.contiguous())
        nll_loss = torch.mean(rec_L1loss)
        loss = nll_loss + self.codebook_weight * codebook_loss.mean()
        
        log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
                "{}/quant_loss".format(split): codebook_loss.detach().mean(),
                "{}/rec_loss".format(split): rec_L1loss.detach().mean(),
                }
        return loss, log


class VQL1GradmaskPerceptualloss(nn.Module):
    def __init__(self, codebook_weight=1.0, pixelloss_weight=1.0, gradloss_weight=1.0,
                 perceptual_weight=1.0, use_grad = False, use_percep=False):
        super().__init__()
        self.codebook_weight = codebook_weight
        self.pixel_weight = pixelloss_weight
        self.gradloss_weight = gradloss_weight
        self.perceptual_weight = perceptual_weight
        self.use_grad = use_grad
        self.use_percep = use_percep
        if use_percep:
                self.perceptual_loss = LPIPS().eval()


    def forward(self, codebook_loss, targets, reconstructions, grad_mask=None,
                split="train"):
        rec_L1loss = torch.abs(targets.contiguous() - reconstructions.contiguous())
        if self.use_grad:
                rec_L1loss = rec_L1loss * self.pixel_weight+grad_mask*self.gradloss_weight
        if self.use_percep:
                p_loss = self.perceptual_loss(targets.contiguous(), reconstructions.contiguous())
                rec_L1loss = rec_L1loss + self.perceptual_weight * p_loss
        else:
                p_loss = torch.tensor([0.0])
        nll_loss = torch.mean(rec_L1loss)
        loss = nll_loss + self.codebook_weight * codebook_loss.mean()
        
        log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
                "{}/quant_loss".format(split): codebook_loss.detach().mean(),
                "{}/p_loss".format(split): p_loss.detach().mean(),
                "{}/rec_loss".format(split): rec_L1loss.detach().mean(),
                }
        return loss, log
