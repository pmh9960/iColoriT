import torch.nn as nn
import torch


class HuberLoss(nn.Module):
    def __init__(self, delta=.01):
        super(HuberLoss, self).__init__()
        self.delta = delta

    def __call__(self, input, target):
        mask = torch.zeros_like(input)
        mann = torch.abs(input - target)
        eucl = .5 * (mann**2)
        mask[...] = mann < self.delta

        # loss = eucl * mask + self.delta * (mann - .5 * self.delta) * (1 - mask)
        loss = eucl * mask / self.delta + (mann - .5 * self.delta) * (1 - mask)
        return torch.sum(loss, dim=-1, keepdim=False).mean()


class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()

    def __call__(self, input, target):
        return torch.sum(torch.abs(input - target), dim=-1, keepdim=False).mean()


class L2Loss(nn.Module):
    def __init__(self):
        super(L2Loss, self).__init__()

    def __call__(self, input, target):
        return torch.sum((input - target)**2, dim=-1, keepdim=False).mean()
