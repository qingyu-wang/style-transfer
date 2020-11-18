import torch
import torch.nn as nn


class ContentLoss(nn.Module):

    def __init__(self):
        super(ContentLoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, output_features, target_features):
        loss = self.mse_loss(output_features, target_features)
        return loss


class StyleLoss(nn.Module):

    def __init__(self, eps=1e-5):
        super(StyleLoss, self).__init__()
        self.eps = eps
        self.mse_loss = nn.MSELoss()

    def calc_mean_std(self, tensor):
        _N, _C, _H, _W = tensor.shape
        var = tensor.view(_N, _C, -1).var(dim=2) + self.eps # channel-wise
        std = var.sqrt().view(_N, _C, 1, 1)
        mean = tensor.view(_N, _C, -1).mean(dim=2).view(_N, _C, 1, 1)
        return mean, std

    def forward(self, output_features, style_features):
        output_mean, output_std = self.calc_mean_std(output_features)
        style_mean, style_std = self.calc_mean_std(style_features)
        loss = self.mse_loss(output_mean, style_mean) + self.mse_loss(output_std, style_std)
        return loss
