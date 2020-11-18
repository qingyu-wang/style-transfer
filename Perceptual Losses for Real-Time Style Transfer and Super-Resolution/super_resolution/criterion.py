import torch
import torch.nn as nn


class PixelLoss(nn.Module):

    def __init__(self):
        super(PixelLoss, self).__init__()
        self.mse_loss = torch.nn.MSELoss()

    def forward(self, outputs, targets):
        loss = self.mse_loss(outputs, targets)
        return loss


class ContentLoss(nn.Module):

    def __init__(self):
        super(ContentLoss, self).__init__()
        self.mse_loss = torch.nn.MSELoss()

    def forward(self, output_features, target_features):
        loss = self.mse_loss(output_features, target_features)
        return loss


def calc_gram_matrix(feature, normalize=True):
    _C, _H, _W = feature.shape
    x = feature.view(_C, _H * _W)
    gram_matrix = torch.mm(x, x.t())
    if normalize:
        gram_matrix = gram_matrix / (_C * _H * _W)
    return gram_matrix


def calc_gram_matrice(features, normalize=True):
    _B, _C, _H, _W = features.shape
    x = features.view(_B, _C, _H * _W)
    gram_matrice = torch.bmm(x, x.permute(0, 2, 1))
    if normalize:
        gram_matrice = gram_matrice / (_C * _H * _W)
    return gram_matrice


class StyleLoss(nn.Module):

    def __init__(self):
        super(StyleLoss, self).__init__()
        self.mse_loss = torch.nn.MSELoss()

    def forward(self, output_features, target_features):
        loss = self.mse_loss(calc_gram_matrice(output_features), calc_gram_matrice(target_features))
        return loss


class TVLoss(nn.Module):

    def __init__(self):
        super(TVLoss, self).__init__()

    def forward(self, outputs):
        x = outputs[:, :, 1:, :] - outputs[:, :, :-1, :]
        y = outputs[:, :, :, 1:] - outputs[:, :, :, :-1]
        loss = torch.sum(torch.abs(x)) + torch.sum(torch.abs(y))
        return loss
