import torch
import torch.nn as nn


class ContentLoss(nn.Module):

    def __init__(self):
        super(ContentLoss, self).__init__()
        self.criterion = nn.MSELoss()

    def forward(self, output_feature, content_feature):
        loss = self.criterion(output_feature, content_feature)
        return loss


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
        self.criterion = nn.MSELoss()
    
    def forward(self, output_gram_matrice, style_gram_matrice):
        loss = self.criterion(output_gram_matrice, style_gram_matrice)
        return loss


class TVLoss(nn.Module):

    def __init__(self):
        super(TVLoss, self).__init__()

    def forward(self, outputs):
        x = outputs[:, :, 1:, :] - outputs[:, :, :-1, :]
        y = outputs[:, :, :, 1:] - outputs[:, :, :, :-1]
        loss = torch.sum(torch.abs(x)) + torch.sum(torch.abs(y))
        return loss
